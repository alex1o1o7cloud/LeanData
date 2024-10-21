import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_zero_range_of_a_l554_55493

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.exp x - 2 * x - 1

-- Define the function g
noncomputable def g (a x : ℝ) : ℝ := a * f x + (1 - a) * Real.exp x

-- Theorem for the tangent line
theorem tangent_line_at_zero :
  ∃ (m b : ℝ), ∀ x, m * x + b = f x + (deriv f 0) * (x - 0) :=
sorry

-- Theorem for the range of a
theorem range_of_a (a : ℝ) :
  (∃ x₁ x₂, x₁ ≠ x₂ ∧ g a x₁ = 0 ∧ g a x₂ = 0) ↔ a > Real.sqrt (Real.exp 1) / 2 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_zero_range_of_a_l554_55493


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_bounds_l554_55408

/-- Given a square P and a positive integer n, f(n) represents the maximum number of 
    elements in a partition of P into rectangles such that each line parallel to a side 
    of P intersects at most n rectangle interiors. -/
def f (n : ℕ+) : ℕ := sorry

/-- Theorem stating the bounds for f(n) -/
theorem f_bounds (n : ℕ+) : 3 * 2^(n.val - 1) - 2 ≤ f n ∧ f n ≤ 3^n.val - 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_bounds_l554_55408


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_valid_integers_l554_55459

def move_initial_digit_to_end (n : Int) : Int :=
  sorry

theorem no_valid_integers : 
  ∀ n : Int, n ≠ 0 → 
    (let N := move_initial_digit_to_end n;
     N ≠ 5 * n ∧ N ≠ 6 * n ∧ N ≠ 8 * n) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_valid_integers_l554_55459


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_between_curves_l554_55447

-- Define the two curves
noncomputable def f (x : ℝ) := x^2
noncomputable def g (x : ℝ) := Real.sqrt x

-- Define the area enclosed by the curves
noncomputable def enclosed_area : ℝ := ∫ x in (0)..(1), g x - f x

-- Theorem statement
theorem area_between_curves : enclosed_area = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_between_curves_l554_55447


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_n_with_five_pairs_l554_55416

/-- Count the number of distinct ordered pairs of positive integers (a, b) such that a^2 + b^2 = n -/
def count_pairs (n : ℕ) : ℕ :=
  (Finset.filter (fun p : ℕ × ℕ => p.1^2 + p.2^2 = n ∧ p.1 > 0 ∧ p.2 > 0) (Finset.product (Finset.range (n+1)) (Finset.range (n+1)))).card

/-- The smallest positive integer n for which there are precisely five distinct ordered pairs of positive integers (a, b) such that a^2 + b^2 = n -/
theorem smallest_n_with_five_pairs :
  200 = (Finset.filter (fun n : ℕ => n > 0 ∧ count_pairs n = 5) (Finset.range 201)).min' sorry :=
sorry

#eval count_pairs 200

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_n_with_five_pairs_l554_55416


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_amanda_max_servings_l554_55427

/-- Represents the recipe and available ingredients --/
structure Recipe :=
  (servings : ℕ)
  (chocolate : ℚ)
  (sugar : ℚ)
  (milk : ℚ)

/-- Calculates the maximum number of servings possible given a recipe and available ingredients --/
def max_servings (recipe : Recipe) (available_chocolate : ℚ) (available_sugar : ℚ) : ℕ :=
  Int.toNat <| min 
    (Int.floor (available_chocolate / recipe.chocolate * recipe.servings))
    (Int.floor (available_sugar / recipe.sugar * recipe.servings))

/-- The recipe for 4 servings --/
def base_recipe : Recipe :=
  { servings := 4
  , chocolate := 3
  , sugar := 1/3
  , milk := 5 }

/-- Theorem stating that Amanda can make at most 12 servings --/
theorem amanda_max_servings :
  max_servings base_recipe 9 2 = 12 := by
  sorry

#eval max_servings base_recipe 9 2

end NUMINAMATH_CALUDE_ERRORFEEDBACK_amanda_max_servings_l554_55427


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_approximate_overall_savings_percentage_l554_55413

/-- Calculates the approximate overall percentage of savings given the discounts and total spent on three items. -/
theorem approximate_overall_savings_percentage
  (kitchen_appliance_savings : ℝ)
  (kitchen_appliance_discount : ℝ)
  (home_decor_savings : ℝ)
  (home_decor_discount : ℝ)
  (gardening_tool_savings : ℝ)
  (gardening_tool_discount : ℝ)
  (total_spent : ℝ)
  (h1 : kitchen_appliance_savings = 8)
  (h2 : kitchen_appliance_discount = 0.20)
  (h3 : home_decor_savings = 12)
  (h4 : home_decor_discount = 0.15)
  (h5 : gardening_tool_savings = 4)
  (h6 : gardening_tool_discount = 0.10)
  (h7 : total_spent = 95) :
  ∃ overall_savings_percentage : ℝ,
    abs (overall_savings_percentage - 40.63) < 0.01 := by
  let kitchen_appliance_original := kitchen_appliance_savings / kitchen_appliance_discount
  let home_decor_original := home_decor_savings / home_decor_discount
  let gardening_tool_original := gardening_tool_savings / gardening_tool_discount
  let total_original := kitchen_appliance_original + home_decor_original + gardening_tool_original
  let total_savings := total_original - total_spent
  let overall_savings_percentage := (total_savings / total_original) * 100
  use overall_savings_percentage
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_approximate_overall_savings_percentage_l554_55413


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_integer_values_l554_55406

def sequence_a : ℕ → ℚ
  | 0 => 1  -- Adding this case for 0
  | 1 => 1
  | 2 => 10/9
  | (n + 3) => 3 * sequence_a (n + 2) - 2 * sequence_a (n + 1)

theorem infinite_integer_values :
  ∃ (S : Set ℕ), Set.Infinite S ∧ ∀ n ∈ S, ∃ m : ℤ, sequence_a n = m := by
  -- Proof goes here
  sorry

#eval sequence_a 7  -- This will help verify if the function is working correctly

end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_integer_values_l554_55406


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_maximizes_sum_of_squares_l554_55440

-- Define a circle
structure Circle where
  radius : ℝ
  center : ℝ × ℝ

-- Define a polygon inscribed in a circle
structure InscribedPolygon (c : Circle) where
  vertices : List (ℝ × ℝ)
  inscribed : ∀ v ∈ vertices, (v.1 - c.center.1)^2 + (v.2 - c.center.2)^2 = c.radius^2

-- Function to calculate the sum of squares of sides of a polygon
noncomputable def sumOfSquaresOfSides {c : Circle} (p : InscribedPolygon c) : ℝ :=
  let sides := List.zipWith (λ a b => (a.1 - b.1)^2 + (a.2 - b.2)^2) p.vertices (p.vertices.rotate 1)
  List.sum sides

-- Define an equilateral triangle inscribed in a circle
noncomputable def EquilateralTriangle (c : Circle) : InscribedPolygon c :=
  { vertices := [
      (c.center.1 + c.radius, c.center.2),
      (c.center.1 - c.radius / 2, c.center.2 + c.radius * Real.sqrt 3 / 2),
      (c.center.1 - c.radius / 2, c.center.2 - c.radius * Real.sqrt 3 / 2)
    ],
    inscribed := by sorry }

-- Theorem statement
theorem equilateral_triangle_maximizes_sum_of_squares :
  ∀ (c : Circle) (p : InscribedPolygon c),
    sumOfSquaresOfSides (EquilateralTriangle c) ≥ sumOfSquaresOfSides p :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_maximizes_sum_of_squares_l554_55440


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_l554_55474

/-- Given an arithmetic sequence, S_m is the sum of the first m terms -/
def S (m : ℕ) : ℝ := sorry

/-- The conditions of the problem -/
axiom S_m_value : S 1 = 30
axiom S_2m_value : S 2 = 100

/-- The theorem to prove -/
theorem arithmetic_sequence_sum : S 3 = 170 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_l554_55474


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_final_water_level_is_34cm_l554_55432

/-- Represents the height of a liquid in centimeters -/
def Height := ℝ

/-- Represents the density of a liquid in kg/m³ -/
def Density := ℝ

/-- Represents a cylindrical vessel containing a liquid -/
structure Vessel where
  liquid_height : Height
  liquid_density : Density

/-- Represents a system of two connected vessels -/
structure ConnectedVessels where
  vessel1 : Vessel
  vessel2 : Vessel
  initial_height : Height

/-- Calculates the final water level in the first vessel after opening the valve -/
noncomputable def calculate_final_water_level (system : ConnectedVessels) : Height :=
  sorry

/-- The theorem stating that the final water level is 34 cm -/
theorem final_water_level_is_34cm (water_density oil_density : Density)
    (h : Height) (system : ConnectedVessels) :
    water_density = (1000 : ℝ) →
    oil_density = (700 : ℝ) →
    h = (40 : ℝ) →
    system.initial_height = h →
    system.vessel1.liquid_height = h →
    system.vessel1.liquid_density = water_density →
    system.vessel2.liquid_height = h →
    system.vessel2.liquid_density = oil_density →
    calculate_final_water_level system = (34 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_final_water_level_is_34cm_l554_55432


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_series_sum_l554_55401

noncomputable def geometric_sum (a : ℝ) (r : ℝ) (n : ℕ) : ℝ :=
  a * (1 - r^n) / (1 - r)

theorem geometric_series_sum :
  let a : ℝ := 2
  let r : ℝ := -2
  let n : ℕ := 11
  geometric_sum a r n = 1366 := by
  -- Unfold the definitions
  unfold geometric_sum
  -- Simplify the expression
  simp [pow_succ]
  -- The rest of the proof would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_series_sum_l554_55401


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_2theta_value_l554_55480

/-- Given vectors a, b, c in R^2 and a condition on their parallelism, 
    prove that sin(2θ) equals -12/13 -/
theorem sin_2theta_value (θ : ℝ) 
  (a b c : Fin 2 → ℝ)
  (ha : a = λ i => if i = 0 then Real.sin θ else 1)
  (hb : b = λ i => if i = 0 then -Real.sin θ else 0)
  (hc : c = λ i => if i = 0 then Real.cos θ else -1)
  (h_parallel : ∃ (k : ℝ), (2 • a - b) = k • c) :
  Real.sin (2 * θ) = -12/13 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_2theta_value_l554_55480


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_monic_cubic_polynomial_l554_55437

-- Define a monic cubic polynomial with real coefficients
def monicCubicPolynomial (a b c : ℝ) (x : ℂ) : ℂ :=
  x^3 + a*x^2 + b*x + c

-- State the theorem
theorem unique_monic_cubic_polynomial :
  ∃! (a b c : ℝ),
    let q := monicCubicPolynomial a b c
    q (3 - 2*I) = 0 ∧ q 0 = -20 ∧
    ∀ x, q x = x^3 - (58/13)*x^2 + (49/13)*x + 20 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_monic_cubic_polynomial_l554_55437


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_zero_difference_bound_g_increasing_g_zero_between_3_and_4_f_zero_at_3_5_l554_55470

-- Define the function g(x)
noncomputable def g (x : ℝ) : ℝ := Real.log x + 2 * x - 8

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := Real.log (x - 5/2)

-- State the theorem
theorem zero_difference_bound :
  ∃ (z_g z_f : ℝ), g z_g = 0 ∧ f z_f = 0 ∧ |z_g - z_f| ≤ 0.5 := by
  sorry

-- Additional helper theorem to show g(x) is increasing
theorem g_increasing : StrictMono g := by
  sorry

-- Theorem to show g has a zero between 3 and 4
theorem g_zero_between_3_and_4 :
  ∃ z : ℝ, 3 < z ∧ z < 4 ∧ g z = 0 := by
  sorry

-- Theorem to show f has a zero at 3.5
theorem f_zero_at_3_5 : f 3.5 = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_zero_difference_bound_g_increasing_g_zero_between_3_and_4_f_zero_at_3_5_l554_55470


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_expression_g_expression_l554_55425

-- Part 1: Function f
noncomputable def f : ℝ → ℝ := sorry

axiom f_def : ∀ x : ℝ, f (x + 1) = x^2 - 2*x

theorem f_expression : ∀ x : ℝ, f x = x^2 - 4*x + 3 := by sorry

-- Part 2: Function g
noncomputable def g : ℝ → ℝ := sorry

axiom g_quadratic : ∃ a b c : ℝ, ∀ x : ℝ, g x = a * x^2 + b * x + c
axiom g_root_neg_two : g (-2) = 0
axiom g_root_three : g 3 = 0
axiom g_at_zero : g 0 = -3

theorem g_expression : ∀ x : ℝ, g x = (1/2) * x^2 - (1/2) * x - 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_expression_g_expression_l554_55425


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_least_non_representable_l554_55417

def representable (n : ℕ) : Prop :=
  ∃ (a b c d : ℕ), a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧
    n * (2^c - 2^d) = 2^a - 2^b

theorem least_non_representable : 
  (∀ m : ℕ, m > 0 ∧ m < 11 → representable m) ∧ ¬representable 11 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_least_non_representable_l554_55417


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hours_worked_per_day_l554_55409

/-- Represents the number of hours worked per day -/
noncomputable def hours_per_day : ℝ := sorry

/-- Hourly wage in dollars -/
def hourly_wage : ℝ := 10

/-- Number of working days per week -/
def days_per_week : ℕ := 5

/-- Number of weeks of savings -/
def weeks_of_savings : ℕ := 4

/-- Robby's savings rate -/
noncomputable def robby_savings_rate : ℝ := 2 / 5

/-- Jaylen's savings rate -/
noncomputable def jaylen_savings_rate : ℝ := 3 / 5

/-- Miranda's savings rate -/
noncomputable def miranda_savings_rate : ℝ := 1 / 2

/-- Total combined savings in dollars -/
def total_savings : ℝ := 3000

theorem hours_worked_per_day :
  hourly_wage * hours_per_day * (days_per_week : ℝ) * (weeks_of_savings : ℝ) *
  (robby_savings_rate + jaylen_savings_rate + miranda_savings_rate) = total_savings →
  hours_per_day = 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hours_worked_per_day_l554_55409


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_susan_disproves_tom_l554_55452

-- Define the structure of a card
structure Card where
  letter : Char
  number : Nat

-- Define what a consonant is
def isConsonant (c : Char) : Bool :=
  c ∈ ['B', 'C', 'D']

-- Define what a prime number is
def isPrime (n : Nat) : Prop :=
  n > 1 ∧ ∀ m : Nat, 1 < m → m < n → ¬(n % m = 0)

-- Define Tom's statement
def tomStatement (card : Card) : Prop :=
  isConsonant card.letter → isPrime card.number

-- Define the set of cards
def cards : List Card :=
  [⟨'A', 2⟩, ⟨'B', 5⟩, ⟨'C', 8⟩, ⟨'D', 3⟩]

theorem susan_disproves_tom :
  ∃! card : Card, card ∈ cards ∧ ¬(tomStatement card) :=
by
  sorry

#check susan_disproves_tom

end NUMINAMATH_CALUDE_ERRORFEEDBACK_susan_disproves_tom_l554_55452


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_common_ratio_l554_55429

/-- The common ratio of a geometric sequence -/
noncomputable def common_ratio (a : ℕ → ℝ) : ℝ :=
  if a 1 = 0 then 0 else a 2 / a 1

/-- Given a geometric sequence with sum S_n of the first n terms, 
    if 8S_6 = 7S_3, then the common ratio is -1/2 -/
theorem geometric_sequence_common_ratio 
  (a : ℕ → ℝ) 
  (S : ℕ → ℝ) 
  (h1 : ∀ n, S n = (a 1) * (1 - (common_ratio a)^n) / (1 - common_ratio a))
  (h2 : 8 * S 6 = 7 * S 3) :
  common_ratio a = -1/2 :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_common_ratio_l554_55429


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_positive_expressions_l554_55466

theorem positive_expressions (x : ℝ) (h : x < 0) :
  -x^3 > 0 ∧ -Real.exp (Real.log 3 * x) + 1 > 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_positive_expressions_l554_55466


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_f_l554_55420

open Real

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 2 * (sin x)^3 + (cos x)^2

-- State the theorem
theorem min_value_f :
  ∃ (x : ℝ), 0 ≤ x ∧ x ≤ π/2 ∧ f x = 26/27 ∧ ∀ (y : ℝ), 0 ≤ y ∧ y ≤ π/2 → f y ≥ 26/27 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_f_l554_55420


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_n_for_triple_l554_55477

def S : Finset ℕ := Finset.range 16

theorem smallest_n_for_triple (n : ℕ) : n = 13 ↔ 
  (∀ T : Finset ℕ, T ⊆ S → T.card = n → 
    ∃ a b, a ∈ T ∧ b ∈ T ∧ b = 3 * a) ∧ 
  (∀ m : ℕ, m < n → 
    ∃ T : Finset ℕ, T ⊆ S ∧ T.card = m ∧ 
      ∀ a b, a ∈ T → b ∈ T → b ≠ 3 * a) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_n_for_triple_l554_55477


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_ratio_l554_55451

-- Define arithmetic sequences a_n and b_n
def a (n : ℕ) : ℚ := 76 * n - 24
def b (n : ℕ) : ℚ := 4 * n - 1

-- Define sums of first n terms
def S (n : ℕ) : ℚ := (n : ℚ) * (38 * n + 14)
def T (n : ℕ) : ℚ := (n : ℚ) * (2 * n + 1)

-- State the theorem
theorem arithmetic_sequence_ratio 
  (h_arithmetic : ∀ n : ℕ, n > 0 → S n / T n = (38 * n + 14) / (2 * n + 1)) :
  a 6 / b 7 = 16 := by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_ratio_l554_55451


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_time_l554_55426

/-- The time (in seconds) it takes for two trains to cross each other -/
noncomputable def crossing_time (train1_length train2_length : ℝ) (train1_speed train2_speed : ℝ) : ℝ :=
  (train1_length + train2_length) / (train1_speed + train2_speed)

/-- Conversion factor from km/hr to m/s -/
noncomputable def km_per_hr_to_m_per_s : ℝ := 1000 / 3600

theorem train_crossing_time :
  let train1_length : ℝ := 250
  let train2_length : ℝ := 250.04
  let train1_speed : ℝ := 120 * km_per_hr_to_m_per_s
  let train2_speed : ℝ := 80 * km_per_hr_to_m_per_s
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.1 ∧ 
    |crossing_time train1_length train2_length train1_speed train2_speed - 9| < ε :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_time_l554_55426


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_vertex_y_coord_l554_55497

/-- The parabola equation -/
def parabola_equation (x : ℝ) : ℝ := -2 * x^2 - 16 * x - 42

/-- The vertex of a parabola y = ax² + bx + c is at x = -b/(2a) -/
noncomputable def vertex_x (a b : ℝ) : ℝ := -b / (2 * a)

/-- The y-coordinate of the vertex -/
noncomputable def vertex_y (a b c : ℝ) : ℝ := 
  let x := vertex_x a b
  a * x^2 + b * x + c

theorem parabola_vertex_y_coord : 
  vertex_y (-2) (-16) (-42) = -10 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_vertex_y_coord_l554_55497


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_APF_l554_55482

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 - y^2/3 = 1

-- Define the right focus F
def F : ℝ × ℝ := (2, 0)

-- Define point A
def A : ℝ × ℝ := (1, 3)

-- Define point P on the hyperbola
def P : ℝ × ℝ := (2, 3)

-- State that P is on the hyperbola
axiom P_on_hyperbola : hyperbola P.1 P.2

-- State that PF is perpendicular to x-axis
axiom PF_perpendicular : P.2 = F.2

-- Define the area of a triangle
noncomputable def triangle_area (a b c : ℝ × ℝ) : ℝ :=
  (1/2) * abs ((b.1 - a.1) * (c.2 - a.2) - (c.1 - a.1) * (b.2 - a.2))

-- Theorem statement
theorem area_of_triangle_APF :
  triangle_area A P F = 3/2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_APF_l554_55482


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_percent_error_5_factorial_l554_55450

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def correct_calculation (n : ℕ) : ℕ := (factorial n) * 3

def incorrect_calculation (n : ℕ) : ℕ := (factorial n) / 3

noncomputable def error_percentage (n : ℕ) : ℚ :=
  let error := (correct_calculation n - incorrect_calculation n : ℚ)
  let correct := (correct_calculation n : ℚ)
  (error / correct) * 100

noncomputable def rounded_error_percentage (n : ℕ) : ℤ :=
  Int.floor ((error_percentage n) + 0.5)

theorem percent_error_5_factorial :
  rounded_error_percentage 5 = 89 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_percent_error_5_factorial_l554_55450


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pentagon_side_length_when_perimeter_equals_area_l554_55469

/-- The side length of a regular pentagon where the perimeter equals the area -/
noncomputable def pentagon_side_length : ℝ := 4 * Real.sqrt (1 + 0.4 * Real.sqrt 5)

/-- Perimeter of a regular pentagon with side length s -/
def pentagon_perimeter (s : ℝ) : ℝ := 5 * s

/-- Area of a regular pentagon with side length s -/
noncomputable def pentagon_area (s : ℝ) : ℝ := (1 / 4) * Real.sqrt (5 * (5 + 2 * Real.sqrt 5)) * s^2

theorem pentagon_side_length_when_perimeter_equals_area :
  pentagon_perimeter pentagon_side_length = pentagon_area pentagon_side_length := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_pentagon_side_length_when_perimeter_equals_area_l554_55469


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_difference_l554_55421

theorem quadratic_difference (a b c m n : ℤ) :
  let f (x : ℤ) := a * x^2 + b * x + c
  (f m - f n = 1) → (|m - n| = 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_difference_l554_55421


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_h_l554_55488

noncomputable def f (x : ℝ) : ℝ := 1 / Real.cos x

noncomputable def g (x : ℝ) : ℝ := f (x - Real.pi/3)

noncomputable def h (x : ℝ) : ℝ := f x + g x

def domain : Set ℝ := { x | Real.pi/12 ≤ x ∧ x ≤ Real.pi/4 }

theorem max_value_of_h :
  ∃ (M : ℝ), M = Real.sqrt 6 ∧ ∀ x ∈ domain, h x ≤ M := by
  sorry

#check max_value_of_h

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_h_l554_55488


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_and_line_properties_l554_55424

-- Define the parabola
noncomputable def parabola (p : ℝ) (x y : ℝ) : Prop := y^2 = 2*p*x ∧ p > 0

-- Define the focus
noncomputable def focus (p : ℝ) : ℝ × ℝ := (1/2, 0)

-- Define a line with slope 2
noncomputable def line (t : ℝ) (x y : ℝ) : Prop := y = 2*x + t

-- Define perpendicularity
noncomputable def perpendicular (x₁ y₁ x₂ y₂ : ℝ) : Prop := x₁*x₂ + y₁*y₂ = 0

theorem parabola_and_line_properties :
  ∃ (p t : ℝ) (x₁ y₁ x₂ y₂ : ℝ),
    -- Parabola conditions
    parabola p x₁ y₁ ∧ parabola p x₂ y₂ ∧
    -- Line conditions
    line t x₁ y₁ ∧ line t x₂ y₂ ∧
    -- Points are different from origin
    (x₁ ≠ 0 ∨ y₁ ≠ 0) ∧ (x₂ ≠ 0 ∨ y₂ ≠ 0) ∧
    -- Perpendicularity condition
    perpendicular x₁ y₁ x₂ y₂ ∧
    -- Focus condition
    focus p = (1/2, 0) →
    -- Conclusions
    p = 1 ∧ t = -4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_and_line_properties_l554_55424


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_floor_sum_specific_l554_55476

noncomputable def arithmetic_floor_sum (start : ℝ) (diff : ℝ) (terms : ℕ) : ℝ :=
  (Finset.range terms).sum (fun i => ⌊start + i * diff⌋)

theorem arithmetic_floor_sum_specific : 
  arithmetic_floor_sum 2 0.8 102 = 4294.2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_floor_sum_specific_l554_55476


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_inscribed_square_l554_55434

/-- A square with specified properties --/
structure Square where
  side_length : ℝ
  area : ℝ
  sides_parallel_to : ℝ → Prop
  sides_tangent_to_semicircles : ℝ → Prop

/-- The area of the inscribed square EFGH in a 6x6 square with semicircles on each side --/
theorem area_of_inscribed_square (original_side : ℝ) (efgh : Square) : 
  original_side = 6 →
  efgh.sides_parallel_to original_side →
  efgh.sides_tangent_to_semicircles original_side →
  efgh.area = 144 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_inscribed_square_l554_55434


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_parallel_and_angle_l554_55444

theorem vector_parallel_and_angle (θ x : Real) : 
  0 < θ ∧ θ < Real.pi / 2 →
  0 < x ∧ x < Real.pi / 2 →
  (Real.sin θ, 1) = (Real.cos θ * k, Real.sqrt 3 * k) →
  Real.sin (x - θ) = 3 / 5 →
  θ = Real.pi / 6 ∧ Real.cos x = (4 * Real.sqrt 3 - 3) / 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_parallel_and_angle_l554_55444


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_conic_sections_equations_l554_55443

-- Define the ellipse
def ellipse (a b : ℝ) (x y : ℝ) : Prop :=
  x^2 / b^2 + y^2 / a^2 = 1

-- Define the hyperbola
def hyperbola (l : ℝ) (x y : ℝ) : Prop :=
  -y^2 / l + x^2 / l = 1

-- Theorem statement
theorem conic_sections_equations :
  ∃ (a b l : ℝ),
    -- Ellipse conditions
    (∀ x y, ellipse a b x y → (x = 0 ∧ y = 2) ∨ (x = 1 ∧ y = 0)) ∧
    (a > b) ∧ (b > 0) ∧
    -- Hyperbola conditions
    (∀ x y, hyperbola l x y → (x = 0 → y = 6 ∨ y = -6)) ∧
    (l ≠ 0) ∧
    -- Equations
    (a = 2 ∧ b = 2) ∧
    (l = -12) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_conic_sections_equations_l554_55443


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_even_function_phi_l554_55412

noncomputable def f (x φ : ℝ) : ℝ := 2 * Real.sin (2 * x + φ - Real.pi / 6)

theorem even_function_phi (φ : ℝ) (h1 : 0 < φ) (h2 : φ < Real.pi) :
  (∀ x, f x φ = f (-x) φ) → φ = 2 * Real.pi / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_even_function_phi_l554_55412


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_solution_l554_55453

noncomputable section

open Real

def angle_at_vertex (triangle : Set (ℝ × ℝ)) (vertex : ℝ × ℝ) : ℝ := sorry

theorem isosceles_triangle_solution (x : ℝ) : 
  (∃ (triangle : Set (ℝ × ℝ)),
    -- The triangle is isosceles
    (∃ (a b : ℝ × ℝ), a ∈ triangle ∧ b ∈ triangle ∧ 
      dist a b = tan x ∧ 
      (∃ (c : ℝ × ℝ), c ∈ triangle ∧ 
        dist a c = tan x ∧ 
        dist b c = tan (5 * x))) ∧
    -- The vertex angle is 4x
    (∃ (vertex : ℝ × ℝ), vertex ∈ triangle ∧ 
      angle_at_vertex triangle vertex = 4 * x * (π / 180)) ∧
    -- x is measured in degrees
    0 < x ∧ x < 90) →
  x = 20 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_solution_l554_55453


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_theorem_l554_55462

/-- The polynomial we're dividing -/
def f (x : ℝ) : ℝ := 8 * x^3 - 22 * x^2 + 30 * x - 45

/-- The divisor polynomial -/
def g (x : ℝ) : ℝ := 4 * x - 8

theorem remainder_theorem :
  ∃ (q : ℝ → ℝ), ∀ x, f x = g x * q x + (-9) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_theorem_l554_55462


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rush_hour_equation_quiet_day_equation_return_equation_distance_to_school_l554_55407

/-- Represents the distance from Jeremy's house to school in miles -/
noncomputable def distance : ℝ := sorry

/-- Represents the normal speed during rush hour in miles per hour -/
noncomputable def normal_speed : ℝ := sorry

/-- The time taken to drive to school in rush hour traffic (in hours) -/
noncomputable def rush_hour_time : ℝ := 30 / 60

/-- The time taken to drive to school on a quiet day (in hours) -/
noncomputable def quiet_day_time : ℝ := 15 / 60

/-- The time taken to drive back from school (in hours) -/
noncomputable def return_time : ℝ := 40 / 60

/-- States that the distance equals the normal speed multiplied by the rush hour time -/
theorem rush_hour_equation : distance = normal_speed * rush_hour_time := by sorry

/-- States that the distance equals the increased speed multiplied by the quiet day time -/
theorem quiet_day_equation : distance = (normal_speed + 25) * quiet_day_time := by sorry

/-- States that the distance equals the decreased speed multiplied by the return time -/
theorem return_equation : distance = (normal_speed - 10) * return_time := by sorry

/-- Proves that the distance from Jeremy's house to school is 12.5 miles -/
theorem distance_to_school : distance = 12.5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rush_hour_equation_quiet_day_equation_return_equation_distance_to_school_l554_55407


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_find_y_proof_l554_55449

def find_y (y : ℝ) : Prop :=
  (55 + 48 + 507 + 2 + 684 + y) / 6 = 223

/-- The proof of the theorem -/
theorem find_y_proof : ∃ y, find_y y ∧ y = 42 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_find_y_proof_l554_55449


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_T_max_at_6_l554_55478

/-- A geometric sequence with first term 30 and satisfying 8S_6 = 9S_3 -/
def GeometricSequence : Type :=
  {a : ℕ → ℝ // a 1 = 30 ∧ ∃ q : ℝ, q ≠ 1 ∧
    (∀ n, a n = 30 * q^(n-1)) ∧
    8 * (30 * (q^6 - 1) / (q - 1)) = 9 * (30 * (q^3 - 1) / (q - 1))}

/-- The product of the first n terms of a geometric sequence -/
def T (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  (Finset.range n).prod (fun i => a (i + 1))

/-- The theorem stating that T_n is maximized when n = 6 for the given geometric sequence -/
theorem T_max_at_6 (a : GeometricSequence) :
  ∃ (n : ℕ), (∀ (m : ℕ), T a.val n ≥ T a.val m) ∧ n = 6 :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_T_max_at_6_l554_55478


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mung_beans_import_range_l554_55400

noncomputable def initial_price : ℝ := 16
noncomputable def price_decrease_per_100_tons : ℝ := 1
noncomputable def min_target_price : ℝ := 8
noncomputable def max_target_price : ℝ := 10

noncomputable def price_after_import (tons_imported : ℝ) : ℝ :=
  initial_price - (tons_imported / 100) * price_decrease_per_100_tons

theorem mung_beans_import_range :
  ∃ (min_tons max_tons : ℝ),
    min_tons = 600 ∧ 
    max_tons = 800 ∧
    (∀ tons, min_tons ≤ tons ∧ tons ≤ max_tons →
      min_target_price ≤ price_after_import tons ∧ 
      price_after_import tons ≤ max_target_price) ∧
    (∀ tons, tons < min_tons ∨ tons > max_tons →
      price_after_import tons < min_target_price ∨ 
      price_after_import tons > max_target_price) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mung_beans_import_range_l554_55400


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_identities_l554_55431

theorem trig_identities (α : Real) (h : Real.sin α = 2 * Real.cos α) :
  (2 * Real.sin α - Real.cos α) / (Real.sin α + 2 * Real.cos α) = 3 / 4 ∧
  Real.sin α ^ 2 + Real.sin α * Real.cos α - 2 * Real.cos α ^ 2 = 4 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_identities_l554_55431


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_volume_ratio_a_to_c_l554_55485

/-- The volume of a cone given its radius and height -/
noncomputable def cone_volume (r h : ℝ) : ℝ := (1/3) * Real.pi * r^2 * h

/-- The ratio of volumes of two cones -/
noncomputable def cone_volume_ratio (r1 h1 r2 h2 : ℝ) : ℝ :=
  (cone_volume r1 h1) / (cone_volume r2 h2)

/-- Theorem: The ratio of the volume of cone A to cone C is 8/27 -/
theorem cone_volume_ratio_a_to_c :
  cone_volume_ratio 10 20 15 30 = 8/27 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_volume_ratio_a_to_c_l554_55485


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_crosses_asymptote_at_15_7_l554_55439

/-- The function g(x) -/
noncomputable def g (x : ℝ) : ℝ := (3 * x^2 - 8 * x - 9) / (x^2 - 5 * x + 2)

/-- The horizontal asymptote of g(x) -/
def horizontal_asymptote : ℝ := 3

theorem g_crosses_asymptote_at_15_7 :
  ∃ (x : ℝ), x = 15 / 7 ∧ g x = horizontal_asymptote := by
  use 15 / 7
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_crosses_asymptote_at_15_7_l554_55439


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_k_multiple_of_360_l554_55465

theorem smallest_k_multiple_of_360 : 
  (∃ k : ℕ, k > 0 ∧ (k * (k + 1) * (2 * k + 1)) / 6 % 360 = 0 ∧ 
    ∀ m : ℕ, m > 0 → m < k → (m * (m + 1) * (2 * m + 1)) / 6 % 360 ≠ 0) → 
  (∃ k : ℕ, k = 360 ∧ (k * (k + 1) * (2 * k + 1)) / 6 % 360 = 0 ∧ 
    ∀ m : ℕ, m > 0 → m < k → (m * (m + 1) * (2 * m + 1)) / 6 % 360 ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_k_multiple_of_360_l554_55465


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hunter_winning_strategy_l554_55419

/-- Represents a cell on the infinite square grid -/
structure Cell where
  x : ℤ
  y : ℤ

/-- Represents a color in the hunter's coloring scheme -/
inductive Color
  | C1 : Fin 3 → Color
  | C2 : Fin 3 → Color
  | C3 : Bool → Color
  | C4 : Bool → Color
  | C5 : Bool → Color

/-- The hunter's coloring function -/
def coloring : Cell → Color := sorry

/-- Checks if two cells are adjacent -/
def isAdjacent (c1 c2 : Cell) : Prop := sorry

/-- Represents a valid path for the rabbit -/
def isValidPath (path : List Cell) : Prop := sorry

/-- Represents the sequence of colors observed by the hunter -/
def observedColors (path : List Cell) : List Color := sorry

/-- The hunter's strategy to determine the starting cell or if the rabbit is stuck -/
def hunterStrategy (colors : List Color) : Option Cell := sorry

/-- The main theorem: The hunter has a winning strategy -/
theorem hunter_winning_strategy :
  ∃ (coloring : Cell → Color),
    ∀ (path : List Cell),
      isValidPath path →
        (∃ (start : Cell), hunterStrategy (observedColors path) = some start ∧ start ∈ path) ∨
        hunterStrategy (observedColors path) = none :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hunter_winning_strategy_l554_55419


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_not_invertible_and_zero_solution_l554_55415

def A : Matrix (Fin 2) (Fin 2) ℝ := !![5, 10; -15, -30]

theorem matrix_not_invertible_and_zero_solution :
  ¬(IsUnit A) ∧ (0 : Matrix (Fin 2) (Fin 2) ℝ) = !![0, 0; 0, 0] := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_not_invertible_and_zero_solution_l554_55415


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_determinant_l554_55489

/-- The problem setup and proof statement -/
theorem matrix_determinant : 
  let v : Fin 3 → ℝ := ![3, 2, -2]
  let w : Fin 3 → ℝ := ![-1, 1, 4]
  let u_dir : Fin 3 → ℝ := ![-1, 1, 0]
  let u : Fin 3 → ℝ := (1 / Real.sqrt 2) • u_dir
  let A : Matrix (Fin 3) (Fin 3) ℝ := Matrix.of ![u, v, w]
  Matrix.det A = -4 * Real.sqrt 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_determinant_l554_55489


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_is_positive_reals_l554_55441

open Set
open Function
open Real

-- Define the function f and its derivative f'
variable (f : ℝ → ℝ)
variable (f' : ℝ → ℝ)

-- Define the conditions
variable (h1 : ∀ x, HasDerivAt f (f' x) x)
variable (h2 : ∀ x, f x + f' x > 1)
variable (h3 : f 0 = 2018)

-- Define the solution set
def solution_set (f : ℝ → ℝ) := {x : ℝ | f x > 2017 / Real.exp x + 1}

-- State the theorem
theorem solution_set_is_positive_reals (f : ℝ → ℝ) :
  solution_set f = Ioi 0 :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_is_positive_reals_l554_55441


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l554_55487

-- Define the functions f and g
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (a^x - 1) / (a^x + 1)
noncomputable def g (x : ℝ) : ℝ := 3^x

-- Theorem statement
theorem function_properties (a : ℝ) (h1 : a > 1) (h2 : g (a + 2) = 81) :
  (a = 2 ∧ ∀ x, f a (-x) = -(f a x)) ∧
  (∀ x y, x < y → f a x < f a y) ∧
  (∀ y, y ∈ Set.range (f a) ↔ -1 < y ∧ y < 1) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l554_55487


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ratio_product_equals_one_l554_55479

-- Define points A, B, C, D, and I
variable (A B C D I : EuclideanSpace ℝ (Fin 2))

-- Define the ratios
noncomputable def ratio_BC_AD (A B C D : EuclideanSpace ℝ (Fin 2)) : ℝ := 
  dist B C / dist A D

noncomputable def ratio_AI_BI (A B I : EuclideanSpace ℝ (Fin 2)) : ℝ := 
  dist A I / dist B I

noncomputable def ratio_DI_CI (C D I : EuclideanSpace ℝ (Fin 2)) : ℝ := 
  dist D I / dist C I

-- State the theorem
theorem ratio_product_equals_one 
  (A B C D I : EuclideanSpace ℝ (Fin 2)) : 
  ratio_BC_AD A B C D * ratio_AI_BI A B I * ratio_DI_CI C D I = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ratio_product_equals_one_l554_55479


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_janet_current_rate_l554_55467

/-- Represents Janet's work and financial situation --/
structure JanetFinances where
  weekly_hours : ℕ
  freelance_hourly_rate : ℚ
  extra_weekly_fica : ℚ
  monthly_healthcare : ℚ
  monthly_income_difference : ℚ

/-- Calculates Janet's current hourly rate --/
noncomputable def current_hourly_rate (j : JanetFinances) : ℚ :=
  j.freelance_hourly_rate - (j.monthly_income_difference + 4 * j.extra_weekly_fica + j.monthly_healthcare) / (4 * j.weekly_hours)

/-- Theorem stating Janet's current hourly rate is $30 --/
theorem janet_current_rate (j : JanetFinances) 
    (h1 : j.weekly_hours = 40)
    (h2 : j.freelance_hourly_rate = 40)
    (h3 : j.extra_weekly_fica = 25)
    (h4 : j.monthly_healthcare = 400)
    (h5 : j.monthly_income_difference = 1100) : 
  current_hourly_rate j = 30 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_janet_current_rate_l554_55467


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_l554_55428

noncomputable def m (x : ℝ) : ℝ × ℝ := (2 * Real.cos x + 2 * Real.sqrt 3 * Real.sin x, 1)

noncomputable def n (x y : ℝ) : ℝ × ℝ := (Real.cos x, -y)

theorem triangle_area (x A : ℝ) (a b c : ℝ) :
  (m x).1 * (n x (1 + 2 * Real.sin (A + π/6))).1 + (m x).2 * (n x (1 + 2 * Real.sin (A + π/6))).2 = 0 →
  1 + 2 * Real.sin (A + π/6) = 3 →
  a = 2 →
  b + c = 4 →
  (1/2) * b * c * Real.sin A = Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_l554_55428


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_salesman_sales_l554_55435

/-- Represents the salesman's sales in Rupees -/
def S : ℝ := sorry

/-- Old commission rate -/
def old_rate : ℝ := 0.05

/-- New commission rate -/
def new_rate : ℝ := 0.025

/-- Fixed salary in new scheme -/
def fixed_salary : ℝ := 1300

/-- Sales threshold for new commission -/
def threshold : ℝ := 4000

/-- Difference between new and old scheme remuneration -/
def remuneration_difference : ℝ := 600

/-- Theorem stating the salesman's sales equal 24000 Rupees -/
theorem salesman_sales : 
  fixed_salary + new_rate * (S - threshold) = old_rate * S + remuneration_difference →
  S = 24000 := by
  intro h
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_salesman_sales_l554_55435


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dot_product_range_l554_55499

-- Define the triangle ABC
def Triangle (A B C : ℝ × ℝ) : Prop :=
  A.1^2 + A.2^2 = 4 ∧ B.1^2 + B.2^2 = 4 ∧ C = (0, 0)

-- Define points M and N on AB
def PointsOnAB (A B M N : ℝ × ℝ) : Prop :=
  ∃ t s : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ 0 ≤ s ∧ s ≤ 1 ∧
  M = (t * A.1 + (1 - t) * B.1, t * A.2 + (1 - t) * B.2) ∧
  N = (s * A.1 + (1 - s) * B.1, s * A.2 + (1 - s) * B.2)

-- Define the distance between M and N
noncomputable def DistanceMN (M N : ℝ × ℝ) : ℝ :=
  Real.sqrt ((M.1 - N.1)^2 + (M.2 - N.2)^2)

-- Define the dot product of CM and CN
def DotProductCMCN (C M N : ℝ × ℝ) : ℝ :=
  (M.1 - C.1) * (N.1 - C.1) + (M.2 - C.2) * (N.2 - C.2)

-- Theorem statement
theorem dot_product_range (A B C M N : ℝ × ℝ) :
  Triangle A B C →
  PointsOnAB A B M N →
  DistanceMN M N = Real.sqrt 2 →
  3/2 ≤ DotProductCMCN C M N ∧ DotProductCMCN C M N ≤ 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_dot_product_range_l554_55499


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_for_decreasing_f_l554_55460

-- Define the function
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log (2 - a * x) / Real.log a

-- State the theorem
theorem range_of_a_for_decreasing_f :
  ∀ a : ℝ, a > 0 →
  (∀ x₁ x₂ : ℝ, -1 < x₁ ∧ x₁ < x₂ ∧ x₂ < 1 → f a x₁ > f a x₂) →
  1 < a ∧ a ≤ 2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_for_decreasing_f_l554_55460


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_reading_comparison_l554_55404

/-- Given the following conditions:
  - Summer break is 80 days long
  - DeShaun read 60 books over the summer
  - Each book DeShaun read averaged 320 pages long
  - The second person read on average 180 pages each day of break
  Prove that the ratio of pages read by the second person to pages read by DeShaun is 3/4 -/
theorem reading_comparison (summer_days : ℕ) (deshaun_books : ℕ) (deshaun_pages_per_book : ℕ) (second_person_pages_per_day : ℕ)
  (h1 : summer_days = 80)
  (h2 : deshaun_books = 60)
  (h3 : deshaun_pages_per_book = 320)
  (h4 : second_person_pages_per_day = 180) :
  (second_person_pages_per_day * summer_days : ℚ) / (deshaun_books * deshaun_pages_per_book) = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_reading_comparison_l554_55404


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ball_prices_and_max_basketballs_l554_55491

/-- Represents the prices and quantities of basketballs and soccer balls --/
structure BallPrices where
  basketball_price : ℝ
  soccer_ball_price : ℝ
  basketball_quantity : ℕ
  soccer_ball_quantity : ℕ

/-- Calculates the total cost of balls --/
def total_cost (prices : BallPrices) : ℝ :=
  prices.basketball_price * prices.basketball_quantity +
  prices.soccer_ball_price * prices.soccer_ball_quantity

/-- Calculates the discounted cost of balls --/
def discounted_cost (prices : BallPrices) (basketball_discount soccer_discount : ℝ) : ℝ :=
  (1 - basketball_discount) * prices.basketball_price * prices.basketball_quantity +
  (1 - soccer_discount) * prices.soccer_ball_price * prices.soccer_ball_quantity

/-- Theorem stating the properties of ball prices and maximum basketballs that can be purchased --/
theorem ball_prices_and_max_basketballs :
  ∃ (prices : BallPrices),
    prices.basketball_price = 150 ∧
    prices.soccer_ball_price = 100 ∧
    prices.basketball_price - prices.soccer_ball_price = 50 ∧
    total_cost { basketball_price := prices.basketball_price,
                 soccer_ball_price := prices.soccer_ball_price,
                 basketball_quantity := 6,
                 soccer_ball_quantity := 8 } = 1700 ∧
    (∀ (b : ℕ),
      b ≤ 10 ∧
      discounted_cost { basketball_price := prices.basketball_price,
                        soccer_ball_price := prices.soccer_ball_price,
                        basketball_quantity := b,
                        soccer_ball_quantity := 10 - b }
                      0.1 0.15 ≤ 1150 →
      b ≤ 6) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ball_prices_and_max_basketballs_l554_55491


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lambda_range_l554_55492

theorem lambda_range (l : ℝ) : (∀ a b : ℝ, a^2 + 8*b^2 ≥ l*b*(a+b)) → -8 ≤ l ∧ l ≤ 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lambda_range_l554_55492


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sticker_distribution_probability_l554_55475

/-- The number of sticker types -/
def num_types : ℕ := 4

/-- The number of students -/
def num_students : ℕ := 4

/-- The total number of stickers -/
def total_stickers : ℕ := num_types * num_students

/-- The number of stickers each student receives -/
def stickers_per_student : ℕ := num_types

/-- The probability of each student receiving one sticker of each type -/
def probability : ℚ := 32 / 50050

theorem sticker_distribution_probability :
  (Nat.choose total_stickers stickers_per_student *
   Nat.choose (total_stickers - stickers_per_student) stickers_per_student *
   (Nat.factorial num_types) ^ 2) /
  (Nat.factorial total_stickers : ℚ) = probability := by
  sorry

#eval num_types + num_students

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sticker_distribution_probability_l554_55475


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_lines_theorem_PQ_length_theorem_l554_55414

-- Define the circle C
def C : Set (ℝ × ℝ) := {p | (p.1 - 3)^2 + (p.2 - 4)^2 = 4}

-- Define point A
def A : ℝ × ℝ := (1, 0)

-- Define the set of lines passing through A
def lines_through_A : Set (Set (ℝ × ℝ)) :=
  {l | ∃ k, l = {p | p.2 = k * (p.1 - A.1)} ∨ l = {p | p.1 = A.1}}

-- Tangent lines theorem
theorem tangent_lines_theorem :
  ∃ l₁ l₂, l₁ ∈ lines_through_A ∧ l₂ ∈ lines_through_A ∧
    (∀ p ∈ l₁, p ∈ C → (∀ q ∈ l₁, q ∈ C → p = q)) ∧
    (∀ p ∈ l₂, p ∈ C → (∀ q ∈ l₂, q ∈ C → p = q)) ∧
    l₁ = {p | p.1 = 1} ∧
    l₂ = {p | 3 * p.1 - 4 * p.2 - 3 = 0} := by sorry

-- Define the line with 45° slope angle passing through A
def l_45 : Set (ℝ × ℝ) := {p | p.2 = p.1 - 1}

-- Intersection points
noncomputable def P : ℝ × ℝ := sorry
noncomputable def Q : ℝ × ℝ := sorry

-- PQ length theorem
theorem PQ_length_theorem :
  P ∈ C ∧ Q ∈ C ∧ P ∈ l_45 ∧ Q ∈ l_45 ∧ P ≠ Q →
  Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2) = 2 * Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_lines_theorem_PQ_length_theorem_l554_55414


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_zeros_less_than_two_l554_55430

noncomputable section

variable (a b c : ℝ) (f : ℝ → ℝ)

-- Define the function f
def f_def (a b c : ℝ) : ℝ → ℝ := λ x ↦ a * x^3 + b * x^2 + c * x

-- Define the conditions
axiom a_nonzero : a ≠ 0
axiom condition_1 : 6 * a + b = 0
axiom condition_2 : f 1 = 4 * a

-- Define the function y
def y (f : ℝ → ℝ) : ℝ → ℝ := λ x ↦ f x - x * Real.exp (-x)

-- Define the existence of three zeros
axiom zeros_exist : ∃ (x₁ x₂ x₃ : ℝ), 
  0 ≤ x₁ ∧ x₁ < x₂ ∧ x₂ < x₃ ∧ x₃ ≤ 3 ∧ 
  y f x₁ = 0 ∧ y f x₂ = 0 ∧ y f x₃ = 0

-- State the theorem
theorem sum_of_zeros_less_than_two :
  ∀ x₁ x₂ x₃, (0 ≤ x₁ ∧ x₁ < x₂ ∧ x₂ < x₃ ∧ x₃ ≤ 3 ∧ 
               y f x₁ = 0 ∧ y f x₂ = 0 ∧ y f x₃ = 0) →
              x₁ + x₂ + x₃ < 2 :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_zeros_less_than_two_l554_55430


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_bound_l554_55498

-- Define the polynomial f(x)
def f (a b c x : ℝ) : ℝ := x^3 + a*x^2 + b*x + c

-- State the theorem
theorem max_value_bound (a b c lambda : ℝ) (x1 x2 x3 : ℝ) 
  (h_lambda_pos : lambda > 0)
  (h_roots : ∀ x, f a b c x = 0 ↔ x = x1 ∨ x = x2 ∨ x = x3)
  (h_diff : x2 - x1 = lambda)
  (h_x3 : x3 > (x1 + x2) / 2) :
  (2 * a^3 + 27 * c - 9 * a * b) / lambda^3 ≤ 3 * Real.sqrt 3 / 2 :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_bound_l554_55498


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinitely_many_points_l554_55445

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- The distance squared between two points -/
def distanceSquared (p1 p2 : Point) : ℝ :=
  (p1.x - p2.x)^2 + (p1.y - p2.y)^2

/-- The unit circle centered at the origin -/
def unitCircle (p : Point) : Prop :=
  p.x^2 + p.y^2 ≤ 1

/-- The sum of squared distances from a point to (-1,0) and (1,0) -/
def sumSquaredDistances (p : Point) : ℝ :=
  distanceSquared p ⟨-1, 0⟩ + distanceSquared p ⟨1, 0⟩

/-- There are infinitely many points satisfying both conditions -/
theorem infinitely_many_points :
  ∃ (S : Set Point), (∀ p ∈ S, unitCircle p ∧ sumSquaredDistances p = 3) ∧ Set.Infinite S := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinitely_many_points_l554_55445


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_relationship_holds_for_all_points_l554_55494

/-- Represents a data point of myopia degree and focal length -/
structure DataPoint where
  y : ℝ  -- degree of myopia glasses
  x : ℝ  -- focal length of lenses

/-- The set of given data points -/
noncomputable def dataPoints : List DataPoint := [
  ⟨100, 1.00⟩,
  ⟨200, 0.50⟩,
  ⟨400, 0.25⟩,
  ⟨500, 0.20⟩
]

/-- The proposed functional relationship between y and x -/
noncomputable def relationship (x : ℝ) : ℝ := 100 / x

/-- Theorem stating that the relationship holds for all given data points -/
theorem relationship_holds_for_all_points : 
  ∀ point ∈ dataPoints, point.y = relationship point.x := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_relationship_holds_for_all_points_l554_55494


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_implies_a_value_l554_55461

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (2 : ℝ)^x + (2 : ℝ)^(-x) * Real.log a

-- State the theorem
theorem odd_function_implies_a_value (a : ℝ) :
  (∀ x : ℝ, f a x = -(f a (-x))) → a = (1 : ℝ) / 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_implies_a_value_l554_55461


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_z_location_l554_55471

noncomputable def complex_plane_quadrant (z : ℂ) : ℕ :=
  if z.re ≥ 0 ∧ z.im ≥ 0 then 1
  else if z.re < 0 ∧ z.im ≥ 0 then 2
  else if z.re < 0 ∧ z.im < 0 then 3
  else 4

theorem z_location (z : ℂ) (h : (1 + Complex.I) * z = 2 * Complex.I) :
  z = 1 + Complex.I ∧ complex_plane_quadrant z = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_z_location_l554_55471


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_grape_juice_mixture_result_l554_55468

/-- Given a mixture of grape juice and other liquids, this function calculates
    the percentage of grape juice in the resulting mixture after adding pure grape juice. -/
noncomputable def grape_juice_percentage (initial_volume : ℝ) (initial_percentage : ℝ) (added_volume : ℝ) : ℝ :=
  let initial_grape_juice := initial_volume * (initial_percentage / 100)
  let total_grape_juice := initial_grape_juice + added_volume
  let final_volume := initial_volume + added_volume
  (total_grape_juice / final_volume) * 100

/-- Theorem stating that given the specific conditions of the problem,
    the resulting mixture is 36% grape juice. -/
theorem grape_juice_mixture_result :
  grape_juice_percentage 40 20 10 = 36 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_grape_juice_mixture_result_l554_55468


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_answer_is_C_l554_55405

inductive AnswerChoice
  | A
  | B
  | C
  | D

def correctAnswer : AnswerChoice := AnswerChoice.C

theorem correct_answer_is_C : correctAnswer = AnswerChoice.C := by rfl

#check correct_answer_is_C

end NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_answer_is_C_l554_55405


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_equals_20_sqrt47_div_7_l554_55403

-- Define the circles and their properties
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the problem setup
def problem_setup (C_a C_b C_c : Circle) : Prop :=
  -- C_a and C_b are externally tangent
  ∃ p : ℝ × ℝ, (dist p C_a.center = C_a.radius) ∧ (dist p C_b.center = C_b.radius) ∧
  (dist C_a.center C_b.center = C_a.radius + C_b.radius) ∧
  -- C_a and C_b are internally tangent to C_c
  (dist C_a.center C_c.center = C_c.radius - C_a.radius) ∧
  (dist C_b.center C_c.center = C_c.radius - C_b.radius) ∧
  -- Radii of C_a and C_b
  C_a.radius = 5 ∧ C_b.radius = 12 ∧
  -- Centers are collinear
  ∃ t : ℝ, C_c.center = (1 - t) • C_a.center + t • C_b.center

-- Define the chord
noncomputable def chord_length (C_a C_b C_c : Circle) : ℝ :=
  let r := C_c.radius
  let d := dist C_a.center C_b.center
  2 * Real.sqrt (r^2 - ((r^2 - C_b.radius^2 + C_a.radius^2) / (2 * d))^2)

-- Theorem statement
theorem chord_length_equals_20_sqrt47_div_7 (C_a C_b C_c : Circle) :
  problem_setup C_a C_b C_c →
  chord_length C_a C_b C_c = 20 * Real.sqrt 47 / 7 :=
by
  intro h
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_equals_20_sqrt47_div_7_l554_55403


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_center_to_face_formula_l554_55433

/-- A regular tetrahedron with base side length a and right angle between lateral faces -/
structure RegularTetrahedron (a : ℝ) where
  base_side_length : a > 0
  lateral_face_angle : ℝ
  angle_is_right : lateral_face_angle = 90

/-- The distance from the center of the base to a lateral face in a regular tetrahedron -/
noncomputable def distance_center_to_face (t : RegularTetrahedron a) : ℝ :=
  a * Real.sqrt 2 / 2

/-- Theorem: The distance from the center of the base to a lateral face 
    in a regular tetrahedron with base side length a and right angle 
    between lateral faces is a√2/2 -/
theorem distance_center_to_face_formula (a : ℝ) (t : RegularTetrahedron a) :
  distance_center_to_face t = a * Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_center_to_face_formula_l554_55433


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_set_l554_55442

theorem inequality_solution_set (x : ℝ) : 
  x ∈ Set.Icc (-5 * Real.pi / 4) (3 * Real.pi / 4) →
  (Real.sin x) ^ 2018 + (Real.cos x) ^ (-2019 : ℤ) ≤ (Real.cos x) ^ 2018 + (Real.sin x) ^ (-2019 : ℤ) ↔
  x ∈ Set.Ico (-5 * Real.pi / 4) (-Real.pi) ∪ 
      Set.Ico (-3 * Real.pi / 4) (-Real.pi / 2) ∪ 
      Set.Ioc 0 (Real.pi / 4) ∪ 
      Set.Ioc (Real.pi / 2) (3 * Real.pi / 4) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_set_l554_55442


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_berry_gathering_time_l554_55423

/-- Represents the productivity of a child in terms of fraction of work done per hour -/
structure Productivity where
  rate : ℚ
  work_per_hour : rate > 0

/-- Represents the time taken to complete the work -/
def time_to_complete (p : ℚ) : ℚ := 1 / p

theorem berry_gathering_time 
  (masha dasha sasha : Productivity)
  (h1 : time_to_complete (masha.rate + dasha.rate) = 15/2)
  (h2 : time_to_complete (masha.rate + sasha.rate) = 6)
  (h3 : time_to_complete (dasha.rate + sasha.rate) = 5) :
  time_to_complete (masha.rate + dasha.rate + sasha.rate) = 4 := by
  sorry

#eval time_to_complete (1/4 : ℚ)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_berry_gathering_time_l554_55423


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_asymptote_sum_l554_55490

/-- Given a function g(x) = (x+3) / (x^2 + cx + d) with vertical asymptotes at x = 2 and x = -3,
    the sum of c and d is -5. -/
theorem asymptote_sum (c d : ℝ) : 
  (∀ x : ℝ, x ≠ 2 ∧ x ≠ -3 → 
    (fun x ↦ (x + 3) / (x^2 + c*x + d)) x = (fun x ↦ (x + 3) / ((x - 2)*(x + 3))) x) →
  c + d = -5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_asymptote_sum_l554_55490


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_A_and_B_l554_55438

-- Define the sets A and B
def A : Set ℝ := {y | ∃ x, y = -x^2 - 2*x}
def B : Set ℝ := {y | ∃ x, y = x + 1}

-- State the theorem
theorem intersection_of_A_and_B :
  A ∩ B = Set.Iic 1 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_A_and_B_l554_55438


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_transportation_is_fifteen_percent_l554_55455

/-- Represents the budget allocation for a company -/
structure BudgetAllocation where
  research_dev : ℚ
  utilities : ℚ
  equipment : ℚ
  supplies : ℚ
  salaries_degrees : ℚ

/-- Calculates the percentage of the budget spent on transportation -/
def transportation_percentage (budget : BudgetAllocation) : ℚ :=
  100 - (budget.research_dev + budget.utilities + budget.equipment + budget.supplies + 
         (budget.salaries_degrees / 360) * 100)

/-- Theorem stating that the transportation percentage is 15% for the given budget allocation -/
theorem transportation_is_fifteen_percent (budget : BudgetAllocation) 
  (h1 : budget.research_dev = 9)
  (h2 : budget.utilities = 5)
  (h3 : budget.equipment = 4)
  (h4 : budget.supplies = 2)
  (h5 : budget.salaries_degrees = 234) :
  transportation_percentage budget = 15 := by
  sorry

#eval transportation_percentage { research_dev := 9, utilities := 5, equipment := 4, supplies := 2, salaries_degrees := 234 }

end NUMINAMATH_CALUDE_ERRORFEEDBACK_transportation_is_fifteen_percent_l554_55455


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_value_l554_55402

-- Define the function f as noncomputable
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 0 then 2^x else a * Real.sqrt x

-- State the theorem
theorem a_value (a : ℝ) : f a (-1) + f a 1 = 1 → a = 1/2 := by
  intro h
  -- The proof steps would go here, but we'll use sorry for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_value_l554_55402


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_and_range_l554_55484

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * Real.log x + x^2

theorem max_value_and_range (a : ℝ) :
  -- Part 1: Maximum value when a = -4
  (∀ x ∈ Set.Icc 1 (Real.exp 1), f (-4) x ≤ f (-4) (Real.exp 1)) ∧
  (f (-4) (Real.exp 1) = (Real.exp 1)^2 - 4) ∧
  -- Part 2: Range of a for non-negative f
  (∀ x ∈ Set.Ioo 1 (Real.exp 1), f a x ≥ 0) ↔ a ≥ -2 * (Real.exp 1) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_and_range_l554_55484


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_F_l554_55410

noncomputable def F (x y : ℝ) : ℝ := (x + 1) / y

theorem min_value_F :
  ∀ x y : ℝ, x^2 + y^2 - 2*x - 2*y + 1 = 0 → 
  ∃ m : ℝ, m = 3/4 ∧ ∀ a b : ℝ, a^2 + b^2 - 2*a - 2*b + 1 = 0 → F a b ≥ m :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_F_l554_55410


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_and_S_closed_form_l554_55418

/-- Sequence a_n defined recursively -/
def a : ℕ → ℕ
  | 0 => 3  -- Changed from 1 to 0 for zero-based indexing
  | n + 1 => a n + 2^(n + 1)

/-- Partial sum S_n of the sequence a_n -/
def S : ℕ → ℕ
  | 0 => 3  -- S_0 is a_0
  | n + 1 => S n + a (n + 1)

/-- The main theorem stating the closed forms for a_n and S_n -/
theorem a_and_S_closed_form : 
  (∀ n : ℕ, a n = (n + 3) * 2^n) ∧ 
  (∀ n : ℕ, S n = (n + 2) * 2^(n + 1)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_and_S_closed_form_l554_55418


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_ratio_triangle_hexagon_l554_55473

theorem area_ratio_triangle_hexagon (R : ℝ) (R_pos : R > 0) : 
  (4 * Real.sqrt 3 * R^2) / 3 / ((3 * Real.sqrt 3 * R^2) / 2) = 8 / 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_ratio_triangle_hexagon_l554_55473


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_petya_wins_l554_55464

-- Define the initial stick length
noncomputable def initial_length : ℝ := 10

-- Define Petya's first move
noncomputable def petya_first_move (l : ℝ) : ℝ × ℝ := (l/2, l/2)

-- Define a function to check if three lengths can form a triangle
def can_form_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

-- Theorem statement
theorem petya_wins (l : ℝ) (h : l = initial_length) :
  let (p1, p2) := petya_first_move l
  ∀ (a b : ℝ), a + b = p1 →
    ∀ (c d : ℝ), c + d = b →
      ¬(can_form_triangle p2 a c ∨ 
        can_form_triangle p2 a d ∨ 
        can_form_triangle p2 c d ∨ 
        can_form_triangle a c d) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_petya_wins_l554_55464


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_non_right_triangle_with_sum_of_squares_equal_diameter_squared_l554_55422

/-- A triangle with sides a, b, c and circumradius r -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  r : ℝ
  ha : a > 0
  hb : b > 0
  hc : c > 0
  hr : r > 0

/-- The angle C of the triangle -/
noncomputable def angle_C (t : Triangle) : ℝ := Real.arccos ((t.a^2 + t.b^2 - t.c^2) / (2 * t.a * t.b))

theorem exists_non_right_triangle_with_sum_of_squares_equal_diameter_squared :
  ∃ (t : Triangle), t.a^2 + t.b^2 = (2 * t.r)^2 ∧ angle_C t ≠ Real.pi / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_non_right_triangle_with_sum_of_squares_equal_diameter_squared_l554_55422


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fishing_solution_l554_55411

/-- Represents the number of fish caught in each round -/
structure FishCatch where
  first : ℕ
  second : ℕ
  third : ℕ

/-- The conditions of the fishing problem -/
def fishing_conditions (c : FishCatch) : Prop :=
  c.first = 8 ∧
  c.second > c.first ∧
  c.third = (c.second + (c.second * 6 / 10 : ℕ)) ∧
  c.first + c.second + c.third = 60

/-- The theorem stating the solution to the fishing problem -/
theorem fishing_solution (c : FishCatch) 
  (h : fishing_conditions c) : 
  c.second - c.first = 12 := by
  sorry

#check fishing_solution

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fishing_solution_l554_55411


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_N_satisfies_conditions_l554_55486

theorem matrix_N_satisfies_conditions : ∃ (N : Matrix (Fin 3) (Fin 3) ℝ),
  N = !![3, 0, 0.5; 1, 0, -0.5; 2, 1, 1.5] ∧
  N.mulVec ![1, 3, -1] = ![-7, 5, 3] ∧
  N.mulVec ![2, -2, 1] = ![4, -3, 0] ∧
  N.mulVec ![0, 1, 2] = ![1, -1, 4] := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_N_satisfies_conditions_l554_55486


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gcd_polynomial_and_b_l554_55463

theorem gcd_polynomial_and_b (b : ℤ) (h : ∃ k : ℤ, b = 780 * k) :
  Nat.gcd (Int.natAbs (5 * b^4 + 2 * b^3 + 6 * b^2 + 5 * b + 65)) b.natAbs = 65 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_gcd_polynomial_and_b_l554_55463


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gcd_problem_l554_55457

theorem gcd_problem (a : ℤ) (h : ∃ k : ℤ, a = 1836 * k) :
  Int.gcd (2 * a^2 + 11 * a + 40) (a + 4) = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_gcd_problem_l554_55457


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_incircle_function_theorem_l554_55448

/-- Given a triangle ABC with incircle touching side AB at D, prove that for the function 
    f(x) = x^2 - cx + ab sin^2(C/2), both f(AD) and f(BD) equal 0. -/
theorem incircle_function_theorem (a b c : ℝ) (C : ℝ) (AD BD : ℝ) 
    (h1 : AD + BD = c)
    (h2 : AD - BD = b - a) : 
  let f := λ x => x^2 - c*x + a*b*(Real.sin (C/2))^2
  f AD = 0 ∧ f BD = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_incircle_function_theorem_l554_55448


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_balance_theorem_l554_55483

/-- Represents the weight of a ball of a specific color -/
structure BallWeight where
  weight : ℚ

/-- The weight of a green ball -/
noncomputable def green : BallWeight := ⟨1⟩

/-- The weight of a blue ball -/
noncomputable def blue : BallWeight := ⟨1⟩

/-- The weight of a yellow ball -/
noncomputable def yellow : BallWeight := ⟨1⟩

/-- The weight of a white ball -/
noncomputable def white : BallWeight := ⟨1⟩

instance : HMul ℕ BallWeight BallWeight where
  hMul n b := ⟨n * b.weight⟩

instance : HAdd BallWeight BallWeight BallWeight where
  hAdd b1 b2 := ⟨b1.weight + b2.weight⟩

/-- 4 green balls balance 8 blue balls -/
axiom green_blue_balance : (4 : ℕ) * green = (8 : ℕ) * blue

/-- 3 yellow balls balance 6 blue balls -/
axiom yellow_blue_balance : (3 : ℕ) * yellow = (6 : ℕ) * blue

/-- 8 blue balls balance 6 white balls -/
axiom blue_white_balance : (8 : ℕ) * blue = (6 : ℕ) * white

/-- Theorem: 64/3 blue balls balance 5 green, 3 yellow, and 4 white balls -/
theorem balance_theorem : 
  (64/3 : ℚ) * blue.weight = ((5 : ℕ) * green + (3 : ℕ) * yellow + (4 : ℕ) * white).weight := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_balance_theorem_l554_55483


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_focus_determines_m_l554_55472

/-- Represents an ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_pos_a : 0 < a
  h_pos_b : 0 < b

/-- The distance from the center to a focus of an ellipse -/
noncomputable def Ellipse.focalDistance (e : Ellipse) : ℝ :=
  Real.sqrt (e.a^2 - e.b^2)

theorem ellipse_focus_determines_m (m : ℝ) (h_pos_m : 0 < m) :
  let e := Ellipse.mk 5 m (by norm_num) h_pos_m
  e.focalDistance = 4 → m = 3 := by
  intro h
  sorry

#check ellipse_focus_determines_m

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_focus_determines_m_l554_55472


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parametric_to_general_l554_55446

-- Define the parameter θ
variable (θ : ℝ)

-- Define the parametric equations
noncomputable def x (θ : ℝ) : ℝ := 2 + Real.sin θ ^ 2
noncomputable def y (θ : ℝ) : ℝ := -1 + 2 * Real.cos θ ^ 2

-- State the theorem
theorem parametric_to_general :
  ∀ θ, 2 * x θ + y θ - 5 = 0 ∧ 2 ≤ x θ ∧ x θ ≤ 3 :=
by
  intro θ
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parametric_to_general_l554_55446


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_2alpha_value_l554_55436

theorem sin_2alpha_value (α : ℝ) 
  (h1 : Real.cos (5 * Real.pi / 2 + α) = 3 / 5)
  (h2 : -Real.pi / 2 < α)
  (h3 : α < 0) : 
  Real.sin (2 * α) = -24 / 25 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_2alpha_value_l554_55436


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_proportion_constant_l554_55458

-- Define the inverse proportion function
noncomputable def inverse_proportion (k : ℝ) (x : ℝ) : ℝ := k / x

-- Define the theorem
theorem inverse_proportion_constant (k : ℝ) :
  k ≠ 0 →
  inverse_proportion k 1 = 3 →
  k = 3 :=
by
  intros h1 h2
  unfold inverse_proportion at h2
  simp at h2
  exact h2

-- The proof is completed without using sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_proportion_constant_l554_55458


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_calculation_l554_55495

-- Define the train's speed in km/hr
noncomputable def train_speed : ℝ := 90

-- Define the time taken to cross the platform in minutes
noncomputable def crossing_time : ℝ := 1

-- Define the function to convert km/hr to m/s
noncomputable def km_hr_to_m_s (speed : ℝ) : ℝ := speed * (1000 / 3600)

-- Define the function to calculate distance traveled
noncomputable def distance_traveled (speed : ℝ) (time : ℝ) : ℝ := speed * time

-- Theorem statement
theorem train_length_calculation :
  let speed_m_s := km_hr_to_m_s train_speed
  let distance := distance_traveled speed_m_s (crossing_time * 60)
  let train_length := distance / 2
  train_length = 750 := by
    -- The proof steps would go here, but we'll use 'sorry' to skip the proof
    sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_calculation_l554_55495


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tournament_players_count_l554_55454

theorem tournament_players_count :
  ∃ (n : ℕ),
  let total_players := n + 10
  let total_games := total_players * (total_players - 1) / 2
  let top_n_points := n * (n - 1) / 2
  let bottom_10_points := 45
  (2 * top_n_points + 2 * bottom_10_points = total_games) ∧
  (n - 1 > 9) ∧
  (total_players = 25) :=
by
  -- The proof goes here
  sorry

#check tournament_players_count

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tournament_players_count_l554_55454


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_number_line_problem_l554_55481

-- Define the points on the number line
def A : ℝ := 3
def B : ℝ := -5

-- Define the conditions
theorem number_line_problem :
  (|A - 3| + |B + 5| = 0) →
  (∃ M₁ M₂ : ℝ, (M₁ = -8 ∧ M₂ = 6) ∧ 
    (|A - M₁| + |B - M₁| = 7/4 * |A - B|) ∧
    (|A - M₂| + |B - M₂| = 7/4 * |A - B|)) ∧
  (∀ P : ℝ, |A - P| + |B - P| ≥ 8) ∧
  (∀ P : ℝ, |((A + B)/2) - P| - |P| ≥ -1) ∧
  (∃ min_P : ℝ, ∀ P : ℝ, 
    |A - min_P| + |B - min_P| + |((A + B)/2) - min_P| - |min_P| ≤
    |A - P| + |B - P| + |((A + B)/2) - P| - |P|) ∧
  (∀ P : ℝ, 
    (∀ Q : ℝ, |A - P| + |B - P| + |((A + B)/2) - P| - |P| ≤
              |A - Q| + |B - Q| + |((A + B)/2) - Q| - |Q|) →
    -5 ≤ P ∧ P ≤ -1) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_number_line_problem_l554_55481


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_head_start_is_64_meters_l554_55496

/-- Represents the race scenario where A runs twice as fast as B -/
structure RaceScenario where
  course_length : ℝ
  speed_ratio : ℝ
  head_start : ℝ

/-- Calculates the time taken by runner A to complete the race -/
noncomputable def time_A (scenario : RaceScenario) : ℝ :=
  scenario.course_length / (scenario.speed_ratio * 1)

/-- Calculates the time taken by runner B to complete the race -/
noncomputable def time_B (scenario : RaceScenario) : ℝ :=
  (scenario.course_length - scenario.head_start) / 1

/-- Theorem stating that the head start must be 64 meters 
    given the conditions of the race -/
theorem head_start_is_64_meters (scenario : RaceScenario) 
  (h1 : scenario.course_length = 128)
  (h2 : scenario.speed_ratio = 2)
  (h3 : time_A scenario = time_B scenario) :
  scenario.head_start = 64 := by
  sorry

#check head_start_is_64_meters

end NUMINAMATH_CALUDE_ERRORFEEDBACK_head_start_is_64_meters_l554_55496


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mass_percentage_Ca_in_CaI2_approx_l554_55456

/-- The mass percentage of calcium in calcium iodide -/
noncomputable def mass_percentage_Ca_in_CaI2 : ℝ :=
  let molar_mass_Ca : ℝ := 40.08
  let molar_mass_I : ℝ := 126.90
  let molar_mass_CaI2 : ℝ := molar_mass_Ca + 2 * molar_mass_I
  (molar_mass_Ca / molar_mass_CaI2) * 100

/-- Theorem: The mass percentage of calcium in calcium iodide is approximately 13.63% -/
theorem mass_percentage_Ca_in_CaI2_approx :
  |mass_percentage_Ca_in_CaI2 - 13.63| < 0.01 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_mass_percentage_Ca_in_CaI2_approx_l554_55456
