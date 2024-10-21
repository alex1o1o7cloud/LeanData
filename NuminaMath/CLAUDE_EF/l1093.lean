import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pork_sales_theorem_l1093_109333

/-- Represents the daily sales and pricing of pork --/
structure PorkSales where
  tibetan_price : ℝ
  black_price : ℝ
  total_weight : ℝ
  min_revenue : ℝ

/-- Represents the adjusted sales and pricing of pork --/
structure AdjustedPorkSales where
  base : PorkSales
  price_adjust : ℝ
  volume_adjust : ℝ
  revenue_increase : ℝ

/-- Calculates the minimum daily sales of Tibetan fragrant pork --/
noncomputable def min_tibetan_sales (s : PorkSales) : ℝ :=
  (s.min_revenue - s.black_price * s.total_weight) / (s.tibetan_price - s.black_price)

/-- Calculates the value of 'a' for the adjusted sales --/
noncomputable def calculate_a (s : AdjustedPorkSales) : ℝ :=
  let base_tibetan_sales := min_tibetan_sales s.base
  let base_black_sales := s.base.total_weight - base_tibetan_sales
  let new_tibetan_price := s.base.tibetan_price * (1 + 2 * s.price_adjust / 100)
  let new_black_price := s.base.black_price * (1 - s.price_adjust / 100)
  let new_tibetan_sales := base_tibetan_sales * (1 - s.volume_adjust / 100)
  let new_black_sales := base_black_sales * 2
  let new_revenue := s.base.min_revenue + s.revenue_increase
  (new_revenue - new_tibetan_price * new_tibetan_sales - new_black_price * new_black_sales) / 
    (new_tibetan_price * new_tibetan_sales * 2 / 100 + new_black_price * new_black_sales / 100)

/-- Theorem stating the correct minimum daily sales of Tibetan fragrant pork and value of 'a' --/
theorem pork_sales_theorem (s : PorkSales) (as : AdjustedPorkSales) : 
  s.tibetan_price = 30 ∧ 
  s.black_price = 20 ∧ 
  s.total_weight = 300 ∧ 
  s.min_revenue = 8000 ∧
  as.base = s ∧
  as.revenue_increase = 1750 →
  min_tibetan_sales s = 200 ∧ calculate_a as = 25 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pork_sales_theorem_l1093_109333


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_rotation_ratio_sum_of_m_and_n_l1093_109381

theorem cone_rotation_ratio (r h : ℝ) (h1 : r > 0) (h2 : h > 0) :
  h / r = 4 * Real.sqrt 14 :=
by
  -- Define the slant height
  let s := Real.sqrt (r^2 + h^2)

  -- The path length equality
  have path_equality : 2 * Real.pi * s = 30 * Real.pi * r :=
    sorry

  -- Simplify the equation
  have simplified_eq : s = 15 * r :=
    sorry

  -- Square both sides
  have squared_eq : s^2 = (15 * r)^2 :=
    sorry

  -- Substitute s^2 with r^2 + h^2
  have substituted_eq : r^2 + h^2 = 225 * r^2 :=
    sorry

  -- Solve for h^2
  have h_squared : h^2 = 224 * r^2 :=
    sorry

  -- Take the square root of both sides
  have h_over_r : h / r = Real.sqrt 224 :=
    sorry

  -- Simplify the square root
  calc h / r
    = Real.sqrt 224 := h_over_r
    _ = 4 * Real.sqrt 14 := by sorry

theorem sum_of_m_and_n : 4 + 14 = 18 :=
by rfl

#eval 4 + 14  -- This will output 18

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_rotation_ratio_sum_of_m_and_n_l1093_109381


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_new_person_weight_is_90_l1093_109305

/-- Calculates the weight of a new person joining a group, given the change in average weight -/
def new_person_weight
  (n : ℕ)                      -- number of people in the group
  (avg_increase : ℝ)           -- increase in average weight
  (replaced_weight : ℝ)        -- weight of the person being replaced
  : ℝ :=
  replaced_weight + n * avg_increase

/-- Proves that the weight of the new person is 90 kg -/
theorem new_person_weight_is_90
  (n : ℕ)
  (avg_increase : ℝ)
  (replaced_weight : ℝ)
  (h1 : n = 10)
  (h2 : avg_increase = 2.5)
  (h3 : replaced_weight = 65)
  : new_person_weight n avg_increase replaced_weight = 90 := by
  -- Unfold the definition of new_person_weight
  unfold new_person_weight
  -- Substitute the given values
  rw [h1, h2, h3]
  -- Perform the calculation
  norm_num

-- Example calculation
#eval new_person_weight 10 2.5 65


end NUMINAMATH_CALUDE_ERRORFEEDBACK_new_person_weight_is_90_l1093_109305


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_a_eq_one_solution_set_all_reals_l1093_109374

-- Define the inequality function as noncomputable
noncomputable def f (x : ℝ) := Real.log (|x + 3| + |x - 7|)

-- Theorem 1: Solution set when a = 1
theorem solution_set_a_eq_one :
  {x : ℝ | f x > 1} = {x : ℝ | x < -3 ∨ x > 7} := by
  sorry

-- Theorem 2: Condition for solution set to be ℝ
theorem solution_set_all_reals (a : ℝ) :
  ({x : ℝ | f x > a} = Set.univ) ↔ a < 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_a_eq_one_solution_set_all_reals_l1093_109374


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_product_equation_l1093_109351

theorem cosine_product_equation (x : ℝ) :
  (Real.cos x) * (Real.cos (2 * x)) * (Real.cos (4 * x)) = 1 / 16 →
  (∃ k : ℤ, x = (2 * k * Real.pi) / 7 ∨ x = ((2 * k + 1) * Real.pi) / 9) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_product_equation_l1093_109351


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_triangle_area_l1093_109320

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := 2 * Real.sqrt 3 * Real.sin x ^ 2 + 2 * Real.sin x * Real.cos x - Real.sqrt 3

-- Define the domain
def domain : Set ℝ := { x | Real.pi / 3 ≤ x ∧ x ≤ 11 * Real.pi / 24 }

-- Theorem for the range of f(x)
theorem f_range : ∀ x ∈ domain, Real.sqrt 3 ≤ f x ∧ f x ≤ 2 := by
  sorry

-- Theorem for the area of triangle ABC
theorem triangle_area (a b r : ℝ) (ha : a = Real.sqrt 3) (hb : b = 2) (hr : r = 3 * Real.sqrt 2 / 4) :
  let s := (a + b + (a * b / (2 * r))) / 2
  Real.sqrt (s * (s - a) * (s - b) * (s - (a * b / (2 * r)))) = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_triangle_area_l1093_109320


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_car_sales_profit_percent_l1093_109373

/-- Calculates the overall profit percent for a trader selling five cars with given prices and profit/loss percentages. -/
noncomputable def overallProfitPercent (price1 price2 price3 price4 price5 : ℝ)
  (profit_percent1 profit_percent2 profit_percent3 profit_percent4 profit_percent5 : ℝ) : ℝ :=
  let profit1 := price1 * profit_percent1
  let profit2 := price2 * profit_percent2
  let profit3 := price3 * profit_percent3
  let profit4 := price4 * profit_percent4
  let profit5 := price5 * profit_percent5
  let total_profit := profit1 + profit2 + profit3 + profit4 + profit5
  let total_cost := price1 + price2 + price3 + price4 + price5
  (total_profit / total_cost) * 100

/-- The overall profit percent for the given car sales scenario is approximately 5.18%. -/
theorem car_sales_profit_percent :
  let ε := 0.01
  abs (overallProfitPercent 325475 375825 450000 287500 600000 0.12 (-0.12) 0.08 (-0.05) 0.15 - 5.18) < ε :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_car_sales_profit_percent_l1093_109373


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_same_color_unit_distance_l1093_109391

-- Define the color type
inductive Color
| Red
| Black

-- Define a point in the plane
structure Point where
  x : ℝ
  y : ℝ

-- Define the coloring function
def coloring : Point → Color := sorry

-- Define the distance function between two points
noncomputable def distance (p q : Point) : ℝ :=
  Real.sqrt ((p.x - q.x)^2 + (p.y - q.y)^2)

-- Theorem statement
theorem same_color_unit_distance :
  ∃ (p q : Point), coloring p = coloring q ∧ distance p q = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_same_color_unit_distance_l1093_109391


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_zero_at_negative_one_l1093_109348

-- Define the fraction
noncomputable def f (x : ℝ) : ℝ := (|x| - 1) / (x^2 - 2*x + 1)

-- Theorem statement
theorem fraction_zero_at_negative_one :
  ∀ x : ℝ, f x = 0 ∧ x^2 - 2*x + 1 ≠ 0 → x = -1 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_zero_at_negative_one_l1093_109348


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_intersection_l1093_109371

-- Define the functions f and g as noncomputable
noncomputable def f (x : ℝ) : ℝ := 1 / Real.sqrt (1 - x)
noncomputable def g (x : ℝ) : ℝ := Real.log (1 + x)

-- Define the domains of f and g
def M : Set ℝ := {x | x < 1}
def N : Set ℝ := {x | x > -1}

-- State the theorem
theorem domain_intersection :
  M ∩ N = {x : ℝ | -1 < x ∧ x < 1} :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_intersection_l1093_109371


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_triangle_area_l1093_109386

-- Define the triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- Define the conditions
def triangle_conditions (t : Triangle) : Prop :=
  t.c = 2 ∧ t.C = Real.pi/3

-- Part I
theorem equilateral_triangle (t : Triangle) 
  (h : triangle_conditions t) 
  (area : (1/2) * t.a * t.b * Real.sin t.C = Real.sqrt 3) :
  t.a = t.b ∧ t.b = t.c := by sorry

-- Part II
theorem triangle_area (t : Triangle) 
  (h : triangle_conditions t)
  (eq : Real.sin t.C + Real.sin (t.B - t.A) = 2 * Real.sin (2 * t.A)) :
  (1/2) * t.a * t.b * Real.sin t.C = (2 * Real.sqrt 3) / 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_triangle_area_l1093_109386


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mushroom_count_l1093_109300

def Basket :=
  {mushrooms : Finset ℕ // mushrooms.card = 30}

def SaffronMilkCap (basket : Basket) (mushroom : ℕ) : Prop :=
  mushroom ∈ basket.val

def MilkCap (basket : Basket) (mushroom : ℕ) : Prop :=
  mushroom ∈ basket.val

def AtLeastOneSaffronIn12 (basket : Basket) : Prop :=
  ∀ (subset : Finset ℕ), subset ⊆ basket.val → subset.card = 12 →
    ∃ (mushroom : ℕ), mushroom ∈ subset ∧ SaffronMilkCap basket mushroom

def AtLeastOneMilkCapIn20 (basket : Basket) : Prop :=
  ∀ (subset : Finset ℕ), subset ⊆ basket.val → subset.card = 20 →
    ∃ (mushroom : ℕ), mushroom ∈ subset ∧ MilkCap basket mushroom

theorem mushroom_count (basket : Basket)
  (h1 : AtLeastOneSaffronIn12 basket)
  (h2 : AtLeastOneMilkCapIn20 basket) :
  ∃ (saffron milkcap : Finset ℕ),
    saffron ⊆ basket.val ∧ milkcap ⊆ basket.val ∧
    saffron.card = 19 ∧ milkcap.card = 11 ∧
    ∀ m ∈ basket.val, (m ∈ saffron ∨ m ∈ milkcap) ∧ ¬(m ∈ saffron ∧ m ∈ milkcap) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mushroom_count_l1093_109300


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_final_liquid_x_percentage_is_25_percent_l1093_109302

/-- Represents the composition of a solution -/
structure Solution where
  total_mass : ℚ
  liquid_x_percentage : ℚ
  water_percentage : ℚ

/-- The initial solution Y -/
def initial_solution_y : Solution :=
  { total_mass := 8
  , liquid_x_percentage := 1/5
  , water_percentage := 4/5 }

/-- Mass of water that evaporates -/
def evaporated_water : ℚ := 2

/-- Mass of solution Y added after evaporation -/
def added_solution_y : ℚ := 2

/-- Calculates the final percentage of liquid X in the solution -/
def final_liquid_x_percentage (s : Solution) (evaporated : ℚ) (added : ℚ) : ℚ :=
  let initial_liquid_x := s.total_mass * s.liquid_x_percentage
  let initial_water := s.total_mass * s.water_percentage
  let remaining_water := initial_water - evaporated
  let added_liquid_x := added * s.liquid_x_percentage
  let added_water := added * s.water_percentage
  let final_liquid_x := initial_liquid_x + added_liquid_x
  let final_water := remaining_water + added_water
  let final_total := final_liquid_x + final_water
  final_liquid_x / final_total

theorem final_liquid_x_percentage_is_25_percent :
  final_liquid_x_percentage initial_solution_y evaporated_water added_solution_y = 1/4 := by
  sorry

#eval final_liquid_x_percentage initial_solution_y evaporated_water added_solution_y

end NUMINAMATH_CALUDE_ERRORFEEDBACK_final_liquid_x_percentage_is_25_percent_l1093_109302


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_valid_sequences_l1093_109355

def is_valid_sequence (b : Fin 12 → ℕ) : Prop :=
  (∀ i j : Fin 12, i ≤ j → b i ≤ b j) ∧
  (∀ i : Fin 12, b i ≤ 2010) ∧
  (∀ i : Fin 12, Odd (b i - i.val - 1))

-- We need to prove that the set of valid sequences is finite
instance : Fintype {b : Fin 12 → ℕ | is_valid_sequence b} := by sorry

theorem count_valid_sequences :
  Fintype.card {b : Fin 12 → ℕ | is_valid_sequence b} = Nat.choose 1016 12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_valid_sequences_l1093_109355


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_boat_journey_time_l1093_109367

/-- Represents the speed of a boat in still water -/
noncomputable def boat_speed : ℝ := 6

/-- Represents the speed of the stream -/
noncomputable def stream_speed : ℝ := 2

/-- The time taken for a boat to travel a given distance downstream -/
noncomputable def time_downstream (distance : ℝ) : ℝ :=
  distance / (boat_speed + stream_speed)

/-- The time taken for a boat to travel a given distance upstream -/
noncomputable def time_upstream (distance : ℝ) : ℝ :=
  distance / (boat_speed - stream_speed)

theorem boat_journey_time :
  (time_downstream 16 + time_upstream 8 = 4) ∧
  (time_downstream 12 + time_upstream 10 = 4) →
  (time_downstream 24 + time_upstream 24 = 9) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_boat_journey_time_l1093_109367


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pipe_weight_approx_l1093_109332

/-- Calculates the weight of a hollow iron pipe -/
noncomputable def pipeWeight (length : ℝ) (externalDiameter : ℝ) (thickness : ℝ) (ironWeight : ℝ) : ℝ :=
  let externalRadius := externalDiameter / 2
  let internalRadius := externalRadius - thickness
  let volumeSolid := Real.pi * externalRadius^2 * length
  let volumeHollow := Real.pi * internalRadius^2 * length
  let volumeIron := volumeSolid - volumeHollow
  volumeIron * ironWeight

/-- The weight of the specified hollow iron pipe is approximately 2736.1416 grams -/
theorem pipe_weight_approx :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.0001 ∧ 
  |pipeWeight 21 8 1 8 - 2736.1416| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_pipe_weight_approx_l1093_109332


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_extrema_l1093_109317

-- Define the function f(x) as noncomputable due to the use of Real.sqrt
noncomputable def f (x m : ℝ) : ℝ := Real.sqrt 3 * Real.sin (2 * x) + 2 * (Real.cos x) ^ 2 + m

-- State the theorem
theorem function_extrema (m : ℝ) :
  (∀ x ∈ Set.Icc 0 (Real.pi / 2), f x m ≥ 3) ∧
  (∃ x ∈ Set.Icc 0 (Real.pi / 2), f x m = 3) →
  m = 3 ∧ ∀ a : ℝ, ∃ x ∈ Set.Icc a (a + Real.pi), f x m = 6 ∧ ∀ y ∈ Set.Icc a (a + Real.pi), f y m ≤ 6 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_extrema_l1093_109317


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_AMC_is_7pi_12_l1093_109385

-- Define the triangle and point
def Triangle (A B C M : ℝ × ℝ) : Prop :=
  -- ABC is isosceles right triangle
  ∃ a : ℝ, A = (a, 0) ∧ B = (0, a) ∧ C = (0, 0)

-- Define distances
noncomputable def Distance (P Q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)

-- Theorem statement
theorem angle_AMC_is_7pi_12 (A B C M : ℝ × ℝ) :
  Triangle A B C M →
  Distance M A = 5 →
  Distance M B = 7 →
  Distance M C = 4 * Real.sqrt 2 →
  ∃ angle : ℝ, angle = (7 * Real.pi) / 12 ∧
             angle = Real.arccos ((Distance M A)^2 + (Distance M C)^2 - (Distance A C)^2) / (2 * Distance M A * Distance M C) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_AMC_is_7pi_12_l1093_109385


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_exponent_sum_l1093_109363

theorem smallest_exponent_sum (p q : ℕ) :
  (∃ (k : ℕ), (7^(p + 4) * 5^q * 2^3 = k^3)) →
  (∀ (r s : ℕ), (∃ (m : ℕ), (7^(r + 4) * 5^s * 2^3 = m^3)) → p + q ≤ r + s) →
  p + q = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_exponent_sum_l1093_109363


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_roots_for_special_quartic_l1093_109379

noncomputable section

def Q (a b c d : ℝ) (x : ℂ) : ℂ := x^4 + a*x^3 + b*x^2 + c*x + d

def area_of_quadrilateral_formed_by_roots (Q : ℂ → ℂ) : ℝ := sorry

def sum_of_roots (Q : ℂ → ℂ) : ℝ := sorry

theorem sum_of_roots_for_special_quartic (a b c d : ℝ) (φ : ℝ) :
  (0 < φ) → (φ < π/6) →
  (Q a b c d (Complex.cos φ + Complex.I * Complex.sin (2*φ)) = 0) →
  (Q a b c d (Complex.sin (2*φ) + Complex.I * Complex.cos φ) = 0) →
  (Complex.abs (Q a b c d 0) = area_of_quadrilateral_formed_by_roots (Q a b c d)) →
  (sum_of_roots (Q a b c d) = (Real.sqrt 6 + Real.sqrt 2) / 2 + 1) :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_roots_for_special_quartic_l1093_109379


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_special_sequence_length_l1093_109370

/-- A sequence of natural numbers with the given properties -/
def SpecialSequence (x : ℕ → ℕ) : Prop :=
  (∀ i, x i ≤ 1998) ∧ 
  (∀ i ≥ 3, x i = Int.natAbs (x (i-1) - x (i-2)))

/-- The length of a sequence -/
def SequenceLength (x : ℕ → ℕ) : ℕ → ℕ
  | 0 => 0
  | n+1 => if n ≥ 2 ∧ x (n+1) = x n ∧ x (n+1) = x (n-1) then n else SequenceLength x n

theorem max_special_sequence_length :
  ∃ (x : ℕ → ℕ), SpecialSequence x ∧ SequenceLength x 2999 = 2998 ∧
  ∀ (y : ℕ → ℕ), SpecialSequence y → SequenceLength y 2999 ≤ 2998 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_special_sequence_length_l1093_109370


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_general_solution_satisfies_equation_l1093_109347

-- Define the differential equation
def differential_equation (y : ℝ → ℝ) (x : ℝ) : Prop :=
  (deriv (deriv y)) x + 3 * (deriv y) x + 2 * y x = 2 * x^2 - 4 * x - 17

-- Define the general solution
noncomputable def general_solution (C₁ C₂ : ℝ) (x : ℝ) : ℝ :=
  C₁ * Real.exp (-x) + C₂ * Real.exp (-2 * x) + x^2 - 5 * x - 2

-- Theorem statement
theorem general_solution_satisfies_equation (C₁ C₂ : ℝ) :
  ∀ x, differential_equation (general_solution C₁ C₂) x := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_general_solution_satisfies_equation_l1093_109347


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_nonconstant_coefficients_l1093_109389

-- Define the binomial expression
noncomputable def binomial_expr (x : ℝ) : ℝ := (2 / Real.sqrt x - x) ^ 9

-- Define the sum of all coefficients
noncomputable def sum_all_coefficients : ℝ := binomial_expr 1

-- Define the constant term
def constant_term : ℝ := -5376

-- Theorem statement
theorem sum_nonconstant_coefficients :
  sum_all_coefficients - constant_term = 5377 := by
  sorry

#eval constant_term

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_nonconstant_coefficients_l1093_109389


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_polynomial_theorem_l1093_109366

/-- A cubic polynomial with real coefficients -/
structure CubicPolynomial where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

/-- The value of the polynomial at a given x -/
def CubicPolynomial.eval (p : CubicPolynomial) (x : ℝ) : ℝ :=
  p.a * x^3 + p.b * x^2 + p.c * x + p.d

/-- The condition that Q(x^2 + x - 1) ≥ Q(x^2 + 2) for all real x -/
def satisfiesCondition (Q : CubicPolynomial) : Prop :=
  ∀ x : ℝ, Q.eval (x^2 + x - 1) ≥ Q.eval (x^2 + 2)

/-- The sum of the roots of a quadratic polynomial bx^2 + cx + d -/
noncomputable def sumOfRoots (b c : ℝ) : ℝ := -c / b

theorem cubic_polynomial_theorem (Q : CubicPolynomial) 
  (h : satisfiesCondition Q) : 
  Q.a = 0 ∧ sumOfRoots Q.b Q.c = -Q.c / Q.b := by
  sorry

#check cubic_polynomial_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_polynomial_theorem_l1093_109366


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_invariance_l1093_109375

-- Define a function that checks if a number remains unchanged when raised to any positive integer power
def unchangedWhenPowered (x : ℂ) : Prop :=
  ∀ n : ℕ+, x^(n : ℕ) = x

-- Define a function that checks if a number retains its absolute value when raised to any positive integer power
def retainsAbsValueWhenPowered (z : ℂ) : Prop :=
  ∀ n : ℕ+, Complex.abs (z^(n : ℕ)) = Complex.abs z

-- Theorem statement
theorem power_invariance :
  (∀ x : ℂ, unchangedWhenPowered x ↔ (x = 1 ∨ x = 0)) ∧
  (∀ z : ℂ, retainsAbsValueWhenPowered z ↔ Complex.abs z = 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_invariance_l1093_109375


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_of_three_difference_l1093_109312

theorem power_of_three_difference (a b : ℝ) (h1 : (3 : ℝ)^a = 5) (h2 : (3 : ℝ)^b = 8) :
  (3 : ℝ)^(3*a - 2*b) = 125/64 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_of_three_difference_l1093_109312


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_temperature_variance_l1093_109359

noncomputable def temperatures : List ℝ := [28, 21, 22, 26, 28, 25]

noncomputable def mean (xs : List ℝ) : ℝ :=
  xs.sum / xs.length

noncomputable def variance (xs : List ℝ) : ℝ :=
  (xs.map (fun x => (x - mean xs) ^ 2)).sum / xs.length

theorem temperature_variance :
  variance temperatures = 22 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_temperature_variance_l1093_109359


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_negative_three_pi_half_l1093_109390

theorem cos_negative_three_pi_half : Real.cos (-3 * π / 2) = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_negative_three_pi_half_l1093_109390


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_magnitude_b_when_perpendicular_x_range_when_dot_product_negative_l1093_109387

noncomputable section

-- Define the vectors
def a (x : ℝ) : ℝ × ℝ := (x, -1)
def b (x : ℝ) : ℝ × ℝ := (x - 2, 3)
def c (x : ℝ) : ℝ × ℝ := (1 - 2*x, 6)

-- Define dot product
def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

-- Define vector addition
def vec_add (v w : ℝ × ℝ) : ℝ × ℝ := (v.1 + w.1, v.2 + w.2)

-- Define scalar multiplication
def scalar_mult (k : ℝ) (v : ℝ × ℝ) : ℝ × ℝ := (k * v.1, k * v.2)

-- Define vector magnitude
noncomputable def vec_magnitude (v : ℝ × ℝ) : ℝ := Real.sqrt (v.1^2 + v.2^2)

-- Theorem 1
theorem magnitude_b_when_perpendicular (x : ℝ) : 
  dot_product (a x) (vec_add (scalar_mult 2 (b x)) (c x)) = 0 → 
  vec_magnitude (b x) = 3 * Real.sqrt 5 := by
  sorry

-- Theorem 2
theorem x_range_when_dot_product_negative (x : ℝ) :
  dot_product (a x) (b x) < 0 →
  -1 < x ∧ x < 3 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_magnitude_b_when_perpendicular_x_range_when_dot_product_negative_l1093_109387


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_burglar_stolen_goods_value_l1093_109383

-- Define the parameters of the problem
noncomputable def base_sentence_rate : ℝ := 1 / 5000
noncomputable def third_offense_increase : ℝ := 0.25
noncomputable def resisting_arrest_penalty : ℝ := 2
noncomputable def total_sentence : ℝ := 12

-- Define the theorem
theorem burglar_stolen_goods_value :
  ∃ (stolen_value : ℝ),
    (stolen_value * base_sentence_rate * (1 + third_offense_increase) + resisting_arrest_penalty = total_sentence) ∧
    (stolen_value = 40000) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_burglar_stolen_goods_value_l1093_109383


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solve_exponential_equation_l1093_109303

theorem solve_exponential_equation (x : ℝ) : (4 : ℝ)^x * (4 : ℝ)^x * (4 : ℝ)^x = (16 : ℝ)^3 → x = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solve_exponential_equation_l1093_109303


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_sum_l1093_109345

-- Define the curves C1 and C2
def C1 (x y : ℝ) : Prop := x + y - 3 = 0
def C2 (x y : ℝ) : Prop := y^2 = 2*x

-- Define point P
def P : ℝ × ℝ := (1, 2)

-- Define the distance function
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Theorem statement
theorem intersection_distance_sum :
  ∃ (A B : ℝ × ℝ),
    C1 A.1 A.2 ∧ C2 A.1 A.2 ∧
    C1 B.1 B.2 ∧ C2 B.1 B.2 ∧
    distance P A + distance P B = 2 * Real.sqrt 22 := by
  sorry

#check intersection_distance_sum

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_sum_l1093_109345


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombus_side_length_l1093_109341

/-- The length of a side of a rhombus, given its diagonals -/
theorem rhombus_side_length (d₁ d₂ : ℝ) : 
  d₁^2 - 3*d₁ + 2 = 0 →
  d₂^2 - 3*d₂ + 2 = 0 →
  d₁ ≠ d₂ →
  ∃ (s : ℝ), s^2 = (d₁^2 + d₂^2) / 4 ∧ s = Real.sqrt 5 / 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombus_side_length_l1093_109341


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_s_squared_l1093_109330

/-- A hyperbola centered at the origin passing through specific points -/
structure Hyperbola where
  a : ℝ
  s : ℝ
  -- The hyperbola passes through (5, -3)
  point1 : (5 : ℝ)^2 / 25 - (-3 : ℝ)^2 / a^2 = 1
  -- The hyperbola passes through (0, 3)
  point2 : (0 : ℝ)^2 / 25 - (3 : ℝ)^2 / a^2 = 1
  -- The hyperbola passes through (s, -4)
  point3 : s^2 / 25 - (-4 : ℝ)^2 / a^2 = 1
  -- a is positive (for a vertically opening hyperbola)
  a_pos : a > 0

/-- The theorem stating that s² = 175/9 for the given hyperbola -/
theorem hyperbola_s_squared (h : Hyperbola) : h.s^2 = 175/9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_s_squared_l1093_109330


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_encryption_decryption_l1093_109313

-- Define the encryption function
noncomputable def encrypt (a : ℝ) (x : ℝ) : ℝ := a^x - 2

-- State the theorem
theorem encryption_decryption :
  ∃ a : ℝ,
  (encrypt a 3 = 6) ∧
  (∃ x : ℝ, encrypt a x = 14 ∧ x = 4) :=
by
  -- We'll use 'sorry' to skip the proof for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_encryption_decryption_l1093_109313


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_board_symbols_l1093_109324

theorem board_symbols (n : ℕ) (plus minus : Finset ℕ) : 
  n = 23 →
  plus ∪ minus = Finset.range n →
  plus ∩ minus = ∅ →
  (∀ s : Finset ℕ, s ⊆ Finset.range n → s.card = 10 → (s ∩ plus).Nonempty) →
  (∀ s : Finset ℕ, s ⊆ Finset.range n → s.card = 15 → (s ∩ minus).Nonempty) →
  plus.card = 14 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_board_symbols_l1093_109324


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_coaches_l1093_109376

/-- The speed-coach relationship for a train --/
structure TrainSystem where
  initialSpeed : ℚ
  k : ℚ

/-- Calculate the speed of the train given the number of coaches --/
def TrainSystem.speed (system : TrainSystem) (coaches : ℚ) : ℚ :=
  system.k / (coaches.sqrt)

/-- The specific train system described in the problem --/
def problemTrainSystem : TrainSystem :=
  { initialSpeed := 60
  , k := 96 }

theorem train_speed_coaches : 
  (problemTrainSystem.speed 16 = 24) ∧ 
  (problemTrainSystem.speed 4 = 48) ∧ 
  (problemTrainSystem.initialSpeed = 60) := by
  sorry

#check train_speed_coaches

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_coaches_l1093_109376


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_property_l1093_109396

-- Define a power function
noncomputable def power_function (α : ℝ) : ℝ → ℝ := fun x ↦ x ^ α

-- State the theorem
theorem power_function_property :
  ∃ α : ℝ, power_function α 2 = 8 ∧
  ∃ x : ℝ, power_function α x = 64 ∧ x = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_property_l1093_109396


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_odd_sufficient_not_necessary_l1093_109382

-- Define the function f
noncomputable def f (φ : ℝ) (x : ℝ) : ℝ := Real.sin (x + φ)

-- Define what it means for a function to be odd
def is_odd (g : ℝ → ℝ) : Prop := ∀ x, g (-x) = -g x

-- State the theorem
theorem sin_odd_sufficient_not_necessary :
  (∃ φ : ℝ, φ ≠ π ∧ is_odd (f φ)) ∧
  (is_odd (f π)) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_odd_sufficient_not_necessary_l1093_109382


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_l1093_109395

theorem range_of_m (m : ℝ) : 
  (∀ (a b c x : ℝ), a^2 + b^2 + c^2 = 1 → Real.sqrt 2*a + Real.sqrt 3*b + 2*c ≤ |x - 1| + |x + m|) →
  m ≤ -4 ∨ m ≥ 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_l1093_109395


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_paths_count_l1093_109346

/-- A chessboard represented as a 10x10 grid --/
def Chessboard := Fin 10 → Fin 10 → Bool

/-- A position on the chessboard --/
structure Position where
  row : Fin 10
  col : Fin 10

/-- A step on the chessboard --/
inductive Step
  | UpLeft
  | Up
  | UpRight

/-- A path on the chessboard --/
def ChessPath := List Step

/-- Checks if a position is on a white square --/
def isWhite (board : Chessboard) (pos : Position) : Bool :=
  board pos.row pos.col

/-- Checks if a step is valid (moves to an adjacent white square in the row above) --/
def isValidStep (board : Chessboard) (start : Position) (step : Step) : Bool :=
  sorry

/-- Checks if a path is valid (all steps are valid) --/
def isValidPath (board : Chessboard) (start : Position) (path : ChessPath) : Bool :=
  sorry

/-- Counts the number of valid 9-step paths between two positions --/
def countValidPaths (board : Chessboard) (start finish : Position) : Nat :=
  sorry

/-- The main theorem to prove --/
theorem valid_paths_count (board : Chessboard) (P Q : Position) :
  isWhite board P ∧ 
  isWhite board Q ∧ 
  P.row = 0 ∧ 
  Q.row = 9 ∧ 
  P.col = Q.col ∧
  (∀ (i j : Fin 10), board i j = (i.val + j.val).bodd) →
  countValidPaths board P Q = 457 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_paths_count_l1093_109346


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vertical_asymptotes_sum_l1093_109350

/-- The function for which we're finding vertical asymptotes -/
noncomputable def f (x : ℝ) : ℝ := (6 * x^2 - 13) / (4 * x^2 + 8 * x + 3)

/-- The denominator of the function -/
def denominator (x : ℝ) : ℝ := 4 * x^2 + 8 * x + 3

theorem vertical_asymptotes_sum :
  ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ 
  denominator x₁ = 0 ∧ 
  denominator x₂ = 0 ∧ 
  x₁ + x₂ = -2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vertical_asymptotes_sum_l1093_109350


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sector_area_l1093_109314

/-- The area of a sector with radius 3 cm and central angle 120° is 3π cm² -/
theorem sector_area : 
  (1/2 : ℝ) * 3^2 * (120 * π / 180) = 3 * π := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sector_area_l1093_109314


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagon_vertex_d_y_coordinate_l1093_109356

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Calculates the area of a triangle given its base and height -/
noncomputable def triangleArea (base height : ℝ) : ℝ := (1/2) * base * height

/-- Represents a hexagon with specific properties -/
structure Hexagon where
  A : Point
  B : Point
  C : Point
  D : Point
  E : Point
  F : Point
  symmetrical : D.x = C.x -- Vertical line of symmetry
  area : ℝ

/-- Theorem: The y-coordinate of vertex D in the specified hexagon is 22.5 -/
theorem hexagon_vertex_d_y_coordinate (h : Hexagon) 
    (h_A : h.A = ⟨0, 0⟩)
    (h_B : h.B = ⟨0, 6⟩)
    (h_C : h.C = ⟨2, 6⟩)
    (h_E : h.E = ⟨4, 6⟩)
    (h_F : h.F = ⟨4, 0⟩)
    (h_area : h.area = 90) :
  h.D.y = 22.5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagon_vertex_d_y_coordinate_l1093_109356


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_asymptotes_intersection_l1093_109380

-- Define the function
noncomputable def f (x : ℝ) : ℝ := (x^2 - 6*x + 8) / (x^2 - 6*x + 9)

-- Define the vertical asymptote
def vertical_asymptote : ℝ := 3

-- Define the horizontal asymptote
def horizontal_asymptote : ℝ := 1

-- Theorem statement
theorem asymptotes_intersection :
  (vertical_asymptote, horizontal_asymptote) = (3, 1) := by
  -- The proof would go here, but we'll use sorry for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_asymptotes_intersection_l1093_109380


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_impossibility_l1093_109311

def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, n = m * m

def is_triangle (a b c : ℕ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

theorem triangle_impossibility : ∃ a b : ℕ, 
  (a > 1000 ∧ b > 1000) ∧ 
  (∀ c : ℕ, is_perfect_square c → ¬is_triangle a b c) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_impossibility_l1093_109311


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_journey_average_speed_l1093_109326

/-- Represents a journey with three segments -/
structure Journey where
  north_distance : ℝ
  north_speed : ℝ
  east_distance : ℝ
  east_speed : ℝ
  return_speed : ℝ

/-- Calculates the average speed of a journey -/
noncomputable def average_speed (j : Journey) : ℝ :=
  let north_time := j.north_distance / j.north_speed
  let east_time := j.east_distance / j.east_speed
  let return_distance := Real.sqrt (j.north_distance^2 + j.east_distance^2)
  let return_time := return_distance / j.return_speed
  let total_distance := j.north_distance + j.east_distance + return_distance
  let total_time := north_time + east_time + return_time
  total_distance / total_time

/-- The specific journey described in the problem -/
def specific_journey : Journey :=
  { north_distance := 10
  , north_speed := 10
  , east_distance := 24
  , east_speed := 12
  , return_speed := 13 }

/-- Theorem stating that the average speed of the specific journey is 12 mph -/
theorem specific_journey_average_speed :
  average_speed specific_journey = 12 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_journey_average_speed_l1093_109326


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_existence_l1093_109344

/-- A regular polygon with n sides --/
structure RegularPolygon (n : ℕ) where
  vertices : Fin n → ℝ × ℝ

/-- A set of marked vertices in a polygon --/
def MarkedVertices (n : ℕ) := Fin n → Bool

/-- A trapezoid is a quadrilateral with at least one pair of parallel sides --/
def IsTrapezoid (v₁ v₂ v₃ v₄ : ℝ × ℝ) : Prop :=
  ∃ (i j : Fin 4), i ≠ j ∧ (v₁.1 - v₂.1) / (v₁.2 - v₂.2) = (v₃.1 - v₄.1) / (v₃.2 - v₄.2)

/-- The main theorem --/
theorem trapezoid_existence
  (polygon : RegularPolygon 1981)
  (marked : MarkedVertices 1981)
  (h : (Finset.filter (fun i => marked i) Finset.univ).card = 64) :
  ∃ (v₁ v₂ v₃ v₄ : Fin 1981),
    marked v₁ ∧ marked v₂ ∧ marked v₃ ∧ marked v₄ ∧
    IsTrapezoid (polygon.vertices v₁) (polygon.vertices v₂) (polygon.vertices v₃) (polygon.vertices v₄) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_existence_l1093_109344


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_determinant_solution_l1093_109328

theorem matrix_determinant_solution (a : ℝ) (h1 : a ≠ 0) : 
  ∀ x : ℝ, Matrix.det (
    !![x + a, x, x, x;
       x, x + a, x, x;
       x, x, x + a, x;
       x, x, x, x + a]
  ) = 0 → x = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_determinant_solution_l1093_109328


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_of_w_and_x_l1093_109310

theorem average_of_w_and_x (w x a b : ℝ) :
  (6 / w + 6 / x = 6 / (Complex.ofReal a + Complex.I * Complex.ofReal b)) →
  (w * x = Complex.ofReal a + Complex.I * Complex.ofReal b) →
  (w + x) / 2 = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_of_w_and_x_l1093_109310


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_C_in_triangle_l1093_109334

-- Define IsTriangle
def IsTriangle (A B C : Real) : Prop :=
  A + B + C = Real.pi ∧ A > 0 ∧ B > 0 ∧ C > 0

theorem tan_C_in_triangle (A B C : Real) :
  -- Triangle ABC exists
  IsTriangle A B C →
  -- tan A and tan B are roots of 3x^2 - 7x + 2 = 0
  (∃ x y : Real, x ≠ y ∧ 
    Real.tan A = x ∧ Real.tan B = y ∧ 
    3 * x^2 - 7 * x + 2 = 0 ∧ 
    3 * y^2 - 7 * y + 2 = 0) →
  -- Then tan C = -7
  Real.tan C = -7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_C_in_triangle_l1093_109334


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_area_increase_l1093_109393

theorem rectangle_area_increase (l w : ℝ) (hl : l > 0) (hw : w > 0) : 
  (1.25 * l * 1.20 * w - l * w) / (l * w) = 0.5 := by
  -- Simplify the expression
  calc (1.25 * l * 1.20 * w - l * w) / (l * w)
     = (1.5 * l * w - l * w) / (l * w) := by ring_nf
  -- Further simplification
   _ = (0.5 * l * w) / (l * w) := by ring_nf
  -- Final step
   _ = 0.5 := by
      field_simp
      ring_nf


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_area_increase_l1093_109393


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equations_l1093_109323

/-- A line that does not pass through the second quadrant -/
def line_l (a : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | a * p.1 - p.2 - 4 = 0}

/-- The circle with equation x^2 + (y - 1)^2 = 5 -/
def circle_set : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 + (p.2 - 1)^2 = 5}

/-- Line l_1 passing through (3, -1) and parallel to line_l -/
def line_l1 (b : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | 2 * p.1 - p.2 + b = 0}

/-- Line l_2 symmetric to line_l1 about y = 1 -/
def line_l2 : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | 2 * p.1 + p.2 - 9 = 0}

/-- The main theorem -/
theorem line_equations
  (h_tangent : ∃ p, p ∈ circle_set ∩ line_l 2 ∧ ∀ q ∈ circle_set, q ∈ line_l 2 → q = p)
  (h_not_second_quadrant : ∀ p ∈ line_l 2, p.1 ≤ 0 → p.2 ≤ 0)
  (h_l1_point : (3, -1) ∈ line_l1 (-7))
  (h_l1_parallel : ∀ p q : ℝ × ℝ, p ∈ line_l 2 ∧ q ∈ line_l1 (-7) → p.2 - q.2 = 2 * (p.1 - q.1))
  (h_l2_symmetric : ∀ p ∈ line_l1 (-7), ∃ q ∈ line_l2, q.1 = p.1 ∧ q.2 = 2 - p.2) :
  (line_l 2 = {p : ℝ × ℝ | 2 * p.1 - p.2 - 4 = 0}) ∧
  (line_l2 = {p : ℝ × ℝ | 2 * p.1 + p.2 - 9 = 0}) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equations_l1093_109323


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_stormi_needs_additional_money_l1093_109315

def car_wash_earnings : ℚ := 5 * (85 / 10)
def dog_walk_earnings : ℚ := 4 * (27 / 4)
def lawn_mow_earnings : ℚ := 3 * (49 / 4)
def gardening_earnings : ℚ := 2 * (37 / 5)

def bicycle_cost : ℚ := 15025 / 100
def helmet_cost : ℚ := 3575 / 100
def lock_cost : ℚ := 245 / 10

def bicycle_discount_rate : ℚ := 15 / 100
def helmet_discount : ℚ := 5
def sales_tax_rate : ℚ := 5 / 100

def total_earnings : ℚ := car_wash_earnings + dog_walk_earnings + lawn_mow_earnings + gardening_earnings

def discounted_bicycle_cost : ℚ := bicycle_cost * (1 - bicycle_discount_rate)
def discounted_helmet_cost : ℚ := helmet_cost - helmet_discount

def total_cost_before_tax : ℚ := discounted_bicycle_cost + discounted_helmet_cost + lock_cost
def total_cost_after_tax : ℚ := total_cost_before_tax * (1 + sales_tax_rate)

def additional_money_needed : ℚ := total_cost_after_tax - total_earnings

theorem stormi_needs_additional_money :
  (⌈additional_money_needed * 100⌉ : ℚ) / 100 = 7106 / 100 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_stormi_needs_additional_money_l1093_109315


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_plates_for_matching_pair_l1093_109360

theorem min_plates_for_matching_pair (n : ℕ) (h : n = 5) : 
  ∀ (f : Fin n → ℕ), ∃ (i j : Fin (n + 1)), i ≠ j ∧ f ⟨i.val, sorry⟩ = f ⟨j.val, sorry⟩ :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_plates_for_matching_pair_l1093_109360


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_science_students_count_l1093_109327

/-- Represents the number of students in different departments and their local percentages --/
structure CollegeData where
  arts_total : ℕ
  arts_local_percent : ℚ
  science_local_percent : ℚ
  commerce_total : ℕ
  commerce_local_percent : ℚ
  total_local_percent : ℚ

/-- Calculates the number of science students based on the given college data --/
def calculate_science_students (data : CollegeData) : ℚ :=
  let arts_local := data.arts_total * data.arts_local_percent
  let commerce_local := data.commerce_total * data.commerce_local_percent
  let total_local := (data.arts_total + data.commerce_total) * data.total_local_percent
  (total_local - arts_local - commerce_local) / data.science_local_percent

/-- Theorem stating that the number of science students is 100 given the specified conditions --/
theorem science_students_count (data : CollegeData) 
  (h1 : data.arts_total = 400)
  (h2 : data.arts_local_percent = 1/2)
  (h3 : data.science_local_percent = 1/4)
  (h4 : data.commerce_total = 120)
  (h5 : data.commerce_local_percent = 17/20)
  (h6 : data.total_local_percent = 327/100) : 
  calculate_science_students data = 100 := by
  sorry

#eval calculate_science_students { 
  arts_total := 400, 
  arts_local_percent := 1/2, 
  science_local_percent := 1/4, 
  commerce_total := 120, 
  commerce_local_percent := 17/20, 
  total_local_percent := 327/100 
}

end NUMINAMATH_CALUDE_ERRORFEEDBACK_science_students_count_l1093_109327


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_diameter_and_circumference_l1093_109354

-- Define the circle's area
noncomputable def circle_area : ℝ := 225 * Real.pi

-- Theorem stating the diameter and circumference of the circle
theorem circle_diameter_and_circumference :
  ∃ (radius : ℝ),
    radius^2 * Real.pi = circle_area ∧
    2 * radius = 30 ∧
    2 * Real.pi * radius = 30 * Real.pi := by
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_diameter_and_circumference_l1093_109354


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_complex_l1093_109336

theorem max_distance_complex (z : ℂ) (h : Complex.abs z = 3) :
  ∃ w : ℂ, Complex.abs w = 3 ∧
    Complex.abs ((1 + 2*Complex.I) * w^3 - w^4) = 216 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_complex_l1093_109336


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_squares_of_roots_l1093_109365

theorem sum_of_squares_of_roots : ∀ (r s t : ℝ),
  (r ≥ 0) → (s ≥ 0) → (t ≥ 0) →
  (r * Real.sqrt r - 8 * r + 9 * Real.sqrt r + 2 = 0) →
  (s * Real.sqrt s - 8 * s + 9 * Real.sqrt s + 2 = 0) →
  (t * Real.sqrt t - 8 * t + 9 * Real.sqrt t + 2 = 0) →
  r^2 + s^2 + t^2 = 46 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_squares_of_roots_l1093_109365


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l1093_109304

noncomputable def f (x : ℝ) := Real.exp (x - 1) + x^3 - 2

def g (a x : ℝ) := x^2 - a*x - a + 3

theorem range_of_a :
  ∃ (m n a : ℝ),
    f m = 0 ∧
    g a n = 0 ∧
    |m - n| ≤ 1 ∧
    ∀ a', (∃ n', g a' n' = 0 ∧ |m - n'| ≤ 1) → 2 ≤ a' ∧ a' ≤ 3 :=
by
  sorry

#check range_of_a

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l1093_109304


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_closest_point_on_line_l1093_109353

noncomputable section

/-- The line y = -4x + 3 --/
def line (x : ℝ) : ℝ := -4 * x + 3

/-- The point we're finding the closest point to --/
def point : ℝ × ℝ := (1, 3)

/-- The proposed closest point on the line --/
def closest_point : ℝ × ℝ := (-1/17, 55/17)

/-- Theorem stating that closest_point is the point on the line closest to point --/
theorem closest_point_on_line :
  ∀ x : ℝ, 
  (x, line x) ≠ closest_point →
  (point.1 - closest_point.1)^2 + (point.2 - closest_point.2)^2 < 
  (point.1 - x)^2 + (point.2 - line x)^2 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_closest_point_on_line_l1093_109353


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_inequality_l1093_109364

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 0 then 2^(-x) else 1

-- State the theorem
theorem range_of_inequality (x : ℝ) :
  f (x + 1) < f (2 * x) ↔ x < 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_inequality_l1093_109364


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_canal_properties_l1093_109331

/-- Represents the properties of a trapezoidal canal -/
structure TrapezoidalCanal where
  length : ℝ
  cross_sectional_area : ℝ
  depth : ℝ
  top_width : ℝ
  bottom_width : ℝ

/-- Defines the canal according to the problem conditions -/
def canal : TrapezoidalCanal where
  length := 750
  cross_sectional_area := 1.6
  depth := 0.8  -- This is derived from the solution, but it's necessary for the structure
  top_width := 0.8 + 2
  bottom_width := 0.8 + 0.4

/-- The daily excavation rate in cubic meters -/
def daily_excavation_rate : ℝ := 48

/-- Theorem stating the properties of the canal and the time to excavate it -/
theorem canal_properties :
  canal.top_width = 2.8 ∧
  canal.bottom_width = 1.2 ∧
  (canal.length * canal.cross_sectional_area) / daily_excavation_rate = 25 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_canal_properties_l1093_109331


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_special_intersection_l1093_109361

/-- Represents a hyperbola with semi-major axis a and semi-minor axis b -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  h_pos_a : a > 0
  h_pos_b : b > 0

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- The eccentricity of a hyperbola -/
noncomputable def eccentricity (h : Hyperbola) : ℝ :=
  Real.sqrt (1 + h.b^2 / h.a^2)

/-- The right focus of a hyperbola -/
noncomputable def right_focus (h : Hyperbola) : Point :=
  ⟨h.a * eccentricity h, 0⟩

/-- Membership of a point on a hyperbola -/
def on_hyperbola (p : Point) (h : Hyperbola) : Prop :=
  (p.x^2 / h.a^2) - (p.y^2 / h.b^2) = 1

/-- Theorem: If a line through the right focus of a hyperbola intersects
    the right branch at points A and B such that triangle F₁AB is right-angled
    with the right angle at A, then the eccentricity is 5 - 2√2 -/
theorem hyperbola_special_intersection (h : Hyperbola)
  (A B : Point) (hA : on_hyperbola A h) (hB : on_hyperbola B h)
  (hline : ∃ t, B = ⟨A.x + t * (A.x - (right_focus h).x), A.y + t * (A.y - (right_focus h).y)⟩)
  (hright_angle : (A.x - (right_focus h).x) * (B.x - A.x) + (A.y - (right_focus h).y) * (B.y - A.y) = 0) :
  eccentricity h = 5 - 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_special_intersection_l1093_109361


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integer_root_cases_l1093_109357

/-- The number of real values of a for which x^2 + ax + 9a = 0 has only integer roots --/
def count_integer_root_cases : ℕ := 10

/-- A pair of integers representing the roots of the quadratic equation --/
structure IntegerRoots where
  r : ℤ
  s : ℤ

/-- The set of all valid integer root pairs --/
def valid_integer_roots : Set IntegerRoots :=
  { roots | 
    (roots.r + 9) * (roots.s + 9) = 81 ∨
    (roots.r + 9) * (roots.s + 9) = -81
  }

/-- The theorem stating the number of cases where x^2 + ax + 9a = 0 has only integer roots --/
theorem integer_root_cases : 
  ∃ (S : Finset IntegerRoots), 
    (∀ roots ∈ S, roots ∈ valid_integer_roots) ∧ 
    (Finset.card (Finset.image (λ roots => -(roots.r + roots.s)) S) = count_integer_root_cases) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integer_root_cases_l1093_109357


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_at_one_l1093_109343

-- Define the function f(x) = x^α
noncomputable def f (α : ℝ) (x : ℝ) : ℝ := x^α

-- State the theorem
theorem derivative_at_one (α : ℝ) :
  f α 2 = 8 → deriv (f α) 1 = 3 := by
  intro h
  -- The proof steps would go here, but we'll use sorry for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_at_one_l1093_109343


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_problem_l1093_109340

/-- An arithmetic sequence -/
structure ArithmeticSequence where
  a : ℕ → ℚ
  d : ℚ
  h : ∀ n, a (n + 1) = a n + d

/-- Sum of first n terms of an arithmetic sequence -/
noncomputable def sum_n (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  (n : ℚ) * (2 * seq.a 1 + (n - 1) * seq.d) / 2

theorem arithmetic_sequence_problem (seq : ArithmeticSequence) 
  (h1 : seq.a 7 + seq.a 9 = 10)
  (h2 : sum_n seq 11 = 11) :
  seq.a 10 = 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_problem_l1093_109340


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_digging_efficiency_theorem_l1093_109325

/-- Represents the digging capability of a team --/
structure DiggingTeam where
  size : ℕ
  trenchLength : ℝ
  individualRate : ℝ

/-- Calculates the length of trench that could be dug if all team members worked together --/
noncomputable def totalDiggingLength (team : DiggingTeam) : ℝ :=
  let totalTime := team.trenchLength / team.individualRate
  team.size * team.individualRate * totalTime

/-- The main theorem --/
theorem digging_efficiency_theorem (team : DiggingTeam) 
  (h1 : team.size = 5)
  (h2 : team.trenchLength = 30)
  (h3 : team.individualRate * 4 = team.trenchLength / 3) :
  totalDiggingLength team = 80 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_digging_efficiency_theorem_l1093_109325


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_route_gender_related_prob_reward_one_each_l1093_109384

/-- Represents the contingency table for route choice and gender --/
structure ContingencyTable :=
  (male_A male_B female_A female_B : ℕ)

/-- Calculates the chi-square statistic for a 2x2 contingency table --/
noncomputable def chi_square (table : ContingencyTable) : ℝ :=
  let n := table.male_A + table.male_B + table.female_A + table.female_B
  let ad := table.male_A * table.female_B
  let bc := table.male_B * table.female_A
  (n * (ad - bc)^2 : ℝ) / ((table.male_A + table.male_B) * 
    (table.female_A + table.female_B) * 
    (table.male_A + table.female_A) * 
    (table.male_B + table.female_B))

/-- The critical value for chi-square test at p = 0.001 --/
def critical_value : ℝ := 10.828

/-- Theorem stating that the chi-square statistic is greater than the critical value --/
theorem route_gender_related (table : ContingencyTable) 
  (h1 : table.male_A + table.male_B = 120)
  (h2 : table.female_A + table.female_B = 180)
  (h3 : table.male_A + table.female_A = 150)
  (h4 : table.male_B + table.female_B = 150) :
  chi_square table > critical_value := by
  sorry

/-- Calculates the probability of selecting exactly one male and one female --/
def prob_one_male_one_female (total males : ℕ) : ℚ :=
  (males * (total - males)) / (total * (total - 1) / 2)

/-- Theorem stating that the probability of selecting one male and one female is 3/5 --/
theorem prob_reward_one_each :
  prob_one_male_one_female 5 2 = 3/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_route_gender_related_prob_reward_one_each_l1093_109384


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_ratio_of_edge_length_ratio_l1093_109338

/-- Regular tetrahedron in 3D space -/
structure RegularTetrahedron where
  edge_length : ℝ
  edge_length_pos : edge_length > 0

/-- Volume of a regular tetrahedron -/
noncomputable def volume (t : RegularTetrahedron) : ℝ :=
  (t.edge_length ^ 3) * (Real.sqrt 2) / 12

/-- Theorem: Volume ratio of two regular tetrahedrons with edge length ratio 1:2 is 1:8 -/
theorem volume_ratio_of_edge_length_ratio (t1 t2 : RegularTetrahedron)
  (h : t2.edge_length = 2 * t1.edge_length) :
  volume t2 = 8 * volume t1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_ratio_of_edge_length_ratio_l1093_109338


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_circle_intersection_l1093_109321

-- Define the line equation
def line_eq (a x y : ℝ) : Prop := x + a * y + 2 = 0

-- Define the circle equation
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 + 2*x - 2*y + 1 = 0

-- Define the theorem
theorem line_circle_intersection (a : ℝ) :
  (∃ x y : ℝ, line_eq a x y ∧ circle_eq x y) → a ≤ 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_circle_intersection_l1093_109321


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_tosses_after_60_seconds_l1093_109318

/-- The number of balls Marisela has at time t -/
def num_balls (t : ℕ) : ℕ := t / 5 + 1

/-- The total number of tosses Marisela has made up to time t -/
def total_tosses (t : ℕ) : ℕ :=
  Finset.sum (Finset.range (t + 1)) (λ i => num_balls i)

/-- The theorem stating that the total number of tosses after 60 seconds is 390 -/
theorem total_tosses_after_60_seconds :
  total_tosses 60 = 390 := by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_tosses_after_60_seconds_l1093_109318


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_profit_on_10th_day_l1093_109307

-- Define the profit function
noncomputable def profit (x : ℝ) : ℝ :=
  if 1 ≤ x ∧ x < 30 then
    -2 * x^2 + 40 * x + 3000
  else if 30 ≤ x ∧ x ≤ 50 then
    -120 * x + 6000
  else
    0

-- Define the domain of x
def valid_x (x : ℝ) : Prop :=
  1 ≤ x ∧ x ≤ 50

-- Theorem statement
theorem max_profit_on_10th_day :
  ∃ (x : ℝ), valid_x x ∧ 
    (∀ (y : ℝ), valid_x y → profit y ≤ profit x) ∧
    x = 10 ∧ profit x = 3200 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_profit_on_10th_day_l1093_109307


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_five_subset_M_l1093_109388

noncomputable def M : Set ℝ := {x | x > 2}
noncomputable def a : ℝ := Real.sqrt 5

theorem sqrt_five_subset_M : {a} ⊆ M := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_five_subset_M_l1093_109388


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_sixty_degrees_l1093_109368

/-- The cosine of 60 degrees is equal to 1/2. -/
theorem cos_sixty_degrees :
  Real.cos (60 * Real.pi / 180) = 1/2 := by
  -- We define the angle in radians
  let angle : Real := 60 * Real.pi / 180
  -- We define the point on the unit circle corresponding to 60 degrees
  let unit_circle_point : (Real × Real) := (1/2, Real.sqrt 3 / 2)
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_sixty_degrees_l1093_109368


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_when_a_zero_one_zero_iff_a_positive_l1093_109309

-- Define the function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x - 1 / x - (a + 1) * Real.log x

-- Part 1: Maximum value when a = 0
theorem max_value_when_a_zero :
  ∃ (max : ℝ), max = -1 ∧ ∀ x > 0, f 0 x ≤ max :=
sorry

-- Part 2: Range of a for exactly one zero
theorem one_zero_iff_a_positive :
  ∀ a : ℝ, (∃! x, x > 0 ∧ f a x = 0) ↔ a > 0 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_when_a_zero_one_zero_iff_a_positive_l1093_109309


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_translation_for_even_function_l1093_109362

noncomputable def determinant (a b c d : ℝ) : ℝ := a * d - b * c

noncomputable def f (x : ℝ) : ℝ := determinant (Real.sqrt 3) (Real.sin x) 1 (Real.cos x)

theorem min_translation_for_even_function :
  ∀ t : ℝ, t > 0 →
  (∀ x : ℝ, f (x + t) = f (-x + t)) →
  t ≥ 5 * Real.pi / 6 :=
by
  sorry

#check min_translation_for_even_function

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_translation_for_even_function_l1093_109362


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distinct_parenthesizations_all_parenthesizations_different_l1093_109306

-- Define a type for the possible parenthesizations
inductive Parenthesization
  | Standard
  | Left
  | Middle
  | Right
  | LeftMiddle
deriving Fintype, Repr

-- Define a function to evaluate the expression based on parenthesization
noncomputable def evaluate (p : Parenthesization) : ℕ :=
  match p with
  | Parenthesization.Standard => 3^(3^(3^3))
  | Parenthesization.Left => ((3^3)^3)^3
  | Parenthesization.Middle => 3^((3^3)^3)
  | Parenthesization.Right => 3^(3^(3^3))
  | Parenthesization.LeftMiddle => (3^3)^(3^3)

-- Theorem stating that there are exactly 3 distinct values other than the standard result
theorem distinct_parenthesizations :
  (Finset.filter (fun p => evaluate p ≠ evaluate Parenthesization.Standard)
    Finset.univ).card = 3 := by
  sorry

-- Theorem stating that all parenthesizations yield different results
theorem all_parenthesizations_different :
  ∀ p q : Parenthesization, p ≠ q → evaluate p ≠ evaluate q := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distinct_parenthesizations_all_parenthesizations_different_l1093_109306


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_student_attention_analysis_l1093_109378

-- Define the piecewise function f(x)
noncomputable def f (x : ℝ) : ℝ :=
  if 0 < x ∧ x ≤ 10 then -0.1 * x^2 + 2.6 * x + 43
  else if 10 < x ∧ x ≤ 16 then 59
  else if 16 < x ∧ x ≤ 30 then -3 * x + 107
  else 0  -- Default value for x outside the specified ranges

-- Theorem statement
theorem student_attention_analysis :
  (f 5 > f 20) ∧
  (∀ x : ℝ, 0 < x ∧ x ≤ 30 → f x ≤ f 10) ∧
  (∀ x : ℝ, 10 < x ∧ x ≤ 16 → f x = f 10) ∧
  (¬ ∃ t : ℝ, 0 < t ∧ t ≤ 17 ∧ (∀ x : ℝ, t ≤ x ∧ x ≤ t + 13 → f x ≥ 55)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_student_attention_analysis_l1093_109378


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_little_d_can_win_l1093_109397

/-- Represents a point in the 3D lattice -/
structure LatticePoint where
  x : Int
  y : Int
  z : Int

/-- Represents a plane perpendicular to one of the coordinate axes -/
inductive Plane
  | XY (z : Int)
  | YZ (x : Int)
  | XZ (y : Int)

/-- Represents a player's move -/
inductive Move
  | LittleD (point : LatticePoint)
  | BigZ (plane : Plane)

/-- The game state -/
structure GameState where
  shoeLocs : List LatticePoint
  munchedPlanes : List Plane

/-- Checks if a point is on a munched plane -/
def isOnMunchedPlane (point : LatticePoint) (planes : List Plane) : Bool := sorry

/-- Checks if a move is valid given the current game state -/
def isValidMove (move : Move) (state : GameState) : Bool := sorry

/-- Updates the game state after a move -/
def updateState (state : GameState) (move : Move) : GameState := sorry

/-- Checks if there are n consecutive shoe locations on a line parallel to a coordinate axis -/
def hasConsecutiveShoes (n : Nat) (state : GameState) : Bool := sorry

/-- The main theorem: Little D can always win for any n -/
theorem little_d_can_win (n : Nat) : 
  ∃ (strategy : List Move), 
    (∀ (opponent_moves : List Move), 
      let game_moves := strategy.zipWith (λ s o => [s, o]) opponent_moves
      let final_state := game_moves.foldl (λ state moves => moves.foldl updateState state) ⟨[], []⟩
      hasConsecutiveShoes n final_state) := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_little_d_can_win_l1093_109397


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_diamond_properties_l1093_109342

-- Define the set T as all real numbers except -1 and 0
def T : Set ℝ := {x : ℝ | x ≠ -1 ∧ x ≠ 0}

-- Define the diamond operation
def diamond (x y : ℝ) : ℝ := 3 * x * y + x + y

theorem diamond_properties :
  (∀ (x y : ℝ), x ∈ T → y ∈ T → diamond x y = diamond y x) ∧ 
  (∃ (x y z : ℝ), x ∈ T ∧ y ∈ T ∧ z ∈ T ∧ diamond (diamond x y) z ≠ diamond x (diamond y z)) ∧
  (¬ ∃ (e : ℝ), e ∈ T ∧ ∀ (x : ℝ), x ∈ T → diamond x e = x ∧ diamond e x = x) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_diamond_properties_l1093_109342


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_and_min_a1_l1093_109301

-- Define the arithmetic sequence a_n
def arithmetic_sequence (a₁ d : ℝ) : ℕ → ℝ := λ n ↦ a₁ + (n - 1 : ℝ) * d

-- Define the sequence b_n
def b_sequence (a : ℕ → ℝ) : ℕ → ℝ := λ n ↦ a (n + 1) * a (n + 2) - a n ^ 2

theorem arithmetic_and_min_a1 (a₁ d : ℝ) (s t : ℕ+) :
  let a := arithmetic_sequence a₁ d
  let b := b_sequence a
  d ≠ 0 ∧
  (∃ k : ℤ, a s + b t = k) ∧
  (∀ n : ℕ, b (n + 1) - b n = d) →
  (∀ n : ℕ, b (n + 1) - b n = 3 * d ^ 2) ∧
  |a₁| ≥ 1 / 18 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_and_min_a1_l1093_109301


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jasmine_max_cards_l1093_109372

/-- The maximum number of trading cards Jasmine can purchase -/
def max_cards (total_money : ℚ) (card_cost : ℚ) : ℕ :=
  (total_money / card_cost).floor.toNat

/-- Theorem: Given $9.00 and $1.00 per card, Jasmine can buy at most 9 cards -/
theorem jasmine_max_cards :
  max_cards 9 1 = 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_jasmine_max_cards_l1093_109372


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polar_line_at_angle_l1093_109377

/-- The set of points in polar coordinates satisfying θ = π/6 forms a line making an angle with the x-axis -/
theorem polar_line_at_angle : 
  ∃ (m b : ℝ), ∀ (r x y : ℝ), 
    let θ : ℝ := π/6
    (x = r * Real.cos θ ∧ y = r * Real.sin θ) → y = m * x + b :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polar_line_at_angle_l1093_109377


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_property_l1093_109399

noncomputable def geometric_sequence (a q : ℝ) : ℕ → ℝ := fun n => a * q ^ (n - 1)

noncomputable def geometric_sum (a q : ℝ) (n : ℕ) : ℝ :=
  if q = 1 then n * a else a * (1 - q^n) / (1 - q)

theorem geometric_sequence_property (a q : ℝ) :
  let S := geometric_sum a q
  (S 18 : ℝ) / (S 9 : ℝ) = 7 / 8 →
  (∃ d : ℝ, S 9 - S 3 = S 6 - S 9 ∧ S 9 = (S 3 + S 6) / 2) ∧
  (geometric_sequence a q 4 = (geometric_sequence a q 7 + geometric_sequence a q 10) / 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_property_l1093_109399


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_condition_implies_range_of_a_l1093_109398

-- Define the sets A and B
def A : Set ℝ := {x | x^2 + 4*x = 0}
def B (a : ℝ) : Set ℝ := {x | x^2 + 2*(a+1)*x + a^2 - 1 = 0}

-- Define the range of a
def range_of_a : Set ℝ := {1} ∪ Set.Ici (-1)

-- State the theorem
theorem intersection_condition_implies_range_of_a :
  ∀ a : ℝ, (A ∩ B a = B a) → a ∈ range_of_a :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_condition_implies_range_of_a_l1093_109398


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_equals_four_thirds_pi_l1093_109329

open Real
open MeasureTheory

-- Define the function that bounds the region
def f (x : ℝ) : ℝ := (x - 1)^2

-- Define the volume of the solid of revolution
noncomputable def volume_of_solid : ℝ :=
  Real.pi * ∫ y in Set.Icc 0 1, (2^2 - (1 + sqrt y)^2) + (1 - sqrt y)^2

-- Theorem statement
theorem volume_equals_four_thirds_pi : volume_of_solid = (4 / 3) * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_equals_four_thirds_pi_l1093_109329


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_female_population_l1093_109308

/-- Proves that in a town with a total population of 600, where males represent one-third of the population, the number of females is 400. -/
theorem female_population (total_population : ℕ) (male_fraction : ℚ) (female_population : ℕ) : 
  total_population = 600 →
  male_fraction = 1 / 3 →
  female_population = total_population - (male_fraction * ↑total_population).floor →
  female_population = 400 := by
  sorry

#check female_population

end NUMINAMATH_CALUDE_ERRORFEEDBACK_female_population_l1093_109308


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_p_income_l1093_109394

noncomputable def average_income (x y : ℝ) : ℝ := (x + y) / 2

theorem p_income 
  (p q r : ℝ)
  (h1 : average_income p q = 5050)
  (h2 : average_income q r = 6250)
  (h3 : average_income p r = 5200)
  : p = 4000 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_p_income_l1093_109394


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_period_of_tan_shifted_l1093_109339

open Real

-- Define the function (marked as noncomputable due to dependency on Real.tan)
noncomputable def f (x : ℝ) : ℝ := tan (x + π / 3)

-- State the theorem
theorem smallest_positive_period_of_tan_shifted :
  ∃ (T : ℝ), T > 0 ∧ (∀ y, f (y + T) = f y) ∧
  (∀ S, S > 0 → (∀ y, f (y + S) = f y) → T ≤ S) ∧
  T = π :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_period_of_tan_shifted_l1093_109339


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_speed_calculation_l1093_109358

/-- Calculates the speed in km/h given distance in meters and time in minutes -/
noncomputable def calculate_speed (distance_meters : ℝ) (time_minutes : ℝ) : ℝ :=
  (distance_meters / 1000) / (time_minutes / 60)

/-- Proves that crossing 300 meters in 4 minutes results in a speed of 4.5 km/h -/
theorem speed_calculation :
  calculate_speed 300 4 = 4.5 := by
  -- Unfold the definition of calculate_speed
  unfold calculate_speed
  -- Simplify the expression
  simp [div_div_eq_mul_div]
  -- Evaluate the numerical expression
  norm_num
  -- QED

end NUMINAMATH_CALUDE_ERRORFEEDBACK_speed_calculation_l1093_109358


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_ratio_sum_l1093_109392

noncomputable section

/-- The equation of the ellipse -/
def ellipse_eq (x y : ℝ) : Prop :=
  3 * x^2 + 2 * x * y + 4 * y^2 - 15 * x - 25 * y + 55 = 0

/-- The ellipse is in the first quadrant -/
def first_quadrant (x y : ℝ) : Prop :=
  x > 0 ∧ y > 0

/-- The set of points on the ellipse in the first quadrant -/
def ellipse_points : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | ellipse_eq p.1 p.2 ∧ first_quadrant p.1 p.2}

/-- The ratio y/x for a point (x,y) -/
def ratio (p : ℝ × ℝ) : ℝ :=
  p.2 / p.1

theorem ellipse_ratio_sum :
  ∃ (c d : ℝ), (∀ p ∈ ellipse_points, ratio p ≤ c ∧ ratio p ≥ d) ∧ c + d = 62 / 51 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_ratio_sum_l1093_109392


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_c_min_value_c_achieved_l1093_109349

theorem min_value_c (a b c : ℝ) (h1 : (2 : ℝ)^a + (4 : ℝ)^b = (2 : ℝ)^c) (h2 : (4 : ℝ)^a + (2 : ℝ)^b = (4 : ℝ)^c) :
  c ≥ Real.log 3 / Real.log 2 - 5/3 :=
by sorry

theorem min_value_c_achieved (ε : ℝ) (hε : ε > 0) :
  ∃ a b c : ℝ, (2 : ℝ)^a + (4 : ℝ)^b = (2 : ℝ)^c ∧ (4 : ℝ)^a + (2 : ℝ)^b = (4 : ℝ)^c ∧ c < Real.log 3 / Real.log 2 - 5/3 + ε :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_c_min_value_c_achieved_l1093_109349


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_projection_area_of_specific_tetrahedron_l1093_109319

/-- Represents a tetrahedron with specific properties -/
structure Tetrahedron where
  side_length : ℝ
  dihedral_angle : ℝ

/-- The maximum projection area of a rotating tetrahedron -/
noncomputable def max_projection_area (t : Tetrahedron) : ℝ := 
  (9 * Real.sqrt 3) / 4

/-- Theorem stating the maximum projection area of a specific tetrahedron -/
theorem max_projection_area_of_specific_tetrahedron :
  let t : Tetrahedron := { side_length := 3, dihedral_angle := π/6 }
  max_projection_area t = (9 * Real.sqrt 3) / 4 := by
  sorry

#check max_projection_area_of_specific_tetrahedron

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_projection_area_of_specific_tetrahedron_l1093_109319


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ratio_c_d_value_l1093_109369

-- Define the variables
variable (a b c d e f : ℝ)

-- Define the given conditions
def condition1 (a b : ℝ) : Prop := a / b = 1 / 3
def condition2 (b c : ℝ) : Prop := b / c = 2
def condition3 (d e : ℝ) : Prop := d / e = 3
def condition4 (e f : ℝ) : Prop := e / f = 1 / 6
def condition5 (a b c d e f : ℝ) : Prop := a * b * c / (d * e * f) = 1 / 4

-- Define the theorem
theorem ratio_c_d_value 
  (h1 : condition1 a b)
  (h2 : condition2 b c)
  (h3 : condition3 d e)
  (h4 : condition4 e f)
  (h5 : condition5 a b c d e f) :
  ∃ (ε : ℝ), abs (c / d - 0.2887) < ε ∧ ε > 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ratio_c_d_value_l1093_109369


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_mumbo_yumbo_alphabet_l1093_109337

/-- Represents an alphabet as a list of characters -/
def Alphabet := List Char

/-- The word central to the Mumbo-Yumbo tribe's language -/
def centralWord : String := "Mumbo-Jumbo"

/-- Checks if all characters in a string are present in an alphabet -/
def containsAllChars (a : Alphabet) (s : String) : Prop :=
  s.toList.all (λ c => a.contains c)

/-- Checks if an alphabet has a defined order (no duplicates) -/
def hasDefinedOrder (a : Alphabet) : Prop :=
  a.Nodup

/-- Theorem stating that a valid Mumbo-Yumbo alphabet contains all letters from the central word
    and has a defined order -/
theorem valid_mumbo_yumbo_alphabet (a : Alphabet) :
  containsAllChars a centralWord ∧ hasDefinedOrder a →
  a.length ≥ centralWord.toList.eraseDups.length := by
  sorry

#check valid_mumbo_yumbo_alphabet

end NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_mumbo_yumbo_alphabet_l1093_109337


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_drug_a_effective_expected_value_drug_a_ineffective_l1093_109316

-- Define the contingency table
def contingency_table : Matrix (Fin 2) (Fin 2) ℕ :=
  ![![30, 20],
    ![10, 40]]

-- Define the K² formula
def k_squared (table : Matrix (Fin 2) (Fin 2) ℕ) : ℚ :=
  let n := (table 0 0) + (table 0 1) + (table 1 0) + (table 1 1)
  let a := table 0 0
  let b := table 0 1
  let c := table 1 0
  let d := table 1 1
  (n * (a * d - b * c)^2 : ℚ) / ((a + b) * (c + d) * (a + c) * (b + d))

-- Define the critical value for 99% confidence
def critical_value : ℚ := 6635 / 1000

-- Theorem 1: K² > critical value
theorem drug_a_effective : k_squared contingency_table > critical_value := by sorry

-- Define the binomial distribution
def binomial (n : ℕ) (p : ℚ) (k : ℕ) : ℚ :=
  (n.choose k : ℚ) * p^k * (1 - p)^(n - k)

-- Define the expected value of a binomial distribution
def expected_value (n : ℕ) (p : ℚ) : ℚ := n * p

-- Theorem 2: E(X) = 8/5 for X ~ B(4, 2/5)
theorem expected_value_drug_a_ineffective : expected_value 4 (2/5) = 8/5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_drug_a_effective_expected_value_drug_a_ineffective_l1093_109316


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1093_109335

open Real
open Set

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := 
  Real.cos (ω * x) * Real.sin (ω * x) + Real.sqrt 3 * (Real.cos (ω * x))^2 - Real.sqrt 3 / 2

def g (m : ℝ) (f : ℝ → ℝ) (x : ℝ) : ℝ := 3 * (f x)^2 + m * f x - 1

theorem f_properties (ω : ℝ) (h₁ : 0 < ω) (h₂ : ω ≤ 1) 
  (h₃ : ∀ x, f ω (x + π) = f ω x) :
  (∀ x, f ω x = Real.sin (2*x + π/3)) ∧
  (Icc (-π/12) (5*π/12) ⊆ (f ω) ⁻¹' (Icc (-1/2) 1)) ∧
  (∀ m, 
    (m > 3 → ∀ x ∈ Icc (-π/12) (5*π/12), g m (f ω) x ≥ -1/4 - m/2) ∧
    (-6 ≤ m ∧ m ≤ 3 → ∀ x ∈ Icc (-π/12) (5*π/12), g m (f ω) x ≥ -1 - m^2/12) ∧
    (m < -6 → ∀ x ∈ Icc (-π/12) (5*π/12), g m (f ω) x ≥ 2 + m)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1093_109335


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_inequality_l1093_109322

theorem sin_inequality (d n : ℕ) (hd : d ≥ 1) (hn : n ≥ 1) (h_not_square : ¬ ∃ m : ℕ, d = m * m) :
  (n * Real.sqrt (d : ℝ) + 1) * |Real.sin (n * Real.pi * Real.sqrt (d : ℝ))| ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_inequality_l1093_109322


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_value_theorem_l1093_109352

-- Define the piecewise function
noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ -1 then x + 2
  else if x < 2 then x^2
  else 2*x

-- State the theorem
theorem function_value_theorem (b : ℝ) :
  f b = 1/2 → (b = -3/2 ∨ b = Real.sqrt 2 / 2 ∨ b = -Real.sqrt 2 / 2) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_value_theorem_l1093_109352
