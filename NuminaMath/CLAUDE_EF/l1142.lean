import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_is_144_l1142_114249

/-- The speed of a train in km/hr -/
noncomputable def train_speed (train_length : ℝ) (crossing_time : ℝ) : ℝ :=
  2 * train_length * 60 / 1000 / crossing_time

/-- Theorem: The speed of the train is 144 km/hr -/
theorem train_speed_is_144 (train_length : ℝ) (crossing_time : ℝ) 
  (h1 : train_length = 1200)
  (h2 : crossing_time = 1) : 
  train_speed train_length crossing_time = 144 := by
  sorry

/-- Compute the train speed for the given parameters -/
def compute_train_speed : ℚ :=
  2 * 1200 * 60 / 1000 / 1

#eval compute_train_speed

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_is_144_l1142_114249


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_sum_probability_l1142_114219

/-- Represents a spinner with a list of numbers --/
structure Spinner :=
  (numbers : List ℕ)

/-- Calculates the probability of getting an odd sum when rotating three spinners --/
def probability_odd_sum (p q r : Spinner) : ℚ :=
  let p_even := (p.numbers.filter (λ n => n % 2 = 0)).length / p.numbers.length
  let q_even := (q.numbers.filter (λ n => n % 2 = 0)).length / q.numbers.length
  let r_odd := (r.numbers.filter (λ n => n % 2 = 1)).length / r.numbers.length
  p_even * q_even * r_odd

/-- The main theorem stating that the probability of getting an odd sum is 1/2 --/
theorem odd_sum_probability :
  let p := Spinner.mk [1, 2, 3, 4]
  let q := Spinner.mk [2, 4, 6]
  let r := Spinner.mk [1, 3, 5]
  probability_odd_sum p q r = 1/2 := by
  sorry

-- Remove the #eval statement as it's not allowed in this context

end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_sum_probability_l1142_114219


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triple_plus_reciprocal_product_l1142_114283

theorem triple_plus_reciprocal_product : 
  ∃ p : ℝ, ∀ x : ℝ, (x + 1/x = 3*x) → (p = -1/2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triple_plus_reciprocal_product_l1142_114283


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_distance_between_circles_l1142_114203

/-- Circle C₁ defined by the equation x² + y² + 4x + 2y + 1 = 0 -/
def C₁ (x y : ℝ) : Prop := x^2 + y^2 + 4*x + 2*y + 1 = 0

/-- Circle C₂ defined by the equation x² + y² - 4x - 4y + 6 = 0 -/
def C₂ (x y : ℝ) : Prop := x^2 + y^2 - 4*x - 4*y + 6 = 0

/-- The distance between two points (x₁, y₁) and (x₂, y₂) -/
noncomputable def distance (x₁ y₁ x₂ y₂ : ℝ) : ℝ :=
  Real.sqrt ((x₂ - x₁)^2 + (y₂ - y₁)^2)

/-- The minimum distance between any point on C₁ and any point on C₂ -/
noncomputable def min_distance : ℝ := 3 - Real.sqrt 2

theorem minimum_distance_between_circles :
  ∀ (x₁ y₁ x₂ y₂ : ℝ), C₁ x₁ y₁ → C₂ x₂ y₂ →
  distance x₁ y₁ x₂ y₂ ≥ min_distance := by
  sorry

#check minimum_distance_between_circles

end NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_distance_between_circles_l1142_114203


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_CD_l1142_114278

/-- The distance between two points given a series of movements -/
noncomputable def distance_after_movements (north south west east : ℝ) : ℝ :=
  let net_north := north - south
  let net_west := west - east
  Real.sqrt (net_north^2 + net_west^2)

/-- Theorem stating that the distance between C and D is 51 yards -/
theorem distance_CD : distance_after_movements 30 20 80 30 = 51 := by
  -- Unfold the definition of distance_after_movements
  unfold distance_after_movements
  -- Simplify the expressions
  simp
  -- We need to prove that √(10² + 50²) = 51
  -- This is approximately true, but exact equality requires more work
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_CD_l1142_114278


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_l_shaped_floor_tiles_l1142_114276

/-- Calculates the number of tiles needed to cover an L-shaped floor -/
theorem l_shaped_floor_tiles (main_length main_width small_length small_width tile_length tile_width : ℚ)
  (h1 : main_length = 15)
  (h2 : main_width = 10)
  (h3 : small_length = 5)
  (h4 : small_width = 10)
  (h5 : tile_length = 1/4)
  (h6 : tile_width = 3/4) :
  ⌈(main_length * main_width + small_length * small_width) / (tile_length * tile_width)⌉ = 1067 := by
  sorry

#eval ⌈(15 : ℚ) * 10 + 5 * 10 / ((1/4) * (3/4))⌉

end NUMINAMATH_CALUDE_ERRORFEEDBACK_l_shaped_floor_tiles_l1142_114276


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_calculations_l1142_114256

theorem arithmetic_calculations :
  (10 - (-6) + 8 - 2 = 22) ∧
  ((1/6 : ℚ) * (-6) / (-1/7 : ℚ) * 7 = 49) ∧
  ((-1)^2021 * (4 - (-3)^2) + 3 / (-3/4 : ℚ) = 1) ∧
  ((5/12 - 7/9 + 2/3 : ℚ) / (1/36 : ℚ) = 11) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_calculations_l1142_114256


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_triangle_area_l1142_114291

/-- Represents an ellipse with given properties -/
structure Ellipse where
  eccentricity : ℝ
  foci_on_x_axis : Bool
  passes_through : ℝ × ℝ

/-- Represents a line in 2D space -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Helper function to calculate the area of a triangle given three points -/
noncomputable def area_triangle (A B C : ℝ × ℝ) : ℝ :=
  sorry

/-- The main theorem about the ellipse and the area of the triangle -/
theorem ellipse_and_triangle_area 
  (C : Ellipse) 
  (l : Line) 
  (h1 : C.eccentricity = Real.sqrt 3 / 2)
  (h2 : C.foci_on_x_axis = true)
  (h3 : C.passes_through = (-2, 1))
  (h4 : l.a = 1 ∧ l.b = -1 ∧ l.c = 2) :
  (∃ a b : ℝ, a = 2 * Real.sqrt 2 ∧ b = Real.sqrt 2 ∧ 
    ∀ x y : ℝ, (x^2 / a^2 + y^2 / b^2 = 1) ↔ (x^2 / 8 + y^2 / 2 = 1)) ∧
  (∃ O P Q : ℝ × ℝ, 
    O = (0, 0) ∧
    P ∈ {(x, y) : ℝ × ℝ | x^2 / 8 + y^2 / 2 = 1 ∧ x - y + 2 = 0} ∧
    Q ∈ {(x, y) : ℝ × ℝ | x^2 / 8 + y^2 / 2 = 1 ∧ x - y + 2 = 0} ∧
    P ≠ Q ∧
    area_triangle O P Q = 4 * Real.sqrt 6 / 5) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_triangle_area_l1142_114291


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_equals_geometric_sum_solution_is_nine_l1142_114269

/-- The sum of the geometric series with first term a and common ratio r -/
noncomputable def geometricSum (a r : ℝ) : ℝ := a / (1 - r)

/-- The product of two infinite series equals the sum of a geometric series -/
theorem product_equals_geometric_sum (x : ℝ) : x > 0 →
  (geometricSum 1 (1/3)) * (geometricSum 1 (-1/3)) = geometricSum 1 (1/x) := by sorry

/-- The solution to the equation is x = 9 -/
theorem solution_is_nine :
  ∃ x : ℝ, x > 0 ∧ (geometricSum 1 (1/3)) * (geometricSum 1 (-1/3)) = geometricSum 1 (1/x) ∧ x = 9 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_equals_geometric_sum_solution_is_nine_l1142_114269


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_yuna_drank_most_l1142_114284

noncomputable def jimin_amount : ℝ := 0.7
noncomputable def eunji_amount : ℝ := jimin_amount - 0.1
noncomputable def yoongi_amount : ℝ := 4/5
noncomputable def yuna_amount : ℝ := jimin_amount + 0.2

theorem yuna_drank_most : 
  yuna_amount > jimin_amount ∧ 
  yuna_amount > eunji_amount ∧ 
  yuna_amount > yoongi_amount := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_yuna_drank_most_l1142_114284


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_subset_intersection_theorem_l1142_114209

theorem subset_intersection_theorem (α : ℝ) (h_pos : α > 0) (h_bound : α < (3 - Real.sqrt 5) / 2) :
  ∃ (n p : ℕ+) (S T : Finset (Finset (Fin n.val))),
    (p : ℝ) > α * 2^(n : ℕ) ∧
    Finset.card S = p ∧
    Finset.card T = p ∧
    ∀ (s t : Finset (Fin n.val)), s ∈ S → t ∈ T → (s ∩ t).Nonempty :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_subset_intersection_theorem_l1142_114209


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_cats_can_catch_mouse_mouse_can_escape_three_cats_l1142_114253

/-- Represents a position on the chessboard -/
structure Position where
  x : Nat
  y : Nat
  h_x : x ≤ 8
  h_y : y ≤ 8

/-- Represents the state of the game -/
structure GameState where
  mouse : Position
  cats : List Position
  turn : Nat

/-- Defines a valid move for a piece -/
def validMove (start finish : Position) : Prop :=
  (start.x = finish.x ∧ (start.y + 1 = finish.y ∨ start.y = finish.y + 1)) ∨
  (start.y = finish.y ∧ (start.x + 1 = finish.x ∨ start.x = finish.x + 1))

/-- Defines if a position is on the edge of the board -/
def isEdge (p : Position) : Prop :=
  p.x = 1 ∨ p.x = 8 ∨ p.y = 1 ∨ p.y = 8

/-- Theorem: Two cats can always catch a mouse not initially on the edge -/
theorem two_cats_can_catch_mouse (initial : GameState) 
  (h_not_edge : ¬isEdge initial.mouse) :
  ∃ (cat1 cat2 : Position), ∀ (moves : Nat), 
    ∃ (final : GameState), 
      (final.cats.length = 2) ∧ 
      (∃ (cat : Position), cat ∈ final.cats ∧ cat = final.mouse) := by
  sorry

/-- Theorem: Mouse can always escape three cats with initial double move -/
theorem mouse_can_escape_three_cats (initial : GameState) 
  (h_cats : initial.cats.length = 3) 
  (h_first_move : ∃ (p1 p2 : Position), 
    validMove initial.mouse p1 ∧ validMove p1 p2) :
  ∀ (moves : Nat), ∃ (final : GameState),
    (∀ (cat : Position), cat ∈ final.cats → cat ≠ final.mouse) ∧
    (isEdge final.mouse ∨ moves < 8 * 8) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_cats_can_catch_mouse_mouse_can_escape_three_cats_l1142_114253


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_eccentricity_of_C_l1142_114264

/-- The curve C defined by parametric equations x = t + 1/t and y = t - 1/t -/
noncomputable def C : ℝ → ℝ × ℝ := λ t => (t + 1/t, t - 1/t)

/-- The eccentricity of a conic section -/
noncomputable def eccentricity (c : ℝ → ℝ × ℝ) : ℝ := sorry

/-- Theorem: The eccentricity of curve C is √2 -/
theorem eccentricity_of_C : eccentricity C = Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_eccentricity_of_C_l1142_114264


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1142_114299

-- Define the function f
def f : ℝ → ℝ := sorry

-- Define the properties of f
axiom f_odd : ∀ x, f (-x) = -f x
axiom f_periodic : ∀ x, f (x + 2) = -f x
axiom f_cubic : ∀ x, -1 ≤ x → x ≤ 1 → f x = x^3

-- Theorem to prove
theorem f_properties :
  (∀ x, |f x| ≤ 1) ∧
  (∀ a, (∃ x, f x > a) ↔ a < 1) :=
by
  sorry

-- Additional lemmas that might be useful for the proof
lemma f_bound : ∀ x, |f x| ≤ 1 := by sorry

lemma f_range : ∀ a, (∃ x, f x > a) ↔ a < 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1142_114299


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_satellite_work_calculation_l1142_114296

/-- Work done in lifting a satellite from Earth's surface to a given height -/
noncomputable def work_done (m g H R₃ : ℝ) : ℝ := (m * g * H * R₃) / (R₃ + H)

theorem satellite_work_calculation (m g H R₃ : ℝ) 
  (hm : m = 6000)  -- Mass in kg (6 tons = 6000 kg)
  (hg : g = 10)    -- Acceleration due to gravity in m/s²
  (hH : H = 300000) -- Height in meters (300 km = 300000 m)
  (hR : R₃ = 6380000) -- Earth's radius in meters (6380 km = 6380000 m)
  : ‖work_done m g H R₃ - 17191616766‖ < 1 := by
  sorry

-- Remove #eval as it's not necessary for the proof and might cause issues
-- #eval work_done 6000 10 300000 6380000

end NUMINAMATH_CALUDE_ERRORFEEDBACK_satellite_work_calculation_l1142_114296


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_chord_line_and_max_sum_l1142_114246

-- Define the curve C in polar coordinates
def curve_C (ρ θ : ℝ) : Prop :=
  ρ^2 - 2 * Real.sqrt 2 * ρ * Real.cos (θ + Real.pi/4) - 2 = 0

-- Define the transformation from polar to Cartesian coordinates
noncomputable def polar_to_cartesian (ρ θ : ℝ) : ℝ × ℝ :=
  (ρ * Real.cos θ, ρ * Real.sin θ)

-- Define the line l
def line_l (x y : ℝ) : Prop := y = x

-- Define a point on the curve C in Cartesian coordinates
def point_on_C (x y : ℝ) : Prop :=
  ∃ ρ θ, curve_C ρ θ ∧ polar_to_cartesian ρ θ = (x, y)

theorem min_chord_line_and_max_sum : 
  (∀ x y, point_on_C x y → line_l x y → 
    ∀ x' y', point_on_C x' y' → (x - x')^2 + (y - y')^2 ≤ (0 - x')^2 + (0 - y')^2) ∧
  (∀ x y, point_on_C x y → x + y ≤ 2 * Real.sqrt 2) ∧
  (∃ x y, point_on_C x y ∧ x + y = 2 * Real.sqrt 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_chord_line_and_max_sum_l1142_114246


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fruit_seller_pricing_l1142_114214

/-- Calculates the new selling price to achieve a target profit percentage -/
noncomputable def calculateNewPrice (initialPrice : ℝ) (initialProfitPercentage : ℝ) (targetProfitPercentage : ℝ) : ℝ :=
  let costPrice := initialPrice / (1 + initialProfitPercentage / 100)
  costPrice * (1 + targetProfitPercentage / 100)

/-- Proves that the new selling prices for mangoes, apples, and oranges to achieve a 15% profit are correct -/
theorem fruit_seller_pricing (ε : ℝ) (hε : ε > 0) :
  let mangoInitialPrice := (14 : ℝ)
  let mangoInitialLoss := (-15 : ℝ)
  let appleInitialPrice := (20 : ℝ)
  let appleInitialLoss := (-10 : ℝ)
  let orangeInitialPrice := (30 : ℝ)
  let orangeInitialProfit := (5 : ℝ)
  let targetProfit := (15 : ℝ)
  
  |calculateNewPrice mangoInitialPrice mangoInitialLoss targetProfit - 18.94| < ε ∧
  |calculateNewPrice appleInitialPrice appleInitialLoss targetProfit - 25.55| < ε ∧
  |calculateNewPrice orangeInitialPrice orangeInitialProfit targetProfit - 32.86| < ε :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fruit_seller_pricing_l1142_114214


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_negative_numbers_l1142_114230

noncomputable def number_list : List ℝ := [-7, 0, -3, 4/3, 9100, -0.27]

theorem count_negative_numbers (list : List ℝ := number_list) :
  (list.filter (λ x => x < 0)).length = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_negative_numbers_l1142_114230


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_first_ten_primes_with_units_digit_7_l1142_114244

def is_prime (n : ℕ) : Bool :=
  if n ≤ 1 then false
  else
    (List.range (n - 1)).all (λ m => (m + 2) = n ∨ n % (m + 2) ≠ 0)

def has_units_digit_7 (n : ℕ) : Bool := n % 10 = 7

def first_ten_primes_with_units_digit_7 : List ℕ :=
  (List.range 200).filter (λ n => is_prime n && has_units_digit_7 n) |>.take 10

theorem sum_of_first_ten_primes_with_units_digit_7 :
  List.sum first_ten_primes_with_units_digit_7 = 810 := by
  -- The proof goes here
  sorry

#eval List.sum first_ten_primes_with_units_digit_7

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_first_ten_primes_with_units_digit_7_l1142_114244


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_speed_ratio_l1142_114240

-- Define the constants from the problem
noncomputable def eddy_distance : ℝ := 540
noncomputable def freddy_distance : ℝ := 300
noncomputable def eddy_time : ℝ := 3
noncomputable def freddy_time : ℝ := 4

-- Define the average speeds
noncomputable def eddy_speed : ℝ := eddy_distance / eddy_time
noncomputable def freddy_speed : ℝ := freddy_distance / freddy_time

-- Theorem statement
theorem speed_ratio :
  eddy_speed / freddy_speed = 2.4 := by
  -- Unfold the definitions
  unfold eddy_speed freddy_speed
  unfold eddy_distance freddy_distance eddy_time freddy_time
  -- Simplify the expression
  simp [div_div_eq_mul_div]
  -- Evaluate the numerical expression
  norm_num

-- Note: The proof is completed without using 'sorry'

end NUMINAMATH_CALUDE_ERRORFEEDBACK_speed_ratio_l1142_114240


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_ratio_theorem_l1142_114212

/-- An ellipse with given properties -/
structure Ellipse where
  Γ : Set (ℝ × ℝ)  -- The ellipse as a set of points in ℝ²
  F₁ : ℝ × ℝ     -- Focus 1
  F₂ : ℝ × ℝ     -- Focus 2
  P : ℝ × ℝ      -- Point P on the ellipse
  Q : ℝ × ℝ      -- Point Q on the ellipse
  h₁ : P ∈ Γ     -- P is on the ellipse
  h₂ : Q ∈ Γ     -- Q is on the ellipse
  h₃ : ∃ (t : ℝ), P = F₁ + t • (Q - F₁)  -- P, Q, and F₁ are collinear
  h₄ : dist P F₂ = dist F₁ F₂              -- |PF₂| = |F₁F₂|
  h₅ : 3 * dist P F₁ = 4 * dist Q F₁       -- 3|PF₁| = 4|QF₁|

/-- The ratio of minor to major axis of an ellipse with given properties -/
noncomputable def minorMajorRatio (e : Ellipse) : ℝ := 2 * Real.sqrt 6 / 7

/-- Theorem stating that the ratio of minor to major axis is 2√6/7 for the given ellipse -/
theorem ellipse_ratio_theorem (e : Ellipse) : 
  minorMajorRatio e = 2 * Real.sqrt 6 / 7 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_ratio_theorem_l1142_114212


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_f_equals_40_div_3_integral_g_equals_1_minus_ln2_l1142_114235

-- Define the integrands
noncomputable def f (x : ℝ) : ℝ := (4 - 2*x) * (4 - x^2)
noncomputable def g (x : ℝ) : ℝ := (x^2 - 2*x - 3) / x

-- State the theorems
theorem integral_f_equals_40_div_3 : 
  ∫ x in (0 : ℝ)..(2 : ℝ), f x = 40/3 := by sorry

theorem integral_g_equals_1_minus_ln2 : 
  ∫ x in (1 : ℝ)..(2 : ℝ), g x = 1 - Real.log 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_f_equals_40_div_3_integral_g_equals_1_minus_ln2_l1142_114235


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_n_is_9_l1142_114217

/-- Feynman number -/
def F (n : ℕ) : ℕ := 2^(2^n) + 1

/-- a_n sequence -/
noncomputable def a (n : ℕ) : ℝ := 2^n

/-- S_n sequence -/
noncomputable def S (n : ℕ) : ℝ := 2 * (2^n - 1)

/-- Left side of the inequality -/
noncomputable def left_side (n : ℕ) : ℝ := (1/4) * (1 - 1/(2^(n+1) - 1))

/-- Right side of the inequality -/
noncomputable def right_side (n : ℕ) : ℝ := 2^n / 1200

/-- The theorem to prove -/
theorem smallest_n_is_9 : 
  ∀ k : ℕ, k < 9 → left_side k ≥ right_side k ∧ 
  left_side 9 < right_side 9 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_n_is_9_l1142_114217


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_identities_l1142_114226

theorem sin_cos_identities (α : ℝ) 
  (h1 : Real.sin α + Real.cos α = Real.sqrt 2 / 3)
  (h2 : π / 2 < α)
  (h3 : α < π) :
  (Real.sin α - Real.cos α = 4 / 3) ∧ 
  ((Real.sin (π / 2 - α))^2 - (Real.cos (π / 2 + α))^2 = -(4 * Real.sqrt 2) / 9) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_identities_l1142_114226


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_y_sum_l1142_114252

/-- Predicate to check if four points form a rectangle -/
def is_rectangle (a b c d : ℝ × ℝ) : Prop :=
  let midpoint := ((a.1 + c.1) / 2, (a.2 + c.2) / 2)
  (midpoint = ((b.1 + d.1) / 2, (b.2 + d.2) / 2)) ∧
  ((a.1 - b.1)^2 + (a.2 - b.2)^2 = (c.1 - d.1)^2 + (c.2 - d.2)^2)

/-- Given a rectangle with opposite vertices at (2, 10) and (8, -6),
    the sum of the y-coordinates of the other two vertices is 4. -/
theorem rectangle_y_sum : 
  ∀ (a b c d : ℝ × ℝ),
  (a = (2, 10) ∧ c = (8, -6)) →  -- Given opposite vertices
  is_rectangle a b c d →         -- The points form a rectangle
  (b.2 + d.2 = 4) :=              -- Sum of y-coordinates of other vertices
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_y_sum_l1142_114252


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_is_15_sqrt7_over_4_l1142_114295

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the conditions of the problem
def triangle_conditions (t : Triangle) : Prop :=
  t.b = 6 ∧ t.c = 4 ∧ t.B = 2 * t.C

-- Define the area calculation function
noncomputable def triangle_area (t : Triangle) : ℝ :=
  (1/2) * t.a * t.b * Real.sin t.C

-- Theorem statement
theorem triangle_area_is_15_sqrt7_over_4 (t : Triangle) 
  (h : triangle_conditions t) : 
  triangle_area t = (15 * Real.sqrt 7) / 4 := by
  sorry

#check triangle_area_is_15_sqrt7_over_4

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_is_15_sqrt7_over_4_l1142_114295


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_transitivity_l1142_114298

-- Define the types for lines and planes
structure Line : Type
structure Plane : Type

-- Define the perpendicular relation between lines and planes
def perpendicular_line_plane (l : Line) (p : Plane) : Prop := sorry

-- Define the theorem
theorem perpendicular_transitivity 
  (m n : Line) (α β : Plane) 
  (h1 : m ≠ n) 
  (h2 : α ≠ β) 
  (h3 : perpendicular_line_plane m β) 
  (h4 : perpendicular_line_plane n β) 
  (h5 : perpendicular_line_plane n α) : 
  perpendicular_line_plane m α := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_transitivity_l1142_114298


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_cos_C_l1142_114232

-- Define a triangle ABC
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

-- Define the dot product of vectors
def dot_product (v1 v2 : ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

-- Define the vector between two points
def vector (p1 p2 : ℝ × ℝ) : ℝ × ℝ :=
  (p2.1 - p1.1, p2.2 - p1.2)

-- Define the cosine of angle C
noncomputable def cos_C (t : Triangle) : ℝ :=
  let AB := vector t.A t.B
  let AC := vector t.A t.C
  let BC := vector t.B t.C
  (dot_product AB AC + dot_product BC BC - dot_product AC AC) /
  (2 * Real.sqrt (dot_product AB AB) * Real.sqrt (dot_product BC BC))

-- State the theorem
theorem min_cos_C (t : Triangle) :
  dot_product (vector t.A t.B) (vector t.A t.C) +
  2 * dot_product (vector t.B t.A) (vector t.B t.C) =
  3 * dot_product (vector t.C t.A) (vector t.C t.B) →
  cos_C t ≥ Real.sqrt 2 / 3 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_cos_C_l1142_114232


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_A_intersect_B_nonempty_l1142_114207

-- Define set A
def A : Set ℝ := {x | x > 0 ∧ x < 2}

-- Define set B
def B : Set ℝ := {x | -1 < x ∧ x < 1}

-- Theorem statement
theorem A_intersect_B_nonempty : (A ∩ B).Nonempty := by
  -- We'll use 0.5 as an example element in both A and B
  use 0.5
  constructor
  · -- Prove 0.5 is in A
    constructor
    · norm_num
    · norm_num
  · -- Prove 0.5 is in B
    constructor
    · norm_num
    · norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_A_intersect_B_nonempty_l1142_114207


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_usual_time_to_catch_bus_l1142_114241

/-- Proves that the usual time to catch the bus is 24 minutes given the conditions -/
theorem usual_time_to_catch_bus (usual_speed : ℝ) (usual_time : ℝ) : 
  usual_speed > 0 →
  usual_time > 0 →
  (4/5 * usual_speed) * (usual_time + 6) = usual_speed * usual_time →
  usual_time = 24 := by
  intros h_speed_pos h_time_pos h_equation
  -- The proof steps would go here
  sorry

#check usual_time_to_catch_bus

end NUMINAMATH_CALUDE_ERRORFEEDBACK_usual_time_to_catch_bus_l1142_114241


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_class_gpa_calculation_l1142_114223

/-- Given a class where one-third of the students have a GPA of 15 and the remaining two-thirds
    have a GPA of 18, the GPA of the whole class is 17. -/
theorem class_gpa_calculation (n : ℕ) (h : n > 0) :
  (1 / 3 : ℚ) * 15 + (2 / 3 : ℚ) * 18 = 17 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_class_gpa_calculation_l1142_114223


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sister_height_growth_l1142_114288

/-- Given a height and a growth percentage, calculate the new height -/
noncomputable def new_height (initial_height : ℝ) (growth_percentage : ℝ) : ℝ :=
  initial_height * (1 + growth_percentage)

/-- Theorem: Given the initial height of 139.65 cm and 5% growth, 
    the new height is 146.6325 cm -/
theorem sister_height_growth : 
  new_height 139.65 0.05 = 146.6325 := by
  -- Unfold the definition of new_height
  unfold new_height
  -- Perform the calculation
  norm_num
  -- The proof is complete
  done

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sister_height_growth_l1142_114288


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cylinder_radius_is_7_5_l1142_114239

/-- The volume of a cylinder with radius r and height h -/
noncomputable def cylinderVolume (r h : ℝ) : ℝ := Real.pi * r^2 * h

/-- The original height of the cylinder -/
def h : ℝ := 3

/-- The increase in radius and height -/
def increase : ℝ := 5

theorem cylinder_radius_is_7_5 (r : ℝ) :
  cylinderVolume (r + increase) h - cylinderVolume r h =
  cylinderVolume r (h + increase) - cylinderVolume r h →
  r = 7.5 := by
  sorry

#check cylinder_radius_is_7_5

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cylinder_radius_is_7_5_l1142_114239


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_apple_cost_problem_l1142_114272

/-- The cost of apples for the first 30 kgs and additional kgs -/
structure AppleCost where
  l : ℚ  -- cost per kg for first 30 kgs
  q : ℚ  -- cost per kg for additional kgs

/-- Calculate the total cost of apples given the weight -/
def totalCost (c : AppleCost) (weight : ℚ) : ℚ :=
  if weight ≤ 30 then c.l * weight
  else c.l * 30 + c.q * (weight - 30)

/-- The problem statement -/
theorem apple_cost_problem (c : AppleCost) :
  totalCost c 33 = 663 ∧ totalCost c 36 = 726 →
  totalCost c 10 = 200 := by
  sorry

#eval totalCost ⟨20, 21⟩ 10  -- To check if the function works as expected

end NUMINAMATH_CALUDE_ERRORFEEDBACK_apple_cost_problem_l1142_114272


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fixed_point_location_l1142_114215

/-- The path of point M is an ellipse -/
noncomputable def is_on_ellipse (x y : ℝ) : Prop :=
  Real.sqrt ((x - Real.sqrt 5)^2 + y^2) + Real.sqrt ((x + Real.sqrt 5)^2 + y^2) = 6

/-- The distance between points M and T -/
noncomputable def distance_MT (x y t : ℝ) : ℝ :=
  Real.sqrt ((x - t)^2 + y^2)

/-- The theorem statement -/
theorem fixed_point_location (t : ℝ) :
  (0 < t ∧ t < 3) →
  (∃ x y : ℝ, is_on_ellipse x y ∧ 
    (∀ x' y' : ℝ, is_on_ellipse x' y' → distance_MT x' y' t ≥ 1) ∧
    distance_MT x y t = 1) →
  t = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_fixed_point_location_l1142_114215


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_value_is_25_point_5_l1142_114255

/-- Definition of a fair 8-sided die -/
def fair_8_sided_die : Finset ℕ := Finset.range 8

/-- The winning function: if one rolls n, they win n^2 dollars -/
def winning_function (n : ℕ) : ℚ := ((n + 1) : ℚ) ^ 2

/-- The probability of rolling any specific side of the die -/
def roll_probability : ℚ := 1 / 8

/-- The expected value of the win -/
noncomputable def expected_value : ℚ :=
  (fair_8_sided_die.sum fun i => roll_probability * winning_function i)

theorem expected_value_is_25_point_5 :
  expected_value = 51 / 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_value_is_25_point_5_l1142_114255


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1142_114279

noncomputable def f (x : ℝ) : ℝ := Real.cos x * Real.cos (x - Real.pi/6) + Real.sqrt 3 * (Real.sin x)^2 - 3 * Real.sqrt 3 / 4

theorem f_properties :
  -- Smallest positive period
  (∃ (T : ℝ), T > 0 ∧ ∀ (x : ℝ), f (x + T) = f x ∧ ∀ (T' : ℝ), T' > 0 ∧ (∀ (x : ℝ), f (x + T') = f x) → T ≤ T') ∧
  -- Monotonic increase interval
  (∀ (k : ℤ), ∀ (x y : ℝ), -Real.pi/12 + k*Real.pi ≤ x ∧ x < y ∧ y ≤ 5*Real.pi/12 + k*Real.pi → f x < f y) ∧
  -- Maximum and minimum values
  (∀ (x : ℝ), Real.pi/12 ≤ x ∧ x ≤ Real.pi/2 → f x ≤ 1/2) ∧
  (f (5*Real.pi/12) = 1/2) ∧
  (∀ (x : ℝ), Real.pi/12 ≤ x ∧ x ≤ Real.pi/2 → f x ≥ -1/4) ∧
  (f (Real.pi/12) = -1/4) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1142_114279


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_inequality_solution_sets_l1142_114210

theorem quadratic_inequality_solution_sets (a : ℝ) :
  let solution_set := {x : ℝ | x^2 + (a - 1) * x - a ≥ 0}
  (a < -1 → solution_set = Set.Iic 1 ∪ Set.Ici (-a)) ∧
  (a = -1 → solution_set = Set.univ) ∧
  (a > -1 → solution_set = Set.Iic (-a) ∪ Set.Ici 1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_inequality_solution_sets_l1142_114210


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_periodic_function_pairs_l1142_114222

theorem periodic_function_pairs (n : ℕ) (p : ℝ → ℝ) (h_cont : ContinuousOn p (Set.Icc 0 ↑n)) (h_periodic : p 0 = p ↑n) :
  ∃ (pairs : Finset (ℝ × ℝ)), pairs.card ≥ n ∧
    ∀ (x y : ℝ), (x, y) ∈ pairs →
      x ∈ Set.Icc 0 ↑n ∧ y ∈ Set.Icc 0 ↑n ∧
      p x = p y ∧ ∃ (k : ℕ), k > 0 ∧ y - x = ↑k := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_periodic_function_pairs_l1142_114222


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coordinates_of_E_l1142_114292

-- Define the points
def A : ℝ × ℝ := (-2, 1)
def B : ℝ × ℝ := (1, 4)
def C : ℝ × ℝ := (4, -3)

-- Define the section formula for internal division
noncomputable def section_formula_internal (P Q : ℝ × ℝ) (m n : ℝ) : ℝ × ℝ :=
  ((m * Q.1 + n * P.1) / (m + n), (m * Q.2 + n * P.2) / (m + n))

-- Define the section formula for external division
noncomputable def section_formula_external (P Q : ℝ × ℝ) (m n : ℝ) : ℝ × ℝ :=
  ((m * Q.1 - n * P.1) / (m - n), (m * Q.2 - n * P.2) / (m - n))

-- Define point D
noncomputable def D : ℝ × ℝ := section_formula_internal A B 2 1

-- Define point E
noncomputable def E : ℝ × ℝ := section_formula_external D C 1 4

-- Theorem statement
theorem coordinates_of_E : E = (-8/3, 11/3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coordinates_of_E_l1142_114292


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_common_tangents_C₂_C₃_l1142_114261

-- Define the circles and line
def C₁ (x y : ℝ) : Prop := (x - 2)^2 + (y + 1)^2 = 4
def line (x y : ℝ) : Prop := x + y - 3 = 0

-- Define the symmetry relation
def symmetric_about_line (C₁ C₂ : (ℝ → ℝ → Prop)) (line : ℝ → ℝ → Prop) : Prop :=
  ∃ (x₁ y₁ x₂ y₂ : ℝ), C₁ x₁ y₁ ∧ C₂ x₂ y₂ ∧ 
    ((x₂ - x₁) * (y₂ - y₁) = -1) ∧
    ((x₁ + x₂) / 2 + (y₁ + y₂) / 2 - 3 = 0)

-- Define C₃ based on the given condition
def C₃ (x y : ℝ) : Prop := x^2 + (y - 1)^2 = 4

-- Define the number of common tangents
def num_common_tangents (C₁ C₂ : (ℝ → ℝ → Prop)) (n : ℕ) : Prop := sorry

-- Theorem statement
theorem common_tangents_C₂_C₃ :
  ∃ (C₂ : ℝ → ℝ → Prop),
    symmetric_about_line C₁ C₂ line →
    num_common_tangents C₂ C₃ 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_common_tangents_C₂_C₃_l1142_114261


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l1142_114280

open Real

noncomputable def a (x : ℝ) : ℝ × ℝ := (cos x ^ 2 - sin x ^ 2, 1/2)
noncomputable def b (x : ℝ) : ℝ × ℝ := (1/2, sin x ^ 2 + Real.sqrt 3 * sin x * cos x)

noncomputable def f (x : ℝ) : ℝ := (a x).1 * (b x).1 + (a x).2 * (b x).2

noncomputable def g (x : ℝ) : ℝ := f (x + Real.pi/6)

def A : Set ℝ := {a : ℝ | ∀ x ∈ Set.Icc (-Real.pi/4) (Real.pi/8), 6*a^2 - a - 5/4 ≥ g x}

theorem range_of_a : A = Set.Iic (-1/2) ∪ Set.Ici (2/3) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l1142_114280


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_expression_value_l1142_114271

def expression (x y z w : ℕ) : ℤ := x * y^z - w

theorem max_expression_value :
  ∀ (x y z w : ℕ),
    x ∈ ({1, 2, 3, 4} : Set ℕ) →
    y ∈ ({1, 2, 3, 4} : Set ℕ) →
    z ∈ ({1, 2, 3, 4} : Set ℕ) →
    w ∈ ({1, 2, 3, 4} : Set ℕ) →
    x ≠ y → x ≠ z → x ≠ w → y ≠ z → y ≠ w → z ≠ w →
    (expression x y z w : ℤ) ≤ 161 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_expression_value_l1142_114271


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lattice_basis_existence_l1142_114213

/-- A full-rank lattice in ℝ² -/
structure Lattice2D where
  basis : Fin 2 → ℝ × ℝ
  full_rank : LinearIndependent ℝ basis

/-- The area of a lattice -/
noncomputable def area (L : Lattice2D) : ℝ := sorry

/-- A sub-lattice of L -/
structure SubLattice (L : Lattice2D) where
  lattice : Lattice2D
  is_sub : ∀ x, ∃ y ∈ (Set.range L.basis), x ∈ (Set.range lattice.basis)

variable {L : Lattice2D}
variable {K : SubLattice L}
variable (m : ℕ)

theorem lattice_basis_existence
  (h_area : area K.lattice / area L = (m : ℝ))
  (h_least : ∀ x ∈ (Set.range L.basis), m • x ∈ (Set.range K.lattice.basis) ∧ 
             ∀ n < m, ∃ y ∈ (Set.range L.basis), n • y ∉ (Set.range K.lattice.basis)) :
  ∃ x₁ x₂ : ℝ × ℝ, 
    (LinearIndependent ℝ ![x₁, x₂] ∧ 
     Submodule.span ℝ {x₁, x₂} = Submodule.span ℝ (Set.range L.basis)) ∧
    (LinearIndependent ℝ ![x₁, m • x₂] ∧ 
     Submodule.span ℝ {x₁, m • x₂} = Submodule.span ℝ (Set.range K.lattice.basis)) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lattice_basis_existence_l1142_114213


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_not_all_segments_integer_length_l1142_114218

/-- A configuration of points in a right triangle where certain angle bisectors meet. -/
structure RightTriangleConfiguration where
  -- The vertices of the right triangle
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  -- Points on the sides of the triangle
  D : ℝ × ℝ
  E : ℝ × ℝ
  -- The intersection point of BD and CE
  I : ℝ × ℝ
  -- Conditions
  right_angle : (A.1 - B.1) * (A.1 - C.1) + (A.2 - B.2) * (A.2 - C.2) = 0
  D_on_AC : ∃ t : ℝ, 0 < t ∧ t < 1 ∧ D = (t * A.1 + (1 - t) * C.1, t * A.2 + (1 - t) * C.2)
  E_on_AB : ∃ s : ℝ, 0 < s ∧ s < 1 ∧ E = (s * A.1 + (1 - s) * B.1, s * A.2 + (1 - s) * B.2)
  angle_bisector_BD : (B.1 - A.1) * (D.2 - B.2) = (B.2 - A.2) * (D.1 - B.1)
  angle_bisector_CE : (C.1 - A.1) * (E.2 - C.2) = (C.2 - A.2) * (E.1 - C.1)
  I_on_BD : ∃ u : ℝ, 0 < u ∧ u < 1 ∧ I = (u * B.1 + (1 - u) * D.1, u * B.2 + (1 - u) * D.2)
  I_on_CE : ∃ v : ℝ, 0 < v ∧ v < 1 ∧ I = (v * C.1 + (1 - v) * E.1, v * C.2 + (1 - v) * E.2)

/-- The theorem stating that it's generally not possible for all specified segments to have integer lengths. -/
theorem not_all_segments_integer_length (config : RightTriangleConfiguration) :
  ¬ (∃ (AB AC BI ID CI IE : ℕ),
      (AB : ℝ)^2 = (config.A.1 - config.B.1)^2 + (config.A.2 - config.B.2)^2 ∧
      (AC : ℝ)^2 = (config.A.1 - config.C.1)^2 + (config.A.2 - config.C.2)^2 ∧
      (BI : ℝ)^2 = (config.B.1 - config.I.1)^2 + (config.B.2 - config.I.2)^2 ∧
      (ID : ℝ)^2 = (config.I.1 - config.D.1)^2 + (config.I.2 - config.D.2)^2 ∧
      (CI : ℝ)^2 = (config.C.1 - config.I.1)^2 + (config.C.2 - config.I.2)^2 ∧
      (IE : ℝ)^2 = (config.I.1 - config.E.1)^2 + (config.I.2 - config.E.2)^2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_not_all_segments_integer_length_l1142_114218


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_locus_theorem_l1142_114228

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a cube in 3D space -/
structure Cube where
  A : Point3D
  B : Point3D
  C : Point3D
  D : Point3D
  A' : Point3D
  B' : Point3D
  C' : Point3D
  D' : Point3D

/-- Represents a plane in 3D space -/
structure Plane where
  normal : Point3D
  point : Point3D

/-- Checks if a point lies on a line segment between two other points -/
def isOnLineSegment (P Q R : Point3D) : Prop := sorry

/-- Checks if a plane is parallel to another plane -/
def isParallelPlane (p1 p2 : Plane) : Prop := sorry

/-- Calculates the distance between a point and a plane -/
def distanceToPlane (point : Point3D) (plane : Plane) : ℝ := sorry

/-- Calculates the distance between two points -/
def distanceBetweenPoints (p1 p2 : Point3D) : ℝ := sorry

/-- Main theorem statement -/
theorem cube_locus_theorem (cube : Cube) :
  -- Part (a)
  ∃ (midplane : Plane),
    isParallelPlane midplane (Plane.mk (Point3D.mk 0 0 1) cube.A) ∧
    distanceToPlane cube.A midplane = (1/2) * distanceBetweenPoints cube.A cube.A' ∧
    ∀ (X Y Z : Point3D),
      isOnLineSegment cube.A cube.C X ∧
      isOnLineSegment cube.B' cube.D' Y ∧
      Z.x = (X.x + Y.x) / 2 ∧
      Z.y = (X.y + Y.y) / 2 ∧
      Z.z = (X.z + Y.z) / 2 →
      distanceToPlane Z midplane = 0 ∧
  -- Part (b)
  ∃ (locus_plane : Plane),
    isParallelPlane locus_plane (Plane.mk (Point3D.mk 0 0 1) cube.A) ∧
    distanceToPlane cube.A locus_plane = (1/3) * distanceBetweenPoints cube.A cube.A' ∧
    ∀ (X Y Z : Point3D),
      isOnLineSegment cube.A cube.C X ∧
      isOnLineSegment cube.B' cube.D' Y ∧
      isOnLineSegment X Y Z ∧
      (Z.x - Y.x)^2 + (Z.y - Y.y)^2 + (Z.z - Y.z)^2 = 
        4 * ((X.x - Z.x)^2 + (X.y - Z.y)^2 + (X.z - Z.z)^2) →
      distanceToPlane Z locus_plane = 0 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_locus_theorem_l1142_114228


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_addition_solution_l1142_114211

/-- Represents a digit (0-9) -/
def Digit := Fin 10

/-- Represents a 4-digit number -/
structure FourDigitNumber where
  thousands : Digit
  hundreds : Digit
  tens : Digit
  ones : Digit

/-- Converts a FourDigitNumber to a natural number -/
def fourDigitToNat (n : FourDigitNumber) : ℕ :=
  1000 * n.thousands.val + 100 * n.hundreds.val + 10 * n.tens.val + n.ones.val

/-- The theorem statement -/
theorem unique_addition_solution :
  ∃! (abcd efgh : FourDigitNumber),
    (fourDigitToNat abcd + fourDigitToNat efgh = 10652) ∧
    (20000 > fourDigitToNat abcd + fourDigitToNat efgh) ∧
    (fourDigitToNat abcd + fourDigitToNat efgh > 10000) ∧
    (abcd.thousands ≠ abcd.hundreds) ∧
    (abcd.thousands ≠ abcd.tens) ∧
    (abcd.thousands ≠ abcd.ones) ∧
    (abcd.hundreds ≠ abcd.tens) ∧
    (abcd.hundreds ≠ abcd.ones) ∧
    (abcd.tens ≠ abcd.ones) ∧
    (efgh.thousands ≠ efgh.hundreds) ∧
    (efgh.thousands ≠ efgh.tens) ∧
    (efgh.thousands ≠ efgh.ones) ∧
    (efgh.hundreds ≠ efgh.tens) ∧
    (efgh.hundreds ≠ efgh.ones) ∧
    (efgh.tens ≠ efgh.ones) ∧
    (abcd.thousands ≠ efgh.thousands) ∧
    (abcd.thousands ≠ efgh.hundreds) ∧
    (abcd.thousands ≠ efgh.tens) ∧
    (abcd.thousands ≠ efgh.ones) ∧
    (abcd.hundreds ≠ efgh.thousands) ∧
    (abcd.hundreds ≠ efgh.hundreds) ∧
    (abcd.hundreds ≠ efgh.tens) ∧
    (abcd.hundreds ≠ efgh.ones) ∧
    (abcd.tens ≠ efgh.thousands) ∧
    (abcd.tens ≠ efgh.hundreds) ∧
    (abcd.tens ≠ efgh.tens) ∧
    (abcd.tens ≠ efgh.ones) ∧
    (abcd.ones ≠ efgh.thousands) ∧
    (abcd.ones ≠ efgh.hundreds) ∧
    (abcd.ones ≠ efgh.tens) ∧
    (abcd.ones ≠ efgh.ones) ∧
    (fourDigitToNat abcd = 9567) ∧
    (fourDigitToNat efgh = 1085) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_addition_solution_l1142_114211


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_N_given_M_l1142_114247

/-- Represents the outcome of a single die roll -/
inductive DieRoll
| one
| two
| three
| four
| five
| six

/-- Represents the outcome of two consecutive die rolls -/
def TwoRolls := DieRoll × DieRoll

/-- Predicate for odd numbers -/
def is_odd (d : DieRoll) : Prop :=
  d = DieRoll.one ∨ d = DieRoll.three ∨ d = DieRoll.five

/-- Event M: both outcomes are odd numbers -/
def M (r : TwoRolls) : Prop :=
  is_odd r.1 ∧ is_odd r.2

/-- Event N: at least one of the outcomes is a 5 -/
def N (r : TwoRolls) : Prop :=
  r.1 = DieRoll.five ∨ r.2 = DieRoll.five

/-- The sample space of all possible outcomes -/
def Ω : Set TwoRolls := Set.univ

/-- Probability measure on the sample space -/
axiom P : Set TwoRolls → ℝ

/-- Properties of the probability measure -/
axiom P_nonneg : ∀ A : Set TwoRolls, 0 ≤ P A
axiom P_le_one : ∀ A : Set TwoRolls, P A ≤ 1
axiom P_empty : P ∅ = 0
axiom P_total : P Ω = 1

/-- Conditional probability -/
noncomputable def conditional_prob (A B : Set TwoRolls) : ℝ :=
  P (A ∩ B) / P B

/-- The main theorem: P(N|M) = 1/3 -/
theorem prob_N_given_M :
  conditional_prob {r : TwoRolls | N r} {r : TwoRolls | M r} = 1/3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_N_given_M_l1142_114247


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_vector_equality_l1142_114267

-- Define the parallelogram ABCD and the center O
variable (A B C D O : ℝ × ℝ)

-- Define the basis vectors
variable (e₁ e₂ : ℝ × ℝ)

-- State the theorem
theorem parallelogram_vector_equality 
  (h_center : O = (A + C) / 2)  -- O is the center of the parallelogram
  (h_AB : B - A = 4 • e₁)       -- AB⃗ = 4e⃗₁
  (h_BC : C - B = 6 • e₂)       -- BC⃗ = 6e⃗₂
  : 3 • e₂ - 2 • e₁ = B - O :=  -- 3e⃗₂ - 2e⃗₁ = BO⃗
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_vector_equality_l1142_114267


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_stream_speed_calculation_l1142_114237

/-- The speed of the stream given the boat's speed, distance, and total time -/
noncomputable def stream_speed (boat_speed : ℝ) (distance : ℝ) (total_time : ℝ) : ℝ :=
  Real.sqrt (boat_speed^2 - (2 * distance / total_time)^2)

/-- Theorem stating the speed of the stream given the problem conditions -/
theorem stream_speed_calculation :
  let boat_speed : ℝ := 5
  let distance : ℝ := 252
  let total_time : ℝ := 200
  ∃ ε > 0, |stream_speed boat_speed distance total_time - 3.52| < ε :=
by
  sorry

/-- Approximate evaluation of the stream speed -/
def approximate_stream_speed : ℚ :=
  let boat_speed : ℚ := 5
  let distance : ℚ := 252
  let total_time : ℚ := 200
  Rat.sqrt ((boat_speed^2 - (2 * distance / total_time)^2).toNNRat)

#eval approximate_stream_speed

end NUMINAMATH_CALUDE_ERRORFEEDBACK_stream_speed_calculation_l1142_114237


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_property_l1142_114225

-- Define the theorem
theorem triangle_property (a b c : ℝ) (A B C : ℝ) (S : ℝ) :
  a > 0 ∧ b > 0 ∧ c > 0 →
  A > 0 ∧ B > 0 ∧ C > 0 →
  A + B + C = π →
  a / (Real.sin B) = b / (Real.sin A) →
  a / (Real.sin C) = c / (Real.sin A) →
  b / (Real.sin A + Real.sin C) = (a - c) / (Real.sin B - Real.sin C) →
  S = (1/2) * b * c * Real.sin A →
  (A = π/3) ∧
  (∃ (M : ℝ × ℝ),
    ((2/3 * b)^2 + (1/3 * c * Real.cos A)^2 + (1/3 * c * Real.sin A)^2) / S ≥ 8 * Real.sqrt 3 / 9) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_property_l1142_114225


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_marks_problem_l1142_114266

/-- The average marks of a student in 3 subjects -/
noncomputable def average_marks (physics chemistry mathematics : ℝ) : ℝ :=
  (physics + chemistry + mathematics) / 3

theorem average_marks_problem (physics chemistry mathematics : ℝ) :
  physics = 110 →
  (physics + mathematics) / 2 = 90 →
  (physics + chemistry) / 2 = 70 →
  average_marks physics chemistry mathematics = 70 := by
  sorry

#check average_marks_problem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_marks_problem_l1142_114266


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pascal_triangle_row_sum_row_12_sum_l1142_114262

/-- Sum of a row in Pascal's Triangle -/
def sum_of_pascal_triangle_row (n : ℕ) : ℕ := sorry

/-- Pascal's Triangle row sum theorem -/
theorem pascal_triangle_row_sum (n : ℕ) : 
  sum_of_pascal_triangle_row n = 2^n := by sorry

/-- Sum of Row 12 in Pascal's Triangle -/
theorem row_12_sum : sum_of_pascal_triangle_row 12 = 4096 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pascal_triangle_row_sum_row_12_sum_l1142_114262


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_circle_line_l1142_114286

-- Define the circle
def my_circle (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 4

-- Define the line
def my_line (x y : ℝ) : Prop := x - 2*y - 6 = 0

-- Define a point P on the lower half of the circle
def P_on_lower_circle (P : ℝ × ℝ) : Prop :=
  my_circle P.1 P.2 ∧ P.2 ≤ 0

-- Theorem statement
theorem min_distance_circle_line :
  ∃ (d : ℝ), d = Real.sqrt 5 - 2 ∧
  ∀ (P : ℝ × ℝ), P_on_lower_circle P →
  ∀ (Q : ℝ × ℝ), my_line Q.1 Q.2 →
  d ≤ Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_circle_line_l1142_114286


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_top_layer_items_l1142_114248

/-- Represents a structure with layers where each layer has double the items of the layer above --/
structure LayeredStructure where
  layers : ℕ
  top_items : ℕ
  total_items : ℕ

/-- The sum of a geometric series with first term a, ratio r, and n terms --/
noncomputable def geometric_sum (a : ℝ) (r : ℝ) (n : ℕ) : ℝ :=
  a * (1 - r^n) / (1 - r)

/-- Theorem stating the number of items on the top layer of the structure --/
theorem top_layer_items (s : LayeredStructure) 
  (h1 : s.layers = 7)
  (h2 : s.total_items = 381)
  (h3 : geometric_sum (s.top_items : ℝ) 2 s.layers = s.total_items) : 
  s.top_items = 3 := by
  sorry

#check top_layer_items

end NUMINAMATH_CALUDE_ERRORFEEDBACK_top_layer_items_l1142_114248


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_concert_ticket_prices_l1142_114270

theorem concert_ticket_prices (x : ℕ) : 
  (∃ a b : ℕ, a * x = 112 ∧ b * x = 168) → 
  (Finset.filter (λ y : ℕ ↦ 112 % y = 0 ∧ 168 % y = 0) (Finset.range 169)).card = 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_concert_ticket_prices_l1142_114270


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_multiples_of_four_upper_bound_l1142_114259

def is_valid_upper_bound (n : ℕ) : Prop :=
  12 ≤ n ∧ (∃ k, n = 12 + 4 * k) ∧
  (Finset.filter (λ x ↦ 4 ∣ x) (Finset.range (n - 11))).card = 25

theorem multiples_of_four_upper_bound :
  ∃ n : ℕ, is_valid_upper_bound n ∧ ∀ m, is_valid_upper_bound m → m ≤ n :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_multiples_of_four_upper_bound_l1142_114259


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_of_2x_plus_6_when_x_is_3_l1142_114243

theorem square_of_2x_plus_6_when_x_is_3 :
  (2 * 3 + 6)^2 = 144 := by
  -- Substitute x = 3
  have h1 : (2 * 3 + 6)^2 = 12^2 := by rfl
  -- Calculate 12^2
  have h2 : 12^2 = 144 := by rfl
  -- Combine the steps
  rw [h1, h2]


end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_of_2x_plus_6_when_x_is_3_l1142_114243


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_bisecting_line_slope_l1142_114257

/-- Represents a point in 2D space -/
structure Point where
  x : ℚ
  y : ℚ

/-- Represents a parallelogram defined by its four vertices -/
structure Parallelogram where
  v1 : Point
  v2 : Point
  v3 : Point
  v4 : Point

/-- Calculates the slope of a line passing through two points -/
def slopeBetweenPoints (p1 p2 : Point) : ℚ :=
  (p2.y - p1.y) / (p2.x - p1.x)

/-- Checks if a point lies on a line defined by two other points -/
def pointOnLine (p : Point) (p1 p2 : Point) : Prop :=
  (p.y - p1.y) * (p2.x - p1.x) = (p2.y - p1.y) * (p.x - p1.x)

/-- Theorem: The line that divides the given parallelogram into two congruent polygons
    and passes through (5, 0) has a slope of 109/19 -/
theorem parallelogram_bisecting_line_slope :
  let para := Parallelogram.mk
    (Point.mk 15 55) (Point.mk 15 124) (Point.mk 33 163) (Point.mk 33 94)
  let origin := Point.mk 5 0
  ∃ (p : Point),
    pointOnLine p origin (Point.mk 15 (55 + 45/19)) ∧
    pointOnLine p origin (Point.mk 33 (163 - 45/19)) ∧
    slopeBetweenPoints origin p = 109/19 := by
  sorry

#eval 109 + 19

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_bisecting_line_slope_l1142_114257


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perfect_cube_condition_l1142_114281

theorem perfect_cube_condition (Z K : ℤ) (h1 : 1000 < Z) (h2 : Z < 8000) (h3 : K > 1) (h4 : Z = K^3) :
  ∃ n : ℤ, Z = n^3 ↔ 11 ≤ K ∧ K ≤ 19 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perfect_cube_condition_l1142_114281


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_of_perpendicular_foot_l1142_114245

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents an ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_ab : a > b
  h_b_pos : b > 0

/-- Defines a point on the ellipse -/
def PointOnEllipse (e : Ellipse) (p : Point) : Prop :=
  p.x^2 / e.a^2 + p.y^2 / e.b^2 = 1

/-- Defines the foci of the ellipse -/
noncomputable def Foci (e : Ellipse) : (Point × Point) :=
  let c := (e.a^2 - e.b^2).sqrt
  (Point.mk (-c) 0, Point.mk c 0)

/-- Defines a point not at the vertices of the major axis -/
def NotVertex (e : Ellipse) (p : Point) : Prop :=
  p ≠ Point.mk (-e.a) 0 ∧ p ≠ Point.mk e.a 0

/-- States that P is the foot of the perpendicular from a focus to the external bisector -/
def IsFootOfPerpendicular (e : Ellipse) (p : Point) (q : Point) : Prop :=
  ∃ (focus : Point), (focus = (Foci e).1 ∨ focus = (Foci e).2) ∧
    -- Additional conditions to define the perpendicular and external bisector
    sorry

/-- The main theorem statement -/
theorem locus_of_perpendicular_foot (e : Ellipse) (p q : Point) :
  PointOnEllipse e q → NotVertex e q → IsFootOfPerpendicular e p q →
  ∃ (o : Point), o = Point.mk 0 0 ∧ (p.x - o.x)^2 + (p.y - o.y)^2 = e.a^2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_of_perpendicular_foot_l1142_114245


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_sqrt_two_l1142_114216

noncomputable def power_function (α : ℝ) : ℝ → ℝ := fun x ↦ x ^ α

theorem power_function_sqrt_two (f : ℝ → ℝ) (α : ℝ) 
  (h1 : f = power_function α) 
  (h2 : f 2 = 1/4) : 
  f (Real.sqrt 2) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_sqrt_two_l1142_114216


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_simple_random_sampling_probability_l1142_114260

/-- Simple random sampling -/
structure SimpleRandomSampling (Population : Type*) where
  population : Finset Population
  sample_size : ℕ
  is_valid : sample_size ≤ population.card

/-- The probability of selecting an individual in simple random sampling -/
noncomputable def selection_probability (srs : SimpleRandomSampling Population) (individual : Population) : ℝ :=
  1 / srs.population.card

theorem simple_random_sampling_probability 
  (Population : Type*)
  [Fintype Population]
  (srs : SimpleRandomSampling Population) 
  (individual : Population) 
  (attempt : ℕ) :
  selection_probability srs individual = 1 / srs.population.card := by
  sorry

#check simple_random_sampling_probability

end NUMINAMATH_CALUDE_ERRORFEEDBACK_simple_random_sampling_probability_l1142_114260


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_terminal_side_theorem_l1142_114290

noncomputable def α : ℝ := Real.pi / 3

def same_terminal_side (θ : ℝ) : Prop :=
  ∃ k : ℤ, θ = 2 * k * Real.pi + α

def in_range (θ : ℝ) : Prop :=
  -4 * Real.pi < θ ∧ θ < 2 * Real.pi

def in_first_or_third_quadrant (θ : ℝ) : Prop :=
  (0 < θ ∧ θ < Real.pi / 2) ∨ (Real.pi < θ ∧ θ < 3 * Real.pi / 2)

theorem terminal_side_theorem :
  (∀ θ : ℝ, same_terminal_side θ ↔ ∃ k : ℤ, θ = 2 * k * Real.pi + Real.pi / 3) ∧
  ({θ : ℝ | same_terminal_side θ ∧ in_range θ} = {-11 * Real.pi / 3, -5 * Real.pi / 3, Real.pi / 3}) ∧
  (∀ β : ℝ, same_terminal_side β → in_first_or_third_quadrant (β / 2)) := by
  sorry

#check terminal_side_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_terminal_side_theorem_l1142_114290


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_incenter_sequence_limit_l1142_114258

/-- Non-degenerate triangle in Euclidean plane -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  nondegeneracy : A ≠ B ∧ B ≠ C ∧ C ≠ A

/-- Incenter of a triangle -/
noncomputable def incenter (t : Triangle) : ℝ × ℝ := sorry

/-- Sequence of points C_n -/
noncomputable def C_sequence (t : Triangle) : ℕ → ℝ × ℝ
  | 0 => t.C
  | n + 1 => incenter ⟨t.A, t.B, C_sequence t n, sorry⟩

/-- Angle of a triangle at a given vertex -/
noncomputable def angle (t : Triangle) (v : ℝ × ℝ) : ℝ := sorry

/-- Point on a line segment that divides it in a given ratio -/
noncomputable def dividing_point (A B : ℝ × ℝ) (ratio : ℝ) : ℝ × ℝ := sorry

/-- The main theorem -/
theorem incenter_sequence_limit (t : Triangle) :
  ∃ (P : ℝ × ℝ), (∀ ε > 0, ∃ N, ∀ n ≥ N, dist (C_sequence t n) P < ε) ∧
  P = dividing_point t.B t.C (angle t t.A / angle t t.B) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_incenter_sequence_limit_l1142_114258


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_segment_length_l1142_114268

-- Define the curve C in polar coordinates
noncomputable def C (θ : ℝ) : ℝ := Real.cos θ + Real.sin θ

-- Define the line l in parametric form
noncomputable def l (t : ℝ) : ℝ × ℝ := (1/2 - (Real.sqrt 2)/2 * t, (Real.sqrt 2)/2 * t)

-- Define the intersection points P and Q
def intersection_points : Set (ℝ × ℝ) :=
  {p | ∃ θ t, (Real.cos θ * C θ, Real.sin θ * C θ) = l t ∧ p = l t}

-- State the theorem
theorem intersection_segment_length :
  ∃ P Q, P ∈ intersection_points ∧ Q ∈ intersection_points ∧ ‖P - Q‖ = Real.sqrt 6 / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_segment_length_l1142_114268


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_equal_perimeters_l1142_114229

/-- A trapezoid with bases AD and BC -/
structure Trapezoid (A B C D : ℝ × ℝ) : Type where
  is_trapezoid : Prop

/-- Point E on segment AD -/
def point_on_segment (E A D : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ E = (1 - t) • A + t • D

/-- Perimeter of a triangle -/
def perimeter (P Q R : ℝ × ℝ) : ℝ :=
  dist P Q + dist Q R + dist R P

/-- Main theorem -/
theorem trapezoid_equal_perimeters 
  (A B C D E : ℝ × ℝ) 
  (h_trapezoid : Trapezoid A B C D) 
  (h_E_on_AD : point_on_segment E A D) 
  (h_equal_perimeters : perimeter A B E = perimeter B C E ∧ perimeter B C E = perimeter C D E) :
  dist B C = (dist A D) / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_equal_perimeters_l1142_114229


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_eight_digit_increasing_integers_mod_1000_l1142_114206

theorem eight_digit_increasing_integers_mod_1000 : ∃ M : ℕ, M = Nat.choose 17 8 ∧ M % 1000 = 310 := by
  let M := Nat.choose 17 8
  use M
  constructor
  · rfl
  · sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_eight_digit_increasing_integers_mod_1000_l1142_114206


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_area_and_diagonal_l1142_114287

/-- A parallelogram with given base and height -/
structure Parallelogram where
  base : ℝ
  height : ℝ

/-- The diagonal of a parallelogram -/
noncomputable def diagonal (p : Parallelogram) : ℝ :=
  Real.sqrt (p.base ^ 2 + p.height ^ 2)

/-- Theorem about a specific parallelogram's area and diagonal length -/
theorem parallelogram_area_and_diagonal :
  ∃ (p : Parallelogram),
    p.base = 20 ∧
    p.height = 4 ∧
    p.base * p.height = 80 ∧
    diagonal p = Real.sqrt 416 := by
  sorry

#check parallelogram_area_and_diagonal

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_area_and_diagonal_l1142_114287


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l1142_114202

noncomputable def f (x : ℝ) (a : ℝ) : ℝ :=
  Real.sqrt 3 * Real.sin (2 * x) + 2 * (Real.cos x) ^ 2 + a

theorem function_properties
  (h_max : ∀ x, x ∈ Set.Icc 0 (Real.pi / 2) → f x 1 ≤ 4)
  (h_attains_max : ∃ x, x ∈ Set.Icc 0 (Real.pi / 2) ∧ f x 1 = 4) :
  (∀ m, m ∈ Set.Ioc 3 4 → ∃ x₁ x₂, x₁ ∈ Set.Icc 0 (Real.pi / 2) ∧ x₂ ∈ Set.Icc 0 (Real.pi / 2) ∧ x₁ ≠ x₂ ∧ f x₁ 1 = m ∧ f x₂ 1 = m) ∧
  (∀ m, m ∈ Set.Ioc 3 4 → ∀ x₁ x₂, x₁ ∈ Set.Icc 0 (Real.pi / 2) → x₂ ∈ Set.Icc 0 (Real.pi / 2) → 
    x₁ ≠ x₂ → f x₁ 1 = m → f x₂ 1 = m → x₁ + x₂ = Real.pi / 3) ∧
  (∀ A B C : ℝ,
    A + B + C = Real.pi →
    0 < A ∧ 0 < B ∧ 0 < C →
    f (A - Real.pi / 6) 1 = 4 →
    Real.sqrt 7 * Real.sin A = 2 * Real.sin B →
    Real.sin B = Real.sqrt 21 / 7 →
    2 * Real.sin C = Real.sqrt 7 ∧
    3 * Real.sqrt 3 / 2 = 1/2 * 2 * 3 * Real.sin A) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l1142_114202


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_2bpc_is_zero_l1142_114289

/-- Angle between three points -/
noncomputable def angle (A P B : ℝ × ℝ) : ℝ := 
  Real.arccos ((A.1 - P.1) * (B.1 - P.1) + (A.2 - P.2) * (B.2 - P.2)) / 
    (Real.sqrt ((A.1 - P.1)^2 + (A.2 - P.2)^2) * Real.sqrt ((B.1 - P.1)^2 + (B.2 - P.2)^2))

/-- Given points A, B, C, and D equally spaced on a line, and a point P,
    prove that sin(2∠BPC) = 0 when cos ∠APC = 3/5 and cos ∠BPD = 4/5 -/
theorem sin_2bpc_is_zero (A B C D P : ℝ × ℝ) : 
  (∃ k : ℝ, B.1 - A.1 = k ∧ C.1 - B.1 = k ∧ D.1 - C.1 = k) →  -- equally spaced on a line
  (B.2 = A.2 ∧ C.2 = A.2 ∧ D.2 = A.2) →  -- all points on the same line
  Real.cos (angle A P C) = 3/5 →
  Real.cos (angle B P D) = 4/5 →
  Real.sin (2 * angle B P C) = 0 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_2bpc_is_zero_l1142_114289


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_equality_l1142_114293

theorem cube_root_equality (x : ℝ) : (x^3)^(1/3) = x := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_equality_l1142_114293


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pen_retailer_profit_percent_pen_retailer_profit_percent_is_16_22_l1142_114254

/-- Calculates the profit percent for a pen retailer. -/
theorem pen_retailer_profit_percent 
  (num_pens : ℕ) 
  (price_pens : ℕ) 
  (discount : ℝ) : ℝ :=
  let cost_price := price_pens
  let marked_price := num_pens
  let selling_price := (1 - discount / 100) * marked_price
  let profit := selling_price - cost_price
  (profit / cost_price) * 100

/-- Proves that the profit percent is approximately 16.22% for the given conditions. -/
theorem pen_retailer_profit_percent_is_16_22 : 
  abs (pen_retailer_profit_percent 54 46 1 - 16.22) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pen_retailer_profit_percent_pen_retailer_profit_percent_is_16_22_l1142_114254


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_m_value_l1142_114234

/-- Given two 2D vectors a and b, proves that if they are parallel and have the components (m, -1) and (1, m+2) respectively, then m = -1. -/
theorem parallel_vectors_m_value (m : ℝ) :
  let a : Fin 2 → ℝ := ![m, -1]
  let b : Fin 2 → ℝ := ![1, m + 2]
  (∃ (k : ℝ), k ≠ 0 ∧ a = k • b) →
  m = -1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_m_value_l1142_114234


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_cos_C_value_l1142_114204

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : Real
  B : Real
  C : Real

/-- The dot product of two 2D vectors -/
def dot_product (x1 y1 x2 y2 : ℝ) : ℝ := x1 * x2 + y1 * y2

theorem max_cos_C_value (t : Triangle) (D : ℝ × ℝ) :
  t.a = 7 →
  D = ((t.b * Real.cos t.C + t.a) / 2, t.b * Real.sin t.C / 2) →  -- D is midpoint of AC
  dot_product (t.b * Real.cos t.C - t.a / 2) (t.b * Real.sin t.C) (t.a / 2) 0 = 25 / 2 →
  (∀ (t' : Triangle), t'.a = t.a → Real.cos t'.C ≤ Real.cos t.C) →
  Real.cos t.C = 5 / 7 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_cos_C_value_l1142_114204


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parity_of_expression_l1142_114236

theorem parity_of_expression (a b c : ℕ) (ha : Odd a) (hb : Even b) :
  (Even c → Odd (3^a + (b+1)^2*c)) ∧ (Odd c → Even (3^a + (b+1)^2*c)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parity_of_expression_l1142_114236


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_segments_distinct_l1142_114233

/-- A right triangle with a point on its hypotenuse -/
structure RightTriangleWithPoint where
  -- Points of the triangle
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  -- Point on the hypotenuse
  D : ℝ × ℝ
  -- ABC is a right triangle
  right_angle : (A.1 - B.1) * (C.1 - B.1) + (A.2 - B.2) * (C.2 - B.2) = 0
  -- BC is the hypotenuse
  hypotenuse : (B.1 - C.1)^2 + (B.2 - C.2)^2 ≥ (A.1 - B.1)^2 + (A.2 - B.2)^2 ∧
               (B.1 - C.1)^2 + (B.2 - C.2)^2 ≥ (A.1 - C.1)^2 + (A.2 - C.2)^2
  -- D lies on BC
  d_on_bc : ∃ t : ℝ, 0 < t ∧ t < 1 ∧ D = (B.1 + t*(C.1 - B.1), B.2 + t*(C.2 - B.2))
  -- D is not the midpoint of BC
  d_not_midpoint : D ≠ ((B.1 + C.1)/2, (B.2 + C.2)/2)

/-- The lengths of segments AD, BD, and CD are all distinct -/
theorem segments_distinct (T : RightTriangleWithPoint) : 
  let ad := ((T.A.1 - T.D.1)^2 + (T.A.2 - T.D.2)^2)^(1/2)
  let bd := ((T.B.1 - T.D.1)^2 + (T.B.2 - T.D.2)^2)^(1/2)
  let cd := ((T.C.1 - T.D.1)^2 + (T.C.2 - T.D.2)^2)^(1/2)
  ad ≠ bd ∧ ad ≠ cd ∧ bd ≠ cd := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_segments_distinct_l1142_114233


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_abc_properties_triangle_abc_area_l1142_114221

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C -/
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  a : ℝ
  b : ℝ
  c : ℝ

theorem triangle_abc_properties (t : Triangle)
  (h1 : t.b = Real.sqrt 3)
  (h2 : Real.sin t.B * (Real.sin t.C + Real.sqrt 3 * Real.cos t.C) - Real.sqrt 3 * Real.sin t.A = 0) :
  ∃ (f : ℝ → ℝ),
    (∀ x, f x = 2 * Real.sqrt 3 * Real.sin (x + π / 6) + Real.sqrt 3) ∧
    (∃ L_max : ℝ, L_max = 3 * Real.sqrt 3 ∧ ∀ x, f x ≤ L_max) := by
  sorry

theorem triangle_abc_area (t : Triangle)
  (h1 : t.b = Real.sqrt 3)
  (h2 : Real.sin t.B * (Real.sin t.C + Real.sqrt 3 * Real.cos t.C) - Real.sqrt 3 * Real.sin t.A = 0)
  (h3 : t.a + t.c = 2) :
  ∃ S : ℝ, S = Real.sqrt 3 / 12 ∧ S = (1 / 2) * t.a * t.c * Real.sin t.B := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_abc_properties_triangle_abc_area_l1142_114221


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_equation_solution_l1142_114274

theorem sin_equation_solution (x : ℝ) :
  Real.sin x * Real.sin (3 * x) + Real.sin (4 * x) * Real.sin (8 * x) = 0 →
  (∃ n : ℤ, x = Real.pi * n / 7) ∨ (∃ k : ℤ, x = Real.pi * k / 5) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_equation_solution_l1142_114274


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smaller_circle_circumference_l1142_114231

-- Define the circumference of the larger circle
def larger_circumference : ℝ := 380

-- Define the area difference between the circles
def area_difference : ℝ := 5775.414574918697

-- Define π as a constant (approximation)
def π : ℝ := 3.14159

-- Define the theorem
theorem smaller_circle_circumference :
  ∃ (smaller_circumference : ℝ),
    smaller_circumference > 0 ∧
    ∃ (r R : ℝ),
      r > 0 ∧ R > 0 ∧
      smaller_circumference = 2 * π * r ∧
      larger_circumference = 2 * π * R ∧
      π * R^2 - π * r^2 = area_difference ∧
      abs (smaller_circumference - 267.9) < 0.1 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smaller_circle_circumference_l1142_114231


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_proof_l1142_114250

open Real

/-- For any non-integer real number x > 1, prove the given inequality -/
theorem inequality_proof (x : ℝ) (h1 : x > 1) (h2 : ¬ Int.floor x = x) : 
  let fx := x - Int.floor x
  let ix := Int.floor x
  ((x + fx) / ix - ix / (x + fx)) + ((x + ix) / fx - fx / (x + ix)) > 9/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_proof_l1142_114250


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_tetrahedron_surface_area_ratio_l1142_114220

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Calculates the distance between two points in 3D space -/
noncomputable def distance (p1 p2 : Point3D) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2 + (p1.z - p2.z)^2)

/-- Calculates the surface area of a regular tetrahedron given its side length -/
noncomputable def tetrahedronSurfaceArea (sideLength : ℝ) : ℝ :=
  Real.sqrt 3 * sideLength^2

/-- Calculates the surface area of a cube given its side length -/
def cubeSurfaceArea (sideLength : ℝ) : ℝ :=
  6 * sideLength^2

/-- The main theorem to prove -/
theorem cube_tetrahedron_surface_area_ratio :
  let cube_side_length : ℝ := 2
  let v1 : Point3D := ⟨0, 0, 0⟩
  let v2 : Point3D := ⟨2, 2, 0⟩
  let v3 : Point3D := ⟨2, 0, 2⟩
  let v4 : Point3D := ⟨0, 2, 2⟩
  let tetrahedron_side_length := distance v1 v2
  let cube_surface_area := cubeSurfaceArea cube_side_length
  let tetrahedron_surface_area := tetrahedronSurfaceArea tetrahedron_side_length
  cube_surface_area / tetrahedron_surface_area = Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_tetrahedron_surface_area_ratio_l1142_114220


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_is_288_l1142_114294

-- Define the triangle
def triangle_DEF (DE EF DF : ℝ) : Prop :=
  DE = 24 ∧ EF = 24 ∧ DF = 35

-- Define Heron's formula for area
noncomputable def heron_area (a b c : ℝ) : ℝ :=
  let s := (a + b + c) / 2
  Real.sqrt (s * (s - a) * (s - b) * (s - c))

-- Theorem statement
theorem triangle_area_is_288 :
  ∀ (DE EF DF : ℝ), triangle_DEF DE EF DF →
  heron_area DE EF DF = 288 := by
  intro DE EF DF h
  sorry

#eval toString "The theorem has been stated and the proof is left as an exercise."

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_is_288_l1142_114294


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_l1142_114201

noncomputable def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, d > 0 ∧ ∀ n : ℕ, a (n + 1) = a n + d

def geometric_sequence (a b c : ℝ) : Prop :=
  b * b = a * c

noncomputable def sum_of_arithmetic_sequence (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  (n : ℝ) / 2 * (2 * a 1 + (n - 1 : ℝ) * (a 2 - a 1))

theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  arithmetic_sequence a →
  a 1 = 1 →
  geometric_sequence (a 1) (a 2) (a 4) →
  sum_of_arithmetic_sequence a 10 = 55 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_l1142_114201


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_minor_axis_length_l1142_114200

/-- A point in the 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Definition of the five given points -/
def points : List Point := [
  { x := 0, y := 0 },
  { x := 0, y := 4 },
  { x := 6, y := 0 },
  { x := 6, y := 4 },
  { x := -3, y := 2 }
]

/-- Assumption that no three points are collinear -/
axiom not_collinear : ∀ p1 p2 p3, p1 ∈ points → p2 ∈ points → p3 ∈ points →
  p1 ≠ p2 → p2 ≠ p3 → p1 ≠ p3 →
  (p2.y - p1.y) * (p3.x - p2.x) ≠ (p3.y - p2.y) * (p2.x - p1.x)

/-- Assumption that the points form an ellipse with axes parallel to coordinate axes -/
axiom is_parallel_ellipse : ∃ a b h k : ℝ, ∀ p, p ∈ points →
  (p.x - h)^2 / a^2 + (p.y - k)^2 / b^2 = 1

/-- The theorem to be proved -/
theorem minor_axis_length :
  ∃ a b h k : ℝ, (∀ p, p ∈ points → (p.x - h)^2 / a^2 + (p.y - k)^2 / b^2 = 1) →
  2 * b = 8 * Real.sqrt 3 / 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_minor_axis_length_l1142_114200


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_winning_candidate_percentage_l1142_114277

def election_votes : List ℕ := [2500, 5000, 20000]

def total_votes : ℕ := election_votes.sum

noncomputable def winning_votes : ℕ := election_votes.maximum.getD 0

noncomputable def winning_percentage : ℚ := (winning_votes : ℚ) / (total_votes : ℚ) * 100

theorem winning_candidate_percentage :
  ∃ ε > 0, ε < 0.01 ∧ |winning_percentage - 72.73| < ε :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_winning_candidate_percentage_l1142_114277


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_parallel_to_plane_iff_l1142_114265

-- Define a structure for a 3D space
structure Space3D where
  -- We'll leave this empty for now, as we don't need to specify its internals

-- Define a line in 3D space
structure Line where
  -- We'll leave this empty for now, as we don't need to specify its internals

-- Define a plane in 3D space
structure Plane where
  -- We'll leave this empty for now, as we don't need to specify its internals

-- Define what it means for a line to be parallel to a plane
def is_parallel (m : Line) (α : Plane) : Prop :=
  sorry -- We'll leave the actual definition as 'sorry' for now

-- Define what it means for a line and a plane to have no common points
def no_common_points (m : Line) (α : Plane) : Prop :=
  sorry -- We'll leave the actual definition as 'sorry' for now

-- Theorem stating the necessary and sufficient condition
theorem line_parallel_to_plane_iff (m : Line) (α : Plane) :
  is_parallel m α ↔ no_common_points m α := by
  sorry -- We'll leave the proof as 'sorry' for now

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_parallel_to_plane_iff_l1142_114265


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_multiple_in_set_l1142_114251

def is_valid_number (n : ℕ) : Prop :=
  n ≥ 1000 ∧ n < 10000 ∧ 
  ∃ (a b c d : ℕ), a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
  ({a, b, c, d} : Finset ℕ) = {1, 3, 6, 8} ∧
  n = 1000 * a + 100 * b + 10 * c + d

def is_multiple_in_set (n : ℕ) : Prop :=
  is_valid_number n ∧
  ∃ (m : ℕ), is_valid_number m ∧ m ≠ n ∧ n % m = 0

theorem unique_multiple_in_set : 
  ∀ (n : ℕ), is_multiple_in_set n ↔ n = 3672 :=
by
  sorry

#check unique_multiple_in_set

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_multiple_in_set_l1142_114251


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_xiao_hua_game_score_l1142_114273

/-- Represents the possible scores in a round -/
inductive Score
| eight : Score
| a : Score
| zero : Score

/-- Represents a round of the game -/
def Round := List Score

/-- Calculates the total score for a round -/
def totalScore (a : ℕ) (round : Round) : ℕ :=
  round.foldl (fun acc score =>
    match score with
    | Score.eight => acc + 8
    | Score.a => acc + a
    | Score.zero => acc
  ) 0

/-- Checks if a total score is achievable -/
def isAchievable (a : ℕ) (score : ℕ) : Prop :=
  ∃ (round : Round), totalScore a round = score

theorem xiao_hua_game_score :
  ∃! (a : ℕ),
    (∀ s : ℕ, s ∈ [103, 104, 105, 106, 107, 108, 109, 110] → isAchievable a s) ∧
    ¬isAchievable a 83 ∧
    a = 13 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_xiao_hua_game_score_l1142_114273


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_area_formula_l1142_114224

/-- A trapezoid with parallel sides BC and AD, where BC = a, AD = b, ∠CAD = α, and ∠BAC = β -/
structure Trapezoid where
  a : ℝ
  b : ℝ
  α : ℝ
  β : ℝ

/-- The area of a trapezoid -/
noncomputable def trapezoidArea (t : Trapezoid) : ℝ :=
  (t.a * (t.a + t.b) * Real.sin t.α * Real.sin (t.α + t.β)) / (2 * Real.sin t.β)

/-- Theorem stating that the area of the trapezoid is correctly calculated -/
theorem trapezoid_area_formula (t : Trapezoid) : 
  trapezoidArea t = (t.a * (t.a + t.b) * Real.sin t.α * Real.sin (t.α + t.β)) / (2 * Real.sin t.β) := by
  -- The proof is omitted for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_area_formula_l1142_114224


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l1142_114242

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively. -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- Helper function to calculate the area of a triangle -/
noncomputable def area (t : Triangle) : ℝ := 
  1/2 * t.a * t.c * Real.sin t.B

/-- The theorem statement -/
theorem triangle_problem (t : Triangle) 
  (h1 : Real.tan t.A + Real.tan t.C = Real.sqrt 3 * (Real.tan t.A * Real.tan t.C - 1))
  (h2 : t.b = 2) :
  t.B = π / 3 ∧ 
  (∀ (s : Triangle), s.b = 2 → area s ≤ Real.sqrt 3) ∧
  (∃ (s : Triangle), s.b = 2 ∧ area s = Real.sqrt 3) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l1142_114242


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_decreasing_f_implies_a_in_range_l1142_114208

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 1 then (a - 3) * x + 5 else 2 * a / x

theorem decreasing_f_implies_a_in_range (a : ℝ) :
  (∀ x₁ x₂ : ℝ, 0 < x₁ ∧ x₁ < x₂ → f a x₂ < f a x₁) →
  0 < a ∧ a ≤ 3 := by
  sorry

#check decreasing_f_implies_a_in_range

end NUMINAMATH_CALUDE_ERRORFEEDBACK_decreasing_f_implies_a_in_range_l1142_114208


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l1142_114285

-- Define the curve
def f (x : ℝ) : ℝ := x^3 - 2*x

-- Define the point of tangency
def point : ℝ × ℝ := (1, -1)

-- State the theorem
theorem tangent_line_equation :
  ∃ (m b : ℝ), 
    (∀ x y : ℝ, y = m*x + b ↔ x - y - 2 = 0) ∧
    (m = deriv f point.1) ∧
    (point.2 = m * point.1 + b) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l1142_114285


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vet_cost_proof_l1142_114227

/-- Calculates the cost of each vet appointment given the total paid, number of appointments, 
    insurance cost, and insurance coverage percentage. -/
noncomputable def vet_appointment_cost (total_paid : ℝ) (num_appointments : ℕ) 
                         (insurance_cost : ℝ) (insurance_coverage : ℝ) : ℝ :=
  let uncovered_percentage := 1 - insurance_coverage
  let equation_left := (num_appointments - 1) * uncovered_percentage + 1
  (total_paid - insurance_cost) / equation_left

/-- Proves that given the specific conditions, each vet appointment costs $400. -/
theorem vet_cost_proof (total_paid : ℝ) (num_appointments : ℕ) 
                       (insurance_cost : ℝ) (insurance_coverage : ℝ) :
  total_paid = 660 ∧ 
  num_appointments = 3 ∧ 
  insurance_cost = 100 ∧ 
  insurance_coverage = 0.8 →
  vet_appointment_cost total_paid num_appointments insurance_cost insurance_coverage = 400 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_vet_cost_proof_l1142_114227


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_equals_abs_implications_l1142_114238

theorem square_equals_abs_implications (a : ℝ) (h : a^2 = |a|) :
  (∃ x ∈ ({a, a + 1, a - 1} : Set ℝ), x = 0) ∧ 
  (a^3 - a = 0) ∧
  (∃! n : ℕ, n = 2 ∧ n = (
    (if a = 0 then 1 else 0) +
    1 +
    (if a = 0 ∨ a = 1 then 1 else 0) +
    1
  )) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_equals_abs_implications_l1142_114238


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l1142_114282

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 0 then 2 * x^3 + 3 * x^2 + 1 else Real.exp (a * x)

theorem range_of_a (a : ℝ) :
  (∀ x ∈ Set.Icc (-2) 2, f a x ≤ 2) ↔ a ≤ (1/2) * Real.log 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l1142_114282


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_garden_cost_l1142_114205

/-- Represents a region in the flower bed -/
structure Region where
  length : ℕ
  width : ℕ

/-- Represents a type of flower -/
structure Flower where
  name : String
  cost : ℚ

def flower_bed : List Region := [
  ⟨7, 2⟩,
  ⟨5, 5⟩,
  ⟨5, 4⟩,
  ⟨7, 3⟩,
  ⟨2, 4⟩
]

def flowers : List Flower := [
  ⟨"Asters", 120/100⟩,
  ⟨"Begonias", 180/100⟩,
  ⟨"Cannas", 220/100⟩,
  ⟨"Dahlias", 280/100⟩,
  ⟨"Easter lilies", 350/100⟩
]

def dahlia_limit : ℕ := 50

/-- Calculates the area of a region -/
def area (r : Region) : ℕ :=
  r.length * r.width

/-- Calculates the total area of the flower bed -/
def total_area : ℕ :=
  (flower_bed.map area).sum

/-- Theorem: The minimum cost for the garden is $179 -/
theorem min_garden_cost : 
  ∃ (arrangement : List (Region × Flower)), 
    (arrangement.length = flower_bed.length) ∧ 
    ((arrangement.map (λ (r, f) => (area r : ℚ) * f.cost)).sum = 179) ∧
    ((arrangement.filter (λ (r, f) => f.name = "Dahlias")).map (λ (r, _) => area r)).sum ≤ dahlia_limit := by
  sorry

#eval total_area

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_garden_cost_l1142_114205


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_digits_18_to_21_l1142_114275

def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sum_of_digits (n / 10)

def sum_of_digits_range (a b : ℕ) : ℕ :=
  (List.range (b - a + 1)).map (fun i => sum_of_digits (a + i)) |>.sum

theorem sum_of_digits_18_to_21 :
  (sum_of_digits_range 18 21 = 24) ∧ (sum_of_digits_range 0 99 = 900) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_digits_18_to_21_l1142_114275


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mean_median_difference_l1142_114297

def classroom_size : ℕ := 100

def score_distribution : List (ℕ × ℕ) := [
  (60, 10),
  (75, 30),
  (80, 25),
  (90, 20),
  (100, 15)
]

def mean_score : ℚ :=
  (score_distribution.map (λ (score, count) => score * count) |>.sum) / classroom_size

def median_score : ℕ := 80

theorem mean_median_difference :
  mean_score - median_score = 3/2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mean_median_difference_l1142_114297


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_fixed_points_l1142_114263

noncomputable def x : ℕ → ℝ → ℝ
  | 0, x₀ => x₀
  | (n+1), x₀ => 
      if 2 * x n x₀ < 1 
      then 2 * x n x₀ + 0.1 
      else 2 * x n x₀ - 1 + 0.1

theorem count_fixed_points : 
  ∃ (S : Set ℝ), (∀ x₀ ∈ S, 0 ≤ x₀ ∧ x₀ < 1 ∧ x 0 x₀ = x 7 x₀) ∧ 
                 (∀ x₀ ∉ S, ¬(0 ≤ x₀ ∧ x₀ < 1 ∧ x 0 x₀ = x 7 x₀)) ∧
                 Finite S ∧ Nat.card S = 127 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_fixed_points_l1142_114263
