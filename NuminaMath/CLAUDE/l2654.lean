import Mathlib

namespace NUMINAMATH_CALUDE_fraction_difference_equals_one_l2654_265459

theorem fraction_difference_equals_one (x y : ℝ) (h : x ≠ y) :
  x / (x - y) - y / (x - y) = 1 := by
  sorry

end NUMINAMATH_CALUDE_fraction_difference_equals_one_l2654_265459


namespace NUMINAMATH_CALUDE_stream_speed_proof_l2654_265408

/-- Proves that the speed of a stream is 5 km/hr given the conditions of a boat's travel --/
theorem stream_speed_proof (boat_speed : ℝ) (downstream_distance : ℝ) (downstream_time : ℝ) :
  boat_speed = 25 →
  downstream_distance = 120 →
  downstream_time = 4 →
  ∃ stream_speed : ℝ, stream_speed = 5 ∧ downstream_distance = (boat_speed + stream_speed) * downstream_time :=
by
  sorry

#check stream_speed_proof

end NUMINAMATH_CALUDE_stream_speed_proof_l2654_265408


namespace NUMINAMATH_CALUDE_function_inequality_condition_l2654_265461

theorem function_inequality_condition (k : ℝ) : 
  (∀ (a x₁ x₂ : ℝ), 1 ≤ a ∧ a ≤ 2 ∧ 2 ≤ x₁ ∧ x₁ < x₂ ∧ x₂ ≤ 4 →
    |x₁ + a / x₁ - 4| - |x₂ + a / x₂ - 4| < k * (x₁ - x₂)) ↔
  k ≤ 6 - 4 * Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_function_inequality_condition_l2654_265461


namespace NUMINAMATH_CALUDE_triangle_theorem_l2654_265499

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the theorem
theorem triangle_theorem (t : Triangle) 
  (h1 : t.a * Real.sin t.C = Real.sqrt (3 * t.c) * Real.cos t.A) :
  t.A = π / 3 ∧ 
  (t.c = 4 ∧ t.a = 5 * Real.sqrt 3 → 
    Real.cos (2 * t.C - t.A) = (17 + 12 * Real.sqrt 7) / 50) := by
  sorry


end NUMINAMATH_CALUDE_triangle_theorem_l2654_265499


namespace NUMINAMATH_CALUDE_place_left_representation_l2654_265494

/-- Represents a three-digit number -/
def ThreeDigitNumber (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

/-- Represents a two-digit number -/
def TwoDigitNumber (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

/-- Represents the operation of placing a three-digit number to the left of a two-digit number -/
def PlaceLeft (x y : ℕ) : ℕ := 100 * x + y

theorem place_left_representation (x y : ℕ) 
  (hx : ThreeDigitNumber x) (hy : TwoDigitNumber y) :
  PlaceLeft x y = 100 * x + y :=
by sorry

end NUMINAMATH_CALUDE_place_left_representation_l2654_265494


namespace NUMINAMATH_CALUDE_physics_class_grades_l2654_265478

theorem physics_class_grades (total_students : ℕ) (prob_A prob_B prob_C : ℚ) :
  total_students = 42 →
  prob_A = 2 * prob_B →
  prob_C = 1.2 * prob_B →
  prob_A + prob_B + prob_C = 1 →
  (prob_B * total_students : ℚ) = 10 :=
by sorry

end NUMINAMATH_CALUDE_physics_class_grades_l2654_265478


namespace NUMINAMATH_CALUDE_sticks_form_triangle_l2654_265421

-- Define the lengths of the sticks
def a : ℝ := 3
def b : ℝ := 4
def c : ℝ := 5

-- Define the triangle inequality theorem
def triangle_inequality (x y z : ℝ) : Prop :=
  x + y > z ∧ x + z > y ∧ y + z > x

-- Theorem statement
theorem sticks_form_triangle : triangle_inequality a b c := by
  sorry

end NUMINAMATH_CALUDE_sticks_form_triangle_l2654_265421


namespace NUMINAMATH_CALUDE_subset_ratio_theorem_l2654_265475

theorem subset_ratio_theorem (n k : ℕ) (h1 : n ≥ 2*k) (h2 : 2*k > 3) :
  (Nat.choose n k = (2*n - k) * Nat.choose n 2) ↔ (n = 27 ∧ k = 4) := by
  sorry

end NUMINAMATH_CALUDE_subset_ratio_theorem_l2654_265475


namespace NUMINAMATH_CALUDE_fence_length_15m_l2654_265437

/-- The length of a fence surrounding a square swimming pool -/
def fence_length (side_length : ℝ) : ℝ := 4 * side_length

/-- Theorem: The length of a fence surrounding a square swimming pool with side length 15 meters is 60 meters -/
theorem fence_length_15m : fence_length 15 = 60 := by
  sorry

end NUMINAMATH_CALUDE_fence_length_15m_l2654_265437


namespace NUMINAMATH_CALUDE_chocolate_bar_count_l2654_265404

theorem chocolate_bar_count (milk_chocolate dark_chocolate white_chocolate : ℕ) 
  (h1 : milk_chocolate = 25)
  (h2 : dark_chocolate = 25)
  (h3 : white_chocolate = 25)
  (h4 : ∃ (total : ℕ), total > 0 ∧ 
    milk_chocolate = total / 4 ∧ 
    dark_chocolate = total / 4 ∧ 
    white_chocolate = total / 4) :
  ∃ (almond_chocolate : ℕ), almond_chocolate = 25 :=
sorry

end NUMINAMATH_CALUDE_chocolate_bar_count_l2654_265404


namespace NUMINAMATH_CALUDE_impossible_division_l2654_265405

/-- Represents a chess-like board with alternating colors -/
def Board := Fin 8 → Fin 8 → Bool

/-- Represents an L-shaped piece on the board -/
structure LPiece :=
  (x : Fin 8) (y : Fin 8)

/-- Checks if an L-piece is valid (within bounds and not in the cut-out corner) -/
def isValidPiece (b : Board) (p : LPiece) : Prop :=
  p.x < 6 ∧ p.y < 6 ∧ ¬(p.x = 0 ∧ p.y = 0)

/-- Counts the number of squares of each color covered by an L-piece -/
def colorCount (b : Board) (p : LPiece) : Nat × Nat :=
  let trueCount := (b p.x p.y).toNat + (b p.x (p.y + 1)).toNat + (b (p.x + 1) p.y).toNat + (b (p.x + 1) (p.y + 1)).toNat
  (trueCount, 4 - trueCount)

/-- The main theorem stating that it's impossible to divide the board as required -/
theorem impossible_division (b : Board) : ¬ ∃ (pieces : List LPiece),
  pieces.length = 15 ∧ 
  (∀ p ∈ pieces, isValidPiece b p) ∧
  (∀ p ∈ pieces, (colorCount b p).1 = 3 ∨ (colorCount b p).2 = 3) ∧
  (pieces.map (λ p => (colorCount b p).1)).sum = 30 :=
sorry

end NUMINAMATH_CALUDE_impossible_division_l2654_265405


namespace NUMINAMATH_CALUDE_sales_volume_correct_profit_at_95_max_profit_at_110_l2654_265466

/-- Represents the weekly sales volume as a function of selling price -/
def sales_volume (x : ℝ) : ℝ := -10 * x + 1500

/-- Represents the weekly profit as a function of selling price -/
def profit (x : ℝ) : ℝ := (x - 80) * sales_volume x

/-- The cost price of each shirt -/
def cost_price : ℝ := 80

/-- The minimum allowed selling price -/
def min_price : ℝ := 90

/-- The maximum allowed selling price -/
def max_price : ℝ := 110

theorem sales_volume_correct :
  sales_volume 90 = 600 ∧ 
  ∀ x, sales_volume (x + 1) = sales_volume x - 10 := by sorry

theorem profit_at_95 : profit 95 = 8250 := by sorry

theorem max_profit_at_110 : 
  profit 110 = 12000 ∧
  ∀ x, min_price ≤ x ∧ x ≤ max_price → profit x ≤ 12000 := by sorry

end NUMINAMATH_CALUDE_sales_volume_correct_profit_at_95_max_profit_at_110_l2654_265466


namespace NUMINAMATH_CALUDE_vasya_fish_count_l2654_265446

/-- Represents the number of fish Vasya caught -/
def total_fish : ℕ := 10

/-- Represents the weight of the three largest fish as a fraction of the total catch -/
def largest_fish_weight_fraction : ℚ := 35 / 100

/-- Represents the weight of the three smallest fish as a fraction of the remaining catch -/
def smallest_fish_weight_fraction : ℚ := 5 / 13

/-- Represents the number of largest fish -/
def num_largest_fish : ℕ := 3

/-- Represents the number of smallest fish -/
def num_smallest_fish : ℕ := 3

theorem vasya_fish_count :
  ∃ (x : ℕ),
    total_fish = num_largest_fish + x + num_smallest_fish ∧
    (1 - largest_fish_weight_fraction) * smallest_fish_weight_fraction = 
      (25 : ℚ) / 100 ∧
    (35 : ℚ) / 3 ≤ (40 : ℚ) / x ∧
    (40 : ℚ) / x ≤ (25 : ℚ) / 3 :=
sorry

end NUMINAMATH_CALUDE_vasya_fish_count_l2654_265446


namespace NUMINAMATH_CALUDE_point_C_coordinates_l2654_265465

/-- Triangle ABC with right angle at B and altitude from A to D on BC -/
structure RightTriangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ
  h_A : A = (7, 1)
  h_B : B = (5, -3)
  h_D : D = (5, 1)
  h_right_angle : (A.1 - B.1) * (C.1 - B.1) + (A.2 - B.2) * (C.2 - B.2) = 0
  h_altitude : (A.1 - D.1) * (C.1 - B.1) + (A.2 - D.2) * (C.2 - B.2) = 0

/-- The coordinates of point C in the given right triangle -/
theorem point_C_coordinates (t : RightTriangle) : t.C = (5, 5) := by
  sorry

end NUMINAMATH_CALUDE_point_C_coordinates_l2654_265465


namespace NUMINAMATH_CALUDE_sin_alpha_value_l2654_265412

theorem sin_alpha_value (α : Real) 
  (h : Real.cos (α - π / 2) = 1 / 3) : 
  Real.sin α = 1 / 3 := by
sorry

end NUMINAMATH_CALUDE_sin_alpha_value_l2654_265412


namespace NUMINAMATH_CALUDE_total_skips_is_33_l2654_265492

/-- Represents the number of skips for each throw -/
structure ThrowSkips :=
  (first : ℕ)
  (second : ℕ)
  (third : ℕ)
  (fourth : ℕ)
  (fifth : ℕ)

/-- Conditions for the stone skipping problem -/
def SkipConditions (t : ThrowSkips) : Prop :=
  t.second = t.first + 2 ∧
  t.third = 2 * t.second ∧
  t.fourth = t.third - 3 ∧
  t.fifth = t.fourth + 1 ∧
  t.fifth = 8

/-- The total number of skips across all throws -/
def TotalSkips (t : ThrowSkips) : ℕ :=
  t.first + t.second + t.third + t.fourth + t.fifth

/-- Theorem stating that the total number of skips is 33 -/
theorem total_skips_is_33 (t : ThrowSkips) (h : SkipConditions t) :
  TotalSkips t = 33 := by
  sorry

end NUMINAMATH_CALUDE_total_skips_is_33_l2654_265492


namespace NUMINAMATH_CALUDE_customized_notebook_combinations_l2654_265429

/-- The number of different notebook designs available. -/
def notebook_designs : ℕ := 12

/-- The number of different pen types available. -/
def pen_types : ℕ := 3

/-- The number of different sticker varieties available. -/
def sticker_varieties : ℕ := 5

/-- The total number of possible combinations for a customized notebook package. -/
def total_combinations : ℕ := notebook_designs * pen_types * sticker_varieties

/-- Theorem stating that the total number of combinations is 180. -/
theorem customized_notebook_combinations :
  total_combinations = 180 := by sorry

end NUMINAMATH_CALUDE_customized_notebook_combinations_l2654_265429


namespace NUMINAMATH_CALUDE_prism_volume_l2654_265417

/-- The volume of a right rectangular prism with specific face areas and dimension ratio -/
theorem prism_volume (a b c : ℝ) : 
  a * b = 64 → 
  b * c = 81 → 
  a * c = 72 → 
  b = 2 * a → 
  |a * b * c - 1629| < 1 := by
sorry

end NUMINAMATH_CALUDE_prism_volume_l2654_265417


namespace NUMINAMATH_CALUDE_magnitude_AC_l2654_265487

def vector_AB : Fin 2 → ℝ := ![1, 2]
def vector_BC : Fin 2 → ℝ := ![3, 4]

theorem magnitude_AC : 
  let vector_AC := (vector_BC 0 - (-vector_AB 0), vector_BC 1 - (-vector_AB 1))
  Real.sqrt ((vector_AC.1)^2 + (vector_AC.2)^2) = 2 * Real.sqrt 13 := by
  sorry

end NUMINAMATH_CALUDE_magnitude_AC_l2654_265487


namespace NUMINAMATH_CALUDE_circle_equation_l2654_265498

/-- A circle with center (1, -2) and radius 3 -/
structure Circle where
  center : ℝ × ℝ := (1, -2)
  radius : ℝ := 3

/-- A point on the 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Predicate to check if a point lies on the circle -/
def onCircle (c : Circle) (p : Point) : Prop :=
  (p.x - c.center.1)^2 + (p.y - c.center.2)^2 = c.radius^2

theorem circle_equation (c : Circle) (p : Point) :
  onCircle c p ↔ (p.x - 1)^2 + (p.y + 2)^2 = 9 := by
  sorry

end NUMINAMATH_CALUDE_circle_equation_l2654_265498


namespace NUMINAMATH_CALUDE_min_sum_squares_l2654_265433

theorem min_sum_squares (a b c d e f g h : ℤ) : 
  a ∈ ({-7, -5, -3, -1, 0, 2, 4, 5} : Set ℤ) →
  b ∈ ({-7, -5, -3, -1, 0, 2, 4, 5} : Set ℤ) →
  c ∈ ({-7, -5, -3, -1, 0, 2, 4, 5} : Set ℤ) →
  d ∈ ({-7, -5, -3, -1, 0, 2, 4, 5} : Set ℤ) →
  e ∈ ({-7, -5, -3, -1, 0, 2, 4, 5} : Set ℤ) →
  f ∈ ({-7, -5, -3, -1, 0, 2, 4, 5} : Set ℤ) →
  g ∈ ({-7, -5, -3, -1, 0, 2, 4, 5} : Set ℤ) →
  h ∈ ({-7, -5, -3, -1, 0, 2, 4, 5} : Set ℤ) →
  a ≠ b → a ≠ c → a ≠ d → a ≠ e → a ≠ f → a ≠ g → a ≠ h →
  b ≠ c → b ≠ d → b ≠ e → b ≠ f → b ≠ g → b ≠ h →
  c ≠ d → c ≠ e → c ≠ f → c ≠ g → c ≠ h →
  d ≠ e → d ≠ f → d ≠ g → d ≠ h →
  e ≠ f → e ≠ g → e ≠ h →
  f ≠ g → f ≠ h →
  g ≠ h →
  (a + b + c + d)^2 + (e + f + g + h)^2 ≥ 34 := by
sorry

end NUMINAMATH_CALUDE_min_sum_squares_l2654_265433


namespace NUMINAMATH_CALUDE_family_weight_theorem_l2654_265426

/-- Represents the weights of a family with three generations -/
structure FamilyWeights where
  mother : ℝ
  daughter : ℝ
  grandchild : ℝ

/-- The total weight of the family -/
def FamilyWeights.total (w : FamilyWeights) : ℝ :=
  w.mother + w.daughter + w.grandchild

/-- The conditions given in the problem -/
def WeightConditions (w : FamilyWeights) : Prop :=
  w.daughter + w.grandchild = 60 ∧
  w.grandchild = (1/5) * w.mother ∧
  w.daughter = 50

/-- Theorem stating that given the conditions, the total weight is 110 kg -/
theorem family_weight_theorem (w : FamilyWeights) (h : WeightConditions w) :
  w.total = 110 := by
  sorry


end NUMINAMATH_CALUDE_family_weight_theorem_l2654_265426


namespace NUMINAMATH_CALUDE_tully_kate_age_ratio_l2654_265484

/-- Represents a person's age --/
structure Person where
  name : String
  current_age : ℕ

/-- Calculates the ratio of two numbers --/
def ratio (a b : ℕ) : ℚ := a / b

theorem tully_kate_age_ratio :
  let tully : Person := { name := "Tully", current_age := 61 }
  let kate : Person := { name := "Kate", current_age := 29 }
  let tully_future_age := tully.current_age + 3
  let kate_future_age := kate.current_age + 3
  ratio tully_future_age kate_future_age = 2 := by
  sorry

end NUMINAMATH_CALUDE_tully_kate_age_ratio_l2654_265484


namespace NUMINAMATH_CALUDE_area_ratio_quad_to_decagon_l2654_265414

-- Define a regular decagon
structure RegularDecagon where
  vertices : Fin 10 → ℝ × ℝ
  is_regular : sorry

-- Define the area of a polygon
def area (polygon : List (ℝ × ℝ)) : ℝ := sorry

-- Define the quadrilateral ACEG within the decagon
def quadACEG (d : RegularDecagon) : List (ℝ × ℝ) :=
  [d.vertices 0, d.vertices 2, d.vertices 4, d.vertices 6]

-- Define the decagon as a list of points
def decagonPoints (d : RegularDecagon) : List (ℝ × ℝ) :=
  (List.range 10).map d.vertices

theorem area_ratio_quad_to_decagon (d : RegularDecagon) :
  area (quadACEG d) / area (decagonPoints d) = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_area_ratio_quad_to_decagon_l2654_265414


namespace NUMINAMATH_CALUDE_investment_loss_calculation_l2654_265400

/-- Represents the capital and loss of two investors -/
structure InvestmentScenario where
  capital_ratio : ℚ  -- Ratio of smaller capital to larger capital
  larger_loss : ℚ    -- Loss of the investor with larger capital
  total_loss : ℚ     -- Total loss of both investors

/-- Theorem stating the relationship between capital ratio, larger investor's loss, and total loss -/
theorem investment_loss_calculation (scenario : InvestmentScenario) 
  (h1 : scenario.capital_ratio = 1 / 9)
  (h2 : scenario.larger_loss = 1080) :
  scenario.total_loss = 1200 := by
  sorry

end NUMINAMATH_CALUDE_investment_loss_calculation_l2654_265400


namespace NUMINAMATH_CALUDE_kim_integer_problem_l2654_265471

theorem kim_integer_problem (x y : ℤ) : 
  3 * x + 2 * y = 145 → (x = 35 ∨ y = 35) → (x = 20 ∨ y = 20) :=
by sorry

end NUMINAMATH_CALUDE_kim_integer_problem_l2654_265471


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l2654_265469

theorem sufficient_not_necessary :
  (∀ x y : ℝ, x > 1 ∧ y > 1 → x + y > 2) ∧
  (∃ x y : ℝ, x + y > 2 ∧ ¬(x > 1 ∧ y > 1)) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l2654_265469


namespace NUMINAMATH_CALUDE_article_cost_l2654_265442

theorem article_cost (decreased_price : ℝ) (decrease_percentage : ℝ) (actual_cost : ℝ) :
  decreased_price = 200 ∧
  decrease_percentage = 20 ∧
  decreased_price = actual_cost * (1 - decrease_percentage / 100) →
  actual_cost = 250 := by
  sorry

end NUMINAMATH_CALUDE_article_cost_l2654_265442


namespace NUMINAMATH_CALUDE_product_remainder_l2654_265496

theorem product_remainder (a b c d : ℕ) (h1 : a = 1729) (h2 : b = 1865) (h3 : c = 1912) (h4 : d = 2023) :
  (a * b * c * d) % 7 = 6 := by
  sorry

end NUMINAMATH_CALUDE_product_remainder_l2654_265496


namespace NUMINAMATH_CALUDE_salary_sum_l2654_265463

/-- Given 5 individuals with an average salary and one known salary, 
    prove the sum of the other four salaries. -/
theorem salary_sum (average_salary : ℕ) (known_salary : ℕ) : 
  average_salary = 8800 → known_salary = 8000 → 
  (5 * average_salary) - known_salary = 36000 := by
  sorry

end NUMINAMATH_CALUDE_salary_sum_l2654_265463


namespace NUMINAMATH_CALUDE_power_two_gt_two_n_plus_one_power_two_le_two_n_plus_one_for_small_n_smallest_n_for_inequality_l2654_265460

theorem power_two_gt_two_n_plus_one (n : ℕ) : n ≥ 3 → 2^n > 2*n + 1 :=
  sorry

theorem power_two_le_two_n_plus_one_for_small_n :
  (2^1 ≤ 2*1 + 1) ∧ (2^2 ≤ 2*2 + 1) :=
  sorry

theorem smallest_n_for_inequality : ∀ n : ℕ, n ≥ 3 ↔ 2^n > 2*n + 1 :=
  sorry

end NUMINAMATH_CALUDE_power_two_gt_two_n_plus_one_power_two_le_two_n_plus_one_for_small_n_smallest_n_for_inequality_l2654_265460


namespace NUMINAMATH_CALUDE_ratio_to_percent_l2654_265476

theorem ratio_to_percent (a b : ℚ) (h : a / b = 2 / 10) : (a / b) * 100 = 20 := by
  sorry

end NUMINAMATH_CALUDE_ratio_to_percent_l2654_265476


namespace NUMINAMATH_CALUDE_data_mode_is_neg_one_l2654_265472

def data : List Int := [-1, 0, 2, -1, 3]

def mode (l : List α) [DecidableEq α] : Option α :=
  l.foldl (λ acc x => 
    match acc with
    | none => some x
    | some y => if l.count x > l.count y then some x else some y
  ) none

theorem data_mode_is_neg_one : mode data = some (-1) := by
  sorry

end NUMINAMATH_CALUDE_data_mode_is_neg_one_l2654_265472


namespace NUMINAMATH_CALUDE_intersection_A_B_complement_union_A_B_l2654_265490

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | -4 < x ∧ x < 4}
def B : Set ℝ := {x : ℝ | x ≤ 1 ∨ x ≥ 3}

-- Theorem for part (1)
theorem intersection_A_B : A ∩ B = {x : ℝ | (-4 < x ∧ x ≤ 1) ∨ (3 ≤ x ∧ x < 4)} := by
  sorry

-- Theorem for part (2)
theorem complement_union_A_B : (A ∪ B)ᶜ = ∅ := by
  sorry

end NUMINAMATH_CALUDE_intersection_A_B_complement_union_A_B_l2654_265490


namespace NUMINAMATH_CALUDE_drawing_red_ball_certain_l2654_265410

/-- A bag containing only red balls -/
structure RedBallBag where
  num_balls : ℕ
  all_red : True

/-- The probability of drawing a red ball from a bag of red balls -/
def prob_draw_red (bag : RedBallBag) : ℝ :=
  1

/-- An event is certain if its probability is 1 -/
def is_certain_event (p : ℝ) : Prop :=
  p = 1

/-- Theorem: Drawing a red ball from a bag containing only 5 red balls is a certain event -/
theorem drawing_red_ball_certain (bag : RedBallBag) (h : bag.num_balls = 5) :
    is_certain_event (prob_draw_red bag) := by
  sorry

end NUMINAMATH_CALUDE_drawing_red_ball_certain_l2654_265410


namespace NUMINAMATH_CALUDE_shooting_test_probability_l2654_265415

/-- The probability of a successful shot -/
def p : ℚ := 2/3

/-- The number of successful shots required to pass -/
def required_successes : ℕ := 3

/-- The maximum number of shots allowed -/
def max_shots : ℕ := 5

/-- The probability of passing the shooting test -/
def pass_probability : ℚ := 64/81

theorem shooting_test_probability :
  (p^required_successes) +
  (Nat.choose 4 required_successes * p^required_successes * (1-p)) +
  (Nat.choose 5 required_successes * p^required_successes * (1-p)^2) = pass_probability :=
sorry

end NUMINAMATH_CALUDE_shooting_test_probability_l2654_265415


namespace NUMINAMATH_CALUDE_question_mark_solution_l2654_265488

theorem question_mark_solution : ∃! x : ℤ, x + 3699 + 1985 - 2047 = 31111 :=
  sorry

end NUMINAMATH_CALUDE_question_mark_solution_l2654_265488


namespace NUMINAMATH_CALUDE_unique_solution_l2654_265482

-- Define the function g
def g (x : ℝ) : ℝ := (x - 1)^5 + (x - 1) - 34

-- State the theorem
theorem unique_solution :
  ∃! x : ℝ, g x = 0 ∧ x = 3 :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_l2654_265482


namespace NUMINAMATH_CALUDE_valid_n_set_l2654_265409

def is_valid_n (n : ℕ) : Prop :=
  ∃ m : ℕ,
    n > 1 ∧
    (∀ d : ℕ, d ∣ n ∧ d ≠ 1 ∧ d ≠ n → ∃ k : ℕ, k ∣ m ∧ k ≠ 1 ∧ k ≠ m ∧ k = d + 1) ∧
    (∀ k : ℕ, k ∣ m ∧ k ≠ 1 ∧ k ≠ m → ∃ d : ℕ, d ∣ n ∧ d ≠ 1 ∧ d ≠ n ∧ k = d + 1)

theorem valid_n_set : {n : ℕ | is_valid_n n} = {4, 8} := by sorry

end NUMINAMATH_CALUDE_valid_n_set_l2654_265409


namespace NUMINAMATH_CALUDE_greatest_integer_radius_l2654_265430

theorem greatest_integer_radius (A : ℝ) (h : A < 80 * Real.pi) :
  ∃ (r : ℕ), r * r * Real.pi = A ∧ r ≤ 8 ∧ ∀ (s : ℕ), s * s * Real.pi = A → s ≤ r :=
sorry

end NUMINAMATH_CALUDE_greatest_integer_radius_l2654_265430


namespace NUMINAMATH_CALUDE_complement_A_intersect_B_l2654_265402

-- Define set A
def A : Set ℝ := {x | x^2 + x - 2 > 0}

-- Define set B
def B : Set ℝ := {y | ∃ x > 0, y = Real.log x / Real.log 2}

-- Define the complement of A in ℝ
def complement_A : Set ℝ := {x | x ∉ A}

-- Theorem statement
theorem complement_A_intersect_B : complement_A ∩ B = Set.Icc (-2) 1 := by
  sorry

end NUMINAMATH_CALUDE_complement_A_intersect_B_l2654_265402


namespace NUMINAMATH_CALUDE_sin_arctan_x_equals_x_l2654_265447

theorem sin_arctan_x_equals_x (x : ℝ) :
  x > 0 →
  Real.sin (Real.arctan x) = x →
  x^4 = (3 - Real.sqrt 5) / 2 := by
sorry

end NUMINAMATH_CALUDE_sin_arctan_x_equals_x_l2654_265447


namespace NUMINAMATH_CALUDE_min_abs_GB_is_392_l2654_265401

-- Define the Revolution polynomial
def Revolution (U S A : ℤ) (x : ℤ) : ℤ := x^3 + U*x^2 + S*x + A

-- State the theorem
theorem min_abs_GB_is_392 
  (U S A G B : ℤ) 
  (h1 : U + S + A + 1 = 1773)
  (h2 : ∀ x, Revolution U S A x = 0 ↔ x = G ∨ x = B)
  (h3 : G ≠ B)
  (h4 : G ≠ 0)
  (h5 : B ≠ 0) :
  ∃ (G' B' : ℤ), G' * B' = 392 ∧ 
    ∀ (G'' B'' : ℤ), G'' ≠ 0 ∧ B'' ≠ 0 ∧ 
      (∀ x, Revolution U S A x = 0 ↔ x = G'' ∨ x = B'') → 
      abs (G'' * B'') ≥ 392 :=
sorry

end NUMINAMATH_CALUDE_min_abs_GB_is_392_l2654_265401


namespace NUMINAMATH_CALUDE_triangle_satisfies_equation_l2654_265418

/-- Converts a number from base 5 to base 10 -/
def base5To10 (d1 d2 : ℕ) : ℕ := 5 * d1 + d2

/-- Converts a number from base 12 to base 10 -/
def base12To10 (d1 d2 : ℕ) : ℕ := 12 * d1 + d2

/-- The digit satisfying the equation in base 5 and base 12 -/
def triangle : ℕ := 2

theorem triangle_satisfies_equation :
  base5To10 5 triangle = base12To10 triangle 3 ∧ triangle < 10 := by sorry

end NUMINAMATH_CALUDE_triangle_satisfies_equation_l2654_265418


namespace NUMINAMATH_CALUDE_total_pencils_l2654_265464

theorem total_pencils (jessica_pencils sandy_pencils jason_pencils : ℕ) 
  (h1 : jessica_pencils = 8)
  (h2 : sandy_pencils = 8)
  (h3 : jason_pencils = 8) :
  jessica_pencils + sandy_pencils + jason_pencils = 24 := by
  sorry

end NUMINAMATH_CALUDE_total_pencils_l2654_265464


namespace NUMINAMATH_CALUDE_bernardo_wins_l2654_265486

def game_winner (N : ℕ) : Prop :=
  N ≤ 999 ∧
  2 * N < 1000 ∧
  2 * N + 100 < 1000 ∧
  4 * N + 200 < 1000 ∧
  4 * N + 300 < 1000 ∧
  8 * N + 600 < 1000 ∧
  8 * N + 700 ≥ 1000

def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sum_of_digits (n / 10)

theorem bernardo_wins :
  ∃ N, game_winner N ∧ 
    (∀ M, M < N → ¬game_winner M) ∧
    N = 38 ∧
    sum_of_digits N = 11 :=
  sorry

end NUMINAMATH_CALUDE_bernardo_wins_l2654_265486


namespace NUMINAMATH_CALUDE_coefficient_x_cubed_expansion_l2654_265436

theorem coefficient_x_cubed_expansion : 
  let expansion := (1 + X : Polynomial ℤ)^5 - (1 + X : Polynomial ℤ)^6
  expansion.coeff 3 = -10 := by
  sorry

end NUMINAMATH_CALUDE_coefficient_x_cubed_expansion_l2654_265436


namespace NUMINAMATH_CALUDE_octal_multiplication_53_26_l2654_265427

/-- Converts an octal number to decimal --/
def octal_to_decimal (n : ℕ) : ℕ := sorry

/-- Converts a decimal number to octal --/
def decimal_to_octal (n : ℕ) : ℕ := sorry

/-- Multiplies two octal numbers --/
def octal_multiply (a b : ℕ) : ℕ :=
  decimal_to_octal (octal_to_decimal a * octal_to_decimal b)

theorem octal_multiplication_53_26 :
  octal_multiply 53 26 = 1662 := by sorry

end NUMINAMATH_CALUDE_octal_multiplication_53_26_l2654_265427


namespace NUMINAMATH_CALUDE_gcd_g_50_52_eq_one_l2654_265422

/-- The function g(x) = x^2 - 3x + 2023 -/
def g (x : ℤ) : ℤ := x^2 - 3*x + 2023

/-- Theorem: The greatest common divisor of g(50) and g(52) is 1 -/
theorem gcd_g_50_52_eq_one : Int.gcd (g 50) (g 52) = 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_g_50_52_eq_one_l2654_265422


namespace NUMINAMATH_CALUDE_trig_identity_1_trig_identity_2_l2654_265407

-- Part 1
theorem trig_identity_1 (α β : Real) 
  (h1 : Real.tan (α + β) = 2/5)
  (h2 : Real.tan (β - π/4) = 1/4) :
  (Real.cos α + Real.sin α) / (Real.cos α - Real.sin α) = 3/22 := by
  sorry

-- Part 2
theorem trig_identity_2 (α β : Real) 
  (h1 : 0 < α ∧ α < π/2)
  (h2 : 0 < β ∧ β < π/2)
  (h3 : Real.cos (α + β) = Real.sqrt 5 / 5)
  (h4 : Real.sin (α - β) = Real.sqrt 10 / 10) :
  2 * β = π/4 := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_1_trig_identity_2_l2654_265407


namespace NUMINAMATH_CALUDE_airplane_seats_l2654_265479

theorem airplane_seats : 
  ∀ (total : ℕ),
  (24 : ℕ) + (total / 4 : ℕ) + (2 * total / 3 : ℕ) = total →
  total = 288 := by
sorry

end NUMINAMATH_CALUDE_airplane_seats_l2654_265479


namespace NUMINAMATH_CALUDE_polynomial_simplification_l2654_265467

theorem polynomial_simplification (x : ℝ) : 
  (2 * x^6 + 3 * x^5 + x^4 + x^3 + x + 10) - (x^6 + 4 * x^5 + 2 * x^4 - x^3 + 12) = 
  x^6 - x^5 - x^4 + 2 * x^3 + x - 2 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l2654_265467


namespace NUMINAMATH_CALUDE_roots_theorem_l2654_265425

theorem roots_theorem :
  (∃ x : ℝ, x > 0 ∧ x^2 = 4 ∧ x = 2) ∧
  (∃ x y : ℝ, x^2 = 9 ∧ y^2 = 9 ∧ x = 3 ∧ y = -3) ∧
  (∃ x : ℝ, x^3 = -27 ∧ x = -3) :=
by sorry

end NUMINAMATH_CALUDE_roots_theorem_l2654_265425


namespace NUMINAMATH_CALUDE_cyclic_inequality_sqrt_l2654_265435

theorem cyclic_inequality_sqrt (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (Real.sqrt (3 * x * (x + y) * (y + z)) +
   Real.sqrt (3 * y * (y + z) * (z + x)) +
   Real.sqrt (3 * z * (z + x) * (x + y))) ≤
  Real.sqrt (4 * (x + y + z)^3) := by
  sorry

end NUMINAMATH_CALUDE_cyclic_inequality_sqrt_l2654_265435


namespace NUMINAMATH_CALUDE_chromatic_number_le_max_degree_plus_one_l2654_265473

/-- A graph is represented by its vertex set and an adjacency relation -/
structure Graph (V : Type*) where
  adj : V → V → Prop

/-- The degree of a vertex in a graph -/
def degree (G : Graph V) (v : V) : ℕ := sorry

/-- The maximum degree of a graph -/
def maxDegree (G : Graph V) : ℕ := sorry

/-- A coloring of a graph is a function from vertices to colors -/
def isColoring (G : Graph V) (f : V → ℕ) : Prop :=
  ∀ u v : V, G.adj u v → f u ≠ f v

/-- The chromatic number of a graph -/
def chromaticNumber (G : Graph V) : ℕ := sorry

/-- Theorem: The chromatic number of a graph is at most one more than its maximum degree -/
theorem chromatic_number_le_max_degree_plus_one (V : Type*) (G : Graph V) :
  chromaticNumber G ≤ maxDegree G + 1 := by sorry

end NUMINAMATH_CALUDE_chromatic_number_le_max_degree_plus_one_l2654_265473


namespace NUMINAMATH_CALUDE_sandwich_meat_cost_l2654_265438

/-- The cost of a pack of sandwich meat given the following conditions:
  * 1 loaf of bread, 2 packs of sandwich meat, and 2 packs of sliced cheese make 10 sandwiches
  * Bread costs $4.00
  * Cheese costs $4.00 per pack
  * There's a $1.00 off coupon for one pack of cheese
  * There's a $1.00 off coupon for one pack of meat
  * Each sandwich costs $2.00
-/
theorem sandwich_meat_cost :
  let bread_cost : ℚ := 4
  let cheese_cost : ℚ := 4
  let cheese_discount : ℚ := 1
  let meat_discount : ℚ := 1
  let sandwich_cost : ℚ := 2
  let sandwich_count : ℕ := 10
  let total_cost : ℚ := sandwich_cost * sandwich_count
  let cheese_total : ℚ := 2 * cheese_cost - cheese_discount
  ∃ meat_cost : ℚ,
    bread_cost + cheese_total + 2 * meat_cost - meat_discount = total_cost ∧
    meat_cost = 5 :=
by sorry

end NUMINAMATH_CALUDE_sandwich_meat_cost_l2654_265438


namespace NUMINAMATH_CALUDE_roller_coaster_runs_l2654_265445

def people_in_line : ℕ := 1532
def num_cars : ℕ := 8
def seats_per_car : ℕ := 3

def capacity_per_ride : ℕ := num_cars * seats_per_car

theorem roller_coaster_runs : 
  ∃ (runs : ℕ), runs * capacity_per_ride ≥ people_in_line ∧ 
  ∀ (k : ℕ), k * capacity_per_ride ≥ people_in_line → k ≥ runs :=
by sorry

end NUMINAMATH_CALUDE_roller_coaster_runs_l2654_265445


namespace NUMINAMATH_CALUDE_point_on_y_axis_has_x_zero_l2654_265493

/-- A point on a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Definition of a point lying on the y-axis -/
def lies_on_y_axis (p : Point) : Prop := p.x = 0

/-- Theorem: If a point lies on the y-axis, its x-coordinate is 0 -/
theorem point_on_y_axis_has_x_zero (M : Point) (h : lies_on_y_axis M) : M.x = 0 := by
  sorry

end NUMINAMATH_CALUDE_point_on_y_axis_has_x_zero_l2654_265493


namespace NUMINAMATH_CALUDE_simplified_fraction_sum_l2654_265419

theorem simplified_fraction_sum (c d : ℕ+) : 
  (c : ℚ) / d = 0.375 ∧ 
  ∀ (a b : ℕ+), (a : ℚ) / b = 0.375 → c ≤ a ∧ d ≤ b → 
  c + d = 11 := by sorry

end NUMINAMATH_CALUDE_simplified_fraction_sum_l2654_265419


namespace NUMINAMATH_CALUDE_cube_root_of_negative_eight_l2654_265431

theorem cube_root_of_negative_eight (x : ℝ) : x^3 = -8 → x = -2 := by
  sorry

end NUMINAMATH_CALUDE_cube_root_of_negative_eight_l2654_265431


namespace NUMINAMATH_CALUDE_percentage_error_calculation_l2654_265455

theorem percentage_error_calculation (N : ℝ) (h : N > 0) : 
  let correct := N * 5
  let incorrect := N / 10
  let absolute_error := |correct - incorrect|
  let percentage_error := (absolute_error / correct) * 100
  percentage_error = 98 := by
sorry

end NUMINAMATH_CALUDE_percentage_error_calculation_l2654_265455


namespace NUMINAMATH_CALUDE_motorcyclists_speeds_l2654_265440

/-- The length of the circular track in meters -/
def track_length : ℝ := 1000

/-- The time interval between overtakes in minutes -/
def overtake_interval : ℝ := 2

/-- The initial speed of motorcyclist A in meters per minute -/
def speed_A : ℝ := 1000

/-- The initial speed of motorcyclist B in meters per minute -/
def speed_B : ℝ := 1500

/-- Theorem stating the conditions and the conclusion about the motorcyclists' speeds -/
theorem motorcyclists_speeds :
  (speed_B - speed_A) * overtake_interval = track_length ∧
  (2 * speed_A - speed_B) * overtake_interval = track_length →
  speed_A = 1000 ∧ speed_B = 1500 := by
  sorry

end NUMINAMATH_CALUDE_motorcyclists_speeds_l2654_265440


namespace NUMINAMATH_CALUDE_unique_score_with_four_ways_l2654_265406

/-- AMC scoring system -/
structure AMCScore where
  correct : ℕ
  unanswered : ℕ
  incorrect : ℕ
  total_questions : ℕ
  score : ℕ

/-- Predicate for valid AMC score -/
def is_valid_score (s : AMCScore) : Prop :=
  s.correct + s.unanswered + s.incorrect = s.total_questions ∧
  s.score = 7 * s.correct + 3 * s.unanswered

/-- Theorem: Unique score with exactly four distinct ways to achieve it -/
theorem unique_score_with_four_ways :
  ∃! S : ℕ, 
    (∃ scores : Finset AMCScore, 
      (∀ s ∈ scores, is_valid_score s ∧ s.total_questions = 30 ∧ s.score = S) ∧
      scores.card = 4 ∧
      (∀ s : AMCScore, is_valid_score s ∧ s.total_questions = 30 ∧ s.score = S → s ∈ scores)) ∧
    S = 240 := by
  sorry

end NUMINAMATH_CALUDE_unique_score_with_four_ways_l2654_265406


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l2654_265444

theorem imaginary_part_of_z (i : ℂ) (h : i^2 = -1) : 
  Complex.im (i * (i - 3)) = -3 := by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l2654_265444


namespace NUMINAMATH_CALUDE_average_goals_l2654_265468

theorem average_goals (layla_goals : ℕ) (kristin_difference : ℕ) (num_games : ℕ) :
  layla_goals = 104 →
  kristin_difference = 24 →
  num_games = 4 →
  (layla_goals + (layla_goals - kristin_difference)) / num_games = 46 :=
by
  sorry

end NUMINAMATH_CALUDE_average_goals_l2654_265468


namespace NUMINAMATH_CALUDE_mutual_fund_yield_range_l2654_265489

/-- The range of annual yields for mutual funds increased by 15% --/
def yield_increase_rate : ℝ := 0.15

/-- The number of mutual funds --/
def num_funds : ℕ := 100

/-- The new range of annual yields after the increase --/
def new_range : ℝ := 11500

/-- The original range of annual yields --/
def original_range : ℝ := 10000

theorem mutual_fund_yield_range : 
  (1 + yield_increase_rate) * original_range = new_range := by
  sorry

end NUMINAMATH_CALUDE_mutual_fund_yield_range_l2654_265489


namespace NUMINAMATH_CALUDE_min_value_inequality_l2654_265491

theorem min_value_inequality (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  (x + y) * (1/x + 4/y) ≥ 9 ∧
  ((x + y) * (1/x + 4/y) = 9 ↔ y/x = 4*x/y) :=
sorry

end NUMINAMATH_CALUDE_min_value_inequality_l2654_265491


namespace NUMINAMATH_CALUDE_class_size_l2654_265424

theorem class_size (french : ℕ) (german : ℕ) (both : ℕ) (neither : ℕ)
  (h1 : french = 41)
  (h2 : german = 22)
  (h3 : both = 9)
  (h4 : neither = 40) :
  french + german - both + neither = 94 := by
  sorry

end NUMINAMATH_CALUDE_class_size_l2654_265424


namespace NUMINAMATH_CALUDE_least_sum_of_equal_multiples_l2654_265480

theorem least_sum_of_equal_multiples (x y z : ℕ+) (h : (4 : ℕ) * x.val = (5 : ℕ) * y.val ∧ (5 : ℕ) * y.val = (6 : ℕ) * z.val) :
  ∃ (a b c : ℕ+), (4 : ℕ) * a.val = (5 : ℕ) * b.val ∧ (5 : ℕ) * b.val = (6 : ℕ) * c.val ∧
    a.val + b.val + c.val ≤ x.val + y.val + z.val ∧ a.val + b.val + c.val = 37 :=
by
  sorry

end NUMINAMATH_CALUDE_least_sum_of_equal_multiples_l2654_265480


namespace NUMINAMATH_CALUDE_trigonometric_inequality_l2654_265420

theorem trigonometric_inequality (x : ℝ) : 
  (9.276 * Real.sin (2 * x) * Real.sin (3 * x) - Real.cos (2 * x) * Real.cos (3 * x) > Real.sin (10 * x)) ↔ 
  (∃ n : ℤ, ((-Real.pi / 10 + 2 * Real.pi * n / 5 < x ∧ x < -Real.pi / 30 + 2 * Real.pi * n) ∨ 
             (Real.pi / 10 + 2 * Real.pi * n / 5 < x ∧ x < 7 * Real.pi / 30 + 2 * Real.pi * n))) :=
by sorry

end NUMINAMATH_CALUDE_trigonometric_inequality_l2654_265420


namespace NUMINAMATH_CALUDE_passing_mark_is_40_l2654_265477

/-- Represents the exam results of a class -/
structure ExamResults where
  total_students : ℕ
  absent_percentage : ℚ
  failed_percentage : ℚ
  just_passed_percentage : ℚ
  remaining_average : ℚ
  class_average : ℚ
  fail_margin : ℕ

/-- Calculates the passing mark for the exam given the exam results -/
def calculate_passing_mark (results : ExamResults) : ℚ :=
  let absent := results.total_students * results.absent_percentage
  let failed := results.total_students * results.failed_percentage
  let just_passed := results.total_students * results.just_passed_percentage
  let remaining := results.total_students - (absent + failed + just_passed)
  let total_marks := results.class_average * results.total_students
  let remaining_marks := remaining * results.remaining_average
  (total_marks - remaining_marks) / (failed + just_passed) + results.fail_margin

/-- Theorem stating that given the exam results, the passing mark is 40 -/
theorem passing_mark_is_40 (results : ExamResults) 
  (h1 : results.total_students = 100)
  (h2 : results.absent_percentage = 1/5)
  (h3 : results.failed_percentage = 3/10)
  (h4 : results.just_passed_percentage = 1/10)
  (h5 : results.remaining_average = 65)
  (h6 : results.class_average = 36)
  (h7 : results.fail_margin = 20) :
  calculate_passing_mark results = 40 := by
  sorry

end NUMINAMATH_CALUDE_passing_mark_is_40_l2654_265477


namespace NUMINAMATH_CALUDE_rational_number_ordering_l2654_265448

theorem rational_number_ordering (a b c : ℚ) 
  (h1 : a - b > 0) (h2 : b - c > 0) : c < b ∧ b < a := by
  sorry

end NUMINAMATH_CALUDE_rational_number_ordering_l2654_265448


namespace NUMINAMATH_CALUDE_expression_evaluation_l2654_265423

theorem expression_evaluation : 
  202.2 * 89.8 - 20.22 * 186 + 2.022 * 3570 - 0.2022 * 16900 = 18198 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2654_265423


namespace NUMINAMATH_CALUDE_divisibility_property_not_true_for_p_2_l2654_265453

theorem divisibility_property (p a n : ℕ) : 
  Nat.Prime p → p ≠ 2 → a > 0 → n > 0 → (p^n ∣ a^p - 1) → (p^(n-1) ∣ a - 1) :=
by sorry

-- The statement is not true for p = 2
theorem not_true_for_p_2 : 
  ∃ (a n : ℕ), a > 0 ∧ n > 0 ∧ (2^n ∣ a^2 - 1) ∧ ¬(2^(n-1) ∣ a - 1) :=
by sorry

end NUMINAMATH_CALUDE_divisibility_property_not_true_for_p_2_l2654_265453


namespace NUMINAMATH_CALUDE_sam_watermelons_l2654_265451

/-- The number of watermelons Sam grew initially -/
def initial_watermelons : ℕ := 4

/-- The number of additional watermelons Sam grew -/
def additional_watermelons : ℕ := 3

/-- The total number of watermelons Sam has -/
def total_watermelons : ℕ := initial_watermelons + additional_watermelons

theorem sam_watermelons : total_watermelons = 7 := by
  sorry

end NUMINAMATH_CALUDE_sam_watermelons_l2654_265451


namespace NUMINAMATH_CALUDE_kevins_watermelons_l2654_265495

/-- The weight of the first watermelon in pounds -/
def first_watermelon : ℝ := 9.91

/-- The weight of the second watermelon in pounds -/
def second_watermelon : ℝ := 4.11

/-- The total weight of watermelons Kevin bought -/
def total_weight : ℝ := first_watermelon + second_watermelon

/-- Theorem stating that the total weight of watermelons Kevin bought is 14.02 pounds -/
theorem kevins_watermelons : total_weight = 14.02 := by sorry

end NUMINAMATH_CALUDE_kevins_watermelons_l2654_265495


namespace NUMINAMATH_CALUDE_custom_op_nested_result_l2654_265485

/-- Custom binary operation ⊗ -/
def custom_op (x y : ℝ) : ℝ := x^3 - 3*x*y

/-- Theorem stating the result of h ⊗ (h ⊗ h) -/
theorem custom_op_nested_result (h : ℝ) : custom_op h (custom_op h h) = h^3 * (10 - 3*h) := by
  sorry

end NUMINAMATH_CALUDE_custom_op_nested_result_l2654_265485


namespace NUMINAMATH_CALUDE_gcd_84_36_l2654_265497

theorem gcd_84_36 : Nat.gcd 84 36 = 12 := by
  sorry

end NUMINAMATH_CALUDE_gcd_84_36_l2654_265497


namespace NUMINAMATH_CALUDE_lena_tennis_win_probability_l2654_265454

theorem lena_tennis_win_probability :
  ∀ (p_lose : ℚ),
  p_lose = 3/7 →
  (∀ (p_win : ℚ), p_win + p_lose = 1 → p_win = 4/7) :=
by sorry

end NUMINAMATH_CALUDE_lena_tennis_win_probability_l2654_265454


namespace NUMINAMATH_CALUDE_geometric_sequence_fifth_term_l2654_265416

/-- A geometric sequence of positive integers with first term 3 and fourth term 243 has fifth term 243. -/
theorem geometric_sequence_fifth_term :
  ∀ (a : ℕ → ℕ),
  (∀ n : ℕ, a (n + 1) / a n = a 2 / a 1) →  -- Geometric sequence condition
  (∀ n : ℕ, a n > 0) →  -- Positive integer condition
  a 1 = 3 →  -- First term is 3
  a 4 = 243 →  -- Fourth term is 243
  a 5 = 243 :=  -- Fifth term is 243
by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_fifth_term_l2654_265416


namespace NUMINAMATH_CALUDE_tricycle_count_l2654_265411

theorem tricycle_count (num_bicycles : ℕ) (bicycle_wheels : ℕ) (tricycle_wheels : ℕ) (total_wheels : ℕ) :
  num_bicycles = 16 →
  bicycle_wheels = 2 →
  tricycle_wheels = 3 →
  total_wheels = 53 →
  ∃ num_tricycles : ℕ, num_bicycles * bicycle_wheels + num_tricycles * tricycle_wheels = total_wheels ∧ num_tricycles = 7 :=
by sorry

end NUMINAMATH_CALUDE_tricycle_count_l2654_265411


namespace NUMINAMATH_CALUDE_min_pencils_divisible_by_3_and_4_l2654_265457

theorem min_pencils_divisible_by_3_and_4 : 
  ∃ n : ℕ, n > 0 ∧ n % 3 = 0 ∧ n % 4 = 0 ∧ ∀ m : ℕ, m > 0 → m % 3 = 0 → m % 4 = 0 → n ≤ m :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_min_pencils_divisible_by_3_and_4_l2654_265457


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l2654_265452

theorem quadratic_equation_solution (x c d : ℝ) : 
  x^2 + 14*x = 92 → 
  (∃ c d : ℕ+, x = Real.sqrt c - d) →
  (∃ c d : ℕ+, x = Real.sqrt c - d ∧ c + d = 148) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l2654_265452


namespace NUMINAMATH_CALUDE_geometric_series_problem_l2654_265483

/-- Given two infinite geometric series with the specified conditions, prove that m = 8 -/
theorem geometric_series_problem (m : ℝ) : 
  let a₁ : ℝ := 18
  let b₁ : ℝ := 6
  let a₂ : ℝ := 18
  let b₂ : ℝ := 6 + m
  let r₁ := b₁ / a₁
  let r₂ := b₂ / a₂
  let S₁ := a₁ / (1 - r₁)
  let S₂ := a₂ / (1 - r₂)
  S₂ = 3 * S₁ → m = 8 := by
sorry


end NUMINAMATH_CALUDE_geometric_series_problem_l2654_265483


namespace NUMINAMATH_CALUDE_string_average_length_l2654_265450

theorem string_average_length :
  let string1 : ℚ := 4
  let string2 : ℚ := 5
  let string3 : ℚ := 7
  let num_strings : ℕ := 3
  (string1 + string2 + string3) / num_strings = 16 / 3 := by
  sorry

end NUMINAMATH_CALUDE_string_average_length_l2654_265450


namespace NUMINAMATH_CALUDE_equation_condition_l2654_265434

theorem equation_condition (a b c : ℤ) :
  a * (a - b) + b * (b - c) + c * (c - a) = 0 ↔ 
  (a = c ∧ b - 2 = c ∧ a = 0 ∧ b = 0 ∧ c = 0) := by
sorry

end NUMINAMATH_CALUDE_equation_condition_l2654_265434


namespace NUMINAMATH_CALUDE_middle_integer_of_three_consecutive_l2654_265432

/-- Given three consecutive integers whose sum is 360, the middle integer is 120. -/
theorem middle_integer_of_three_consecutive (n : ℤ) : 
  (n - 1) + n + (n + 1) = 360 → n = 120 := by
  sorry

end NUMINAMATH_CALUDE_middle_integer_of_three_consecutive_l2654_265432


namespace NUMINAMATH_CALUDE_factor_calculation_l2654_265456

theorem factor_calculation (x : ℝ) (h : x = 36) : 
  ∃ f : ℝ, ((x + 10) * f / 2) - 2 = 88 / 2 ∧ f = 2 := by
  sorry

end NUMINAMATH_CALUDE_factor_calculation_l2654_265456


namespace NUMINAMATH_CALUDE_dart_probability_l2654_265443

/-- The probability of a dart landing within a circular target area inscribed in a regular hexagonal dartboard -/
theorem dart_probability (s : ℝ) (h : s > 0) : 
  let hexagon_area := 3 * Real.sqrt 3 / 2 * s^2
  let circle_area := Real.pi * s^2
  circle_area / hexagon_area = 2 * Real.pi / (3 * Real.sqrt 3) := by
  sorry

end NUMINAMATH_CALUDE_dart_probability_l2654_265443


namespace NUMINAMATH_CALUDE_sqrt_twelve_minus_abs_one_minus_sqrt_three_plus_seven_plus_pi_pow_zero_l2654_265403

theorem sqrt_twelve_minus_abs_one_minus_sqrt_three_plus_seven_plus_pi_pow_zero :
  Real.sqrt 12 - |1 - Real.sqrt 3| + (7 + Real.pi)^0 = Real.sqrt 3 + 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_twelve_minus_abs_one_minus_sqrt_three_plus_seven_plus_pi_pow_zero_l2654_265403


namespace NUMINAMATH_CALUDE_train_length_l2654_265441

/-- The length of a train given its speed and time to cross a pole -/
theorem train_length (speed : ℝ) (time : ℝ) : 
  speed = 300 → time = 15 → ∃ length : ℝ, 
  (abs (length - 1249.95) < 0.01) ∧ 
  (length = speed * 1000 / 3600 * time) := by
  sorry

end NUMINAMATH_CALUDE_train_length_l2654_265441


namespace NUMINAMATH_CALUDE_flower_arrangement_theorem_l2654_265481

/-- The number of ways to arrange flowers of three different hues -/
def flower_arrangements (X : ℕ+) : ℕ :=
  30

/-- Theorem stating that the number of valid flower arrangements is always 30 -/
theorem flower_arrangement_theorem (X : ℕ+) :
  flower_arrangements X = 30 :=
by
  sorry

end NUMINAMATH_CALUDE_flower_arrangement_theorem_l2654_265481


namespace NUMINAMATH_CALUDE_quadratic_root_values_l2654_265470

theorem quadratic_root_values (b c : ℝ) : 
  (Complex.I * Real.sqrt 2 + 1) ^ 2 + b * (Complex.I * Real.sqrt 2 + 1) + c = 0 → 
  b = -2 ∧ c = 3 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_values_l2654_265470


namespace NUMINAMATH_CALUDE_candy_bar_earnings_difference_l2654_265413

/-- The problem of calculating the difference in earnings between Tina and Marvin from selling candy bars. -/
theorem candy_bar_earnings_difference :
  let candy_bar_price : ℕ := 2
  let marvins_sales : ℕ := 35
  let tinas_sales : ℕ := 3 * marvins_sales
  let marvins_earnings : ℕ := candy_bar_price * marvins_sales
  let tinas_earnings : ℕ := candy_bar_price * tinas_sales
  tinas_earnings - marvins_earnings = 140 := by
sorry

end NUMINAMATH_CALUDE_candy_bar_earnings_difference_l2654_265413


namespace NUMINAMATH_CALUDE_managers_salary_correct_managers_salary_l2654_265439

theorem managers_salary (num_employees : ℕ) (avg_salary : ℝ) (avg_increase : ℝ) : ℝ :=
  let total_salary := num_employees * avg_salary
  let new_avg_salary := avg_salary + avg_increase
  let new_total_salary := (num_employees + 1) * new_avg_salary
  new_total_salary - total_salary

theorem correct_managers_salary :
  managers_salary 50 2000 250 = 14750 := by sorry

end NUMINAMATH_CALUDE_managers_salary_correct_managers_salary_l2654_265439


namespace NUMINAMATH_CALUDE_garrison_reinforcement_reinforcement_size_l2654_265474

theorem garrison_reinforcement (initial_garrison : ℕ) (initial_days : ℕ) 
  (days_before_reinforcement : ℕ) (days_after_reinforcement : ℕ) : ℕ :=
  let total_provisions := initial_garrison * initial_days
  let remaining_provisions := total_provisions - (initial_garrison * days_before_reinforcement)
  let reinforcement := (remaining_provisions / days_after_reinforcement) - initial_garrison
  reinforcement

theorem reinforcement_size :
  garrison_reinforcement 1000 60 15 20 = 1250 := by sorry

end NUMINAMATH_CALUDE_garrison_reinforcement_reinforcement_size_l2654_265474


namespace NUMINAMATH_CALUDE_farm_corn_cobs_l2654_265458

/-- Calculates the total number of corn cobs grown on a farm with two fields -/
def total_corn_cobs (field1_rows : ℕ) (field2_rows : ℕ) (cobs_per_row : ℕ) : ℕ :=
  (field1_rows * cobs_per_row) + (field2_rows * cobs_per_row)

/-- Theorem stating that the total number of corn cobs on the farm is 116 -/
theorem farm_corn_cobs : total_corn_cobs 13 16 4 = 116 := by
  sorry

end NUMINAMATH_CALUDE_farm_corn_cobs_l2654_265458


namespace NUMINAMATH_CALUDE_xy_value_l2654_265428

theorem xy_value (x y : ℝ) (sum_eq : x + y = 10) (sum_cubes_eq : x^3 + y^3 = 370) : x * y = 21 := by
  sorry

end NUMINAMATH_CALUDE_xy_value_l2654_265428


namespace NUMINAMATH_CALUDE_captains_age_and_crew_size_l2654_265449

theorem captains_age_and_crew_size (l k : ℕ) : 
  l * (l - 1) = k * (l - 2) + 15 → 
  ((l = 1 ∧ k = 15) ∨ (l = 15 ∧ k = 15)) := by
sorry

end NUMINAMATH_CALUDE_captains_age_and_crew_size_l2654_265449


namespace NUMINAMATH_CALUDE_thirty_six_is_triangular_and_square_l2654_265462

/-- Definition of triangular numbers -/
def is_triangular (n : ℕ) : Prop :=
  ∃ k : ℕ, n = k * (k + 1) / 2

/-- Definition of square numbers -/
def is_square (n : ℕ) : Prop :=
  ∃ k : ℕ, n = k^2

/-- Theorem: 36 is both a triangular number and a square number -/
theorem thirty_six_is_triangular_and_square :
  is_triangular 36 ∧ is_square 36 :=
sorry

end NUMINAMATH_CALUDE_thirty_six_is_triangular_and_square_l2654_265462
