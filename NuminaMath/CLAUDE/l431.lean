import Mathlib

namespace NUMINAMATH_CALUDE_two_digit_product_1365_l431_43177

/-- Represents a two-digit number --/
structure TwoDigitNumber where
  tens : Nat
  ones : Nat
  is_valid : 1 ≤ tens ∧ tens ≤ 9 ∧ 0 ≤ ones ∧ ones ≤ 9

/-- Converts a TwoDigitNumber to a natural number --/
def TwoDigitNumber.toNat (n : TwoDigitNumber) : Nat :=
  10 * n.tens + n.ones

theorem two_digit_product_1365 :
  ∀ (ab cd : TwoDigitNumber),
    ab.toNat * cd.toNat = 1365 →
    ab.tens ≠ ab.ones →
    cd.tens ≠ cd.ones →
    ab.tens ≠ cd.tens →
    ab.tens ≠ cd.ones →
    ab.ones ≠ cd.tens →
    ab.ones ≠ cd.ones →
    ((ab.tens = 2 ∧ ab.ones = 1) ∧ (cd.tens = 6 ∧ cd.ones = 5)) ∨
    ((ab.tens = 6 ∧ ab.ones = 5) ∧ (cd.tens = 2 ∧ cd.ones = 1)) :=
by sorry

end NUMINAMATH_CALUDE_two_digit_product_1365_l431_43177


namespace NUMINAMATH_CALUDE_range_of_x_l431_43148

theorem range_of_x (x : ℝ) : 
  (∃ m : ℝ, m ∈ Set.Icc 1 3 ∧ x + 3 * m + 5 > 0) → x > -14 := by
  sorry

end NUMINAMATH_CALUDE_range_of_x_l431_43148


namespace NUMINAMATH_CALUDE_brothers_age_difference_l431_43100

theorem brothers_age_difference (michael_age younger_brother_age older_brother_age : ℕ) : 
  younger_brother_age = 5 →
  older_brother_age = 3 * younger_brother_age →
  michael_age + older_brother_age + younger_brother_age = 28 →
  older_brother_age - 2 * (michael_age - 1) = 1 := by
  sorry

end NUMINAMATH_CALUDE_brothers_age_difference_l431_43100


namespace NUMINAMATH_CALUDE_triangle_theorem_l431_43108

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    if a*sin(A) = 4b*sin(B) and a*c = √5*(a^2 - b^2 - c^2),
    then cos(A) = -√5/5 and sin(2B - A) = -2√5/5 -/
theorem triangle_theorem (a b c A B C : ℝ) 
  (h1 : a * Real.sin A = 4 * b * Real.sin B)
  (h2 : a * c = Real.sqrt 5 * (a^2 - b^2 - c^2)) :
  Real.cos A = -(Real.sqrt 5 / 5) ∧ 
  Real.sin (2 * B - A) = -(2 * Real.sqrt 5 / 5) := by
  sorry

end NUMINAMATH_CALUDE_triangle_theorem_l431_43108


namespace NUMINAMATH_CALUDE_prove_late_time_l431_43101

def late_time_problem (charlize_late : ℕ) (classmates_extra : ℕ) (num_classmates : ℕ) : Prop :=
  let classmate_late := charlize_late + classmates_extra
  let total_classmates_late := num_classmates * classmate_late
  let total_late := total_classmates_late + charlize_late
  total_late = 140

theorem prove_late_time : late_time_problem 20 10 4 := by
  sorry

end NUMINAMATH_CALUDE_prove_late_time_l431_43101


namespace NUMINAMATH_CALUDE_kylie_piggy_bank_coins_kylie_piggy_bank_coins_value_l431_43198

/-- The number of coins Kylie got from her piggy bank -/
def coins_from_piggy_bank : ℕ := sorry

/-- The number of coins Kylie got from her brother -/
def coins_from_brother : ℕ := 13

/-- The number of coins Kylie got from her father -/
def coins_from_father : ℕ := 8

/-- The number of coins Kylie gave to Laura -/
def coins_given_to_laura : ℕ := 21

/-- The number of coins Kylie had left -/
def coins_left : ℕ := 15

theorem kylie_piggy_bank_coins :
  coins_from_piggy_bank + coins_from_brother + coins_from_father - coins_given_to_laura = coins_left :=
by sorry

theorem kylie_piggy_bank_coins_value : coins_from_piggy_bank = 15 :=
by sorry

end NUMINAMATH_CALUDE_kylie_piggy_bank_coins_kylie_piggy_bank_coins_value_l431_43198


namespace NUMINAMATH_CALUDE_solve_equation_y_l431_43153

theorem solve_equation_y (y : ℝ) (hy : y ≠ 0) :
  (7 * y)^4 = (14 * y)^3 ↔ y = 8 / 7 := by
sorry

end NUMINAMATH_CALUDE_solve_equation_y_l431_43153


namespace NUMINAMATH_CALUDE_cos_45_cos_15_plus_sin_45_sin_15_l431_43125

theorem cos_45_cos_15_plus_sin_45_sin_15 :
  Real.cos (45 * π / 180) * Real.cos (15 * π / 180) + Real.sin (45 * π / 180) * Real.sin (15 * π / 180) = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_45_cos_15_plus_sin_45_sin_15_l431_43125


namespace NUMINAMATH_CALUDE_fraction_decomposition_l431_43103

theorem fraction_decomposition (x A B C : ℚ) : 
  (6*x^2 - 13*x + 6) / (2*x^3 + 3*x^2 - 11*x - 6) = 
  A / (x + 1) + B / (2*x - 3) + C / (x - 2) →
  A = 1 ∧ B = 4 ∧ C = 1 := by
sorry

end NUMINAMATH_CALUDE_fraction_decomposition_l431_43103


namespace NUMINAMATH_CALUDE_A_subseteq_C_l431_43138

-- Define the universe
def U : Set ℝ := {x | -1 ≤ x ∧ x ≤ 3}

-- Define set A
def A : Set ℝ := {x | -1 ≤ x ∧ x < 3}

-- Define set B
def B : Set ℝ := {x | x^2 - 2*x - 3 = 0}

-- Define set C
def C : Set ℝ := {x | -1 < x ∧ x < 3}

-- Theorem statement
theorem A_subseteq_C : C ⊆ A := by sorry

end NUMINAMATH_CALUDE_A_subseteq_C_l431_43138


namespace NUMINAMATH_CALUDE_product_factors_l431_43195

/-- Given three different natural numbers, each with exactly three factors,
    the product a³b⁴c⁵ has 693 factors. -/
theorem product_factors (a b c : ℕ) (ha : a.factors.length = 3)
    (hb : b.factors.length = 3) (hc : c.factors.length = 3)
    (hab : a ≠ b) (hbc : b ≠ c) (hac : a ≠ c) :
    (a^3 * b^4 * c^5).factors.length = 693 := by
  sorry

end NUMINAMATH_CALUDE_product_factors_l431_43195


namespace NUMINAMATH_CALUDE_fence_bricks_l431_43167

/-- Calculates the number of bricks needed for a rectangular fence -/
def bricks_needed (length width height depth : ℕ) : ℕ :=
  4 * length * width * depth

theorem fence_bricks :
  bricks_needed 20 5 2 1 = 800 := by
  sorry

end NUMINAMATH_CALUDE_fence_bricks_l431_43167


namespace NUMINAMATH_CALUDE_odometer_reading_l431_43191

theorem odometer_reading (initial_reading lunch_reading total_distance : ℝ) :
  lunch_reading - initial_reading = 372.0 →
  total_distance = 584.3 →
  initial_reading = 212.3 :=
by sorry

end NUMINAMATH_CALUDE_odometer_reading_l431_43191


namespace NUMINAMATH_CALUDE_log_simplification_l431_43144

theorem log_simplification : (2 * (Real.log 3 / Real.log 4) + Real.log 3 / Real.log 8) * 
  (Real.log 2 / Real.log 3 + Real.log 2 / Real.log 9) = 2 := by
  sorry

end NUMINAMATH_CALUDE_log_simplification_l431_43144


namespace NUMINAMATH_CALUDE_triangle_properties_l431_43179

-- Define the triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- Define the theorem
theorem triangle_properties (t : Triangle) 
  (h1 : 2 * Real.cos t.C * (t.a * Real.cos t.B + t.b * Real.cos t.A) = t.c)
  (h2 : t.c = Real.sqrt 7)
  (h3 : (1/2) * t.a * t.b * Real.sin t.C = (3 * Real.sqrt 3) / 2) :
  t.C = π/3 ∧ t.a + t.b + t.c = 5 + Real.sqrt 7 := by
  sorry


end NUMINAMATH_CALUDE_triangle_properties_l431_43179


namespace NUMINAMATH_CALUDE_similarity_of_reasoning_types_l431_43126

-- Define the types of reasoning
inductive ReasoningType
| Inductive
| Analogical

-- Define the characteristics of reasoning types
def characteristicsOf (r : ReasoningType) : String :=
  match r with
  | ReasoningType.Inductive => "deriving general principles from specific facts"
  | ReasoningType.Analogical => "going from specific to specific, based on shared attributes"

-- Define the concept of necessary correctness
def isNecessarilyCorrect (r : ReasoningType) : Prop := False

-- Theorem statement
theorem similarity_of_reasoning_types :
  ∀ (r : ReasoningType), ¬(isNecessarilyCorrect r) :=
by sorry

end NUMINAMATH_CALUDE_similarity_of_reasoning_types_l431_43126


namespace NUMINAMATH_CALUDE_monotonic_f_implies_a_eq_one_l431_43194

/-- Piecewise function f(x) as defined in the problem -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≤ -1 then -x^2 + 2*a else a*x + 4

/-- Theorem stating that if f is monotonic on ℝ, then a = 1 -/
theorem monotonic_f_implies_a_eq_one (a : ℝ) :
  Monotone (f a) → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_monotonic_f_implies_a_eq_one_l431_43194


namespace NUMINAMATH_CALUDE_sum_of_solutions_quadratic_sum_of_solutions_specific_l431_43115

theorem sum_of_solutions_quadratic (a b c : ℝ) (h : a ≠ 0) :
  let equation := fun x => a * x^2 + b * x + c
  let sum_of_solutions := -b / a
  equation 0 = 0 → sum_of_solutions = -b / a :=
by
  sorry

-- The specific problem
theorem sum_of_solutions_specific :
  let equation := fun x : ℝ => -48 * x^2 + 100 * x + 200
  let sum_of_solutions := 25 / 12
  (∀ x, equation x = 0 → x = sum_of_solutions / 2 ∨ x = sum_of_solutions / 2) ∧
  sum_of_solutions = 25 / 12 :=
by
  sorry

end NUMINAMATH_CALUDE_sum_of_solutions_quadratic_sum_of_solutions_specific_l431_43115


namespace NUMINAMATH_CALUDE_first_number_value_l431_43187

theorem first_number_value (x y : ℝ) : 
  x - y = 88 → y = 0.2 * x → x = 110 := by sorry

end NUMINAMATH_CALUDE_first_number_value_l431_43187


namespace NUMINAMATH_CALUDE_train_crossing_time_l431_43140

def train_length : ℝ := 450
def train_speed_kmh : ℝ := 54

theorem train_crossing_time : 
  ∀ (platform_length : ℝ) (train_speed_ms : ℝ),
    platform_length = train_length →
    train_speed_ms = train_speed_kmh * (1000 / 3600) →
    (2 * train_length) / train_speed_ms = 60 := by
  sorry

end NUMINAMATH_CALUDE_train_crossing_time_l431_43140


namespace NUMINAMATH_CALUDE_possible_sum_BC_ge_90_l431_43155

/-- Represents an acute triangle with angles A, B, and C --/
structure AcuteTriangle where
  A : Real
  B : Real
  C : Real
  acute : A < 90 ∧ B < 90 ∧ C < 90
  sum_180 : A + B + C = 180
  ordered : A > B ∧ B > C

/-- 
Theorem: In an acute triangle with angles A > B > C, 
it's possible for the sum of B and C to be greater than or equal to 90°
--/
theorem possible_sum_BC_ge_90 (t : AcuteTriangle) : 
  ∃ (x y z : Real), x > y ∧ y > z ∧ x < 90 ∧ y < 90 ∧ z < 90 ∧ x + y + z = 180 ∧ y + z ≥ 90 := by
  sorry

end NUMINAMATH_CALUDE_possible_sum_BC_ge_90_l431_43155


namespace NUMINAMATH_CALUDE_percentage_problem_l431_43172

/-- Given a number N and a percentage P, this theorem proves that
    if P% of N is 24 less than 50% of N, and N = 160, then P = 35. -/
theorem percentage_problem (N : ℝ) (P : ℝ) : 
  N = 160 → 
  (P / 100) * N = (50 / 100) * N - 24 → 
  P = 35 := by
sorry

end NUMINAMATH_CALUDE_percentage_problem_l431_43172


namespace NUMINAMATH_CALUDE_find_B_value_l431_43174

theorem find_B_value (A B : ℕ) : 
  (100 ≤ 6 * 100 + A * 10 + 5) ∧ (6 * 100 + A * 10 + 5 < 1000) ∧ 
  (100 ≤ 1 * 100 + 0 * 10 + B) ∧ (1 * 100 + 0 * 10 + B < 1000) ∧
  (6 * 100 + A * 10 + 5 + 1 * 100 + 0 * 10 + B = 748) →
  B = 3 := by sorry

end NUMINAMATH_CALUDE_find_B_value_l431_43174


namespace NUMINAMATH_CALUDE_inequality_solution_l431_43107

theorem inequality_solution (x : ℝ) : (x - 5) / ((x - 2) * (x^2 - 1)) < 0 ↔ x < -1 ∨ (1 < x ∧ x < 5) := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_l431_43107


namespace NUMINAMATH_CALUDE_isosceles_triangle_base_length_l431_43156

/-- An isosceles triangle with congruent sides of 8 cm and perimeter of 25 cm has a base length of 9 cm. -/
theorem isosceles_triangle_base_length :
  ∀ (base congruent_side : ℝ),
  congruent_side = 8 →
  base + 2 * congruent_side = 25 →
  base = 9 := by
sorry

end NUMINAMATH_CALUDE_isosceles_triangle_base_length_l431_43156


namespace NUMINAMATH_CALUDE_shaded_areas_comparison_l431_43151

/-- Represents a square with its division and shading pattern -/
structure Square where
  total_divisions : ℕ
  shaded_divisions : ℕ

/-- The three squares in the problem -/
def square_I : Square := { total_divisions := 4, shaded_divisions := 2 }
def square_II : Square := { total_divisions := 9, shaded_divisions := 3 }
def square_III : Square := { total_divisions := 12, shaded_divisions := 4 }

/-- Calculates the shaded area fraction of a square -/
def shaded_area_fraction (s : Square) : ℚ :=
  s.shaded_divisions / s.total_divisions

/-- Theorem stating the relationship between the shaded areas -/
theorem shaded_areas_comparison :
  shaded_area_fraction square_II = shaded_area_fraction square_III ∧
  shaded_area_fraction square_I ≠ shaded_area_fraction square_II :=
by sorry

end NUMINAMATH_CALUDE_shaded_areas_comparison_l431_43151


namespace NUMINAMATH_CALUDE_binomial_expansion_sum_zero_l431_43117

theorem binomial_expansion_sum_zero (n : ℕ) (b : ℕ) (h1 : n ≥ 2) (h2 : b > 0) :
  let a := 3 * b
  (n.choose 1 * (a - 2 * b) ^ (n - 1) + n.choose 2 * (a - 2 * b) ^ (n - 2) = 0) ↔ n = 3 :=
by sorry

end NUMINAMATH_CALUDE_binomial_expansion_sum_zero_l431_43117


namespace NUMINAMATH_CALUDE_triangle_area_sines_l431_43113

theorem triangle_area_sines (a b c : ℝ) (h_a : a = 5) (h_b : b = 4 * Real.sqrt 2) (h_c : c = 7) :
  let R := (a * b * c) / (4 * Real.sqrt (((a + b + c)/2) * (((a + b + c)/2) - a) * (((a + b + c)/2) - b) * (((a + b + c)/2) - c)));
  let sin_A := a / (2 * R);
  let sin_B := b / (2 * R);
  let sin_C := c / (2 * R);
  let s := (sin_A + sin_B + sin_C) / 2;
  Real.sqrt (s * (s - sin_A) * (s - sin_B) * (s - sin_C)) = 7 / 25 := by
sorry

end NUMINAMATH_CALUDE_triangle_area_sines_l431_43113


namespace NUMINAMATH_CALUDE_football_team_right_handed_players_l431_43192

theorem football_team_right_handed_players
  (total_players : ℕ)
  (throwers : ℕ)
  (multiple_position : ℕ)
  (left_to_right_ratio : ℚ)
  (h1 : total_players = 120)
  (h2 : throwers = 60)
  (h3 : multiple_position = 20)
  (h4 : left_to_right_ratio = 2 / 3)
  (h5 : throwers + multiple_position ≤ total_players) :
  throwers + multiple_position + ((total_players - (throwers + multiple_position)) / (1 + left_to_right_ratio⁻¹)) = 104 :=
by sorry

end NUMINAMATH_CALUDE_football_team_right_handed_players_l431_43192


namespace NUMINAMATH_CALUDE_eggs_in_club_house_l431_43128

theorem eggs_in_club_house (total eggs_in_park eggs_in_town_hall eggs_in_club_house : ℕ) :
  total = eggs_in_club_house + eggs_in_park + eggs_in_town_hall →
  eggs_in_park = 25 →
  eggs_in_town_hall = 15 →
  total = 80 →
  eggs_in_club_house = 40 := by
sorry

end NUMINAMATH_CALUDE_eggs_in_club_house_l431_43128


namespace NUMINAMATH_CALUDE_airplane_passengers_l431_43193

theorem airplane_passengers (total_passengers men : ℕ) 
  (h1 : total_passengers = 80)
  (h2 : men = 30)
  (h3 : ∃ women : ℕ, women = men) :
  ∃ children : ℕ, children = 20 ∧ total_passengers = men + men + children :=
by sorry

end NUMINAMATH_CALUDE_airplane_passengers_l431_43193


namespace NUMINAMATH_CALUDE_angle_sum_inequality_l431_43188

theorem angle_sum_inequality (θ₁ θ₂ θ₃ θ₄ : Real)
  (h₁ : 0 < θ₁ ∧ θ₁ < π/2)
  (h₂ : 0 < θ₂ ∧ θ₂ < π/2)
  (h₃ : 0 < θ₃ ∧ θ₃ < π/2)
  (h₄ : 0 < θ₄ ∧ θ₄ < π/2)
  (h_sum : θ₁ + θ₂ + θ₃ + θ₄ = π) :
  (Real.sqrt 2 * Real.sin θ₁ - 1) / Real.cos θ₁ +
  (Real.sqrt 2 * Real.sin θ₂ - 1) / Real.cos θ₂ +
  (Real.sqrt 2 * Real.sin θ₃ - 1) / Real.cos θ₃ +
  (Real.sqrt 2 * Real.sin θ₄ - 1) / Real.cos θ₄ ≥ 0 :=
by sorry

end NUMINAMATH_CALUDE_angle_sum_inequality_l431_43188


namespace NUMINAMATH_CALUDE_circle_y_axis_intersection_sum_l431_43168

theorem circle_y_axis_intersection_sum (h k r : ℝ) : 
  h = -3 → k = 5 → r = 8 → 
  (k + (r^2 - h^2).sqrt) + (k - (r^2 - h^2).sqrt) = 10 := by sorry

end NUMINAMATH_CALUDE_circle_y_axis_intersection_sum_l431_43168


namespace NUMINAMATH_CALUDE_equation_solution_l431_43109

theorem equation_solution : ∃! x : ℝ, (x^2 + 2*x + 3) / (x^2 - 1) = x + 3 :=
by
  -- The unique solution is x = 1
  use 1
  constructor
  -- Prove that x = 1 satisfies the equation
  · sorry
  -- Prove that any other solution must equal 1
  · sorry

end NUMINAMATH_CALUDE_equation_solution_l431_43109


namespace NUMINAMATH_CALUDE_overall_profit_percentage_l431_43165

def book_a_cost : ℚ := 50
def book_b_cost : ℚ := 75
def book_c_cost : ℚ := 100
def book_a_sell : ℚ := 60
def book_b_sell : ℚ := 90
def book_c_sell : ℚ := 120

def total_investment_cost : ℚ := book_a_cost + book_b_cost + book_c_cost
def total_revenue : ℚ := book_a_sell + book_b_sell + book_c_sell
def total_profit : ℚ := total_revenue - total_investment_cost
def profit_percentage : ℚ := (total_profit / total_investment_cost) * 100

theorem overall_profit_percentage :
  profit_percentage = 20 := by sorry

end NUMINAMATH_CALUDE_overall_profit_percentage_l431_43165


namespace NUMINAMATH_CALUDE_pyramid_surface_area_change_l431_43102

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a rectangular parallelepiped -/
structure Parallelepiped where
  a : ℝ  -- length
  b : ℝ  -- width
  c : ℝ  -- height

/-- Represents a quadrilateral pyramid -/
structure QuadPyramid where
  base : Point3D  -- center of the base
  apex : Point3D

/-- Calculates the surface area of a quadrilateral pyramid -/
def surfaceArea (p : Parallelepiped) (q : QuadPyramid) : ℝ := sorry

/-- Position of the apex on L₂ -/
inductive ApexPosition
  | Midpoint
  | Between
  | Vertex

/-- Theorem about the surface area of the pyramid -/
theorem pyramid_surface_area_change
  (p : Parallelepiped)
  (q : QuadPyramid)
  (h₁ : q.base.z = 0)  -- base is on the xy-plane
  (h₂ : q.apex.z = p.c)  -- apex is on the top face
  :
  (∀ (pos₁ pos₂ : ApexPosition),
    pos₁ = ApexPosition.Midpoint ∧ pos₂ = ApexPosition.Between →
      surfaceArea p q < surfaceArea p { q with apex := sorry }) ∧
  (∀ (pos₁ pos₂ : ApexPosition),
    pos₁ = ApexPosition.Between ∧ pos₂ = ApexPosition.Vertex →
      surfaceArea p q < surfaceArea p { q with apex := sorry }) ∧
  (∀ (pos : ApexPosition),
    pos = ApexPosition.Vertex →
      ∀ (q' : QuadPyramid), surfaceArea p q ≤ surfaceArea p q') :=
sorry

end NUMINAMATH_CALUDE_pyramid_surface_area_change_l431_43102


namespace NUMINAMATH_CALUDE_jaewoong_ran_most_l431_43154

-- Define the athletes and their distances
def jaewoong_distance : ℕ := 20  -- in kilometers
def seongmin_distance : ℕ := 2600  -- in meters
def eunseong_distance : ℕ := 5000  -- in meters

-- Define the conversion factor from kilometers to meters
def km_to_m : ℕ := 1000

-- Theorem to prove Jaewoong ran the most
theorem jaewoong_ran_most :
  (jaewoong_distance * km_to_m > seongmin_distance) ∧
  (jaewoong_distance * km_to_m > eunseong_distance) :=
by
  sorry

#check jaewoong_ran_most

end NUMINAMATH_CALUDE_jaewoong_ran_most_l431_43154


namespace NUMINAMATH_CALUDE_closest_point_l431_43133

def v (t : ℝ) : Fin 3 → ℝ := fun i =>
  match i with
  | 0 => 3 + 5*t
  | 1 => -2 + 4*t
  | 2 => 1 + 2*t

def a : Fin 3 → ℝ := fun i =>
  match i with
  | 0 => -1
  | 1 => 1
  | 2 => -3

def direction : Fin 3 → ℝ := fun i =>
  match i with
  | 0 => 5
  | 1 => 4
  | 2 => 2

theorem closest_point (t : ℝ) :
  (∀ s : ℝ, ‖v t - a‖ ≤ ‖v s - a‖) ↔ t = -16/45 := by sorry

end NUMINAMATH_CALUDE_closest_point_l431_43133


namespace NUMINAMATH_CALUDE_partition_modular_sum_l431_43129

theorem partition_modular_sum (p : ℕ) (h_prime : Nat.Prime p) (h_p_ge_5 : p ≥ 5) :
  ∀ (A B C : Set ℕ), 
    (A ∪ B ∪ C = Finset.range (p - 1)) →
    (A ∩ B = ∅) → (B ∩ C = ∅) → (A ∩ C = ∅) →
    ∃ (x y z : ℕ), x ∈ A ∧ y ∈ B ∧ z ∈ C ∧ (x + y) % p = z % p :=
by sorry

end NUMINAMATH_CALUDE_partition_modular_sum_l431_43129


namespace NUMINAMATH_CALUDE_star_emilio_sum_difference_l431_43180

/-- The sum of numbers from 1 to 50 -/
def starSum : ℕ := (List.range 50).map (· + 1) |>.sum

/-- The sum of numbers from 1 to 50 with '3' replaced by '2' -/
def emilioSum : ℕ := (List.range 50).map (· + 1) |>.map (replaceThreeWithTwo) |>.sum
  where
    replaceThreeWithTwo (n : ℕ) : ℕ :=
      let tens := n / 10
      let ones := n % 10
      if tens = 3 then 20 + ones
      else if ones = 3 then 10 * tens + 2
      else n

/-- The difference between Star's sum and Emilio's sum is 105 -/
theorem star_emilio_sum_difference : starSum - emilioSum = 105 := by
  sorry

end NUMINAMATH_CALUDE_star_emilio_sum_difference_l431_43180


namespace NUMINAMATH_CALUDE_certain_number_value_l431_43111

theorem certain_number_value (t b c : ℝ) :
  (t + b + c + 14 + 15) / 5 = 12 ∧ (t + b + c + 29) / 4 = 15 → 14 = 14 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_value_l431_43111


namespace NUMINAMATH_CALUDE_arithmetic_seq_product_l431_43197

/-- An increasing arithmetic sequence of integers -/
def is_increasing_arithmetic_seq (b : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, d > 0 ∧ ∀ n : ℕ, b (n + 1) = b n + d

theorem arithmetic_seq_product (b : ℕ → ℤ) 
  (h_seq : is_increasing_arithmetic_seq b)
  (h_prod : b 4 * b 5 = 15) :
  b 3 * b 6 = 7 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_seq_product_l431_43197


namespace NUMINAMATH_CALUDE_sum_of_60_digits_eq_180_l431_43146

/-- The sum of the first 60 digits after the decimal point in the decimal expansion of 1/1234 -/
def sum_of_60_digits : ℕ :=
  -- Define the sum here
  180

/-- Theorem stating that the sum of the first 60 digits after the decimal point
    in the decimal expansion of 1/1234 is equal to 180 -/
theorem sum_of_60_digits_eq_180 :
  sum_of_60_digits = 180 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_60_digits_eq_180_l431_43146


namespace NUMINAMATH_CALUDE_k_range_proof_l431_43104

-- Define the propositions p and q
def p (x k : ℝ) : Prop := x ≥ k
def q (x : ℝ) : Prop := (2 - x) / (x + 1) < 0

-- Define the range of k
def k_range (k : ℝ) : Prop := k > 2

-- State the theorem
theorem k_range_proof :
  (∀ k, (∀ x, p x k ↔ q x) → k_range k) ∧
  (∀ k, k_range k → (∀ x, p x k ↔ q x)) :=
sorry

end NUMINAMATH_CALUDE_k_range_proof_l431_43104


namespace NUMINAMATH_CALUDE_prime_iff_sum_four_integers_l431_43182

theorem prime_iff_sum_four_integers (n : ℕ) (h : n ≥ 5) :
  Nat.Prime n ↔ ∀ (a b c d : ℕ), a > 0 → b > 0 → c > 0 → d > 0 → n = a + b + c + d → a * b ≠ c * d := by
  sorry

end NUMINAMATH_CALUDE_prime_iff_sum_four_integers_l431_43182


namespace NUMINAMATH_CALUDE_intersection_equals_open_interval_l431_43119

-- Define the sets M and N
def M : Set ℝ := {x : ℝ | |x - 1| < 2}
def N : Set ℝ := {x : ℝ | x < 2}

-- Define the open interval (-1, 2)
def open_interval : Set ℝ := {x : ℝ | -1 < x ∧ x < 2}

-- Theorem statement
theorem intersection_equals_open_interval : M ∩ N = open_interval := by
  sorry

end NUMINAMATH_CALUDE_intersection_equals_open_interval_l431_43119


namespace NUMINAMATH_CALUDE_buttons_per_shirt_proof_l431_43161

/-- The number of shirts Sally sews on Monday -/
def monday_shirts : ℕ := 4

/-- The number of shirts Sally sews on Tuesday -/
def tuesday_shirts : ℕ := 3

/-- The number of shirts Sally sews on Wednesday -/
def wednesday_shirts : ℕ := 2

/-- The total number of buttons Sally needs for all shirts -/
def total_buttons : ℕ := 45

/-- The number of buttons per shirt -/
def buttons_per_shirt : ℕ := 5

theorem buttons_per_shirt_proof :
  (monday_shirts + tuesday_shirts + wednesday_shirts) * buttons_per_shirt = total_buttons :=
by sorry

end NUMINAMATH_CALUDE_buttons_per_shirt_proof_l431_43161


namespace NUMINAMATH_CALUDE_unique_function_satisfying_equation_l431_43159

/-- A function from non-negative reals to non-negative reals. -/
def NonNegativeRealFunction := {f : ℝ → ℝ // ∀ x, 0 ≤ x → 0 ≤ f x}

/-- The functional equation f(f(x)) + f(x) = 6x for all x ≥ 0. -/
def FunctionalEquation (f : NonNegativeRealFunction) : Prop :=
  ∀ x : ℝ, 0 ≤ x → f.val (f.val x) + f.val x = 6 * x

theorem unique_function_satisfying_equation :
  ∀ f : NonNegativeRealFunction, FunctionalEquation f → 
    ∀ x : ℝ, 0 ≤ x → f.val x = 2 * x :=
sorry

end NUMINAMATH_CALUDE_unique_function_satisfying_equation_l431_43159


namespace NUMINAMATH_CALUDE_base_8_to_10_98765_l431_43184

-- Define the base-8 number as a list of digits
def base_8_number : List Nat := [9, 8, 7, 6, 5]

-- Define the function to convert a base-8 number to base-10
def base_8_to_10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (8 ^ (digits.length - 1 - i))) 0

-- Theorem statement
theorem base_8_to_10_98765 :
  base_8_to_10 base_8_number = 41461 := by
  sorry

end NUMINAMATH_CALUDE_base_8_to_10_98765_l431_43184


namespace NUMINAMATH_CALUDE_solution_set_equivalence_l431_43120

theorem solution_set_equivalence (x : ℝ) :
  (|(8 - x) / 4| < 3) ↔ (4 < x ∧ x < 20) := by
  sorry

end NUMINAMATH_CALUDE_solution_set_equivalence_l431_43120


namespace NUMINAMATH_CALUDE_radius_of_C₁_is_8_l431_43150

-- Define the points and circles
variable (O X Y Z : ℝ × ℝ)
variable (C₁ C₂ : Set (ℝ × ℝ))

-- Define the conditions
variable (h₁ : O ∈ C₂)
variable (h₂ : X ∈ C₁ ∩ C₂)
variable (h₃ : Y ∈ C₁ ∩ C₂)
variable (h₄ : Z ∈ C₂)
variable (h₅ : Z ∉ C₁)
variable (h₆ : ‖X - Z‖ = 15)
variable (h₇ : ‖O - Z‖ = 17)
variable (h₈ : ‖Y - Z‖ = 8)
variable (h₉ : (X - O) • (Z - O) = 0)  -- Right angle at X

-- Define the radius of C₁
def radius_C₁ (O X : ℝ × ℝ) : ℝ := ‖X - O‖

-- Theorem statement
theorem radius_of_C₁_is_8 :
  radius_C₁ O X = 8 :=
sorry

end NUMINAMATH_CALUDE_radius_of_C₁_is_8_l431_43150


namespace NUMINAMATH_CALUDE_tony_fish_count_l431_43145

def fish_count (initial : ℕ) (years : ℕ) (yearly_addition : ℕ) (yearly_loss : ℕ) : ℕ :=
  initial + years * (yearly_addition - yearly_loss)

theorem tony_fish_count :
  fish_count 2 5 2 1 = 7 := by
  sorry

end NUMINAMATH_CALUDE_tony_fish_count_l431_43145


namespace NUMINAMATH_CALUDE_gcd_lcm_sum_divisibility_l431_43132

theorem gcd_lcm_sum_divisibility (a b : ℕ) (h : a > 0 ∧ b > 0) :
  Nat.gcd a b + Nat.lcm a b = a + b → a ∣ b ∨ b ∣ a := by
  sorry

end NUMINAMATH_CALUDE_gcd_lcm_sum_divisibility_l431_43132


namespace NUMINAMATH_CALUDE_geometric_sequence_term_l431_43114

/-- A geometric sequence is a sequence where each term after the first is found by multiplying the previous term by a fixed, non-zero number called the common ratio. -/
def IsGeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r ≠ 0 ∧ ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_term (a : ℕ → ℝ) :
  IsGeometricSequence a → a 3 = 2 → a 6 = 16 → ∀ n : ℕ, a n = 2^(n - 2) := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_term_l431_43114


namespace NUMINAMATH_CALUDE_equation_solutions_l431_43110

theorem equation_solutions :
  (∀ x : ℚ, x + 1/4 = 7/4 → x = 3/2) ∧
  (∀ x : ℚ, 2/3 + x = 3/4 → x = 1/12) := by
sorry

end NUMINAMATH_CALUDE_equation_solutions_l431_43110


namespace NUMINAMATH_CALUDE_parabola_y_axis_intersection_l431_43162

/-- The parabola y = x^2 - 4 intersects the y-axis at the point (0, -4) -/
theorem parabola_y_axis_intersection :
  let f : ℝ → ℝ := λ x ↦ x^2 - 4
  ∃! p : ℝ × ℝ, p.1 = 0 ∧ p.2 = f p.1 ∧ p = (0, -4) :=
by sorry

end NUMINAMATH_CALUDE_parabola_y_axis_intersection_l431_43162


namespace NUMINAMATH_CALUDE_pythagorean_preservation_l431_43143

theorem pythagorean_preservation (a b c α β γ : ℝ) 
  (h1 : a^2 + b^2 = c^2)
  (h2 : α^2 + β^2 - γ^2 = 2)
  (s := a * α + b * β - c * γ)
  (p := a - α * s)
  (q := b - β * s)
  (r := c - γ * s) :
  p^2 + q^2 = r^2 := by
sorry

end NUMINAMATH_CALUDE_pythagorean_preservation_l431_43143


namespace NUMINAMATH_CALUDE_total_amount_is_234_l431_43142

/-- Represents the share of each person in rupees -/
structure Share where
  x : ℝ
  y : ℝ
  z : ℝ

/-- The total amount distributed -/
def total_amount (s : Share) : ℝ := s.x + s.y + s.z

/-- The condition that y gets 45 paisa for each rupee x gets -/
def y_ratio (s : Share) : Prop := s.y = 0.45 * s.x

/-- The condition that z gets 50 paisa for each rupee x gets -/
def z_ratio (s : Share) : Prop := s.z = 0.50 * s.x

/-- The condition that y's share is 54 rupees -/
def y_share (s : Share) : Prop := s.y = 54

theorem total_amount_is_234 (s : Share) 
  (hy : y_ratio s) (hz : z_ratio s) (hy_share : y_share s) : 
  total_amount s = 234 := by
  sorry


end NUMINAMATH_CALUDE_total_amount_is_234_l431_43142


namespace NUMINAMATH_CALUDE_zero_exponent_rule_l431_43141

theorem zero_exponent_rule (a b : ℤ) (hb : b ≠ 0) : (a / b : ℚ) ^ (0 : ℕ) = 1 := by
  sorry

end NUMINAMATH_CALUDE_zero_exponent_rule_l431_43141


namespace NUMINAMATH_CALUDE_root_sum_reciprocal_shifted_l431_43105

theorem root_sum_reciprocal_shifted (a b c : ℂ) : 
  (a^3 - 2*a - 5 = 0) → 
  (b^3 - 2*b - 5 = 0) → 
  (c^3 - 2*c - 5 = 0) → 
  (1/(a-2) + 1/(b-2) + 1/(c-2) = 10) := by
sorry

end NUMINAMATH_CALUDE_root_sum_reciprocal_shifted_l431_43105


namespace NUMINAMATH_CALUDE_rectangle_area_increase_l431_43173

theorem rectangle_area_increase (L B : ℝ) (h1 : L > 0) (h2 : B > 0) : 
  1.11 * L * (B * (1 + 22/100)) = 1.3542 * (L * B) := by sorry

end NUMINAMATH_CALUDE_rectangle_area_increase_l431_43173


namespace NUMINAMATH_CALUDE_smallest_undefined_inverse_l431_43134

theorem smallest_undefined_inverse (b : ℕ) : 
  (b > 0) → 
  (¬ ∃ x, x * b ≡ 1 [ZMOD 75]) → 
  (¬ ∃ x, x * b ≡ 1 [ZMOD 90]) → 
  (∀ a < b, a > 0 → (∃ x, x * a ≡ 1 [ZMOD 75]) ∨ (∃ x, x * a ≡ 1 [ZMOD 90])) → 
  b = 15 := by
sorry

end NUMINAMATH_CALUDE_smallest_undefined_inverse_l431_43134


namespace NUMINAMATH_CALUDE_number_operation_l431_43170

theorem number_operation (x : ℝ) : x - 10 = 15 → x + 5 = 30 := by
  sorry

end NUMINAMATH_CALUDE_number_operation_l431_43170


namespace NUMINAMATH_CALUDE_equal_cost_miles_l431_43166

/-- Represents the cost of a car rental plan as a function of miles driven -/
def PlanCost (initialFee : ℝ) (costPerMile : ℝ) (miles : ℝ) : ℝ :=
  initialFee + costPerMile * miles

theorem equal_cost_miles : ∃ (miles : ℝ), 
  PlanCost 65 0.40 miles = PlanCost 0 0.60 miles ∧ miles = 325 := by
  sorry

end NUMINAMATH_CALUDE_equal_cost_miles_l431_43166


namespace NUMINAMATH_CALUDE_triangle_abc_properties_l431_43163

theorem triangle_abc_properties (A B C : Real) (a b c : Real) (S : Real) :
  c = Real.sqrt 3 →
  b = 1 →
  C = 2 * π / 3 →  -- 120° in radians
  B = π / 6 ∧      -- 30° in radians
  S = Real.sqrt 3 / 4 :=
by sorry

end NUMINAMATH_CALUDE_triangle_abc_properties_l431_43163


namespace NUMINAMATH_CALUDE_final_algae_count_l431_43158

/-- The number of algae plants in Milford Lake -/
def algae_count : ℕ → ℕ
| 0 => 809  -- Original count
| (n + 1) => algae_count n + 2454  -- Increase

theorem final_algae_count : algae_count 1 = 3263 := by
  sorry

end NUMINAMATH_CALUDE_final_algae_count_l431_43158


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_sum_l431_43178

theorem geometric_sequence_ratio_sum (k p r : ℝ) (hk : k ≠ 0) (hp : p ≠ 1) (hr : r ≠ 1) (hpr : p ≠ r) :
  k * p^2 - k * r^2 = 5 * (k * p - k * r) → p + r = 5 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_ratio_sum_l431_43178


namespace NUMINAMATH_CALUDE_range_of_a_l431_43116

/-- The condition for two distinct real roots -/
def has_two_distinct_real_roots (a : ℝ) : Prop :=
  ∃ x y : ℝ, x ≠ y ∧ x^2 - 2*x + a = 0 ∧ y^2 - 2*y + a = 0

/-- The condition for a hyperbola -/
def is_hyperbola (a : ℝ) : Prop :=
  (a - 3) * (a + 1) < 0

/-- The main theorem -/
theorem range_of_a (a : ℝ) : 
  ¬(has_two_distinct_real_roots a ∨ is_hyperbola a) → a ≥ 3 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l431_43116


namespace NUMINAMATH_CALUDE_ones_digit_of_34_power_power_4_cycle_seventeen_power_odd_main_theorem_l431_43118

theorem ones_digit_of_34_power (n : ℕ) : n > 0 → (34^n) % 10 = (4^n) % 10 := by sorry

theorem power_4_cycle : ∀ n : ℕ, n > 0 → (4^n) % 10 = if n % 2 = 1 then 4 else 6 := by sorry

theorem seventeen_power_odd : (17^17) % 2 = 1 := by sorry

theorem main_theorem : (34^(34*(17^17))) % 10 = 4 := by sorry

end NUMINAMATH_CALUDE_ones_digit_of_34_power_power_4_cycle_seventeen_power_odd_main_theorem_l431_43118


namespace NUMINAMATH_CALUDE_max_workers_l431_43139

/-- Represents the number of workers on the small field -/
def n : ℕ := sorry

/-- The total number of workers in the crew -/
def total_workers : ℕ := 2 * n + 4

/-- The area of the small field -/
def small_area : ℝ := sorry

/-- The area of the large field -/
def large_area : ℝ := 2 * small_area

/-- The time taken to complete work on the small field -/
def small_field_time : ℝ := sorry

/-- The time taken to complete work on the large field -/
def large_field_time : ℝ := sorry

/-- The condition that the small field is still being worked on when the large field is finished -/
axiom work_condition : small_field_time > large_field_time

/-- The theorem stating the maximum number of workers in the crew -/
theorem max_workers : total_workers ≤ 10 := by sorry

end NUMINAMATH_CALUDE_max_workers_l431_43139


namespace NUMINAMATH_CALUDE_sphere_volume_in_cube_l431_43199

/-- The volume of a sphere inscribed in a cube with surface area 24 cm² is (4/3)π cm³ -/
theorem sphere_volume_in_cube (cube_surface_area : ℝ) (sphere_volume : ℝ) : 
  cube_surface_area = 24 →
  sphere_volume = (4/3) * Real.pi := by
  sorry

#check sphere_volume_in_cube

end NUMINAMATH_CALUDE_sphere_volume_in_cube_l431_43199


namespace NUMINAMATH_CALUDE_abc_sum_and_squares_l431_43112

theorem abc_sum_and_squares (a b c : ℝ) 
  (sum_zero : a + b + c = 0) 
  (sum_squares_one : a^2 + b^2 + c^2 = 1) : 
  (a*b + b*c + c*a = -1/2) ∧ (a^4 + b^4 + c^4 = 1/2) := by
  sorry

end NUMINAMATH_CALUDE_abc_sum_and_squares_l431_43112


namespace NUMINAMATH_CALUDE_set_problem_l431_43137

theorem set_problem (U A B C : Finset ℕ) 
  (h_U : U.card = 300)
  (h_A : A.card = 80)
  (h_B : B.card = 70)
  (h_C : C.card = 60)
  (h_AB : (A ∩ B).card = 30)
  (h_AC : (A ∩ C).card = 25)
  (h_BC : (B ∩ C).card = 20)
  (h_ABC : (A ∩ B ∩ C).card = 15)
  (h_outside : (U \ (A ∪ B ∪ C)).card = 65)
  (h_subset : A ∪ B ∪ C ⊆ U) :
  (A \ (B ∪ C)).card = 40 := by
sorry

end NUMINAMATH_CALUDE_set_problem_l431_43137


namespace NUMINAMATH_CALUDE_same_grade_percentage_l431_43130

/-- Given a class of students who took two tests, this theorem proves
    the percentage of students who received the same grade on both tests. -/
theorem same_grade_percentage
  (total_students : ℕ)
  (same_grade_students : ℕ)
  (h1 : total_students = 30)
  (h2 : same_grade_students = 12) :
  (same_grade_students : ℚ) / total_students * 100 = 40 := by
  sorry

end NUMINAMATH_CALUDE_same_grade_percentage_l431_43130


namespace NUMINAMATH_CALUDE_special_permutations_count_l431_43171

/-- The number of permutations of 5 distinct elements where 2 specific elements are not placed at the ends -/
def special_permutations : ℕ :=
  -- Number of ways to choose 2 positions out of 3 for A and E
  (3 * 2) *
  -- Number of ways to arrange the remaining 3 elements
  (3 * 2 * 1)

theorem special_permutations_count : special_permutations = 36 := by
  sorry

end NUMINAMATH_CALUDE_special_permutations_count_l431_43171


namespace NUMINAMATH_CALUDE_rationalize_denominator_l431_43164

theorem rationalize_denominator :
  (1 : ℝ) / (Real.rpow 3 (1/3) + Real.rpow 27 (1/3)) = Real.rpow 9 (1/3) / 12 := by
  sorry

end NUMINAMATH_CALUDE_rationalize_denominator_l431_43164


namespace NUMINAMATH_CALUDE_inverse_157_mod_263_l431_43121

/-- The multiplicative inverse of 157 modulo 263 is 197 -/
theorem inverse_157_mod_263 : ∃ x : ℕ, x < 263 ∧ (157 * x) % 263 = 1 :=
by
  use 197
  sorry

end NUMINAMATH_CALUDE_inverse_157_mod_263_l431_43121


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_sum_l431_43157

theorem quadratic_equation_solution_sum : ∀ c d : ℝ,
  (c^2 - 6*c + 15 = 27) →
  (d^2 - 6*d + 15 = 27) →
  c ≥ d →
  3*c + 2*d = 15 + Real.sqrt 21 := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_sum_l431_43157


namespace NUMINAMATH_CALUDE_smallest_number_divisibility_l431_43176

theorem smallest_number_divisibility (n : ℕ) : 
  (∀ m : ℕ, m < 1572 → ¬(
    (m + 3) % 9 = 0 ∧ 
    (m + 3) % 35 = 0 ∧ 
    (m + 3) % 25 = 0 ∧ 
    (m + 3) % 21 = 0
  )) ∧
  (1572 + 3) % 9 = 0 ∧ 
  (1572 + 3) % 35 = 0 ∧ 
  (1572 + 3) % 25 = 0 ∧ 
  (1572 + 3) % 21 = 0 :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_divisibility_l431_43176


namespace NUMINAMATH_CALUDE_division_problem_l431_43183

theorem division_problem (dividend quotient remainder : ℕ) 
  (h1 : dividend = 3086)
  (h2 : quotient = 36)
  (h3 : remainder = 26)
  : ∃ divisor : ℕ, 
    dividend = divisor * quotient + remainder ∧ 
    divisor = 85 := by
  sorry

end NUMINAMATH_CALUDE_division_problem_l431_43183


namespace NUMINAMATH_CALUDE_product_of_distances_l431_43124

-- Define the ellipse C
def C (x y : ℝ) : Prop := x^2 / 5 + y^2 = 1

-- Define the foci F₁ and F₂
def F₁ : ℝ × ℝ := sorry
def F₂ : ℝ × ℝ := sorry

-- Define a point P on the ellipse
def P : ℝ × ℝ := sorry

-- State that P is on the ellipse C
axiom P_on_C : C P.1 P.2

-- Define the dot product of vectors PF₁ and PF₂
def PF₁_dot_PF₂ : ℝ := sorry

-- State that the dot product of PF₁ and PF₂ is zero
axiom PF₁_perp_PF₂ : PF₁_dot_PF₂ = 0

-- Define the distances |PF₁| and |PF₂|
def dist_PF₁ : ℝ := sorry
def dist_PF₂ : ℝ := sorry

-- Theorem to prove
theorem product_of_distances : dist_PF₁ * dist_PF₂ = 2 := by sorry

end NUMINAMATH_CALUDE_product_of_distances_l431_43124


namespace NUMINAMATH_CALUDE_t_is_perfect_square_l431_43135

theorem t_is_perfect_square (n : ℕ+) (t : ℕ+) (h : t = 2 + 2 * Real.sqrt (1 + 12 * n.val ^ 2)) :
  ∃ (x : ℕ), t = x ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_t_is_perfect_square_l431_43135


namespace NUMINAMATH_CALUDE_plane_equation_l431_43169

/-- A plane in 3D space --/
structure Plane where
  a : ℤ
  b : ℤ
  c : ℤ
  d : ℤ
  a_pos : a > 0
  coprime : Nat.gcd (Int.natAbs a) (Nat.gcd (Int.natAbs b) (Nat.gcd (Int.natAbs c) (Int.natAbs d))) = 1

/-- A point in 3D space --/
structure Point3D where
  x : ℤ
  y : ℤ
  z : ℤ

def is_parallel (p1 p2 : Plane) : Prop :=
  ∃ (k : ℚ), k ≠ 0 ∧ p1.a = k * p2.a ∧ p1.b = k * p2.b ∧ p1.c = k * p2.c

def point_on_plane (pt : Point3D) (p : Plane) : Prop :=
  p.a * pt.x + p.b * pt.y + p.c * pt.z + p.d = 0

theorem plane_equation :
  ∃ (p : Plane),
    is_parallel p { a := 3, b := -2, c := 4, d := -6, a_pos := by simp, coprime := by sorry } ∧
    point_on_plane { x := 2, y := 3, z := -1 } p ∧
    p.a = 3 ∧ p.b = -2 ∧ p.c = 4 ∧ p.d = 4 :=
by sorry

end NUMINAMATH_CALUDE_plane_equation_l431_43169


namespace NUMINAMATH_CALUDE_multiply_mixed_number_l431_43136

theorem multiply_mixed_number : 7 * (9 + 2/5) = 65 + 4/5 := by
  sorry

end NUMINAMATH_CALUDE_multiply_mixed_number_l431_43136


namespace NUMINAMATH_CALUDE_paco_cookies_theorem_l431_43175

/-- Calculates the number of sweet cookies Paco ate given the initial quantities and eating conditions -/
def sweet_cookies_eaten (initial_sweet : ℕ) (initial_salty : ℕ) (sweet_salty_difference : ℕ) : ℕ :=
  initial_salty + sweet_salty_difference

theorem paco_cookies_theorem (initial_sweet initial_salty sweet_salty_difference : ℕ) 
  (h1 : initial_sweet = 39)
  (h2 : initial_salty = 6)
  (h3 : sweet_salty_difference = 9) :
  sweet_cookies_eaten initial_sweet initial_salty sweet_salty_difference = 15 := by
  sorry

#eval sweet_cookies_eaten 39 6 9

end NUMINAMATH_CALUDE_paco_cookies_theorem_l431_43175


namespace NUMINAMATH_CALUDE_m_value_range_l431_43190

/-- The equation x^2 + 2√2x + m = 0 has two distinct real roots -/
def p (m : ℝ) : Prop := ∃ x y : ℝ, x ≠ y ∧ x^2 + 2 * Real.sqrt 2 * x + m = 0 ∧ y^2 + 2 * Real.sqrt 2 * y + m = 0

/-- The solution set of the inequality 4x^2 + 4(m-2)x + 1 > 0 is ℝ -/
def q (m : ℝ) : Prop := ∀ x : ℝ, 4 * x^2 + 4 * (m - 2) * x + 1 > 0

/-- The range of values for m -/
def m_range (m : ℝ) : Prop := m ≤ 1 ∨ (2 ≤ m ∧ m < 3)

theorem m_value_range :
  ∀ m : ℝ, (p m ∨ q m) ∧ ¬(p m ∧ q m) → m_range m :=
by sorry

end NUMINAMATH_CALUDE_m_value_range_l431_43190


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_range_l431_43185

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

theorem geometric_sequence_sum_range
  (a : ℕ → ℝ)
  (h_geom : is_geometric_sequence a)
  (h_prod : a 4 * a 8 = 9) :
  ∀ x : ℝ, (x ∈ Set.Iic (-6) ∪ Set.Ici 6) ↔ ∃ (a₃ a₉ : ℝ), a 3 = a₃ ∧ a 9 = a₉ ∧ a₃ + a₉ = x :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_range_l431_43185


namespace NUMINAMATH_CALUDE_S_max_at_9_l431_43181

/-- An arithmetic sequence -/
def arithmetic_sequence : ℕ → ℝ := sorry

/-- Sum of the first n terms of the arithmetic sequence -/
def S (n : ℕ) : ℝ := sorry

/-- Conditions of the problem -/
axiom S_18_positive : S 18 > 0
axiom S_19_negative : S 19 < 0

/-- Theorem: S_n is maximum when n = 9 -/
theorem S_max_at_9 : ∀ k : ℕ, S 9 ≥ S k := by sorry

end NUMINAMATH_CALUDE_S_max_at_9_l431_43181


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_l431_43189

theorem simplify_and_evaluate (x : ℝ) (h : x = Real.sqrt 2 - 1) :
  (x^2 + 2*x + 1) / (x + 1) = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_l431_43189


namespace NUMINAMATH_CALUDE_chess_tournament_games_l431_43127

/-- The number of games played in a round-robin tournament -/
def gamesPlayed (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Theorem: In a chess group with 15 players, where each player plays every other player exactly once,
    and each game is played by two players, the total number of games played is 105. -/
theorem chess_tournament_games :
  gamesPlayed 15 = 105 := by
  sorry

#eval gamesPlayed 15  -- This will evaluate to 105

end NUMINAMATH_CALUDE_chess_tournament_games_l431_43127


namespace NUMINAMATH_CALUDE_price_difference_l431_43160

def original_price : ℝ := 1200

def price_after_increase (p : ℝ) : ℝ := p * 1.1

def price_after_decrease (p : ℝ) : ℝ := p * 0.85

def final_price : ℝ := price_after_decrease (price_after_increase original_price)

theorem price_difference : original_price - final_price = 78 := by
  sorry

end NUMINAMATH_CALUDE_price_difference_l431_43160


namespace NUMINAMATH_CALUDE_max_value_of_prime_sum_diff_l431_43147

theorem max_value_of_prime_sum_diff (a b c : ℕ) : 
  Nat.Prime a ∧ Nat.Prime b ∧ Nat.Prime c ∧  -- a, b, c are prime
  a ≠ b ∧ b ≠ c ∧ a ≠ c ∧                    -- a, b, c are distinct
  a + b * c = 37 →                           -- given equation
  ∀ x y z : ℕ, 
    Nat.Prime x ∧ Nat.Prime y ∧ Nat.Prime z ∧
    x ≠ y ∧ y ≠ z ∧ x ≠ z ∧
    x + y * z = 37 →
    x + y - z ≤ a + b - c ∧                  -- a + b - c is maximum
    a + b - c = 32                           -- the maximum value is 32
  := by sorry

end NUMINAMATH_CALUDE_max_value_of_prime_sum_diff_l431_43147


namespace NUMINAMATH_CALUDE_division_multiplication_result_l431_43131

theorem division_multiplication_result : 
  let x : ℝ := 5.5
  let y : ℝ := (x / 6) * 12
  y = 11 := by sorry

end NUMINAMATH_CALUDE_division_multiplication_result_l431_43131


namespace NUMINAMATH_CALUDE_possible_a3_values_l431_43123

/-- An arithmetic sequence with a non-zero common difference -/
structure ArithmeticSequence where
  a : ℕ → ℚ
  d : ℚ
  h_d : d ≠ 0
  h_arithmetic : ∀ n, a (n + 1) = a n + d

/-- Theorem: Possible values of a_3 in the arithmetic sequence -/
theorem possible_a3_values (seq : ArithmeticSequence) 
  (h_a5 : seq.a 5 = 6)
  (h_a3_gt_1 : seq.a 3 > 1)
  (h_geometric : ∃ (m : ℕ → ℕ), 
    (∀ t, 5 < m t ∧ (t > 0 → m (t-1) < m t)) ∧ 
    (∀ t, ∃ r, seq.a (m t) = seq.a 3 * r^(t+1) ∧ seq.a 5 = seq.a 3 * r^2)) :
  seq.a 3 = 3 ∨ seq.a 3 = 2 ∨ seq.a 3 = 3/2 :=
sorry

end NUMINAMATH_CALUDE_possible_a3_values_l431_43123


namespace NUMINAMATH_CALUDE_projectile_meeting_time_l431_43122

/-- Time for two projectiles to meet --/
theorem projectile_meeting_time (initial_distance : ℝ) (speed1 speed2 : ℝ) :
  initial_distance = 1998 →
  speed1 = 444 →
  speed2 = 555 →
  (initial_distance / (speed1 + speed2)) * 60 = 120 := by
  sorry

end NUMINAMATH_CALUDE_projectile_meeting_time_l431_43122


namespace NUMINAMATH_CALUDE_paint_project_total_l431_43152

/-- The total amount of paint needed for a project, given the amount left from a previous project and the amount that needs to be bought. -/
def total_paint (left_over : ℕ) (to_buy : ℕ) : ℕ :=
  left_over + to_buy

/-- Theorem stating that the total amount of paint needed is 333 liters. -/
theorem paint_project_total :
  total_paint 157 176 = 333 := by
  sorry

end NUMINAMATH_CALUDE_paint_project_total_l431_43152


namespace NUMINAMATH_CALUDE_three_squares_before_2300_l431_43186

theorem three_squares_before_2300 : 
  ∃ (n : ℕ), n = 2025 ∧ 
  (∃ (a b c : ℕ), 
    n < a^2 ∧ a^2 < b^2 ∧ b^2 < c^2 ∧ c^2 ≤ 2300 ∧
    ∀ (x : ℕ), n < x^2 ∧ x^2 ≤ 2300 → x^2 = a^2 ∨ x^2 = b^2 ∨ x^2 = c^2) ∧
  ∀ (m : ℕ), m > n → 
    ¬(∃ (a b c : ℕ), 
      m < a^2 ∧ a^2 < b^2 ∧ b^2 < c^2 ∧ c^2 ≤ 2300 ∧
      ∀ (x : ℕ), m < x^2 ∧ x^2 ≤ 2300 → x^2 = a^2 ∨ x^2 = b^2 ∨ x^2 = c^2) :=
by sorry

end NUMINAMATH_CALUDE_three_squares_before_2300_l431_43186


namespace NUMINAMATH_CALUDE_hyperbola_equation_l431_43149

/-- Given a hyperbola with the equation x²/a² - y²/b² = 1, where a > 0 and b > 0,
    left focus F₁ and right focus F₂ on the x-axis,
    point P(3,4) on an asymptote, and |PF₁ + PF₂| = |F₁F₂|,
    prove that the equation of the hyperbola is x²/9 - y²/16 = 1 -/
theorem hyperbola_equation (a b : ℝ) (h1 : a > 0) (h2 : b > 0)
  (F₁ F₂ : ℝ × ℝ) (hF : ∃ c : ℝ, F₁ = (-c, 0) ∧ F₂ = (c, 0))
  (P : ℝ × ℝ) (hP : P = (3, 4))
  (h_asymptote : ∃ k : ℝ, k * 3 = 4 ∧ (∀ x y : ℝ, y = k * x → x^2/a^2 - y^2/b^2 = 1))
  (h_vector_sum : ‖P - F₁ + (P - F₂)‖ = ‖F₂ - F₁‖) :
  ∀ x y : ℝ, x^2/9 - y^2/16 = 1 ↔ x^2/a^2 - y^2/b^2 = 1 :=
sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l431_43149


namespace NUMINAMATH_CALUDE_functional_equation_solution_l431_43196

/-- A function g: ℝ → ℝ satisfying the given functional equation -/
def FunctionalEquation (g : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, g (g x - y) = g x + g (g y - g (-x)) + 2 * x

/-- Theorem stating that any function satisfying the functional equation must be g(x) = -2x -/
theorem functional_equation_solution (g : ℝ → ℝ) (h : FunctionalEquation g) :
  ∀ x : ℝ, g x = -2 * x := by
  sorry

end NUMINAMATH_CALUDE_functional_equation_solution_l431_43196


namespace NUMINAMATH_CALUDE_complex_sum_of_parts_l431_43106

theorem complex_sum_of_parts (z : ℂ) (h : z * (1 + Complex.I) = 1 - Complex.I) :
  (z.re : ℝ) + (z.im : ℝ) = -1 := by sorry

end NUMINAMATH_CALUDE_complex_sum_of_parts_l431_43106
