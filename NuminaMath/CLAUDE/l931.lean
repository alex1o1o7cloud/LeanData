import Mathlib

namespace NUMINAMATH_CALUDE_bank_transfer_theorem_l931_93134

def calculate_final_balance (initial_balance : ℚ) (transfer1 : ℚ) (transfer2 : ℚ) (service_charge_rate : ℚ) : ℚ :=
  let service_charge1 := transfer1 * service_charge_rate
  let service_charge2 := transfer2 * service_charge_rate
  initial_balance - (transfer1 + service_charge1) - service_charge2

theorem bank_transfer_theorem (initial_balance : ℚ) (transfer1 : ℚ) (transfer2 : ℚ) (service_charge_rate : ℚ) 
  (h1 : initial_balance = 400)
  (h2 : transfer1 = 90)
  (h3 : transfer2 = 60)
  (h4 : service_charge_rate = 2/100) :
  calculate_final_balance initial_balance transfer1 transfer2 service_charge_rate = 307 := by
  sorry

end NUMINAMATH_CALUDE_bank_transfer_theorem_l931_93134


namespace NUMINAMATH_CALUDE_count_integer_lengths_specific_triangle_l931_93157

/-- Represents a right triangle with integer side lengths -/
structure RightTriangle where
  a : ℕ  -- length of first leg
  b : ℕ  -- length of second leg
  c : ℕ  -- length of hypotenuse
  right_angle : c^2 = a^2 + b^2  -- Pythagorean theorem

/-- Counts the number of distinct integer lengths of line segments
    that can be drawn from a vertex to the opposite side -/
def count_integer_lengths (t : RightTriangle) : ℕ :=
  -- Implementation details omitted
  sorry

/-- The main theorem -/
theorem count_integer_lengths_specific_triangle :
  ∃ t : RightTriangle, t.a = 15 ∧ t.b = 20 ∧ count_integer_lengths t = 9 :=
by
  sorry

end NUMINAMATH_CALUDE_count_integer_lengths_specific_triangle_l931_93157


namespace NUMINAMATH_CALUDE_tile1_in_position_B_l931_93197

-- Define a tile with numbers on its sides
structure Tile where
  top : ℕ
  right : ℕ
  bottom : ℕ
  left : ℕ

-- Define the four tiles
def tile1 : Tile := ⟨5, 3, 2, 4⟩
def tile2 : Tile := ⟨3, 1, 5, 2⟩
def tile3 : Tile := ⟨4, 0, 6, 5⟩
def tile4 : Tile := ⟨2, 4, 3, 0⟩

-- Define the possible positions
inductive Position
  | A | B | C | D

-- Function to check if two tiles can be adjacent
def canBeAdjacent (t1 t2 : Tile) : Bool :=
  (t1.right = t2.left) ∨ (t1.left = t2.right) ∨ (t1.top = t2.bottom) ∨ (t1.bottom = t2.top)

-- Theorem: Tile 1 must be in position B
theorem tile1_in_position_B :
  ∃ (p2 p3 p4 : Position), 
    p2 ≠ Position.B ∧ p3 ≠ Position.B ∧ p4 ≠ Position.B ∧
    p2 ≠ p3 ∧ p2 ≠ p4 ∧ p3 ≠ p4 ∧
    (canBeAdjacent tile1 tile2 → 
      (p2 = Position.A ∨ p2 = Position.C ∨ p2 = Position.D)) ∧
    (canBeAdjacent tile1 tile3 → 
      (p3 = Position.A ∨ p3 = Position.C ∨ p3 = Position.D)) ∧
    (canBeAdjacent tile1 tile4 → 
      (p4 = Position.A ∨ p4 = Position.C ∨ p4 = Position.D)) :=
by
  sorry


end NUMINAMATH_CALUDE_tile1_in_position_B_l931_93197


namespace NUMINAMATH_CALUDE_sum_of_solutions_is_zero_l931_93192

theorem sum_of_solutions_is_zero :
  ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ 
  (9 * x₁) / 27 = 6 / x₁ ∧
  (9 * x₂) / 27 = 6 / x₂ ∧
  x₁ + x₂ = 0 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_solutions_is_zero_l931_93192


namespace NUMINAMATH_CALUDE_amit_work_days_l931_93171

theorem amit_work_days (amit_rate : ℚ) (ananthu_rate : ℚ) : 
  ananthu_rate = 1 / 45 →
  amit_rate * 3 + ananthu_rate * 36 = 1 →
  amit_rate = 1 / 15 :=
by
  sorry

end NUMINAMATH_CALUDE_amit_work_days_l931_93171


namespace NUMINAMATH_CALUDE_right_triangle_rotation_creates_cone_l931_93103

/-- A right triangle is a triangle with one right angle -/
structure RightTriangle where
  -- We don't need to define the specifics of a right triangle for this statement
  mk :: 

/-- A cone is a three-dimensional geometric shape with a circular base that tapers to a point -/
structure Cone where
  -- We don't need to define the specifics of a cone for this statement
  mk ::

/-- Rotation of a right triangle around one of its legs -/
def rotateAroundLeg (t : RightTriangle) : Cone :=
  sorry

theorem right_triangle_rotation_creates_cone (t : RightTriangle) :
  ∃ (c : Cone), rotateAroundLeg t = c :=
sorry

end NUMINAMATH_CALUDE_right_triangle_rotation_creates_cone_l931_93103


namespace NUMINAMATH_CALUDE_caiden_roofing_cost_l931_93166

def metal_roofing_cost (total_feet : ℕ) (free_feet : ℕ) (cost_per_foot : ℚ) 
  (discount_rate : ℚ) (sales_tax_rate : ℚ) : ℚ :=
  let paid_feet := total_feet - free_feet
  let initial_cost := paid_feet * cost_per_foot
  let discounted_cost := initial_cost * (1 - discount_rate)
  let total_before_tax := discounted_cost
  let total_cost := total_before_tax * (1 + sales_tax_rate)
  total_cost

theorem caiden_roofing_cost :
  metal_roofing_cost 300 250 8 (15/100) (5/100) = 357 := by
  sorry

end NUMINAMATH_CALUDE_caiden_roofing_cost_l931_93166


namespace NUMINAMATH_CALUDE_geometric_sequence_property_l931_93187

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

theorem geometric_sequence_property (a : ℕ → ℝ) (h_geom : is_geometric_sequence a) 
  (h1 : a 3 * a 6 = 9) (h2 : a 2 * a 4 * a 5 = 27) : a 2 = 3 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_property_l931_93187


namespace NUMINAMATH_CALUDE_systematic_sample_theorem_l931_93175

/-- Represents a systematic sample from a class -/
structure SystematicSample where
  totalStudents : ℕ
  sampleSize : ℕ
  firstSample : ℕ
  knownSamples : Finset ℕ

/-- Checks if a number is part of the systematic sample -/
def isInSample (s : SystematicSample) (n : ℕ) : Prop :=
  ∃ k : ℕ, n = s.firstSample + k * (s.totalStudents / s.sampleSize)

theorem systematic_sample_theorem (s : SystematicSample)
  (h1 : s.totalStudents = 48)
  (h2 : s.sampleSize = 6)
  (h3 : s.firstSample = 5)
  (h4 : s.knownSamples = {5, 21, 29, 37, 45})
  (h5 : ∀ n ∈ s.knownSamples, isInSample s n) :
  isInSample s 13 ∧ (∀ n, isInSample s n → n = 13 ∨ n ∈ s.knownSamples) :=
sorry

end NUMINAMATH_CALUDE_systematic_sample_theorem_l931_93175


namespace NUMINAMATH_CALUDE_complex_number_quadrant_l931_93139

theorem complex_number_quadrant : 
  let z : ℂ := (3 * Complex.I) / (1 + 2 * Complex.I)
  (0 < z.re) ∧ (0 < z.im) := by
  sorry

end NUMINAMATH_CALUDE_complex_number_quadrant_l931_93139


namespace NUMINAMATH_CALUDE_modulus_v_is_five_l931_93150

/-- Given two complex numbers u and v, prove that |v| = 5 when uv = 15 - 20i and |u| = 5 -/
theorem modulus_v_is_five (u v : ℂ) (h1 : u * v = 15 - 20 * I) (h2 : Complex.abs u = 5) : 
  Complex.abs v = 5 := by
sorry

end NUMINAMATH_CALUDE_modulus_v_is_five_l931_93150


namespace NUMINAMATH_CALUDE_exists_long_period_in_range_l931_93106

/-- The length of the period of the decimal expansion of 1/n -/
def period_length (n : ℕ) : ℕ := sorry

theorem exists_long_period_in_range :
  ∀ (start : ℕ), 
  (10^99 ≤ start) →
  ∃ (n : ℕ), 
    (start ≤ n) ∧ 
    (n < start + 100000) ∧ 
    (period_length n > 2011) := by
  sorry

end NUMINAMATH_CALUDE_exists_long_period_in_range_l931_93106


namespace NUMINAMATH_CALUDE_cow_field_theorem_l931_93117

theorem cow_field_theorem (total_cows : ℕ) (female_cows : ℕ) (male_cows : ℕ) 
  (spotted_females : ℕ) (horned_males : ℕ) : 
  total_cows = 300 →
  female_cows = 2 * male_cows →
  female_cows + male_cows = total_cows →
  spotted_females = female_cows / 2 →
  horned_males = male_cows / 2 →
  spotted_females - horned_males = 50 := by
sorry

end NUMINAMATH_CALUDE_cow_field_theorem_l931_93117


namespace NUMINAMATH_CALUDE_promotion_savings_l931_93168

/-- Calculates the total cost of two pairs of shoes under Promotion A -/
def cost_promotion_a (price : ℝ) : ℝ :=
  price + price * 0.6

/-- Calculates the total cost of two pairs of shoes under Promotion B -/
def cost_promotion_b (price : ℝ) : ℝ :=
  price + (price - 15)

/-- The price of each pair of shoes -/
def shoe_price : ℝ := 50

theorem promotion_savings : 
  cost_promotion_b shoe_price - cost_promotion_a shoe_price = 5 := by
sorry

end NUMINAMATH_CALUDE_promotion_savings_l931_93168


namespace NUMINAMATH_CALUDE_daily_tylenol_intake_l931_93102

def tablets_per_dose : ℕ := 2
def mg_per_tablet : ℕ := 375
def hours_between_doses : ℕ := 6
def hours_per_day : ℕ := 24

theorem daily_tylenol_intake :
  tablets_per_dose * mg_per_tablet * (hours_per_day / hours_between_doses) = 3000 := by
  sorry

end NUMINAMATH_CALUDE_daily_tylenol_intake_l931_93102


namespace NUMINAMATH_CALUDE_max_dominos_with_room_for_one_l931_93186

/-- Represents a chessboard -/
structure Chessboard :=
  (rows : Nat)
  (cols : Nat)

/-- Represents a domino -/
structure Domino :=
  (width : Nat)
  (height : Nat)

/-- Represents a placement of dominos on a chessboard -/
def DominoPlacement := List (Nat × Nat)

/-- Function to check if a domino placement is valid -/
def isValidPlacement (board : Chessboard) (domino : Domino) (placement : DominoPlacement) : Prop :=
  sorry

/-- Function to check if there's room for one more domino -/
def hasRoomForOne (board : Chessboard) (domino : Domino) (placement : DominoPlacement) : Prop :=
  sorry

/-- The main theorem -/
theorem max_dominos_with_room_for_one (board : Chessboard) (domino : Domino) :
  board.rows = 6 →
  board.cols = 6 →
  domino.width = 1 →
  domino.height = 2 →
  (∃ (n : Nat) (placement : DominoPlacement),
    n = 11 ∧
    isValidPlacement board domino placement ∧
    placement.length = n ∧
    hasRoomForOne board domino placement) ∧
  (∀ (m : Nat) (placement : DominoPlacement),
    m > 11 →
    isValidPlacement board domino placement →
    placement.length = m →
    ¬hasRoomForOne board domino placement) :=
  by sorry

end NUMINAMATH_CALUDE_max_dominos_with_room_for_one_l931_93186


namespace NUMINAMATH_CALUDE_complex_equation_solution_l931_93199

theorem complex_equation_solution (z : ℂ) (h : (3 - 4 * Complex.I) * z = 25) : z = 3 + 4 * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l931_93199


namespace NUMINAMATH_CALUDE_orange_juice_price_l931_93105

/-- The cost of a glass of orange juice -/
def orange_juice_cost : ℚ := 85/100

/-- The cost of a bagel -/
def bagel_cost : ℚ := 95/100

/-- The cost of a sandwich -/
def sandwich_cost : ℚ := 465/100

/-- The cost of milk -/
def milk_cost : ℚ := 115/100

/-- The additional amount spent on lunch compared to breakfast -/
def lunch_breakfast_difference : ℚ := 4

theorem orange_juice_price : 
  bagel_cost + orange_juice_cost + lunch_breakfast_difference = sandwich_cost + milk_cost := by
  sorry

end NUMINAMATH_CALUDE_orange_juice_price_l931_93105


namespace NUMINAMATH_CALUDE_tan_domain_theorem_l931_93140

open Set
open Real

noncomputable def f (x : ℝ) : ℝ := Real.tan (2 * x - π / 4)

def domain_set : Set ℝ := ⋃ k : ℤ, Ioo ((k : ℝ) * π / 2 - π / 8) ((k : ℝ) * π / 2 + 3 * π / 8)

theorem tan_domain_theorem :
  {x : ℝ | ∃ y, f x = y} = domain_set :=
sorry

end NUMINAMATH_CALUDE_tan_domain_theorem_l931_93140


namespace NUMINAMATH_CALUDE_max_product_range_l931_93111

-- Define the functions h and k
def h : ℝ → ℝ := sorry
def k : ℝ → ℝ := sorry

-- State the theorem
theorem max_product_range (h k : ℝ → ℝ) 
  (h_range : ∀ x, -3 ≤ h x ∧ h x ≤ 5)
  (k_range : ∀ x, 0 ≤ k x ∧ k x ≤ 4) :
  ∃ d, ∀ x, h x ^ 2 * k x ≤ d ∧ d = 100 :=
sorry

end NUMINAMATH_CALUDE_max_product_range_l931_93111


namespace NUMINAMATH_CALUDE_expand_product_l931_93169

theorem expand_product (y : ℝ) (h : y ≠ 0) :
  (3 / 7) * ((7 / y) + 8 * y^2 - 3 * y) = 3 / y + (24 * y^2 - 9 * y) / 7 := by
  sorry

end NUMINAMATH_CALUDE_expand_product_l931_93169


namespace NUMINAMATH_CALUDE_crayons_per_row_l931_93133

/-- Given that Faye has 16 rows of crayons and pencils, with a total of 96 crayons,
    prove that there are 6 crayons in each row. -/
theorem crayons_per_row (total_rows : ℕ) (total_crayons : ℕ) (h1 : total_rows = 16) (h2 : total_crayons = 96) :
  total_crayons / total_rows = 6 := by
  sorry

end NUMINAMATH_CALUDE_crayons_per_row_l931_93133


namespace NUMINAMATH_CALUDE_f_composition_equals_one_over_e_l931_93174

noncomputable def f (x : ℝ) : ℝ :=
  if x < 0 then Real.exp x else Real.log x

theorem f_composition_equals_one_over_e :
  f (f (1 / Real.exp 1)) = 1 / Real.exp 1 := by sorry

end NUMINAMATH_CALUDE_f_composition_equals_one_over_e_l931_93174


namespace NUMINAMATH_CALUDE_max_min_f_on_interval_l931_93164

-- Define the function f(x) = -3x + 1
def f (x : ℝ) : ℝ := -3 * x + 1

-- State the theorem
theorem max_min_f_on_interval :
  (∀ x ∈ Set.Icc 0 1, f x ≤ 1) ∧
  (∃ x ∈ Set.Icc 0 1, f x = 1) ∧
  (∀ x ∈ Set.Icc 0 1, f x ≥ -2) ∧
  (∃ x ∈ Set.Icc 0 1, f x = -2) :=
sorry

end NUMINAMATH_CALUDE_max_min_f_on_interval_l931_93164


namespace NUMINAMATH_CALUDE_problem_solution_l931_93113

/-- Given that 2x^5 - x^3 + 4x^2 + 3x - 5 + g(x) = 7x^3 - 4x + 2,
    prove that g(x) = -2x^5 + 6x^3 - 4x^2 - x + 7 -/
theorem problem_solution (x : ℝ) :
  let g : ℝ → ℝ := λ x => -2*x^5 + 6*x^3 - 4*x^2 - x + 7
  2*x^5 - x^3 + 4*x^2 + 3*x - 5 + g x = 7*x^3 - 4*x + 2 :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l931_93113


namespace NUMINAMATH_CALUDE_son_work_time_l931_93119

/-- Given a task that can be completed by a man in 7 days or by the man and his son together in 3 days, 
    this theorem proves that the son can complete the task alone in 5.25 days. -/
theorem son_work_time (man_time : ℝ) (combined_time : ℝ) (son_time : ℝ) : 
  man_time = 7 → combined_time = 3 → son_time = 21 / 4 := by
  sorry

end NUMINAMATH_CALUDE_son_work_time_l931_93119


namespace NUMINAMATH_CALUDE_motorcycle_license_count_l931_93118

/-- The number of possible letters for a motorcycle license -/
def num_letters : ℕ := 3

/-- The number of digits in a motorcycle license -/
def num_digits : ℕ := 6

/-- The number of possible choices for each digit -/
def choices_per_digit : ℕ := 10

/-- The total number of possible motorcycle licenses -/
def total_licenses : ℕ := num_letters * (choices_per_digit ^ num_digits)

theorem motorcycle_license_count :
  total_licenses = 3000000 := by
  sorry

end NUMINAMATH_CALUDE_motorcycle_license_count_l931_93118


namespace NUMINAMATH_CALUDE_bowling_ball_weight_l931_93104

theorem bowling_ball_weight : 
  ∀ (b c : ℝ),
  5 * b = 2 * c →
  3 * c = 72 →
  b = 9.6 := by
sorry

end NUMINAMATH_CALUDE_bowling_ball_weight_l931_93104


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l931_93124

theorem sqrt_equation_solution (x : ℝ) :
  Real.sqrt x / 15 = 4 → x = 3600 := by sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l931_93124


namespace NUMINAMATH_CALUDE_root_difference_l931_93180

/-- The difference between the larger and smaller roots of the quadratic equation
    x^2 - 2px + (p^2 - 4p + 4) = 0, where p is a real number. -/
theorem root_difference (p : ℝ) : 
  let a := 1
  let b := -2*p
  let c := p^2 - 4*p + 4
  let discriminant := b^2 - 4*a*c
  let larger_root := (-b + Real.sqrt discriminant) / (2*a)
  let smaller_root := (-b - Real.sqrt discriminant) / (2*a)
  larger_root - smaller_root = 4 * Real.sqrt (p - 1) := by
sorry

end NUMINAMATH_CALUDE_root_difference_l931_93180


namespace NUMINAMATH_CALUDE_lowest_possible_score_l931_93190

def total_tests : ℕ := 6
def max_score : ℕ := 200
def target_average : ℕ := 170

def first_four_scores : List ℕ := [150, 180, 175, 160]

theorem lowest_possible_score :
  ∃ (score1 score2 : ℕ),
    score1 ≤ max_score ∧ 
    score2 ≤ max_score ∧
    (List.sum first_four_scores + score1 + score2) / total_tests = target_average ∧
    (∀ (s1 s2 : ℕ), 
      s1 ≤ max_score → 
      s2 ≤ max_score → 
      (List.sum first_four_scores + s1 + s2) / total_tests = target_average → 
      min s1 s2 ≥ min score1 score2) ∧
    min score1 score2 = 155 :=
by sorry

end NUMINAMATH_CALUDE_lowest_possible_score_l931_93190


namespace NUMINAMATH_CALUDE_point_coordinates_l931_93151

/-- A point in the 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents the second quadrant of the 2D plane -/
def SecondQuadrant (p : Point) : Prop :=
  p.x < 0 ∧ p.y > 0

/-- Distance from a point to the x-axis -/
def DistToXAxis (p : Point) : ℝ :=
  |p.y|

/-- Distance from a point to the y-axis -/
def DistToYAxis (p : Point) : ℝ :=
  |p.x|

theorem point_coordinates :
  ∀ (p : Point),
    SecondQuadrant p →
    DistToXAxis p = 3 →
    DistToYAxis p = 7 →
    p.x = -7 ∧ p.y = 3 :=
by sorry

end NUMINAMATH_CALUDE_point_coordinates_l931_93151


namespace NUMINAMATH_CALUDE_liam_fourth_week_l931_93156

/-- A sequence of four numbers representing chapters read each week -/
def ChapterSequence := Fin 4 → ℕ

/-- The properties of Liam's reading sequence -/
def IsLiamSequence (s : ChapterSequence) : Prop :=
  (∀ i : Fin 3, s (i + 1) = s i + 3) ∧
  (s 0 + s 1 + s 2 + s 3 = 50)

/-- Theorem stating that the fourth number in Liam's sequence is 17 -/
theorem liam_fourth_week (s : ChapterSequence) (h : IsLiamSequence s) : s 3 = 17 := by
  sorry

end NUMINAMATH_CALUDE_liam_fourth_week_l931_93156


namespace NUMINAMATH_CALUDE_min_distance_to_curve_l931_93183

theorem min_distance_to_curve :
  let f (x y : ℝ) := Real.sqrt (x^2 + y^2)
  let g (x y : ℝ) := 6*x + 8*y - 4*x^2
  ∃ (min : ℝ), min = Real.sqrt 2061 / 8 ∧
    (∀ x y : ℝ, g x y = 48 → f x y ≥ min) ∧
    (∃ x y : ℝ, g x y = 48 ∧ f x y = min) :=
by sorry

end NUMINAMATH_CALUDE_min_distance_to_curve_l931_93183


namespace NUMINAMATH_CALUDE_maggie_bouncy_balls_l931_93131

/-- The number of packs of red bouncy balls Maggie bought -/
def red_packs : ℕ := 4

/-- The number of packs of yellow bouncy balls Maggie bought -/
def yellow_packs : ℕ := 8

/-- The number of packs of green bouncy balls Maggie bought -/
def green_packs : ℕ := 4

/-- The number of bouncy balls in each pack -/
def balls_per_pack : ℕ := 10

/-- The total number of bouncy balls Maggie bought -/
def total_balls : ℕ := red_packs * balls_per_pack + yellow_packs * balls_per_pack + green_packs * balls_per_pack

theorem maggie_bouncy_balls : total_balls = 160 := by
  sorry

end NUMINAMATH_CALUDE_maggie_bouncy_balls_l931_93131


namespace NUMINAMATH_CALUDE_odd_square_minus_one_div_eight_l931_93162

theorem odd_square_minus_one_div_eight (a : ℕ) (h1 : a > 0) (h2 : Odd a) :
  ∃ k : ℤ, a^2 - 1 = 8 * k :=
sorry

end NUMINAMATH_CALUDE_odd_square_minus_one_div_eight_l931_93162


namespace NUMINAMATH_CALUDE_car_trip_duration_l931_93114

/-- Proves that a car trip with given conditions has a total duration of 8 hours -/
theorem car_trip_duration (initial_speed : ℝ) (initial_time : ℝ) (later_speed : ℝ) (avg_speed : ℝ) 
  (h1 : initial_speed = 30)
  (h2 : initial_time = 6)
  (h3 : later_speed = 46)
  (h4 : avg_speed = 34) :
  ∃ (total_time : ℝ), 
    (initial_speed * initial_time + later_speed * (total_time - initial_time)) / total_time = avg_speed ∧
    total_time = 8 := by
  sorry


end NUMINAMATH_CALUDE_car_trip_duration_l931_93114


namespace NUMINAMATH_CALUDE_imaginary_part_of_one_plus_i_to_fifth_l931_93120

theorem imaginary_part_of_one_plus_i_to_fifth (i : ℂ) : i * i = -1 → Complex.im ((1 + i)^5) = -4 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_one_plus_i_to_fifth_l931_93120


namespace NUMINAMATH_CALUDE_classroom_handshakes_l931_93115

theorem classroom_handshakes (m n : ℕ) (h1 : m ≥ 3) (h2 : n ≥ 3) 
  (h3 : 2 * m * n - m - n = 252) : m * n = 72 := by
  sorry

end NUMINAMATH_CALUDE_classroom_handshakes_l931_93115


namespace NUMINAMATH_CALUDE_nonagon_diagonal_intersection_probability_l931_93161

/-- The probability of two randomly chosen diagonals intersecting in a regular nonagon -/
theorem nonagon_diagonal_intersection_probability :
  let n : ℕ := 9  -- number of sides in a nonagon
  let total_diagonals : ℕ := n.choose 2 - n
  let diagonal_pairs : ℕ := total_diagonals.choose 2
  let intersecting_pairs : ℕ := n.choose 4
  (intersecting_pairs : ℚ) / diagonal_pairs = 14 / 39 :=
by sorry


end NUMINAMATH_CALUDE_nonagon_diagonal_intersection_probability_l931_93161


namespace NUMINAMATH_CALUDE_turnips_sum_l931_93123

/-- The number of turnips Keith grew -/
def keith_turnips : ℕ := 6

/-- The number of turnips Alyssa grew -/
def alyssa_turnips : ℕ := 9

/-- The total number of turnips grown by Keith and Alyssa -/
def total_turnips : ℕ := keith_turnips + alyssa_turnips

theorem turnips_sum :
  total_turnips = 15 := by sorry

end NUMINAMATH_CALUDE_turnips_sum_l931_93123


namespace NUMINAMATH_CALUDE_three_number_ratio_problem_l931_93144

theorem three_number_ratio_problem (a b c : ℝ) 
  (h_sum : a + b + c = 120)
  (h_ratio1 : a / b = 3 / 4)
  (h_ratio2 : b / c = 3 / 5)
  (h_positive : a > 0 ∧ b > 0 ∧ c > 0) :
  b = 1440 / 41 := by
sorry

end NUMINAMATH_CALUDE_three_number_ratio_problem_l931_93144


namespace NUMINAMATH_CALUDE_hot_water_bottle_price_l931_93130

/-- Proves that the price of a hot-water bottle is 6 dollars given the problem conditions --/
theorem hot_water_bottle_price :
  let thermometer_price : ℚ := 2
  let total_sales : ℚ := 1200
  let thermometer_to_bottle_ratio : ℕ := 7
  let bottles_sold : ℕ := 60
  let thermometers_sold : ℕ := thermometer_to_bottle_ratio * bottles_sold
  let bottle_price : ℚ := (total_sales - (thermometer_price * thermometers_sold)) / bottles_sold
  bottle_price = 6 :=
by sorry

end NUMINAMATH_CALUDE_hot_water_bottle_price_l931_93130


namespace NUMINAMATH_CALUDE_min_distance_is_zero_l931_93159

-- Define the two functions
def f (x : ℝ) : ℝ := |x|
def g (x : ℝ) : ℝ := x^2 - 5*x + 4

-- Define the distance function between the two graphs
def distance (x : ℝ) : ℝ := |f x - g x|

-- Theorem statement
theorem min_distance_is_zero :
  ∃ (x : ℝ), distance x = 0 ∧ ∀ (y : ℝ), distance y ≥ 0 :=
sorry

end NUMINAMATH_CALUDE_min_distance_is_zero_l931_93159


namespace NUMINAMATH_CALUDE_M_subset_range_l931_93193

def M (a : ℝ) := {x : ℝ | x^2 + 2*(1-a)*x + 3-a ≤ 0}

theorem M_subset_range (a : ℝ) : M a ⊆ Set.Icc 0 3 ↔ -1 ≤ a ∧ a ≤ 18/7 := by sorry

end NUMINAMATH_CALUDE_M_subset_range_l931_93193


namespace NUMINAMATH_CALUDE_polynomial_irreducibility_l931_93188

theorem polynomial_irreducibility (a b c : ℤ) : 
  (0 < |c| ∧ |c| < |b| ∧ |b| < |a|) →
  (∀ x : ℤ, Irreducible (x * (x - a) * (x - b) * (x - c) + 1)) ↔
  (a ≠ 1 ∨ b ≠ 2 ∨ c ≠ 3) ∧ (a ≠ -1 ∨ b ≠ -2 ∨ c ≠ -3) :=
by sorry

end NUMINAMATH_CALUDE_polynomial_irreducibility_l931_93188


namespace NUMINAMATH_CALUDE_tensor_equation_solution_l931_93141

/-- Custom binary operation ⊗ for positive real numbers -/
def tensor (a b : ℝ) : ℝ := a * b + a + b^2

/-- Theorem stating that if 1⊗m = 3, then m = 1 -/
theorem tensor_equation_solution (m : ℝ) (h1 : m > 0) (h2 : tensor 1 m = 3) : m = 1 := by
  sorry

end NUMINAMATH_CALUDE_tensor_equation_solution_l931_93141


namespace NUMINAMATH_CALUDE_l_shape_area_l931_93170

/-- Represents a rectangle with given width and height -/
structure Rectangle where
  width : ℝ
  height : ℝ

/-- Calculates the area of a rectangle -/
def Rectangle.area (r : Rectangle) : ℝ := r.width * r.height

/-- Represents the cut-out position -/
structure CutOutPosition where
  fromRight : ℝ
  fromBottom : ℝ

theorem l_shape_area (large : Rectangle) (cutOut : Rectangle) (pos : CutOutPosition) : 
  large.width = 12 →
  large.height = 7 →
  cutOut.width = 4 →
  cutOut.height = 3 →
  pos.fromRight = large.width / 2 →
  pos.fromBottom = large.height / 2 →
  large.area - cutOut.area = 72 := by
  sorry

end NUMINAMATH_CALUDE_l_shape_area_l931_93170


namespace NUMINAMATH_CALUDE_not_sufficient_not_necessary_condition_l931_93185

theorem not_sufficient_not_necessary_condition (a b : ℝ) :
  ¬(∀ a b : ℝ, a + b > 0 → a * b > 0) ∧ ¬(∀ a b : ℝ, a * b > 0 → a + b > 0) := by
  sorry

end NUMINAMATH_CALUDE_not_sufficient_not_necessary_condition_l931_93185


namespace NUMINAMATH_CALUDE_students_in_both_math_and_science_l931_93129

theorem students_in_both_math_and_science 
  (total : ℕ) 
  (not_math : ℕ) 
  (not_science : ℕ) 
  (not_either : ℕ) 
  (h1 : total = 40) 
  (h2 : not_math = 10) 
  (h3 : not_science = 15) 
  (h4 : not_either = 2) : 
  total - not_math + total - not_science - (total - not_either) = 17 := by
sorry

end NUMINAMATH_CALUDE_students_in_both_math_and_science_l931_93129


namespace NUMINAMATH_CALUDE_milo_run_distance_l931_93160

/-- Milo's running speed in miles per hour -/
def milo_run_speed : ℝ := 3

/-- Milo's skateboard rolling speed in miles per hour -/
def milo_roll_speed : ℝ := milo_run_speed * 2

/-- Cory's wheelchair driving speed in miles per hour -/
def cory_drive_speed : ℝ := 12

/-- Time Milo runs in hours -/
def run_time : ℝ := 2

theorem milo_run_distance : 
  (milo_roll_speed = milo_run_speed * 2) →
  (cory_drive_speed = milo_roll_speed * 2) →
  (cory_drive_speed = 12) →
  (milo_run_speed * run_time = 6) :=
by
  sorry


end NUMINAMATH_CALUDE_milo_run_distance_l931_93160


namespace NUMINAMATH_CALUDE_product_equals_simplified_fraction_l931_93127

/-- The repeating decimal 0.456̅ as a rational number -/
def repeating_decimal : ℚ := 456 / 999

/-- The product of 0.456̅ and 8 -/
def product : ℚ := repeating_decimal * 8

/-- Theorem stating that the product of 0.456̅ and 8 is equal to 1216/333 -/
theorem product_equals_simplified_fraction : product = 1216 / 333 := by
  sorry

end NUMINAMATH_CALUDE_product_equals_simplified_fraction_l931_93127


namespace NUMINAMATH_CALUDE_fourth_root_cube_root_equality_l931_93109

theorem fourth_root_cube_root_equality : 
  (0.000008 : ℝ)^((1/3) * (1/4)) = (2 : ℝ)^(1/4) / (10 : ℝ)^(1/2) :=
sorry

end NUMINAMATH_CALUDE_fourth_root_cube_root_equality_l931_93109


namespace NUMINAMATH_CALUDE_percent_relation_l931_93177

theorem percent_relation (x y : ℝ) (h : 0.2 * (x - y) = 0.14 * (x + y)) :
  y = (3 / 17) * x := by
  sorry

end NUMINAMATH_CALUDE_percent_relation_l931_93177


namespace NUMINAMATH_CALUDE_first_number_equation_l931_93148

theorem first_number_equation : ∃ x : ℝ, 
  x + 17.0005 - 9.1103 = 20.011399999999995 ∧ 
  x = 12.121199999999995 := by sorry

end NUMINAMATH_CALUDE_first_number_equation_l931_93148


namespace NUMINAMATH_CALUDE_max_profit_at_50_l931_93153

/-- Profit function given the price increase x -/
def profit (x : ℕ) : ℤ := -5 * x^2 + 500 * x + 20000

/-- The maximum allowed price increase -/
def max_increase : ℕ := 200

/-- Theorem stating the maximum profit and the price increase that achieves it -/
theorem max_profit_at_50 :
  ∃ (x : ℕ), x ≤ max_increase ∧ 
  profit x = 32500 ∧ 
  ∀ (y : ℕ), y ≤ max_increase → profit y ≤ profit x :=
sorry

end NUMINAMATH_CALUDE_max_profit_at_50_l931_93153


namespace NUMINAMATH_CALUDE_train_travel_distance_l931_93142

/-- Represents the efficiency of a coal-powered train in miles per pound of coal -/
def train_efficiency : ℚ := 5 / 2

/-- Represents the amount of coal remaining in pounds -/
def coal_remaining : ℕ := 160

/-- Calculates the distance a train can travel given its efficiency and remaining coal -/
def distance_traveled (efficiency : ℚ) (coal : ℕ) : ℚ :=
  efficiency * coal

/-- Theorem stating that the train can travel 400 miles before running out of fuel -/
theorem train_travel_distance :
  distance_traveled train_efficiency coal_remaining = 400 := by
  sorry

end NUMINAMATH_CALUDE_train_travel_distance_l931_93142


namespace NUMINAMATH_CALUDE_perpendicular_bisector_of_intersection_points_l931_93128

open Real

/-- Represents a curve in polar coordinates -/
structure PolarCurve where
  equation : ℝ → ℝ → Prop

/-- The first curve: ρ = 2sin θ -/
def C₁ : PolarCurve :=
  ⟨λ ρ θ ↦ ρ = 2 * sin θ⟩

/-- The second curve: ρ = 2cos θ -/
def C₂ : PolarCurve :=
  ⟨λ ρ θ ↦ ρ = 2 * cos θ⟩

/-- Represents a point in polar coordinates -/
structure PolarPoint where
  ρ : ℝ
  θ : ℝ

/-- Finds the intersection points of two polar curves -/
def intersectionPoints (c₁ c₂ : PolarCurve) : Set PolarPoint :=
  {p | c₁.equation p.ρ p.θ ∧ c₂.equation p.ρ p.θ}

/-- The perpendicular bisector equation -/
def perpendicularBisector (ρ θ : ℝ) : Prop :=
  ρ = 1 / (sin θ + cos θ)

theorem perpendicular_bisector_of_intersection_points :
  ∀ (A B : PolarPoint), A ∈ intersectionPoints C₁ C₂ → B ∈ intersectionPoints C₁ C₂ → A ≠ B →
  ∀ ρ θ, perpendicularBisector ρ θ ↔ 
    (∃ t, ρ * cos θ = A.ρ * cos A.θ + t * (B.ρ * cos B.θ - A.ρ * cos A.θ) ∧
          ρ * sin θ = A.ρ * sin A.θ + t * (B.ρ * sin B.θ - A.ρ * sin A.θ) ∧
          0 < t ∧ t < 1) :=
sorry

end NUMINAMATH_CALUDE_perpendicular_bisector_of_intersection_points_l931_93128


namespace NUMINAMATH_CALUDE_shaded_area_semicircles_l931_93136

/-- The area of the shaded region in the given semicircle configuration -/
theorem shaded_area_semicircles (r_ADB r_BEC : ℝ) (h_ADB : r_ADB = 2) (h_BEC : r_BEC = 3) : 
  let r_DFE := (r_ADB + r_BEC) / 2
  (π * r_ADB^2 / 2 + π * r_BEC^2 / 2) - (π * r_DFE^2 / 2) = 3.375 * π := by
  sorry

end NUMINAMATH_CALUDE_shaded_area_semicircles_l931_93136


namespace NUMINAMATH_CALUDE_remainder_problem_l931_93122

theorem remainder_problem (n : ℤ) (h : n % 22 = 12) : (2 * n) % 11 = 2 := by
  sorry

end NUMINAMATH_CALUDE_remainder_problem_l931_93122


namespace NUMINAMATH_CALUDE_train_speed_l931_93181

/-- The speed of a train given its length and time to cross a point -/
theorem train_speed (length : ℝ) (time : ℝ) (h1 : length = 560) (h2 : time = 16) :
  length / time = 35 := by
  sorry

end NUMINAMATH_CALUDE_train_speed_l931_93181


namespace NUMINAMATH_CALUDE_ice_cream_stacking_problem_l931_93165

/-- The number of permutations of n distinct objects -/
def permutations (n : ℕ) : ℕ := Nat.factorial n

/-- The problem statement -/
theorem ice_cream_stacking_problem :
  permutations 5 = 120 := by sorry

end NUMINAMATH_CALUDE_ice_cream_stacking_problem_l931_93165


namespace NUMINAMATH_CALUDE_tylers_eggs_count_l931_93182

/-- The number of eggs required for a cake serving a given number of people -/
def eggs_required (people : ℕ) : ℕ := 2 * (people / 4)

/-- The number of eggs Tyler has in the fridge -/
def tylers_eggs : ℕ := eggs_required 8 - 1

theorem tylers_eggs_count : tylers_eggs = 3 := by
  sorry

end NUMINAMATH_CALUDE_tylers_eggs_count_l931_93182


namespace NUMINAMATH_CALUDE_blue_socks_count_l931_93145

/-- Represents the number of pairs of socks Luis bought -/
structure SockPurchase where
  red : ℕ
  blue : ℕ

/-- Represents the cost of socks in dollars -/
structure SockCost where
  red : ℕ
  blue : ℕ

/-- Calculates the total cost of the sock purchase -/
def totalCost (purchase : SockPurchase) (cost : SockCost) : ℕ :=
  purchase.red * cost.red + purchase.blue * cost.blue

theorem blue_socks_count (purchase : SockPurchase) (cost : SockCost) :
  purchase.red = 4 →
  cost.red = 3 →
  cost.blue = 5 →
  totalCost purchase cost = 42 →
  purchase.blue = 6 := by
  sorry

end NUMINAMATH_CALUDE_blue_socks_count_l931_93145


namespace NUMINAMATH_CALUDE_train_speed_l931_93110

/-- The speed of a train given the time it takes to pass a pole and cross a stationary train -/
theorem train_speed
  (pole_pass_time : ℝ)
  (stationary_train_length : ℝ)
  (crossing_time : ℝ)
  (h1 : pole_pass_time = 5)
  (h2 : stationary_train_length = 360)
  (h3 : crossing_time = 25) :
  let train_speed := stationary_train_length / (crossing_time - pole_pass_time)
  train_speed = 18 := by
  sorry

end NUMINAMATH_CALUDE_train_speed_l931_93110


namespace NUMINAMATH_CALUDE_skaters_meeting_time_l931_93108

/-- The time it takes for two skaters to meet on a circular rink -/
theorem skaters_meeting_time 
  (rink_circumference : ℝ) 
  (speed_skater1 : ℝ) 
  (speed_skater2 : ℝ) 
  (h1 : rink_circumference = 3000) 
  (h2 : speed_skater1 = 100) 
  (h3 : speed_skater2 = 150) : 
  rink_circumference / (speed_skater1 + speed_skater2) = 12 := by
  sorry

#check skaters_meeting_time

end NUMINAMATH_CALUDE_skaters_meeting_time_l931_93108


namespace NUMINAMATH_CALUDE_binomial_12_6_l931_93198

theorem binomial_12_6 : Nat.choose 12 6 = 924 := by
  sorry

end NUMINAMATH_CALUDE_binomial_12_6_l931_93198


namespace NUMINAMATH_CALUDE_factor_polynomial_l931_93132

theorem factor_polynomial (x : ℝ) : 98 * x^7 - 266 * x^13 = 14 * x^7 * (7 - 19 * x^6) := by
  sorry

end NUMINAMATH_CALUDE_factor_polynomial_l931_93132


namespace NUMINAMATH_CALUDE_price_before_increase_l931_93179

/-- Proves that the total price before the increase was 25 pounds, given the original prices and percentage increases. -/
theorem price_before_increase 
  (candy_price : ℝ) 
  (soda_price : ℝ) 
  (candy_increase : ℝ) 
  (soda_increase : ℝ) 
  (h1 : candy_price = 10)
  (h2 : soda_price = 15)
  (h3 : candy_increase = 0.25)
  (h4 : soda_increase = 0.50) :
  candy_price + soda_price = 25 := by
  sorry

#check price_before_increase

end NUMINAMATH_CALUDE_price_before_increase_l931_93179


namespace NUMINAMATH_CALUDE_not_multiple_of_five_l931_93138

theorem not_multiple_of_five : ¬ (∃ k : ℤ, (2015^2 / 5^2) = 5 * k) ∧
  (∃ k : ℤ, (2019^2 - 2014^2) = 5 * k) ∧
  (∃ k : ℤ, (2019^2 * 10^2) = 5 * k) ∧
  (∃ k : ℤ, (2020^2 / 101^2) = 5 * k) ∧
  (∃ k : ℤ, (2010^2 - 2005^2) = 5 * k) :=
by sorry

#check not_multiple_of_five

end NUMINAMATH_CALUDE_not_multiple_of_five_l931_93138


namespace NUMINAMATH_CALUDE_unique_solution_exists_l931_93158

theorem unique_solution_exists (y : ℝ) (h : y > 0) :
  ∃! x : ℝ, (2 ^ (4 * x + 2)) * (4 ^ (2 * x + 3)) = 8 ^ (3 * x + 4) * y :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_exists_l931_93158


namespace NUMINAMATH_CALUDE_angle_of_inclination_slope_one_l931_93163

/-- The angle of inclination of a line with slope 1 in the Cartesian coordinate system is π/4 -/
theorem angle_of_inclination_slope_one :
  let line := {(x, y) : ℝ × ℝ | x - y - 3 = 0}
  let slope : ℝ := 1
  let angle_of_inclination := Real.arctan slope
  angle_of_inclination = π / 4 := by
  sorry

end NUMINAMATH_CALUDE_angle_of_inclination_slope_one_l931_93163


namespace NUMINAMATH_CALUDE_units_digit_of_product_l931_93137

def product : ℕ := 1 * 3 * 5 * 79 * 97 * 113

theorem units_digit_of_product :
  (product % 10) = 5 := by sorry

end NUMINAMATH_CALUDE_units_digit_of_product_l931_93137


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l931_93167

/-- A geometric sequence with the given properties -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_sum (a : ℕ → ℝ) :
  GeometricSequence a → a 3 + a 5 = 20 → a 4 = 8 → a 2 + a 6 = 34 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l931_93167


namespace NUMINAMATH_CALUDE_max_value_of_g_l931_93155

/-- The function g(x) = 4x - x^4 -/
def g (x : ℝ) : ℝ := 4*x - x^4

/-- The theorem stating that the maximum value of g(x) on [0, 2] is 3 -/
theorem max_value_of_g :
  ∃ (c : ℝ), c ∈ Set.Icc 0 2 ∧ 
  (∀ x, x ∈ Set.Icc 0 2 → g x ≤ g c) ∧
  g c = 3 :=
sorry

end NUMINAMATH_CALUDE_max_value_of_g_l931_93155


namespace NUMINAMATH_CALUDE_a_range_l931_93172

theorem a_range (a : ℝ) : 
  (∀ x ∈ Set.Icc 1 2, |2^x - a| < |5 - 2^x|) → 
  3 < a ∧ a < 5 := by
sorry

end NUMINAMATH_CALUDE_a_range_l931_93172


namespace NUMINAMATH_CALUDE_elder_age_l931_93194

/-- The age difference between two people -/
def age_difference : ℕ := 20

/-- The number of years ago when the elder was 5 times as old as the younger -/
def years_ago : ℕ := 8

/-- The ratio of elder's age to younger's age in the past -/
def age_ratio : ℕ := 5

theorem elder_age (younger_age elder_age : ℕ) : 
  (elder_age = younger_age + age_difference) → 
  (elder_age - years_ago = age_ratio * (younger_age - years_ago)) →
  elder_age = 33 := by
sorry

end NUMINAMATH_CALUDE_elder_age_l931_93194


namespace NUMINAMATH_CALUDE_sum_of_reciprocals_squared_l931_93152

theorem sum_of_reciprocals_squared (a b c d : ℝ) : 
  a = Real.sqrt 5 + Real.sqrt 7 + Real.sqrt 35 →
  b = -Real.sqrt 5 + Real.sqrt 7 + Real.sqrt 35 →
  c = Real.sqrt 5 - Real.sqrt 7 + Real.sqrt 35 →
  d = -Real.sqrt 5 - Real.sqrt 7 + Real.sqrt 35 →
  (1/a + 1/b + 1/c + 1/d)^2 = 560/83521 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_reciprocals_squared_l931_93152


namespace NUMINAMATH_CALUDE_students_playing_both_sports_l931_93125

def total_students : ℕ := 470
def football_players : ℕ := 325
def cricket_players : ℕ := 175
def neither_players : ℕ := 50

theorem students_playing_both_sports : ℕ := by
  sorry

#check students_playing_both_sports = 80

end NUMINAMATH_CALUDE_students_playing_both_sports_l931_93125


namespace NUMINAMATH_CALUDE_min_perimeter_52_l931_93116

/-- Represents the side lengths of the squares in the rectangle --/
structure SquareSides where
  a : ℕ
  b : ℕ

/-- Calculates the perimeter of a rectangle given its length and width --/
def rectangle_perimeter (length width : ℕ) : ℕ :=
  2 * (length + width)

/-- Represents the configuration of squares in the rectangle --/
def square_configuration (sides : SquareSides) : Prop :=
  ∃ (left_column middle_column right_column bottom_row : ℕ),
    left_column = 2 * sides.a + sides.b ∧
    middle_column = 3 * sides.a + sides.b ∧
    right_column = 12 * sides.a - 2 * sides.b ∧
    bottom_row = 8 * sides.a - sides.b ∧
    left_column > 0 ∧ middle_column > 0 ∧ right_column > 0 ∧ bottom_row > 0

theorem min_perimeter_52 :
  ∀ (sides : SquareSides),
    square_configuration sides →
    ∀ (length width : ℕ),
      length = 2 * sides.a + sides.b + 3 * sides.a + sides.b + 12 * sides.a - 2 * sides.b →
      width = 2 * sides.a + sides.b + 8 * sides.a - sides.b →
      rectangle_perimeter length width ≥ 52 :=
sorry

end NUMINAMATH_CALUDE_min_perimeter_52_l931_93116


namespace NUMINAMATH_CALUDE_table_formula_proof_l931_93101

theorem table_formula_proof : 
  (∀ (x y : ℕ), (x = 1 ∧ y = 3) ∨ (x = 2 ∧ y = 7) ∨ (x = 3 ∧ y = 13) ∨ 
   (x = 4 ∧ y = 21) ∨ (x = 5 ∧ y = 31) → y = x^2 + x + 1) :=
by sorry

end NUMINAMATH_CALUDE_table_formula_proof_l931_93101


namespace NUMINAMATH_CALUDE_percentage_of_160_l931_93184

theorem percentage_of_160 : (3 / 8 : ℚ) / 100 * 160 = 0.6 := by
  sorry

end NUMINAMATH_CALUDE_percentage_of_160_l931_93184


namespace NUMINAMATH_CALUDE_cookie_recipe_average_l931_93176

/-- Represents the cookie recipe and calculates the average pieces per cookie. -/
def average_pieces_per_cookie (total_cookies : ℕ) (chocolate_chips : ℕ) : ℚ :=
  let mms : ℕ := chocolate_chips / 3
  let white_chips : ℕ := mms / 2
  let raisins : ℕ := white_chips * 2
  let total_pieces : ℕ := chocolate_chips + mms + white_chips + raisins
  (total_pieces : ℚ) / total_cookies

/-- Theorem stating that the average pieces per cookie is 4.125 given the specified recipe. -/
theorem cookie_recipe_average :
  average_pieces_per_cookie 48 108 = 4.125 := by
  sorry


end NUMINAMATH_CALUDE_cookie_recipe_average_l931_93176


namespace NUMINAMATH_CALUDE_min_value_quadratic_l931_93154

theorem min_value_quadratic (x y : ℝ) : x^2 + x*y + y^2 + 7 ≥ 7 := by
  sorry

end NUMINAMATH_CALUDE_min_value_quadratic_l931_93154


namespace NUMINAMATH_CALUDE_sequence_properties_l931_93196

def sequence_a (n : ℕ) : ℚ := sorry

def sequence_S (n : ℕ) : ℚ := sorry

axiom a_1 : sequence_a 1 = 3

axiom S_def : ∀ n : ℕ, n ≥ 2 → 2 * sequence_a n = sequence_S n * sequence_S (n - 1)

theorem sequence_properties :
  (∃ d : ℚ, ∀ n : ℕ, n ≥ 1 → (1 / sequence_S (n + 1) - 1 / sequence_S n = d)) ∧
  (∀ n : ℕ, n ≥ 2 → sequence_a n = 18 / ((5 - 3 * n) * (8 - 3 * n))) :=
sorry

end NUMINAMATH_CALUDE_sequence_properties_l931_93196


namespace NUMINAMATH_CALUDE_truck_fuel_efficiency_l931_93189

theorem truck_fuel_efficiency 
  (distance : ℝ) 
  (current_gas : ℝ) 
  (additional_gas : ℝ) 
  (h1 : distance = 90) 
  (h2 : current_gas = 12) 
  (h3 : additional_gas = 18) : 
  distance / (current_gas + additional_gas) = 3 := by
sorry

end NUMINAMATH_CALUDE_truck_fuel_efficiency_l931_93189


namespace NUMINAMATH_CALUDE_smallest_fourth_power_b_l931_93173

theorem smallest_fourth_power_b : ∃ (n : ℕ), 
  (7 + 7 * 18 + 7 * 18^2 = n^4) ∧ 
  (∀ (b : ℕ), b > 0 → b < 18 → ¬∃ (m : ℕ), 7 + 7 * b + 7 * b^2 = m^4) := by
  sorry

end NUMINAMATH_CALUDE_smallest_fourth_power_b_l931_93173


namespace NUMINAMATH_CALUDE_sector_area_l931_93146

/-- The area of a circular sector with central angle π/3 and radius 3 is 3π/2 -/
theorem sector_area (θ : Real) (r : Real) (h1 : θ = π / 3) (h2 : r = 3) :
  (1 / 2) * θ * r^2 = 3 * π / 2 := by
  sorry

end NUMINAMATH_CALUDE_sector_area_l931_93146


namespace NUMINAMATH_CALUDE_bus_train_speed_ratio_l931_93195

/-- The fraction of the speed of a bus compared to the speed of a train -/
theorem bus_train_speed_ratio :
  -- The ratio between the speed of a train and a car
  ∀ (train_speed car_speed : ℝ),
  train_speed / car_speed = 16 / 15 →
  -- A bus covered 320 km in 5 hours
  ∀ (bus_speed : ℝ),
  bus_speed * 5 = 320 →
  -- The car will cover 525 km in 7 hours
  car_speed * 7 = 525 →
  -- The fraction of the speed of the bus compared to the speed of the train
  bus_speed / train_speed = 4 / 5 := by
sorry

end NUMINAMATH_CALUDE_bus_train_speed_ratio_l931_93195


namespace NUMINAMATH_CALUDE_ana_beto_game_l931_93112

def is_valid_sequence (seq : List Int) : Prop :=
  seq.length = 2016 ∧ (seq.count 1 = 1008) ∧ (seq.count (-1) = 1008)

def block_sum_squares (blocks : List (List Int)) : Int :=
  (blocks.map (λ block => (block.sum)^2)).sum

theorem ana_beto_game (N : Nat) :
  (∃ (seq : List Int) (blocks : List (List Int)),
    is_valid_sequence seq ∧
    seq = blocks.join ∧
    block_sum_squares blocks = N) ↔
  (N % 2 = 0 ∧ N ≤ 2016) :=
sorry

end NUMINAMATH_CALUDE_ana_beto_game_l931_93112


namespace NUMINAMATH_CALUDE_sqrt_fraction_simplification_l931_93121

theorem sqrt_fraction_simplification :
  Real.sqrt (25 / 36 + 16 / 9) = Real.sqrt 89 / 6 :=
by sorry

end NUMINAMATH_CALUDE_sqrt_fraction_simplification_l931_93121


namespace NUMINAMATH_CALUDE_biology_homework_pages_l931_93126

/-- The number of pages of math homework -/
def math_pages : ℕ := 8

/-- The total number of pages of math and biology homework -/
def total_math_biology_pages : ℕ := 11

/-- The number of pages of biology homework -/
def biology_pages : ℕ := total_math_biology_pages - math_pages

theorem biology_homework_pages : biology_pages = 3 := by
  sorry

end NUMINAMATH_CALUDE_biology_homework_pages_l931_93126


namespace NUMINAMATH_CALUDE_sqrt_product_simplification_l931_93191

theorem sqrt_product_simplification (x : ℝ) (h : x ≥ 0) :
  Real.sqrt (3 * x) * Real.sqrt (2 * x^2) = Real.sqrt 6 * x^(3/2) :=
sorry

end NUMINAMATH_CALUDE_sqrt_product_simplification_l931_93191


namespace NUMINAMATH_CALUDE_grain_movement_representation_l931_93149

-- Define the type for grain movement
inductive GrainMovement
  | arrival
  | departure

-- Define a function to represent the sign of grain movement
def signOfMovement (g : GrainMovement) : Int :=
  match g with
  | GrainMovement.arrival => 1
  | GrainMovement.departure => -1

-- Define the theorem
theorem grain_movement_representation :
  ∀ (quantity : ℕ),
  (signOfMovement GrainMovement.arrival * quantity = 30) →
  (signOfMovement GrainMovement.departure * quantity = -30) :=
by
  sorry


end NUMINAMATH_CALUDE_grain_movement_representation_l931_93149


namespace NUMINAMATH_CALUDE_rectangle_area_l931_93135

/-- Given a rectangle where the sum of width and length is half of 28, and the width is 6,
    prove that its area is 48 square units. -/
theorem rectangle_area (w l : ℝ) : w = 6 → w + l = 28 / 2 → w * l = 48 := by sorry

end NUMINAMATH_CALUDE_rectangle_area_l931_93135


namespace NUMINAMATH_CALUDE_last_disc_is_blue_l931_93178

/-- Represents the color of a disc --/
inductive Color
  | Red
  | Blue
  | Yellow

/-- Represents the state of the bag --/
structure BagState where
  red : ℕ
  blue : ℕ
  yellow : ℕ

/-- Initial state of the bag --/
def initial_state : BagState :=
  { red := 7, blue := 8, yellow := 9 }

/-- Represents the rules for drawing and replacing discs --/
def draw_and_replace (state : BagState) : BagState :=
  sorry

/-- Represents the process of repeatedly drawing and replacing discs until the end condition is met --/
def process (state : BagState) : BagState :=
  sorry

/-- Theorem stating that the last remaining disc(s) will be blue --/
theorem last_disc_is_blue :
  ∃ (final_state : BagState), process initial_state = final_state ∧ 
  final_state.blue > 0 ∧ final_state.red = 0 ∧ final_state.yellow = 0 :=
sorry

end NUMINAMATH_CALUDE_last_disc_is_blue_l931_93178


namespace NUMINAMATH_CALUDE_num_valid_codes_correct_num_valid_codes_positive_l931_93147

/-- The number of possible 5-digit codes where no digit is used more than twice. -/
def num_valid_codes : ℕ := 102240

/-- A function that calculates the number of valid 5-digit codes. -/
def calculate_valid_codes : ℕ :=
  let all_different := 10 * 9 * 8 * 7 * 6
  let one_digit_repeated := 10 * (5 * 4 / 2) * 9 * 8 * 7
  let two_digits_repeated := 10 * 9 * (5 * 4 / 2) * (3 * 2 / 2) * 8
  all_different + one_digit_repeated + two_digits_repeated

/-- Theorem stating that the number of valid codes is correct. -/
theorem num_valid_codes_correct : calculate_valid_codes = num_valid_codes := by
  sorry

/-- Theorem stating that the calculated number of valid codes is positive. -/
theorem num_valid_codes_positive : 0 < num_valid_codes := by
  sorry

end NUMINAMATH_CALUDE_num_valid_codes_correct_num_valid_codes_positive_l931_93147


namespace NUMINAMATH_CALUDE_simplify_square_roots_l931_93100

theorem simplify_square_roots : 
  (Real.sqrt 288 / Real.sqrt 32) - (Real.sqrt 242 / Real.sqrt 121) = 3 - Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_square_roots_l931_93100


namespace NUMINAMATH_CALUDE_cos_270_degrees_l931_93107

theorem cos_270_degrees : Real.cos (270 * π / 180) = 0 := by
  sorry

end NUMINAMATH_CALUDE_cos_270_degrees_l931_93107


namespace NUMINAMATH_CALUDE_product_of_logs_l931_93143

theorem product_of_logs (a b : ℕ+) : 
  (b - a = 1560) →
  (Real.log b / Real.log a = 3) →
  (a + b : ℕ) = 1740 := by sorry

end NUMINAMATH_CALUDE_product_of_logs_l931_93143
