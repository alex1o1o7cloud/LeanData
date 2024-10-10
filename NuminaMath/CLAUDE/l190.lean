import Mathlib

namespace arithmetic_sequence_middle_term_l190_19003

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n m : ℕ, a (n + m) - a n = m * (a (n + 1) - a n)

theorem arithmetic_sequence_middle_term 
  (a : ℕ → ℝ) 
  (h_arith : arithmetic_sequence a) 
  (h_sum : a 3 + a 5 = 16) : 
  a 4 = 8 := by
sorry

end arithmetic_sequence_middle_term_l190_19003


namespace find_other_number_l190_19097

theorem find_other_number (a b : ℕ+) (hcf lcm : ℕ+) : 
  Nat.gcd a.val b.val = hcf.val →
  Nat.lcm a.val b.val = lcm.val →
  hcf * lcm = a * b →
  a = 154 →
  hcf = 14 →
  lcm = 396 →
  b = 36 := by
sorry

end find_other_number_l190_19097


namespace simplify_expression_l190_19009

theorem simplify_expression (x : ℝ) : 4*x + 6*x^3 + 8 - (3 - 6*x^3 - 4*x) = 12*x^3 + 8*x + 5 := by
  sorry

end simplify_expression_l190_19009


namespace hexagon_centers_square_area_ratio_l190_19091

/-- Square represents a square in 2D space -/
structure Square where
  side : ℝ
  center : ℝ × ℝ

/-- RegularHexagon represents a regular hexagon in 2D space -/
structure RegularHexagon where
  side : ℝ
  center : ℝ × ℝ

/-- Configuration represents the problem setup -/
structure Configuration where
  square : Square
  hexagons : Fin 4 → RegularHexagon

/-- Defines the specific configuration described in the problem -/
def problem_configuration : Configuration :=
  sorry

/-- Calculate the area of a square given its side length -/
def square_area (s : Square) : ℝ :=
  s.side * s.side

/-- Calculate the area of the square formed by the centers of the hexagons -/
def hexagon_centers_square_area (c : Configuration) : ℝ :=
  sorry

/-- The main theorem to be proved -/
theorem hexagon_centers_square_area_ratio (c : Configuration) :
  hexagon_centers_square_area c / square_area c.square = 4.5 := by
  sorry

end hexagon_centers_square_area_ratio_l190_19091


namespace cone_rolling_ratio_l190_19010

/-- 
Theorem: For a right circular cone with base radius r and height h, 
if the cone makes 23 complete rotations when rolled on its side, 
then h/r = 4√33.
-/
theorem cone_rolling_ratio (r h : ℝ) (hr : r > 0) (hh : h > 0) : 
  (2 * Real.pi * Real.sqrt (r^2 + h^2) = 46 * Real.pi * r) → 
  (h / r = 4 * Real.sqrt 33) := by
sorry

end cone_rolling_ratio_l190_19010


namespace unique_integer_satisfying_conditions_l190_19032

theorem unique_integer_satisfying_conditions (x : ℤ) 
  (h1 : 0 < x ∧ x < 7)
  (h2 : 0 < x ∧ x < 15)
  (h3 : -1 < x ∧ x < 5)
  (h4 : 0 < x ∧ x < 3)
  (h5 : x + 2 < 4) : 
  x = 1 := by
sorry

end unique_integer_satisfying_conditions_l190_19032


namespace sum_squares_consecutive_integers_l190_19047

theorem sum_squares_consecutive_integers (a : ℤ) :
  let S := (a - 2)^2 + (a - 1)^2 + a^2 + (a + 1)^2 + (a + 2)^2
  ∃ k : ℤ, S = 5 * k ∧ ¬∃ m : ℤ, S = 25 * m :=
by sorry

end sum_squares_consecutive_integers_l190_19047


namespace female_students_count_l190_19042

theorem female_students_count (total_average : ℝ) (male_count : ℕ) (male_average : ℝ) (female_average : ℝ)
  (h1 : total_average = 90)
  (h2 : male_count = 8)
  (h3 : male_average = 83)
  (h4 : female_average = 92) :
  ∃ (female_count : ℕ),
    female_count = 28 ∧
    (male_count * male_average + female_count * female_average) / (male_count + female_count) = total_average :=
by sorry

end female_students_count_l190_19042


namespace letter_value_puzzle_l190_19035

theorem letter_value_puzzle (L E A D : ℤ) : 
  L = 15 →
  L + E + A + D = 41 →
  D + E + A + L = 45 →
  A + D + D + E + D = 53 →
  D = 4 :=
by
  sorry

end letter_value_puzzle_l190_19035


namespace negation_of_existential_absolute_value_l190_19081

theorem negation_of_existential_absolute_value (x : ℝ) :
  (¬ ∃ x : ℝ, |x| ≤ 2) ↔ (∀ x : ℝ, |x| > 2) := by
sorry

end negation_of_existential_absolute_value_l190_19081


namespace angle_bisector_ratio_l190_19048

-- Define the triangle XYZ
structure Triangle :=
  (X Y Z : ℝ × ℝ)

-- Define the angle bisectors
def angleBisector (T : Triangle) (vertex : ℝ × ℝ) (side1 side2 : ℝ × ℝ) : ℝ × ℝ := sorry

-- Define the intersection point Q
def intersectionPoint (T : Triangle) : ℝ × ℝ := sorry

-- Define the distance between two points
def distance (p1 p2 : ℝ × ℝ) : ℝ := sorry

-- Theorem statement
theorem angle_bisector_ratio (T : Triangle) :
  let X := T.X
  let Y := T.Y
  let Z := T.Z
  let U := angleBisector T X Y Z
  let V := angleBisector T Y X Z
  let Q := intersectionPoint T
  distance X Y = 8 ∧ distance X Z = 6 ∧ distance Y Z = 4 →
  distance Y Q / distance Q V = 2 := by sorry

end angle_bisector_ratio_l190_19048


namespace sampling_properties_l190_19007

/-- Represents a club with male and female members -/
structure Club where
  male_members : ℕ
  female_members : ℕ

/-- Represents a sample drawn from the club -/
structure Sample where
  size : ℕ
  males_selected : ℕ
  females_selected : ℕ

/-- The probability of selecting a male from the club -/
def prob_select_male (c : Club) (s : Sample) : ℚ :=
  s.males_selected / c.male_members

/-- The probability of selecting a female from the club -/
def prob_select_female (c : Club) (s : Sample) : ℚ :=
  s.females_selected / c.female_members

/-- Theorem about the sampling properties of a specific club and sample -/
theorem sampling_properties (c : Club) (s : Sample) 
    (h_male : c.male_members = 30)
    (h_female : c.female_members = 20)
    (h_sample_size : s.size = 5)
    (h_males_selected : s.males_selected = 2)
    (h_females_selected : s.females_selected = 3) :
  (∃ (sampling_method : String), sampling_method = "random") ∧
  (¬ ∃ (sampling_method : String), sampling_method = "stratified") ∧
  prob_select_male c s < prob_select_female c s :=
by sorry

end sampling_properties_l190_19007


namespace quadratic_roots_problem_l190_19083

theorem quadratic_roots_problem (a b m p r : ℝ) : 
  (a^2 - m*a + 3 = 0) →
  (b^2 - m*b + 3 = 0) →
  ((a + 2/b)^2 - p*(a + 2/b) + r = 0) →
  ((b + 2/a)^2 - p*(b + 2/a) + r = 0) →
  r = 25/3 := by
sorry

end quadratic_roots_problem_l190_19083


namespace quadratic_function_comparison_l190_19036

theorem quadratic_function_comparison (y₁ y₂ : ℝ) : 
  ((-1 : ℝ)^2 - 2*(-1) = y₁) → 
  ((2 : ℝ)^2 - 2*2 = y₂) → 
  y₁ > y₂ := by sorry

end quadratic_function_comparison_l190_19036


namespace fidos_yard_area_fraction_l190_19084

theorem fidos_yard_area_fraction :
  ∀ (s : ℝ), s > 0 →
  (π * s^2) / (4 * s^2) = π / 4 := by
  sorry

end fidos_yard_area_fraction_l190_19084


namespace matching_pair_probability_is_0_5226_l190_19090

/-- Represents the types of shoes in the warehouse -/
inductive ShoeType
  | Sneaker
  | Boot
  | DressShoe

/-- Represents the shoe warehouse inventory -/
structure ShoeWarehouse where
  sneakers : ℕ
  boots : ℕ
  dressShoes : ℕ
  sneakerProb : ℝ
  bootProb : ℝ
  dressShoeProb : ℝ

/-- Calculates the probability of selecting a matching pair of shoes -/
def matchingPairProbability (warehouse : ShoeWarehouse) : ℝ :=
  let sneakerProb := warehouse.sneakers * warehouse.sneakerProb * (warehouse.sneakers - 1) * warehouse.sneakerProb
  let bootProb := warehouse.boots * warehouse.bootProb * (warehouse.boots - 1) * warehouse.bootProb
  let dressShoeProb := warehouse.dressShoes * warehouse.dressShoeProb * (warehouse.dressShoes - 1) * warehouse.dressShoeProb
  sneakerProb + bootProb + dressShoeProb

/-- Theorem stating the probability of selecting a matching pair of shoes -/
theorem matching_pair_probability_is_0_5226 :
  let warehouse : ShoeWarehouse := {
    sneakers := 12,
    boots := 15,
    dressShoes := 18,
    sneakerProb := 0.04,
    bootProb := 0.03,
    dressShoeProb := 0.02
  }
  matchingPairProbability warehouse = 0.5226 := by sorry

end matching_pair_probability_is_0_5226_l190_19090


namespace simplify_radical_expression_l190_19057

theorem simplify_radical_expression :
  ∃ (a b c : ℕ+), 
    (((Real.sqrt 2 - 1) ^ (2 - Real.sqrt 3)) / ((Real.sqrt 2 + 1) ^ (2 + Real.sqrt 3)) = 
     (3 + 2 * Real.sqrt 2) ^ Real.sqrt 3) ∧
    (∀ (p : ℕ), Nat.Prime p → ¬(p ^ 2 ∣ c.val)) :=
by sorry

end simplify_radical_expression_l190_19057


namespace speeding_proof_l190_19065

theorem speeding_proof (distance : ℝ) (time : ℝ) (speed_limit : ℝ)
  (h1 : distance = 165)
  (h2 : time = 2)
  (h3 : speed_limit = 80)
  : ∃ t : ℝ, 0 ≤ t ∧ t ≤ time ∧ (distance / time > speed_limit) :=
by
  sorry

#check speeding_proof

end speeding_proof_l190_19065


namespace piggy_bank_equality_days_l190_19070

def minjoo_initial : ℕ := 12000
def siwoo_initial : ℕ := 4000
def minjoo_daily : ℕ := 300
def siwoo_daily : ℕ := 500

theorem piggy_bank_equality_days : 
  ∃ d : ℕ, d = 40 ∧ 
  minjoo_initial + d * minjoo_daily = siwoo_initial + d * siwoo_daily :=
sorry

end piggy_bank_equality_days_l190_19070


namespace open_box_volume_formula_l190_19060

/-- Represents the volume of an open box constructed from a rectangular metal sheet. -/
def boxVolume (sheetLength sheetWidth x : ℝ) : ℝ :=
  (sheetLength - 2*x) * (sheetWidth - 2*x) * x

theorem open_box_volume_formula (x : ℝ) 
  (h1 : x > 0) 
  (h2 : x < 10) : 
  boxVolume 30 20 x = 600*x - 100*x^2 + 4*x^3 := by
  sorry

end open_box_volume_formula_l190_19060


namespace parallel_lines_and_not_always_parallel_planes_l190_19096

-- Define the line equations
def line1 (a x y : ℝ) : Prop := a * x + 3 * y + 1 = 0
def line2 (a x y : ℝ) : Prop := 2 * x + (a + 1) * y + 1 = 0

-- Define parallel lines
def parallel (a : ℝ) : Prop := ∀ x y, line1 a x y ↔ line2 a x y

-- Define a plane
def Plane : Type := ℝ × ℝ × ℝ

-- Define a point in 3D space
def Point : Type := ℝ × ℝ × ℝ

-- Define distance between a point and a plane
def distance (p : Point) (plane : Plane) : ℝ := sorry

-- Define non-collinear points
def nonCollinear (p1 p2 p3 : Point) : Prop := sorry

-- Define parallel planes
def parallelPlanes (α β : Plane) : Prop := sorry

-- Statement of the theorem
theorem parallel_lines_and_not_always_parallel_planes :
  (∀ a, parallel a ↔ a = -3) ∧
  ¬(∀ α β : Plane, ∀ p1 p2 p3 : Point,
    nonCollinear p1 p2 p3 →
    distance p1 β = distance p2 β ∧ distance p2 β = distance p3 β →
    parallelPlanes α β) := by sorry

end parallel_lines_and_not_always_parallel_planes_l190_19096


namespace abs_neg_one_third_l190_19039

theorem abs_neg_one_third : |(-1 : ℚ) / 3| = 1 / 3 := by
  sorry

end abs_neg_one_third_l190_19039


namespace value_exceeds_initial_price_min_avg_value_l190_19095

-- Define the value of M at the beginning of the nth year
def value (n : ℕ) : ℚ :=
  if n ≤ 3 then
    20 * (1/2)^(n-1)
  else
    4 * n - 7

-- Define the sum of values for the first n years
def sum_values (n : ℕ) : ℚ :=
  if n ≤ 3 then
    40 - 5 * 2^(3-n)
  else
    2 * n^2 - 5 * n + 32

-- Define the average value over n years
def avg_value (n : ℕ) : ℚ :=
  sum_values n / n

-- Theorem 1: Value exceeds initial price at the beginning of the 7th year
theorem value_exceeds_initial_price :
  ∀ k < 7, value k ≤ 20 ∧ value 7 > 20 :=
sorry

-- Theorem 2: Minimum average value is 11, occurring at n = 4
theorem min_avg_value :
  ∀ n : ℕ, n ≥ 1 → avg_value n ≥ 11 ∧ avg_value 4 = 11 :=
sorry

end value_exceeds_initial_price_min_avg_value_l190_19095


namespace soccer_committee_count_l190_19005

/-- The number of teams in the soccer league -/
def num_teams : ℕ := 5

/-- The number of members in each team -/
def team_size : ℕ := 7

/-- The number of members selected from the host team -/
def host_selection : ℕ := 4

/-- The number of members selected from each non-host team -/
def non_host_selection : ℕ := 2

/-- The total number of possible organizing committees -/
def total_committees : ℕ := 34134175

theorem soccer_committee_count :
  (num_teams * (Nat.choose team_size host_selection) *
   (Nat.choose team_size non_host_selection ^ (num_teams - 1))) = total_committees := by
  sorry

end soccer_committee_count_l190_19005


namespace demand_analysis_l190_19077

def f (x : ℕ) : ℚ := (1 / 150) * x * (x + 1) * (35 - 2 * x)

def g (x : ℕ) : ℚ := (1 / 25) * x * (12 - x)

theorem demand_analysis (x : ℕ) (h : x ≤ 12) :
  -- 1. The demand in the x-th month
  g x = f x - f (x - 1) ∧
  -- 2. The maximum monthly demand occurs when x = 6 and is equal to 36/25
  (∀ y : ℕ, y ≤ 12 → g y ≤ g 6) ∧ g 6 = 36 / 25 ∧
  -- 3. The total demand for the first 6 months is 161/25
  f 6 = 161 / 25 :=
sorry

end demand_analysis_l190_19077


namespace square_diff_plus_six_b_l190_19001

theorem square_diff_plus_six_b (a b : ℝ) (h : a + b = 3) : a^2 - b^2 + 6*b = 9 := by
  sorry

end square_diff_plus_six_b_l190_19001


namespace difference_of_squares_l190_19063

theorem difference_of_squares (m n : ℝ) : (-m - n) * (-m + n) = (-m)^2 - n^2 := by
  sorry

end difference_of_squares_l190_19063


namespace new_homes_theorem_l190_19074

/-- The number of original trailer homes -/
def original_homes : ℕ := 30

/-- The initial average age of original trailer homes 5 years ago -/
def initial_avg_age : ℚ := 15

/-- The current average age of all trailer homes -/
def current_avg_age : ℚ := 12

/-- The number of years that have passed -/
def years_passed : ℕ := 5

/-- Function to calculate the number of new trailer homes added -/
def new_homes_added : ℚ :=
  (original_homes * (initial_avg_age + years_passed) - original_homes * current_avg_age) /
  (current_avg_age - years_passed)

theorem new_homes_theorem :
  new_homes_added = 240 / 7 :=
sorry

end new_homes_theorem_l190_19074


namespace cookies_per_bag_l190_19087

theorem cookies_per_bag (chocolate_chip : ℕ) (oatmeal : ℕ) (bags : ℕ) :
  chocolate_chip = 13 →
  oatmeal = 41 →
  bags = 6 →
  (chocolate_chip + oatmeal) / bags = 9 :=
by sorry

end cookies_per_bag_l190_19087


namespace quadratic_function_range_l190_19043

/-- Given a quadratic function y = x^2 - 2bx + b^2 + c whose graph intersects
    the line y = 1 - x at only one point, and its vertex is on the graph of
    y = ax^2 (a ≠ 0), prove that the range of values for a is a ≥ -1/5 and a ≠ 0. -/
theorem quadratic_function_range (b c : ℝ) (a : ℝ) 
  (h1 : ∃! x, x^2 - 2*b*x + b^2 + c = 1 - x) 
  (h2 : c = a * b^2) 
  (h3 : a ≠ 0) : 
  a ≥ -1/5 ∧ a ≠ 0 := by
  sorry

end quadratic_function_range_l190_19043


namespace log_equation_solution_l190_19034

theorem log_equation_solution :
  ∀ y : ℝ, (Real.log y / Real.log 9 = Real.log 8 / Real.log 2) → y = 729 :=
by sorry

end log_equation_solution_l190_19034


namespace strap_mask_probability_is_0_12_l190_19023

/-- Represents a mask factory with two types of products -/
structure MaskFactory where
  regularRatio : ℝ
  surgicalRatio : ℝ
  regularStrapRatio : ℝ
  surgicalStrapRatio : ℝ

/-- The probability of selecting a strap mask from the factory -/
def strapMaskProbability (factory : MaskFactory) : ℝ :=
  factory.regularRatio * factory.regularStrapRatio +
  factory.surgicalRatio * factory.surgicalStrapRatio

/-- Theorem stating the probability of selecting a strap mask -/
theorem strap_mask_probability_is_0_12 :
  let factory : MaskFactory := {
    regularRatio := 0.8,
    surgicalRatio := 0.2,
    regularStrapRatio := 0.1,
    surgicalStrapRatio := 0.2
  }
  strapMaskProbability factory = 0.12 := by
  sorry

end strap_mask_probability_is_0_12_l190_19023


namespace twenty_in_base_five_l190_19059

/-- Converts a decimal number to its base-5 representation -/
def to_base_five (n : ℕ) : List ℕ :=
  if n = 0 then [0]
  else
    let rec aux (m : ℕ) (acc : List ℕ) : List ℕ :=
      if m = 0 then acc
      else aux (m / 5) ((m % 5) :: acc)
    aux n []

/-- Checks if a list of digits represents a valid base-5 number -/
def is_valid_base_five (digits : List ℕ) : Prop :=
  ∀ d ∈ digits, d < 5

/-- Converts a list of base-5 digits to its decimal value -/
def from_base_five (digits : List ℕ) : ℕ :=
  digits.foldl (fun acc d => acc * 5 + d) 0

theorem twenty_in_base_five :
  to_base_five 20 = [4, 0] ∧
  is_valid_base_five [4, 0] ∧
  from_base_five [4, 0] = 20 :=
sorry

end twenty_in_base_five_l190_19059


namespace pasture_problem_l190_19053

/-- The number of horses b put in the pasture -/
def b_horses : ℕ := 16

/-- The total cost of the pasture in Rs -/
def total_cost : ℕ := 435

/-- The amount b should pay in Rs -/
def b_payment : ℕ := 180

/-- The number of horses a put in -/
def a_horses : ℕ := 12

/-- The number of months a's horses stayed -/
def a_months : ℕ := 8

/-- The number of months b's horses stayed -/
def b_months : ℕ := 9

/-- The number of horses c put in -/
def c_horses : ℕ := 18

/-- The number of months c's horses stayed -/
def c_months : ℕ := 6

theorem pasture_problem :
  b_horses = 16 ∧
  (b_horses * b_months : ℚ) / (a_horses * a_months + b_horses * b_months + c_horses * c_months : ℚ) =
  b_payment / total_cost := by
  sorry

end pasture_problem_l190_19053


namespace truck_distance_l190_19037

/-- Prove that a truck traveling b/4 feet every t seconds will cover 20b/t yards in 4 minutes -/
theorem truck_distance (b t : ℝ) (h1 : t > 0) : 
  let feet_per_t_seconds := b / 4
  let seconds_in_4_minutes := 4 * 60
  let feet_in_yard := 3
  let yards_in_4_minutes := (feet_per_t_seconds * seconds_in_4_minutes / t) / feet_in_yard
  yards_in_4_minutes = 20 * b / t :=
by sorry

end truck_distance_l190_19037


namespace sum_1_to_50_base6_l190_19056

/-- Converts a base 10 number to base 6 --/
def toBase6 (n : ℕ) : ℕ := sorry

/-- Converts a base 6 number to base 10 --/
def fromBase6 (n : ℕ) : ℕ := sorry

/-- Calculates the sum of integers from 1 to n in base 6 --/
def sumInBase6 (n : ℕ) : ℕ := sorry

theorem sum_1_to_50_base6 :
  sumInBase6 (fromBase6 50) = toBase6 55260 := by sorry

end sum_1_to_50_base6_l190_19056


namespace f_derivative_at_zero_l190_19006

noncomputable def f (x : ℝ) : ℝ :=
  if x ≠ 0 then 
    Real.sin (Real.exp (x^2 * Real.sin (5/x)) - 1) + x
  else 
    0

theorem f_derivative_at_zero : 
  deriv f 0 = 1 := by sorry

end f_derivative_at_zero_l190_19006


namespace no_roots_of_composite_l190_19031

-- Define the function f
def f (b c x : ℝ) : ℝ := x^2 + b*x + c

-- State the theorem
theorem no_roots_of_composite (b c : ℝ) :
  (∀ x : ℝ, f b c x ≠ x) →
  (∀ x : ℝ, f b c (f b c x) ≠ x) :=
by sorry

end no_roots_of_composite_l190_19031


namespace solution_set_quadratic_inequality_l190_19013

theorem solution_set_quadratic_inequality :
  let f : ℝ → ℝ := fun x ↦ 2 * x^2 - x - 1
  {x : ℝ | f x > 0} = {x : ℝ | x < -1/2 ∨ x > 1} := by sorry

end solution_set_quadratic_inequality_l190_19013


namespace factorial_sum_equality_l190_19089

theorem factorial_sum_equality : 7 * Nat.factorial 7 + 5 * Nat.factorial 5 - 3 * Nat.factorial 3 + 2 * Nat.factorial 2 = 35866 := by
  sorry

end factorial_sum_equality_l190_19089


namespace parallel_vectors_x_value_l190_19025

/-- Two vectors are parallel if one is a scalar multiple of the other -/
def parallel (a b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, a.1 = k * b.1 ∧ a.2 = k * b.2

theorem parallel_vectors_x_value :
  let a : ℝ × ℝ := (1, -2)
  let b : ℝ × ℝ := (-2, x)
  parallel a b → x = 4 := by
sorry

end parallel_vectors_x_value_l190_19025


namespace jane_reading_probability_l190_19067

theorem jane_reading_probability (p : ℚ) (h : p = 5/8) :
  1 - p = 3/8 := by
  sorry

end jane_reading_probability_l190_19067


namespace range_of_m_l190_19018

theorem range_of_m (m : ℝ) : 
  (∀ x : ℝ, m * x^2 + m * x + 1 > 0) → 0 ≤ m ∧ m < 4 := by
  sorry

end range_of_m_l190_19018


namespace square_roots_problem_l190_19015

theorem square_roots_problem (a : ℝ) :
  (∃ x : ℝ, x > 0 ∧ (2*a + 6)^2 = x ∧ (3 - a)^2 = x) → a = -9 := by
  sorry

end square_roots_problem_l190_19015


namespace fifteenth_student_age_l190_19076

theorem fifteenth_student_age 
  (total_students : Nat) 
  (avg_age_all : ℕ) 
  (group1_size : Nat) 
  (avg_age_group1 : ℕ) 
  (group2_size : Nat) 
  (avg_age_group2 : ℕ) 
  (h1 : total_students = 15) 
  (h2 : avg_age_all = 15) 
  (h3 : group1_size = 5) 
  (h4 : avg_age_group1 = 14) 
  (h5 : group2_size = 9) 
  (h6 : avg_age_group2 = 16) :
  total_students * avg_age_all = 
    group1_size * avg_age_group1 + 
    group2_size * avg_age_group2 + 11 :=
by sorry

end fifteenth_student_age_l190_19076


namespace rectangle_dimension_change_l190_19068

theorem rectangle_dimension_change (L B : ℝ) (L' B' : ℝ) (h1 : L' = 1.03 * L) (h2 : B' = B * (1 + 0.06)) :
  L' * B' = 1.0918 * (L * B) :=
sorry

end rectangle_dimension_change_l190_19068


namespace sqrt_meaningful_range_l190_19062

theorem sqrt_meaningful_range (m : ℝ) : 
  (∃ (x : ℝ), x^2 = m + 3) ↔ m ≥ -3 := by sorry

end sqrt_meaningful_range_l190_19062


namespace platform_length_l190_19024

/-- Calculates the length of a platform given the speed of a train, time to cross the platform,
    and the length of the train. -/
theorem platform_length (train_speed : ℝ) (crossing_time : ℝ) (train_length : ℝ) :
  train_speed = 72 * (5 / 18) →  -- Convert km/hr to m/s
  crossing_time = 26 →
  train_length = 440 →
  train_speed * crossing_time - train_length = 80 := by
  sorry

#check platform_length

end platform_length_l190_19024


namespace max_notebooks_proof_l190_19092

/-- The maximum number of notebooks that can be bought given the constraints -/
def max_notebooks : ℕ := 5

/-- The total budget in yuan -/
def total_budget : ℚ := 30

/-- The total number of books -/
def total_books : ℕ := 30

/-- The cost of each notebook in yuan -/
def notebook_cost : ℚ := 4

/-- The cost of each exercise book in yuan -/
def exercise_book_cost : ℚ := 0.4

theorem max_notebooks_proof :
  (∀ n : ℕ, n ≤ total_books →
    n * notebook_cost + (total_books - n) * exercise_book_cost ≤ total_budget) →
  (max_notebooks * notebook_cost + (total_books - max_notebooks) * exercise_book_cost ≤ total_budget) ∧
  (∀ m : ℕ, m > max_notebooks →
    m * notebook_cost + (total_books - m) * exercise_book_cost > total_budget) :=
by sorry

end max_notebooks_proof_l190_19092


namespace power_multiplication_l190_19046

theorem power_multiplication (a : ℝ) : a^3 * a^4 = a^7 := by sorry

end power_multiplication_l190_19046


namespace us_stripes_count_l190_19073

/-- The number of stars on the US flag -/
def us_stars : ℕ := 50

/-- The number of circles on Pete's flag -/
def pete_circles : ℕ := us_stars / 2 - 3

/-- The number of squares on Pete's flag as a function of US flag stripes -/
def pete_squares (s : ℕ) : ℕ := 2 * s + 6

/-- The total number of shapes on Pete's flag -/
def pete_total_shapes : ℕ := 54

/-- Theorem: The number of stripes on the US flag is 13 -/
theorem us_stripes_count : 
  ∃ (s : ℕ), s = 13 ∧ pete_circles + pete_squares s = pete_total_shapes :=
sorry

end us_stripes_count_l190_19073


namespace total_time_cutting_grass_l190_19085

-- Define the time to cut one lawn in minutes
def time_per_lawn : ℕ := 30

-- Define the number of lawns cut on Saturday
def lawns_saturday : ℕ := 8

-- Define the number of lawns cut on Sunday
def lawns_sunday : ℕ := 8

-- Define the number of minutes in an hour
def minutes_per_hour : ℕ := 60

-- Theorem to prove
theorem total_time_cutting_grass :
  (time_per_lawn * (lawns_saturday + lawns_sunday)) / minutes_per_hour = 8 := by
  sorry

end total_time_cutting_grass_l190_19085


namespace four_digit_with_four_or_five_l190_19038

/-- The number of four-digit positive integers -/
def total_four_digit : ℕ := 9000

/-- The number of four-digit positive integers without 4 or 5 -/
def without_four_or_five : ℕ := 3584

/-- The number of four-digit positive integers with at least one 4 or 5 -/
def with_four_or_five : ℕ := total_four_digit - without_four_or_five

theorem four_digit_with_four_or_five : with_four_or_five = 5416 := by
  sorry

end four_digit_with_four_or_five_l190_19038


namespace consecutive_even_sum_42_square_diff_l190_19078

theorem consecutive_even_sum_42_square_diff (n m : ℤ) : 
  (Even n) → (Even m) → (m = n + 2) → (n + m = 42) → 
  (m ^ 2 - n ^ 2 = 84) := by
sorry

end consecutive_even_sum_42_square_diff_l190_19078


namespace tshirt_cost_l190_19040

theorem tshirt_cost (initial_amount : ℕ) (sweater_cost : ℕ) (shoes_cost : ℕ) (remaining_amount : ℕ) :
  initial_amount = 91 →
  sweater_cost = 24 →
  shoes_cost = 11 →
  remaining_amount = 50 →
  initial_amount - remaining_amount - sweater_cost - shoes_cost = 6 :=
by sorry

end tshirt_cost_l190_19040


namespace min_value_reciprocal_sum_min_value_is_nine_l190_19075

theorem min_value_reciprocal_sum (a b : ℝ) (h1 : a * b > 0) (h2 : a + 4 * b = 1) :
  ∀ x y : ℝ, x * y > 0 ∧ x + 4 * y = 1 → 1 / x + 1 / y ≥ 1 / a + 1 / b :=
by sorry

theorem min_value_is_nine (a b : ℝ) (h1 : a * b > 0) (h2 : a + 4 * b = 1) :
  1 / a + 1 / b = 9 :=
by sorry

end min_value_reciprocal_sum_min_value_is_nine_l190_19075


namespace trigonometric_identities_l190_19099

theorem trigonometric_identities (x : ℝ) : 
  ((Real.sqrt 3) / 2 * Real.cos x - (1 / 2) * Real.sin x = Real.cos (x + π / 6)) ∧ 
  (Real.sin x + Real.cos x = Real.sqrt 2 * Real.sin (x + π / 4)) := by
  sorry

end trigonometric_identities_l190_19099


namespace cone_volume_theorem_l190_19066

-- Define the cone properties
def base_radius : ℝ := 1
def lateral_area_ratio : ℝ := 2

-- Theorem statement
theorem cone_volume_theorem :
  let r := base_radius
  let l := lateral_area_ratio * r -- slant height
  let h := Real.sqrt (l^2 - r^2) -- height
  (1/3 : ℝ) * π * r^2 * h = (Real.sqrt 3 * π) / 3 := by
  sorry

end cone_volume_theorem_l190_19066


namespace range_of_m_l190_19033

def is_circle (m : ℝ) : Prop :=
  ∃ (x y : ℝ), x^2 + y^2 - 2*m*x + 2*m^2 - 2*m = 0

def hyperbola_eccentricity_in_range (m : ℝ) : Prop :=
  ∃ (e : ℝ), 1 < e ∧ e < 2 ∧ e^2 = 1 + m/5

def p (m : ℝ) : Prop := is_circle m
def q (m : ℝ) : Prop := hyperbola_eccentricity_in_range m

theorem range_of_m :
  ∀ m : ℝ, (¬(p m ∧ q m) ∧ (p m ∨ q m)) → (2 ≤ m ∧ m < 15) :=
sorry

end range_of_m_l190_19033


namespace percentage_of_pine_trees_l190_19008

theorem percentage_of_pine_trees (total_trees : ℕ) (non_pine_trees : ℕ) : 
  total_trees = 350 → non_pine_trees = 105 → 
  (((total_trees - non_pine_trees) : ℚ) / total_trees) * 100 = 70 := by
  sorry

end percentage_of_pine_trees_l190_19008


namespace parallel_vectors_x_value_l190_19021

/-- Two 2D vectors are parallel if their cross product is zero -/
def parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 - a.2 * b.1 = 0

/-- Given vectors a and b, prove that if they are parallel, then x = 3 -/
theorem parallel_vectors_x_value (x : ℝ) :
  let a : ℝ × ℝ := (3, -2)
  let b : ℝ × ℝ := (x, -2)
  parallel a b → x = 3 := by
  sorry

end parallel_vectors_x_value_l190_19021


namespace parabola_vertex_l190_19086

/-- Given a parabola with equation y = -x^2 + ax + b where the solution to y ≤ 0
    is (-∞, -1] ∪ [7, ∞), prove that the vertex of this parabola is (3, 16) -/
theorem parabola_vertex (a b : ℝ) :
  (∀ x, -x^2 + a*x + b ≤ 0 ↔ x ≤ -1 ∨ x ≥ 7) →
  ∃ (vertex_x vertex_y : ℝ), vertex_x = 3 ∧ vertex_y = 16 ∧
    ∀ x, -x^2 + a*x + b = -(x - vertex_x)^2 + vertex_y :=
by sorry

end parabola_vertex_l190_19086


namespace perpendicular_lines_parallel_l190_19058

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular and parallel relations
variable (perpendicular : Line → Plane → Prop)
variable (parallel : Line → Line → Prop)

-- State the theorem
theorem perpendicular_lines_parallel (m n : Line) (α : Plane) :
  perpendicular m α → perpendicular n α → parallel m n := by
  sorry

end perpendicular_lines_parallel_l190_19058


namespace inequality_proof_l190_19080

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h_sum : a^2 + b^2 + c^2 = 1) :
  1/a^2 + 1/b^2 + 1/c^2 ≥ 2*(a^3 + b^3 + c^3)/(a*b*c) + 3 := by
  sorry

end inequality_proof_l190_19080


namespace line_segment_length_l190_19079

/-- Represents a line segment in 3D space -/
structure LineSegment3D where
  length : ℝ

/-- Represents a space region around a line segment -/
structure SpaceRegion where
  segment : LineSegment3D
  radius : ℝ
  volume : ℝ

/-- Theorem: If a space region containing all points within 5 units of a line segment 
    in three-dimensional space has a volume of 500π, then the length of the line segment is 40/3 units. -/
theorem line_segment_length (region : SpaceRegion) 
  (h1 : region.radius = 5)
  (h2 : region.volume = 500 * Real.pi) : 
  region.segment.length = 40 / 3 := by
  sorry

end line_segment_length_l190_19079


namespace cos_20_minus_cos_40_l190_19072

theorem cos_20_minus_cos_40 : Real.cos (20 * π / 180) - Real.cos (40 * π / 180) = -1 / (2 * Real.sqrt 5) := by
  sorry

end cos_20_minus_cos_40_l190_19072


namespace prob_non_blue_specific_cube_l190_19093

/-- A cube with colored faces -/
structure ColoredCube where
  green_faces : ℕ
  yellow_faces : ℕ
  blue_faces : ℕ

/-- The probability of rolling a non-blue face on a colored cube -/
def prob_non_blue (cube : ColoredCube) : ℚ :=
  (cube.green_faces + cube.yellow_faces) / (cube.green_faces + cube.yellow_faces + cube.blue_faces)

/-- Theorem: The probability of rolling a non-blue face on a cube with 3 green faces, 2 yellow faces, and 1 blue face is 5/6 -/
theorem prob_non_blue_specific_cube :
  prob_non_blue ⟨3, 2, 1⟩ = 5/6 := by sorry

end prob_non_blue_specific_cube_l190_19093


namespace apple_cost_theorem_l190_19064

/-- The cost of groceries for Olivia -/
def grocery_problem (total_cost banana_cost bread_cost milk_cost apple_cost : ℕ) : Prop :=
  total_cost = 42 ∧
  banana_cost = 12 ∧
  bread_cost = 9 ∧
  milk_cost = 7 ∧
  apple_cost = total_cost - (banana_cost + bread_cost + milk_cost)

theorem apple_cost_theorem :
  ∃ (apple_cost : ℕ), grocery_problem 42 12 9 7 apple_cost ∧ apple_cost = 14 := by
  sorry

end apple_cost_theorem_l190_19064


namespace parabola_focus_l190_19054

/-- The focus of a parabola y^2 = -4x is at (-1, 0) -/
theorem parabola_focus (x y : ℝ) :
  y^2 = -4*x → (x + 1)^2 + y^2 = 1 := by
  sorry

end parabola_focus_l190_19054


namespace parking_lot_bikes_l190_19082

/-- The number of bikes in a parking lot with cars and bikes. -/
def numBikes (numCars : ℕ) (totalWheels : ℕ) (wheelsPerCar : ℕ) (wheelsPerBike : ℕ) : ℕ :=
  (totalWheels - numCars * wheelsPerCar) / wheelsPerBike

theorem parking_lot_bikes :
  numBikes 14 76 4 2 = 10 := by
  sorry

end parking_lot_bikes_l190_19082


namespace line_segment_endpoint_l190_19019

-- Define the start point of the line segment
def start_point : ℝ × ℝ := (1, 3)

-- Define the end point of the line segment
def end_point (x : ℝ) : ℝ × ℝ := (x, 7)

-- Define the length of the line segment
def segment_length : ℝ := 15

-- Theorem statement
theorem line_segment_endpoint (x : ℝ) : 
  x < 0 → 
  Real.sqrt ((x - 1)^2 + (7 - 3)^2) = segment_length → 
  x = 1 - Real.sqrt 209 := by
sorry

end line_segment_endpoint_l190_19019


namespace total_seedlings_transferred_l190_19020

def seedlings_day1 : ℕ := 200

def seedlings_day2 (day1 : ℕ) : ℕ := 2 * day1

theorem total_seedlings_transferred : 
  seedlings_day1 + seedlings_day2 seedlings_day1 = 600 := by
  sorry

end total_seedlings_transferred_l190_19020


namespace complex_modulus_problem_l190_19051

theorem complex_modulus_problem (z : ℂ) : z = (-1 + I) / (1 + I) → Complex.abs z = 1 := by
  sorry

end complex_modulus_problem_l190_19051


namespace no_positive_integer_solutions_l190_19022

theorem no_positive_integer_solutions : ∀ A : ℕ, 
  1 ≤ A → A ≤ 9 → 
  ¬∃ p q : ℕ, p > 0 ∧ q > 0 ∧ p * q = Nat.factorial A ∧ p + q = 10 * A + A := by
  sorry

end no_positive_integer_solutions_l190_19022


namespace garden_land_ratio_l190_19044

/-- Represents a rectangle with width 3/5 of its length -/
structure Rectangle where
  length : ℝ
  width : ℝ
  width_prop : width = 3/5 * length

theorem garden_land_ratio (land garden : Rectangle) 
  (h : garden.length = 3/5 * land.length) :
  (garden.length * garden.width) / (land.length * land.width) = 36/100 := by
  sorry

end garden_land_ratio_l190_19044


namespace initial_nickels_l190_19052

/-- Given the current number of nickels and the number of borrowed nickels,
    prove that the initial number of nickels is their sum. -/
theorem initial_nickels (current_nickels borrowed_nickels : ℕ) :
  let initial_nickels := current_nickels + borrowed_nickels
  initial_nickels = current_nickels + borrowed_nickels :=
by sorry

end initial_nickels_l190_19052


namespace octagon_area_in_circle_l190_19041

theorem octagon_area_in_circle (circle_area : ℝ) (octagon_area : ℝ) :
  circle_area = 256 * Real.pi →
  octagon_area = 8 * (1 / 2 * (Real.sqrt (circle_area / Real.pi))^2 * Real.sin (Real.pi / 4)) →
  octagon_area = 256 * Real.sqrt 2 :=
by sorry

end octagon_area_in_circle_l190_19041


namespace p_sufficient_not_necessary_for_q_l190_19004

theorem p_sufficient_not_necessary_for_q :
  (∀ x : ℝ, x - 1 = 0 → (x - 1) * (x + 2) = 0) ∧
  (∃ x : ℝ, (x - 1) * (x + 2) = 0 ∧ x - 1 ≠ 0) :=
by sorry

end p_sufficient_not_necessary_for_q_l190_19004


namespace fiona_reach_food_prob_l190_19094

-- Define the number of lily pads
def num_pads : ℕ := 16

-- Define the predator pads
def predator_pads : Set ℕ := {2, 5, 8}

-- Define the food pad
def food_pad : ℕ := 14

-- Define Fiona's starting pad
def start_pad : ℕ := 0

-- Define the probability of hopping to the next pad
def hop_prob : ℚ := 1/2

-- Define the probability of jumping 3 pads
def jump_prob : ℚ := 1/2

-- Define a function to calculate the probability of reaching a pad safely
def safe_prob (pad : ℕ) : ℚ := sorry

-- Theorem statement
theorem fiona_reach_food_prob : 
  safe_prob food_pad = 5/1024 := sorry

end fiona_reach_food_prob_l190_19094


namespace tetrahedron_circumscribed_sphere_area_l190_19014

theorem tetrahedron_circumscribed_sphere_area (edge_length : ℝ) : 
  edge_length = 4 → 
  ∃ (sphere_area : ℝ), sphere_area = 24 * Real.pi := by
  sorry

end tetrahedron_circumscribed_sphere_area_l190_19014


namespace cauchy_schwarz_and_inequality_sum_of_roots_inequality_l190_19017

theorem cauchy_schwarz_and_inequality (a b c x y z : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (a^2 + b^2 + c^2) * (x^2 + y^2 + z^2) ≥ (a*x + b*y + c*z)^2 :=
sorry

theorem sum_of_roots_inequality (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hsum : a + b + c = 1) :
  Real.sqrt a + Real.sqrt (2 * b) + Real.sqrt (3 * c) ≤ Real.sqrt 6 :=
sorry

end cauchy_schwarz_and_inequality_sum_of_roots_inequality_l190_19017


namespace G_simplification_l190_19027

noncomputable def F (x : ℝ) : ℝ := Real.log ((1 + x) / (1 - x))

noncomputable def G (x : ℝ) : ℝ := F ((2 * x - x^2) / (1 + 2 * x + x^2))

theorem G_simplification (x : ℝ) (h : x ≠ -1/2) : G x = Real.log (1 + 4 * x) - Real.log (1 + 2 * x) :=
by sorry

end G_simplification_l190_19027


namespace sarah_candy_duration_l190_19071

/-- The number of days Sarah's candy will last -/
def candy_duration (neighbors_candy : ℕ) (sister_candy : ℕ) (daily_consumption : ℕ) : ℕ :=
  (neighbors_candy + sister_candy) / daily_consumption

/-- Proof that Sarah's candy will last 9 days -/
theorem sarah_candy_duration :
  candy_duration 66 15 9 = 9 := by
  sorry

end sarah_candy_duration_l190_19071


namespace quadratic_inequality_solution_l190_19045

/-- Given that the solution set of ax² + 5x - 2 > 0 is {x | 1/2 < x < 2},
    prove that a = -2 and the solution set of ax² + 5x + a² - 1 > 0 is {x | -1/2 < x < 3} -/
theorem quadratic_inequality_solution (a : ℝ) : 
  (∀ x : ℝ, ax^2 + 5*x - 2 > 0 ↔ 1/2 < x ∧ x < 2) → 
  (a = -2 ∧ ∀ x : ℝ, a*x^2 + 5*x + a^2 - 1 > 0 ↔ -1/2 < x ∧ x < 3) :=
by sorry

end quadratic_inequality_solution_l190_19045


namespace complement_A_U_eq_l190_19012

-- Define the universal set U
def U : Set ℝ := {x : ℝ | -4 < x ∧ x < 4}

-- Define set A
def A : Set ℝ := {x : ℝ | -3 ≤ x ∧ x < 2}

-- Define the complement of A with respect to U
def complement_A_U : Set ℝ := {x : ℝ | x ∈ U ∧ x ∉ A}

-- Theorem statement
theorem complement_A_U_eq :
  complement_A_U = {x : ℝ | (-4 < x ∧ x < -3) ∨ (2 ≤ x ∧ x < 4)} :=
by sorry

end complement_A_U_eq_l190_19012


namespace three_digit_power_of_2_and_5_l190_19026

theorem three_digit_power_of_2_and_5 : ∃ n : ℕ, 
  (100 ≤ n ∧ n < 1000) ∧ 
  (∃ m : ℕ, n = 2^m) ∧ 
  (∃ k : ℕ, n = 5^k) :=
sorry

end three_digit_power_of_2_and_5_l190_19026


namespace x_plus_2y_inequality_l190_19011

theorem x_plus_2y_inequality (x y m : ℝ) (hx : x > 0) (hy : y > 0) (h : 2/x + 1/y = 1) :
  x + 2*y > m^2 + 2*m ↔ m > -4 ∧ m < 2 := by
sorry

end x_plus_2y_inequality_l190_19011


namespace eliminate_denominators_l190_19000

theorem eliminate_denominators (x : ℝ) :
  (x - 1) / 3 = 4 - (2 * x + 1) / 2 ↔ 2 * (x - 1) = 24 - 3 * (2 * x + 1) :=
by sorry

end eliminate_denominators_l190_19000


namespace opponents_total_score_l190_19049

def baseball_problem (team_scores : List ℕ) (games_lost : ℕ) : Prop :=
  let total_games := team_scores.length
  let lost_scores := team_scores.take games_lost
  let won_scores := team_scores.drop games_lost
  
  -- Conditions
  total_games = 7 ∧
  team_scores = [1, 3, 5, 6, 7, 8, 10] ∧
  games_lost = 3 ∧
  
  -- Lost games: opponent scored 2 more than the team
  (List.sum (lost_scores.map (· + 2))) +
  -- Won games: team scored 3 times opponent's score
  (List.sum (won_scores.map (· / 3))) = 24

theorem opponents_total_score :
  baseball_problem [1, 3, 5, 6, 7, 8, 10] 3 := by
  sorry

end opponents_total_score_l190_19049


namespace book_has_180_pages_l190_19002

/-- Calculates the number of pages in a book given reading habits and time to finish --/
def book_pages (weekday_pages : ℕ) (weekend_pages : ℕ) (weeks : ℕ) : ℕ :=
  let weekdays := 5 * weeks
  let weekends := 2 * weeks
  weekday_pages * weekdays + weekend_pages * weekends

/-- Theorem stating that a book has 180 pages given specific reading habits and time --/
theorem book_has_180_pages :
  book_pages 10 20 2 = 180 := by
  sorry

#eval book_pages 10 20 2

end book_has_180_pages_l190_19002


namespace power_equation_solution_l190_19050

theorem power_equation_solution (n : ℕ) : 2^(2*n) + 2^(2*n) + 2^(2*n) + 2^(2*n) = 4^22 ↔ n = 21 := by
  sorry

end power_equation_solution_l190_19050


namespace tangent_line_at_neg_one_l190_19069

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := x^3 - 2*x^2 + a*x - 1

-- Define the derivative of f(x)
def f_derivative (a : ℝ) (x : ℝ) : ℝ := 3*x^2 - 4*x + a

-- Theorem statement
theorem tangent_line_at_neg_one (a : ℝ) :
  f_derivative a 1 = 1 →
  ∃ y : ℝ, 9 * (-1) - y + 3 = 0 ∧
  y = f a (-1) ∧
  f_derivative a (-1) = 9 :=
by sorry

end tangent_line_at_neg_one_l190_19069


namespace smallest_angle_is_three_l190_19055

/-- Represents a polygon divided into sectors with central angles forming an arithmetic sequence -/
structure PolygonSectors where
  num_sectors : ℕ
  angle_sum : ℕ
  is_arithmetic_sequence : Bool
  all_angles_integer : Bool

/-- The smallest possible sector angle for a polygon with given properties -/
def smallest_sector_angle (p : PolygonSectors) : ℕ :=
  sorry

/-- Theorem stating the smallest possible sector angle for a specific polygon configuration -/
theorem smallest_angle_is_three :
  ∀ (p : PolygonSectors),
    p.num_sectors = 16 ∧
    p.angle_sum = 360 ∧
    p.is_arithmetic_sequence = true ∧
    p.all_angles_integer = true →
    smallest_sector_angle p = 3 :=
  sorry

end smallest_angle_is_three_l190_19055


namespace problem_statement_l190_19029

theorem problem_statement (x : ℝ) (h : x + 1/x = 7) : 
  (x - 3)^2 + 49 / (x - 3)^2 = 23 := by
sorry

end problem_statement_l190_19029


namespace least_subtraction_for_divisibility_problem_solution_l190_19088

theorem least_subtraction_for_divisibility (n : ℕ) (d : ℕ) (h : d > 0) :
  ∃ (x : ℕ), x ≤ d - 1 ∧ (n - x) % d = 0 ∧ ∀ (y : ℕ), y < x → (n - y) % d ≠ 0 :=
by sorry

theorem problem_solution :
  let n : ℕ := 102932847
  let d : ℕ := 25
  ∃ (x : ℕ), x = 22 ∧ x ≤ d - 1 ∧ (n - x) % d = 0 ∧ ∀ (y : ℕ), y < x → (n - y) % d ≠ 0 :=
by sorry

end least_subtraction_for_divisibility_problem_solution_l190_19088


namespace sine_derivative_2007_l190_19061

open Real

theorem sine_derivative_2007 (f : ℝ → ℝ) (x : ℝ) :
  f = sin →
  (∀ n : ℕ, deriv^[n] f = fun x ↦ f (x + n * (π / 2))) →
  deriv^[2007] f = fun x ↦ -cos x := by
  sorry

end sine_derivative_2007_l190_19061


namespace simplify_expression_l190_19030

theorem simplify_expression (z y : ℝ) : (4 - 5*z + 2*y) - (6 + 7*z - 3*y) = -2 - 12*z + 5*y := by
  sorry

end simplify_expression_l190_19030


namespace sum_reciprocals_perpendicular_lines_equilateral_triangle_perimeter_polynomial_divisibility_l190_19016

-- Problem 1
theorem sum_reciprocals (x y : ℝ) (h1 : x + y = 50) (h2 : x * y = 25) :
  1 / x + 1 / y = 2 := by sorry

-- Problem 2
theorem perpendicular_lines (a b : ℝ) 
  (h : ∀ x y, (a * x + 2 * y + 1 = 0 ∧ 3 * x + b * y + 5 = 0) → 
    ((-a / 2) * (-3 / b) = -1)) :
  b = -3 := by sorry

-- Problem 3
theorem equilateral_triangle_perimeter (A : ℝ) (h : A = 100 * Real.sqrt 3) :
  let s := Real.sqrt (4 * A / Real.sqrt 3);
  3 * s = 60 := by sorry

-- Problem 4
theorem polynomial_divisibility (p q : ℝ) 
  (h : ∀ x, (x + 2) ∣ (x^3 - 2*x^2 + p*x + q)) 
  (h_p : p = 60) :
  q = 136 := by sorry

end sum_reciprocals_perpendicular_lines_equilateral_triangle_perimeter_polynomial_divisibility_l190_19016


namespace largest_of_consecutive_odd_divisible_by_3_l190_19098

/-- Three consecutive odd natural numbers divisible by 3 whose sum is 72 -/
def ConsecutiveOddDivisibleBy3 (a b c : ℕ) : Prop :=
  (Odd a ∧ Odd b ∧ Odd c) ∧
  (a % 3 = 0 ∧ b % 3 = 0 ∧ c % 3 = 0) ∧
  (b = a + 6 ∧ c = a + 12) ∧
  (a + b + c = 72)

theorem largest_of_consecutive_odd_divisible_by_3 {a b c : ℕ} 
  (h : ConsecutiveOddDivisibleBy3 a b c) : 
  max a (max b c) = 30 := by
  sorry

end largest_of_consecutive_odd_divisible_by_3_l190_19098


namespace roots_condition_inequality_condition_max_value_condition_l190_19028

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^2 - 2*a*x + a + 2

-- Part 1
theorem roots_condition (a : ℝ) :
  (∃ x y, x ≠ y ∧ x < 2 ∧ y < 2 ∧ f a x = 0 ∧ f a y = 0) → a < -1 := by sorry

-- Part 2
theorem inequality_condition (a : ℝ) :
  (∀ x, f a x ≥ -1 - a*x) → -2 ≤ a ∧ a ≤ 6 := by sorry

-- Part 3
theorem max_value_condition (a : ℝ) :
  (∀ x ∈ Set.Icc 0 2, f a x ≤ 4) ∧ (∃ x ∈ Set.Icc 0 2, f a x = 4) → a = 2/3 ∨ a = 2 := by sorry

end roots_condition_inequality_condition_max_value_condition_l190_19028
