import Mathlib

namespace NUMINAMATH_CALUDE_f_max_min_implies_a_range_l1705_170548

/-- The function f(x) defined in terms of a real parameter a -/
def f (a : ℝ) (x : ℝ) : ℝ := x^3 + a*x^2 + (a+6)*x + 1

/-- The derivative of f(x) with respect to x -/
def f' (a : ℝ) (x : ℝ) : ℝ := 3*x^2 + 2*a*x + (a+6)

/-- Theorem stating that if f(x) has both a maximum and a minimum, then a is in the specified range -/
theorem f_max_min_implies_a_range (a : ℝ) :
  (∃ (x_max x_min : ℝ), ∀ x, f a x ≤ f a x_max ∧ f a x_min ≤ f a x) →
  a < -3 ∨ a > 6 :=
sorry

end NUMINAMATH_CALUDE_f_max_min_implies_a_range_l1705_170548


namespace NUMINAMATH_CALUDE_sqrt_six_star_sqrt_six_l1705_170524

-- Define the ¤ operation
def star (x y : ℝ) : ℝ := (x + y)^2 - (x - y)^2

-- State the theorem
theorem sqrt_six_star_sqrt_six : star (Real.sqrt 6) (Real.sqrt 6) = 24 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_six_star_sqrt_six_l1705_170524


namespace NUMINAMATH_CALUDE_mans_speed_against_current_l1705_170504

/-- Given a man's speed with the current and the speed of the current, 
    calculate the man's speed against the current. -/
theorem mans_speed_against_current 
  (speed_with_current : ℝ) 
  (current_speed : ℝ) 
  (h1 : speed_with_current = 16) 
  (h2 : current_speed = 3.2) : 
  speed_with_current - 2 * current_speed = 9.6 := by
  sorry

#check mans_speed_against_current

end NUMINAMATH_CALUDE_mans_speed_against_current_l1705_170504


namespace NUMINAMATH_CALUDE_equation_solution_l1705_170563

theorem equation_solution :
  ∀ y : ℝ, (((36 * y + (36 * y + 55) ^ (1/3)) ^ (1/4)) = 11) → y = 7315/18 :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l1705_170563


namespace NUMINAMATH_CALUDE_no_perfect_squares_l1705_170516

theorem no_perfect_squares (n : ℕ+) : 
  ¬(∃ (a b c : ℕ), (2 * n.val^2 - 1 = a^2) ∧ (3 * n.val^2 - 1 = b^2) ∧ (6 * n.val^2 - 1 = c^2)) :=
by sorry

end NUMINAMATH_CALUDE_no_perfect_squares_l1705_170516


namespace NUMINAMATH_CALUDE_banana_arrangements_l1705_170559

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

theorem banana_arrangements :
  let total_letters : ℕ := 6
  let a_count : ℕ := 3
  let n_count : ℕ := 2
  let b_count : ℕ := 1
  factorial total_letters / (factorial a_count * factorial n_count * factorial b_count) = 60 := by
  sorry

end NUMINAMATH_CALUDE_banana_arrangements_l1705_170559


namespace NUMINAMATH_CALUDE_tan_beta_value_l1705_170580

theorem tan_beta_value (α β : Real) 
  (h1 : Real.tan α = 2) 
  (h2 : Real.tan (α + β) = -1) : 
  Real.tan β = 3 := by
sorry

end NUMINAMATH_CALUDE_tan_beta_value_l1705_170580


namespace NUMINAMATH_CALUDE_school_gender_ratio_l1705_170590

theorem school_gender_ratio (boys girls : ℕ) : 
  boys * 13 = girls * 5 →  -- ratio of boys to girls is 5:13
  girls = boys + 80 →      -- there are 80 more girls than boys
  boys = 50 :=             -- prove that the number of boys is 50
by sorry

end NUMINAMATH_CALUDE_school_gender_ratio_l1705_170590


namespace NUMINAMATH_CALUDE_two_propositions_are_true_l1705_170546

/-- Represents the truth value of a proposition -/
inductive PropositionTruth
  | True
  | False

/-- The four propositions in the problem -/
def proposition1 : PropositionTruth := PropositionTruth.False
def proposition2 : PropositionTruth := PropositionTruth.True
def proposition3 : PropositionTruth := PropositionTruth.False
def proposition4 : PropositionTruth := PropositionTruth.True

/-- Counts the number of true propositions -/
def countTruePropositions (p1 p2 p3 p4 : PropositionTruth) : Nat :=
  match p1, p2, p3, p4 with
  | PropositionTruth.True, PropositionTruth.True, PropositionTruth.True, PropositionTruth.True => 4
  | PropositionTruth.True, PropositionTruth.True, PropositionTruth.True, PropositionTruth.False => 3
  | PropositionTruth.True, PropositionTruth.True, PropositionTruth.False, PropositionTruth.True => 3
  | PropositionTruth.True, PropositionTruth.True, PropositionTruth.False, PropositionTruth.False => 2
  | PropositionTruth.True, PropositionTruth.False, PropositionTruth.True, PropositionTruth.True => 3
  | PropositionTruth.True, PropositionTruth.False, PropositionTruth.True, PropositionTruth.False => 2
  | PropositionTruth.True, PropositionTruth.False, PropositionTruth.False, PropositionTruth.True => 2
  | PropositionTruth.True, PropositionTruth.False, PropositionTruth.False, PropositionTruth.False => 1
  | PropositionTruth.False, PropositionTruth.True, PropositionTruth.True, PropositionTruth.True => 3
  | PropositionTruth.False, PropositionTruth.True, PropositionTruth.True, PropositionTruth.False => 2
  | PropositionTruth.False, PropositionTruth.True, PropositionTruth.False, PropositionTruth.True => 2
  | PropositionTruth.False, PropositionTruth.True, PropositionTruth.False, PropositionTruth.False => 1
  | PropositionTruth.False, PropositionTruth.False, PropositionTruth.True, PropositionTruth.True => 2
  | PropositionTruth.False, PropositionTruth.False, PropositionTruth.True, PropositionTruth.False => 1
  | PropositionTruth.False, PropositionTruth.False, PropositionTruth.False, PropositionTruth.True => 1
  | PropositionTruth.False, PropositionTruth.False, PropositionTruth.False, PropositionTruth.False => 0

/-- Theorem stating that exactly 2 out of 4 given propositions are true -/
theorem two_propositions_are_true :
  countTruePropositions proposition1 proposition2 proposition3 proposition4 = 2 := by
  sorry

end NUMINAMATH_CALUDE_two_propositions_are_true_l1705_170546


namespace NUMINAMATH_CALUDE_relay_race_second_leg_time_l1705_170583

theorem relay_race_second_leg_time 
  (first_leg_time : ℝ) 
  (average_time : ℝ) 
  (h1 : first_leg_time = 58) 
  (h2 : average_time = 42) : 
  let second_leg_time := 2 * average_time - first_leg_time
  second_leg_time = 26 := by
  sorry

end NUMINAMATH_CALUDE_relay_race_second_leg_time_l1705_170583


namespace NUMINAMATH_CALUDE_mango_rate_per_kg_l1705_170579

/-- The rate per kg for mangoes given the purchase details --/
theorem mango_rate_per_kg (grape_quantity grape_rate mango_quantity total_payment : ℕ) : 
  grape_quantity = 9 →
  grape_rate = 70 →
  mango_quantity = 9 →
  total_payment = 1125 →
  (total_payment - grape_quantity * grape_rate) / mango_quantity = 55 := by
sorry

end NUMINAMATH_CALUDE_mango_rate_per_kg_l1705_170579


namespace NUMINAMATH_CALUDE_polar_equivalence_l1705_170565

/-- Two points in polar coordinates are equivalent if they represent the same point in the plane. -/
def polar_equivalent (r1 : ℝ) (θ1 : ℝ) (r2 : ℝ) (θ2 : ℝ) : Prop :=
  r1 * (Real.cos θ1) = r2 * (Real.cos θ2) ∧ r1 * (Real.sin θ1) = r2 * (Real.sin θ2)

/-- The theorem stating that (-3, 7π/6) is equivalent to (3, π/6) in polar coordinates. -/
theorem polar_equivalence :
  polar_equivalent (-3) (7 * Real.pi / 6) 3 (Real.pi / 6) ∧ 
  3 > 0 ∧ 
  0 ≤ Real.pi / 6 ∧ 
  Real.pi / 6 < 2 * Real.pi :=
sorry

end NUMINAMATH_CALUDE_polar_equivalence_l1705_170565


namespace NUMINAMATH_CALUDE_imaginary_part_of_i_minus_one_l1705_170570

theorem imaginary_part_of_i_minus_one :
  Complex.im (Complex.I - 1) = 1 :=
sorry

end NUMINAMATH_CALUDE_imaginary_part_of_i_minus_one_l1705_170570


namespace NUMINAMATH_CALUDE_maggie_red_packs_l1705_170553

/-- The number of packs of red bouncy balls Maggie bought -/
def red_packs : ℕ := sorry

/-- The number of packs of yellow bouncy balls Maggie bought -/
def yellow_packs : ℕ := 8

/-- The number of packs of green bouncy balls Maggie bought -/
def green_packs : ℕ := 4

/-- The number of bouncy balls in each package -/
def balls_per_pack : ℕ := 10

/-- The total number of bouncy balls Maggie bought -/
def total_balls : ℕ := 160

theorem maggie_red_packs : red_packs = 4 := by
  sorry

end NUMINAMATH_CALUDE_maggie_red_packs_l1705_170553


namespace NUMINAMATH_CALUDE_adam_total_earnings_l1705_170552

/-- Calculates Adam's earnings given task rates, completion numbers, and exchange rates -/
def adam_earnings (dollar_per_lawn : ℝ) (euro_per_car : ℝ) (peso_per_dog : ℝ)
                  (lawns_total : ℕ) (cars_total : ℕ) (dogs_total : ℕ)
                  (lawns_forgot : ℕ) (cars_forgot : ℕ) (dogs_forgot : ℕ)
                  (euro_to_dollar : ℝ) (peso_to_dollar : ℝ) : ℝ :=
  let lawns_done := lawns_total - lawns_forgot
  let cars_done := cars_total - cars_forgot
  let dogs_done := dogs_total - dogs_forgot
  
  let lawn_earnings := dollar_per_lawn * lawns_done
  let car_earnings := euro_per_car * cars_done * euro_to_dollar
  let dog_earnings := peso_per_dog * dogs_done * peso_to_dollar
  
  lawn_earnings + car_earnings + dog_earnings

/-- Theorem stating Adam's earnings based on given conditions -/
theorem adam_total_earnings :
  adam_earnings 9 10 50 12 6 4 8 2 1 1.1 0.05 = 87.5 := by
  sorry

#eval adam_earnings 9 10 50 12 6 4 8 2 1 1.1 0.05

end NUMINAMATH_CALUDE_adam_total_earnings_l1705_170552


namespace NUMINAMATH_CALUDE_shaded_area_of_circles_l1705_170562

theorem shaded_area_of_circles (r : ℝ) (h1 : r > 0) (h2 : π * r^2 = 81 * π) : 
  (π * r^2) / 2 + (π * (r/2)^2) / 2 = 50.625 * π := by sorry

end NUMINAMATH_CALUDE_shaded_area_of_circles_l1705_170562


namespace NUMINAMATH_CALUDE_quadratic_roots_range_l1705_170502

theorem quadratic_roots_range (m : ℝ) :
  (∃ x₁ x₂ : ℝ, (m + 3) * x₁^2 - 4 * m * x₁ + 2 * m - 1 = 0 ∧
                (m + 3) * x₂^2 - 4 * m * x₂ + 2 * m - 1 = 0 ∧
                x₁ * x₂ < 0 ∧
                x₁ < 0 ∧ x₂ > 0 ∧
                abs x₁ > x₂) →
  m > -3 ∧ m < 0 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_range_l1705_170502


namespace NUMINAMATH_CALUDE_largest_integer_for_negative_quadratic_l1705_170587

theorem largest_integer_for_negative_quadratic :
  ∀ m : ℤ, m^2 - 11*m + 24 < 0 → m ≤ 7 ∧ 
  ∃ n : ℤ, n^2 - 11*n + 24 < 0 ∧ n = 7 :=
by sorry

end NUMINAMATH_CALUDE_largest_integer_for_negative_quadratic_l1705_170587


namespace NUMINAMATH_CALUDE_matching_socks_probability_l1705_170535

def blue_socks : ℕ := 12
def green_socks : ℕ := 10
def red_socks : ℕ := 9

def total_socks : ℕ := blue_socks + green_socks + red_socks

def matching_pairs : ℕ := (blue_socks.choose 2) + (green_socks.choose 2) + (red_socks.choose 2)

def total_pairs : ℕ := total_socks.choose 2

theorem matching_socks_probability :
  (matching_pairs : ℚ) / total_pairs = 147 / 465 :=
sorry

end NUMINAMATH_CALUDE_matching_socks_probability_l1705_170535


namespace NUMINAMATH_CALUDE_mrs_hilt_chickens_l1705_170588

theorem mrs_hilt_chickens (total_legs : ℕ) (num_dogs : ℕ) (dog_legs : ℕ) (chicken_legs : ℕ) 
  (h1 : total_legs = 12)
  (h2 : num_dogs = 2)
  (h3 : dog_legs = 4)
  (h4 : chicken_legs = 2) :
  (total_legs - num_dogs * dog_legs) / chicken_legs = 2 := by
sorry

end NUMINAMATH_CALUDE_mrs_hilt_chickens_l1705_170588


namespace NUMINAMATH_CALUDE_greatest_multiple_of_eight_remainder_l1705_170582

/-- A function that checks if a natural number uses only unique digits from 1 to 9 -/
def uniqueDigits (n : ℕ) : Prop := sorry

/-- The greatest integer multiple of 8 using unique digits from 1 to 9 -/
noncomputable def M : ℕ := sorry

theorem greatest_multiple_of_eight_remainder :
  M % 1000 = 976 ∧ M % 8 = 0 ∧ uniqueDigits M ∧ ∀ k : ℕ, k > M → k % 8 = 0 → ¬(uniqueDigits k) := by
  sorry

end NUMINAMATH_CALUDE_greatest_multiple_of_eight_remainder_l1705_170582


namespace NUMINAMATH_CALUDE_probability_equals_frequency_l1705_170514

/-- Represents the number of red balls in the bag -/
def red_balls : ℕ := 2

/-- Represents the number of yellow balls in the bag -/
def yellow_balls : ℕ := 3

/-- Represents the total number of balls in the bag -/
def total_balls : ℕ := red_balls + yellow_balls

/-- Represents the observed limiting frequency from the experiment -/
def observed_frequency : ℚ := 2/5

/-- Theorem stating that the probability of selecting a red ball equals the observed frequency -/
theorem probability_equals_frequency : 
  (red_balls : ℚ) / (total_balls : ℚ) = observed_frequency :=
sorry

end NUMINAMATH_CALUDE_probability_equals_frequency_l1705_170514


namespace NUMINAMATH_CALUDE_number_difference_l1705_170564

theorem number_difference (L S : ℕ) (h1 : L = 1584) (h2 : L = 6 * S + 15) : L - S = 1323 := by
  sorry

end NUMINAMATH_CALUDE_number_difference_l1705_170564


namespace NUMINAMATH_CALUDE_treadmill_time_difference_l1705_170581

theorem treadmill_time_difference : 
  let total_distance : ℝ := 8
  let constant_speed : ℝ := 3
  let day1_speed : ℝ := 6
  let day2_speed : ℝ := 3
  let day3_speed : ℝ := 4
  let day4_speed : ℝ := 3
  let daily_distance : ℝ := 2
  let constant_time := total_distance / constant_speed
  let varied_time := daily_distance / day1_speed + daily_distance / day2_speed + 
                     daily_distance / day3_speed + daily_distance / day4_speed
  (constant_time - varied_time) * 60 = 80 := by sorry

end NUMINAMATH_CALUDE_treadmill_time_difference_l1705_170581


namespace NUMINAMATH_CALUDE_problem_solution_l1705_170515

-- Define the sets A and B
def A (a : ℝ) : Set ℝ := {x | x^2 - x - a = 0}
def B : Set ℝ := {2, -5}

-- Define the theorem
theorem problem_solution :
  ∃ (a : ℝ),
    (2 ∈ A a) ∧
    (a = 2) ∧
    (A a = {-1, 2}) ∧
    (let U := A a ∪ B;
     U = {-5, -1, 2} ∧
     (U \ A a) ∪ (U \ B) = {-5, -1}) :=
by
  sorry


end NUMINAMATH_CALUDE_problem_solution_l1705_170515


namespace NUMINAMATH_CALUDE_bean_region_probability_l1705_170523

noncomputable def probability_bean_region : ℝ :=
  let total_area := (1 - 0) * ((Real.exp 1 + 1) - 0)
  let specific_area := ∫ x in (0)..(1), (Real.exp x + 1) - (Real.exp 1 + 1)
  specific_area / total_area

theorem bean_region_probability : probability_bean_region = 1 / (Real.exp 1 + 1) := by
  sorry

end NUMINAMATH_CALUDE_bean_region_probability_l1705_170523


namespace NUMINAMATH_CALUDE_division_property_l1705_170532

theorem division_property (a b : ℕ+) :
  (∃ (q r : ℕ), a.val^2 + b.val^2 = q * (a.val + b.val) + r ∧
                0 ≤ r ∧ r < a.val + b.val ∧
                q^2 + r = 1977) →
  ((a.val = 50 ∧ b.val = 37) ∨
   (a.val = 50 ∧ b.val = 7) ∨
   (a.val = 37 ∧ b.val = 50) ∨
   (a.val = 7 ∧ b.val = 50)) :=
by sorry

end NUMINAMATH_CALUDE_division_property_l1705_170532


namespace NUMINAMATH_CALUDE_relay_race_arrangements_l1705_170571

/-- The number of female students --/
def total_students : ℕ := 6

/-- The number of students to be selected for the relay race --/
def selected_students : ℕ := 4

/-- A function to calculate the number of arrangements when only one of A or B participates --/
def one_participates : ℕ := 2 * (total_students - 2).choose (selected_students - 1) * (selected_students).factorial

/-- A function to calculate the number of arrangements when both A and B participate --/
def both_participate : ℕ := selected_students.choose 2 * (selected_students - 1).factorial

/-- The total number of different arrangements --/
def total_arrangements : ℕ := one_participates + both_participate

theorem relay_race_arrangements :
  total_arrangements = 264 := by sorry

end NUMINAMATH_CALUDE_relay_race_arrangements_l1705_170571


namespace NUMINAMATH_CALUDE_two_numbers_problem_l1705_170517

theorem two_numbers_problem (A B : ℝ) (h1 : A + B = 40) (h2 : A * B = 375) (h3 : A / B = 3/2) 
  (h4 : A > 0) (h5 : B > 0) : A = 24 ∧ B = 16 ∧ A - B = 8 := by
  sorry

end NUMINAMATH_CALUDE_two_numbers_problem_l1705_170517


namespace NUMINAMATH_CALUDE_specific_circle_diameter_l1705_170594

/-- A circle tangent to the y-axis and a line, passing through a specific point -/
structure TangentCircle where
  /-- The circle is tangent to the y-axis -/
  tangent_y_axis : Bool
  /-- The slope of the line the circle is tangent to -/
  line_slope : ℝ
  /-- The x-coordinate of the point the circle passes through -/
  point_x : ℝ
  /-- The y-coordinate of the point the circle passes through -/
  point_y : ℝ

/-- The diameter of a TangentCircle -/
def circle_diameter (c : TangentCircle) : Set ℝ :=
  {d : ℝ | d = 2 ∨ d = 14/3}

/-- Theorem stating the diameter of the specific TangentCircle -/
theorem specific_circle_diameter :
  let c : TangentCircle := {
    tangent_y_axis := true,
    line_slope := Real.sqrt 3 / 3,
    point_x := 2,
    point_y := Real.sqrt 3
  }
  ∀ d ∈ circle_diameter c, d = 2 ∨ d = 14/3 := by
  sorry

end NUMINAMATH_CALUDE_specific_circle_diameter_l1705_170594


namespace NUMINAMATH_CALUDE_jerry_collection_cost_l1705_170511

/-- The amount of money Jerry needs to finish his action figure collection -/
def money_needed (current : ℕ) (total : ℕ) (cost : ℕ) : ℕ :=
  (total - current) * cost

/-- Theorem: Jerry needs $72 to finish his collection -/
theorem jerry_collection_cost : money_needed 7 16 8 = 72 := by
  sorry

end NUMINAMATH_CALUDE_jerry_collection_cost_l1705_170511


namespace NUMINAMATH_CALUDE_other_solution_quadratic_l1705_170549

theorem other_solution_quadratic (h : 49 * (5/7)^2 - 88 * (5/7) + 40 = 0) :
  49 * (8/7)^2 - 88 * (8/7) + 40 = 0 := by
  sorry

end NUMINAMATH_CALUDE_other_solution_quadratic_l1705_170549


namespace NUMINAMATH_CALUDE_abc_equation_solutions_l1705_170586

theorem abc_equation_solutions :
  ∃! (s : Finset (ℕ × ℕ × ℕ)), 
    s.card = 5 ∧ 
    (∀ (a b c : ℕ), (a, b, c) ∈ s ↔ 
      a > 0 ∧ b > 0 ∧ c > 0 ∧ 
      a ≥ b ∧ b ≥ c ∧ 
      a * b * c = 2 * (a - 1) * (b - 1) * (c - 1)) :=
by sorry

end NUMINAMATH_CALUDE_abc_equation_solutions_l1705_170586


namespace NUMINAMATH_CALUDE_no_valid_reassignment_l1705_170541

/-- Represents a seating arrangement in a classroom -/
structure Classroom :=
  (rows : Nat)
  (cols : Nat)
  (students : Nat)
  (center_empty : Bool)

/-- Checks if a reassignment is possible given the classroom setup -/
def reassignment_possible (c : Classroom) : Prop :=
  c.rows = 5 ∧ c.cols = 7 ∧ c.students = 34 ∧ c.center_empty = true →
  ∃ (new_arrangement : Fin c.students → Fin (c.rows * c.cols)),
    ∀ i : Fin c.students,
      let old_pos := i.val
      let new_pos := (new_arrangement i).val
      (new_pos ≠ old_pos) ∧
      ((new_pos = old_pos + 1 ∨ new_pos = old_pos - 1) ∨
       (new_pos = old_pos + c.cols ∨ new_pos = old_pos - c.cols))

theorem no_valid_reassignment (c : Classroom) :
  ¬(reassignment_possible c) :=
sorry

end NUMINAMATH_CALUDE_no_valid_reassignment_l1705_170541


namespace NUMINAMATH_CALUDE_valid_numbers_count_l1705_170567

/-- Counts the number of valid eight-digit numbers where each digit appears exactly as many times as its value. -/
def count_valid_numbers : ℕ :=
  let single_eight := 1
  let seven_sevens_one_one := 8
  let six_sixes_two_twos := 28
  let five_fives_two_twos_one_one := 168
  let five_fives_three_threes := 56
  let four_fours_three_threes_one_one := 280
  single_eight + seven_sevens_one_one + six_sixes_two_twos + 
  five_fives_two_twos_one_one + five_fives_three_threes + 
  four_fours_three_threes_one_one

theorem valid_numbers_count : count_valid_numbers = 541 := by
  sorry

end NUMINAMATH_CALUDE_valid_numbers_count_l1705_170567


namespace NUMINAMATH_CALUDE_time_to_paint_one_room_l1705_170510

/-- Given a painting job with a total number of rooms, rooms already painted,
    and time to paint the remaining rooms, calculate the time to paint one room. -/
theorem time_to_paint_one_room
  (total_rooms : ℕ)
  (painted_rooms : ℕ)
  (time_for_remaining : ℕ)
  (h1 : total_rooms = 10)
  (h2 : painted_rooms = 8)
  (h3 : time_for_remaining = 16)
  (h4 : painted_rooms < total_rooms) :
  (time_for_remaining : ℚ) / (total_rooms - painted_rooms : ℚ) = 8 := by
  sorry

end NUMINAMATH_CALUDE_time_to_paint_one_room_l1705_170510


namespace NUMINAMATH_CALUDE_sprinter_target_heart_rate_l1705_170573

/-- Calculates the maximum heart rate given the age --/
def maxHeartRate (age : ℕ) : ℕ := 225 - age

/-- Calculates the target heart rate as a percentage of the maximum heart rate --/
def targetHeartRate (maxRate : ℕ) : ℚ := 0.85 * maxRate

/-- Rounds a rational number to the nearest integer --/
def roundToNearest (x : ℚ) : ℤ := (x + 1/2).floor

theorem sprinter_target_heart_rate :
  let age : ℕ := 30
  let max_rate := maxHeartRate age
  let target_rate := targetHeartRate max_rate
  roundToNearest target_rate = 166 := by sorry

end NUMINAMATH_CALUDE_sprinter_target_heart_rate_l1705_170573


namespace NUMINAMATH_CALUDE_population_average_age_l1705_170521

/-- Proves that given a population with a specific ratio of women to men and their respective average ages, the average age of the entire population can be calculated. -/
theorem population_average_age 
  (total_population : ℕ) 
  (women_ratio : ℚ) 
  (men_ratio : ℚ) 
  (women_avg_age : ℚ) 
  (men_avg_age : ℚ) 
  (h1 : women_ratio + men_ratio = 1) 
  (h2 : women_ratio = 11 / 21) 
  (h3 : men_ratio = 10 / 21) 
  (h4 : women_avg_age = 34) 
  (h5 : men_avg_age = 32) : 
  (women_ratio * women_avg_age + men_ratio * men_avg_age : ℚ) = 33 + 1 / 21 := by
  sorry

#check population_average_age

end NUMINAMATH_CALUDE_population_average_age_l1705_170521


namespace NUMINAMATH_CALUDE_triangle_arithmetic_sequence_properties_l1705_170533

/-- Given a triangle with sides a > b > c forming an arithmetic sequence with difference d,
    and inscribed circle radius r, prove the following properties. -/
theorem triangle_arithmetic_sequence_properties
  (a b c d r : ℝ)
  (α γ : ℝ)
  (h1 : a > b)
  (h2 : b > c)
  (h3 : a - b = d)
  (h4 : b - c = d)
  (h5 : r > 0)
  (h6 : α > 0)
  (h7 : γ > 0) :
  (Real.tan (α / 2) * Real.tan (γ / 2) = 1 / 3) ∧
  (r = 2 * d / (3 * (Real.tan (α / 2) - Real.tan (γ / 2)))) :=
by sorry

end NUMINAMATH_CALUDE_triangle_arithmetic_sequence_properties_l1705_170533


namespace NUMINAMATH_CALUDE_farmer_radishes_per_row_l1705_170503

/-- Represents the farmer's planting scenario -/
structure FarmerPlanting where
  bean_seedlings : ℕ
  bean_per_row : ℕ
  pumpkin_seeds : ℕ
  pumpkin_per_row : ℕ
  total_radishes : ℕ
  rows_per_bed : ℕ
  total_beds : ℕ

/-- Calculates the number of radishes per row -/
def radishes_per_row (fp : FarmerPlanting) : ℕ :=
  let bean_rows := fp.bean_seedlings / fp.bean_per_row
  let pumpkin_rows := fp.pumpkin_seeds / fp.pumpkin_per_row
  let total_rows := fp.rows_per_bed * fp.total_beds
  let radish_rows := total_rows - (bean_rows + pumpkin_rows)
  fp.total_radishes / radish_rows

/-- Theorem stating that given the farmer's planting conditions, 
    the number of radishes per row is 6 -/
theorem farmer_radishes_per_row :
  let fp : FarmerPlanting := {
    bean_seedlings := 64,
    bean_per_row := 8,
    pumpkin_seeds := 84,
    pumpkin_per_row := 7,
    total_radishes := 48,
    rows_per_bed := 2,
    total_beds := 14
  }
  radishes_per_row fp = 6 := by
  sorry

end NUMINAMATH_CALUDE_farmer_radishes_per_row_l1705_170503


namespace NUMINAMATH_CALUDE_linear_function_properties_l1705_170555

/-- A linear function passing through two given points -/
structure LinearFunction where
  b : ℝ
  k : ℝ
  point1 : b * (-2) + k = -3
  point2 : b * 1 + k = 3

/-- Theorem stating the properties of the linear function -/
theorem linear_function_properties (f : LinearFunction) :
  f.k = 1 ∧ f.b = 2 ∧ f.b * (-2) + f.k ≠ 3 := by sorry

end NUMINAMATH_CALUDE_linear_function_properties_l1705_170555


namespace NUMINAMATH_CALUDE_carpet_area_calculation_l1705_170509

theorem carpet_area_calculation (room_length room_width wardrobe_side feet_per_yard : ℝ) 
  (h1 : room_length = 18)
  (h2 : room_width = 12)
  (h3 : wardrobe_side = 3)
  (h4 : feet_per_yard = 3) : 
  (room_length * room_width - wardrobe_side * wardrobe_side) / (feet_per_yard * feet_per_yard) = 23 := by
  sorry

end NUMINAMATH_CALUDE_carpet_area_calculation_l1705_170509


namespace NUMINAMATH_CALUDE_screen_time_calculation_l1705_170560

/-- Calculates the remaining screen time for the evening given the total recommended time and time already used. -/
def remaining_screen_time (total_recommended : ℕ) (time_used : ℕ) : ℕ :=
  total_recommended - time_used

/-- Converts hours to minutes. -/
def hours_to_minutes (hours : ℕ) : ℕ :=
  hours * 60

theorem screen_time_calculation :
  let total_recommended := hours_to_minutes 2
  let time_used := 45
  remaining_screen_time total_recommended time_used = 75 := by
  sorry

end NUMINAMATH_CALUDE_screen_time_calculation_l1705_170560


namespace NUMINAMATH_CALUDE_scramble_language_word_count_l1705_170542

/-- The number of letters in the Scramble alphabet -/
def alphabet_size : ℕ := 25

/-- The maximum word length in the Scramble language -/
def max_word_length : ℕ := 5

/-- Calculates the number of words of a given length that contain at least one 'B' -/
def words_with_b (length : ℕ) : ℕ :=
  alphabet_size ^ length - (alphabet_size - 1) ^ length

/-- The total number of valid words in the Scramble language -/
def total_valid_words : ℕ :=
  words_with_b 1 + words_with_b 2 + words_with_b 3 + words_with_b 4 + words_with_b 5

theorem scramble_language_word_count :
  total_valid_words = 1863701 :=
by sorry

end NUMINAMATH_CALUDE_scramble_language_word_count_l1705_170542


namespace NUMINAMATH_CALUDE_polygon_similarity_nesting_l1705_170531

-- Define polygons
variable (Polygon : Type)

-- Define similarity relation between polygons
variable (similar : Polygon → Polygon → Prop)

-- Define nesting relation between polygons
variable (nesting : Polygon → Polygon → Prop)

-- Main theorem
theorem polygon_similarity_nesting 
  (p q : Polygon) : 
  (¬ similar p q) ↔ 
  (∃ r : Polygon, similar r q ∧ ¬ nesting r p) :=
sorry

end NUMINAMATH_CALUDE_polygon_similarity_nesting_l1705_170531


namespace NUMINAMATH_CALUDE_solve_sticker_problem_l1705_170599

def sticker_problem (initial : ℝ) (bought : ℝ) (birthday : ℝ) (mother : ℝ) (total : ℝ) : Prop :=
  let from_sister := total - (initial + bought + birthday + mother)
  from_sister = 6.0

theorem solve_sticker_problem :
  sticker_problem 20.0 26.0 20.0 58.0 130.0 := by
  sorry

end NUMINAMATH_CALUDE_solve_sticker_problem_l1705_170599


namespace NUMINAMATH_CALUDE_mutually_exclusive_events_l1705_170522

-- Define the sample space for two coin tosses
inductive CoinToss
  | HH  -- Two heads
  | HT  -- Head then tail
  | TH  -- Tail then head
  | TT  -- Two tails

-- Define the event "At least one head"
def atLeastOneHead (outcome : CoinToss) : Prop :=
  outcome = CoinToss.HH ∨ outcome = CoinToss.HT ∨ outcome = CoinToss.TH

-- Define the event "Both tosses are tails"
def bothTails (outcome : CoinToss) : Prop :=
  outcome = CoinToss.TT

-- Theorem stating that "Both tosses are tails" is mutually exclusive to "At least one head"
theorem mutually_exclusive_events :
  ∀ (outcome : CoinToss), ¬(atLeastOneHead outcome ∧ bothTails outcome) :=
by sorry

end NUMINAMATH_CALUDE_mutually_exclusive_events_l1705_170522


namespace NUMINAMATH_CALUDE_inequalities_proof_l1705_170597

theorem inequalities_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a^2 / b ≥ 2*a - b) ∧ (a^2 / b + b^2 / c + c^2 / a ≥ a + b + c) := by
  sorry

end NUMINAMATH_CALUDE_inequalities_proof_l1705_170597


namespace NUMINAMATH_CALUDE_fraction_division_problem_l1705_170500

theorem fraction_division_problem : (4 + 2 / 3) / (9 / 7) = 98 / 27 := by
  sorry

end NUMINAMATH_CALUDE_fraction_division_problem_l1705_170500


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l1705_170525

/-- A geometric sequence with positive terms -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r ≠ 0 ∧ ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_sum (a : ℕ → ℝ) :
  GeometricSequence a →
  (∀ n : ℕ, a n > 0) →
  a 2 * a 4 + 2 * a 3 * a 5 + a 4 * a 6 = 25 →
  a 3 + a 5 = 5 :=
by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l1705_170525


namespace NUMINAMATH_CALUDE_richard_david_age_difference_l1705_170554

/-- Represents the ages of the three sons -/
structure Ages where
  richard : ℕ
  david : ℕ
  scott : ℕ

/-- The conditions of the problem -/
def family_conditions (ages : Ages) : Prop :=
  ages.richard > ages.david ∧
  ages.david = ages.scott + 8 ∧
  ages.richard + 8 = 2 * (ages.scott + 8) ∧
  ages.david = 14

/-- The theorem to prove -/
theorem richard_david_age_difference (ages : Ages) 
  (h : family_conditions ages) : ages.richard - ages.david = 6 := by
  sorry

end NUMINAMATH_CALUDE_richard_david_age_difference_l1705_170554


namespace NUMINAMATH_CALUDE_shadow_height_calculation_l1705_170543

/-- Given a tree and a person casting shadows, calculate the person's height -/
theorem shadow_height_calculation (tree_height tree_shadow alex_shadow : ℚ) 
  (h1 : tree_height = 50)
  (h2 : tree_shadow = 25)
  (h3 : alex_shadow = 20 / 12) : -- Convert 20 inches to feet
  tree_height / tree_shadow * alex_shadow = 10 / 3 := by
  sorry

#check shadow_height_calculation

end NUMINAMATH_CALUDE_shadow_height_calculation_l1705_170543


namespace NUMINAMATH_CALUDE_rectangle_area_l1705_170547

/-- Given a rectangle with perimeter 40 and one side length 5, prove its area is 75 -/
theorem rectangle_area (perimeter : ℝ) (side : ℝ) (h1 : perimeter = 40) (h2 : side = 5) :
  let other_side := perimeter / 2 - side
  side * other_side = 75 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_l1705_170547


namespace NUMINAMATH_CALUDE_angle_triple_supplement_l1705_170550

theorem angle_triple_supplement (x : ℝ) : 
  x = 3 * (180 - x) → x = 135 := by
  sorry

end NUMINAMATH_CALUDE_angle_triple_supplement_l1705_170550


namespace NUMINAMATH_CALUDE_subtraction_problem_l1705_170589

theorem subtraction_problem : 240 - (35 * 4 + 6 * 3) = 82 := by
  sorry

end NUMINAMATH_CALUDE_subtraction_problem_l1705_170589


namespace NUMINAMATH_CALUDE_vertex_of_quadratic_l1705_170527

-- Define the quadratic function
def f (x : ℝ) : ℝ := -2 * (x - 1)^2 - 3

-- Theorem stating that the vertex of the quadratic function is at (1, -3)
theorem vertex_of_quadratic :
  ∃ (x y : ℝ), x = 1 ∧ y = -3 ∧ ∀ (t : ℝ), f t ≤ f x :=
sorry

end NUMINAMATH_CALUDE_vertex_of_quadratic_l1705_170527


namespace NUMINAMATH_CALUDE_reflection_line_sum_l1705_170596

/-- Given a line y = mx + b, if the reflection of point (2,3) across this line
    is (4,9), then m + b = 20/3 -/
theorem reflection_line_sum (m b : ℚ) : 
  (∀ (x y : ℚ), y = m * x + b →
    (2 + (2 * m * (m * 2 + b - 3) / (1 + m^2)) = 4 ∧
     3 + (2 * (m * 2 + b - 3) / (1 + m^2)) = 9)) →
  m + b = 20/3 := by sorry

end NUMINAMATH_CALUDE_reflection_line_sum_l1705_170596


namespace NUMINAMATH_CALUDE_sum_of_integers_l1705_170556

theorem sum_of_integers (x y : ℕ+) 
  (h1 : x^2 + y^2 = 145)
  (h2 : x * y = 40) : 
  x + y = 15 := by sorry

end NUMINAMATH_CALUDE_sum_of_integers_l1705_170556


namespace NUMINAMATH_CALUDE_number_ratio_l1705_170584

theorem number_ratio (x y z : ℝ) (k : ℝ) : 
  x = 18 →
  y = k * x →
  z = 2 * y →
  (x + y + z) / 3 = 78 →
  y / x = 4 := by
sorry

end NUMINAMATH_CALUDE_number_ratio_l1705_170584


namespace NUMINAMATH_CALUDE_central_angle_values_l1705_170501

/-- A circular sector with perimeter p and area a -/
structure CircularSector where
  p : ℝ  -- perimeter
  a : ℝ  -- area
  h_p_pos : p > 0
  h_a_pos : a > 0

/-- The central angle (in radians) of a circular sector -/
def central_angle (s : CircularSector) : Set ℝ :=
  {θ : ℝ | ∃ r : ℝ, r > 0 ∧ 2 * r + r * θ = s.p ∧ 1/2 * r^2 * θ = s.a}

/-- Theorem: For a circular sector with perimeter 6 and area 2, 
    the central angle is either 1 or 4 radians -/
theorem central_angle_values (s : CircularSector) 
  (h_p : s.p = 6) (h_a : s.a = 2) : 
  central_angle s = {1, 4} := by sorry

end NUMINAMATH_CALUDE_central_angle_values_l1705_170501


namespace NUMINAMATH_CALUDE_certain_number_proof_l1705_170518

theorem certain_number_proof (n : ℕ) (h1 : n > 0) :
  let m := 72 * 14
  Nat.gcd m 72 = 72 :=
by sorry

end NUMINAMATH_CALUDE_certain_number_proof_l1705_170518


namespace NUMINAMATH_CALUDE_four_number_average_l1705_170577

theorem four_number_average (a b c d : ℝ) 
  (h1 : b + c + d = 24)
  (h2 : a + c + d = 36)
  (h3 : a + b + d = 28)
  (h4 : a + b + c = 32) :
  (a + b + c + d) / 4 = 10 := by
sorry

end NUMINAMATH_CALUDE_four_number_average_l1705_170577


namespace NUMINAMATH_CALUDE_base_conversion_1729_to_base7_l1705_170568

/-- Converts a list of digits in base 7 to a natural number in base 10 -/
def fromBase7 (digits : List Nat) : Nat :=
  digits.foldl (fun acc d => 7 * acc + d) 0

/-- Theorem: 1729 in base 10 is equal to 5020 in base 7 -/
theorem base_conversion_1729_to_base7 :
  1729 = fromBase7 [5, 0, 2, 0] := by
  sorry

#eval fromBase7 [5, 0, 2, 0]  -- Should output 1729

end NUMINAMATH_CALUDE_base_conversion_1729_to_base7_l1705_170568


namespace NUMINAMATH_CALUDE_two_intersection_points_l1705_170569

def quadratic_function (c : ℝ) (x : ℝ) : ℝ := 2*x^2 - 3*x - c

theorem two_intersection_points (c : ℝ) (h : c > 0) :
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ quadratic_function c x₁ = 0 ∧ quadratic_function c x₂ = 0 ∧
  ∀ x : ℝ, quadratic_function c x = 0 → x = x₁ ∨ x = x₂ :=
sorry

end NUMINAMATH_CALUDE_two_intersection_points_l1705_170569


namespace NUMINAMATH_CALUDE_video_votes_l1705_170585

theorem video_votes (total_votes : ℕ) (likes : ℕ) (dislikes : ℕ) (score : ℤ) : 
  total_votes = likes + dislikes →
  likes = (6 : ℕ) * total_votes / 10 →
  dislikes = (4 : ℕ) * total_votes / 10 →
  score = likes - dislikes →
  score = 150 →
  total_votes = 750 := by
sorry


end NUMINAMATH_CALUDE_video_votes_l1705_170585


namespace NUMINAMATH_CALUDE_harry_sister_stamp_ratio_l1705_170530

/-- Proves the ratio of Harry's stamps to his sister's stamps -/
theorem harry_sister_stamp_ratio :
  let total_stamps : ℕ := 240
  let sister_stamps : ℕ := 60
  let harry_stamps : ℕ := total_stamps - sister_stamps
  (harry_stamps : ℚ) / sister_stamps = 3 := by
  sorry

end NUMINAMATH_CALUDE_harry_sister_stamp_ratio_l1705_170530


namespace NUMINAMATH_CALUDE_line_division_theorem_l1705_170595

/-- A line on a plane represented by ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Check if three lines divide the plane into six parts -/
def divides_into_six_parts (l1 l2 l3 : Line) : Prop :=
  sorry

/-- The set of k values that satisfy the condition -/
def k_values : Set ℝ := {0, -1, -2}

/-- Theorem stating the relationship between the lines and k values -/
theorem line_division_theorem (k : ℝ) :
  let l1 : Line := ⟨1, -2, 1⟩
  let l2 : Line := ⟨1, 0, -1⟩
  let l3 : Line := ⟨1, k, 0⟩
  divides_into_six_parts l1 l2 l3 → k ∈ k_values := by
  sorry

end NUMINAMATH_CALUDE_line_division_theorem_l1705_170595


namespace NUMINAMATH_CALUDE_quadratic_inequality_empty_solution_set_l1705_170505

/-- The solution set of the quadratic inequality kx^2 + 2kx + 2 < 0 is empty if and only if k is in the closed interval [0, 2]. -/
theorem quadratic_inequality_empty_solution_set (k : ℝ) : 
  (∀ x : ℝ, k * x^2 + 2 * k * x + 2 ≥ 0) ↔ 0 ≤ k ∧ k ≤ 2 :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_empty_solution_set_l1705_170505


namespace NUMINAMATH_CALUDE_angle_between_a_and_b_l1705_170512

/-- The angle between two 3D vectors -/
def angle_between_vectors (a b : ℝ × ℝ × ℝ) : ℝ := by sorry

/-- The vector a -/
def a : ℝ × ℝ × ℝ := (1, 1, -4)

/-- The vector b -/
def b : ℝ × ℝ × ℝ := (1, -2, 2)

/-- The theorem stating that the angle between vectors a and b is 135 degrees -/
theorem angle_between_a_and_b : 
  angle_between_vectors a b = 135 * Real.pi / 180 := by sorry

end NUMINAMATH_CALUDE_angle_between_a_and_b_l1705_170512


namespace NUMINAMATH_CALUDE_triangle_angle_measure_l1705_170576

theorem triangle_angle_measure (A B C : ℝ) : 
  A > 0 → B > 0 → C > 0 →
  A + B + C = 180 →
  C = 2 * B →
  B = A / 3 →
  A = 90 := by sorry

end NUMINAMATH_CALUDE_triangle_angle_measure_l1705_170576


namespace NUMINAMATH_CALUDE_condition_equivalence_l1705_170598

theorem condition_equivalence (p q : Prop) :
  (¬(p ∧ q) ∧ (p ∨ q)) ↔ (p ≠ q) :=
sorry

end NUMINAMATH_CALUDE_condition_equivalence_l1705_170598


namespace NUMINAMATH_CALUDE_triangle_problem_l1705_170534

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
theorem triangle_problem (A B C : ℝ) (a b c : ℝ) :
  -- Condition: sin²A + sin A sin C + sin²C + cos²B = 1
  (Real.sin A)^2 + (Real.sin A) * (Real.sin C) + (Real.sin C)^2 + (Real.cos B)^2 = 1 →
  -- Condition: a = 5
  a = 5 →
  -- Condition: b = 7
  b = 7 →
  -- Prove: B = 2π/3
  B = 2 * Real.pi / 3 ∧
  -- Prove: sin C = 3√3/14
  Real.sin C = 3 * Real.sqrt 3 / 14 := by
  sorry


end NUMINAMATH_CALUDE_triangle_problem_l1705_170534


namespace NUMINAMATH_CALUDE_smallest_gcd_yz_l1705_170508

theorem smallest_gcd_yz (x y z : ℕ+) 
  (hxy : Nat.gcd x.val y.val = 270)
  (hxz : Nat.gcd x.val z.val = 105) :
  ∃ (y' z' : ℕ+), 
    Nat.gcd y'.val z'.val = 15 ∧
    (∀ (y'' z'' : ℕ+), 
      Nat.gcd x.val y''.val = 270 → 
      Nat.gcd x.val z''.val = 105 → 
      Nat.gcd y''.val z''.val ≥ 15) :=
sorry

end NUMINAMATH_CALUDE_smallest_gcd_yz_l1705_170508


namespace NUMINAMATH_CALUDE_remi_bottle_capacity_l1705_170591

/-- The capacity of Remi's water bottle in ounces -/
def bottle_capacity : ℕ := 20

/-- The number of times Remi refills his bottle per day -/
def refills_per_day : ℕ := 3

/-- The number of days Remi drinks from his bottle -/
def days : ℕ := 7

/-- The amount of water Remi spills in ounces -/
def spilled_water : ℕ := 5 + 8

/-- The total amount of water Remi drinks in ounces -/
def total_water_drunk : ℕ := 407

/-- Theorem stating that given the conditions, Remi's water bottle capacity is 20 ounces -/
theorem remi_bottle_capacity :
  bottle_capacity * refills_per_day * days - spilled_water = total_water_drunk :=
by sorry

end NUMINAMATH_CALUDE_remi_bottle_capacity_l1705_170591


namespace NUMINAMATH_CALUDE_system_solution_l1705_170551

theorem system_solution (x y : ℝ) (eq1 : x + 2*y = 6) (eq2 : 2*x + y = 21) : x + y = 9 := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l1705_170551


namespace NUMINAMATH_CALUDE_division_problem_l1705_170520

theorem division_problem : ∃ (dividend : Nat) (divisor : Nat),
  dividend = 10004678 ∧ 
  divisor = 142 ∧ 
  100 ≤ divisor ∧ 
  divisor < 1000 ∧
  10000000 ≤ dividend ∧ 
  dividend < 100000000 ∧
  dividend / divisor = 70709 := by
  sorry

end NUMINAMATH_CALUDE_division_problem_l1705_170520


namespace NUMINAMATH_CALUDE_arithmetic_progression_cos_sum_l1705_170537

/-- An arithmetic progression is a sequence where the difference between
    successive terms is constant. -/
def is_arithmetic_progression (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_progression_cos_sum (a : ℕ → ℝ) :
  is_arithmetic_progression a →
  a 1 + a 7 + a 13 = 4 * Real.pi →
  Real.cos (a 2 + a 12) = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_progression_cos_sum_l1705_170537


namespace NUMINAMATH_CALUDE_max_sum_of_squares_l1705_170539

/-- Given that x₁ and x₂ are real roots of the equation x² - (k-2)x + (k² + 3k + 5) = 0,
    where k is a real number, prove that the maximum value of x₁² + x₂² is 18. -/
theorem max_sum_of_squares (k : ℝ) (x₁ x₂ : ℝ) 
    (h₁ : x₁^2 - (k-2)*x₁ + (k^2 + 3*k + 5) = 0)
    (h₂ : x₂^2 - (k-2)*x₂ + (k^2 + 3*k + 5) = 0)
    (h₃ : x₁ ≠ x₂) : 
  ∃ (M : ℝ), M = 18 ∧ x₁^2 + x₂^2 ≤ M :=
by sorry

end NUMINAMATH_CALUDE_max_sum_of_squares_l1705_170539


namespace NUMINAMATH_CALUDE_rectangles_in_5x5_grid_l1705_170540

/-- The number of rectangles in a n×n grid -/
def rectangles_in_grid (n : ℕ) : ℕ := (n.choose 2) * (n.choose 2)

/-- Theorem: The number of rectangles in a 5×5 grid is 100 -/
theorem rectangles_in_5x5_grid : rectangles_in_grid 5 = 100 := by
  sorry

end NUMINAMATH_CALUDE_rectangles_in_5x5_grid_l1705_170540


namespace NUMINAMATH_CALUDE_sixty_degrees_is_hundred_clerts_l1705_170574

/-- Represents the number of clerts in a full circle in the Martian system -/
def full_circle_clerts : ℕ := 600

/-- Represents the number of degrees in a full circle in the Earth system -/
def full_circle_degrees : ℕ := 360

/-- Converts degrees to clerts -/
def degrees_to_clerts (degrees : ℕ) : ℚ :=
  (degrees : ℚ) * full_circle_clerts / full_circle_degrees

theorem sixty_degrees_is_hundred_clerts :
  degrees_to_clerts 60 = 100 := by sorry

end NUMINAMATH_CALUDE_sixty_degrees_is_hundred_clerts_l1705_170574


namespace NUMINAMATH_CALUDE_smallest_number_of_eggs_l1705_170557

/-- Represents the number of eggs in a container --/
def EggsPerContainer := 12

/-- Represents the number of containers with fewer eggs --/
def FewerEggsContainers := 3

/-- Represents the number of eggs in containers with fewer eggs --/
def EggsInFewerEggsContainers := 10

/-- Calculates the total number of eggs given the number of containers --/
def totalEggs (numContainers : ℕ) : ℕ :=
  numContainers * EggsPerContainer - FewerEggsContainers * (EggsPerContainer - EggsInFewerEggsContainers)

theorem smallest_number_of_eggs :
  ∃ (n : ℕ), (n > 100 ∧ totalEggs n = 102 ∧ ∀ m, m > 100 → totalEggs m ≥ 102) := by
  sorry

end NUMINAMATH_CALUDE_smallest_number_of_eggs_l1705_170557


namespace NUMINAMATH_CALUDE_symmetric_circle_equation_l1705_170566

/-- The equation of a circle symmetric to (x+2)^2 + y^2 = 5 with respect to y = x -/
theorem symmetric_circle_equation :
  let original_circle := (fun (x y : ℝ) => (x + 2)^2 + y^2 = 5)
  let symmetry_line := (fun (x y : ℝ) => y = x)
  let symmetric_circle := (fun (x y : ℝ) => x^2 + (y + 2)^2 = 5)
  ∀ x y : ℝ, symmetric_circle x y ↔ 
    ∃ x' y' : ℝ, original_circle x' y' ∧ 
    ((x + y = x' + y') ∧ (y - x = x' - y')) :=
by sorry


end NUMINAMATH_CALUDE_symmetric_circle_equation_l1705_170566


namespace NUMINAMATH_CALUDE_sqrt_equation_average_zero_l1705_170529

theorem sqrt_equation_average_zero :
  let solutions := {x : ℝ | Real.sqrt (3 * x^2 + 4) = Real.sqrt 40}
  ∃ (x₁ x₂ : ℝ), x₁ ∈ solutions ∧ x₂ ∈ solutions ∧ x₁ ≠ x₂ ∧
  ∀ x ∈ solutions, x = x₁ ∨ x = x₂ ∧
  (x₁ + x₂) / 2 = 0 :=
by sorry

end NUMINAMATH_CALUDE_sqrt_equation_average_zero_l1705_170529


namespace NUMINAMATH_CALUDE_intersection_dot_product_l1705_170519

-- Define the line l: 4x + 3y - 5 = 0
def line_l (x y : ℝ) : Prop := 4 * x + 3 * y - 5 = 0

-- Define the circle C: x² + y² - 4 = 0
def circle_C (x y : ℝ) : Prop := x^2 + y^2 - 4 = 0

-- Define the intersection points A and B
def is_intersection (x y : ℝ) : Prop := line_l x y ∧ circle_C x y

-- Define the origin O
def origin : ℝ × ℝ := (0, 0)

-- Theorem statement
theorem intersection_dot_product :
  ∃ (A B : ℝ × ℝ),
    is_intersection A.1 A.2 ∧
    is_intersection B.1 B.2 ∧
    A ≠ B ∧
    (A.1 * B.1 + A.2 * B.2 = -2) :=
sorry

end NUMINAMATH_CALUDE_intersection_dot_product_l1705_170519


namespace NUMINAMATH_CALUDE_area_transformation_l1705_170544

-- Define a function representing the area between a curve and the x-axis
noncomputable def area_between_curve_and_xaxis (f : ℝ → ℝ) : ℝ := sorry

-- Theorem statement
theorem area_transformation (f : ℝ → ℝ) (h : area_between_curve_and_xaxis f = 12) :
  area_between_curve_and_xaxis (λ x => 4 * f (x + 3)) = 48 :=
by sorry

end NUMINAMATH_CALUDE_area_transformation_l1705_170544


namespace NUMINAMATH_CALUDE_pyramid_base_edge_length_l1705_170575

theorem pyramid_base_edge_length 
  (r : ℝ) 
  (h : ℝ) 
  (hemisphere_radius : r = 3) 
  (pyramid_height : h = 4) 
  (hemisphere_tangent : True)  -- This represents the tangency condition
  : ∃ (s : ℝ), s = (12 * Real.sqrt 14) / 7 ∧ s > 0 := by
  sorry

end NUMINAMATH_CALUDE_pyramid_base_edge_length_l1705_170575


namespace NUMINAMATH_CALUDE_tan_160_gt_tan_neg_23_l1705_170536

theorem tan_160_gt_tan_neg_23 : Real.tan (160 * π / 180) > Real.tan (-23 * π / 180) := by
  sorry

end NUMINAMATH_CALUDE_tan_160_gt_tan_neg_23_l1705_170536


namespace NUMINAMATH_CALUDE_range_of_a_l1705_170592

-- Define the propositions p and q
def p (a : ℝ) : Prop := ∀ x ∈ Set.Icc 1 2, x^2 - a ≥ 0

def q (a : ℝ) : Prop := ∃ x : ℝ, x^2 + 2*a*x + 2 - a = 0

-- State the theorem
theorem range_of_a (a : ℝ) (h : p a ∧ q a) : a = 1 ∨ a ≤ -2 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l1705_170592


namespace NUMINAMATH_CALUDE_whale_first_hour_consumption_l1705_170558

/-- Represents the whale's plankton consumption pattern --/
structure WhaleConsumption where
  duration : Nat
  hourlyIncrease : Nat
  totalConsumption : Nat
  sixthHourConsumption : Nat

/-- Calculates the first hour's consumption given the whale's consumption pattern --/
def firstHourConsumption (w : WhaleConsumption) : Nat :=
  w.sixthHourConsumption - (w.hourlyIncrease * 5)

/-- Theorem stating that for the given whale consumption pattern, 
    the first hour's consumption is 38 kilos --/
theorem whale_first_hour_consumption 
  (w : WhaleConsumption) 
  (h1 : w.duration = 9)
  (h2 : w.hourlyIncrease = 3)
  (h3 : w.totalConsumption = 450)
  (h4 : w.sixthHourConsumption = 53) : 
  firstHourConsumption w = 38 := by
  sorry

#eval firstHourConsumption ⟨9, 3, 450, 53⟩

end NUMINAMATH_CALUDE_whale_first_hour_consumption_l1705_170558


namespace NUMINAMATH_CALUDE_min_red_cells_correct_l1705_170593

/-- Minimum number of red cells needed in an n x n grid such that at least one red cell remains
    after erasing any 2 rows and 2 columns -/
def min_red_cells (n : ℕ) : ℕ :=
  if n = 4 then 7 else n + 3

theorem min_red_cells_correct (n : ℕ) (h : n ≥ 4) :
  ∀ (red_cells : Finset (Fin n × Fin n)),
    (∀ (rows cols : Finset (Fin n)), rows.card = 2 → cols.card = 2 →
      ∃ (i j : Fin n), i ∉ rows ∧ j ∉ cols ∧ (i, j) ∈ red_cells) →
    red_cells.card ≥ min_red_cells n :=
by sorry

end NUMINAMATH_CALUDE_min_red_cells_correct_l1705_170593


namespace NUMINAMATH_CALUDE_triangle_problem_l1705_170578

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the theorem
theorem triangle_problem (t : Triangle) 
  (h1 : (2 * t.b - t.c) * Real.cos t.A = t.a * Real.cos t.C)
  (h2 : t.a = Real.sqrt 13)
  (h3 : t.b + t.c = 5) :
  t.A = π / 3 ∧ 
  (1 / 2 : ℝ) * t.b * t.c * Real.sin t.A = Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_triangle_problem_l1705_170578


namespace NUMINAMATH_CALUDE_bucket_weight_l1705_170528

theorem bucket_weight (p q : ℝ) : ℝ :=
  let one_quarter_full := p
  let three_quarters_full := q
  let full_weight := -1/2 * p + 3/2 * q
  full_weight

#check bucket_weight

end NUMINAMATH_CALUDE_bucket_weight_l1705_170528


namespace NUMINAMATH_CALUDE_star_difference_equals_28_l1705_170507

def star (a b : ℝ) : ℝ := a^2 + 2*a*b + b^2

theorem star_difference_equals_28 : (star 3 5) - (star 2 4) = 28 := by
  sorry

end NUMINAMATH_CALUDE_star_difference_equals_28_l1705_170507


namespace NUMINAMATH_CALUDE_parabola_equation_l1705_170545

/-- A parabola is defined by the equation y = ax^2 + bx + c where a, b, and c are real numbers and a ≠ 0 -/
def Parabola (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

/-- A parabola opens upwards if a > 0 -/
def OpensUpwards (a b c : ℝ) : Prop := a > 0

/-- A parabola intersects the y-axis at the point (0, y) if y = c -/
def IntersectsYAxisAt (a b c y : ℝ) : Prop := c = y

theorem parabola_equation : ∃ (a b : ℝ), 
  OpensUpwards a b 2 ∧ 
  IntersectsYAxisAt a b 2 2 ∧ 
  (∀ x, Parabola a b 2 x = x^2 + 2) := by sorry

end NUMINAMATH_CALUDE_parabola_equation_l1705_170545


namespace NUMINAMATH_CALUDE_arithmetic_computation_l1705_170561

theorem arithmetic_computation : -7 * 3 - (-4 * -2) + (-9 * -6) / 3 = -11 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_computation_l1705_170561


namespace NUMINAMATH_CALUDE_mean_chocolate_sales_l1705_170526

def week1_sales : ℕ := 75
def week2_sales : ℕ := 67
def week3_sales : ℕ := 75
def week4_sales : ℕ := 70
def week5_sales : ℕ := 68
def num_weeks : ℕ := 5

def total_sales : ℕ := week1_sales + week2_sales + week3_sales + week4_sales + week5_sales

theorem mean_chocolate_sales :
  (total_sales : ℚ) / num_weeks = 71 := by sorry

end NUMINAMATH_CALUDE_mean_chocolate_sales_l1705_170526


namespace NUMINAMATH_CALUDE_triangle_isosceles_if_cosine_condition_l1705_170506

-- Define a triangle ABC
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  a : ℝ
  b : ℝ
  c : ℝ

-- Define what it means for a triangle to be isosceles
def isIsosceles (t : Triangle) : Prop :=
  t.A = t.B ∨ t.B = t.C ∨ t.A = t.C

-- State the theorem
theorem triangle_isosceles_if_cosine_condition (t : Triangle) 
  (h : t.a * Real.cos t.B = t.b * Real.cos t.A) : 
  isIsosceles t := by
  sorry

end NUMINAMATH_CALUDE_triangle_isosceles_if_cosine_condition_l1705_170506


namespace NUMINAMATH_CALUDE_equation_solution_for_all_y_l1705_170513

theorem equation_solution_for_all_y :
  ∃! x : ℚ, ∀ y : ℚ, 10 * x * y - 15 * y + 4 * x - 6 = 0 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_equation_solution_for_all_y_l1705_170513


namespace NUMINAMATH_CALUDE_bicycle_journey_l1705_170538

theorem bicycle_journey (t₅ t₁₅ : ℝ) (h_positive : t₅ > 0 ∧ t₁₅ > 0) :
  (5 * t₅ + 15 * t₁₅) / (t₅ + t₁₅) = 10 → t₁₅ / (t₅ + t₁₅) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_bicycle_journey_l1705_170538


namespace NUMINAMATH_CALUDE_units_digit_sum_of_powers_l1705_170572

theorem units_digit_sum_of_powers : (42^5 + 24^5 + 2^5) % 10 = 8 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_sum_of_powers_l1705_170572
