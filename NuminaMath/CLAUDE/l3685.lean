import Mathlib

namespace NUMINAMATH_CALUDE_problem_statement_l3685_368556

noncomputable def f (x : ℝ) : ℝ := Real.log x + 1 / x

noncomputable def g (x : ℝ) : ℝ := x - Real.log x

theorem problem_statement (x x₁ x₂ : ℝ) :
  (∀ x > 0, f x ≥ 1) ∧
  (∀ x > 1, f x < g x) ∧
  (x₁ > x₂ ∧ x₁ > 0 ∧ x₂ > 0 ∧ g x₁ = g x₂ → x₁ * x₂ < 1) :=
by sorry

end NUMINAMATH_CALUDE_problem_statement_l3685_368556


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l3685_368592

theorem sufficient_not_necessary_condition (a b : ℝ) :
  (∀ a b : ℝ, a > 1 ∧ b > 2 → a + b > 3) ∧
  (∃ a b : ℝ, a + b > 3 ∧ ¬(a > 1 ∧ b > 2)) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l3685_368592


namespace NUMINAMATH_CALUDE_two_heads_probability_l3685_368541

/-- The probability of getting heads on a single fair coin toss -/
def prob_heads : ℚ := 1 / 2

/-- The probability of getting two heads when tossing two fair coins simultaneously -/
def prob_two_heads : ℚ := prob_heads * prob_heads

/-- Theorem: The probability of getting two heads when tossing two fair coins simultaneously is 1/4 -/
theorem two_heads_probability : prob_two_heads = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_two_heads_probability_l3685_368541


namespace NUMINAMATH_CALUDE_third_number_problem_l3685_368587

theorem third_number_problem (x : ℝ) : 
  (14 + 32 + x) / 3 = (21 + 47 + 22) / 3 + 3 → x = 53 := by
  sorry

end NUMINAMATH_CALUDE_third_number_problem_l3685_368587


namespace NUMINAMATH_CALUDE_absolute_value_equality_l3685_368531

theorem absolute_value_equality (x : ℝ) : |x - 2| = |x + 3| → x = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_equality_l3685_368531


namespace NUMINAMATH_CALUDE_divisibility_of_concatenated_numbers_l3685_368597

theorem divisibility_of_concatenated_numbers (a b : ℕ) : 
  100 ≤ a ∧ a < 1000 →
  100 ≤ b ∧ b < 1000 →
  37 ∣ (a + b) →
  37 ∣ (1000 * a + b) := by
sorry

end NUMINAMATH_CALUDE_divisibility_of_concatenated_numbers_l3685_368597


namespace NUMINAMATH_CALUDE_steve_monday_pounds_l3685_368564

/-- The amount of money Steve wants to make in total -/
def total_money : ℕ := 100

/-- The pay rate per pound of lingonberries -/
def pay_rate : ℕ := 2

/-- The number of pounds Steve picked on Thursday -/
def thursday_pounds : ℕ := 18

/-- The factor by which Tuesday's harvest was greater than Monday's -/
def tuesday_factor : ℕ := 3

theorem steve_monday_pounds : 
  ∃ (monday_pounds : ℕ), 
    monday_pounds + tuesday_factor * monday_pounds + thursday_pounds = total_money / pay_rate ∧ 
    monday_pounds = 8 := by
  sorry

end NUMINAMATH_CALUDE_steve_monday_pounds_l3685_368564


namespace NUMINAMATH_CALUDE_is_14th_term_l3685_368506

/-- The sequence term for a given index -/
def sequenceTerm (n : ℕ) : ℚ := (n + 3 : ℚ) / (n + 1 : ℚ)

/-- Theorem stating that 17/15 is the 14th term of the sequence -/
theorem is_14th_term : sequenceTerm 14 = 17 / 15 := by
  sorry

end NUMINAMATH_CALUDE_is_14th_term_l3685_368506


namespace NUMINAMATH_CALUDE_freddy_travel_time_l3685_368554

/-- Represents the travel details of a person --/
structure TravelDetails where
  startCity : String
  endCity : String
  distance : Real
  time : Real

/-- The problem setup --/
def problem : Prop := ∃ (eddySpeed freddySpeed : Real),
  let eddy : TravelDetails := ⟨"A", "B", 900, 3⟩
  let freddy : TravelDetails := ⟨"A", "C", 300, freddySpeed / 300⟩
  eddySpeed = eddy.distance / eddy.time ∧
  eddySpeed / freddySpeed = 4 ∧
  freddy.time = 4

/-- The theorem to be proved --/
theorem freddy_travel_time : problem := by sorry

end NUMINAMATH_CALUDE_freddy_travel_time_l3685_368554


namespace NUMINAMATH_CALUDE_prime_divisor_of_fermat_number_l3685_368539

theorem prime_divisor_of_fermat_number (n : ℕ) (p : ℕ) (h_prime : Nat.Prime p) 
  (h_divides : p ∣ 2^(2^n) + 1) : 2^(n+1) ∣ p - 1 := by
  sorry

end NUMINAMATH_CALUDE_prime_divisor_of_fermat_number_l3685_368539


namespace NUMINAMATH_CALUDE_barbaras_score_l3685_368504

theorem barbaras_score (total_students : ℕ) (students_without_barbara : ℕ) 
  (avg_without_barbara : ℚ) (avg_with_barbara : ℚ) :
  total_students = 20 →
  students_without_barbara = 19 →
  avg_without_barbara = 78 →
  avg_with_barbara = 79 →
  (total_students * avg_with_barbara - students_without_barbara * avg_without_barbara : ℚ) = 98 := by
  sorry

#check barbaras_score

end NUMINAMATH_CALUDE_barbaras_score_l3685_368504


namespace NUMINAMATH_CALUDE_village_population_is_100_l3685_368581

/-- Represents the number of people in a youth summer village with specific characteristics. -/
def village_population (total : ℕ) (not_working : ℕ) (with_families : ℕ) (shower_singers : ℕ) (working_no_family_singers : ℕ) : Prop :=
  not_working = 50 ∧
  with_families = 25 ∧
  shower_singers = 75 ∧
  working_no_family_singers = 50 ∧
  total = not_working + with_families + shower_singers - working_no_family_singers

theorem village_population_is_100 :
  ∃ (total : ℕ), village_population total 50 25 75 50 ∧ total = 100 := by
  sorry

end NUMINAMATH_CALUDE_village_population_is_100_l3685_368581


namespace NUMINAMATH_CALUDE_digit_equation_solution_l3685_368543

theorem digit_equation_solution : ∃! (X : ℕ), X < 10 ∧ (510 : ℚ) / X = 40 + 3 * X :=
by sorry

end NUMINAMATH_CALUDE_digit_equation_solution_l3685_368543


namespace NUMINAMATH_CALUDE_max_value_of_function_l3685_368582

theorem max_value_of_function (x : ℝ) (hx : x < 0) :
  ∃ (max : ℝ), max = -4 * Real.sqrt 3 ∧ ∀ y, y = 3 * x + 4 / x → y ≤ max :=
sorry

end NUMINAMATH_CALUDE_max_value_of_function_l3685_368582


namespace NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l3685_368516

/-- Given a geometric sequence {a_n} with common ratio q, prove that if the sum of the first n terms S_n
    satisfies S_2 = 2a_2 + 3 and S_3 = 2a_3 + 3, then q = 2. -/
theorem geometric_sequence_common_ratio
  (a : ℕ → ℝ)  -- The geometric sequence
  (q : ℝ)      -- The common ratio
  (S : ℕ → ℝ)  -- The sum function
  (h1 : ∀ n, a (n + 1) = a n * q)  -- Definition of geometric sequence
  (h2 : ∀ n, S n = (a 1) * (1 - q^n) / (1 - q))  -- Sum formula for geometric sequence
  (h3 : S 2 = 2 * a 2 + 3)  -- Given condition for S_2
  (h4 : S 3 = 2 * a 3 + 3)  -- Given condition for S_3
  : q = 2 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l3685_368516


namespace NUMINAMATH_CALUDE_count_ways_2024_l3685_368584

/-- The target sum -/
def target_sum : ℕ := 2024

/-- The set of allowed numbers -/
def allowed_numbers : Finset ℕ := {2, 3, 4}

/-- A function that counts the number of ways to express a target sum
    as a sum of non-negative integer multiples of allowed numbers,
    ignoring the order of summands -/
noncomputable def count_ways (target : ℕ) (allowed : Finset ℕ) : ℕ :=
  sorry

/-- The main theorem stating that there are 57231 ways to express 2024
    as a sum of non-negative integer multiples of 2, 3, and 4,
    ignoring the order of summands -/
theorem count_ways_2024 :
  count_ways target_sum allowed_numbers = 57231 :=
by sorry

end NUMINAMATH_CALUDE_count_ways_2024_l3685_368584


namespace NUMINAMATH_CALUDE_sum_a_b_is_8_l3685_368569

/-- A quadrilateral PQRS with specific properties -/
structure Quadrilateral where
  a : ℤ
  b : ℤ
  a_gt_b : a > b
  b_pos : b > 0
  is_rectangle : True  -- We assume PQRS is a rectangle
  area_is_32 : 2 * (a - b).natAbs * (a + b).natAbs = 32

/-- The sum of a and b in a quadrilateral with specific properties is 8 -/
theorem sum_a_b_is_8 (q : Quadrilateral) : q.a + q.b = 8 := by
  sorry

#check sum_a_b_is_8

end NUMINAMATH_CALUDE_sum_a_b_is_8_l3685_368569


namespace NUMINAMATH_CALUDE_cos_alpha_value_l3685_368594

theorem cos_alpha_value (α : Real) (h : Real.cos (Real.pi + α) = -1/3) : Real.cos α = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_cos_alpha_value_l3685_368594


namespace NUMINAMATH_CALUDE_red_peppers_weight_l3685_368571

/-- The weight of red peppers at Dale's Vegetarian Restaurant -/
def weight_red_peppers : ℝ := 5.666666666666667 - 2.8333333333333335

/-- Theorem: The weight of red peppers is equal to the total weight of peppers minus the weight of green peppers -/
theorem red_peppers_weight :
  weight_red_peppers = 5.666666666666667 - 2.8333333333333335 := by
  sorry

end NUMINAMATH_CALUDE_red_peppers_weight_l3685_368571


namespace NUMINAMATH_CALUDE_problem_solution_l3685_368598

theorem problem_solution : 
  (0.027 ^ (-1/3 : ℝ)) + (16 ^ 3) ^ (1/4 : ℝ) - 3⁻¹ + ((2 : ℝ).sqrt - 1) ^ (0 : ℝ) = 12 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3685_368598


namespace NUMINAMATH_CALUDE_condition_implies_isosceles_l3685_368518

-- Define a structure for a triangle in a plane
structure Triangle :=
  (A B C : ℝ × ℝ)

-- Define a function to represent a vector from one point to another
def vector (p q : ℝ × ℝ) : ℝ × ℝ := (q.1 - p.1, q.2 - p.2)

-- Define dot product for 2D vectors
def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

-- Define the condition given in the problem
def satisfies_condition (t : Triangle) : Prop :=
  ∀ O : ℝ × ℝ, 
    let OB := vector O t.B
    let OC := vector O t.C
    let OA := vector O t.A
    dot_product (OB - OC) (OB + OC - 2 • OA) = 0

-- Define what it means for a triangle to be isosceles
def is_isosceles (t : Triangle) : Prop :=
  let AB := vector t.A t.B
  let AC := vector t.A t.C
  dot_product AB AB = dot_product AC AC

-- State the theorem
theorem condition_implies_isosceles (t : Triangle) :
  satisfies_condition t → is_isosceles t :=
by
  sorry

end NUMINAMATH_CALUDE_condition_implies_isosceles_l3685_368518


namespace NUMINAMATH_CALUDE_division_problem_l3685_368580

theorem division_problem (a b c : ℚ) 
  (h1 : a / b = 3) 
  (h2 : b / c = 2/5) : 
  c / a = 5/6 := by sorry

end NUMINAMATH_CALUDE_division_problem_l3685_368580


namespace NUMINAMATH_CALUDE_audrey_heracles_age_difference_l3685_368583

/-- The age difference between Audrey and Heracles -/
def ageDifference (audreyAge : ℕ) (heraclesAge : ℕ) : ℕ :=
  audreyAge - heraclesAge

theorem audrey_heracles_age_difference :
  ∃ (audreyAge : ℕ),
    heraclesAge = 10 ∧
    audreyAge + 3 = 2 * heraclesAge ∧
    ageDifference audreyAge heraclesAge = 7 :=
by
  sorry

end NUMINAMATH_CALUDE_audrey_heracles_age_difference_l3685_368583


namespace NUMINAMATH_CALUDE_cat_gemstone_difference_l3685_368529

/-- Given three cats with gemstone collars, prove the difference between Spaatz's 
    gemstones and half of Frankie's gemstones. -/
theorem cat_gemstone_difference (binkie frankie spaatz : ℕ) : 
  binkie = 24 →
  spaatz = 1 →
  binkie = 4 * frankie →
  spaatz = frankie →
  spaatz - (frankie / 2) = 2 := by
  sorry

end NUMINAMATH_CALUDE_cat_gemstone_difference_l3685_368529


namespace NUMINAMATH_CALUDE_largest_among_four_l3685_368537

theorem largest_among_four (a b : ℝ) (h1 : b > a) (h2 : a > 0) (h3 : a + b = 1) :
  b > (1/2 : ℝ) ∧ b > 2*a*b ∧ b > a^2 + b^2 := by
  sorry

end NUMINAMATH_CALUDE_largest_among_four_l3685_368537


namespace NUMINAMATH_CALUDE_lucia_weekly_dance_cost_l3685_368505

/-- Represents the cost of dance classes for a week -/
def total_dance_cost (hip_hop_classes ballet_classes jazz_classes : ℕ) 
  (hip_hop_cost ballet_cost jazz_cost : ℕ) : ℕ :=
  hip_hop_classes * hip_hop_cost + ballet_classes * ballet_cost + jazz_classes * jazz_cost

/-- Proves that Lucia's weekly dance class cost is $52 -/
theorem lucia_weekly_dance_cost : 
  total_dance_cost 2 2 1 10 12 8 = 52 := by
  sorry

end NUMINAMATH_CALUDE_lucia_weekly_dance_cost_l3685_368505


namespace NUMINAMATH_CALUDE_range_of_m_l3685_368549

theorem range_of_m (f : ℝ → ℝ → ℝ) (x₀ : ℝ) (h_nonzero : x₀ ≠ 0) :
  (∀ m : ℝ, f m x = 9*x - m) →
  f x₀ x₀ = f 0 x₀ →
  ∃ m : ℝ, 0 < m ∧ m < 1/2 :=
sorry

end NUMINAMATH_CALUDE_range_of_m_l3685_368549


namespace NUMINAMATH_CALUDE_choir_members_count_l3685_368522

theorem choir_members_count : ∃! n : ℕ, 
  150 ≤ n ∧ n ≤ 300 ∧ 
  n % 10 = 6 ∧ 
  n % 11 = 6 := by
sorry

end NUMINAMATH_CALUDE_choir_members_count_l3685_368522


namespace NUMINAMATH_CALUDE_points_deducted_for_incorrect_l3685_368589

def test_questions : ℕ := 30
def correct_answer_points : ℕ := 20
def maria_final_score : ℕ := 325
def maria_correct_answers : ℕ := 19

theorem points_deducted_for_incorrect (deducted_points : ℕ) : 
  (maria_correct_answers * correct_answer_points) - 
  ((test_questions - maria_correct_answers) * deducted_points) = 
  maria_final_score → 
  deducted_points = 5 := by
sorry

end NUMINAMATH_CALUDE_points_deducted_for_incorrect_l3685_368589


namespace NUMINAMATH_CALUDE_coefficient_x_squared_in_expansion_l3685_368528

theorem coefficient_x_squared_in_expansion :
  (Finset.range 5).sum (fun k => (Nat.choose 4 k : ℤ) * (-2)^(4 - k) * (if k = 2 then 1 else 0)) = 24 := by
  sorry

end NUMINAMATH_CALUDE_coefficient_x_squared_in_expansion_l3685_368528


namespace NUMINAMATH_CALUDE_basketball_campers_count_l3685_368511

theorem basketball_campers_count (total_campers soccer_campers football_campers : ℕ) 
  (h1 : total_campers = 88)
  (h2 : soccer_campers = 32)
  (h3 : football_campers = 32) :
  total_campers - soccer_campers - football_campers = 24 := by
  sorry

end NUMINAMATH_CALUDE_basketball_campers_count_l3685_368511


namespace NUMINAMATH_CALUDE_savings_distribution_l3685_368527

/-- Calculates the amount each child receives from the couple's savings --/
theorem savings_distribution (husband_contribution : ℕ) (wife_contribution : ℕ)
  (husband_interval : ℕ) (wife_interval : ℕ) (months : ℕ) (days_per_month : ℕ)
  (savings_percentage : ℚ) (num_children : ℕ) :
  husband_contribution = 450 →
  wife_contribution = 315 →
  husband_interval = 10 →
  wife_interval = 5 →
  months = 8 →
  days_per_month = 30 →
  savings_percentage = 3/4 →
  num_children = 6 →
  (((months * days_per_month / husband_interval) * husband_contribution +
    (months * days_per_month / wife_interval) * wife_contribution) *
    savings_percentage / num_children : ℚ) = 3240 := by
  sorry

end NUMINAMATH_CALUDE_savings_distribution_l3685_368527


namespace NUMINAMATH_CALUDE_first_friend_shells_l3685_368559

/-- Proves the amount of shells added by the first friend given initial conditions --/
theorem first_friend_shells (initial_shells : ℕ) (second_friend_shells : ℕ) (total_shells : ℕ)
  (h1 : initial_shells = 5)
  (h2 : second_friend_shells = 17)
  (h3 : total_shells = 37)
  : total_shells - initial_shells - second_friend_shells = 15 := by
  sorry

end NUMINAMATH_CALUDE_first_friend_shells_l3685_368559


namespace NUMINAMATH_CALUDE_triangle_centroid_inequality_l3685_368562

/-- Given a triangle ABC with side lengths a, b, and c, centroid G, and an arbitrary point P,
    prove that a⋅PA³ + b⋅PB³ + c⋅PC³ ≥ 3abc⋅PG -/
theorem triangle_centroid_inequality (A B C P : ℝ × ℝ) 
    (a b c : ℝ) (h_positive : a > 0 ∧ b > 0 ∧ c > 0) :
  let G := ((A.1 + B.1 + C.1) / 3, (A.2 + B.2 + C.2) / 3)
  let PA := Real.sqrt ((P.1 - A.1)^2 + (P.2 - A.2)^2)
  let PB := Real.sqrt ((P.1 - B.1)^2 + (P.2 - B.2)^2)
  let PC := Real.sqrt ((P.1 - C.1)^2 + (P.2 - C.2)^2)
  let PG := Real.sqrt ((P.1 - G.1)^2 + (P.2 - G.2)^2)
  a * PA^3 + b * PB^3 + c * PC^3 ≥ 3 * a * b * c * PG := by
sorry


end NUMINAMATH_CALUDE_triangle_centroid_inequality_l3685_368562


namespace NUMINAMATH_CALUDE_vlads_pen_price_ratio_l3685_368585

/-- The ratio of gel pen price to ballpoint pen price given the conditions in Vlad's pen purchase problem -/
theorem vlads_pen_price_ratio :
  ∀ (x y : ℕ) (b g : ℝ),
  x > 0 → y > 0 → b > 0 → g > 0 →
  (x + y) * g = 4 * (x * b + y * g) →
  (x + y) * b = (1 / 2) * (x * b + y * g) →
  g = 8 * b := by
sorry

end NUMINAMATH_CALUDE_vlads_pen_price_ratio_l3685_368585


namespace NUMINAMATH_CALUDE_prob_sum_24_four_dice_l3685_368545

/-- The probability of rolling a specific number on a fair six-sided die -/
def prob_single_die : ℚ := 1 / 6

/-- The number of dice rolled -/
def num_dice : ℕ := 4

/-- The desired sum of the dice rolls -/
def target_sum : ℕ := 24

/-- The number of sides on each die -/
def die_sides : ℕ := 6

theorem prob_sum_24_four_dice : 
  (prob_single_die ^ num_dice : ℚ) = 1 / 1296 :=
sorry

end NUMINAMATH_CALUDE_prob_sum_24_four_dice_l3685_368545


namespace NUMINAMATH_CALUDE_bug_crawl_distance_l3685_368500

-- Define the bug's starting position
def start : Int := 3

-- Define the bug's first destination
def first_dest : Int := 9

-- Define the bug's final destination
def final_dest : Int := -4

-- Define the function to calculate distance between two points
def distance (a b : Int) : Nat := Int.natAbs (b - a)

-- Theorem statement
theorem bug_crawl_distance : 
  distance start first_dest + distance first_dest final_dest = 19 := by
  sorry

end NUMINAMATH_CALUDE_bug_crawl_distance_l3685_368500


namespace NUMINAMATH_CALUDE_fifteenth_term_of_sequence_l3685_368574

def geometric_sequence (a₁ : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a₁ * r^(n - 1)

theorem fifteenth_term_of_sequence (a₁ a₂ : ℚ) (h₁ : a₁ = 12) (h₂ : a₂ = 4) :
  geometric_sequence a₁ (a₂ / a₁) 15 = 12 / 4782969 := by
  sorry

end NUMINAMATH_CALUDE_fifteenth_term_of_sequence_l3685_368574


namespace NUMINAMATH_CALUDE_inequality_equivalence_l3685_368586

theorem inequality_equivalence (x : ℝ) : 
  (x / 4 ≤ 3 + x ∧ 3 + x < -3 * (1 + x)) ↔ -4 ≤ x ∧ x < -3/2 :=
by sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l3685_368586


namespace NUMINAMATH_CALUDE_marble_probability_l3685_368524

/-- The probability of drawing a red, blue, or green marble from a bag -/
theorem marble_probability (red blue green yellow : ℕ) : 
  red = 4 → blue = 3 → green = 2 → yellow = 6 → 
  (red + blue + green : ℚ) / (red + blue + green + yellow) = 0.6 := by
sorry

end NUMINAMATH_CALUDE_marble_probability_l3685_368524


namespace NUMINAMATH_CALUDE_altitude_sum_of_specific_triangle_l3685_368563

/-- The sum of altitudes of a triangle formed by the line 10x + 8y = 80 and coordinate axes --/
theorem altitude_sum_of_specific_triangle : 
  let line : ℝ → ℝ → Prop := λ x y => 10 * x + 8 * y = 80
  let triangle := {p : ℝ × ℝ | p.1 ≥ 0 ∧ p.2 ≥ 0 ∧ line p.1 p.2}
  let altitudes := 
    [10, 8, (40 : ℝ) / Real.sqrt 41]  -- altitudes to y-axis, x-axis, and hypotenuse
  (altitudes.sum : ℝ) = 18 + 40 / Real.sqrt 41 := by
  sorry

end NUMINAMATH_CALUDE_altitude_sum_of_specific_triangle_l3685_368563


namespace NUMINAMATH_CALUDE_charging_time_is_112_5_l3685_368535

/-- Represents the charging time for each device type -/
structure ChargingTimes where
  smartphone : ℝ
  tablet : ℝ
  laptop : ℝ

/-- Represents the charging percentages for each device -/
structure ChargingPercentages where
  smartphone : ℝ
  tablet : ℝ
  laptop : ℝ

/-- Calculates the total charging time given the full charging times and charging percentages -/
def totalChargingTime (times : ChargingTimes) (percentages : ChargingPercentages) : ℝ :=
  times.tablet * percentages.tablet +
  times.smartphone * percentages.smartphone +
  times.laptop * percentages.laptop

/-- Theorem stating that the total charging time is 112.5 minutes -/
theorem charging_time_is_112_5 (times : ChargingTimes) (percentages : ChargingPercentages) :
  times.smartphone = 26 →
  times.tablet = 53 →
  times.laptop = 80 →
  percentages.smartphone = 0.75 →
  percentages.tablet = 1 →
  percentages.laptop = 0.5 →
  totalChargingTime times percentages = 112.5 := by
  sorry

end NUMINAMATH_CALUDE_charging_time_is_112_5_l3685_368535


namespace NUMINAMATH_CALUDE_odd_even_intersection_empty_l3685_368595

def odd_integers : Set ℤ := {x | ∃ k : ℤ, x = 2 * k + 1}
def even_integers : Set ℤ := {x | ∃ k : ℤ, x = 2 * k}

theorem odd_even_intersection_empty :
  odd_integers ∩ even_integers = ∅ := by sorry

end NUMINAMATH_CALUDE_odd_even_intersection_empty_l3685_368595


namespace NUMINAMATH_CALUDE_expression_evaluation_equation_solutions_l3685_368525

-- Part 1
theorem expression_evaluation :
  |Real.sqrt 3 - 1| - 2 * Real.cos (60 * π / 180) + (Real.sqrt 3 - 2)^2 + Real.sqrt 12 = 5 - Real.sqrt 3 := by
  sorry

-- Part 2
theorem equation_solutions (x : ℝ) :
  2 * (x - 3)^2 = x^2 - 9 ↔ x = 3 ∨ x = 9 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_equation_solutions_l3685_368525


namespace NUMINAMATH_CALUDE_least_number_with_remainder_one_l3685_368565

theorem least_number_with_remainder_one (n : ℕ) : 
  (∀ m : ℕ, m > 0 ∧ m < 115 → (m % 38 ≠ 1 ∨ m % 3 ≠ 1)) ∧ 
  (115 % 38 = 1 ∧ 115 % 3 = 1) := by
  sorry

end NUMINAMATH_CALUDE_least_number_with_remainder_one_l3685_368565


namespace NUMINAMATH_CALUDE_simplify_square_roots_l3685_368521

theorem simplify_square_roots : Real.sqrt (5 * 3) * Real.sqrt (3^3 * 5^3) = 225 := by
  sorry

end NUMINAMATH_CALUDE_simplify_square_roots_l3685_368521


namespace NUMINAMATH_CALUDE_gecko_insect_consumption_l3685_368513

theorem gecko_insect_consumption (geckos lizards total_insects : ℕ) 
  (h1 : geckos = 5)
  (h2 : lizards = 3)
  (h3 : total_insects = 66) :
  ∃ (gecko_consumption : ℕ), 
    gecko_consumption * geckos + (2 * gecko_consumption) * lizards = total_insects ∧ 
    gecko_consumption = 6 := by
  sorry

end NUMINAMATH_CALUDE_gecko_insect_consumption_l3685_368513


namespace NUMINAMATH_CALUDE_parallel_vectors_t_value_l3685_368555

/-- Two vectors in ℝ² -/
def Vector2 := ℝ × ℝ

/-- Check if two vectors are parallel -/
def are_parallel (v w : Vector2) : Prop :=
  ∃ (k : ℝ), v.1 = k * w.1 ∧ v.2 = k * w.2

theorem parallel_vectors_t_value :
  ∀ (t : ℝ),
  let a : Vector2 := (t, -6)
  let b : Vector2 := (-3, 2)
  are_parallel a b → t = 9 := by
sorry

end NUMINAMATH_CALUDE_parallel_vectors_t_value_l3685_368555


namespace NUMINAMATH_CALUDE_extreme_value_implies_a_equals_negative_one_l3685_368573

/-- Given a function f(x) = ax^2 + x^2 that reaches an extreme value at x = -2,
    prove that a = -1 --/
theorem extreme_value_implies_a_equals_negative_one (a : ℝ) :
  let f : ℝ → ℝ := λ x ↦ a * x^2 + x^2
  (∃ ε > 0, ∀ x ∈ Set.Ioo (-2 - ε) (-2 + ε), f x ≤ f (-2) ∨ f x ≥ f (-2)) →
  a = -1 :=
by sorry

end NUMINAMATH_CALUDE_extreme_value_implies_a_equals_negative_one_l3685_368573


namespace NUMINAMATH_CALUDE_missing_number_proof_l3685_368509

theorem missing_number_proof (x : ℝ) : 
  x * 54 = 75625 → 
  ⌊x + 0.5⌋ = 1400 := by
sorry

end NUMINAMATH_CALUDE_missing_number_proof_l3685_368509


namespace NUMINAMATH_CALUDE_evaluate_expression_l3685_368510

theorem evaluate_expression (x y : ℝ) (hx : x = 4) (hy : y = 2) :
  y * (y - 2 * x)^2 = 72 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l3685_368510


namespace NUMINAMATH_CALUDE_function_domain_range_implies_interval_l3685_368523

open Real

theorem function_domain_range_implies_interval (f : ℝ → ℝ) (m n : ℝ) :
  (∀ x ∈ Set.Icc m n, f x ∈ Set.Icc (-1/2) (1/4)) →
  (∀ x, f x = sin x * sin (x + π/3) - 1/4) →
  m < n →
  π/3 ≤ n - m ∧ n - m ≤ 2*π/3 := by
  sorry

end NUMINAMATH_CALUDE_function_domain_range_implies_interval_l3685_368523


namespace NUMINAMATH_CALUDE_odd_function_symmetry_l3685_368517

/-- A function f is odd if f(-x) = -f(x) for all x -/
def IsOdd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

theorem odd_function_symmetry (f : ℝ → ℝ) (a : ℝ) (h : IsOdd f) :
  f (-a) = -f a := by sorry

end NUMINAMATH_CALUDE_odd_function_symmetry_l3685_368517


namespace NUMINAMATH_CALUDE_three_year_deposit_optimal_l3685_368572

/-- Represents the deposit options available --/
inductive DepositOption
  | OneYearRepeated
  | OneYearThenTwoYear
  | TwoYearThenOneYear
  | ThreeYear

/-- Calculates the final amount for a given deposit option --/
def calculateFinalAmount (option : DepositOption) (initialDeposit : ℝ) : ℝ :=
  match option with
  | .OneYearRepeated => initialDeposit * (1 + 0.0414 * 0.8)^3
  | .OneYearThenTwoYear => initialDeposit * (1 + 0.0414 * 0.8) * (1 + 0.0468 * 0.8 * 2)
  | .TwoYearThenOneYear => initialDeposit * (1 + 0.0468 * 0.8 * 2) * (1 + 0.0414 * 0.8)
  | .ThreeYear => initialDeposit * (1 + 0.0540 * 3 * 0.8)

/-- Theorem stating that the three-year fixed deposit option yields the highest return --/
theorem three_year_deposit_optimal (initialDeposit : ℝ) (h : initialDeposit > 0) :
  ∀ option : DepositOption, calculateFinalAmount .ThreeYear initialDeposit ≥ calculateFinalAmount option initialDeposit :=
by sorry

end NUMINAMATH_CALUDE_three_year_deposit_optimal_l3685_368572


namespace NUMINAMATH_CALUDE_difference_of_squares_65_35_l3685_368567

theorem difference_of_squares_65_35 : 65^2 - 35^2 = 3000 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_65_35_l3685_368567


namespace NUMINAMATH_CALUDE_tyrone_nickels_l3685_368578

/-- Represents the contents of Tyrone's piggy bank -/
structure PiggyBank where
  one_dollar_bills : Nat
  five_dollar_bills : Nat
  quarters : Nat
  dimes : Nat
  nickels : Nat
  pennies : Nat

/-- Calculates the total value in dollars of the contents of the piggy bank -/
def total_value (pb : PiggyBank) : Rat :=
  pb.one_dollar_bills + 
  5 * pb.five_dollar_bills + 
  (1/4) * pb.quarters + 
  (1/10) * pb.dimes + 
  (1/20) * pb.nickels + 
  (1/100) * pb.pennies

/-- Tyrone's piggy bank contents -/
def tyrone_piggy_bank : PiggyBank :=
  { one_dollar_bills := 2
  , five_dollar_bills := 1
  , quarters := 13
  , dimes := 20
  , nickels := 8  -- This is what we want to prove
  , pennies := 35 }

theorem tyrone_nickels : 
  total_value tyrone_piggy_bank = 13 := by sorry

end NUMINAMATH_CALUDE_tyrone_nickels_l3685_368578


namespace NUMINAMATH_CALUDE_cinnamon_amount_l3685_368534

/-- The amount of nutmeg used in tablespoons -/
def nutmeg : ℝ := 0.5

/-- The difference in tablespoons between cinnamon and nutmeg -/
def difference : ℝ := 0.17

/-- The amount of cinnamon used in tablespoons -/
def cinnamon : ℝ := nutmeg + difference

theorem cinnamon_amount : cinnamon = 0.67 := by
  sorry

end NUMINAMATH_CALUDE_cinnamon_amount_l3685_368534


namespace NUMINAMATH_CALUDE_odd_numbers_product_equality_l3685_368599

theorem odd_numbers_product_equality (a b c d : ℕ) : 
  Odd a → Odd b → Odd c → Odd d →
  a < b → b < c → c < d →
  a * d = b * c →
  ∃ k l : ℕ, a + d = 2^k ∧ b + c = 2^l →
  a = 1 := by sorry

end NUMINAMATH_CALUDE_odd_numbers_product_equality_l3685_368599


namespace NUMINAMATH_CALUDE_lakota_new_cd_count_l3685_368566

/-- The price of a used CD in dollars -/
def used_cd_price : ℚ := 9.99

/-- The total price of Lakota's purchase in dollars -/
def lakota_total : ℚ := 127.92

/-- The total price of Mackenzie's purchase in dollars -/
def mackenzie_total : ℚ := 133.89

/-- The number of used CDs Lakota bought -/
def lakota_used : ℕ := 2

/-- The number of new CDs Mackenzie bought -/
def mackenzie_new : ℕ := 3

/-- The number of used CDs Mackenzie bought -/
def mackenzie_used : ℕ := 8

/-- The number of new CDs Lakota bought -/
def lakota_new : ℕ := 6

theorem lakota_new_cd_count :
  ∃ (new_cd_price : ℚ),
    new_cd_price * lakota_new + used_cd_price * lakota_used = lakota_total ∧
    new_cd_price * mackenzie_new + used_cd_price * mackenzie_used = mackenzie_total :=
by sorry

end NUMINAMATH_CALUDE_lakota_new_cd_count_l3685_368566


namespace NUMINAMATH_CALUDE_birds_nest_eggs_l3685_368570

theorem birds_nest_eggs (x : ℕ) : 
  (2 * x + 3 + 4 = 17) → x = 5 := by sorry

end NUMINAMATH_CALUDE_birds_nest_eggs_l3685_368570


namespace NUMINAMATH_CALUDE_horner_third_step_value_l3685_368588

def f (x : ℝ) : ℝ := x^5 - 2*x^4 + 3*x^3 - 7*x^2 + 6*x - 3

def horner_step (n : ℕ) (x : ℝ) (coeffs : List ℝ) : ℝ :=
  match n, coeffs with
  | 0, _ => 0
  | n+1, a::rest => a + x * horner_step n x rest
  | _, _ => 0

theorem horner_third_step_value :
  let coeffs := [1, -2, 3, -7, 6, -3]
  let x := 2
  horner_step 3 x coeffs = -1 := by sorry

end NUMINAMATH_CALUDE_horner_third_step_value_l3685_368588


namespace NUMINAMATH_CALUDE_oil_drop_probability_l3685_368544

theorem oil_drop_probability (coin_diameter : ℝ) (hole_side_length : ℝ) : 
  coin_diameter = 2 → hole_side_length = 1 → 
  (hole_side_length^2) / ((coin_diameter/2)^2 * π) = 1/π :=
by sorry

end NUMINAMATH_CALUDE_oil_drop_probability_l3685_368544


namespace NUMINAMATH_CALUDE_evaluate_expression_l3685_368552

theorem evaluate_expression : 5 * 399 + 4 * 399 + 3 * 399 + 397 = 5185 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l3685_368552


namespace NUMINAMATH_CALUDE_postman_june_distance_l3685_368501

/-- Represents a step counter with a maximum count before resetting -/
structure StepCounter where
  max_count : ℕ
  resets : ℕ
  final_count : ℕ

/-- Calculates the total number of steps based on the step counter data -/
def total_steps (counter : StepCounter) : ℕ :=
  counter.max_count * counter.resets + counter.final_count

/-- Converts steps to miles given the number of steps per mile -/
def steps_to_miles (steps : ℕ) (steps_per_mile : ℕ) : ℕ :=
  steps / steps_per_mile

/-- Theorem stating that given the specified conditions, the total distance walked is 2615 miles -/
theorem postman_june_distance :
  let counter : StepCounter := ⟨100000, 52, 30000⟩
  let steps_per_mile : ℕ := 2000
  steps_to_miles (total_steps counter) steps_per_mile = 2615 := by
  sorry

end NUMINAMATH_CALUDE_postman_june_distance_l3685_368501


namespace NUMINAMATH_CALUDE_lines_parallel_iff_m_eq_one_l3685_368532

/-- Two lines in the plane -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The condition for two lines to be parallel -/
def parallel (l1 l2 : Line) : Prop :=
  l1.a * l2.b = l2.a * l1.b ∧ l1.a * l2.c ≠ l2.a * l1.c

/-- The first line: x + (1+m)y = 2-m -/
def line1 (m : ℝ) : Line :=
  { a := 1, b := 1 + m, c := m - 2 }

/-- The second line: 2mx + 4y = -16 -/
def line2 (m : ℝ) : Line :=
  { a := 2 * m, b := 4, c := 16 }

/-- The theorem stating that the lines are parallel iff m = 1 -/
theorem lines_parallel_iff_m_eq_one :
  ∀ m : ℝ, parallel (line1 m) (line2 m) ↔ m = 1 := by
  sorry

end NUMINAMATH_CALUDE_lines_parallel_iff_m_eq_one_l3685_368532


namespace NUMINAMATH_CALUDE_largest_multiple_of_11_below_negative_85_l3685_368575

theorem largest_multiple_of_11_below_negative_85 :
  ∀ n : ℤ, n % 11 = 0 → n < -85 → n ≤ -88 :=
by
  sorry

end NUMINAMATH_CALUDE_largest_multiple_of_11_below_negative_85_l3685_368575


namespace NUMINAMATH_CALUDE_sams_mystery_books_l3685_368507

/-- The number of mystery books Sam bought at the school's book fair -/
def mystery_books : ℕ := sorry

/-- The number of adventure books Sam bought -/
def adventure_books : ℕ := 13

/-- The number of used books Sam bought -/
def used_books : ℕ := 15

/-- The number of new books Sam bought -/
def new_books : ℕ := 15

/-- The total number of books Sam bought -/
def total_books : ℕ := used_books + new_books

theorem sams_mystery_books : 
  mystery_books = total_books - adventure_books ∧ 
  mystery_books = 17 := by sorry

end NUMINAMATH_CALUDE_sams_mystery_books_l3685_368507


namespace NUMINAMATH_CALUDE_elisa_dinner_cost_l3685_368538

/-- The amount of money Elisa had to pay for dinner with her friends -/
def dinner_cost (num_people : ℕ) (meal_cost : ℕ) (ice_cream_cost : ℕ) : ℕ :=
  num_people * (meal_cost + ice_cream_cost)

/-- Theorem stating that Elisa had at least $45 to pay for dinner -/
theorem elisa_dinner_cost :
  ∃ (money : ℕ), money ≥ dinner_cost 3 10 5 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_elisa_dinner_cost_l3685_368538


namespace NUMINAMATH_CALUDE_sum_of_repeating_decimals_l3685_368512

-- Define repeating decimals
def repeating_decimal_2 : ℚ := 2/9
def repeating_decimal_03 : ℚ := 1/33

-- Theorem statement
theorem sum_of_repeating_decimals :
  repeating_decimal_2 + repeating_decimal_03 = 25/99 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_repeating_decimals_l3685_368512


namespace NUMINAMATH_CALUDE_dot_product_of_specific_vectors_l3685_368514

theorem dot_product_of_specific_vectors :
  let A : ℝ × ℝ := (Real.cos (110 * π / 180), Real.sin (110 * π / 180))
  let B : ℝ × ℝ := (Real.cos (50 * π / 180), Real.sin (50 * π / 180))
  let OA : ℝ × ℝ := A
  let OB : ℝ × ℝ := B
  (OA.1 * OB.1 + OA.2 * OB.2) = 1/2 := by
sorry


end NUMINAMATH_CALUDE_dot_product_of_specific_vectors_l3685_368514


namespace NUMINAMATH_CALUDE_complex_expression_equals_four_l3685_368503

theorem complex_expression_equals_four :
  (0.0625 : ℝ) ^ (1/4 : ℝ) + ((-3 : ℝ)^4)^(1/4 : ℝ) - (Real.sqrt 5 - Real.sqrt 3)^0 + (3 + 3/8 : ℝ)^(1/3 : ℝ) = 4 := by
  sorry

end NUMINAMATH_CALUDE_complex_expression_equals_four_l3685_368503


namespace NUMINAMATH_CALUDE_sum_a_c_l3685_368540

theorem sum_a_c (a b c d : ℝ) 
  (h1 : a * b + b * c + c * d + d * a = 42)
  (h2 : b + d = 6) : 
  a + c = 7 := by sorry

end NUMINAMATH_CALUDE_sum_a_c_l3685_368540


namespace NUMINAMATH_CALUDE_remainder_2n_mod_4_l3685_368519

theorem remainder_2n_mod_4 (n : ℕ) (h : n % 4 = 3) : (2 * n) % 4 = 2 := by
  sorry

end NUMINAMATH_CALUDE_remainder_2n_mod_4_l3685_368519


namespace NUMINAMATH_CALUDE_mailman_delivery_l3685_368577

/-- Represents the different types of mail delivered by the mailman -/
structure MailDelivery where
  junkMail : ℕ
  magazines : ℕ
  newspapers : ℕ
  bills : ℕ
  postcards : ℕ

/-- Calculates the total number of mail pieces delivered -/
def totalMail (delivery : MailDelivery) : ℕ :=
  delivery.junkMail + delivery.magazines + delivery.newspapers + delivery.bills + delivery.postcards

/-- Theorem stating that the total mail delivered is 20 pieces -/
theorem mailman_delivery :
  ∃ (delivery : MailDelivery),
    delivery.junkMail = 6 ∧
    delivery.magazines = 5 ∧
    delivery.newspapers = 3 ∧
    delivery.bills = 4 ∧
    delivery.postcards = 2 ∧
    totalMail delivery = 20 := by
  sorry

end NUMINAMATH_CALUDE_mailman_delivery_l3685_368577


namespace NUMINAMATH_CALUDE_simplify_expression_l3685_368558

theorem simplify_expression : (((81 : ℚ) / 16) ^ (3 / 4) - (-1) ^ (0 : ℕ)) = 19 / 8 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l3685_368558


namespace NUMINAMATH_CALUDE_lake_distance_difference_l3685_368561

/-- The difference between the circumference and diameter of a circular lake -/
theorem lake_distance_difference (diameter : ℝ) (pi : ℝ) 
  (h1 : diameter = 2)
  (h2 : pi = 3.14) : 
  2 * pi * (diameter / 2) - diameter = 4.28 := by
  sorry

end NUMINAMATH_CALUDE_lake_distance_difference_l3685_368561


namespace NUMINAMATH_CALUDE_parallel_vectors_m_value_l3685_368596

/-- Two 2D vectors are parallel if their cross product is zero -/
def parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 - a.2 * b.1 = 0

theorem parallel_vectors_m_value :
  let a : ℝ × ℝ := (1, -3)
  let b : ℝ × ℝ := (-2, m)
  parallel a b → m = 6 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_m_value_l3685_368596


namespace NUMINAMATH_CALUDE_transaction_yearly_loss_l3685_368536

/-- Calculates the simple interest for a given principal, rate, and time -/
def simpleInterest (principal : ℚ) (rate : ℚ) (time : ℚ) : ℚ :=
  (principal * rate * time) / 100

/-- Represents the financial transaction described in the problem -/
structure FinancialTransaction where
  borrowAmount : ℚ
  borrowRate : ℚ
  lendRate : ℚ
  timeInYears : ℚ

/-- Calculates the yearly loss in the given financial transaction -/
def yearlyLoss (transaction : FinancialTransaction) : ℚ :=
  let borrowInterest := simpleInterest transaction.borrowAmount transaction.borrowRate transaction.timeInYears
  let lendInterest := simpleInterest transaction.borrowAmount transaction.lendRate transaction.timeInYears
  (borrowInterest - lendInterest) / transaction.timeInYears

/-- Theorem stating that the yearly loss in the given transaction is 140 -/
theorem transaction_yearly_loss :
  let transaction : FinancialTransaction := {
    borrowAmount := 7000
    borrowRate := 4
    lendRate := 6
    timeInYears := 2
  }
  yearlyLoss transaction = 140 := by sorry

end NUMINAMATH_CALUDE_transaction_yearly_loss_l3685_368536


namespace NUMINAMATH_CALUDE_ladder_distance_l3685_368547

theorem ladder_distance (ladder_length : ℝ) (elevation_angle : ℝ) (distance_to_wall : ℝ) :
  ladder_length = 9.2 →
  elevation_angle = 60 * π / 180 →
  distance_to_wall = ladder_length * Real.cos elevation_angle →
  distance_to_wall = 4.6 := by
  sorry

end NUMINAMATH_CALUDE_ladder_distance_l3685_368547


namespace NUMINAMATH_CALUDE_noodles_and_pirates_total_l3685_368526

theorem noodles_and_pirates_total (pirates : ℕ) (noodles : ℕ) : 
  pirates = 45 → noodles = pirates - 7 → noodles + pirates = 83 :=
by sorry

end NUMINAMATH_CALUDE_noodles_and_pirates_total_l3685_368526


namespace NUMINAMATH_CALUDE_betty_books_l3685_368590

theorem betty_books : ∀ (b : ℕ), 
  (b + (b + b / 4) = 45) → b = 20 := by
  sorry

end NUMINAMATH_CALUDE_betty_books_l3685_368590


namespace NUMINAMATH_CALUDE_min_area_PJ1J2_l3685_368515

/-- Triangle PQR with given side lengths -/
structure Triangle (PQ QR PR : ℝ) where
  side_PQ : PQ = 26
  side_QR : QR = 28
  side_PR : PR = 30

/-- Point Y on side QR -/
def Y (QR : ℝ) := ℝ

/-- Incenter of a triangle -/
def incenter (A B C : ℝ × ℝ) : ℝ × ℝ := sorry

/-- Area of a triangle given three points -/
def triangle_area (A B C : ℝ × ℝ) : ℝ := sorry

/-- The main theorem -/
theorem min_area_PJ1J2 (t : Triangle 26 28 30) (y : Y 28) :
  ∃ (P Q R : ℝ × ℝ),
    let J1 := incenter P Q (0, y)
    let J2 := incenter P R (0, y)
    ∀ (y' : Y 28),
      let J1' := incenter P Q (0, y')
      let J2' := incenter P R (0, y')
      triangle_area P J1 J2 ≤ triangle_area P J1' J2' ∧
      (∃ (y_min : Y 28), triangle_area P J1 J2 = 51) := by
  sorry

end NUMINAMATH_CALUDE_min_area_PJ1J2_l3685_368515


namespace NUMINAMATH_CALUDE_barry_total_amount_l3685_368546

/-- Calculates the total amount Barry needs to pay for his purchase --/
def calculate_total_amount (shirt_price pants_price tie_price : ℝ)
  (shirt_discount pants_discount coupon_discount sales_tax : ℝ) : ℝ :=
  let discounted_shirt := shirt_price * (1 - shirt_discount)
  let discounted_pants := pants_price * (1 - pants_discount)
  let subtotal := discounted_shirt + discounted_pants + tie_price
  let after_coupon := subtotal * (1 - coupon_discount)
  let total := after_coupon * (1 + sales_tax)
  total

/-- Theorem stating that the total amount Barry needs to pay is $201.27 --/
theorem barry_total_amount : 
  calculate_total_amount 80 100 40 0.15 0.10 0.05 0.07 = 201.27 := by
  sorry

end NUMINAMATH_CALUDE_barry_total_amount_l3685_368546


namespace NUMINAMATH_CALUDE_smallest_integer_fraction_l3685_368576

theorem smallest_integer_fraction (y : ℤ) : (8 : ℚ) / 12 < (y : ℚ) / 15 ↔ 11 ≤ y := by
  sorry

end NUMINAMATH_CALUDE_smallest_integer_fraction_l3685_368576


namespace NUMINAMATH_CALUDE_triangle_existence_l3685_368502

theorem triangle_existence (a b c d e : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) (he : e > 0) : 
  ∃ (x y z : ℝ), 
    x = Real.sqrt (b^2 + c^2 + d^2) ∧ 
    y = Real.sqrt (a^2 + b^2 + c^2 + e^2 + 2*a*c) ∧ 
    z = Real.sqrt (a^2 + d^2 + e^2 + 2*d*e) ∧ 
    x + y > z ∧ y + z > x ∧ z + x > y :=
sorry

end NUMINAMATH_CALUDE_triangle_existence_l3685_368502


namespace NUMINAMATH_CALUDE_total_balls_l3685_368533

def ball_count (red blue yellow : ℕ) : Prop :=
  red + blue + yellow > 0 ∧ 2 * blue = 3 * red ∧ 4 * red = 2 * yellow

theorem total_balls (red blue yellow : ℕ) :
  ball_count red blue yellow → yellow = 40 → red + blue + yellow = 90 := by
  sorry

end NUMINAMATH_CALUDE_total_balls_l3685_368533


namespace NUMINAMATH_CALUDE_derivative_f_at_zero_l3685_368551

/-- The function f(x) = (2x+1)^2 -/
def f (x : ℝ) : ℝ := (2*x + 1)^2

/-- The derivative of f at x = 0 is 4 -/
theorem derivative_f_at_zero : 
  deriv f 0 = 4 := by sorry

end NUMINAMATH_CALUDE_derivative_f_at_zero_l3685_368551


namespace NUMINAMATH_CALUDE_some_number_value_l3685_368593

theorem some_number_value (a x : ℕ) (h1 : a = 105) (h2 : a^3 = x * 25 * 45 * 49) : x = 21 := by
  sorry

end NUMINAMATH_CALUDE_some_number_value_l3685_368593


namespace NUMINAMATH_CALUDE_ball_hit_ground_time_l3685_368560

/-- The time at which a ball hits the ground when thrown upward -/
theorem ball_hit_ground_time : ∃ t : ℚ, t = 10/7 ∧ -4.9 * t^2 + 3.5 * t + 5 = 0 := by
  sorry

end NUMINAMATH_CALUDE_ball_hit_ground_time_l3685_368560


namespace NUMINAMATH_CALUDE_sequence_equality_l3685_368530

theorem sequence_equality (a : ℕ → ℕ) 
  (h : ∀ (i j : ℕ), i ≠ j → Nat.gcd (a i) (a j) = Nat.gcd i j) :
  ∀ (i : ℕ), a i = i := by
  sorry

end NUMINAMATH_CALUDE_sequence_equality_l3685_368530


namespace NUMINAMATH_CALUDE_hex_conversion_sum_l3685_368591

/-- Converts a hexadecimal number to decimal --/
def hex_to_decimal (hex : String) : ℕ := sorry

/-- Converts a decimal number to radix 7 --/
def decimal_to_radix7 (n : ℕ) : String := sorry

/-- Converts a radix 7 number to decimal --/
def radix7_to_decimal (r7 : String) : ℕ := sorry

/-- Converts a decimal number to hexadecimal --/
def decimal_to_hex (n : ℕ) : String := sorry

/-- Adds two hexadecimal numbers and returns the result in hexadecimal --/
def add_hex (hex1 : String) (hex2 : String) : String := sorry

theorem hex_conversion_sum :
  let initial_hex := "E78"
  let decimal := hex_to_decimal initial_hex
  let radix7 := decimal_to_radix7 decimal
  let back_to_decimal := radix7_to_decimal radix7
  let final_hex := decimal_to_hex back_to_decimal
  add_hex initial_hex final_hex = "1CF0" := by sorry

end NUMINAMATH_CALUDE_hex_conversion_sum_l3685_368591


namespace NUMINAMATH_CALUDE_evaluate_expression_l3685_368557

theorem evaluate_expression : Real.sqrt (Real.sqrt 81) + Real.sqrt 256 - Real.sqrt 49 = 12 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l3685_368557


namespace NUMINAMATH_CALUDE_triangle_properties_l3685_368520

/-- Triangle ABC with vertices A(-1,-1), B(3,2), C(7,-7) -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- The specific triangle ABC given in the problem -/
def triangleABC : Triangle :=
  { A := (-1, -1)
    B := (3, 2)
    C := (7, -7) }

/-- Altitude from a point to a line -/
def altitude (p : ℝ × ℝ) (l : ℝ → ℝ) : ℝ → ℝ := sorry

/-- Area of a triangle -/
def triangleArea (t : Triangle) : ℝ := sorry

/-- Main theorem stating the properties of triangle ABC -/
theorem triangle_properties (t : Triangle) (h : t = triangleABC) :
  (altitude t.C (λ x => (3/4) * x - 5/4) = λ x => (-4/3) * x + 19/3) ∧
  triangleArea t = 24 := by sorry

end NUMINAMATH_CALUDE_triangle_properties_l3685_368520


namespace NUMINAMATH_CALUDE_part_one_part_two_l3685_368508

/-- Part I: Minimum value of m for maximum |f(x)| -/
theorem part_one (a : ℝ) (h_a : a ∈ Set.Icc 4 6) :
  ∃ m : ℝ, m ≥ 6 ∧ ∀ x ∈ Set.Icc 1 m, |x + a / x - 4| ≤ |m + a / m - 4| :=
sorry

/-- Part II: Upper bound for k -/
theorem part_two (a : ℝ) (h_a : a ∈ Set.Icc 1 2) (k : ℝ) :
  (∀ x₁ x₂ : ℝ, x₁ ∈ Set.Icc 2 4 → x₂ ∈ Set.Icc 2 4 → x₁ < x₂ →
    |x₁ + a / x₁ - 4| - |x₂ + a / x₂ - 4| < k * x₁ + 3 - (k * x₂ + 3)) →
  k ≤ 6 - 4 * Real.sqrt 3 :=
sorry

end NUMINAMATH_CALUDE_part_one_part_two_l3685_368508


namespace NUMINAMATH_CALUDE_max_min_values_of_f_l3685_368579

noncomputable def f (x : ℝ) : ℝ := Real.exp (2 * x^2 - 4 * x - 6)

theorem max_min_values_of_f :
  ∃ (max min : ℝ),
    (∀ x ∈ Set.Icc 0 3, f x ≤ max) ∧
    (∃ x ∈ Set.Icc 0 3, f x = max) ∧
    (∀ x ∈ Set.Icc 0 3, min ≤ f x) ∧
    (∃ x ∈ Set.Icc 0 3, f x = min) ∧
    max = 1 ∧
    min = 1 / Real.exp 8 :=
by sorry

end NUMINAMATH_CALUDE_max_min_values_of_f_l3685_368579


namespace NUMINAMATH_CALUDE_logical_equivalence_l3685_368542

theorem logical_equivalence (P Q R : Prop) :
  (¬P ∧ ¬Q → ¬R) ↔ (R → P ∨ Q) := by
  sorry

end NUMINAMATH_CALUDE_logical_equivalence_l3685_368542


namespace NUMINAMATH_CALUDE_profit_doubling_l3685_368553

theorem profit_doubling (cost : ℝ) (original_price : ℝ) :
  original_price = cost * 1.6 →
  let double_price := 2 * original_price
  (double_price - cost) / cost * 100 = 220 := by
sorry

end NUMINAMATH_CALUDE_profit_doubling_l3685_368553


namespace NUMINAMATH_CALUDE_mary_gave_one_blue_crayon_l3685_368548

/-- Given that Mary initially has 5 green crayons and 8 blue crayons,
    gives 3 green crayons to Becky, and has 9 crayons left afterwards,
    prove that Mary gave 1 blue crayon to Becky. -/
theorem mary_gave_one_blue_crayon 
  (initial_green : Nat) 
  (initial_blue : Nat)
  (green_given : Nat)
  (total_left : Nat)
  (h1 : initial_green = 5)
  (h2 : initial_blue = 8)
  (h3 : green_given = 3)
  (h4 : total_left = 9)
  (h5 : total_left = initial_green + initial_blue - green_given - blue_given)
  : blue_given = 1 := by
  sorry

#check mary_gave_one_blue_crayon

end NUMINAMATH_CALUDE_mary_gave_one_blue_crayon_l3685_368548


namespace NUMINAMATH_CALUDE_white_surface_fraction_is_5_16_l3685_368568

/-- Represents a cube with given edge length -/
structure Cube :=
  (edge : ℕ)

/-- Represents the large cube constructed from smaller cubes -/
structure LargeCube :=
  (edge : ℕ)
  (smallCubes : ℕ)
  (redCubes : ℕ)
  (whiteCubes : ℕ)

/-- Calculates the surface area of a cube -/
def surfaceArea (c : Cube) : ℕ :=
  6 * c.edge * c.edge

/-- Calculates the fraction of white surface area -/
def whiteSurfaceFraction (lc : LargeCube) : ℚ :=
  sorry

/-- Theorem stating the fraction of white surface area -/
theorem white_surface_fraction_is_5_16 (lc : LargeCube) 
  (h1 : lc.edge = 4)
  (h2 : lc.smallCubes = 64)
  (h3 : lc.redCubes = 48)
  (h4 : lc.whiteCubes = 16) :
  whiteSurfaceFraction lc = 5 / 16 :=
sorry

end NUMINAMATH_CALUDE_white_surface_fraction_is_5_16_l3685_368568


namespace NUMINAMATH_CALUDE_gcd_of_36_and_54_l3685_368550

theorem gcd_of_36_and_54 : Nat.gcd 36 54 = 18 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_36_and_54_l3685_368550
