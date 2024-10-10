import Mathlib

namespace z_is_real_z_is_complex_z_is_pure_imaginary_z_in_fourth_quadrant_l1096_109666

-- Define the complex number z as a function of real number m
def z (m : ℝ) : ℂ := Complex.mk (m^2 - 1) (m^2 - m - 2)

-- 1. z is a real number iff m = -1 or m = 2
theorem z_is_real (m : ℝ) : (z m).im = 0 ↔ m = -1 ∨ m = 2 := by sorry

-- 2. z is a complex number iff m ≠ -1 and m ≠ 2
theorem z_is_complex (m : ℝ) : (z m).im ≠ 0 ↔ m ≠ -1 ∧ m ≠ 2 := by sorry

-- 3. z is a pure imaginary number iff m = 1
theorem z_is_pure_imaginary (m : ℝ) : (z m).re = 0 ∧ (z m).im ≠ 0 ↔ m = 1 := by sorry

-- 4. z is in the fourth quadrant iff 1 < m < 2
theorem z_in_fourth_quadrant (m : ℝ) : 
  (z m).re > 0 ∧ (z m).im < 0 ↔ 1 < m ∧ m < 2 := by sorry

end z_is_real_z_is_complex_z_is_pure_imaginary_z_in_fourth_quadrant_l1096_109666


namespace tom_gave_sixteen_balloons_l1096_109682

/-- The number of balloons Tom gave to Fred -/
def balloons_given (initial_balloons remaining_balloons : ℕ) : ℕ :=
  initial_balloons - remaining_balloons

/-- Theorem: Tom gave 16 balloons to Fred -/
theorem tom_gave_sixteen_balloons :
  let initial_balloons : ℕ := 30
  let remaining_balloons : ℕ := 14
  balloons_given initial_balloons remaining_balloons = 16 := by
  sorry

end tom_gave_sixteen_balloons_l1096_109682


namespace not_coplanar_implies_not_intersect_exists_not_intersect_but_coplanar_l1096_109689

/-- A point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- A line in 3D space defined by two points -/
structure Line3D where
  p1 : Point3D
  p2 : Point3D

/-- Check if four points are coplanar -/
def areCoplanar (p1 p2 p3 p4 : Point3D) : Prop := sorry

/-- Check if two lines intersect -/
def linesIntersect (l1 l2 : Line3D) : Prop := sorry

theorem not_coplanar_implies_not_intersect 
  (E F G H : Point3D) 
  (EF : Line3D) 
  (GH : Line3D) 
  (h1 : EF = Line3D.mk E F) 
  (h2 : GH = Line3D.mk G H) : 
  ¬(areCoplanar E F G H) → ¬(linesIntersect EF GH) := 
sorry

theorem exists_not_intersect_but_coplanar :
  ∃ (E F G H : Point3D) (EF GH : Line3D),
    EF = Line3D.mk E F ∧ 
    GH = Line3D.mk G H ∧ 
    ¬(linesIntersect EF GH) ∧ 
    areCoplanar E F G H :=
sorry

end not_coplanar_implies_not_intersect_exists_not_intersect_but_coplanar_l1096_109689


namespace total_letters_in_names_l1096_109694

theorem total_letters_in_names (jonathan_first : ℕ) (jonathan_surname : ℕ) 
  (sister_first : ℕ) (sister_surname : ℕ) 
  (h1 : jonathan_first = 8) (h2 : jonathan_surname = 10) 
  (h3 : sister_first = 5) (h4 : sister_surname = 10) : 
  jonathan_first + jonathan_surname + sister_first + sister_surname = 33 := by
  sorry

end total_letters_in_names_l1096_109694


namespace c_oxen_count_l1096_109663

/-- Represents the number of oxen and months for each person --/
structure GrazingData where
  oxen : ℕ
  months : ℕ

/-- Calculates the total oxen-months for a given GrazingData --/
def oxenMonths (data : GrazingData) : ℕ := data.oxen * data.months

/-- Theorem: Given the conditions, c put 15 oxen for grazing --/
theorem c_oxen_count (total_rent : ℚ) (a b : GrazingData) (c_months : ℕ) (c_rent : ℚ) :
  total_rent = 210 →
  a = { oxen := 10, months := 7 } →
  b = { oxen := 12, months := 5 } →
  c_months = 3 →
  c_rent = 54 →
  ∃ (c_oxen : ℕ), 
    let c : GrazingData := { oxen := c_oxen, months := c_months }
    (c_rent / total_rent) * (oxenMonths a + oxenMonths b + oxenMonths c) = oxenMonths c ∧
    c_oxen = 15 := by
  sorry


end c_oxen_count_l1096_109663


namespace odd_numbers_mean_median_impossibility_l1096_109688

theorem odd_numbers_mean_median_impossibility :
  ∀ (a b c d e f g : ℤ),
    a < b ∧ b < c ∧ c < d ∧ d < e ∧ e < f ∧ f < g →
    Odd a ∧ Odd b ∧ Odd c ∧ Odd d ∧ Odd e ∧ Odd f ∧ Odd g →
    (a + b + c + d + e + f + g) / 7 ≠ d + 3 / 7 :=
by sorry

end odd_numbers_mean_median_impossibility_l1096_109688


namespace batsman_score_difference_l1096_109647

theorem batsman_score_difference (total_innings : ℕ) (total_average : ℚ) (reduced_average : ℚ) (highest_score : ℕ) :
  total_innings = 46 →
  total_average = 61 →
  reduced_average = 58 →
  highest_score = 202 →
  ∃ (lowest_score : ℕ),
    (total_average * total_innings : ℚ) = 
      (reduced_average * (total_innings - 2) + (highest_score + lowest_score) : ℚ) ∧
    highest_score - lowest_score = 150 :=
by sorry

end batsman_score_difference_l1096_109647


namespace limit_of_function_l1096_109603

theorem limit_of_function (ε : ℝ) (hε : ε > 0) :
  ∃ δ > 0, ∀ x : ℝ, 0 < |x - π/3| ∧ |x - π/3| < δ →
    |(1 - 2 * Real.cos x) / Real.sin (π - 3 * x) + Real.sqrt 3 / 3| < ε := by
  sorry

end limit_of_function_l1096_109603


namespace union_A_B_when_m_2_range_m_for_A_subset_B_l1096_109628

-- Define the sets A and B
def A : Set ℝ := {x | (x + 2) / (x - 3/2) < 0}
def B (m : ℝ) : Set ℝ := {x | x^2 - (m + 1)*x + m ≤ 0}

-- Theorem for part (1)
theorem union_A_B_when_m_2 : A ∪ B 2 = {x | -2 < x ∧ x ≤ 2} := by sorry

-- Theorem for part (2)
theorem range_m_for_A_subset_B : 
  {m | A ⊆ B m} = {m | -2 < m ∧ m < 3/2} := by sorry

end union_A_B_when_m_2_range_m_for_A_subset_B_l1096_109628


namespace pebble_ratio_l1096_109622

/-- Prove that the ratio of pebbles Lance threw to pebbles Candy threw is 3:1 -/
theorem pebble_ratio : 
  let candy_pebbles : ℕ := 4
  let lance_pebbles : ℕ := candy_pebbles + 8
  (lance_pebbles : ℚ) / (candy_pebbles : ℚ) = 3 := by sorry

end pebble_ratio_l1096_109622


namespace runner_daily_distance_l1096_109674

theorem runner_daily_distance (total_distance : ℝ) (total_weeks : ℝ) (daily_distance : ℝ) : 
  total_distance = 42 ∧ total_weeks = 3 ∧ daily_distance * (total_weeks * 7) = total_distance →
  daily_distance = 2 := by
  sorry

end runner_daily_distance_l1096_109674


namespace correct_miscopied_value_l1096_109646

/-- Given a set of values with an incorrect mean due to one miscopied value,
    calculate the correct value that should have been recorded. -/
theorem correct_miscopied_value
  (n : ℕ) -- Total number of values
  (initial_mean : ℚ) -- Initial (incorrect) mean
  (wrong_value : ℚ) -- Value that was incorrectly recorded
  (correct_mean : ℚ) -- Correct mean after fixing the error
  (h1 : n = 30) -- There are 30 values
  (h2 : initial_mean = 150) -- The initial mean was 150
  (h3 : wrong_value = 135) -- The value was incorrectly recorded as 135
  (h4 : correct_mean = 151) -- The correct mean is 151
  : ℚ := -- The theorem returns a rational number
by
  -- The proof goes here
  sorry

#check correct_miscopied_value

end correct_miscopied_value_l1096_109646


namespace age_ratio_sandy_molly_l1096_109600

/-- Represents a person's age -/
structure Age where
  years : ℕ

/-- Represents the passage of time in years -/
def yearsLater (a : Age) (n : ℕ) : Age :=
  ⟨a.years + n⟩

theorem age_ratio_sandy_molly :
  ∀ (sandy_current : Age) (molly_current : Age),
    yearsLater sandy_current 6 = Age.mk 42 →
    molly_current = Age.mk 27 →
    (sandy_current.years : ℚ) / molly_current.years = 4 / 3 := by
  sorry

end age_ratio_sandy_molly_l1096_109600


namespace divisor_power_result_l1096_109629

theorem divisor_power_result (k : ℕ) : 
  21^k ∣ 435961 → 7^k - k^7 = 1 := by
  sorry

end divisor_power_result_l1096_109629


namespace polynomial_simplification_l1096_109606

theorem polynomial_simplification (x : ℝ) :
  3 - 5*x - 7*x^2 + 9 + 11*x - 13*x^2 - 15 + 17*x + 19*x^2 + 2*x^3 =
  -3 + 23*x - x^2 + 2*x^3 := by
  sorry

end polynomial_simplification_l1096_109606


namespace brothers_total_goals_l1096_109649

/-- The total number of goals scored by Louie and his brother -/
def total_goals (louie_last_match : ℕ) (louie_previous : ℕ) (brother_seasons : ℕ) (games_per_season : ℕ) : ℕ :=
  let louie_total := louie_last_match + louie_previous
  let brother_per_game := 2 * louie_last_match
  let brother_total := brother_seasons * games_per_season * brother_per_game
  louie_total + brother_total

/-- Theorem stating the total number of goals scored by the brothers -/
theorem brothers_total_goals :
  total_goals 4 40 3 50 = 1244 := by
  sorry

end brothers_total_goals_l1096_109649


namespace conference_left_handed_fraction_l1096_109616

theorem conference_left_handed_fraction :
  ∀ (total : ℕ) (red blue left_handed_red left_handed_blue : ℕ),
    red + blue = total →
    red = 7 * (total / 10) →
    blue = 3 * (total / 10) →
    left_handed_red = red / 3 →
    left_handed_blue = 2 * blue / 3 →
    (left_handed_red + left_handed_blue : ℚ) / total = 13 / 30 :=
by sorry

end conference_left_handed_fraction_l1096_109616


namespace fly_can_always_escape_l1096_109618

/-- Represents a bug (fly or spider) in the octahedron -/
structure Bug where
  position : ℝ × ℝ × ℝ
  speed : ℝ

/-- Represents the octahedron -/
structure Octahedron where
  vertices : List (ℝ × ℝ × ℝ)
  edges : List ((ℝ × ℝ × ℝ) × (ℝ × ℝ × ℝ))

/-- Represents the state of the chase -/
structure ChaseState where
  octahedron : Octahedron
  fly : Bug
  spiders : List Bug

/-- Function to determine if the fly can escape -/
def canFlyEscape (state : ChaseState) : Prop :=
  ∃ (nextPosition : ℝ × ℝ × ℝ), nextPosition ∈ state.octahedron.vertices ∧
    ∀ (spider : Bug), spider ∈ state.spiders →
      ‖spider.position - nextPosition‖ > ‖state.fly.position - nextPosition‖ * (spider.speed / state.fly.speed)

/-- The main theorem -/
theorem fly_can_always_escape (r : ℝ) (h : r < 25) :
  ∀ (state : ChaseState),
    state.fly.speed = 50 ∧
    (∀ spider ∈ state.spiders, spider.speed = r) ∧
    state.fly.position ∈ state.octahedron.vertices ∧
    state.spiders.length = 3 →
    canFlyEscape state :=
  sorry

end fly_can_always_escape_l1096_109618


namespace function_range_l1096_109690

theorem function_range : 
  ∃ (min max : ℝ), min = -1 ∧ max = 3 ∧
  (∀ x : ℝ, 0 ≤ x ∧ x ≤ 3 → min ≤ x^2 - 2*x ∧ x^2 - 2*x ≤ max) ∧
  (∃ x₁ x₂ : ℝ, 0 ≤ x₁ ∧ x₁ ≤ 3 ∧ 0 ≤ x₂ ∧ x₂ ≤ 3 ∧ 
    x₁^2 - 2*x₁ = min ∧ x₂^2 - 2*x₂ = max) := by
  sorry

end function_range_l1096_109690


namespace pet_store_puzzle_l1096_109699

theorem pet_store_puzzle (initial_birds initial_puppies initial_cats initial_spiders : ℕ)
  (final_total : ℕ) :
  initial_birds = 12 →
  initial_puppies = 9 →
  initial_cats = 5 →
  initial_spiders = 15 →
  final_total = 25 →
  ∃ (adopted_puppies : ℕ),
    adopted_puppies = initial_puppies - (
      initial_birds + initial_puppies + initial_cats + initial_spiders -
      (initial_birds / 2 + 7 + final_total)
    ) :=
by sorry

end pet_store_puzzle_l1096_109699


namespace quadratic_equal_roots_l1096_109696

/-- Theorem: The quadratic equation x^2 - 6x + 9 = 0 has two equal real roots. -/
theorem quadratic_equal_roots :
  ∃ (x : ℝ), (x^2 - 6*x + 9 = 0) ∧ (∀ y : ℝ, y^2 - 6*y + 9 = 0 → y = x) := by
  sorry

end quadratic_equal_roots_l1096_109696


namespace days_to_complete_correct_l1096_109617

/-- Represents the number of days required to complete the work -/
def days_to_complete : ℕ := 9

/-- Represents the total number of family members -/
def total_members : ℕ := 15

/-- Represents the number of days it takes for a woman to complete the work -/
def woman_days : ℕ := 180

/-- Represents the number of days it takes for a man to complete the work -/
def man_days : ℕ := 120

/-- Represents the number of women in the family -/
def num_women : ℕ := 3

/-- Represents the number of men in the family -/
def num_men : ℕ := total_members - num_women

/-- Represents the fraction of work done by women in one day -/
def women_work_per_day : ℚ := (1 / woman_days : ℚ) * num_women / 3

/-- Represents the fraction of work done by men in one day -/
def men_work_per_day : ℚ := (1 / man_days : ℚ) * num_men / 2

/-- Represents the total fraction of work done by the family in one day -/
def total_work_per_day : ℚ := women_work_per_day + men_work_per_day

/-- Theorem stating that the calculated number of days to complete the work is correct -/
theorem days_to_complete_correct : 
  ⌈(1 : ℚ) / total_work_per_day⌉ = days_to_complete := by sorry

end days_to_complete_correct_l1096_109617


namespace perpendicular_line_polar_equation_l1096_109602

/-- The polar equation of a line perpendicular to the polar axis and passing through 
    the center of the circle ρ = 6cosθ -/
theorem perpendicular_line_polar_equation (ρ θ : ℝ) : 
  (ρ = 6 * Real.cos θ → ∃ c, c = 3 ∧ ρ * Real.cos θ = c) :=
sorry

end perpendicular_line_polar_equation_l1096_109602


namespace ab9_equals_459_implies_a_equals_4_l1096_109678

/-- Represents a three-digit number with 9 as the last digit -/
structure ThreeDigitNumber9 where
  hundreds : Nat
  tens : Nat
  inv_hundreds : hundreds < 10
  inv_tens : tens < 10

/-- Converts a ThreeDigitNumber9 to its numerical value -/
def ThreeDigitNumber9.toNat (n : ThreeDigitNumber9) : Nat :=
  100 * n.hundreds + 10 * n.tens + 9

theorem ab9_equals_459_implies_a_equals_4 (ab9 : ThreeDigitNumber9) 
  (h : ab9.toNat = 459) : ab9.hundreds = 4 := by
  sorry

end ab9_equals_459_implies_a_equals_4_l1096_109678


namespace arithmetic_sequence_property_l1096_109687

def arithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_property (a : ℕ → ℝ) :
  arithmeticSequence a →
  a 4 + a 6 + a 8 + a 10 + a 12 = 120 →
  a 7 - (1/3) * a 5 = 16 := by
  sorry

end arithmetic_sequence_property_l1096_109687


namespace circle_equation_with_tangent_conditions_l1096_109635

/-- The standard equation of a circle with center on y = (1/2)x^2 and tangent to y = 0 and x = 0 -/
theorem circle_equation_with_tangent_conditions (t : ℝ) :
  (∃ (r : ℝ), r > 0 ∧
    (∀ (x y : ℝ), (x - t)^2 + (y - (1/2) * t^2)^2 = r^2 ↔
      ((x = 0 ∨ y = 0) → (x - t)^2 + (y - (1/2) * t^2)^2 = r^2))) →
  (∃ (s : ℝ), s = 1 ∨ s = -1) ∧
    (∀ (x y : ℝ), (x - s)^2 + (y - (1/2))^2 = 1) := by
  sorry

end circle_equation_with_tangent_conditions_l1096_109635


namespace binomial_coefficient_26_6_l1096_109611

theorem binomial_coefficient_26_6 (h1 : Nat.choose 24 5 = 42504) (h2 : Nat.choose 24 6 = 134596) :
  Nat.choose 26 6 = 230230 := by
  sorry

end binomial_coefficient_26_6_l1096_109611


namespace thursday_tuesday_difference_l1096_109692

/-- The amount of money Max's mom gave him on Tuesday -/
def tuesday_amount : ℕ := 8

/-- The amount of money Max's mom gave him on Wednesday -/
def wednesday_amount : ℕ := 5 * tuesday_amount

/-- The amount of money Max's mom gave him on Thursday -/
def thursday_amount : ℕ := wednesday_amount + 9

/-- The theorem stating the difference between Thursday's and Tuesday's amounts -/
theorem thursday_tuesday_difference :
  thursday_amount - tuesday_amount = 41 := by sorry

end thursday_tuesday_difference_l1096_109692


namespace max_sum_of_digits_24hour_format_l1096_109672

/-- Represents a time in 24-hour format -/
structure Time24 where
  hours : Fin 24
  minutes : Fin 60

/-- Calculates the sum of digits in a natural number -/
def sumOfDigits (n : ℕ) : ℕ := sorry

/-- Calculates the sum of digits in a Time24 -/
def sumOfDigitsTime24 (t : Time24) : ℕ :=
  sumOfDigits t.hours.val + sumOfDigits t.minutes.val

/-- The theorem stating the maximum sum of digits in a 24-hour format display -/
theorem max_sum_of_digits_24hour_format :
  ∃ (max : ℕ), ∀ (t : Time24), sumOfDigitsTime24 t ≤ max ∧
  ∃ (t' : Time24), sumOfDigitsTime24 t' = max ∧ max = 24 := by sorry

end max_sum_of_digits_24hour_format_l1096_109672


namespace simplify_expression_l1096_109623

theorem simplify_expression (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hxy : x * y - 4 ≠ 0) :
  (x^2 - 4 / y) / (y^2 - 4 / x) = x / y := by
  sorry

end simplify_expression_l1096_109623


namespace interest_rate_calculation_l1096_109665

/-- Calculates the simple interest rate given loan details and total interest --/
def calculate_interest_rate (loan1_principal loan1_time loan2_principal loan2_time total_interest : ℚ) : ℚ :=
  let total_interest_fraction := (loan1_principal * loan1_time + loan2_principal * loan2_time) / 100
  total_interest / total_interest_fraction

theorem interest_rate_calculation (loan1_principal loan1_time loan2_principal loan2_time total_interest : ℚ) 
  (h1 : loan1_principal = 5000)
  (h2 : loan1_time = 2)
  (h3 : loan2_principal = 3000)
  (h4 : loan2_time = 4)
  (h5 : total_interest = 2200) :
  calculate_interest_rate loan1_principal loan1_time loan2_principal loan2_time total_interest = 10 := by
  sorry

#eval calculate_interest_rate 5000 2 3000 4 2200

end interest_rate_calculation_l1096_109665


namespace megan_seashells_l1096_109655

def current_seashells : ℕ := 19
def needed_seashells : ℕ := 6
def target_seashells : ℕ := 25

theorem megan_seashells : current_seashells + needed_seashells = target_seashells := by
  sorry

end megan_seashells_l1096_109655


namespace three_digit_not_multiple_of_6_or_8_count_l1096_109636

/-- The count of three-digit numbers -/
def three_digit_count : ℕ := 900

/-- The count of three-digit numbers that are multiples of 6 -/
def multiples_of_6_count : ℕ := 150

/-- The count of three-digit numbers that are multiples of 8 -/
def multiples_of_8_count : ℕ := 112

/-- The count of three-digit numbers that are multiples of both 6 and 8 (i.e., multiples of 24) -/
def multiples_of_24_count : ℕ := 37

/-- Theorem: The count of three-digit numbers that are not multiples of 6 or 8 is 675 -/
theorem three_digit_not_multiple_of_6_or_8_count : 
  three_digit_count - (multiples_of_6_count + multiples_of_8_count - multiples_of_24_count) = 675 := by
  sorry

end three_digit_not_multiple_of_6_or_8_count_l1096_109636


namespace percentage_difference_l1096_109640

theorem percentage_difference (x y : ℝ) (h : x = 4 * y) :
  (x - y) / x * 100 = 75 :=
by sorry

end percentage_difference_l1096_109640


namespace yuanxiao_sales_problem_l1096_109693

/-- Yuanxiao sales problem -/
theorem yuanxiao_sales_problem 
  (cost : ℝ) 
  (min_price : ℝ) 
  (base_sales : ℝ) 
  (base_price : ℝ) 
  (price_sensitivity : ℝ) 
  (max_price : ℝ) 
  (min_profit : ℝ)
  (h1 : cost = 20)
  (h2 : min_price = 25)
  (h3 : base_sales = 250)
  (h4 : base_price = 25)
  (h5 : price_sensitivity = 10)
  (h6 : max_price = 38)
  (h7 : min_profit = 2000) :
  let sales_volume (x : ℝ) := -price_sensitivity * x + (base_sales + price_sensitivity * base_price)
  let profit (x : ℝ) := (x - cost) * (sales_volume x)
  ∃ (optimal_price : ℝ) (max_profit : ℝ) (min_sales : ℝ),
    (∀ x, sales_volume x = -10 * x + 500) ∧
    (optimal_price = 35 ∧ max_profit = 2250 ∧ 
     ∀ x, x ≥ min_price → profit x ≤ max_profit) ∧
    (min_sales = 120 ∧
     ∀ x, min_price ≤ x ∧ x ≤ max_price → 
     profit x ≥ min_profit → sales_volume x ≥ min_sales) := by
  sorry

end yuanxiao_sales_problem_l1096_109693


namespace Z_in_first_quadrant_l1096_109619

def Z : ℂ := Complex.I * (1 - 2 * Complex.I)

theorem Z_in_first_quadrant : 
  Complex.re Z > 0 ∧ Complex.im Z > 0 := by
  sorry

end Z_in_first_quadrant_l1096_109619


namespace alex_remaining_money_l1096_109695

theorem alex_remaining_money (weekly_income : ℝ) (tax_rate : ℝ) (water_bill : ℝ) 
  (tithe_rate : ℝ) (groceries : ℝ) (transportation : ℝ) :
  weekly_income = 900 →
  tax_rate = 0.15 →
  water_bill = 75 →
  tithe_rate = 0.20 →
  groceries = 150 →
  transportation = 50 →
  weekly_income - (tax_rate * weekly_income) - water_bill - (tithe_rate * weekly_income) - 
    groceries - transportation = 310 := by
  sorry

end alex_remaining_money_l1096_109695


namespace at_least_one_truth_teller_not_knight_l1096_109658

-- Define the possible types of people
inductive PersonType
  | Knight
  | Liar
  | Normal

-- Define a person with a type and a statement about the other person
structure Person where
  type : PersonType
  statement : PersonType → Prop

-- Define what it means for a person to be telling the truth
def isTellingTruth (p : Person) (otherType : PersonType) : Prop :=
  match p.type with
  | PersonType.Knight => p.statement otherType
  | PersonType.Liar => ¬(p.statement otherType)
  | PersonType.Normal => True

-- Define the specific statements made by A and B
def statementA (typeB : PersonType) : Prop := typeB = PersonType.Knight
def statementB (typeA : PersonType) : Prop := typeA ≠ PersonType.Knight

-- Define A and B
def A : Person := { type := PersonType.Knight, statement := statementA }
def B : Person := { type := PersonType.Knight, statement := statementB }

-- The main theorem
theorem at_least_one_truth_teller_not_knight :
  ∃ p : Person, p ∈ [A, B] ∧ 
    (∃ otherType : PersonType, isTellingTruth p otherType) ∧ 
    p.type ≠ PersonType.Knight :=
sorry

end at_least_one_truth_teller_not_knight_l1096_109658


namespace cubic_root_sum_l1096_109676

theorem cubic_root_sum (p q r : ℝ) : 
  (p^3 - 3*p + 1 = 0) → 
  (q^3 - 3*q + 1 = 0) → 
  (r^3 - 3*r + 1 = 0) → 
  p*(q - r)^2 + q*(r - p)^2 + r*(p - q)^2 = 3 := by
  sorry

end cubic_root_sum_l1096_109676


namespace overlap_area_of_circles_l1096_109641

/-- The area of overlap between two circles with given properties -/
theorem overlap_area_of_circles (r : ℝ) (overlap_percentage : ℝ) : 
  r = 10 ∧ overlap_percentage = 0.25 → 
  2 * (25 * Real.pi - 50) = 
    2 * ((overlap_percentage * 2 * Real.pi * r^2 / 4) - (r^2 / 2)) := by
  sorry

end overlap_area_of_circles_l1096_109641


namespace existence_of_sequence_l1096_109697

/-- The number of positive divisors of n -/
def d (n : ℕ) : ℕ := sorry

/-- The smallest prime divisor of n -/
def s (n : ℕ) : ℕ := sorry

/-- The theorem stating the existence of a sequence satisfying the given conditions -/
theorem existence_of_sequence : ∃ (a : ℕ → ℕ), 
  (∀ k ∈ Finset.range 2022, a (k + 1) > a k + 1) ∧ 
  (∀ k ∈ Finset.range 2022, d (a (k + 1) - a k - 1) > 2023^k) ∧
  (∀ k ∈ Finset.range 2022, s (a (k + 1) - a k) > 2023^k) :=
sorry

end existence_of_sequence_l1096_109697


namespace skee_ball_tickets_proof_l1096_109626

/-- The number of tickets Tom won from 'whack a mole' -/
def whack_a_mole_tickets : ℕ := 32

/-- The number of tickets Tom spent on a hat -/
def spent_tickets : ℕ := 7

/-- The number of tickets Tom has left -/
def remaining_tickets : ℕ := 50

/-- The number of tickets Tom won from 'skee ball' -/
def skee_ball_tickets : ℕ := (remaining_tickets + spent_tickets) - whack_a_mole_tickets

theorem skee_ball_tickets_proof : skee_ball_tickets = 25 := by
  sorry

end skee_ball_tickets_proof_l1096_109626


namespace h_bounds_l1096_109621

/-- The probability that the distance between two randomly chosen points on (0,1) is less than h -/
def probability (h : ℝ) : ℝ := h * (2 - h)

/-- Theorem stating the bounds of h given the probability constraints -/
theorem h_bounds (h : ℝ) (h_pos : 0 < h) (h_lt_one : h < 1) 
  (prob_lower : 1/4 < probability h) (prob_upper : probability h < 3/4) : 
  1/2 - Real.sqrt 3 / 2 < h ∧ h < 1/2 + Real.sqrt 3 / 2 := by
  sorry

#check h_bounds

end h_bounds_l1096_109621


namespace fraction_equality_l1096_109605

theorem fraction_equality (a b c d : ℝ) 
  (h_pos : a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0) 
  (h1 : (5*a + b) / (5*c + d) = (6*a + b) / (6*c + d))
  (h2 : (7*a + b) / (7*c + d) = 9) :
  (9*a + b) / (9*c + d) = 9 := by
  sorry

end fraction_equality_l1096_109605


namespace trajectory_is_ray_l1096_109656

/-- The set of complex numbers z satisfying |z+1| - |z-1| = 2 forms a ray in the complex plane -/
theorem trajectory_is_ray : 
  {z : ℂ | Complex.abs (z + 1) - Complex.abs (z - 1) = 2} = 
  {z : ℂ | ∃ t : ℝ, t ≥ 0 ∧ z = 1 + t} := by sorry

end trajectory_is_ray_l1096_109656


namespace equation_solution_l1096_109684

theorem equation_solution : ∃ (x₁ x₂ : ℝ), 
  (x₁ = 1 + Real.sqrt 3 ∧ x₂ = 1 - Real.sqrt 3) ∧
  (x₁^2 = (4*x₁ - 2)/(x₁ - 2) ∧ x₂^2 = (4*x₂ - 2)/(x₂ - 2)) :=
sorry

end equation_solution_l1096_109684


namespace r_value_when_n_is_2_l1096_109685

theorem r_value_when_n_is_2 (n : ℕ) (s r : ℕ) 
  (h1 : s = 3^(n^2) + 1) 
  (h2 : r = 5^s - s) 
  (h3 : n = 2) : 
  r = 5^82 - 82 := by
sorry

end r_value_when_n_is_2_l1096_109685


namespace arithmetic_geometric_ratio_l1096_109691

/-- An arithmetic sequence with common difference d and first term a_1 -/
def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + d

/-- Three terms form a geometric sequence -/
def geometric_sequence (x y z : ℝ) : Prop :=
  y^2 = x * z

theorem arithmetic_geometric_ratio
  (a : ℕ → ℝ) (d : ℝ)
  (h1 : arithmetic_sequence a d)
  (h2 : d ≠ 0)
  (h3 : a 1 ≠ 0)
  (h4 : geometric_sequence (a 2) (a 4) (a 8)) :
  (a 1 + a 5 + a 9) / (a 2 + a 3) = 3 :=
sorry

end arithmetic_geometric_ratio_l1096_109691


namespace fraction_equals_zero_l1096_109669

theorem fraction_equals_zero (x : ℝ) (h : x ≠ 0) :
  x = 3 → (2 * x - 6) / (5 * x) = 0 := by sorry

end fraction_equals_zero_l1096_109669


namespace garden_remaining_area_l1096_109638

/-- A rectangular garden plot with a shed in one corner -/
structure GardenPlot where
  length : ℝ
  width : ℝ
  shedSide : ℝ

/-- Calculate the remaining area of a garden plot available for planting -/
def remainingArea (garden : GardenPlot) : ℝ :=
  garden.length * garden.width - garden.shedSide * garden.shedSide

/-- Theorem: The remaining area of a 20ft by 18ft garden plot with a 4ft by 4ft shed is 344 sq ft -/
theorem garden_remaining_area :
  let garden : GardenPlot := { length := 20, width := 18, shedSide := 4 }
  remainingArea garden = 344 := by sorry

end garden_remaining_area_l1096_109638


namespace player_matches_l1096_109624

/-- The number of matches played by a player -/
def num_matches : ℕ := sorry

/-- The current average runs per match -/
def current_average : ℚ := 32

/-- The runs to be scored in the next match -/
def next_match_runs : ℕ := 98

/-- The increase in average after the next match -/
def average_increase : ℚ := 6

theorem player_matches :
  (current_average * num_matches + next_match_runs) / (num_matches + 1) = current_average + average_increase →
  num_matches = 10 := by sorry

end player_matches_l1096_109624


namespace smallest_first_term_of_arithmetic_sequence_l1096_109654

def arithmetic_sequence (c₁ : ℚ) (d : ℚ) : ℕ → ℚ
  | 0 => c₁
  | n+1 => arithmetic_sequence c₁ d n + d

def sum_of_terms (c₁ : ℚ) (d : ℚ) (n : ℕ) : ℚ :=
  (n : ℚ) * (2 * c₁ + (n - 1 : ℚ) * d) / 2

theorem smallest_first_term_of_arithmetic_sequence :
  ∃ (c₁ d : ℚ),
    (c₁ ≥ 1/3) ∧
    (∃ (S₃ S₇ : ℕ),
      sum_of_terms c₁ d 3 = S₃ ∧
      sum_of_terms c₁ d 7 = S₇) ∧
    (∀ (c₁' d' : ℚ),
      (c₁' ≥ 1/3) →
      (∃ (S₃' S₇' : ℕ),
        sum_of_terms c₁' d' 3 = S₃' ∧
        sum_of_terms c₁' d' 7 = S₇') →
      c₁' ≥ c₁) ∧
    c₁ = 5/14 :=
sorry

end smallest_first_term_of_arithmetic_sequence_l1096_109654


namespace two_rats_boring_theorem_l1096_109668

/-- The distance burrowed by the larger rat on day n -/
def larger_rat_distance (n : ℕ) : ℚ := 2^(n-1)

/-- The distance burrowed by the smaller rat on day n -/
def smaller_rat_distance (n : ℕ) : ℚ := (1/2)^(n-1)

/-- The total distance burrowed by both rats after n days -/
def total_distance (n : ℕ) : ℚ := 
  (Finset.range n).sum (λ i => larger_rat_distance (i+1) + smaller_rat_distance (i+1))

/-- The theorem stating that the total distance burrowed after 5 days is 32 15/16 -/
theorem two_rats_boring_theorem : total_distance 5 = 32 + 15/16 := by sorry

end two_rats_boring_theorem_l1096_109668


namespace third_boy_age_l1096_109613

theorem third_boy_age (total_age : ℕ) (age_two_boys : ℕ) (num_boys : ℕ) :
  total_age = 29 →
  age_two_boys = 9 →
  num_boys = 3 →
  ∃ (third_boy_age : ℕ), third_boy_age = total_age - 2 * age_two_boys :=
by
  sorry

end third_boy_age_l1096_109613


namespace jerry_shower_limit_l1096_109670

/-- Calculates the number of full showers Jerry can take in July --/
def showers_in_july (total_water : ℕ) (drinking_cooking : ℕ) (shower_water : ℕ)
  (pool_length : ℕ) (pool_width : ℕ) (pool_depth : ℕ)
  (odd_day_leakage : ℕ) (even_day_leakage : ℕ) (evaporation_rate : ℕ)
  (odd_days : ℕ) (even_days : ℕ) : ℕ :=
  let pool_volume := pool_length * pool_width * pool_depth
  let total_leakage := odd_day_leakage * odd_days + even_day_leakage * even_days
  let total_evaporation := evaporation_rate * (odd_days + even_days)
  let pool_water_usage := pool_volume + total_leakage + total_evaporation
  let remaining_water := total_water - drinking_cooking - pool_water_usage
  remaining_water / shower_water

/-- Theorem stating that Jerry can take at most 1 full shower in July --/
theorem jerry_shower_limit :
  showers_in_july 1000 100 20 10 10 6 5 8 2 16 15 ≤ 1 :=
by sorry

end jerry_shower_limit_l1096_109670


namespace line_equation_condition1_line_equation_condition2_line_equation_condition3_l1096_109642

-- Define the line l
def line_l (a b c : ℝ) : Prop := ∀ x y : ℝ, a * x + b * y + c = 0

-- Define the point (1, -2) that the line passes through
def point_condition (a b c : ℝ) : Prop := a * 1 + b * (-2) + c = 0

-- Theorem for condition 1
theorem line_equation_condition1 (a b c : ℝ) :
  point_condition a b c →
  (∃ k : ℝ, k = 1 - π / 12 ∧ b / a = -k) →
  line_l a b c ↔ line_l 1 (-Real.sqrt 3) (-2 * Real.sqrt 3 - 1) :=
sorry

-- Theorem for condition 2
theorem line_equation_condition2 (a b c : ℝ) :
  point_condition a b c →
  (b / a = 1) →
  line_l a b c ↔ line_l 1 (-1) (-3) :=
sorry

-- Theorem for condition 3
theorem line_equation_condition3 (a b c : ℝ) :
  point_condition a b c →
  (c / b = -1) →
  line_l a b c ↔ line_l 1 1 1 :=
sorry

end line_equation_condition1_line_equation_condition2_line_equation_condition3_l1096_109642


namespace f_not_valid_mapping_l1096_109615

-- Define the sets M and P
def M : Set ℝ := {x | 0 ≤ x ∧ x ≤ 6}
def P : Set ℝ := {y | 0 ≤ y ∧ y ≤ 3}

-- Define the function f
def f : ℝ → ℝ := fun x ↦ x

-- Theorem stating that f is not a valid mapping from M to P
theorem f_not_valid_mapping : ¬(∀ x ∈ M, f x ∈ P) := by
  sorry


end f_not_valid_mapping_l1096_109615


namespace choir_members_count_l1096_109645

theorem choir_members_count : ∃! n : ℕ, 
  200 < n ∧ n < 300 ∧ 
  (n + 4) % 10 = 0 ∧ 
  (n + 5) % 11 = 0 ∧ 
  n = 226 := by
sorry

end choir_members_count_l1096_109645


namespace sqrt_equation_solution_l1096_109698

theorem sqrt_equation_solution (a b : ℕ) : 
  (Real.sqrt (8 + b / a) = 2 * Real.sqrt (b / a)) → (a = 63 ∧ b = 8) :=
by sorry

end sqrt_equation_solution_l1096_109698


namespace no_special_two_digit_primes_l1096_109648

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → n % d ≠ 0

def reverse_digits (n : ℕ) : ℕ :=
  let tens := n / 10
  let ones := n % 10
  ones * 10 + tens

def digit_sum (n : ℕ) : ℕ :=
  (n / 10) + (n % 10)

theorem no_special_two_digit_primes :
  ∀ n : ℕ, 10 ≤ n → n < 100 →
    is_prime n ∧ is_prime (reverse_digits n) ∧ is_prime (digit_sum n) → False :=
sorry

end no_special_two_digit_primes_l1096_109648


namespace milk_selling_price_l1096_109664

/-- Proves that the selling price of milk per litre is twice the cost price,
    given the mixing ratio of water to milk and the profit percentage. -/
theorem milk_selling_price 
  (x : ℝ) -- cost price of pure milk per litre
  (water_ratio : ℝ) -- ratio of water added to pure milk
  (milk_ratio : ℝ) -- ratio of pure milk
  (profit_percentage : ℝ) -- profit percentage
  (h1 : water_ratio = 2) -- 2 litres of water are added
  (h2 : milk_ratio = 6) -- to every 6 litres of pure milk
  (h3 : profit_percentage = 166.67) -- profit percentage is 166.67%
  : ∃ (selling_price : ℝ), selling_price = 2 * x := by
  sorry

end milk_selling_price_l1096_109664


namespace sugar_solution_percentage_l1096_109634

theorem sugar_solution_percentage (x : ℝ) : 
  x > 0 ∧ x < 100 →
  (3/4 * x + 1/4 * 34) / 100 = 16 / 100 →
  x = 10 := by
sorry

end sugar_solution_percentage_l1096_109634


namespace vector_magnitude_problem_l1096_109667

theorem vector_magnitude_problem (a b : ℝ × ℝ) :
  let angle := Real.pi / 3  -- 60 degrees in radians
  let dot_product := a.1 * b.1 + a.2 * b.2
  let magnitude (v : ℝ × ℝ) := Real.sqrt (v.1^2 + v.2^2)
  angle = Real.arccos (dot_product / (magnitude a * magnitude b)) →
  magnitude a = 2 →
  magnitude b = 5 →
  magnitude (2 • a - b) = Real.sqrt 21 := by
sorry

end vector_magnitude_problem_l1096_109667


namespace distance_AC_l1096_109637

theorem distance_AC (A B C : ℝ × ℝ) : 
  let AB := Real.sqrt 27
  let BC := 2
  let angle_ABC := 150 * Real.pi / 180
  let AC := Real.sqrt ((AB^2 + BC^2) - 2 * AB * BC * Real.cos angle_ABC)
  AC = 7 := by sorry

end distance_AC_l1096_109637


namespace bridge_units_correct_l1096_109662

-- Define the units
inductive LengthUnit
| Kilometers

inductive LoadUnit
| Tons

-- Define the bridge properties
structure Bridge where
  length : ℕ
  loadCapacity : ℕ

-- Define the function to assign units
def assignUnits (b : Bridge) : (LengthUnit × LoadUnit) :=
  (LengthUnit.Kilometers, LoadUnit.Tons)

-- Theorem statement
theorem bridge_units_correct (b : Bridge) (h1 : b.length = 1) (h2 : b.loadCapacity = 50) :
  assignUnits b = (LengthUnit.Kilometers, LoadUnit.Tons) := by
  sorry

#check bridge_units_correct

end bridge_units_correct_l1096_109662


namespace sqrt_eighteen_minus_sqrt_two_l1096_109652

theorem sqrt_eighteen_minus_sqrt_two : Real.sqrt 18 - Real.sqrt 2 = 2 * Real.sqrt 2 := by
  sorry

end sqrt_eighteen_minus_sqrt_two_l1096_109652


namespace fourth_sample_number_l1096_109686

/-- Represents a systematic sample from a population -/
structure SystematicSample where
  population_size : ℕ
  sample_size : ℕ
  interval : ℕ
  start : ℕ

/-- Checks if a number is in the systematic sample -/
def in_sample (s : SystematicSample) (n : ℕ) : Prop :=
  ∃ k : ℕ, n = s.start + k * s.interval ∧ n ≤ s.population_size

theorem fourth_sample_number 
  (s : SystematicSample)
  (h_pop : s.population_size = 56)
  (h_sample : s.sample_size = 4)
  (h_6 : in_sample s 6)
  (h_34 : in_sample s 34)
  (h_48 : in_sample s 48) :
  in_sample s 20 :=
sorry

end fourth_sample_number_l1096_109686


namespace probability_both_in_photo_l1096_109675

/-- Represents a runner on a circular track -/
structure Runner where
  name : String
  lapTime : ℝ
  direction : Bool  -- true for counterclockwise, false for clockwise

/-- Represents the photography setup -/
structure PhotoSetup where
  trackCoverage : ℝ  -- fraction of track covered by the photo
  minTime : ℝ        -- minimum time after start for taking the photo (in seconds)
  maxTime : ℝ        -- maximum time after start for taking the photo (in seconds)

/-- Calculate the probability of both runners being in the photo -/
def probabilityBothInPhoto (ann : Runner) (ben : Runner) (setup : PhotoSetup) : ℝ :=
  sorry

/-- Theorem statement for the probability problem -/
theorem probability_both_in_photo 
  (ann : Runner) 
  (ben : Runner) 
  (setup : PhotoSetup) 
  (h1 : ann.name = "Ann" ∧ ann.lapTime = 75 ∧ ann.direction = true)
  (h2 : ben.name = "Ben" ∧ ben.lapTime = 60 ∧ ben.direction = false)
  (h3 : setup.trackCoverage = 1/6)
  (h4 : setup.minTime = 12 * 60)
  (h5 : setup.maxTime = 15 * 60) :
  probabilityBothInPhoto ann ben setup = 1/6 :=
sorry

end probability_both_in_photo_l1096_109675


namespace banquet_plates_l1096_109630

/-- The total number of plates served at a banquet -/
theorem banquet_plates (lobster_rolls : ℕ) (spicy_hot_noodles : ℕ) (seafood_noodles : ℕ)
  (h1 : lobster_rolls = 25)
  (h2 : spicy_hot_noodles = 14)
  (h3 : seafood_noodles = 16) :
  lobster_rolls + spicy_hot_noodles + seafood_noodles = 55 := by
  sorry

end banquet_plates_l1096_109630


namespace min_xy_m_range_l1096_109679

-- Define the conditions
def condition (x y : ℝ) : Prop := x > 0 ∧ y > 0 ∧ 1/x + 3/y = 2

-- Theorem for the minimum value of xy
theorem min_xy (x y : ℝ) (h : condition x y) : 
  ∀ a b : ℝ, condition a b → x * y ≤ a * b ∧ x * y ≥ 3 :=
sorry

-- Theorem for the range of m
theorem m_range (x y : ℝ) (h : condition x y) :
  ∀ m : ℝ, (∀ a b : ℝ, condition a b → 3*a + b ≥ m^2 - m) → 
  -2 ≤ m ∧ m ≤ 3 :=
sorry

end min_xy_m_range_l1096_109679


namespace track_width_l1096_109620

theorem track_width (r : ℝ) 
  (h1 : 2 * π * (2 * r) - 2 * π * r = 16 * π) 
  (h2 : 2 * r - r = r) : r = 8 := by
  sorry

end track_width_l1096_109620


namespace smallest_k_no_real_roots_l1096_109651

theorem smallest_k_no_real_roots : 
  ∀ k : ℤ, (∀ x : ℝ, 2 * x * (k * x - 5) - x^2 + 12 ≠ 0) → k ≥ 3 :=
by sorry

end smallest_k_no_real_roots_l1096_109651


namespace intersection_complement_equals_singleton_l1096_109601

-- Define the universal set U
def U : Set (ℝ × ℝ) := Set.univ

-- Define set M
def M : Set (ℝ × ℝ) := {p : ℝ × ℝ | (p.2 - 3) / (p.1 - 2) = 1}

-- Define set N
def N : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.2 = p.1 + 1}

-- Theorem statement
theorem intersection_complement_equals_singleton :
  N ∩ (U \ M) = {(2, 3)} := by sorry

end intersection_complement_equals_singleton_l1096_109601


namespace concentric_circles_radii_difference_l1096_109671

theorem concentric_circles_radii_difference (s L : ℝ) (h : s > 0) :
  (L^2 / s^2 = 9 / 4) → (L - s = 0.5 * s) := by
  sorry

end concentric_circles_radii_difference_l1096_109671


namespace unsold_bag_weights_l1096_109661

def bag_weights : List Nat := [3, 7, 12, 15, 17, 28, 30]

def total_weight : Nat := bag_weights.sum

structure SalesDistribution where
  day1 : Nat
  day2 : Nat
  day3 : Nat
  unsold : Nat

def is_valid_distribution (d : SalesDistribution) : Prop :=
  d.day1 + d.day2 + d.day3 + d.unsold = total_weight ∧
  d.day2 = 2 * d.day1 ∧
  d.day3 = 2 * d.day2 ∧
  d.unsold ∈ bag_weights

theorem unsold_bag_weights :
  ∀ d : SalesDistribution, is_valid_distribution d → d.unsold = 7 ∨ d.unsold = 28 :=
by sorry

end unsold_bag_weights_l1096_109661


namespace sum_of_roots_l1096_109657

theorem sum_of_roots (p q : ℝ) (hp_neq_q : p ≠ q) : 
  (∃ x : ℝ, x^2 + p*x + q = 0 ∧ x^2 + q*x + p = 0) → p + q = -1 := by
  sorry

end sum_of_roots_l1096_109657


namespace initial_loss_percentage_l1096_109650

/-- Proves that for an article with a cost price of $400, if increasing the selling price
    by $100 results in a 5% gain, then the initial loss percentage is 20%. -/
theorem initial_loss_percentage 
  (cost_price : ℝ) 
  (selling_price : ℝ) 
  (h1 : cost_price = 400)
  (h2 : selling_price + 100 = 1.05 * cost_price) :
  (cost_price - selling_price) / cost_price * 100 = 20 := by
  sorry

#check initial_loss_percentage

end initial_loss_percentage_l1096_109650


namespace simple_interest_rate_calculation_l1096_109614

theorem simple_interest_rate_calculation 
  (initial_sum : ℝ) 
  (final_amount : ℝ) 
  (time : ℝ) 
  (h1 : initial_sum = 12500)
  (h2 : final_amount = 15500)
  (h3 : time = 4)
  (h4 : final_amount = initial_sum * (1 + time * (rate / 100))) :
  rate = 6 :=
by
  sorry

end simple_interest_rate_calculation_l1096_109614


namespace parabola_symmetric_points_l1096_109680

/-- Prove that for a parabola y = ax^2 (a > 0) where the distance from focus to directrix is 1/4,
    and two points A(x₁, y₁) and B(x₂, y₂) on the parabola are symmetric about y = x + m,
    and x₁x₂ = -1/2, then m = 3/2 -/
theorem parabola_symmetric_points (a : ℝ) (x₁ y₁ x₂ y₂ m : ℝ) : 
  a > 0 →
  (1 / (4 * a) = 1 / 4) →
  y₁ = a * x₁^2 →
  y₂ = a * x₂^2 →
  y₁ + y₂ = x₁ + x₂ + 2 * m →
  x₁ * x₂ = -1 / 2 →
  m = 3 / 2 := by
  sorry

end parabola_symmetric_points_l1096_109680


namespace even_function_domain_l1096_109633

/-- A function f is even if f(x) = f(-x) for all x in its domain -/
def IsEven (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

/-- The function f(x) = ax^2 + bx + 3a + b -/
def f (a b : ℝ) (x : ℝ) : ℝ :=
  a * x^2 + b * x + 3 * a + b

theorem even_function_domain (a b : ℝ) :
  (IsEven (f a b)) ∧ (Set.Icc (2 * a) (a - 1)).Nonempty → a + b = 1/3 := by
  sorry

end even_function_domain_l1096_109633


namespace anna_truck_meet_once_l1096_109612

/-- Represents the movement of Anna and the garbage truck on a path with trash pails. -/
structure TrashCollection where
  annaSpeed : ℝ
  truckSpeed : ℝ
  pailDistance : ℝ
  truckStopTime : ℝ

/-- Calculates the number of times Anna and the truck meet. -/
def meetingCount (tc : TrashCollection) : ℕ :=
  sorry

/-- The theorem states that Anna and the truck meet exactly once under the given conditions. -/
theorem anna_truck_meet_once :
  ∀ (tc : TrashCollection),
    tc.annaSpeed = 5 ∧
    tc.truckSpeed = 15 ∧
    tc.pailDistance = 300 ∧
    tc.truckStopTime = 40 →
    meetingCount tc = 1 :=
  sorry

end anna_truck_meet_once_l1096_109612


namespace square_sum_value_l1096_109683

theorem square_sum_value (a b : ℝ) (h1 : a - b = 5) (h2 : a * b = 2) : a^2 + b^2 = 29 := by
  sorry

end square_sum_value_l1096_109683


namespace paige_picture_upload_l1096_109643

/-- The number of pictures Paige uploaded to Facebook -/
def total_pictures : ℕ := 35

/-- The number of pictures in the first album -/
def first_album : ℕ := 14

/-- The number of additional albums -/
def additional_albums : ℕ := 3

/-- The number of pictures in each additional album -/
def pictures_per_additional_album : ℕ := 7

/-- Theorem stating that the total number of pictures uploaded is correct -/
theorem paige_picture_upload :
  total_pictures = first_album + additional_albums * pictures_per_additional_album :=
by sorry

end paige_picture_upload_l1096_109643


namespace factorial_ratio_equals_sixty_sevenths_l1096_109631

theorem factorial_ratio_equals_sixty_sevenths : (Nat.factorial 10 * Nat.factorial 6 * Nat.factorial 3) / (Nat.factorial 9 * Nat.factorial 7) = 60 / 7 := by
  sorry

end factorial_ratio_equals_sixty_sevenths_l1096_109631


namespace cousins_initial_money_l1096_109660

/-- Represents the money distribution problem with Carmela and her cousins -/
def money_distribution (carmela_initial : ℕ) (cousin_count : ℕ) (give_amount : ℕ) (cousin_initial : ℕ) : Prop :=
  let carmela_final := carmela_initial - (cousin_count * give_amount)
  let cousin_final := cousin_initial + give_amount
  carmela_final = cousin_final

/-- Proves that given the conditions, each cousin must have had $2 initially -/
theorem cousins_initial_money :
  money_distribution 7 4 1 2 := by
  sorry

end cousins_initial_money_l1096_109660


namespace min_segment_length_for_cyclists_l1096_109653

/-- Represents a cyclist on a circular track -/
structure Cyclist where
  speed : ℝ
  position : ℝ

/-- The circular track -/
def trackLength : ℝ := 300

/-- Theorem stating the minimum length of track segment where all cyclists will eventually appear -/
theorem min_segment_length_for_cyclists (c1 c2 c3 : Cyclist) 
  (h1 : c1.speed ≠ c2.speed)
  (h2 : c2.speed ≠ c3.speed)
  (h3 : c1.speed ≠ c3.speed)
  (h4 : c1.speed > 0 ∧ c2.speed > 0 ∧ c3.speed > 0) :
  ∃ (d : ℝ), d = 75 ∧ 
  (∀ (t : ℝ), ∃ (t' : ℝ), t' ≥ t ∧ 
    (((c1.position + c1.speed * t') % trackLength - 
      (c2.position + c2.speed * t') % trackLength + trackLength) % trackLength ≤ d ∧
     ((c2.position + c2.speed * t') % trackLength - 
      (c3.position + c3.speed * t') % trackLength + trackLength) % trackLength ≤ d ∧
     ((c1.position + c1.speed * t') % trackLength - 
      (c3.position + c3.speed * t') % trackLength + trackLength) % trackLength ≤ d)) :=
sorry

end min_segment_length_for_cyclists_l1096_109653


namespace square_measurement_error_l1096_109639

theorem square_measurement_error (actual_side : ℝ) (measured_side : ℝ) :
  measured_side^2 = actual_side^2 * (1 + 0.050625) →
  (measured_side - actual_side) / actual_side = 0.025 := by
sorry

end square_measurement_error_l1096_109639


namespace total_fruit_mass_is_7425_l1096_109627

/-- The number of apple trees in the orchard -/
def num_apple_trees : ℕ := 30

/-- The mass of apples produced by each apple tree (in kg) -/
def apple_yield_per_tree : ℕ := 150

/-- The number of peach trees in the orchard -/
def num_peach_trees : ℕ := 45

/-- The average mass of peaches produced by each peach tree (in kg) -/
def peach_yield_per_tree : ℕ := 65

/-- The total mass of fruit harvested in the orchard (in kg) -/
def total_fruit_mass : ℕ :=
  num_apple_trees * apple_yield_per_tree + num_peach_trees * peach_yield_per_tree

theorem total_fruit_mass_is_7425 : total_fruit_mass = 7425 := by
  sorry

end total_fruit_mass_is_7425_l1096_109627


namespace function_increasing_l1096_109607

theorem function_increasing (f : ℝ → ℝ) 
  (h : ∀ x₁ x₂ : ℝ, x₁ ≠ x₂ → x₁ * f x₁ + x₂ * f x₂ > x₁ * f x₂ + x₂ * f x₁) : 
  StrictMono f := by
  sorry

end function_increasing_l1096_109607


namespace band_members_count_l1096_109644

/-- Calculates the number of band members given the earnings per member, total earnings, and number of gigs. -/
def band_members (earnings_per_member : ℕ) (total_earnings : ℕ) (num_gigs : ℕ) : ℕ :=
  (total_earnings / num_gigs) / earnings_per_member

/-- Proves that the number of band members is 4 given the specified conditions. -/
theorem band_members_count : band_members 20 400 5 = 4 := by
  sorry

end band_members_count_l1096_109644


namespace right_triangle_with_acute_angle_greater_than_epsilon_l1096_109681

theorem right_triangle_with_acute_angle_greater_than_epsilon :
  ∀ ε : Real, 0 < ε → ε < π / 4 →
  ∃ a b c : ℕ, 
    a * a + b * b = c * c ∧ 
    Real.arctan (min (a / b) (b / a)) > ε :=
by sorry

end right_triangle_with_acute_angle_greater_than_epsilon_l1096_109681


namespace equation_solution_l1096_109610

theorem equation_solution :
  ∃ x : ℝ, (x^2 + x ≠ 0 ∧ x^2 - x ≠ 0) ∧
  (4 / (x^2 + x) - 3 / (x^2 - x) = 0) ∧
  x = 7 :=
by sorry

end equation_solution_l1096_109610


namespace function_is_periodic_l1096_109609

-- Define the function f and the constant a
variable (f : ℝ → ℝ)
variable (a : ℝ)

-- State the theorem
theorem function_is_periodic
  (h1 : ∀ x, f x ≠ 0)
  (h2 : a > 0)
  (h3 : ∀ x, f (x - a) = 1 / f x) :
  ∀ x, f x = f (x + 2 * a) :=
by sorry

end function_is_periodic_l1096_109609


namespace circle_area_ratio_l1096_109659

/-- Given two circles X and Y where an arc of 60° on X has the same length as an arc of 40° on Y,
    the ratio of the area of circle X to the area of circle Y is 9/4. -/
theorem circle_area_ratio (R_X R_Y : ℝ) (h : R_X > 0 ∧ R_Y > 0) :
  (60 / 360 * (2 * Real.pi * R_X) = 40 / 360 * (2 * Real.pi * R_Y)) →
  (Real.pi * R_X^2) / (Real.pi * R_Y^2) = 9 / 4 := by
  sorry

end circle_area_ratio_l1096_109659


namespace bryan_uninterested_offer_is_one_l1096_109608

/-- Represents the record sale scenario --/
structure RecordSale where
  total_records : ℕ
  sammy_offer : ℚ
  bryan_offer_interested : ℚ
  bryan_interested_fraction : ℚ
  profit_difference : ℚ

/-- Calculates Bryan's offer for uninterested records --/
def bryan_uninterested_offer (sale : RecordSale) : ℚ :=
  let sammy_total := sale.total_records * sale.sammy_offer
  let bryan_interested_records := sale.total_records * sale.bryan_interested_fraction
  let bryan_uninterested_records := sale.total_records - bryan_interested_records
  let bryan_interested_total := bryan_interested_records * sale.bryan_offer_interested
  (sammy_total - bryan_interested_total - sale.profit_difference) / bryan_uninterested_records

/-- Theorem stating Bryan's offer for uninterested records is $1 --/
theorem bryan_uninterested_offer_is_one (sale : RecordSale)
    (h1 : sale.total_records = 200)
    (h2 : sale.sammy_offer = 4)
    (h3 : sale.bryan_offer_interested = 6)
    (h4 : sale.bryan_interested_fraction = 1/2)
    (h5 : sale.profit_difference = 100) :
    bryan_uninterested_offer sale = 1 := by
  sorry

end bryan_uninterested_offer_is_one_l1096_109608


namespace average_annual_reduction_l1096_109677

theorem average_annual_reduction (total_reduction : ℝ) (years : ℕ) (average_reduction : ℝ) : 
  total_reduction = 0.19 → years = 2 → (1 - average_reduction) ^ years = 1 - total_reduction → average_reduction = 0.1 := by
  sorry

end average_annual_reduction_l1096_109677


namespace max_volume_cuboid_l1096_109604

/-- Represents a cuboid with integer dimensions -/
structure Cuboid where
  length : ℕ+
  width : ℕ+
  height : ℕ+

/-- Calculates the surface area of a cuboid -/
def surfaceArea (c : Cuboid) : ℕ :=
  2 * (c.length * c.width + c.length * c.height + c.width * c.height)

/-- Calculates the volume of a cuboid -/
def volume (c : Cuboid) : ℕ :=
  c.length * c.width * c.height

/-- Theorem stating the maximum volume of a cuboid with given conditions -/
theorem max_volume_cuboid :
  ∃ (c : Cuboid), surfaceArea c = 150 ∧
    (∀ (c' : Cuboid), surfaceArea c' = 150 → volume c' ≤ volume c) ∧
    volume c = 125 := by
  sorry


end max_volume_cuboid_l1096_109604


namespace equation_describes_ellipse_l1096_109625

-- Define the equation
def equation (x y : ℝ) : Prop :=
  Real.sqrt (x^2 + (y-2)^2) + Real.sqrt ((x-6)^2 + (y+4)^2) = 12

-- Define the property of being an ellipse
def is_ellipse (f : ℝ → ℝ → Prop) : Prop :=
  ∃ (F₁ F₂ : ℝ × ℝ) (a : ℝ),
    a > Real.sqrt ((F₁.1 - F₂.1)^2 + (F₁.2 - F₂.2)^2) / 2 ∧
    ∀ (x y : ℝ), f x y ↔ 
      Real.sqrt ((x - F₁.1)^2 + (y - F₁.2)^2) +
      Real.sqrt ((x - F₂.1)^2 + (y - F₂.2)^2) = 2 * a

-- Theorem statement
theorem equation_describes_ellipse : is_ellipse equation := by
  sorry

end equation_describes_ellipse_l1096_109625


namespace fraction_irreducible_l1096_109632

theorem fraction_irreducible (n : ℕ+) : Nat.gcd (21 * n + 4) (14 * n + 3) = 1 := by
  sorry

end fraction_irreducible_l1096_109632


namespace bread_distribution_l1096_109673

theorem bread_distribution (total_loaves : ℕ) (num_people : ℕ) : 
  total_loaves = 100 →
  num_people = 5 →
  ∃ (a d : ℚ), 
    (∀ i : ℕ, i ≤ 5 → a + (i - 1) * d ≥ 0) ∧
    (a + (a + d) + (a + 2*d) + (a + 3*d) + (a + 4*d) = total_loaves) ∧
    ((a + 2*d) + (a + 3*d) + (a + 4*d) = 3 * (a + (a + d))) →
  (a + 4*d ≤ 30) :=
by sorry

end bread_distribution_l1096_109673
