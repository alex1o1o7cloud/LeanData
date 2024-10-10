import Mathlib

namespace parabola_shift_l3447_344739

/-- Given a parabola y = x^2, shifting it 3 units right and 4 units up results in y = (x-3)^2 + 4 -/
theorem parabola_shift (x y : ℝ) : 
  (y = x^2) → 
  (∃ (y' : ℝ → ℝ), 
    (∀ x, y' x = (x - 3)^2) ∧ 
    (∀ x, y' x + 4 = (x - 3)^2 + 4)) := by
  sorry


end parabola_shift_l3447_344739


namespace only_negative_three_squared_positive_l3447_344701

theorem only_negative_three_squared_positive :
  let a := 0 * ((-2019) ^ 2018)
  let b := (-3) ^ 2
  let c := -2 / ((-3) ^ 4)
  let d := (-2) ^ 3
  (a ≤ 0 ∧ b > 0 ∧ c < 0 ∧ d < 0) := by sorry

end only_negative_three_squared_positive_l3447_344701


namespace smallest_five_digit_divisible_by_first_five_primes_l3447_344792

theorem smallest_five_digit_divisible_by_first_five_primes :
  let first_five_primes := [2, 3, 5, 7, 11]
  let is_five_digit (n : ℕ) := 10000 ≤ n ∧ n ≤ 99999
  let divisible_by_all (n : ℕ) := ∀ p ∈ first_five_primes, n % p = 0
  ∃ (n : ℕ), is_five_digit n ∧ divisible_by_all n ∧
    ∀ m, is_five_digit m ∧ divisible_by_all m → n ≤ m ∧ n = 11550 :=
by
  sorry

#eval 11550 % 2  -- Should output 0
#eval 11550 % 3  -- Should output 0
#eval 11550 % 5  -- Should output 0
#eval 11550 % 7  -- Should output 0
#eval 11550 % 11 -- Should output 0

end smallest_five_digit_divisible_by_first_five_primes_l3447_344792


namespace closest_ratio_is_27_26_l3447_344759

/-- The admission fee for adults -/
def adult_fee : ℕ := 30

/-- The admission fee for children -/
def child_fee : ℕ := 15

/-- The total amount collected -/
def total_collected : ℕ := 2400

/-- Represents the number of adults and children at the exhibition -/
structure Attendance where
  adults : ℕ
  children : ℕ
  adults_nonzero : adults > 0
  children_nonzero : children > 0
  total_correct : adult_fee * adults + child_fee * children = total_collected

/-- The ratio of adults to children -/
def attendance_ratio (a : Attendance) : ℚ :=
  a.adults / a.children

/-- Checks if a given ratio is closest to 1 among all possible attendances -/
def is_closest_to_one (r : ℚ) : Prop :=
  ∀ a : Attendance, |attendance_ratio a - 1| ≥ |r - 1|

/-- The main theorem stating that 27/26 is the ratio closest to 1 -/
theorem closest_ratio_is_27_26 :
  is_closest_to_one (27 / 26) :=
sorry

end closest_ratio_is_27_26_l3447_344759


namespace food_allowance_per_teacher_l3447_344745

/-- Calculates the food allowance per teacher given the seminar details and total spent --/
theorem food_allowance_per_teacher
  (regular_fee : ℝ)
  (discount_rate : ℝ)
  (num_teachers : ℕ)
  (total_spent : ℝ)
  (h1 : regular_fee = 150)
  (h2 : discount_rate = 0.05)
  (h3 : num_teachers = 10)
  (h4 : total_spent = 1525)
  : (total_spent - num_teachers * (regular_fee * (1 - discount_rate))) / num_teachers = 10 := by
  sorry

#check food_allowance_per_teacher

end food_allowance_per_teacher_l3447_344745


namespace find_a_value_l3447_344784

theorem find_a_value (a : ℕ) (h : a ^ 3 = 21 * 25 * 45 * 49) : a = 105 := by
  sorry

end find_a_value_l3447_344784


namespace factorial_division_l3447_344766

theorem factorial_division (h : Nat.factorial 10 = 3628800) :
  Nat.factorial 10 / Nat.factorial 4 = 151200 := by
  sorry

end factorial_division_l3447_344766


namespace additional_spheres_in_cone_l3447_344743

/-- Represents a truncated cone -/
structure TruncatedCone where
  height : ℝ
  lower_radius : ℝ
  upper_radius : ℝ

/-- Represents a sphere -/
structure Sphere where
  center : ℝ × ℝ × ℝ
  radius : ℝ

/-- Function to check if a sphere is tangent to the cone's surfaces -/
def is_tangent_to_cone (s : Sphere) (c : TruncatedCone) : Prop :=
  sorry

/-- Function to check if two spheres are tangent -/
def are_spheres_tangent (s1 s2 : Sphere) : Prop :=
  sorry

/-- Function to calculate the maximum number of additional spheres -/
def max_additional_spheres (c : TruncatedCone) (s1 s2 : Sphere) : ℕ :=
  sorry

/-- Main theorem -/
theorem additional_spheres_in_cone 
  (c : TruncatedCone) 
  (s1 s2 : Sphere) :
  c.height = 8 ∧
  s1.radius = 2 ∧
  s2.radius = 3 ∧
  is_tangent_to_cone s1 c ∧
  is_tangent_to_cone s2 c ∧
  are_spheres_tangent s1 s2 →
  max_additional_spheres c s1 s2 = 2 := by
  sorry

end additional_spheres_in_cone_l3447_344743


namespace geometric_sequence_property_l3447_344754

/-- A geometric sequence -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

/-- Theorem: In a geometric sequence, if a₂ * a₆ = 4, then a₄ = 2 or a₄ = -2 -/
theorem geometric_sequence_property (a : ℕ → ℝ) 
  (h_geom : geometric_sequence a) (h_prod : a 2 * a 6 = 4) : 
  a 4 = 2 ∨ a 4 = -2 := by
  sorry

end geometric_sequence_property_l3447_344754


namespace double_infinite_sum_equals_two_l3447_344728

theorem double_infinite_sum_equals_two :
  (∑' m : ℕ+, ∑' n : ℕ+, (1 : ℝ) / (m * n * (m + n + 3))) = 2 := by sorry

end double_infinite_sum_equals_two_l3447_344728


namespace range_of_fraction_l3447_344775

-- Define a monotonically decreasing function on ℝ
def monotonically_decreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f y < f x

-- Define symmetry of f(x-1) with respect to (1,0)
def symmetric_about_one (f : ℝ → ℝ) : Prop :=
  ∀ x, f (2 - x) = -f x

-- Main theorem
theorem range_of_fraction (f : ℝ → ℝ) (h_decr : monotonically_decreasing f)
    (h_sym : symmetric_about_one f) :
    ∀ t : ℝ, f (t^2 - 2*t) + f (-3) > 0 → (t - 1) / (t - 3) < 1/2 :=
by
  sorry


end range_of_fraction_l3447_344775


namespace equation_solution_l3447_344716

theorem equation_solution (x : ℝ) : 
  (x^2 - 36) / 3 = (x^2 + 3*x + 9) / 6 ↔ x = 9 ∨ x = -9 := by
  sorry

end equation_solution_l3447_344716


namespace paint_tins_needed_half_tin_leftover_l3447_344727

-- Define the wall area range
def wall_area_min : ℝ := 1915
def wall_area_max : ℝ := 1925

-- Define the paint coverage range per tin
def coverage_min : ℝ := 17.5
def coverage_max : ℝ := 18.5

-- Define the minimum number of tins needed
def min_tins : ℕ := 111

-- Theorem statement
theorem paint_tins_needed :
  ∀ (wall_area paint_coverage : ℝ),
    wall_area_min ≤ wall_area ∧ wall_area < wall_area_max →
    coverage_min ≤ paint_coverage ∧ paint_coverage < coverage_max →
    (↑min_tins : ℝ) * coverage_min > wall_area_max ∧
    (↑(min_tins - 1) : ℝ) * coverage_min ≤ wall_area_max :=
by sorry

-- Additional theorem to ensure at least half a tin is left over
theorem half_tin_leftover :
  (↑min_tins : ℝ) * coverage_min - wall_area_max ≥ 0.5 * coverage_min :=
by sorry

end paint_tins_needed_half_tin_leftover_l3447_344727


namespace total_good_balls_eq_144_l3447_344755

/-- The total number of soccer balls -/
def total_soccer_balls : ℕ := 180

/-- The total number of basketballs -/
def total_basketballs : ℕ := 75

/-- The total number of tennis balls -/
def total_tennis_balls : ℕ := 90

/-- The total number of volleyballs -/
def total_volleyballs : ℕ := 50

/-- The number of soccer balls with holes -/
def soccer_balls_with_holes : ℕ := 125

/-- The number of basketballs with holes -/
def basketballs_with_holes : ℕ := 49

/-- The number of tennis balls with holes -/
def tennis_balls_with_holes : ℕ := 62

/-- The number of deflated volleyballs -/
def deflated_volleyballs : ℕ := 15

/-- The total number of balls without holes or deflation -/
def total_good_balls : ℕ := 
  (total_soccer_balls - soccer_balls_with_holes) +
  (total_basketballs - basketballs_with_holes) +
  (total_tennis_balls - tennis_balls_with_holes) +
  (total_volleyballs - deflated_volleyballs)

theorem total_good_balls_eq_144 : total_good_balls = 144 := by
  sorry

end total_good_balls_eq_144_l3447_344755


namespace sandy_change_l3447_344771

/-- The change Sandy received from her purchase of toys -/
def change_received (football_price baseball_price paid : ℚ) : ℚ :=
  paid - (football_price + baseball_price)

/-- Theorem stating the correct change Sandy received -/
theorem sandy_change : change_received 9.14 6.81 20 = 4.05 := by
  sorry

end sandy_change_l3447_344771


namespace bobby_candy_total_l3447_344744

theorem bobby_candy_total (initial_candy : ℕ) (more_candy : ℕ) (chocolate : ℕ)
  (h1 : initial_candy = 28)
  (h2 : more_candy = 42)
  (h3 : chocolate = 63) :
  initial_candy + more_candy + chocolate = 133 :=
by sorry

end bobby_candy_total_l3447_344744


namespace count_ordered_quadruples_l3447_344700

theorem count_ordered_quadruples (n : ℕ+) :
  (Finset.filter (fun (quad : ℕ × ℕ × ℕ × ℕ) =>
    let (a, b, c, d) := quad
    0 ≤ a ∧ a ≤ b ∧ b ≤ c ∧ c ≤ d ∧ d ≤ n)
    (Finset.product (Finset.range (n + 1))
      (Finset.product (Finset.range (n + 1))
        (Finset.product (Finset.range (n + 1)) (Finset.range (n + 1)))))).card
  = Nat.choose (n + 4) 4 :=
by sorry

end count_ordered_quadruples_l3447_344700


namespace solve_weeks_worked_problem_l3447_344756

/-- Represents the problem of calculating the number of weeks worked --/
def WeeksWorkedProblem (regular_days_per_week : ℕ) 
                       (hours_per_day : ℕ) 
                       (regular_pay_rate : ℚ) 
                       (overtime_pay_rate : ℚ) 
                       (total_earnings : ℚ) 
                       (total_hours : ℕ) : Prop :=
  let regular_hours_per_week := regular_days_per_week * hours_per_day
  ∃ (weeks_worked : ℕ),
    let regular_hours := weeks_worked * regular_hours_per_week
    let overtime_hours := total_hours - regular_hours
    regular_hours * regular_pay_rate + overtime_hours * overtime_pay_rate = total_earnings ∧
    weeks_worked = 4

/-- The main theorem stating the solution to the problem --/
theorem solve_weeks_worked_problem :
  WeeksWorkedProblem 6 10 (210/100) (420/100) 525 245 := by
  sorry

#check solve_weeks_worked_problem

end solve_weeks_worked_problem_l3447_344756


namespace inequality_proof_l3447_344789

theorem inequality_proof (a b c : ℝ) (h1 : a < 0) (h2 : a < b) (h3 : b < c) :
  (a + b < b + c) ∧ (a / (a + b) < 1) := by
  sorry

end inequality_proof_l3447_344789


namespace colby_remaining_mangoes_l3447_344752

def total_harvest : ℕ := 60
def sold_to_market : ℕ := 20
def mangoes_per_kg : ℕ := 8

def remaining_after_market : ℕ := total_harvest - sold_to_market

def sold_to_community : ℕ := remaining_after_market / 2

def remaining_kg : ℕ := remaining_after_market - sold_to_community

theorem colby_remaining_mangoes :
  remaining_kg * mangoes_per_kg = 160 := by sorry

end colby_remaining_mangoes_l3447_344752


namespace juliet_supporter_in_capulet_l3447_344776

-- Define the population distribution
def montague_pop : ℚ := 4/6
def capulet_pop : ℚ := 1/6
def verona_pop : ℚ := 1/6

-- Define the support percentages for Juliet
def montague_juliet : ℚ := 1/5  -- 20% support Juliet (100% - 80%)
def capulet_juliet : ℚ := 7/10
def verona_juliet : ℚ := 3/5

-- Theorem statement
theorem juliet_supporter_in_capulet :
  let total_juliet := montague_pop * montague_juliet + capulet_pop * capulet_juliet + verona_pop * verona_juliet
  (capulet_pop * capulet_juliet) / total_juliet = 1/3 := by
  sorry

end juliet_supporter_in_capulet_l3447_344776


namespace unique_solution_sqrt_equation_l3447_344796

theorem unique_solution_sqrt_equation :
  ∀ m n : ℕ+, 
    (m : ℝ)^2 = Real.sqrt (n : ℝ) + Real.sqrt ((2 * n + 1) : ℝ) → 
    m = 13 ∧ n = 4900 :=
by
  sorry

end unique_solution_sqrt_equation_l3447_344796


namespace c_rent_share_is_27_l3447_344773

/-- Represents the rental information for a person --/
structure RentalInfo where
  oxen : ℕ
  months : ℕ

/-- Calculates the total rent share for a person --/
def calculateRentShare (totalRent : ℚ) (totalOxenMonths : ℕ) (info : RentalInfo) : ℚ :=
  totalRent * (info.oxen * info.months : ℚ) / totalOxenMonths

theorem c_rent_share_is_27 
  (a b c : RentalInfo)
  (h_a : a = ⟨10, 7⟩)
  (h_b : b = ⟨12, 5⟩)
  (h_c : c = ⟨15, 3⟩)
  (h_total_rent : totalRent = 105)
  (h_total_oxen_months : totalOxenMonths = a.oxen * a.months + b.oxen * b.months + c.oxen * c.months) :
  calculateRentShare totalRent totalOxenMonths c = 27 := by
  sorry


end c_rent_share_is_27_l3447_344773


namespace no_prime_solution_l3447_344795

def base_p_to_decimal (n : ℕ) (p : ℕ) : ℕ :=
  let digits := n.digits p
  (List.range digits.length).foldl (λ acc i => acc + digits[i]! * p ^ i) 0

theorem no_prime_solution :
  ∀ p : ℕ, p.Prime → p ≠ 2 → p ≠ 3 → p ≠ 5 → p ≠ 7 →
    base_p_to_decimal 1014 p + base_p_to_decimal 309 p + base_p_to_decimal 120 p +
    base_p_to_decimal 132 p + base_p_to_decimal 7 p ≠
    base_p_to_decimal 153 p + base_p_to_decimal 276 p + base_p_to_decimal 371 p :=
by
  sorry

end no_prime_solution_l3447_344795


namespace mean_temperature_l3447_344764

def temperatures : List ℝ := [-7, -4, -4, -5, 1, 3, 2, 4]

theorem mean_temperature :
  (temperatures.sum / temperatures.length : ℝ) = -1.25 := by
sorry

end mean_temperature_l3447_344764


namespace lexie_family_age_ratio_l3447_344719

/-- Proves that given the age relationships in Lexie's family, the ratio of her sister's age to Lexie's age is 2. -/
theorem lexie_family_age_ratio :
  ∀ (lexie_age brother_age sister_age : ℕ),
    lexie_age = 8 →
    lexie_age = brother_age + 6 →
    sister_age - brother_age = 14 →
    ∃ (k : ℕ), sister_age = k * lexie_age →
    sister_age / lexie_age = 2 := by
  sorry

end lexie_family_age_ratio_l3447_344719


namespace parallel_postulate_introduction_l3447_344741

-- Define the concept of a geometric theorem
def GeometricTheorem : Type := Unit

-- Define the concept of Euclid's parallel postulate
def EuclidParallelPostulate : Type := Unit

-- Define the property of a theorem being independent of the parallel postulate
def independent (t : GeometricTheorem) (p : EuclidParallelPostulate) : Prop := True

-- Define the concept of introducing a postulate in geometry
def introduced_later (p : EuclidParallelPostulate) : Prop := True

theorem parallel_postulate_introduction 
  (many_theorems : Set GeometricTheorem)
  (parallel_postulate : EuclidParallelPostulate)
  (h : ∀ t ∈ many_theorems, independent t parallel_postulate) :
  introduced_later parallel_postulate :=
by
  sorry

#check parallel_postulate_introduction

end parallel_postulate_introduction_l3447_344741


namespace final_pen_count_l3447_344740

def pen_collection (initial : ℕ) (mike_gave : ℕ) (sharon_took : ℕ) : ℕ :=
  ((initial + mike_gave) * 2) - sharon_took

theorem final_pen_count : pen_collection 25 22 19 = 75 := by
  sorry

end final_pen_count_l3447_344740


namespace gcd_of_powers_minus_one_l3447_344770

theorem gcd_of_powers_minus_one : Nat.gcd (2^300 - 1) (2^315 - 1) = 2^15 - 1 := by
  sorry

end gcd_of_powers_minus_one_l3447_344770


namespace officer_combinations_count_l3447_344734

def totalMembers : ℕ := 25
def officerPositions : ℕ := 3

def chooseOfficers (n m : ℕ) : ℕ := n * (n - 1) * (n - 2)

def officersCombinations : ℕ :=
  let withoutPairs := chooseOfficers (totalMembers - 4) officerPositions
  let withAliceBob := 3 * 2 * (totalMembers - 4)
  let withCharlesDiana := 3 * 2 * (totalMembers - 4)
  withoutPairs + withAliceBob + withCharlesDiana

theorem officer_combinations_count :
  officersCombinations = 8232 :=
sorry

end officer_combinations_count_l3447_344734


namespace consistent_production_rate_l3447_344781

/-- Represents the rate of paint drum production -/
structure PaintProduction where
  days : ℕ
  drums : ℕ

/-- Calculates the daily production rate -/
def dailyRate (p : PaintProduction) : ℚ :=
  p.drums / p.days

theorem consistent_production_rate : 
  let scenario1 : PaintProduction := ⟨3, 18⟩
  let scenario2 : PaintProduction := ⟨60, 360⟩
  dailyRate scenario1 = dailyRate scenario2 ∧ dailyRate scenario1 = 6 := by
  sorry

end consistent_production_rate_l3447_344781


namespace line_through_points_l3447_344750

-- Define a point in 2D space
structure Point2D where
  x : ℝ
  y : ℝ

-- Define a line in general form ax + by + c = 0
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

-- Function to check if a point lies on a line
def pointOnLine (p : Point2D) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

-- Theorem statement
theorem line_through_points :
  let p1 : Point2D := ⟨-2, 2⟩
  let p2 : Point2D := ⟨0, 6⟩
  let l : Line := ⟨2, -1, 6⟩
  pointOnLine p1 l ∧ pointOnLine p2 l := by sorry

end line_through_points_l3447_344750


namespace fair_distribution_l3447_344749

/-- Represents the number of books each player brings to the game -/
def books_per_player : ℕ := 4

/-- Represents the total number of books in the game -/
def total_books : ℕ := 2 * books_per_player

/-- Represents the number of points needed to win the game -/
def points_to_win : ℕ := 3

/-- Represents player A's current points -/
def a_points : ℕ := 2

/-- Represents player B's current points -/
def b_points : ℕ := 1

/-- Represents the probability of player A winning the game -/
def prob_a_wins : ℚ := 3/4

/-- Represents the probability of player B winning the game -/
def prob_b_wins : ℚ := 1/4

/-- Theorem stating the fair distribution of books -/
theorem fair_distribution :
  let a_books := (total_books : ℚ) * prob_a_wins
  let b_books := (total_books : ℚ) * prob_b_wins
  a_books = 6 ∧ b_books = 2 := by
  sorry

end fair_distribution_l3447_344749


namespace five_by_five_not_coverable_l3447_344760

/-- Represents a checkerboard with width and height -/
structure Checkerboard :=
  (width : ℕ)
  (height : ℕ)

/-- Checks if a checkerboard can be covered by dominos -/
def can_be_covered_by_dominos (board : Checkerboard) : Prop :=
  (board.width * board.height) % 2 = 0 ∧
  (board.width * board.height) / 2 = (board.width * board.height + 1) / 2

theorem five_by_five_not_coverable :
  ¬(can_be_covered_by_dominos ⟨5, 5⟩) :=
by sorry

end five_by_five_not_coverable_l3447_344760


namespace coefficient_x_cubed_in_binomial_expansion_l3447_344777

theorem coefficient_x_cubed_in_binomial_expansion :
  (Finset.range 7).sum (λ k => Nat.choose 6 k * 2^(6 - k) * if k = 3 then 1 else 0) = 160 := by
  sorry

end coefficient_x_cubed_in_binomial_expansion_l3447_344777


namespace married_men_fraction_l3447_344799

theorem married_men_fraction (total_women : ℕ) (h_total_women_pos : total_women > 0) :
  let single_women := (3 : ℚ) / 7 * total_women
  let married_women := total_women - single_women
  let married_men := married_women
  let total_people := total_women + married_men
  married_men / total_people = 4 / 11 := by
sorry

end married_men_fraction_l3447_344799


namespace carrot_broccoli_ratio_is_two_to_one_l3447_344738

/-- Represents the sales data for a farmers' market --/
structure MarketSales where
  total : ℕ
  broccoli : ℕ
  cauliflower : ℕ
  spinach_offset : ℕ

/-- Calculates the ratio of carrot sales to broccoli sales --/
def carrot_broccoli_ratio (sales : MarketSales) : ℚ :=
  let carrot_sales := sales.total - sales.broccoli - sales.cauliflower - 
    (sales.spinach_offset + (sales.total - sales.broccoli - sales.cauliflower - sales.spinach_offset) / 2)
  carrot_sales / sales.broccoli

/-- Theorem stating that the ratio of carrot sales to broccoli sales is 2:1 --/
theorem carrot_broccoli_ratio_is_two_to_one (sales : MarketSales) 
  (h1 : sales.total = 380)
  (h2 : sales.broccoli = 57)
  (h3 : sales.cauliflower = 136)
  (h4 : sales.spinach_offset = 16) :
  carrot_broccoli_ratio sales = 2 := by
  sorry

end carrot_broccoli_ratio_is_two_to_one_l3447_344738


namespace porche_homework_time_l3447_344751

/-- Proves that given a total time of 3 hours (180 minutes) and homework assignments
    taking 45, 30, 50, and 25 minutes respectively, the remaining time for a special project
    is 30 minutes. -/
theorem porche_homework_time (total_time : ℕ) (math_time english_time science_time history_time : ℕ) :
  total_time = 180 ∧
  math_time = 45 ∧
  english_time = 30 ∧
  science_time = 50 ∧
  history_time = 25 →
  total_time - (math_time + english_time + science_time + history_time) = 30 :=
by sorry

end porche_homework_time_l3447_344751


namespace second_discount_percentage_l3447_344733

theorem second_discount_percentage (original_price : ℝ) (first_discount : ℝ) (final_price : ℝ) :
  original_price = 400 ∧
  first_discount = 30 ∧
  final_price = 224 →
  ∃ second_discount : ℝ,
    second_discount = 20 ∧
    final_price = original_price * (1 - first_discount / 100) * (1 - second_discount / 100) :=
by sorry

end second_discount_percentage_l3447_344733


namespace four_variable_inequality_l3447_344706

theorem four_variable_inequality (a b c d : ℝ) 
  (non_neg_a : a ≥ 0) (non_neg_b : b ≥ 0) (non_neg_c : c ≥ 0) (non_neg_d : d ≥ 0)
  (sum_one : a + b + c + d = 1) :
  a * b * c * d + b * c * d * a + c * d * a * b + d * a * b * c ≤ 1 / 27 + 176 / 27 * a * b * c * d := by
  sorry

end four_variable_inequality_l3447_344706


namespace marbles_distribution_l3447_344713

theorem marbles_distribution (total_marbles : ℕ) (num_friends : ℕ) (marbles_per_friend : ℕ) :
  total_marbles = 5504 →
  num_friends = 64 →
  marbles_per_friend = total_marbles / num_friends →
  marbles_per_friend = 86 :=
by
  sorry

end marbles_distribution_l3447_344713


namespace smallest_perfect_square_divisible_by_2_3_5_l3447_344763

theorem smallest_perfect_square_divisible_by_2_3_5 : 
  ∀ n : ℕ, n > 0 → n^2 % 2 = 0 → n^2 % 3 = 0 → n^2 % 5 = 0 → n^2 ≥ 225 :=
by sorry

end smallest_perfect_square_divisible_by_2_3_5_l3447_344763


namespace abs_neg_three_eq_three_l3447_344787

theorem abs_neg_three_eq_three : |(-3 : ℤ)| = 3 := by
  sorry

end abs_neg_three_eq_three_l3447_344787


namespace spider_web_paths_l3447_344794

/-- The number of paths in a grid where only right and up moves are allowed -/
def number_of_paths (m n : ℕ) : ℕ :=
  Nat.choose (m + n) m

/-- Theorem: In a 7x3 grid, the number of paths from bottom-left to top-right
    moving only right and up is equal to (10 choose 7) -/
theorem spider_web_paths :
  number_of_paths 7 3 = Nat.choose 10 7 := by
  sorry

end spider_web_paths_l3447_344794


namespace smallest_value_for_x_between_zero_and_one_l3447_344723

theorem smallest_value_for_x_between_zero_and_one (x : ℝ) (h : 0 < x ∧ x < 1) :
  x^2 < min x (min (2*x) (min (Real.sqrt x) (1/x))) :=
by sorry

end smallest_value_for_x_between_zero_and_one_l3447_344723


namespace decimal_100_to_base_4_has_four_digits_l3447_344746

def to_base_4 (n : ℕ) : List ℕ :=
  if n = 0 then [0]
  else
    let rec aux (m : ℕ) (acc : List ℕ) : List ℕ :=
      if m = 0 then acc
      else aux (m / 4) ((m % 4) :: acc)
    aux n []

theorem decimal_100_to_base_4_has_four_digits :
  (to_base_4 100).length = 4 := by
  sorry

end decimal_100_to_base_4_has_four_digits_l3447_344746


namespace sum_of_coefficients_factorized_form_l3447_344793

theorem sum_of_coefficients_factorized_form (x y : ℝ) : 
  ∃ (a b c d e : ℤ), 
    27 * x^6 - 512 * y^6 = (a * x^2 + b * y^2) * (c * x^4 + d * x^2 * y^2 + e * y^4) ∧
    a + b + c + d + e = 92 :=
by sorry

end sum_of_coefficients_factorized_form_l3447_344793


namespace algebraic_expression_properties_l3447_344709

variable (a b c : ℝ)

def f (x : ℝ) : ℝ := a * x^5 + b * x^3 + 3 * x + c

theorem algebraic_expression_properties :
  (f 0 = -1) →
  (f 1 = -1) →
  (f 3 = -10) →
  (c = -1 ∧ a + b + c = -4 ∧ f (-3) = 8) := by
  sorry

end algebraic_expression_properties_l3447_344709


namespace triangle_area_l3447_344718

theorem triangle_area (a b c : ℝ) (A B C : ℝ) :
  a = 5 →
  B = π / 3 →
  Real.cos A = 11 / 14 →
  let S := (1 / 2) * a * c * Real.sin B
  S = 10 * Real.sqrt 3 :=
by sorry

end triangle_area_l3447_344718


namespace both_save_800_l3447_344788

/-- Represents the financial situation of Anand and Balu -/
structure FinancialSituation where
  anand_income : ℕ
  balu_income : ℕ
  anand_expenditure : ℕ
  balu_expenditure : ℕ

/-- Checks if the given financial situation satisfies the problem conditions -/
def satisfies_conditions (fs : FinancialSituation) : Prop :=
  fs.anand_income * 4 = fs.balu_income * 5 ∧
  fs.anand_expenditure * 2 = fs.balu_expenditure * 3 ∧
  fs.anand_income = 2000

/-- Calculates the savings for a person given their income and expenditure -/
def savings (income : ℕ) (expenditure : ℕ) : ℕ :=
  income - expenditure

/-- Theorem stating that both Anand and Balu save 800 each -/
theorem both_save_800 (fs : FinancialSituation) (h : satisfies_conditions fs) :
  savings fs.anand_income fs.anand_expenditure = 800 ∧
  savings fs.balu_income fs.balu_expenditure = 800 := by
  sorry


end both_save_800_l3447_344788


namespace constant_sum_property_l3447_344717

/-- Representation of a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Definition of the ellipse -/
def isOnEllipse (p : Point) : Prop :=
  p.x^2 / 8 + p.y^2 / 4 = 1

/-- Definition of a line passing through a point -/
def Line (p : Point) (m : ℝ) :=
  {q : Point | q.y - p.y = m * (q.x - p.x)}

/-- The focus point of the ellipse -/
def F : Point := ⟨2, 0⟩

/-- Theorem stating the constant sum property -/
theorem constant_sum_property 
  (A B P : Point) 
  (hA : isOnEllipse A) 
  (hB : isOnEllipse B) 
  (hP : P.x = 0) 
  (hline : ∃ (m : ℝ), A ∈ Line F m ∧ B ∈ Line F m ∧ P ∈ Line F m)
  (m n : ℝ)
  (hm : (P.x - A.x, P.y - A.y) = m • (A.x - F.x, A.y - F.y))
  (hn : (P.x - B.x, P.y - B.y) = n • (B.x - F.x, B.y - F.y)) :
  m + n = -4 := by
    sorry


end constant_sum_property_l3447_344717


namespace rational_solutions_quadratic_l3447_344708

theorem rational_solutions_quadratic (k : ℕ+) : 
  (∃ x : ℚ, k * x^2 + 16 * x + k = 0) ↔ k = 8 := by
sorry

end rational_solutions_quadratic_l3447_344708


namespace ball_distribution_l3447_344782

theorem ball_distribution (n : ℕ) (k : ℕ) :
  -- Part (a): No empty boxes
  (Nat.choose (n + k - 1) (k - 1) = Nat.choose 19 5 → n = 20 ∧ k = 6) ∧
  -- Part (b): Some boxes can be empty
  (Nat.choose (n + k - 1) (k - 1) = Nat.choose 25 5 → n = 20 ∧ k = 6) :=
by sorry

end ball_distribution_l3447_344782


namespace meal_price_calculation_l3447_344783

theorem meal_price_calculation (beef_amount : ℝ) (pork_ratio : ℝ) (meat_per_meal : ℝ) (total_revenue : ℝ) :
  beef_amount = 20 →
  pork_ratio = 1 / 2 →
  meat_per_meal = 1.5 →
  total_revenue = 400 →
  (total_revenue / ((beef_amount + beef_amount * pork_ratio) / meat_per_meal) = 20) :=
by sorry

end meal_price_calculation_l3447_344783


namespace binomial_coefficient_sum_l3447_344707

theorem binomial_coefficient_sum (n : ℕ) : 4^n - 2^n = 992 → n = 5 := by
  sorry

end binomial_coefficient_sum_l3447_344707


namespace ball_hitting_ground_time_l3447_344705

/-- The height of a ball thrown upwards is given by y = -20t^2 + 32t + 60,
    where y is the height in feet and t is the time in seconds.
    This theorem proves that the time when the ball hits the ground (y = 0)
    is (4 + √91) / 5 seconds. -/
theorem ball_hitting_ground_time :
  let y (t : ℝ) := -20 * t^2 + 32 * t + 60
  ∃ t : ℝ, y t = 0 ∧ t = (4 + Real.sqrt 91) / 5 :=
by sorry

end ball_hitting_ground_time_l3447_344705


namespace ones_more_frequent_than_fives_l3447_344711

-- Define the upper bound of the sequence
def upperBound : ℕ := 1000000000

-- Define a function that computes the digital root of a number
def digitalRoot (n : ℕ) : ℕ :=
  if n % 9 = 0 then 9 else n % 9

-- Define a function that counts occurrences of a digit in the final sequence
def countDigit (d : ℕ) : ℕ :=
  (upperBound / 9) + if d = 1 then 1 else 0

-- Theorem statement
theorem ones_more_frequent_than_fives :
  countDigit 1 > countDigit 5 := by
sorry

end ones_more_frequent_than_fives_l3447_344711


namespace janet_bird_count_l3447_344720

theorem janet_bird_count (crows hawks : ℕ) : 
  hawks = crows + (crows * 6 / 10) →
  crows + hawks = 78 →
  crows = 30 := by
sorry

end janet_bird_count_l3447_344720


namespace beggars_and_mothers_attitude_l3447_344725

structure Neighborhood where
  has_nearby_railway : Bool
  has_frequent_beggars : Bool

structure Mother where
  treats_beggars_equally : Bool
  provides_newspapers : Bool
  father_helped_in_depression : Bool

def reason_for_beggars_visits (n : Neighborhood) : Bool :=
  n.has_nearby_railway

def mother_treatment_of_beggars (m : Mother) : Bool :=
  m.treats_beggars_equally

def purpose_of_newspapers (m : Mother) : Bool :=
  m.provides_newspapers

def explanation_for_mothers_attitude (m : Mother) : Bool :=
  m.father_helped_in_depression

theorem beggars_and_mothers_attitude 
  (n : Neighborhood) 
  (m : Mother) 
  (h1 : n.has_nearby_railway = true)
  (h2 : n.has_frequent_beggars = true)
  (h3 : m.treats_beggars_equally = true)
  (h4 : m.provides_newspapers = true)
  (h5 : m.father_helped_in_depression = true) :
  reason_for_beggars_visits n = true ∧
  mother_treatment_of_beggars m = true ∧
  purpose_of_newspapers m = true ∧
  explanation_for_mothers_attitude m = true := by
  sorry

end beggars_and_mothers_attitude_l3447_344725


namespace four_row_grid_has_27_triangles_l3447_344798

/-- Represents a triangular grid with a given number of rows -/
structure TriangularGrid :=
  (rows : ℕ)

/-- Counts the number of small triangles in a triangular grid -/
def countSmallTriangles (grid : TriangularGrid) : ℕ :=
  (grid.rows * (grid.rows + 1)) / 2

/-- Counts the number of medium triangles in a triangular grid -/
def countMediumTriangles (grid : TriangularGrid) : ℕ :=
  ((grid.rows - 1) * grid.rows) / 2

/-- Counts the number of large triangles in a triangular grid -/
def countLargeTriangles (grid : TriangularGrid) : ℕ :=
  if grid.rows ≥ 3 then 1 else 0

/-- Counts the total number of triangles in a triangular grid -/
def countTotalTriangles (grid : TriangularGrid) : ℕ :=
  countSmallTriangles grid + countMediumTriangles grid + countLargeTriangles grid

/-- Theorem: A triangular grid with 4 rows contains 27 triangles in total -/
theorem four_row_grid_has_27_triangles :
  countTotalTriangles (TriangularGrid.mk 4) = 27 := by
  sorry

end four_row_grid_has_27_triangles_l3447_344798


namespace no_real_roots_for_specific_k_l3447_344791

theorem no_real_roots_for_specific_k : ∀ x : ℝ, x^2 + 2*x + 2 ≠ 0 := by
  sorry

end no_real_roots_for_specific_k_l3447_344791


namespace only_D_is_comprehensive_l3447_344737

/-- Represents the type of survey --/
inductive SurveyType
  | Comprehensive
  | Sampling

/-- Represents the different survey options --/
inductive SurveyOption
  | A  -- Understanding the service life of a certain light bulb
  | B  -- Understanding whether a batch of cold drinks meets quality standards
  | C  -- Understanding the vision status of eighth-grade students nationwide
  | D  -- Understanding which month has the most births in a certain class

/-- Determines the appropriate survey type for a given option --/
def determineSurveyType (option : SurveyOption) : SurveyType :=
  match option with
  | SurveyOption.A => SurveyType.Sampling
  | SurveyOption.B => SurveyType.Sampling
  | SurveyOption.C => SurveyType.Sampling
  | SurveyOption.D => SurveyType.Comprehensive

/-- Theorem stating that only Option D is suitable for a comprehensive survey --/
theorem only_D_is_comprehensive :
  ∀ (option : SurveyOption),
    determineSurveyType option = SurveyType.Comprehensive ↔ option = SurveyOption.D :=
by sorry

#check only_D_is_comprehensive

end only_D_is_comprehensive_l3447_344737


namespace gcd_12012_18018_l3447_344786

theorem gcd_12012_18018 : Nat.gcd 12012 18018 = 6006 := by
  sorry

end gcd_12012_18018_l3447_344786


namespace cosine_angle_between_vectors_l3447_344762

def a : Fin 2 → ℝ := ![1, 2]
def b : Fin 2 → ℝ := ![1, 3]

theorem cosine_angle_between_vectors :
  let inner_product := (a 0) * (b 0) + (a 1) * (b 1)
  let magnitude_a := Real.sqrt ((a 0)^2 + (a 1)^2)
  let magnitude_b := Real.sqrt ((b 0)^2 + (b 1)^2)
  (inner_product / (magnitude_a * magnitude_b)) = (7 * Real.sqrt 2) / 10 := by
  sorry

end cosine_angle_between_vectors_l3447_344762


namespace f_properties_l3447_344736

noncomputable def f (x : ℝ) : ℝ := Real.log (abs x) / Real.log 2

theorem f_properties :
  (∀ x ≠ 0, f x = f (-x)) ∧
  (∀ x y, 0 < x ∧ x < y → f x < f y) :=
by sorry

end f_properties_l3447_344736


namespace sine_inequality_l3447_344768

theorem sine_inequality (t : ℝ) (h1 : 0 < t) (h2 : t ≤ π / 2) :
  1 / (Real.sin t)^2 ≤ 1 / t^2 + 1 - 4 / π^2 := by
  sorry

end sine_inequality_l3447_344768


namespace right_triangle_of_orthocenters_l3447_344780

-- Define the circle and points
def Circle : Type := ℂ → Prop
def on_circle (c : Circle) (p : ℂ) : Prop := c p

-- Define the orthocenter function
def orthocenter (a b c : ℂ) : ℂ := sorry

-- Main theorem
theorem right_triangle_of_orthocenters 
  (O A B C D E : ℂ) 
  (c : Circle)
  (on_circle_A : on_circle c A)
  (on_circle_B : on_circle c B)
  (on_circle_C : on_circle c C)
  (on_circle_D : on_circle c D)
  (on_circle_E : on_circle c E)
  (consecutive : sorry) -- Represent that A, B, C, D, E are consecutive
  (equal_chords : AC = BD ∧ BD = CE ∧ CE = DO)
  (H₁ : ℂ := orthocenter A C D)
  (H₂ : ℂ := orthocenter B C D)
  (H₃ : ℂ := orthocenter B C E) :
  ∃ (θ : ℝ), Complex.arg ((H₁ - H₂) / (H₁ - H₃)) = θ ∧ θ = π/2 := by sorry

#check right_triangle_of_orthocenters

end right_triangle_of_orthocenters_l3447_344780


namespace school_population_relation_l3447_344790

theorem school_population_relation 
  (X : ℝ) -- Total number of students
  (p : ℝ) -- Percentage of boys that 90 students represent
  (h1 : X > 0) -- Assumption that the school has a positive number of students
  (h2 : 0 < p ∧ p < 100) -- Assumption that p is a valid percentage
  : 90 = p / 100 * 0.5 * X := by
  sorry

end school_population_relation_l3447_344790


namespace short_trees_calculation_l3447_344710

/-- The number of short trees currently in the park -/
def current_short_trees : ℕ := 112

/-- The number of short trees to be planted -/
def trees_to_plant : ℕ := 105

/-- The total number of short trees after planting -/
def total_short_trees : ℕ := 217

/-- Theorem stating that the current number of short trees plus the number of trees to be planted equals the total number of short trees after planting -/
theorem short_trees_calculation :
  current_short_trees + trees_to_plant = total_short_trees := by sorry

end short_trees_calculation_l3447_344710


namespace magic_mike_calculation_l3447_344712

/-- The problem statement --/
theorem magic_mike_calculation (p q r s t : ℝ) : 
  p = 3 ∧ q = 4 ∧ r = 5 ∧ s = 6 →
  (p - q + r * s - t = p - (q - (r * (s - t)))) →
  t = 0 := by
sorry

end magic_mike_calculation_l3447_344712


namespace alcohol_percentage_original_mixture_l3447_344742

/-- Proves that the percentage of alcohol in the original mixture is 20% --/
theorem alcohol_percentage_original_mixture :
  let original_volume : ℝ := 15
  let added_water : ℝ := 5
  let new_alcohol_percentage : ℝ := 15
  let new_volume : ℝ := original_volume + added_water
  let original_alcohol_volume : ℝ := original_volume * (original_alcohol_percentage / 100)
  let new_alcohol_volume : ℝ := new_volume * (new_alcohol_percentage / 100)
  ∀ original_alcohol_percentage : ℝ,
    original_alcohol_volume = new_alcohol_volume →
    original_alcohol_percentage = 20 :=
by
  sorry


end alcohol_percentage_original_mixture_l3447_344742


namespace missing_sale_is_3920_l3447_344767

/-- Calculates the missing sale amount given the sales for 5 months and the desired average -/
def calculate_missing_sale (sales : List ℕ) (average : ℕ) : ℕ :=
  6 * average - sales.sum

/-- The list of known sales amounts -/
def known_sales : List ℕ := [3435, 3855, 4230, 3560, 2000]

/-- The desired average sale -/
def desired_average : ℕ := 3500

theorem missing_sale_is_3920 :
  calculate_missing_sale known_sales desired_average = 3920 := by
  sorry

#eval calculate_missing_sale known_sales desired_average

end missing_sale_is_3920_l3447_344767


namespace isosceles_minimizes_side_l3447_344714

/-- Represents a triangle with sides a, b, c and angle α opposite to side a -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  α : ℝ
  area : ℝ
  h_positive : a > 0 ∧ b > 0 ∧ c > 0 ∧ α > 0 ∧ area > 0
  h_angle : α < π
  h_area : area = (1/2) * b * c * Real.sin α

/-- Given a fixed angle α and area S, the triangle that minimizes side a is isosceles with b = c -/
theorem isosceles_minimizes_side (α S : ℝ) (h_α : 0 < α ∧ α < π) (h_S : S > 0) :
  ∃ (t : Triangle), t.α = α ∧ t.area = S ∧ t.b = t.c ∧
  ∀ (u : Triangle), u.α = α → u.area = S → t.a ≤ u.a :=
sorry

end isosceles_minimizes_side_l3447_344714


namespace age_difference_l3447_344753

/-- Given the ages of Mandy, her brother, and her sister, prove the age difference between Mandy and her sister. -/
theorem age_difference (mandy_age brother_age sister_age : ℕ) 
  (h1 : mandy_age = 3)
  (h2 : brother_age = 4 * mandy_age)
  (h3 : sister_age = brother_age - 5) :
  sister_age - mandy_age = 4 := by
  sorry

end age_difference_l3447_344753


namespace triangle_properties_l3447_344797

theorem triangle_properties (a b c : ℝ) (A B C : ℝ) :
  a > 0 ∧ b > 0 ∧ c > 0 →
  A > 0 ∧ B > 0 ∧ C > 0 →
  A + B + C = π →
  b = a * Real.cos C + (Real.sqrt 3 / 3) * a * Real.sin C →
  a = 2 →
  b + c ≥ 4 →
  A = π / 3 ∧ (1 / 2) * a * b * Real.sin C = Real.sqrt 3 := by
  sorry

end triangle_properties_l3447_344797


namespace greatest_common_divisor_480_90_under_60_l3447_344703

theorem greatest_common_divisor_480_90_under_60 : 
  ∃ n : ℕ, n > 0 ∧ 
    n ∣ 480 ∧ 
    n < 60 ∧ 
    n ∣ 90 ∧ 
    ∀ m : ℕ, m > 0 → m ∣ 480 → m < 60 → m ∣ 90 → m ≤ n :=
by
  use 30
  sorry

end greatest_common_divisor_480_90_under_60_l3447_344703


namespace b_contribution_is_9000_l3447_344729

/-- Represents the business partnership between A and B -/
structure Partnership where
  a_initial_investment : ℕ
  b_join_month : ℕ
  total_months : ℕ
  profit_ratio_a : ℕ
  profit_ratio_b : ℕ

/-- Calculates B's contribution to the capital given the partnership details -/
def calculate_b_contribution (p : Partnership) : ℕ :=
  sorry

/-- Theorem stating that B's contribution is 9000 rupees given the problem conditions -/
theorem b_contribution_is_9000 :
  let p : Partnership := {
    a_initial_investment := 3500,
    b_join_month := 5,
    total_months := 12,
    profit_ratio_a := 2,
    profit_ratio_b := 3
  }
  calculate_b_contribution p = 9000 := by
  sorry

end b_contribution_is_9000_l3447_344729


namespace train_passing_jogger_l3447_344761

/-- Time for a train to pass a jogger given their speeds and initial positions -/
theorem train_passing_jogger 
  (jogger_speed : ℝ) 
  (train_speed : ℝ) 
  (train_length : ℝ) 
  (initial_distance : ℝ) 
  (h1 : jogger_speed = 9 * (1000 / 3600))
  (h2 : train_speed = 45 * (1000 / 3600))
  (h3 : train_length = 120)
  (h4 : initial_distance = 200) :
  (initial_distance + train_length) / (train_speed - jogger_speed) = 32 := by
  sorry

#check train_passing_jogger

end train_passing_jogger_l3447_344761


namespace gcf_32_48_l3447_344724

theorem gcf_32_48 : Nat.gcd 32 48 = 16 := by
  sorry

end gcf_32_48_l3447_344724


namespace boys_from_clay_middle_school_l3447_344779

/-- Represents the three schools in the problem -/
inductive School
| Jonas
| Clay
| Pine

/-- Represents the gender of students -/
inductive Gender
| Boy
| Girl

/-- The total number of students at the camp -/
def total_students : ℕ := 150

/-- The number of boys at the camp -/
def total_boys : ℕ := 80

/-- The number of girls at the camp -/
def total_girls : ℕ := 70

/-- The number of students from each school -/
def students_per_school (s : School) : ℕ :=
  match s with
  | School.Jonas => 50
  | School.Clay => 60
  | School.Pine => 40

/-- The number of girls from Jonas Middle School -/
def girls_from_jonas : ℕ := 30

/-- The number of boys from Pine Middle School -/
def boys_from_pine : ℕ := 15

/-- The main theorem to prove -/
theorem boys_from_clay_middle_school :
  (students_per_school School.Clay) -
  (students_per_school School.Clay - 
   (total_boys - boys_from_pine - (students_per_school School.Jonas - girls_from_jonas))) = 45 := by
  sorry

end boys_from_clay_middle_school_l3447_344779


namespace distance_product_range_l3447_344722

-- Define the curves C₁ and C₂
def C₁ (x y : ℝ) : Prop := y^2 = 4*x
def C₂ (x y : ℝ) : Prop := (x - 4)^2 + y^2 = 8

-- Define the point P on C₁
def P (t : ℝ) : ℝ × ℝ := (t^2, 2*t)

-- Define the line l passing through P with slope 1
def l (t x : ℝ) : ℝ := x + 2*t - t^2

-- Define the product of distances |PQ||PR|
def distance_product (t : ℝ) : ℝ := (t^2 - 2)^2 + 4

-- Main theorem
theorem distance_product_range :
  ∀ t : ℝ, C₁ (P t).1 (P t).2 →
    (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
      C₂ x₁ (l t x₁) ∧ C₂ x₂ (l t x₂)) →
    distance_product t ∈ Set.Icc 4 8 ∪ Set.Ioo 8 36 :=
by sorry

end distance_product_range_l3447_344722


namespace solve_sqrt_equation_l3447_344731

theorem solve_sqrt_equation (x : ℝ) :
  Real.sqrt ((2 / x) + 3) = 4 / 3 → x = -18 / 11 := by
  sorry

end solve_sqrt_equation_l3447_344731


namespace percentage_commutation_l3447_344772

theorem percentage_commutation (x : ℝ) (h : 0.3 * (0.4 * x) = 48) : 
  0.4 * (0.3 * x) = 48 := by
  sorry

end percentage_commutation_l3447_344772


namespace boat_speed_in_still_water_l3447_344778

/-- The speed of a boat in still water, given stream speed and downstream travel data -/
theorem boat_speed_in_still_water (stream_speed : ℝ) (downstream_distance : ℝ) (downstream_time : ℝ) : 
  stream_speed = 4 →
  downstream_distance = 112 →
  downstream_time = 4 →
  (downstream_distance / downstream_time) - stream_speed = 24 := by
  sorry

end boat_speed_in_still_water_l3447_344778


namespace sum_of_distances_is_ten_l3447_344732

/-- Given a circle tangent to the sides of an angle at points A and B, with a point C on the circle,
    this structure represents the distances and conditions of the problem. -/
structure CircleTangentProblem where
  -- Distance from C to line AB
  h : ℝ
  -- Distance from C to the side of the angle passing through A
  h_A : ℝ
  -- Distance from C to the side of the angle passing through B
  h_B : ℝ
  -- Condition: h = 4
  h_eq_four : h = 4
  -- Condition: One distance is four times the other
  one_distance_four_times_other : h_B = 4 * h_A

/-- The theorem stating that the sum of distances from C to the sides of the angle is 10. -/
theorem sum_of_distances_is_ten (p : CircleTangentProblem) : p.h_A + p.h_B = 10 := by
  sorry

end sum_of_distances_is_ten_l3447_344732


namespace correct_rows_per_bus_l3447_344774

/-- Represents the number of rows in each bus -/
def rows_per_bus : ℕ := 10

/-- Represents the number of columns in each bus -/
def columns_per_bus : ℕ := 4

/-- Represents the total number of buses -/
def total_buses : ℕ := 6

/-- Represents the total number of students that can be accommodated -/
def total_students : ℕ := 240

/-- Theorem stating that the number of rows per bus is correct -/
theorem correct_rows_per_bus : 
  rows_per_bus * columns_per_bus * total_buses = total_students := by
  sorry

end correct_rows_per_bus_l3447_344774


namespace deepak_age_l3447_344769

/-- Given the ratio between Rahul and Deepak's ages is 4:3, and that Rahul will be 26 years old after 6 years, prove that Deepak's present age is 15 years. -/
theorem deepak_age (rahul_age deepak_age : ℕ) : 
  rahul_age = 4 * (rahul_age / 4) → 
  deepak_age = 3 * (rahul_age / 4) → 
  rahul_age + 6 = 26 → 
  deepak_age = 15 := by
sorry

end deepak_age_l3447_344769


namespace edward_rides_l3447_344726

theorem edward_rides (total_tickets : ℕ) (spent_tickets : ℕ) (cost_per_ride : ℕ) : 
  total_tickets = 79 → spent_tickets = 23 → cost_per_ride = 7 →
  (total_tickets - spent_tickets) / cost_per_ride = 8 := by
sorry

end edward_rides_l3447_344726


namespace base_8_to_base_7_l3447_344757

def base_8_to_decimal (n : ℕ) : ℕ := n

def decimal_to_base_7 (n : ℕ) : ℕ := n

theorem base_8_to_base_7 :
  decimal_to_base_7 (base_8_to_decimal 536) = 1010 :=
sorry

end base_8_to_base_7_l3447_344757


namespace ellipse_circle_intersection_l3447_344747

/-- Given an ellipse and a circle with specific properties, prove that a line passing through the origin and intersecting the circle at two points satisfying a dot product condition has specific equations. -/
theorem ellipse_circle_intersection (a b : ℝ) (h₁ : a > b) (h₂ : b > 0) : 
  let e := Real.sqrt 3 / 2
  let t_area := Real.sqrt 3
  let c := Real.sqrt (a^2 - b^2)
  let ellipse := fun (x y : ℝ) ↦ x^2 / a^2 + y^2 / b^2 = 1
  let circle := fun (x y : ℝ) ↦ (x - a)^2 + (y - b)^2 = (a / b)^2
  ∀ (x₁ y₁ x₂ y₂ : ℝ),
    e = c / a →
    t_area = c * b →
    (∃ k : ℝ, (y₁ = k * x₁ ∧ y₂ = k * x₂) ∨ (x₁ = 0 ∧ x₂ = 0)) →
    circle x₁ y₁ →
    circle x₂ y₂ →
    (x₁ - a) * (x₂ - a) + (y₁ - b) * (y₂ - b) = -2 →
    (y₁ = 0 ∧ y₂ = 0) ∨ (∃ k : ℝ, k = 4/3 ∧ y₁ = k * x₁ ∧ y₂ = k * x₂) :=
by sorry

end ellipse_circle_intersection_l3447_344747


namespace right_triangle_third_side_l3447_344730

theorem right_triangle_third_side (a b c : ℝ) : 
  a > 0 → b > 0 → c > 0 →
  (a = 6 ∧ b = 8 ∧ c^2 = a^2 + b^2) ∨ 
  (a = 6 ∧ c = 8 ∧ b^2 = c^2 - a^2) ∨
  (b = 6 ∧ c = 8 ∧ a^2 = c^2 - b^2) →
  c = 10 ∨ c = 2 * Real.sqrt 7 ∨ b = 10 ∨ b = 2 * Real.sqrt 7 ∨ a = 10 ∨ a = 2 * Real.sqrt 7 :=
by sorry

end right_triangle_third_side_l3447_344730


namespace min_value_line_circle_l3447_344785

/-- Given a line ax + by + c - 1 = 0 (where b, c > 0) passing through the center of the circle x^2 + y^2 - 2y - 5 = 0, 
    the minimum value of 4/b + 1/c is 9. -/
theorem min_value_line_circle (a b c : ℝ) : 
  b > 0 → c > 0 → 
  (∃ x y : ℝ, a * x + b * y + c - 1 = 0 ∧ x^2 + y^2 - 2*y - 5 = 0) →
  (∀ b' c' : ℝ, b' > 0 → c' > 0 → 
    (∃ x y : ℝ, a * x + b' * y + c' - 1 = 0 ∧ x^2 + y^2 - 2*y - 5 = 0) →
    4/b + 1/c ≤ 4/b' + 1/c') →
  4/b + 1/c = 9 :=
by sorry


end min_value_line_circle_l3447_344785


namespace square_is_quadratic_l3447_344704

/-- A function f: ℝ → ℝ is quadratic if there exist constants a, b, c with a ≠ 0 such that
    f(x) = a * x^2 + b * x + c for all x -/
def IsQuadratic (f : ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

/-- The function f(x) = x^2 is quadratic -/
theorem square_is_quadratic : IsQuadratic (fun x ↦ x^2) := by
  sorry

end square_is_quadratic_l3447_344704


namespace range_of_a_for_sqrt_function_l3447_344735

theorem range_of_a_for_sqrt_function (f : ℝ → ℝ) (a : ℝ) : 
  (∀ x, f x = Real.sqrt (x^2 + 2*a*x + 1)) → 
  (∀ x, ∃ y, f x = y) →
  -1 ≤ a ∧ a ≤ 1 :=
by sorry

end range_of_a_for_sqrt_function_l3447_344735


namespace total_lollipops_eq_twelve_l3447_344758

/-- The number of lollipops Sushi's father brought -/
def total_lollipops : ℕ := sorry

/-- The number of lollipops eaten by the children -/
def eaten_lollipops : ℕ := 5

/-- The number of lollipops left -/
def remaining_lollipops : ℕ := 7

/-- Theorem stating that the total number of lollipops equals 12 -/
theorem total_lollipops_eq_twelve :
  total_lollipops = eaten_lollipops + remaining_lollipops ∧
  total_lollipops = 12 := by sorry

end total_lollipops_eq_twelve_l3447_344758


namespace triple_f_of_3_l3447_344765

def f (x : ℝ) : ℝ := 7 * x - 3

theorem triple_f_of_3 : f (f (f 3)) = 858 := by sorry

end triple_f_of_3_l3447_344765


namespace carla_lemonade_consumption_l3447_344715

/-- The number of glasses of lemonade Carla can drink in a given time period. -/
def glasses_of_lemonade (time_minutes : ℕ) (rate_minutes : ℕ) : ℕ :=
  time_minutes / rate_minutes

/-- Proves that Carla can drink 11 glasses of lemonade in 3 hours and 40 minutes. -/
theorem carla_lemonade_consumption : 
  glasses_of_lemonade 220 20 = 11 := by
  sorry

#eval glasses_of_lemonade 220 20

end carla_lemonade_consumption_l3447_344715


namespace cost_per_item_proof_l3447_344721

/-- The cost per item in the first batch of fruits -/
def cost_per_item_first_batch : ℝ := 120

/-- The total cost of the first batch of fruits -/
def total_cost_first_batch : ℝ := 600

/-- The total cost of the second batch of fruits -/
def total_cost_second_batch : ℝ := 1250

/-- The number of items in the second batch is twice the number in the first batch -/
axiom double_items : ∃ n : ℝ, n * cost_per_item_first_batch = total_cost_first_batch ∧
                               2 * n * (cost_per_item_first_batch + 5) = total_cost_second_batch

theorem cost_per_item_proof : 
  cost_per_item_first_batch = 120 :=
sorry

end cost_per_item_proof_l3447_344721


namespace divisible_by_50_l3447_344702

/-- A polygon drawn on a square grid -/
structure GridPolygon where
  area : ℕ
  divisible_by_2 : ∃ (half : ℕ), area = 2 * half
  divisible_by_25 : ∃ (part : ℕ), area = 25 * part

/-- The main theorem -/
theorem divisible_by_50 (p : GridPolygon) (h : p.area = 100) :
  ∃ (small : ℕ), p.area = 50 * small := by
  sorry

end divisible_by_50_l3447_344702


namespace product_magnitude_l3447_344748

open Complex

theorem product_magnitude (z₁ z₂ : ℂ) (h1 : abs z₁ = 3) (h2 : z₂ = 2 + I) : 
  abs (z₁ * z₂) = 3 * Real.sqrt 5 := by
  sorry

end product_magnitude_l3447_344748
