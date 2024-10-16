import Mathlib

namespace NUMINAMATH_CALUDE_initial_water_was_six_cups_l2667_266704

/-- Represents the water consumption during a hike --/
structure HikeWaterConsumption where
  total_distance : ℝ
  total_time : ℝ
  remaining_water : ℝ
  leak_rate : ℝ
  last_mile_consumption : ℝ
  first_three_miles_rate : ℝ

/-- Calculates the initial amount of water in the canteen --/
def initial_water (h : HikeWaterConsumption) : ℝ :=
  h.remaining_water + h.leak_rate * h.total_time + 
  h.last_mile_consumption + h.first_three_miles_rate * (h.total_distance - 1)

/-- Theorem stating that the initial amount of water in the canteen was 6 cups --/
theorem initial_water_was_six_cups (h : HikeWaterConsumption) 
  (h_distance : h.total_distance = 4)
  (h_time : h.total_time = 2)
  (h_remaining : h.remaining_water = 1)
  (h_leak : h.leak_rate = 1)
  (h_last_mile : h.last_mile_consumption = 1)
  (h_first_three : h.first_three_miles_rate = 2/3) :
  initial_water h = 6 := by
  sorry


end NUMINAMATH_CALUDE_initial_water_was_six_cups_l2667_266704


namespace NUMINAMATH_CALUDE_power_relation_l2667_266757

theorem power_relation (a m n : ℝ) (h1 : a^m = 8) (h2 : a^n = 4) : a^(m-2*n) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_power_relation_l2667_266757


namespace NUMINAMATH_CALUDE_sin_product_equals_one_sixteenth_l2667_266766

theorem sin_product_equals_one_sixteenth : 
  Real.sin (6 * π / 180) * Real.sin (42 * π / 180) * Real.sin (66 * π / 180) * Real.sin (78 * π / 180) = 1 / 16 := by
  sorry

end NUMINAMATH_CALUDE_sin_product_equals_one_sixteenth_l2667_266766


namespace NUMINAMATH_CALUDE_minimum_force_to_submerge_cube_l2667_266714

-- Define constants
def cube_volume : Real := 10e-6  -- 10 cm³ converted to m³
def cube_density : Real := 400   -- kg/m³
def water_density : Real := 1000 -- kg/m³
def gravity : Real := 10         -- m/s²

-- Define the minimum force function
def minimum_submerge_force (v : Real) (ρ_cube : Real) (ρ_water : Real) (g : Real) : Real :=
  (ρ_water - ρ_cube) * v * g

-- Theorem statement
theorem minimum_force_to_submerge_cube :
  minimum_submerge_force cube_volume cube_density water_density gravity = 0.06 := by
  sorry

end NUMINAMATH_CALUDE_minimum_force_to_submerge_cube_l2667_266714


namespace NUMINAMATH_CALUDE_sheep_price_is_30_l2667_266708

/-- Represents the farm animals and their sale --/
structure FarmSale where
  goats : ℕ
  sheep : ℕ
  goat_price : ℕ
  sheep_price : ℕ
  goats_sold_ratio : ℚ
  sheep_sold_ratio : ℚ
  total_sale : ℕ

/-- The conditions of the farm sale problem --/
def farm_conditions (s : FarmSale) : Prop :=
  s.goats * 7 = s.sheep * 5 ∧
  s.goats + s.sheep = 360 ∧
  s.goats_sold_ratio = 1/2 ∧
  s.sheep_sold_ratio = 2/3 ∧
  s.goat_price = 40 ∧
  s.total_sale = 7200

/-- The theorem stating that the sheep price is $30 --/
theorem sheep_price_is_30 (s : FarmSale) (h : farm_conditions s) : s.sheep_price = 30 := by
  sorry


end NUMINAMATH_CALUDE_sheep_price_is_30_l2667_266708


namespace NUMINAMATH_CALUDE_zacks_marbles_l2667_266705

theorem zacks_marbles (initial_marbles : ℕ) (kept_marbles : ℕ) (num_friends : ℕ) : 
  initial_marbles = 65 → 
  kept_marbles = 5 → 
  num_friends = 3 → 
  (initial_marbles - kept_marbles) / num_friends = 20 := by
sorry

end NUMINAMATH_CALUDE_zacks_marbles_l2667_266705


namespace NUMINAMATH_CALUDE_triangle_side_relation_l2667_266789

/-- Given a triangle ABC where the angles satisfy the equation 3α + 2β = 180°,
    prove that a^2 + bc = c^2, where a, b, and c are the lengths of the sides
    opposite to angles α, β, and γ respectively. -/
theorem triangle_side_relation (α β γ a b c : ℝ) : 
  0 < α ∧ 0 < β ∧ 0 < γ ∧ 
  α + β + γ = Real.pi ∧
  3 * α + 2 * β = Real.pi →
  a^2 + b * c = c^2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_relation_l2667_266789


namespace NUMINAMATH_CALUDE_travel_time_equation_l2667_266710

theorem travel_time_equation (x : ℝ) : x > 3 → 
  (30 / (x - 3) - 30 / x = 40 / 60) ↔ 
  (30 = (x - 3) * (40 / 60) ∧ 30 = x * ((40 / 60) + (30 / (x - 3)))) := by
  sorry

#check travel_time_equation

end NUMINAMATH_CALUDE_travel_time_equation_l2667_266710


namespace NUMINAMATH_CALUDE_min_students_theorem_l2667_266771

/-- The minimum number of students that can be divided into either 18 or 24 teams 
    with a maximum difference of 2 students between team sizes. -/
def min_students : ℕ := 70

/-- Checks if a number can be divided into a given number of teams
    with a maximum difference of 2 students between team sizes. -/
def can_divide (n : ℕ) (teams : ℕ) : Prop :=
  ∃ (base_size : ℕ), 
    (n ≥ base_size * teams) ∧ 
    (n ≤ (base_size + 2) * teams)

theorem min_students_theorem : 
  (can_divide min_students 18) ∧ 
  (can_divide min_students 24) ∧ 
  (∀ m : ℕ, m < min_students → ¬(can_divide m 18 ∧ can_divide m 24)) :=
sorry

end NUMINAMATH_CALUDE_min_students_theorem_l2667_266771


namespace NUMINAMATH_CALUDE_max_sum_digits_24hour_clock_l2667_266758

/-- Represents a time in 24-hour format -/
structure Time24 where
  hours : Fin 24
  minutes : Fin 60

/-- Calculates the sum of digits in a natural number -/
def sumDigits (n : ℕ) : ℕ := sorry

/-- Checks if a natural number is even -/
def isEven (n : ℕ) : Prop := ∃ k, n = 2 * k

/-- Calculates the total sum of digits in a Time24 -/
def totalSumDigits (t : Time24) : ℕ :=
  sumDigits t.hours.val + sumDigits t.minutes.val

/-- The theorem to be proved -/
theorem max_sum_digits_24hour_clock :
  ∃ (t : Time24), 
    (isEven (sumDigits t.hours.val)) ∧ 
    (∀ (t' : Time24), isEven (sumDigits t'.hours.val) → totalSumDigits t' ≤ totalSumDigits t) ∧
    totalSumDigits t = 22 := by sorry

end NUMINAMATH_CALUDE_max_sum_digits_24hour_clock_l2667_266758


namespace NUMINAMATH_CALUDE_triangle_right_angle_l2667_266747

theorem triangle_right_angle (A B C : ℝ) (h : A = B - C) : B = 90 := by
  sorry

end NUMINAMATH_CALUDE_triangle_right_angle_l2667_266747


namespace NUMINAMATH_CALUDE_problem_solution_l2667_266718

def sum_of_digits (n : ℕ) : ℕ := sorry

theorem problem_solution (n : ℕ) : 2 * n * sum_of_digits (3 * n) = 2022 → n = 337 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l2667_266718


namespace NUMINAMATH_CALUDE_polynomial_characterization_l2667_266794

/-- A polynomial satisfying the given condition -/
def SatisfyingPolynomial (P : ℝ → ℝ) : Prop :=
  ∀ (a b c : ℝ), a*b + b*c + c*a = 0 →
    P (a-b) + P (b-c) + P (c-a) = 2 * P (a+b+c)

/-- The form of the polynomial that satisfies the condition -/
def PolynomialForm (P : ℝ → ℝ) : Prop :=
  ∃ (r s : ℝ), ∀ x, P x = r * x^4 + s * x^2

theorem polynomial_characterization :
  ∀ P : ℝ → ℝ, SatisfyingPolynomial P ↔ PolynomialForm P :=
sorry

end NUMINAMATH_CALUDE_polynomial_characterization_l2667_266794


namespace NUMINAMATH_CALUDE_problem_solution_l2667_266730

theorem problem_solution (x y z : ℝ) 
  (h_pos_x : x > 0) (h_pos_y : y > 0) (h_pos_z : z > 0)
  (h1 : x^2 / y = 2)
  (h2 : y^2 / z = 3)
  (h3 : z^2 / x = 4) :
  x = 576^(1/7) := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l2667_266730


namespace NUMINAMATH_CALUDE_purely_imaginary_complex_number_l2667_266728

theorem purely_imaginary_complex_number (a : ℝ) :
  let z : ℂ := (a^2 - 2*a) + (a - 2)*Complex.I
  (∀ x : ℝ, z = x*Complex.I) ↔ a = 0 := by
  sorry

end NUMINAMATH_CALUDE_purely_imaginary_complex_number_l2667_266728


namespace NUMINAMATH_CALUDE_product_from_sum_and_difference_l2667_266712

theorem product_from_sum_and_difference (x y : ℝ) 
  (sum_eq : x + y = 19) 
  (diff_eq : x - y = 5) : 
  x * y = 84 := by
sorry

end NUMINAMATH_CALUDE_product_from_sum_and_difference_l2667_266712


namespace NUMINAMATH_CALUDE_smallest_x_solution_l2667_266715

theorem smallest_x_solution (w x y z : ℝ) 
  (non_neg : w ≥ 0 ∧ x ≥ 0 ∧ y ≥ 0 ∧ z ≥ 0)
  (eq1 : y = x - 2003)
  (eq2 : z = 2*y - 2003)
  (eq3 : w = 3*z - 2003) :
  x ≥ 10015/3 ∧ 
  (x = 10015/3 → y = 4006/3 ∧ z = 2003/3 ∧ w = 0) :=
by sorry

end NUMINAMATH_CALUDE_smallest_x_solution_l2667_266715


namespace NUMINAMATH_CALUDE_total_books_combined_l2667_266725

theorem total_books_combined (bryan_books_per_shelf : ℕ) (bryan_shelves : ℕ) 
  (alyssa_books_per_shelf : ℕ) (alyssa_shelves : ℕ) : 
  bryan_books_per_shelf = 56 → 
  bryan_shelves = 9 → 
  alyssa_books_per_shelf = 73 → 
  alyssa_shelves = 12 → 
  bryan_books_per_shelf * bryan_shelves + alyssa_books_per_shelf * alyssa_shelves = 1380 := by
  sorry

end NUMINAMATH_CALUDE_total_books_combined_l2667_266725


namespace NUMINAMATH_CALUDE_share_price_is_31_l2667_266752

/-- The price at which an investor bought shares, given the dividend rate,
    face value, and return on investment. -/
def share_purchase_price (dividend_rate : ℚ) (face_value : ℚ) (roi : ℚ) : ℚ :=
  (dividend_rate * face_value) / roi

/-- Theorem stating that under the given conditions, the share purchase price is 31. -/
theorem share_price_is_31 :
  let dividend_rate : ℚ := 155 / 1000
  let face_value : ℚ := 50
  let roi : ℚ := 1 / 4
  share_purchase_price dividend_rate face_value roi = 31 := by
  sorry

end NUMINAMATH_CALUDE_share_price_is_31_l2667_266752


namespace NUMINAMATH_CALUDE_arithmetic_sequence_first_term_l2667_266711

/-- An arithmetic sequence is a sequence where the difference between each consecutive term is constant. -/
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d

/-- The median of a sequence with an odd number of terms is the middle term. -/
def median (a : ℕ → ℝ) (n : ℕ) : ℝ := a ((n + 1) / 2)

theorem arithmetic_sequence_first_term
  (a : ℕ → ℝ)  -- The sequence
  (n : ℕ)      -- The number of terms in the sequence
  (h1 : is_arithmetic_sequence a)
  (h2 : median a n = 1010)
  (h3 : a n = 2015) :
  a 1 = 5 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_first_term_l2667_266711


namespace NUMINAMATH_CALUDE_squirrel_cones_problem_l2667_266779

theorem squirrel_cones_problem :
  ∃ (x y : ℕ), 
    x + y < 25 ∧
    2 * x > y + 26 ∧
    2 * y > x - 4 ∧
    x = 17 ∧
    y = 7 := by
  sorry

end NUMINAMATH_CALUDE_squirrel_cones_problem_l2667_266779


namespace NUMINAMATH_CALUDE_subcommittee_formation_count_l2667_266792

def senate_committee_size : ℕ := 18
def num_republicans : ℕ := 10
def num_democrats : ℕ := 8
def subcommittee_size : ℕ := 5
def min_republicans : ℕ := 2

theorem subcommittee_formation_count :
  (Finset.range (subcommittee_size - min_republicans + 1)).sum (λ k =>
    Nat.choose num_republicans (min_republicans + k) *
    Nat.choose num_democrats (subcommittee_size - (min_republicans + k))
  ) = 7812 := by
  sorry

end NUMINAMATH_CALUDE_subcommittee_formation_count_l2667_266792


namespace NUMINAMATH_CALUDE_c_investment_is_1200_l2667_266759

/-- Represents the investment and profit distribution of a business partnership --/
structure BusinessPartnership where
  investmentA : ℕ
  investmentB : ℕ
  investmentC : ℕ
  totalProfit : ℕ
  profitShareC : ℕ

/-- Calculates C's investment amount based on the given conditions --/
def calculateInvestmentC (bp : BusinessPartnership) : ℕ :=
  sorry

/-- Theorem stating that C's investment is 1200 given the specified conditions --/
theorem c_investment_is_1200 : 
  ∀ (bp : BusinessPartnership), 
  bp.investmentA = 800 ∧ 
  bp.investmentB = 1000 ∧ 
  bp.totalProfit = 1000 ∧ 
  bp.profitShareC = 400 →
  calculateInvestmentC bp = 1200 :=
sorry

end NUMINAMATH_CALUDE_c_investment_is_1200_l2667_266759


namespace NUMINAMATH_CALUDE_line_perp_parallel_planes_l2667_266760

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular relation between a line and a plane
variable (perp : Line → Plane → Prop)

-- Define the parallel relation between two planes
variable (para : Plane → Plane → Prop)

-- State the theorem
theorem line_perp_parallel_planes 
  (l : Line) (α β : Plane) 
  (h1 : perp l β) (h2 : para α β) : 
  perp l α :=
sorry

end NUMINAMATH_CALUDE_line_perp_parallel_planes_l2667_266760


namespace NUMINAMATH_CALUDE_simple_interest_problem_l2667_266796

/-- Given a principal amount P and an interest rate r, 
    if P(1 + 2r) = 710 and P(1 + 7r) = 1020, then P = 586 -/
theorem simple_interest_problem (P r : ℝ) 
  (h1 : P * (1 + 2 * r) = 710)
  (h2 : P * (1 + 7 * r) = 1020) : 
  P = 586 := by sorry

end NUMINAMATH_CALUDE_simple_interest_problem_l2667_266796


namespace NUMINAMATH_CALUDE_volume_of_regular_triangular_truncated_pyramid_l2667_266790

/-- A regular triangular truncated pyramid -/
structure RegularTriangularTruncatedPyramid where
  /-- Height of the pyramid -/
  H : ℝ
  /-- Angle between lateral edge and base -/
  α : ℝ
  /-- H is positive -/
  H_pos : 0 < H
  /-- α is between 0 and π/2 -/
  α_range : 0 < α ∧ α < Real.pi / 2
  /-- H is the geometric mean between the sides of the bases -/
  H_is_geometric_mean : ∃ a b : ℝ, 0 < b ∧ b < a ∧ H^2 = a * b

/-- Volume of a regular triangular truncated pyramid -/
noncomputable def volume (p : RegularTriangularTruncatedPyramid) : ℝ :=
  (p.H^3 * Real.sqrt 3) / (4 * Real.sin p.α ^ 2)

/-- Theorem stating the volume of a regular triangular truncated pyramid -/
theorem volume_of_regular_triangular_truncated_pyramid (p : RegularTriangularTruncatedPyramid) :
  volume p = (p.H^3 * Real.sqrt 3) / (4 * Real.sin p.α ^ 2) := by sorry

end NUMINAMATH_CALUDE_volume_of_regular_triangular_truncated_pyramid_l2667_266790


namespace NUMINAMATH_CALUDE_triangle_ABC_properties_l2667_266701

theorem triangle_ABC_properties (A B C : ℝ) :
  0 < A ∧ A < 2 * π / 3 →
  0 < B ∧ B < π →
  0 < C ∧ C < π →
  A + B + C = π →
  Real.cos C + (Real.cos A - Real.sqrt 3 * Real.sin A) * Real.cos B = 0 →
  Real.sin (A - π / 3) = 3 / 5 →
  B = π / 3 ∧ Real.sin (2 * C) = (24 + 7 * Real.sqrt 3) / 50 := by
  sorry

end NUMINAMATH_CALUDE_triangle_ABC_properties_l2667_266701


namespace NUMINAMATH_CALUDE_smallest_n_for_inequality_l2667_266777

theorem smallest_n_for_inequality : ∃ (n : ℕ), n = 4 ∧ (∀ x y z w : ℝ, (x^2 + y^2 + z^2 + w^2)^2 ≤ n * (x^4 + y^4 + z^4 + w^4)) ∧ (∀ m : ℕ, m < n → ∃ x y z w : ℝ, (x^2 + y^2 + z^2 + w^2)^2 > m * (x^4 + y^4 + z^4 + w^4)) :=
sorry

end NUMINAMATH_CALUDE_smallest_n_for_inequality_l2667_266777


namespace NUMINAMATH_CALUDE_parallelogram_area_l2667_266798

/-- The area of a parallelogram with one angle measuring 100 degrees and two consecutive sides of lengths 10 inches and 18 inches is equal to 180 sin(10°) square inches. -/
theorem parallelogram_area (a b : ℝ) (θ : ℝ) (h1 : a = 10) (h2 : b = 18) (h3 : θ = 100 * π / 180) :
  a * b * Real.sin ((π / 2) - (θ / 2)) = 180 * Real.sin (10 * π / 180) := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_area_l2667_266798


namespace NUMINAMATH_CALUDE_distribution_difference_l2667_266786

theorem distribution_difference (total_amount : ℕ) (group1 : ℕ) (group2 : ℕ)
  (h1 : total_amount = 5040)
  (h2 : group1 = 14)
  (h3 : group2 = 18)
  (h4 : group1 < group2) :
  (total_amount / group1) - (total_amount / group2) = 80 := by
  sorry

end NUMINAMATH_CALUDE_distribution_difference_l2667_266786


namespace NUMINAMATH_CALUDE_high_school_nine_games_l2667_266734

/-- The number of teams in the league -/
def num_teams : ℕ := 9

/-- The number of non-league games each team plays -/
def non_league_games : ℕ := 6

/-- Calculate the total number of games in a season -/
def total_games : ℕ := 
  (num_teams * (num_teams - 1) / 2) + (num_teams * non_league_games)

/-- Theorem stating that the total number of games is 90 -/
theorem high_school_nine_games : total_games = 90 := by
  sorry

end NUMINAMATH_CALUDE_high_school_nine_games_l2667_266734


namespace NUMINAMATH_CALUDE_distribute_five_balls_two_boxes_l2667_266791

/-- The number of ways to distribute n distinct objects into two boxes,
    where box 1 contains at least k objects and box 2 contains at least m objects. -/
def distribute (n k m : ℕ) : ℕ :=
  (Finset.range (n - k - m + 1)).sum (λ i => Nat.choose n (k + i))

/-- Theorem: There are 25 ways to distribute 5 distinct objects into two boxes,
    where box 1 contains at least 1 object and box 2 contains at least 2 objects. -/
theorem distribute_five_balls_two_boxes : distribute 5 1 2 = 25 := by
  sorry

end NUMINAMATH_CALUDE_distribute_five_balls_two_boxes_l2667_266791


namespace NUMINAMATH_CALUDE_absolute_value_inequality_l2667_266731

theorem absolute_value_inequality (a : ℝ) :
  (∀ x : ℝ, |x - 4| + |x - 5| > a) ↔ a < 1 := by sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_l2667_266731


namespace NUMINAMATH_CALUDE_sqrt_product_equality_l2667_266764

theorem sqrt_product_equality : Real.sqrt 2 * Real.sqrt 3 = Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_product_equality_l2667_266764


namespace NUMINAMATH_CALUDE_complement_union_is_empty_l2667_266780

universe u

def U : Set ℕ := {1, 2, 3, 4, 5}
def M : Set ℕ := {1, 3, 4}
def N : Set ℕ := {2, 4, 5}

theorem complement_union_is_empty :
  (U \ (M ∪ N)) = ∅ := by
  sorry

end NUMINAMATH_CALUDE_complement_union_is_empty_l2667_266780


namespace NUMINAMATH_CALUDE_library_books_remaining_l2667_266737

/-- Calculates the number of remaining books after two days of borrowing. -/
def remaining_books (initial : ℕ) (day1_borrowed : ℕ) (day2_borrowed : ℕ) : ℕ :=
  initial - (day1_borrowed + day2_borrowed)

/-- Theorem stating the number of remaining books in the library scenario. -/
theorem library_books_remaining :
  remaining_books 100 10 20 = 70 := by
  sorry

end NUMINAMATH_CALUDE_library_books_remaining_l2667_266737


namespace NUMINAMATH_CALUDE_rearrangement_writing_time_l2667_266797

/-- The number of distinct letters in the name --/
def name_length : ℕ := 7

/-- The number of rearrangements that can be written per minute --/
def rearrangements_per_minute : ℕ := 15

/-- The total number of minutes required to write all rearrangements --/
def total_minutes : ℕ := 336

/-- Theorem stating that the total time to write all rearrangements of a 7-letter name
    at a rate of 15 rearrangements per minute is 336 minutes --/
theorem rearrangement_writing_time :
  (Nat.factorial name_length) / rearrangements_per_minute = total_minutes := by
  sorry

end NUMINAMATH_CALUDE_rearrangement_writing_time_l2667_266797


namespace NUMINAMATH_CALUDE_parallelogram_sum_impossibility_l2667_266702

theorem parallelogram_sum_impossibility :
  ¬ ∃ (a b h : ℕ+), (b * h : ℕ) + 2 * a + 2 * b + 6 = 102 :=
by sorry

end NUMINAMATH_CALUDE_parallelogram_sum_impossibility_l2667_266702


namespace NUMINAMATH_CALUDE_total_turtles_l2667_266746

def turtle_problem (lucas rebecca miguel tran pedro kristen kris trey : ℕ) : Prop :=
  lucas = 8 ∧
  rebecca = 2 * lucas ∧
  miguel = rebecca + 10 ∧
  tran = miguel + 5 ∧
  pedro = 2 * tran ∧
  kristen = 3 * pedro ∧
  kris = kristen / 4 ∧
  trey = 5 * kris ∧
  lucas + rebecca + miguel + tran + pedro + kristen + kris + trey = 605

theorem total_turtles :
  ∃ (lucas rebecca miguel tran pedro kristen kris trey : ℕ),
    turtle_problem lucas rebecca miguel tran pedro kristen kris trey :=
by
  sorry

end NUMINAMATH_CALUDE_total_turtles_l2667_266746


namespace NUMINAMATH_CALUDE_cube_root_of_fourth_root_l2667_266717

theorem cube_root_of_fourth_root (a : ℝ) (h : a > 0) :
  (a * a^(1/4))^(1/3) = a^(5/12) := by
  sorry

end NUMINAMATH_CALUDE_cube_root_of_fourth_root_l2667_266717


namespace NUMINAMATH_CALUDE_root_power_sum_relation_l2667_266745

theorem root_power_sum_relation (t : ℕ → ℝ) (d e f : ℝ) : 
  (∃ (r₁ r₂ r₃ : ℝ), r₁^3 - 7*r₁^2 + 12*r₁ - 20 = 0 ∧ 
                      r₂^3 - 7*r₂^2 + 12*r₂ - 20 = 0 ∧ 
                      r₃^3 - 7*r₃^2 + 12*r₃ - 20 = 0 ∧ 
                      ∀ k, t k = r₁^k + r₂^k + r₃^k) →
  t 0 = 3 →
  t 1 = 7 →
  t 2 = 15 →
  (∀ k ≥ 2, t (k+1) = d * t k + e * t (k-1) + f * t (k-2) - 5) →
  d + e + f = 15 := by
sorry

end NUMINAMATH_CALUDE_root_power_sum_relation_l2667_266745


namespace NUMINAMATH_CALUDE_profit_distribution_l2667_266761

theorem profit_distribution (total_profit : ℕ) (a_prop b_prop c_prop : ℕ) 
  (h_total : total_profit = 20000)
  (h_prop : a_prop = 2 ∧ b_prop = 3 ∧ c_prop = 5) :
  let total_parts := a_prop + b_prop + c_prop
  let part_value := total_profit / total_parts
  let b_share := b_prop * part_value
  let c_share := c_prop * part_value
  c_share - b_share = 4000 := by
sorry

end NUMINAMATH_CALUDE_profit_distribution_l2667_266761


namespace NUMINAMATH_CALUDE_proportional_relationship_l2667_266787

/-- Given that x is directly proportional to y^4, y is inversely proportional to z^2,
    and x = 4 when z = 3, prove that x = 3/192 when z = 6. -/
theorem proportional_relationship (x y z : ℝ) (k : ℝ) 
    (h1 : ∃ m : ℝ, x = m * y^4)
    (h2 : ∃ n : ℝ, y = n / z^2)
    (h3 : x = 4 ∧ z = 3 → x * z^8 = k)
    (h4 : x * z^8 = k) :
    z = 6 → x = 3 / 192 := by
  sorry

end NUMINAMATH_CALUDE_proportional_relationship_l2667_266787


namespace NUMINAMATH_CALUDE_pyramid_properties_l2667_266707

/-- Given a sphere of radius r and a regular four-sided pyramid constructed
    such that:
    1. The sphere is divided into two parts
    2. The part towards the center is the mean proportional between the entire radius and the other part
    3. A plane is placed perpendicularly to the radius at the dividing point
    4. The pyramid is constructed in the larger segment of the sphere
    5. The apex of the pyramid is on the surface of the sphere

    Then the following properties hold for the pyramid:
    1. Its volume is 2/3 * r^3
    2. Its surface area is r^2 * (√(2√5 + 10) + √5 - 1)
    3. The tangent of its inclination angle is 1/2 * (√(√5 + 1))^3
-/
theorem pyramid_properties (r : ℝ) (h : r > 0) :
  ∃ (V F : ℝ) (tan_α : ℝ),
    V = 2/3 * r^3 ∧
    F = r^2 * (Real.sqrt (2 * Real.sqrt 5 + 10) + Real.sqrt 5 - 1) ∧
    tan_α = 1/2 * (Real.sqrt (Real.sqrt 5 + 1))^3 :=
sorry

end NUMINAMATH_CALUDE_pyramid_properties_l2667_266707


namespace NUMINAMATH_CALUDE_soccer_handshakes_l2667_266788

theorem soccer_handshakes (team_size : Nat) (referee_count : Nat) : 
  team_size = 11 → referee_count = 3 → 
  (team_size * team_size) + (2 * team_size * referee_count) = 187 := by
  sorry

#check soccer_handshakes

end NUMINAMATH_CALUDE_soccer_handshakes_l2667_266788


namespace NUMINAMATH_CALUDE_max_x0_value_l2667_266748

theorem max_x0_value (x : Fin 1996 → ℝ) 
  (h1 : x 0 = x 1995)
  (h2 : ∀ i : Fin 1995, x i.val + 2 / x i.val = 2 * x (i.val + 1) + 1 / x (i.val + 1))
  (h3 : ∀ i : Fin 1996, x i > 0) :
  x 0 ≤ 2^997 ∧ ∃ y : Fin 1996 → ℝ, 
    y 0 = 2^997 ∧ 
    y 1995 = y 0 ∧ 
    (∀ i : Fin 1995, y i + 2 / y i = 2 * y (i.val + 1) + 1 / y (i.val + 1)) ∧
    (∀ i : Fin 1996, y i > 0) :=
by sorry

end NUMINAMATH_CALUDE_max_x0_value_l2667_266748


namespace NUMINAMATH_CALUDE_multiplicative_inverse_203_mod_301_l2667_266738

theorem multiplicative_inverse_203_mod_301 : ∃ x : ℕ, x < 301 ∧ (203 * x) % 301 = 1 :=
by
  use 29
  sorry

end NUMINAMATH_CALUDE_multiplicative_inverse_203_mod_301_l2667_266738


namespace NUMINAMATH_CALUDE_largest_c_for_three_in_range_l2667_266784

/-- The function f(x) = x^2 - 7x + c -/
def f (c : ℝ) (x : ℝ) : ℝ := x^2 - 7*x + c

/-- 3 is in the range of f -/
def three_in_range (c : ℝ) : Prop := ∃ x, f c x = 3

/-- The largest value of c such that 3 is in the range of f(x) = x^2 - 7x + c is 61/4 -/
theorem largest_c_for_three_in_range :
  (∃ c, three_in_range c ∧ ∀ c', three_in_range c' → c' ≤ c) ∧
  (∀ c, three_in_range c → c ≤ 61/4) :=
sorry

end NUMINAMATH_CALUDE_largest_c_for_three_in_range_l2667_266784


namespace NUMINAMATH_CALUDE_orthocenter_on_altitude_ratio_HD_HA_is_zero_l2667_266772

/-- A triangle with sides 11, 12, and 13 -/
structure Triangle :=
  (a : ℝ) (b : ℝ) (c : ℝ)
  (ha : a = 11)
  (hb : b = 12)
  (hc : c = 13)

/-- The orthocenter of the triangle -/
def orthocenter (t : Triangle) : ℝ × ℝ := sorry

/-- The altitude from vertex A to side BC -/
def altitude_AD (t : Triangle) : ℝ := sorry

/-- The point D where the altitude AD intersects BC -/
def point_D (t : Triangle) : ℝ × ℝ := sorry

/-- The point A of the triangle -/
def point_A (t : Triangle) : ℝ × ℝ := sorry

/-- The distance between two points -/
def distance (p1 p2 : ℝ × ℝ) : ℝ := sorry

theorem orthocenter_on_altitude (t : Triangle) :
  let H := orthocenter t
  let D := point_D t
  distance H D = 0 := by sorry

theorem ratio_HD_HA_is_zero (t : Triangle) :
  let H := orthocenter t
  let D := point_D t
  let A := point_A t
  distance H D / distance H A = 0 := by sorry

end NUMINAMATH_CALUDE_orthocenter_on_altitude_ratio_HD_HA_is_zero_l2667_266772


namespace NUMINAMATH_CALUDE_f_max_and_inequality_l2667_266763

def f (x : ℝ) : ℝ := |x - 1| - 2 * |x + 1|

theorem f_max_and_inequality :
  (∃ (a : ℝ), ∀ x, f x ≤ a ∧ ∃ y, f y = a) ∧
  (∀ m n : ℝ, m > 0 → n > 0 → 1/m + 1/(2*n) = 2 → 2*m + n > 2) :=
sorry

end NUMINAMATH_CALUDE_f_max_and_inequality_l2667_266763


namespace NUMINAMATH_CALUDE_smaller_field_area_l2667_266709

/-- Given two square fields where one is 1% broader than the other,
    and the difference in their areas is 201 square meters,
    prove that the area of the smaller field is 10,000 square meters. -/
theorem smaller_field_area (s : ℝ) (h1 : s > 0) :
  (s * 1.01)^2 - s^2 = 201 → s^2 = 10000 := by sorry

end NUMINAMATH_CALUDE_smaller_field_area_l2667_266709


namespace NUMINAMATH_CALUDE_age_problem_l2667_266732

/-- Given three people a, b, and c, where:
  * a is two years older than b
  * b is twice as old as c
  * The sum of their ages is 22
  Prove that b is 8 years old. -/
theorem age_problem (a b c : ℕ) 
  (h1 : a = b + 2) 
  (h2 : b = 2 * c) 
  (h3 : a + b + c = 22) : 
  b = 8 := by
  sorry

end NUMINAMATH_CALUDE_age_problem_l2667_266732


namespace NUMINAMATH_CALUDE_double_reflection_of_D_l2667_266751

-- Define a point in 2D space
structure Point where
  x : ℝ
  y : ℝ

-- Define reflection over y-axis
def reflectOverYAxis (p : Point) : Point :=
  { x := -p.x, y := p.y }

-- Define reflection over x-axis
def reflectOverXAxis (p : Point) : Point :=
  { x := p.x, y := -p.y }

-- Define the composition of reflections
def doubleReflection (p : Point) : Point :=
  reflectOverXAxis (reflectOverYAxis p)

-- Theorem statement
theorem double_reflection_of_D :
  let D : Point := { x := 3, y := 3 }
  doubleReflection D = { x := -3, y := -3 } := by
  sorry

end NUMINAMATH_CALUDE_double_reflection_of_D_l2667_266751


namespace NUMINAMATH_CALUDE_total_pears_is_105_l2667_266753

/-- The number of pears picked by Jason -/
def jason_pears : ℕ := 46

/-- The number of pears picked by Keith -/
def keith_pears : ℕ := 47

/-- The number of pears picked by Mike -/
def mike_pears : ℕ := 12

/-- The total number of pears picked -/
def total_pears : ℕ := jason_pears + keith_pears + mike_pears

/-- Theorem stating that the total number of pears picked is 105 -/
theorem total_pears_is_105 : total_pears = 105 := by
  sorry

end NUMINAMATH_CALUDE_total_pears_is_105_l2667_266753


namespace NUMINAMATH_CALUDE_midsegment_inequality_l2667_266743

/-- Midsegment theorem for triangles -/
theorem midsegment_inequality (a b c : ℝ) (h : a > 0 ∧ b > 0 ∧ c > 0) :
  let perimeter := a + b + c
  let midsegment_sum := (b + c) / 2 + (a + c) / 2 + (a + b) / 2
  midsegment_sum < perimeter ∧ midsegment_sum > 3 / 4 * perimeter :=
by sorry

end NUMINAMATH_CALUDE_midsegment_inequality_l2667_266743


namespace NUMINAMATH_CALUDE_fraction_comparison_l2667_266719

theorem fraction_comparison (x : ℝ) :
  x ∈ Set.Icc (-1 : ℝ) 3 →
  (5 * x + 2 > 8 - 3 * x) ↔ (x ∈ Set.Ioo (3/4 : ℝ) 3) :=
by sorry

end NUMINAMATH_CALUDE_fraction_comparison_l2667_266719


namespace NUMINAMATH_CALUDE_prob_at_least_three_out_of_five_is_half_l2667_266739

def probability_at_least_three_out_of_five : ℚ :=
  let n : ℕ := 5  -- total number of games
  let p : ℚ := 1/2  -- probability of winning a single game
  let winning_prob : ℕ → ℚ := λ k => Nat.choose n k * p^k * (1-p)^(n-k)
  (winning_prob 3) + (winning_prob 4) + (winning_prob 5)

theorem prob_at_least_three_out_of_five_is_half :
  probability_at_least_three_out_of_five = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_prob_at_least_three_out_of_five_is_half_l2667_266739


namespace NUMINAMATH_CALUDE_detergent_loads_theorem_l2667_266767

/-- Represents the number of loads of laundry that can be washed with one bottle of detergent -/
def loads_per_bottle (regular_price sale_price cost_per_load : ℚ) : ℚ :=
  (2 * sale_price) / (2 * cost_per_load)

/-- Theorem stating the number of loads that can be washed with one bottle of detergent -/
theorem detergent_loads_theorem (regular_price sale_price cost_per_load : ℚ) 
  (h1 : regular_price = 25)
  (h2 : sale_price = 20)
  (h3 : cost_per_load = 1/4) :
  loads_per_bottle regular_price sale_price cost_per_load = 80 := by
  sorry

#eval loads_per_bottle 25 20 (1/4)

end NUMINAMATH_CALUDE_detergent_loads_theorem_l2667_266767


namespace NUMINAMATH_CALUDE_work_completion_time_l2667_266782

theorem work_completion_time 
  (a_rate : ℝ) (b_rate : ℝ) (work_left : ℝ) (days_worked : ℝ) : 
  a_rate = 1 / 15 →
  b_rate = 1 / 20 →
  work_left = 0.41666666666666663 →
  (1 - work_left) = (a_rate + b_rate) * days_worked →
  days_worked = 5 := by
sorry

end NUMINAMATH_CALUDE_work_completion_time_l2667_266782


namespace NUMINAMATH_CALUDE_f_zeros_f_min_max_on_interval_l2667_266744

def f (x : ℝ) : ℝ := x^2 + x - 2

theorem f_zeros (x : ℝ) : f x = 0 ↔ x = 1 ∨ x = -2 := by sorry

theorem f_min_max_on_interval :
  let a : ℝ := -1
  let b : ℝ := 1
  (∀ x ∈ Set.Icc a b, f x ≥ -9/4) ∧
  (∃ x ∈ Set.Icc a b, f x = -9/4) ∧
  (∀ x ∈ Set.Icc a b, f x ≤ 0) ∧
  (∃ x ∈ Set.Icc a b, f x = 0) := by sorry

end NUMINAMATH_CALUDE_f_zeros_f_min_max_on_interval_l2667_266744


namespace NUMINAMATH_CALUDE_third_term_is_eight_thirds_l2667_266724

/-- The sequence defined by a_n = n - 1/n -/
def a (n : ℕ) : ℚ := n - 1 / n

/-- Theorem: The third term of the sequence a_n is 8/3 -/
theorem third_term_is_eight_thirds : a 3 = 8 / 3 := by
  sorry

end NUMINAMATH_CALUDE_third_term_is_eight_thirds_l2667_266724


namespace NUMINAMATH_CALUDE_angle_on_line_l2667_266716

theorem angle_on_line (α : Real) : 
  (0 ≤ α) ∧ (α < π) ∧ 
  (∃ (x y : Real), x + 2 * y = 0 ∧ 
    x = Real.cos α ∧ y = Real.sin α) →
  Real.sin (π / 2 - 2 * α) = 3 / 5 := by
sorry

end NUMINAMATH_CALUDE_angle_on_line_l2667_266716


namespace NUMINAMATH_CALUDE_notebook_distribution_l2667_266727

theorem notebook_distribution (total_notebooks : ℕ) (half_students_notebooks : ℕ) :
  total_notebooks = 512 →
  half_students_notebooks = 16 →
  ∃ (num_students : ℕ) (fraction : ℚ),
    num_students > 0 ∧
    fraction > 0 ∧
    fraction < 1 ∧
    (num_students / 2 : ℚ) * half_students_notebooks = total_notebooks ∧
    (num_students : ℚ) * (fraction * num_students) = total_notebooks ∧
    fraction = 1 / 8 :=
by sorry

end NUMINAMATH_CALUDE_notebook_distribution_l2667_266727


namespace NUMINAMATH_CALUDE_jan_paid_288_dollars_l2667_266785

def roses_per_dozen : ℕ := 12
def dozen_bought : ℕ := 5
def cost_per_rose : ℚ := 6
def discount_percentage : ℚ := 80

def total_roses : ℕ := dozen_bought * roses_per_dozen

def full_price : ℚ := (total_roses : ℚ) * cost_per_rose

def discounted_price : ℚ := full_price * (discount_percentage / 100)

theorem jan_paid_288_dollars : discounted_price = 288 := by
  sorry

end NUMINAMATH_CALUDE_jan_paid_288_dollars_l2667_266785


namespace NUMINAMATH_CALUDE_line_equation_forms_l2667_266775

theorem line_equation_forms (A B C : ℝ) :
  ∃ (φ p : ℝ), ∀ (x y : ℝ),
    A * x + B * y + C = 0 ↔ x * Real.cos φ + y * Real.sin φ = p :=
by sorry

end NUMINAMATH_CALUDE_line_equation_forms_l2667_266775


namespace NUMINAMATH_CALUDE_video_game_players_l2667_266721

/-- The number of players who joined a video game -/
def players_joined : ℕ := 5

theorem video_game_players :
  let initial_players : ℕ := 4
  let lives_per_player : ℕ := 3
  let total_lives : ℕ := 27
  players_joined = (total_lives - initial_players * lives_per_player) / lives_per_player :=
by
  sorry

#check video_game_players

end NUMINAMATH_CALUDE_video_game_players_l2667_266721


namespace NUMINAMATH_CALUDE_specific_tetrahedron_volume_l2667_266706

/-- Regular tetrahedron with given midpoint distances -/
structure RegularTetrahedron where
  midpoint_to_face : ℝ
  midpoint_to_edge : ℝ

/-- Volume of a regular tetrahedron -/
def volume (t : RegularTetrahedron) : ℝ := sorry

/-- Theorem stating the volume of the specific regular tetrahedron -/
theorem specific_tetrahedron_volume :
  ∃ (t : RegularTetrahedron),
    t.midpoint_to_face = 2 ∧
    t.midpoint_to_edge = Real.sqrt 10 ∧
    volume t = 80 * Real.sqrt 15 := by sorry

end NUMINAMATH_CALUDE_specific_tetrahedron_volume_l2667_266706


namespace NUMINAMATH_CALUDE_equilateral_triangle_to_three_layered_quadrilateral_l2667_266770

/-- Represents a polygon with a specified number of sides -/
structure Polygon where
  sides : ℕ
  deriving Repr

/-- Represents a folded shape -/
structure FoldedShape where
  shape : Polygon
  layers : ℕ
  deriving Repr

/-- Represents an equilateral triangle -/
def EquilateralTriangle : Polygon :=
  { sides := 3 }

/-- Represents a quadrilateral -/
def Quadrilateral : Polygon :=
  { sides := 4 }

/-- Folding operation that transforms one shape into another -/
def fold (start : Polygon) (result : FoldedShape) : Prop :=
  sorry

/-- Theorem stating that an equilateral triangle can be folded into a three-layered quadrilateral -/
theorem equilateral_triangle_to_three_layered_quadrilateral :
  ∃ (result : FoldedShape), 
    result.shape = Quadrilateral ∧ 
    result.layers = 3 ∧ 
    fold EquilateralTriangle result :=
by sorry

end NUMINAMATH_CALUDE_equilateral_triangle_to_three_layered_quadrilateral_l2667_266770


namespace NUMINAMATH_CALUDE_profit_share_difference_l2667_266756

/-- Given investments and B's profit share, calculate the difference between A's and C's profit shares -/
theorem profit_share_difference (a b c b_profit : ℕ) 
  (h1 : a = 8000) 
  (h2 : b = 10000) 
  (h3 : c = 12000) 
  (h4 : b_profit = 1700) : 
  (c * b_profit / b) - (a * b_profit / b) = 680 := by
  sorry

end NUMINAMATH_CALUDE_profit_share_difference_l2667_266756


namespace NUMINAMATH_CALUDE_parallel_vectors_m_value_l2667_266762

/-- Two vectors are parallel if their corresponding components are proportional -/
def parallel (a b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, k ≠ 0 ∧ a.1 * b.2 = k * a.2 * b.1

theorem parallel_vectors_m_value :
  ∀ m : ℝ,
  let a : ℝ × ℝ := (2, 1)
  let b : ℝ × ℝ := (m, 2)
  parallel a b → m = 4 := by
sorry

end NUMINAMATH_CALUDE_parallel_vectors_m_value_l2667_266762


namespace NUMINAMATH_CALUDE_symmetric_complex_product_l2667_266776

theorem symmetric_complex_product :
  ∀ (z₁ z₂ : ℂ),
  (Complex.re z₁ = -Complex.re z₂) →
  (Complex.im z₁ = Complex.im z₂) →
  (z₁ = 3 + Complex.I) →
  z₁ * z₂ = -10 := by
sorry

end NUMINAMATH_CALUDE_symmetric_complex_product_l2667_266776


namespace NUMINAMATH_CALUDE_prime_pair_sum_both_prime_prime_pair_product_l2667_266774

/-- Two prime numbers that sum to 101 -/
def prime_pair : (ℕ × ℕ) := sorry

/-- The sum of the prime pair is 101 -/
theorem prime_pair_sum : prime_pair.1 + prime_pair.2 = 101 := sorry

/-- Both numbers in the pair are prime -/
theorem both_prime : 
  Nat.Prime prime_pair.1 ∧ Nat.Prime prime_pair.2 := sorry

/-- The product of the prime pair is 194 -/
theorem prime_pair_product : 
  prime_pair.1 * prime_pair.2 = 194 := sorry

end NUMINAMATH_CALUDE_prime_pair_sum_both_prime_prime_pair_product_l2667_266774


namespace NUMINAMATH_CALUDE_x_value_l2667_266769

theorem x_value : ∃ x : ℚ, (2 / 5 * x) - (1 / 3 * x) = 110 ∧ x = 1650 := by
  sorry

end NUMINAMATH_CALUDE_x_value_l2667_266769


namespace NUMINAMATH_CALUDE_circle_theorem_sphere_theorem_l2667_266781

-- Define a circle and a sphere
def Circle : Type := Unit
def Sphere : Type := Unit

-- Define a point on a circle and a sphere
def PointOnCircle : Type := Unit
def PointOnSphere : Type := Unit

-- Define a semicircle and a hemisphere
def Semicircle : Type := Unit
def Hemisphere : Type := Unit

-- Define a function to check if a point is in a semicircle or hemisphere
def isIn : PointOnCircle → Semicircle → Prop := sorry
def isInHemisphere : PointOnSphere → Hemisphere → Prop := sorry

-- Theorem for the circle problem
theorem circle_theorem (c : Circle) (p1 p2 p3 p4 : PointOnCircle) :
  ∃ (s : Semicircle), (isIn p1 s ∧ isIn p2 s ∧ isIn p3 s) ∨
                      (isIn p1 s ∧ isIn p2 s ∧ isIn p4 s) ∨
                      (isIn p1 s ∧ isIn p3 s ∧ isIn p4 s) ∨
                      (isIn p2 s ∧ isIn p3 s ∧ isIn p4 s) :=
sorry

-- Theorem for the sphere problem
theorem sphere_theorem (s : Sphere) (p1 p2 p3 p4 p5 : PointOnSphere) :
  ∃ (h : Hemisphere), (isInHemisphere p1 h ∧ isInHemisphere p2 h ∧ isInHemisphere p3 h ∧ isInHemisphere p4 h) ∨
                      (isInHemisphere p1 h ∧ isInHemisphere p2 h ∧ isInHemisphere p3 h ∧ isInHemisphere p5 h) ∨
                      (isInHemisphere p1 h ∧ isInHemisphere p2 h ∧ isInHemisphere p4 h ∧ isInHemisphere p5 h) ∨
                      (isInHemisphere p1 h ∧ isInHemisphere p3 h ∧ isInHemisphere p4 h ∧ isInHemisphere p5 h) ∨
                      (isInHemisphere p2 h ∧ isInHemisphere p3 h ∧ isInHemisphere p4 h ∧ isInHemisphere p5 h) :=
sorry

end NUMINAMATH_CALUDE_circle_theorem_sphere_theorem_l2667_266781


namespace NUMINAMATH_CALUDE_geometric_sequence_tan_property_l2667_266793

/-- Given a geometric sequence {a_n} where a₂a₆ + 2a₄² = π, prove that tan(a₃a₅) = √3 -/
theorem geometric_sequence_tan_property (a : ℕ → ℝ) 
  (h_geometric : ∀ n : ℕ, a (n + 2) / a (n + 1) = a (n + 1) / a n)
  (h_condition : a 2 * a 6 + 2 * (a 4)^2 = Real.pi) :
  Real.tan (a 3 * a 5) = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_tan_property_l2667_266793


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l2667_266735

theorem sqrt_equation_solution (x : ℝ) :
  Real.sqrt (x + 8) = 10 → x = 92 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l2667_266735


namespace NUMINAMATH_CALUDE_cricket_runs_l2667_266754

theorem cricket_runs (a b c : ℕ) : 
  a + b + c = 95 →
  3 * a = b →
  5 * b = c →
  c = 75 := by
sorry

end NUMINAMATH_CALUDE_cricket_runs_l2667_266754


namespace NUMINAMATH_CALUDE_combined_salaries_l2667_266750

/-- The combined salaries of four employees given the salary of the fifth and the average of all five -/
theorem combined_salaries (salary_C average_salary : ℕ) 
  (hC : salary_C = 14000)
  (havg : average_salary = 8600) :
  salary_C + 4 * average_salary - 5 * average_salary = 29000 := by
  sorry

end NUMINAMATH_CALUDE_combined_salaries_l2667_266750


namespace NUMINAMATH_CALUDE_function_f_is_identity_l2667_266765

/-- A function satisfying the given conditions -/
def FunctionF (f : ℝ → ℝ) : Prop :=
  f 1 = 1 ∧ ∀ x y : ℝ, f (x * y + f x) = x * f y + f x

/-- Theorem stating that the only function satisfying the conditions is the identity function -/
theorem function_f_is_identity (f : ℝ → ℝ) (hf : FunctionF f) : 
  ∀ x : ℝ, f x = x := by
  sorry

end NUMINAMATH_CALUDE_function_f_is_identity_l2667_266765


namespace NUMINAMATH_CALUDE_toms_profit_l2667_266799

/-- Calculate Tom's profit from the world's largest dough ball event -/
theorem toms_profit (flour_needed : ℕ) (flour_bag_size : ℕ) (flour_bag_cost : ℕ)
                    (salt_needed : ℕ) (salt_cost_per_pound : ℚ)
                    (promotion_cost : ℕ) (ticket_price : ℕ) (tickets_sold : ℕ) :
  flour_needed = 500 →
  flour_bag_size = 50 →
  flour_bag_cost = 20 →
  salt_needed = 10 →
  salt_cost_per_pound = 1/5 →
  promotion_cost = 1000 →
  ticket_price = 20 →
  tickets_sold = 500 →
  (tickets_sold * ticket_price : ℤ) - 
  (((flour_needed / flour_bag_size) * flour_bag_cost : ℕ) + 
   (salt_needed * salt_cost_per_pound).num + 
   promotion_cost : ℤ) = 8798 :=
by sorry

end NUMINAMATH_CALUDE_toms_profit_l2667_266799


namespace NUMINAMATH_CALUDE_sandy_fish_count_l2667_266741

def final_fish_count (initial : ℕ) (bought : ℕ) (given_away : ℕ) (babies : ℕ) : ℕ :=
  initial + bought - given_away + babies

theorem sandy_fish_count :
  final_fish_count 26 6 10 15 = 37 := by
  sorry

end NUMINAMATH_CALUDE_sandy_fish_count_l2667_266741


namespace NUMINAMATH_CALUDE_distribute_5_3_l2667_266703

/-- The number of ways to distribute indistinguishable objects into distinguishable containers -/
def distribute (n : ℕ) (k : ℕ) : ℕ := Nat.choose (n + k - 1) (k - 1)

/-- Theorem: There are 21 ways to distribute 5 indistinguishable balls into 3 distinguishable boxes -/
theorem distribute_5_3 : distribute 5 3 = 21 := by
  sorry

end NUMINAMATH_CALUDE_distribute_5_3_l2667_266703


namespace NUMINAMATH_CALUDE_linear_equation_solution_l2667_266755

theorem linear_equation_solution (x y m : ℝ) : 
  x = 2 ∧ y = -3 ∧ 5 * x + m * y + 2 = 0 → m = 4 := by
  sorry

end NUMINAMATH_CALUDE_linear_equation_solution_l2667_266755


namespace NUMINAMATH_CALUDE_two_digit_multiple_of_35_l2667_266768

theorem two_digit_multiple_of_35 (n : ℕ) (h1 : 10 ≤ n ∧ n < 100) (h2 : n % 35 = 0) : 
  n % 10 = 5 :=
sorry

end NUMINAMATH_CALUDE_two_digit_multiple_of_35_l2667_266768


namespace NUMINAMATH_CALUDE_cubic_sum_minus_product_l2667_266720

theorem cubic_sum_minus_product (a b c : ℝ) 
  (sum_eq : a + b + c = 12) 
  (sum_prod_eq : a * b + a * c + b * c = 30) : 
  a^3 + b^3 + c^3 - 3*a*b*c = 648 := by
sorry

end NUMINAMATH_CALUDE_cubic_sum_minus_product_l2667_266720


namespace NUMINAMATH_CALUDE_max_sphere_radius_squared_max_sphere_radius_squared_achievable_l2667_266749

/-- Represents a right circular cone -/
structure Cone where
  baseRadius : ℝ
  height : ℝ

/-- Represents the configuration of three cones and a sphere -/
structure ConeConfiguration where
  cone : Cone
  axisIntersectionDistance : ℝ
  sphereRadius : ℝ

/-- Checks if the configuration is valid -/
def isValidConfiguration (config : ConeConfiguration) : Prop :=
  config.cone.baseRadius = 4 ∧
  config.cone.height = 10 ∧
  config.axisIntersectionDistance = 5

/-- Theorem stating the maximum possible value of r^2 -/
theorem max_sphere_radius_squared (config : ConeConfiguration) 
  (h : isValidConfiguration config) : 
  config.sphereRadius ^ 2 ≤ 100 / 29 := by
  sorry

/-- Theorem stating that the maximum value is achievable -/
theorem max_sphere_radius_squared_achievable : 
  ∃ (config : ConeConfiguration), isValidConfiguration config ∧ config.sphereRadius ^ 2 = 100 / 29 := by
  sorry

end NUMINAMATH_CALUDE_max_sphere_radius_squared_max_sphere_radius_squared_achievable_l2667_266749


namespace NUMINAMATH_CALUDE_performance_selection_ways_l2667_266729

/-- The number of students who can sing -/
def num_singers : ℕ := 3

/-- The number of students who can dance -/
def num_dancers : ℕ := 2

/-- The number of students who can both sing and dance -/
def num_both : ℕ := 1

/-- The total number of students -/
def total_students : ℕ := 6

/-- The number of students to be selected for singing -/
def singers_to_select : ℕ := 2

/-- The number of students to be selected for dancing -/
def dancers_to_select : ℕ := 1

/-- The number of ways to select the required students for the performance -/
def num_ways : ℕ := Nat.choose (num_singers + num_both) singers_to_select * num_dancers - 1

theorem performance_selection_ways :
  num_ways = Nat.choose (num_singers + num_both) singers_to_select * num_dancers - 1 :=
by sorry

end NUMINAMATH_CALUDE_performance_selection_ways_l2667_266729


namespace NUMINAMATH_CALUDE_trent_kept_tadpoles_l2667_266773

/-- The number of tadpoles Trent initially caught -/
def initial_tadpoles : ℕ := 180

/-- The percentage of tadpoles Trent let go -/
def percent_released : ℚ := 75 / 100

/-- The number of tadpoles Trent kept -/
def tadpoles_kept : ℕ := 45

/-- Theorem stating that the number of tadpoles Trent kept is equal to 45 -/
theorem trent_kept_tadpoles : 
  (initial_tadpoles : ℚ) * (1 - percent_released) = tadpoles_kept := by sorry

end NUMINAMATH_CALUDE_trent_kept_tadpoles_l2667_266773


namespace NUMINAMATH_CALUDE_final_amount_after_bets_l2667_266722

/-- Calculates the final amount after a series of bets -/
def finalAmount (initialAmount : ℚ) (numBets numWins numLosses : ℕ) : ℚ :=
  initialAmount * (3/2)^numWins * (1/2)^numLosses

/-- Theorem stating the final amount after 7 bets with 4 wins and 3 losses -/
theorem final_amount_after_bets :
  finalAmount 128 7 4 3 = 81 := by
  sorry

end NUMINAMATH_CALUDE_final_amount_after_bets_l2667_266722


namespace NUMINAMATH_CALUDE_fixed_fee_is_5_20_l2667_266742

/-- Represents a music streaming service with a fixed monthly fee and a per-song fee -/
structure StreamingService where
  fixedFee : ℝ
  perSongFee : ℝ

/-- Calculates the total bill for a given number of songs -/
def bill (s : StreamingService) (songs : ℕ) : ℝ :=
  s.fixedFee + s.perSongFee * songs

theorem fixed_fee_is_5_20 (s : StreamingService) :
  bill s 20 = 15.20 ∧ bill s 40 = 25.20 → s.fixedFee = 5.20 := by
  sorry

#check fixed_fee_is_5_20

end NUMINAMATH_CALUDE_fixed_fee_is_5_20_l2667_266742


namespace NUMINAMATH_CALUDE_value_of_y_l2667_266700

theorem value_of_y (x y : ℚ) : 
  x = 51 → x^3*y - 2*x^2*y + x*y = 127500 → y = 1/51 := by sorry

end NUMINAMATH_CALUDE_value_of_y_l2667_266700


namespace NUMINAMATH_CALUDE_arithmetic_calculation_l2667_266736

theorem arithmetic_calculation : 4 * 6 * 8 + 24 / 4 = 198 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculation_l2667_266736


namespace NUMINAMATH_CALUDE_perpendicular_line_through_point_perpendicular_line_equation_l2667_266723

-- Define a point in 2D space
structure Point2D where
  x : ℝ
  y : ℝ

-- Define a line in 2D space (ax + by + c = 0)
structure Line2D where
  a : ℝ
  b : ℝ
  c : ℝ

def perpendicular (l1 l2 : Line2D) : Prop :=
  l1.a * l2.a + l1.b * l2.b = 0

def passes_through (l : Line2D) (p : Point2D) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

theorem perpendicular_line_through_point 
  (P : Point2D)
  (given_line : Line2D)
  (result_line : Line2D) : Prop :=
  P.x = -1 ∧ 
  P.y = 3 ∧ 
  given_line.a = 1 ∧ 
  given_line.b = -2 ∧ 
  given_line.c = 3 ∧
  result_line.a = 2 ∧ 
  result_line.b = 1 ∧ 
  result_line.c = -1 ∧
  perpendicular given_line result_line ∧
  passes_through result_line P ∧
  ∀ (other_line : Line2D), 
    perpendicular given_line other_line ∧ 
    passes_through other_line P → 
    other_line = result_line

theorem perpendicular_line_equation : 
  perpendicular_line_through_point 
    (Point2D.mk (-1) 3) 
    (Line2D.mk 1 (-2) 3) 
    (Line2D.mk 2 1 (-1)) := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_line_through_point_perpendicular_line_equation_l2667_266723


namespace NUMINAMATH_CALUDE_inscribed_circle_diameter_right_triangle_l2667_266733

/-- In a right-angled triangle with legs a and b, hypotenuse c, and inscribed circle of radius r,
    the diameter of the inscribed circle is a + b - c. -/
theorem inscribed_circle_diameter_right_triangle 
  (a b c r : ℝ) 
  (h_right_triangle : a^2 + b^2 = c^2) 
  (h_positive : a > 0 ∧ b > 0 ∧ c > 0 ∧ r > 0) 
  (h_inscribed : r = (a + b - c) / 2) : 
  2 * r = a + b - c := by
  sorry

end NUMINAMATH_CALUDE_inscribed_circle_diameter_right_triangle_l2667_266733


namespace NUMINAMATH_CALUDE_equal_share_money_l2667_266795

theorem equal_share_money (total_amount : ℚ) (num_people : ℕ) 
  (h1 : total_amount = 3.75)
  (h2 : num_people = 3) : 
  total_amount / num_people = 1.25 := by
  sorry

end NUMINAMATH_CALUDE_equal_share_money_l2667_266795


namespace NUMINAMATH_CALUDE_prob_at_most_one_mistake_value_l2667_266726

/-- Probability of correct answer for the first question -/
def p1 : ℚ := 3/4

/-- Probability of correct answer for the second question -/
def p2 : ℚ := 1/2

/-- Probability of correct answer for the third question -/
def p3 : ℚ := 1/6

/-- Probability of at most one mistake in the first three questions -/
def prob_at_most_one_mistake : ℚ := 
  p1 * p2 * p3 + 
  (1 - p1) * p2 * p3 + 
  p1 * (1 - p2) * p3 + 
  p1 * p2 * (1 - p3)

theorem prob_at_most_one_mistake_value : 
  prob_at_most_one_mistake = 11/24 := by sorry

end NUMINAMATH_CALUDE_prob_at_most_one_mistake_value_l2667_266726


namespace NUMINAMATH_CALUDE_S_is_circle_l2667_266713

-- Define the set of complex numbers satisfying the condition
def S : Set ℂ := {z : ℂ | Complex.abs (z - Complex.I) = Complex.abs (3 + 4 * Complex.I)}

-- Theorem stating that S is a circle
theorem S_is_circle : ∃ (c : ℂ) (r : ℝ), S = {z : ℂ | Complex.abs (z - c) = r} :=
sorry

end NUMINAMATH_CALUDE_S_is_circle_l2667_266713


namespace NUMINAMATH_CALUDE_ratio_expression_value_l2667_266783

theorem ratio_expression_value (A B C : ℚ) (h : A / B = 3 / 2 ∧ B / C = 2 / 5) :
  (4 * A + 3 * B) / (5 * C - 2 * A) = 18 / 19 := by
  sorry

end NUMINAMATH_CALUDE_ratio_expression_value_l2667_266783


namespace NUMINAMATH_CALUDE_similar_triangle_scaling_l2667_266778

theorem similar_triangle_scaling (base1 height1 base2 : ℝ) (height2 : ℝ) : 
  base1 = 12 → height1 = 6 → base2 = 8 → 
  (base1 / height1 = base2 / height2) → 
  height2 = 4 := by sorry

end NUMINAMATH_CALUDE_similar_triangle_scaling_l2667_266778


namespace NUMINAMATH_CALUDE_ali_flower_sales_l2667_266740

def flower_problem (monday_sales : ℕ) (friday_multiplier : ℕ) (total_sales : ℕ) : Prop :=
  let friday_sales := friday_multiplier * monday_sales
  let tuesday_sales := total_sales - monday_sales - friday_sales
  tuesday_sales = 8

theorem ali_flower_sales : flower_problem 4 2 20 := by
  sorry

end NUMINAMATH_CALUDE_ali_flower_sales_l2667_266740
