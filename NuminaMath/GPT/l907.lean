import Mathlib

namespace bob_monthly_hours_l907_90776

noncomputable def total_hours_in_month : ℝ :=
  let daily_hours := 10
  let weekly_days := 5
  let weeks_in_month := 4.33
  daily_hours * weekly_days * weeks_in_month

theorem bob_monthly_hours :
  total_hours_in_month = 216.5 :=
by
  sorry

end bob_monthly_hours_l907_90776


namespace quad_condition_l907_90723

noncomputable def quadratic (a : ℝ) (x : ℝ) : ℝ := x^2 + a * x - 4 * a

theorem quad_condition (a : ℝ) : (-16 ≤ a ∧ a ≤ 0) → (∀ x : ℝ, quadratic a x > 0) ↔ (¬ ∃ x : ℝ, quadratic a x ≤ 0) := by
  sorry

end quad_condition_l907_90723


namespace exists_unique_c_l907_90793

theorem exists_unique_c (a : ℝ) (h₁ : 1 < a) :
  (∃ (c : ℝ), ∀ (x : ℝ), x ∈ Set.Icc a (2 * a) → ∃ (y : ℝ), y ∈ Set.Icc a (a ^ 2) ∧ (Real.log x / Real.log a + Real.log y / Real.log a = c)) ↔ a = 2 :=
by
  sorry

end exists_unique_c_l907_90793


namespace profit_achieved_at_50_yuan_l907_90731

theorem profit_achieved_at_50_yuan :
  ∀ (x : ℝ), (30 ≤ x ∧ x ≤ 54) → 
  ((x - 30) * (80 - 2 * (x - 40)) = 1200) →
  x = 50 :=
by
  intros x h_range h_profit
  sorry

end profit_achieved_at_50_yuan_l907_90731


namespace polynomial_roots_arithmetic_progression_complex_root_l907_90707

theorem polynomial_roots_arithmetic_progression_complex_root :
  ∃ a : ℝ, (∀ (r d : ℂ), (r - d) + r + (r + d) = 9 → (r - d) * r + (r - d) * (r + d) + r * (r + d) = 30 → d^2 = -3 → 
  (r - d) * r * (r + d) = -a) → a = -12 :=
by sorry

end polynomial_roots_arithmetic_progression_complex_root_l907_90707


namespace fifth_team_points_l907_90770

theorem fifth_team_points (points_A points_B points_C points_D points_E : ℕ) 
(hA : points_A = 1) 
(hB : points_B = 2) 
(hC : points_C = 5) 
(hD : points_D = 7) 
(h_sum : points_A + points_B + points_C + points_D + points_E = 20) : 
points_E = 5 := 
sorry

end fifth_team_points_l907_90770


namespace abs_m_minus_n_eq_five_l907_90711

theorem abs_m_minus_n_eq_five (m n : ℝ) (h₁ : m * n = 6) (h₂ : m + n = 7) : |m - n| = 5 :=
sorry

end abs_m_minus_n_eq_five_l907_90711


namespace cars_on_happy_street_l907_90740

theorem cars_on_happy_street :
  let cars_tuesday := 25
  let cars_monday := cars_tuesday - cars_tuesday * 20 / 100
  let cars_wednesday := cars_monday + 2
  let cars_thursday : ℕ := 10
  let cars_friday : ℕ := 10
  let cars_saturday : ℕ := 5
  let cars_sunday : ℕ := 5
  let total_cars := cars_monday + cars_tuesday + cars_wednesday + cars_thursday + cars_friday + cars_saturday + cars_sunday
  total_cars = 97 :=
by
  sorry

end cars_on_happy_street_l907_90740


namespace upper_bound_of_n_l907_90787

theorem upper_bound_of_n (m n : ℕ) (h_m : m ≥ 2)
  (h_div : ∀ a : ℕ, gcd a n = 1 → n ∣ a^m - 1) : 
  n ≤ 4 * m * (2^m - 1) := 
sorry

end upper_bound_of_n_l907_90787


namespace common_difference_of_arithmetic_sequence_l907_90792

variable {α : Type*} [LinearOrderedField α]

def arithmetic_sequence_sum (n : ℕ) (an : ℕ → α) : α :=
  (n : α) * an 1 + (n * (n - 1) / 2 * (an 2 - an 1))

theorem common_difference_of_arithmetic_sequence (S : ℕ → ℕ) (d : ℕ) (a1 a2 : ℕ)
  (h1 : ∀ n, S n = 4 * n ^ 2 - n)
  (h2 : a1 = S 1)
  (h3 : a2 = S 2 - S 1) :
  d = a2 - a1 → d = 8 := by
  sorry

end common_difference_of_arithmetic_sequence_l907_90792


namespace percentage_exceeds_l907_90751

theorem percentage_exceeds (N P : ℕ) (h1 : N = 50) (h2 : N = (P / 100) * N + 42) : P = 16 :=
sorry

end percentage_exceeds_l907_90751


namespace domain_of_log_sqrt_l907_90720

theorem domain_of_log_sqrt (x : ℝ) : (-1 < x ∧ x ≤ 3) ↔ (0 < x + 1 ∧ 3 - x ≥ 0) :=
by
  sorry

end domain_of_log_sqrt_l907_90720


namespace range_of_a_l907_90732

theorem range_of_a (a : ℝ) : (∃ x : ℝ, 0 < x ∧ x^2 + 2 * a * x + 2 * a + 3 < 0) ↔ a < -1 :=
sorry

end range_of_a_l907_90732


namespace y_intercept_exists_l907_90759

def line_eq (x y : ℝ) : Prop := x + 2 * y + 2 = 0

theorem y_intercept_exists : ∃ y : ℝ, line_eq 0 y ∧ y = -1 :=
by
  sorry

end y_intercept_exists_l907_90759


namespace B_squared_ge_AC_l907_90734

variable {a b c A B C : ℝ}

theorem B_squared_ge_AC
  (h1 : b^2 < a * c)
  (h2 : a * C - 2 * b * B + c * A = 0) :
  B^2 ≥ A * C := 
sorry

end B_squared_ge_AC_l907_90734


namespace add_pure_alcohol_to_achieve_percentage_l907_90777

-- Define the initial conditions
def initial_solution_volume : ℝ := 6
def initial_alcohol_percentage : ℝ := 0.30
def initial_pure_alcohol : ℝ := initial_solution_volume * initial_alcohol_percentage

-- Define the final conditions
def final_alcohol_percentage : ℝ := 0.50

-- Define the unknown to prove
def amount_of_alcohol_to_add : ℝ := 2.4

-- The target statement to prove
theorem add_pure_alcohol_to_achieve_percentage :
  (initial_pure_alcohol + amount_of_alcohol_to_add) / (initial_solution_volume + amount_of_alcohol_to_add) = final_alcohol_percentage :=
by
  sorry

end add_pure_alcohol_to_achieve_percentage_l907_90777


namespace percent_games_lost_l907_90709

theorem percent_games_lost
  (w l t : ℕ)
  (h_ratio : 7 * l = 3 * w)
  (h_tied : t = 5) :
  (l : ℝ) / (w + l + t) * 100 = 20 :=
by
  sorry

end percent_games_lost_l907_90709


namespace garden_roller_area_l907_90761

theorem garden_roller_area (length : ℝ) (area_5rev : ℝ) (d1 d2 : ℝ) (π : ℝ) :
  length = 4 ∧ area_5rev = 88 ∧ π = 22 / 7 ∧ d2 = 1.4 →
  let circumference := π * d2
  let area_rev := circumference * length
  let new_area_5rev := 5 * area_rev
  new_area_5rev = 88 :=
by
  sorry

end garden_roller_area_l907_90761


namespace tanner_savings_in_november_l907_90789

theorem tanner_savings_in_november(savings_sep : ℕ) (savings_oct : ℕ) 
(spending : ℕ) (leftover : ℕ) (N : ℕ) :
savings_sep = 17 →
savings_oct = 48 →
spending = 49 →
leftover = 41 →
((savings_sep + savings_oct + N - spending) = leftover) →
N = 25 :=
by
  intros h_sep h_oct h_spending h_leftover h_equation
  sorry

end tanner_savings_in_november_l907_90789


namespace incircle_tangent_distance_l907_90774

theorem incircle_tangent_distance (a b c : ℝ) (M : ℝ) (BM : ℝ) (x1 y1 z1 x2 y2 z2 : ℝ) 
  (h1 : BM = y1 + z1)
  (h2 : BM = y2 + z2)
  (h3 : x1 + y1 = x2 + y2)
  (h4 : x1 + z1 = c)
  (h5 : x2 + z2 = a) :
  |y1 - y2| = |(a - c) / 2| := by 
  sorry

end incircle_tangent_distance_l907_90774


namespace average_age_of_girls_l907_90757

theorem average_age_of_girls (total_students : ℕ) (boys_avg_age : ℝ) (school_avg_age : ℚ)
    (girls_count : ℕ) (total_age_school : ℝ) (boys_count : ℕ) 
    (total_age_boys : ℝ) (total_age_girls : ℝ): (total_students = 640) →
    (boys_avg_age = 12) →
    (school_avg_age = 47 / 4) →
    (girls_count = 160) →
    (total_students - girls_count = boys_count) →
    (boys_avg_age * boys_count = total_age_boys) →
    (school_avg_age * total_students = total_age_school) →
    (total_age_school - total_age_boys = total_age_girls) →
    total_age_girls / girls_count = 11 :=
by
  intros h_total_students h_boys_avg_age h_school_avg_age h_girls_count 
         h_boys_count h_total_age_boys h_total_age_school h_total_age_girls
  sorry

end average_age_of_girls_l907_90757


namespace order_of_x_y_z_l907_90716

variable (x : ℝ)
variable (y : ℝ)
variable (z : ℝ)

-- Conditions
axiom h1 : 0.9 < x
axiom h2 : x < 1.0
axiom h3 : y = x^x
axiom h4 : z = x^(x^x)

-- Theorem to be proved
theorem order_of_x_y_z (h1 : 0.9 < x) (h2 : x < 1.0) (h3 : y = x^x) (h4 : z = x^(x^x)) : x < z ∧ z < y :=
by
  sorry

end order_of_x_y_z_l907_90716


namespace point_on_x_axis_l907_90797

theorem point_on_x_axis (m : ℝ) (h : 3 * m + 1 = 0) : m = -1 / 3 :=
by 
  sorry

end point_on_x_axis_l907_90797


namespace leap_years_among_given_years_l907_90715

-- Definitions for conditions
def is_divisible (a b : Nat) : Prop := b ≠ 0 ∧ a % b = 0

def is_leap_year (y : Nat) : Prop :=
  is_divisible y 4 ∧ (¬ is_divisible y 100 ∨ is_divisible y 400)

-- Statement of the problem
theorem leap_years_among_given_years :
  is_leap_year 1996 ∧ is_leap_year 2036 ∧ (¬ is_leap_year 1700) ∧ (¬ is_leap_year 1998) :=
by
  -- Proof would go here
  sorry

end leap_years_among_given_years_l907_90715


namespace three_sum_xyz_l907_90768

theorem three_sum_xyz (x y z : ℝ) 
  (h1 : y + z = 18 - 4 * x) 
  (h2 : x + z = 22 - 4 * y) 
  (h3 : x + y = 15 - 4 * z) : 
  3 * x + 3 * y + 3 * z = 55 / 2 := 
  sorry

end three_sum_xyz_l907_90768


namespace simplify_polynomial_l907_90726

theorem simplify_polynomial :
  (3 * x^3 + 4 * x^2 + 8 * x - 5) - (2 * x^3 + x^2 + 6 * x - 7) = x^3 + 3 * x^2 + 2 * x + 2 := 
by
  sorry

end simplify_polynomial_l907_90726


namespace price_per_box_l907_90796

theorem price_per_box (total_apples : ℕ) (apples_per_box : ℕ) (total_revenue : ℕ) : 
  total_apples = 10000 → apples_per_box = 50 → total_revenue = 7000 → 
  total_revenue / (total_apples / apples_per_box) = 35 :=
by
  intros h1 h2 h3
  -- we can skip the actual proof with sorry. This indicates that the proof is not provided,
  -- but the statement is what needs to be proven.
  sorry

end price_per_box_l907_90796


namespace equation_of_latus_rectum_l907_90791

theorem equation_of_latus_rectum (p : ℝ) (h1 : p = 6) :
  (∀ x y : ℝ, y ^ 2 = -12 * x → x = 3) :=
sorry

end equation_of_latus_rectum_l907_90791


namespace bread_per_day_baguettes_per_day_croissants_per_day_l907_90721

-- Define the conditions
def loaves_per_hour : ℕ := 10
def hours_per_day : ℕ := 6
def baguettes_per_2hours : ℕ := 30
def croissants_per_75minutes : ℕ := 20

-- Conversion factors
def minutes_per_hour : ℕ := 60
def minutes_per_block : ℕ := 75
def blocks_per_75minutes : ℕ := 360 / 75

-- Proof statements
theorem bread_per_day :
  loaves_per_hour * hours_per_day = 60 := by sorry

theorem baguettes_per_day :
  (hours_per_day / 2) * baguettes_per_2hours = 90 := by sorry

theorem croissants_per_day :
  (blocks_per_75minutes * croissants_per_75minutes) = 80 := by sorry

end bread_per_day_baguettes_per_day_croissants_per_day_l907_90721


namespace proof_part1_proof_part2_l907_90779

noncomputable def f (x a : ℝ) : ℝ := x^3 - a * x^2 + 3 * x

def condition1 (a : ℝ) : Prop := ∀ x : ℝ, x ≥ 1 → 3 * x^2 - 2 * a * x + 3 ≥ 0

def condition2 (a : ℝ) : Prop := 3 * 3^2 - 2 * a * 3 + 3 = 0

theorem proof_part1 (a : ℝ) : condition1 a → a ≤ 3 := 
sorry

theorem proof_part2 (a : ℝ) (ha : a = 5) : 
  f 1 a = -1 ∧ f 3 a = -9 ∧ f 5 a = 15 :=
sorry

end proof_part1_proof_part2_l907_90779


namespace symmetric_curve_eq_l907_90703

theorem symmetric_curve_eq : 
  (∃ x' y', (x' - 3)^2 + 4*(y' - 5)^2 = 4 ∧ (x' - 6 = x' + x) ∧ (y' - 10 = y' + y)) ->
  (∃ x y, (x - 6) ^ 2 + 4 * (y - 10) ^ 2 = 4) :=
by
  sorry

end symmetric_curve_eq_l907_90703


namespace limit_one_minus_reciprocal_l907_90764

theorem limit_one_minus_reciprocal (h : Filter.Tendsto (fun (n : ℕ) => 1 / n) Filter.atTop (nhds 0)) :
  Filter.Tendsto (fun (n : ℕ) => 1 - 1 / n) Filter.atTop (nhds 1) :=
sorry

end limit_one_minus_reciprocal_l907_90764


namespace union_of_A_and_B_is_R_l907_90743

open Set Real

def A := {x : ℝ | log x > 0}
def B := {x : ℝ | x ≤ 1}

theorem union_of_A_and_B_is_R : A ∪ B = univ := by
  sorry

end union_of_A_and_B_is_R_l907_90743


namespace coefficient_of_x_eq_2_l907_90754

variable (a : ℝ)

theorem coefficient_of_x_eq_2 (h : (5 * (-2)) + (4 * a) = 2) : a = 3 :=
sorry

end coefficient_of_x_eq_2_l907_90754


namespace min_students_same_place_l907_90712

-- Define the context of the problem
def classSize := 45
def numberOfChoices := 6

-- The proof statement
theorem min_students_same_place : 
  ∃ (n : ℕ), 8 ≤ n ∧ n = Nat.ceil (classSize / numberOfChoices) :=
by
  sorry

end min_students_same_place_l907_90712


namespace difference_of_squares_evaluation_l907_90786

theorem difference_of_squares_evaluation :
  49^2 - 16^2 = 2145 :=
by sorry

end difference_of_squares_evaluation_l907_90786


namespace distance_to_post_office_l907_90735

theorem distance_to_post_office
  (D : ℝ)
  (travel_rate : ℝ) (walk_rate : ℝ)
  (total_time_hours : ℝ)
  (h1 : travel_rate = 25)
  (h2 : walk_rate = 4)
  (h3 : total_time_hours = 5 + 48 / 60) :
  D = 20 :=
by
  sorry

end distance_to_post_office_l907_90735


namespace geometric_sequences_l907_90767

variable (a_n b_n : ℕ → ℕ) -- Geometric sequences
variable (S_n T_n : ℕ → ℕ) -- Sums of first n terms
variable (h : ∀ n, S_n n / T_n n = (3^n + 1) / 4)

theorem geometric_sequences (n : ℕ) (h : ∀ n, S_n n / T_n n = (3^n + 1) / 4) : 
  (a_n 3) / (b_n 3) = 9 := 
sorry

end geometric_sequences_l907_90767


namespace unique_line_equal_intercepts_l907_90739

-- Definitions of the point and line
structure Point where
  x : ℝ
  y : ℝ

def passesThrough (L : ℝ → ℝ) (P : Point) : Prop :=
  L P.x = P.y

noncomputable def hasEqualIntercepts (L : ℝ → ℝ) : Prop :=
  ∃ a, L 0 = a ∧ L a = 0

-- The main theorem statement
theorem unique_line_equal_intercepts (L : ℝ → ℝ) (P : Point) (hP : P.x = 2 ∧ P.y = 1) (h_equal_intercepts : hasEqualIntercepts L) :
  ∃! (L : ℝ → ℝ), passesThrough L P ∧ hasEqualIntercepts L :=
sorry

end unique_line_equal_intercepts_l907_90739


namespace velocity_of_current_l907_90700

theorem velocity_of_current (v : ℝ) 
  (row_speed : ℝ) (distance : ℝ) (total_time : ℝ) 
  (h_row_speed : row_speed = 5)
  (h_distance : distance = 2.4)
  (h_total_time : total_time = 1)
  (h_equation : distance / (row_speed + v) + distance / (row_speed - v) = total_time) :
  v = 1 :=
sorry

end velocity_of_current_l907_90700


namespace units_digits_no_match_l907_90729

theorem units_digits_no_match : ∀ x : ℕ, 1 ≤ x ∧ x ≤ 100 → (x % 10 ≠ (101 - x) % 10) :=
by
  intro x hx
  sorry

end units_digits_no_match_l907_90729


namespace player1_coins_l907_90744

theorem player1_coins (coin_distribution : Fin 9 → ℕ) :
  let rotations := 11
  let player_4_coins := 90
  let player_8_coins := 35
  ∀ player : Fin 9, player = 0 → 
    let player_1_coins := coin_distribution player
    (coin_distribution 3 = player_4_coins) →
    (coin_distribution 7 = player_8_coins) →
    player_1_coins = 57 := 
sorry

end player1_coins_l907_90744


namespace recipe_sugar_amount_l907_90762

-- Definitions from A)
def cups_of_salt : ℕ := 9
def additional_cups_of_sugar (sugar salt : ℕ) : Prop := sugar = salt + 2

-- Statement to prove
theorem recipe_sugar_amount (salt : ℕ) (h : salt = cups_of_salt) : ∃ sugar : ℕ, additional_cups_of_sugar sugar salt ∧ sugar = 11 :=
by
  sorry

end recipe_sugar_amount_l907_90762


namespace sum_of_digits_133131_l907_90769

noncomputable def extract_digits_sum (n : Nat) : Nat :=
  let digits := n.digits 10
  digits.foldl (· + ·) 0

theorem sum_of_digits_133131 :
  let ABCDEF := 665655 / 5
  extract_digits_sum ABCDEF = 12 :=
by
  sorry

end sum_of_digits_133131_l907_90769


namespace carl_lawn_area_l907_90773

theorem carl_lawn_area :
  ∃ (width height : ℤ), 
    (width + 1) + (height + 1) - 4 = 24 ∧
    3 * width = height ∧
    3 * ((width + 1) * 3) * ((height + 1) * 3) = 243 :=
by
  sorry

end carl_lawn_area_l907_90773


namespace find_dividend_l907_90745

-- Define the given constants
def quotient : ℕ := 909899
def divisor : ℕ := 12

-- Define the dividend as the product of divisor and quotient
def dividend : ℕ := divisor * quotient

-- The theorem stating the equality we need to prove
theorem find_dividend : dividend = 10918788 := by
  sorry

end find_dividend_l907_90745


namespace average_length_of_strings_l907_90785

-- Define lengths of the three strings
def length1 := 4  -- length of the first string in inches
def length2 := 5  -- length of the second string in inches
def length3 := 7  -- length of the third string in inches

-- Define the total length and number of strings
def total_length := length1 + length2 + length3
def num_strings := 3

-- Define the average length calculation
def average_length := total_length / num_strings

-- The proof statement
theorem average_length_of_strings : average_length = 16 / 3 := 
by 
  sorry

end average_length_of_strings_l907_90785


namespace decimal_equivalent_of_quarter_cubed_l907_90738

theorem decimal_equivalent_of_quarter_cubed :
    (1 / 4 : ℝ) ^ 3 = 0.015625 := 
by
    sorry

end decimal_equivalent_of_quarter_cubed_l907_90738


namespace product_remainder_31_l907_90749

theorem product_remainder_31 (m n : ℕ) (h₁ : m % 31 = 7) (h₂ : n % 31 = 12) : (m * n) % 31 = 22 :=
by
  sorry

end product_remainder_31_l907_90749


namespace inequality_holds_l907_90725

theorem inequality_holds (a : ℝ) : 
  (∀ x : ℝ, 1 ≤ x ∧ x ≤ 12 → x^2 + 25 + |x^3 - 5 * x^2| ≥ a * x) ↔ a ≤ 2.5 := 
by
  sorry

end inequality_holds_l907_90725


namespace cindy_correct_answer_l907_90736

theorem cindy_correct_answer (x : ℤ) (h : (x - 7) / 5 = 37) : (x - 5) / 7 = 26 :=
sorry

end cindy_correct_answer_l907_90736


namespace students_in_each_class_l907_90799

-- Define the conditions
def sheets_per_student : ℕ := 5
def total_sheets : ℕ := 400
def number_of_classes : ℕ := 4

-- Define the main proof theorem
theorem students_in_each_class : (total_sheets / sheets_per_student) / number_of_classes = 20 := by
  sorry -- Proof goes here

end students_in_each_class_l907_90799


namespace revenue_growth_20_percent_l907_90724

noncomputable def revenue_increase (R2000 R2003 R2005 : ℝ) : ℝ :=
  ((R2005 - R2003) / R2003) * 100

theorem revenue_growth_20_percent (R2000 : ℝ) (h1 : R2003 = 1.5 * R2000) (h2 : R2005 = 1.8 * R2000) :
  revenue_increase R2000 R2003 R2005 = 20 :=
by
  sorry

end revenue_growth_20_percent_l907_90724


namespace height_of_triangle_l907_90719

theorem height_of_triangle
    (A : ℝ) (b : ℝ) (h : ℝ)
    (h1 : A = 30)
    (h2 : b = 12)
    (h3 : A = (b * h) / 2) :
    h = 5 :=
by
  sorry

end height_of_triangle_l907_90719


namespace part_a_part_b_l907_90755

-- Define the tower of exponents function for convenience
def tower (base : ℕ) (height : ℕ) : ℕ :=
  if height = 0 then 1 else base^(tower base (height - 1))

-- Part a: Tower of 3s with height 99 is greater than Tower of 2s with height 100
theorem part_a : tower 3 99 > tower 2 100 := sorry

-- Part b: Tower of 3s with height 100 is greater than Tower of 3s with height 99
theorem part_b : tower 3 100 > tower 3 99 := sorry

end part_a_part_b_l907_90755


namespace find_polynomials_l907_90756

-- Definition of polynomials in Lean
noncomputable def polynomials : Type := Polynomial ℝ

-- Main theorem statement
theorem find_polynomials : 
  ∀ p : polynomials, 
    (∀ x : ℝ, p.eval (5 * x) ^ 2 - 3 = p.eval (5 * x^2 + 1)) → 
    (p.eval 0 ≠ 0 → (∃ c : ℝ, (p = Polynomial.C c) ∧ (c = (1 + Real.sqrt 13) / 2 ∨ c = (1 - Real.sqrt 13) / 2))) ∧ 
    (p.eval 0 = 0 → ∀ x : ℝ, p.eval x = 0) :=
by
  sorry

end find_polynomials_l907_90756


namespace floor_neg_seven_four_is_neg_two_l907_90790

noncomputable def floor_neg_seven_four : Int :=
  Int.floor (-7 / 4)

theorem floor_neg_seven_four_is_neg_two : floor_neg_seven_four = -2 := by
  sorry

end floor_neg_seven_four_is_neg_two_l907_90790


namespace total_number_of_coins_l907_90760

-- Define conditions
def pennies : Nat := 38
def nickels : Nat := 27
def dimes : Nat := 19
def quarters : Nat := 24
def half_dollars : Nat := 13
def one_dollar_coins : Nat := 17
def two_dollar_coins : Nat := 5
def australian_fifty_cent_coins : Nat := 4
def mexican_one_peso_coins : Nat := 12

-- Define the problem as a theorem
theorem total_number_of_coins : 
  pennies + nickels + dimes + quarters + half_dollars + one_dollar_coins + two_dollar_coins + australian_fifty_cent_coins + mexican_one_peso_coins = 159 := by
  sorry

end total_number_of_coins_l907_90760


namespace find_k_l907_90771

theorem find_k 
  (k : ℤ) 
  (h : 2^2000 - 2^1999 - 2^1998 + 2^1997 = k * 2^1997) : 
  k = 3 :=
sorry

end find_k_l907_90771


namespace sanAntonioToAustin_passes_austinToSanAntonio_l907_90722

noncomputable def buses_passed : ℕ :=
  let austinToSanAntonio (n : ℕ) : ℕ := n * 2
  let sanAntonioToAustin (n : ℕ) : ℕ := n * 2 + 1
  let tripDuration : ℕ := 3
  if (austinToSanAntonio 3 - 0) <= tripDuration then 2 else 0

-- Proof statement
theorem sanAntonioToAustin_passes_austinToSanAntonio :
  buses_passed = 2 :=
  sorry

end sanAntonioToAustin_passes_austinToSanAntonio_l907_90722


namespace minimum_zeros_l907_90746

theorem minimum_zeros (n : ℕ) (a : Fin n → ℤ) (h : n = 2011)
  (H : ∀ i j k : Fin n, a i + a j + a k ∈ Set.range a) : 
  ∃ (num_zeros : ℕ), num_zeros ≥ 2009 ∧ (∃ f : Fin (num_zeros) → Fin n, ∀ i : Fin (num_zeros), a (f i) = 0) :=
sorry

end minimum_zeros_l907_90746


namespace probability_of_square_product_is_17_over_96_l907_90763

def num_tiles : Nat := 12
def num_die_faces : Nat := 8

def is_perfect_square (n : Nat) : Prop :=
  ∃ k : Nat, k * k = n

def favorable_outcomes_count : Nat :=
  -- Valid pairs where tile's number and die's number product is a perfect square
  List.length [ (1, 1), (1, 4), (2, 2), (4, 1),
                (1, 9), (3, 3), (9, 1), (4, 4),
                (2, 8), (8, 2), (5, 5), (6, 6),
                (4, 9), (9, 4), (7, 7), (8, 8),
                (9, 9) ] -- Equals 17 pairs

def total_outcomes_count : Nat :=
  num_tiles * num_die_faces

def probability_square_product : ℚ :=
  favorable_outcomes_count / total_outcomes_count

theorem probability_of_square_product_is_17_over_96 :
  probability_square_product = (17 : ℚ) / 96 := 
  by sorry

end probability_of_square_product_is_17_over_96_l907_90763


namespace neg_P_l907_90794

def P := ∃ x : ℝ, (0 < x) ∧ (3^x < x^3)

theorem neg_P : ¬P ↔ ∀ x : ℝ, (0 < x) → (3^x ≥ x^3) :=
by
  sorry

end neg_P_l907_90794


namespace lower_amount_rent_l907_90784

theorem lower_amount_rent (L : ℚ) (total_rent : ℚ) (reduction : ℚ)
  (h1 : total_rent = 2000)
  (h2 : reduction = 200)
  (h3 : 10 * (60 - L) = reduction) :
  L = 40 := by
  sorry

end lower_amount_rent_l907_90784


namespace smallest_c1_in_arithmetic_sequence_l907_90765

theorem smallest_c1_in_arithmetic_sequence (S3 S7 : ℕ) (S3_natural : S3 > 0) (S7_natural : S7 > 0)
    (c1_geq_one_third : ∀ d : ℚ, (c1 : ℚ) = (7*S3 - S7) / 14 → c1 ≥ 1/3) : 
    ∃ c1 : ℚ, c1 = 5/14 ∧ c1 ≥ 1/3 := 
by 
  sorry

end smallest_c1_in_arithmetic_sequence_l907_90765


namespace find_y_l907_90727

theorem find_y (y : ℕ) : (1 / 8) * 2^36 = 8^y → y = 11 := by
  sorry

end find_y_l907_90727


namespace smallest_n_sum_gt_10_pow_5_l907_90742

theorem smallest_n_sum_gt_10_pow_5 :
  ∃ (n : ℕ), (n ≥ 142) ∧ (5 * n^2 + 4 * n ≥ 100000) :=
by
  use 142
  sorry

end smallest_n_sum_gt_10_pow_5_l907_90742


namespace solve_inequality_system_l907_90783

theorem solve_inequality_system (x : ℝ) :
  (4 * x + 5 > x - 1) ∧ ((3 * x - 1) / 2 < x) ↔ (-2 < x ∧ x < 1) :=
by
  sorry

end solve_inequality_system_l907_90783


namespace area_triangle_PZQ_l907_90772

/-- 
In rectangle PQRS, side PQ measures 8 units and side QR measures 4 units.
Points X and Y are on side RS such that segment RX measures 2 units and
segment SY measures 3 units. Lines PX and QY intersect at point Z.
Prove the area of triangle PZQ is 128/3 square units.
-/

theorem area_triangle_PZQ {PQ QR RX SY : ℝ} (h1 : PQ = 8) (h2 : QR = 4) (h3 : RX = 2) (h4 : SY = 3) :
  let area_PZQ : ℝ := 8 * 4 / 2 * 8 / (3 * 2)
  area_PZQ = 128 / 3 :=
by
  sorry

end area_triangle_PZQ_l907_90772


namespace ground_beef_lean_beef_difference_l907_90780

theorem ground_beef_lean_beef_difference (x y z : ℕ) 
  (h1 : x + y + z = 20) 
  (h2 : y + 2 * z = 18) :
  x - z = 2 :=
sorry

end ground_beef_lean_beef_difference_l907_90780


namespace altered_solution_contains_60_liters_of_detergent_l907_90713

-- Definitions corresponding to the conditions
def initial_ratio_bleach_to_detergent_to_water : ℚ := 2 / 40 / 100
def initial_ratio_bleach_to_detergent : ℚ := 1 / 20
def initial_ratio_detergent_to_water : ℚ := 1 / 5

def altered_ratio_bleach_to_detergent : ℚ := 3 / 20
def altered_ratio_detergent_to_water : ℚ := 1 / 5

def water_in_altered_solution : ℚ := 300

-- We need to find the amount of detergent in the altered solution
def amount_of_detergent_in_altered_solution : ℚ := 20

-- The proportion and the final amount calculation
theorem altered_solution_contains_60_liters_of_detergent :
  (300 / 100) * (20) = 60 :=
by
  sorry

end altered_solution_contains_60_liters_of_detergent_l907_90713


namespace arithmetic_sequence_common_difference_l907_90701

theorem arithmetic_sequence_common_difference (a : ℕ → ℤ) (d : ℤ) (h_arith_seq: ∀ n, a n = a 1 + (n - 1) * d) 
  (h_cond1 : a 3 + a 9 = 4 * a 5) (h_cond2 : a 2 = -8) : 
  d = 4 :=
sorry

end arithmetic_sequence_common_difference_l907_90701


namespace range_f_l907_90781

noncomputable def f (a b x : ℝ) : ℝ :=
  Real.sqrt (a * Real.cos x ^ 2 + b * Real.sin x ^ 2) + 
  Real.sqrt (a * Real.sin x ^ 2 + b * Real.cos x ^ 2)

theorem range_f (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a ≠ b) :
  Set.range (f a b) = Set.Icc (Real.sqrt a + Real.sqrt b) (Real.sqrt (2 * (a + b))) :=
sorry

end range_f_l907_90781


namespace average_weight_of_girls_l907_90708

theorem average_weight_of_girls :
  ∀ (total_students boys girls total_weight class_average_weight boys_average_weight girls_average_weight : ℝ),
  total_students = 25 →
  boys = 15 →
  girls = 10 →
  boys + girls = total_students →
  class_average_weight = 45 →
  boys_average_weight = 48 →
  total_weight = 1125 →
  girls_average_weight = (total_weight - (boys * boys_average_weight)) / girls →
  total_weight = class_average_weight * total_students →
  girls_average_weight = 40.5 :=
by
  intros total_students boys girls total_weight class_average_weight boys_average_weight girls_average_weight
  sorry

end average_weight_of_girls_l907_90708


namespace yogurt_packs_ordered_l907_90747

theorem yogurt_packs_ordered (P : ℕ) (price_per_pack refund_amount : ℕ) (expired_percentage : ℚ)
  (h1 : price_per_pack = 12)
  (h2 : refund_amount = 384)
  (h3 : expired_percentage = 0.40)
  (h4 : refund_amount / price_per_pack = 32)
  (h5 : 32 / expired_percentage = P) :
  P = 80 :=
sorry

end yogurt_packs_ordered_l907_90747


namespace polygon_E_has_largest_area_l907_90705

-- Define the areas of square and right triangle
def area_square (side : ℕ): ℕ := side * side
def area_right_triangle (leg : ℕ): ℕ := (leg * leg) / 2

-- Define the areas of each polygon
def area_polygon_A : ℕ := 2 * (area_square 2) + (area_right_triangle 2)
def area_polygon_B : ℕ := 3 * (area_square 2)
def area_polygon_C : ℕ := (area_square 2) + 4 * (area_right_triangle 2)
def area_polygon_D : ℕ := 3 * (area_right_triangle 2)
def area_polygon_E : ℕ := 4 * (area_square 2)

-- The theorem assertion
theorem polygon_E_has_largest_area : 
  area_polygon_E = 16 ∧ 
  16 > area_polygon_A ∧
  16 > area_polygon_B ∧
  16 > area_polygon_C ∧
  16 > area_polygon_D := 
sorry

end polygon_E_has_largest_area_l907_90705


namespace smallest_k_divides_l907_90752

noncomputable def f (z : ℂ) : ℂ := z^12 + z^11 + z^8 + z^7 + z^6 + z^3 + 1

noncomputable def g (z : ℂ) (k : ℕ) : ℂ := z^k - 1

theorem smallest_k_divides (k : ℕ) : k = 84 :=
by
  sorry

end smallest_k_divides_l907_90752


namespace proof1_proof2_l907_90706

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ :=
  |a * x - 2| - |x + 2|

-- Statement for proof 1
theorem proof1 (x : ℝ)
  (a : ℝ) (h : a = 2) (hx : f 2 x ≤ 1) : -1/3 ≤ x ∧ x ≤ 5 :=
sorry

-- Statement for proof 2
theorem proof2 (a : ℝ)
  (h : ∀ x : ℝ, -4 ≤ f a x ∧ f a x ≤ 4) : a = 1 ∨ a = -1 :=
sorry

end proof1_proof2_l907_90706


namespace inequality_holds_l907_90733

theorem inequality_holds (a b c d : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) (h_pos_d : 0 < d) :
  (a^3 / (a^3 + 15 * b * c * d))^(1/2) ≥ a^(15/8) / (a^(15/8) + b^(15/8) + c^(15/8) + d^(15/8)) :=
sorry

end inequality_holds_l907_90733


namespace negation_of_prop_p_l907_90710

theorem negation_of_prop_p (p : Prop) (h : ∀ x: ℝ, 0 < x → x > Real.log x) :
  (¬ (∀ x: ℝ, 0 < x → x > Real.log x)) ↔ (∃ x_0: ℝ, 0 < x_0 ∧ x_0 ≤ Real.log x_0) :=
by sorry

end negation_of_prop_p_l907_90710


namespace intersection_P_Q_eq_Q_l907_90728

-- Definitions of P and Q
def P : Set ℝ := {y | ∃ x : ℝ, y = x^2}
def Q : Set ℝ := {y | ∃ x : ℝ, y = x^2 + 1}

-- Statement to prove P ∩ Q = Q
theorem intersection_P_Q_eq_Q : P ∩ Q = Q := 
by 
  sorry

end intersection_P_Q_eq_Q_l907_90728


namespace bowling_average_decrease_l907_90788

theorem bowling_average_decrease 
  (original_average : ℚ) 
  (wickets_last_match : ℚ) 
  (runs_last_match : ℚ) 
  (original_wickets : ℚ) 
  (original_total_runs : ℚ := original_wickets * original_average) 
  (new_total_wickets : ℚ := original_wickets + wickets_last_match) 
  (new_total_runs : ℚ := original_total_runs + runs_last_match)
  (new_average : ℚ := new_total_runs / new_total_wickets) :
  original_wickets = 85 → original_average = 12.4 → wickets_last_match = 5 → runs_last_match = 26 → new_average = 12 →
  original_average - new_average = 0.4 := 
by 
  intros 
  sorry

end bowling_average_decrease_l907_90788


namespace gabby_mom_gave_20_l907_90775

theorem gabby_mom_gave_20 (makeup_set_cost saved_money more_needed total_needed mom_money : ℕ)
  (h1 : makeup_set_cost = 65)
  (h2 : saved_money = 35)
  (h3 : more_needed = 10)
  (h4 : total_needed = makeup_set_cost - saved_money)
  (h5 : total_needed - mom_money = more_needed) :
  mom_money = 20 :=
by
  sorry

end gabby_mom_gave_20_l907_90775


namespace smallest_boxes_l907_90748

-- Definitions based on the conditions:
def divisible_by (n d : Nat) : Prop := ∃ k, n = d * k

-- The statement to be proved:
theorem smallest_boxes (n : Nat) : 
  divisible_by n 5 ∧ divisible_by n 24 -> n = 120 :=
by sorry

end smallest_boxes_l907_90748


namespace mcq_options_l907_90717

theorem mcq_options :
  ∃ n : ℕ, (1/n : ℝ) * (1/2) * (1/2) = (1/12) ∧ n = 3 :=
by
  sorry

end mcq_options_l907_90717


namespace grains_in_one_tsp_l907_90714

-- Definitions based on conditions
def grains_in_one_cup : Nat := 480
def half_cup_is_8_tbsp : Nat := 8
def one_tbsp_is_3_tsp : Nat := 3

-- Theorem statement
theorem grains_in_one_tsp :
  let grains_in_half_cup := grains_in_one_cup / 2
  let grains_in_one_tbsp := grains_in_half_cup / half_cup_is_8_tbsp
  grains_in_one_tbsp / one_tbsp_is_3_tsp = 10 :=
by
  sorry

end grains_in_one_tsp_l907_90714


namespace greatest_price_book_l907_90730

theorem greatest_price_book (p : ℕ) (B : ℕ) (D : ℕ) (F : ℕ) (T : ℚ) 
  (h1 : B = 20) 
  (h2 : D = 200) 
  (h3 : F = 5)
  (h4 : T = 0.07) 
  (h5 : ∀ p, 20 * p * (1 + T) ≤ (D - F)) : 
  p ≤ 9 :=
by
  sorry

end greatest_price_book_l907_90730


namespace find_D_E_l907_90737

/--
Consider the circle given by \( x^2 + y^2 + D \cdot x + E \cdot y + F = 0 \) that is symmetrical with
respect to the line \( l_1: x - y + 4 = 0 \) and the line \( l_2: x + 3y = 0 \). Prove that the values 
of \( D \) and \( E \) are \( 12 \) and \( -4 \), respectively.
-/
theorem find_D_E (D E F : ℝ) (h1 : -D/2 + E/2 + 4 = 0) (h2 : -D/2 - 3*E/2 = 0) : D = 12 ∧ E = -4 :=
by
  sorry

end find_D_E_l907_90737


namespace area_proof_l907_90782

def square_side_length : ℕ := 2
def triangle_leg_length : ℕ := 2

-- Definition of the initial square area
def square_area (side_length : ℕ) : ℕ := side_length * side_length

-- Definition of the area for one isosceles right triangle
def triangle_area (leg_length : ℕ) : ℕ := (leg_length * leg_length) / 2

-- Area of the initial square
def R_square_area : ℕ := square_area square_side_length

-- Area of the 12 isosceles right triangles
def total_triangle_area : ℕ := 12 * triangle_area triangle_leg_length

-- Total area of region R
def R_area : ℕ := R_square_area + total_triangle_area

-- Smallest convex polygon S is a larger square with side length 8
def S_area : ℕ := square_area (4 * square_side_length)

-- Area inside S but outside R
def area_inside_S_outside_R : ℕ := S_area - R_area

theorem area_proof : area_inside_S_outside_R = 36 :=
by
  sorry

end area_proof_l907_90782


namespace find_a_5_in_arithmetic_sequence_l907_90741

variable {a : ℕ → ℕ}

def arithmetic_sequence (a : ℕ → ℕ) (a1 d : ℕ) : Prop :=
  a 1 = a1 ∧ ∀ n, a (n + 1) = a n + d

theorem find_a_5_in_arithmetic_sequence (h : arithmetic_sequence a 1 2) : a 5 = 9 :=
sorry

end find_a_5_in_arithmetic_sequence_l907_90741


namespace max_viewers_per_week_l907_90718

theorem max_viewers_per_week :
  ∃ (x y : ℕ), 80 * x + 40 * y ≤ 320 ∧ x + y ≥ 6 ∧ 600000 * x + 200000 * y = 2000000 :=
by
  sorry

end max_viewers_per_week_l907_90718


namespace train_ride_duration_is_360_minutes_l907_90704

-- Define the conditions given in the problem
def arrived_at_station_at_8 (t : ℕ) : Prop := t = 8 * 60
def train_departed_at_835 (t_depart : ℕ) : Prop := t_depart = 8 * 60 + 35
def train_arrived_at_215 (t_arrive : ℕ) : Prop := t_arrive = 14 * 60 + 15
def exited_station_at_3 (t_exit : ℕ) : Prop := t_exit = 15 * 60

-- Define the problem statement
theorem train_ride_duration_is_360_minutes (boarding alighting : ℕ) :
  arrived_at_station_at_8 boarding ∧ 
  train_departed_at_835 boarding ∧ 
  train_arrived_at_215 alighting ∧ 
  exited_station_at_3 alighting → 
  alighting - boarding = 360 := 
by
  sorry

end train_ride_duration_is_360_minutes_l907_90704


namespace inversely_proportional_decrease_l907_90798

theorem inversely_proportional_decrease :
  ∀ {x y q c : ℝ}, 
  0 < x ∧ 0 < y ∧ 0 < c ∧ 0 < q →
  (x * y = c) →
  (((1 + q / 100) * x) * ((100 / (100 + q)) * y) = c) →
  ((y - (100 / (100 + q)) * y) / y) * 100 = 100 * q / (100 + q) :=
by
  intros x y q c hb hxy hxy'
  sorry

end inversely_proportional_decrease_l907_90798


namespace extreme_point_of_f_l907_90795

open Real

noncomputable def f (x : ℝ) : ℝ := (3 / 2) * x^2 - log x

theorem extreme_point_of_f : ∃ x₀ > 0, f x₀ = f (sqrt 3 / 3) ∧ 
  (∀ x < sqrt 3 / 3, f x > f (sqrt 3 / 3)) ∧
  (∀ x > sqrt 3 / 3, f x > f (sqrt 3 / 3)) :=
sorry

end extreme_point_of_f_l907_90795


namespace intersection_of_M_and_N_l907_90758

noncomputable def M : Set ℝ := {y | ∃ x : ℝ, y = x ^ 2 - 1}
noncomputable def N : Set ℝ := {x | -3 ≤ x ∧ x ≤ 3}
noncomputable def intersection : Set ℝ := {z | -1 ≤ z ∧ z ≤ 3}

theorem intersection_of_M_and_N : M ∩ N = {z | -1 ≤ z ∧ z ≤ 3} := 
sorry

end intersection_of_M_and_N_l907_90758


namespace correct_proposition_l907_90766

-- Defining the function f(x)
def f (x : ℝ) : ℝ := x^3 - x^2 - x + 2

-- Defining proposition p
def p : Prop := ∃ x : ℝ, 0 < x ∧ x < 2 ∧ f x < 0

-- Defining proposition q
def q : Prop := ∀ x y : ℝ, x + y > 4 → x > 2 ∧ y > 2

-- Theorem statement to prove the correct answer
theorem correct_proposition : (¬ p) ∧ (¬ q) :=
by
  sorry

end correct_proposition_l907_90766


namespace barney_extra_weight_l907_90753

-- Define the weight of a regular dinosaur
def regular_dinosaur_weight : ℕ := 800

-- Define the combined weight of five regular dinosaurs
def five_regular_dinosaurs_weight : ℕ := 5 * regular_dinosaur_weight

-- Define the total weight of Barney and the five regular dinosaurs together
def total_combined_weight : ℕ := 9500

-- Define the weight of Barney
def barney_weight : ℕ := total_combined_weight - five_regular_dinosaurs_weight

-- The proof statement
theorem barney_extra_weight : barney_weight - five_regular_dinosaurs_weight = 1500 :=
by sorry

end barney_extra_weight_l907_90753


namespace no_nat_number_with_perfect_square_l907_90750

theorem no_nat_number_with_perfect_square (n : Nat) : 
  ¬ ∃ m : Nat, m * m = n^6 + 3 * n^5 - 5 * n^4 - 15 * n^3 + 4 * n^2 + 12 * n + 3 := 
  by
  sorry

end no_nat_number_with_perfect_square_l907_90750


namespace sqrt5_lt_sqrt2_plus_1_l907_90702

theorem sqrt5_lt_sqrt2_plus_1 : Real.sqrt 5 < Real.sqrt 2 + 1 :=
sorry

end sqrt5_lt_sqrt2_plus_1_l907_90702


namespace min_value_expression_l907_90778

theorem min_value_expression (α β : ℝ) :
  ∃ x y, x = 3 * Real.cos α + 6 * Real.sin β ∧
         y = 3 * Real.sin α + 6 * Real.cos β ∧
         (x - 10)^2 + (y - 18)^2 = 121 :=
by
  sorry

end min_value_expression_l907_90778
