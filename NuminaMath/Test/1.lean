import Mathlib
import Mathlib.Data.Complex.Exponential
import Mathlib.Data.Nat.Choose.Basic
import Mathlib.Data.Rat.Defs
import Mathlib.Data.Real.Basic

namespace grunters_win_4_out_of_6_1_1569

/-- The Grunters have a probability of winning any given game as 60% --/
def p : ℚ := 3 / 5

/-- The Grunters have a probability of losing any given game as 40% --/
def q : ℚ := 1 - p

/-- The binomial coefficient for choosing exactly 4 wins out of 6 games --/
def binomial_6_4 : ℚ := Nat.choose 6 4

/-- The probability that the Grunters win exactly 4 out of the 6 games --/
def prob_4_wins : ℚ := binomial_6_4 * (p ^ 4) * (q ^ 2)

/--
The probability that the Grunters win exactly 4 out of the 6 games is
exactly $\frac{4860}{15625}$.
--/
theorem grunters_win_4_out_of_6 : prob_4_wins = 4860 / 15625 := by
  sorry

end grunters_win_4_out_of_6_1_1569


namespace math_competition_rankings_1_1639

noncomputable def rankings (n : ℕ) : ℕ → Prop := sorry

theorem math_competition_rankings :
  (∀ (A B C D E : ℕ), 
    A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ A ≠ E ∧
    B ≠ C ∧ B ≠ D ∧ B ≠ E ∧
    C ≠ D ∧ C ≠ E ∧
    D ≠ E ∧
    
    -- A's guesses
    (rankings A 1 → rankings B 3 ∧ rankings C 5) →
    -- B's guesses
    (rankings B 2 → rankings E 4 ∧ rankings D 5) →
    -- C's guesses
    (rankings C 3 → rankings A 1 ∧ rankings E 4) →
    -- D's guesses
    (rankings D 4 → rankings C 1 ∧ rankings D 2) →
    -- E's guesses
    (rankings E 5 → rankings A 3 ∧ rankings D 4) →
    -- Condition that each position is guessed correctly by someone
    (∃ i, rankings A i) ∧
    (∃ i, rankings B i) ∧
    (∃ i, rankings C i) ∧
    (∃ i, rankings D i) ∧
    (∃ i, rankings E i) →
    
    -- The actual placing according to derived solution
    rankings A 1 ∧ 
    rankings D 2 ∧ 
    rankings B 3 ∧ 
    rankings E 4 ∧ 
    rankings C 5) :=
sorry

end math_competition_rankings_1_1639


namespace plane_through_intersection_1_1353

def plane1 (x y z : ℝ) : Prop := x + y + 5 * z - 1 = 0
def plane2 (x y z : ℝ) : Prop := 2 * x + 3 * y - z + 2 = 0
def pointM (x y z : ℝ) : Prop := (x, y, z) = (3, 2, 1)

theorem plane_through_intersection (x y z : ℝ) :
  plane1 x y z ∧ plane2 x y z ∧ pointM x y z → 5 * x + 14 * y - 74 * z + 31 = 0 := by
  intro h
  sorry

end plane_through_intersection_1_1353


namespace simplify_fraction_1_1147

theorem simplify_fraction :
  ((1 / 4) + (1 / 6)) / ((3 / 8) - (1 / 3)) = 10 := by
  sorry

end simplify_fraction_1_1147


namespace figure_perimeter_1_1475

-- Define the side length of the square and the triangles.
def square_side_length : ℕ := 3
def triangle_side_length : ℕ := 2

-- Calculate the perimeter of the figure
def perimeter (a b : ℕ) : ℕ := 2 * a + 2 * b

-- Statement to prove
theorem figure_perimeter : perimeter square_side_length triangle_side_length = 10 := 
by 
  -- "sorry" denotes that the proof is omitted.
  sorry

end figure_perimeter_1_1475


namespace twenty_eight_is_seventy_percent_of_what_number_1_1672

theorem twenty_eight_is_seventy_percent_of_what_number (x : ℝ) (h : 28 / x = 70 / 100) : x = 40 :=
by
  sorry

end twenty_eight_is_seventy_percent_of_what_number_1_1672


namespace commercial_break_duration_1_1097

theorem commercial_break_duration (n1 n2 t1 t2 : ℕ) (h1 : n1 = 3) (h2: t1 = 5) (h3 : n2 = 11) (h4 : t2 = 2) : 
  n1 * t1 + n2 * t2 = 37 := 
by 
  sorry

end commercial_break_duration_1_1097


namespace num_solutions_20_1_1346

def num_solutions (n : ℕ) : ℕ :=
  4 * n

theorem num_solutions_20 : num_solutions 20 = 80 := by
  sorry

end num_solutions_20_1_1346


namespace rationalize_denominator_correct_1_1159

noncomputable def rationalize_denominator : ℚ :=
  let A := 5
  let B := 49
  let C := 21
  A + B + C

theorem rationalize_denominator_correct :
  (let A := 5
   let B := 49
   let C := 21
   (A + B + C) = 75) :=
by
  sorry

end rationalize_denominator_correct_1_1159


namespace find_rectangle_width_1_1755

variable (length_square : ℕ) (length_rectangle : ℕ) (width_rectangle : ℕ)

-- Given conditions
def square_side_length := 700
def rectangle_length := 400
def square_perimeter := 4 * square_side_length
def rectangle_perimeter := square_perimeter / 2
def rectangle_perimeter_eq := 2 * length_rectangle + 2 * width_rectangle

-- Statement to prove
theorem find_rectangle_width :
  (square_perimeter = 2800) →
  (rectangle_perimeter = 1400) →
  (length_rectangle = 400) →
  (rectangle_perimeter_eq = 1400) →
  (width_rectangle = 300) :=
by
  intros
  sorry

end find_rectangle_width_1_1755


namespace arithmetic_expression_1_1323

theorem arithmetic_expression :
  7 / 2 - 3 - 5 + 3 * 4 = 7.5 :=
by {
  -- We state the main equivalence to be proven
  sorry
}

end arithmetic_expression_1_1323


namespace expense_recording_1_1370

-- Define the recording of income and expenses
def record_income (amount : Int) : Int := amount
def record_expense (amount : Int) : Int := -amount

-- Given conditions
def income_example := record_income 500
def expense_example := record_expense 400

-- Prove that an expense of 400 yuan is recorded as -400 yuan
theorem expense_recording : record_expense 400 = -400 :=
  by sorry

end expense_recording_1_1370


namespace find_sister_candy_initially_1_1257

-- Defining the initial pieces of candy Katie had.
def katie_candy : ℕ := 8

-- Defining the pieces of candy Katie's sister had initially.
def sister_candy_initially : ℕ := sorry -- To be determined

-- The total number of candy pieces they had after eating 8 pieces.
def total_remaining_candy : ℕ := 23

theorem find_sister_candy_initially : 
  (katie_candy + sister_candy_initially - 8 = total_remaining_candy) → (sister_candy_initially = 23) :=
by
  sorry

end find_sister_candy_initially_1_1257


namespace trip_duration_is_6_hours_1_1496

def distance_1 := 55 * 4
def total_distance (A : ℕ) := distance_1 + 70 * A
def total_time (A : ℕ) := 4 + A
def average_speed (A : ℕ) := total_distance A / total_time A

theorem trip_duration_is_6_hours (A : ℕ) (h : 60 = average_speed A) : total_time A = 6 :=
by
  sorry

end trip_duration_is_6_hours_1_1496


namespace simplified_expression_evaluates_to_2_1_1892

-- Definitions based on given conditions:
def x := 2 -- where x = (1/2)^(-1)
def y := 1 -- where y = (-2023)^0

-- Main statement to prove:
theorem simplified_expression_evaluates_to_2 :
  ((2 * x - y) / (x + y) - (x * x - 2 * x * y + y * y) / (x * x - y * y)) / (x - y) / (x + y) = 2 :=
by
  sorry

end simplified_expression_evaluates_to_2_1_1892


namespace sequence_term_is_square_1_1378

noncomputable def sequence_term (n : ℕ) : ℕ :=
  let part1 := (10 ^ (n + 1) - 1) / 9
  let part2 := (10 ^ (2 * n + 2) - 10 ^ (n + 1)) / 9
  1 + 4 * part1 + 4 * part2

theorem sequence_term_is_square (n : ℕ) : ∃ k : ℕ, k^2 = sequence_term n :=
by
  sorry

end sequence_term_is_square_1_1378


namespace find_number_1_1653

theorem find_number (n : ℕ) (some_number : ℕ) 
  (h : (1/5 : ℝ)^n * (1/4 : ℝ)^(18 : ℕ) = 1 / (2 * (some_number : ℝ)^n))
  (hn : n = 35) : some_number = 10 := 
by 
  sorry

end find_number_1_1653


namespace cylinder_surface_area_1_1649

theorem cylinder_surface_area
  (l : ℝ) (r : ℝ) (unfolded_square_side : ℝ) (base_circumference : ℝ)
  (hl : unfolded_square_side = 2 * π)
  (hl_gen : l = 2 * π)
  (hc : base_circumference = 2 * π)
  (hr : r = 1) :
  2 * π * r * (r + l) = 2 * π + 4 * π^2 :=
by
  sorry

end cylinder_surface_area_1_1649


namespace length_of_top_side_1_1196

def height_of_trapezoid : ℝ := 8
def area_of_trapezoid : ℝ := 72
def top_side_is_shorter (b : ℝ) : Prop := ∃ t : ℝ, t = b - 6

theorem length_of_top_side (b t : ℝ) (h_height : height_of_trapezoid = 8)
  (h_area : area_of_trapezoid = 72) 
  (h_top_side : top_side_is_shorter b)
  (h_area_formula : (1/2) * (b + t) * 8 = 72) : t = 6 := 
by 
  sorry

end length_of_top_side_1_1196


namespace exists_x_abs_ge_one_fourth_1_1086

theorem exists_x_abs_ge_one_fourth :
  ∀ (a b c : ℝ), ∃ x : ℝ, |x| ≤ 1 ∧ |x^3 + a * x^2 + b * x + c| ≥ 1 / 4 :=
by sorry

end exists_x_abs_ge_one_fourth_1_1086


namespace bagel_pieces_after_10_cuts_1_1984

def bagel_pieces_after_cuts (initial_pieces : ℕ) (cuts : ℕ) : ℕ :=
  initial_pieces + cuts

theorem bagel_pieces_after_10_cuts : bagel_pieces_after_cuts 1 10 = 11 := by
  sorry

end bagel_pieces_after_10_cuts_1_1984


namespace smallest_positive_multiple_1_1234

theorem smallest_positive_multiple (a : ℕ) (h : a > 0) : ∃ a > 0, (31 * a) % 103 = 7 := 
sorry

end smallest_positive_multiple_1_1234


namespace right_triangle_sqrt_1_1366

noncomputable def sqrt_2 := Real.sqrt 2
noncomputable def sqrt_3 := Real.sqrt 3
noncomputable def sqrt_5 := Real.sqrt 5

theorem right_triangle_sqrt: 
  (sqrt_2 ^ 2 + sqrt_3 ^ 2 = sqrt_5 ^ 2) :=
by
  sorry

end right_triangle_sqrt_1_1366


namespace solution_set_of_inequality_1_1527

theorem solution_set_of_inequality :
  { x : ℝ | |x + 1| + |x - 4| ≥ 7 } = { x : ℝ | x ≤ -2 ∨ x ≥ 5 } := sorry

end solution_set_of_inequality_1_1527


namespace initial_salt_percentage_1_1956

theorem initial_salt_percentage (P : ℕ) : 
  let initial_solution := 100 
  let added_salt := 20 
  let final_solution := initial_solution + added_salt 
  (P / 100) * initial_solution + added_salt = (25 / 100) * final_solution → 
  P = 10 := 
by
  sorry

end initial_salt_percentage_1_1956


namespace uncle_zhang_age_1_1565

theorem uncle_zhang_age (z l : ℕ) (h1 : z + l = 56) (h2 : z = l - (l / 2)) : z = 24 :=
by sorry

end uncle_zhang_age_1_1565


namespace black_region_area_is_correct_1_1277

noncomputable def area_of_black_region : ℕ :=
  let area_large_square := 10 * 10
  let area_first_smaller_square := 4 * 4
  let area_second_smaller_square := 2 * 2
  area_large_square - (area_first_smaller_square + area_second_smaller_square)

theorem black_region_area_is_correct :
  area_of_black_region = 80 :=
by
  sorry

end black_region_area_is_correct_1_1277


namespace rex_lesson_schedule_1_1208

-- Define the total lessons and weeks
def total_lessons : ℕ := 40
def weeks_completed : ℕ := 6
def weeks_remaining : ℕ := 4

-- Define the proof statement
theorem rex_lesson_schedule : (weeks_completed + weeks_remaining) * 4 = total_lessons := by
  -- Proof placeholder, to be filled in 
  sorry

end rex_lesson_schedule_1_1208


namespace james_has_43_oreos_1_1396

def james_oreos (jordan : ℕ) : ℕ := 7 + 4 * jordan

theorem james_has_43_oreos (jordan : ℕ) (total : ℕ) (h1 : total = jordan + james_oreos jordan) (h2 : total = 52) : james_oreos jordan = 43 :=
by
  sorry

end james_has_43_oreos_1_1396


namespace proof_numbers_exist_1_1678

noncomputable def exists_numbers : Prop :=
  ∃ a b c : ℕ, a > 10^10 ∧ b > 10^10 ∧ c > 10^10 ∧
  (a * b % (a + 2012) = 0) ∧
  (a * c % (a + 2012) = 0) ∧
  (b * c % (b + 2012) = 0) ∧
  (a * b * c % (b + 2012) = 0) ∧
  (a * b * c % (c + 2012) = 0)

theorem proof_numbers_exist : exists_numbers :=
  sorry

end proof_numbers_exist_1_1678


namespace roots_equal_of_quadratic_eq_zero_1_1994

theorem roots_equal_of_quadratic_eq_zero (a : ℝ) :
  (∃ x : ℝ, (x^2 - a*x + 1) = 0 ∧ (∀ y : ℝ, (y^2 - a*y + 1) = 0 → y = x)) → (a = 2 ∨ a = -2) :=
by
  sorry

end roots_equal_of_quadratic_eq_zero_1_1994


namespace find_A_in_terms_of_B_and_C_1_1905

noncomputable def f (A B : ℝ) (x : ℝ) := A * x - 3 * B^2
noncomputable def g (B C : ℝ) (x : ℝ) := B * x + C

theorem find_A_in_terms_of_B_and_C (A B C : ℝ) (h : B ≠ 0) (h1 : f A B (g B C 1) = 0) : A = 3 * B^2 / (B + C) :=
by sorry

end find_A_in_terms_of_B_and_C_1_1905


namespace arctan_addition_formula_1_1466

noncomputable def arctan_add : ℝ :=
  Real.arctan (1 / 3) + Real.arctan (3 / 8)

theorem arctan_addition_formula :
  arctan_add = Real.arctan (17 / 21) :=
by
  sorry

end arctan_addition_formula_1_1466


namespace amount_paid_to_Y_1_1852

-- Definition of the conditions.
def total_payment (X Y : ℕ) : Prop := X + Y = 330
def payment_relation (X Y : ℕ) : Prop := X = 12 * Y / 10

-- The theorem we want to prove.
theorem amount_paid_to_Y (X Y : ℕ) (h1 : total_payment X Y) (h2 : payment_relation X Y) : Y = 150 := 
by 
  sorry

end amount_paid_to_Y_1_1852


namespace calculate_expression_1_1536

theorem calculate_expression : 
  (2^10 + (3^6 / 3^2)) = 1105 := 
by 
  -- Steps involve intermediate calculations
  -- for producing (2^10 = 1024), (3^6 = 729), (3^2 = 9)
  -- and then finding (729 / 9 = 81), (1024 + 81 = 1105)
  sorry

end calculate_expression_1_1536


namespace find_whole_number_M_1_1123

-- Define the conditions
def condition (M : ℕ) : Prop :=
  21 < M ∧ M < 23

-- Define the main theorem to be proven
theorem find_whole_number_M (M : ℕ) (h : condition M) : M = 22 := by
  sorry

end find_whole_number_M_1_1123


namespace cookies_last_days_1_1598

variable (c1 c2 t : ℕ)

/-- Jackson's oldest son gets 4 cookies after school each day, and his youngest son gets 2 cookies. 
There are 54 cookies in the box, so the number of days the box will last is 9. -/
theorem cookies_last_days (h1 : c1 = 4) (h2 : c2 = 2) (h3 : t = 54) : 
  t / (c1 + c2) = 9 := by
  sorry

end cookies_last_days_1_1598


namespace repeating_decimal_base_1_1908

theorem repeating_decimal_base (k : ℕ) (h_pos : 0 < k) (h_repr : (9 : ℚ) / 61 = (3 * k + 4) / (k^2 - 1)) : k = 21 :=
  sorry

end repeating_decimal_base_1_1908


namespace subtraction_division_1_1982

theorem subtraction_division : 3550 - (1002 / 20.04) = 3499.9501 := by
  sorry

end subtraction_division_1_1982


namespace range_of_m_1_1128

theorem range_of_m (m : ℝ) : 
  (∀ x : ℝ, x ≤ 3 → (x ≤ m → (x < y → y < m))) → m ≥ 3 := 
by
  sorry

end range_of_m_1_1128


namespace president_and_committee_1_1507

def combinatorial (n k : ℕ) : ℕ := Nat.choose n k

theorem president_and_committee :
  let num_people := 10
  let num_president := 1
  let num_committee := 3
  let num_ways_president := 10
  let num_remaining_people := num_people - num_president
  let num_ways_committee := combinatorial num_remaining_people num_committee
  num_ways_president * num_ways_committee = 840 := 
by
  sorry

end president_and_committee_1_1507


namespace increased_work_1_1072

variable (W p : ℕ)

theorem increased_work (hW : W > 0) (hp : p > 0) : 
  (W / (7 * p / 8)) - (W / p) = W / (7 * p) := 
sorry

end increased_work_1_1072


namespace binomial_divisible_by_prime_1_1099

theorem binomial_divisible_by_prime (p n : ℕ) (hp : Nat.Prime p) (hn : n ≥ p) :
  (Nat.choose n p) - (n / p) % p = 0 := 
sorry

end binomial_divisible_by_prime_1_1099


namespace frank_reading_days_1_1724

-- Define the parameters
def pages_weekdays : ℚ := 5.7
def pages_weekends : ℚ := 9.5
def total_pages : ℚ := 576
def pages_per_week : ℚ := (pages_weekdays * 5) + (pages_weekends * 2)

-- Define the property to be proved
theorem frank_reading_days : 
  (total_pages / pages_per_week).floor * 7 + 
  (total_pages - (total_pages / pages_per_week).floor * pages_per_week) / pages_weekdays 
  = 85 := 
  by
    sorry

end frank_reading_days_1_1724


namespace correct_statements_count_1_1183

theorem correct_statements_count :
  (∀ x > 0, x > Real.sin x) ∧
  (¬ (∀ x > 0, x - Real.log x > 0) ↔ (∃ x > 0, x - Real.log x ≤ 0)) ∧
  ¬ (∀ p q : Prop, (p ∨ q) → (p ∧ q)) →
  2 = 2 :=
by sorry

end correct_statements_count_1_1183


namespace range_of_m_1_1033

theorem range_of_m (x : ℝ) (m : ℝ) (hx : 0 < x ∧ x < π) 
  (h : Real.cot (x / 3) = m * Real.cot x): m > 3 ∨ m < 0 :=
sorry

end range_of_m_1_1033


namespace x_coordinate_incenter_eq_1_1864

theorem x_coordinate_incenter_eq {x y : ℝ} :
  (y = 0 → x + y = 3 → x = 0) → 
  (y = x → y = -x + 3 → x = 3 / 2) :=
by
  sorry

end x_coordinate_incenter_eq_1_1864


namespace number_x_is_divided_by_1_1121

-- Define the conditions
variable (x y n : ℕ)
variable (cond1 : x = n * y + 4)
variable (cond2 : 2 * x = 8 * 3 * y + 3)
variable (cond3 : 13 * y - x = 1)

-- Define the statement to be proven
theorem number_x_is_divided_by : n = 11 :=
by
  sorry

end number_x_is_divided_by_1_1121


namespace stools_chopped_up_1_1058

variable (chairs tables stools : ℕ)
variable (sticks_per_chair sticks_per_table sticks_per_stool : ℕ)
variable (sticks_per_hour hours total_sticks_from_chairs tables_sticks required_sticks : ℕ)

theorem stools_chopped_up (h1 : sticks_per_chair = 6)
                         (h2 : sticks_per_table = 9)
                         (h3 : sticks_per_stool = 2)
                         (h4 : sticks_per_hour = 5)
                         (h5 : chairs = 18)
                         (h6 : tables = 6)
                         (h7 : hours = 34)
                         (h8 : total_sticks_from_chairs = chairs * sticks_per_chair)
                         (h9 : tables_sticks = tables * sticks_per_table)
                         (h10 : required_sticks = hours * sticks_per_hour)
                         (h11 : total_sticks_from_chairs + tables_sticks = 162) :
                         stools = 4 := by
  sorry

end stools_chopped_up_1_1058


namespace books_of_jason_1_1874

theorem books_of_jason (M J : ℕ) (hM : M = 42) (hTotal : M + J = 60) : J = 18 :=
by
  sorry

end books_of_jason_1_1874


namespace surface_area_spherical_segment_1_1698

-- Definitions based on given conditions
variables {R h : ℝ}

-- The theorem to be proven
theorem surface_area_spherical_segment (h_pos : 0 < h) (R_pos : 0 < R)
  (planes_not_intersect_sphere : h < 2 * R) :
  S = 2 * π * R * h := by
  sorry

end surface_area_spherical_segment_1_1698


namespace ratio_of_saramago_readers_1_1586

theorem ratio_of_saramago_readers 
  (W : ℕ) (S K B N : ℕ)
  (h1 : W = 42)
  (h2 : K = W / 6)
  (h3 : B = 3)
  (h4 : N = (S - B) - 1)
  (h5 : W = (S - B) + (K - B) + B + N) :
  S / W = 1 / 2 :=
by
  sorry

end ratio_of_saramago_readers_1_1586


namespace total_selling_price_is_correct_1_1961

def original_price : ℝ := 120
def discount_rate : ℝ := 0.30
def tax_rate : ℝ := 0.15

def discount : ℝ := discount_rate * original_price
def sale_price : ℝ := original_price - discount
def tax : ℝ := tax_rate * sale_price
def total_selling_price : ℝ := sale_price + tax

theorem total_selling_price_is_correct : total_selling_price = 96.6 := by
  sorry

end total_selling_price_is_correct_1_1961


namespace fraction_of_usual_speed_1_1331

-- Definitions based on conditions
variable (S R : ℝ)
variable (h1 : S * 60 = R * 72)

-- Goal statement
theorem fraction_of_usual_speed (h1 : S * 60 = R * 72) : R / S = 5 / 6 :=
by
  sorry

end fraction_of_usual_speed_1_1331


namespace solve_frac_eqn_1_1048

theorem solve_frac_eqn (x : ℝ) :
  (1 / ((x - 1) * (x - 3)) + 1 / ((x - 3) * (x - 5)) +
   1 / ((x - 5) * (x - 7)) + 1 / ((x - 7) * (x - 9)) = 1 / 8) ↔ 
  (x = 13 ∨ x = -3) :=
by
  sorry

end solve_frac_eqn_1_1048


namespace exists_irrationals_floor_neq_1_1693

-- Define irrationality of a number
def irrational (x : ℝ) : Prop :=
  ¬ ∃ (r : ℚ), x = r

theorem exists_irrationals_floor_neq :
  ∃ (a b : ℝ), irrational a ∧ irrational b ∧ 1 < a ∧ 1 < b ∧ 
  ∀ (m n : ℕ), ⌊a ^ m⌋ ≠ ⌊b ^ n⌋ :=
by
  sorry

end exists_irrationals_floor_neq_1_1693


namespace unique_positive_integer_solution_1_1796

theorem unique_positive_integer_solution :
  ∃! n : ℕ, n > 0 ∧ ∃ k : ℕ, n^4 - n^3 + 3*n^2 + 5 = k^2 :=
by
  sorry

end unique_positive_integer_solution_1_1796


namespace ending_number_is_54_1_1092

def first_even_after_15 : ℕ := 16
def evens_between (a b : ℕ) : ℕ := (b - first_even_after_15) / 2 + 1

theorem ending_number_is_54 (n : ℕ) (h : evens_between 15 n = 20) : n = 54 :=
by {
  sorry
}

end ending_number_is_54_1_1092


namespace village_Y_initial_population_1_1204

def population_X := 76000
def decrease_rate_X := 1200
def increase_rate_Y := 800
def years := 17

def population_X_after_17_years := population_X - decrease_rate_X * years
def population_Y_after_17_years (P : Nat) := P + increase_rate_Y * years

theorem village_Y_initial_population (P : Nat) (h : population_Y_after_17_years P = population_X_after_17_years) : P = 42000 :=
by
  sorry

end village_Y_initial_population_1_1204


namespace solution_system_inequalities_1_1082

theorem solution_system_inequalities (x : ℝ) : 
  (x - 4 ≤ 0 ∧ 2 * (x + 1) < 3 * x) ↔ (2 < x ∧ x ≤ 4) := 
sorry

end solution_system_inequalities_1_1082


namespace positive_even_integers_less_than_1000_not_divisible_by_3_or_11_1_1893

theorem positive_even_integers_less_than_1000_not_divisible_by_3_or_11 :
  ∃ n : ℕ, n = 108 ∧
    (∀ m : ℕ, 0 < m → 2 ∣ m → m < 1000 → (¬ (3 ∣ m) ∧ ¬ (11 ∣ m) ↔ m ≤ n)) :=
sorry

end positive_even_integers_less_than_1000_not_divisible_by_3_or_11_1_1893


namespace initially_calculated_average_1_1780

open List

theorem initially_calculated_average (numbers : List ℝ) (h_len : numbers.length = 10) 
  (h_wrong_reading : ∃ (n : ℝ), n ∈ numbers ∧ n ≠ 26 ∧ (numbers.erase n).sum + 26 = numbers.sum - 36 + 26) 
  (h_correct_avg : numbers.sum / 10 = 16) : 
  ((numbers.sum - 10) / 10 = 15) := 
sorry

end initially_calculated_average_1_1780


namespace solution_A_1_1705

def P : Set ℕ := {1, 2, 3, 4}

theorem solution_A (A : Set ℕ) (h1 : A ⊆ P) 
  (h2 : ∀ x ∈ A, 2 * x ∉ A) 
  (h3 : ∀ x ∈ (P \ A), 2 * x ∉ (P \ A)): 
    A = {2} ∨ A = {1, 4} ∨ A = {2, 3} ∨ A = {1, 3, 4} :=
sorry

end solution_A_1_1705


namespace move_right_by_three_units_1_1303

theorem move_right_by_three_units :
  (-1 + 3 = 2) :=
  by { sorry }

end move_right_by_three_units_1_1303


namespace range_of_real_number_m_1_1613

open Set

variable {m : ℝ}

theorem range_of_real_number_m (U : Set ℝ) (A : Set ℝ) (B : Set ℝ) (h1 : U = univ) (h2 : A = { x | x < 1 }) (h3 : B = { x | x ≥ m }) (h4 : compl A ⊆ B) : m ≤ 1 := by
  sorry

end range_of_real_number_m_1_1613


namespace no_solution_xyz_1_1472

theorem no_solution_xyz : ∀ (x y z : Nat), (1 ≤ x) → (x ≤ 9) → (0 ≤ y) → (y ≤ 9) → (0 ≤ z) → (z ≤ 9) →
    100 * x + 10 * y + z ≠ 10 * x * y + x * z :=
by
  intros x y z hx1 hx9 hy1 hy9 hz1 hz9
  sorry

end no_solution_xyz_1_1472


namespace find_value_1_1137

variable (a b c : Int)

-- Conditions from the problem
axiom abs_a_eq_two : |a| = 2
axiom b_eq_neg_seven : b = -7
axiom neg_c_eq_neg_five : -c = -5

-- Proof problem
theorem find_value : a^2 + (-b) + (-c) = 6 := by
  sorry

end find_value_1_1137


namespace valid_base6_number_2015_1_1839

def is_valid_base6_digit (d : Nat) : Prop :=
  d = 0 ∨ d = 1 ∨ d = 2 ∨ d = 3 ∨ d = 4 ∨ d = 5

def is_base6_number (n : Nat) : Prop :=
  ∀ (digit : Nat), digit ∈ (n.digits 10) → is_valid_base6_digit digit

theorem valid_base6_number_2015 : is_base6_number 2015 := by
  sorry

end valid_base6_number_2015_1_1839


namespace largest_k_exists_1_1823

noncomputable def largest_k := 3

theorem largest_k_exists :
  ∃ (k : ℕ), (k = largest_k) ∧ ∀ m : ℕ, 
    (∀ n : ℕ, ∃ a b : ℕ, m + n = a^2 + b^2) ∧ 
    (∀ n : ℕ, ∃ seq : ℕ → ℕ,
      (∀ i : ℕ, seq i = a^2 + b^2) ∧
      (∀ j : ℕ, m ≤ j → a^2 + b^2 ≠ 3 + 4 * j)
    ) := ⟨3, rfl, sorry⟩

end largest_k_exists_1_1823


namespace triangle_acute_angle_contradiction_1_1185

theorem triangle_acute_angle_contradiction
  (α β γ : ℝ)
  (h_sum : α + β + γ = 180)
  (h_tri : 0 < α ∧ 0 < β ∧ 0 < γ)
  (h_at_most_one_acute : (α < 90 ∧ β ≥ 90 ∧ γ ≥ 90) 
                         ∨ (α ≥ 90 ∧ β < 90 ∧ γ ≥ 90) 
                         ∨ (α ≥ 90 ∧ β ≥ 90 ∧ γ < 90)) :
  false :=
by
  sorry

end triangle_acute_angle_contradiction_1_1185


namespace water_current_speed_1_1296

-- Definitions based on the conditions
def swimmer_speed : ℝ := 4  -- The swimmer's speed in still water (km/h)
def swim_time : ℝ := 2  -- Time taken to swim against the current (hours)
def swim_distance : ℝ := 6  -- Distance swum against the current (km)

-- The effective speed against the current
noncomputable def effective_speed_against_current (v : ℝ) : ℝ := swimmer_speed - v

-- Lean statement that formalizes proving the speed of the current
theorem water_current_speed (v : ℝ) (h : effective_speed_against_current v = swim_distance / swim_time) : v = 1 :=
by
  sorry

end water_current_speed_1_1296


namespace cos_seventh_eq_sum_of_cos_1_1014

theorem cos_seventh_eq_sum_of_cos:
  ∃ (b₁ b₂ b₃ b₄ b₅ b₆ b₇ : ℝ),
  (∀ θ : ℝ, (Real.cos θ) ^ 7 = b₁ * Real.cos θ + b₂ * Real.cos (2 * θ) + b₃ * Real.cos (3 * θ) + b₄ * Real.cos (4 * θ) + b₅ * Real.cos (5 * θ) + b₆ * Real.cos (6 * θ) + b₇ * Real.cos (7 * θ)) ∧
  (b₁ ^ 2 + b₂ ^ 2 + b₃ ^ 2 + b₄ ^ 2 + b₅ ^ 2 + b₆ ^ 2 + b₇ ^ 2 = 1555 / 4096) :=
sorry

end cos_seventh_eq_sum_of_cos_1_1014


namespace tile_covering_problem_1_1126

theorem tile_covering_problem :
  let tile_length := 5
  let tile_width := 3
  let region_length := 5 * 12  -- converting feet to inches
  let region_width := 3 * 12   -- converting feet to inches
  let tile_area := tile_length * tile_width
  let region_area := region_length * region_width
  region_area / tile_area = 144 := 
by 
  let tile_length := 5
  let tile_width := 3
  let region_length := 5 * 12
  let region_width := 3 * 12
  let tile_area := tile_length * tile_width
  let region_area := region_length * region_width
  sorry

end tile_covering_problem_1_1126


namespace solve_for_x_1_1448

theorem solve_for_x (x : ℝ) (h : (x / 4) / 2 = 4 / (x / 2)) : x = 8 ∨ x = -8 :=
by
  sorry

end solve_for_x_1_1448


namespace right_triangle_height_1_1029

theorem right_triangle_height
  (h : ℕ)
  (base : ℕ)
  (rectangle_area : ℕ)
  (same_area : (1 / 2 : ℚ) * base * h = rectangle_area)
  (base_eq_width : base = 5)
  (rectangle_area_eq : rectangle_area = 45) :
  h = 18 :=
by
  sorry

end right_triangle_height_1_1029


namespace average_score_for_girls_1_1335

variable (A a B b : ℕ)
variable (h1 : 71 * A + 76 * a = 74 * (A + a))
variable (h2 : 81 * B + 90 * b = 84 * (B + b))
variable (h3 : 71 * A + 81 * B = 79 * (A + B))

theorem average_score_for_girls
  (h1 : 71 * A + 76 * a = 74 * (A + a))
  (h2 : 81 * B + 90 * b = 84 * (B + b))
  (h3 : 71 * A + 81 * B = 79 * (A + B))
  : (76 * a + 90 * b) / (a + b) = 84 := by
  sorry

end average_score_for_girls_1_1335


namespace factorize1_factorize2_factorize3_factorize4_1_1357

-- 1. Factorize 3x - 12x^3
theorem factorize1 (x : ℝ) : 3 * x - 12 * x^3 = 3 * x * (1 - 2 * x) * (1 + 2 * x) := 
sorry

-- 2. Factorize 9m^2 - 4n^2
theorem factorize2 (m n : ℝ) : 9 * m^2 - 4 * n^2 = (3 * m + 2 * n) * (3 * m - 2 * n) := 
sorry

-- 3. Factorize a^2(x - y) + b^2(y - x)
theorem factorize3 (a b x y : ℝ) : a^2 * (x - y) + b^2 * (y - x) = (x - y) * (a + b) * (a - b) := 
sorry

-- 4. Factorize x^2 - 4xy + 4y^2 - 1
theorem factorize4 (x y : ℝ) : x^2 - 4 * x * y + 4 * y^2 - 1 = (x - y + 1) * (x - y - 1) := 
sorry

end factorize1_factorize2_factorize3_factorize4_1_1357


namespace number_of_small_jars_1_1512

theorem number_of_small_jars (S L : ℕ) (h1 : S + L = 100) (h2 : 3 * S + 5 * L = 376) : S = 62 := 
sorry

end number_of_small_jars_1_1512


namespace coordinates_of_P_1_1364

theorem coordinates_of_P (A B : ℝ × ℝ × ℝ) (m : ℝ) :
  A = (1, 0, 2) ∧ B = (1, -3, 1) ∧ (0, 0, m) = (0, 0, -3) :=
by 
  sorry

end coordinates_of_P_1_1364


namespace product_of_two_numbers_1_1050

theorem product_of_two_numbers 
  (x y : ℝ) 
  (h1 : x - y = 2) 
  (h2 : x + y = 8 * (x - y)) 
  (h3 : x * y = 40 * (x - y)) 
  : x * y = 63 := 
by 
  sorry

end product_of_two_numbers_1_1050


namespace total_bricks_calculation_1_1849

def bricks_in_row : Nat := 30
def rows_in_wall : Nat := 50
def number_of_walls : Nat := 2
def total_bricks_for_both_walls : Nat := 3000

theorem total_bricks_calculation (h1 : bricks_in_row = 30) 
                                      (h2 : rows_in_wall = 50) 
                                      (h3 : number_of_walls = 2) : 
                                      bricks_in_row * rows_in_wall * number_of_walls = total_bricks_for_both_walls :=
by
  sorry

end total_bricks_calculation_1_1849


namespace volunteer_org_percentage_change_1_1260

theorem volunteer_org_percentage_change 
  (initial_membership : ℝ)
  (fall_increase_rate : ℝ)
  (spring_decrease_rate : ℝ) :
  (initial_membership = 100) →
  (fall_increase_rate = 0.05) →
  (spring_decrease_rate = 0.19) →
  (14.95 : ℝ) =
  ((initial_membership * (1 + fall_increase_rate)) * (1 - spring_decrease_rate)
  - initial_membership) / initial_membership * 100 := by
  sorry

end volunteer_org_percentage_change_1_1260


namespace factorization_correct_1_1675

theorem factorization_correct (x : ℝ) : 
  (hxA : x^2 + 2*x + 1 ≠ x*(x + 2) + 1) → 
  (hxB : x^2 + 2*x + 1 ≠ (x + 1)*(x - 1)) → 
  (hxC : x^2 + x ≠ (x + 1/2)^2 - 1/4) →
  x^2 + x = x * (x + 1) := 
by sorry

end factorization_correct_1_1675


namespace question_true_1_1308
noncomputable def a := (1/2) * Real.cos (7 * Real.pi / 180) - (Real.sqrt 3 / 2) * Real.sin (7 * Real.pi / 180)
noncomputable def b := (2 * Real.tan (12 * Real.pi / 180)) / (1 + Real.tan (12 * Real.pi / 180)^2)
noncomputable def c := Real.sqrt ((1 - Real.cos (44 * Real.pi / 180)) / 2)

theorem question_true :
  b > a ∧ a > c :=
by
  sorry

end question_true_1_1308


namespace Eric_test_score_1_1167

theorem Eric_test_score (n : ℕ) (old_avg new_avg : ℚ) (Eric_score : ℚ) :
  n = 22 →
  old_avg = 84 →
  new_avg = 85 →
  Eric_score = (n * new_avg) - ((n - 1) * old_avg) →
  Eric_score = 106 :=
by
  intros h1 h2 h3 h4
  sorry

end Eric_test_score_1_1167


namespace units_digit_17_times_29_1_1945

theorem units_digit_17_times_29 :
  (17 * 29) % 10 = 3 :=
by
  sorry

end units_digit_17_times_29_1_1945


namespace root_in_interval_sum_eq_three_1_1462

noncomputable def f (x : ℝ) : ℝ := 2^x + x - 5

theorem root_in_interval_sum_eq_three {a b : ℤ} (h1 : b - a = 1) (h2 : ∃ x : ℝ, a < x ∧ x < b ∧ f x = 0) :
  a + b = 3 :=
by
  sorry

end root_in_interval_sum_eq_three_1_1462


namespace number_of_real_b_1_1006

noncomputable def count_integer_roots_of_quadratic_eq_b : ℕ :=
  let pairs := [(1, 64), (2, 32), (4, 16), (8, 8), (-1, -64), (-2, -32), (-4, -16), (-8, -8)]
  pairs.length

theorem number_of_real_b : count_integer_roots_of_quadratic_eq_b = 8 :=
by {
  -- sorry is used to skip the proof
  sorry
}

end number_of_real_b_1_1006


namespace inequality_a3_b3_c3_1_1258

theorem inequality_a3_b3_c3 (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) : 
  a^3 + b^3 + c^3 ≥ (1/3) * (a^2 + b^2 + c^2) * (a + b + c) := 
by 
  sorry

end inequality_a3_b3_c3_1_1258


namespace find_length_of_rod_1_1040

-- Constants representing the given conditions
def weight_6m_rod : ℝ := 6.1
def length_6m_rod : ℝ := 6
def weight_unknown_rod : ℝ := 12.2

-- Proof statement ensuring the length of the rod that weighs 12.2 kg is 12 meters
theorem find_length_of_rod (L : ℝ) (h : weight_6m_rod / length_6m_rod = weight_unknown_rod / L) : 
  L = 12 := by
  sorry

end find_length_of_rod_1_1040


namespace square_distance_between_intersections_1_1022

-- Definitions of the circles
def circle1 (x y : ℝ) : Prop := (x - 2)^2 + y^2 = 25
def circle2 (x y : ℝ) : Prop := (x - 7)^2 + y^2 = 4

-- Problem: Prove the square of the distance between intersection points P and Q
theorem square_distance_between_intersections :
  (∃ (x y1 y2 : ℝ), circle1 x y1 ∧ circle2 x y1 ∧ circle1 x y2 ∧ circle2 x y2 ∧ y1 ≠ y2) →
  ∃ d : ℝ, d^2 = 15.3664 :=
by
  sorry

end square_distance_between_intersections_1_1022


namespace problem_statement_1_1581

theorem problem_statement : (-1:ℤ) ^ 4 - (2 - (-3:ℤ) ^ 2) = 6 := by
  sorry  -- Proof will be provided separately

end problem_statement_1_1581


namespace decreasing_exponential_range_1_1658

theorem decreasing_exponential_range {a : ℝ} (h : ∀ x y : ℝ, x < y → (a + 1)^x > (a + 1)^y) : -1 < a ∧ a < 0 :=
sorry

end decreasing_exponential_range_1_1658


namespace feet_more_than_heads_1_1818

def num_hens := 50
def num_goats := 45
def num_camels := 8
def num_keepers := 15

def feet_per_hen := 2
def feet_per_goat := 4
def feet_per_camel := 4
def feet_per_keeper := 2

def total_heads := num_hens + num_goats + num_camels + num_keepers
def total_feet := (num_hens * feet_per_hen) + (num_goats * feet_per_goat) + (num_camels * feet_per_camel) + (num_keepers * feet_per_keeper)

-- Theorem to prove:
theorem feet_more_than_heads : total_feet - total_heads = 224 := by
  -- proof goes here
  sorry

end feet_more_than_heads_1_1818


namespace problem_a_problem_d_1_1003

theorem problem_a (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + b = 4) : (1 / (a * b)) ≥ 1 / 4 :=
by
  sorry

theorem problem_d (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + b = 4) : a^2 + b^2 ≥ 8 :=
by
  sorry

end problem_a_problem_d_1_1003


namespace part_I_part_II_1_1554

-- Define the conditions given in the problem
def set_A : Set ℝ := { x | -1 < x ∧ x < 3 }
def set_B (a b : ℝ) : Set ℝ := { x | x^2 - a * x + b < 0 }

-- Part I: Prove that if A = B, then a = 2 and b = -3
theorem part_I (a b : ℝ) (h : set_A = set_B a b) : a = 2 ∧ b = -3 :=
sorry

-- Part II: Prove that if b = 3 and A ∩ B ⊇ B, then the range of a is [-2√3, 4]
theorem part_II (a : ℝ) (b : ℝ := 3) (h : set_A ∩ set_B a b ⊇ set_B a b) : -2 * Real.sqrt 3 ≤ a ∧ a ≤ 4 :=
sorry

end part_I_part_II_1_1554


namespace marble_solid_color_percentage_1_1582

theorem marble_solid_color_percentage (a b : ℕ) (h1 : a = 5) (h2 : b = 85) : a + b = 90 := 
by
  sorry

end marble_solid_color_percentage_1_1582


namespace number_of_adults_1_1927

-- Given constants
def children : ℕ := 200
def price_child (price_adult : ℕ) : ℕ := price_adult / 2
def total_amount : ℕ := 16000

-- Based on the problem conditions
def price_adult := 32

-- The generated proof problem
theorem number_of_adults 
    (price_adult_gt_0 : price_adult > 0)
    (h_price_adult : price_adult = 32)
    (h_total_amount : total_amount = 16000) 
    (h_price_relation : ∀ price_adult, price_adult / 2 * 2 = price_adult) :
  ∃ A : ℕ, 32 * A + 16 * 200 = 16000 ∧ price_child price_adult = 16 := by
  sorry

end number_of_adults_1_1927


namespace airplane_average_speed_1_1127

-- Define the conditions
def miles_to_kilometers (miles : ℕ) : ℝ :=
  miles * 1.60934

def distance_miles : ℕ := 1584
def time_hours : ℕ := 24

-- Define the problem to prove
theorem airplane_average_speed : 
  (miles_to_kilometers distance_miles) / (time_hours : ℝ) = 106.24 :=
by
  sorry

end airplane_average_speed_1_1127


namespace total_license_groups_1_1083

-- Defining the given conditions
def letter_choices : Nat := 3
def digit_choices_per_slot : Nat := 10
def number_of_digit_slots : Nat := 5

-- Statement to prove that the total number of different license groups is 300000
theorem total_license_groups : letter_choices * (digit_choices_per_slot ^ number_of_digit_slots) = 300000 := by
  sorry

end total_license_groups_1_1083


namespace original_average_1_1533

theorem original_average (n : ℕ) (A : ℝ) (new_avg : ℝ) 
  (h1 : n = 25) 
  (h2 : new_avg = 140) 
  (h3 : 2 * A = new_avg) : A = 70 :=
sorry

end original_average_1_1533


namespace non_similar_triangles_with_arithmetic_angles_1_1302

theorem non_similar_triangles_with_arithmetic_angles : 
  ∃! (d : ℕ), d > 0 ∧ d ≤ 50 := 
sorry

end non_similar_triangles_with_arithmetic_angles_1_1302


namespace find_theta_in_interval_1_1120

variable (θ : ℝ)

def angle_condition (θ : ℝ) (x : ℝ) : Prop :=
  0 ≤ x ∧ x ≤ 1 ∧ (x^3 * Real.cos θ - x * (1 - x) + (1 - x)^3 * Real.tan θ > 0)

theorem find_theta_in_interval :
  (∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → angle_condition θ x) →
  0 < θ ∧ θ < Real.pi / 2 :=
by
  sorry

end find_theta_in_interval_1_1120


namespace sum_f_div_2009_equals_1005_1_1534

def f (x : ℚ) : ℚ := x^5 / (5*x^4 - 10*x^3 + 10*x^2 - 5*x + 1)

theorem sum_f_div_2009_equals_1005 :
  (∑ i in Finset.range (2009+1).succ, f (i / 2009)) = 1005 :=
sorry

end sum_f_div_2009_equals_1005_1_1534


namespace max_a_avoiding_lattice_points_1_1921

def is_lattice_point (x y : ℤ) : Prop :=
  true  -- Placeholder for (x, y) being in lattice points.

def passes_through_lattice_point (m : ℚ) (x : ℤ) : Prop :=
  is_lattice_point x (⌊m * x + 2⌋)

theorem max_a_avoiding_lattice_points :
  ∀ {a : ℚ}, (∀ x : ℤ, (0 < x ∧ x ≤ 100) → ¬passes_through_lattice_point ((1 : ℚ) / 2) x ∧ ¬passes_through_lattice_point (a - 1) x) →
  a = 50 / 99 :=
by
  sorry

end max_a_avoiding_lattice_points_1_1921


namespace gardener_cabbages_this_year_1_1747

-- Definitions for the conditions
def side_length_last_year (x : ℕ) := true
def area_last_year (x : ℕ) := x * x
def increase_in_output := 197

-- Proposition to prove the number of cabbages this year
theorem gardener_cabbages_this_year (x : ℕ) (hx : side_length_last_year x) : 
  (area_last_year x + increase_in_output) = 9801 :=
by 
  sorry

end gardener_cabbages_this_year_1_1747


namespace q_zero_1_1474

noncomputable def q (x : ℝ) : ℝ := sorry -- Definition of the polynomial q(x) is required here.

theorem q_zero : 
  (∀ n : ℕ, n ≤ 7 → q (3^n) = 1 / 3^n) →
  q 0 = 0 :=
by 
  sorry

end q_zero_1_1474


namespace necessary_and_sufficient_condition_1_1416

-- Define the first circle
def circle1 (m : ℝ) : Set (ℝ × ℝ) :=
  { p | (p.1 + m)^2 + p.2^2 = 1 }

-- Define the second circle
def circle2 : Set (ℝ × ℝ) :=
  { p | (p.1 - 2)^2 + p.2^2 = 4 }

-- Define the condition -1 ≤ m ≤ 1
def condition (m : ℝ) : Prop :=
  -1 ≤ m ∧ m ≤ 1

-- Define the property for circles having common points
def circlesHaveCommonPoints (m : ℝ) : Prop :=
  ∃ p : ℝ × ℝ, p ∈ circle1 m ∧ p ∈ circle2

-- The final statement
theorem necessary_and_sufficient_condition (m : ℝ) :
  condition m → circlesHaveCommonPoints m ↔ (-5 ≤ m ∧ m ≤ 1) :=
by
  sorry

end necessary_and_sufficient_condition_1_1416


namespace square_units_digit_1_1914

theorem square_units_digit (n : ℕ) (h : (n^2 / 10) % 10 = 7) : n^2 % 10 = 6 := 
sorry

end square_units_digit_1_1914


namespace simplify_and_compute_1_1044

theorem simplify_and_compute (x : ℝ) (h : x = 4) : (x^8 - 32*x^4 + 256) / (x^4 - 16) = 240 := by
  sorry

end simplify_and_compute_1_1044


namespace sum_largest_and_smallest_1_1487

-- Define the three-digit number properties
def hundreds_digit := 4
def tens_digit := 8
def A : ℕ := sorry  -- Placeholder for the digit A

-- Define the number based on the digits
def number (A : ℕ) : ℕ := 100 * hundreds_digit + 10 * tens_digit + A

-- Hypotheses
axiom A_range : 0 ≤ A ∧ A ≤ 9

-- Largest and smallest possible numbers
def largest_number := number 9
def smallest_number := number 0

-- Prove the sum
theorem sum_largest_and_smallest : largest_number + smallest_number = 969 :=
by
  sorry

end sum_largest_and_smallest_1_1487


namespace algebraic_expression_value_1_1717

theorem algebraic_expression_value (m : ℝ) (h : m^2 - m - 2 = 0) : 2 * m^2 - 2 * m = 4 := by
  sorry

end algebraic_expression_value_1_1717


namespace remainder_when_xyz_divided_by_9_is_0_1_1703

theorem remainder_when_xyz_divided_by_9_is_0
  (x y z : ℕ)
  (hx : x < 9)
  (hy : y < 9)
  (hz : z < 9)
  (h1 : x + 3 * y + 2 * z ≡ 0 [MOD 9])
  (h2 : 3 * x + 2 * y + z ≡ 5 [MOD 9])
  (h3 : 2 * x + y + 3 * z ≡ 5 [MOD 9]) :
  (x * y * z) % 9 = 0 := by
  sorry

end remainder_when_xyz_divided_by_9_is_0_1_1703


namespace john_total_amount_1_1972

def grandpa_amount : ℕ := 30
def grandma_amount : ℕ := 3 * grandpa_amount
def aunt_amount : ℕ := 3 / 2 * grandpa_amount
def uncle_amount : ℕ := 2 / 3 * grandma_amount

def total_amount : ℕ :=
  grandpa_amount + grandma_amount + aunt_amount + uncle_amount

theorem john_total_amount : total_amount = 225 := by sorry

end john_total_amount_1_1972


namespace slope_of_perpendicular_line_1_1170

theorem slope_of_perpendicular_line 
  (x1 y1 x2 y2 : ℤ)
  (h : x1 = 3 ∧ y1 = -4 ∧ x2 = -6 ∧ y2 = 2) : 
∃ m : ℚ, m = 3/2 :=
by
  sorry

end slope_of_perpendicular_line_1_1170


namespace six_letter_words_count_1_1719

def first_letter_possibilities := 26
def second_letter_possibilities := 26
def third_letter_possibilities := 26
def fourth_letter_possibilities := 26

def number_of_six_letter_words : Nat := 
  first_letter_possibilities * 
  second_letter_possibilities * 
  third_letter_possibilities * 
  fourth_letter_possibilities

theorem six_letter_words_count : number_of_six_letter_words = 456976 := by
  sorry

end six_letter_words_count_1_1719


namespace find_c_1_1101

noncomputable def f (x c : ℝ) := x * (x - c) ^ 2
noncomputable def f' (x c : ℝ) := 3 * x ^ 2 - 4 * c * x + c ^ 2
noncomputable def f'' (x c : ℝ) := 6 * x - 4 * c

theorem find_c (c : ℝ) : f' 2 c = 0 ∧ f'' 2 c < 0 → c = 6 :=
by {
  sorry
}

end find_c_1_1101


namespace banana_price_1_1860

theorem banana_price (x y : ℕ) (b : ℕ) 
  (hx : x + y = 4) 
  (cost_eq : 50 * x + 60 * y + b = 275) 
  (banana_cheaper_than_pear : b < 60) 
  : b = 35 ∨ b = 45 ∨ b = 55 :=
by
  sorry

end banana_price_1_1860


namespace time_spent_on_type_a_problems_1_1169

-- Define the conditions
def total_questions := 200
def examination_duration_hours := 3
def type_a_problems := 100
def type_b_problems := total_questions - type_a_problems
def type_a_time_coeff := 2

-- Convert examination duration to minutes
def examination_duration_minutes := examination_duration_hours * 60

-- Variables for time per problem
variable (x : ℝ)

-- The total time spent
def total_time_spent : ℝ := type_a_problems * (type_a_time_coeff * x) + type_b_problems * x

-- Statement we need to prove
theorem time_spent_on_type_a_problems :
  total_time_spent x = examination_duration_minutes → type_a_problems * (type_a_time_coeff * x) = 120 :=
by
  sorry

end time_spent_on_type_a_problems_1_1169


namespace students_in_second_class_1_1862

variable (x : ℕ)

theorem students_in_second_class :
  (∃ x, 30 * 40 + 70 * x = (30 + x) * 58.75) → x = 50 :=
by
  sorry

end students_in_second_class_1_1862


namespace largest_binomial_coefficient_term_largest_coefficient_term_remainder_mod_7_1_1230

-- Definitions and conditions
def binomial_expansion (x : ℕ) (n : ℕ) : ℕ := (x + (3 * x^2))^n

-- Problem statements
theorem largest_binomial_coefficient_term :
  ∀ x n,
  (x + 3 * x^2)^n = (x + 3 * x^2)^7 →
  (2^n = 128) →
  ∃ t : ℕ, t = 2835 * x^11 := 
by sorry

theorem largest_coefficient_term :
  ∀ x n,
  (x + 3 * x^2)^n = (x + 3 * x^2)^7 →
  exists t, t = 5103 * x^13 :=
by sorry

theorem remainder_mod_7 :
  ∀ x n,
  x = 3 →
  n = 2016 →
  (x + (3 * x^2))^n % 7 = 1 :=
by sorry

end largest_binomial_coefficient_term_largest_coefficient_term_remainder_mod_7_1_1230


namespace rahul_share_is_100_1_1981

-- Definitions of the conditions
def rahul_rate := 1/3
def rajesh_rate := 1/2
def total_payment := 250

-- Definition of their work rate when they work together
def combined_rate := rahul_rate + rajesh_rate

-- Definition of the total value of the work done in one day when both work together
noncomputable def combined_work_value := total_payment / combined_rate

-- Definition of Rahul's share for the work done in one day
noncomputable def rahul_share := rahul_rate * combined_work_value

-- The theorem we need to prove
theorem rahul_share_is_100 : rahul_share = 100 := by
  sorry

end rahul_share_is_100_1_1981


namespace divisor_is_twelve_1_1540

theorem divisor_is_twelve (d : ℕ) (h : 64 = 5 * d + 4) : d = 12 := 
sorry

end divisor_is_twelve_1_1540


namespace bricks_in_chimney_900_1_1420

theorem bricks_in_chimney_900 (h : ℕ) :
  let Brenda_rate := h / 9
  let Brandon_rate := h / 10
  let combined_rate := (Brenda_rate + Brandon_rate) - 10
  5 * combined_rate = h → h = 900 :=
by
  intros Brenda_rate Brandon_rate combined_rate
  sorry

end bricks_in_chimney_900_1_1420


namespace sum_of_squares_of_reciprocals_1_1232

-- Definitions based on the problem's conditions
variables (a b : ℝ) (hab : a + b = 3 * a * b + 1) (h_an : a ≠ 0) (h_bn : b ≠ 0)

-- Statement of the problem to be proved
theorem sum_of_squares_of_reciprocals :
  (1 / a^2) + (1 / b^2) = (4 * a * b + 10) / (a^2 * b^2) :=
sorry

end sum_of_squares_of_reciprocals_1_1232


namespace find_c_1_1349

theorem find_c
  (m b d c : ℝ)
  (h : m = b * d * c / (d + c)) :
  c = m * d / (b * d - m) :=
sorry

end find_c_1_1349


namespace find_angle_A_1_1504

-- Conditions
def is_triangle (A B C : ℝ) : Prop := A + B + C = 180
def B_is_two_C (B C : ℝ) : Prop := B = 2 * C
def B_is_80 (B : ℝ) : Prop := B = 80

-- Theorem statement
theorem find_angle_A (A B C : ℝ) (h₁ : is_triangle A B C) (h₂ : B_is_two_C B C) (h₃ : B_is_80 B) : A = 60 := by
  sorry

end find_angle_A_1_1504


namespace grid_covering_impossible_1_1726

theorem grid_covering_impossible :
  ∀ (x y : ℕ), x + y = 19 → 6 * x + 7 * y = 132 → False :=
by
  intros x y h1 h2
  -- Proof would go here.
  sorry

end grid_covering_impossible_1_1726


namespace bead_necklaces_sold_1_1279

def cost_per_necklace : ℕ := 7
def total_earnings : ℕ := 70
def gemstone_necklaces_sold : ℕ := 7

theorem bead_necklaces_sold (B : ℕ) 
  (h1 : total_earnings = cost_per_necklace * (B + gemstone_necklaces_sold))  :
  B = 3 :=
by {
  sorry
}

end bead_necklaces_sold_1_1279


namespace triangle_side_s_1_1473

/-- The sides of a triangle have lengths 8, 13, and s where s is a whole number.
    What is the smallest possible value of s?
    We need to show that the minimum possible value of s such that 8 + s > 13,
    s < 21, and 13 + s > 8 is s = 6. -/
theorem triangle_side_s (s : ℕ) : 
  (8 + s > 13) ∧ (8 + 13 > s) ∧ (13 + s > 8) → s = 6 :=
by
  sorry

end triangle_side_s_1_1473


namespace prob_draw_1_1284

-- Define the probabilities as constants
def prob_A_winning : ℝ := 0.4
def prob_A_not_losing : ℝ := 0.9

-- Prove that the probability of a draw is 0.5
theorem prob_draw : prob_A_not_losing - prob_A_winning = 0.5 :=
by sorry

end prob_draw_1_1284


namespace neg_distance_represents_west_1_1873

def represents_east (distance : Int) : Prop :=
  distance > 0

def represents_west (distance : Int) : Prop :=
  distance < 0

theorem neg_distance_represents_west (pos_neg : represents_east 30) :
  represents_west (-50) :=
by
  sorry

end neg_distance_represents_west_1_1873


namespace dan_present_age_1_1884

-- Let x be Dan's present age
variable (x : ℤ)

-- Condition: Dan's age after 18 years will be 8 times his age 3 years ago
def condition (x : ℤ) : Prop :=
  x + 18 = 8 * (x - 3)

-- The goal is to prove that Dan's present age is 6
theorem dan_present_age (x : ℤ) (h : condition x) : x = 6 :=
by
  sorry

end dan_present_age_1_1884


namespace remainder_when_7x_div_9_1_1709

theorem remainder_when_7x_div_9 (x : ℕ) (h : x % 9 = 5) : (7 * x) % 9 = 8 :=
sorry

end remainder_when_7x_div_9_1_1709


namespace intercepts_1_1547

def line_equation (x y : ℝ) : Prop :=
  5 * x + 3 * y - 15 = 0

theorem intercepts (a b : ℝ) : line_equation a 0 ∧ line_equation 0 b → (a = 3 ∧ b = 5) :=
  sorry

end intercepts_1_1547


namespace muscovy_more_than_cayuga_1_1417

theorem muscovy_more_than_cayuga
  (M C K : ℕ)
  (h1 : M + C + K = 90)
  (h2 : M = 39)
  (h3 : M = 2 * C + 3 + C) :
  M - C = 27 := by
  sorry

end muscovy_more_than_cayuga_1_1417


namespace scientific_notation_correct_1_1451

noncomputable def scientific_notation_139000 : Prop :=
  139000 = 1.39 * 10^5

theorem scientific_notation_correct : scientific_notation_139000 :=
by
  -- The proof would be included here, but we add sorry to skip it
  sorry

end scientific_notation_correct_1_1451


namespace compute_100p_plus_q_1_1888

-- Given constants p, q under the provided conditions,
-- prove the result: 100p + q = 430 / 3.
theorem compute_100p_plus_q (p q : ℚ) 
  (h1 : ∀ x : ℚ, (x + p) * (x + q) * (x + 20) = 0 → x ≠ -4)
  (h2 : ∀ x : ℚ, (x + 3 * p) * (x + 4) * (x + 10) = 0 → (x = -4 ∨ x ≠ -4)) :
  100 * p + q = 430 / 3 := 
by 
  sorry

end compute_100p_plus_q_1_1888


namespace part_I_part_II_1_1455

noncomputable def A : Set ℝ := {x | 2*x^2 - 5*x - 3 <= 0}
noncomputable def B (a : ℝ) : Set ℝ := {x | (x - (2*a + 1)) * (x - (a - 1)) < 0}

theorem part_I :
  (A ∪ B 0 = {x : ℝ | -1 < x ∧ x ≤ 3}) :=
by sorry

theorem part_II (a : ℝ) :
  (A ∩ B a = ∅) →
  (a ≤ -3/4 ∨ a ≥ 4) ∧ a ≠ -2 :=
by sorry


end part_I_part_II_1_1455


namespace routes_from_A_to_B_in_4_by_3_grid_1_1367

-- Problem: Given a 4 by 3 rectangular grid, and movement allowing only right (R) or down (D),
-- prove that the number of different routes from point A to point B is 35.
def routes_4_by_3 : ℕ :=
  let n_moves := 3 + 4  -- Total moves required are 3 Rs and 4 Ds
  let r_moves := 3      -- Number of Right moves (R)
  Nat.choose (n_moves) (r_moves) -- Number of ways to choose 3 Rs from 7 moves

theorem routes_from_A_to_B_in_4_by_3_grid : routes_4_by_3 = 35 := by {
  sorry -- Proof omitted
}

end routes_from_A_to_B_in_4_by_3_grid_1_1367


namespace sum_of_first_41_terms_is_94_1_1315

def equal_product_sequence (a : ℕ → ℕ) (k : ℕ) : Prop := 
∀ (n : ℕ), a (n+1) * a (n+2) * a (n+3) = k

theorem sum_of_first_41_terms_is_94
  (a : ℕ → ℕ)
  (h1 : equal_product_sequence a 8)
  (h2 : a 1 = 1)
  (h3 : a 2 = 2) :
  (Finset.range 41).sum a = 94 :=
by
  sorry

end sum_of_first_41_terms_is_94_1_1315


namespace relationship_abc_1_1941

noncomputable def a (x : ℝ) : ℝ := Real.log x
noncomputable def b (x : ℝ) : ℝ := Real.exp (Real.log x)
noncomputable def c (x : ℝ) : ℝ := Real.exp (Real.log (1 / x))

theorem relationship_abc (x : ℝ) (h : (1 / Real.exp 1) < x ∧ x < 1) : a x < b x ∧ b x < c x :=
by
  have ha : a x = Real.log x := rfl
  have hb : b x = Real.exp (Real.log x) := rfl
  have hc : c x = Real.exp (Real.log (1 / x)) := rfl
  sorry

end relationship_abc_1_1941


namespace largest_number_is_correct_1_1866

theorem largest_number_is_correct (a b c : ℝ) (h1 : a + b + c = 100) (h2 : c - b = 10) (h3 : b - a = 3) : c = 33.25 :=
by
  sorry

end largest_number_is_correct_1_1866


namespace compute_fg_1_1398

def f (x : ℤ) : ℤ := x * x
def g (x : ℤ) : ℤ := 3 * x + 4

theorem compute_fg : f (g (-3)) = 25 := by
  sorry

end compute_fg_1_1398


namespace max_eq_zero_max_two_solutions_1_1026

theorem max_eq_zero_max_two_solutions {a b : Fin 10 → ℝ}
  (h : ∀ i, a i ≠ 0) : 
  ∃ (solution_count : ℕ), solution_count <= 2 ∧
  ∃ (solutions : Fin solution_count → ℝ), 
    ∀ (x : ℝ), (∀ i, max (a i * x + b i) = 0) ↔ ∃ j, x = solutions j := sorry

end max_eq_zero_max_two_solutions_1_1026


namespace feb_03_2013_nine_day_1_1379

-- Definitions of the main dates involved
def dec_21_2012 : Nat := 0  -- Assuming day 0 is Dec 21, 2012
def feb_03_2013 : Nat := 45  -- 45 days after Dec 21, 2012

-- Definition to determine the Nine-day period
def nine_day_period (x : Nat) : (Nat × Nat) :=
  let q := x / 9
  let r := x % 9
  (q + 1, r + 1)

-- Theorem we want to prove
theorem feb_03_2013_nine_day : nine_day_period feb_03_2013 = (5, 9) :=
by
  sorry

end feb_03_2013_nine_day_1_1379


namespace lowest_price_for_16_oz_butter_1_1877

-- Define the constants
def price_single_16_oz_package : ℝ := 7
def price_8_oz_package : ℝ := 4
def price_4_oz_package : ℝ := 2
def discount_4_oz_package : ℝ := 0.5

-- Calculate the discounted price for a 4 oz package
def discounted_price_4_oz_package : ℝ := price_4_oz_package * discount_4_oz_package

-- Calculate the total price for two discounted 4 oz packages
def total_price_two_discounted_4_oz_packages : ℝ := 2 * discounted_price_4_oz_package

-- Calculate the total price using the 8 oz package and two discounted 4 oz packages
def total_price_using_coupon : ℝ := price_8_oz_package + total_price_two_discounted_4_oz_packages

-- State the property to prove
theorem lowest_price_for_16_oz_butter :
  min price_single_16_oz_package total_price_using_coupon = 6 :=
sorry

end lowest_price_for_16_oz_butter_1_1877


namespace smallest_value_4x_plus_3y_1_1669

-- Define the condition as a predicate
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 = 18 * x + 8 * y + 10

-- Prove the smallest possible value of 4x + 3y given the condition
theorem smallest_value_4x_plus_3y : ∃ x y : ℝ, circle_eq x y ∧ (4 * x + 3 * y = -40) :=
by
  -- Placeholder for the proof
  sorry

end smallest_value_4x_plus_3y_1_1669


namespace amount_made_per_jersey_1_1712

-- Definitions based on conditions
def total_revenue_from_jerseys : ℕ := 25740
def number_of_jerseys_sold : ℕ := 156

-- Theorem statement
theorem amount_made_per_jersey : 
  total_revenue_from_jerseys / number_of_jerseys_sold = 165 := 
by
  sorry

end amount_made_per_jersey_1_1712


namespace fraction_simplest_sum_1_1759

theorem fraction_simplest_sum (a b : ℕ) (ha : a > 0) (hb : b > 0) 
  (h : (3975 : ℚ) / 10000 = (a : ℚ) / b) 
  (simp : ∀ (c : ℕ), c ∣ a ∧ c ∣ b → c = 1) : a + b = 559 :=
sorry

end fraction_simplest_sum_1_1759


namespace shaded_areas_total_1_1096

theorem shaded_areas_total (r R : ℝ) (h_divides : ∀ (A : ℝ), ∃ (B : ℝ), B = A / 3)
  (h_center : True) (h_area : π * R^2 = 81 * π) :
  (π * R^2 / 3) + (π * (R / 2)^2 / 3) = 33.75 * π :=
by
  -- The proof here will be added.
  sorry

end shaded_areas_total_1_1096


namespace steve_halfway_time_longer_1_1282

theorem steve_halfway_time_longer :
  ∀ (Td: ℝ) (Ts: ℝ),
  Td = 33 →
  Ts = 2 * Td →
  (Ts / 2) - (Td / 2) = 16.5 :=
by
  intros Td Ts hTd hTs
  rw [hTd, hTs]
  sorry

end steve_halfway_time_longer_1_1282


namespace seth_oranges_1_1309

def initial_boxes := 9
def boxes_given_to_mother := 1

def remaining_boxes_after_giving_to_mother := initial_boxes - boxes_given_to_mother
def boxes_given_away := remaining_boxes_after_giving_to_mother / 2
def boxes_left := remaining_boxes_after_giving_to_mother - boxes_given_away

theorem seth_oranges : boxes_left = 4 := by
  sorry

end seth_oranges_1_1309


namespace cups_of_flour_put_in_1_1328

-- Conditions
def recipeSugar : ℕ := 3
def recipeFlour : ℕ := 10
def neededMoreFlourThanSugar : ℕ := 5

-- Question: How many cups of flour did she put in?
-- Answer: 5 cups of flour
theorem cups_of_flour_put_in : (recipeSugar + neededMoreFlourThanSugar = recipeFlour) → recipeFlour - neededMoreFlourThanSugar = 5 := 
by
  intros h
  sorry

end cups_of_flour_put_in_1_1328


namespace integral_value_1_1434

theorem integral_value : ∫ x in (1:ℝ)..(2:ℝ), (x^2 + 1) / x = (3 / 2) + Real.log 2 :=
by sorry

end integral_value_1_1434


namespace susan_added_oranges_1_1226

-- Conditions as definitions
def initial_oranges_in_box : ℝ := 55.0
def final_oranges_in_box : ℝ := 90.0

-- Define the quantity of oranges Susan put into the box
def susan_oranges := final_oranges_in_box - initial_oranges_in_box

-- Theorem statement to prove that the number of oranges Susan put into the box is 35.0
theorem susan_added_oranges : susan_oranges = 35.0 := by
  unfold susan_oranges
  sorry

end susan_added_oranges_1_1226


namespace union_complement_set_1_1803

open Set

variable (U : Set ℕ) (M : Set ℕ) (N : Set ℕ)

def complement_in_U (U N : Set ℕ) : Set ℕ :=
  U \ N

theorem union_complement_set :
  U = {0, 1, 2, 4, 6, 8} →
  M = {0, 4, 6} →
  N = {0, 1, 6} →
  M ∪ (complement_in_U U N) = {0, 2, 4, 6, 8} :=
by
  intros
  rw [complement_in_U, union_comm]
  sorry

end union_complement_set_1_1803


namespace range_of_a_1_1075

variable (x a : ℝ)
def inequality_sys := x < a ∧ x < 3
def solution_set := x < a

theorem range_of_a (h : ∀ x, inequality_sys x a → solution_set x a) : a ≤ 3 := by
  sorry

end range_of_a_1_1075


namespace average_speed_of_car_1_1865

/-- The car's average speed given it travels 65 km in the first hour and 45 km in the second hour. -/
theorem average_speed_of_car (d1 d2 : ℕ) (t : ℕ) (h1 : d1 = 65) (h2 : d2 = 45) (h3 : t = 2) :
  (d1 + d2) / t = 55 :=
by
  sorry

end average_speed_of_car_1_1865


namespace train_length_proof_1_1491

-- Define the conditions
def time_to_cross := 12 -- Time in seconds
def speed_km_per_h := 75 -- Speed in km/h

-- Convert the speed to m/s
def speed_m_per_s := speed_km_per_h * (5 / 18 : ℚ)

-- The length of the train using the formula: length = speed * time
def length_of_train := speed_m_per_s * (time_to_cross : ℚ)

-- The theorem to prove
theorem train_length_proof : length_of_train = 250 := by
  sorry

end train_length_proof_1_1491


namespace min_a_b_1_1314

theorem min_a_b : 
  (∀ x : ℝ, 3 * a * (Real.sin x + Real.cos x) + 2 * b * Real.sin (2 * x) ≤ 3) →
  a + b = -2 →
  a = -4 / 5 :=
by
  sorry

end min_a_b_1_1314


namespace ceil_sum_sqrt_eval_1_1529

theorem ceil_sum_sqrt_eval : 
  (⌈Real.sqrt 2⌉ + ⌈Real.sqrt 22⌉ + ⌈Real.sqrt 222⌉) = 22 := 
by
  sorry

end ceil_sum_sqrt_eval_1_1529


namespace ellipse_with_foci_on_x_axis_1_1679

theorem ellipse_with_foci_on_x_axis (a : ℝ) :
  (∀ x y : ℝ, (x^2) / (a - 5) + (y^2) / 2 = 1 →  
   (∃ cx cy : ℝ, ∀ x', cx - x' = a - 5 ∧ cy = 2)) → 
  a > 7 :=
by sorry

end ellipse_with_foci_on_x_axis_1_1679


namespace number_of_white_balls_1_1027

theorem number_of_white_balls (total_balls : ℕ) (red_prob black_prob : ℝ)
  (h_total : total_balls = 50)
  (h_red_prob : red_prob = 0.15)
  (h_black_prob : black_prob = 0.45) :
  ∃ (white_balls : ℕ), white_balls = 20 :=
by
  sorry

end number_of_white_balls_1_1027


namespace total_amount_is_2500_1_1735

noncomputable def total_amount_divided (P1 : ℝ) (annual_income : ℝ) : ℝ :=
  let P2 := 2500 - P1
  let income_from_P1 := (5 / 100) * P1
  let income_from_P2 := (6 / 100) * P2
  income_from_P1 + income_from_P2

theorem total_amount_is_2500 : 
  (total_amount_divided 2000 130) = 130 :=
by
  sorry

end total_amount_is_2500_1_1735


namespace projection_ratio_zero_1_1909

variables (v w u p q : ℝ → ℝ) -- Assuming vectors are functions from ℝ to ℝ
variables (norm : (ℝ → ℝ) → ℝ) -- norm is a function from vectors to ℝ
variables (proj : (ℝ → ℝ) → (ℝ → ℝ) → (ℝ → ℝ)) -- proj is the projection function

-- Assume the conditions
axiom proj_p : p = proj v w
axiom proj_q : q = proj p u
axiom perp_uv : ∀ t, v t * u t = 0 -- u is perpendicular to v
axiom norm_ratio : norm p / norm v = 3 / 8

theorem projection_ratio_zero : norm q / norm v = 0 :=
by sorry

end projection_ratio_zero_1_1909


namespace difference_between_advertised_and_actual_mileage_1_1951

def advertised_mileage : ℕ := 35

def city_mileage_regular : ℕ := 30
def highway_mileage_premium : ℕ := 40
def traffic_mileage_diesel : ℕ := 32

def gallons_regular : ℕ := 4
def gallons_premium : ℕ := 4
def gallons_diesel : ℕ := 4

def total_miles_driven : ℕ :=
  (gallons_regular * city_mileage_regular) + 
  (gallons_premium * highway_mileage_premium) + 
  (gallons_diesel * traffic_mileage_diesel)

def total_gallons_used : ℕ :=
  gallons_regular + gallons_premium + gallons_diesel

def weighted_average_mpg : ℤ :=
  total_miles_driven / total_gallons_used

theorem difference_between_advertised_and_actual_mileage :
  advertised_mileage - weighted_average_mpg = 1 :=
by
  -- proof to be filled in
  sorry

end difference_between_advertised_and_actual_mileage_1_1951


namespace problem_1_1485

variables {b1 b2 b3 a1 a2 : ℤ}

-- Condition: five numbers -9, b1, b2, b3, -1 form a geometric sequence.
def is_geometric_seq (b1 b2 b3 : ℤ) : Prop :=
b1^2 = -9 * b2 ∧ b2^2 = b1 * b3 ∧ b1 * b3 = 9

-- Condition: four numbers -9, a1, a2, -3 form an arithmetic sequence.
def is_arithmetic_seq (a1 a2 : ℤ) : Prop :=
2 * a1 = -9 + a2 ∧ 2 * a2 = a1 - 3

-- Proof problem: prove that b2(a2 - a1) = -6
theorem problem (h_geom : is_geometric_seq b1 b2 b3) (h_arith : is_arithmetic_seq a1 a2) : 
  b2 * (a2 - a1) = -6 :=
by sorry

end problem_1_1485


namespace circle_chord_length_equal_1_1665

def equation_of_circle (D E F : ℝ) : Prop :=
  ∀ x y : ℝ, x^2 + y^2 + D * x + E * y + F = 0

def distances_equal (D E F : ℝ) : Prop :=
  (D^2 ≠ E^2 ∧ E^2 > 4 * F) → 
  (∀ x y : ℝ, (x^2 + y^2 + D * x + E * y + F = 0) → (x = -D/2) ∧ (y = -E/2) → (abs x = abs y))

theorem circle_chord_length_equal (D E F : ℝ) (h : D^2 ≠ E^2 ∧ E^2 > 4 * F) :
  distances_equal D E F :=
by
  sorry

end circle_chord_length_equal_1_1665


namespace range_of_a_1_1294

noncomputable def f (a x : ℝ) : ℝ := x^3 - 3 * x + a

theorem range_of_a (a : ℝ) :
  (∃ x₁ x₂ x₃ : ℝ, x₁ ≠ x₂ ∧ x₂ ≠ x₃ ∧ x₁ ≠ x₃ ∧ f a x₁ = 0 ∧ f a x₂ = 0 ∧ f a x₃ = 0) ↔ (-2 < a ∧ a < 2) :=
by
  sorry

end range_of_a_1_1294


namespace find_a_1_1332

def is_odd_function (f : ℝ → ℝ) : Prop := ∀ x : ℝ, x ≠ 0 → f (-x) = -f x

noncomputable def f (a x : ℝ) : ℝ := (1 / (2^x - 1)) + a

theorem find_a (a : ℝ) : 
  is_odd_function (f a) → a = 1 / 2 :=
by
  sorry

end find_a_1_1332


namespace right_triangle_area_1_1412

theorem right_triangle_area (a b c : ℝ) (h1 : c = 13) (h2 : a = 5) (h3 : a^2 + b^2 = c^2) : 0.5 * a * b = 30 := by
  sorry

end right_triangle_area_1_1412


namespace long_furred_brown_dogs_1_1549

-- Definitions based on given conditions
def T : ℕ := 45
def L : ℕ := 36
def B : ℕ := 27
def N : ℕ := 8

-- The number of long-furred brown dogs (LB) that needs to be proved
def LB : ℕ := 26

-- Lean 4 statement to prove LB
theorem long_furred_brown_dogs :
  L + B - LB = T - N :=
by 
  unfold T L B N LB -- we unfold definitions to simplify the theorem
  sorry

end long_furred_brown_dogs_1_1549


namespace time_to_cross_pole_is_correct_1_1573

-- Define the conversion factor to convert km/hr to m/s
def km_per_hr_to_m_per_s (speed_km_per_hr : ℕ) : ℕ := speed_km_per_hr * 1000 / 3600

-- Define the speed of the train in m/s
def train_speed_m_per_s : ℕ := km_per_hr_to_m_per_s 216

-- Define the length of the train
def train_length_m : ℕ := 480

-- Define the time to cross an electric pole
def time_to_cross_pole : ℕ := train_length_m / train_speed_m_per_s

-- Theorem stating that the computed time to cross the pole is 8 seconds
theorem time_to_cross_pole_is_correct :
  time_to_cross_pole = 8 := by
  sorry

end time_to_cross_pole_is_correct_1_1573


namespace sum_x_coordinates_eq_3_1_1700

def f : ℝ → ℝ := sorry -- definition of the function f as given by the five line segments

theorem sum_x_coordinates_eq_3 :
  (∃ x1 x2 x3 : ℝ, (f x1 = x1 + 1 ∧ f x2 = x2 + 1 ∧ f x3 = x3 + 1) ∧ (x1 + x2 + x3 = 3)) :=
sorry

end sum_x_coordinates_eq_3_1_1700


namespace max_n_value_1_1056

theorem max_n_value (a b c d : ℝ) (h1 : a > b) (h2 : b > c) (h3 : c > d) :
  (1 / (a - b)) + (1 / (b - c)) + (1 / (c - d)) ≥ (9 / (a - d)) :=
by
  sorry

end max_n_value_1_1056


namespace max_subsequences_2001_1_1555

theorem max_subsequences_2001 (seq : List ℕ) (h_len : seq.length = 2001) : 
  ∃ n : ℕ, n = 667^3 :=
sorry

end max_subsequences_2001_1_1555


namespace blocks_combination_count_1_1067

-- Definition statements reflecting all conditions in the problem
def select_4_blocks_combinations : ℕ :=
  let choose (n k : ℕ) := Nat.choose n k
  let factorial (n : ℕ) := Nat.factorial n
  choose 6 4 * choose 6 4 * factorial 4

-- Theorem stating the result we want to prove
theorem blocks_combination_count : select_4_blocks_combinations = 5400 :=
by
  -- We will provide the proof steps here
  sorry

end blocks_combination_count_1_1067


namespace find_angle_C_1_1784

variable {A B C : ℝ} -- Angles of triangle ABC
variable {a b c : ℝ} -- Sides opposite the respective angles

theorem find_angle_C
  (h1 : 2 * c * Real.cos B = 2 * a + b) : 
  C = 120 :=
  sorry

end find_angle_C_1_1784


namespace range_of_a_1_1267

def discriminant (a b c : ℝ) : ℝ :=
  b^2 - 4 * a * c

def no_real_roots (a b c : ℝ) : Prop :=
  discriminant a b c < 0

theorem range_of_a (a : ℝ) :
  no_real_roots 1 (2 * a - 1) 1 ↔ -1 / 2 < a ∧ a < 3 / 2 := 
by sorry

end range_of_a_1_1267


namespace Petya_wins_optimally_1_1538

-- Defining the game state and rules
inductive GameState
| PetyaWin
| VasyaWin

-- Rules of the game
def game_rule (n : ℕ) : Prop :=
  n > 0 ∧ (n % 3 = 0 ∨ n % 3 = 1 ∨ n % 3 = 2)

-- Determine the winner given the initial number of minuses
def determine_winner (n : ℕ) : GameState :=
  if n % 3 = 0 then GameState.PetyaWin else GameState.VasyaWin

-- Theorem: Petya will win the game if both play optimally
theorem Petya_wins_optimally (n : ℕ) (h1 : n = 2021) (h2 : game_rule n) : determine_winner n = GameState.PetyaWin :=
by {
  sorry
}

end Petya_wins_optimally_1_1538


namespace algebraic_expression_value_1_1938

theorem algebraic_expression_value (a : ℝ) (h : 2 * a^2 + 3 * a - 5 = 0) : 6 * a^2 + 9 * a - 5 = 10 :=
by
  sorry

end algebraic_expression_value_1_1938


namespace Eddy_travel_time_1_1437

theorem Eddy_travel_time (T V_e V_f : ℝ) 
  (dist_AB dist_AC : ℝ) 
  (time_Freddy : ℝ) 
  (speed_ratio : ℝ) 
  (h1 : dist_AB = 600) 
  (h2 : dist_AC = 300) 
  (h3 : time_Freddy = 3) 
  (h4 : speed_ratio = 2)
  (h5 : V_f = dist_AC / time_Freddy)
  (h6 : V_e = speed_ratio * V_f)
  (h7 : T = dist_AB / V_e) :
  T = 3 :=
by
  sorry

end Eddy_travel_time_1_1437


namespace number_of_friends_1_1078

-- Define the initial amount of money John had
def initial_money : ℝ := 20.10 

-- Define the amount spent on sweets
def sweets_cost : ℝ := 1.05 

-- Define the amount given to each friend
def money_per_friend : ℝ := 1.00 

-- Define the amount of money left after giving to friends
def final_money : ℝ := 17.05 

-- Define a theorem to find the number of friends John gave money to
theorem number_of_friends (init_money sweets_cost money_per_friend final_money : ℝ) : 
  (init_money - sweets_cost - final_money) / money_per_friend = 2 :=
by
  sorry

end number_of_friends_1_1078


namespace malvina_correct_1_1579
noncomputable def angle (x : ℝ) : Prop := 0 < x ∧ x < 180
noncomputable def malvina_identifies (x : ℝ) : Prop := x > 90

noncomputable def sum_of_values := (Real.sqrt 5 + Real.sqrt 2) / 2

theorem malvina_correct (x : ℝ) (h1 : angle x) (h2 : malvina_identifies x) :
  sum_of_values = (Real.sqrt 5 + Real.sqrt 2) / 2 :=
by sorry

end malvina_correct_1_1579


namespace arnold_total_protein_1_1848

-- Definitions and conditions
def collagen_protein_per_scoop : ℕ := 18 / 2
def protein_powder_protein_per_scoop : ℕ := 21
def steak_protein : ℕ := 56

def collagen_scoops : ℕ := 1
def protein_powder_scoops : ℕ := 1
def steaks : ℕ := 1

-- Statement of the theorem/problem
theorem arnold_total_protein : 
  (collagen_protein_per_scoop * collagen_scoops) + 
  (protein_powder_protein_per_scoop * protein_powder_scoops) + 
  (steak_protein * steaks) = 86 :=
by
  sorry

end arnold_total_protein_1_1848


namespace cauchy_bunyakovsky_inequality_1_1464

theorem cauchy_bunyakovsky_inequality 
  (n : ℕ) 
  (a b k A B K : Fin n → ℝ) : 
  (∑ i, a i * A i)^2 ≤ (∑ i, (a i)^2) * (∑ i, (A i)^2) :=
by
  sorry

end cauchy_bunyakovsky_inequality_1_1464


namespace largest_possible_d_plus_r_1_1165

theorem largest_possible_d_plus_r :
  ∃ d r : ℕ, 0 < d ∧ 468 % d = r ∧ 636 % d = r ∧ 867 % d = r ∧ d + r = 27 := by
  sorry

end largest_possible_d_plus_r_1_1165


namespace two_digit_numbers_1_1728

def is_digit (n : ℕ) : Prop := n ≤ 9

theorem two_digit_numbers (a b : ℕ) (h1 : is_digit a) (h2 : is_digit b) 
  (h3 : a ≠ b) (h4 : (a + b) = 11) : 
  (∃ n m : ℕ, (n = 10 * a + b) ∧ (m = 10 * b + a) ∧ (∃ k : ℕ, (10 * a + b)^2 - (10 * b + a)^2 = k^2)) := 
sorry

end two_digit_numbers_1_1728


namespace complex_sum_identity_1_1142

theorem complex_sum_identity (z : ℂ) (h : z^2 + z + 1 = 0) : 
  z^100 + z^101 + z^102 + z^103 + z^104 = -1 := 
by 
  sorry

end complex_sum_identity_1_1142


namespace find_unit_price_B_1_1106

variable (x : ℕ)

def unit_price_B := x
def unit_price_A := x + 50

theorem find_unit_price_B (h : (2000 / unit_price_A x = 1500 / unit_price_B x)) : unit_price_B x = 150 :=
by
  sorry

end find_unit_price_B_1_1106


namespace relationship_between_a_and_b_1_1187

theorem relationship_between_a_and_b {a b : ℝ} (h1 : a > 0) (h2 : b > 0)
  (h3 : ∀ x : ℝ, |(2 * x + 2)| < a → |(x + 1)| < b) : b ≥ a / 2 :=
by
  -- The proof steps will be inserted here
  sorry

end relationship_between_a_and_b_1_1187


namespace count_integer_values_of_x_1_1903

theorem count_integer_values_of_x (x : ℕ) (h : ⌈Real.sqrt x⌉ = 12) : ∃ (n : ℕ), n = 23 :=
by
  sorry

end count_integer_values_of_x_1_1903


namespace original_savings_1_1655

variable (A B : ℕ)

-- A's savings are 5 times that of B's savings
def cond1 : Prop := A = 5 * B

-- If A withdraws 60 yuan and B deposits 60 yuan, then B's savings will be twice that of A's savings
def cond2 : Prop := (B + 60) = 2 * (A - 60)

-- Prove the original savings of A and B
theorem original_savings (h1 : cond1 A B) (h2 : cond2 A B) : A = 100 ∧ B = 20 := by
  sorry

end original_savings_1_1655


namespace garden_perimeter_ratio_1_1880

theorem garden_perimeter_ratio (side_length : ℕ) (tripled_side_length : ℕ) (original_perimeter : ℕ) (new_perimeter : ℕ) (ratio : ℚ) :
  side_length = 50 →
  tripled_side_length = 3 * side_length →
  original_perimeter = 4 * side_length →
  new_perimeter = 4 * tripled_side_length →
  ratio = original_perimeter / new_perimeter →
  ratio = 1 / 3 :=
by
  sorry

end garden_perimeter_ratio_1_1880


namespace system_solution_1_1018

theorem system_solution (x y : ℝ) (h1 : x + y = 1) (h2 : x - y = 3) : x = 2 ∧ y = -1 :=
by
  sorry

end system_solution_1_1018


namespace part1_part2_1_1510

def f (x a : ℝ) : ℝ := abs (x - a) + 2 * x

theorem part1 (x : ℝ) : f x (-1) ≤ 0 ↔ x ≤ -1/3 :=
by
  sorry

theorem part2 (a : ℝ) : (∀ x : ℝ, x ≥ -1 → f x a ≥ 0) ↔ (a ≤ -3 ∨ a ≥ 1) :=
by
  sorry

end part1_part2_1_1510


namespace propA_propB_relation_1_1638

variable (x y : ℤ)

theorem propA_propB_relation :
  (x + y ≠ 5 → x ≠ 2 ∨ y ≠ 3) ∧ ¬(x ≠ 2 ∨ y ≠ 3 → x + y ≠ 5) :=
by
  sorry

end propA_propB_relation_1_1638


namespace value_of_f_12_1_1955

theorem value_of_f_12 (f : ℕ → ℤ) 
  (h1 : f 2 = 5)
  (h2 : f 3 = 7)
  (h3 : ∀ m n : ℕ, 0 < m → 0 < n → f m + f n = f (m * n)) :
  f 12 = 17 :=
by
  sorry

end value_of_f_12_1_1955


namespace pieces_of_candy_1_1890

def total_items : ℝ := 3554
def secret_eggs : ℝ := 145.0

theorem pieces_of_candy : (total_items - secret_eggs) = 3409 :=
by 
  sorry

end pieces_of_candy_1_1890


namespace x_y_solution_1_1740

variable (x y : ℕ)

noncomputable def x_wang_speed : ℕ := x - 6

theorem x_y_solution (hx : (5 : ℚ) / 6 * x = y) (hy : (2 : ℚ) / 3 * (x - 6) = y - 10) : x = 36 ∧ y = 30 :=
by {
  sorry
}

end x_y_solution_1_1740


namespace num_partitions_of_staircase_1_1243

-- Definition of a staircase
def is_staircase (n : ℕ) (cells : ℕ × ℕ → Prop) : Prop :=
  ∀ (i j : ℕ), 1 ≤ j → j ≤ i → i ≤ n → cells (i, j)

-- Number of partitions of a staircase of height n
def num_partitions (n : ℕ) : ℕ :=
  2^(n-1)

theorem num_partitions_of_staircase (n : ℕ) (cells : ℕ × ℕ → Prop) :
  is_staircase n cells → (∃ p : ℕ, p = num_partitions n) :=
by
  intro h
  use (2^(n-1))
  sorry

end num_partitions_of_staircase_1_1243


namespace xy_range_1_1498

theorem xy_range (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x + 1/x + y + 1/y = 5) :
  1/4 ≤ x * y ∧ x * y ≤ 4 :=
sorry

end xy_range_1_1498


namespace base_case_inequality_induction_inequality_1_1180

theorem base_case_inequality : 2^5 > 5^2 + 1 := by
  -- Proof not required
  sorry

theorem induction_inequality (n : ℕ) (h : n ≥ 5) : 2^n > n^2 + 1 := by
  -- Proof not required
  sorry

end base_case_inequality_induction_inequality_1_1180


namespace number_of_ways_to_assign_roles_1_1887

theorem number_of_ways_to_assign_roles :
  let men := 6
  let women := 5
  let male_roles := 3
  let female_roles := 2
  let either_gender_roles := 1
  let total_men := men - male_roles
  let total_women := women - female_roles
  (men.choose male_roles) * (women.choose female_roles) * (total_men + total_women).choose either_gender_roles = 14400 := by 
sorry

end number_of_ways_to_assign_roles_1_1887


namespace zoes_apartment_number_units_digit_is_1_1_1742

-- Defining the conditions as the initial problem does
def is_two_digit_number (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100

def is_square (n : ℕ) : Prop :=
  ∃ k : ℕ, k * k = n

def is_odd (n : ℕ) : Prop :=
  n % 2 = 1

def is_divisible_by_3 (n : ℕ) : Prop :=
  n % 3 = 0

def has_digit_two (n : ℕ) : Prop :=
  n / 10 = 2 ∨ n % 10 = 2

def three_out_of_four (n : ℕ) : Prop :=
  (is_square n ∧ is_odd n ∧ is_divisible_by_3 n ∧ ¬ has_digit_two n) ∨
  (is_square n ∧ is_odd n ∧ ¬ is_divisible_by_3 n ∧ has_digit_two n) ∨
  (is_square n ∧ ¬ is_odd n ∧ is_divisible_by_3 n ∧ has_digit_two n) ∨
  (¬ is_square n ∧ is_odd n ∧ is_divisible_by_3 n ∧ has_digit_two n)

theorem zoes_apartment_number_units_digit_is_1 : ∃ n : ℕ, is_two_digit_number n ∧ three_out_of_four n ∧ n % 10 = 1 :=
by
  sorry

end zoes_apartment_number_units_digit_is_1_1_1742


namespace ratio_sum_2_or_4_1_1627

theorem ratio_sum_2_or_4 (a b c d : ℝ) 
  (h1 : a / b + b / c + c / d + d / a = 6)
  (h2 : a / c + b / d + c / a + d / b = 8) : 
  (a / b + c / d = 2) ∨ (a / b + c / d = 4) :=
sorry

end ratio_sum_2_or_4_1_1627


namespace distance_between_trees_1_1223

def yard_length : ℝ := 1530
def number_of_trees : ℝ := 37
def number_of_gaps := number_of_trees - 1

theorem distance_between_trees :
  number_of_gaps ≠ 0 →
  (yard_length / number_of_gaps) = 42.5 :=
by
  sorry

end distance_between_trees_1_1223


namespace calculate_product_1_1631

noncomputable def complex_number_r (r : ℂ) : Prop :=
r^6 = 1 ∧ r ≠ 1

theorem calculate_product (r : ℂ) (h₁ : r^6 = 1) (h₂ : r ≠ 1) :
  (r - 1) * (r^2 - 1) * (r^3 - 1) * (r^4 - 1) * (r^5 - 1) = 2 := 
sorry

end calculate_product_1_1631


namespace stream_speed_1_1084

def upstream_time : ℝ := 4  -- time in hours
def downstream_time : ℝ := 4  -- time in hours
def upstream_distance : ℝ := 32  -- distance in km
def downstream_distance : ℝ := 72  -- distance in km

-- Speed equations based on given conditions
def effective_speed_upstream (vj vs : ℝ) : Prop := vj - vs = upstream_distance / upstream_time
def effective_speed_downstream (vj vs : ℝ) : Prop := vj + vs = downstream_distance / downstream_time

theorem stream_speed (vj vs : ℝ)  
  (h1 : effective_speed_upstream vj vs)
  (h2 : effective_speed_downstream vj vs) : 
  vs = 5 := sorry

end stream_speed_1_1084


namespace smallest_n_for_inequality_1_1812

theorem smallest_n_for_inequality (n : ℕ) : 5 + 3 * n > 300 ↔ n = 99 := by
  sorry

end smallest_n_for_inequality_1_1812


namespace negative_column_exists_1_1920

theorem negative_column_exists
  (table : Fin 1999 → Fin 2001 → ℤ)
  (H : ∀ i : Fin 1999, (∏ j : Fin 2001, table i j) < 0) :
  ∃ j : Fin 2001, (∏ i : Fin 1999, table i j) < 0 :=
sorry

end negative_column_exists_1_1920


namespace area_of_quadrilateral_1_1674

theorem area_of_quadrilateral (A B C : ℝ) (triangle1 triangle2 triangle3 quadrilateral : ℝ)
  (hA : A = 5) (hB : B = 9) (hC : C = 9)
  (h_sum : quadrilateral = triangle1 + triangle2 + triangle3)
  (h1 : triangle1 = A)
  (h2 : triangle2 = B)
  (h3 : triangle3 = C) :
  quadrilateral = 40 :=
by
  sorry

end area_of_quadrilateral_1_1674


namespace total_earnings_first_three_months_1_1500

-- Definitions
def earning_first_month : ℕ := 350
def earning_second_month : ℕ := 2 * earning_first_month + 50
def earning_third_month : ℕ := 4 * (earning_first_month + earning_second_month)

-- Question restated as a theorem
theorem total_earnings_first_three_months : 
  (earning_first_month + earning_second_month + earning_third_month = 5500) :=
by 
  -- Placeholder for the proof
  sorry

end total_earnings_first_three_months_1_1500


namespace find_second_number_1_1423

theorem find_second_number 
    (lcm : ℕ) (gcf : ℕ) (num1 : ℕ) (num2 : ℕ)
    (h_lcm : lcm = 56) (h_gcf : gcf = 10) (h_num1 : num1 = 14) 
    (h_product : lcm * gcf = num1 * num2) : 
    num2 = 40 :=
by
  sorry

end find_second_number_1_1423


namespace max_min_z_1_1085

-- Define the ellipse
def on_ellipse (x y : ℝ) : Prop :=
  x^2 + 4*y^2 = 4*x

-- Define the function z
def z (x y : ℝ) : ℝ :=
  x^2 - y^2

-- Define the required points
def P1 (x y : ℝ) :=
  x = 4 ∧ y = 0

def P2 (x y : ℝ) :=
  x = 2/5 ∧ (y = 3/5 ∨ y = -3/5)

-- Theorem stating the required conditions
theorem max_min_z (x y : ℝ) (h : on_ellipse x y) :
  (P1 x y → z x y = 16) ∧ (P2 x y → z x y = -1/5) :=
by
  sorry

end max_min_z_1_1085


namespace sequence_bound_1_1184

theorem sequence_bound (a : ℕ → ℝ) (h_pos : ∀ n, a n > 0) (h_seq : ∀ n, (a n) ^ 2 ≤ a (n + 1)) :
  ∀ n, a n < 1 / n :=
by
  intros
  sorry

end sequence_bound_1_1184


namespace distance_between_polar_points_1_1996

noncomputable def polar_to_rect (r θ : ℝ) : ℝ × ℝ :=
  (r * Real.cos θ, r * Real.sin θ)

noncomputable def distance (A B : ℝ × ℝ) : ℝ :=
  Real.sqrt ((B.1 - A.1) ^ 2 + (B.2 - A.2) ^ 2)

theorem distance_between_polar_points :
  let A := polar_to_rect 1 (Real.pi / 6)
  let B := polar_to_rect 2 (-Real.pi / 2)
  distance A B = Real.sqrt 7 :=
by
  sorry

end distance_between_polar_points_1_1996


namespace tan_8pi_over_3_eq_neg_sqrt3_1_1566

open Real

theorem tan_8pi_over_3_eq_neg_sqrt3 : tan (8 * π / 3) = -√3 :=
by
  sorry

end tan_8pi_over_3_eq_neg_sqrt3_1_1566


namespace playgroup_count_1_1867

-- Definitions based on the conditions
def total_people (girls boys parents : ℕ) := girls + boys + parents
def playgroups (total size_per_group : ℕ) := total / size_per_group

-- Statement of the problem
theorem playgroup_count (girls boys parents size_per_group : ℕ)
  (h_girls : girls = 14)
  (h_boys : boys = 11)
  (h_parents : parents = 50)
  (h_size_per_group : size_per_group = 25) :
  playgroups (total_people girls boys parents) size_per_group = 3 :=
by {
  -- This is just the statement, the proof is skipped with sorry
  sorry
}

end playgroup_count_1_1867


namespace triangle_angles_sum_1_1682

theorem triangle_angles_sum (x : ℝ) (h : 40 + 3 * x + (x + 10) = 180) : x = 32.5 := by
  sorry

end triangle_angles_sum_1_1682


namespace find_k_1_1846

theorem find_k (a : ℕ → ℕ) (h₀ : a 1 = 2) (h₁ : ∀ m n, a (m + n) = a m * a n) (hk : a (k + 1) = 1024) : k = 9 := 
sorry

end find_k_1_1846


namespace part_I_part_II_1_1443

noncomputable
def x₀ : ℝ := 2

noncomputable
def f (x m : ℝ) : ℝ := |x - m| + |x + 1/m| - x₀

theorem part_I (x : ℝ) : |x + 3| - 2 * x - 1 < 0 ↔ x > 2 :=
by sorry

theorem part_II (m : ℝ) (h : m > 0) :
  (∃ x : ℝ, f x m = 0) → m = 1 :=
by sorry

end part_I_part_II_1_1443


namespace find_f_x_minus_1_1_1739

-- Define the function f
def f (x : ℝ) : ℝ := 2 * x - 1

-- State the theorem
theorem find_f_x_minus_1 (x : ℝ) : f (x - 1) = 2 * x - 3 := by
  sorry

end find_f_x_minus_1_1_1739


namespace circles_intersect_iff_1_1288

-- Definitions of the two circles and their parameters
def circle1 (x y : ℝ) : Prop := x^2 + y^2 = 9

def circle2 (x y r : ℝ) : Prop := x^2 + y^2 + 8 * x - 6 * y + 25 - r^2 = 0

-- Lean statement to prove the range of r
theorem circles_intersect_iff (r : ℝ) (hr : 0 < r) :
  (∃ x y : ℝ, circle1 x y ∧ circle2 x y r) ↔ (2 < r ∧ r < 8) :=
by
  sorry

end circles_intersect_iff_1_1288


namespace defective_chip_ratio_1_1115

theorem defective_chip_ratio (defective_chips total_chips : ℕ)
  (h1 : defective_chips = 15)
  (h2 : total_chips = 60000) :
  defective_chips / total_chips = 1 / 4000 :=
by
  sorry

end defective_chip_ratio_1_1115


namespace sum_and_times_1_1139

theorem sum_and_times 
  (a : ℕ) (ha : a = 99) 
  (b : ℕ) (hb : b = 301) 
  (c : ℕ) (hc : c = 200) : 
  a + b = 2 * c :=
by 
  -- skipping proof 
  sorry

end sum_and_times_1_1139


namespace total_enemies_1_1250

theorem total_enemies (n : ℕ) : (n - 3) * 9 = 72 → n = 11 :=
by
  sorry

end total_enemies_1_1250


namespace find_all_pairs_1_1745

def is_solution (m n : ℕ) : Prop := 200 * m + 6 * n = 2006

def valid_pairs : List (ℕ × ℕ) := [(1, 301), (4, 201), (7, 101), (10, 1)]

theorem find_all_pairs :
  ∀ (m n : ℕ), is_solution m n ↔ (m, n) ∈ valid_pairs := by sorry

end find_all_pairs_1_1745


namespace Rohit_is_to_the_east_of_starting_point_1_1926

-- Define the conditions and the problem statement.
def Rohit's_movements_proof
  (distance_south : ℕ) (distance_first_left : ℕ) (distance_second_left : ℕ) (distance_right : ℕ)
  (final_distance : ℕ) : Prop :=
  distance_south = 25 ∧
  distance_first_left = 20 ∧
  distance_second_left = 25 ∧
  distance_right = 15 ∧
  final_distance = 35 →
  (direction : String) → (distance : ℕ) →
  direction = "east" ∧ distance = final_distance

-- We can now state the theorem
theorem Rohit_is_to_the_east_of_starting_point :
  Rohit's_movements_proof 25 20 25 15 35 :=
by
  sorry

end Rohit_is_to_the_east_of_starting_point_1_1926


namespace smallest_number_is_27_1_1614

theorem smallest_number_is_27 (a b c : ℕ) (h_mean : (a + b + c) / 3 = 30) (h_median : b = 28) (h_largest : c = b + 7) : a = 27 :=
by {
  sorry
}

end smallest_number_is_27_1_1614


namespace average_of_numbers_eq_x_1_1689

theorem average_of_numbers_eq_x (x : ℝ) (h : (2 + x + 10) / 3 = x) : x = 6 := 
by sorry

end average_of_numbers_eq_x_1_1689


namespace product_of_terms_eq_72_1_1757

theorem product_of_terms_eq_72
  (a b c : ℝ)
  (h1 : 0 < a ∧ 0 < b ∧ 0 < c)
  (h2 : (a / b + b / c + c / a) + (b / a + c / b + a / c) = 12) :
  (a / b + b / c + c / a) * (b / a + c / b + a / c) = 72 :=
by
  sorry

end product_of_terms_eq_72_1_1757


namespace roy_missed_days_1_1460

theorem roy_missed_days {hours_per_day days_per_week actual_hours_week missed_days : ℕ}
    (h1 : hours_per_day = 2)
    (h2 : days_per_week = 5)
    (h3 : actual_hours_week = 6)
    (expected_hours_week : ℕ := hours_per_day * days_per_week)
    (missed_hours : ℕ := expected_hours_week - actual_hours_week)
    (missed_days := missed_hours / hours_per_day) :
  missed_days = 2 := by
  sorry

end roy_missed_days_1_1460


namespace michelle_total_payment_1_1916
noncomputable def michelle_base_cost := 25
noncomputable def included_talk_time := 40 -- in hours
noncomputable def text_cost := 10 -- in cents per message
noncomputable def extra_talk_cost := 15 -- in cents per minute
noncomputable def february_texts_sent := 200
noncomputable def february_talk_time := 41 -- in hours

theorem michelle_total_payment : 
  25 + ((200 * 10) / 100) + (((41 - 40) * 60 * 15) / 100) = 54 := by
  sorry

end michelle_total_payment_1_1916


namespace find_c_1_1481

noncomputable def g (x c : ℝ) : ℝ := 1 / (3 * x + c)
noncomputable def g_inv (x : ℝ) : ℝ := (2 - 3 * x) / (3 * x)

theorem find_c (c : ℝ) : (∀ x : ℝ, g_inv (g x c) = x) -> c = 3 :=
by 
  intro h
  sorry

end find_c_1_1481


namespace sum_first_m_terms_inequality_always_holds_1_1248

noncomputable def f (x : ℝ) : ℝ := 1 / (4^x + 2)

-- Define the sequence {a_n}
noncomputable def a_n (n m : ℕ) : ℝ := f (n / m)

-- Define the sum S_m
noncomputable def S_m (m : ℕ) : ℝ := ∑ n in Finset.range m, a_n n m

theorem sum_first_m_terms (m : ℕ) : S_m m = (1 / 12) * (3 * m - 1) := sorry

theorem inequality_always_holds (m : ℕ) (a : ℝ) (h : ∀ m : ℕ, (a^m / S_m m) < (a^(m+1) / S_m (m+1))) : a > 5/2 := sorry

end sum_first_m_terms_inequality_always_holds_1_1248


namespace simplify_fraction_1_1813

theorem simplify_fraction (b : ℕ) (hb : b = 5) : (15 * b^4) / (90 * b^3 * b) = 1 / 6 := by
  sorry

end simplify_fraction_1_1813


namespace determine_marriages_1_1986

-- Definitions of the items each person bought
variable (a_items b_items c_items : ℕ) -- Number of items bought by wives a, b, and c
variable (A_items B_items C_items : ℕ) -- Number of items bought by husbands A, B, and C

-- Conditions
variable (spend_eq_square_a : a_items * a_items = a_spend) -- Spending equals square of items
variable (spend_eq_square_b : b_items * b_items = b_spend)
variable (spend_eq_square_c : c_items * c_items = c_spend)
variable (spend_eq_square_A : A_items * A_items = A_spend)
variable (spend_eq_square_B : B_items * B_items = B_spend)
variable (spend_eq_square_C : C_items * C_items = C_spend)

variable (A_spend_eq : A_spend = a_spend + 48) -- Husbands spent 48 yuan more than wives
variable (B_spend_eq : B_spend = b_spend + 48)
variable (C_spend_eq : C_spend = c_spend + 48)

variable (A_bought_9_more : A_items = b_items + 9) -- A bought 9 more items than b
variable (B_bought_7_more : B_items = a_items + 7) -- B bought 7 more items than a

-- Theorem statement
theorem determine_marriages (hA : A_items ≥ b_items + 9) (hB : B_items ≥ a_items + 7) :
  (A_spend = A_items * A_items) ∧ (B_spend = B_items * B_items) ∧ (C_spend = C_items * C_items) ∧
  (a_spend = a_items * a_items) ∧ (b_spend = b_items * b_items) ∧ (c_spend = c_items * c_items) →
  (A_spend = a_spend + 48) ∧ (B_spend = b_spend + 48) ∧ (C_spend = c_spend + 48) →
  (A_items = b_items + 9) ∧ (B_items = a_items + 7) →
  (A_items = 13 ∧ c_items = 11) ∧ (B_items = 8 ∧ b_items = 4) ∧ (C_items = 7 ∧ a_items = 1) :=
by
  sorry

end determine_marriages_1_1986


namespace sphere_surface_area_ratio_1_1405

axiom prism_has_circumscribed_sphere : Prop
axiom prism_has_inscribed_sphere : Prop

theorem sphere_surface_area_ratio 
  (h1 : prism_has_circumscribed_sphere)
  (h2 : prism_has_inscribed_sphere) : 
  ratio_surface_area_of_circumscribed_to_inscribed_sphere = 5 :=
sorry

end sphere_surface_area_ratio_1_1405


namespace muffin_count_1_1723

theorem muffin_count (doughnuts cookies muffins : ℕ) (h1 : doughnuts = 50) (h2 : cookies = (3 * doughnuts) / 5) (h3 : muffins = (1 * doughnuts) / 5) : muffins = 10 :=
by sorry

end muffin_count_1_1723


namespace minimal_volume_block_1_1964

theorem minimal_volume_block (l m n : ℕ) (h : (l - 1) * (m - 1) * (n - 1) = 297) : l * m * n = 192 :=
sorry

end minimal_volume_block_1_1964


namespace parcels_division_1_1352

theorem parcels_division (x y n : ℕ) (h : 5 + 2 * x + 3 * y = 4 * n) (hn : n = x + y) :
    n = 3 ∨ n = 4 ∨ n = 5 := 
sorry

end parcels_division_1_1352


namespace simultaneous_equations_solution_1_1949

-- Definition of the two equations
def eq1 (m x y : ℝ) : Prop := y = m * x + 5
def eq2 (m x y : ℝ) : Prop := y = (3 * m - 2) * x + 6

-- Lean theorem statement to check if the equations have a solution
theorem simultaneous_equations_solution (m : ℝ) :
  (m ≠ 1) ↔ ∃ x y : ℝ, eq1 m x y ∧ eq2 m x y := 
sorry

end simultaneous_equations_solution_1_1949


namespace additional_savings_in_cents_1_1968

/-
The book has a cover price of $30.
There are two discount methods to compare:
1. First $5 off, then 25% off.
2. First 25% off, then $5 off.
Prove that the difference in final costs (in cents) between these two discount methods is 125 cents.
-/
def book_price : ℝ := 30
def discount_cash : ℝ := 5
def discount_percentage : ℝ := 0.25

def final_price_apply_cash_first (price : ℝ) (cash_discount : ℝ) (percentage_discount : ℝ) : ℝ :=
  (price - cash_discount) * (1 - percentage_discount)

def final_price_apply_percentage_first (price : ℝ) (percentage_discount : ℝ) (cash_discount : ℝ) : ℝ :=
  (price * (1 - percentage_discount)) - cash_discount

def savings_comparison (price : ℝ) (cash_discount : ℝ) (percentage_discount : ℝ) : ℝ :=
  (final_price_apply_cash_first price cash_discount percentage_discount) - 
  (final_price_apply_percentage_first price percentage_discount cash_discount)

theorem additional_savings_in_cents : 
  savings_comparison book_price discount_cash discount_percentage * 100 = 125 :=
  by sorry

end additional_savings_in_cents_1_1968


namespace Annabelle_saved_12_dollars_1_1470

def weekly_allowance : ℕ := 30
def spent_on_junk_food : ℕ := weekly_allowance / 3
def spent_on_sweets : ℕ := 8
def total_spent : ℕ := spent_on_junk_food + spent_on_sweets
def saved_amount : ℕ := weekly_allowance - total_spent

theorem Annabelle_saved_12_dollars : saved_amount = 12 := by
  -- proof goes here
  sorry

end Annabelle_saved_12_dollars_1_1470


namespace still_need_more_volunteers_1_1816

def total_volunteers_needed : ℕ := 80
def students_volunteering_per_class : ℕ := 4
def number_of_classes : ℕ := 5
def teacher_volunteers : ℕ := 10
def total_student_volunteers : ℕ := students_volunteering_per_class * number_of_classes
def total_volunteers_so_far : ℕ := total_student_volunteers + teacher_volunteers

theorem still_need_more_volunteers : total_volunteers_needed - total_volunteers_so_far = 50 := by
  sorry

end still_need_more_volunteers_1_1816


namespace part1_part2_1_1045

namespace Problem

open Set

def A : Set ℤ := {x | -6 ≤ x ∧ x ≤ 6}
def B : Set ℤ := {1, 2, 3}
def C : Set ℤ := {3, 4, 5, 6}

-- Part (1)
theorem part1 : A ∩ (B ∩ C) = {3} := by 
  sorry

-- Part (2)
theorem part2 : A ∩ (A \ (B ∪ C)) = {-6, -5, -4, -3, -2, -1, 0} := by 
  sorry

end Problem

end part1_part2_1_1045


namespace water_in_pool_after_35_days_1_1787

theorem water_in_pool_after_35_days :
  ∀ (initial_amount : ℕ) (evap_rate : ℕ) (cycle_days : ℕ) (add_amount : ℕ) (total_days : ℕ),
  initial_amount = 300 → evap_rate = 1 → cycle_days = 5 → add_amount = 5 → total_days = 35 →
  initial_amount - evap_rate * total_days + (total_days / cycle_days) * add_amount = 300 :=
by
  intros initial_amount evap_rate cycle_days add_amount total_days h₁ h₂ h₃ h₄ h₅
  sorry

end water_in_pool_after_35_days_1_1787


namespace rate_of_mangoes_per_kg_1_1228

variable (grapes_qty : ℕ := 8)
variable (grapes_rate_per_kg : ℕ := 70)
variable (mangoes_qty : ℕ := 9)
variable (total_amount_paid : ℕ := 1055)

theorem rate_of_mangoes_per_kg :
  (total_amount_paid - grapes_qty * grapes_rate_per_kg) / mangoes_qty = 55 :=
by
  sorry

end rate_of_mangoes_per_kg_1_1228


namespace geometric_sum_first_six_terms_1_1824

variable (a_n : ℕ → ℝ)

axiom geometric_seq (r a1 : ℝ) : ∀ n, a_n n = a1 * r ^ (n - 1)
axiom a2_val : a_n 2 = 2
axiom a5_val : a_n 5 = 16

theorem geometric_sum_first_six_terms (S6 : ℝ) : S6 = 1 * (1 - 2^6) / (1 - 2) := by
  sorry

end geometric_sum_first_six_terms_1_1824


namespace certain_number_divisible_1_1010

theorem certain_number_divisible (x : ℤ) (n : ℤ) (h1 : 0 < n ∧ n < 11) (h2 : x - n = 11 * k) (h3 : n = 1) : x = 12 :=
by sorry

end certain_number_divisible_1_1010


namespace geometric_seq_value_1_1422

theorem geometric_seq_value (a : ℕ → ℝ) (h : a 4 + a 8 = -2) :
  a 6 * (a 2 + 2 * a 6 + a 10) = 4 :=
sorry

end geometric_seq_value_1_1422


namespace incircle_hexagon_area_ratio_1_1748

noncomputable def area_hexagon (s : ℝ) : ℝ :=
  (3 * Real.sqrt 3 / 2) * s^2

noncomputable def radius_incircle (s : ℝ) : ℝ :=
  (s * Real.sqrt 3) / 2

noncomputable def area_incircle (r : ℝ) : ℝ :=
  Real.pi * r^2

noncomputable def area_ratio (s : ℝ) : ℝ :=
  let A_hexagon := area_hexagon s
  let r := radius_incircle s
  let A_incircle := area_incircle r
  A_incircle / A_hexagon

theorem incircle_hexagon_area_ratio (s : ℝ) (h : s = 1) :
  area_ratio s = (Real.pi * Real.sqrt 3) / 6 :=
by
  sorry

end incircle_hexagon_area_ratio_1_1748


namespace minimum_value_inequality_1_1053

theorem minimum_value_inequality (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z)
    (h : x + 2 * y + 3 * z = 1) :
  (16 / x^3 + 81 / (8 * y^3) + 1 / (27 * z^3)) ≥ 1296 := sorry

end minimum_value_inequality_1_1053


namespace perimeter_large_star_1_1237

theorem perimeter_large_star (n m : ℕ) (P : ℕ)
  (triangle_perimeter : ℕ) (quad_perimeter : ℕ) (small_star_perimeter : ℕ)
  (hn : n = 5) (hm : m = 5)
  (h_triangle_perimeter : triangle_perimeter = 7)
  (h_quad_perimeter : quad_perimeter = 18)
  (h_small_star_perimeter : small_star_perimeter = 3) :
  m * quad_perimeter + small_star_perimeter = n * triangle_perimeter + P → P = 58 :=
by 
  -- Placeholder proof
  sorry

end perimeter_large_star_1_1237


namespace horse_revolutions_1_1202

theorem horse_revolutions (r1 r2  : ℝ) (rev1 rev2 : ℕ)
  (h1 : r1 = 30) (h2 : rev1 = 20) (h3 : r2 = 10) : rev2 = 60 :=
by
  sorry

end horse_revolutions_1_1202


namespace polynomial_coeff_fraction_eq_neg_122_div_121_1_1467

theorem polynomial_coeff_fraction_eq_neg_122_div_121
  (a0 a1 a2 a3 a4 a5 : ℤ)
  (h1 : (2 - 1) ^ 5 = a0 + a1 * 1 + a2 * 1^2 + a3 * 1^3 + a4 * 1^4 + a5 * 1^5)
  (h2 : (2 - (-1)) ^ 5 = a0 + a1 * (-1) + a2 * (-1)^2 + a3 * (-1)^3 + a4 * (-1)^4 + a5 * (-1)^5)
  (h_sum1 : a0 + a1 + a2 + a3 + a4 + a5 = 1)
  (h_sum2 : a0 - a1 + a2 - a3 + a4 - a5 = 243) :
  (a0 + a2 + a4) / (a1 + a3 + a5) = - 122 / 121 :=
sorry

end polynomial_coeff_fraction_eq_neg_122_div_121_1_1467


namespace intersection_complement_1_1425

open Set

def M : Set ℝ := {x | 0 < x ∧ x < 3}
def N : Set ℝ := {x | 2 < x}
def R_complement_N : Set ℝ := {x | x ≤ 2}

theorem intersection_complement : M ∩ R_complement_N = {x | 0 < x ∧ x ≤ 2} :=
by
  sorry

end intersection_complement_1_1425


namespace gnomes_telling_the_truth_1_1134

-- Conditions
def gnome_height_claims : List ℕ := [60, 61, 62, 63, 64, 65, 66]

-- Question and Proof Problem in Lean 4
theorem gnomes_telling_the_truth :
  (∀ (actual_heights : List ℕ), 
    actual_heights.length = 7 →
    (∀ i, i < 7 → i > 0 → 
    actual_heights.get! i > actual_heights.get! (i - 1) → gnome_height_claims.get! i ≠ actual_heights.get! i)) →
  -- conclusion
  (∃ count, count = 1) :=
by
  sorry

end gnomes_telling_the_truth_1_1134


namespace find_rth_term_1_1484

def S (n : ℕ) : ℕ := 2 * n + 3 * (n^3)

def a (r : ℕ) : ℕ := S r - S (r - 1)

theorem find_rth_term (r : ℕ) : a r = 9 * r^2 - 9 * r + 5 := by
  sorry

end find_rth_term_1_1484


namespace hayley_friends_1_1339

theorem hayley_friends (total_stickers : ℕ) (stickers_per_friend : ℕ) (h1 : total_stickers = 72) (h2 : stickers_per_friend = 8) : (total_stickers / stickers_per_friend) = 9 :=
by
  sorry

end hayley_friends_1_1339


namespace number_of_boys_in_second_class_1_1753

def boys_in_first_class : ℕ := 28
def portion_of_second_class (b2 : ℕ) : ℚ := 7 / 8 * b2

theorem number_of_boys_in_second_class (b2 : ℕ) (h : portion_of_second_class b2 = boys_in_first_class) : b2 = 32 :=
by 
  sorry

end number_of_boys_in_second_class_1_1753


namespace find_numbers_1_1489

-- Define the conditions
def geometric_mean_condition (a b : ℝ) : Prop :=
  a * b = 3

def harmonic_mean_condition (a b : ℝ) : Prop :=
  2 / (1 / a + 1 / b) = 3 / 2

-- State the theorem to be proven
theorem find_numbers (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  geometric_mean_condition a b ∧ harmonic_mean_condition a b → (a = 1 ∧ b = 3) ∨ (a = 3 ∧ b = 1) := 
by 
  sorry

end find_numbers_1_1489


namespace least_n_condition_1_1225

-- Define the conditions and the question in Lean 4
def jackson_position (n : ℕ) : ℕ := sorry  -- Defining the position of Jackson after n steps

def expected_value (n : ℕ) : ℝ := sorry  -- Defining the expected value E_n

theorem least_n_condition : ∃ n : ℕ, (1 / expected_value n > 2017) ∧ (∀ m < n, 1 / expected_value m ≤ 2017) ∧ n = 13446 :=
by {
  -- Jackson starts at position 1
  -- The conditions described in the problem will be formulated here
  -- We need to show that the least n such that 1 / E_n > 2017 is 13446
  sorry
}

end least_n_condition_1_1225


namespace exists_integer_K_1_1556

theorem exists_integer_K (Z : ℕ) (K : ℕ) : 
  1000 < Z ∧ Z < 2000 ∧ Z = K^4 → 
  ∃ K, K = 6 := 
by
  sorry

end exists_integer_K_1_1556


namespace polynomial_solution_1_1593

theorem polynomial_solution (P : ℝ → ℝ) (hP : ∀ x : ℝ, (x + 1) * P (x - 1) + (x - 1) * P (x + 1) = 2 * x * P x) :
  ∃ (a d : ℝ), ∀ x : ℝ, P x = a * x^3 - a * x + d := 
sorry

end polynomial_solution_1_1593


namespace butterflies_count_1_1071

theorem butterflies_count (total_black_dots : ℕ) (black_dots_per_butterfly : ℕ) 
                          (h1 : total_black_dots = 4764) 
                          (h2 : black_dots_per_butterfly = 12) :
                          total_black_dots / black_dots_per_butterfly = 397 :=
by
  sorry

end butterflies_count_1_1071


namespace relationship_among_a_b_c_1_1374

-- Defining the properties and conditions of the function
def is_even (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (-x) = f x

-- Defining the function f based on the condition
noncomputable def f (x m : ℝ) : ℝ := 2 ^ |x - m| - 1

-- Defining the constants a, b, c
noncomputable def a : ℝ := f (Real.log 3 / Real.log 0.5) 0
noncomputable def b : ℝ := f (Real.log 5 / Real.log 2) 0
noncomputable def c : ℝ := f 0 0

-- The theorem stating the relationship among a, b, and c
theorem relationship_among_a_b_c : c < a ∧ a < b := by
  sorry

end relationship_among_a_b_c_1_1374


namespace nth_term_correct_1_1336

noncomputable def term_in_sequence (n : ℕ) : ℚ :=
  2^n / (2^n + 3)

theorem nth_term_correct (n : ℕ) : term_in_sequence n = 2^n / (2^n + 3) :=
by
  sorry

end nth_term_correct_1_1336


namespace decimal_division_1_1047

theorem decimal_division : (0.05 : ℝ) / (0.005 : ℝ) = 10 := 
by 
  sorry

end decimal_division_1_1047


namespace find_n_1_1065

def sum_for (x : ℕ) : ℕ :=
  if x > 1 then (List.range (2*x)).sum else 0

theorem find_n (n : ℕ) (h : n * (sum_for 4) = 360) : n = 10 :=
by
  sorry

end find_n_1_1065


namespace find_reflection_line_1_1468

-- Definition of the original and reflected vertices
structure Point :=
  (x : ℝ)
  (y : ℝ)

def D : Point := {x := 1, y := 2}
def E : Point := {x := 6, y := 7}
def F : Point := {x := -5, y := 5}
def D' : Point := {x := 1, y := -4}
def E' : Point := {x := 6, y := -9}
def F' : Point := {x := -5, y := -7}

theorem find_reflection_line (M : ℝ) :
  (D.y + D'.y) / 2 = M ∧ (E.y + E'.y) / 2 = M ∧ (F.y + F'.y) / 2 = M → M = -1 :=
by
  intros
  sorry

end find_reflection_line_1_1468


namespace carina_coffee_1_1392

def total_coffee (t f : ℕ) : ℕ := 10 * t + 5 * f

theorem carina_coffee (t : ℕ) (h1 : t = 3) (f : ℕ) (h2 : f = t + 2) : total_coffee t f = 55 := by
  sorry

end carina_coffee_1_1392


namespace positive_integer_product_divisibility_1_1856

theorem positive_integer_product_divisibility (x : ℕ → ℕ) (n p k : ℕ)
    (P : ℕ) (hx : ∀ i, 1 ≤ i → i ≤ n → x i < 2 * x 1)
    (hpos : ∀ i, 1 ≤ i → i ≤ n → 0 < x i)
    (hstrict : ∀ i j, 1 ≤ i → i < j → j ≤ n → x i < x j)
    (hn : 3 ≤ n)
    (hp : Nat.Prime p)
    (hk : 0 < k)
    (hP : P = ∏ i in Finset.range n, x (i + 1))
    (hdiv : p ^ k ∣ P) : 
  (P / p^k) ≥ Nat.factorial n := by
  sorry

end positive_integer_product_divisibility_1_1856


namespace wood_burned_afternoon_1_1011

theorem wood_burned_afternoon (burned_morning burned_afternoon bundles_start bundles_end : ℕ) 
  (h_burned_morning : burned_morning = 4)
  (h_bundles_start : bundles_start = 10) 
  (h_bundles_end : bundles_end = 3)
  (total_burned : bundles_start - bundles_end = burned_morning + burned_afternoon) :
  burned_afternoon = 3 :=
by {
  -- Proof placeholder
  sorry
}

end wood_burned_afternoon_1_1011


namespace number_with_150_quarters_is_37_point_5_1_1792

theorem number_with_150_quarters_is_37_point_5 (n : ℝ) (h : n / (1/4) = 150) : n = 37.5 := 
by 
  sorry

end number_with_150_quarters_is_37_point_5_1_1792


namespace increase_80_by_50_percent_1_1118

theorem increase_80_by_50_percent :
  let initial_number : ℕ := 80
  let increase_percentage : ℝ := 0.5
  initial_number + (initial_number * increase_percentage) = 120 :=
by
  sorry

end increase_80_by_50_percent_1_1118


namespace Chloe_final_points_1_1108

-- Define the points scored (or lost) in each round
def round1_points : ℤ := 40
def round2_points : ℤ := 50
def round3_points : ℤ := 60
def round4_points : ℤ := 70
def round5_points : ℤ := -4
def round6_points : ℤ := 80
def round7_points : ℤ := -6

-- Statement to prove: Chloe's total points at the end of the game
theorem Chloe_final_points : 
  round1_points + round2_points + round3_points + round4_points + round5_points + round6_points + round7_points = 290 :=
by
  sorry

end Chloe_final_points_1_1108


namespace sufficient_but_not_necessary_condition_1_1923

theorem sufficient_but_not_necessary_condition (a : ℝ) : (a = 2 → |a| = 2) ∧ (|a| = 2 → a = 2 ∨ a = -2) :=
by
  sorry

end sufficient_but_not_necessary_condition_1_1923


namespace neither_5_nor_6_nice_1200_1_1526

def is_k_nice (N k : ℕ) : Prop := N % k = 1

def count_k_nice_up_to (k n : ℕ) : ℕ :=
(n + (k - 1)) / k

def count_neither_5_nor_6_nice_up_to (n : ℕ) : ℕ :=
  let count_5_nice := count_k_nice_up_to 5 n
  let count_6_nice := count_k_nice_up_to 6 n
  let count_5_and_6_nice := count_k_nice_up_to 30 n
  n - (count_5_nice + count_6_nice - count_5_and_6_nice)

theorem neither_5_nor_6_nice_1200 : count_neither_5_nor_6_nice_up_to 1200 = 800 := 
by
  sorry

end neither_5_nor_6_nice_1200_1_1526


namespace sampling_method_D_is_the_correct_answer_1_1343

def sampling_method_A_is_simple_random_sampling : Prop :=
  false

def sampling_method_B_is_simple_random_sampling : Prop :=
  false

def sampling_method_C_is_simple_random_sampling : Prop :=
  false

def sampling_method_D_is_simple_random_sampling : Prop :=
  true

theorem sampling_method_D_is_the_correct_answer :
  sampling_method_A_is_simple_random_sampling = false ∧
  sampling_method_B_is_simple_random_sampling = false ∧
  sampling_method_C_is_simple_random_sampling = false ∧
  sampling_method_D_is_simple_random_sampling = true :=
by
  sorry

end sampling_method_D_is_the_correct_answer_1_1343


namespace problem_proof_1_1299

variable {x y z : ℝ}

theorem problem_proof (h1 : 0 < x) (h2 : 0 < y) (h3 : 0 < z) (h4 : x^2 + y^2 + z^2 + 2 * x * y * z = 1) : 2 * (x + y + z) ≤ 3 := 
sorry

end problem_proof_1_1299


namespace meet_at_centroid_1_1316

-- Definitions of positions
def Harry : ℝ × ℝ := (10, -3)
def Sandy : ℝ × ℝ := (2, 7)
def Ron : ℝ × ℝ := (6, 1)

-- Mathematical proof problem statement
theorem meet_at_centroid : 
    (Harry.1 + Sandy.1 + Ron.1) / 3 = 6 ∧ (Harry.2 + Sandy.2 + Ron.2) / 3 = 5 / 3 := 
by
  sorry

end meet_at_centroid_1_1316


namespace probability_two_heads_one_tail_in_three_tosses_1_1567

theorem probability_two_heads_one_tail_in_three_tosses
(P : ℕ → Prop) (pr : ℤ) : 
  (∀ n, P n → pr = 1 / 2) -> 
  P 3 → pr = 3 / 8 :=
by
  sorry

end probability_two_heads_one_tail_in_three_tosses_1_1567


namespace ratio_greater_than_one_ratio_greater_than_one_neg_1_1426

theorem ratio_greater_than_one (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a / b > 1) : a > b :=
by
  sorry

theorem ratio_greater_than_one_neg (a b : ℝ) (h1 : a < 0) (h2 : b < 0) (h3 : a / b > 1) : a < b :=
by
  sorry

end ratio_greater_than_one_ratio_greater_than_one_neg_1_1426


namespace find_roots_1_1869

noncomputable def P (x : ℝ) : ℝ := x^4 - 3 * x^3 + 3 * x^2 - x - 6

theorem find_roots : {x : ℝ | P x = 0} = {-1, 1, 2} :=
by
  sorry

end find_roots_1_1869


namespace least_positive_integer_condition_1_1438

theorem least_positive_integer_condition
  (a : ℤ) (ha1 : a % 4 = 1) (ha2 : a % 5 = 2) (ha3 : a % 6 = 3) :
  a > 0 → a = 57 :=
by
  intro ha_pos
  -- Proof omitted for brevity
  sorry

end least_positive_integer_condition_1_1438


namespace smallest_b_factors_1_1393

theorem smallest_b_factors (p q b : ℤ) (hpq : p * q = 1764) (hb : b = p + q) (hposp : p > 0) (hposq : q > 0) :
  b = 84 :=
by
  sorry

end smallest_b_factors_1_1393


namespace find_a8_a12_sum_1_1930

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

theorem find_a8_a12_sum
  (a : ℕ → ℝ) 
  (h_geom : geometric_sequence a) 
  (h1 : a 2 + a 6 = 3) 
  (h2 : a 6 + a 10 = 12) : 
  a 8 + a 12 = 24 :=
sorry

end find_a8_a12_sum_1_1930


namespace intersection_M_N_1_1948

def M : Set ℕ := {3, 5, 6, 8}
def N : Set ℕ := {4, 5, 7, 8}

theorem intersection_M_N : M ∩ N = {5, 8} :=
  sorry

end intersection_M_N_1_1948


namespace expression_equals_five_1_1794

theorem expression_equals_five (a : ℝ) (h : 2 * a^2 - 3 * a + 4 = 5) : 7 + 6 * a - 4 * a^2 = 5 :=
by
  sorry

end expression_equals_five_1_1794


namespace tens_digit_of_3_pow_2010_1_1544

theorem tens_digit_of_3_pow_2010 : (3^2010 / 10) % 10 = 4 := by
  sorry

end tens_digit_of_3_pow_2010_1_1544


namespace parabola_directrix_distance_1_1049

theorem parabola_directrix_distance (m : ℝ) (h : |1 / (4 * m)| = 2) : m = 1/8 ∨ m = -1/8 :=
by { sorry }

end parabola_directrix_distance_1_1049


namespace seats_not_occupied_1_1832

def seats_per_row : ℕ := 8
def total_rows : ℕ := 12
def seat_utilization_ratio : ℚ := 3 / 4

theorem seats_not_occupied : 
  (seats_per_row * total_rows) - (seats_per_row * seat_utilization_ratio * total_rows) = 24 := 
by
  sorry

end seats_not_occupied_1_1832


namespace determine_a_1_1625

def A := {x : ℝ | x < 6}
def B (a : ℝ) := {x : ℝ | x - a < 0}

theorem determine_a (a : ℝ) (h : A ⊆ B a) : 6 ≤ a := 
sorry

end determine_a_1_1625


namespace max_cars_and_quotient_1_1372

-- Definition of the problem parameters
def car_length : ℕ := 5
def speed_per_car_length : ℕ := 10
def hour_in_seconds : ℕ := 3600
def one_kilometer_in_meters : ℕ := 1000
def distance_in_meters_per_hour (n : ℕ) : ℕ := (10 * n) * one_kilometer_in_meters
def unit_distance (n : ℕ) : ℕ := car_length * (n + 1)

-- Hypotheses
axiom car_spacing : ∀ n : ℕ, unit_distance n = car_length * (n + 1)
axiom car_speed : ∀ n : ℕ, distance_in_meters_per_hour n = (10 * n) * one_kilometer_in_meters

-- Maximum whole number of cars M that can pass in one hour and the quotient when M is divided by 10
theorem max_cars_and_quotient : ∃ (M : ℕ), M = 3000 ∧ M / 10 = 300 := by
  sorry

end max_cars_and_quotient_1_1372


namespace reliability_is_correct_1_1394

-- Define the probabilities of each switch functioning properly.
def P_A : ℝ := 0.9
def P_B : ℝ := 0.8
def P_C : ℝ := 0.7

-- Define the system reliability.
def reliability : ℝ := P_A * P_B * P_C

-- The theorem stating the reliability of the system.
theorem reliability_is_correct : reliability = 0.504 := by
  sorry

end reliability_is_correct_1_1394


namespace problem1_problem2_1_1131

def setA (a : ℝ) : Set ℝ := {x | a ≤ x ∧ x ≤ a + 3}
def setB : Set ℝ := {x | x > 1 ∨ x < -6}

theorem problem1 (a : ℝ) : (setA a ∩ setB = ∅) → (-6 ≤ a ∧ a ≤ -2) := by
  intro h
  sorry

theorem problem2 (a : ℝ) : (setA a ∪ setB = setB) → (a < -9 ∨ a > 1) := by
  intro h
  sorry

end problem1_problem2_1_1131


namespace total_circles_1_1008

theorem total_circles (n : ℕ) (h1 : ∀ k : ℕ, k = n + 14 → n^2 = (k * (k + 1) / 2)) : 
  n = 35 → n^2 = 1225 :=
by
  sorry

end total_circles_1_1008


namespace johns_father_age_1_1399

variable {Age : Type} [OrderedRing Age]
variables (J M F : Age)

def john_age := J
def mother_age := M
def father_age := F

def john_younger_than_father (F J : Age) : Prop := F = 2 * J
def father_older_than_mother (F M : Age) : Prop := F = M + 4
def age_difference_between_john_and_mother (M J : Age) : Prop := M = J + 16

-- The question to be proved in Lean:
theorem johns_father_age :
  john_younger_than_father F J →
  father_older_than_mother F M →
  age_difference_between_john_and_mother M J →
  F = 40 := 
by
  intros h1 h2 h3
  sorry

end johns_father_age_1_1399


namespace mans_rate_in_still_water_1_1622

-- Definitions from the conditions
def speed_with_stream : ℝ := 10
def speed_against_stream : ℝ := 6

-- The statement to prove the man's rate in still water is as expected.
theorem mans_rate_in_still_water : (speed_with_stream + speed_against_stream) / 2 = 8 := by
  sorry

end mans_rate_in_still_water_1_1622


namespace moving_circle_fixed_point_1_1932

def parabola (p : ℝ × ℝ) : Prop := p.2^2 = 4 * p.1

def tangent_line (c : ℝ × ℝ) (r : ℝ) : Prop :=
  abs (c.1 + 1) = r

theorem moving_circle_fixed_point :
  ∀ (c : ℝ × ℝ) (r : ℝ),
    parabola c →
    tangent_line c r →
    (1, 0) ∈ {p : ℝ × ℝ | dist c p = r} :=
by
  intro c r hc ht
  sorry

end moving_circle_fixed_point_1_1932


namespace average_score_1_1354

theorem average_score (s1 s2 s3 : ℕ) (n : ℕ) (h1 : s1 = 115) (h2 : s2 = 118) (h3 : s3 = 115) (h4 : n = 3) :
    (s1 + s2 + s3) / n = 116 :=
by
    sorry

end average_score_1_1354


namespace max_value_m_1_1213

variable {a b m : ℝ}

theorem max_value_m (ha : a > 0) (hb : b > 0) 
  (h : ∀ a b, (3 / a) + (1 / b) ≥ m / (a + 3 * b)) : m ≤ 12 :=
by 
  sorry

end max_value_m_1_1213


namespace cost_of_bananas_and_cantaloupe_1_1060

-- Define variables representing the prices
variables (a b c d : ℝ)

-- Define the given conditions as hypotheses
def conditions : Prop :=
  a + b + c + d = 33 ∧
  d = 3 * a ∧
  c = a + 2 * b

-- State the main theorem
theorem cost_of_bananas_and_cantaloupe (h : conditions a b c d) : b + c = 13 :=
by {
  sorry
}

end cost_of_bananas_and_cantaloupe_1_1060


namespace find_a_plus_b_1_1999

noncomputable def f (a b x : ℝ) : ℝ := (a * x^3) / 3 - b * x^2 + a^2 * x - 1 / 3
noncomputable def f_prime (a b x : ℝ) : ℝ := a * x^2 - 2 * b * x + a^2

theorem find_a_plus_b 
  (a b : ℝ)
  (h_deriv : f_prime a b 1 = 0)
  (h_extreme : f a b 1 = 0) :
  a + b = -7 / 9 := 
sorry

end find_a_plus_b_1_1999


namespace maximize_tetrahedron_volume_1_1516

noncomputable def volume_of_tetrahedron (a : ℝ) : ℝ :=
  a / 6

theorem maximize_tetrahedron_volume (a : ℝ) (h_a : 0 < a) 
  (P Q X Y : ℝ × ℝ × ℝ) (h_PQ : dist P Q = 1) (h_XY : dist X Y = 1) :
  volume_of_tetrahedron a = a / 6 :=
by
  sorry

end maximize_tetrahedron_volume_1_1516


namespace find_n_1_1151

theorem find_n (n : ℕ) (h : 2 ^ 3 * 5 * n = Nat.factorial 10) : n = 45360 :=
sorry

end find_n_1_1151


namespace preservation_time_at_33_degrees_1_1604

noncomputable def preservation_time (x : ℝ) (k : ℝ) (b : ℝ) : ℝ :=
  Real.exp (k * x + b)

theorem preservation_time_at_33_degrees (k b : ℝ) 
  (h1 : Real.exp b = 192)
  (h2 : Real.exp (22 * k + b) = 48) :
  preservation_time 33 k b = 24 := by
  sorry

end preservation_time_at_33_degrees_1_1604


namespace arithmetic_sequence_sum_1_1732

theorem arithmetic_sequence_sum :
  ∀ (a₁ : ℕ) (d : ℕ) (a_n : ℕ) (n : ℕ),
    a₁ = 1 →
    d = 2 →
    a_n = 29 →
    a_n = a₁ + (n - 1) * d →
    (n : ℕ) = 15 →
    (∑ k in Finset.range n, a₁ + k * d) = 225 :=
by
  intros a₁ d a_n n h₁ h_d hₐ h_an h_n
  sorry

end arithmetic_sequence_sum_1_1732


namespace half_of_number_1_1334

theorem half_of_number (x : ℝ) (h : (4 / 15 * 5 / 7 * x - 4 / 9 * 2 / 5 * x = 8)) : (1 / 2 * x = 315) :=
sorry

end half_of_number_1_1334


namespace lunch_break_duration_1_1355

theorem lunch_break_duration (m a : ℝ) (L : ℝ) :
  (9 - L) * (m + a) = 0.6 → 
  (7 - L) * a = 0.3 → 
  (5 - L) * m = 0.1 → 
  L = 42 / 60 :=
by sorry

end lunch_break_duration_1_1355


namespace find_interest_rate_1_1588

noncomputable def amount : ℝ := 896
noncomputable def principal : ℝ := 799.9999999999999
noncomputable def time : ℝ := 2 + 2 / 5
noncomputable def interest : ℝ := amount - principal
noncomputable def rate : ℝ := interest / (principal * time)

theorem find_interest_rate :
  rate * 100 = 5 := by
  sorry

end find_interest_rate_1_1588


namespace Julie_can_print_complete_newspapers_1_1660

def sheets_in_box_A : ℕ := 4 * 200
def sheets_in_box_B : ℕ := 3 * 350
def total_sheets : ℕ := sheets_in_box_A + sheets_in_box_B

def front_section_sheets : ℕ := 10
def sports_section_sheets : ℕ := 7
def arts_section_sheets : ℕ := 5
def events_section_sheets : ℕ := 3

def sheets_per_newspaper : ℕ := front_section_sheets + sports_section_sheets + arts_section_sheets + events_section_sheets

theorem Julie_can_print_complete_newspapers : total_sheets / sheets_per_newspaper = 74 := by
  sorry

end Julie_can_print_complete_newspapers_1_1660


namespace series_equality_1_1163

theorem series_equality :
  (∑ n in Finset.range 200, (-1)^(n+1) * (1:ℚ) / (n+1)) = (∑ n in Finset.range 100, 1 / (100 + n + 1)) :=
by sorry

end series_equality_1_1163


namespace passing_marks_1_1834

theorem passing_marks (T P : ℝ) (h1 : 0.30 * T = P - 30) (h2 : 0.45 * T = P + 15) : P = 120 := 
by
  sorry

end passing_marks_1_1834


namespace painting_perimeter_1_1844

-- Definitions for the problem conditions
def frame_thickness : ℕ := 3
def frame_area : ℕ := 108

-- Declaration that expresses the given conditions and the problem's conclusion
theorem painting_perimeter {w h : ℕ} (h_frame : (w + 2 * frame_thickness) * (h + 2 * frame_thickness) - w * h = frame_area) :
  2 * (w + h) = 24 :=
by
  sorry

end painting_perimeter_1_1844


namespace lines_coplanar_iff_k_eq_neg2_1_1465

noncomputable def line1 (s k : ℝ) : ℝ × ℝ × ℝ :=
(2 + s, 4 - k * s, 2 + k * s)

noncomputable def line2 (t : ℝ) : ℝ × ℝ × ℝ :=
(t, 2 + 2 * t, 3 - t)

theorem lines_coplanar_iff_k_eq_neg2 :
  (∃ s t : ℝ, line1 s k = line2 t) → k = -2 :=
by
  sorry

end lines_coplanar_iff_k_eq_neg2_1_1465


namespace remainder_of_exponentiation_is_correct_1_1561

-- Define the given conditions
def modulus := 500
def exponent := 5 ^ (5 ^ 5)
def carmichael_500 := 100
def carmichael_100 := 20

-- Prove the main theorem
theorem remainder_of_exponentiation_is_correct :
  (5 ^ exponent) % modulus = 125 := 
by
  -- Skipping the proof
  sorry

end remainder_of_exponentiation_is_correct_1_1561


namespace find_m_for_even_function_1_1268

def f (x : ℝ) (m : ℝ) := x^2 + (m - 1) * x + 3

theorem find_m_for_even_function : ∃ m : ℝ, (∀ x : ℝ, f (-x) m = f x m) ∧ m = 1 :=
sorry

end find_m_for_even_function_1_1268


namespace calculate_expression_1_1360

theorem calculate_expression :
  ((2000000000000 - 1234567890123) * 3 = 2296296329631) :=
by 
  sorry

end calculate_expression_1_1360


namespace jack_last_10_shots_made_1_1407

theorem jack_last_10_shots_made (initial_shots : ℕ) (initial_percentage : ℚ)
  (additional_shots : ℕ) (new_percentage : ℚ)
  (initial_successful_shots : initial_shots * initial_percentage = 18)
  (total_shots : initial_shots + additional_shots = 40)
  (total_successful_shots : (initial_shots + additional_shots) * new_percentage = 25) :
  ∃ x : ℕ, x = 7 := by
sorry

end jack_last_10_shots_made_1_1407


namespace intersection_complement_1_1129

def U : Set ℕ := {0, 1, 2, 3, 4}
def A : Set ℕ := {1, 2, 3}
def B : Set ℕ := {2, 4}

theorem intersection_complement :
  A ∩ (U \ B) = {1, 3} :=
by {
  -- To ensure the validity of the theorem, the proof goes here
  sorry
}

end intersection_complement_1_1129


namespace solve_inequality_1_find_range_of_a_1_1692

def f (x : ℝ) : ℝ := |x - 1| + |x + 2|

theorem solve_inequality_1 :
  {x : ℝ | f x ≥ 5} = {x : ℝ | x ≤ -3} ∪ {x : ℝ | x ≥ 2} :=
by
  sorry
  
theorem find_range_of_a (a : ℝ) :
  (∀ x : ℝ, f x > a^2 - 2 * a - 5) ↔ -2 < a ∧ a < 4 :=
by
  sorry

end solve_inequality_1_find_range_of_a_1_1692


namespace square_of_1024_1_1595

theorem square_of_1024 : 1024^2 = 1048576 :=
by
  sorry

end square_of_1024_1_1595


namespace lambda_range_1_1100

noncomputable def sequence_a (n : ℕ) : ℝ :=
  if n = 0 then 1 else
  sequence_a (n - 1) / (sequence_a (n - 1) + 2)

noncomputable def sequence_b (lambda : ℝ) (n : ℕ) : ℝ :=
  if n = 0 then -3/2 * lambda else
  (n - 2 * lambda) * (1 / sequence_a (n - 1) + 1)

def is_monotonically_increasing (seq : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, n > 0 → seq (n+1) > seq n

theorem lambda_range (lambda : ℝ) (hn : is_monotonically_increasing (sequence_b lambda)) : lambda < 4/5 := sorry

end lambda_range_1_1100


namespace num_occupied_third_floor_rooms_1_1059

-- Definitions based on conditions
def first_floor_rent : Int := 15
def second_floor_rent : Int := 20
def third_floor_rent : Int := 2 * first_floor_rent
def rooms_per_floor : Int := 3
def monthly_earnings : Int := 165

-- The proof statement
theorem num_occupied_third_floor_rooms : 
  let total_full_occupancy_cost := rooms_per_floor * first_floor_rent + rooms_per_floor * second_floor_rent + rooms_per_floor * third_floor_rent
  let revenue_difference := total_full_occupancy_cost - monthly_earnings
  revenue_difference / third_floor_rent = 1 → rooms_per_floor - revenue_difference / third_floor_rent = 2 :=
by
  sorry

end num_occupied_third_floor_rooms_1_1059


namespace distinct_ordered_pairs_solution_1_1193

theorem distinct_ordered_pairs_solution :
  (∃ n : ℕ, ∀ x y : ℕ, (x > 0 ∧ y > 0 ∧ x^4 * y^4 - 24 * x^2 * y^2 + 35 = 0) ↔ n = 1) :=
sorry

end distinct_ordered_pairs_solution_1_1193


namespace transmitted_word_is_PAROHOD_1_1828

-- Define the binary representation of each letter in the Russian alphabet.
def binary_repr : String → String
| "А" => "00000"
| "Б" => "00001"
| "В" => "00011"
| "Г" => "00111"
| "Д" => "00101"
| "Е" => "00110"
| "Ж" => "01100"
| "З" => "01011"
| "И" => "01001"
| "Й" => "11000"
| "К" => "01010"
| "Л" => "01011"
| "М" => "01101"
| "Н" => "01111"
| "О" => "01100"
| "П" => "01110"
| "Р" => "01010"
| "С" => "01100"
| "Т" => "01001"
| "У" => "01111"
| "Ф" => "11101"
| "Х" => "11011"
| "Ц" => "11100"
| "Ч" => "10111"
| "Ш" => "11110"
| "Щ" => "11110"
| "Ь" => "00010"
| "Ы" => "00011"
| "Ъ" => "00101"
| "Э" => "11100"
| "Ю" => "01111"
| "Я" => "11111"
| _  => "00000" -- default case

-- Define the received scrambled word.
def received_word : List String := ["Э", "А", "В", "Щ", "О", "Щ", "И"]

-- The target transmitted word is "ПАРОХОД" which corresponds to ["П", "А", "Р", "О", "Х", "О", "Д"]
def transmitted_word : List String := ["П", "А", "Р", "О", "Х", "О", "Д"]

-- Lean 4 proof statement to show that the received scrambled word reconstructs to the transmitted word.
theorem transmitted_word_is_PAROHOD (b_repr : String → String)
(received : List String) :
  received = received_word →
  transmitted_word.map b_repr = received.map b_repr → transmitted_word = ["П", "А", "Р", "О", "Х", "О", "Д"] :=
by 
  intros h_received h_repr_eq
  exact sorry

end transmitted_word_is_PAROHOD_1_1828


namespace angle_complement_1_1506

-- Conditions: The complement of angle A is 60 degrees
def complement (α : ℝ) : ℝ := 90 - α 

theorem angle_complement (A : ℝ) : complement A = 60 → A = 30 :=
by
  sorry

end angle_complement_1_1506


namespace minimum_norm_of_v_1_1493

open Real 

-- Define the vector v and condition
noncomputable def v : ℝ × ℝ := sorry

-- Define the condition
axiom v_condition : ‖(v.1 + 4, v.2 + 2)‖ = 10

-- The statement that we need to prove
theorem minimum_norm_of_v : ‖v‖ = 10 - 2 * sqrt 5 :=
by
  sorry

end minimum_norm_of_v_1_1493


namespace find_principal_1_1786

theorem find_principal (P : ℝ) (r : ℝ) (t : ℝ) (CI SI : ℝ) 
  (h_r : r = 0.20) 
  (h_t : t = 2) 
  (h_diff : CI - SI = 144) 
  (h_CI : CI = P * (1 + r)^t - P) 
  (h_SI : SI = P * r * t) : 
  P = 3600 :=
by
  sorry

end find_principal_1_1786


namespace relationship_m_n_1_1192

variable (a b : ℝ)
variable (m n : ℝ)

theorem relationship_m_n (h1 : a > b) (h2 : b > 0) (hm : m = Real.sqrt a - Real.sqrt b) (hn : n = Real.sqrt (a - b)) : m < n := sorry

end relationship_m_n_1_1192


namespace find_e_1_1306

noncomputable def f (x : ℝ) (c : ℝ) := 5 * x + 2 * c

noncomputable def g (x : ℝ) (c : ℝ) := c * x^2 + 3

noncomputable def fg (x : ℝ) (c : ℝ) := f (g x c) c

theorem find_e (c : ℝ) (e : ℝ) (h1 : f (g x c) c = 15 * x^2 + e) (h2 : 5 * c = 15) : e = 21 :=
by
  sorry

end find_e_1_1306


namespace tickets_to_be_sold_1_1408

theorem tickets_to_be_sold : 
  let total_tickets := 200
  let jude_tickets := 16
  let andrea_tickets := 4 * jude_tickets
  let sandra_tickets := 2 * jude_tickets + 8
  total_tickets - (jude_tickets + andrea_tickets + sandra_tickets) = 80 := by
  sorry

end tickets_to_be_sold_1_1408


namespace probability_at_least_one_consonant_1_1584

def letters := ["k", "h", "a", "n", "t", "k", "a", "r"]
def consonants := ["k", "h", "n", "t", "r"]
def vowels := ["a", "a"]

def num_letters := 7
def num_consonants := 5
def num_vowels := 2

def probability_no_consonants : ℚ := (num_vowels / num_letters) * ((num_vowels - 1) / (num_letters - 1))

def complement_rule (p: ℚ) : ℚ := 1 - p

theorem probability_at_least_one_consonant :
  complement_rule probability_no_consonants = 20/21 :=
by
  sorry

end probability_at_least_one_consonant_1_1584


namespace gcd_expression_1_1430

theorem gcd_expression (n : ℕ) (h : n > 2) : Nat.gcd (n^5 - 5 * n^3 + 4 * n) 120 = 120 :=
by
  sorry

end gcd_expression_1_1430


namespace angle_Y_measure_1_1287

def hexagon_interior_angle_sum (n : ℕ) : ℕ :=
  180 * (n - 2)

def supplementary (α β : ℕ) : Prop :=
  α + β = 180

def equal_angles (α β γ δ : ℕ) : Prop :=
  α = β ∧ β = γ ∧ γ = δ

theorem angle_Y_measure :
  ∀ (C H E S1 S2 Y : ℕ),
    C = E ∧ E = S1 ∧ S1 = Y →
    supplementary H S2 →
    hexagon_interior_angle_sum 6 = C + H + E + S1 + S2 + Y →
    Y = 135 :=
by
  intros C H E S1 S2 Y h1 h2 h3
  sorry

end angle_Y_measure_1_1287


namespace total_number_of_people_1_1326

variables (A : ℕ) -- Number of adults in the group

-- Conditions
-- Each adult meal costs $8 and the total cost was $72
def cost_per_adult_meal : ℕ := 8
def total_cost : ℕ := 72
def number_of_kids : ℕ := 2

-- Proof problem: Given the conditions, prove the total number of people in the group is 11
theorem total_number_of_people (h : A * cost_per_adult_meal = total_cost) : A + number_of_kids = 11 :=
sorry

end total_number_of_people_1_1326


namespace hexagon_pillar_height_1_1218

noncomputable def height_of_pillar_at_vertex_F (s : ℝ) (hA hB hC : ℝ) (A : ℝ × ℝ) : ℝ :=
  10

theorem hexagon_pillar_height :
  ∀ (s hA hB hC : ℝ) (A : ℝ × ℝ),
  s = 8 ∧ hA = 15 ∧ hB = 10 ∧ hC = 12 ∧ A = (3, 3 * Real.sqrt 3) →
  height_of_pillar_at_vertex_F s hA hB hC A = 10 := by
  sorry

end hexagon_pillar_height_1_1218


namespace side_length_of_smaller_square_1_1648

theorem side_length_of_smaller_square (s : ℝ) (A1 A2 : ℝ) (h1 : 5 * 5 = A1 + A2) (h2 : 2 * A2 = A1 + 25)  : s = 5 * Real.sqrt 3 / 3 :=
by
  sorry

end side_length_of_smaller_square_1_1648


namespace stickers_distribution_1_1657

-- Define the mathematical problem: distributing 10 stickers among 5 sheets with each sheet getting at least one sticker.

def partitions_count (n k : ℕ) : ℕ := sorry

theorem stickers_distribution (n : ℕ) (k : ℕ) (h₁ : n = 10) (h₂ : k = 5) :
  partitions_count (n - k) k = 7 := by
  sorry

end stickers_distribution_1_1657


namespace calc_xy_square_1_1651

theorem calc_xy_square
  (x y z : ℝ)
  (h1 : 2 * x * (y + z) = 1 + y * z)
  (h2 : 1 / x - 2 / y = 3 / 2)
  (h3 : x + y + 1 / 2 = 0) :
  (x + y + z) ^ 2 = 1 :=
by
  sorry

end calc_xy_square_1_1651


namespace bridge_weight_requirement_1_1616

def weight_soda_can : ℕ := 12
def weight_empty_soda_can : ℕ := 2
def num_soda_cans : ℕ := 6

def weight_empty_other_can : ℕ := 3
def num_other_cans : ℕ := 2

def wind_force_eq_soda_cans : ℕ := 2

def total_weight_bridge_must_hold : ℕ :=
  weight_soda_can * num_soda_cans + weight_empty_soda_can * num_soda_cans +
  weight_empty_other_can * num_other_cans +
  wind_force_eq_soda_cans * (weight_soda_can + weight_empty_soda_can)

theorem bridge_weight_requirement :
  total_weight_bridge_must_hold = 118 :=
by
  unfold total_weight_bridge_must_hold weight_soda_can weight_empty_soda_can num_soda_cans
    weight_empty_other_can num_other_cans wind_force_eq_soda_cans
  sorry

end bridge_weight_requirement_1_1616


namespace trajectory_equation_1_1560

-- Define the condition that the distance to the coordinate axes is equal.
def equidistantToAxes (x y : ℝ) : Prop :=
  abs x = abs y

-- State the theorem that we need to prove.
theorem trajectory_equation (x y : ℝ) (h : equidistantToAxes x y) : y^2 = x^2 :=
by sorry

end trajectory_equation_1_1560


namespace milk_left_1_1205

theorem milk_left (initial_milk : ℝ) (given_away : ℝ) (h_initial : initial_milk = 5) (h_given : given_away = 18 / 4) :
  ∃ remaining_milk : ℝ, remaining_milk = initial_milk - given_away ∧ remaining_milk = 1 / 2 :=
by
  use 1 / 2
  sorry

end milk_left_1_1205


namespace max_value_of_f_1_1694

noncomputable def f (x a b : ℝ) := (1 - x ^ 2) * (x ^ 2 + a * x + b)

theorem max_value_of_f (a b : ℝ) (x : ℝ) 
  (h1 : (∀ x, f x 8 15 = (1 - x^2) * (x^2 + 8*x + 15)))
  (h2 : ∀ x, f x a b = f (-(x + 4)) a b) :
  ∃ m, (∀ x, f x 8 15 ≤ m) ∧ m = 16 :=
by
  sorry

end max_value_of_f_1_1694


namespace problem_a_range_1_1710

theorem problem_a_range (a : ℝ) :
  (∀ x : ℝ, (a - 1) * x^2 - 2 * (a - 1) * x - 2 < 0) ↔ (-1 < a ∧ a ≤ 1) :=
by
  sorry

end problem_a_range_1_1710


namespace largest_divisor_for_odd_n_1_1389

noncomputable def is_odd (n : ℤ) : Prop := ∃ k : ℤ, n = 2 * k + 1

theorem largest_divisor_for_odd_n (n : ℤ) (h : is_odd n ∧ n > 0) : 
  15 ∣ (n + 3) * (n + 5) * (n + 7) * (n + 9) * (n + 11) := 
by 
  sorry

end largest_divisor_for_odd_n_1_1389


namespace christian_age_in_eight_years_1_1477

-- Definitions from the conditions
def christian_current_age : ℕ := 72
def brian_age_in_eight_years : ℕ := 40

-- Theorem to prove
theorem christian_age_in_eight_years : ∃ (age : ℕ), age = christian_current_age + 8 ∧ age = 80 := by
  sorry

end christian_age_in_eight_years_1_1477


namespace angle_measure_1_1960

variable (x : ℝ)

noncomputable def is_supplement (x : ℝ) : Prop := 180 - x = 3 * (90 - x) - 60

theorem angle_measure : is_supplement x → x = 15 :=
by
  sorry

end angle_measure_1_1960


namespace sufficient_but_not_necessary_1_1769

theorem sufficient_but_not_necessary (x : ℝ) : (x > 1/2 → 2 * x^2 + x - 1 > 0) ∧ ¬(2 * x^2 + x - 1 > 0 → x > 1 / 2) := 
by
  sorry

end sufficient_but_not_necessary_1_1769


namespace tangent_line_equation_1_1578

noncomputable def y (x : ℝ) := (2 * x - 1) / (x + 2)
def point : ℝ × ℝ := (-1, -3)
def tangent_eq (x y : ℝ) : Prop := 5 * x - y + 2 = 0

theorem tangent_line_equation :
  tangent_eq (-1) (-3) := 
sorry

end tangent_line_equation_1_1578


namespace plane_centroid_1_1133

theorem plane_centroid (a b : ℝ) (h : 1 / a ^ 2 + 1 / b ^ 2 + 1 / 25 = 1 / 4) :
  let p := a / 3
  let q := b / 3
  let r := 5 / 3
  (1 / p ^ 2 + 1 / q ^ 2 + 1 / r ^ 2) = 369 / 400 :=
by
  sorry

end plane_centroid_1_1133


namespace depth_of_well_1_1160

theorem depth_of_well (d : ℝ) (t1 t2 : ℝ)
  (h1 : d = 15 * t1^2)
  (h2 : t2 = d / 1100)
  (h3 : t1 + t2 = 9.5) :
  d = 870.25 := 
sorry

end depth_of_well_1_1160


namespace prove_scientific_notation_1_1089

def scientific_notation_correct : Prop :=
  340000 = 3.4 * (10 ^ 5)

theorem prove_scientific_notation : scientific_notation_correct :=
  by
    sorry

end prove_scientific_notation_1_1089


namespace commute_time_difference_1_1254

theorem commute_time_difference (x y : ℝ) 
  (h1 : x + y = 39)
  (h2 : (x - 10)^2 + (y - 10)^2 = 10) :
  |x - y| = 4 :=
by
  sorry

end commute_time_difference_1_1254


namespace fraction_meaningful_1_1558

theorem fraction_meaningful (a : ℝ) : (∃ b, b = 2 / (a + 1)) → a ≠ -1 :=
by
  sorry

end fraction_meaningful_1_1558


namespace smallest_value_of_y_1_1690

theorem smallest_value_of_y : 
  (∃ y : ℝ, 6 * y^2 - 41 * y + 55 = 0 ∧ ∀ z : ℝ, 6 * z^2 - 41 * z + 55 = 0 → y ≤ z) →
  ∃ y : ℝ, y = 2.5 :=
by sorry

end smallest_value_of_y_1_1690


namespace a_n_formula_T_n_formula_1_1446

variable (a : Nat → Int) (b : Nat → Int)
variable (S : Nat → Int) (T : Nat → Int)
variable (d a_1 : Int)

-- Conditions:
axiom a_seq_arith : ∀ n, a (n + 1) = a n + d
axiom S_arith : ∀ n, S n = n * (a 1 + a n) / 2
axiom S_10 : S 10 = 110
axiom geo_seq : (a 2) ^ 2 = a 1 * a 4
axiom b_def : ∀ n, b n = 1 / ((a n - 1) * (a n + 1))

-- Goals: 
-- 1. Find the general formula for the terms of sequence {a_n}
theorem a_n_formula : ∀ n, a n = 2 * n := sorry

-- 2. Find the sum of the first n terms T_n of the sequence {b_n} given b_n
theorem T_n_formula : ∀ n, T n = 1 / 2 - 1 / (4 * n + 2) := sorry

end a_n_formula_T_n_formula_1_1446


namespace x_sq_plus_3x_eq_1_1_1112

theorem x_sq_plus_3x_eq_1 (x : ℝ) (h : (x^2 + 3*x)^2 + 2*(x^2 + 3*x) - 3 = 0) : x^2 + 3*x = 1 :=
sorry

end x_sq_plus_3x_eq_1_1_1112


namespace ben_eggs_left_1_1503

def initial_eggs : ℕ := 50
def day1_morning : ℕ := 5
def day1_afternoon : ℕ := 4
def day2_morning : ℕ := 8
def day2_evening : ℕ := 3
def day3_afternoon : ℕ := 6
def day3_night : ℕ := 2

theorem ben_eggs_left : initial_eggs - (day1_morning + day1_afternoon + day2_morning + day2_evening + day3_afternoon + day3_night) = 22 := 
by
  sorry

end ben_eggs_left_1_1503


namespace ram_ravi_selected_probability_1_1110

noncomputable def probability_both_selected : ℝ := 
  let probability_ram_80 := (1 : ℝ) / 7
  let probability_ravi_80 := (1 : ℝ) / 5
  let probability_both_80 := probability_ram_80 * probability_ravi_80
  let num_applicants := 200
  let num_spots := 4
  let probability_single_selection := (num_spots : ℝ) / (num_applicants : ℝ)
  let probability_both_selected_given_80 := probability_single_selection * probability_single_selection
  probability_both_80 * probability_both_selected_given_80

theorem ram_ravi_selected_probability :
  probability_both_selected = 1 / 87500 := 
by
  sorry

end ram_ravi_selected_probability_1_1110


namespace smallest_k_for_inequality_1_1300

theorem smallest_k_for_inequality : 
  ∃ k : ℕ,  k > 0 ∧ ( (k-10) ^ 5026 ≥ 2013 ^ 2013 ) ∧ 
  (∀ m : ℕ, m > 0 ∧ ((m-10) ^ 5026) ≥ 2013 ^ 2013 → m ≥ 55) :=
sorry

end smallest_k_for_inequality_1_1300


namespace min_value_inequality_equality_condition_1_1605

theorem min_value_inequality (a b : ℝ) (ha : 1 < a) (hb : 1 < b) :
  (b^2 / (a - 1) + a^2 / (b - 1)) ≥ 8 :=
sorry

theorem equality_condition (a b : ℝ) (ha : 1 < a) (hb : 1 < b) :
  (b^2 / (a - 1) + a^2 / (b - 1) = 8) ↔ ((a = 2) ∧ (b = 2)) :=
sorry

end min_value_inequality_equality_condition_1_1605


namespace perp_lines_iff_m_values_1_1397

section
variables (m x y : ℝ)

def l1 := (m * x + y - 2 = 0)
def l2 := ((m + 1) * x - 2 * m * y + 1 = 0)

theorem perp_lines_iff_m_values (h1 : l1 m x y) (h2 : l2 m x y) (h_perp : (m * (m + 1) + (-2 * m) = 0)) : m = 0 ∨ m = 1 :=
by {
  sorry
}
end

end perp_lines_iff_m_values_1_1397


namespace length_of_platform_is_350_1_1311

-- Define the parameters as given in the problem
def train_length : ℕ := 300
def time_to_cross_post : ℕ := 18
def time_to_cross_platform : ℕ := 39

-- Define the speed of the train as a ratio of the length of the train and the time to cross the post
def train_speed : ℚ := train_length / time_to_cross_post

-- Formalize the problem statement: Prove that the length of the platform is 350 meters
theorem length_of_platform_is_350 : ∃ (L : ℕ), (train_speed * time_to_cross_platform) = train_length + L := by
  use 350
  sorry

end length_of_platform_is_350_1_1311


namespace find_g7_1_1664

namespace ProofProblem

variable (g : ℝ → ℝ)
variable (h1 : ∀ x y : ℝ, g (x + y) = g x + g y)
variable (h2 : g 6 = 8)

theorem find_g7 : g 7 = 28 / 3 := by
  sorry

end ProofProblem

end find_g7_1_1664


namespace sum_of_readings_ammeters_1_1911

variables (I1 I2 I3 I4 I5 : ℝ)

noncomputable def sum_of_ammeters (I1 I2 I3 I4 I5 : ℝ) : ℝ :=
  I1 + I2 + I3 + I4 + I5

theorem sum_of_readings_ammeters :
  I1 = 2 ∧ I2 = I1 ∧ I3 = 2 * I1 ∧ I5 = I3 + I1 ∧ I4 = (5 / 3) * I5 →
  sum_of_ammeters I1 I2 I3 I4 I5 = 24 :=
by
  sorry

end sum_of_readings_ammeters_1_1911


namespace quadratic_inequality_range_1_1993

theorem quadratic_inequality_range (a x : ℝ) :
  (∀ x : ℝ, x^2 - x - a^2 + a + 1 > 0) ↔ (-1/2 < a ∧ a < 3/2) :=
by
  sorry

end quadratic_inequality_range_1_1993


namespace a1_lt_a3_iff_an_lt_an1_1_1068

-- Define arithmetic sequence and required properties
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop := 
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

variables (a : ℕ → ℝ)

-- Define the necessary and sufficient condition theorem
theorem a1_lt_a3_iff_an_lt_an1 (h_arith : is_arithmetic_sequence a) :
  (a 1 < a 3) ↔ (∀ n : ℕ, a n < a (n + 1)) :=
sorry

end a1_lt_a3_iff_an_lt_an1_1_1068


namespace by_how_much_were_the_numerator_and_denominator_increased_1_1601

noncomputable def original_fraction_is_six_over_eleven (n : ℕ) : Prop :=
  n / (n + 5) = 6 / 11

noncomputable def resulting_fraction_is_seven_over_twelve (n x : ℕ) : Prop :=
  (n + x) / (n + 5 + x) = 7 / 12

theorem by_how_much_were_the_numerator_and_denominator_increased :
  ∃ (n x : ℕ), original_fraction_is_six_over_eleven n ∧ resulting_fraction_is_seven_over_twelve n x ∧ x = 1 :=
by
  sorry

end by_how_much_were_the_numerator_and_denominator_increased_1_1601


namespace circle_center_and_radius_sum_1_1936

theorem circle_center_and_radius_sum :
  let a := -4
  let b := -8
  let r := Real.sqrt 17
  a + b + r = -12 + Real.sqrt 17 :=
by
  sorry

end circle_center_and_radius_sum_1_1936


namespace proof_problem_1_1122

noncomputable def f (x : ℝ) (a : ℝ) (b : ℝ) : ℝ := a * Real.sin x + b * x^3 + 4

noncomputable def f' (x : ℝ) (a : ℝ) (b : ℝ) : ℝ := a * Real.cos x + 3 * b * x^2

theorem proof_problem (a b : ℝ) :
  f 2016 a b + f (-2016) a b + f' 2017 a b - f' (-2017) a b = 8 := by
  sorry

end proof_problem_1_1122


namespace expression_range_1_1704

theorem expression_range (a b c x : ℝ) (h : a^2 + b^2 + c^2 ≠ 0) :
  ∃ y : ℝ, y = (a * Real.cos x + b * Real.sin x + c) / (Real.sqrt (a^2 + b^2 + c^2)) 
           ∧ y ∈ Set.Icc (-Real.sqrt 3) (Real.sqrt 3) :=
sorry

end expression_range_1_1704


namespace max_areas_in_disk_1_1023

noncomputable def max_non_overlapping_areas (n : ℕ) : ℕ := 5 * n + 1

theorem max_areas_in_disk (n : ℕ) : 
  let disk_divided_by_2n_radii_and_two_secant_lines_areas  := (5 * n + 1)
  disk_divided_by_2n_radii_and_two_secant_lines_areas = max_non_overlapping_areas n := by sorry

end max_areas_in_disk_1_1023


namespace beautiful_39th_moment_1_1440

def is_beautiful (h : ℕ) (mm : ℕ) : Prop :=
  (h + mm) % 12 = 0

def start_time := (7, 49)

noncomputable def find_39th_beautiful_moment : ℕ × ℕ :=
  (15, 45)

theorem beautiful_39th_moment :
  find_39th_beautiful_moment = (15, 45) :=
by
  sorry

end beautiful_39th_moment_1_1440


namespace percent_of_a_is_4b_1_1628

variable (a b : ℝ)
variable (h : a = 1.2 * b)

theorem percent_of_a_is_4b :
  (4 * b) = (10 / 3 * 100 * a) / 100 :=
by sorry

end percent_of_a_is_4b_1_1628


namespace negation_exists_eq_forall_1_1924

theorem negation_exists_eq_forall (h : ¬ ∃ x : ℝ, x^2 + 2*x + 5 = 0) : ∀ x : ℝ, x^2 + 2*x + 5 ≠ 0 := 
by
  sorry

end negation_exists_eq_forall_1_1924


namespace q1_q2_1_1178

variable (a b : ℝ)

-- Definition of the conditions
def conditions : Prop := a + b = 7 ∧ a * b = 6

-- Statement of the first question
theorem q1 (h : conditions a b) : a^2 + b^2 = 37 := sorry

-- Statement of the second question
theorem q2 (h : conditions a b) : a^3 * b - 2 * a^2 * b^2 + a * b^3 = 150 := sorry

end q1_q2_1_1178


namespace find_a8_1_1715

theorem find_a8 (a : ℕ → ℤ) (x : ℤ) :
  (1 + x)^10 = a 0 + a 1 * (1 - x) + a 2 * (1 - x)^2 + a 3 * (1 - x)^3 +
               a 4 * (1 - x)^4 + a 5 * (1 - x)^5 + a 6 * (1 - x)^6 +
               a 7 * (1 - x)^7 + a 8 * (1 - x)^8 + a 9 * (1 - x)^9 +
               a 10 * (1 - x)^10 → a 8 = 180 := by
  sorry

end find_a8_1_1715


namespace julia_age_correct_1_1833

def julia_age_proof : Prop :=
  ∃ (j : ℚ) (m : ℚ), m = 15 * j ∧ m - j = 40 ∧ j = 20 / 7

theorem julia_age_correct : julia_age_proof :=
by
  sorry

end julia_age_correct_1_1833


namespace second_discount_percentage_1_1428

theorem second_discount_percentage (x : ℝ) :
  9356.725146198829 * 0.8 * (1 - x / 100) * 0.95 = 6400 → x = 10 :=
by
  sorry

end second_discount_percentage_1_1428


namespace find_number_1_1330

theorem find_number (N : ℝ) (h : 0.60 * N = 0.50 * 720) : N = 600 :=
sorry

end find_number_1_1330


namespace model_N_completion_time_1_1021

variable (T : ℕ)

def model_M_time : ℕ := 36
def number_of_M_computers : ℕ := 12
def number_of_N_computers := number_of_M_computers -- given that they are the same.

-- Statement of the problem: Given the conditions, prove T = 18
theorem model_N_completion_time :
  (number_of_M_computers : ℝ) * (1 / model_M_time) + (number_of_N_computers : ℝ) * (1 / T) = 1 →
  T = 18 :=
by
  sorry

end model_N_completion_time_1_1021


namespace pyramid_layers_total_1_1418

-- Since we are dealing with natural number calculations, noncomputable is generally not needed.

-- Definition of the pyramid layers and the number of balls in each layer
def number_of_balls (n : ℕ) : ℕ := n ^ 2

-- Given conditions for the layers
def third_layer_balls : ℕ := number_of_balls 3
def fifth_layer_balls : ℕ := number_of_balls 5

-- Statement of the problem proving that their sum is 34
theorem pyramid_layers_total : third_layer_balls + fifth_layer_balls = 34 := by
  sorry -- proof to be provided

end pyramid_layers_total_1_1418


namespace no_such_n_1_1015

open Nat

def is_power (m : ℕ) : Prop :=
  ∃ r ≥ 2, ∃ b, m = b ^ r

theorem no_such_n (n : ℕ) (A : Fin n → ℕ) :
  (2 ≤ n) →
  (∀ i j : Fin n, i ≠ j → A i ≠ A j) →
  (∀ k : ℕ, is_power (∏ i, (A i + k))) →
  False :=
by
  sorry

end no_such_n_1_1015


namespace circle_reflection_1_1117

variable (x₀ y₀ : ℝ)

def reflect_over_line_y_eq_neg_x (p : ℝ × ℝ) : ℝ × ℝ :=
  (-p.2, -p.1)

theorem circle_reflection 
  (h₁ : x₀ = 8)
  (h₂ : y₀ = -3) :
  reflect_over_line_y_eq_neg_x (x₀, y₀) = (3, -8) := by
  sorry

end circle_reflection_1_1117


namespace geometric_series_sum_1_1602

theorem geometric_series_sum :
  let a := (1 : ℚ) / 3
  let r := -(1 / 3)
  let n := 5
  let S₅ := (a * (1 - r ^ n)) / (1 - r)
  S₅ = 61 / 243 := by
  let a := (1 : ℚ) / 3
  let r := -(1 / 3)
  let n := 5
  let S₅ := (a * (1 - r ^ n)) / (1 - r)
  sorry

end geometric_series_sum_1_1602


namespace factorization1_factorization2_factorization3_1_1563

-- Problem 1
theorem factorization1 (a x : ℝ) : a * x^2 - 4 * a = a * (x + 2) * (x - 2) :=
sorry

-- Problem 2
theorem factorization2 (m x y : ℝ) : m * x^2 + 2 * m * x * y + m * y^2 = m * (x + y)^2 :=
sorry

-- Problem 3
theorem factorization3 (a b : ℝ) : (1 / 2) * a^2 - a * b + (1 / 2) * b^2 = (1 / 2) * (a - b)^2 :=
sorry

end factorization1_factorization2_factorization3_1_1563


namespace perimeter_of_similar_triangle_1_1070

def is_isosceles (a b c : ℕ) : Prop :=
  (a = b) ∨ (b = c) ∨ (a = c)

theorem perimeter_of_similar_triangle (a b c : ℕ) (d e f : ℕ) 
  (h1 : is_isosceles a b c) (h2 : min a b = 15) (h3 : min (min a b) c = 15)
  (h4 : d = 75) (h5 : (d / 15) = e / b) (h6 : f = e) :
  d + e + f = 375 :=
by
  sorry

end perimeter_of_similar_triangle_1_1070


namespace points_distance_le_sqrt5_1_1436

theorem points_distance_le_sqrt5 :
  ∀ (points : Fin 6 → ℝ × ℝ), 
  (∀ i, (0 ≤ (points i).1 ∧ (points i).1 ≤ 4) ∧ (0 ≤ (points i).2 ∧ (points i).2 ≤ 3)) →
  ∃ (i j : Fin 6), i ≠ j ∧ dist (points i) (points j) ≤ Real.sqrt 5 :=
by
  sorry

end points_distance_le_sqrt5_1_1436


namespace MrsHiltReadTotalChapters_1_1337

-- Define the number of books and chapters per book
def numberOfBooks : ℕ := 4
def chaptersPerBook : ℕ := 17

-- Define the total number of chapters Mrs. Hilt read
def totalChapters (books : ℕ) (chapters : ℕ) : ℕ := books * chapters

-- The main statement to be proved
theorem MrsHiltReadTotalChapters : totalChapters numberOfBooks chaptersPerBook = 68 := by
  sorry

end MrsHiltReadTotalChapters_1_1337


namespace croissant_to_orange_ratio_1_1382

-- Define the conditions as given in the problem
variables (c o : ℝ)
variable (emily_expenditure : ℝ)
variable (lucas_expenditure : ℝ)

-- Given conditions of expenditures
axiom emily_expenditure_is : emily_expenditure = 5 * c + 4 * o
axiom lucas_expenditure_is : lucas_expenditure = 3 * emily_expenditure
axiom lucas_expenditure_as_purchased : lucas_expenditure = 4 * c + 10 * o

-- Prove the ratio of the cost of a croissant to an orange
theorem croissant_to_orange_ratio : (c / o) = 2 / 11 :=
by sorry

end croissant_to_orange_ratio_1_1382


namespace CD_is_b_minus_a_minus_c_1_1589

variables (V : Type) [AddCommGroup V] [Module ℝ V]
variables (A B C D : V) (a b c : V)

def AB : V := a
def AD : V := b
def BC : V := c

theorem CD_is_b_minus_a_minus_c (h1 : A + a = B) (h2 : A + b = D) (h3 : B + c = C) :
  D - C = b - a - c :=
by sorry

end CD_is_b_minus_a_minus_c_1_1589


namespace cosine_of_eight_times_alpha_1_1667

theorem cosine_of_eight_times_alpha (α : ℝ) (hypotenuse : ℝ) 
  (cos_α : ℝ) (cos_2α : ℝ) (cos_4α : ℝ) 
  (h₀ : hypotenuse = Real.sqrt (1^2 + (Real.sqrt 2)^2))
  (h₁ : cos_α = (Real.sqrt 2) / hypotenuse)
  (h₂ : cos_2α = 2 * cos_α^2 - 1)
  (h₃ : cos_4α = 2 * cos_2α^2 - 1)
  (h₄ : cos_8α = 2 * cos_4α^2 - 1) :
  cos_8α = 17 / 81 := 
  by
  sorry

end cosine_of_eight_times_alpha_1_1667


namespace find_ks_1_1829

def is_valid_function (f : ℕ → ℤ) (k : ℤ) : Prop :=
  ∀ x y : ℕ, f (x * y) = f x + f y + k * f (Nat.gcd x y)

theorem find_ks (f : ℕ → ℤ) :
  (f 2006 = 2007) →
  is_valid_function f k →
  k = 0 ∨ k = -1 :=
sorry

end find_ks_1_1829


namespace log_27_gt_point_53_1_1433

open Real

theorem log_27_gt_point_53 :
  log 27 > 0.53 :=
by
  sorry

end log_27_gt_point_53_1_1433


namespace evaluate_expression_1_1720

theorem evaluate_expression :
  let x := (1 : ℚ) / 2
  let y := (3 : ℚ) / 4
  let z := -6
  let w := 2
  (x^2 * y^4 * z * w = - (243 / 256)) := 
by {
  let x := (1 : ℚ) / 2
  let y := (3 : ℚ) / 4
  let z := -6
  let w := 2
  sorry
}

end evaluate_expression_1_1720


namespace part1_monotonically_increasing_interval_part1_symmetry_axis_part2_find_a_b_1_1952

noncomputable def f (a b x : ℝ) : ℝ :=
  a * (2 * (Real.cos (x/2))^2 + Real.sin x) + b

theorem part1_monotonically_increasing_interval (b : ℝ) (k : ℤ) :
  let f := f 1 b
  ∀ x, x ∈ Set.Icc (-3 * Real.pi / 4 + 2 * k * Real.pi) (Real.pi / 4 + 2 * k * Real.pi) ->
    f x <= f (x + Real.pi) :=
sorry

theorem part1_symmetry_axis (b : ℝ) (k : ℤ) :
  let f := f 1 b
  ∀ x, f x = f (2 * (Real.pi / 4 + k * Real.pi) - x) :=
sorry

theorem part2_find_a_b (x : ℝ) (h1 : 0 ≤ x ∧ x ≤ Real.pi)
  (h2 : ∃ (a b : ℝ), ∀ x, x ∈ Set.Icc 0 Real.pi → (a > 0 ∧ 3 ≤ f a b x ∧ f a b x ≤ 4)) :
  (1 - Real.sqrt 2 < a ∧ a < 1 + Real.sqrt 2) ∧ b = 3 :=
sorry

end part1_monotonically_increasing_interval_part1_symmetry_axis_part2_find_a_b_1_1952


namespace quadratic_function_has_specific_k_1_1215

theorem quadratic_function_has_specific_k (k : ℤ) :
  (∀ x : ℝ, ∃ y : ℝ, y = (k-1)*x^(k^2-k+2) + k*x - 1) ↔ k = 0 :=
by
  sorry

end quadratic_function_has_specific_k_1_1215


namespace painting_time_1_1114

variable (a d e : ℕ)

theorem painting_time (h : a * e * d = a * d * e) : (d * x = a^2 * e) := 
by
   sorry

end painting_time_1_1114


namespace purple_gumdrops_after_replacement_1_1431

def total_gumdrops : Nat := 200
def orange_percentage : Nat := 40
def purple_percentage : Nat := 10
def yellow_percentage : Nat := 25
def white_percentage : Nat := 15
def black_percentage : Nat := 10

def initial_orange_gumdrops := (orange_percentage * total_gumdrops) / 100
def initial_purple_gumdrops := (purple_percentage * total_gumdrops) / 100
def orange_to_purple := initial_orange_gumdrops / 3
def final_purple_gumdrops := initial_purple_gumdrops + orange_to_purple

theorem purple_gumdrops_after_replacement : final_purple_gumdrops = 47 := by
  sorry

end purple_gumdrops_after_replacement_1_1431


namespace remaining_slices_correct_1_1091

-- Define initial slices of pie and cake
def initial_pie_slices : Nat := 2 * 8
def initial_cake_slices : Nat := 12

-- Define slices eaten on Friday
def friday_pie_slices_eaten : Nat := 2
def friday_cake_slices_eaten : Nat := 2

-- Define slices eaten on Saturday
def saturday_pie_slices_eaten (remaining: Nat) : Nat := remaining / 2 -- 50%
def saturday_cake_slices_eaten (remaining: Nat) : Nat := remaining / 4 -- 25%

-- Define slices eaten on Sunday morning
def sunday_morning_pie_slices_eaten : Nat := 2
def sunday_morning_cake_slices_eaten : Nat := 3

-- Define slices eaten on Sunday evening
def sunday_evening_pie_slices_eaten : Nat := 4
def sunday_evening_cake_slices_eaten : Nat := 1

-- Function to calculate remaining slices
def remaining_slices : Nat × Nat :=
  let after_friday_pies := initial_pie_slices - friday_pie_slices_eaten
  let after_friday_cake := initial_cake_slices - friday_cake_slices_eaten
  let after_saturday_pies := after_friday_pies - saturday_pie_slices_eaten after_friday_pies
  let after_saturday_cake := after_friday_cake - saturday_cake_slices_eaten after_friday_cake
  let after_sunday_morning_pies := after_saturday_pies - sunday_morning_pie_slices_eaten
  let after_sunday_morning_cake := after_saturday_cake - sunday_morning_cake_slices_eaten
  let final_pies := after_sunday_morning_pies - sunday_evening_pie_slices_eaten
  let final_cake := after_sunday_morning_cake - sunday_evening_cake_slices_eaten
  (final_pies, final_cake)

theorem remaining_slices_correct :
  remaining_slices = (1, 4) :=
  by {
    sorry -- Proof is omitted
  }

end remaining_slices_correct_1_1091


namespace train_speed_ratio_1_1066

theorem train_speed_ratio 
  (v_A v_B : ℝ)
  (h1 : v_A = 2 * v_B)
  (h2 : 27 = L_A / v_A)
  (h3 : 17 = L_B / v_B)
  (h4 : 22 = (L_A + L_B) / (v_A + v_B))
  (h5 : v_A + v_B ≤ 60) :
  v_A / v_B = 2 := by
  sorry

-- Conditions given must be defined properly
variables (L_A L_B : ℝ)

end train_speed_ratio_1_1066


namespace inequality_solution_1_1542

theorem inequality_solution (a : ℝ) : (∀ x : ℝ, (a + 1) * x > a + 1 → x < 1) ↔ a < -1 := by
  sorry

end inequality_solution_1_1542


namespace molecular_weight_K3AlC2O4_3_1_1928

noncomputable def molecularWeightOfCompound : ℝ :=
  let potassium_weight : ℝ := 39.10
  let aluminum_weight  : ℝ := 26.98
  let carbon_weight    : ℝ := 12.01
  let oxygen_weight    : ℝ := 16.00
  let total_potassium_weight : ℝ := 3 * potassium_weight
  let total_aluminum_weight  : ℝ := aluminum_weight
  let total_carbon_weight    : ℝ := 3 * 2 * carbon_weight
  let total_oxygen_weight    : ℝ := 3 * 4 * oxygen_weight
  total_potassium_weight + total_aluminum_weight + total_carbon_weight + total_oxygen_weight

theorem molecular_weight_K3AlC2O4_3 : molecularWeightOfCompound = 408.34 := by
  sorry

end molecular_weight_K3AlC2O4_3_1_1928


namespace smallest_number_of_coins_1_1551

theorem smallest_number_of_coins :
  ∃ (n : ℕ), (∀ (a : ℕ), 5 ≤ a ∧ a < 100 → 
    ∃ (c : ℕ → ℕ), (a = 5 * c 0 + 10 * c 1 + 25 * c 2) ∧ 
    (c 0 + c 1 + c 2 = n)) ∧ n = 9 :=
by
  sorry

end smallest_number_of_coins_1_1551


namespace probability_of_MATHEMATICS_letter_1_1136

def unique_letters_in_mathematics : Finset Char := {'M', 'A', 'T', 'H', 'E', 'I', 'C', 'S'}

theorem probability_of_MATHEMATICS_letter :
  let total_letters := 26
  let unique_letters_count := unique_letters_in_mathematics.card
  (unique_letters_count / total_letters : ℝ) = 8 / 26 := by
  sorry

end probability_of_MATHEMATICS_letter_1_1136


namespace otimes_identity_1_1969

def otimes (x y : ℝ) : ℝ := x^2 - y^2

theorem otimes_identity (h : ℝ) : otimes h (otimes h h) = h^2 :=
by
  sorry

end otimes_identity_1_1969


namespace min_side_length_1_1917

def table_diagonal (w h : ℕ) : ℕ :=
  Nat.sqrt (w * w + h * h)

theorem min_side_length (w h : ℕ) (S : ℕ) (dw : w = 9) (dh : h = 12) (dS : S = 15) :
  S >= table_diagonal w h :=
by
  sorry

end min_side_length_1_1917


namespace symmetric_points_origin_1_1154

theorem symmetric_points_origin (a b : ℝ) (h : (1, 2) = (-a, -b)) : a = -1 ∧ b = -2 :=
sorry

end symmetric_points_origin_1_1154


namespace least_number_subtracted_1_1791

theorem least_number_subtracted (n k : ℕ) (h₁ : n = 123457) (h₂ : k = 79) : ∃ r, n % k = r ∧ r = 33 :=
by
  sorry

end least_number_subtracted_1_1791


namespace time_2556_hours_from_now_main_1_1189

theorem time_2556_hours_from_now (h : ℕ) (mod_res : h % 12 = 0) :
  (3 + h) % 12 = 3 :=
by {
  sorry
}

-- Constants
def current_time : ℕ := 3
def hours_passed : ℕ := 2556
-- Proof input
def modular_result : hours_passed % 12 = 0 := by {
 sorry -- In the real proof, we should show that 2556 is divisible by 12
}

-- Main theorem instance
theorem main : (current_time + hours_passed) % 12 = 3 := 
  time_2556_hours_from_now hours_passed modular_result

end time_2556_hours_from_now_main_1_1189


namespace solve_eq_integers_1_1350

theorem solve_eq_integers (x y : ℤ) : 
    x^2 - x * y - 6 * y^2 + 2 * x + 19 * y = 18 ↔ (x = 2 ∧ y = 2) ∨ (x = -2 ∧ y = 2) := by
    sorry

end solve_eq_integers_1_1350


namespace sum_of_decimals_is_one_1_1519

-- Define digits for each decimal place
def digit_a : ℕ := 2
def digit_b : ℕ := 3
def digit_c : ℕ := 2
def digit_d : ℕ := 2

-- Define the decimal numbers with these digits
def decimal1 : Rat := (digit_b * 10 + digit_a) / 100
def decimal2 : Rat := (digit_d * 10 + digit_c) / 100
def decimal3 : Rat := (2 * 10 + 2) / 100
def decimal4 : Rat := (2 * 10 + 3) / 100

-- The main theorem that states the sum of these decimals is 1
theorem sum_of_decimals_is_one : decimal1 + decimal2 + decimal3 + decimal4 = 1 := by
  sorry

end sum_of_decimals_is_one_1_1519


namespace field_perimeter_1_1659

noncomputable def outer_perimeter (posts : ℕ) (post_width_inches : ℝ) (spacing_feet : ℝ) : ℝ :=
  let posts_per_side := posts / 4
  let gaps_per_side := posts_per_side - 1
  let post_width_feet := post_width_inches / 12
  let side_length := gaps_per_side * spacing_feet + posts_per_side * post_width_feet
  4 * side_length

theorem field_perimeter : 
  outer_perimeter 32 5 4 = 125 + 1/3 := 
by
  sorry

end field_perimeter_1_1659


namespace james_marbles_left_1_1307

def total_initial_marbles : Nat := 28
def marbles_in_bag_A : Nat := 4
def marbles_in_bag_B : Nat := 6
def marbles_in_bag_C : Nat := 2
def marbles_in_bag_D : Nat := 8
def marbles_in_bag_E : Nat := 4
def marbles_in_bag_F : Nat := 4

theorem james_marbles_left : total_initial_marbles - marbles_in_bag_D = 20 := by
  -- James has 28 marbles initially.
  -- He gives away Bag D which has 8 marbles.
  -- 28 - 8 = 20
  sorry

end james_marbles_left_1_1307


namespace bottles_in_cups_1_1891

-- Defining the given conditions
variables (BOTTLE GLASS CUP JUG : ℕ)

axiom h1 : JUG = BOTTLE + GLASS
axiom h2 : 2 * JUG = 7 * GLASS
axiom h3 : BOTTLE = CUP + 2 * GLASS

theorem bottles_in_cups : BOTTLE = 5 * CUP :=
sorry

end bottles_in_cups_1_1891


namespace proof_strictly_increasing_sequence_1_1907

noncomputable def exists_strictly_increasing_sequence : Prop :=
  ∃ a : ℕ → ℕ, 
    (∀ n m : ℕ, n < m → a n < a m) ∧
    (∀ m : ℕ, ∃ i j : ℕ, m = a i + a j) ∧
    (∀ n : ℕ, 0 < n → a n > n^2 / 16)

theorem proof_strictly_increasing_sequence : exists_strictly_increasing_sequence :=
  sorry

end proof_strictly_increasing_sequence_1_1907


namespace find_xy_1_1046

variable (x y : ℚ)

theorem find_xy (h1 : 1/x + 3/y = 1/2) (h2 : 1/y - 3/x = 1/3) : 
    x = -20 ∧ y = 60/11 := 
by
  sorry

end find_xy_1_1046


namespace black_rectangle_ways_1_1365

theorem black_rectangle_ways : ∑ a in Finset.range 5, ∑ b in Finset.range 5, (5 - a) * (5 - b) = 225 := sorry

end black_rectangle_ways_1_1365


namespace range_of_a_1_1087

noncomputable def inequality_condition (x : ℝ) (a : ℝ) : Prop :=
  a - 2 * x - |Real.log x| ≤ 0

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, x > 0 → inequality_condition x a) ↔ a ≤ 1 + Real.log 2 :=
sorry

end range_of_a_1_1087


namespace maximize_inscribed_polygons_1_1041

theorem maximize_inscribed_polygons : 
  ∃ (n : ℕ) (m : ℕ → ℕ), 
    (∀ i j, 1 ≤ i ∧ i < j ∧ j ≤ n → m i < m j) ∧ 
    (∑ i in Finset.range n, m i = 1996) ∧ 
    (n = 61) ∧ 
    (∀ k, 0 ≤ k ∧ k < n → m k = k + 2) :=
by
  sorry

end maximize_inscribed_polygons_1_1041


namespace interest_rate_A_1_1878

-- Given conditions
variables (Principal : ℝ := 4000)
variables (interestRate_C : ℝ := 11.5 / 100)
variables (gain_B : ℝ := 180)
variables (time : ℝ := 3)
variables (interest_from_C : ℝ := Principal * interestRate_C * time)
variables (interest_to_A : ℝ := interest_from_C - gain_B)

-- The proof goal
theorem interest_rate_A (R : ℝ) : 
  1200 = Principal * (R / 100) * time → 
  R = 10 :=
by
  sorry

end interest_rate_A_1_1878


namespace hot_sauce_container_size_1_1750

theorem hot_sauce_container_size :
  let serving_size := 0.5
  let servings_per_day := 3
  let days := 20
  let total_consumed := servings_per_day * serving_size * days
  let one_quart := 32
  one_quart - total_consumed = 2 :=
by
  sorry

end hot_sauce_container_size_1_1750


namespace race_distance_1_1492

theorem race_distance (D : ℝ)
  (A_speed : ℝ := D / 20)
  (B_speed : ℝ := D / 25)
  (A_beats_B_by : ℝ := 18)
  (h1 : A_speed * 25 = D + A_beats_B_by)
  : D = 72 := 
by
  sorry

end race_distance_1_1492


namespace fraction_ordering_1_1266

theorem fraction_ordering :
  let a := (6 : ℚ) / 22
  let b := (8 : ℚ) / 32
  let c := (10 : ℚ) / 29
  a < b ∧ b < c :=
by
  sorry

end fraction_ordering_1_1266


namespace solve_equation_1_1795

theorem solve_equation (x : ℚ) : 3 * (x - 2) = 2 - 5 * (x - 2) ↔ x = 9 / 4 := by
  sorry

end solve_equation_1_1795


namespace point_in_second_quadrant_1_1764

theorem point_in_second_quadrant (P : ℝ × ℝ)
  (h1 : P.1 < 0) -- Point P is in the second quadrant, so its x-coordinate is negative
  (h2 : 0 < P.2) -- Point P is in the second quadrant, so its y-coordinate is positive
  (h3 : |P.2| = 3) -- The distance from P to the x-axis is 3
  (h4 : |P.1| = 4) -- The distance from P to the y-axis is 4
  : P = (-4, 3) := 
  sorry

end point_in_second_quadrant_1_1764


namespace sum_leq_six_of_quadratic_roots_1_1774

theorem sum_leq_six_of_quadratic_roots (a b : ℤ) (h1 : a ≠ -1) (h2 : b ≠ -1) 
  (h3 : ∃ r1 r2 : ℤ, r1 ≠ r2 ∧ x^2 + ab * x + (a + b) = 0 ∧ 
         x = r1 ∧ x = r2) : a + b ≤ 6 :=
by
  sorry

end sum_leq_six_of_quadratic_roots_1_1774


namespace y_coordinate_sum_of_circle_on_y_axis_1_1652

-- Define the properties of the circle
def center := (-3, 1)
def radius := 8

-- Define the equation of the circle
def circle_eq (x y : ℝ) : Prop :=
  (x + 3) ^ 2 + (y - 1) ^ 2 = 64

-- Define the Lean theorem statement
theorem y_coordinate_sum_of_circle_on_y_axis 
  (h₁ : center = (-3, 1)) 
  (h₂ : radius = 8) 
  (h₃ : ∀ y : ℝ, circle_eq 0 y → (∃ y1 y2 : ℝ, y = y1 ∨ y = y2) ) : 
  ∃ y1 y2 : ℝ, (y1 + y2 = 2) ∧ (circle_eq 0 y1) ∧ (circle_eq 0 y2) := 
by 
  sorry

end y_coordinate_sum_of_circle_on_y_axis_1_1652


namespace digit_B_divisible_by_9_1_1206

theorem digit_B_divisible_by_9 (B : ℕ) (h1 : B ≤ 9) (h2 : (4 + B + B + 1 + 3) % 9 = 0) : B = 5 := 
by {
  /- Proof omitted -/
  sorry
}

end digit_B_divisible_by_9_1_1206


namespace polynomial_g_1_1400

def f (x : ℝ) : ℝ := x^2

theorem polynomial_g (g : ℝ → ℝ) :
  (∀ x, f (g x) = 9 * x ^ 2 - 6 * x + 1) →
  (∀ x, g x = 3 * x - 1 ∨ g x = -3 * x + 1) :=
by
  sorry

end polynomial_g_1_1400


namespace combinations_x_eq_2_or_8_1_1236

theorem combinations_x_eq_2_or_8 (x : ℕ) (h_pos : 0 < x) (h_comb : Nat.choose 10 x = Nat.choose 10 2) : x = 2 ∨ x = 8 :=
sorry

end combinations_x_eq_2_or_8_1_1236


namespace total_growth_of_trees_1_1894

theorem total_growth_of_trees :
  let t1_growth_rate := 1 -- first tree grows 1 meter/day
  let t2_growth_rate := 2 -- second tree grows 2 meters/day
  let t3_growth_rate := 2 -- third tree grows 2 meters/day
  let t4_growth_rate := 3 -- fourth tree grows 3 meters/day
  let days := 4
  t1_growth_rate * days + t2_growth_rate * days + t3_growth_rate * days + t4_growth_rate * days = 32 :=
by
  let t1_growth_rate := 1
  let t2_growth_rate := 2
  let t3_growth_rate := 2
  let t4_growth_rate := 3
  let days := 4
  sorry

end total_growth_of_trees_1_1894


namespace group_capacity_1_1857

theorem group_capacity (total_students : ℕ) (selected_students : ℕ) (removed_students : ℕ) :
  total_students = 5008 → selected_students = 200 → removed_students = 8 →
  (total_students - removed_students) / selected_students = 25 :=
by
  intros h1 h2 h3
  sorry

end group_capacity_1_1857


namespace parallel_lines_slope_1_1525

theorem parallel_lines_slope (m : ℚ) (h : (x - y = 1) → (m + 3) * x + m * y - 8 = 0) :
  m = -3 / 2 :=
sorry

end parallel_lines_slope_1_1525


namespace find_multiplier_1_1900

variable {a b : ℝ} 

theorem find_multiplier (h1 : 3 * a = x * b) (h2 : a ≠ 0 ∧ b ≠ 0) (h3 : a / 4 = b / 3) : x = 4 :=
by
  sorry

end find_multiplier_1_1900


namespace fruit_weights_determined_1_1617

noncomputable def fruit_weight_configuration : Prop :=
  let weights : List ℕ := [100, 150, 170, 200, 280]
  let mandarin := 100
  let apple := 150
  let banana := 170
  let peach := 200
  let orange := 280
  (peach < orange) ∧
  (apple < banana ∧ banana < peach) ∧
  (mandarin < banana) ∧
  (apple + banana > orange)
  
theorem fruit_weights_determined :
  fruit_weight_configuration :=
by
  sorry

end fruit_weights_determined_1_1617


namespace ladder_base_distance_1_1531

theorem ladder_base_distance
  (c : ℕ) (b : ℕ) (hypotenuse : c = 13) (wall_height : b = 12) :
  ∃ x : ℕ, x^2 + b^2 = c^2 ∧ x = 5 := by
  sorry

end ladder_base_distance_1_1531


namespace probability_sum_of_five_1_1320

def total_outcomes : ℕ := 36
def favorable_outcomes : ℕ := 4

theorem probability_sum_of_five :
  favorable_outcomes / total_outcomes = 1 / 9 := 
by
  sorry

end probability_sum_of_five_1_1320


namespace polynomial_solution_1_1518

theorem polynomial_solution (x : ℝ) (h : (2 * x - 1) ^ 2 = 9) : x = 2 ∨ x = -1 :=
by
  sorry

end polynomial_solution_1_1518


namespace find_xyz_sum_1_1707

variables {x y z : ℝ}

def system_of_equations (x y z : ℝ) : Prop :=
  x > 0 ∧ y > 0 ∧ z > 0 ∧
  (x^2 + x * y + y^2 = 12) ∧
  (y^2 + y * z + z^2 = 9) ∧
  (z^2 + z * x + x^2 = 21)

theorem find_xyz_sum (x y z : ℝ) (h : system_of_equations x y z) : 
  x * y + y * z + z * x = 12 :=
sorry

end find_xyz_sum_1_1707


namespace number_of_true_propositions_1_1965

theorem number_of_true_propositions : 
  (∃ x y : ℝ, (x * y = 1) ↔ (x = y⁻¹ ∨ y = x⁻¹)) ∧
  (¬(∀ x : ℝ, (x > -3) → x^2 - x - 6 ≤ 0)) ∧
  (¬(∀ a b : ℝ, (a > b) → (a^2 < b^2))) ∧
  (¬(∀ x : ℝ, (x - 1/x > 0) → (x > -1))) →
  True := by
  sorry

end number_of_true_propositions_1_1965


namespace step_count_initial_1_1587

theorem step_count_initial :
  ∃ (x y : ℕ), 1 ≤ x ∧ x ≤ 9 ∧ 0 ≤ y ∧ y ≤ 9 ∧ (11 * y - x = 64) ∧ (10 * x + y = 26) :=
by
  sorry

end step_count_initial_1_1587


namespace statues_ratio_1_1837

theorem statues_ratio :
  let y1 := 4                  -- Number of statues after first year.
  let y2 := 4 * y1             -- Number of statues after second year.
  let y3 := (y2 + 12) - 3      -- Number of statues after third year.
  let y4 := 31                 -- Number of statues after fourth year.
  let added_fourth_year := y4 - y3  -- Statues added in the fourth year.
  let broken_third_year := 3        -- Statues broken in the third year.
  added_fourth_year / broken_third_year = 2 :=
by
  sorry

end statues_ratio_1_1837


namespace opposite_neg_two_1_1790

def opposite (x : Int) : Int := -x

theorem opposite_neg_two : opposite (-2) = 2 := by
  sorry

end opposite_neg_two_1_1790


namespace find_other_number_1_1889

-- Defining the two numbers and their properties
def sum_is_84 (a b : ℕ) : Prop := a + b = 84
def one_is_36 (a b : ℕ) : Prop := a = 36 ∨ b = 36
def other_is_48 (a b : ℕ) : Prop := a = 48 ∨ b = 48

-- The theorem statement
theorem find_other_number (a b : ℕ) (h1 : sum_is_84 a b) (h2 : one_is_36 a b) : other_is_48 a b :=
by {
  sorry
}

end find_other_number_1_1889


namespace most_convincing_method_for_relationship_1_1919

-- Definitions from conditions
def car_owners : ℕ := 300
def car_owners_opposed_policy : ℕ := 116
def non_car_owners : ℕ := 200
def non_car_owners_opposed_policy : ℕ := 121

-- The theorem statement
theorem most_convincing_method_for_relationship : 
  (owning_a_car_related_to_opposing_policy : Bool) :=
by
  -- Proof of the statement
  sorry

end most_convincing_method_for_relationship_1_1919


namespace squares_difference_1_1199

theorem squares_difference (x y : ℝ) (h1 : x + y = 10) (h2 : x - y = 4) : x^2 - y^2 = 40 :=
by sorry

end squares_difference_1_1199


namespace solution_set_ineq_1_1842

theorem solution_set_ineq (x : ℝ) : (1 < x ∧ x ≤ 3) ↔ (x - 3) / (x - 1) ≤ 0 := sorry

end solution_set_ineq_1_1842


namespace katherine_time_20_1_1532

noncomputable def time_katherine_takes (k : ℝ) :=
  let time_naomi_takes_per_website := (5/4) * k
  let total_websites := 30
  let total_time_naomi := 750
  time_naomi_takes_per_website = 25 ∧ k = 20

theorem katherine_time_20 :
  ∃ k : ℝ, time_katherine_takes k :=
by
  use 20
  sorry

end katherine_time_20_1_1532


namespace sum_of_80th_equation_1_1463

theorem sum_of_80th_equation : (2 * 80 + 1) + (5 * 80 - 1) = 560 := by
  sorry

end sum_of_80th_equation_1_1463


namespace thabo_books_1_1415

theorem thabo_books (H P F : ℕ) (h1 : P = H + 20) (h2 : F = 2 * P) (h3 : H + P + F = 280) : H = 55 :=
by
  sorry

end thabo_books_1_1415


namespace A_plus_B_eq_one_fourth_1_1032

noncomputable def A : ℚ := 1 / 3
noncomputable def B : ℚ := -1 / 12

theorem A_plus_B_eq_one_fourth :
  A + B = 1 / 4 := by
  sorry

end A_plus_B_eq_one_fourth_1_1032


namespace fill_time_1_1521

-- Definition of the conditions
def faster_pipe_rate (t : ℕ) := 1 / t
def slower_pipe_rate (t : ℕ) := 1 / (4 * t)
def combined_rate (t : ℕ) := faster_pipe_rate t + slower_pipe_rate t
def time_to_fill_tank (t : ℕ) := 1 / combined_rate t

-- Given t = 50, prove the combined fill time is 40 minutes which is equal to the target time to fill the tank.
theorem fill_time (t : ℕ) (h : 4 * t = 200) : t = 50 → time_to_fill_tank t = 40 :=
by
  intros ht
  rw [ht]
  sorry

end fill_time_1_1521


namespace cos_double_angle_1_1102
open Real

theorem cos_double_angle (α : ℝ) (h : tan (α - π / 4) = 2) : cos (2 * α) = -4 / 5 := 
sorry

end cos_double_angle_1_1102


namespace minimum_value_of_expression_1_1711

theorem minimum_value_of_expression (x : ℝ) (hx : x > 0) : 6 * x + 1 / x ^ 6 ≥ 7 :=
sorry

end minimum_value_of_expression_1_1711


namespace hard_candy_food_colouring_1_1409

noncomputable def food_colouring_per_hard_candy (lollipop_use : ℕ) (gummy_use : ℕ)
    (lollipops_per_day : ℕ) (gummies_per_day : ℕ) (hard_candies_per_day : ℕ)
    (total_food_colouring : ℕ) : ℕ := 
by
  -- Let ml_lollipops be the total amount needed for lollipops
  let ml_lollipops := lollipop_use * lollipops_per_day
  -- Let ml_gummy be the total amount needed for gummy candies
  let ml_gummy := gummy_use * gummies_per_day
  -- Let ml_non_hard be the amount for lollipops and gummy candies combined
  let ml_non_hard := ml_lollipops + ml_gummy
  -- Let ml_hard be the amount used for hard candies alone
  let ml_hard := total_food_colouring - ml_non_hard
  -- Compute the food colouring used per hard candy
  exact ml_hard / hard_candies_per_day

theorem hard_candy_food_colouring :
  food_colouring_per_hard_candy 8 3 150 50 20 1950 = 30 :=
by
  unfold food_colouring_per_hard_candy
  sorry

end hard_candy_food_colouring_1_1409


namespace cuckoo_chime_78_1_1608

-- Define the arithmetic sum for the cuckoo clock problem
def cuckoo_chime_sum (n a l : Nat) : Nat :=
  n * (a + l) / 2

-- Main theorem
theorem cuckoo_chime_78 : 
  cuckoo_chime_sum 12 1 12 = 78 := 
by
  -- Proof part can be written here
  sorry

end cuckoo_chime_78_1_1608


namespace juan_distance_1_1132

def time : ℝ := 80.0
def speed : ℝ := 10.0
def distance (t : ℝ) (s : ℝ) : ℝ := t * s

theorem juan_distance : distance time speed = 800.0 := by
  sorry

end juan_distance_1_1132


namespace creative_sum_1_1200

def letterValue (ch : Char) : Int :=
  let n := (ch.toNat - 'a'.toNat + 1) % 12
  if n = 0 then 2
  else if n = 1 then 1
  else if n = 2 then 2
  else if n = 3 then 3
  else if n = 4 then 2
  else if n = 5 then 1
  else if n = 6 then 0
  else if n = 7 then -1
  else if n = 8 then -2
  else if n = 9 then -3
  else if n = 10 then -2
  else if n = 11 then -1
  else 0 -- this should never happen

def wordValue (word : String) : Int :=
  word.foldl (λ acc ch => acc + letterValue ch) 0

theorem creative_sum : wordValue "creative" = -2 :=
  by
    sorry

end creative_sum_1_1200


namespace calc_factorial_sum_1_1953

theorem calc_factorial_sum : 6 * Nat.factorial 6 + 5 * Nat.factorial 5 + Nat.factorial 5 = 5040 := by
  sorry

end calc_factorial_sum_1_1953


namespace nth_inequality_1_1176

theorem nth_inequality (n : ℕ) : 
  (∑ k in Finset.range (2^(n+1) - 1), (1/(k+1))) > (↑(n+1) / 2) := 
by sorry

end nth_inequality_1_1176


namespace discounted_price_1_1847

variable (marked_price : ℝ) (discount_rate : ℝ)
variable (marked_price_def : marked_price = 150)
variable (discount_rate_def : discount_rate = 20)

theorem discounted_price (hmp : marked_price = 150) (hdr : discount_rate = 20) : 
  marked_price - (discount_rate / 100) * marked_price = 120 := by
  rw [hmp, hdr]
  sorry

end discounted_price_1_1847


namespace ratio_of_voters_1_1826

theorem ratio_of_voters (V_X V_Y : ℝ) 
  (h1 : 0.62 * V_X + 0.38 * V_Y = 0.54 * (V_X + V_Y)) : V_X / V_Y = 2 :=
by
  sorry

end ratio_of_voters_1_1826


namespace replace_asterisk_1_1321

theorem replace_asterisk :
  ∃ x : ℤ, (x / 21) * (63 / 189) = 1 ∧ x = 63 := sorry

end replace_asterisk_1_1321


namespace inscribed_cube_volume_1_1612

noncomputable def side_length_of_inscribed_cube (d : ℝ) : ℝ :=
d / Real.sqrt 3

noncomputable def volume_of_inscribed_cube (s : ℝ) : ℝ :=
s^3

theorem inscribed_cube_volume :
  (volume_of_inscribed_cube (side_length_of_inscribed_cube 12)) = 192 * Real.sqrt 3 :=
by
  sorry

end inscribed_cube_volume_1_1612


namespace ratio_a_c_1_1283

variables (a b c d : ℚ)

axiom ratio_a_b : a / b = 5 / 4
axiom ratio_c_d : c / d = 4 / 3
axiom ratio_d_b : d / b = 1 / 8

theorem ratio_a_c : a / c = 15 / 2 :=
by sorry

end ratio_a_c_1_1283


namespace number_of_arrangements_1_1344

theorem number_of_arrangements (P : Fin 5 → Type) (youngest : Fin 5) 
  (h_in_not_first_last : ∀ (i : Fin 5), i ≠ 0 → i ≠ 4 → i ≠ youngest) : 
  ∃ n, n = 72 := 
by
  sorry

end number_of_arrangements_1_1344


namespace M_sufficient_not_necessary_for_N_1_1819

def M : Set ℝ := {x | x^2 < 4}
def N : Set ℝ := {x | x < 2}

theorem M_sufficient_not_necessary_for_N (a : ℝ) :
  (a ∈ M → a ∈ N) ∧ (a ∈ N → ¬ (a ∈ M)) :=
sorry

end M_sufficient_not_necessary_for_N_1_1819


namespace lowest_score_to_average_90_1_1825

theorem lowest_score_to_average_90 {s1 s2 s3 max_score avg_score : ℕ} 
    (h1: s1 = 88) 
    (h2: s2 = 96) 
    (h3: s3 = 105) 
    (hmax: max_score = 120) 
    (havg: avg_score = 90) 
    : ∃ s4 s5, s4 ≤ max_score ∧ s5 ≤ max_score ∧ (s1 + s2 + s3 + s4 + s5) / 5 = avg_score ∧ (min s4 s5 = 41) :=
by {
    sorry
}

end lowest_score_to_average_90_1_1825


namespace harmonic_sum_ratio_1_1517

theorem harmonic_sum_ratio :
  (∑ k in Finset.range (2020 + 1), (2021 - k) / k) /
  (∑ k in Finset.range (2021 - 1), 1 / (k + 2)) = 2021 :=
by
  sorry

end harmonic_sum_ratio_1_1517


namespace horner_example_1_1872

def horner (coeffs : List ℤ) (x : ℤ) : ℤ :=
  coeffs.foldr (λ a acc => a + x * acc) 0

theorem horner_example : horner [12, 35, -8, 79, 6, 5, 3] (-4) = 220 := by
  sorry

end horner_example_1_1872


namespace solve_equation_correctly_1_1671

theorem solve_equation_correctly : 
  ∀ x : ℝ, (x - 1) / 2 - 1 = (2 * x + 1) / 3 → x = -11 :=
by
  intro x h
  sorry

end solve_equation_correctly_1_1671


namespace square_area_from_diagonal_1_1280

theorem square_area_from_diagonal (d : ℝ) (h : d = 12) :
  ∃ (A : ℝ), A = 72 :=
by
  sorry

end square_area_from_diagonal_1_1280


namespace sin_arith_seq_1_1017

theorem sin_arith_seq (a : ℕ → ℝ) (d : ℝ)
  (h_arith_seq : ∀ n, a (n + 1) = a n + d)
  (h_sum : a 1 + a 5 + a 9 = 5 * Real.pi) :
  Real.sin (a 2 + a 8) = - (Real.sqrt 3) / 2 :=
sorry

end sin_arith_seq_1_1017


namespace symmetric_points_sum_1_1640

theorem symmetric_points_sum
  (a b : ℝ)
  (h1 : a = -3)
  (h2 : b = 2) :
  a + b = -1 := by
  sorry

end symmetric_points_sum_1_1640


namespace max_ab_1_1815

theorem max_ab (a b : ℝ) (h : a + b = 1) : ab ≤ 1 / 4 :=
by
  sorry

end max_ab_1_1815


namespace apples_left_total_1_1610

-- Define the initial conditions
def FrankApples : ℕ := 36
def SusanApples : ℕ := 3 * FrankApples
def SusanLeft : ℕ := SusanApples / 2
def FrankLeft : ℕ := (2 / 3) * FrankApples

-- Define the total apples left
def total_apples_left (SusanLeft FrankLeft : ℕ) : ℕ := SusanLeft + FrankLeft

-- Given conditions transformed to Lean
theorem apples_left_total : 
  total_apples_left (SusanApples / 2) ((2 / 3) * FrankApples) = 78 := by
  sorry

end apples_left_total_1_1610


namespace minimum_seats_occupied_1_1624

theorem minimum_seats_occupied (total_seats : ℕ) (h : total_seats = 180) : 
  ∃ occupied_seats : ℕ, occupied_seats = 45 ∧ 
  ∀ additional_person,
    (∀ i : ℕ, i < total_seats → 
     (occupied_seats ≤ i → i < occupied_seats + 1 ∨ i > occupied_seats + 1)) →
    additional_person = occupied_seats + 1  :=
by
  sorry

end minimum_seats_occupied_1_1624


namespace multiple_of_regular_rate_is_1_5_1_1810

-- Definitions
def hourly_rate := 5.50
def regular_hours := 7.5
def total_hours := 10.5
def total_earnings := 66.0
def excess_hours := total_hours - regular_hours
def regular_earnings := regular_hours * hourly_rate
def excess_earnings := total_earnings - regular_earnings
def rate_per_excess_hour := excess_earnings / excess_hours
def multiple_of_regular_rate := rate_per_excess_hour / hourly_rate

-- Statement of the problem
theorem multiple_of_regular_rate_is_1_5 : multiple_of_regular_rate = 1.5 :=
by
  -- Note: The proof is not required, hence sorry is used.
  sorry

end multiple_of_regular_rate_is_1_5_1_1810


namespace series_sum_eq_1_1901

-- Definitions from conditions
def a : ℚ := 1 / 2
def r : ℚ := 1 / 2
def n : ℕ := 8

-- Theorem statement
theorem series_sum_eq :
  (∑ i in Finset.range n, a * r^i) = 255 / 256 :=
sorry

end series_sum_eq_1_1901


namespace food_cost_max_1_1019

theorem food_cost_max (x : ℝ) (total_cost : ℝ) (tax_rate : ℝ) (tip_rate : ℝ) (max_total : ℝ) (food_cost_max : ℝ) :
  total_cost = x * (1 + tax_rate + tip_rate) →
  tax_rate = 0.07 →
  tip_rate = 0.15 →
  max_total = 50 →
  total_cost ≤ max_total →
  food_cost_max = 50 / 1.22 →
  x ≤ food_cost_max :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end food_cost_max_1_1019


namespace simplify_expression_1_1606

variable (b : ℝ)

theorem simplify_expression (b : ℝ) : 
  (3 * b + 7 - 5 * b) / 3 = (-2 / 3) * b + (7 / 3) :=
by
  sorry

end simplify_expression_1_1606


namespace quadratic_one_real_root_positive_m_1_1840

theorem quadratic_one_real_root_positive_m (m : ℝ) (h : m > 0) :
  (∀ x : ℝ, x^2 + 6 * m * x + 2 * m = 0 → ((6 * m)^2 - 4 * 1 * (2 * m) = 0)) → m = 2 / 9 :=
by
  sorry

end quadratic_one_real_root_positive_m_1_1840


namespace fraction_value_1_1229

theorem fraction_value : (20 * 21) / (2 + 0 + 2 + 1) = 84 := by
  sorry

end fraction_value_1_1229


namespace Thomas_speed_greater_than_Jeremiah_1_1251

-- Define constants
def Thomas_passes_kilometers_per_hour := 5
def Jeremiah_passes_kilometers_per_hour := 6

-- Define speeds (in meters per hour)
def Thomas_speed := Thomas_passes_kilometers_per_hour * 1000
def Jeremiah_speed := Jeremiah_passes_kilometers_per_hour * 1000

-- Define hypothetical additional distances
def Thomas_hypothetical_additional_distance := 600 * 2
def Jeremiah_hypothetical_additional_distance := 50 * 2

-- Define effective distances traveled
def Thomas_effective_distance := Thomas_speed + Thomas_hypothetical_additional_distance
def Jeremiah_effective_distance := Jeremiah_speed + Jeremiah_hypothetical_additional_distance

-- Theorem to prove
theorem Thomas_speed_greater_than_Jeremiah : Thomas_effective_distance > Jeremiah_effective_distance := by
  -- Placeholder for the proof
  sorry

end Thomas_speed_greater_than_Jeremiah_1_1251


namespace cos_double_angle_1_1182

-- Definition of the terminal condition
def terminal_side_of_angle (α : ℝ) (x y : ℝ) : Prop :=
  (x = 1) ∧ (y = Real.sqrt 3) ∧ (x^2 + y^2 = 1)

-- Prove the required statement
theorem cos_double_angle (α : ℝ) :
  (terminal_side_of_angle α 1 (Real.sqrt 3)) →
  Real.cos (2 * α + Real.pi / 2) = - Real.sqrt 3 / 2 :=
by
  sorry

end cos_double_angle_1_1182


namespace solve_system_eqs_1_1338

theorem solve_system_eqs (x y : ℝ) (h1 : (1 + x) * (1 + x^2) * (1 + x^4) = 1 + y^7)
                            (h2 : (1 + y) * (1 + y^2) * (1 + y^4) = 1 + x^7) :
  (x = 0 ∧ y = 0) ∨ (x = -1 ∧ y = -1) :=
sorry

end solve_system_eqs_1_1338


namespace total_amount_paid_1_1093

theorem total_amount_paid :
  let pizzas := 3
  let cost_per_pizza := 8
  let total_cost := pizzas * cost_per_pizza
  total_cost = 24 :=
by
  sorry

end total_amount_paid_1_1093


namespace categorize_numbers_1_1773

def numbers : List ℚ := [-16/10, -5/6, 89/10, -7, 1/12, 0, 25]

def is_positive (x : ℚ) : Prop := x > 0
def is_negative_fraction (x : ℚ) : Prop := x < 0 ∧ x.den ≠ 1
def is_negative_integer (x : ℚ) : Prop := x < 0 ∧ x.den = 1

theorem categorize_numbers :
  { x | x ∈ numbers ∧ is_positive x } = { 89 / 10, 1 / 12, 25 } ∧
  { x | x ∈ numbers ∧ is_negative_fraction x } = { -5 / 6 } ∧
  { x | x ∈ numbers ∧ is_negative_integer x } = { -7 } := by
  sorry

end categorize_numbers_1_1773


namespace max_pies_without_ingredients_1_1174

theorem max_pies_without_ingredients :
  let total_pies := 36
  let chocolate_pies := total_pies / 3
  let marshmallow_pies := total_pies / 4
  let cayenne_pies := total_pies / 2
  let soy_nuts_pies := total_pies / 8
  let max_ingredient_pies := max (max chocolate_pies marshmallow_pies) (max cayenne_pies soy_nuts_pies)
  total_pies - max_ingredient_pies = 18 :=
by
  sorry

end max_pies_without_ingredients_1_1174


namespace solution_set_of_inequality_1_1772

theorem solution_set_of_inequality :
  {x : ℝ | -x^2 + 3 * x - 2 ≥ 0} = {x : ℝ | 1 ≤ x ∧ x ≤ 2} :=
by
  sorry

end solution_set_of_inequality_1_1772


namespace number_of_dots_in_120_circles_1_1304

theorem number_of_dots_in_120_circles :
  ∃ n : ℕ, (n = 14) ∧ (∀ m : ℕ, m * (m + 1) / 2 + m ≤ 120 → m ≤ n) :=
by
  sorry

end number_of_dots_in_120_circles_1_1304


namespace oldest_child_age_1_1937

def avg (a b c : ℕ) : ℕ := (a + b + c) / 3

theorem oldest_child_age (a b : ℕ) (h1 : avg a b x = 10) (h2 : a = 8) (h3 : b = 11) : x = 11 :=
by
  sorry

end oldest_child_age_1_1937


namespace smallest_portion_is_two_1_1524

theorem smallest_portion_is_two (a1 a2 a3 a4 a5 : ℕ) (d : ℕ) (h1 : a1 = a3 - 2 * d) (h2 : a2 = a3 - d) (h3 : a4 = a3 + d) (h4 : a5 = a3 + 2 * d) (h5 : a1 + a2 + a3 + a4 + a5 = 120) (h6 : a3 + a4 + a5 = 7 * (a1 + a2)) : a1 = 2 :=
by sorry

end smallest_portion_is_two_1_1524


namespace volume_remaining_proof_1_1141

noncomputable def volume_remaining_part (v_original v_total_small : ℕ) : ℕ := v_original - v_total_small

def original_edge_length := 9
def small_edge_length := 3
def num_edges := 12

def volume_original := original_edge_length ^ 3
def volume_small := small_edge_length ^ 3
def volume_total_small := num_edges * volume_small

theorem volume_remaining_proof : volume_remaining_part volume_original volume_total_small = 405 := by
  sorry

end volume_remaining_proof_1_1141


namespace initial_average_marks_is_90_1_1222

def incorrect_average_marks (A : ℝ) : Prop :=
  let wrong_sum := 10 * A
  let correct_sum := 10 * 95
  wrong_sum + 50 = correct_sum

theorem initial_average_marks_is_90 : ∃ A : ℝ, incorrect_average_marks A ∧ A = 90 :=
by
  use 90
  unfold incorrect_average_marks
  simp
  sorry

end initial_average_marks_is_90_1_1222


namespace calculate_value_1_1902

theorem calculate_value : (245^2 - 225^2) / 20 = 470 :=
by
  sorry

end calculate_value_1_1902


namespace rosie_pie_count_1_1583

def total_apples : ℕ := 40
def initial_apples_required : ℕ := 3
def apples_per_pie : ℕ := 5

theorem rosie_pie_count : (total_apples - initial_apples_required) / apples_per_pie = 7 :=
by
  sorry

end rosie_pie_count_1_1583


namespace simplify_exponent_multiplication_1_1080

theorem simplify_exponent_multiplication :
  (10 ^ 0.25) * (10 ^ 0.15) * (10 ^ 0.35) * (10 ^ 0.05) * (10 ^ 0.85) * (10 ^ 0.35) = 10 ^ 2 := by
  sorry

end simplify_exponent_multiplication_1_1080


namespace find_abc_sum_1_1695

-- Definitions and statements directly taken from conditions
def Q1 (x y : ℝ) : Prop := y = x^2 + 51/50
def Q2 (x y : ℝ) : Prop := x = y^2 + 23/2
def common_tangent_rational_slope (a b c : ℤ) : Prop :=
  ∃ (x y : ℝ), (a * x + b * y = c) ∧ (Q1 x y ∨ Q2 x y)

theorem find_abc_sum :
  ∃ (a b c : ℕ), 
    gcd (a) (gcd (b) (c)) = 1 ∧
    common_tangent_rational_slope (a) (b) (c) ∧
    a + b + c = 9 :=
  by sorry

end find_abc_sum_1_1695


namespace jeff_total_run_is_290_1_1273

variables (monday_to_wednesday_run : ℕ)
variables (thursday_run : ℕ)
variables (friday_run : ℕ)

def jeff_weekly_run_total : ℕ :=
  monday_to_wednesday_run + thursday_run + friday_run

theorem jeff_total_run_is_290 :
  (60 * 3) + (60 - 20) + (60 + 10) = 290 :=
by
  sorry

end jeff_total_run_is_290_1_1273


namespace michael_water_left_1_1763

theorem michael_water_left :
  let initial_water := 5
  let given_water := (18 / 7 : ℚ) -- using rational number to represent the fractions
  let remaining_water := initial_water - given_water
  remaining_water = 17 / 7 :=
by
  sorry

end michael_water_left_1_1763


namespace ab_divisibility_1_1749

theorem ab_divisibility (a b : ℕ) (h_a : a ≥ 2) (h_b : b ≥ 2) : 
  (ab - 1) % ((a - 1) * (b - 1)) = 0 ↔ (a = 2 ∧ b = 2) ∨ (a = 3 ∧ b = 3) :=
sorry

end ab_divisibility_1_1749


namespace proof_problem_1_1977

axiom sqrt (x : ℝ) : ℝ
axiom cbrt (x : ℝ) : ℝ
noncomputable def sqrtValue : ℝ :=
  sqrt 81

theorem proof_problem (m n : ℝ) (hm : sqrt m = 3) (hn : cbrt n = -4) : sqrt (2 * m - n - 1) = 9 ∨ sqrt (2 * m - n - 1) = -9 :=
by
  sorry

end proof_problem_1_1977


namespace common_remainder_proof_1_1156

def least_subtracted := 6
def original_number := 1439
def reduced_number := original_number - least_subtracted
def divisors := [5, 11, 13]
def common_remainder := 3

theorem common_remainder_proof :
  ∀ d ∈ divisors, reduced_number % d = common_remainder := by
  sorry

end common_remainder_proof_1_1156


namespace absolute_value_positive_1_1030

theorem absolute_value_positive (a : ℝ) (h : a ≠ 0) : |a| > 0 := by
  sorry

end absolute_value_positive_1_1030


namespace probability_intersection_1_1310

variables (A B : Type → Prop)

-- Assuming we have a measure space (probability) P
variables {P : Type → Prop}

-- Given probabilities
def p_A := 0.65
def p_B := 0.55
def p_Ac_Bc := 0.20

-- The theorem to be proven
theorem probability_intersection :
  (p_A + p_B - (1 - p_Ac_Bc) = 0.40) :=
by
  sorry

end probability_intersection_1_1310


namespace part_1_part_2_part_3_1_1592

variable {f : ℝ → ℝ}

axiom C1 : ∀ x y : ℝ, f (x + y) = f x + f y
axiom C2 : ∀ x : ℝ, x > 0 → f x < 0
axiom C3 : f 3 = -4

theorem part_1 : f 0 = 0 :=
by
  sorry

theorem part_2 : ∀ x : ℝ, f (-x) = -f x :=
by
  sorry

theorem part_3 : ∀ x : ℝ, -9 ≤ x ∧ x ≤ 9 → f x ≤ 12 ∧ f x ≥ -12 :=
by
  sorry

end part_1_part_2_part_3_1_1592


namespace baker_remaining_cakes_1_1743

def initial_cakes : ℝ := 167.3
def sold_cakes : ℝ := 108.2
def remaining_cakes : ℝ := initial_cakes - sold_cakes

theorem baker_remaining_cakes : remaining_cakes = 59.1 := by
  sorry

end baker_remaining_cakes_1_1743


namespace imag_part_z_is_3_1_1861

namespace ComplexMultiplication

-- Define the imaginary unit i
def i := Complex.I

-- Define the complex number z
def z := (1 + 2 * i) * (2 - i)

-- Define the imaginary part of a complex number
def imag_part (z : ℂ) : ℂ := Complex.im z

-- Statement to prove: The imaginary part of z = 3
theorem imag_part_z_is_3 : imag_part z = 3 := by
  sorry

end ComplexMultiplication

end imag_part_z_is_3_1_1861


namespace series_sum_is_6_over_5_1_1261

noncomputable def series_sum : ℝ := ∑' n : ℕ, if n % 4 == 0 then 1 / (4^(n/4)) else 
                                          if n % 4 == 1 then 1 / (2 * 4^(n/4)) else 
                                          if n % 4 == 2 then -1 / (4^(n/4) * 4^(1/2)) else 
                                          -1 / (2 * 4^(n/4 + 1/2))

theorem series_sum_is_6_over_5 : series_sum = 6 / 5 := 
  sorry

end series_sum_is_6_over_5_1_1261


namespace evaluate_f_at_3_1_1988

noncomputable def f : ℝ → ℝ := sorry

axiom h_odd : ∀ x : ℝ, f (-x) = -f x 
axiom h_periodic : ∀ x : ℝ, f x = f (x + 4)
axiom h_def : ∀ x : ℝ, 0 < x ∧ x < 2 → f x = 2 * x^2

theorem evaluate_f_at_3 : f 3 = -2 := by
  sorry

end evaluate_f_at_3_1_1988


namespace concert_revenue_1_1830

-- Define the prices and attendees
def adult_price := 26
def teenager_price := 18
def children_price := adult_price / 2
def num_adults := 183
def num_teenagers := 75
def num_children := 28

-- Calculate total revenue
def total_revenue := num_adults * adult_price + num_teenagers * teenager_price + num_children * children_price

-- The goal is to prove that total_revenue equals 6472
theorem concert_revenue : total_revenue = 6472 :=
by
  sorry

end concert_revenue_1_1830


namespace abc_le_one_eighth_1_1340

theorem abc_le_one_eighth (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c)
  (h : a / (1 + a) + b / (1 + b) + c / (1 + c) = 1) : a * b * c ≤ 1 / 8 :=
by
  sorry

end abc_le_one_eighth_1_1340


namespace smallest_possible_sum_1_1488

-- Defining the conditions for x and y.
variables (x y : ℕ)

-- We need a theorem to formalize our question with the given conditions.
theorem smallest_possible_sum (hx : x > 0) (hy : y > 0) (hne : x ≠ y) (hxy : 1/x + 1/y = 1/24) : x + y = 100 :=
by
  sorry

end smallest_possible_sum_1_1488


namespace absolute_difference_probability_1_1191

-- Define the conditions
def num_red_marbles : ℕ := 1500
def num_black_marbles : ℕ := 2000
def total_marbles : ℕ := num_red_marbles + num_black_marbles

def P_s : ℚ :=
  let ways_to_choose_2_red := (num_red_marbles * (num_red_marbles - 1)) / 2
  let ways_to_choose_2_black := (num_black_marbles * (num_black_marbles - 1)) / 2
  let total_favorable_outcomes := ways_to_choose_2_red + ways_to_choose_2_black
  total_favorable_outcomes / (total_marbles * (total_marbles - 1) / 2)

def P_d : ℚ :=
  (num_red_marbles * num_black_marbles) / (total_marbles * (total_marbles - 1) / 2)

-- Prove the statement
theorem absolute_difference_probability : |P_s - P_d| = 1 / 50 := by
  sorry

end absolute_difference_probability_1_1191


namespace eccentricity_range_1_1687

def ellipse (a b : ℝ) (h : a > 0 ∧ b > 0 ∧ a > b) :=
  ∀ x y : ℝ, x^2 / a^2 + y^2 / b^2 = 1

def c_squared (a b : ℝ) : ℝ := a^2 - b^2

def perpendicular_condition (a b c x y : ℝ) : Prop :=
  (x - c, y).fst * (x + c, y).fst + (x - c, y).snd * (x + c, y).snd = 0

theorem eccentricity_range (a b e c x y : ℝ)
  (h : a > 0 ∧ b > 0 ∧ a > b)
  (h_ellipse : ellipse a b h)
  (h_perp : perpendicular_condition a b c x y) :
  (e = c / a ∧ 0 < e ∧ e < 1) →
  (√2 / 2 ≤ e ∧ e < 1) :=
by
  sorry

end eccentricity_range_1_1687


namespace total_area_of_forest_and_fields_1_1681

theorem total_area_of_forest_and_fields (r p k : ℝ) (h1 : k = 12) 
  (h2 : r^2 + 4 * p^2 + 45 = 12 * k) :
  (r^2 + 4 * p^2 + 12 * k = 135) :=
by
  -- Proof goes here
  sorry

end total_area_of_forest_and_fields_1_1681


namespace howard_items_1_1971

theorem howard_items (a b c : ℕ) (h1 : a + b + c = 40) (h2 : 40 * a + 300 * b + 400 * c = 5000) : a = 20 :=
by
  sorry

end howard_items_1_1971


namespace B_share_correct_1_1661

noncomputable def total_share : ℕ := 120
noncomputable def B_share : ℕ := 20
noncomputable def A_share (x : ℕ) : ℕ := x + 20
noncomputable def C_share (x : ℕ) : ℕ := x + 40

theorem B_share_correct : ∃ x : ℕ, total_share = (A_share x) + x + (C_share x) ∧ x = B_share := by
  sorry

end B_share_correct_1_1661


namespace solution_set_of_abs_inequality_1_1347

theorem solution_set_of_abs_inequality :
  { x : ℝ | |x^2 - 2| < 2 } = { x : ℝ | -2 < x ∧ x < 0 ∨ 0 < x ∧ x < 2 } :=
sorry

end solution_set_of_abs_inequality_1_1347


namespace star_neg5_4_star_neg3_neg6_1_1135

-- Definition of the new operation
def star (a b : ℤ) : ℤ := 2 * a * b - b / 2

-- The first proof problem
theorem star_neg5_4 : star (-5) 4 = -42 := by sorry

-- The second proof problem
theorem star_neg3_neg6 : star (-3) (-6) = 39 := by sorry

end star_neg5_4_star_neg3_neg6_1_1135


namespace find_a_range_for_two_distinct_roots_1_1252

def f (x : ℝ) : ℝ := x^3 - 3 * x + 5

theorem find_a_range_for_two_distinct_roots :
  ∀ (a : ℝ), 3 ≤ a ∧ a ≤ 7 → ∃ (x1 x2 : ℝ), x1 ≠ x2 ∧ f x1 = a ∧ f x2 = a :=
by
  -- The proof will be here
  sorry

end find_a_range_for_two_distinct_roots_1_1252


namespace sum_of_n_terms_1_1281

noncomputable def S : ℕ → ℕ :=
sorry -- We define S, but its exact form is not used in the statement directly

noncomputable def a : ℕ → ℕ := 
sorry -- We define a, but its exact form is not used in the statement directly

-- Conditions
axiom S3_eq : S 3 = 1
axiom a_rec : ∀ n : ℕ, 0 < n → a (n + 3) = 2 * (a n)

-- Proof problem
theorem sum_of_n_terms : S 2019 = 2^673 - 1 :=
sorry

end sum_of_n_terms_1_1281


namespace min_value_of_function_1_1263

theorem min_value_of_function (x : ℝ) (hx : x > 0) :
  ∃ y, y = (3 + x + x^2) / (1 + x) ∧ y = -1 + 2 * Real.sqrt 3 :=
sorry

end min_value_of_function_1_1263


namespace tangent_line_computation_1_1684

variables (f : ℝ → ℝ)

theorem tangent_line_computation (h_tangent : ∀ x, (f x = -x + 8) ∧ (∃ y, y = -x + 8 → (f y) = -x + 8 → deriv f x = -1)) :
    f 5 + deriv f 5 = 2 :=
sorry

end tangent_line_computation_1_1684


namespace power_mean_inequality_1_1113

variables {a b c : ℝ}
variables {n p q r : ℕ}

theorem power_mean_inequality 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (hn : 0 < n)
  (hpqr_nonneg : 0 ≤ p ∧ 0 ≤ q ∧ 0 ≤ r)
  (sum_pqr : p + q + r = n) :
  a^n + b^n + c^n ≥ a^p * b^q * c^r + a^r * b^p * c^q + a^q * b^r * c^p :=
sorry

end power_mean_inequality_1_1113


namespace estimate_time_pm_1_1798

-- Definitions from the conditions
def school_start_time : ℕ := 12
def classes : List String := ["Maths", "History", "Geography", "Science", "Music"]
def class_time : ℕ := 45  -- in minutes
def break_time : ℕ := 15  -- in minutes
def classes_up_to_science : List String := ["Maths", "History", "Geography", "Science"]
def total_classes_time : ℕ := classes_up_to_science.length * (class_time + break_time)

-- Lean statement to prove that given the conditions, the time is 4 pm
theorem estimate_time_pm :
  school_start_time + (total_classes_time / 60) = 16 :=
by
  sorry

end estimate_time_pm_1_1798


namespace correct_propositions_1_1918

variables {Line Plane : Type}
variables (m n : Line) (α β : Plane)

-- Assume basic predicates for lines and planes
variable (parallel : Line → Line → Prop)
variable (perp : Line → Plane → Prop)
variable (subset : Line → Plane → Prop)
variable (planar_parallel : Plane → Plane → Prop)

-- Stating the theorem to be proved
theorem correct_propositions :
  (parallel m n ∧ perp m α → perp n α) ∧ 
  (planar_parallel α β ∧ parallel m n ∧ perp m α → perp n β) :=
by
  sorry

end correct_propositions_1_1918


namespace sin_inequality_iff_angle_inequality_1_1642

section
variables {A B : ℝ} {a b : ℝ} (R : ℝ) (hA : A = Real.sin a) (hB : B = Real.sin b)

theorem sin_inequality_iff_angle_inequality (A B : ℝ) :
  (A > B) ↔ (Real.sin A > Real.sin B) :=
sorry
end

end sin_inequality_iff_angle_inequality_1_1642


namespace Emily_used_10_dimes_1_1020

theorem Emily_used_10_dimes
  (p n d : ℕ)
  (h1 : p + n + d = 50)
  (h2 : p + 5 * n + 10 * d = 200) :
  d = 10 := by
  sorry

end Emily_used_10_dimes_1_1020


namespace total_amount_spent_on_cookies_1_1597

def days_in_april : ℕ := 30
def cookies_per_day : ℕ := 3
def cost_per_cookie : ℕ := 18

theorem total_amount_spent_on_cookies : days_in_april * cookies_per_day * cost_per_cookie = 1620 := by
  sorry

end total_amount_spent_on_cookies_1_1597


namespace cubic_root_equality_1_1453

theorem cubic_root_equality (a b c : ℝ) (h1 : a + b + c = 12) (h2 : a * b + b * c + c * a = 14) (h3 : a * b * c = -3) :
  (1 / a^2) + (1 / b^2) + (1 / c^2) = 268 / 9 := 
by
  sorry

end cubic_root_equality_1_1453


namespace Jessie_lost_7_kilograms_1_1322

def Jessie_previous_weight : ℕ := 74
def Jessie_current_weight : ℕ := 67
def Jessie_weight_lost : ℕ := Jessie_previous_weight - Jessie_current_weight

theorem Jessie_lost_7_kilograms : Jessie_weight_lost = 7 :=
by
  sorry

end Jessie_lost_7_kilograms_1_1322


namespace max_ounces_among_items_1_1144

theorem max_ounces_among_items
  (budget : ℝ)
  (candy_cost : ℝ)
  (candy_ounces : ℝ)
  (candy_stock : ℕ)
  (chips_cost : ℝ)
  (chips_ounces : ℝ)
  (chips_stock : ℕ)
  : budget = 7 → candy_cost = 1.25 → candy_ounces = 12 →
    candy_stock = 5 → chips_cost = 1.40 → chips_ounces = 17 → chips_stock = 4 →
    max (min ((budget / candy_cost) * candy_ounces) (candy_stock * candy_ounces))
        (min ((budget / chips_cost) * chips_ounces) (chips_stock * chips_ounces)) = 68 := 
by
  intros h_budget h_candy_cost h_candy_ounces h_candy_stock h_chips_cost h_chips_ounces h_chips_stock
  sorry

end max_ounces_among_items_1_1144


namespace opposite_sqrt3_1_1007

def opposite (x : ℝ) : ℝ := -x

theorem opposite_sqrt3 :
  opposite (Real.sqrt 3) = -Real.sqrt 3 :=
by
  sorry

end opposite_sqrt3_1_1007


namespace fraction_of_innocent_cases_1_1992

-- Definitions based on the given conditions
def total_cases : ℕ := 17
def dismissed_cases : ℕ := 2
def delayed_cases : ℕ := 1
def guilty_cases : ℕ := 4

-- The remaining cases after dismissals
def remaining_cases : ℕ := total_cases - dismissed_cases

-- The remaining cases that are not innocent
def non_innocent_cases : ℕ := delayed_cases + guilty_cases

-- The innocent cases
def innocent_cases : ℕ := remaining_cases - non_innocent_cases

-- The fraction of the remaining cases that were ruled innocent
def fraction_innocent : Rat := innocent_cases / remaining_cases

-- The theorem we want to prove
theorem fraction_of_innocent_cases :
  fraction_innocent = 2 / 3 := by
  sorry

end fraction_of_innocent_cases_1_1992


namespace find_swimming_speed_1_1997

variable (S : ℝ)

def is_average_speed (x y avg : ℝ) : Prop :=
  avg = 2 * x * y / (x + y)

theorem find_swimming_speed
  (running_speed : ℝ := 7)
  (average_speed : ℝ := 4)
  (h : is_average_speed S running_speed average_speed) :
  S = 2.8 :=
by sorry

end find_swimming_speed_1_1997


namespace rainfall_difference_1_1203

-- Define the conditions
def first_day_rainfall : ℕ := 26
def second_day_rainfall : ℕ := 34
def third_day_rainfall : ℕ := second_day_rainfall - 12
def total_rainfall_this_year : ℕ := first_day_rainfall + second_day_rainfall + third_day_rainfall
def average_rainfall : ℕ := 140

-- Define the statement to prove
theorem rainfall_difference : average_rainfall - total_rainfall_this_year = 58 := by
  -- Add your proof here
  sorry

end rainfall_difference_1_1203


namespace paint_required_for_small_statues_1_1369

-- Constants definition
def pint_per_8ft_statue : ℕ := 1
def height_original_statue : ℕ := 8
def height_small_statue : ℕ := 2
def number_of_small_statues : ℕ := 400

-- Theorem statement
theorem paint_required_for_small_statues :
  pint_per_8ft_statue = 1 →
  height_original_statue = 8 →
  height_small_statue = 2 →
  number_of_small_statues = 400 →
  (number_of_small_statues * (pint_per_8ft_statue * (height_small_statue / height_original_statue)^2)) = 25 :=
by
  intros h1 h2 h3 h4
  sorry

end paint_required_for_small_statues_1_1369


namespace completing_the_square_transformation_1_1696

theorem completing_the_square_transformation (x : ℝ) : 
  (x^2 - 4 * x + 1 = 0) → ((x - 2)^2 = 3) :=
by
  intro h
  -- Transformation steps will be shown in the proof
  sorry

end completing_the_square_transformation_1_1696


namespace infinitely_many_c_exist_1_1069

theorem infinitely_many_c_exist :
  ∃ c: ℕ, ∃ x y z: ℕ, (x^2 - c) * (y^2 - c) = z^2 - c ∧ (x^2 + c) * (y^2 - c) = z^2 - c :=
by
  sorry

end infinitely_many_c_exist_1_1069


namespace valentines_given_1_1025

theorem valentines_given (x y : ℕ) (h : x * y = x + y + 40) : x * y = 84 :=
by
  -- solving for x, y based on the factors of 41
  sorry

end valentines_given_1_1025


namespace eval_expr_1_1111

theorem eval_expr : (3^3)^2 = 729 := 
by
  sorry

end eval_expr_1_1111


namespace matinee_ticket_price_1_1219

theorem matinee_ticket_price
  (M : ℝ)  -- Denote M as the price of a matinee ticket
  (evening_ticket_price : ℝ := 12)  -- Price of an evening ticket
  (ticket_3D_price : ℝ := 20)  -- Price of a 3D ticket
  (matinee_tickets_sold : ℕ := 200)  -- Number of matinee tickets sold
  (evening_tickets_sold : ℕ := 300)  -- Number of evening tickets sold
  (tickets_3D_sold : ℕ := 100)  -- Number of 3D tickets sold
  (total_revenue : ℝ := 6600) -- Total revenue
  (h : matinee_tickets_sold * M + evening_tickets_sold * evening_ticket_price + tickets_3D_sold * ticket_3D_price = total_revenue) :
  M = 5 :=
by
  sorry

end matinee_ticket_price_1_1219


namespace common_solutions_y_values_1_1530

theorem common_solutions_y_values :
  (∃ x : ℝ, x^2 + y^2 - 16 = 0 ∧ x^2 - 3*y - 12 = 0) ↔ (y = -4 ∨ y = 1) :=
by {
  sorry
}

end common_solutions_y_values_1_1530


namespace fixed_line_of_midpoint_1_1688

theorem fixed_line_of_midpoint
  (A B : ℝ × ℝ)
  (H : ∀ (P : ℝ × ℝ), (P = A ∨ P = B) → (P.1^2 / 3 - P.2^2 / 6 = 1))
  (slope_l : (B.2 - A.2) / (B.1 - A.1) = 2)
  (midpoint_lies : (A.1 + B.1) / 2 = (A.2 + B.2) / 2) :
  ∀ (M : ℝ × ℝ), (M.1 = (A.1 + B.1) / 2 ∧ M.2 = (A.2 + B.2) / 2) → M.1 - M.2 = 0 :=
by
  sorry

end fixed_line_of_midpoint_1_1688


namespace total_area_of_field_1_1770

noncomputable def total_field_area (A1 A2 : ℝ) : ℝ := A1 + A2

theorem total_area_of_field :
  ∀ (A1 A2 : ℝ),
    A1 = 405 ∧ (A2 - A1 = (1/5) * ((A1 + A2) / 2)) →
    total_field_area A1 A2 = 900 :=
by
  intros A1 A2 h
  sorry

end total_area_of_field_1_1770


namespace find_sale_month_4_1_1325

-- Definitions based on the given conditions
def avg_sale_per_month : ℕ := 6500
def num_months : ℕ := 6
def sale_month_1 : ℕ := 6435
def sale_month_2 : ℕ := 6927
def sale_month_3 : ℕ := 6855
def sale_month_5 : ℕ := 6562
def sale_month_6 : ℕ := 4991

theorem find_sale_month_4 : 
  (avg_sale_per_month * num_months) - (sale_month_1 + sale_month_2 + sale_month_3 + sale_month_5 + sale_month_6) = 7230 :=
by
  -- The proof will be provided below
  sorry

end find_sale_month_4_1_1325


namespace minimum_n_minus_m_1_1630

noncomputable def f (x : Real) : Real :=
    (Real.sin x) * (Real.sin (x + Real.pi / 3)) - 1 / 4

theorem minimum_n_minus_m (m n : Real) (h : m < n) 
  (h_domain : ∀ x, m ≤ x ∧ x ≤ n → -1 / 2 ≤ f x ∧ f x ≤ 1 / 4) :
  n - m = 2 * Real.pi / 3 :=
by
  sorry

end minimum_n_minus_m_1_1630


namespace ellipse_equation_and_fixed_point_proof_1_1883

theorem ellipse_equation_and_fixed_point_proof :
  (∀ (m n : ℝ), (m > 0) → (n > 0) → (m ≠ n) →
                (m * 0^2 + n * (-2)^2 = 1) ∧ (m * (3/2)^2 + n * (-1)^2 = 1) → 
                (m = 1/3 ∧ n = 1/4)) ∧
                (∀ (M N : ℝ × ℝ), ∃ H : ℝ × ℝ,
                (∃ (P : ℝ × ℝ), P = (1, -2)) ∧
                (∃ (A : ℝ × ℝ), A = (0, -2)) ∧
                (∃ (B : ℝ × ℝ), B = (3/2, -1)) ∧
                (∃ (T : ℝ × ℝ), ∀ x, M.1 = x) ∧
                (∃ K : ℝ × ℝ, K = (0, -2)) →
                M.1 * N.2 - M.2 * N.1 = 0) :=
by
  sorry

end ellipse_equation_and_fixed_point_proof_1_1883


namespace smallest_positive_period_symmetry_axis_range_of_f_1_1808
noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x + (Real.pi / 6))

theorem smallest_positive_period :
  ∃ T > 0, ∀ x, f (x + T) = f x ∧ T = Real.pi := sorry

theorem symmetry_axis (k : ℤ) :
  ∃ k : ℤ, ∃ x : ℝ, f x = f (x + k * (Real.pi / 2)) ∧ x = (Real.pi / 6) + k * (Real.pi / 2) := sorry

theorem range_of_f : 
  ∀ x, -Real.pi / 12 ≤ x ∧ x ≤ Real.pi / 2 → -1/2 ≤ f x ∧ f x ≤ 1 := sorry

end smallest_positive_period_symmetry_axis_range_of_f_1_1808


namespace polynomial_sum_at_points_1_1212

def P (x : ℝ) : ℝ := x^5 - 1.7 * x^3 + 2.5

theorem polynomial_sum_at_points :
  P 19.1 + P (-19.1) = 5 := by
  sorry

end polynomial_sum_at_points_1_1212


namespace condition_an_necessary_but_not_sufficient_1_1537

-- Definitions for the sequence and properties
def is_geometric_sequence (a : ℕ → ℝ) (r : ℝ) : Prop :=
  ∀ n, n ≥ 1 → a (n + 1) = r * (a n)

def condition_an (a : ℕ → ℝ) : Prop :=
  ∀ n, n ≥ 2 → a n = 2 * a (n - 1)

-- The theorem statement
theorem condition_an_necessary_but_not_sufficient (a : ℕ → ℝ) :
  (∀ n, n ≥ 1 → a (n + 1) = 2 * (a n)) → (condition_an a) ∧ ¬(is_geometric_sequence a 2) :=
by
  sorry

end condition_an_necessary_but_not_sufficient_1_1537


namespace solve_ineq_system_1_1105

theorem solve_ineq_system (x : ℝ) :
  (x - 1) / (x + 2) ≤ 0 ∧ x^2 - 2 * x - 3 < 0 ↔ -1 < x ∧ x ≤ 1 :=
by sorry

end solve_ineq_system_1_1105


namespace spinsters_count_1_1806

theorem spinsters_count (S C : ℕ) (h1 : S / C = 2 / 9) (h2 : C = S + 42) : S = 12 := by
  sorry

end spinsters_count_1_1806


namespace volume_hemisphere_from_sphere_1_1435

theorem volume_hemisphere_from_sphere (r : ℝ) (V_sphere : ℝ) (V_hemisphere : ℝ) 
  (h1 : V_sphere = 150 * Real.pi) 
  (h2 : V_sphere = (4 / 3) * Real.pi * r^3) : 
  V_hemisphere = 75 * Real.pi :=
by
  sorry

end volume_hemisphere_from_sphere_1_1435


namespace factor_poly_eq_factored_form_1_1130

-- Defining the polynomial expressions
def poly1 (x : ℝ) := x^2 + 4 * x + 3
def poly2 (x : ℝ) := x^2 + 8 * x + 15
def poly3 (x : ℝ) := x^2 + 6 * x - 8

-- The main expression which needs to be factored
def main_expr (x : ℝ) := (poly1 x) * (poly2 x) + (poly3 x)

-- Stating the goal factored form
def factored_form (x : ℝ) := (x^2 + 6 * x + 19) * (x^2 + 6 * x - 2)

-- The theorem statement
theorem factor_poly_eq_factored_form (x : ℝ) : 
  main_expr x = factored_form x := 
by
  sorry

end factor_poly_eq_factored_form_1_1130


namespace find_sum_of_natural_numbers_1_1457

theorem find_sum_of_natural_numbers :
  ∃ (square triangle : ℕ), square^2 + 12 = triangle^2 ∧ square + triangle = 6 :=
by
  sorry

end find_sum_of_natural_numbers_1_1457


namespace combined_weight_is_correct_1_1836

def EvanDogWeight := 63
def IvanDogWeight := EvanDogWeight / 7
def CombinedWeight := EvanDogWeight + IvanDogWeight

theorem combined_weight_is_correct 
: CombinedWeight = 72 :=
by 
  sorry

end combined_weight_is_correct_1_1836


namespace problem_1_1064

theorem problem (x y : ℕ) (hxpos : 0 < x ∧ x < 20) (hypos : 0 < y ∧ y < 20) (h : x + y + x * y = 119) : 
  x + y = 24 ∨ x + y = 21 ∨ x + y = 20 :=
by sorry

end problem_1_1064


namespace function_relation_1_1239

theorem function_relation (f : ℝ → ℝ) 
  (h0 : ∀ x, f (-x) = f x)
  (h1 : ∀ x, f (x + 2) = f x)
  (h2 : ∀ x y, 0 ≤ x ∧ x < y ∧ y ≤ 1 → f x < f y) :
  f 0 < f (-6.5) ∧ f (-6.5) < f (-1) := 
by
  sorry

end function_relation_1_1239


namespace vector_dot_product_1_1376

-- Definitions of the vectors
def a : ℝ × ℝ := (2, 1)
def b : ℝ × ℝ := (-1, 2)

-- Definition of the dot product for 2D vectors
def dot_product (u v : ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2

-- Main statement to prove
theorem vector_dot_product :
  dot_product (a.1 + b.1, a.2 + b.2) (a.1 - b.1, a.2 - b.2) = 0 :=
by
  sorry

end vector_dot_product_1_1376


namespace jake_earnings_per_hour_1_1116

-- Definitions for conditions
def initialDebt : ℕ := 100
def payment : ℕ := 40
def hoursWorked : ℕ := 4
def remainingDebt : ℕ := initialDebt - payment

-- Theorem stating Jake's earnings per hour
theorem jake_earnings_per_hour : remainingDebt / hoursWorked = 15 := by
  sorry

end jake_earnings_per_hour_1_1116


namespace necessary_but_not_sufficient_condition_for_a_lt_neg_one_1_1285

theorem necessary_but_not_sufficient_condition_for_a_lt_neg_one (a : ℝ) : 
  (1 / a > -1) ↔ (a < -1) :=
by sorry

end necessary_but_not_sufficient_condition_for_a_lt_neg_one_1_1285


namespace find_m_1_1231

noncomputable def f (x a : ℝ) : ℝ := x - a

theorem find_m (a m : ℝ) :
  (∀ x, 0 ≤ x ∧ x ≤ 4 → f x a ≤ 2) →
  (∃ x, -2 ≤ x ∧ x ≤ 4 ∧ -1 - f (x + 1) a ≤ m) :=
sorry

end find_m_1_1231


namespace evaluate_expression_1_1896

theorem evaluate_expression : (2^3002 * 3^3004) / (6^3003) = (3 / 2) := by
  sorry

end evaluate_expression_1_1896


namespace hexagon_unique_intersection_points_are_45_1_1596

-- Definitions related to hexagon for the proof problem
def hexagon_vertices : ℕ := 6
def sides_of_hexagon : ℕ := 6
def diagonals_of_hexagon : ℕ := 9
def total_line_segments : ℕ := 15
def total_intersections : ℕ := 105
def vertex_intersections_per_vertex : ℕ := 10
def total_vertex_intersections : ℕ := 60

-- Final Proof Statement that needs to be proved
theorem hexagon_unique_intersection_points_are_45 :
  total_intersections - total_vertex_intersections = 45 :=
by
  sorry

end hexagon_unique_intersection_points_are_45_1_1596


namespace tarun_garden_area_1_1831

theorem tarun_garden_area :
  ∀ (side : ℝ), 
  (1500 / 8 = 4 * side) → 
  (30 * side = 1500) → 
  side^2 = 2197.265625 :=
by
  sorry

end tarun_garden_area_1_1831


namespace blocks_per_box_1_1635

theorem blocks_per_box (total_blocks : ℕ) (boxes : ℕ) (h1 : total_blocks = 16) (h2 : boxes = 8) : total_blocks / boxes = 2 :=
by
  sorry

end blocks_per_box_1_1635


namespace starting_number_of_sequence_1_1978

theorem starting_number_of_sequence :
  ∃ (start : ℤ), 
    (∀ n, 0 ≤ n ∧ n < 8 → start + n * 11 ≤ 119) ∧ 
    (∃ k, 1 ≤ k ∧ k ≤ 8 ∧ 119 = start + (k - 1) * 11) ↔ start = 33 :=
by
  sorry

end starting_number_of_sequence_1_1978


namespace optimal_years_minimize_cost_1_1201

noncomputable def initial_cost : ℝ := 150000
noncomputable def annual_expenses (n : ℕ) : ℝ := 15000 * n
noncomputable def maintenance_cost (n : ℕ) : ℝ := (n * (3000 + 3000 * n)) / 2
noncomputable def total_cost (n : ℕ) : ℝ := initial_cost + annual_expenses n + maintenance_cost n
noncomputable def average_annual_cost (n : ℕ) : ℝ := total_cost n / n

theorem optimal_years_minimize_cost : ∀ n : ℕ, n = 10 ↔ average_annual_cost 10 ≤ average_annual_cost n :=
by sorry

end optimal_years_minimize_cost_1_1201


namespace sum_of_exponents_1_1152

def power_sum_2021 (a : ℕ → ℤ) (n : ℕ → ℕ) (r : ℕ) : Prop :=
  (∀ k, 1 ≤ k ∧ k ≤ r → (a k = 1 ∨ a k = -1)) ∧
  (a 1 * 3 ^ n 1 + a 2 * 3 ^ n 2 + a 3 * 3 ^ n 3 + a 4 * 3 ^ n 4 + a 5 * 3 ^ n 5 + a 6 * 3 ^ n 6 = 2021) ∧
  (n 1 = 7 ∧ n 2 = 5 ∧ n 3 = 4 ∧ n 4 = 2 ∧ n 5 = 1 ∧ n 6 = 0) ∧
  (a 1 = 1 ∧ a 2 = -1 ∧ a 3 = 1 ∧ a 4 = -1 ∧ a 5 = 1 ∧ a 6 = -1)

theorem sum_of_exponents : ∃ (a : ℕ → ℤ) (n : ℕ → ℕ) (r : ℕ), power_sum_2021 a n r ∧ (n 1 + n 2 + n 3 + n 4 + n 5 + n 6 = 19) :=
by {
  sorry
}

end sum_of_exponents_1_1152


namespace area_of_trapezoid_1_1371

-- Definitions of geometric properties and conditions
def is_perpendicular (a b c : ℝ) : Prop := a + b = 90 -- representing ∠ABC = 90°
def tangent_length (bc ad : ℝ) (O : ℝ) : Prop := bc * ad = O -- representing BC tangent to O with diameter AD
def is_diameter (ad r : ℝ) : Prop := ad = 2 * r -- AD being the diameter of the circle with radius r

-- Given conditions in the problem
variables (AB BC CD AD r O : ℝ) (h1 : is_perpendicular AB BC 90) (h2 : is_perpendicular BC CD 90)
          (h3 : tangent_length BC AD O) (h4 : is_diameter AD r) (h5 : BC = 2 * CD)
          (h6 : AB = 9) (h7 : CD = 3)

-- Statement to prove the area is 36
theorem area_of_trapezoid : (AB + CD) * CD = 36 := by
  sorry

end area_of_trapezoid_1_1371


namespace opposite_of_neg_six_1_1737

theorem opposite_of_neg_six : -(-6) = 6 := 
by
  sorry

end opposite_of_neg_six_1_1737


namespace cost_price_of_apple_is_18_1_1233

noncomputable def cp (sp : ℝ) (loss_fraction : ℝ) : ℝ := sp / (1 - loss_fraction)

theorem cost_price_of_apple_is_18 :
  cp 15 (1/6) = 18 :=
by
  sorry

end cost_price_of_apple_is_18_1_1233


namespace annette_miscalculation_1_1313

theorem annette_miscalculation :
  let x := 6
  let y := 3
  let x' := 5
  let y' := 4
  x' - y' = 1 :=
by
  let x := 6
  let y := 3
  let x' := 5
  let y' := 4
  sorry

end annette_miscalculation_1_1313


namespace isosceles_triangle_side_length_1_1523

theorem isosceles_triangle_side_length (a b : ℝ) (h : a < b) : 
  ∃ l : ℝ, l = (b - a) / 2 := 
sorry

end isosceles_triangle_side_length_1_1523


namespace Adam_bought_9_cat_food_packages_1_1767

def num_cat_food_packages (c : ℕ) : Prop :=
  let cat_cans := 10 * c
  let dog_cans := 7 * 5
  cat_cans = dog_cans + 55

theorem Adam_bought_9_cat_food_packages : num_cat_food_packages 9 :=
by
  unfold num_cat_food_packages
  sorry

end Adam_bought_9_cat_food_packages_1_1767


namespace contestant_wins_probability_1_1876

-- Define the basic parameters: number of questions and number of choices
def num_questions : ℕ := 4
def num_choices : ℕ := 3

-- Define the probability of getting a single question right
def prob_right : ℚ := 1 / num_choices

-- Define the probability of guessing all questions right
def prob_all_right : ℚ := prob_right ^ num_questions

-- Define the probability of guessing exactly three questions right (one wrong)
def prob_one_wrong : ℚ := (prob_right ^ 3) * (2 / num_choices)

-- Calculate the total probability of winning
def total_prob_winning : ℚ := prob_all_right + 4 * prob_one_wrong

-- The final statement to prove
theorem contestant_wins_probability :
  total_prob_winning = 1 / 9 := 
sorry

end contestant_wins_probability_1_1876


namespace seating_arrangement_1_1387

def numWaysCableCars (adults children cars capacity : ℕ) : ℕ := 
  sorry 

theorem seating_arrangement :
  numWaysCableCars 4 2 3 3 = 348 :=
by {
  sorry
}

end seating_arrangement_1_1387


namespace line_circle_intersection_1_1935

theorem line_circle_intersection (a : ℝ) : 
  (∀ x y : ℝ, (4 * x + 3 * y + a = 0) → ((x - 1)^2 + (y - 2)^2 = 9)) ∧
  (∃ A B : ℝ, dist A B = 4 * Real.sqrt 2) →
  (a = -5 ∨ a = -15) :=
by 
  sorry

end line_circle_intersection_1_1935


namespace pears_value_equivalence_1_1359

-- Condition: $\frac{3}{4}$ of $16$ apples are worth $12$ pears
def apples_to_pears (a p : ℕ) : Prop :=
  (3 * 16 / 4 * a = 12 * p)

-- Question: How many pears (p) are equivalent in value to $\frac{2}{3}$ of $9$ apples?
def pears_equivalent_to_apples (p : ℕ) : Prop :=
  (2 * 9 / 3 * p = 6)

theorem pears_value_equivalence (p : ℕ) (a : ℕ) (h1 : apples_to_pears a p) (h2 : pears_equivalent_to_apples p) : 
  p = 6 :=
sorry

end pears_value_equivalence_1_1359


namespace complement_of_M_in_U_1_1915

def U := Set.univ (α := ℝ)
def M := {x : ℝ | x < -2 ∨ x > 8}
def compl_M := {x : ℝ | -2 ≤ x ∧ x ≤ 8}

theorem complement_of_M_in_U : compl_M = U \ M :=
by
  sorry

end complement_of_M_in_U_1_1915


namespace kangaroo_chase_1_1882

noncomputable def time_to_catch_up (jumps_baby: ℕ) (jumps_mother: ℕ) (time_period: ℕ) 
  (jump_dist_mother: ℕ) (jump_dist_reduction_factor: ℕ) 
  (initial_baby_jumps: ℕ): ℕ :=
  let jump_dist_baby := jump_dist_mother / jump_dist_reduction_factor
  let distance_mother := jumps_mother * jump_dist_mother
  let distance_baby := jumps_baby * jump_dist_baby
  let relative_velocity := distance_mother - distance_baby
  let initial_distance := initial_baby_jumps * jump_dist_baby
  (initial_distance / relative_velocity) * time_period

theorem kangaroo_chase :
 ∀ (jumps_baby: ℕ) (jumps_mother: ℕ) (time_period: ℕ) 
   (jump_dist_mother: ℕ) (jump_dist_reduction_factor: ℕ) 
   (initial_baby_jumps: ℕ),
  jumps_baby = 5 ∧ jumps_mother = 3 ∧ time_period = 2 ∧ 
  jump_dist_mother = 6 ∧ jump_dist_reduction_factor = 3 ∧ 
  initial_baby_jumps = 12 →
  time_to_catch_up jumps_baby jumps_mother time_period jump_dist_mother 
    jump_dist_reduction_factor initial_baby_jumps = 6 := 
by
  intros jumps_baby jumps_mother time_period jump_dist_mother 
    jump_dist_reduction_factor initial_baby_jumps _; sorry

end kangaroo_chase_1_1882


namespace jogging_walking_ratio_1_1256

theorem jogging_walking_ratio (total_time walk_time jog_time: ℕ) (h1 : total_time = 21) (h2 : walk_time = 9) (h3 : jog_time = total_time - walk_time) :
  (jog_time : ℚ) / walk_time = 4 / 3 :=
by
  sorry

end jogging_walking_ratio_1_1256


namespace quadrilateral_side_squares_inequality_1_1730

theorem quadrilateral_side_squares_inequality :
  ∀ (x1 y1 x2 y2 : ℝ),
    0 ≤ x1 ∧ x1 ≤ 1 ∧ 0 ≤ y1 ∧ y1 ≤ 1 ∧
    0 ≤ x2 ∧ x2 ≤ 1 ∧ 0 ≤ y2 ∧ y2 ≤ 1 →
    2 ≤ (x1 - 1)^2 + y1^2 + (x2 - 1)^2 + (y1 - 1)^2 + x2^2 + (y2 - 1)^2 + x1^2 + y2^2 ∧ 
          (x1 - 1)^2 + y1^2 + (x2 - 1)^2 + (y1 - 1)^2 + x2^2 + (y2 - 1)^2 + x1^2 + y2^2 ≤ 4 :=
by
  intro x1 y1 x2 y2 h
  sorry

end quadrilateral_side_squares_inequality_1_1730


namespace part1_part2_1_1073

-- Define all given conditions
variable {A B C AC BC : ℝ}
variable (A_in_range : 0 < A ∧ A < π/2)
variable (B_in_range : 0 < B ∧ B < π/2)
variable (C_in_range : 0 < C ∧ C < π/2)
variable (m_perp_n : (Real.cos (A + π/3) * Real.cos B) + (Real.sin (A + π/3) * Real.sin B) = 0)
variable (cos_B : Real.cos B = 3/5)
variable (AC_value : AC = 8)

-- First part: Prove A - B = π/6
theorem part1 : A - B = π / 6 :=
by
  sorry

-- Second part: Prove BC = 4√3 + 3 given additional conditions
theorem part2 : BC = 4 * Real.sqrt 3 + 3 :=
by
  sorry

end part1_part2_1_1073


namespace smallest_odd_digit_number_gt_1000_mult_5_1_1987

def is_odd_digit (n : ℕ) : Prop := n = 1 ∨ n = 3 ∨ n = 5 ∨ n = 7 ∨ n = 9

def valid_number (n : ℕ) : Prop :=
  n > 1000 ∧ (∃ d1 d2 d3 d4, n = d1 * 1000 + d2 * 100 + d3 * 10 + d4 ∧ 
  is_odd_digit d1 ∧ is_odd_digit d2 ∧ is_odd_digit d3 ∧ is_odd_digit d4 ∧ 
  d4 = 5)

theorem smallest_odd_digit_number_gt_1000_mult_5 : ∃ n : ℕ, valid_number n ∧ 
  ∀ m : ℕ, valid_number m → m ≥ n := 
by
  use 1115
  simp [valid_number, is_odd_digit]
  sorry

end smallest_odd_digit_number_gt_1000_mult_5_1_1987


namespace mutually_exclusive_events_1_1241

-- Define the conditions
variable (redBalls greenBalls : ℕ)
variable (n : ℕ) -- Number of balls drawn
variable (event_one_red_ball event_two_green_balls : Prop)

-- Assumptions: more than two red balls and more than two green balls
axiom H1 : 2 < redBalls
axiom H2 : 2 < greenBalls

-- Assume that exactly one red ball and exactly two green balls are events
axiom H3 : event_one_red_ball = (n = 2 ∧ 1 ≤ redBalls ∧ 1 ≤ greenBalls)
axiom H4 : event_two_green_balls = (n = 2 ∧ greenBalls ≥ 2)

-- Definition of mutually exclusive events
def mutually_exclusive (A B : Prop) : Prop :=
  A ∧ B → false

-- Statement of the theorem
theorem mutually_exclusive_events :
  mutually_exclusive event_one_red_ball event_two_green_balls :=
by {
  sorry
}

end mutually_exclusive_events_1_1241


namespace ted_alex_age_ratio_1_1209

theorem ted_alex_age_ratio (t a : ℕ) 
  (h1 : t - 3 = 4 * (a - 3))
  (h2 : t - 5 = 5 * (a - 5)) : 
  ∃ x : ℕ, (t + x) / (a + x) = 3 ∧ x = 1 :=
by
  sorry

end ted_alex_age_ratio_1_1209


namespace y_share_1_1845

theorem y_share (total_amount : ℝ) (x_share y_share z_share : ℝ)
  (hx : x_share = 1) (hy : y_share = 0.45) (hz : z_share = 0.30)
  (h_total : total_amount = 105) :
  (60 * y_share) = 27 :=
by
  have h_cycle : 1 + y_share + z_share = 1.75 := by sorry
  have h_num_cycles : total_amount / 1.75 = 60 := by sorry
  sorry

end y_share_1_1845


namespace games_within_division_1_1677

theorem games_within_division (N M : ℕ) (h1 : N > 3 * M) (h2 : M > 5) (h3 : 2 * N + 6 * M = 76) : 2 * N = 40 :=
by {
  sorry
}

end games_within_division_1_1677


namespace frequency_of_middle_group_1_1666

theorem frequency_of_middle_group (sample_size : ℕ) (x : ℝ) (h : sample_size = 160) (h_rel_freq : x = 0.2) 
  (h_relation : x = (1 / 4) * (10 * x)) : 
  sample_size * x = 32 :=
by
  sorry

end frequency_of_middle_group_1_1666


namespace sum_of_x_values_1_1483

theorem sum_of_x_values (y x : ℝ) (h1 : y = 6) (h2 : x^2 + y^2 = 144) : x + (-x) = 0 :=
by
  sorry

end sum_of_x_values_1_1483


namespace geometric_product_formula_1_1449

variable (b : ℕ → ℝ) (n : ℕ)
variable (h_pos : ∀ i, b i > 0) (h_npos : n > 0)

noncomputable def T_n := (∏ i in Finset.range n, b i)

theorem geometric_product_formula
  (hn : 0 < n) (hb : ∀ i < n, 0 < b i):
  T_n b n = (b 0 * b (n-1)) ^ (n / 2) :=
sorry

end geometric_product_formula_1_1449


namespace students_like_burgers_1_1286

theorem students_like_burgers (total_students : ℕ) (french_fries_likers : ℕ) (both_likers : ℕ) (neither_likers : ℕ) 
    (h1 : total_students = 25) (h2 : french_fries_likers = 15) (h3 : both_likers = 6) (h4 : neither_likers = 6) : 
    (total_students - neither_likers) - (french_fries_likers - both_likers) = 10 :=
by
  -- The proof will go here.
  sorry

end students_like_burgers_1_1286


namespace polynomial_q_correct_1_1445

noncomputable def polynomial_q (x : ℝ) : ℝ :=
  -x^6 + 12*x^5 + 9*x^4 + 14*x^3 - 5*x^2 + 17*x + 1

noncomputable def polynomial_rhs (x : ℝ) : ℝ :=
  x^6 + 12*x^5 + 13*x^4 + 14*x^3 + 17*x + 3

noncomputable def polynomial_2 (x : ℝ) : ℝ :=
  2*x^6 + 4*x^4 + 5*x^2 + 2

theorem polynomial_q_correct (x : ℝ) : 
  polynomial_q x = polynomial_rhs x - polynomial_2 x := 
by
  sorry

end polynomial_q_correct_1_1445


namespace f_sum_zero_1_1198

noncomputable def f : ℝ → ℝ := sorry

axiom f_property_1 : ∀ x : ℝ, f (x ^ 3) = (f x) ^ 3
axiom f_property_2 : ∀ x1 x2 : ℝ, x1 ≠ x2 → f x1 ≠ f x2

theorem f_sum_zero : f 0 + f (-1) + f 1 = 0 := by
  sorry

end f_sum_zero_1_1198


namespace sum_of_first_nine_terms_1_1746

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ (d : ℝ), ∀ (n : ℕ), a n = a 1 + d * (n - 1)

variables (a : ℕ → ℝ) (h_seq : arithmetic_sequence a)

-- Given condition: a₂ + a₃ + a₇ + a₈ = 20
def condition : Prop := a 2 + a 3 + a 7 + a 8 = 20

-- Statement: Prove that the sum of the first 9 terms is 45
theorem sum_of_first_nine_terms (h : condition a) : 
  (a 1 + a 2 + a 3 + a 4 + a 5 + a 6 + a 7 + a 8 + a 9) = 45 :=
by sorry

end sum_of_first_nine_terms_1_1746


namespace winning_percentage_is_70_1_1494

def percentage_of_votes (P : ℝ) : Prop :=
  ∃ (P : ℝ), (7 * P - 7 * (100 - P) = 280 ∧ 0 ≤ P ∧ P ≤ 100)

theorem winning_percentage_is_70 :
  percentage_of_votes 70 :=
by
  sorry

end winning_percentage_is_70_1_1494


namespace symmetric_points_x_axis_1_1912

theorem symmetric_points_x_axis (a b : ℤ) 
  (h1 : a - 1 = 2) (h2 : 5 = -(b - 1)) : (a + b) ^ 2023 = -1 := 
by
  -- The proof steps will go here.
  sorry

end symmetric_points_x_axis_1_1912


namespace final_silver_tokens_1_1439

structure TokenCounts :=
  (red : ℕ)
  (blue : ℕ)

def initial_tokens : TokenCounts := { red := 100, blue := 50 }

def exchange_booth1 (tokens : TokenCounts) : TokenCounts :=
  { red := tokens.red - 3, blue := tokens.blue + 2 }

def exchange_booth2 (tokens : TokenCounts) : TokenCounts :=
  { red := tokens.red + 1, blue := tokens.blue - 3 }

noncomputable def max_exchanges (initial : TokenCounts) : ℕ × ℕ :=
  let x := 48
  let y := 47
  (x, y)

noncomputable def silver_tokens (x y : ℕ) : ℕ := x + y

theorem final_silver_tokens (x y : ℕ) (tokens : TokenCounts) 
  (hx : tokens.red = initial_tokens.red - 3 * x + y)
  (hy : tokens.blue = initial_tokens.blue + 2 * x - 3 * y) 
  (hx_le : tokens.red >= 3 → false)
  (hy_le : tokens.blue >= 3 → false) : 
  silver_tokens x y = 95 :=
by {
  sorry
}

end final_silver_tokens_1_1439


namespace math_proof_1_1479

variable {a b c A B C : ℝ}
variable {S : ℝ}

noncomputable def problem_statement (h1 : b + c = 2 * a * Real.cos B)
    (h2 : S = a^2 / 4) : Prop :=
    (∃ A B : ℝ, (A = 2 * B) ∧ (A = 90)) 

theorem math_proof (h1 : b + c = 2 * a * Real.cos B)
    (h2 : S = a^2 / 4) :
    problem_statement h1 h2 :=
    sorry

end math_proof_1_1479


namespace line_intersects_ellipse_all_possible_slopes_1_1958

theorem line_intersects_ellipse_all_possible_slopes (m : ℝ) :
  m^2 ≥ 1 / 5 ↔ ∃ x y : ℝ, (y = m * x - 3) ∧ (4 * x^2 + 25 * y^2 = 100) := sorry

end line_intersects_ellipse_all_possible_slopes_1_1958


namespace machine_A_sprockets_per_hour_1_1301

theorem machine_A_sprockets_per_hour :
  ∀ (A T : ℝ),
    (T > 0 ∧
    (∀ P Q, P = 1.1 * A ∧ Q = 330 / P ∧ Q = 330 / A + 10) →
      A = 3) := 
by
  intro A T
  intro h
  sorry

end machine_A_sprockets_per_hour_1_1301


namespace union_A_B_1_1104

def set_A : Set ℝ := { x | 1 / x ≤ 0 }
def set_B : Set ℝ := { x | x^2 - 1 < 0 }

theorem union_A_B : set_A ∪ set_B = { x | x < 1 } :=
by
  sorry

end union_A_B_1_1104


namespace sam_catches_alice_in_40_minutes_1_1255

def sam_speed := 7 -- mph
def alice_speed := 4 -- mph
def initial_distance := 2 -- miles

theorem sam_catches_alice_in_40_minutes : 
  (initial_distance / (sam_speed - alice_speed)) * 60 = 40 :=
by sorry

end sam_catches_alice_in_40_minutes_1_1255


namespace range_of_u_1_1495

variable (a b u : ℝ)

theorem range_of_u (ha : a > 0) (hb : b > 0) (hab : a + b = 1) : 
  (∀ x : ℝ, x > 0 → a^2 + b^2 ≥ x ↔ x ≤ 16) :=
sorry

end range_of_u_1_1495


namespace find_a_in_triangle_1_1820

theorem find_a_in_triangle
  (A B C : ℝ)
  (a b c : ℝ)
  (h1 : c = 3)
  (h2 : C = Real.pi / 3)
  (h3 : Real.sin B = 2 * Real.sin A)
  (h4 : a = 3) :
  a = Real.sqrt 3 := by
  sorry

end find_a_in_triangle_1_1820


namespace geometric_sequence_sum_inverse_equals_1_1270

variable (a : ℕ → ℝ)
variable (n : ℕ)

noncomputable def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃(r : ℝ), ∀ (n : ℕ), a (n + 1) = r * a n

theorem geometric_sequence_sum_inverse_equals (a : ℕ → ℝ)
  (h_geo : is_geometric_sequence a)
  (h_sum : a 5 + a 6 + a 7 + a 8 = 15 / 8)
  (h_prod : a 6 * a 7 = -9 / 8) :
  (1 / a 5) + (1 / a 6) + (1 / a 7) + (1 / a 8) = -5 / 3 :=
by
  sorry

end geometric_sequence_sum_inverse_equals_1_1270


namespace total_oranges_picked_1_1508

-- Defining the number of oranges picked by Mary, Jason, and Sarah
def maryOranges := 122
def jasonOranges := 105
def sarahOranges := 137

-- The theorem to prove that the total number of oranges picked is 364
theorem total_oranges_picked : maryOranges + jasonOranges + sarahOranges = 364 := by
  sorry

end total_oranges_picked_1_1508


namespace triangle_inequality_1_1024

theorem triangle_inequality
  (a b c : ℝ)
  (habc : ¬(a + b ≤ c ∨ a + c ≤ b ∨ b + c ≤ a)) :
  a * (b - c)^2 + b * (c - a)^2 + c * (a - b)^2 + 4 * a * b * c > a^3 + b^3 + c^3 := 
by {
  sorry
}

end triangle_inequality_1_1024


namespace maximum_value_of_m_1_1706

theorem maximum_value_of_m (x y : ℝ) (hx : x > 1 / 2) (hy : y > 1) : 
    (4 * x^2 / (y - 1) + y^2 / (2 * x - 1)) ≥ 8 := 
sorry

end maximum_value_of_m_1_1706


namespace danny_distance_to_work_1_1995

-- Define the conditions and the problem in terms of Lean definitions
def distance_to_first_friend : ℕ := 8
def distance_to_second_friend : ℕ := distance_to_first_friend / 2
def total_distance_driven_so_far : ℕ := distance_to_first_friend + distance_to_second_friend
def distance_to_work : ℕ := 3 * total_distance_driven_so_far

-- Lean statement to be proven
theorem danny_distance_to_work :
  distance_to_work = 36 :=
by
  -- This is the proof placeholder
  sorry

end danny_distance_to_work_1_1995


namespace scientific_notation_example_1_1164

def scientific_notation (n : ℝ) (a : ℝ) (b : ℤ) : Prop :=
  n = a * 10^b

theorem scientific_notation_example : 
  scientific_notation 0.00519 5.19 (-3) :=
by 
  sorry

end scientific_notation_example_1_1164


namespace find_beta_1_1511

variable (α β : ℝ)

theorem find_beta 
  (h1 : Real.cos α = 1 / 7)
  (h2 : Real.cos (α + β) = -11 / 14)
  (h3 : 0 < α ∧ α < Real.pi / 2)
  (h4 : Real.pi / 2 < α + β ∧ α + β < Real.pi) : β = Real.pi / 3 := sorry

end find_beta_1_1511


namespace cost_price_of_watch_1_1497

variable (CP : ℝ)
variable (SP_loss SP_gain : ℝ)
variable (h1 : SP_loss = CP * 0.725)
variable (h2 : SP_gain = CP * 1.125)
variable (h3 : SP_gain - SP_loss = 275)

theorem cost_price_of_watch : CP = 687.50 :=
by
  sorry

end cost_price_of_watch_1_1497


namespace correct_operation_1_1641

theorem correct_operation (a : ℝ) : a^5 / a^2 = a^3 := by
  -- Proof steps will be supplied here
  sorry

end correct_operation_1_1641


namespace complement_of_P_in_U_1_1975

/-- Definitions of sets U and P -/
def U := { y : ℝ | ∃ x : ℝ, x > 1 ∧ y = Real.log x / Real.log 2 }
def P := { y : ℝ | ∃ x : ℝ, x > 2 ∧ y = 1 / x }

/-- The complement of P in U -/
def complement_U_P := { y : ℝ | y = 0 ∨ y ≥ 1 / 2 }

/-- Proving the complement of P in U is as expected -/
theorem complement_of_P_in_U : { y : ℝ | y ∈ U ∧ y ∉ P } = complement_U_P := by
  sorry

end complement_of_P_in_U_1_1975


namespace cube_distance_1_1190

-- The Lean 4 statement
theorem cube_distance (side_length : ℝ) (h1 h2 h3 : ℝ) (r s t : ℕ) 
  (h1_eq : h1 = 18) (h2_eq : h2 = 20) (h3_eq : h3 = 22) (side_length_eq : side_length = 15) :
  r = 57 ∧ s = 597 ∧ t = 3 ∧ r + s + t = 657 :=
by
  sorry

end cube_distance_1_1190


namespace cupcakes_difference_1_1541

theorem cupcakes_difference (h : ℕ) (betty_rate : ℕ) (dora_rate : ℕ) (betty_break : ℕ) 
  (cupcakes_difference : ℕ) 
  (H₁ : betty_rate = 10) 
  (H₂ : dora_rate = 8) 
  (H₃ : betty_break = 2) 
  (H₄ : cupcakes_difference = 10) : 
  8 * h - 10 * (h - 2) = 10 → h = 5 :=
by
  intro H
  sorry

end cupcakes_difference_1_1541


namespace sec_120_eq_neg_2_1_1162

noncomputable def sec (θ : ℝ) : ℝ := 1 / Real.cos θ

theorem sec_120_eq_neg_2 : sec 120 = -2 := by
  sorry

end sec_120_eq_neg_2_1_1162


namespace solve_system_eqs_1_1028

theorem solve_system_eqs : 
    ∃ (x y z : ℚ), 
    4 * x - 3 * y + z = -10 ∧
    3 * x + 5 * y - 2 * z = 8 ∧
    x - 2 * y + 7 * z = 5 ∧
    x = -51 / 61 ∧ 
    y = 378 / 61 ∧ 
    z = 728 / 61 := by
  sorry

end solve_system_eqs_1_1028


namespace learn_at_least_537_words_1_1962

theorem learn_at_least_537_words (total_words : ℕ) (guess_percentage : ℝ) (required_percentage : ℝ) :
  total_words = 600 → guess_percentage = 0.05 → required_percentage = 0.90 → 
  ∀ (words_learned : ℕ), words_learned ≥ 537 → 
  (words_learned + guess_percentage * (total_words - words_learned)) / total_words ≥ required_percentage :=
by
  intros h_total_words h_guess_percentage h_required_percentage words_learned h_words_learned
  sorry

end learn_at_least_537_words_1_1962


namespace largest_square_perimeter_1_1765

-- Define the conditions
def rectangle_length : ℕ := 80
def rectangle_width : ℕ := 60

-- Define the theorem to prove
theorem largest_square_perimeter : 4 * rectangle_width = 240 := by
  -- The proof steps are omitted
  sorry

end largest_square_perimeter_1_1765


namespace problem_statement_1_1172

noncomputable def x : ℝ := (3 + Real.sqrt 8) ^ 30
noncomputable def n : ℤ := Int.floor x
noncomputable def f : ℝ := x - n

theorem problem_statement : x * (1 - f) = 1 := 
by 
  sorry

end problem_statement_1_1172


namespace grill_burns_fifteen_coals_in_twenty_minutes_1_1851

-- Define the problem conditions
def total_coals (bags : ℕ) (coals_per_bag : ℕ) : ℕ :=
  bags * coals_per_bag

def burning_ratio (total_coals : ℕ) (total_minutes : ℕ) : ℕ :=
  total_minutes / total_coals

-- Given conditions
def bags := 3
def coals_per_bag := 60
def total_minutes := 240
def fifteen_coals := 15

-- Problem statement
theorem grill_burns_fifteen_coals_in_twenty_minutes :
  total_minutes / total_coals bags coals_per_bag * fifteen_coals = 20 :=
by
  sorry

end grill_burns_fifteen_coals_in_twenty_minutes_1_1851


namespace positive_diff_after_add_five_1_1194

theorem positive_diff_after_add_five (y : ℝ) 
  (h : (45 + y)/2 = 32) : |45 - (y + 5)| = 21 :=
by 
  sorry

end positive_diff_after_add_five_1_1194


namespace sum_of_squares_equality_1_1744

theorem sum_of_squares_equality (n : ℕ) (h : n = 5) :
  (∑ i in Finset.range (n + 1), i^2) = (∑ i in Finset.range (2 * n + 1), i) := by
  sorry

end sum_of_squares_equality_1_1744


namespace unique_root_condition_1_1683

theorem unique_root_condition (a : ℝ) : 
  (∀ x : ℝ, x^3 + a*x^2 - 4*a*x + a^2 - 4 = 0 → ∃! x₀ : ℝ, x = x₀) ↔ a < 1 :=
by sorry

end unique_root_condition_1_1683


namespace number_of_lines_1_1051

-- Define point P
structure Point where
  x : ℝ
  y : ℝ

-- Define a line by its equation ax + by + c = 0
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ
  nonzero: a ≠ 0 ∨ b ≠ 0

-- Definition of a line passing through a point P
def passes_through (l : Line) (P : Point) : Prop :=
  l.a * P.x + l.b * P.y + l.c = 0

-- Definition of a line having equal intercepts on x-axis and y-axis
def equal_intercepts (l : Line) : Prop :=
  l.a ≠ 0 ∧ l.b ≠ 0 ∧ l.a = l.b

-- Definition of a specific point P
def P : Point := { x := 1, y := 2 }

-- The theorem statement
theorem number_of_lines : ∃ (lines : Finset Line), (∀ l ∈ lines, passes_through l P ∧ equal_intercepts l) ∧ lines.card = 2 := by
  sorry

end number_of_lines_1_1051


namespace donut_selection_count_1_1629

def num_donut_selections : ℕ :=
  Nat.choose 9 3

theorem donut_selection_count : num_donut_selections = 84 := 
by
  sorry

end donut_selection_count_1_1629


namespace sin_B_value_triangle_area_1_1242

-- Problem 1: sine value of angle B given the conditions
theorem sin_B_value (a b c : ℝ) (A B C : ℝ)
  (h1 : 3 * b = 4 * c)
  (h2 : B = 2 * C) :
  Real.sin B = (4 * Real.sqrt 5) / 9 :=
sorry

-- Problem 2: Area of triangle ABC given the conditions and b = 4
theorem triangle_area (a b c : ℝ) (A B C : ℝ)
  (h1 : 3 * b = 4 * c)
  (h2 : B = 2 * C)
  (h3 : b = 4) :
  (1 / 2) * b * c * Real.sin A = (14 * Real.sqrt 5) / 9 :=
sorry

end sin_B_value_triangle_area_1_1242


namespace subsets_neither_A_nor_B_1_1799

def U : Finset ℕ := {1, 2, 3, 4, 5, 6, 7, 8}
def A : Finset ℕ := {1, 2, 3, 4, 5}
def B : Finset ℕ := {4, 5, 6, 7, 8}

theorem subsets_neither_A_nor_B : 
  (U.powerset.card - A.powerset.card - B.powerset.card + (A ∩ B).powerset.card) = 196 := by 
  sorry

end subsets_neither_A_nor_B_1_1799


namespace remainder_of_max_6_multiple_no_repeated_digits_1_1983

theorem remainder_of_max_6_multiple_no_repeated_digits (M : ℕ) 
  (hM : ∃ n, M = 6 * n) 
  (h_unique_digits : ∀ (d : ℕ), d ∈ (M.digits 10) → (M.digits 10).count d = 1) 
  (h_max_M : ∀ (k : ℕ), (∃ n, k = 6 * n) ∧ (∀ (d : ℕ), d ∈ (k.digits 10) → (k.digits 10).count d = 1) → k ≤ M) :
  M % 100 = 78 := 
sorry

end remainder_of_max_6_multiple_no_repeated_digits_1_1983


namespace min_value_expression_1_1348

variable (a b c : ℝ)
variable (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
variable (h_eq : a * b * c = 64)

theorem min_value_expression :
  a^2 + 6 * a * b + 9 * b^2 + 3 * c^2 ≥ 192 :=
by {
  sorry
}

end min_value_expression_1_1348


namespace rational_b_if_rational_a_1_1153

theorem rational_b_if_rational_a (x : ℚ) (h_rational : ∃ a : ℚ, a = x / (x^2 - x + 1)) :
  ∃ b : ℚ, b = x^2 / (x^4 - x^2 + 1) :=
by
  sorry

end rational_b_if_rational_a_1_1153


namespace total_time_to_complete_work_1_1406

noncomputable def mahesh_work_rate (W : ℕ) := W / 40
noncomputable def mahesh_work_done_in_20_days (W : ℕ) := 20 * (mahesh_work_rate W)
noncomputable def remaining_work (W : ℕ) := W - (mahesh_work_done_in_20_days W)
noncomputable def rajesh_work_rate (W : ℕ) := (remaining_work W) / 30

theorem total_time_to_complete_work (W : ℕ) :
    (mahesh_work_rate W) + (rajesh_work_rate W) = W / 24 →
    (mahesh_work_done_in_20_days W) = W / 2 →
    (remaining_work W) = W / 2 →
    (rajesh_work_rate W) = W / 60 →
    20 + 30 = 50 :=
by 
  intros _ _ _ _
  sorry

end total_time_to_complete_work_1_1406


namespace stations_visited_1_1275

-- Define the total number of nails
def total_nails : ℕ := 560

-- Define the number of nails left at each station
def nails_per_station : ℕ := 14

-- Main theorem statement
theorem stations_visited : total_nails / nails_per_station = 40 := by
  sorry

end stations_visited_1_1275


namespace problem1_problem2_1_1691

-- Problem (1) proof statement
theorem problem1 (a : ℝ) (h : a ≠ 0) : 
  3 * a^2 * a^3 + a^7 / a^2 = 4 * a^5 :=
by
  sorry

-- Problem (2) proof statement
theorem problem2 (x : ℝ) : 
  (x - 1)^2 - x * (x + 1) + (-2023)^0 = -3 * x + 2 :=
by
  sorry

end problem1_problem2_1_1691


namespace compute_expression_1_1528

theorem compute_expression : 1013^2 - 987^2 - 1007^2 + 993^2 = 24000 := by
  sorry

end compute_expression_1_1528


namespace find_v2002_1_1944

def g (x : ℕ) : ℕ :=
  match x with
  | 1 => 5
  | 2 => 3
  | 3 => 6
  | 4 => 2
  | 5 => 1
  | 6 => 7
  | 7 => 4
  | _ => 0

def seq_v : ℕ → ℕ
| 0       => 5
| (n + 1) => g (seq_v n)

theorem find_v2002 : seq_v 2002 = 5 :=
  sorry

end find_v2002_1_1944


namespace max_price_of_product_1_1061

theorem max_price_of_product (x : ℝ) 
  (cond1 : (x - 10) * 0.1 = (x - 20) * 0.2) : 
  x = 30 := 
by 
  sorry

end max_price_of_product_1_1061


namespace center_of_gravity_shift_center_of_gravity_shift_result_1_1171

variable (l s : ℝ) (s_val : s = 60)
#check (s_val : s = 60)

theorem center_of_gravity_shift : abs ((l / 2) - ((l - s) / 2)) = s / 2 := 
by sorry

theorem center_of_gravity_shift_result : (s / 2 = 30) :=
by sorry

end center_of_gravity_shift_center_of_gravity_shift_result_1_1171


namespace average_price_of_cow_1_1419

theorem average_price_of_cow (total_price_cows_and_goats rs: ℕ) (num_cows num_goats: ℕ)
    (avg_price_goat: ℕ) (total_price: total_price_cows_and_goats = 1400)
    (num_cows_eq: num_cows = 2) (num_goats_eq: num_goats = 8)
    (avg_price_goat_eq: avg_price_goat = 60) :
    let total_price_goats := avg_price_goat * num_goats
    let total_price_cows := total_price_cows_and_goats - total_price_goats
    let avg_price_cow := total_price_cows / num_cows
    avg_price_cow = 460 :=
by
  sorry

end average_price_of_cow_1_1419


namespace perpendicular_bisector_eq_1_1373

theorem perpendicular_bisector_eq (A B: (ℝ × ℝ)) (hA: A = (1, 3)) (hB: B = (-5, 1)) :
  ∃ m c, (m = -3) ∧ (c = 4) ∧ (∀ x y, y = m * x + c ↔ 3 * x + y + 4 = 0) := 
by
  sorry

end perpendicular_bisector_eq_1_1373


namespace binomial_12_6_eq_924_1_1585

theorem binomial_12_6_eq_924 : Nat.choose 12 6 = 924 := 
by
  sorry

end binomial_12_6_eq_924_1_1585


namespace min_value_frac_1_1559

variable (a b : ℝ) (h₀ : a > 0) (h₁ : b > 0) (h₂ : a + b = 1)

theorem min_value_frac : (1 / a + 4 / b) = 9 :=
by sorry

end min_value_frac_1_1559


namespace trapezoid_bisector_segment_length_1_1148

-- Definitions of the conditions
variables {a b c d t : ℝ}

noncomputable def semiperimeter (a b c d : ℝ) : ℝ := (a + b + c + d) / 2

-- The theorem statement
theorem trapezoid_bisector_segment_length
  (p : ℝ)
  (h_p : p = semiperimeter a b c d) :
  t^2 = (4 * b * d) / (b + d)^2 * (p - a) * (p - c) :=
sorry

end trapezoid_bisector_segment_length_1_1148


namespace geometric_arithmetic_seq_unique_ratio_1_1363

variable (d : ℚ) (q : ℚ) (k : ℤ)
variable (h_d_nonzero : d ≠ 0)
variable (h_q_pos : 0 < q) (h_q_lt_one : q < 1)
variable (h_integer : 14 / (1 + q + q^2) = k)

theorem geometric_arithmetic_seq_unique_ratio :
  q = 1 / 2 :=
by
  sorry

end geometric_arithmetic_seq_unique_ratio_1_1363


namespace cubes_with_odd_red_faces_1_1341

-- Define the dimensions and conditions of the block
def block_length : ℕ := 6
def block_width: ℕ := 6
def block_height : ℕ := 2

-- The block is painted initially red on all sides
-- Then the bottom face is painted blue
-- The block is cut into 1-inch cubes
-- 

noncomputable def num_cubes_with_odd_red_faces (length width height : ℕ) : ℕ :=
  -- Only edge cubes have odd number of red faces in this configuration
  let corner_count := 8  -- 4 on top + 4 on bottom (each has 4 red faces)
  let edge_count := 40   -- 20 on top + 20 on bottom (each has 3 red faces)
  let face_only_count := 32 -- 16 on top + 16 on bottom (each has 2 red faces)
  -- The resulting total number of cubes with odd red faces
  edge_count

-- The theorem we need to prove
theorem cubes_with_odd_red_faces : num_cubes_with_odd_red_faces block_length block_width block_height = 40 :=
  by 
    -- Proof goes here
    sorry

end cubes_with_odd_red_faces_1_1341


namespace find_value_added_1_1571

theorem find_value_added :
  ∀ (n x : ℤ), (2 * n + x = 8 * n - 4) → (n = 4) → (x = 20) :=
by
  intros n x h1 h2
  sorry

end find_value_added_1_1571


namespace negation_correct_1_1207

-- Define the original statement as a predicate
def original_statement (x : ℝ) : Prop := x > 1 → x^2 ≤ x

-- Define the negation of the original statement as a predicate
def negated_statement : Prop := ∃ x : ℝ, x > 1 ∧ x^2 > x

-- Define the theorem that the negation of the original statement implies the negated statement
theorem negation_correct :
  ¬ (∀ x : ℝ, original_statement x) ↔ negated_statement := by
  sorry

end negation_correct_1_1207


namespace prove_3a_3b_3c_1_1077

variable (a b c : ℝ)

def condition1 := b + c = 15 - 2 * a
def condition2 := a + c = -18 - 3 * b
def condition3 := a + b = 8 - 4 * c
def condition4 := a - b + c = 3

theorem prove_3a_3b_3c (h1 : condition1 a b c) (h2 : condition2 a b c) (h3 : condition3 a b c) (h4 : condition4 a b c) :
  3 * a + 3 * b + 3 * c = 24 / 5 :=
sorry

end prove_3a_3b_3c_1_1077


namespace solution_system_of_equations_1_1413

theorem solution_system_of_equations : 
  ∃ (x y : ℝ), (2 * x - y = 3 ∧ x + y = 3) ∧ (x = 2 ∧ y = 1) := 
by
  sorry

end solution_system_of_equations_1_1413


namespace calculate_expression_1_1177

theorem calculate_expression :
  (6 * 5 * 4 * 3 * 2 * 1 - 5 * 4 * 3 * 2 * 1) / (4 * 3 * 2 * 1) = 25 := 
by sorry

end calculate_expression_1_1177


namespace future_value_option_B_correct_1_1668

noncomputable def future_value_option_B (p q : ℝ) : ℝ :=
  150 * (1 + p / 100 / 2) ^ 6 * (1 + q / 100 / 4) ^ 12

theorem future_value_option_B_correct (p q A₂ : ℝ) :
  A₂ = 150 * (1 + p / 100 / 2) ^ 6 * (1 + q / 100 / 4) ^ 12 →
  ∃ A₂, A₂ = future_value_option_B p q :=
by
  intro h
  use A₂
  exact h

end future_value_option_B_correct_1_1668


namespace positive_integer_solutions_inequality_1_1054

theorem positive_integer_solutions_inequality :
  {x : ℕ | 2 * x + 9 ≥ 3 * (x + 2)} = {1, 2, 3} :=
by
  sorry

end positive_integer_solutions_inequality_1_1054


namespace right_triangle_condition_1_1600

theorem right_triangle_condition (a b c : ℝ) (h : c^2 - a^2 = b^2) : 
  ∃ (A B C : ℝ), A + B + C = 180 ∧ A = 90 ∧ B + C = 90 :=
by sorry

end right_triangle_condition_1_1600


namespace lizzy_final_amount_1_1777

-- Define constants
def m : ℕ := 80   -- cents from mother
def f : ℕ := 40   -- cents from father
def s : ℕ := 50   -- cents spent on candy
def u : ℕ := 70   -- cents from uncle
def t : ℕ := 90   -- cents for the toy
def c : ℕ := 110  -- cents change she received

-- Define the final amount calculation
def final_amount : ℕ := m + f - s + u - t + c

-- Prove the final amount is 160
theorem lizzy_final_amount : final_amount = 160 := by
  sorry

end lizzy_final_amount_1_1777


namespace cost_to_fill_pool_1_1594

-- Definitions based on the conditions
def cubic_foot_to_liters: ℕ := 25
def pool_depth: ℕ := 10
def pool_width: ℕ := 6
def pool_length: ℕ := 20
def cost_per_liter: ℕ := 3

-- Statement to be proved
theorem cost_to_fill_pool : 
  (pool_depth * pool_width * pool_length * cubic_foot_to_liters * cost_per_liter) = 90000 := 
by 
  sorry

end cost_to_fill_pool_1_1594


namespace chelsea_sugar_bags_1_1401

variable (n : ℕ)

-- Defining the conditions as hypotheses
def initial_sugar : ℕ := 24
def remaining_sugar : ℕ := 21
def sugar_lost : ℕ := initial_sugar - remaining_sugar
def torn_bag_sugar : ℕ := 2 * sugar_lost

-- Define the statement to prove
theorem chelsea_sugar_bags :
  n = initial_sugar / torn_bag_sugar → n = 4 :=
by
  sorry

end chelsea_sugar_bags_1_1401


namespace john_needs_packs_1_1414

-- Definitions based on conditions
def utensils_per_pack : Nat := 30
def utensils_types : Nat := 3
def spoons_per_pack : Nat := utensils_per_pack / utensils_types
def spoons_needed : Nat := 50

-- Statement to prove
theorem john_needs_packs : (50 / spoons_per_pack) = 5 :=
by
  -- To complete the proof
  sorry

end john_needs_packs_1_1414


namespace range_cos_2alpha_cos_2beta_1_1404

variable (α β : ℝ)
variable (h : Real.sin α + Real.cos β = 3 / 2)

theorem range_cos_2alpha_cos_2beta :
  -3/2 ≤ Real.cos (2 * α) + Real.cos (2 * β) ∧ Real.cos (2 * α) + Real.cos (2 * β) ≤ 3/2 :=
sorry

end range_cos_2alpha_cos_2beta_1_1404


namespace nine_otimes_three_1_1513

def otimes (a b : ℤ) : ℤ := a + (4 * a) / (3 * b)

theorem nine_otimes_three : otimes 9 3 = 13 := by
  sorry

end nine_otimes_three_1_1513


namespace point_B_in_first_quadrant_1_1043

theorem point_B_in_first_quadrant 
  (a b : ℝ) 
  (h1 : a > 0) 
  (h2 : -b > 0) : 
  (a > 0) ∧ (b > 0) := 
by 
  sorry

end point_B_in_first_quadrant_1_1043


namespace prime_has_property_p_1_1009

theorem prime_has_property_p (n : ℕ) (hn : Prime n) (a : ℕ) (h : n ∣ a^n - 1) : n^2 ∣ a^n - 1 := by
  sorry

end prime_has_property_p_1_1009


namespace smallest_m_plus_n_1_1973

theorem smallest_m_plus_n (m n : ℕ) (h1 : m > n) (h2 : n ≥ 1) 
(h3 : 1000 ∣ 1978^m - 1978^n) : m + n = 106 :=
sorry

end smallest_m_plus_n_1_1973


namespace average_value_f_1_1146

def f (x : ℝ) : ℝ := (1 + x)^3

theorem average_value_f : (1 / (4 - 2)) * (∫ x in (2:ℝ)..(4:ℝ), f x) = 68 :=
by
  sorry

end average_value_f_1_1146


namespace rice_pounds_1_1574

noncomputable def pounds_of_rice (r p : ℝ) : Prop :=
  r + p = 30 ∧ 1.10 * r + 0.55 * p = 23.50

theorem rice_pounds (r p : ℝ) (h : pounds_of_rice r p) : r = 12.7 :=
sorry

end rice_pounds_1_1574


namespace train_length_1_1388

theorem train_length (speed_kmh : ℕ) (time_s : ℕ) (length_m : ℚ) : 
  speed_kmh = 120 → 
  time_s = 25 → 
  length_m = 833.25 → 
  (speed_kmh * 1000 / 3600) * time_s = length_m :=
by
  intros
  sorry

end train_length_1_1388


namespace tulips_sum_1_1150

def tulips_total (arwen_tulips : ℕ) (elrond_tulips : ℕ) : ℕ := arwen_tulips + elrond_tulips

theorem tulips_sum : tulips_total 20 (2 * 20) = 60 := by
  sorry

end tulips_sum_1_1150


namespace max_candies_theorem_1_1644

-- Defining constants: the number of students and the total number of candies.
def n : ℕ := 40
def T : ℕ := 200

-- Defining the condition that each student takes at least 2 candies.
def min_candies_per_student : ℕ := 2

-- Calculating the minimum total number of candies taken by 39 students.
def min_total_for_39_students := min_candies_per_student * (n - 1)

-- The maximum number of candies one student can take.
def max_candies_one_student_can_take := T - min_total_for_39_students

-- The statement to prove.
theorem max_candies_theorem : max_candies_one_student_can_take = 122 :=
by
  sorry

end max_candies_theorem_1_1644


namespace probability_problems_1_1469

theorem probability_problems (x : ℕ) :
  (0 = (if 8 + 12 > 8 then 0 else 1)) ∧
  (1 = (1 - 0)) ∧
  (3 / 5 = 12 / 20) ∧
  (4 / 5 = (8 + x) / 20 → x = 8) := by sorry

end probability_problems_1_1469


namespace function_value_1_1486

noncomputable def log_base (a x : ℝ) : ℝ := Real.log x / Real.log a

theorem function_value (a b : ℝ) (h₀ : a > 0) (h₁ : a ≠ 1) (h₂ : log_base a (2 + b) = 1) (h₃ : log_base a (8 + b) = 2) : a + b = 4 :=
by
  sorry

end function_value_1_1486


namespace sum_of_squares_of_coefficients_1_1476

theorem sum_of_squares_of_coefficients :
  let poly := 5 * (X^6 + 4 * X^4 + 2 * X^2 + 1)
  let coeffs := [5, 20, 10, 5]
  (coeffs.map (λ c => c * c)).sum = 550 := 
by
  sorry

end sum_of_squares_of_coefficients_1_1476


namespace eccentricity_of_ellipse_1_1771

def ellipse_equation (x y : ℝ) (m : ℝ) : Prop := x^2 / 4 + y^2 / m = 1
def sum_of_distances_to_foci (x y : ℝ) (m : ℝ) : Prop := 
  ∃ fx₁ fy₁ fx₂ fy₂ : ℝ, (ellipse_equation x y m) ∧ 
  (dist (x, y) (fx₁, fy₁) + dist (x, y) (fx₂, fy₂) = m - 3)

theorem eccentricity_of_ellipse (m : ℝ) (h₁ : 4 < m) 
  (h₂ : sum_of_distances_to_foci x y m) : 
  ∃ e : ℝ, e = √5 / 3 :=
sorry

end eccentricity_of_ellipse_1_1771


namespace total_profit_correct_1_1173

noncomputable def total_profit (a b c : ℕ) (c_share : ℕ) : ℕ :=
  let ratio := a + b + c
  let part_value := c_share / c
  ratio * part_value

theorem total_profit_correct (h_a : ℕ := 5000) (h_b : ℕ := 8000) (h_c : ℕ := 9000) (h_c_share : ℕ := 36000) :
  total_profit h_a h_b h_c h_c_share = 88000 :=
by
  sorry

end total_profit_correct_1_1173


namespace polynomial_simplified_1_1515

def polynomial (x : ℝ) : ℝ := 4 - 6 * x - 8 * x^2 + 12 - 14 * x + 16 * x^2 - 18 + 20 * x + 24 * x^2

theorem polynomial_simplified (x : ℝ) : polynomial x = 32 * x^2 - 2 :=
by
  sorry

end polynomial_simplified_1_1515


namespace part1_part2_1_1957

section
  variable {x a : ℝ}

  def f (x a : ℝ) := |x - a| + 3 * x

  theorem part1 (h : a = 1) : 
    (∀ x, f x a ≥ 3 * x + 2 ↔ (x ≥ 3 ∨ x ≤ -1)) :=
    sorry

  theorem part2 : 
    (∀ x, (f x a) ≤ 0 ↔ (x ≤ -1)) → a = 2 :=
    sorry
end

end part1_part2_1_1957


namespace a6_value_1_1291

variable (a_n : ℕ → ℤ)

/-- Given conditions in the arithmetic sequence -/
def arithmetic_sequence_property (a_n : ℕ → ℤ) :=
  ∀ n, a_n n = a_n 0 + n * (a_n 1 - a_n 0)

/-- Given sum condition a_4 + a_5 + a_6 + a_7 + a_8 = 150 -/
def sum_condition :=
  a_n 4 + a_n 5 + a_n 6 + a_n 7 + a_n 8 = 150

theorem a6_value (h : arithmetic_sequence_property a_n) (hsum : sum_condition a_n) :
  a_n 6 = 30 := 
by
  sorry

end a6_value_1_1291


namespace solve_for_k_1_1042

def f (n : ℤ) : ℤ :=
  if n % 2 = 1 then n + 5 else n / 2

theorem solve_for_k (k : ℤ) (h1 : k % 2 = 1) (h2 : f (f (f k)) = 57) : k = 223 :=
by
  -- Proof will be provided here
  sorry

end solve_for_k_1_1042


namespace probability_red_or_white_is_11_over_13_1_1858

-- Given data
def total_marbles : ℕ := 60
def blue_marbles : ℕ := 5
def red_marbles : ℕ := 9
def white_marbles : ℕ := total_marbles - blue_marbles - red_marbles

def blue_size : ℕ := 2
def red_size : ℕ := 1
def white_size : ℕ := 1

-- Total size value of all marbles
def total_size_value : ℕ := (blue_size * blue_marbles) + (red_size * red_marbles) + (white_size * white_marbles)

-- Probability of selecting a red or white marble
def probability_red_or_white : ℚ := (red_size * red_marbles + white_size * white_marbles) / total_size_value

-- Theorem to prove
theorem probability_red_or_white_is_11_over_13 : probability_red_or_white = 11 / 13 :=
by sorry

end probability_red_or_white_is_11_over_13_1_1858


namespace investment_ratio_same_period_1_1246

-- Define the profits of A and B
def profit_A : ℕ := 60000
def profit_B : ℕ := 6000

-- Define their investment ratio given the same time period
theorem investment_ratio_same_period : profit_A / profit_B = 10 :=
by
  -- Proof skipped 
  sorry

end investment_ratio_same_period_1_1246


namespace evaluate_expression_1_1615

theorem evaluate_expression : 101^3 + 3 * 101^2 * 2 + 3 * 101 * 2^2 + 2^3 = 1092727 := by
  sorry

end evaluate_expression_1_1615


namespace ratio_of_men_to_women_1_1570

def num_cannoneers : ℕ := 63
def num_people : ℕ := 378
def num_women (C : ℕ) : ℕ := 2 * C
def num_men (total : ℕ) (women : ℕ) : ℕ := total - women

theorem ratio_of_men_to_women : 
  let C := num_cannoneers
  let total := num_people
  let W := num_women C
  let M := num_men total W
  M / W = 2 :=
by
  sorry

end ratio_of_men_to_women_1_1570


namespace find_common_difference_1_1632

-- Definitions for arithmetic sequences and sums
def S (a1 d : ℕ) (n : ℕ) := (n * (2 * a1 + (n - 1) * d)) / 2
def a (a1 d : ℕ) (n : ℕ) := a1 + (n - 1) * d

theorem find_common_difference (a1 d : ℕ) :
  S a1 d 3 = 6 → a a1 d 3 = 4 → d = 2 :=
by
  intros S3_eq a3_eq
  sorry

end find_common_difference_1_1632


namespace min_value_expression_1_1821

theorem min_value_expression (x : ℝ) (h : x > 1) : 
  ∃ min_val, min_val = 6 ∧ ∀ y > 1, 2 * y + 2 / (y - 1) ≥ min_val :=
by  
  use 6
  sorry

end min_value_expression_1_1821


namespace trains_crossing_time_correct_1_1119

def convert_kmph_to_mps (speed_kmph : ℕ) : ℚ := (speed_kmph * 5) / 18

def time_to_cross_each_other 
  (length_train1 length_train2 speed_kmph_train1 speed_kmph_train2 : ℕ) : ℚ :=
  let speed_train1 := convert_kmph_to_mps speed_kmph_train1
  let speed_train2 := convert_kmph_to_mps speed_kmph_train2
  let relative_speed := speed_train2 - speed_train1
  let total_distance := length_train1 + length_train2
  (total_distance : ℚ) / relative_speed

theorem trains_crossing_time_correct :
  time_to_cross_each_other 200 150 40 46 = 210 := by
  sorry

end trains_crossing_time_correct_1_1119


namespace sum_of_discount_rates_1_1342

theorem sum_of_discount_rates : 
  let fox_price := 15
  let pony_price := 20
  let fox_pairs := 3
  let pony_pairs := 2
  let total_savings := 9
  let pony_discount := 18.000000000000014
  let fox_discount := 4
  let total_discount_rate := fox_discount + pony_discount
  total_discount_rate = 22.000000000000014 := by
sorry

end sum_of_discount_rates_1_1342


namespace emerson_row_distance_1_1424

theorem emerson_row_distance (d1 d2 total : ℕ) (h1 : d1 = 6) (h2 : d2 = 18) (h3 : total = 39) :
  15 = total - (d1 + d2) :=
by sorry

end emerson_row_distance_1_1424


namespace sin_600_eq_neg_sqrt_3_div_2_1_1259

theorem sin_600_eq_neg_sqrt_3_div_2 : Real.sin (600 * (Real.pi / 180)) = - (Real.sqrt 3 / 2) :=
by
  sorry

end sin_600_eq_neg_sqrt_3_div_2_1_1259


namespace no_real_solution_x_4_plus_x_plus1_4_plus_x_plus2_4_eq_x_plus3_4_plus_10_1_1985

theorem no_real_solution_x_4_plus_x_plus1_4_plus_x_plus2_4_eq_x_plus3_4_plus_10 :
  ¬ ∃ x : ℝ, x^4 + (x + 1)^4 + (x + 2)^4 = (x + 3)^4 + 10 :=
by {
  sorry
}

end no_real_solution_x_4_plus_x_plus1_4_plus_x_plus2_4_eq_x_plus3_4_plus_10_1_1985


namespace arithmetic_sequence_a3_1_1318

theorem arithmetic_sequence_a3 (a : ℕ → ℤ) (h1 : a 1 = 4) (h10 : a 10 = 22) (d : ℤ) (hd : ∀ n, a n = a 1 + (n - 1) * d) :
  a 3 = 8 :=
by
  -- Skipping the proof
  sorry

end arithmetic_sequence_a3_1_1318


namespace largest_divisor_of_n5_minus_n_1_1680

theorem largest_divisor_of_n5_minus_n (n : ℤ) : 
  ∃ d : ℤ, (∀ n : ℤ, d ∣ (n^5 - n)) ∧ d = 30 :=
sorry

end largest_divisor_of_n5_minus_n_1_1680


namespace factorial_sum_perfect_square_iff_1_1603

def is_perfect_square (n : Nat) : Prop := ∃ m : Nat, m * m = n

def sum_of_factorials (n : Nat) : Nat :=
  (List.range (n + 1)).map Nat.factorial |>.sum

theorem factorial_sum_perfect_square_iff (n : Nat) :
  n = 1 ∨ n = 3 ↔ is_perfect_square (sum_of_factorials n) := by {
  sorry
}

end factorial_sum_perfect_square_iff_1_1603


namespace Cartesian_eq_C2_correct_distance_AB_correct_1_1922

-- Part I: Proving the Cartesian equation of curve (C2)
noncomputable def equation_of_C2 (x y : ℝ) (α : ℝ) : Prop :=
  x = 4 * Real.cos α ∧ y = 4 + 4 * Real.sin α

def Cartesian_eq_C2 (x y : ℝ) : Prop :=
  x^2 + (y - 4)^2 = 16

theorem Cartesian_eq_C2_correct (x y α : ℝ) (h : equation_of_C2 x y α) : Cartesian_eq_C2 x y :=
by sorry

-- Part II: Proving the distance |AB| given polar equations
noncomputable def polar_eq_C1 (theta : ℝ) : ℝ :=
  4 * Real.sin theta

noncomputable def polar_eq_C2 (theta : ℝ) : ℝ :=
  8 * Real.sin theta

def distance_AB (rho1 rho2 : ℝ) : ℝ :=
  abs (rho1 - rho2)

theorem distance_AB_correct : distance_AB (polar_eq_C1 (π / 3)) (polar_eq_C2 (π / 3)) = 2 * Real.sqrt 3 :=
by sorry

end Cartesian_eq_C2_correct_distance_AB_correct_1_1922


namespace probability_of_two_white_balls_1_1471

-- Define the total number of balls
def total_balls : ℕ := 11

-- Define the number of white balls
def white_balls : ℕ := 5

-- Define the number of ways to choose 2 out of n (combinations)
def choose (n r : ℕ) : ℕ := n.choose r

-- Define the total combinations of drawing 2 balls out of 11
def total_combinations : ℕ := choose total_balls 2

-- Define the combinations of drawing 2 white balls out of 5
def white_combinations : ℕ := choose white_balls 2

-- Define the probability of drawing 2 white balls
noncomputable def probability_white : ℚ := (white_combinations : ℚ) / (total_combinations : ℚ)

-- Now, state the theorem that states the desired result
theorem probability_of_two_white_balls : probability_white = 2 / 11 := sorry

end probability_of_two_white_balls_1_1471


namespace at_least_two_consecutive_heads_probability_1_1441

noncomputable def probability_at_least_two_consecutive_heads : ℚ := 
  let total_outcomes := 16
  let unfavorable_outcomes := 8
  1 - (unfavorable_outcomes / total_outcomes)

theorem at_least_two_consecutive_heads_probability :
  probability_at_least_two_consecutive_heads = 1 / 2 := 
by
  sorry

end at_least_two_consecutive_heads_probability_1_1441


namespace arithmetic_geometric_relation_1_1094

variable (a₁ a₂ b₁ b₂ b₃ : ℝ)

-- Conditions
def is_arithmetic_sequence (a₁ a₂ : ℝ) : Prop :=
  ∃ (d : ℝ), -2 + d = a₁ ∧ a₁ + d = a₂ ∧ a₂ + d = -8

def is_geometric_sequence (b₁ b₂ b₃ : ℝ) : Prop :=
  ∃ (r : ℝ), -2 * r = b₁ ∧ b₁ * r = b₂ ∧ b₂ * r = b₃ ∧ b₃ * r = -8

-- The problem statement
theorem arithmetic_geometric_relation (h₁ : is_arithmetic_sequence a₁ a₂) (h₂ : is_geometric_sequence b₁ b₂ b₃) :
  (a₂ - a₁) / b₂ = 1 / 2 := by
    sorry

end arithmetic_geometric_relation_1_1094


namespace inheritance_amount_1_1804

def is_inheritance_amount (x : ℝ) : Prop :=
  let federal_tax := 0.25 * x
  let remaining_after_fed := x - federal_tax
  let state_tax := 0.12 * remaining_after_fed
  let total_tax_paid := federal_tax + state_tax
  total_tax_paid = 15600

theorem inheritance_amount : 
  ∃ x, is_inheritance_amount x ∧ x = 45882 := 
by
  sorry

end inheritance_amount_1_1804


namespace jerky_remaining_after_giving_half_1_1645

-- Define the main conditions as variables
def days := 5
def initial_jerky := 40
def jerky_per_day := 1 + 1 + 2

-- Calculate total consumption
def total_consumption := jerky_per_day * days

-- Calculate remaining jerky
def remaining_jerky := initial_jerky - total_consumption

-- Calculate final jerky after giving half to her brother
def jerky_left := remaining_jerky / 2

-- Statement to be proved
theorem jerky_remaining_after_giving_half :
  jerky_left = 10 :=
by
  -- Proof will go here
  sorry

end jerky_remaining_after_giving_half_1_1645


namespace one_of_18_consecutive_is_divisible_1_1450

-- Define the sum of digits function
def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

-- Define what it means for one number to be divisible by another
def divisible (a b : ℕ) : Prop :=
  b ≠ 0 ∧ a % b = 0

-- The main theorem
theorem one_of_18_consecutive_is_divisible : 
  ∀ (n : ℕ), 100 ≤ n ∧ n + 17 ≤ 999 → ∃ (k : ℕ), n ≤ k ∧ k ≤ (n + 17) ∧ divisible k (sum_of_digits k) :=
by
  intros n h
  sorry

end one_of_18_consecutive_is_divisible_1_1450


namespace find_smallest_subtract_1_1618

-- Definitions for multiples
def is_mul_2 (n : ℕ) : Prop := 2 ∣ n
def is_mul_3 (n : ℕ) : Prop := 3 ∣ n
def is_mul_5 (n : ℕ) : Prop := 5 ∣ n

-- Statement of the problem
theorem find_smallest_subtract (x : ℕ) :
  (is_mul_2 (134 - x)) ∧ (is_mul_3 (134 - x)) ∧ (is_mul_5 (134 - x)) → x = 14 :=
by
  sorry

end find_smallest_subtract_1_1618


namespace election_votes_total_1_1074

-- Definitions representing the conditions
def CandidateAVotes (V : ℕ) := 45 * V / 100
def CandidateBVotes (V : ℕ) := 35 * V / 100
def CandidateCVotes (V : ℕ) := 20 * V / 100

-- Main theorem statement
theorem election_votes_total (V : ℕ) (h1: CandidateAVotes V = 45 * V / 100) (h2: CandidateBVotes V = 35 * V / 100) (h3: CandidateCVotes V = 20 * V / 100)
  (h4: CandidateAVotes V - CandidateBVotes V = 1800) : V = 18000 :=
  sorry

end election_votes_total_1_1074


namespace derivative_f_at_1_1_1809

-- Define the function f(x) = x * ln(x)
noncomputable def f (x : ℝ) : ℝ := x * Real.log x

-- State the theorem to prove f'(1) = 1
theorem derivative_f_at_1 : (deriv f 1) = 1 :=
sorry

end derivative_f_at_1_1_1809


namespace range_of_a_minus_b_1_1637

theorem range_of_a_minus_b {a b : ℝ} (h1 : -2 < a) (h2 : a < 3) (h3 : 1 < b) (h4 : b < 2) : 
  -4 < a - b ∧ a - b < 2 := 
by
  sorry

end range_of_a_minus_b_1_1637


namespace jill_spent_30_percent_on_food_1_1247

variables (T F : ℝ)

theorem jill_spent_30_percent_on_food
  (h1 : 0.04 * T = 0.016 * T + 0.024 * T)
  (h2 : 0.40 + 0.30 + F = 1) :
  F = 0.30 :=
by 
  sorry

end jill_spent_30_percent_on_food_1_1247


namespace geometric_sequence_proof_1_1039

-- Define a geometric sequence with first term 1 and common ratio q with |q| ≠ 1
noncomputable def geometric_sequence (q : ℝ) (n : ℕ) : ℝ :=
  if h : |q| ≠ 1 then (1 : ℝ) * q ^ (n - 1) else 0

-- m should be 11 given the conditions
theorem geometric_sequence_proof (q : ℝ) (m : ℕ) (h : |q| ≠ 1) 
  (hm : geometric_sequence q m = geometric_sequence q 1 * geometric_sequence q 2 * geometric_sequence q 3 * geometric_sequence q 4 * geometric_sequence q 5 ) : 
  m = 11 :=
by
  sorry

end geometric_sequence_proof_1_1039


namespace factorial_power_of_two_1_1741

theorem factorial_power_of_two solutions (a b c : ℕ) (h_a_pos : 0 < a) (h_b_pos : 0 < b) (h_c_pos : 0 < c) (h_equation : a.factorial + b.factorial = 2^(c.factorial)) :
  solutions = [(1, 1, 1), (2, 2, 2)] :=
sorry

end factorial_power_of_two_1_1741


namespace tank_full_capacity_1_1535

theorem tank_full_capacity (w c : ℕ) (h1 : w = c / 6) (h2 : w + 4 = c / 3) : c = 12 :=
sorry

end tank_full_capacity_1_1535


namespace a_equals_5_1_1290

def f (x : ℝ) (a : ℝ) : ℝ := x^3 + a*x^2 + 3*x - 9
def f' (x : ℝ) (a : ℝ) : ℝ := 3*x^2 + 2*a*x + 3

theorem a_equals_5 (a : ℝ) : 
  (∃ x : ℝ, x = -3 ∧ f' x a = 0) → a = 5 := 
by
  sorry

end a_equals_5_1_1290


namespace calculate_arithmetic_expression_1_1779

noncomputable def arithmetic_sum (a d l : ℕ) : ℕ :=
  let n := (l - a) / d + 1
  n * (a + l) / 2

theorem calculate_arithmetic_expression :
  3 * (arithmetic_sum 71 2 99) = 3825 :=
by
  sorry

end calculate_arithmetic_expression_1_1779


namespace find_triples_1_1998

theorem find_triples (x y z : ℕ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (hxy : x ≤ y) (hyz : y ≤ z) 
  (h_eq : x * y + y * z + z * x - x * y * z = 2) : (x = 1 ∧ y = 1 ∧ z = 1) ∨ (x = 2 ∧ y = 3 ∧ z = 4) := 
by 
  sorry

end find_triples_1_1998


namespace find_x_1_1514

theorem find_x (x : ℝ) (h : (x * (x ^ 4) ^ (1/2)) ^ (1/4) = 2) : 
  x = 16 ^ (1/3) :=
sorry

end find_x_1_1514


namespace minimum_value_of_sum_1_1734

theorem minimum_value_of_sum (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h : a + b + c = 1) : 
    1 / (2 * a + b) + 1 / (2 * b + c) + 1 / (2 * c + a) >= 3 :=
by
  sorry

end minimum_value_of_sum_1_1734


namespace flower_bed_dimensions_1_1459

variable (l w : ℕ)

theorem flower_bed_dimensions :
  (l + 3) * (w + 2) = l * w + 64 →
  (l + 2) * (w + 3) = l * w + 68 →
  l = 14 ∧ w = 10 :=
by
  intro h1 h2
  sorry

end flower_bed_dimensions_1_1459


namespace not_p_suff_not_q_1_1271

theorem not_p_suff_not_q (x : ℝ) :
  ¬(|x| ≥ 1) → ¬(x^2 + x - 6 ≥ 0) :=
sorry

end not_p_suff_not_q_1_1271


namespace beef_weight_after_processing_1_1838

def original_weight : ℝ := 861.54
def weight_loss_percentage : ℝ := 0.35
def retained_percentage : ℝ := 1 - weight_loss_percentage
def weight_after_processing (w : ℝ) := retained_percentage * w

theorem beef_weight_after_processing :
  weight_after_processing original_weight = 560.001 :=
by
  sorry

end beef_weight_after_processing_1_1838


namespace side_length_S2_1_1333

def square_side_length 
  (w h : ℕ)
  (R1 R2 : ℕ → ℕ → Prop) 
  (S1 S2 S3 : ℕ → Prop) 
  (r s : ℕ) 
  (combined_rectangle : ℕ × ℕ → Prop)
  (cond1 : combined_rectangle (3330, 2030))
  (cond2 : R1 r s) 
  (cond3 : R2 r s) 
  (cond4 : S1 (r + s)) 
  (cond5 : S2 s) 
  (cond6 : S3 (r + s)) 
  (cond7 : 2 * r + s = 2030) 
  (cond8 : 2 * r + 3 * s = 3330) : Prop :=
  s = 650

theorem side_length_S2 (w h : ℕ)
  (R1 R2 : ℕ → ℕ → Prop) 
  (S1 S2 S3 : ℕ → Prop) 
  (r s : ℕ) 
  (combined_rectangle : ℕ × ℕ → Prop)
  (cond1 : combined_rectangle (3330, 2030))
  (cond2 : R1 r s) 
  (cond3 : R2 r s) 
  (cond4 : S1 (r + s)) 
  (cond5 : S2 s) 
  (cond6 : S3 (r + s)) 
  (cond7 : 2 * r + s = 2030) 
  (cond8 : 2 * r + 3 * s = 3330) : square_side_length w h R1 R2 S1 S2 S3 r s combined_rectangle cond1 cond2 cond3 cond4 cond5 cond6 cond7 cond8 :=
sorry

end side_length_S2_1_1333


namespace min_value_of_quadratic_function_1_1654

def quadratic_function (x : ℝ) : ℝ :=
  x^2 + 2 * x - 5

theorem min_value_of_quadratic_function :
  ∃ x : ℝ, quadratic_function x = -1 :=
by
  sorry

end min_value_of_quadratic_function_1_1654


namespace probability_of_log_ge_than_1_1_1545

noncomputable def probability_log_greater_than_one : ℝ := sorry

theorem probability_of_log_ge_than_1 :
  probability_log_greater_than_one = 1 / 2 :=
sorry

end probability_of_log_ge_than_1_1_1545


namespace sum_of_repeating_decimals_1_1227

-- Definitions of repeating decimals x and y
def x : ℚ := 25 / 99
def y : ℚ := 87 / 99

-- The assertion that the sum of these repeating decimals is equal to 112/99 as a fraction
theorem sum_of_repeating_decimals: x + y = 112 / 99 := by
  sorry

end sum_of_repeating_decimals_1_1227


namespace sufficient_but_not_necessary_condition_1_1807

theorem sufficient_but_not_necessary_condition (a : ℝ) :
  (∃ a, (1 + a) ^ 6 = 64) →
  (a = 1 → (1 + a) ^ 6 = 64) ∧ ¬(∀ a, ((1 + a) ^ 6 = 64 → a = 1)) :=
by
  sorry

end sufficient_but_not_necessary_condition_1_1807


namespace page_copy_cost_1_1762

theorem page_copy_cost (cost_per_4_pages : ℕ) (page_count : ℕ) (dollar_to_cents : ℕ) : cost_per_4_pages = 8 → page_count = 4 → dollar_to_cents = 100 → (1500 * (page_count / cost_per_4_pages) = 750) :=
by
  intros
  sorry

end page_copy_cost_1_1762


namespace fewerEmployeesAbroadThanInKorea_1_1575

def totalEmployees : Nat := 928
def employeesInKorea : Nat := 713
def employeesAbroad : Nat := totalEmployees - employeesInKorea

theorem fewerEmployeesAbroadThanInKorea :
  employeesInKorea - employeesAbroad = 498 :=
by
  sorry

end fewerEmployeesAbroadThanInKorea_1_1575


namespace percentage_of_all_students_with_cars_1_1959

def seniors := 300
def percent_seniors_with_cars := 0.40
def lower_grades := 1500
def percent_lower_grades_with_cars := 0.10

theorem percentage_of_all_students_with_cars :
  (120 + 150) / 1800 * 100 = 15 := by
  sorry

end percentage_of_all_students_with_cars_1_1959


namespace domain_of_function_1_1543

section
variable (x : ℝ)

def condition_1 := x + 4 ≥ 0
def condition_2 := x + 2 ≠ 0
def domain := { x : ℝ | x ≥ -4 ∧ x ≠ -2 }

theorem domain_of_function : (condition_1 x ∧ condition_2 x) ↔ (x ∈ domain) :=
by
  sorry
end

end domain_of_function_1_1543


namespace monotonic_intervals_a1_decreasing_on_1_to_2_exists_a_for_minimum_value_1_1854

-- Proof Problem I
noncomputable def f1 (x : ℝ) := x^2 + x - Real.log x

theorem monotonic_intervals_a1 : 
  (∀ x, 0 < x ∧ x < 1 / 2 → f1 x < 0) ∧ (∀ x, 1 / 2 < x → f1 x > 0) := 
sorry

-- Proof Problem II
noncomputable def f2 (x : ℝ) (a : ℝ) := x^2 + a * x - Real.log x

theorem decreasing_on_1_to_2 (a : ℝ) :
  (∀ x, 1 ≤ x ∧ x ≤ 2 → f2 x a ≤ 0) → a ≤ -7 / 2 :=
sorry

-- Proof Problem III
noncomputable def g (x : ℝ) (a : ℝ) := a * x - Real.log x

theorem exists_a_for_minimum_value :
  ∃ a : ℝ, (∀ x, 0 < x ∧ x ≤ Real.exp 1 → g x a = 3) ∧ a = Real.exp 2 :=
sorry

end monotonic_intervals_a1_decreasing_on_1_to_2_exists_a_for_minimum_value_1_1854


namespace smallest_lcm_value_1_1676

def is_five_digit (x : ℕ) : Prop :=
  10000 ≤ x ∧ x < 100000

theorem smallest_lcm_value :
  ∃ (m n : ℕ), is_five_digit m ∧ is_five_digit n ∧ Nat.gcd m n = 5 ∧ Nat.lcm m n = 20030010 :=
by
  sorry

end smallest_lcm_value_1_1676


namespace slope_of_line_1_1699

theorem slope_of_line (x y : ℝ) : 
  3 * y + 9 = -6 * x - 15 → 
  ∃ m b, y = m * x + b ∧ m = -2 := 
by {
  sorry
}

end slope_of_line_1_1699


namespace rosy_current_age_1_1220

theorem rosy_current_age 
  (R : ℕ) 
  (h1 : ∀ (david_age rosy_age : ℕ), david_age = rosy_age + 12) 
  (h2 : ∀ (david_age_plus_4 rosy_age_plus_4 : ℕ), david_age_plus_4 = 2 * rosy_age_plus_4) : 
  R = 8 := 
sorry

end rosy_current_age_1_1220


namespace curve_B_is_not_good_1_1197

-- Define the points A and B
def A : ℝ × ℝ := (-5, 0)
def B : ℝ × ℝ := (5, 0)

-- Define the condition for being a "good curve"
def is_good_curve (C : ℝ × ℝ → Prop) : Prop :=
  ∃ M : ℝ × ℝ, C M ∧ abs (dist M A - dist M B) = 8

-- Define the curves
def curve_A (p : ℝ × ℝ) : Prop := p.1 + p.2 = 5
def curve_B (p : ℝ × ℝ) : Prop := p.1 ^ 2 + p.2 ^ 2 = 9
def curve_C (p : ℝ × ℝ) : Prop := (p.1 ^ 2) / 25 + (p.2 ^ 2) / 9 = 1
def curve_D (p : ℝ × ℝ) : Prop := p.1 ^ 2 = 16 * p.2

-- Prove that curve_B is not a "good curve"
theorem curve_B_is_not_good : ¬ is_good_curve curve_B := by
  sorry

end curve_B_is_not_good_1_1197


namespace num_positive_k_for_solution_to_kx_minus_18_eq_3k_1_1783

theorem num_positive_k_for_solution_to_kx_minus_18_eq_3k : 
  ∃ (k_vals : Finset ℕ), 
  (∀ k ∈ k_vals, ∃ x : ℤ, k * x - 18 = 3 * k) ∧ 
  k_vals.card = 6 :=
by
  sorry

end num_positive_k_for_solution_to_kx_minus_18_eq_3k_1_1783


namespace proof_problem_1_1879

variable {a b c : ℝ}

-- Condition: a < 0
variable (ha : a < 0)
-- Condition: b > 0
variable (hb : b > 0)
-- Condition: c > 0
variable (hc : c > 0)
-- Condition: a < b < c
variable (hab : a < b) (hbc : b < c)

-- Proof statement
theorem proof_problem :
  (ab * b < b * c) ∧
  (a * c < b * c) ∧
  (a + c < b + c) ∧
  (c / a < 1) :=
  by
    sorry

end proof_problem_1_1879


namespace insert_zeros_between_digits_is_cube_1_1611

def is_perfect_cube (n : ℕ) : Prop := ∃ k : ℕ, k^3 = n

theorem insert_zeros_between_digits_is_cube (k b : ℕ) (h_b : b ≥ 4) 
  : is_perfect_cube (1 * b^(3*(1+k)) + 3 * b^(2*(1+k)) + 3 * b^(1+k) + 1) :=
sorry

end insert_zeros_between_digits_is_cube_1_1611


namespace car_highway_mileage_1_1324

theorem car_highway_mileage :
  (∀ (H : ℝ), 
    (H > 0) → 
    (4 / H + 4 / 20 = (8 / H) * 1.4000000000000001) → 
    (H = 36)) :=
by
  intros H H_pos h_cond
  have : H = 36 := 
    sorry
  exact this

end car_highway_mileage_1_1324


namespace sum_and_difference_repeating_decimals_1_1822

noncomputable def repeating_decimal_6 : ℚ := 2 / 3
noncomputable def repeating_decimal_2 : ℚ := 2 / 9
noncomputable def repeating_decimal_9 : ℚ := 1
noncomputable def repeating_decimal_3 : ℚ := 1 / 3

theorem sum_and_difference_repeating_decimals :
  repeating_decimal_6 + repeating_decimal_2 - repeating_decimal_9 + repeating_decimal_3 = 2 / 9 := 
by 
  sorry

end sum_and_difference_repeating_decimals_1_1822


namespace problem_1_1362

variable (α : ℝ)

def setA : Set ℝ := {Real.sin α, Real.cos α, 1}
def setB : Set ℝ := {Real.sin α ^ 2, Real.sin α + Real.cos α, 0}
theorem problem (h : setA α = setB α) : Real.sin α ^ 2009 + Real.cos α ^ 2009 = -1 := 
by 
  sorry

end problem_1_1362


namespace alice_marble_groups_1_1262

-- Define the number of each colored marble Alice has
def pink_marble := 1
def blue_marble := 1
def white_marble := 1
def black_marbles := 4

-- The function to count the number of different groups of two marbles Alice can choose
noncomputable def count_groups : Nat :=
  let total_colors := 4  -- Pink, Blue, White, and one representative black
  1 + (total_colors.choose 2)

-- The theorem statement 
theorem alice_marble_groups : count_groups = 7 := by 
  sorry

end alice_marble_groups_1_1262


namespace minimum_value_f_1_1946

noncomputable def f (a b c : ℝ) : ℝ :=
  a / (Real.sqrt (a^2 + 8*b*c)) + b / (Real.sqrt (b^2 + 8*a*c)) + c / (Real.sqrt (c^2 + 8*a*b))

theorem minimum_value_f (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) : 
  1 ≤ f a b c := by
  sorry

end minimum_value_f_1_1946


namespace eight_bees_have_48_legs_1_1863

  def legs_per_bee : ℕ := 6
  def number_of_bees : ℕ := 8
  def total_legs : ℕ := 48

  theorem eight_bees_have_48_legs :
    number_of_bees * legs_per_bee = total_legs :=
  by
    sorry
  
end eight_bees_have_48_legs_1_1863


namespace irreducible_fraction_for_any_n_1_1452

theorem irreducible_fraction_for_any_n (n : ℤ) : Int.gcd (14 * n + 3) (21 * n + 4) = 1 := 
by {
  sorry
}

end irreducible_fraction_for_any_n_1_1452


namespace find_arrays_1_1081

-- Defines a condition where positive integers satisfy the given properties
def satisfies_conditions (a b c d : ℕ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ 
  a ∣ b * c * d - 1 ∧ 
  b ∣ a * c * d - 1 ∧ 
  c ∣ a * b * d - 1 ∧ 
  d ∣ a * b * c - 1

-- The theorem that any four positive integers satisfying the conditions are either (2, 3, 7, 11) or (2, 3, 11, 13)
theorem find_arrays :
  ∀ a b c d : ℕ, satisfies_conditions a b c d → 
    (a = 2 ∧ b = 3 ∧ c = 7 ∧ d = 11) ∨ 
    (a = 2 ∧ b = 3 ∧ c = 11 ∧ d = 13) :=
by
  intro a b c d h
  sorry

end find_arrays_1_1081


namespace find_cost_of_pencil_and_pen_1_1013

variable (p q r : ℝ)

-- Definitions based on conditions
def condition1 := 3 * p + 2 * q + r = 4.20
def condition2 := p + 3 * q + 2 * r = 4.75
def condition3 := 2 * r = 3.00

-- The theorem to prove
theorem find_cost_of_pencil_and_pen (p q r : ℝ) (h1 : condition1 p q r) (h2 : condition2 p q r) (h3 : condition3 r) :
  p + q = 1.12 :=
by
  sorry

end find_cost_of_pencil_and_pen_1_1013


namespace value_of_a_1_1276

theorem value_of_a (a b c : ℕ) (h1 : a + b = 12) (h2 : b + c = 16) (h3 : c = 7) : a = 3 := by
  sorry

end value_of_a_1_1276


namespace compare_trig_values_1_1444

noncomputable def a : ℝ := Real.sin (2 * Real.pi / 7)
noncomputable def b : ℝ := Real.tan (5 * Real.pi / 7)
noncomputable def c : ℝ := Real.cos (5 * Real.pi / 7)

theorem compare_trig_values :
  (0 < 2 * Real.pi / 7 ∧ 2 * Real.pi / 7 < Real.pi / 2) →
  (Real.pi / 2 < 5 * Real.pi / 7 ∧ 5 * Real.pi / 7 < 3 * Real.pi / 4) →
  b < c ∧ c < a :=
by
  intro h1 h2
  sorry

end compare_trig_values_1_1444


namespace max_non_overlapping_areas_1_1090

theorem max_non_overlapping_areas (n : ℕ) : 
  ∃ (max_areas : ℕ), max_areas = 3 * n := by
  sorry

end max_non_overlapping_areas_1_1090


namespace analytical_expression_range_of_t_1_1793

noncomputable def f (x : ℝ) : ℝ := x^2 - 3 * x

theorem analytical_expression (x : ℝ) :
  (f (x + 1) - f x = 2 * x - 2) ∧ (f 1 = -2) :=
by
  sorry

theorem range_of_t (t : ℝ) :
  (∀ x : ℝ, f x > 0 ∧ f (x + t) < 0 → x = 1) ↔ (-2 <= t ∧ t < -1) :=
by
  sorry

end analytical_expression_range_of_t_1_1793


namespace find_positive_number_1_1685

theorem find_positive_number (m : ℝ) 
  (h : (m - 1)^2 = (3 * m - 5)^2) : 
  (m - 1)^2 = 1 ∨ (m - 1)^2 = 1 / 4 :=
by sorry

end find_positive_number_1_1685


namespace sequence_periodic_1_1768

def last_digit (n : ℕ) : ℕ := n % 10

noncomputable def a_n (n : ℕ) : ℕ := last_digit (n^(n^n))

theorem sequence_periodic :
  ∃ period : ℕ, period = 20 ∧ ∀ n m : ℕ, n ≡ m [MOD period] → a_n n = a_n m :=
sorry

end sequence_periodic_1_1768


namespace union_complement_eq_1_1800

def U : Set Nat := {0, 1, 2, 4, 6, 8}
def M : Set Nat := {0, 4, 6}
def N : Set Nat := {0, 1, 6}
def complement (u : Set α) (s : Set α) : Set α := {x ∈ u | x ∉ s}

theorem union_complement_eq :
  M ∪ (complement U N) = {0, 2, 4, 6, 8} :=
by sorry

end union_complement_eq_1_1800


namespace largest_n_for_crates_1_1731

theorem largest_n_for_crates (total_crates : ℕ) (min_oranges max_oranges : ℕ)
  (h1 : total_crates = 145)
  (h2 : min_oranges = 110)
  (h3 : max_oranges = 140) : 
  ∃ n : ℕ, n = 5 ∧ ∀ k : ℕ, k ≤ max_oranges - min_oranges + 1 → total_crates / k ≤ n :=
  by {
    sorry
  }

end largest_n_for_crates_1_1731


namespace find_current_1_1385

open Complex

noncomputable def V : ℂ := 2 + I
noncomputable def Z : ℂ := 2 - 4 * I

theorem find_current :
  V / Z = (1 / 2) * I := 
sorry

end find_current_1_1385


namespace units_of_Product_C_sold_1_1980

-- Definitions of commission rates
def commission_rate_A : ℝ := 0.05
def commission_rate_B : ℝ := 0.07
def commission_rate_C : ℝ := 0.10

-- Definitions of revenues per unit
def revenue_A : ℝ := 1500
def revenue_B : ℝ := 2000
def revenue_C : ℝ := 3500

-- Definition of units sold
def units_A : ℕ := 5
def units_B : ℕ := 3

-- Commission calculations for Product A and B
def commission_A : ℝ := commission_rate_A * revenue_A * units_A
def commission_B : ℝ := commission_rate_B * revenue_B * units_B

-- Previous average commission and new average commission
def previous_avg_commission : ℝ := 100
def new_avg_commission : ℝ := 250

-- The main proof statement
theorem units_of_Product_C_sold (x : ℝ) (h1 : new_avg_commission = previous_avg_commission + 150)
  (h2 : total_units = units_A + units_B + x)
  (h3 : total_new_commission = commission_A + commission_B + (commission_rate_C * revenue_C * x))
  : x = 12 :=
by
  sorry

end units_of_Product_C_sold_1_1980


namespace termite_ridden_fraction_1_1913

theorem termite_ridden_fraction (T : ℝ)
  (h1 : (3 / 10) * T = 0.1) : T = 1 / 3 :=
by
  -- proof goes here
  sorry

end termite_ridden_fraction_1_1913


namespace length_of_bridge_1_1942

-- Define the conditions
def train_length : ℕ := 130 -- length of the train in meters
def train_speed : ℕ := 45  -- speed of the train in km/hr
def crossing_time : ℕ := 30  -- time to cross the bridge in seconds

-- Prove that the length of the bridge is 245 meters
theorem length_of_bridge : 
  (train_speed * 1000 / 3600 * crossing_time) - train_length = 245 := 
by
  sorry

end length_of_bridge_1_1942


namespace problem1_problem2_1_1760

-- Statement for Question (1)
theorem problem1 (x : ℝ) (h : |x - 1| + x ≥ x + 2) : x ≤ -1 ∨ x ≥ 3 :=
  sorry

-- Statement for Question (2)
theorem problem2 (a : ℝ) (h : ∀ x : ℝ, |x - a| + x ≤ 3 * x → x ≥ 2) : a = 6 :=
  sorry

end problem1_problem2_1_1760


namespace perimeter_of_intersection_triangle_1_1179

theorem perimeter_of_intersection_triangle :
  ∀ (P Q R : Type) (dist : P → Q → ℝ) (length_PQ length_QR length_PR seg_ellP seg_ellQ seg_ellR : ℝ),
  (length_PQ = 150) →
  (length_QR = 250) →
  (length_PR = 200) →
  (seg_ellP = 75) →
  (seg_ellQ = 50) →
  (seg_ellR = 25) →
  let TU := seg_ellP + seg_ellQ
  let US := seg_ellQ + seg_ellR
  let ST := seg_ellR + (seg_ellR * (length_QR / length_PQ))
  TU + US + ST = 266.67 :=
by
  intros P Q R dist length_PQ length_QR length_PR seg_ellP seg_ellQ seg_ellR hPQ hQR hPR hP hQ hR
  let TU := seg_ellP + seg_ellQ
  let US := seg_ellQ + seg_ellR
  let ST := seg_ellR + (seg_ellR * (length_QR / length_PQ))
  have : TU + US + ST = 266.67 := sorry
  exact this

end perimeter_of_intersection_triangle_1_1179


namespace slope_of_line_1_1954

theorem slope_of_line (x1 y1 x2 y2 : ℝ)
  (h1 : 4 * y1 + 6 * x1 = 0)
  (h2 : 4 * y2 + 6 * x2 = 0)
  (h1x2 : x1 ≠ x2) :
  (y2 - y1) / (x2 - x1) = -3 / 2 :=
by sorry

end slope_of_line_1_1954


namespace james_muffins_correct_1_1572

-- Arthur baked 115 muffins
def arthur_muffins : ℕ := 115

-- James baked 12 times as many muffins as Arthur
def james_multiplier : ℕ := 12

-- The number of muffins James baked
def james_muffins : ℕ := arthur_muffins * james_multiplier

-- The expected result
def expected_james_muffins : ℕ := 1380

-- The statement we want to prove
theorem james_muffins_correct : james_muffins = expected_james_muffins := by
  sorry

end james_muffins_correct_1_1572


namespace probability_of_all_heads_or_tails_1_1633

theorem probability_of_all_heads_or_tails :
  let possible_outcomes := 256
  let favorable_outcomes := 2
  favorable_outcomes / possible_outcomes = 1 / 128 := by
  sorry

end probability_of_all_heads_or_tails_1_1633


namespace max_value_of_XYZ_XY_YZ_ZX_1_1725

theorem max_value_of_XYZ_XY_YZ_ZX (X Y Z : ℕ) (h : X + Y + Z = 15) : 
  X * Y * Z + X * Y + Y * Z + Z * X ≤ 200 := 
sorry

end max_value_of_XYZ_XY_YZ_ZX_1_1725


namespace remainder_when_2x_divided_by_7_1_1751

theorem remainder_when_2x_divided_by_7 (x y r : ℤ) (h1 : x = 10 * y + 3)
    (h2 : 2 * x = 7 * (3 * y) + r) (h3 : 11 * y - x = 2) : r = 1 := by
  sorry

end remainder_when_2x_divided_by_7_1_1751


namespace solution_set_of_inequality_1_1963

theorem solution_set_of_inequality (x : ℝ) : x * (9 - x) > 0 ↔ 0 < x ∧ x < 9 := by
  sorry

end solution_set_of_inequality_1_1963


namespace solve_for_x_1_1697

theorem solve_for_x (x : ℝ) (h₁ : (x + 2) ≠ 0) (h₂ : (|x| - 2) / (x + 2) = 0) : x = 2 := by
  sorry

end solve_for_x_1_1697


namespace point_translation_1_1775

theorem point_translation :
  ∃ (x_old y_old x_new y_new : ℤ),
  (x_old = 1 ∧ y_old = -2) ∧
  (x_new = x_old + 2) ∧
  (y_new = y_old + 3) ∧
  (x_new = 3) ∧
  (y_new = 1) :=
sorry

end point_translation_1_1775


namespace percentage_error_in_area_1_1835

-- Definitions based on conditions
def actual_side (s : ℝ) := s
def measured_side (s : ℝ) := s * 1.01
def actual_area (s : ℝ) := s^2
def calculated_area (s : ℝ) := (measured_side s)^2

-- Theorem statement of the proof problem
theorem percentage_error_in_area (s : ℝ) : 
  (calculated_area s - actual_area s) / actual_area s * 100 = 2.01 := 
by 
  -- Proof is omitted
  sorry

end percentage_error_in_area_1_1835


namespace swimming_time_1_1269

theorem swimming_time (c t : ℝ) 
  (h1 : 10.5 + c ≠ 0)
  (h2 : 10.5 - c ≠ 0)
  (h3 : t = 45 / (10.5 + c))
  (h4 : t = 18 / (10.5 - c)) :
  t = 3 := 
by
  sorry

end swimming_time_1_1269


namespace min_value_x_plus_y_1_1721

theorem min_value_x_plus_y (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : 4 / x + 1 / y = 1 / 2) : x + y ≥ 18 := sorry

end min_value_x_plus_y_1_1721


namespace harper_water_duration_1_1427

theorem harper_water_duration
  (half_bottle_per_day : ℝ)
  (bottles_per_case : ℕ)
  (cost_per_case : ℝ)
  (total_spending : ℝ)
  (cases_bought : ℕ)
  (days_per_case : ℕ)
  (total_days : ℕ) :
  half_bottle_per_day = 1/2 →
  bottles_per_case = 24 →
  cost_per_case = 12 →
  total_spending = 60 →
  cases_bought = total_spending / cost_per_case →
  days_per_case = bottles_per_case * 2 →
  total_days = days_per_case * cases_bought →
  total_days = 240 :=
by
  -- The proof will be added here
  sorry

end harper_water_duration_1_1427


namespace union_complement_set_1_1802

open Set

variable (U : Set ℕ) (M : Set ℕ) (N : Set ℕ)

def complement_in_U (U N : Set ℕ) : Set ℕ :=
  U \ N

theorem union_complement_set :
  U = {0, 1, 2, 4, 6, 8} →
  M = {0, 4, 6} →
  N = {0, 1, 6} →
  M ∪ (complement_in_U U N) = {0, 2, 4, 6, 8} :=
by
  intros
  rw [complement_in_U, union_comm]
  sorry

end union_complement_set_1_1802


namespace red_tint_percentage_new_mixture_1_1037

-- Definitions of the initial conditions
def original_volume : ℝ := 50
def red_tint_percentage : ℝ := 0.20
def added_red_tint : ℝ := 6

-- Definition for the proof
theorem red_tint_percentage_new_mixture : 
  let original_red_tint := red_tint_percentage * original_volume
  let new_red_tint := original_red_tint + added_red_tint
  let new_total_volume := original_volume + added_red_tint
  (new_red_tint / new_total_volume) * 100 = 28.57 :=
by
  sorry

end red_tint_percentage_new_mixture_1_1037


namespace no_naturals_satisfy_divisibility_condition_1_1461

theorem no_naturals_satisfy_divisibility_condition :
  ∀ (a b c : ℕ), ¬ (2013 * (a * b + b * c + c * a) ∣ a^2 + b^2 + c^2) :=
by
  sorry

end no_naturals_satisfy_divisibility_condition_1_1461


namespace equation_solutions_1_1636

theorem equation_solutions :
  ∀ x y : ℤ, x^2 + x * y + y^2 + x + y - 5 = 0 → (x = 1 ∧ y = 1) ∨ (x = 1 ∧ y = -3) ∨ (x = -3 ∧ y = 1) :=
by
  intro x y h
  sorry

end equation_solutions_1_1636


namespace man_speed_against_current_1_1607

-- Definitions for the problem conditions
def man_speed_with_current : ℝ := 21
def current_speed : ℝ := 4.3

-- Main proof statement
theorem man_speed_against_current : man_speed_with_current - 2 * current_speed = 12.4 :=
by
  sorry

end man_speed_against_current_1_1607


namespace shortest_wire_length_1_1224

theorem shortest_wire_length (d1 d2 : ℝ) (r1 r2 : ℝ) (t : ℝ) :
  d1 = 8 ∧ d2 = 20 ∧ r1 = 4 ∧ r2 = 10 ∧ t = 8 * Real.sqrt 10 + 17.4 * Real.pi → 
  ∃ l : ℝ, l = t :=
by 
  sorry

end shortest_wire_length_1_1224


namespace exp_mono_increasing_1_1211

theorem exp_mono_increasing (x y : ℝ) (h : x ≤ y) : (2:ℝ)^x ≤ (2:ℝ)^y :=
sorry

end exp_mono_increasing_1_1211


namespace spadesuit_eval_1_1410

def spadesuit (x y : ℝ) : ℝ :=
  (x + y) * (x - y)

theorem spadesuit_eval :
  spadesuit 5 (spadesuit 6 3) = -704 := by
  sorry

end spadesuit_eval_1_1410


namespace fraction_of_second_year_given_not_third_year_1_1390

theorem fraction_of_second_year_given_not_third_year (total_students : ℕ) 
  (third_year_students : ℕ) (second_year_students : ℕ) :
  third_year_students = total_students * 30 / 100 →
  second_year_students = total_students * 10 / 100 →
  ↑second_year_students / (total_students - third_year_students) = (1 : ℚ) / 7 :=
by
  -- Proof omitted
  sorry

end fraction_of_second_year_given_not_third_year_1_1390


namespace daisies_multiple_of_4_1_1509

def num_roses := 8
def num_daisies (D : ℕ) := D
def num_marigolds := 48
def num_arrangements := 4

theorem daisies_multiple_of_4 (D : ℕ) 
  (h_roses_div_4 : num_roses % num_arrangements = 0)
  (h_marigolds_div_4 : num_marigolds % num_arrangements = 0)
  (h_total_div_4 : (num_roses + num_daisies D + num_marigolds) % num_arrangements = 0) :
  D % 4 = 0 :=
sorry

end daisies_multiple_of_4_1_1509


namespace units_digit_of_7_pow_6_pow_5_1_1052

theorem units_digit_of_7_pow_6_pow_5 : (7^(6^5)) % 10 = 1 := by
  -- Proof goes here
  sorry

end units_digit_of_7_pow_6_pow_5_1_1052


namespace train_crosses_platform_in_15_seconds_1_1345

-- Definitions based on conditions
def length_of_train : ℝ := 330 -- in meters
def tunnel_length : ℝ := 1200 -- in meters
def time_to_cross_tunnel : ℝ := 45 -- in seconds
def platform_length : ℝ := 180 -- in meters

-- Definition based on the solution but directly asserting the correct answer.
def time_to_cross_platform : ℝ := 15 -- in seconds

-- Lean statement
theorem train_crosses_platform_in_15_seconds :
  (length_of_train + platform_length) / ((length_of_train + tunnel_length) / time_to_cross_tunnel) = time_to_cross_platform :=
by
  sorry

end train_crosses_platform_in_15_seconds_1_1345


namespace medians_inequality_1_1904

  variable {a b c : ℝ} (h_triangle : a + b > c ∧ a + c > b ∧ b + c > a)

  noncomputable def median_length (a b c : ℝ) : ℝ :=
    1 / 2 * Real.sqrt (2 * b^2 + 2 * c^2 - a^2)

  noncomputable def semiperimeter (a b c : ℝ) : ℝ :=
    (a + b + c) / 2

  theorem medians_inequality (m_a m_b m_c s: ℝ)
    (h_ma : m_a = median_length a b c)
    (h_mb : m_b = median_length b c a)
    (h_mc : m_c = median_length c a b)
    (h_s : s = semiperimeter a b c) :
    m_a^2 + m_b^2 + m_c^2 ≥ s^2 := by
  sorry
  
end medians_inequality_1_1904


namespace polynomial_remainder_1_1421

theorem polynomial_remainder (y : ℂ) (h1 : y^5 + y^4 + y^3 + y^2 + y + 1 = 0) (h2 : y^6 = 1) :
  (y^55 + y^40 + y^25 + y^10 + 1) % (y^5 + y^4 + y^3 + y^2 + y + 1) = 2 * y + 3 :=
sorry

end polynomial_remainder_1_1421


namespace general_formula_an_sum_first_n_terms_cn_1_1272

-- Define sequences and conditions
def arithmetic_sequence (a : ℕ → ℤ) (d : ℤ) : Prop := ∀ n, a (n + 1) = a n + d
def geometric_sequence (b : ℕ → ℤ) (r : ℤ) : Prop := ∀ n, b (n + 1) = b n * r

variables (a : ℕ → ℤ) (b : ℕ → ℤ)

-- Given conditions
axiom a_is_arithmetic : arithmetic_sequence a 2
axiom b_is_geometric : geometric_sequence b 3
axiom b2_eq_3 : b 2 = 3
axiom b3_eq_9 : b 3 = 9
axiom a1_eq_b1 : a 1 = b 1
axiom a14_eq_b4 : a 14 = b 4

-- Results to be proved
theorem general_formula_an : ∀ n, a n = 2 * n - 1 := sorry
theorem sum_first_n_terms_cn : ∀ n, (∑ i in Finset.range n, (a i + b i)) = n^2 + (3^n - 1) / 2 := sorry

end general_formula_an_sum_first_n_terms_cn_1_1272


namespace area_of_isosceles_trapezoid_1_1195

variable (a b c d : ℝ) -- Variables for sides and bases of the trapezoid

-- Define isosceles trapezoid with given sides and bases
def is_isosceles_trapezoid (a b c d : ℝ) (h : ℝ) :=
  a = b ∧ c = 10 ∧ d = 16 ∧ (∃ (h : ℝ), a^2 = h^2 + ((d - c) / 2)^2 ∧ a = 5)

-- Lean theorem for the area of the isosceles trapezoid
theorem area_of_isosceles_trapezoid :
  ∀ (a b c d : ℝ) (h : ℝ), is_isosceles_trapezoid a b c d h
  → (1 / 2) * (c + d) * h = 52 :=
by
  sorry

end area_of_isosceles_trapezoid_1_1195


namespace ages_correct_1_1929

-- Definitions of the given conditions
def john_age : ℕ := 42
def tim_age : ℕ := 79
def james_age : ℕ := 30
def lisa_age : ℚ := 54.5
def kate_age : ℕ := 34
def michael_age : ℚ := 61.5
def anna_age : ℚ := 54.5

-- Mathematically equivalent proof problem
theorem ages_correct :
  (james_age = 30) ∧
  (lisa_age = 54.5) ∧
  (kate_age = 34) ∧
  (michael_age = 61.5) ∧
  (anna_age = 54.5) :=
by {
  sorry  -- Proof to be filled in
}

end ages_correct_1_1929


namespace solve_eq_1_1701

theorem solve_eq (x y : ℕ) (h : x^2 - 2 * x * y + y^2 + 5 * x + 5 * y = 1500) :
  (x = 150 ∧ y = 150) ∨ (x = 150 ∧ y = 145) ∨ (x = 145 ∧ y = 135) ∨
  (x = 135 ∧ y = 120) ∨ (x = 120 ∧ y = 100) ∨ (x = 100 ∧ y = 75) ∨
  (x = 75 ∧ y = 45) ∨ (x = 45 ∧ y = 10) ∨ (x = 145 ∧ y = 150) ∨
  (x = 135 ∧ y = 145) ∨ (x = 120 ∧ y = 135) ∨ (x = 100 ∧ y = 120) ∨
  (x = 75 ∧ y = 100) ∨ (x = 45 ∧ y = 75) ∨ (x = 10 ∧ y = 45) :=
sorry

end solve_eq_1_1701


namespace anthony_pencils_1_1454

def initial_pencils : ℝ := 56.0  -- Condition 1
def pencils_left : ℝ := 47.0     -- Condition 2
def pencils_given : ℝ := 9.0     -- Correct Answer

theorem anthony_pencils :
  initial_pencils - pencils_left = pencils_given :=
by
  sorry

end anthony_pencils_1_1454


namespace triangle_area_1_1377

variable (a b c : ℕ)
variable (s : ℕ := 21)
variable (area : ℕ := 84)

theorem triangle_area 
(h1 : c = a + b - 12) 
(h2 : (a + b + c) / 2 = s) 
(h3 : c - a = 2) 
: (21 * (21 - a) * (21 - b) * (21 - c)).sqrt = area := 
sorry

end triangle_area_1_1377


namespace cos_sin_value_1_1501

theorem cos_sin_value (α : ℝ) (h : Real.tan α = Real.sqrt 2) : Real.cos α * Real.sin α = Real.sqrt 2 / 3 :=
sorry

end cos_sin_value_1_1501


namespace product_base9_conversion_1_1522

noncomputable def base_9_to_base_10 (n : ℕ) : ℕ :=
match n with
| 237 => 2 * 9^2 + 3 * 9^1 + 7
| 17 => 9 + 7
| _ => 0

noncomputable def base_10_to_base_9 (n : ℕ) : ℕ :=
match n with
-- Step of conversion from example: 3136 => 4*9^3 + 2*9^2 + 6*9^1 + 4*9^0
| 3136 => 4 * 1000 + 2 * 100 + 6 * 10 + 4 -- representing 4264 in base 9
| _ => 0

theorem product_base9_conversion :
  base_10_to_base_9 ((base_9_to_base_10 237) * (base_9_to_base_10 17)) = 4264 := by
  sorry

end product_base9_conversion_1_1522


namespace sqrt_7_estimate_1_1974

theorem sqrt_7_estimate (h1 : 4 < 7) (h2 : 7 < 9) (h3 : Nat.sqrt 4 = 2) (h4 : Nat.sqrt 9 = 3) : 2 < Real.sqrt 7 ∧ Real.sqrt 7 < 3 :=
  by {
    -- the proof would go here, but use 'sorry' to omit it
    sorry
  }

end sqrt_7_estimate_1_1974


namespace executed_is_9_1_1552

-- Define the conditions based on given problem
variables (x K I : ℕ)

-- Condition 1: Number of killed
def number_killed (x : ℕ) : ℕ := 2 * x + 4

-- Condition 2: Number of injured
def number_injured (x : ℕ) : ℕ := (16 * x) / 3 + 8

-- Condition 3: Total of killed, injured, and executed is less than 98
def total_less_than_98 (x : ℕ) (k : ℕ) (i : ℕ) : Prop := k + i + x < 98

-- Condition 4: Relation between killed and executed
def killed_relation (x : ℕ) (k : ℕ) : Prop := k - 4 = 2 * x

-- The final theorem statement to prove
theorem executed_is_9 : ∃ x, number_killed x = 2 * x + 4 ∧
                       number_injured x = (16 * x) / 3 + 8 ∧
                       total_less_than_98 x (number_killed x) (number_injured x) ∧
                       killed_relation x (number_killed x) ∧
                       x = 9 :=
by
  sorry

end executed_is_9_1_1552


namespace rectangle_difference_length_width_1_1841

theorem rectangle_difference_length_width (x y p d : ℝ) (h1 : x + y = p / 2) (h2 : x^2 + y^2 = d^2) (h3 : x > y) : 
  x - y = (Real.sqrt (8 * d^2 - p^2)) / 2 := sorry

end rectangle_difference_length_width_1_1841


namespace ratio_pentagon_side_length_to_rectangle_width_1_1546

def pentagon_side_length (p : ℕ) (n : ℕ) := p / n
def rectangle_width (p : ℕ) (ratio : ℕ) := p / (2 * (1 + ratio))

theorem ratio_pentagon_side_length_to_rectangle_width :
  pentagon_side_length 60 5 / rectangle_width 80 3 = (6 : ℚ) / 5 :=
by {
  sorry
}

end ratio_pentagon_side_length_to_rectangle_width_1_1546


namespace angles_relation_1_1827

/-- Given angles α and β from two right-angled triangles in a 3x3 grid such that α + β = 90°,
    prove that 2α + β = 90°. -/
theorem angles_relation (α β : ℝ) (h1 : α + β = 90) : 2 * α + β = 90 := by
  sorry

end angles_relation_1_1827


namespace negation_of_all_students_are_punctual_1_1161

variable (Student : Type)
variable (student : Student → Prop)
variable (punctual : Student → Prop)

theorem negation_of_all_students_are_punctual :
  ¬ (∀ x, student x → punctual x) ↔ (∃ x, student x ∧ ¬ punctual x) := by
  sorry

end negation_of_all_students_are_punctual_1_1161


namespace inequality_solution_1_1109

theorem inequality_solution (a x : ℝ) : 
  (x^2 - (a + 1) * x + a) ≤ 0 ↔ 
  (a > 1 → (1 ≤ x ∧ x ≤ a)) ∧ 
  (a = 1 → x = 1) ∧ 
  (a < 1 → (a ≤ x ∧ x ≤ 1)) :=
by 
  sorry

end inequality_solution_1_1109


namespace travel_time_total_1_1403

theorem travel_time_total (dist1 dist2 dist3 speed1 speed2 speed3 : ℝ)
  (h_dist1 : dist1 = 50) (h_dist2 : dist2 = 100) (h_dist3 : dist3 = 150)
  (h_speed1 : speed1 = 50) (h_speed2 : speed2 = 80) (h_speed3 : speed3 = 120) :
  dist1 / speed1 + dist2 / speed2 + dist3 / speed3 = 3.5 :=
by
  sorry

end travel_time_total_1_1403


namespace overall_loss_is_450_1_1005

noncomputable def total_worth_stock : ℝ := 22499.999999999996

noncomputable def selling_price_20_percent_stock (W : ℝ) : ℝ :=
    0.20 * W * 1.10

noncomputable def selling_price_80_percent_stock (W : ℝ) : ℝ :=
    0.80 * W * 0.95

noncomputable def total_selling_price (W : ℝ) : ℝ :=
    selling_price_20_percent_stock W + selling_price_80_percent_stock W

noncomputable def overall_loss (W : ℝ) : ℝ :=
    W - total_selling_price W

theorem overall_loss_is_450 :
  overall_loss total_worth_stock = 450 := by
  sorry

end overall_loss_is_450_1_1005


namespace range_of_f_1_1553

noncomputable def f (x : ℕ) : ℤ := x^2 - 2*x

theorem range_of_f : 
  {y : ℤ | ∃ x ∈ ({0, 1, 2, 3} : Finset ℕ), f x = y} = {-1, 0, 3} :=
by
  sorry

end range_of_f_1_1553


namespace integral_solution_unique_1_1217

theorem integral_solution_unique (a b c : ℤ) : a^2 + b^2 + c^2 = a^2 * b^2 → a = 0 ∧ b = 0 ∧ c = 0 :=
by
  sorry

end integral_solution_unique_1_1217


namespace adah_practiced_total_hours_1_1656

theorem adah_practiced_total_hours :
  let minutes_per_day := 86
  let days_practiced := 2
  let minutes_other_days := 278
  let total_minutes := (minutes_per_day * days_practiced) + minutes_other_days
  let total_hours := total_minutes / 60
  total_hours = 7.5 :=
by
  sorry

end adah_practiced_total_hours_1_1656


namespace fencing_required_1_1925

theorem fencing_required (L W : ℝ) (hL : L = 20) (hA : 20 * W = 60) : 2 * W + L = 26 :=
by
  sorry

end fencing_required_1_1925


namespace fractions_sum_1_1143

theorem fractions_sum (a : ℝ) (h : a ≠ 0) : (1 / a) + (2 / a) = 3 / a := 
by 
  sorry

end fractions_sum_1_1143


namespace lcm_36_225_1_1805

theorem lcm_36_225 : Nat.lcm 36 225 = 900 := by
  -- Defining the factorizations as given
  let fact_36 : 36 = 2^2 * 3^2 := by rfl
  let fact_225 : 225 = 3^2 * 5^2 := by rfl

  -- Indicating what LCM we need to prove
  show Nat.lcm 36 225 = 900

  -- Proof (skipped)
  sorry

end lcm_36_225_1_1805


namespace max_min_x_plus_inv_x_1_1004

-- We're assuming existence of 101 positive numbers with given conditions.
variable {x : ℝ}
variable {y : Fin 100 → ℝ}

-- Conditions given in the problem
def cumulative_sum (x : ℝ) (y : Fin 100 → ℝ) : Prop :=
  0 < x ∧ (∀ i, 0 < y i) ∧ x + (∑ i, y i) = 102 ∧ 1 / x + (∑ i, 1 / y i) = 102

-- The theorem to prove the maximum and minimum value of x + 1/x
theorem max_min_x_plus_inv_x (x : ℝ) (y : Fin 100 → ℝ) (h : cumulative_sum x y) : 
  (x + 1 / x ≤ 405 / 102) ∧ (x + 1 / x ≥ 399 / 102) := 
  sorry

end max_min_x_plus_inv_x_1_1004


namespace find_second_number_1_1722

-- The Lean statement for the given math problem:

theorem find_second_number
  (x y z : ℝ)  -- Represent the three numbers
  (h1 : x = 2 * y)  -- The first number is twice the second
  (h2 : z = (1/3) * x)  -- The third number is one-third of the first
  (h3 : x + y + z = 110)  -- The sum of the three numbers is 110
  : y = 30 :=  -- The second number is 30
sorry

end find_second_number_1_1722


namespace percentage_increase_each_job_1_1031

-- Definitions of original and new amounts for each job as given conditions
def original_first_job : ℝ := 65
def new_first_job : ℝ := 70

def original_second_job : ℝ := 240
def new_second_job : ℝ := 315

def original_third_job : ℝ := 800
def new_third_job : ℝ := 880

-- Proof problem statement
theorem percentage_increase_each_job :
  (new_first_job - original_first_job) / original_first_job * 100 = 7.69 ∧
  (new_second_job - original_second_job) / original_second_job * 100 = 31.25 ∧
  (new_third_job - original_third_job) / original_third_job * 100 = 10 := by
  sorry

end percentage_increase_each_job_1_1031


namespace divisibility_equivalence_distinct_positive_1_1238

variable (a b c : ℕ)

theorem divisibility_equivalence_distinct_positive (h1 : a ≠ b) (h2 : b ≠ c) (h3 : c ≠ a) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  ((a + b + c) ∣ (a^3 * b + b^3 * c + c^3 * a)) ↔ ((a + b + c) ∣ (a * b^3 + b * c^3 + c * a^3)) :=
by sorry

end divisibility_equivalence_distinct_positive_1_1238


namespace john_ate_half_package_1_1079

def fraction_of_package_john_ate (servings : ℕ) (calories_per_serving : ℕ) (calories_consumed : ℕ) : ℚ :=
  calories_consumed / (servings * calories_per_serving : ℚ)

theorem john_ate_half_package (servings : ℕ) (calories_per_serving : ℕ) (calories_consumed : ℕ) 
    (h_servings : servings = 3) (h_calories_per_serving : calories_per_serving = 120) (h_calories_consumed : calories_consumed = 180) :
    fraction_of_package_john_ate servings calories_per_serving calories_consumed = 1 / 2 :=
by
  -- Replace the actual proof with sorry to ensure the statement compiles.
  sorry

end john_ate_half_package_1_1079


namespace sin_range_1_1245

theorem sin_range (x : ℝ) (h : x ∈ Set.Icc (Real.pi / 6) (Real.pi / 2)) : 
  Set.range (fun x => Real.sin x) = Set.Icc (1/2 : ℝ) 1 :=
sorry

end sin_range_1_1245


namespace election_winner_votes_1_1351

theorem election_winner_votes (V : ℝ) : (0.62 * V = 806) → (0.62 * V) - (0.38 * V) = 312 → 0.62 * V = 806 :=
by
  intro hWin hDiff
  exact hWin

end election_winner_votes_1_1351


namespace incorrect_reciprocal_quotient_1_1274

-- Definitions based on problem conditions
def identity_property (x : ℚ) : x * 1 = x := by sorry
def division_property (a b : ℚ) (h : b ≠ 0) : a / b = 0 → a = 0 := by sorry
def additive_inverse_property (x : ℚ) : x * (-1) = -x := by sorry

-- Statement that needs to be proved
theorem incorrect_reciprocal_quotient (a b : ℚ) (h1 : a ≠ 0) (h2 : b = 1 / a) : a / b ≠ 1 :=
by sorry

end incorrect_reciprocal_quotient_1_1274


namespace triangle_side_relation_1_1859

theorem triangle_side_relation 
  (a b c : ℝ) 
  (A : ℝ) 
  (h : b^2 + c^2 = a * ((√3 / 3) * b * c + a)) : 
  a = 2 * √3 * Real.cos A := 
sorry

end triangle_side_relation_1_1859


namespace average_speed_of_car_1_1098

theorem average_speed_of_car : 
  let distance1 := 30
  let speed1 := 60
  let distance2 := 35
  let speed2 := 70
  let distance3 := 36
  let speed3 := 80
  let distance4 := 20
  let speed4 := 55
  let total_distance := distance1 + distance2 + distance3 + distance4
  let time1 := distance1 / speed1
  let time2 := distance2 / speed2
  let time3 := distance3 / speed3
  let time4 := distance4 / speed4
  let total_time := time1 + time2 + time3 + time4
  let average_speed := total_distance / total_time
  average_speed = 66.70 := sorry

end average_speed_of_car_1_1098


namespace line_circle_no_intersection_1_1754

theorem line_circle_no_intersection :
  ∀ (x y : ℝ), (5 * x + 8 * y = 10) → ¬ (x^2 + y^2 = 1) :=
by
  intro x y hline hcirc
  -- Proof omitted
  sorry

end line_circle_no_intersection_1_1754


namespace incorrect_expression_D_1_1939

noncomputable def E : ℝ := sorry
def R : ℕ := sorry
def S : ℕ := sorry
def m : ℕ := sorry
def t : ℕ := sorry

-- E is a repeating decimal
-- R is the non-repeating part of E with m digits
-- S is the repeating part of E with t digits

theorem incorrect_expression_D : ¬ (10^m * (10^t - 1) * E = S * (R - 1)) :=
sorry

end incorrect_expression_D_1_1939


namespace find_fraction_1_1329

variable (N : ℕ) (F : ℚ)
theorem find_fraction (h1 : N = 90) (h2 : 3 + (1/2 : ℚ) * (1/3 : ℚ) * (1/5 : ℚ) * N = F * N) : F = 1 / 15 :=
sorry

end find_fraction_1_1329


namespace simplify_polynomial_1_1855

variable {x : ℝ} -- Assume x is a real number

theorem simplify_polynomial :
  (3 * x^2 + 8 * x - 5) - (2 * x^2 + 6 * x - 15) = x^2 + 2 * x + 10 :=
sorry

end simplify_polynomial_1_1855


namespace find_tan_of_cos_in_4th_quadrant_1_1947

-- Given conditions
variable (α : ℝ) (h1 : Real.cos α = 3/5) (h2 : α > 3*Real.pi/2 ∧ α < 2*Real.pi)

-- Lean statement to prove the question
theorem find_tan_of_cos_in_4th_quadrant : Real.tan α = - (4 / 3) := 
by
  sorry

end find_tan_of_cos_in_4th_quadrant_1_1947


namespace sufficient_but_not_necessary_1_1035

variables {p q : Prop}

theorem sufficient_but_not_necessary :
  (p → q) ∧ (¬q → ¬p) ∧ ¬(q → p) → (¬q → ¬p) ∧ (¬(q → p)) :=
by
  sorry

end sufficient_but_not_necessary_1_1035


namespace triangle_equilateral_if_arithmetic_sequences_1_1395

theorem triangle_equilateral_if_arithmetic_sequences
  (A B C : ℝ) (a b c : ℝ)
  (h_angles : A + B + C = 180)
  (h_angle_seq : ∃ (N : ℝ), A = B - N ∧ C = B + N)
  (h_sides : ∃ (n : ℝ), a = b - n ∧ c = b + n) :
  A = B ∧ B = C ∧ a = b ∧ b = c :=
sorry

end triangle_equilateral_if_arithmetic_sequences_1_1395


namespace proof_problem_1_1626

-- Given conditions: 
variables (a b c d : ℝ)
axiom condition : (2 * a + b) / (b + 2 * c) = (c + 3 * d) / (4 * d + a)

-- Proof problem statement:
theorem proof_problem : (a = c ∨ 3 * a + 4 * b + 5 * c + 6 * d = 0 ∨ (a = c ∧ 3 * a + 4 * b + 5 * c + 6 * d = 0)) :=
by
  sorry

end proof_problem_1_1626


namespace arithmetic_sequence_k_1_1875

theorem arithmetic_sequence_k (d : ℤ) (h_d : d ≠ 0) (a : ℕ → ℤ) 
  (h_seq : ∀ n, a n = 0 + n * d) (h_k : a 21 = a 0 + a 1 + a 2 + a 3 + a 4 + a 5 + a 6):
  21 = 21 :=
by
  -- This would be the problem setup
  -- The proof would go here
  sorry

end arithmetic_sequence_k_1_1875


namespace find_x_for_parallel_vectors_1_1643

-- Definitions for the given conditions
def a : ℝ × ℝ := (4, 2)
def b (x : ℝ) : ℝ × ℝ := (x, 3)
def parallel (u v : ℝ × ℝ) : Prop := u.1 * v.2 = u.2 * v.1

-- The proof statement
theorem find_x_for_parallel_vectors (x : ℝ) (h : parallel a (b x)) : x = 6 :=
  sorry

end find_x_for_parallel_vectors_1_1643


namespace laura_saves_more_with_promotion_A_1_1789

def promotion_A_cost (pair_price : ℕ) : ℕ :=
  let second_pair_price := pair_price / 2
  pair_price + second_pair_price

def promotion_B_cost (pair_price : ℕ) : ℕ :=
  let discount := pair_price * 20 / 100
  pair_price + (pair_price - discount)

def savings (pair_price : ℕ) : ℕ :=
  promotion_B_cost pair_price - promotion_A_cost pair_price

theorem laura_saves_more_with_promotion_A :
  savings 50 = 15 :=
  by
  -- The detailed proof will be added here
  sorry

end laura_saves_more_with_promotion_A_1_1789


namespace min_tip_percentage_1_1278

noncomputable def meal_cost : ℝ := 37.25
noncomputable def total_paid : ℝ := 40.975
noncomputable def tip_percentage (P : ℝ) : Prop := P > 0 ∧ P < 15 ∧ (meal_cost + (P/100) * meal_cost = total_paid)

theorem min_tip_percentage : ∃ P : ℝ, tip_percentage P ∧ P = 10 := by
  sorry

end min_tip_percentage_1_1278


namespace combined_marbles_1_1736

def Rhonda_marbles : ℕ := 80
def Amon_marbles : ℕ := Rhonda_marbles + 55

theorem combined_marbles : Amon_marbles + Rhonda_marbles = 215 :=
by
  sorry

end combined_marbles_1_1736


namespace find_m_value_1_1038

-- Define the quadratic function
def quadratic (x m : ℝ) : ℝ := x^2 - 6 * x + m

-- Define the condition that the quadratic function has a minimum value of 1
def has_minimum_value_of_one (m : ℝ) : Prop := ∃ x : ℝ, quadratic x m = 1

-- The main theorem statement
theorem find_m_value : ∀ m : ℝ, has_minimum_value_of_one m → m = 10 :=
by sorry

end find_m_value_1_1038


namespace digit_theta_1_1124

noncomputable def theta : ℕ := 7

theorem digit_theta (Θ : ℕ) (h1 : 378 / Θ = 40 + Θ + Θ) : Θ = theta :=
by {
  sorry
}

end digit_theta_1_1124


namespace average_production_per_day_for_entire_month_1_1562

-- Definitions based on the conditions
def average_first_25_days := 65
def average_last_5_days := 35
def number_of_days_in_first_period := 25
def number_of_days_in_last_period := 5
def total_days_in_month := 30

-- The goal is to prove that the average production per day for the entire month is 60 TVs/day.
theorem average_production_per_day_for_entire_month :
  (average_first_25_days * number_of_days_in_first_period + 
   average_last_5_days * number_of_days_in_last_period) / total_days_in_month = 60 := 
by
  sorry

end average_production_per_day_for_entire_month_1_1562


namespace warehouse_bins_total_1_1634

theorem warehouse_bins_total (x : ℕ) (h1 : 12 * 20 + x * 15 = 510) : 12 + x = 30 :=
by
  sorry

end warehouse_bins_total_1_1634


namespace abs_eq_1_solution_set_1_1727

theorem abs_eq_1_solution_set (x : ℝ) : (|x| + |x + 1| = 1) ↔ (x ∈ Set.Icc (-1 : ℝ) 0) := by
  sorry

end abs_eq_1_solution_set_1_1727


namespace value_of_t_plus_k_1_1885

noncomputable def f (x t : ℝ) : ℝ := x^3 + (t - 1) * x^2 - 1

theorem value_of_t_plus_k (k t : ℝ)
  (h1 : k ≠ 0)
  (h2 : ∀ x, f x t = 2 * x - 1)
  (h3 : ∃ x₁ x₂, f x₁ t = 2 * x₁ - 1 ∧ f x₂ t = 2 * x₂ - 1) :
  t + k = 7 :=
sorry

end value_of_t_plus_k_1_1885


namespace typing_speed_in_6_minutes_1_1312

theorem typing_speed_in_6_minutes (total_chars : ℕ) (chars_first_minute : ℕ) (chars_last_minute : ℕ) (chars_other_minutes : ℕ) :
  total_chars = 2098 →
  chars_first_minute = 112 →
  chars_last_minute = 97 →
  chars_other_minutes = 1889 →
  (1889 / 6 : ℝ) < 315 → 
  ¬(∀ n, 1 ≤ n ∧ n ≤ 14 - 6 + 1 → chars_other_minutes / 6 ≥ 946) :=
by
  -- Given that analyzing the content, 
  -- proof is skipped here, replace this line with the actual proof.
  sorry

end typing_speed_in_6_minutes_1_1312


namespace blonde_hair_count_1_1002

theorem blonde_hair_count (total_people : ℕ) (percentage_blonde : ℕ) (h_total : total_people = 600) (h_percentage : percentage_blonde = 30) : 
  (percentage_blonde * total_people / 100) = 180 :=
by
  -- Conditions from the problem
  have h1 : total_people = 600 := h_total
  have h2 : percentage_blonde = 30 := h_percentage
  -- Start the proof
  sorry

end blonde_hair_count_1_1002


namespace range_of_a_1_1221

noncomputable def f (x : ℝ) : ℝ := Real.exp x + 2 * x

noncomputable def f' (x : ℝ) : ℝ := Real.exp x + 2

theorem range_of_a (a : ℝ) : (∀ x : ℝ, f' x ≥ a) → (a ≤ 2) :=
by
  sorry

end range_of_a_1_1221


namespace winner_collected_1_1853

variable (M : ℕ)
variable (last_year_rate this_year_rate : ℝ)
variable (extra_miles : ℕ)
variable (money_collected_last_year money_collected_this_year : ℝ)

axiom rate_last_year : last_year_rate = 4
axiom rate_this_year : this_year_rate = 2.75
axiom extra_miles_eq : extra_miles = 5

noncomputable def money_eq (M : ℕ) : ℝ :=
  last_year_rate * M

theorem winner_collected :
  ∃ M : ℕ, money_eq M = 44 :=
by
  sorry

end winner_collected_1_1853


namespace bridge_construction_1_1001

-- Definitions used in the Lean statement based on conditions.
def rate (workers : ℕ) (days : ℕ) : ℚ := 1 / (workers * days)

-- The problem statement: prove that if 60 workers working together can build the bridge in 3 days, 
-- then 120 workers will take 1.5 days to build the bridge.
theorem bridge_construction (t : ℚ) : 
  (rate 60 3) * 120 * t = 1 → t = 1.5 := by
  sorry

end bridge_construction_1_1001


namespace sum_of_consecutive_integers_between_ln20_1_1381

theorem sum_of_consecutive_integers_between_ln20 : ∃ a b : ℤ, a < b ∧ b = a + 1 ∧ 1 ≤ a ∧ a + 1 ≤ 3 ∧ (a + b = 4) :=
by
  sorry

end sum_of_consecutive_integers_between_ln20_1_1381


namespace units_digit_fraction_1_1950

-- Given conditions
def numerator : ℕ := 30 * 31 * 32 * 33 * 34 * 35
def denominator : ℕ := 1500
def simplified_fraction : ℕ := 2^5 * 3 * 31 * 33 * 17 * 7

-- Statement of the proof goal
theorem units_digit_fraction :
  (simplified_fraction) % 10 = 2 := by
  sorry

end units_digit_fraction_1_1950


namespace sum_of_powers_eq_zero_1_1989

theorem sum_of_powers_eq_zero
  (a b c : ℝ)
  (n : ℝ)
  (h1 : a + b + c = 0)
  (h2 : a^3 + b^3 + c^3 = 0) :
  a^(2* ⌊n⌋ + 1) + b^(2* ⌊n⌋ + 1) + c^(2* ⌊n⌋ + 1) = 0 := by
  sorry

end sum_of_powers_eq_zero_1_1989


namespace mn_eq_one_1_1103

noncomputable def f (x : ℝ) : ℝ := |Real.log x / Real.log 2|

variables (m n : ℝ) (hmn : m < n) (hm_pos : 0 < m) (hn_pos : 0 < n) (hmn_equal : f m = f n)

theorem mn_eq_one : m * n = 1 := by
  sorry

end mn_eq_one_1_1103


namespace flower_team_participation_1_1298

-- Definitions based on the conditions in the problem
def num_rows : ℕ := 60
def first_row_people : ℕ := 40
def people_increment : ℕ := 1

-- Statement to be proved in Lean
theorem flower_team_participation (x : ℕ) (hx : 1 ≤ x ∧ x ≤ num_rows) : 
  ∃ y : ℕ, y = first_row_people - people_increment + x :=
by
  -- Placeholder for the proof
  sorry

end flower_team_participation_1_1298


namespace correct_option_is_c_1_1782

variable {x y : ℕ}

theorem correct_option_is_c (hx : (x^2)^3 = x^6) :
  (∀ x : ℕ, x * x^2 ≠ x^2) →
  (∀ x y : ℕ, (x + y)^2 ≠ x^2 + y^2) →
  (∃ x : ℕ, x^2 + x^2 ≠ x^4) →
  (x^2)^3 = x^6 :=
by
  intros h1 h2 h3
  exact hx

end correct_option_is_c_1_1782


namespace boats_left_1_1292

def initial_boats : ℕ := 30
def percentage_eaten_by_fish : ℕ := 20
def boats_shot_with_arrows : ℕ := 2
def boats_blown_by_wind : ℕ := 3
def boats_sank : ℕ := 4

def boats_eaten_by_fish : ℕ := (initial_boats * percentage_eaten_by_fish) / 100

theorem boats_left : initial_boats - boats_eaten_by_fish - boats_shot_with_arrows - boats_blown_by_wind - boats_sank = 15 := by
  sorry

end boats_left_1_1292


namespace jill_spent_more_1_1623

def cost_per_ball_red : ℝ := 1.50
def cost_per_ball_yellow : ℝ := 1.25
def cost_per_ball_blue : ℝ := 1.00

def packs_red : ℕ := 5
def packs_yellow : ℕ := 4
def packs_blue : ℕ := 3

def balls_per_pack_red : ℕ := 18
def balls_per_pack_yellow : ℕ := 15
def balls_per_pack_blue : ℕ := 12

def balls_red : ℕ := packs_red * balls_per_pack_red
def balls_yellow : ℕ := packs_yellow * balls_per_pack_yellow
def balls_blue : ℕ := packs_blue * balls_per_pack_blue

def cost_red : ℝ := balls_red * cost_per_ball_red
def cost_yellow : ℝ := balls_yellow * cost_per_ball_yellow
def cost_blue : ℝ := balls_blue * cost_per_ball_blue

def combined_cost_yellow_blue : ℝ := cost_yellow + cost_blue

theorem jill_spent_more : cost_red = combined_cost_yellow_blue + 24 := by
  sorry

end jill_spent_more_1_1623


namespace change_is_13_82_1_1411

def sandwich_cost : ℝ := 5
def num_sandwiches : ℕ := 3
def discount_rate : ℝ := 0.10
def tax_rate : ℝ := 0.05
def payment : ℝ := 20 + 5 + 3

def total_cost_before_discount : ℝ := num_sandwiches * sandwich_cost
def discount_amount : ℝ := total_cost_before_discount * discount_rate
def discounted_cost : ℝ := total_cost_before_discount - discount_amount
def tax_amount : ℝ := discounted_cost * tax_rate
def total_cost_after_tax : ℝ := discounted_cost + tax_amount

def change (payment total_cost : ℝ) : ℝ := payment - total_cost

theorem change_is_13_82 : change payment total_cost_after_tax = 13.82 := 
by
  -- Proof will be provided here
  sorry

end change_is_13_82_1_1411


namespace hexagon_interior_angles_1_1055

theorem hexagon_interior_angles
  (A B C D E F : ℝ)
  (hA : A = 90)
  (hB : B = 120)
  (hCD : C = D)
  (hE : E = 2 * C + 20)
  (hF : F = 60)
  (hsum : A + B + C + D + E + F = 720) :
  D = 107.5 := 
by
  -- formal proof required here
  sorry

end hexagon_interior_angles_1_1055


namespace negation_of_p_1_1240

variable (x : ℝ)

-- Define the original proposition p
def p := ∀ x, x^2 < 1 → x < 1

-- Define the negation of p
def neg_p := ∃ x₀, x₀^2 ≥ 1 ∧ x₀ < 1

-- State the theorem that negates p
theorem negation_of_p : ¬ p ↔ neg_p :=
by
  sorry

end negation_of_p_1_1240


namespace total_boys_went_down_slide_1_1062

-- Definitions according to the conditions given
def boys_went_down_slide1 : ℕ := 22
def boys_went_down_slide2 : ℕ := 13

-- The statement to be proved
theorem total_boys_went_down_slide : boys_went_down_slide1 + boys_went_down_slide2 = 35 := 
by 
  sorry

end total_boys_went_down_slide_1_1062


namespace min_ratio_ax_1_1520

theorem min_ratio_ax (a x y : ℕ) (ha : a > 100) (hx : x > 100) (hy : y > 100) 
: y^2 - 1 = a^2 * (x^2 - 1) → ∃ (k : ℕ), k = 2 ∧ (a = k * x) := 
sorry

end min_ratio_ax_1_1520


namespace multiply_identity_1_1940

variable (x y : ℝ)

theorem multiply_identity :
  (3 * x ^ 4 - 2 * y ^ 3) * (9 * x ^ 8 + 6 * x ^ 4 * y ^ 3 + 4 * y ^ 6) = 27 * x ^ 12 - 8 * y ^ 9 := by
  sorry

end multiply_identity_1_1940


namespace find_x_1_1898

def op (a b : ℕ) : ℕ := a * b - b + b ^ 2

theorem find_x (x : ℕ) : (∃ x : ℕ, op x 8 = 80) :=
  sorry

end find_x_1_1898


namespace arrange_letters_1_1621

-- Definitions based on conditions
def total_letters := 6
def identical_bs := 2 -- Number of B's that are identical
def distinct_as := 3  -- Number of A's that are distinct
def distinct_ns := 1  -- Number of N's that are distinct

-- Now formulate the proof statement
theorem arrange_letters :
    (Nat.factorial total_letters) / (Nat.factorial identical_bs) = 360 :=
by
  sorry

end arrange_letters_1_1621


namespace div_by_133_1_1785

theorem div_by_133 (n : ℕ) : 133 ∣ 11^(n+2) + 12^(2*n+1) :=
by sorry

end div_by_133_1_1785


namespace problem_solution_1_1778

theorem problem_solution :
  (- (5 : ℚ) / 12) ^ 2023 * (12 / 5) ^ 2023 = -1 := 
by
  sorry

end problem_solution_1_1778


namespace value_of_X_1_1138

def M : ℕ := 2024 / 4
def N : ℕ := M / 2
def X : ℕ := M + N

theorem value_of_X : X = 759 := by
  sorry

end value_of_X_1_1138


namespace proof_statement_1_1447

open Classical

variable (Person : Type) (Nationality : Type) (Occupation : Type)

variable (A B C D : Person)
variable (UnitedKingdom UnitedStates Germany France : Nationality)
variable (Doctor Teacher : Occupation)

variable (nationality : Person → Nationality)
variable (occupation : Person → Occupation)
variable (can_swim : Person → Prop)
variable (play_sports_together : Person → Person → Prop)

noncomputable def proof :=
  (nationality A = UnitedKingdom ∧ nationality D = Germany)

axiom condition1 : occupation A = Doctor ∧ ∃ x : Person, nationality x = UnitedStates ∧ occupation x = Doctor
axiom condition2 : occupation B = Teacher ∧ ∃ x : Person, nationality x = Germany ∧ occupation x = Teacher 
axiom condition3 : can_swim C ∧ ∀ x : Person, nationality x = Germany → ¬ can_swim x
axiom condition4 : ∃ x : Person, nationality x = France ∧ play_sports_together A x

theorem proof_statement : 
  (nationality A = UnitedKingdom ∧ nationality D = Germany) :=
by {
  sorry
}

end proof_statement_1_1447


namespace sellingPrice_is_459_1_1967

-- Definitions based on conditions
def costPrice : ℝ := 540
def markupPercentage : ℝ := 0.15
def discountPercentage : ℝ := 0.2608695652173913

-- Calculating the marked price based on the given conditions
def markedPrice (cp : ℝ) (markup : ℝ) : ℝ := cp + (markup * cp)

-- Calculating the discount amount based on the marked price and the discount percentage
def discount (mp : ℝ) (discountPct : ℝ) : ℝ := discountPct * mp

-- Calculating the selling price
def sellingPrice (mp : ℝ) (discountAmt : ℝ) : ℝ := mp - discountAmt

-- Stating the final proof problem
theorem sellingPrice_is_459 :
  sellingPrice (markedPrice costPrice markupPercentage) (discount (markedPrice costPrice markupPercentage) discountPercentage) = 459 :=
by
  sorry

end sellingPrice_is_459_1_1967


namespace union_complement_eq_1_1801

def U : Set Nat := {0, 1, 2, 4, 6, 8}
def M : Set Nat := {0, 4, 6}
def N : Set Nat := {0, 1, 6}
def complement (u : Set α) (s : Set α) : Set α := {x ∈ u | x ∉ s}

theorem union_complement_eq :
  M ∪ (complement U N) = {0, 2, 4, 6, 8} :=
by sorry

end union_complement_eq_1_1801


namespace fixed_point_always_1_1871

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := 2^x + Real.logb a (x + 1) + 3

theorem fixed_point_always (a : ℝ) (h : a > 0 ∧ a ≠ 1) : f 0 a = 4 :=
by
  sorry

end fixed_point_always_1_1871


namespace project_selection_probability_1_1482

/-- Each employee can randomly select one project from four optional assessment projects. -/
def employees : ℕ := 4

def projects : ℕ := 4

def total_events (e : ℕ) (p : ℕ) : ℕ := p^e

def choose_exactly_one_project_not_selected_probability (e : ℕ) (p : ℕ) : ℚ :=
  (Nat.choose p 2 * Nat.factorial 3) / (p^e : ℚ)

theorem project_selection_probability :
  choose_exactly_one_project_not_selected_probability employees projects = 9 / 16 :=
by
  sorry

end project_selection_probability_1_1482


namespace cost_per_pack_1_1934

variable (total_amount : ℕ) (number_of_packs : ℕ)

theorem cost_per_pack (h1 : total_amount = 132) (h2 : number_of_packs = 11) : 
  total_amount / number_of_packs = 12 := by
  sorry

end cost_per_pack_1_1934


namespace find_sum_abc_1_1265

-- Define the real numbers a, b, c
variables {a b c : ℝ}

-- Define the conditions that a, b, c are positive reals.
axiom ha_pos : 0 < a
axiom hb_pos : 0 < b
axiom hc_pos : 0 < c

-- Define the condition that a^2 + b^2 + c^2 = 989
axiom habc_sq : a^2 + b^2 + c^2 = 989

-- Define the condition that (a+b)^2 + (b+c)^2 + (c+a)^2 = 2013
axiom habc_sq_sum : (a+b)^2 + (b+c)^2 + (c+a)^2 = 2013

-- The proposition to be proven
theorem find_sum_abc : a + b + c = 32 :=
by
  -- ...(proof goes here)
  sorry

end find_sum_abc_1_1265


namespace bill_toys_1_1620

variable (B H : ℕ)

theorem bill_toys (h1 : H = B / 2 + 9) (h2 : B + H = 99) : B = 60 := by
  sorry

end bill_toys_1_1620


namespace x_add_one_greater_than_x_1_1713

theorem x_add_one_greater_than_x (x : ℝ) : x + 1 > x :=
by
  sorry

end x_add_one_greater_than_x_1_1713


namespace inscribed_rectangle_area_1_1149

variables (b h x : ℝ)
variables (h_isosceles_triangle : b > 0 ∧ h > 0 ∧ x > 0 ∧ x < h)

noncomputable def rectangle_area (b h x : ℝ) : ℝ :=
  (b * x / h) * (h - x)

theorem inscribed_rectangle_area :
  rectangle_area b h x = (b * x / h) * (h - x) :=
by
  unfold rectangle_area
  sorry

end inscribed_rectangle_area_1_1149


namespace only_solution_1_1881

def pythagorean_euler_theorem (p r : ℕ) : Prop :=
  ∃ (p r : ℕ), Nat.Prime p ∧ r > 0 ∧ (∑ i in Finset.range (r + 1), (p + i)^p) = (p + r + 1)^p

theorem only_solution (p r : ℕ) : pythagorean_euler_theorem p r ↔ p = 3 ∧ r = 2 :=
by
  sorry

end only_solution_1_1881


namespace dino_dolls_count_1_1966

theorem dino_dolls_count (T : ℝ) (H : 0.7 * T = 140) : T = 200 :=
sorry

end dino_dolls_count_1_1966


namespace common_ratio_geometric_sequence_1_1181

theorem common_ratio_geometric_sequence (a b c d : ℤ) (h1 : a = 10) (h2 : b = -20) (h3 : c = 40) (h4 : d = -80) :
    b / a = -2 ∧ c = b * -2 ∧ d = c * -2 := by
  sorry

end common_ratio_geometric_sequence_1_1181


namespace work_completion_days_1_1076

theorem work_completion_days (A_days B_days : ℕ) (hA : A_days = 3) (hB : B_days = 6) : 
  (1 / ((1 / (A_days : ℚ)) + (1 / (B_days : ℚ)))) = 2 := 
by
  sorry

end work_completion_days_1_1076


namespace maximize_k_1_1557

open Real

theorem maximize_k (x y : ℝ) (h₁ : 0 < x) (h₂ : 0 < y) (h₃ : log x + log y = 0)
  (h₄ : ∀ x y : ℝ, 0 < x → 0 < y → k * (x + 2 * y) ≤ x^2 + 4 * y^2) : k ≤ sqrt 2 :=
sorry

end maximize_k_1_1557


namespace cost_of_large_fries_1_1910

noncomputable def cost_of_cheeseburger : ℝ := 3.65
noncomputable def cost_of_milkshake : ℝ := 2
noncomputable def cost_of_coke : ℝ := 1
noncomputable def cost_of_cookie : ℝ := 0.5
noncomputable def tax : ℝ := 0.2
noncomputable def toby_initial_amount : ℝ := 15
noncomputable def toby_remaining_amount : ℝ := 7
noncomputable def split_bill : ℝ := 2

theorem cost_of_large_fries : 
  let total_meal_cost := (split_bill * (toby_initial_amount - toby_remaining_amount))
  let total_cost_so_far := (2 * cost_of_cheeseburger) + cost_of_milkshake + cost_of_coke + (3 * cost_of_cookie) + tax
  total_meal_cost - total_cost_so_far = 4 := 
by
  sorry

end cost_of_large_fries_1_1910


namespace triangle_formation_and_acuteness_1_1502

variables {a b c : ℝ} {k n : ℕ}

theorem triangle_formation_and_acuteness (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (hn : 2 ≤ n) (hk : k < n) (hp : a^n + b^n = c^n) : 
  (a^k + b^k > c^k ∧ b^k + c^k > a^k ∧ c^k + a^k > b^k) ∧ (k < n / 2 → (a^k)^2 + (b^k)^2 > (c^k)^2) :=
sorry

end triangle_formation_and_acuteness_1_1502


namespace vertex_coordinates_quadratic_through_point_a_less_than_neg_2_fifth_1_1166

noncomputable def quadratic_function (a : ℝ) (x : ℝ) : ℝ := (x + 1) * (a * x + 2 * a + 2)

theorem vertex_coordinates (a : ℝ) (H : a = 1) : 
    (∃ v_x v_y : ℝ, quadratic_function a v_x = v_y ∧ v_x = -5 / 2 ∧ v_y = -9 / 4) := 
by {
    sorry
}

theorem quadratic_through_point : 
    (∃ a : ℝ, (quadratic_function a 0 = -2) ∧ (∀ x, quadratic_function a x = -2 * (x + 1)^2)) := 
by {
    sorry
}

theorem a_less_than_neg_2_fifth 
  (x1 x2 y1 y2 a : ℝ) (H1 : x1 + x2 = 2) (H2 : x1 < x2) (H3 : y1 > y2) 
  (Hfunc : ∀ x, quadratic_function (a * x + 2 * a + 2) (x + 1) = quadratic_function x y) :
    a < -2 / 5 := 
by {
    sorry
}

end vertex_coordinates_quadratic_through_point_a_less_than_neg_2_fifth_1_1166


namespace direct_proportion_1_1797

theorem direct_proportion : 
  ∃ k, (∀ x, y = k * x) ↔ (y = -2 * x) :=
by
  sorry

end direct_proportion_1_1797


namespace root_expression_value_1_1716

theorem root_expression_value 
  (m : ℝ) 
  (h : 2 * m^2 - 3 * m - 1 = 0) : 
  6 * m^2 - 9 * m + 2021 = 2024 := 
by 
  sorry

end root_expression_value_1_1716


namespace expected_groups_1_1295

/-- Given a sequence of k zeros and m ones arranged in random order and divided into 
alternating groups of identical digits, the expected value of the total number of 
groups is 1 + 2 * k * m / (k + m). -/
theorem expected_groups (k m : ℕ) (h_pos : k + m > 0) :
  let total_groups := 1 + 2 * k * m / (k + m)
  total_groups = (1 + 2 * k * m / (k + m)) := by
  sorry

end expected_groups_1_1295


namespace carpet_needed_for_room_1_1380

theorem carpet_needed_for_room
  (length_feet : ℕ) (width_feet : ℕ)
  (area_conversion_factor : ℕ)
  (length_given : length_feet = 12)
  (width_given : width_feet = 6)
  (conversion_given : area_conversion_factor = 9) :
  (length_feet * width_feet) / area_conversion_factor = 8 := 
by
  sorry

end carpet_needed_for_room_1_1380


namespace problem1_problem2_1_1107

-- Define the conditions
variable (a x : ℝ)
variable (h_gt_zero : x > 0) (a_gt_zero : a > 0)

-- Problem 1: Prove that 0 < x ≤ 300
theorem problem1 (h: 12 * (500 - x) * (1 + 0.005 * x) ≥ 12 * 500) : 0 < x ∧ x ≤ 300 := 
sorry

-- Problem 2: Prove that 0 < a ≤ 5.5 given the conditions
theorem problem2 (h1 : 12 * (a - 13 / 1000 * x) * x ≤ 12 * (500 - x) * (1 + 0.005 * x))
                (h2 : x = 250) : 0 < a ∧ a ≤ 5.5 := 
sorry

end problem1_problem2_1_1107


namespace intersection_A_B_1_1317

-- Define the sets A and B based on given conditions
def A : Set ℝ := { x | x^2 ≤ 1 }
def B : Set ℝ := { x | (x - 2) / x ≤ 0 }

-- State the proof problem
theorem intersection_A_B : A ∩ B = { x | 0 < x ∧ x ≤ 1 } :=
by
  sorry

end intersection_A_B_1_1317


namespace seq_is_geometric_from_second_1_1619

namespace sequence_problem

-- Definitions and conditions
def S : ℕ → ℕ
| 0 => 0
| 1 => 1
| 2 => 2
| (n + 1) => 3 * S n - 2 * S (n - 1)

-- Recursive definition for sum of sequence terms
axiom S_rec_relation (n : ℕ) (h : n ≥ 2) : 
  S (n + 1) - 3 * S n + 2 * S (n - 1) = 0

-- Prove the sequence is geometric from the second term
theorem seq_is_geometric_from_second :
  ∃ (a : ℕ → ℕ), (∀ n ≥ 2, a (n + 1) = 2 * a n) ∧ 
  (a 1 = 1) ∧ 
  (a 2 = 1) :=
by
  sorry

end sequence_problem

end seq_is_geometric_from_second_1_1619


namespace janets_total_pockets_1_1906

-- Define the total number of dresses
def totalDresses : ℕ := 36

-- Define the dresses with pockets
def dressesWithPockets : ℕ := totalDresses / 2

-- Define the dresses without pockets
def dressesWithoutPockets : ℕ := totalDresses - dressesWithPockets

-- Define the dresses with one hidden pocket
def dressesWithOneHiddenPocket : ℕ := (40 * dressesWithoutPockets) / 100

-- Define the dresses with 2 pockets
def dressesWithTwoPockets : ℕ := dressesWithPockets / 3

-- Define the dresses with 3 pockets
def dressesWithThreePockets : ℕ := dressesWithPockets / 4

-- Define the dresses with 4 pockets
def dressesWithFourPockets : ℕ := dressesWithPockets - dressesWithTwoPockets - dressesWithThreePockets

-- Calculate the total number of pockets
def totalPockets : ℕ := 
  2 * dressesWithTwoPockets + 
  3 * dressesWithThreePockets + 
  4 * dressesWithFourPockets + 
  dressesWithOneHiddenPocket

-- The theorem to prove the total number of pockets
theorem janets_total_pockets : totalPockets = 63 :=
  by
    -- Proof is omitted, use 'sorry'
    sorry

end janets_total_pockets_1_1906


namespace ways_to_divide_day_1_1788

theorem ways_to_divide_day : 
  ∃ nm_count: ℕ, nm_count = 72 ∧ ∀ n m: ℕ, 0 < n ∧ 0 < m ∧ n * m = 72000 → 
  ∃ nm_pairs: ℕ, nm_pairs = 72 * 2 :=
sorry

end ways_to_divide_day_1_1788


namespace krista_driving_hours_each_day_1_1175

-- Define the conditions as constants
def road_trip_days : ℕ := 3
def jade_hours_per_day : ℕ := 8
def total_hours : ℕ := 42

-- Define the function to calculate Krista's hours per day
noncomputable def krista_hours_per_day : ℕ :=
  (total_hours - road_trip_days * jade_hours_per_day) / road_trip_days

-- State the theorem to prove Krista drove 6 hours each day
theorem krista_driving_hours_each_day : krista_hours_per_day = 6 := by
  sorry

end krista_driving_hours_each_day_1_1175


namespace value_of_a_1_1817

-- Define the three lines as predicates
def line1 (x y : ℝ) : Prop := x + y = 1
def line2 (x y : ℝ) : Prop := x - y = 1
def line3 (a x y : ℝ) : Prop := a * x + y = 1

-- Define the condition that the lines do not form a triangle
def lines_do_not_form_triangle (a x y : ℝ) : Prop :=
  (∀ x y, line1 x y → ¬line3 a x y) ∨
  (∀ x y, line2 x y → ¬line3 a x y) ∨
  (a = 1)

theorem value_of_a (a : ℝ) :
  (¬ ∃ x y, line1 x y ∧ line2 x y ∧ line3 a x y) →
  lines_do_not_form_triangle a 1 0 →
  a = -1 :=
by
  intro h1 h2
  sorry

end value_of_a_1_1817


namespace range_of_m_1_1738

noncomputable def point := (ℝ × ℝ)
noncomputable def P : point := (-1, 1)
noncomputable def Q : point := (2, 2)
noncomputable def M : point := (0, -1)
noncomputable def line_eq (m : ℝ) := ∀ p : point, p.1 + m * p.2 + m = 0

theorem range_of_m (m : ℝ) (l : line_eq m) : -3 < m ∧ m < -2/3 := 
by
  sorry

end range_of_m_1_1738


namespace polynomial_divisibility_1_1016

theorem polynomial_divisibility (m : ℤ) : (3 * (-2)^2 + 5 * (-2) + m = 0) ↔ (m = -2) :=
by
  sorry

end polynomial_divisibility_1_1016


namespace problem_statement_1_1490

noncomputable def f (x : ℝ) : ℝ := (2 * x + 3) / (x + 2)

def S : Set ℝ := {y | ∃ x ≥ 0, y = f x}

theorem problem_statement :
  (∀ y ∈ S, y ≤ 2) ∧ (¬ (2 ∈ S)) ∧ (∀ y ∈ S, y ≥ 3 / 2) ∧ (3 / 2 ∈ S) :=
by
  sorry

end problem_statement_1_1490


namespace box_volume_1_1036

theorem box_volume (l w h : ℝ)
  (A1 : l * w = 30)
  (A2 : w * h = 20)
  (A3 : l * h = 12) :
  l * w * h = 60 :=
sorry

end box_volume_1_1036


namespace rectangle_ratio_1_1761

theorem rectangle_ratio (t a b : ℝ) (h₀ : b = 2 * a) (h₁ : (t + 2 * a) ^ 2 = 3 * t ^ 2) : b / a = 2 :=
by
  sorry

end rectangle_ratio_1_1761


namespace euler_totient_bound_1_1714

theorem euler_totient_bound (n k : ℕ) (h1 : 0 < n) (h2 : 0 < k) (h3 : (Nat.totient^[k]) n = 1) :
  n ≤ 3^k :=
sorry

end euler_totient_bound_1_1714


namespace f_value_at_4_1_1895

def f : ℝ → ℝ := sorry  -- Define f as a function from ℝ to ℝ

-- Specify the condition that f satisfies for all real numbers x
axiom f_condition (x : ℝ) : f (2^x) + x * f (2^(-x)) = 3

-- Statement to be proven: f(4) = -3
theorem f_value_at_4 : f 4 = -3 :=
by {
  -- Proof goes here
  sorry
}

end f_value_at_4_1_1895


namespace avg_speed_is_40_1_1850

noncomputable def average_speed (x : ℝ) : ℝ :=
  let time1 := x / 40
  let time2 := 2 * x / 20
  let total_time := time1 + time2
  let total_distance := 5 * x
  total_distance / total_time

theorem avg_speed_is_40 (x : ℝ) (hx : x > 0) :
  average_speed x = 40 := by
  sorry

end avg_speed_is_40_1_1850


namespace linear_function_passing_origin_1_1478

theorem linear_function_passing_origin (m : ℝ) :
  (∃ (y x : ℝ), y = -2 * x + (m - 5) ∧ y = 0 ∧ x = 0) → m = 5 :=
by
  sorry

end linear_function_passing_origin_1_1478


namespace binomial_divisible_by_prime_1_1758

-- Define the conditions: p is prime and 0 < k < p
variables (p k : ℕ)
variable (hp : Nat.Prime p)
variable (hk : 0 < k ∧ k < p)

-- State that the binomial coefficient \(\binom{p}{k}\) is divisible by \( p \)
theorem binomial_divisible_by_prime
  (p k : ℕ) (hp : Nat.Prime p) (hk : 0 < k ∧ k < p) :
  p ∣ Nat.choose p k :=
by
  sorry

end binomial_divisible_by_prime_1_1758


namespace sqrt_domain_1_1564

def inequality_holds (x : ℝ) : Prop := x + 5 ≥ 0

theorem sqrt_domain (x : ℝ) : inequality_holds x ↔ x ≥ -5 := by
  sorry

end sqrt_domain_1_1564


namespace solve_for_r_1_1235

theorem solve_for_r (r : ℝ) (h: (r + 9) / (r - 3) = (r - 2) / (r + 5)) : r = -39 / 19 :=
sorry

end solve_for_r_1_1235


namespace no_intersection_of_sets_1_1264

noncomputable def A (a b x y : ℝ) :=
  a * (Real.sin x + Real.sin y) + (b - 1) * (Real.cos x + Real.cos y) = 0

noncomputable def B (a b x y : ℝ) :=
  (b + 1) * Real.sin (x + y) - a * Real.cos (x + y) = a

noncomputable def C (a b : ℝ) :=
  ∀ z : ℝ, z^2 - 2 * (a - b) * z + (a + b)^2 - 2 > 0

theorem no_intersection_of_sets (a b x y : ℝ) (h1 : 0 < x) (h2 : x < Real.pi / 2) (h3 : 0 < y) (h4 : y < Real.pi / 2) :
  (C a b) → ¬(∃ x y, A a b x y ∧ B a b x y) :=
by 
  sorry

end no_intersection_of_sets_1_1264


namespace find_number_of_rabbits_1_1216

def total_heads (R P : ℕ) : ℕ := R + P
def total_legs (R P : ℕ) : ℕ := 4 * R + 2 * P

theorem find_number_of_rabbits (R P : ℕ)
  (h1 : total_heads R P = 60)
  (h2 : total_legs R P = 192) :
  R = 36 := by
  sorry

end find_number_of_rabbits_1_1216


namespace certain_number_is_1_1_1550

theorem certain_number_is_1 (z : ℕ) (hz : z % 4 = 0) :
  ∃ n : ℕ, (z * (6 + z) + n) % 2 = 1 ∧ n = 1 :=
by
  sorry

end certain_number_is_1_1_1550


namespace number_of_children_1_1293

variables (n : ℕ) (y : ℕ) (d : ℕ)

def sum_of_ages (n : ℕ) (y : ℕ) (d : ℕ) : ℕ :=
  n * y + d * (n * (n - 1) / 2)

theorem number_of_children (H1 : sum_of_ages n 6 3 = 60) : n = 6 :=
by {
  sorry
}

end number_of_children_1_1293


namespace watermelon_cost_100_1_1391

variable (a b : ℕ) -- costs of one watermelon and one melon respectively
variable (x : ℕ) -- number of watermelons in the container

theorem watermelon_cost_100 :
  (∀ x, (1 : ℚ) = x / 160 + (150 - x) / 120 ∧ 120 * a = 30 * b ∧ 120 * a + 30 * b = 24000 ∧ x = 120) →
  a = 100 :=
by
  intro h
  sorry

end watermelon_cost_100_1_1391


namespace small_triangle_perimeter_1_1244

theorem small_triangle_perimeter (P : ℕ) (P₁ : ℕ) (P₂ : ℕ) (P₃ : ℕ)
  (h₁ : P = 11) (h₂ : P₁ = 5) (h₃ : P₂ = 7) (h₄ : P₃ = 9) :
  (P₁ + P₂ + P₃) - P = 10 :=
by
  sorry

end small_triangle_perimeter_1_1244


namespace remainder_91_pow_91_mod_100_1_1356

-- Definitions
def large_power_mod (a b n : ℕ) : ℕ :=
  (a^b) % n

-- Statement
theorem remainder_91_pow_91_mod_100 : large_power_mod 91 91 100 = 91 :=
by
  sorry

end remainder_91_pow_91_mod_100_1_1356


namespace degree_product_1_1383

-- Define the degrees of the polynomials p and q
def degree_p : ℕ := 3
def degree_q : ℕ := 4

-- Define the functions p(x) and q(x) as polynomials and their respective degrees
axiom degree_p_definition (p : Polynomial ℝ) : p.degree = degree_p
axiom degree_q_definition (q : Polynomial ℝ) : q.degree = degree_q

-- Define the degree of the product p(x^2) * q(x^4)
noncomputable def degree_p_x2_q_x4 (p q : Polynomial ℝ) : ℕ :=
  2 * degree_p + 4 * degree_q

-- Prove that the degree of p(x^2) * q(x^4) is 22
theorem degree_product (p q : Polynomial ℝ) (hp : p.degree = degree_p) (hq : q.degree = degree_q) :
  degree_p_x2_q_x4 p q = 22 :=
by
  sorry

end degree_product_1_1383


namespace alice_wins_chomp_1_1718

def symmetrical_strategy (n : ℕ) : Prop :=
  ∃ strategy : (ℕ × ℕ) → (ℕ × ℕ), 
  (∀ turn : ℕ × ℕ, 
    strategy turn = 
      if turn = (1,1) then (1,1)
      else if turn.fst = 2 ∧ turn.snd = 2 then (2,2)
      else if turn.fst = 1 then (turn.snd, 1)
      else (1, turn.fst)) 

theorem alice_wins_chomp (n : ℕ) (h : 1 ≤ n) : 
  symmetrical_strategy n := 
sorry

end alice_wins_chomp_1_1718


namespace area_square_given_diagonal_1_1458

theorem area_square_given_diagonal (d : ℝ) (h : d = 16) : (∃ A : ℝ, A = 128) :=
by 
  sorry

end area_square_given_diagonal_1_1458


namespace victor_earnings_1_1991

def hourly_wage := 6 -- dollars per hour
def hours_monday := 5 -- hours
def hours_tuesday := 5 -- hours

theorem victor_earnings : (hourly_wage * (hours_monday + hours_tuesday)) = 60 :=
by
  sorry

end victor_earnings_1_1991


namespace coffee_mix_price_per_pound_1_1599

-- Definitions based on conditions
def total_weight : ℝ := 100
def columbian_price_per_pound : ℝ := 8.75
def brazilian_price_per_pound : ℝ := 3.75
def columbian_weight : ℝ := 52
def brazilian_weight : ℝ := total_weight - columbian_weight

-- Goal to prove
theorem coffee_mix_price_per_pound :
  (columbian_weight * columbian_price_per_pound + brazilian_weight * brazilian_price_per_pound) / total_weight = 6.35 :=
by
  sorry

end coffee_mix_price_per_pound_1_1599


namespace prime_quadratic_root_range_1_1358

theorem prime_quadratic_root_range (p : ℕ) (hprime : Prime p) 
  (hroots : ∃ x1 x2 : ℤ, x1 * x2 = -580 * p ∧ x1 + x2 = p) : 20 < p ∧ p < 30 :=
by
  sorry

end prime_quadratic_root_range_1_1358


namespace taller_tree_height_1_1943

-- Definitions and Variables
variables (h : ℝ)

-- Conditions as Definitions
def top_difference_condition := (h - 20) / h = 5 / 7

-- Proof Statement
theorem taller_tree_height (h : ℝ) (H : top_difference_condition h) : h = 70 := 
by {
  sorry
}

end taller_tree_height_1_1943


namespace find_triple_sum_1_1548

theorem find_triple_sum (x y z : ℝ) 
  (h1 : y + z = 20 - 4 * x)
  (h2 : x + z = 1 - 4 * y)
  (h3 : x + y = -12 - 4 * z) :
  3 * x + 3 * y + 3 * z = 9 / 2 := 
sorry

end find_triple_sum_1_1548


namespace problem_statement_1_1756

variable (a b c d x : ℕ)

theorem problem_statement
  (h1 : a + b = x)
  (h2 : b + c = 9)
  (h3 : c + d = 3)
  (h4 : a + d = 6) :
  x = 12 :=
by
  sorry

end problem_statement_1_1756


namespace football_cost_correct_1_1214

variable (total_spent_on_toys : ℝ := 12.30)
variable (spent_on_marbles : ℝ := 6.59)

theorem football_cost_correct :
  (total_spent_on_toys - spent_on_marbles = 5.71) :=
by
  sorry

end football_cost_correct_1_1214


namespace non_real_roots_b_range_1_1670

theorem non_real_roots_b_range (b : ℝ) : 
  ∃ (x : ℂ), x^2 + (b : ℂ) * x + 16 = 0 ∧ (¬ ∃ (x : ℝ), x^2 + b * x + 16 = 0) ↔ -8 < b ∧ b < 8 := 
by
  sorry

end non_real_roots_b_range_1_1670


namespace cucumbers_after_purchase_1_1539

theorem cucumbers_after_purchase (C U : ℕ) (h1 : C + U = 10) (h2 : C = 4) : U + 2 = 8 := by
  sorry

end cucumbers_after_purchase_1_1539


namespace soccer_league_fraction_female_proof_1_1186

variable (m f : ℝ)

def soccer_league_fraction_female : Prop :=
  let males_last_year := m
  let females_last_year := f
  let males_this_year := 1.05 * m
  let females_this_year := 1.2 * f
  let total_this_year := 1.1 * (m + f)
  (1.05 * m + 1.2 * f = 1.1 * (m + f)) → ((0.6 * m) / (1.65 * m) = 4 / 11)

theorem soccer_league_fraction_female_proof (m f : ℝ) : soccer_league_fraction_female m f :=
by {
  sorry
}

end soccer_league_fraction_female_proof_1_1186


namespace find_m_no_solution_1_1752

-- Define the condition that the equation has no solution
def no_solution (m : ℤ) : Prop :=
  ∀ x : ℤ, (x + m)/(4 - x^2) + x / (x - 2) ≠ 1

-- State the proof problem in Lean 4
theorem find_m_no_solution : ∀ m : ℤ, no_solution m → (m = 2 ∨ m = 6) :=
by
  sorry

end find_m_no_solution_1_1752


namespace min_chord_length_intercepted_line_eq_1_1168

theorem min_chord_length_intercepted_line_eq (m : ℝ)
  (hC : ∀ (x y : ℝ), (x-1)^2 + (y-1)^2 = 16)
  (hL : ∀ (x y : ℝ), (2*m-1)*x + (m-1)*y - 3*m + 1 = 0)
  : ∃ x y : ℝ, x - 2*y - 4 = 0 := sorry

end min_chord_length_intercepted_line_eq_1_1168


namespace find_c_1_1868

-- Define the polynomial P(x)
def P (c : ℚ) (x : ℚ) : ℚ := x^3 + 4 * x^2 + c * x + 20

-- Given that x - 3 is a factor of P(x), prove that c = -83/3
theorem find_c (c : ℚ) (h : P c 3 = 0) : c = -83 / 3 :=
by
  sorry

end find_c_1_1868


namespace total_number_of_members_1_1702

-- Define the basic setup
def committees := Fin 5
def members := {m : Finset committees // m.card = 2}

-- State the theorem
theorem total_number_of_members :
  (∃ s : Finset members, s.card = 10) :=
sorry

end total_number_of_members_1_1702


namespace calculate_unoccupied_volume_1_1432

def tank_length : ℕ := 12
def tank_width : ℕ := 10
def tank_height : ℕ := 8
def tank_volume : ℕ := tank_length * tank_width * tank_height

def water_volume : ℕ := tank_volume / 3
def ice_cube_volume : ℕ := 1
def ice_cubes_count : ℕ := 12
def total_ice_volume : ℕ := ice_cubes_count * ice_cube_volume
def occupied_volume : ℕ := water_volume + total_ice_volume

def unoccupied_volume : ℕ := tank_volume - occupied_volume

theorem calculate_unoccupied_volume : unoccupied_volume = 628 := by
  sorry

end calculate_unoccupied_volume_1_1432


namespace jack_mopping_rate_1_1429

variable (bathroom_floor_area : ℕ) (kitchen_floor_area : ℕ) (time_mopped : ℕ)

theorem jack_mopping_rate
  (h_bathroom : bathroom_floor_area = 24)
  (h_kitchen : kitchen_floor_area = 80)
  (h_time : time_mopped = 13) :
  (bathroom_floor_area + kitchen_floor_area) / time_mopped = 8 :=
by
  sorry

end jack_mopping_rate_1_1429


namespace verify_extrema_1_1650

noncomputable def f (x : ℝ) : ℝ := (1 / 4) * x^4 - 2 * x^3 + (11 / 2) * x^2 - 6 * x + (9 / 4)

theorem verify_extrema :
  f 1 = 0 ∧ f 2 = 1 ∧ f 3 = 0 := by
  sorry

end verify_extrema_1_1650


namespace temperature_difference_correct_1_1210

def avg_high : ℝ := 9
def avg_low : ℝ := -5
def temp_difference : ℝ := avg_high - avg_low

theorem temperature_difference_correct : temp_difference = 14 := by
  sorry

end temperature_difference_correct_1_1210


namespace magnitude_of_z_1_1480

theorem magnitude_of_z (z : ℂ) (h : z * (2 - Complex.I) = Complex.I) : Complex.abs z = 1 / Real.sqrt 5 := by
  sorry

end magnitude_of_z_1_1480


namespace isosceles_triangle_perimeter_1_1386

def is_isosceles_triangle (a b c : ℕ) : Prop :=
  (a = b) ∨ (b = c) ∨ (a = c)

def is_valid_triangle (a b c : ℕ) : Prop :=
  (a + b > c) ∧ (a + c > b) ∧ (b + c > a)

theorem isosceles_triangle_perimeter {a b : ℕ} (h₁ : is_isosceles_triangle a b b) (h₂ : is_valid_triangle a b b) : a + b + b = 15 :=
  sorry

end isosceles_triangle_perimeter_1_1386


namespace min_value_SN64_by_aN_is_17_over_2_1_1402

noncomputable def a_n (n : ℕ) : ℕ := 2 * n
noncomputable def S_n (n : ℕ) : ℕ := n^2 + n

theorem min_value_SN64_by_aN_is_17_over_2 :
  ∃ (n : ℕ), 2 ≤ n ∧ (a_2 = 4 ∧ S_10 = 110) →
  ((S_n n + 64) / a_n n) = 17 / 2 :=
by
  sorry

end min_value_SN64_by_aN_is_17_over_2_1_1402


namespace lemonade_lemons_per_glass_1_1289

def number_of_glasses : ℕ := 9
def total_lemons : ℕ := 18
def lemons_per_glass : ℕ := 2

theorem lemonade_lemons_per_glass :
  total_lemons / number_of_glasses = lemons_per_glass :=
by
  sorry

end lemonade_lemons_per_glass_1_1289


namespace line_intersects_plane_at_angle_1_1327

def direction_vector : ℝ × ℝ × ℝ := (1, -1, 2)
def normal_vector : ℝ × ℝ × ℝ := (-2, 2, -4)

theorem line_intersects_plane_at_angle :
  let a := direction_vector
  let u := normal_vector
  a ≠ (0, 0, 0) → u ≠ (0, 0, 0) →
  ∃ θ : ℝ, 0 < θ ∧ θ < π :=
by
  sorry

end line_intersects_plane_at_angle_1_1327


namespace find_a_for_arithmetic_progression_roots_1_1609

theorem find_a_for_arithmetic_progression_roots (x a : ℝ) : 
  (∀ (x : ℝ), x^4 - a*x^2 + 1 = 0) → 
  (∃ (t1 t2 : ℝ), t1 > 0 ∧ t2 > 0 ∧ (t2 = 9*t1) ∧ (t1 + t2 = a) ∧ (t1 * t2 = 1)) → 
  (a = 10/3) := 
  by 
    intros h1 h2
    sorry

end find_a_for_arithmetic_progression_roots_1_1609


namespace jessica_current_age_1_1766

-- Definitions and conditions from the problem
def J (M : ℕ) : ℕ := M / 2
def M : ℕ := 60

-- Lean statement for the proof problem
theorem jessica_current_age : J M + 10 = 40 :=
by
  sorry

end jessica_current_age_1_1766


namespace find_constants_to_satisfy_equation_1_1886

-- Define the condition
def equation_condition (x : ℝ) (A B C : ℝ) :=
  -2 * x^2 + 5 * x - 6 = A * (x^2 + 1) + (B * x + C) * x

-- Define the proof problem as a Lean 4 statement
theorem find_constants_to_satisfy_equation (A B C : ℝ) :
  A = -6 ∧ B = 4 ∧ C = 5 ↔ ∀ x : ℝ, x ≠ 0 → x^2 + 1 ≠ 0 → equation_condition x A B C := 
by
  sorry

end find_constants_to_satisfy_equation_1_1886


namespace parallelogram_angle_1_1140

theorem parallelogram_angle (a b : ℕ) (h : a + b = 180) (exceed_by_10 : b = a + 10) : a = 85 := by
  -- proof skipped
  sorry

end parallelogram_angle_1_1140


namespace common_point_exists_1_1368

theorem common_point_exists (a b c : ℝ) :
  ∃ x y : ℝ, y = a * x ^ 2 - b * x + c ∧ y = b * x ^ 2 - c * x + a ∧ y = c * x ^ 2 - a * x + b :=
  sorry

end common_point_exists_1_1368


namespace probability_not_touch_outer_edge_1_1590

def checkerboard : ℕ := 10

def total_squares : ℕ := checkerboard * checkerboard

def perimeter_squares : ℕ := 4 * checkerboard - 4

def inner_squares : ℕ := total_squares - perimeter_squares

def probability : ℚ := inner_squares / total_squares

theorem probability_not_touch_outer_edge : probability = 16 / 25 :=
by
  sorry

end probability_not_touch_outer_edge_1_1590


namespace equilateral_triangle_perimeter_1_1843

-- Definitions based on conditions
def equilateral_triangle_side : ℕ := 8

-- The statement we need to prove
theorem equilateral_triangle_perimeter : 3 * equilateral_triangle_side = 24 := by
  sorry

end equilateral_triangle_perimeter_1_1843


namespace parrots_count_1_1505

theorem parrots_count (p r : ℕ) : 2 * p + 4 * r = 26 → p + r = 10 → p = 7 := by
  intros h1 h2
  sorry

end parrots_count_1_1505


namespace quadratic_to_standard_form_div_1_1361

theorem quadratic_to_standard_form_div (b c : ℤ)
  (h : ∀ x : ℤ, x^2 - 2100 * x - 8400 = (x + b)^2 + c) :
  c / b = 1058 :=
sorry

end quadratic_to_standard_form_div_1_1361


namespace neg_one_power_zero_1_1249

theorem neg_one_power_zero : (-1: ℤ)^0 = 1 := 
sorry

end neg_one_power_zero_1_1249


namespace sum_of_distinct_integers_eq_zero_1_1733

theorem sum_of_distinct_integers_eq_zero 
  (a b c d : ℤ) 
  (distinct : (a ≠ b) ∧ (a ≠ c) ∧ (a ≠ d) ∧ (b ≠ c) ∧ (b ≠ d) ∧ (c ≠ d))
  (prod_eq_25 : a * b * c * d = 25) : a + b + c + d = 0 := by
  sorry

end sum_of_distinct_integers_eq_zero_1_1733


namespace fraction_pizza_covered_by_pepperoni_1_1814

theorem fraction_pizza_covered_by_pepperoni :
  (∀ (r_pizz : ℝ) (n_pepp : ℕ) (d_pepp : ℝ),
      r_pizz = 8 ∧ n_pepp = 32 ∧ d_pepp = 2 →
      (n_pepp * π * (d_pepp / 2)^2) / (π * r_pizz^2) = 1 / 2) :=
sorry

end fraction_pizza_covered_by_pepperoni_1_1814


namespace number_of_participants_1_1976

-- Define the conditions and theorem
theorem number_of_participants (n : ℕ) (h : n * (n - 1) / 2 = 231) : n = 22 :=
  sorry

end number_of_participants_1_1976


namespace integer_pairs_satisfy_equation_1_1305

theorem integer_pairs_satisfy_equation :
  ∀ (x y : ℤ), (x^2 * y + y^2 = x^3) → (x = 0 ∧ y = 0) ∨ (x = -4 ∧ y = -8) :=
by
  sorry

end integer_pairs_satisfy_equation_1_1305


namespace valid_lineups_count_1_1297

-- Define the conditions
def total_players : ℕ := 15
def max : ℕ := 1
def rex : ℕ := 1
def tex : ℕ := 1

-- Proving the number of valid lineups
theorem valid_lineups_count :
  ∃ n, n = 5 ∧ total_players = 15 ∧ max + rex + tex ≤ 1 → n = 2277 :=
sorry

end valid_lineups_count_1_1297


namespace second_number_value_1_1647

theorem second_number_value (A B C : ℝ) 
    (h1 : A + B + C = 98) 
    (h2 : A = (2/3) * B) 
    (h3 : C = (8/5) * B) : 
    B = 30 :=
by 
  sorry

end second_number_value_1_1647


namespace balloon_ratio_1_1899

theorem balloon_ratio 
  (initial_blue : ℕ) (initial_purple : ℕ) (balloons_left : ℕ)
  (h1 : initial_blue = 303)
  (h2 : initial_purple = 453)
  (h3 : balloons_left = 378) :
  (balloons_left / (initial_blue + initial_purple) : ℚ) = (1 / 2 : ℚ) :=
by
  sorry

end balloon_ratio_1_1899


namespace converse_equivalence_1_1000

-- Definition of the original proposition
def original_proposition : Prop := ∀ (x : ℝ), x < 0 → x^2 > 0

-- Definition of the converse proposition
def converse_proposition : Prop := ∀ (x : ℝ), x^2 > 0 → x < 0

-- Theorem statement asserting the equivalence
theorem converse_equivalence : (converse_proposition = ¬ original_proposition) :=
sorry

end converse_equivalence_1_1000


namespace right_triangle_third_angle_1_1456

-- Define the problem
def sum_of_angles_in_triangle (a b c : ℝ) : Prop := a + b + c = 180

-- Define the given angles
def is_right_angle (a : ℝ) : Prop := a = 90
def given_angle (b : ℝ) : Prop := b = 25

-- Define the third angle
def third_angle (a b c : ℝ) : Prop := a + b + c = 180

-- The theorem to prove 
theorem right_triangle_third_angle : ∀ (a b c : ℝ), 
  is_right_angle a → given_angle b → third_angle a b c → c = 65 :=
by
  intros a b c ha hb h_triangle
  sorry

end right_triangle_third_angle_1_1456


namespace smallest_x_for_non_prime_expression_1_1646

/-- The smallest positive integer x for which x^2 + x + 41 is not a prime number is 40. -/
theorem smallest_x_for_non_prime_expression : ∃ x : ℕ, x > 0 ∧ x^2 + x + 41 = 41 * 41 ∧ (∀ y : ℕ, 0 < y ∧ y < x → Prime (y^2 + y + 41)) := 
sorry

end smallest_x_for_non_prime_expression_1_1646


namespace math_proof_problem_1_1375

-- Definitions
def PropA : Prop := ¬ (∀ n : ℤ, (3 ∣ n → ¬ (n % 2 = 1)))
def PropB : Prop := ¬ (¬ (∃ x : ℝ, x^2 + x + 1 ≥ 0))
def PropC : Prop := ∀ (α β : ℝ) (k : ℤ), α = k * Real.pi + β ↔ Real.tan α = Real.tan β
def PropD : Prop := ∀ (a b : ℝ), a ≠ 0 → a * b ≠ 0 → b ≠ 0

def correct_options : Prop := PropA ∧ PropC ∧ ¬PropB ∧ PropD

-- The theorem to be proven
theorem math_proof_problem : correct_options :=
by
  sorry

end math_proof_problem_1_1375


namespace mod_multiplication_1_1568

theorem mod_multiplication :
  (176 * 929) % 50 = 4 :=
by
  sorry

end mod_multiplication_1_1568


namespace maximum_abc_827_1_1012

noncomputable def maximum_abc (a b c : ℝ) := (a * b * c)

theorem maximum_abc_827 (a b c : ℝ) 
  (h1: a > 0) 
  (h2: b > 0) 
  (h3: c > 0) 
  (h4: (a * b) + c = (a + c) * (b + c)) 
  (h5: a + b + c = 2) : 
  maximum_abc a b c = 8 / 27 := 
by 
  sorry

end maximum_abc_827_1_1012


namespace line_does_not_pass_second_quadrant_1_1686

theorem line_does_not_pass_second_quadrant (a : ℝ) :
  (∀ x y : ℝ, (a + 1) * x + y + 2 - a = 0 → ¬(x < 0 ∧ y > 0)) ↔ a ≤ -1 :=
by
  sorry

end line_does_not_pass_second_quadrant_1_1686


namespace range_of_x_in_function_1_1897

theorem range_of_x_in_function : ∀ (x : ℝ), (2 - x ≥ 0) ∧ (x + 2 ≠ 0) ↔ (x ≤ 2 ∧ x ≠ -2) :=
by
  intro x
  sorry

end range_of_x_in_function_1_1897


namespace factor_y6_plus_64_1_1576

theorem factor_y6_plus_64 : (y^2 + 4) ∣ (y^6 + 64) :=
sorry

end factor_y6_plus_64_1_1576


namespace total_wet_surface_area_is_correct_1_1708

def cisternLength : ℝ := 8
def cisternWidth : ℝ := 4
def waterDepth : ℝ := 1.25

def bottomSurfaceArea : ℝ := cisternLength * cisternWidth
def longerSideSurfaceArea (depth : ℝ) : ℝ := depth * cisternLength * 2
def shorterSideSurfaceArea (depth : ℝ) : ℝ := depth * cisternWidth * 2

def totalWetSurfaceArea : ℝ :=
  bottomSurfaceArea + longerSideSurfaceArea waterDepth + shorterSideSurfaceArea waterDepth

theorem total_wet_surface_area_is_correct :
  totalWetSurfaceArea = 62 := by
  sorry

end total_wet_surface_area_is_correct_1_1708


namespace DianasInitialSpeed_1_1125

open Nat

theorem DianasInitialSpeed
  (total_distance : ℕ)
  (initial_time : ℕ)
  (tired_speed : ℕ)
  (total_time : ℕ)
  (distance_when_tired : ℕ)
  (initial_distance : ℕ)
  (initial_speed : ℕ)
  (initial_hours : ℕ) :
  total_distance = 10 →
  initial_time = 2 →
  tired_speed = 1 →
  total_time = 6 →
  distance_when_tired = tired_speed * (total_time - initial_time) →
  initial_distance = total_distance - distance_when_tired →
  initial_distance = initial_speed * initial_time →
  initial_speed = 3 := by
  sorry

end DianasInitialSpeed_1_1125


namespace max_value_exponential_and_power_functions_1_1970

variable (a b : ℝ)

-- Given conditions
axiom condition : 0 < b ∧ b < a ∧ a < 1

-- Problem statement
theorem max_value_exponential_and_power_functions : 
  a^b = max (max (a^b) (b^a)) (max (a^a) (b^b)) :=
by
  sorry

end max_value_exponential_and_power_functions_1_1970


namespace arrangement_of_70616_1_1580

def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

def arrangement_count (digits : List ℕ) : ℕ :=
  let count := digits.length
  let duplicates := List.length (List.filter (fun x => x = 6) digits)
  factorial count / factorial duplicates

theorem arrangement_of_70616 : arrangement_count [7, 0, 6, 6, 1] = 4 * 12 := by
  -- We need to prove that the number of ways to arrange the digits 7, 0, 6, 6, 1 without starting with 0 is 48
  sorry

end arrangement_of_70616_1_1580


namespace sum_combinatorial_identity_1_1933

theorem sum_combinatorial_identity (k m n : ℕ) (hk : 0 < k) (hm : 0 < m) (hn : 0 < n) (hkn : k ≤ n) :
  (∑ r in Finset.range (m + 1), k * (Nat.choose m r) * (Nat.choose n k) / ((r + k) * (Nat.choose (m + n) (r + k)))) = 1 :=
sorry

end sum_combinatorial_identity_1_1933


namespace fibonacci_series_sum_1_1158

def fibonacci (n : ℕ) : ℕ :=
  if n = 0 then 0
  else if n = 1 then 1
  else fibonacci (n-1) + fibonacci (n-2)

noncomputable def sum_fibonacci_fraction : ℚ :=
  ∑' (n : ℕ), (fibonacci n : ℚ) / (5^n : ℚ)

theorem fibonacci_series_sum : sum_fibonacci_fraction = 5 / 19 := by
  sorry

end fibonacci_series_sum_1_1158


namespace remainder_when_divided_by_10_1_1811

theorem remainder_when_divided_by_10 : 
  (2468 * 7391 * 90523) % 10 = 4 :=
by
  sorry

end remainder_when_divided_by_10_1_1811


namespace solve_for_x2_plus_9y2_1_1188

variable (x y : ℝ)

def condition1 : Prop := x + 3 * y = 3
def condition2 : Prop := x * y = -6

theorem solve_for_x2_plus_9y2 (h1 : condition1 x y) (h2 : condition2 x y) :
  x^2 + 9 * y^2 = 45 :=
by
  sorry

end solve_for_x2_plus_9y2_1_1188


namespace twelve_edge_cubes_painted_faces_1_1673

theorem twelve_edge_cubes_painted_faces :
  let painted_faces_per_edge_cube := 2
  let num_edge_cubes := 12
  painted_faces_per_edge_cube * num_edge_cubes = 24 :=
by
  sorry

end twelve_edge_cubes_painted_faces_1_1673


namespace find_pairs_1_1057

def is_power_of_two (n : ℕ) : Prop :=
  ∃ k : ℕ, n = 2^k

theorem find_pairs (a n : ℕ) (h1 : a ≥ n) (h2 : is_power_of_two ((a + 1)^n + a - 1)) :
  (a = 4 ∧ n = 3) ∨ (∃ k : ℕ, a = 2^k ∧ n = 1) :=
by
  sorry

end find_pairs_1_1057


namespace initial_contribution_amount_1_1729

variable (x : ℕ)
variable (workers : ℕ := 1200)
variable (total_with_extra_contribution: ℕ := 360000)
variable (extra_contribution_each: ℕ := 50)

theorem initial_contribution_amount :
  (workers * x = total_with_extra_contribution - workers * extra_contribution_each) →
  workers * x = 300000 :=
by
  intro h
  sorry

end initial_contribution_amount_1_1729


namespace larry_gave_52_apples_1_1776

-- Define the initial and final count of Joyce's apples
def initial_apples : ℝ := 75.0
def final_apples : ℝ := 127.0

-- Define the number of apples Larry gave Joyce
def apples_given : ℝ := final_apples - initial_apples

-- The theorem stating that Larry gave Joyce 52 apples
theorem larry_gave_52_apples : apples_given = 52 := by
  sorry

end larry_gave_52_apples_1_1776


namespace sum_of_first_five_terms_is_31_1_1063

variable (a : ℕ → ℝ) (q : ℝ)

-- The geometric sequence a_n
def is_geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n * q

-- Condition 1: a_2 * a_3 = 2 * a_1
def condition1 (a : ℕ → ℝ) (q : ℝ) : Prop :=
  a 2 * a 3 = 2 * a 1

-- Condition 2: The arithmetic mean of a_4 and 2 * a_7 is 5/4
def condition2 (a : ℕ → ℝ) (q : ℝ) : Prop :=
  (a 4 + 2 * a 7) / 2 = 5 / 4

-- Sum of the first 5 terms of the geometric sequence
def S_5 (a : ℕ → ℝ) : ℝ :=
  a 0 + a 1 + a 2 + a 3 + a 4

-- The theorem to prove
theorem sum_of_first_five_terms_is_31 (a : ℕ → ℝ) (q : ℝ) 
  (hg : is_geometric_sequence a q) 
  (hc1 : condition1 a q) 
  (hc2 : condition2 a q) : 
  S_5 a = 31 := by
  sorry

end sum_of_first_five_terms_is_31_1_1063


namespace scalene_triangle_cannot_be_divided_into_two_congruent_triangles_1_1095

-- Definitions and Conditions
structure Triangle :=
(a b c : ℝ) (h1 : a ≠ b) (h2 : b ≠ c) (h3 : a ≠ c)

-- Statement of the problem
theorem scalene_triangle_cannot_be_divided_into_two_congruent_triangles (T : Triangle) :
  ¬(∃ (D : ℝ) (ABD ACD : Triangle), ABD.a = ACD.a ∧ ABD.b = ACD.b ∧ ABD.c = ACD.c) :=
sorry

end scalene_triangle_cannot_be_divided_into_two_congruent_triangles_1_1095


namespace construction_company_total_weight_1_1979

noncomputable def total_weight_of_materials_in_pounds : ℝ :=
  let weight_of_concrete := 12568.3
  let weight_of_bricks := 2108 * 2.20462
  let weight_of_stone := 7099.5
  let weight_of_wood := 3778 * 2.20462
  let weight_of_steel := 5879 * (1 / 16)
  let weight_of_glass := 12.5 * 2000
  let weight_of_sand := 2114.8
  weight_of_concrete + weight_of_bricks + weight_of_stone + weight_of_wood + weight_of_steel + weight_of_glass + weight_of_sand

theorem construction_company_total_weight : total_weight_of_materials_in_pounds = 60129.72 :=
by
  sorry

end construction_company_total_weight_1_1979


namespace line_parallel_slope_1_1870

theorem line_parallel_slope (m : ℝ) :
  (2 * 8 = m * m) →
  m = -4 :=
by
  intro h
  sorry

end line_parallel_slope_1_1870


namespace general_term_1_1088

def S (n : ℕ) : ℕ := n^2 + 3 * n

def a (n : ℕ) : ℕ :=
  if n = 1 then S 1
  else S n - S (n - 1)

theorem general_term (n : ℕ) (hn : n ≥ 1) : a n = 2 * n + 2 :=
by {
  sorry
}

end general_term_1_1088


namespace count_three_digit_numbers_1_1384

def count_decreasing_digit_numbers : ℕ :=
  ∑ h in Finset.range 10 \ {0, 1}, ∑ t in Finset.range h, t

theorem count_three_digit_numbers :
  count_decreasing_digit_numbers = 120 :=
sorry

end count_three_digit_numbers_1_1384


namespace triangle_area_is_2_1_1253

noncomputable def area_of_triangle_OAB {x₀ : ℝ} (h₀ : 0 < x₀) : ℝ :=
  let y₀ := 1 / x₀
  let slope := -1 / x₀^2
  let tangent_line (x : ℝ) := y₀ + slope * (x - x₀)
  let A : ℝ × ℝ := (2 * x₀, 0) -- Intersection with x-axis
  let B : ℝ × ℝ := (0, 2 * y₀) -- Intersection with y-axis
  1 / 2 * abs (2 * y₀ * 2 * x₀)

theorem triangle_area_is_2 (x₀ : ℝ) (h₀ : 0 < x₀) : area_of_triangle_OAB h₀ = 2 :=
by
  sorry

end triangle_area_is_2_1_1253


namespace min_value_4x_plus_inv_1_1499

noncomputable def min_value_function (x : ℝ) := 4 * x + 1 / (4 * x - 5)

theorem min_value_4x_plus_inv (x : ℝ) (h : x > 5 / 4) : min_value_function x = 7 :=
sorry

end min_value_4x_plus_inv_1_1499


namespace rectangle_area_1_1931

-- Define the rectangular properties
variables {w l d x : ℝ}
def width (w : ℝ) : ℝ := w
def length (w : ℝ) : ℝ := 3 * w
def diagonal (w : ℝ) : ℝ := x

theorem rectangle_area (w x : ℝ) (hw : w ^ 2 + (3 * w) ^ 2 = x ^ 2) : w * 3 * w = 3 / 10 * x ^ 2 :=
by 
  sorry

end rectangle_area_1_1931


namespace find_the_triplet_1_1157

theorem find_the_triplet (x y z : ℕ) (h : x + y + z = x * y * z) : (x = 1 ∧ y = 2 ∧ z = 3) ∨ (x = 1 ∧ y = 3 ∧ z = 2) ∨ (x = 2 ∧ y = 1 ∧ z = 3) ∨ (x = 2 ∧ y = 3 ∧ z = 1) ∨ (x = 3 ∧ y = 1 ∧ z = 2) ∨ (x = 3 ∧ y = 2 ∧ z = 1) :=
by
  sorry

end find_the_triplet_1_1157


namespace ratio_of_volume_to_surface_area_1_1442

-- Definitions of the given conditions
def unit_cube_volume : ℕ := 1
def total_cubes : ℕ := 8
def volume := total_cubes * unit_cube_volume
def exposed_faces (center_cube_faces : ℕ) (side_cube_faces : ℕ) (top_cube_faces : ℕ) : ℕ :=
  center_cube_faces + 6 * side_cube_faces + top_cube_faces
def surface_area := exposed_faces 1 5 5
def ratio := volume / surface_area

-- The main theorem statement
theorem ratio_of_volume_to_surface_area : ratio = 2 / 9 := by
  sorry

end ratio_of_volume_to_surface_area_1_1442


namespace sum_in_base7_1_1577

-- An encoder function for base 7 integers
def to_base7 (n : ℕ) : string :=
sorry -- skipping the implementation for brevity

-- Decoding the string representation back to a natural number
def from_base7 (s : string) : ℕ :=
sorry -- skipping the implementation for brevity

-- The provided numbers in base 7
def x : ℕ := from_base7 "666"
def y : ℕ := from_base7 "66"
def z : ℕ := from_base7 "6"

-- The expected sum in base 7
def expected_sum : ℕ := from_base7 "104"

-- The statement to be proved
theorem sum_in_base7 : x + y + z = expected_sum :=
sorry -- The proof is omitted

end sum_in_base7_1_1577


namespace number_solution_1_1781

theorem number_solution : ∃ x : ℝ, x + 9 = x^2 ∧ x = (1 + Real.sqrt 37) / 2 :=
by
  use (1 + Real.sqrt 37) / 2
  simp
  sorry

end number_solution_1_1781


namespace probability_of_top_card_heart_1_1145

-- Define the total number of cards in the deck.
def total_cards : ℕ := 39

-- Define the number of hearts in the deck.
def hearts : ℕ := 13

-- Define the probability that the top card is a heart.
def probability_top_card_heart : ℚ := hearts / total_cards

-- State the theorem to prove.
theorem probability_of_top_card_heart : probability_top_card_heart = 1 / 3 :=
by
  sorry

end probability_of_top_card_heart_1_1145


namespace geometric_sequence_seventh_term_1_1663

theorem geometric_sequence_seventh_term
  (a r : ℝ)
  (h1 : a * r^4 = 16)
  (h2 : a * r^10 = 4) :
  a * r^6 = 4 * (2^(2/3)) :=
by
  sorry

end geometric_sequence_seventh_term_1_1663


namespace apples_difference_1_1034

-- Definitions for initial and remaining apples
def initial_apples : ℕ := 46
def remaining_apples : ℕ := 14

-- The theorem to prove the difference between initial and remaining apples is 32
theorem apples_difference : initial_apples - remaining_apples = 32 := by
  -- proof is omitted
  sorry

end apples_difference_1_1034


namespace prove_a5_1_1319

-- Definition of the conditions
def expansion (x : ℤ) (a_0 a_1 a_2 a_3 a_4 a_5 a_6 a_7 a_8 : ℤ) :=
  (x - 1) ^ 8 = a_0 + a_1 * (1 + x) + a_2 * (1 + x)^2 + a_3 * (1 + x)^3 + a_4 * (1 + x)^4 + 
               a_5 * (1 + x)^5 + a_6 * (1 + x)^6 + a_7 * (1 + x)^7 + a_8 * (1 + x)^8

-- Given condition
axiom condition (a_0 a_1 a_2 a_3 a_4 a_5 a_6 a_7 a_8 : ℤ) : ∀ x : ℤ, expansion x a_0 a_1 a_2 a_3 a_4 a_5 a_6 a_7 a_8

-- The target problem: proving a_5 = -448
theorem prove_a5 (a_0 a_1 a_2 a_3 a_4 a_5 a_6 a_7 a_8 : ℤ) : a_5 = -448 :=
by
  sorry

end prove_a5_1_1319


namespace cute_angle_of_isosceles_cute_triangle_1_1155

theorem cute_angle_of_isosceles_cute_triangle (A B C : ℝ) 
    (h1 : B = 2 * C)
    (h2 : A + B + C = 180)
    (h3 : A = B ∨ A = C) :
    A = 45 ∨ A = 72 :=
sorry

end cute_angle_of_isosceles_cute_triangle_1_1155


namespace olivia_money_left_1_1990

-- Defining hourly wages
def wage_monday : ℕ := 10
def wage_wednesday : ℕ := 12
def wage_friday : ℕ := 14
def wage_saturday : ℕ := 20

-- Defining hours worked each day
def hours_monday : ℕ := 5
def hours_wednesday : ℕ := 4
def hours_friday : ℕ := 3
def hours_saturday : ℕ := 2

-- Defining business-related expenses and tax rate
def expenses : ℕ := 50
def tax_rate : ℝ := 0.15

-- Calculate total earnings
def total_earnings : ℕ :=
  (hours_monday * wage_monday) +
  (hours_wednesday * wage_wednesday) +
  (hours_friday * wage_friday) +
  (hours_saturday * wage_saturday)

-- Earnings after expenses
def earnings_after_expenses : ℕ :=
  total_earnings - expenses

-- Calculate tax amount
def tax_amount : ℝ :=
  tax_rate * (total_earnings : ℝ)

-- Final amount Olivia has left
def remaining_amount : ℝ :=
  (earnings_after_expenses : ℝ) - tax_amount

theorem olivia_money_left : remaining_amount = 103 := by
  sorry

end olivia_money_left_1_1990
