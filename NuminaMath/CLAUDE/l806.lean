import Mathlib

namespace smaller_number_in_ratio_l806_80664

theorem smaller_number_in_ratio (a b c x y : ℝ) : 
  0 < a → a < b → x > 0 → y > 0 → x / y = (a + 1) / (b + 1) → x + y = 2 * c →
  min x y = (2 * c * (a + 1)) / (a + b + 2) := by
sorry

end smaller_number_in_ratio_l806_80664


namespace meal_combinations_l806_80683

/-- Number of meat options -/
def meatOptions : ℕ := 4

/-- Number of vegetable options -/
def vegOptions : ℕ := 5

/-- Number of dessert options -/
def dessertOptions : ℕ := 5

/-- Number of meat choices -/
def meatChoices : ℕ := 2

/-- Number of vegetable choices -/
def vegChoices : ℕ := 3

/-- Number of dessert choices -/
def dessertChoices : ℕ := 1

/-- The total number of meal combinations -/
theorem meal_combinations : 
  (meatOptions.choose meatChoices) * (vegOptions.choose vegChoices) * dessertOptions = 300 := by
  sorry

end meal_combinations_l806_80683


namespace min_type_c_cards_l806_80679

/-- Represents the number of cards sold of each type -/
structure CardSales where
  a : ℕ
  b : ℕ
  c : ℕ

/-- Calculates the total number of cards sold -/
def total_cards (sales : CardSales) : ℕ :=
  sales.a + sales.b + sales.c

/-- Calculates the total income from card sales -/
def total_income (sales : CardSales) : ℚ :=
  0.5 * sales.a + 1 * sales.b + 2.5 * sales.c

/-- Theorem stating the minimum number of type C cards sold -/
theorem min_type_c_cards (sales : CardSales) 
  (h1 : total_cards sales = 150)
  (h2 : total_income sales = 180) :
  sales.c ≥ 20 := by
  sorry

#check min_type_c_cards

end min_type_c_cards_l806_80679


namespace unique_divisor_sum_product_l806_80669

def sum_of_divisors (n : ℕ) : ℕ := (Finset.filter (· ∣ n) (Finset.range (n + 1))).sum id

def product_of_divisors (n : ℕ) : ℕ := (Finset.filter (· ∣ n) (Finset.range (n + 1))).prod id

theorem unique_divisor_sum_product :
  ∃! P : ℕ, P > 0 ∧ sum_of_divisors P = 2 * P ∧ product_of_divisors P = P ^ 2 ∧ P = 6 := by
  sorry

end unique_divisor_sum_product_l806_80669


namespace triangle_kp_r3_bound_l806_80696

/-- For any triangle with circumradius R, perimeter P, and area K, KP/R³ ≤ 27/4 -/
theorem triangle_kp_r3_bound (R P K : ℝ) (hR : R > 0) (hP : P > 0) (hK : K > 0) :
  K * P / R^3 ≤ 27 / 4 := by
  sorry


end triangle_kp_r3_bound_l806_80696


namespace product_of_roots_equals_one_l806_80692

theorem product_of_roots_equals_one :
  let A := Real.sqrt 2019 + Real.sqrt 2020 + 1
  let B := -Real.sqrt 2019 - Real.sqrt 2020 - 1
  let C := Real.sqrt 2019 - Real.sqrt 2020 + 1
  let D := Real.sqrt 2020 - Real.sqrt 2019 - 1
  A * B * C * D = 1 := by sorry

end product_of_roots_equals_one_l806_80692


namespace highest_power_of_two_and_three_l806_80689

def n : ℤ := 15^4 - 11^4

theorem highest_power_of_two_and_three (n : ℤ) (h : n = 15^4 - 11^4) :
  (∃ m : ℕ, 2^4 * m = n ∧ ¬(∃ k : ℕ, 2^5 * k = n)) ∧
  (∃ m : ℕ, 3^0 * m = n ∧ ¬(∃ k : ℕ, 3^1 * k = n)) :=
sorry

end highest_power_of_two_and_three_l806_80689


namespace density_of_S_l806_80619

def S : Set ℚ := {q : ℚ | ∃ (m n : ℕ), q = (m * n : ℚ) / ((m^2 + n^2) : ℚ)}

theorem density_of_S (x y : ℚ) (hx : x ∈ S) (hy : y ∈ S) (hxy : x < y) :
  ∃ z : ℚ, z ∈ S ∧ x < z ∧ z < y := by
  sorry

end density_of_S_l806_80619


namespace bird_count_theorem_l806_80684

theorem bird_count_theorem (initial_parrots : ℕ) (remaining_parrots : ℕ) 
  (remaining_crows : ℕ) (remaining_pigeons : ℕ) : 
  initial_parrots = 15 →
  remaining_parrots = 5 →
  remaining_crows = 3 →
  remaining_pigeons = 2 →
  ∃ (flew_away : ℕ), 
    flew_away = initial_parrots - remaining_parrots ∧
    initial_parrots + (flew_away + remaining_crows) + (flew_away + remaining_pigeons) = 40 :=
by sorry

end bird_count_theorem_l806_80684


namespace chocolate_distribution_l806_80627

theorem chocolate_distribution (x y : ℕ) (h1 : y = x + 1) (h2 : ∃ z : ℕ, y = (x - 1) * z + z) : 
  ∃ z : ℕ, y = (x - 1) * z + z ∧ z = 2 := by
sorry

end chocolate_distribution_l806_80627


namespace triangle_inequality_l806_80611

/-- For any triangle with side lengths a, b, and c, 
    3(b+c-a)(c+a-b)(a+b-c) ≤ a²(b+c-a) + b²(c+a-b) + c²(a+b-c) holds. -/
theorem triangle_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (hab : a + b > c) (hbc : b + c > a) (hca : c + a > b) :
  3 * (b + c - a) * (c + a - b) * (a + b - c) ≤ 
  a^2 * (b + c - a) + b^2 * (c + a - b) + c^2 * (a + b - c) := by
  sorry

end triangle_inequality_l806_80611


namespace ticket_sales_revenue_ticket_sales_problem_l806_80637

theorem ticket_sales_revenue 
  (student_price : ℕ) 
  (general_price : ℕ) 
  (total_tickets : ℕ) 
  (general_tickets : ℕ) : ℕ :=
  let student_tickets := total_tickets - general_tickets
  let student_revenue := student_tickets * student_price
  let general_revenue := general_tickets * general_price
  student_revenue + general_revenue

theorem ticket_sales_problem : 
  ticket_sales_revenue 4 6 525 388 = 2876 := by
  sorry

end ticket_sales_revenue_ticket_sales_problem_l806_80637


namespace point_A_distance_theorem_l806_80660

-- Define the point A on the number line
def A : ℝ → ℝ := λ a ↦ 2 * a + 1

-- Define the distance function from a point to the origin
def distance_to_origin (x : ℝ) : ℝ := |x|

-- Theorem statement
theorem point_A_distance_theorem :
  ∀ a : ℝ, distance_to_origin (A a) = 3 → a = -2 ∨ a = 1 :=
by
  sorry

end point_A_distance_theorem_l806_80660


namespace quadratic_roots_difference_l806_80639

theorem quadratic_roots_difference (p q : ℝ) (hp : p > 0) (hq : q > 0) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧
    x₁^2 + p*x₁ + q = 0 ∧
    x₂^2 + p*x₂ + q = 0 ∧
    |x₁ - x₂| = 2) →
  p = 2 * Real.sqrt (q + 1) :=
by sorry

end quadratic_roots_difference_l806_80639


namespace max_fourth_root_sum_max_fourth_root_sum_achievable_l806_80656

theorem max_fourth_root_sum (a b c d : ℝ) 
  (h_nonneg : a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧ d ≥ 0) 
  (h_sum : a + b + c + d = 1) : 
  (abcd : ℝ)^(1/4) + ((1-a)*(1-b)*(1-c)*(1-d) : ℝ)^(1/4) ≤ 1 :=
by sorry

theorem max_fourth_root_sum_achievable : 
  ∃ (a b c d : ℝ), 
    a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧ d ≥ 0 ∧ 
    a + b + c + d = 1 ∧ 
    (abcd : ℝ)^(1/4) + ((1-a)*(1-b)*(1-c)*(1-d) : ℝ)^(1/4) = 1 :=
by sorry

end max_fourth_root_sum_max_fourth_root_sum_achievable_l806_80656


namespace quadratic_roots_l806_80685

/-- A quadratic function f(x) = x^2 - px + q -/
def f (p q x : ℝ) : ℝ := x^2 - p*x + q

/-- Theorem: If f(p + q) = 0 and f(p - q) = 0, then either q = 0 (and p can be any value) or (p, q) = (0, -1) -/
theorem quadratic_roots (p q : ℝ) : 
  f p q (p + q) = 0 ∧ f p q (p - q) = 0 → 
  (q = 0 ∨ (p = 0 ∧ q = -1)) := by
sorry

end quadratic_roots_l806_80685


namespace common_intersection_point_l806_80624

-- Define the circle S
def S : Set (ℝ × ℝ) := {p | p.1^2 + p.2^2 = 1}

-- Define points A and B
def A : ℝ × ℝ := (1, 0)
def B : ℝ × ℝ := (-1, 0)

-- Define the tangent line L at A
def L : Set (ℝ × ℝ) := {p | p.1 = 1}

-- Define the function for points X and Y on L
def X (p : ℝ) : ℝ × ℝ := (1, 2*p)
def Y (q : ℝ) : ℝ × ℝ := (1, -2*q)

-- Define the condition for X and Y
def XYCondition (p q c : ℝ) : Prop := p * q = c / 4

-- Define the theorem
theorem common_intersection_point (c : ℝ) (h : c > 0) :
  ∀ (p q : ℝ), p > 0 → q > 0 → XYCondition p q c →
  ∃ (R : ℝ × ℝ), R.1 = (4 - c) / (4 + c) ∧ R.2 = 0 ∧
  (∀ (P Q : ℝ × ℝ), P ∈ S → Q ∈ S →
   (∃ (t : ℝ), P = (1 - t) • B + t • X p) →
   (∃ (s : ℝ), Q = (1 - s) • B + s • Y q) →
   ∃ (k : ℝ), R = (1 - k) • P + k • Q) :=
sorry

end common_intersection_point_l806_80624


namespace eggs_per_group_l806_80681

/-- Given 9 eggs split into 3 groups, prove that there are 3 eggs in each group. -/
theorem eggs_per_group (total_eggs : ℕ) (num_groups : ℕ) (h1 : total_eggs = 9) (h2 : num_groups = 3) :
  total_eggs / num_groups = 3 := by
  sorry

end eggs_per_group_l806_80681


namespace perpendicular_line_to_plane_is_perpendicular_to_all_lines_in_plane_l806_80622

-- Define a plane
structure Plane :=
  (α : Type*)

-- Define a line
structure Line :=
  (l : Type*)

-- Define perpendicular relation between a line and a plane
def perpendicular_to_plane (l : Line) (α : Plane) : Prop :=
  sorry

-- Define a line being contained within a plane
def contained_in_plane (m : Line) (α : Plane) : Prop :=
  sorry

-- Define perpendicular relation between two lines
def perpendicular_lines (l m : Line) : Prop :=
  sorry

-- Theorem statement
theorem perpendicular_line_to_plane_is_perpendicular_to_all_lines_in_plane
  (l m : Line) (α : Plane)
  (h1 : perpendicular_to_plane l α)
  (h2 : contained_in_plane m α) :
  perpendicular_lines l m :=
sorry

end perpendicular_line_to_plane_is_perpendicular_to_all_lines_in_plane_l806_80622


namespace xy_value_l806_80615

theorem xy_value (x y : ℝ) (h1 : x + y = 10) (h2 : x^2 + y^2 = 58) : x * y = 21 := by
  sorry

end xy_value_l806_80615


namespace brianne_alex_yard_ratio_l806_80667

/-- Proves that Brianne's yard is 6 times larger than Alex's yard given the conditions -/
theorem brianne_alex_yard_ratio :
  ∀ (derrick_yard alex_yard brianne_yard : ℝ),
  derrick_yard = 10 →
  alex_yard = derrick_yard / 2 →
  brianne_yard = 30 →
  brianne_yard / alex_yard = 6 := by
sorry

end brianne_alex_yard_ratio_l806_80667


namespace lily_catches_mary_l806_80653

/-- Mary's walking speed in miles per hour -/
def mary_speed : ℝ := 4

/-- Lily's walking speed in miles per hour -/
def lily_speed : ℝ := 6

/-- Initial distance between Mary and Lily in miles -/
def initial_distance : ℝ := 2

/-- Time in minutes for Lily to catch up to Mary -/
def catch_up_time : ℝ := 60

theorem lily_catches_mary : 
  (lily_speed - mary_speed) * catch_up_time / 60 = initial_distance := by
  sorry

end lily_catches_mary_l806_80653


namespace sufficient_conditions_for_inequality_l806_80623

theorem sufficient_conditions_for_inequality (f : ℝ → ℝ) :
  (((∀ x y : ℝ, x < y → f x > f y) ∧ (∀ x : ℝ, f x > 0)) →
    (∃ a : ℝ, a ≠ 0 ∧ ∀ x : ℝ, f (x + a) < f x + f a)) ∧
  ((∀ x y : ℝ, x < y → f x < f y) ∧ (∃ x₀ : ℝ, x₀ < 0 ∧ f x₀ = 0) →
    (∃ a : ℝ, a ≠ 0 ∧ ∀ x : ℝ, f (x + a) < f x + f a)) :=
by sorry

end sufficient_conditions_for_inequality_l806_80623


namespace circle_travel_in_triangle_l806_80687

/-- The distance traveled by the center of a circle rolling inside a triangle -/
def circle_travel_distance (a b c r : ℝ) : ℝ :=
  (a - 2 * r) + (b - 2 * r) + (c - 2 * r)

/-- Theorem: The distance traveled by the center of a circle with radius 2
    rolling inside a 6-8-10 triangle is 8 -/
theorem circle_travel_in_triangle :
  circle_travel_distance 6 8 10 2 = 8 := by
  sorry

end circle_travel_in_triangle_l806_80687


namespace solve_exponential_equation_l806_80690

theorem solve_exponential_equation :
  ∃ y : ℝ, 4^(3*y) = (64 : ℝ)^(1/3) ∧ y = 1/3 := by
  sorry

end solve_exponential_equation_l806_80690


namespace largest_multiple_of_11_less_than_neg_150_l806_80686

theorem largest_multiple_of_11_less_than_neg_150 :
  ∀ n : ℤ, n * 11 < -150 → n * 11 ≤ -154 :=
by
  sorry

end largest_multiple_of_11_less_than_neg_150_l806_80686


namespace probability_even_sum_two_wheels_l806_80699

theorem probability_even_sum_two_wheels (wheel1_total : ℕ) (wheel1_even : ℕ) (wheel2_total : ℕ) (wheel2_even : ℕ) :
  wheel1_total = 2 * wheel1_even ∧ 
  wheel2_total = 5 ∧ 
  wheel2_even = 2 →
  (wheel1_even : ℚ) / wheel1_total * (wheel2_even : ℚ) / wheel2_total + 
  (wheel1_even : ℚ) / wheel1_total * ((wheel2_total - wheel2_even) : ℚ) / wheel2_total = 1 / 2 := by
  sorry

end probability_even_sum_two_wheels_l806_80699


namespace quadratic_roots_l806_80616

theorem quadratic_roots (a b c : ℝ) (ha : a ≠ 0) 
  (h1 : a + b + c = 0) (h2 : 4*a - 2*b + c = 0) : 
  ∃ (x y : ℝ), x = 1 ∧ y = -2 ∧ 
  (∀ z : ℝ, a*z^2 + b*z + c = 0 ↔ z = x ∨ z = y) := by
sorry

end quadratic_roots_l806_80616


namespace gcd_18_30_l806_80694

theorem gcd_18_30 : Nat.gcd 18 30 = 6 := by
  sorry

end gcd_18_30_l806_80694


namespace polar_to_rectangular_l806_80695

theorem polar_to_rectangular (x y ρ : ℝ) :
  ρ = 2 → x^2 + y^2 = ρ^2 → x^2 + y^2 = 4 := by
sorry

end polar_to_rectangular_l806_80695


namespace sniper_B_wins_l806_80665

/-- Represents the probabilities of scoring 1, 2, and 3 points for a sniper -/
structure SniperProbabilities where
  one : Real
  two : Real
  three : Real
  sum_to_one : one + two + three = 1
  non_negative : one ≥ 0 ∧ two ≥ 0 ∧ three ≥ 0

/-- Calculates the expected score for a sniper given their probabilities -/
def expectedScore (p : SniperProbabilities) : Real :=
  1 * p.one + 2 * p.two + 3 * p.three

/-- Sniper A's probabilities -/
def sniperA : SniperProbabilities where
  one := 0.4
  two := 0.1
  three := 0.5
  sum_to_one := by sorry
  non_negative := by sorry

/-- Sniper B's probabilities -/
def sniperB : SniperProbabilities where
  one := 0.1
  two := 0.6
  three := 0.3
  sum_to_one := by sorry
  non_negative := by sorry

/-- Theorem stating that Sniper B has a higher expected score than Sniper A -/
theorem sniper_B_wins : expectedScore sniperB > expectedScore sniperA := by
  sorry

end sniper_B_wins_l806_80665


namespace probability_of_square_l806_80672

/-- The probability of selecting a square from a set of figures -/
theorem probability_of_square (total_figures : ℕ) (square_count : ℕ) 
  (h1 : total_figures = 10) (h2 : square_count = 3) : 
  (square_count : ℚ) / total_figures = 3 / 10 := by
  sorry

#check probability_of_square

end probability_of_square_l806_80672


namespace problem_solution_l806_80654

theorem problem_solution : 
  (∃ x : ℝ, x^2 = 6) ∧ (∃ y : ℝ, y^2 = 2) ∧ (∃ z : ℝ, z^2 = 27) ∧ (∃ w : ℝ, w^2 = 9) ∧ (∃ v : ℝ, v^2 = 1/3) →
  (∃ a b : ℝ, 
    (a^2 = 6 ∧ b^2 = 2 ∧ a * b + Real.sqrt 27 / Real.sqrt 9 - Real.sqrt (1/3) = 8 * Real.sqrt 3 / 3) ∧
    ((Real.sqrt 5 - 1)^2 - (Real.sqrt 2 - 1) * (Real.sqrt 2 + 1) = 5 - 2 * Real.sqrt 5)) := by
  sorry

end problem_solution_l806_80654


namespace sphere_volume_condition_l806_80649

theorem sphere_volume_condition (R V : ℝ) : 
  (V = (4 / 3) * π * R^3) → (R > Real.sqrt 10 → V > 36 * π) := by
  sorry

end sphere_volume_condition_l806_80649


namespace multiply_72519_by_9999_l806_80645

theorem multiply_72519_by_9999 : 72519 * 9999 = 725117481 := by
  sorry

end multiply_72519_by_9999_l806_80645


namespace least_number_of_pennies_l806_80604

theorem least_number_of_pennies : ∃ (a : ℕ), a > 0 ∧ 
  a % 7 = 3 ∧ 
  a % 5 = 4 ∧ 
  a % 3 = 2 ∧ 
  ∀ (b : ℕ), b > 0 → b % 7 = 3 → b % 5 = 4 → b % 3 = 2 → a ≤ b :=
by sorry

end least_number_of_pennies_l806_80604


namespace monomial_difference_implies_m_pow_n_eq_nine_l806_80612

-- Define the variables
variable (a b m n : ℕ)

-- Define the condition that the difference is a monomial
def is_monomial_difference : Prop :=
  ∃ (k : ℕ) (c : ℤ), 2 * a * b^(2*m+n) - a^(m-n) * b^8 = c * a^k * b^k

-- State the theorem
theorem monomial_difference_implies_m_pow_n_eq_nine
  (h : is_monomial_difference a b m n) : m^n = 9 := by
  sorry

end monomial_difference_implies_m_pow_n_eq_nine_l806_80612


namespace political_science_majors_l806_80610

/-- The number of applicants who majored in political science -/
def P : ℕ := 15

theorem political_science_majors :
  (40 : ℕ) = P + 15 + 10 ∧ 
  (20 : ℕ) = 5 + 15 ∧
  (10 : ℕ) = 40 - (P + 20) :=
by sorry

end political_science_majors_l806_80610


namespace ceiling_neg_sqrt_36_l806_80659

theorem ceiling_neg_sqrt_36 : ⌈-Real.sqrt 36⌉ = -6 := by
  sorry

end ceiling_neg_sqrt_36_l806_80659


namespace unique_three_digit_factorial_sum_l806_80621

def factorial (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

def digit_sum_factorial (n : ℕ) : ℕ :=
  let hundreds := n / 100
  let tens := (n / 10) % 10
  let ones := n % 10
  factorial hundreds + factorial tens + factorial ones

theorem unique_three_digit_factorial_sum : ∀ n : ℕ, 
  100 ≤ n ∧ n ≤ 999 → (n = digit_sum_factorial n ↔ n = 145) :=
by sorry

end unique_three_digit_factorial_sum_l806_80621


namespace staff_dress_price_l806_80662

/-- The final price of a dress for staff members after discounts -/
theorem staff_dress_price (d : ℝ) : 
  let initial_discount : ℝ := 0.65
  let staff_discount : ℝ := 0.60
  let price_after_initial_discount : ℝ := d * (1 - initial_discount)
  let final_price : ℝ := price_after_initial_discount * (1 - staff_discount)
  final_price = d * 0.14 := by
sorry

end staff_dress_price_l806_80662


namespace unique_N_value_l806_80606

theorem unique_N_value (a b N : ℕ) (h1 : N = (a^2 + b^2) / (a*b - 1)) : N = 5 := by
  sorry

end unique_N_value_l806_80606


namespace greatest_multiple_of_5_and_7_under_800_l806_80657

theorem greatest_multiple_of_5_and_7_under_800 : 
  ∀ n : ℕ, n % 5 = 0 ∧ n % 7 = 0 ∧ n < 800 → n ≤ 770 :=
by sorry

end greatest_multiple_of_5_and_7_under_800_l806_80657


namespace aleesia_weight_loss_l806_80628

/-- Aleesia's weekly weight loss problem -/
theorem aleesia_weight_loss 
  (aleesia_weeks : ℕ) 
  (alexei_weeks : ℕ) 
  (alexei_weekly_loss : ℝ) 
  (total_loss : ℝ) 
  (h1 : aleesia_weeks = 10) 
  (h2 : alexei_weeks = 8) 
  (h3 : alexei_weekly_loss = 2.5) 
  (h4 : total_loss = 35) :
  ∃ (aleesia_weekly_loss : ℝ), 
    aleesia_weekly_loss * aleesia_weeks + alexei_weekly_loss * alexei_weeks = total_loss ∧ 
    aleesia_weekly_loss = 1.5 := by
  sorry


end aleesia_weight_loss_l806_80628


namespace lcm_of_36_and_12_l806_80693

theorem lcm_of_36_and_12 (a b : ℕ+) (h1 : a = 36) (h2 : b = 12) (h3 : Nat.gcd a b = 8) :
  Nat.lcm a b = 54 := by
  sorry

end lcm_of_36_and_12_l806_80693


namespace count_parallel_edges_l806_80636

structure RectangularPrism where
  length : ℝ
  width : ℝ
  height : ℝ
  distinct : length ≠ width ∧ width ≠ height ∧ length ≠ height

def parallel_edge_pairs (prism : RectangularPrism) : ℕ := 6

theorem count_parallel_edges (prism : RectangularPrism) :
  parallel_edge_pairs prism = 6 := by
  sorry

end count_parallel_edges_l806_80636


namespace right_triangle_hypotenuse_l806_80646

theorem right_triangle_hypotenuse (QR RS QS : ℝ) (cos_R : ℝ) : 
  cos_R = 3/5 →  -- Given condition
  RS = 10 →     -- Given condition
  QR = RS * cos_R →  -- Definition of cosine in right triangle
  QS^2 = RS^2 - QR^2 →  -- Pythagorean theorem
  QS = 8 :=  -- Conclusion to prove
by sorry  -- Proof omitted

end right_triangle_hypotenuse_l806_80646


namespace triangle_third_height_bound_l806_80650

theorem triangle_third_height_bound (a b c : ℝ) (ha hb : ℝ) (h : ℝ) : 
  ha = 12 → hb = 20 → 
  a * ha = b * hb → 
  c * h = a * ha → 
  c > a - b → 
  h < 30 := by sorry

end triangle_third_height_bound_l806_80650


namespace prob_two_odd_dice_l806_80647

/-- The number of faces on a standard die -/
def num_faces : ℕ := 6

/-- The number of odd faces on a standard die -/
def num_odd_faces : ℕ := 3

/-- The total number of possible outcomes when throwing two dice -/
def total_outcomes : ℕ := num_faces * num_faces

/-- The number of outcomes where both dice show odd numbers -/
def favorable_outcomes : ℕ := num_odd_faces * num_odd_faces

/-- The probability of rolling two odd numbers when throwing two dice simultaneously -/
theorem prob_two_odd_dice : 
  (favorable_outcomes : ℚ) / total_outcomes = 1 / 4 := by
  sorry

end prob_two_odd_dice_l806_80647


namespace largest_solution_of_equation_l806_80691

theorem largest_solution_of_equation (a b c d : ℤ) (x : ℝ) :
  (4 * x / 5 - 2 = 5 / x) →
  (x = (a + b * Real.sqrt c) / d) →
  (∀ y, (4 * y / 5 - 2 = 5 / y) → y ≤ x) →
  (x = (5 + 5 * Real.sqrt 5) / 4) :=
by sorry

end largest_solution_of_equation_l806_80691


namespace school_bus_time_theorem_l806_80698

/-- Represents time in hours and minutes -/
structure Time where
  hours : Nat
  minutes : Nat

/-- Converts 24-hour format to 12-hour format -/
def to12HourFormat (t : Time) : Time :=
  if t.hours ≤ 12 then t else { hours := t.hours - 12, minutes := t.minutes }

/-- Calculates the time difference in minutes between two Time values -/
def timeDifference (t1 t2 : Time) : Nat :=
  (t2.hours * 60 + t2.minutes) - (t1.hours * 60 + t1.minutes)

theorem school_bus_time_theorem :
  let schoolEndTime : Time := { hours := 16, minutes := 45 }
  let arrivalTime : Time := { hours := 17, minutes := 20 }
  (to12HourFormat schoolEndTime = { hours := 4, minutes := 45 }) ∧
  (timeDifference schoolEndTime arrivalTime = 35) :=
by sorry

end school_bus_time_theorem_l806_80698


namespace cost_per_shot_l806_80661

def number_of_dogs : ℕ := 3
def puppies_per_dog : ℕ := 4
def shots_per_puppy : ℕ := 2
def total_cost : ℕ := 120

theorem cost_per_shot :
  (total_cost : ℚ) / (number_of_dogs * puppies_per_dog * shots_per_puppy) = 5 := by
  sorry

end cost_per_shot_l806_80661


namespace min_n_for_sqrt_27n_integer_l806_80655

theorem min_n_for_sqrt_27n_integer (n : ℕ+) (h : ∃ k : ℕ, k^2 = 27 * n) :
  ∀ m : ℕ+, (∃ j : ℕ, j^2 = 27 * m) → n ≤ m :=
by sorry

end min_n_for_sqrt_27n_integer_l806_80655


namespace slope_condition_implies_coefficient_bound_l806_80630

/-- Given two distinct points on a linear function, if the slope between them is negative, then the coefficient of x in the function is less than 1. -/
theorem slope_condition_implies_coefficient_bound
  (x₁ x₂ y₁ y₂ a : ℝ)
  (h_distinct : x₁ ≠ x₂)
  (h_on_graph₁ : y₁ = (a - 1) * x₁ + 1)
  (h_on_graph₂ : y₂ = (a - 1) * x₂ + 1)
  (h_slope_neg : (y₁ - y₂) / (x₁ - x₂) < 0) :
  a < 1 := by
  sorry

end slope_condition_implies_coefficient_bound_l806_80630


namespace line_segment_parameterization_l806_80673

theorem line_segment_parameterization (m n p q : ℝ) : 
  (∃ (t : ℝ), -1 ≤ t ∧ t ≤ 1 ∧ 
    1 = m * (-1) + n ∧ 
    -3 = p * (-1) + q ∧
    6 = m * 1 + n ∧ 
    5 = p * 1 + q) →
  m^2 + n^2 + p^2 + q^2 = 99 := by
sorry

end line_segment_parameterization_l806_80673


namespace average_not_1380_l806_80608

def numbers : List ℕ := [1200, 1300, 1400, 1510, 1520, 1200]

def average (lst : List ℕ) : ℚ :=
  (lst.sum : ℚ) / lst.length

theorem average_not_1380 : average numbers ≠ 1380 := by
  sorry

end average_not_1380_l806_80608


namespace sequence_general_term_l806_80697

/-- Given a sequence a_n with sum S_n satisfying S_n = 3 - 2a_n,
    prove that the general term of a_n is (2/3)^(n-1) -/
theorem sequence_general_term (a : ℕ → ℚ) (S : ℕ → ℚ)
    (h : ∀ n, S n = 3 - 2 * a n) :
  ∀ n, a n = (2/3)^(n-1) := by
sorry

end sequence_general_term_l806_80697


namespace final_expression_l806_80638

theorem final_expression (b : ℚ) : ((3 * b + 6) - 5 * b) / 3 = -2/3 * b + 2 := by
  sorry

end final_expression_l806_80638


namespace zero_point_condition_l806_80631

theorem zero_point_condition (a : ℝ) : 
  (∃ x ∈ Set.Ioo (-1 : ℝ) 1, 3 * a * x + 1 - 2 * a = 0) ↔ 
  (a < -1 ∨ a > 1/5) := by sorry

end zero_point_condition_l806_80631


namespace zara_age_l806_80668

def guesses : List Nat := [26, 29, 31, 34, 37, 39, 42, 45, 47, 50, 52]

def is_prime (n : Nat) : Prop := Nat.Prime n

def more_than_half_low (age : Nat) : Prop :=
  (guesses.filter (· < age)).length > guesses.length / 2

def three_off_by_one (age : Nat) : Prop :=
  (guesses.filter (fun x => x = age - 1 ∨ x = age + 1)).length = 3

theorem zara_age : ∃! age : Nat, 
  age ∈ guesses ∧
  is_prime age ∧
  more_than_half_low age ∧
  three_off_by_one age ∧
  age = 47 :=
sorry

end zara_age_l806_80668


namespace min_value_of_f_range_of_t_l806_80688

-- Define the function f
def f (x : ℝ) : ℝ := |x + 2| + |x - 4|

-- Theorem for the minimum value of f
theorem min_value_of_f : ∀ x : ℝ, f x ≥ 6 ∧ ∃ x₀ : ℝ, f x₀ = 6 := by sorry

-- Define the set A
def A (t : ℝ) : Set ℝ := {x | f x ≤ t^2 - t}

-- Define the set B
def B : Set ℝ := {x | -3 ≤ x ∧ x ≤ 5}

-- Theorem for the range of t
theorem range_of_t : ∀ t : ℝ, (A t ∩ B).Nonempty ↔ t ≤ -2 ∨ t ≥ 3 := by sorry

end min_value_of_f_range_of_t_l806_80688


namespace min_odd_integers_l806_80635

theorem min_odd_integers (a b c d e f : ℤ) 
  (sum1 : a + b = 28)
  (sum2 : a + b + c + d = 46)
  (sum3 : a + b + c + d + e + f = 66) :
  ∃ (a' b' c' d' e' f' : ℤ), 
    (a' + b' = 28) ∧ 
    (a' + b' + c' + d' = 46) ∧ 
    (a' + b' + c' + d' + e' + f' = 66) ∧
    (Even a') ∧ (Even b') ∧ (Even c') ∧ (Even d') ∧ (Even e') ∧ (Even f') :=
by
  sorry

end min_odd_integers_l806_80635


namespace female_officers_count_l806_80676

theorem female_officers_count (total_on_duty : ℕ) (female_on_duty_ratio : ℚ) (female_duty_percentage : ℚ) :
  total_on_duty = 360 →
  female_on_duty_ratio = 1/2 →
  female_duty_percentage = 3/5 →
  (↑total_on_duty * female_on_duty_ratio / female_duty_percentage : ℚ) = 300 := by
  sorry

end female_officers_count_l806_80676


namespace average_price_per_book_l806_80677

theorem average_price_per_book (books_shop1 : ℕ) (price_shop1 : ℕ) 
  (books_shop2 : ℕ) (price_shop2 : ℕ) 
  (h1 : books_shop1 = 65) (h2 : price_shop1 = 1380) 
  (h3 : books_shop2 = 55) (h4 : price_shop2 = 900) : 
  (price_shop1 + price_shop2) / (books_shop1 + books_shop2) = 19 := by
  sorry

end average_price_per_book_l806_80677


namespace sunday_price_calculation_l806_80603

def original_price : ℝ := 250
def regular_discount : ℝ := 0.4
def sunday_discount : ℝ := 0.25

theorem sunday_price_calculation : 
  original_price * (1 - regular_discount) * (1 - sunday_discount) = 112.5 := by
  sorry

end sunday_price_calculation_l806_80603


namespace algebraic_expression_evaluation_l806_80613

theorem algebraic_expression_evaluation (a b : ℤ) (h1 : a = 1) (h2 : b = -1) :
  a + 2*b + 2*(a + 2*b) + 1 = -2 := by sorry

end algebraic_expression_evaluation_l806_80613


namespace certain_number_power_l806_80625

theorem certain_number_power (k : ℕ) (h : k = 11) :
  (1/2)^22 * (1/81)^k = (1/354294)^22 := by
  sorry

end certain_number_power_l806_80625


namespace tangent_line_properties_l806_80607

/-- Parabola C: x^2 = 4y with focus F(0, 1) -/
structure Parabola where
  C : ℝ → ℝ
  F : ℝ × ℝ
  h : C = fun x ↦ (x^2) / 4
  focus : F = (0, 1)

/-- Line through P(a, -2) forming tangents to C at A(x₁, y₁) and B(x₂, y₂) -/
structure TangentLine (C : Parabola) where
  a : ℝ
  P : ℝ × ℝ
  A : ℝ × ℝ
  B : ℝ × ℝ
  h₁ : P = (a, -2)
  h₂ : A.2 = C.C A.1
  h₃ : B.2 = C.C B.1

/-- Circumcenter of triangle PAB -/
def circumcenter (C : Parabola) (L : TangentLine C) : ℝ × ℝ := sorry

/-- Main theorem -/
theorem tangent_line_properties (C : Parabola) (L : TangentLine C) :
  let (x₁, y₁) := L.A
  let (x₂, y₂) := L.B
  let M := circumcenter C L
  (x₁ * x₂ + y₁ * y₂ = -4) ∧
  (∃ r : ℝ, (M.1 - C.F.1)^2 + (M.2 - C.F.2)^2 = r^2 ∧
            (L.P.1 - C.F.1)^2 + (L.P.2 - C.F.2)^2 = r^2) := by
  sorry

end tangent_line_properties_l806_80607


namespace complex_number_problem_l806_80620

theorem complex_number_problem (α β : ℂ) :
  (α - β).re > 0 →
  (2 * Complex.I * (α + β)).re > 0 →
  β = 4 + Complex.I →
  α = -4 + Complex.I :=
by
  sorry

end complex_number_problem_l806_80620


namespace test_scores_l806_80629

theorem test_scores (scores : Finset ℕ) (petya_score : ℕ) : 
  scores.card = 7317 →
  (∀ (x y : ℕ), x ∈ scores → y ∈ scores → x ≠ y) →
  (∀ (x y z : ℕ), x ∈ scores → y ∈ scores → z ∈ scores → x < y + z) →
  petya_score ∈ scores →
  petya_score > 15 :=
by sorry

end test_scores_l806_80629


namespace sector_area_with_diameter_4_and_angle_90_l806_80651

theorem sector_area_with_diameter_4_and_angle_90 (π : Real) :
  let diameter : Real := 4
  let centralAngle : Real := 90
  let radius : Real := diameter / 2
  let sectorArea : Real := (centralAngle / 360) * π * radius^2
  sectorArea = π := by sorry

end sector_area_with_diameter_4_and_angle_90_l806_80651


namespace correct_mean_calculation_l806_80602

theorem correct_mean_calculation (n : ℕ) (original_mean : ℚ) (incorrect_value correct_value : ℚ) :
  n = 20 ∧ original_mean = 150 ∧ incorrect_value = 135 ∧ correct_value = 160 →
  (n * original_mean - incorrect_value + correct_value) / n = 151.25 := by
sorry

end correct_mean_calculation_l806_80602


namespace platform_length_l806_80680

/-- Given a train with speed 54 km/hr passing a platform in 32 seconds
    and passing a man standing on the platform in 20 seconds,
    prove that the length of the platform is 180 meters. -/
theorem platform_length (train_speed : ℝ) (platform_time : ℝ) (man_time : ℝ) :
  train_speed = 54 →
  platform_time = 32 →
  man_time = 20 →
  (train_speed * 1000 / 3600) * platform_time - (train_speed * 1000 / 3600) * man_time = 180 :=
by sorry

end platform_length_l806_80680


namespace no_solution_quadratic_inequality_l806_80666

theorem no_solution_quadratic_inequality :
  ¬∃ (x : ℝ), 3 * x^2 + 9 * x ≤ -12 := by sorry

end no_solution_quadratic_inequality_l806_80666


namespace min_disks_is_twelve_l806_80652

/-- Represents the number of files of each size --/
structure FileCount where
  large : Nat  -- 0.85 MB files
  medium : Nat -- 0.65 MB files
  small : Nat  -- 0.5 MB files

/-- Represents the constraints of the problem --/
structure DiskProblem where
  totalFiles : Nat
  diskCapacity : Float
  maxFilesPerDisk : Nat
  fileSizes : FileCount
  largeSizeMB : Float
  mediumSizeMB : Float
  smallSizeMB : Float

def problem : DiskProblem := {
  totalFiles := 35,
  diskCapacity := 1.44,
  maxFilesPerDisk := 4,
  fileSizes := { large := 5, medium := 15, small := 15 },
  largeSizeMB := 0.85,
  mediumSizeMB := 0.65,
  smallSizeMB := 0.5
}

/-- Calculates the minimum number of disks required --/
def minDisksRequired (p : DiskProblem) : Nat :=
  sorry -- Proof goes here

theorem min_disks_is_twelve : minDisksRequired problem = 12 := by
  sorry -- Proof goes here

end min_disks_is_twelve_l806_80652


namespace games_in_division_l806_80658

/-- Represents a baseball league with the given conditions -/
structure BaseballLeague where
  P : ℕ  -- Number of games played against each team in own division
  Q : ℕ  -- Number of games played against each team in other divisions
  p_gt_3q : P > 3 * Q
  q_gt_3 : Q > 3
  total_games : 2 * P + 6 * Q = 78

/-- Theorem stating that each team plays 54 games within its own division -/
theorem games_in_division (league : BaseballLeague) : 2 * league.P = 54 := by
  sorry

end games_in_division_l806_80658


namespace greatest_b_for_no_minus_six_l806_80634

theorem greatest_b_for_no_minus_six (b : ℤ) : 
  (∀ x : ℝ, x^2 + b*x + 20 ≠ -6) ↔ b ≤ 10 :=
by sorry

end greatest_b_for_no_minus_six_l806_80634


namespace geometric_sequence_second_term_l806_80663

theorem geometric_sequence_second_term (b : ℝ) (h1 : b > 0) :
  (∃ r : ℝ, r > 0 ∧ b = 15 * r ∧ 45/4 = b * r) → b = 15 * Real.sqrt 3 / 2 := by
  sorry

end geometric_sequence_second_term_l806_80663


namespace function_arithmetic_l806_80678

def StrictlyIncreasing (f : ℕ → ℕ) : Prop :=
  ∀ x y : ℕ, x < y → f x < f y

theorem function_arithmetic (f : ℕ → ℕ) 
  (h_increasing : StrictlyIncreasing f)
  (h_two : f 2 = 2)
  (h_coprime : ∀ m n : ℕ, Nat.Coprime m n → f (m * n) = f m * f n) :
  ∀ n : ℕ, f n = n :=
sorry

end function_arithmetic_l806_80678


namespace third_quadrant_angle_tangent_l806_80674

theorem third_quadrant_angle_tangent (α β : Real) : 
  (2 * Real.pi - Real.pi < α) ∧ (α < 2 * Real.pi - Real.pi/2) →
  (Real.sin (α + β) * Real.cos β - Real.sin β * Real.cos (α + β) = -12/13) →
  Real.tan (α/2) = -3/2 := by
  sorry

end third_quadrant_angle_tangent_l806_80674


namespace decimalRep_periodic_first_seven_digits_digit_150_l806_80642

/-- The decimal representation of 17/70 as a sequence of digits after the decimal point -/
def decimalRep : ℕ → ℕ := sorry

/-- The decimal representation of 17/70 is periodic with period 7 -/
theorem decimalRep_periodic : ∀ n : ℕ, decimalRep (n + 7) = decimalRep n := sorry

/-- The first 7 digits of the decimal representation of 17/70 -/
theorem first_seven_digits : 
  (decimalRep 0, decimalRep 1, decimalRep 2, decimalRep 3, decimalRep 4, decimalRep 5, decimalRep 6) 
  = (2, 4, 2, 8, 5, 7, 1) := sorry

/-- The 150th digit after the decimal point in the decimal representation of 17/70 is 2 -/
theorem digit_150 : decimalRep 149 = 2 := sorry

end decimalRep_periodic_first_seven_digits_digit_150_l806_80642


namespace store_sales_growth_rate_l806_80671

theorem store_sales_growth_rate 
  (initial_sales : ℝ) 
  (final_sales : ℝ) 
  (months : ℕ) 
  (h1 : initial_sales = 20000)
  (h2 : final_sales = 45000)
  (h3 : months = 2) :
  ∃ (growth_rate : ℝ), 
    growth_rate = 0.5 ∧ 
    final_sales = initial_sales * (1 + growth_rate) ^ months :=
sorry

end store_sales_growth_rate_l806_80671


namespace freshman_class_size_l806_80670

theorem freshman_class_size :
  ∃! n : ℕ, 0 < n ∧ n < 450 ∧ n % 19 = 18 ∧ n % 17 = 10 ∧ n = 265 := by
  sorry

end freshman_class_size_l806_80670


namespace fraction_equality_l806_80643

theorem fraction_equality (a b : ℚ) (h : (a - b) / b = 2 / 3) : a / b = 5 / 3 := by
  sorry

end fraction_equality_l806_80643


namespace president_and_vp_from_seven_l806_80618

/-- The number of ways to choose a President and a Vice-President from a group of n people,
    where the two positions must be held by different people. -/
def choose_president_and_vp (n : ℕ) : ℕ := n * (n - 1)

/-- Theorem: There are 42 ways to choose a President and a Vice-President from a group of 7 people,
    where the two positions must be held by different people. -/
theorem president_and_vp_from_seven : choose_president_and_vp 7 = 42 := by
  sorry

end president_and_vp_from_seven_l806_80618


namespace congruence_problem_l806_80641

theorem congruence_problem (n : ℤ) : 
  0 ≤ n ∧ n < 127 ∧ (126 * n) % 127 = 103 % 127 → n % 127 = 24 % 127 := by
  sorry

end congruence_problem_l806_80641


namespace number_of_black_marbles_l806_80633

/-- Given a bag of marbles with white and black marbles, prove the number of black marbles. -/
theorem number_of_black_marbles
  (total_marbles : ℕ)
  (white_marbles : ℕ)
  (h1 : total_marbles = 37)
  (h2 : white_marbles = 19) :
  total_marbles - white_marbles = 18 :=
by sorry

end number_of_black_marbles_l806_80633


namespace tylers_meal_combinations_l806_80609

/-- The number of meat options available --/
def meat_options : ℕ := 4

/-- The number of vegetable options available --/
def vegetable_options : ℕ := 5

/-- The number of dessert options available --/
def dessert_options : ℕ := 5

/-- The number of vegetables Tyler must choose --/
def vegetables_to_choose : ℕ := 3

/-- The number of desserts Tyler must choose --/
def desserts_to_choose : ℕ := 2

/-- The number of unique meal combinations Tyler can choose --/
def unique_meals : ℕ := meat_options * (Nat.choose vegetable_options vegetables_to_choose) * (Nat.choose dessert_options desserts_to_choose)

theorem tylers_meal_combinations :
  unique_meals = 400 :=
sorry

end tylers_meal_combinations_l806_80609


namespace missing_sale_is_1000_l806_80648

/-- Calculates the missing sale amount given the sales for 5 months and the average sale for 6 months -/
def calculate_missing_sale (sale1 sale2 sale4 sale5 sale6 average : ℕ) : ℕ :=
  6 * average - (sale1 + sale2 + sale4 + sale5 + sale6)

/-- Theorem stating that given the specific sales and average, the missing sale must be 1000 -/
theorem missing_sale_is_1000 :
  calculate_missing_sale 800 900 700 800 900 850 = 1000 := by
  sorry

end missing_sale_is_1000_l806_80648


namespace no_solution_exists_l806_80675

theorem no_solution_exists : ¬ ∃ (a : ℝ), 
  ({0, 1} : Set ℝ) ∩ ({11 - a, Real.log a, 2^a, a} : Set ℝ) = {1} := by
  sorry

end no_solution_exists_l806_80675


namespace f_periodicity_and_smallest_a_l806_80605

def is_valid_f (f : ℕ+ → ℝ) (a : ℕ+) : Prop :=
  f a = f 1995 ∧
  f (a + 1) = f 1996 ∧
  f (a + 2) = f 1997 ∧
  ∀ n : ℕ+, f (n + a) = (f n - 1) / (f n + 1)

theorem f_periodicity_and_smallest_a :
  ∃ (f : ℕ+ → ℝ) (a : ℕ+),
    is_valid_f f a ∧
    (∀ n : ℕ+, f (n + 4 * a) = f n) ∧
    (∀ a' : ℕ+, a' < a → ¬ is_valid_f f a') :=
  sorry

end f_periodicity_and_smallest_a_l806_80605


namespace a_profit_share_is_3750_l806_80632

/-- Calculates the share of profit for an investor in a partnership business -/
def calculate_profit_share (investment_a investment_b investment_c total_profit : ℚ) : ℚ :=
  (investment_a / (investment_a + investment_b + investment_c)) * total_profit

/-- Theorem: Given the investments and total profit, A's share of the profit is 3750 -/
theorem a_profit_share_is_3750 
  (investment_a : ℚ) 
  (investment_b : ℚ) 
  (investment_c : ℚ) 
  (total_profit : ℚ) 
  (h1 : investment_a = 6300)
  (h2 : investment_b = 4200)
  (h3 : investment_c = 10500)
  (h4 : total_profit = 12500) :
  calculate_profit_share investment_a investment_b investment_c total_profit = 3750 := by
  sorry

#eval calculate_profit_share 6300 4200 10500 12500

end a_profit_share_is_3750_l806_80632


namespace books_read_proof_l806_80640

def total_books (megan_books kelcie_books greg_books : ℕ) : ℕ :=
  megan_books + kelcie_books + greg_books

theorem books_read_proof (megan_books : ℕ) 
  (h1 : megan_books = 32)
  (h2 : ∃ kelcie_books : ℕ, kelcie_books = megan_books / 4)
  (h3 : ∃ greg_books : ℕ, greg_books = 2 * (megan_books / 4) + 9) :
  ∃ total : ℕ, total_books megan_books (megan_books / 4) (2 * (megan_books / 4) + 9) = 65 := by
  sorry

end books_read_proof_l806_80640


namespace thousand_factorization_sum_l806_80682

/-- Checks if a positive integer contains zero in its decimal representation -/
def containsZero (n : Nat) : Bool :=
  n.repr.contains '0'

/-- Theorem stating the existence of two positive integers satisfying the given conditions -/
theorem thousand_factorization_sum :
  ∃ (a b : Nat), a * b = 1000 ∧ ¬containsZero a ∧ ¬containsZero b ∧ a + b = 133 := by
  sorry

end thousand_factorization_sum_l806_80682


namespace sequence_sum_proof_l806_80614

def sequence_sum (n : ℕ) : ℚ := -(n + 1 : ℚ) / (n + 2 : ℚ)

theorem sequence_sum_proof (n : ℕ) :
  let a : ℕ → ℚ := λ k => if k = 1 then -2/3 else sequence_sum k - sequence_sum (k-1)
  let S : ℕ → ℚ := sequence_sum
  (∀ k : ℕ, k ≥ 2 → S k + 1 / S k + 2 = a k) →
  S n = sequence_sum n :=
by sorry

end sequence_sum_proof_l806_80614


namespace parallelogram_area_l806_80600

/-- The area of a parallelogram with given dimensions -/
theorem parallelogram_area (h : ℝ) (angle : ℝ) (s : ℝ) : 
  h = 30 → angle = 60 * π / 180 → s = 15 → h * s * Real.cos angle = 225 := by
  sorry

end parallelogram_area_l806_80600


namespace monotonicity_and_extrema_of_f_l806_80601

-- Define the function f
def f (x : ℝ) : ℝ := x^3 - 3*x^2 - 9*x + 1

-- Define the derivative of f
def f' (x : ℝ) : ℝ := 3*x^2 - 6*x - 9

theorem monotonicity_and_extrema_of_f :
  (∀ x y, x < -1 → y < -1 → f x < f y) ∧ 
  (∀ x y, 3 < x → 3 < y → f x < f y) ∧
  (∀ x y, -1 < x → x < y → y < 3 → f x > f y) ∧
  (∀ ε > 0, ∃ δ > 0, ∀ x, 0 < |x - (-1)| → |x - (-1)| < δ → f x < f (-1)) ∧
  (∀ ε > 0, ∃ δ > 0, ∀ x, 0 < |x - 3| → |x - 3| < δ → f x > f 3) ∧
  f (-1) = 6 ∧
  f 3 = -26 :=
by sorry

end monotonicity_and_extrema_of_f_l806_80601


namespace range_of_a_l806_80644

theorem range_of_a (a : ℝ) : 
  (a > 0) → 
  (∀ x : ℝ, ((x - 3*a) * (x - a) < 0) → 
    ¬(x^2 - 3*x ≤ 0 ∧ x^2 - x - 2 > 0)) ∧ 
  (∃ x : ℝ, ¬(x^2 - 3*x ≤ 0 ∧ x^2 - x - 2 > 0) ∧ 
    ¬((x - 3*a) * (x - a) < 0)) ↔ 
  (0 < a ∧ a ≤ 2/3) ∨ a ≥ 3 := by
sorry

end range_of_a_l806_80644


namespace orange_segments_total_l806_80617

/-- Represents the number of orange segments each animal received -/
structure OrangeDistribution where
  siskin : ℕ
  hedgehog : ℕ
  beaver : ℕ

/-- Defines the conditions of the orange distribution problem -/
def validDistribution (d : OrangeDistribution) : Prop :=
  d.hedgehog = 2 * d.siskin ∧
  d.beaver = 5 * d.siskin ∧
  d.beaver = d.siskin + 8

/-- The theorem stating that the total number of orange segments is 16 -/
theorem orange_segments_total (d : OrangeDistribution) 
  (h : validDistribution d) : d.siskin + d.hedgehog + d.beaver = 16 := by
  sorry

#check orange_segments_total

end orange_segments_total_l806_80617


namespace best_play_win_probability_correct_l806_80626

/-- The probability of the best play winning in a contest where 2m jurors are randomly selected from 2n moms. -/
def best_play_win_probability (n m : ℕ) : ℚ :=
  let C := fun (n k : ℕ) => Nat.choose n k
  1 / (C (2*n) n * C (2*n) (2*m)) *
  (Finset.sum (Finset.range (2*m + 1)) (fun q =>
    C n q * C n (2*m - q) *
    (Finset.sum (Finset.range (min q (m-1) + 1)) (fun t =>
      C q t * C (2*n - q) (n - t)))))

/-- Theorem stating the probability of the best play winning. -/
theorem best_play_win_probability_correct (n m : ℕ) (h : 2*m ≤ n) :
  best_play_win_probability n m = 
  (1 / (Nat.choose (2*n) n * Nat.choose (2*n) (2*m))) *
  (Finset.sum (Finset.range (2*m + 1)) (fun q =>
    Nat.choose n q * Nat.choose n (2*m - q) *
    (Finset.sum (Finset.range (min q (m-1) + 1)) (fun t =>
      Nat.choose q t * Nat.choose (2*n - q) (n - t))))) :=
by sorry

end best_play_win_probability_correct_l806_80626
