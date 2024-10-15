import Mathlib

namespace NUMINAMATH_GPT_max_books_john_can_buy_l687_68796

-- Define the key variables and conditions
def johns_money : ℕ := 3745
def book_cost : ℕ := 285
def sales_tax_rate : ℚ := 0.05

-- Define the total cost per book including tax
def total_cost_per_book : ℝ := book_cost + book_cost * sales_tax_rate

-- Define the inequality problem
theorem max_books_john_can_buy : ∃ (x : ℕ), 300 * x ≤ johns_money ∧ 300 * (x + 1) > johns_money :=
by
  sorry

end NUMINAMATH_GPT_max_books_john_can_buy_l687_68796


namespace NUMINAMATH_GPT_compute_expression_value_l687_68761

-- Define the expression
def expression : ℤ := 1013^2 - 1009^2 - 1011^2 + 997^2

-- State the theorem with the required conditions and conclusions
theorem compute_expression_value : expression = -19924 := 
by 
  -- The proof steps would go here.
  sorry

end NUMINAMATH_GPT_compute_expression_value_l687_68761


namespace NUMINAMATH_GPT_line_passes_through_fixed_point_l687_68770

theorem line_passes_through_fixed_point (k : ℝ) : ∀ x y : ℝ, (y - 1 = k * (x + 2)) → (x = -2 ∧ y = 1) :=
by
  intro x y h
  sorry

end NUMINAMATH_GPT_line_passes_through_fixed_point_l687_68770


namespace NUMINAMATH_GPT_division_result_l687_68723

theorem division_result : (8900 / 6) / 4 = 370.8333 :=
by sorry

end NUMINAMATH_GPT_division_result_l687_68723


namespace NUMINAMATH_GPT_find_tax_percentage_l687_68787

-- Definitions based on given conditions
def income_total : ℝ := 58000
def income_threshold : ℝ := 40000
def tax_above_threshold_percentage : ℝ := 0.2
def total_tax : ℝ := 8000

-- Let P be the percentage taxed on the first $40,000
variable (P : ℝ)

-- Formulate the problem as a proof goal
theorem find_tax_percentage (h : total_tax = 8000) :
  P = ((total_tax - (tax_above_threshold_percentage * (income_total - income_threshold))) / income_threshold) * 100 :=
by sorry

end NUMINAMATH_GPT_find_tax_percentage_l687_68787


namespace NUMINAMATH_GPT_ratio_of_cube_sides_l687_68780

theorem ratio_of_cube_sides 
  (a b : ℝ) 
  (h : (6 * a^2) / (6 * b^2) = 49) :
  a / b = 7 :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_cube_sides_l687_68780


namespace NUMINAMATH_GPT_candy_group_size_l687_68701

-- Define the given conditions
def num_candies : ℕ := 30
def num_groups : ℕ := 10

-- Define the statement that needs to be proven
theorem candy_group_size : num_candies / num_groups = 3 := 
by 
  sorry

end NUMINAMATH_GPT_candy_group_size_l687_68701


namespace NUMINAMATH_GPT_fraction_sum_l687_68702

-- Define the fractions
def frac1: ℚ := 3/9
def frac2: ℚ := 5/12

-- The theorem statement
theorem fraction_sum : frac1 + frac2 = 3/4 := 
sorry

end NUMINAMATH_GPT_fraction_sum_l687_68702


namespace NUMINAMATH_GPT_bananas_bought_l687_68762

theorem bananas_bought (O P B : Nat) (x : Nat) 
  (h1 : P - O = B)
  (h2 : O + P = 120)
  (h3 : P = 90)
  (h4 : 60 * x + 30 * (2 * x) = 24000) : 
  x = 200 := by
  sorry

end NUMINAMATH_GPT_bananas_bought_l687_68762


namespace NUMINAMATH_GPT_focal_length_of_curve_l687_68763

theorem focal_length_of_curve : 
  (∀ θ : ℝ, ∃ x y : ℝ, x = 2 * Real.cos θ ∧ y = Real.sin θ) →
  ∃ f : ℝ, f = 2 * Real.sqrt 3 :=
by sorry

end NUMINAMATH_GPT_focal_length_of_curve_l687_68763


namespace NUMINAMATH_GPT_total_distance_of_ship_l687_68754

-- Define the conditions
def first_day_distance : ℕ := 100
def second_day_distance := 3 * first_day_distance
def third_day_distance := second_day_distance + 110
def total_distance := first_day_distance + second_day_distance + third_day_distance

-- Theorem stating that given the conditions the total distance traveled is 810 miles
theorem total_distance_of_ship :
  total_distance = 810 := by
  sorry

end NUMINAMATH_GPT_total_distance_of_ship_l687_68754


namespace NUMINAMATH_GPT_Luke_piles_of_quarters_l687_68734

theorem Luke_piles_of_quarters (Q : ℕ) (h : 6 * Q = 30) : Q = 5 :=
by
  sorry

end NUMINAMATH_GPT_Luke_piles_of_quarters_l687_68734


namespace NUMINAMATH_GPT_warehouse_box_storage_l687_68732

theorem warehouse_box_storage (S : ℝ) (h1 : (3 - 1/4) * S = 55000) : (1/4) * S = 5000 :=
by
  sorry

end NUMINAMATH_GPT_warehouse_box_storage_l687_68732


namespace NUMINAMATH_GPT_disproof_of_Alitta_l687_68709

-- Definition: A prime number
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- Definition: A number is odd
def is_odd (n : ℕ) : Prop :=
  n % 2 = 1

-- The value is a specific set of odd primes including 11
def contains (p : ℕ) : Prop :=
  p = 3 ∨ p = 5 ∨ p = 7 ∨ p = 11

-- Main statement: There exists an odd prime p in the given options such that p^2 - 2 is not a prime
theorem disproof_of_Alitta :
  ∃ p : ℕ, contains p ∧ is_prime p ∧ is_odd p ∧ ¬ is_prime (p^2 - 2) :=
by
  sorry

end NUMINAMATH_GPT_disproof_of_Alitta_l687_68709


namespace NUMINAMATH_GPT_intersections_line_segment_l687_68730

def intersects_count (a b : ℕ) (x y : ℕ) : ℕ :=
  let steps := gcd x y
  2 * (steps + 1)

theorem intersections_line_segment (x y : ℕ) (h_x : x = 501) (h_y : y = 201) :
  intersects_count 1 1 x y = 336 := by
  sorry

end NUMINAMATH_GPT_intersections_line_segment_l687_68730


namespace NUMINAMATH_GPT_consecutive_nums_sum_as_product_l687_68748

theorem consecutive_nums_sum_as_product {n : ℕ} (h : 100 < n) :
  ∃ (a b c : ℕ), (a ≠ b) ∧ (b ≠ c) ∧ (a ≠ c) ∧ (2 ≤ a) ∧ (2 ≤ b) ∧ (2 ≤ c) ∧ 
  ((n + (n+1) + (n+2) = a * b * c) ∨ ((n+1) + (n+2) + (n+3) = a * b * c)) :=
by
  sorry

end NUMINAMATH_GPT_consecutive_nums_sum_as_product_l687_68748


namespace NUMINAMATH_GPT_findSolutions_l687_68759

-- Define the given mathematical problem
def originalEquation (x : ℝ) : Prop :=
  ((x - 3) * (x - 4) * (x - 5) * (x - 6) * (x - 5) * (x - 4) * (x - 3)) / ((x - 4) * (x - 6) * (x - 4)) = 1

-- Define the conditions where the equation is valid
def validCondition (x : ℝ) : Prop :=
  x ≠ 4 ∧ x ≠ 6

-- Define the set of solutions
def solutions (x : ℝ) : Prop :=
  x = 4 + Real.sqrt 2 ∨ x = 4 - Real.sqrt 2

-- The theorem stating the correct set of solutions
theorem findSolutions (x : ℝ) : originalEquation x ∧ validCondition x ↔ solutions x :=
by sorry

end NUMINAMATH_GPT_findSolutions_l687_68759


namespace NUMINAMATH_GPT_line_through_intersections_l687_68750

def circle1 (x y : ℝ) : Prop := (x - 2)^2 + (y - 1)^2 = 4
def circle2 (x y : ℝ) : Prop := x^2 + (y - 2)^2 = 9

theorem line_through_intersections (x y : ℝ) :
  circle1 x y → circle2 x y → 2 * x - 3 * y = 0 := 
sorry

end NUMINAMATH_GPT_line_through_intersections_l687_68750


namespace NUMINAMATH_GPT_percent_of_x_is_z_l687_68742

def condition1 (z y : ℝ) : Prop := 0.45 * z = 0.72 * y
def condition2 (y x : ℝ) : Prop := y = 0.75 * x
def condition3 (w z : ℝ) : Prop := w = 0.60 * z^2
def condition4 (z w : ℝ) : Prop := z = 0.30 * w^(1/3)

theorem percent_of_x_is_z (x y z w : ℝ) 
  (h1 : condition1 z y) 
  (h2 : condition2 y x)
  (h3 : condition3 w z)
  (h4 : condition4 z w) : 
  z / x = 1.2 :=
sorry

end NUMINAMATH_GPT_percent_of_x_is_z_l687_68742


namespace NUMINAMATH_GPT_min_white_surface_area_is_five_over_ninety_six_l687_68737

noncomputable def fraction_white_surface_area (total_surface_area white_surface_area : ℕ) :=
  (white_surface_area : ℚ) / (total_surface_area : ℚ)

theorem min_white_surface_area_is_five_over_ninety_six :
  let total_surface_area := 96
  let white_surface_area := 5
  fraction_white_surface_area total_surface_area white_surface_area = 5 / 96 :=
by
  sorry

end NUMINAMATH_GPT_min_white_surface_area_is_five_over_ninety_six_l687_68737


namespace NUMINAMATH_GPT_solve_for_x_l687_68767

theorem solve_for_x : ∀ x : ℝ, (x - 27) / 3 = (3 * x + 6) / 8 → x = -234 :=
by
  intro x
  intro h
  sorry

end NUMINAMATH_GPT_solve_for_x_l687_68767


namespace NUMINAMATH_GPT_days_needed_to_wash_all_towels_l687_68726

def towels_per_hour : ℕ := 7
def hours_per_day : ℕ := 2
def total_towels : ℕ := 98

theorem days_needed_to_wash_all_towels :
  (total_towels / (towels_per_hour * hours_per_day)) = 7 :=
by
  sorry

end NUMINAMATH_GPT_days_needed_to_wash_all_towels_l687_68726


namespace NUMINAMATH_GPT_scientific_notation_35100_l687_68738

theorem scientific_notation_35100 : 35100 = 3.51 * 10^4 :=
by
  sorry

end NUMINAMATH_GPT_scientific_notation_35100_l687_68738


namespace NUMINAMATH_GPT_ratio_of_saturday_to_friday_customers_l687_68778

def tips_per_customer : ℝ := 2.0
def customers_friday : ℕ := 28
def customers_sunday : ℕ := 36
def total_tips : ℝ := 296

theorem ratio_of_saturday_to_friday_customers :
  let tips_friday := customers_friday * tips_per_customer
  let tips_sunday := customers_sunday * tips_per_customer
  let tips_friday_and_sunday := tips_friday + tips_sunday
  let tips_saturday := total_tips - tips_friday_and_sunday
  let customers_saturday := tips_saturday / tips_per_customer
  (customers_saturday / customers_friday : ℝ) = 3 := 
by
  sorry

end NUMINAMATH_GPT_ratio_of_saturday_to_friday_customers_l687_68778


namespace NUMINAMATH_GPT_cars_on_river_road_l687_68788

variable (B C M : ℕ)

theorem cars_on_river_road
  (h1 : ∃ B C : ℕ, B / C = 1 / 3) -- ratio of buses to cars is 1:3
  (h2 : ∀ B C : ℕ, C = B + 40) -- 40 fewer buses than cars
  (h3 : ∃ B C M : ℕ, B + C + M = 720) -- total number of vehicles is 720
  : C = 60 :=
sorry

end NUMINAMATH_GPT_cars_on_river_road_l687_68788


namespace NUMINAMATH_GPT_inequality_proof_l687_68765

theorem inequality_proof (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d):
    1/a + 1/b + 4/c + 16/d ≥ 64/(a + b + c + d) :=
by
  sorry

end NUMINAMATH_GPT_inequality_proof_l687_68765


namespace NUMINAMATH_GPT_total_people_large_seats_is_84_l687_68785

-- Definition of the number of large seats
def large_seats : Nat := 7

-- Definition of the number of people each large seat can hold
def people_per_large_seat : Nat := 12

-- Definition of the total number of people that can ride on large seats
def total_people_large_seats : Nat := large_seats * people_per_large_seat

-- Statement that we need to prove
theorem total_people_large_seats_is_84 : total_people_large_seats = 84 := by
  sorry

end NUMINAMATH_GPT_total_people_large_seats_is_84_l687_68785


namespace NUMINAMATH_GPT_train_crossing_time_l687_68740

theorem train_crossing_time
  (train_length : ℝ)
  (platform_length : ℝ)
  (time_to_cross_platform : ℝ)
  (train_speed : ℝ := (train_length + platform_length) / time_to_cross_platform)
  (time_to_cross_signal_pole : ℝ := train_length / train_speed) :
  train_length = 300 ∧ platform_length = 1000 ∧ time_to_cross_platform = 39 → time_to_cross_signal_pole = 9 := by
  intro h
  cases h
  sorry

end NUMINAMATH_GPT_train_crossing_time_l687_68740


namespace NUMINAMATH_GPT_average_distance_per_day_l687_68725

def distance_Monday : ℝ := 4.2
def distance_Tuesday : ℝ := 3.8
def distance_Wednesday : ℝ := 3.6
def distance_Thursday : ℝ := 4.4

def total_distance : ℝ := distance_Monday + distance_Tuesday + distance_Wednesday + distance_Thursday

def number_of_days : ℕ := 4

theorem average_distance_per_day : total_distance / number_of_days = 4 := by
  sorry

end NUMINAMATH_GPT_average_distance_per_day_l687_68725


namespace NUMINAMATH_GPT_complement_U_A_l687_68712

def U : Finset ℤ := {-2, -1, 0, 1, 2}
def A : Finset ℤ := {-2, -1, 1, 2}

theorem complement_U_A : (U \ A) = {0} := by
  sorry

end NUMINAMATH_GPT_complement_U_A_l687_68712


namespace NUMINAMATH_GPT_dave_apps_left_l687_68779

def initial_apps : ℕ := 24
def initial_files : ℕ := 9
def files_left : ℕ := 5
def apps_left (files_left: ℕ) : ℕ := files_left + 7

theorem dave_apps_left :
  apps_left files_left = 12 :=
by
  sorry

end NUMINAMATH_GPT_dave_apps_left_l687_68779


namespace NUMINAMATH_GPT_binomial_square_evaluation_l687_68776

theorem binomial_square_evaluation : 15^2 + 2 * 15 * 3 + 3^2 = 324 := by
  sorry

end NUMINAMATH_GPT_binomial_square_evaluation_l687_68776


namespace NUMINAMATH_GPT_find_m_interval_l687_68766

-- Define the sequence recursively
def sequence_recursive (x : ℕ → ℝ) (n : ℕ) : Prop :=
  x 0 = 5 ∧ ∀ n, x (n + 1) = (x n ^ 2 + 5 * x n + 4) / (x n + 6)

-- The left-hand side of the inequality
noncomputable def target_value : ℝ := 4 + 1 / (2 ^ 20)

-- The condition that the sequence element must satisfy
def condition (x : ℕ → ℝ) (m : ℕ) : Prop :=
  x m ≤ target_value

-- The proof problem statement, m lies within the given interval
theorem find_m_interval (x : ℕ → ℝ) (m : ℕ) :
  sequence_recursive x n →
  condition x m →
  81 ≤ m ∧ m ≤ 242 :=
sorry

end NUMINAMATH_GPT_find_m_interval_l687_68766


namespace NUMINAMATH_GPT_medal_ratio_l687_68739

theorem medal_ratio (total_medals : ℕ) (track_medals : ℕ) (badminton_medals : ℕ) (swimming_medals : ℕ) 
  (h1 : total_medals = 20) 
  (h2 : track_medals = 5) 
  (h3 : badminton_medals = 5) 
  (h4 : swimming_medals = total_medals - track_medals - badminton_medals) : 
  swimming_medals / track_medals = 2 := 
by 
  sorry

end NUMINAMATH_GPT_medal_ratio_l687_68739


namespace NUMINAMATH_GPT_div_condition_nat_l687_68714

theorem div_condition_nat (n : ℕ) : (n + 1) ∣ (n^2 + 1) ↔ n = 0 ∨ n = 1 :=
by
  sorry

end NUMINAMATH_GPT_div_condition_nat_l687_68714


namespace NUMINAMATH_GPT_retirement_year_l687_68753

-- Define the basic conditions
def rule_of_70 (age: ℕ) (years_of_employment: ℕ) : Prop :=
  age + years_of_employment ≥ 70

def age_in_hiring_year : ℕ := 32
def hiring_year : ℕ := 1987

theorem retirement_year : ∃ y: ℕ, rule_of_70 (age_in_hiring_year + y) y ∧ (hiring_year + y = 2006) :=
  sorry

end NUMINAMATH_GPT_retirement_year_l687_68753


namespace NUMINAMATH_GPT_factor_1_factor_2_l687_68731

theorem factor_1 {x : ℝ} : x^2 - 4*x + 3 = (x - 1) * (x - 3) :=
sorry

theorem factor_2 {x : ℝ} : 4*x^2 + 12*x - 7 = (2*x + 7) * (2*x - 1) :=
sorry

end NUMINAMATH_GPT_factor_1_factor_2_l687_68731


namespace NUMINAMATH_GPT_seashells_ratio_l687_68715

theorem seashells_ratio (s_1 s_2 S t s3 : ℕ) (hs1 : s_1 = 5) (hs2 : s_2 = 7) (hS : S = 36)
  (ht : t = s_1 + s_2) (hs3 : s3 = S - t) :
  s3 / t = 2 :=
by
  rw [hs1, hs2] at ht
  simp at ht
  rw [hS, ht] at hs3
  simp at hs3
  sorry

end NUMINAMATH_GPT_seashells_ratio_l687_68715


namespace NUMINAMATH_GPT_harkamal_purchase_mangoes_l687_68741

variable (m : ℕ)

def cost_of_grapes (cost_per_kg grapes_weight : ℕ) : ℕ := cost_per_kg * grapes_weight
def cost_of_mangoes (cost_per_kg mangoes_weight : ℕ) : ℕ := cost_per_kg * mangoes_weight

theorem harkamal_purchase_mangoes :
  (cost_of_grapes 70 10 + cost_of_mangoes 55 m = 1195) → m = 9 :=
by
  sorry

end NUMINAMATH_GPT_harkamal_purchase_mangoes_l687_68741


namespace NUMINAMATH_GPT_cost_price_of_article_l687_68700

theorem cost_price_of_article (SP : ℝ) (profit_percent : ℝ) (CP : ℝ) 
    (h1 : SP = 100) 
    (h2 : profit_percent = 0.20) 
    (h3 : SP = CP * (1 + profit_percent)) : 
    CP = 83.33 :=
by
  sorry

end NUMINAMATH_GPT_cost_price_of_article_l687_68700


namespace NUMINAMATH_GPT_EH_length_l687_68758

structure Rectangle :=
(AB BC CD DA : ℝ)
(horiz: AB=CD)
(verti: BC=DA)
(diag_eq: (AB^2 + BC^2) = (CD^2 + DA^2))

structure Point :=
(x y : ℝ)

noncomputable def H_distance (E D : Point)
    (AB BC : ℝ) : ℝ :=
    (E.y - D.y) -- if we consider D at origin (0,0)

theorem EH_length
    (AB BC : ℝ)
    (H_dist : ℝ)
    (E : Point)
    (rectangle : Rectangle) :
    AB = 50 →
    BC = 60 →
    E.x^2 + BC^2 = 30^2 + 60^2 →
    E.y = 40 →
    H_dist = E.y - CD →
    H_dist = 7.08 :=
by
    sorry

end NUMINAMATH_GPT_EH_length_l687_68758


namespace NUMINAMATH_GPT_fruit_seller_loss_percentage_l687_68771

theorem fruit_seller_loss_percentage :
  ∃ (C : ℝ), 
    (5 : ℝ) = C - (6.25 - C * (1 + 0.05)) → 
    (C = 6.25) → 
    (C - 5 = 1.25) → 
    (1.25 / 6.25 * 100 = 20) :=
by 
  sorry

end NUMINAMATH_GPT_fruit_seller_loss_percentage_l687_68771


namespace NUMINAMATH_GPT_binomial_square_solution_l687_68704

variable (t u b : ℝ)

theorem binomial_square_solution (h1 : 2 * t * u = 12) (h2 : u^2 = 9) : b = t^2 → b = 4 :=
by
  sorry

end NUMINAMATH_GPT_binomial_square_solution_l687_68704


namespace NUMINAMATH_GPT_problem_statement_l687_68764

variable {a x y : ℝ}

theorem problem_statement (hx : 0 < a) (ha : a < 1) (h : a^x < a^y) : x^3 > y^3 :=
sorry

end NUMINAMATH_GPT_problem_statement_l687_68764


namespace NUMINAMATH_GPT_quadratic_real_roots_m_range_l687_68781

theorem quadratic_real_roots_m_range :
  ∀ (m : ℝ), (∃ x : ℝ, x^2 + 4*x + m + 5 = 0) ↔ m ≤ -1 :=
by sorry

end NUMINAMATH_GPT_quadratic_real_roots_m_range_l687_68781


namespace NUMINAMATH_GPT_prob_of_king_or_queen_top_l687_68721

/-- A standard deck comprises 52 cards, with 13 ranks and 4 suits, each rank having one card per suit. -/
def standard_deck : Set (String × String) :=
Set.prod { "Ace", "2", "3", "4", "5", "6", "7", "8", "9", "10", "Jack", "Queen", "King" }
          { "Hearts", "Diamonds", "Clubs", "Spades" }

/-- There are four cards of rank King and four of rank Queen in the standard deck. -/
def count_kings_and_queens : Nat := 
4 + 4

/-- The total number of cards in a standard deck is 52. -/
def total_cards : Nat := 52

/-- The probability that the top card is either a King or a Queen is 2/13. -/
theorem prob_of_king_or_queen_top :
  (count_kings_and_queens / total_cards : ℚ) = (2 / 13 : ℚ) :=
sorry

end NUMINAMATH_GPT_prob_of_king_or_queen_top_l687_68721


namespace NUMINAMATH_GPT_max_expr_under_condition_l687_68798

-- Define the conditions and variables
variable {x : ℝ}

-- State the theorem about the maximum value of the given expression under the given condition
theorem max_expr_under_condition (h : x < -3) : 
  ∃ M, M = -2 * Real.sqrt 2 - 3 ∧ ∀ y, y < -3 → y + 2 / (y + 3) ≤ M :=
sorry

end NUMINAMATH_GPT_max_expr_under_condition_l687_68798


namespace NUMINAMATH_GPT_part_a_l687_68736

theorem part_a 
  (x y u v : ℝ) 
  (h1 : x + y = u + v) 
  (h2 : x^2 + y^2 = u^2 + v^2) : 
  ∀ n : ℕ, x^n + y^n = u^n + v^n := 
by sorry

end NUMINAMATH_GPT_part_a_l687_68736


namespace NUMINAMATH_GPT_output_of_code_snippet_is_six_l687_68711

-- Define the variables and the condition
def a : ℕ := 3
def y : ℕ := if a < 10 then 2 * a else a * a 

-- The statement to be proved
theorem output_of_code_snippet_is_six :
  y = 6 :=
by
  sorry

end NUMINAMATH_GPT_output_of_code_snippet_is_six_l687_68711


namespace NUMINAMATH_GPT_area_of_park_l687_68716

-- Definitions of conditions
def ratio_length_breadth (L B : ℝ) : Prop := L / B = 1 / 3
def cycling_time_distance (speed time perimeter : ℝ) : Prop := perimeter = speed * time

theorem area_of_park :
  ∃ (L B : ℝ),
    ratio_length_breadth L B ∧
    cycling_time_distance 12 (8 / 60) (2 * (L + B)) ∧
    L * B = 120000 := by
  sorry

end NUMINAMATH_GPT_area_of_park_l687_68716


namespace NUMINAMATH_GPT_monkeys_more_than_giraffes_l687_68718

theorem monkeys_more_than_giraffes :
  let zebras := 12
  let camels := zebras / 2
  let monkeys := 4 * camels
  let giraffes := 2
  monkeys - giraffes = 22
:= by
  intros
  let zebras := 12
  let camels := zebras / 2
  let monkeys := 4 * camels
  let giraffes := 2
  have h := monkeys - giraffes
  exact sorry

end NUMINAMATH_GPT_monkeys_more_than_giraffes_l687_68718


namespace NUMINAMATH_GPT_solution_l687_68733

-- Define the vectors and their conditions
variables {u v : ℝ}

def vec1 := (3, -2)
def vec2 := (9, -7)
def vec3 := (-1, 2)
def vec4 := (-3, 4)

-- Condition: The linear combination of vec1 and u*vec2 equals the linear combination of vec3 and v*vec4.
axiom H : (3 + 9 * u, -2 - 7 * u) = (-1 - 3 * v, 2 + 4 * v)

-- Statement of the proof problem:
theorem solution : u = -4/15 ∧ v = -8/15 :=
by {
  sorry
}

end NUMINAMATH_GPT_solution_l687_68733


namespace NUMINAMATH_GPT_no_sum_14_l687_68786

theorem no_sum_14 (x y : ℤ) (h : x * y + 4 = 40) : x + y ≠ 14 :=
by sorry

end NUMINAMATH_GPT_no_sum_14_l687_68786


namespace NUMINAMATH_GPT_jaya_rank_from_bottom_l687_68792

theorem jaya_rank_from_bottom (n t : ℕ) (h_n : n = 53) (h_t : t = 5) : n - t + 1 = 50 := by
  sorry

end NUMINAMATH_GPT_jaya_rank_from_bottom_l687_68792


namespace NUMINAMATH_GPT_abs_h_eq_2_l687_68768

-- Definitions based on the given conditions
def sum_of_squares_of_roots (h : ℝ) : Prop :=
  let a := 1
  let b := -4 * h
  let c := -8
  let sum_of_roots := -b / a
  let prod_of_roots := c / a
  let sum_of_squares := sum_of_roots^2 - 2 * prod_of_roots
  sum_of_squares = 80

-- Theorem to prove the absolute value of h is 2
theorem abs_h_eq_2 (h : ℝ) (h_condition : sum_of_squares_of_roots h) : |h| = 2 :=
by
  sorry

end NUMINAMATH_GPT_abs_h_eq_2_l687_68768


namespace NUMINAMATH_GPT_height_relationship_height_at_90_l687_68703

noncomputable def f (x : ℝ) : ℝ := (1/2) * x

theorem height_relationship :
  (∀ x : ℝ, (x = 10 -> f x = 5) ∧ (x = 30 -> f x = 15) ∧ (x = 50 -> f x = 25) ∧ (x = 70 -> f x = 35)) → (∀ x : ℝ, f x = (1/2) * x) :=
by
  sorry

theorem height_at_90 :
  f 90 = 45 :=
by
  sorry

end NUMINAMATH_GPT_height_relationship_height_at_90_l687_68703


namespace NUMINAMATH_GPT_fraction_is_one_fifth_l687_68749

theorem fraction_is_one_fifth
  (x a b : ℤ)
  (hx : x^2 = 25)
  (h2x : 2 * x = a * x / b + 9) :
  a = 1 ∧ b = 5 :=
by
  sorry

end NUMINAMATH_GPT_fraction_is_one_fifth_l687_68749


namespace NUMINAMATH_GPT_player_matches_average_increase_l687_68773

theorem player_matches_average_increase 
  (n T : ℕ) 
  (h1 : T = 32 * n) 
  (h2 : (T + 76) / (n + 1) = 36) : 
  n = 10 := 
by 
  sorry

end NUMINAMATH_GPT_player_matches_average_increase_l687_68773


namespace NUMINAMATH_GPT_expression_equivalence_l687_68710

theorem expression_equivalence : (2 / 20) + (3 / 30) + (4 / 40) + (5 / 50) = 0.4 := by
  sorry

end NUMINAMATH_GPT_expression_equivalence_l687_68710


namespace NUMINAMATH_GPT_externally_tangent_intersect_two_points_l687_68794

def circle1 (x y : ℝ) : Prop := x^2 + y^2 - 2*x - 2*y + 1 = 0
def circle2 (x y r : ℝ) : Prop := (x - 4)^2 + (y - 5)^2 = r^2 ∧ r > 0

theorem externally_tangent (r : ℝ) : 
  (∃ x y : ℝ, circle2 x y r) →
  (∃ x y : ℝ, circle1 x y) → 
  (dist (1, 1) (4, 5) = r + 1) → 
  r = 4 := 
sorry

theorem intersect_two_points (r : ℝ) : 
  (∃ x y : ℝ, circle2 x y r) → 
  (∃ x y : ℝ, circle1 x y) → 
  (|r - 1| < dist (1, 1) (4, 5) ∧ dist (1, 1) (4, 5) < r + 1) → 
  4 < r ∧ r < 6 :=
sorry

end NUMINAMATH_GPT_externally_tangent_intersect_two_points_l687_68794


namespace NUMINAMATH_GPT_original_remainder_when_dividing_by_44_is_zero_l687_68797

theorem original_remainder_when_dividing_by_44_is_zero 
  (N R : ℕ) 
  (Q : ℕ) 
  (h1 : N = 44 * 432 + R) 
  (h2 : N = 34 * Q + 2) 
  : R = 0 := 
sorry

end NUMINAMATH_GPT_original_remainder_when_dividing_by_44_is_zero_l687_68797


namespace NUMINAMATH_GPT_at_least_one_is_one_l687_68713

theorem at_least_one_is_one (x y z : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0):
  (1/x + 1/y + 1/z = 1) → (1/(x + y + z) = 1) → (x = 1 ∨ y = 1 ∨ z = 1) :=
by
  sorry

end NUMINAMATH_GPT_at_least_one_is_one_l687_68713


namespace NUMINAMATH_GPT_original_number_of_laborers_l687_68755

theorem original_number_of_laborers (L : ℕ) 
  (h : L * 9 = (L - 6) * 15) : L = 15 :=
sorry

end NUMINAMATH_GPT_original_number_of_laborers_l687_68755


namespace NUMINAMATH_GPT_smallest_number_of_rectangles_l687_68782

-- Defining the given problem conditions
def rectangle_area : ℕ := 3 * 4
def smallest_square_side_length : ℕ := 12

-- Lean 4 statement to prove the problem
theorem smallest_number_of_rectangles 
    (h : ∃ n : ℕ, n * n = smallest_square_side_length * smallest_square_side_length)
    (h1 : ∃ m : ℕ, m * rectangle_area = smallest_square_side_length * smallest_square_side_length) :
    m = 9 :=
by
  sorry

end NUMINAMATH_GPT_smallest_number_of_rectangles_l687_68782


namespace NUMINAMATH_GPT_units_digit_of_power_l687_68760

theorem units_digit_of_power (a b : ℕ) : (a % 10 = 7) → (b % 4 = 0) → ((a^b) % 10 = 1) :=
by
  intros
  sorry

end NUMINAMATH_GPT_units_digit_of_power_l687_68760


namespace NUMINAMATH_GPT_Megan_deleted_files_l687_68735

theorem Megan_deleted_files (initial_files folders files_per_folder deleted_files : ℕ) 
    (h1 : initial_files = 93) 
    (h2 : folders = 9)
    (h3 : files_per_folder = 8) 
    (h4 : deleted_files = initial_files - folders * files_per_folder) : 
  deleted_files = 21 :=
by
  sorry

end NUMINAMATH_GPT_Megan_deleted_files_l687_68735


namespace NUMINAMATH_GPT_g_minus_one_eq_zero_l687_68745

def g (x r : ℝ) : ℝ := 3 * x^3 - 2 * x^2 + 4 * x - 5 + r

theorem g_minus_one_eq_zero (r : ℝ) : g (-1) r = 0 → r = 14 := by
  sorry

end NUMINAMATH_GPT_g_minus_one_eq_zero_l687_68745


namespace NUMINAMATH_GPT_remainder_division_l687_68793

theorem remainder_division :
  ∃ N R1 Q2, N = 44 * 432 + R1 ∧ N = 30 * Q2 + 18 ∧ R1 < 44 ∧ 18 = R1 :=
by
  sorry

end NUMINAMATH_GPT_remainder_division_l687_68793


namespace NUMINAMATH_GPT_ratio_a_d_l687_68706

theorem ratio_a_d 
  (a b c d : ℕ) 
  (h1 : a / b = 1 / 4) 
  (h2 : b / c = 13 / 9) 
  (h3 : c / d = 5 / 13) : 
  a / d = 5 / 36 :=
sorry

end NUMINAMATH_GPT_ratio_a_d_l687_68706


namespace NUMINAMATH_GPT_students_not_yes_for_either_subject_l687_68784

variable (total_students yes_m no_m unsure_m yes_r no_r unsure_r yes_only_m : ℕ)

theorem students_not_yes_for_either_subject :
  total_students = 800 →
  yes_m = 500 →
  no_m = 200 →
  unsure_m = 100 →
  yes_r = 400 →
  no_r = 100 →
  unsure_r = 300 →
  yes_only_m = 150 →
  ∃ students_not_yes, students_not_yes = total_students - (yes_only_m + (yes_m - yes_only_m) + (yes_r - (yes_m - yes_only_m))) ∧ students_not_yes = 400 :=
by
  intros ht yt1 nnm um ypr ynr ur yom
  sorry

end NUMINAMATH_GPT_students_not_yes_for_either_subject_l687_68784


namespace NUMINAMATH_GPT_doors_per_apartment_l687_68752

def num_buildings : ℕ := 2
def num_floors_per_building : ℕ := 12
def num_apt_per_floor : ℕ := 6
def total_num_doors : ℕ := 1008

theorem doors_per_apartment : total_num_doors / (num_buildings * num_floors_per_building * num_apt_per_floor) = 7 :=
by
  sorry

end NUMINAMATH_GPT_doors_per_apartment_l687_68752


namespace NUMINAMATH_GPT_smallest_b_factors_l687_68790

theorem smallest_b_factors (b p q : ℤ) (H : p * q = 2016) : 
  (∀ k₁ k₂ : ℤ, k₁ * k₂ = 2016 → k₁ + k₂ ≥ p + q) → 
  b = 90 :=
by
  -- Here, we assume the premises stated for integers p, q such that their product is 2016.
  -- We need to fill in the proof steps which will involve checking all appropriate (p, q) pairs.
  sorry

end NUMINAMATH_GPT_smallest_b_factors_l687_68790


namespace NUMINAMATH_GPT_Andy_earnings_l687_68775

/-- Andy's total earnings during an 8-hour shift. --/
theorem Andy_earnings (hours : ℕ) (hourly_wage : ℕ) (num_racquets : ℕ) (pay_per_racquet : ℕ)
  (num_grommets : ℕ) (pay_per_grommet : ℕ) (num_stencils : ℕ) (pay_per_stencil : ℕ)
  (h_shift : hours = 8) (h_hourly : hourly_wage = 9) (h_racquets : num_racquets = 7)
  (h_pay_racquets : pay_per_racquet = 15) (h_grommets : num_grommets = 2)
  (h_pay_grommets : pay_per_grommet = 10) (h_stencils : num_stencils = 5)
  (h_pay_stencils : pay_per_stencil = 1) :
  (hours * hourly_wage + num_racquets * pay_per_racquet + num_grommets * pay_per_grommet +
  num_stencils * pay_per_stencil) = 202 :=
by
  sorry

end NUMINAMATH_GPT_Andy_earnings_l687_68775


namespace NUMINAMATH_GPT_solve_for_x_l687_68769

theorem solve_for_x : ∀ (x : ℝ), (-3 * x - 8 = 5 * x + 4) → (x = -3 / 2) := by
  intro x
  intro h
  sorry

end NUMINAMATH_GPT_solve_for_x_l687_68769


namespace NUMINAMATH_GPT_hectors_sibling_product_l687_68707

theorem hectors_sibling_product (sisters : Nat) (brothers : Nat) (helen : Nat -> Prop): 
  (helen 4) → (helen 7) → (helen 5) → (helen 6) →
  (sisters + 1 = 5) → (brothers + 1 = 7) → ((sisters * brothers) = 30) :=
by
  sorry

end NUMINAMATH_GPT_hectors_sibling_product_l687_68707


namespace NUMINAMATH_GPT_container_capacity_l687_68774

theorem container_capacity (C : ℝ) (h1 : 0.30 * C + 36 = 0.75 * C) : C = 80 :=
by
  sorry

end NUMINAMATH_GPT_container_capacity_l687_68774


namespace NUMINAMATH_GPT_tan_alpha_l687_68799

theorem tan_alpha (α : ℝ) (hα1 : α > π / 2) (hα2 : α < π) (h_sin : Real.sin α = 4 / 5) : Real.tan α = - (4 / 3) :=
by 
  sorry

end NUMINAMATH_GPT_tan_alpha_l687_68799


namespace NUMINAMATH_GPT_fg_eq_gf_condition_l687_68783

theorem fg_eq_gf_condition (m n p q : ℝ) (f g : ℝ → ℝ)
  (hf : ∀ x, f x = m * x + n) 
  (hg : ∀ x, g x = p * x + q) : 
  (∀ x, f (g x) = g (f x)) ↔ n * (1 - p) = q * (1 - m) := 
sorry

end NUMINAMATH_GPT_fg_eq_gf_condition_l687_68783


namespace NUMINAMATH_GPT_proof_of_problem_l687_68729

noncomputable def f : ℝ → ℝ := sorry  -- define f as a function in ℝ to ℝ

theorem proof_of_problem 
  (h_odd : ∀ x : ℝ, f (-x) = -f x)
  (h_f1 : f 1 = 1)
  (h_periodic : ∀ x : ℝ, f (x + 6) = f x + f 3) :
  f 2015 + f 2016 = -1 := 
sorry

end NUMINAMATH_GPT_proof_of_problem_l687_68729


namespace NUMINAMATH_GPT_solve_inequality_l687_68791

theorem solve_inequality (a : ℝ) :
  (a = 0 → {x : ℝ | x ≤ -1} = {x : ℝ | ax^2 - 2 ≥ 2 * x - a * x}) ∧
  (a > 0 → {x : ℝ | x ≥ 2 / a} ∪ {x : ℝ | x ≤ -1} = {x : ℝ | ax^2 - 2 ≥ 2 * x - a * x}) ∧
  (-2 < a ∧ a < 0 → {x : ℝ | 2 / a ≤ x ∧ x ≤ -1} = {x : ℝ | ax^2 - 2 ≥ 2 * x - a * x}) ∧
  (a = -2 → {x : ℝ | x = -1} = {x : ℝ | ax^2 - 2 ≥ 2 * x - a * x}) ∧
  (a < -2 → {x : ℝ | -1 ≤ x ∧ x ≤ 2 / a} = {x : ℝ | ax^2 - 2 ≥ 2 * x - a * x}) :=
by 
  sorry

end NUMINAMATH_GPT_solve_inequality_l687_68791


namespace NUMINAMATH_GPT_scientific_notation_correct_l687_68717

noncomputable def scientific_notation_139000 : Prop :=
  139000 = 1.39 * 10^5

theorem scientific_notation_correct : scientific_notation_139000 :=
by
  -- The proof would be included here, but we add sorry to skip it
  sorry

end NUMINAMATH_GPT_scientific_notation_correct_l687_68717


namespace NUMINAMATH_GPT_floor_neg_seven_fourths_l687_68708

theorem floor_neg_seven_fourths : Int.floor (-7 / 4 : ℚ) = -2 := 
by 
  sorry

end NUMINAMATH_GPT_floor_neg_seven_fourths_l687_68708


namespace NUMINAMATH_GPT_ferris_wheel_time_10_seconds_l687_68722

noncomputable def time_to_reach_height (R : ℝ) (T : ℝ) (h : ℝ) : ℝ :=
  let ω := 2 * Real.pi / T
  let t := (Real.arcsin (h / R - 1)) / ω
  t

theorem ferris_wheel_time_10_seconds :
  time_to_reach_height 30 120 15 = 10 :=
by
  sorry

end NUMINAMATH_GPT_ferris_wheel_time_10_seconds_l687_68722


namespace NUMINAMATH_GPT_apples_per_hour_l687_68757

def total_apples : ℕ := 15
def hours : ℕ := 3

theorem apples_per_hour : total_apples / hours = 5 := by
  sorry

end NUMINAMATH_GPT_apples_per_hour_l687_68757


namespace NUMINAMATH_GPT_football_team_total_members_l687_68728

-- Definitions from the problem conditions
def initialMembers : ℕ := 42
def newMembers : ℕ := 17

-- Mathematical equivalent proof problem
theorem football_team_total_members : initialMembers + newMembers = 59 := by
  sorry

end NUMINAMATH_GPT_football_team_total_members_l687_68728


namespace NUMINAMATH_GPT_derivative_of_f_l687_68720

noncomputable def f (x : ℝ) : ℝ := x * Real.cos x - Real.sin x

theorem derivative_of_f : ∀ x : ℝ, deriv f x = -x * Real.sin x := by
  sorry

end NUMINAMATH_GPT_derivative_of_f_l687_68720


namespace NUMINAMATH_GPT_three_numbers_lcm_ratio_l687_68777

theorem three_numbers_lcm_ratio
  (x : ℕ)
  (h1 : 3 * x.gcd 4 = 1)
  (h2 : (3 * x * 4 * x) / x.gcd (3 * x) = 180)
  (h3 : ∃ y : ℕ, y = 5 * (3 * x))
  : (3 * x = 45 ∧ 4 * x = 60 ∧ 5 * (3 * x) = 225) ∧
      lcm (lcm (3 * x) (4 * x)) (5 * (3 * x)) = 900 :=
by
  sorry

end NUMINAMATH_GPT_three_numbers_lcm_ratio_l687_68777


namespace NUMINAMATH_GPT_Rick_received_amount_l687_68747

theorem Rick_received_amount :
  let total_promised := 400
  let sally_owes := 35
  let amy_owes := 30
  let derek_owes := amy_owes / 2
  let carl_owes := 35
  let total_owed := sally_owes + amy_owes + derek_owes + carl_owes
  total_promised - total_owed = 285 :=
by
  sorry

end NUMINAMATH_GPT_Rick_received_amount_l687_68747


namespace NUMINAMATH_GPT_independence_events_exactly_one_passing_l687_68744

-- Part 1: Independence of Events

def event_A (die1 : ℕ) : Prop :=
  die1 % 2 = 1

def event_B (die1 die2 : ℕ) : Prop :=
  (die1 + die2) % 3 = 0

def P_event_A : ℚ :=
  1 / 2

def P_event_B : ℚ :=
  1 / 3

def P_event_AB : ℚ :=
  1 / 6

theorem independence_events : P_event_AB = P_event_A * P_event_B :=
by
  sorry

-- Part 2: Probability of Exactly One Passing the Assessment

def probability_of_hitting (p : ℝ) : ℝ :=
  1 - (1 - p)^2

def P_A_hitting : ℝ :=
  0.7

def P_B_hitting : ℝ :=
  0.6

def probability_one_passing : ℝ :=
  (probability_of_hitting P_A_hitting) * (1 - probability_of_hitting P_B_hitting) + (1 - probability_of_hitting P_A_hitting) * (probability_of_hitting P_B_hitting)

theorem exactly_one_passing : probability_one_passing = 0.2212 :=
by
  sorry

end NUMINAMATH_GPT_independence_events_exactly_one_passing_l687_68744


namespace NUMINAMATH_GPT_min_value_l687_68746

theorem min_value (a b : ℝ) (h₁ : a > 0) (h₂ : b > 0) (h₃ : a + b = 1) : 
  ∃ min_val, min_val = 5 + 2 * Real.sqrt 6 ∧ (∀ x, (x = 5 + 2 * Real.sqrt 6) → x ≥ min_val) :=
by
  sorry

end NUMINAMATH_GPT_min_value_l687_68746


namespace NUMINAMATH_GPT_number_of_zeros_of_f_l687_68751

def f (x : ℝ) : ℝ := 2 * x - 3 * x

theorem number_of_zeros_of_f :
  ∃ (n : ℕ), n = 2 ∧ (∀ x, f x = 0 → x ∈ {x | f x = 0}) :=
by {
  sorry
}

end NUMINAMATH_GPT_number_of_zeros_of_f_l687_68751


namespace NUMINAMATH_GPT_lcm_gcd_product_l687_68727

theorem lcm_gcd_product (a b : ℕ) (ha : a = 12) (hb : b = 9) :
  Nat.lcm a b * Nat.gcd a b = 108 := by
  rw [ha, hb]
  -- Replace with Nat library functions and calculate
  sorry

end NUMINAMATH_GPT_lcm_gcd_product_l687_68727


namespace NUMINAMATH_GPT_mouse_lives_correct_l687_68789

def cat_lives : ℕ := 9
def dog_lives : ℕ := cat_lives - 3
def mouse_lives : ℕ := dog_lives + 7

theorem mouse_lives_correct : mouse_lives = 13 :=
by
  sorry

end NUMINAMATH_GPT_mouse_lives_correct_l687_68789


namespace NUMINAMATH_GPT_least_number_1056_div_26_l687_68756

/-- Define the given values and the divisibility condition -/
def least_number_to_add (n : ℕ) (d : ℕ) : ℕ :=
  let remainder := n % d
  d - remainder

/-- State the theorem to prove that the least number to add to 1056 to make it divisible by 26 is 10. -/
theorem least_number_1056_div_26 : least_number_to_add 1056 26 = 10 :=
by
  sorry -- Proof is omitted as per the instruction

end NUMINAMATH_GPT_least_number_1056_div_26_l687_68756


namespace NUMINAMATH_GPT_find_initial_investment_l687_68705

open Real

noncomputable def initial_investment (x : ℝ) (years : ℕ) (final_value : ℝ) : ℝ := 
  final_value / (3 ^ (years / (112 / x)))

theorem find_initial_investment :
  let x := 8
  let years := 28
  let final_value := 31500
  initial_investment x years final_value = 3500 := 
by 
  sorry

end NUMINAMATH_GPT_find_initial_investment_l687_68705


namespace NUMINAMATH_GPT_simplify_fraction_l687_68743

open Real

theorem simplify_fraction (x : ℝ) : (3 + 2 * sin x + 2 * cos x) / (3 + 2 * sin x - 2 * cos x) = 3 / 5 + (2 / 5) * cos x :=
by
  sorry

end NUMINAMATH_GPT_simplify_fraction_l687_68743


namespace NUMINAMATH_GPT_trapezoid_DC_length_l687_68724

theorem trapezoid_DC_length 
  (AB DC: ℝ) (BC: ℝ) 
  (angle_BCD angle_CDA: ℝ)
  (h1: AB = 8)
  (h2: BC = 4 * Real.sqrt 3)
  (h3: angle_BCD = 60)
  (h4: angle_CDA = 45)
  (h5: AB = DC):
  DC = 14 + 4 * Real.sqrt 2 :=
sorry

end NUMINAMATH_GPT_trapezoid_DC_length_l687_68724


namespace NUMINAMATH_GPT_sum_of_B_and_C_in_base_6_l687_68795

def digit_base_6 (n: Nat) : Prop :=
  n > 0 ∧ n < 6

theorem sum_of_B_and_C_in_base_6
  (A B C : Nat)
  (hA : digit_base_6 A)
  (hB : digit_base_6 B)
  (hC : digit_base_6 C)
  (hDistinct : A ≠ B ∧ B ≠ C ∧ A ≠ C)
  (hSum : 43 * (A + B + C) = 216 * A) :
  B + C = 5 := by
  sorry

end NUMINAMATH_GPT_sum_of_B_and_C_in_base_6_l687_68795


namespace NUMINAMATH_GPT_feet_heads_difference_l687_68719

theorem feet_heads_difference :
  let hens := 60
  let goats := 35
  let camels := 6
  let keepers := 10
  let heads := hens + goats + camels + keepers
  let feet := (2 * hens) + (4 * goats) + (4 * camels) + (2 * keepers)
  feet - heads = 193 :=
by
  sorry

end NUMINAMATH_GPT_feet_heads_difference_l687_68719


namespace NUMINAMATH_GPT_solve_equation_x_squared_eq_16x_l687_68772

theorem solve_equation_x_squared_eq_16x :
  ∀ x : ℝ, x^2 = 16 * x ↔ (x = 0 ∨ x = 16) :=
by 
  intro x
  -- Complete proof here
  sorry

end NUMINAMATH_GPT_solve_equation_x_squared_eq_16x_l687_68772
