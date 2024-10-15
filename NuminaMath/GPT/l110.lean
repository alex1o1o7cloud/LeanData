import Mathlib

namespace NUMINAMATH_GPT_books_per_bookshelf_l110_11061

theorem books_per_bookshelf (total_books bookshelves : ℕ) (h_total_books : total_books = 34) (h_bookshelves : bookshelves = 2) : total_books / bookshelves = 17 :=
by
  sorry

end NUMINAMATH_GPT_books_per_bookshelf_l110_11061


namespace NUMINAMATH_GPT_length_first_train_l110_11097

noncomputable def length_second_train : ℝ := 200
noncomputable def speed_first_train_kmh : ℝ := 42
noncomputable def speed_second_train_kmh : ℝ := 30
noncomputable def time_seconds : ℝ := 14.998800095992321

noncomputable def speed_first_train_ms : ℝ := speed_first_train_kmh * 1000 / 3600
noncomputable def speed_second_train_ms : ℝ := speed_second_train_kmh * 1000 / 3600

noncomputable def relative_speed : ℝ := speed_first_train_ms + speed_second_train_ms
noncomputable def combined_length : ℝ := relative_speed * time_seconds

theorem length_first_train : combined_length - length_second_train = 99.9760019198464 :=
by
  sorry

end NUMINAMATH_GPT_length_first_train_l110_11097


namespace NUMINAMATH_GPT_factorize_m_square_minus_16_l110_11095

-- Define the expression
def expr (m : ℝ) : ℝ := m^2 - 16

-- Define the factorized form
def factorized_expr (m : ℝ) : ℝ := (m + 4) * (m - 4)

-- State the theorem
theorem factorize_m_square_minus_16 (m : ℝ) : expr m = factorized_expr m :=
by
  sorry

end NUMINAMATH_GPT_factorize_m_square_minus_16_l110_11095


namespace NUMINAMATH_GPT_reduced_price_proof_l110_11091

noncomputable def reduced_price (P: ℝ) := 0.88 * P

theorem reduced_price_proof :
  ∃ R P : ℝ, R = reduced_price P ∧ 1200 / R = 1200 / P + 6 ∧ R = 24 :=
by
  sorry

end NUMINAMATH_GPT_reduced_price_proof_l110_11091


namespace NUMINAMATH_GPT_men_with_all_attributes_le_l110_11055

theorem men_with_all_attributes_le (total men_with_tv men_with_radio men_with_ac: ℕ) (married_men: ℕ) 
(h_total: total = 100) 
(h_married_men: married_men = 84) 
(h_men_with_tv: men_with_tv = 75) 
(h_men_with_radio: men_with_radio = 85) 
(h_men_with_ac: men_with_ac = 70) : 
  ∃ x, x ≤ men_with_ac ∧ x ≤ married_men ∧ x ≤ men_with_tv ∧ x ≤ men_with_radio ∧ (x ≤ total) := 
sorry

end NUMINAMATH_GPT_men_with_all_attributes_le_l110_11055


namespace NUMINAMATH_GPT_arithmetic_geometric_sequence_l110_11029

theorem arithmetic_geometric_sequence (a : ℕ → ℤ) (h1 : ∀ n, a (n + 1) = a n + 3)
    (h2 : (a 1 + 3) * (a 1 + 21) = (a 1 + 9) ^ 2) : a 3 = 12 :=
by 
  sorry

end NUMINAMATH_GPT_arithmetic_geometric_sequence_l110_11029


namespace NUMINAMATH_GPT_trees_falling_count_l110_11099

/-- Definition of the conditions of the problem. --/
def initial_mahogany_trees : ℕ := 50
def initial_narra_trees : ℕ := 30
def trees_on_farm_after_typhoon : ℕ := 88

/-- The mathematical proof problem statement in Lean 4:
Prove the total number of trees that fell during the typhoon (N + M) is equal to 5,
given the conditions.
--/
theorem trees_falling_count (M N : ℕ) 
  (h1 : M = N + 1)
  (h2 : (initial_mahogany_trees - M + 3 * M) + (initial_narra_trees - N + 2 * N) = trees_on_farm_after_typhoon) :
  N + M = 5 := sorry

end NUMINAMATH_GPT_trees_falling_count_l110_11099


namespace NUMINAMATH_GPT_travel_time_by_raft_l110_11068

variable (U V : ℝ) -- U: speed of the steamboat, V: speed of the river current
variable (S : ℝ) -- S: distance between cities A and B

-- Conditions
variable (h1 : S = 12 * U - 15 * V) -- Distance calculation, city B to city A
variable (h2 : S = 8 * U + 10 * V)  -- Distance calculation, city A to city B
variable (T : ℝ) -- Time taken on a raft

-- Proof problem
theorem travel_time_by_raft : T = 60 :=
by
  sorry


end NUMINAMATH_GPT_travel_time_by_raft_l110_11068


namespace NUMINAMATH_GPT_final_hair_length_l110_11010

theorem final_hair_length (x y z : ℕ) (hx : x = 16) (hy : y = 11) (hz : z = 12) : 
  (x - y) + z = 17 :=
by
  sorry

end NUMINAMATH_GPT_final_hair_length_l110_11010


namespace NUMINAMATH_GPT_ratio_simplified_l110_11089

variable (a b c : ℕ)
variable (n m p : ℕ) (h1 : n > 0) (h2 : m > 0) (h3 : p > 0)

theorem ratio_simplified (h_ratio : a^n = 3 * c^p ∧ b^m = 4 * c^p ∧ c^p = 7 * c^p) :
  (a^n + b^m + c^p) / c^p = 2 := sorry

end NUMINAMATH_GPT_ratio_simplified_l110_11089


namespace NUMINAMATH_GPT_real_solutions_l110_11006

theorem real_solutions (x : ℝ) :
  (1 / ((x - 2) * (x - 3)) + 1 / ((x - 3) * (x - 4)) + 1 / ((x - 4) * (x - 5)) + 1 / ((x - 5) * (x - 6)) = 1 / 12) ↔ (x = 12 ∨ x = -4) :=
by
  sorry

end NUMINAMATH_GPT_real_solutions_l110_11006


namespace NUMINAMATH_GPT_power_multiplication_l110_11035

variable (a : ℝ)

theorem power_multiplication : (-a)^3 * a^2 = -a^5 := 
sorry

end NUMINAMATH_GPT_power_multiplication_l110_11035


namespace NUMINAMATH_GPT_fraction_of_number_is_three_quarters_l110_11047

theorem fraction_of_number_is_three_quarters 
  (f : ℚ) 
  (h1 : 76 ≠ 0) 
  (h2 : f * 76 = 76 - 19) : 
  f = 3 / 4 :=
by
  sorry

end NUMINAMATH_GPT_fraction_of_number_is_three_quarters_l110_11047


namespace NUMINAMATH_GPT_cauchy_schwarz_inequality_l110_11093

theorem cauchy_schwarz_inequality (a b x y : ℝ) :
  (a^2 + b^2) * (x^2 + y^2) ≥ (a * x + b * y)^2 :=
by
  sorry

end NUMINAMATH_GPT_cauchy_schwarz_inequality_l110_11093


namespace NUMINAMATH_GPT_proper_subset_count_of_set_l110_11037

theorem proper_subset_count_of_set (s : Finset ℕ) (h : s = {1, 2, 3}) : s.powerset.card - 1 = 7 := by
  sorry

end NUMINAMATH_GPT_proper_subset_count_of_set_l110_11037


namespace NUMINAMATH_GPT_intersection_of_A_and_B_l110_11034

open Finset

def A : Finset ℤ := {-2, -1, 0, 1, 2}
def B : Finset ℤ := {1, 2, 3}

theorem intersection_of_A_and_B :
  A ∩ B = {1, 2} :=
by sorry

end NUMINAMATH_GPT_intersection_of_A_and_B_l110_11034


namespace NUMINAMATH_GPT_isosceles_triangle_perimeter_l110_11002

theorem isosceles_triangle_perimeter (a b : ℕ) (h1 : a = 6 ∨ a = 9) (h2 : b = 6 ∨ b = 9) (h : a ≠ b) : (a * 2 + b = 21 ∨ a * 2 + b = 24) :=
by
  sorry

end NUMINAMATH_GPT_isosceles_triangle_perimeter_l110_11002


namespace NUMINAMATH_GPT_simplify_expression_l110_11070

theorem simplify_expression (x : ℝ) : 
  (2 * x - 3 * (2 + x) + 4 * (2 - x) - 5 * (2 + 3 * x)) = -20 * x - 8 :=
by
  sorry

end NUMINAMATH_GPT_simplify_expression_l110_11070


namespace NUMINAMATH_GPT_jacob_fraction_of_phoebe_age_l110_11003

-- Definitions
def Rehana_current_age := 25
def Rehana_future_age (years : Nat) := Rehana_current_age + years
def Phoebe_future_age (years : Nat) := (Rehana_future_age years) / 3
def Phoebe_current_age := Phoebe_future_age 5 - 5
def Jacob_age := 3
def fraction_of_Phoebe_age := Jacob_age / Phoebe_current_age

-- Theorem statement
theorem jacob_fraction_of_phoebe_age :
  fraction_of_Phoebe_age = 3 / 5 :=
  sorry

end NUMINAMATH_GPT_jacob_fraction_of_phoebe_age_l110_11003


namespace NUMINAMATH_GPT_locus_of_Q_is_circle_l110_11084

variables {A B C P Q : ℝ}

def point_on_segment (A B C : ℝ) : Prop := C > A ∧ C < B

def variable_point_on_circle (A B P : ℝ) : Prop := (P - A) * (P - B) = 0

def ratio_condition (C P Q A B : ℝ) : Prop := (P - C) / (C - Q) = (A - C) / (C - B)

def locus_of_Q_circle (A B C P Q : ℝ) : Prop := ∃ B', (C > A ∧ C < B) → (P - A) * (P - B) = 0 → (P - C) / (C - Q) = (A - C) / (C - B) → (Q - B') * (Q - B) = 0

theorem locus_of_Q_is_circle (A B C P Q : ℝ) :
  point_on_segment A B C →
  variable_point_on_circle A B P →
  ratio_condition C P Q A B →
  locus_of_Q_circle A B C P Q :=
by
  sorry

end NUMINAMATH_GPT_locus_of_Q_is_circle_l110_11084


namespace NUMINAMATH_GPT_GIMPS_meaning_l110_11075

/--
  Curtis Cooper's team discovered the largest prime number known as \( 2^{74,207,281} - 1 \), which is a Mersenne prime.
  GIMPS stands for "Great Internet Mersenne Prime Search."

  Prove that GIMPS means "Great Internet Mersenne Prime Search".
-/
theorem GIMPS_meaning : GIMPS = "Great Internet Mersenne Prime Search" :=
  sorry

end NUMINAMATH_GPT_GIMPS_meaning_l110_11075


namespace NUMINAMATH_GPT_contingency_fund_l110_11044

theorem contingency_fund:
  let d := 240
  let cp := d * (1.0 / 3)
  let lc := d * (1.0 / 2)
  let r := d - cp - lc
  let lp := r * (1.0 / 4)
  let cf := r - lp
  cf = 30 := 
by
  sorry

end NUMINAMATH_GPT_contingency_fund_l110_11044


namespace NUMINAMATH_GPT_integer_roots_of_quadratic_l110_11001

theorem integer_roots_of_quadratic (a : ℚ) :
  (∃ x₁ x₂ : ℤ, 
    a * x₁ * x₁ + (a + 1) * x₁ + (a - 1) = 0 ∧ 
    a * x₂ * x₂ + (a + 1) * x₂ + (a - 1) = 0 ∧ 
    x₁ ≠ x₂) ↔ 
      a = 0 ∨ a = -1/7 ∨ a = 1 :=
by
  sorry

end NUMINAMATH_GPT_integer_roots_of_quadratic_l110_11001


namespace NUMINAMATH_GPT_front_crawl_speed_l110_11063
   
   def swim_condition := 
     ∃ F : ℝ, -- Speed of front crawl in yards per minute
     (∃ t₁ t₂ d₁ d₂ : ℝ, -- t₁ is time for front crawl, t₂ is time for breaststroke, d₁ and d₂ are distances
               t₁ = 8 ∧
               t₂ = 4 ∧
               d₁ = t₁ * F ∧
               d₂ = t₂ * 35 ∧
               d₁ + d₂ = 500 ∧
               t₁ + t₂ = 12) ∧
     F = 45
   
   theorem front_crawl_speed : swim_condition :=
     by
       sorry -- Proof goes here, with given conditions satisfying F = 45
   
end NUMINAMATH_GPT_front_crawl_speed_l110_11063


namespace NUMINAMATH_GPT_shortest_handspan_is_Doyoon_l110_11036

def Sangwon_handspan_cm : ℝ := 19.8
def Doyoon_handspan_cm : ℝ := 18.9
def Changhyeok_handspan_cm : ℝ := 19.3

theorem shortest_handspan_is_Doyoon :
  Doyoon_handspan_cm < Sangwon_handspan_cm ∧ Doyoon_handspan_cm < Changhyeok_handspan_cm :=
by
  sorry

end NUMINAMATH_GPT_shortest_handspan_is_Doyoon_l110_11036


namespace NUMINAMATH_GPT_parking_savings_l110_11057

theorem parking_savings (weekly_cost : ℕ) (monthly_cost : ℕ) (weeks_in_year : ℕ) (months_in_year : ℕ)
  (h_weekly_cost : weekly_cost = 10)
  (h_monthly_cost : monthly_cost = 42)
  (h_weeks_in_year : weeks_in_year = 52)
  (h_months_in_year : months_in_year = 12) :
  weekly_cost * weeks_in_year - monthly_cost * months_in_year = 16 := 
by
  sorry

end NUMINAMATH_GPT_parking_savings_l110_11057


namespace NUMINAMATH_GPT_naomi_number_of_ways_to_1000_l110_11021

-- Define the initial condition and operations

def start : ℕ := 2

def add1 (n : ℕ) : ℕ := n + 1

def square (n : ℕ) : ℕ := n * n

-- Define a proposition that counts the number of ways to reach 1000 from 2 using these operations
def count_ways (start target : ℕ) : ℕ := sorry  -- We'll need a complex function to literally count the paths, but we'll abstract this here.

-- Theorem stating the number of ways to reach 1000
theorem naomi_number_of_ways_to_1000 : count_ways start 1000 = 128 := 
sorry

end NUMINAMATH_GPT_naomi_number_of_ways_to_1000_l110_11021


namespace NUMINAMATH_GPT_least_positive_integer_satisfying_conditions_l110_11043

theorem least_positive_integer_satisfying_conditions :
  ∃ b : ℕ, b > 0 ∧ (b % 7 = 6) ∧ (b % 11 = 10) ∧ (b % 13 = 12) ∧ b = 1000 :=
by
  sorry

end NUMINAMATH_GPT_least_positive_integer_satisfying_conditions_l110_11043


namespace NUMINAMATH_GPT_congruence_theorem_l110_11065

def triangle_congruent_SSA (a b : ℝ) (gamma : ℝ) :=
  b * b = a * a + (-2 * a * 5 * Real.cos gamma) + 25

theorem congruence_theorem : triangle_congruent_SSA 3 5 (150 * Real.pi / 180) :=
by
  -- Proof is omitted, based on the problem's instruction.
  sorry

end NUMINAMATH_GPT_congruence_theorem_l110_11065


namespace NUMINAMATH_GPT_geometric_series_product_l110_11078

theorem geometric_series_product (y : ℝ) :
  (∑'n : ℕ, (1 / 3 : ℝ) ^ n) * (∑'n : ℕ, (- 1 / 3 : ℝ) ^ n)
  = ∑'n : ℕ, (y⁻¹ : ℝ) ^ n ↔ y = 9 :=
by
  sorry

end NUMINAMATH_GPT_geometric_series_product_l110_11078


namespace NUMINAMATH_GPT_other_liquid_cost_l110_11079

-- Definitions based on conditions
def total_fuel_gallons : ℕ := 12
def fuel_price_per_gallon : ℝ := 8
def oil_price_per_gallon : ℝ := 15
def fuel_cost : ℝ := total_fuel_gallons * fuel_price_per_gallon
def other_liquid_price_per_gallon (x : ℝ) : Prop :=
  (7 * x + 5 * oil_price_per_gallon = fuel_cost) ∨
  (7 * oil_price_per_gallon + 5 * x = fuel_cost)

-- Question: The cost of the other liquid per gallon
theorem other_liquid_cost :
  ∃ x, other_liquid_price_per_gallon x ∧ x = 3 :=
sorry

end NUMINAMATH_GPT_other_liquid_cost_l110_11079


namespace NUMINAMATH_GPT_cost_per_mile_sunshine_is_018_l110_11020

theorem cost_per_mile_sunshine_is_018 :
  ∀ (x : ℝ) (daily_rate_sunshine daily_rate_city cost_per_mile_city : ℝ),
  daily_rate_sunshine = 17.99 →
  daily_rate_city = 18.95 →
  cost_per_mile_city = 0.16 →
  (daily_rate_sunshine + 48 * x = daily_rate_city + cost_per_mile_city * 48) →
  x = 0.18 :=
by
  intros x daily_rate_sunshine daily_rate_city cost_per_mile_city
  intros h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_cost_per_mile_sunshine_is_018_l110_11020


namespace NUMINAMATH_GPT_find_n_l110_11005

theorem find_n : ∃ n : ℕ, n < 200 ∧ ∃ k : ℕ, n^2 + (n + 1)^2 = k^2 ∧ (n = 3 ∨ n = 20 ∨ n = 119) := 
by
  sorry

end NUMINAMATH_GPT_find_n_l110_11005


namespace NUMINAMATH_GPT_Greg_harvested_acres_l110_11000

-- Defining the conditions
def Sharon_harvested : ℝ := 0.1
def Greg_harvested (additional: ℝ) (Sharon: ℝ) : ℝ := Sharon + additional

-- Proving the statement
theorem Greg_harvested_acres : Greg_harvested 0.3 Sharon_harvested = 0.4 :=
by
  sorry

end NUMINAMATH_GPT_Greg_harvested_acres_l110_11000


namespace NUMINAMATH_GPT_range_x_range_a_l110_11048

variable {x a : ℝ}
def p (a x : ℝ) : Prop := x^2 - 4 * a * x + 3 * a^2 < 0
def q (x : ℝ) : Prop := (x - 3) * (2 - x) ≥ 0

-- (1) If a = 1, find the range of x for which p ∧ q is true.
theorem range_x (h : a = 1) : 2 ≤ x ∧ x < 3 ↔ p 1 x ∧ q x := sorry

-- (2) If ¬p is a necessary but not sufficient condition for ¬q, find the range of real number a.
theorem range_a : (¬p a x → ¬q x) → (∃ a : ℝ, 1 < a ∧ a < 2) := sorry

end NUMINAMATH_GPT_range_x_range_a_l110_11048


namespace NUMINAMATH_GPT_number_of_zeros_is_one_l110_11009

noncomputable def f (x : ℝ) : ℝ := Real.exp x + 3 * x

theorem number_of_zeros_is_one : 
  ∃! x : ℝ, f x = 0 :=
sorry

end NUMINAMATH_GPT_number_of_zeros_is_one_l110_11009


namespace NUMINAMATH_GPT_tree_drops_leaves_on_fifth_day_l110_11008

def initial_leaves := 340
def daily_drop_fraction := 1 / 10

noncomputable def leaves_after_day (n: ℕ) : ℕ :=
  match n with
  | 0 => initial_leaves
  | 1 => initial_leaves - Nat.floor (initial_leaves * daily_drop_fraction)
  | 2 => leaves_after_day 1 - Nat.floor (leaves_after_day 1 * daily_drop_fraction)
  | 3 => leaves_after_day 2 - Nat.floor (leaves_after_day 2 * daily_drop_fraction)
  | 4 => leaves_after_day 3 - Nat.floor (leaves_after_day 3 * daily_drop_fraction)
  | _ => 0  -- beyond the 4th day

theorem tree_drops_leaves_on_fifth_day : leaves_after_day 4 = 225 := by
  -- We'll skip the detailed proof here, focusing on the statement
  sorry

end NUMINAMATH_GPT_tree_drops_leaves_on_fifth_day_l110_11008


namespace NUMINAMATH_GPT_melissa_work_hours_l110_11019

theorem melissa_work_hours (total_fabric : ℕ) (fabric_per_dress : ℕ) (hours_per_dress : ℕ) (total_num_dresses : ℕ) (total_hours : ℕ) 
  (h1 : total_fabric = 56) (h2 : fabric_per_dress = 4) (h3 : hours_per_dress = 3) : 
  total_hours = (total_fabric / fabric_per_dress) * hours_per_dress := by
  sorry

end NUMINAMATH_GPT_melissa_work_hours_l110_11019


namespace NUMINAMATH_GPT_nineteen_power_six_l110_11077

theorem nineteen_power_six :
    19^11 / 19^5 = 47045881 := by
  sorry

end NUMINAMATH_GPT_nineteen_power_six_l110_11077


namespace NUMINAMATH_GPT_total_students_in_school_l110_11015

theorem total_students_in_school : 
  let C1 := 32
  let C2 := C1 - 2
  let C3 := C2 - 2
  let C4 := C3 - 2
  let C5 := C4 - 2
  C1 + C2 + C3 + C4 + C5 = 140 :=
by
  let C1 := 32
  let C2 := C1 - 2
  let C3 := C2 - 2
  let C4 := C3 - 2
  let C5 := C4 - 2
  sorry

end NUMINAMATH_GPT_total_students_in_school_l110_11015


namespace NUMINAMATH_GPT_square_diagonal_cut_l110_11074

/--
Given a square with side length 10,
prove that cutting along the diagonal results in two 
right-angled isosceles triangles with dimensions 10, 10, 10*sqrt(2).
-/
theorem square_diagonal_cut (side_length : ℕ) (triangle_side1 triangle_side2 hypotenuse : ℝ) 
  (h_side : side_length = 10)
  (h_triangle_side1 : triangle_side1 = 10) 
  (h_triangle_side2 : triangle_side2 = 10)
  (h_hypotenuse : hypotenuse = 10 * Real.sqrt 2) : 
  triangle_side1 = side_length ∧ triangle_side2 = side_length ∧ hypotenuse = side_length * Real.sqrt 2 :=
by
  sorry

end NUMINAMATH_GPT_square_diagonal_cut_l110_11074


namespace NUMINAMATH_GPT_surface_area_is_correct_volume_is_approximately_correct_l110_11030

noncomputable def surface_area_of_CXYZ (height : ℝ) (side_length : ℝ) : ℝ :=
  let area_CZX_CZY := 48
  let area_CXY := 9 * Real.sqrt 3
  let area_XYZ := 9 * Real.sqrt 15
  2 * area_CZX_CZY + area_CXY + area_XYZ

theorem surface_area_is_correct (height : ℝ) (side_length : ℝ) (h : height = 24) (s : side_length = 18) :
  surface_area_of_CXYZ height side_length = 96 + 9 * Real.sqrt 3 + 9 * Real.sqrt 15 :=
by
  sorry

noncomputable def volume_of_CXYZ (height : ℝ ) (side_length : ℝ) : ℝ :=
  -- Placeholder for the volume calculation approximation method.
  486

theorem volume_is_approximately_correct
  (height : ℝ) (side_length : ℝ) (h : height = 24) (s : side_length = 18) :
  volume_of_CXYZ height side_length = 486 :=
by
  sorry

end NUMINAMATH_GPT_surface_area_is_correct_volume_is_approximately_correct_l110_11030


namespace NUMINAMATH_GPT_increasing_interval_a_geq_neg2_l110_11066

theorem increasing_interval_a_geq_neg2
  (f : ℝ → ℝ)
  (h : ∀ x, f x = x^2 + 2 * (a - 2) * x + 5)
  (h_inc : ∀ x > 4, f (x + 1) > f x) :
  a ≥ -2 :=
sorry

end NUMINAMATH_GPT_increasing_interval_a_geq_neg2_l110_11066


namespace NUMINAMATH_GPT_total_pages_l110_11090

-- Definitions based on conditions
def math_pages : ℕ := 10
def extra_reading_pages : ℕ := 3
def reading_pages : ℕ := math_pages + extra_reading_pages

-- Statement of the proof problem
theorem total_pages : math_pages + reading_pages = 23 := by 
  sorry

end NUMINAMATH_GPT_total_pages_l110_11090


namespace NUMINAMATH_GPT_total_packs_of_groceries_l110_11027

-- Definitions for the conditions
def packs_of_cookies : ℕ := 2
def packs_of_cake : ℕ := 12

-- Theorem stating the total packs of groceries
theorem total_packs_of_groceries : packs_of_cookies + packs_of_cake = 14 :=
by sorry

end NUMINAMATH_GPT_total_packs_of_groceries_l110_11027


namespace NUMINAMATH_GPT_find_value_of_f2_l110_11051

noncomputable def f (x : ℝ) : ℝ := (x + 1) / (x - 1)

theorem find_value_of_f2 : f 2 = 101 / 99 :=
  sorry

end NUMINAMATH_GPT_find_value_of_f2_l110_11051


namespace NUMINAMATH_GPT_total_books_l110_11092

theorem total_books (shelves_mystery shelves_picture : ℕ) (books_per_shelf : ℕ) 
    (h_mystery : shelves_mystery = 5) (h_picture : shelves_picture = 4) (h_books_per_shelf : books_per_shelf = 6) : 
    shelves_mystery * books_per_shelf + shelves_picture * books_per_shelf = 54 := 
by 
  sorry

end NUMINAMATH_GPT_total_books_l110_11092


namespace NUMINAMATH_GPT_integrality_condition_l110_11058

noncomputable def binom (n k : ℕ) : ℕ := 
  n.choose k

theorem integrality_condition (n k : ℕ) (h : 1 ≤ k) (h1 : k < n) (h2 : (k + 1) ∣ (n^2 - 3*k^2 - 2)) : 
  ∃ m : ℕ, m = (n^2 - 3*k^2 - 2) / (k + 1) ∧ (m * binom n k) % 1 = 0 :=
sorry

end NUMINAMATH_GPT_integrality_condition_l110_11058


namespace NUMINAMATH_GPT_valid_x_values_l110_11052

noncomputable def valid_triangle_sides (x : ℕ) : Prop :=
  8 + 11 > x + 3 ∧ 8 + (x + 3) > 11 ∧ 11 + (x + 3) > 8

theorem valid_x_values :
  {x : ℕ | valid_triangle_sides x ∧ x > 0} = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15} :=
by
  sorry

end NUMINAMATH_GPT_valid_x_values_l110_11052


namespace NUMINAMATH_GPT_combined_molecular_weight_l110_11041

def atomic_weight_C : ℝ := 12.01
def atomic_weight_Cl : ℝ := 35.45
def atomic_weight_S : ℝ := 32.07
def atomic_weight_F : ℝ := 19.00

def molecular_weight_CCl4 : ℝ := atomic_weight_C + 4 * atomic_weight_Cl
def molecular_weight_SF6 : ℝ := atomic_weight_S + 6 * atomic_weight_F

def weight_moles_CCl4 (moles : ℝ) : ℝ := moles * molecular_weight_CCl4
def weight_moles_SF6 (moles : ℝ) : ℝ := moles * molecular_weight_SF6

theorem combined_molecular_weight : weight_moles_CCl4 9 + weight_moles_SF6 5 = 2114.64 := by
  sorry

end NUMINAMATH_GPT_combined_molecular_weight_l110_11041


namespace NUMINAMATH_GPT_total_inflation_time_l110_11067

theorem total_inflation_time (time_per_ball : ℕ) (alexia_balls : ℕ) (extra_balls : ℕ) : 
  time_per_ball = 20 → alexia_balls = 20 → extra_balls = 5 →
  (alexia_balls * time_per_ball) + ((alexia_balls + extra_balls) * time_per_ball) = 900 :=
by 
  intros h1 h2 h3
  sorry

end NUMINAMATH_GPT_total_inflation_time_l110_11067


namespace NUMINAMATH_GPT_relation_between_incircle_radius_perimeter_area_l110_11032

theorem relation_between_incircle_radius_perimeter_area (r p S : ℝ) (h : S = (1 / 2) * r * p) : S = (1 / 2) * r * p :=
by {
  sorry
}

end NUMINAMATH_GPT_relation_between_incircle_radius_perimeter_area_l110_11032


namespace NUMINAMATH_GPT_fifteen_percent_minus_70_l110_11054

theorem fifteen_percent_minus_70 (a : ℝ) : 0.15 * a - 70 = (15 / 100) * a - 70 :=
by sorry

end NUMINAMATH_GPT_fifteen_percent_minus_70_l110_11054


namespace NUMINAMATH_GPT_one_million_div_one_fourth_l110_11060

theorem one_million_div_one_fourth : (1000000 : ℝ) / (1 / 4) = 4000000 := by
  sorry

end NUMINAMATH_GPT_one_million_div_one_fourth_l110_11060


namespace NUMINAMATH_GPT_total_cost_price_l110_11085

theorem total_cost_price (SP1 SP2 SP3 : ℝ) (P1 P2 P3 : ℝ) 
  (h1 : SP1 = 120) (h2 : SP2 = 150) (h3 : SP3 = 200)
  (h4 : P1 = 0.20) (h5 : P2 = 0.25) (h6 : P3 = 0.10) : (SP1 / (1 + P1) + SP2 / (1 + P2) + SP3 / (1 + P3) = 401.82) :=
by
  sorry

end NUMINAMATH_GPT_total_cost_price_l110_11085


namespace NUMINAMATH_GPT_pie_chart_probability_l110_11056

theorem pie_chart_probability
  (P_W P_X P_Z : ℚ)
  (h_W : P_W = 1/4)
  (h_X : P_X = 1/3)
  (h_Z : P_Z = 1/6) :
  1 - P_W - P_X - P_Z = 1/4 :=
by
  -- The detailed proof steps are omitted as per the requirement.
  sorry

end NUMINAMATH_GPT_pie_chart_probability_l110_11056


namespace NUMINAMATH_GPT_convex_quadrilateral_division_l110_11062

-- Definitions for convex quadrilateral and some basic geometric objects.
structure Point :=
  (x : ℝ)
  (y : ℝ)

structure Quadrilateral :=
  (A B C D : Point)
  (convex : ∀ (X Y Z : Point), (X ≠ Y) ∧ (Y ≠ Z) ∧ (Z ≠ X))

-- Definitions for lines and midpoints.
def is_midpoint (M X Y : Point) : Prop :=
  M.x = (X.x + Y.x) / 2 ∧ M.y = (X.y + Y.y) / 2

-- Preliminary to determining equal area division.
def equal_area_division (Q : Quadrilateral) (L : Point → Point → Prop) : Prop :=
  ∃ F,
    is_midpoint F Q.A Q.B ∧
    -- Assuming some way to relate area with F and L
    L Q.D F ∧
    -- Placeholder for equality of areas (details depend on how we calculate area)
    sorry

-- Problem statement in Lean 4
theorem convex_quadrilateral_division (Q : Quadrilateral) :
  ∃ L, equal_area_division Q L :=
by
  -- Proof will be constructed here based on steps in the solution
  sorry

end NUMINAMATH_GPT_convex_quadrilateral_division_l110_11062


namespace NUMINAMATH_GPT_arrange_digits_l110_11025

theorem arrange_digits (A B C D E F : ℕ)
  (h1 : A ≠ B) (h2 : A ≠ C) (h3 : A ≠ D) (h4 : A ≠ E) (h5 : A ≠ F)
  (h6 : B ≠ C) (h7 : B ≠ D) (h8 : B ≠ E) (h9 : B ≠ F)
  (h10 : C ≠ D) (h11 : C ≠ E) (h12 : C ≠ F)
  (h13 : D ≠ E) (h14 : D ≠ F) (h15 : E ≠ F)
  (range_A : 1 ≤ A ∧ A ≤ 6) (range_B : 1 ≤ B ∧ B ≤ 6) (range_C : 1 ≤ C ∧ C ≤ 6)
  (range_D : 1 ≤ D ∧ D ≤ 6) (range_E : 1 ≤ E ∧ E ≤ 6) (range_F : 1 ≤ F ∧ F ≤ 6)
  (sum_line1 : A + D + E = 15) (sum_line2 : A + C + 9 = 15) 
  (sum_line3 : B + D + 9 = 15) (sum_line4 : 7 + C + E = 15) 
  (sum_line5 : 9 + C + A = 15) (sum_line6 : A + 8 + F = 15) 
  (sum_line7 : 7 + D + F = 15) : 
  (A = 4) ∧ (B = 1) ∧ (C = 2) ∧ (D = 5) ∧ (E = 6) ∧ (F = 3) :=
sorry

end NUMINAMATH_GPT_arrange_digits_l110_11025


namespace NUMINAMATH_GPT_age_difference_l110_11045

-- Defining the current age of the son
def S : ℕ := 26

-- Defining the current age of the man
def M : ℕ := 54

-- Defining the condition that in two years, the man's age is twice the son's age
def condition : Prop := (M + 2) = 2 * (S + 2)

-- The theorem that states how much older the man is than the son
theorem age_difference : condition → M - S = 28 := by
  sorry

end NUMINAMATH_GPT_age_difference_l110_11045


namespace NUMINAMATH_GPT_B_completes_work_in_18_days_l110_11086

variable {A B : ℝ}
variable (x : ℝ)

-- Conditions provided
def A_works_twice_as_fast_as_B (h1 : A = 2 * B) : Prop := true
def together_finish_work_in_6_days (h2 : B = 1 / x ∧ A = 2 / x ∧ 1 / 6 = (A + B)) : Prop := true

-- Theorem to prove: It takes B 18 days to complete the work independently
theorem B_completes_work_in_18_days (h1 : A = 2 * B) (h2 : B = 1 / x ∧ A = 2 / x ∧ 1 / 6 = (A + B)) : x = 18 := by
  sorry

end NUMINAMATH_GPT_B_completes_work_in_18_days_l110_11086


namespace NUMINAMATH_GPT_matthew_egg_rolls_l110_11023

theorem matthew_egg_rolls (A P M : ℕ) 
  (h1 : M = 3 * P) 
  (h2 : P = A / 2) 
  (h3 : A = 4) : 
  M = 6 :=
by
  sorry

end NUMINAMATH_GPT_matthew_egg_rolls_l110_11023


namespace NUMINAMATH_GPT_largest_angle_of_consecutive_integer_angles_of_hexagon_l110_11080

theorem largest_angle_of_consecutive_integer_angles_of_hexagon 
  (angles : Fin 6 → ℝ)
  (h_consecutive : ∃ (x : ℝ), angles = ![
    x - 3, x - 2, x - 1, x, x + 1, x + 2 ])
  (h_sum : (angles 0 + angles 1 + angles 2 + angles 3 + angles 4 + angles 5) = 720) :
  (angles 5 = 122.5) :=
by
  sorry

end NUMINAMATH_GPT_largest_angle_of_consecutive_integer_angles_of_hexagon_l110_11080


namespace NUMINAMATH_GPT_least_number_to_add_l110_11088

theorem least_number_to_add (k n : ℕ) (h : k = 1015) (m : n = 25) : 
  ∃ x : ℕ, (k + x) % n = 0 ∧ x = 10 := by
  sorry

end NUMINAMATH_GPT_least_number_to_add_l110_11088


namespace NUMINAMATH_GPT_largest_six_consecutive_composites_less_than_40_l110_11028

def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def is_composite (n : ℕ) := ¬ is_prime n ∧ n > 1

theorem largest_six_consecutive_composites_less_than_40 :
  ∃ (seq : ℕ → ℕ) (i : ℕ),
    (∀ j : ℕ, j < 6 → is_composite (seq (i + j))) ∧ 
    (seq i < 40) ∧ 
    (seq (i+1) < 40) ∧ 
    (seq (i+2) < 40) ∧ 
    (seq (i+3) < 40) ∧ 
    (seq (i+4) < 40) ∧ 
    (seq (i+5) < 40) ∧ 
    seq (i+5) = 30 
:= sorry

end NUMINAMATH_GPT_largest_six_consecutive_composites_less_than_40_l110_11028


namespace NUMINAMATH_GPT_value_of_polynomial_l110_11040

theorem value_of_polynomial (x y : ℝ) (h : x - y = 5) : (x - y)^2 + 2 * (x - y) - 10 = 25 :=
by sorry

end NUMINAMATH_GPT_value_of_polynomial_l110_11040


namespace NUMINAMATH_GPT_proof_problem_l110_11033

noncomputable def p : ℝ := -5 / 3
noncomputable def q : ℝ := -1

def A (p : ℝ) : Set ℝ := {x | 2 * x^2 + 3 * p * x + 2 = 0}
def B (q : ℝ) : Set ℝ := {x | 2 * x^2 + x + q = 0}

theorem proof_problem (h : (A p ∩ B q) = {1 / 2}) :
    p = -5 / 3 ∧ q = -1 ∧ (A p ∪ B q) = {-1, 1 / 2, 2} := by
  sorry

end NUMINAMATH_GPT_proof_problem_l110_11033


namespace NUMINAMATH_GPT_negation_of_exists_proposition_l110_11004

theorem negation_of_exists_proposition :
  (¬ ∃ x : ℝ, x^2 - x + 1 = 0) ↔ ∀ x : ℝ, x^2 - x + 1 ≠ 0 :=
sorry

end NUMINAMATH_GPT_negation_of_exists_proposition_l110_11004


namespace NUMINAMATH_GPT_find_a3_a4_a5_l110_11071

open Real

variables {a : ℕ → ℝ} (q : ℝ)

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q

noncomputable def a_1 : ℝ := 3

def sum_of_first_three (a : ℕ → ℝ) : Prop :=
  a 0 + a 1 + a 2 = 21

def all_terms_positive (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, 0 < a n

theorem find_a3_a4_a5 (h1 : is_geometric_sequence a) (h2 : a 0 = a_1) (h3 : sum_of_first_three a) (h4 : all_terms_positive a) :
  a 2 + a 3 + a 4 = 84 :=
sorry

end NUMINAMATH_GPT_find_a3_a4_a5_l110_11071


namespace NUMINAMATH_GPT_is_correct_functional_expression_l110_11046

variable (x : ℝ)

def is_isosceles_triangle (x : ℝ) (y : ℝ) : Prop :=
  2*x + y = 20

theorem is_correct_functional_expression (h1 : 5 < x) (h2 : x < 10) : 
  ∃ y, y = 20 - 2*x :=
by
  sorry

end NUMINAMATH_GPT_is_correct_functional_expression_l110_11046


namespace NUMINAMATH_GPT_number_of_friends_l110_11031

theorem number_of_friends (total_bottle_caps : ℕ) (bottle_caps_per_friend : ℕ) (h1 : total_bottle_caps = 18) (h2 : bottle_caps_per_friend = 3) :
  total_bottle_caps / bottle_caps_per_friend = 6 :=
by
  sorry

end NUMINAMATH_GPT_number_of_friends_l110_11031


namespace NUMINAMATH_GPT_betty_height_in_feet_l110_11083

theorem betty_height_in_feet (dog_height carter_height betty_height : ℕ) (h1 : dog_height = 24) 
  (h2 : carter_height = 2 * dog_height) (h3 : betty_height = carter_height - 12) : betty_height / 12 = 3 :=
by
  sorry

end NUMINAMATH_GPT_betty_height_in_feet_l110_11083


namespace NUMINAMATH_GPT_find_some_number_l110_11016

-- Definitions of symbol replacements
def replacement_minus (a b : Nat) := a + b
def replacement_plus (a b : Nat) := a * b
def replacement_times (a b : Nat) := a / b
def replacement_div (a b : Nat) := a - b

-- The transformed equation using the replacements
def transformed_equation (some_number : Nat) :=
  replacement_minus
    some_number
    (replacement_div
      (replacement_plus 9 (replacement_times 8 3))
      25) = 5

theorem find_some_number : ∃ some_number : Nat, transformed_equation some_number ∧ some_number = 6 :=
by
  exists 6
  unfold transformed_equation
  unfold replacement_minus replacement_plus replacement_times replacement_div
  sorry

end NUMINAMATH_GPT_find_some_number_l110_11016


namespace NUMINAMATH_GPT_sufficient_and_necessary_condition_l110_11059

noncomputable def f (x : ℝ) : ℝ := x + Real.sin x

theorem sufficient_and_necessary_condition (a b : ℝ) : (a + b > 0) ↔ (f a + f b > 0) :=
by sorry

end NUMINAMATH_GPT_sufficient_and_necessary_condition_l110_11059


namespace NUMINAMATH_GPT_necessary_and_sufficient_condition_l110_11082

noncomputable def f (a x : ℝ) : ℝ := a * x - x^2

theorem necessary_and_sufficient_condition (a : ℝ) (h : 0 < a) :
  (∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → f a x ≤ 1) ↔ (0 < a ∧ a ≤ 2) := by
  sorry

end NUMINAMATH_GPT_necessary_and_sufficient_condition_l110_11082


namespace NUMINAMATH_GPT_arith_seq_a4a6_equals_4_l110_11076

variable (a : ℕ → ℝ) (d : ℝ)
variable (h2 : a 2 = a 1 + d)
variable (h4 : a 4 = a 1 + 3 * d)
variable (h6 : a 6 = a 1 + 5 * d)
variable (h8 : a 8 = a 1 + 7 * d)
variable (h10 : a 10 = a 1 + 9 * d)
variable (condition : (a 2)^2 + 2 * a 2 * a 8 + a 6 * a 10 = 16)

theorem arith_seq_a4a6_equals_4 : a 4 * a 6 = 4 := by
  sorry

end NUMINAMATH_GPT_arith_seq_a4a6_equals_4_l110_11076


namespace NUMINAMATH_GPT_number_of_trees_l110_11049

theorem number_of_trees (n : ℕ) (diff : ℕ) (count1 : ℕ) (count2 : ℕ) (timur1 : ℕ) (alexander1 : ℕ) (timur2 : ℕ) (alexander2 : ℕ) : 
  diff = alexander1 - timur1 ∧
  count1 = timur2 + (alexander2 - timur1) ∧
  n = count1 + diff →
  n = 118 :=
by
  sorry

end NUMINAMATH_GPT_number_of_trees_l110_11049


namespace NUMINAMATH_GPT_conic_section_is_hyperbola_l110_11069

-- Definitions for the conditions in the problem
def conic_section_equation (x y : ℝ) := (x - 4) ^ 2 = 5 * (y + 2) ^ 2 - 45

-- The theorem that we need to prove
theorem conic_section_is_hyperbola : ∀ x y : ℝ, (conic_section_equation x y) → "H" = "H" :=
by
  intro x y h
  sorry

end NUMINAMATH_GPT_conic_section_is_hyperbola_l110_11069


namespace NUMINAMATH_GPT_cuboid_surface_area_two_cubes_l110_11098

noncomputable def cuboid_surface_area (b : ℝ) : ℝ :=
  let l := 2 * b
  let w := b
  let h := b
  2 * (l * w + l * h + w * h)

theorem cuboid_surface_area_two_cubes (b : ℝ) : cuboid_surface_area b = 10 * b^2 := by
  sorry

end NUMINAMATH_GPT_cuboid_surface_area_two_cubes_l110_11098


namespace NUMINAMATH_GPT_inequality_solution_non_negative_integer_solutions_l110_11053

theorem inequality_solution (x : ℝ) :
  (x - 2) / 2 ≤ (7 - x) / 3 → x ≤ 4 :=
by
  sorry

theorem non_negative_integer_solutions :
  { n : ℤ | n ≥ 0 ∧ n ≤ 4 } = {0, 1, 2, 3, 4} :=
by
  sorry

end NUMINAMATH_GPT_inequality_solution_non_negative_integer_solutions_l110_11053


namespace NUMINAMATH_GPT_solve_for_x_l110_11038

theorem solve_for_x (y : ℝ) (x : ℝ) (h1 : y = 432) (h2 : 12^2 * x^4 / 432 = y) : x = 6 := by
  sorry

end NUMINAMATH_GPT_solve_for_x_l110_11038


namespace NUMINAMATH_GPT_prob_6_higher_than_3_after_10_shuffles_l110_11012

def p_k (k : Nat) : ℚ := (3^k - 2^k) / (2 * 3^k)

theorem prob_6_higher_than_3_after_10_shuffles :
  p_k 10 = (3^10 - 2^10) / (2 * 3^10) :=
by
  sorry

end NUMINAMATH_GPT_prob_6_higher_than_3_after_10_shuffles_l110_11012


namespace NUMINAMATH_GPT_price_of_case_bulk_is_12_l110_11087

noncomputable def price_per_can_grocery_store : ℚ := 6 / 12
noncomputable def price_per_can_bulk : ℚ := price_per_can_grocery_store - 0.25
def cans_per_case_bulk : ℕ := 48
noncomputable def price_per_case_bulk : ℚ := price_per_can_bulk * cans_per_case_bulk

theorem price_of_case_bulk_is_12 : price_per_case_bulk = 12 :=
by
  sorry

end NUMINAMATH_GPT_price_of_case_bulk_is_12_l110_11087


namespace NUMINAMATH_GPT_john_must_study_4_5_hours_l110_11042

-- Let "study_time" be the amount of time John needs to study for the second exam.

noncomputable def study_time_for_avg_score (hours1 score1 target_avg total_exams : ℝ) (direct_relation : Prop) :=
  2 * target_avg - score1 / (score1 / hours1)

theorem john_must_study_4_5_hours :
  study_time_for_avg_score 3 60 75 2 (60 / 3 = 90 / study_time_for_avg_score 3 60 75 2 (60 / 3 = 90 / study_time_for_avg_score 3 60 75 2 (sorry))) = 4.5 :=
sorry

end NUMINAMATH_GPT_john_must_study_4_5_hours_l110_11042


namespace NUMINAMATH_GPT_exists_parallel_line_l110_11018

variable (P : ℝ × ℝ)
variable (g : ℝ × ℝ)
variable (in_first_quadrant : 0 < P.1 ∧ 0 < P.2)
variable (parallel_to_second_projection_plane : ∃ c : ℝ, g = (c, 0))

theorem exists_parallel_line (P : ℝ × ℝ) (g : ℝ × ℝ) (in_first_quadrant : 0 < P.1 ∧ 0 < P.2)
  (parallel_to_second_projection_plane : ∃ c : ℝ, g = (c, 0)) :
  ∃ a : ℝ × ℝ, (∃ d : ℝ, g = (d, 0)) ∧ (a = P) :=
sorry

end NUMINAMATH_GPT_exists_parallel_line_l110_11018


namespace NUMINAMATH_GPT_find_c_l110_11039

-- Defining the variables and conditions given in the problem
variables (a b c : ℝ)

-- Conditions
def vertex_condition : Prop := (2, -3) = (a * (-3)^2 + b * (-3) + c, -3)
def point_condition : Prop := (7, -1) = (a * (-1)^2 + b * (-1) + c, -1)

-- Problem Statement
theorem find_c 
  (h_vertex : vertex_condition a b c)
  (h_point : point_condition a b c) :
  c = 53 / 4 :=
sorry

end NUMINAMATH_GPT_find_c_l110_11039


namespace NUMINAMATH_GPT_remainder_7_pow_93_mod_12_l110_11072

theorem remainder_7_pow_93_mod_12 : 7 ^ 93 % 12 = 7 := 
by
  -- the sequence repeats every two terms: 7, 1, 7, 1, ...
  sorry

end NUMINAMATH_GPT_remainder_7_pow_93_mod_12_l110_11072


namespace NUMINAMATH_GPT_ax_by_power5_l110_11011

-- Define the real numbers a, b, x, and y
variables (a b x y : ℝ)

-- Define the conditions as assumptions
axiom axiom1 : a * x + b * y = 3
axiom axiom2 : a * x^2 + b * y^2 = 7
axiom axiom3 : a * x^3 + b * y^3 = 16
axiom axiom4 : a * x^4 + b * y^4 = 42

-- State the theorem to prove ax^5 + by^5 = 20
theorem ax_by_power5 : a * x^5 + b * y^5 = 20 :=
  sorry

end NUMINAMATH_GPT_ax_by_power5_l110_11011


namespace NUMINAMATH_GPT_fractions_order_l110_11081

theorem fractions_order :
  let frac1 := (21 : ℚ) / (17 : ℚ)
  let frac2 := (23 : ℚ) / (19 : ℚ)
  let frac3 := (25 : ℚ) / (21 : ℚ)
  frac3 < frac2 ∧ frac2 < frac1 :=
by sorry

end NUMINAMATH_GPT_fractions_order_l110_11081


namespace NUMINAMATH_GPT_find_2a_plus_b_l110_11026

theorem find_2a_plus_b (a b : ℝ) (h1 : 3 * a + 2 * b = 18) (h2 : 5 * a + 4 * b = 31) :
  2 * a + b = 11.5 :=
sorry

end NUMINAMATH_GPT_find_2a_plus_b_l110_11026


namespace NUMINAMATH_GPT_num_values_of_a_l110_11024

def A : Set ℝ := {-1, 1, 3}
def B (a : ℝ) : Set ℝ := {1, a^2 - 2 * a}

theorem num_values_of_a : ∃v : Finset ℝ, (∀ a ∈ v, B a ⊆ A) ∧ v.card = 3 :=
by
  sorry

end NUMINAMATH_GPT_num_values_of_a_l110_11024


namespace NUMINAMATH_GPT_max_gcd_of_15n_plus_4_and_9n_plus_2_eq_2_l110_11073

theorem max_gcd_of_15n_plus_4_and_9n_plus_2_eq_2 (n : ℕ) (hn : n > 0) :
  ∃ m, m = Nat.gcd (15 * n + 4) (9 * n + 2) ∧ m ≤ 2 :=
by
  sorry

end NUMINAMATH_GPT_max_gcd_of_15n_plus_4_and_9n_plus_2_eq_2_l110_11073


namespace NUMINAMATH_GPT_cost_comparison_compare_cost_when_x_30_more_cost_effective_30_l110_11013

-- Definitions for the conditions in the problem
def cost_of_suit : ℕ := 1000
def cost_of_tie : ℕ := 200

-- Definitions for Option 1 and Option 2 calculations
def option1_total_cost (x : ℕ) (h : x > 20) : ℕ := 200 * x + 16000
def option2_total_cost (x : ℕ) (h : x > 20) : ℕ := 180 * x + 18000

-- Case x=30 for comparison
def x : ℕ := 30
def option1_cost_when_x_30 : ℕ := 200 * x + 16000
def option2_cost_when_x_30 : ℕ := 180 * x + 18000

-- More cost-effective plan when x=30
def more_cost_effective_plan_for_x_30 : ℕ := 21800

theorem cost_comparison (x : ℕ) (h1 : x > 20) :
  option1_total_cost x h1 = 200 * x + 16000 ∧
  option2_total_cost x h1 = 180 * x + 18000 := 
by
  sorry

theorem compare_cost_when_x_30 :
  option1_cost_when_x_30 = 22000 ∧
  option2_cost_when_x_30 = 23400 ∧
  option1_cost_when_x_30 < option2_cost_when_x_30 := 
by
  sorry

theorem more_cost_effective_30 :
  more_cost_effective_plan_for_x_30 = 21800 := 
by
  sorry

end NUMINAMATH_GPT_cost_comparison_compare_cost_when_x_30_more_cost_effective_30_l110_11013


namespace NUMINAMATH_GPT_concentric_circle_area_ratio_l110_11017

theorem concentric_circle_area_ratio (r R : ℝ) (h_ratio : (π * R^2) / (π * r^2) = 16 / 3) :
  R - r = 1.309 * r :=
by
  sorry

end NUMINAMATH_GPT_concentric_circle_area_ratio_l110_11017


namespace NUMINAMATH_GPT_trigonometric_identity_simplification_l110_11096

theorem trigonometric_identity_simplification :
  (Real.sin (15 * Real.pi / 180) * Real.cos (75 * Real.pi / 180) + Real.cos (15 * Real.pi / 180) * Real.sin (105 * Real.pi / 180) = 1) :=
by sorry

end NUMINAMATH_GPT_trigonometric_identity_simplification_l110_11096


namespace NUMINAMATH_GPT_ratio_snakes_to_lions_is_S_per_100_l110_11014

variables {S G : ℕ}

/-- Giraffe count in Safari National Park is 10 fewer than snakes -/
def safari_giraffes_minus_ten (S G : ℕ) : Prop := G = S - 10

/-- The number of lions in Safari National Park -/
def safari_lions : ℕ := 100

/-- The ratio of number of snakes to number of lions in Safari National Park -/
def ratio_snakes_to_lions (S : ℕ) : ℕ := S / safari_lions

/-- Prove the ratio of the number of snakes to the number of lions in Safari National Park -/
theorem ratio_snakes_to_lions_is_S_per_100 :
  ∀ S G, safari_giraffes_minus_ten S G → (ratio_snakes_to_lions S = S / 100) :=
by
  intros S G h
  sorry

end NUMINAMATH_GPT_ratio_snakes_to_lions_is_S_per_100_l110_11014


namespace NUMINAMATH_GPT_functional_equation_l110_11022

noncomputable def f : ℝ → ℝ := sorry

theorem functional_equation (f_add : ∀ x y : ℝ, f (x + y) = f x + f y) (f_two : f 2 = 4) : f 1 = 2 :=
sorry

end NUMINAMATH_GPT_functional_equation_l110_11022


namespace NUMINAMATH_GPT_smallest_n_divisibility_l110_11007

theorem smallest_n_divisibility (n : ℕ) (h : n = 5) : 
  ∃ k, (1 ≤ k ∧ k ≤ n + 1) ∧ (n^2 - n) % k = 0 ∧ ∃ i, (1 ≤ i ∧ i ≤ n + 1) ∧ (n^2 - n) % i ≠ 0 :=
by
  sorry

end NUMINAMATH_GPT_smallest_n_divisibility_l110_11007


namespace NUMINAMATH_GPT_proposition_1_proposition_2_proposition_3_proposition_4_l110_11064

axiom p1 : Prop
axiom p2 : Prop
axiom p3 : Prop
axiom p4 : Prop

axiom p1_true : p1 = true
axiom p2_false : p2 = false
axiom p3_false : p3 = false
axiom p4_true : p4 = true

theorem proposition_1 : (p1 ∧ p4) = true := by sorry
theorem proposition_2 : (p1 ∧ p2) = false := by sorry
theorem proposition_3 : (¬p2 ∨ p3) = true := by sorry
theorem proposition_4 : (¬p3 ∨ ¬p4) = true := by sorry

end NUMINAMATH_GPT_proposition_1_proposition_2_proposition_3_proposition_4_l110_11064


namespace NUMINAMATH_GPT_largest_three_digit_multiple_of_9_with_digits_sum_27_l110_11050

theorem largest_three_digit_multiple_of_9_with_digits_sum_27 :
  ∃ n : ℕ, 100 ≤ n ∧ n ≤ 999 ∧ (n % 9 = 0) ∧ ((n / 100) + ((n % 100) / 10) + (n % 10) = 27) ∧ n = 999 :=
by
  sorry

end NUMINAMATH_GPT_largest_three_digit_multiple_of_9_with_digits_sum_27_l110_11050


namespace NUMINAMATH_GPT_xiao_zhang_return_distance_xiao_zhang_no_refuel_needed_l110_11094

def total_distance : ℕ :=
  15 - 3 + 16 - 11 + 10 - 12 + 4 - 15 + 16 - 18

def fuel_consumption_per_km : ℝ := 0.6
def initial_fuel : ℝ := 72.2

theorem xiao_zhang_return_distance :
  total_distance = 2 := by
  sorry

theorem xiao_zhang_no_refuel_needed :
  (initial_fuel - fuel_consumption_per_km * (|15| + |3| + |16| + |11| + |10| + |12| + |4| + |15| + |16| + |18|)) >= 0 := by
  sorry

end NUMINAMATH_GPT_xiao_zhang_return_distance_xiao_zhang_no_refuel_needed_l110_11094
