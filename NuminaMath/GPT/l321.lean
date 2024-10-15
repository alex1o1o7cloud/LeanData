import Mathlib

namespace NUMINAMATH_GPT_exactly_one_box_empty_count_l321_32152

-- Define the setting with four different balls and four boxes.
def numberOfWaysExactlyOneBoxEmpty (balls : Finset ℕ) (boxes : Finset ℕ) : ℕ :=
  if (balls.card = 4 ∧ boxes.card = 4) then
     Nat.choose 4 2 * Nat.factorial 3
  else 0

theorem exactly_one_box_empty_count :
  numberOfWaysExactlyOneBoxEmpty {1, 2, 3, 4} {1, 2, 3, 4} = 144 :=
by
  -- The proof is omitted
  sorry

end NUMINAMATH_GPT_exactly_one_box_empty_count_l321_32152


namespace NUMINAMATH_GPT_allocation_count_l321_32104

def allocate_volunteers (num_service_points : Nat) (num_volunteers : Nat) : Nat :=
  -- Definition that captures the counting logic as per the problem statement
  if num_service_points = 4 ∧ num_volunteers = 6 then 660 else 0

theorem allocation_count :
  allocate_volunteers 4 6 = 660 :=
sorry

end NUMINAMATH_GPT_allocation_count_l321_32104


namespace NUMINAMATH_GPT_wallace_fulfills_orders_in_13_days_l321_32119

def batch_small_bags_production := 12
def batch_large_bags_production := 8
def time_per_small_batch := 8
def time_per_large_batch := 12
def daily_production_limit := 18

def initial_stock_small := 18
def initial_stock_large := 10

def order1_small := 45
def order1_large := 30
def order2_small := 60
def order2_large := 25
def order3_small := 52
def order3_large := 42

def total_small_bags_needed := order1_small + order2_small + order3_small
def total_large_bags_needed := order1_large + order2_large + order3_large
def small_bags_to_produce := total_small_bags_needed - initial_stock_small
def large_bags_to_produce := total_large_bags_needed - initial_stock_large

def small_batches_needed := (small_bags_to_produce + batch_small_bags_production - 1) / batch_small_bags_production
def large_batches_needed := (large_bags_to_produce + batch_large_bags_production - 1) / batch_large_bags_production

def total_time_small_batches := small_batches_needed * time_per_small_batch
def total_time_large_batches := large_batches_needed * time_per_large_batch
def total_production_time := total_time_small_batches + total_time_large_batches

def days_needed := (total_production_time + daily_production_limit - 1) / daily_production_limit

theorem wallace_fulfills_orders_in_13_days :
  days_needed = 13 := by
  sorry

end NUMINAMATH_GPT_wallace_fulfills_orders_in_13_days_l321_32119


namespace NUMINAMATH_GPT_larger_jar_half_full_l321_32192

-- Defining the capacities of the jars
variables (S L W : ℚ)

-- Conditions
def equal_amount_water (S L W : ℚ) : Prop :=
  W = (1/5 : ℚ) * S ∧ W = (1/4 : ℚ) * L

-- Question: What fraction will the larger jar be filled if the water from the smaller jar is added to it?
theorem larger_jar_half_full (S L W : ℚ) (h : equal_amount_water S L W) :
  (2 * W) / L = (1 / 2 : ℚ) :=
sorry

end NUMINAMATH_GPT_larger_jar_half_full_l321_32192


namespace NUMINAMATH_GPT_students_not_reading_novels_l321_32159

theorem students_not_reading_novels
  (total_students : ℕ)
  (students_three_or_more_novels : ℕ)
  (students_two_novels : ℕ)
  (students_one_novel : ℕ)
  (h_total_students : total_students = 240)
  (h_students_three_or_more_novels : students_three_or_more_novels = 1 / 6 * 240)
  (h_students_two_novels : students_two_novels = 35 / 100 * 240)
  (h_students_one_novel : students_one_novel = 5 / 12 * 240)
  :
  total_students - (students_three_or_more_novels + students_two_novels + students_one_novel) = 16 :=
by
  sorry

end NUMINAMATH_GPT_students_not_reading_novels_l321_32159


namespace NUMINAMATH_GPT_ellipse_focus_value_k_l321_32194

theorem ellipse_focus_value_k 
  (k : ℝ)
  (h : ∀ x y, 5 * x^2 + k * y^2 = 5 → abs y ≠ 2 → ∀ c : ℝ, c^2 = 4 → k = 1) :
  ∀ k : ℝ, (5 * (0:ℝ)^2 + k * (2:ℝ)^2 = 5) ∧ (5 * (0:ℝ)^2 + k * (-(2:ℝ))^2 = 5) → k = 1 := by
  sorry

end NUMINAMATH_GPT_ellipse_focus_value_k_l321_32194


namespace NUMINAMATH_GPT_perpendicular_tangent_lines_l321_32198

def f (x : ℝ) : ℝ := x^3 + 1

noncomputable def tangent_line_eqs (x₀ : ℝ) (y₀ : ℝ) : Prop :=
  (3 * x₀ - y₀ - 1 = 0) ∨ (3 * x₀ - y₀ + 3 = 0)

theorem perpendicular_tangent_lines (x₀ : ℝ) (hx₀ : x₀ = 1 ∨ x₀ = -1) :
  tangent_line_eqs x₀ (f x₀) := by
  sorry

end NUMINAMATH_GPT_perpendicular_tangent_lines_l321_32198


namespace NUMINAMATH_GPT_parabola_chord_ratio_is_3_l321_32185

noncomputable def parabola_chord_ratio (p : ℝ) (h : p > 0) : ℝ :=
  let focus_x := p / 2
  let a_x := (3 * p) / 2
  let b_x := p / 6
  let af := a_x + (p / 2)
  let bf := b_x + (p / 2)
  af / bf

theorem parabola_chord_ratio_is_3 (p : ℝ) (h : p > 0) : parabola_chord_ratio p h = 3 := by
  sorry

end NUMINAMATH_GPT_parabola_chord_ratio_is_3_l321_32185


namespace NUMINAMATH_GPT_largest_among_abcd_l321_32154

theorem largest_among_abcd (a b c d k : ℤ) (h : a - 1 = b + 2 ∧ b + 2 = c - 3 ∧ c - 3 = d + 4) :
  c = k + 3 ∧
  a = k + 1 ∧
  b = k - 2 ∧
  d = k - 4 ∧
  c > a ∧
  c > b ∧
  c > d :=
by
  sorry

end NUMINAMATH_GPT_largest_among_abcd_l321_32154


namespace NUMINAMATH_GPT_radishes_times_carrots_l321_32105

theorem radishes_times_carrots (cucumbers radishes carrots : ℕ) 
  (h1 : cucumbers = 15) 
  (h2 : radishes = 3 * cucumbers) 
  (h3 : carrots = 9) : 
  radishes / carrots = 5 :=
by
  sorry

end NUMINAMATH_GPT_radishes_times_carrots_l321_32105


namespace NUMINAMATH_GPT_max_adjacent_distinct_pairs_l321_32101

theorem max_adjacent_distinct_pairs (n : ℕ) (h : n = 100) : 
  ∃ (a : ℕ), a = 50 := 
by 
  -- Here we use the provided constraints and game scenario to state the theorem formally.
  sorry

end NUMINAMATH_GPT_max_adjacent_distinct_pairs_l321_32101


namespace NUMINAMATH_GPT_no_x4_term_expansion_l321_32190

-- Mathematical condition and properties
variable {R : Type*} [CommRing R]

theorem no_x4_term_expansion (a : R) (h : a ≠ 0) :
  ∃ a, (a = 8) := 
by 
  sorry

end NUMINAMATH_GPT_no_x4_term_expansion_l321_32190


namespace NUMINAMATH_GPT_Monica_saved_per_week_l321_32171

theorem Monica_saved_per_week(amount_per_cycle : ℕ) (weeks_per_cycle : ℕ) (num_cycles : ℕ) (total_saved : ℕ) :
  num_cycles = 5 →
  weeks_per_cycle = 60 →
  (amount_per_cycle * num_cycles) = total_saved →
  total_saved = 4500 →
  total_saved / (weeks_per_cycle * num_cycles) = 75 := 
by
  intros
  sorry

end NUMINAMATH_GPT_Monica_saved_per_week_l321_32171


namespace NUMINAMATH_GPT_unique_prime_value_l321_32180

def T : ℤ := 2161

theorem unique_prime_value :
  ∃ p : ℕ, (∃ n : ℤ, n^4 - 898 * n^2 + T - 2160 = p) ∧ Prime p ∧ (∀ q, (∃ n : ℤ, n^4 - 898 * n^2 + T - 2160 = q) → q = p) :=
  sorry

end NUMINAMATH_GPT_unique_prime_value_l321_32180


namespace NUMINAMATH_GPT_find_b_l321_32183

theorem find_b (a b c : ℚ) :
  -- Condition from the problem, equivalence of polynomials for all x
  ((4 : ℚ) * x^2 - 2 * x + 5 / 2) * (a * x^2 + b * x + c) =
    12 * x^4 - 8 * x^3 + 15 * x^2 - 5 * x + 5 / 2 →
  -- Given we found that a = 3 from the solution
  a = 3 →
  -- We need to prove that b = -1/2
  b = -1 / 2 :=
sorry

end NUMINAMATH_GPT_find_b_l321_32183


namespace NUMINAMATH_GPT_girls_in_math_class_l321_32117

theorem girls_in_math_class (x y z : ℕ)
  (boys_girls_ratio : 5 * x = 8 * x)
  (math_science_ratio : 7 * y = 13 * x)
  (science_literature_ratio : 4 * y = 3 * z)
  (total_students : 13 * x + 4 * y + 5 * z = 720) :
  8 * x = 176 :=
by
  sorry

end NUMINAMATH_GPT_girls_in_math_class_l321_32117


namespace NUMINAMATH_GPT_total_surfers_l321_32161

theorem total_surfers (num_surfs_santa_monica : ℝ) (ratio_malibu : ℝ) (ratio_santa_monica : ℝ) (ratio_venice : ℝ) (ratio_huntington : ℝ) (ratio_newport : ℝ) :
    num_surfs_santa_monica = 36 ∧ ratio_malibu = 7 ∧ ratio_santa_monica = 4.5 ∧ ratio_venice = 3.5 ∧ ratio_huntington = 2 ∧ ratio_newport = 1.5 →
    (ratio_malibu * (num_surfs_santa_monica / ratio_santa_monica) +
     num_surfs_santa_monica +
     ratio_venice * (num_surfs_santa_monica / ratio_santa_monica) +
     ratio_huntington * (num_surfs_santa_monica / ratio_santa_monica) +
     ratio_newport * (num_surfs_santa_monica / ratio_santa_monica)) = 148 :=
by
  sorry

end NUMINAMATH_GPT_total_surfers_l321_32161


namespace NUMINAMATH_GPT_cows_to_eat_grass_in_96_days_l321_32102

theorem cows_to_eat_grass_in_96_days (G r : ℕ) : 
  (∀ N : ℕ, (70 * 24 = G + 24 * r) → (30 * 60 = G + 60 * r) → 
  (∃ N : ℕ, 96 * N = G + 96 * r) → N = 20) :=
by
  intro N
  intro h1 h2 h3
  sorry

end NUMINAMATH_GPT_cows_to_eat_grass_in_96_days_l321_32102


namespace NUMINAMATH_GPT_equal_parts_division_l321_32106

theorem equal_parts_division (n : ℕ) (h : (n * n) % 4 = 0) : 
  ∃ parts : ℕ, parts = 4 ∧ ∀ (i : ℕ), i < parts → 
    ∃ p : ℕ, p = (n * n) / parts :=
by sorry

end NUMINAMATH_GPT_equal_parts_division_l321_32106


namespace NUMINAMATH_GPT_yoojung_notebooks_l321_32134

theorem yoojung_notebooks (N : ℕ) (h : (N - 5) / 2 = 4) : N = 13 :=
by
  sorry

end NUMINAMATH_GPT_yoojung_notebooks_l321_32134


namespace NUMINAMATH_GPT_percentage_of_apples_is_50_l321_32108

-- Definitions based on the conditions
def initial_apples : ℕ := 10
def initial_oranges : ℕ := 23
def oranges_removed : ℕ := 13

-- Final percentage calculation after removing 13 oranges
def percentage_apples (apples oranges_removed : ℕ) :=
  let total_initial := initial_apples + initial_oranges
  let oranges_left := initial_oranges - oranges_removed
  let total_after_removal := initial_apples + oranges_left
  (initial_apples * 100) / total_after_removal

-- The theorem to be proved
theorem percentage_of_apples_is_50 : percentage_apples initial_apples oranges_removed = 50 := by
  sorry

end NUMINAMATH_GPT_percentage_of_apples_is_50_l321_32108


namespace NUMINAMATH_GPT_yardsCatchingPasses_l321_32110

-- Definitions from conditions in a)
def totalYardage : ℕ := 150
def runningYardage : ℕ := 90

-- Problem statement (Proof will follow)
theorem yardsCatchingPasses : totalYardage - runningYardage = 60 := by
  sorry

end NUMINAMATH_GPT_yardsCatchingPasses_l321_32110


namespace NUMINAMATH_GPT_smallest_number_diminished_by_10_divisible_l321_32172

theorem smallest_number_diminished_by_10_divisible :
  ∃ (x : ℕ), (x - 10) % 24 = 0 ∧ x = 34 :=
by
  sorry

end NUMINAMATH_GPT_smallest_number_diminished_by_10_divisible_l321_32172


namespace NUMINAMATH_GPT_train_overtake_distance_l321_32178

theorem train_overtake_distance (speed_a speed_b hours_late time_to_overtake distance_a distance_b : ℝ) 
  (h1 : speed_a = 30)
  (h2 : speed_b = 38)
  (h3 : hours_late = 2) 
  (h4 : distance_a = speed_a * hours_late) 
  (h5 : distance_b = speed_b * time_to_overtake) 
  (h6 : time_to_overtake = distance_a / (speed_b - speed_a)) : 
  distance_b = 285 := sorry

end NUMINAMATH_GPT_train_overtake_distance_l321_32178


namespace NUMINAMATH_GPT_sin_inequality_in_triangle_l321_32135

theorem sin_inequality_in_triangle (A B C : ℝ) (hA_leq_B : A ≤ B) (hB_leq_C : B ≤ C)
  (hSum : A + B + C = π) (hA_pos : 0 < A) (hB_pos : 0 < B) (hC_pos : 0 < C)
  (hA_lt_pi : A < π) (hB_lt_pi : B < π) (hC_lt_pi : C < π) :
  0 < Real.sin A + Real.sin B - Real.sin C ∧ Real.sin A + Real.sin B - Real.sin C ≤ Real.sqrt 3 / 2 := 
sorry

end NUMINAMATH_GPT_sin_inequality_in_triangle_l321_32135


namespace NUMINAMATH_GPT_number_of_children_l321_32143

theorem number_of_children (x : ℕ) : 3 * x + 12 = 5 * x - 10 → x = 11 :=
by
  intros h
  have : 3 * x + 12 = 5 * x - 10 := h
  sorry

end NUMINAMATH_GPT_number_of_children_l321_32143


namespace NUMINAMATH_GPT_bus_problem_l321_32145

-- Define the participants in 2005
def participants_2005 (k : ℕ) : ℕ := 27 * k + 19

-- Define the participants in 2006
def participants_2006 (k : ℕ) : ℕ := participants_2005 k + 53

-- Define the total number of buses needed in 2006
def buses_needed_2006 (k : ℕ) : ℕ := (participants_2006 k) / 27 + if (participants_2006 k) % 27 = 0 then 0 else 1

-- Define the total number of buses needed in 2005
def buses_needed_2005 (k : ℕ) : ℕ := k + 1

-- Define the additional buses needed in 2006 compared to 2005
def additional_buses_2006 (k : ℕ) := buses_needed_2006 k - buses_needed_2005 k

-- Define the number of people in the incomplete bus in 2006
def people_in_incomplete_bus_2006 (k : ℕ) := (participants_2006 k) % 27

-- The proof statement to be proved
theorem bus_problem (k : ℕ) : additional_buses_2006 k = 2 ∧ people_in_incomplete_bus_2006 k = 9 := by
  sorry

end NUMINAMATH_GPT_bus_problem_l321_32145


namespace NUMINAMATH_GPT_total_dresses_l321_32175

theorem total_dresses (E M D S: ℕ) 
  (h1 : D = M + 12)
  (h2 : M = E / 2)
  (h3 : E = 16)
  (h4 : S = D - 5) : 
  E + M + D + S = 59 :=
by
  sorry

end NUMINAMATH_GPT_total_dresses_l321_32175


namespace NUMINAMATH_GPT_sin_eq_solutions_l321_32128

theorem sin_eq_solutions :
  (∃ count : ℕ, 
    count = 4007 ∧ 
    (∀ (x : ℝ), 
      0 ≤ x ∧ x ≤ 2 * Real.pi → 
      (∃ (k1 k2 : ℤ), 
        x = -2 * k1 * Real.pi ∨ 
        x = 2 * Real.pi ∨ 
        x = (2 * k2 + 1) * Real.pi / 4005)
    )) :=
sorry

end NUMINAMATH_GPT_sin_eq_solutions_l321_32128


namespace NUMINAMATH_GPT_chocolate_chips_needed_l321_32191

-- Define the variables used in the conditions
def cups_per_recipe := 2
def number_of_recipes := 23

-- State the theorem
theorem chocolate_chips_needed : (cups_per_recipe * number_of_recipes) = 46 := 
by sorry

end NUMINAMATH_GPT_chocolate_chips_needed_l321_32191


namespace NUMINAMATH_GPT_math_problem_l321_32142

theorem math_problem 
  (x y : ℝ) 
  (h1 : 4 * x + y = 12) 
  (h2 : x + 4 * y = 18) : 
  20 * x^2 + 24 * x * y + 20 * y^2 = 468 := 
by
  sorry

end NUMINAMATH_GPT_math_problem_l321_32142


namespace NUMINAMATH_GPT_solve_for_z_l321_32121

open Complex

theorem solve_for_z (z : ℂ) (h : 2 * z * I = 1 + 3 * I) : 
  z = (3 / 2) - (1 / 2) * I :=
by
  sorry

end NUMINAMATH_GPT_solve_for_z_l321_32121


namespace NUMINAMATH_GPT_suit_cost_l321_32173

theorem suit_cost :
  let shirt_cost := 15
  let pants_cost := 40
  let sweater_cost := 30
  let shirts := 4
  let pants := 2
  let sweaters := 2 
  let total_cost := shirts * shirt_cost + pants * pants_cost + sweaters * sweater_cost
  let discount_store := 0.80
  let discount_coupon := 0.90
  ∃ S, discount_coupon * discount_store * (total_cost + S) = 252 → S = 150 :=
by
  let shirt_cost := 15
  let pants_cost := 40
  let sweater_cost := 30
  let shirts := 4
  let pants := 2
  let sweaters := 2 
  let total_cost := shirts * shirt_cost + pants * pants_cost + sweaters * sweater_cost
  let discount_store := 0.80
  let discount_coupon := 0.90
  exists 150
  intro h
  sorry

end NUMINAMATH_GPT_suit_cost_l321_32173


namespace NUMINAMATH_GPT_cost_of_stuffers_number_of_combinations_l321_32186

noncomputable def candy_cane_cost : ℝ := 4 * 0.5
noncomputable def beanie_baby_cost : ℝ := 2 * 3
noncomputable def book_cost : ℝ := 5
noncomputable def toy_cost : ℝ := 3 * 1
noncomputable def gift_card_cost : ℝ := 10
noncomputable def one_child_stuffers_cost : ℝ := candy_cane_cost + beanie_baby_cost + book_cost + toy_cost + gift_card_cost
noncomputable def total_cost : ℝ := one_child_stuffers_cost * 4

def num_books := 5
def num_toys := 10
def toys_combinations : ℕ := Nat.choose num_toys 3
def total_combinations : ℕ := num_books * toys_combinations

theorem cost_of_stuffers (h : total_cost = 104) : total_cost = 104 := by
  sorry

theorem number_of_combinations (h : total_combinations = 600) : total_combinations = 600 := by
  sorry

end NUMINAMATH_GPT_cost_of_stuffers_number_of_combinations_l321_32186


namespace NUMINAMATH_GPT_diet_sodas_sold_l321_32199

theorem diet_sodas_sold (R D : ℕ) (h1 : R + D = 64) (h2 : R / D = 9 / 7) : D = 28 := 
by
  sorry

end NUMINAMATH_GPT_diet_sodas_sold_l321_32199


namespace NUMINAMATH_GPT_range_of_k_l321_32123

theorem range_of_k (k : ℝ) : (∃ (x1 x2 : ℝ), x1 ≠ x2 ∧ (x1^2 + 2*x1 - k = 0) ∧ (x2^2 + 2*x2 - k = 0)) ↔ k > -1 :=
by
  sorry

end NUMINAMATH_GPT_range_of_k_l321_32123


namespace NUMINAMATH_GPT_hyperbola_range_of_m_l321_32177

theorem hyperbola_range_of_m (m : ℝ) : (∃ f : ℝ → ℝ → ℝ, ∀ x y: ℝ, f x y = (x^2 / (4 - m) - y^2 / (2 + m))) → (4 - m) * (2 + m) > 0 → -2 < m ∧ m < 4 :=
by
  intros h_eq h_cond
  sorry

end NUMINAMATH_GPT_hyperbola_range_of_m_l321_32177


namespace NUMINAMATH_GPT_cone_surface_area_l321_32138

theorem cone_surface_area (r l: ℝ) (θ : ℝ) (h₁ : r = 3) (h₂ : θ = 2 * π / 3) (h₃: 2 * π * r = θ * l) :
  π * r * l + π * r ^ 2 = 36 * π :=
by
  sorry

end NUMINAMATH_GPT_cone_surface_area_l321_32138


namespace NUMINAMATH_GPT_final_number_l321_32151

variables (crab goat bear cat hen : ℕ)

-- Given conditions
def row4_sum : Prop := 5 * crab = 10
def col5_sum : Prop := 4 * crab + goat = 11
def row2_sum : Prop := 2 * goat + crab + 2 * bear = 16
def col2_sum : Prop := cat + bear + 2 * goat + crab = 13
def col3_sum : Prop := 2 * crab + 2 * hen + goat = 17

-- Theorem statement
theorem final_number
  (hcrab : row4_sum crab)
  (hgoat_col5 : col5_sum crab goat)
  (hbear_row2 : row2_sum crab goat bear)
  (hcat_col2 : col2_sum cat crab bear goat)
  (hhen_col3 : col3_sum crab goat hen) :
  crab = 2 ∧ goat = 3 ∧ bear = 4 ∧ cat = 1 ∧ hen = 5 → (cat * 10000 + hen * 1000 + crab * 100 + bear * 10 + goat = 15243) :=
sorry

end NUMINAMATH_GPT_final_number_l321_32151


namespace NUMINAMATH_GPT_monochromatic_triangle_in_K6_l321_32196

theorem monochromatic_triangle_in_K6 :
  ∀ (color : Fin 6 → Fin 6 → Prop),
  (∀ (a b : Fin 6), a ≠ b → (color a b ↔ color b a)) →
  (∃ (x y z : Fin 6), x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ (color x y = color y z ∧ color y z = color z x)) :=
by
  sorry

end NUMINAMATH_GPT_monochromatic_triangle_in_K6_l321_32196


namespace NUMINAMATH_GPT_complex_magnitude_l321_32174

theorem complex_magnitude (z : ℂ) (h : z * (2 - 4 * Complex.I) = 1 + 3 * Complex.I) :
  Complex.abs z = Real.sqrt 2 / 2 :=
by
  sorry

end NUMINAMATH_GPT_complex_magnitude_l321_32174


namespace NUMINAMATH_GPT_inner_ring_speed_minimum_train_distribution_l321_32193

theorem inner_ring_speed_minimum
  (l_inner : ℝ) (num_trains_inner : ℕ) (max_wait_inner : ℝ) (speed_min : ℝ) :
  l_inner = 30 →
  num_trains_inner = 9 →
  max_wait_inner = 10 →
  speed_min = 20 :=
by 
  sorry

theorem train_distribution
  (l_inner : ℝ) (speed_inner : ℝ) (speed_outer : ℝ) (total_trains : ℕ) (max_wait_diff : ℝ) (trains_inner : ℕ) (trains_outer : ℕ) :
  l_inner = 30 →
  speed_inner = 25 →
  speed_outer = 30 →
  total_trains = 18 →
  max_wait_diff = 1 →
  trains_inner = 10 →
  trains_outer = 8 :=
by 
  sorry

end NUMINAMATH_GPT_inner_ring_speed_minimum_train_distribution_l321_32193


namespace NUMINAMATH_GPT_perpendicular_lines_from_perpendicular_planes_l321_32118

variable {Line : Type} {Plane : Type}

-- Definitions of non-coincidence, perpendicularity, parallelism
noncomputable def non_coincident_lines (a b : Line) : Prop := sorry
noncomputable def non_coincident_planes (α β : Plane) : Prop := sorry
noncomputable def line_parallel_to_plane (a : Line) (α : Plane) : Prop := sorry
noncomputable def line_perpendicular_to_plane (a : Line) (α : Plane) : Prop := sorry
noncomputable def plane_parallel_to_plane (α β : Plane) : Prop := sorry
noncomputable def plane_perpendicular_to_plane (α β : Plane) : Prop := sorry
noncomputable def line_perpendicular_to_line (a b : Line) : Prop := sorry

-- Given non-coincident lines and planes
variable {a b : Line} {α β : Plane}

-- Problem statement
theorem perpendicular_lines_from_perpendicular_planes (h1 : non_coincident_lines a b)
  (h2 : non_coincident_planes α β)
  (h3 : line_perpendicular_to_plane a α)
  (h4 : line_perpendicular_to_plane b β)
  (h5 : plane_perpendicular_to_plane α β) : line_perpendicular_to_line a b := sorry

end NUMINAMATH_GPT_perpendicular_lines_from_perpendicular_planes_l321_32118


namespace NUMINAMATH_GPT_joe_paint_usage_l321_32148

theorem joe_paint_usage :
  let total_paint := 360
  let paint_first_week := total_paint * (1 / 4)
  let remaining_paint_after_first_week := total_paint - paint_first_week
  let paint_second_week := remaining_paint_after_first_week * (1 / 7)
  paint_first_week + paint_second_week = 128.57 :=
by
  sorry

end NUMINAMATH_GPT_joe_paint_usage_l321_32148


namespace NUMINAMATH_GPT_zookeeper_fish_excess_l321_32158

theorem zookeeper_fish_excess :
  let emperor_ratio := 3
  let adelie_ratio := 5
  let total_penguins := 48
  let total_ratio := emperor_ratio + adelie_ratio
  let emperor_penguins := (emperor_ratio / total_ratio) * total_penguins
  let adelie_penguins := (adelie_ratio / total_ratio) * total_penguins
  let emperor_fish_needed := emperor_penguins * 1.5
  let adelie_fish_needed := adelie_penguins * 2
  let total_fish_needed := emperor_fish_needed + adelie_fish_needed
  let fish_zookeeper_has := total_penguins * 2.5
  (fish_zookeeper_has - total_fish_needed = 33) :=
  
by {
  sorry
}

end NUMINAMATH_GPT_zookeeper_fish_excess_l321_32158


namespace NUMINAMATH_GPT_inequality_proof_l321_32184

theorem inequality_proof (a b c : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) (h_sum : a + b + c = 3) :
  (ab / Real.sqrt (c^2 + 3)) + (bc / Real.sqrt (a^2 + 3)) + (ca / Real.sqrt (b^2 + 3)) ≤ 3 / 2 :=
by
  sorry

end NUMINAMATH_GPT_inequality_proof_l321_32184


namespace NUMINAMATH_GPT_curve_cross_intersection_l321_32144

theorem curve_cross_intersection : 
  ∃ (t_a t_b : ℝ), t_a ≠ t_b ∧ 
  (3 * t_a^2 + 1 = 3 * t_b^2 + 1) ∧
  (t_a^3 - 6 * t_a^2 + 4 = t_b^3 - 6 * t_b^2 + 4) ∧
  (3 * t_a^2 + 1 = 109 ∧ t_a^3 - 6 * t_a^2 + 4 = -428) := by
  sorry

end NUMINAMATH_GPT_curve_cross_intersection_l321_32144


namespace NUMINAMATH_GPT_possible_AC_values_l321_32130

-- Given points A, B, and C on a straight line 
-- with AB = 1 and BC = 3, prove that AC can be 2 or 4.

theorem possible_AC_values (A B C : ℝ) (hAB : abs (B - A) = 1) (hBC : abs (C - B) = 3) : 
  abs (C - A) = 2 ∨ abs (C - A) = 4 :=
sorry

end NUMINAMATH_GPT_possible_AC_values_l321_32130


namespace NUMINAMATH_GPT_all_three_use_media_l321_32149

variable (U T R M T_and_M T_and_R R_and_M T_and_R_and_M : ℕ)

theorem all_three_use_media (hU : U = 180)
  (hT : T = 115)
  (hR : R = 110)
  (hM : M = 130)
  (hT_and_M : T_and_M = 85)
  (hT_and_R : T_and_R = 75)
  (hR_and_M : R_and_M = 95)
  (h_union : U = T + R + M - T_and_R - T_and_M - R_and_M + T_and_R_and_M) :
  T_and_R_and_M = 80 :=
by
  sorry

end NUMINAMATH_GPT_all_three_use_media_l321_32149


namespace NUMINAMATH_GPT_range_of_a_l321_32179

noncomputable def common_point_ellipse_parabola (a : ℝ) : Prop :=
  ∃ x y : ℝ, x^2 + 4 * (y - a)^2 = 4 ∧ x^2 = 2 * y

theorem range_of_a : ∀ a : ℝ, common_point_ellipse_parabola a → -1 ≤ a ∧ a ≤ 17 / 8 :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l321_32179


namespace NUMINAMATH_GPT_max_value_inequality_max_value_equality_l321_32120

theorem max_value_inequality (x : ℝ) (hx : x < 0) : 
  3 * x + 4 / x ≤ -4 * Real.sqrt 3 :=
sorry

theorem max_value_equality (x : ℝ) (hx : x = -2 * Real.sqrt 3 / 3) : 
  3 * x + 4 / x = -4 * Real.sqrt 3 :=
sorry

end NUMINAMATH_GPT_max_value_inequality_max_value_equality_l321_32120


namespace NUMINAMATH_GPT_quadratic_root_l321_32116

theorem quadratic_root (m : ℝ) (h : m^2 + 2 * m - 1 = 0) : 2 * m^2 + 4 * m = 2 := by
  sorry

end NUMINAMATH_GPT_quadratic_root_l321_32116


namespace NUMINAMATH_GPT_mixture_alcohol_quantity_l321_32141

theorem mixture_alcohol_quantity:
  ∀ (A W : ℝ), 
    A / W = 4 / 3 ∧ A / (W + 7) = 4 / 5 → A = 14 :=
by
  intros A W h
  sorry

end NUMINAMATH_GPT_mixture_alcohol_quantity_l321_32141


namespace NUMINAMATH_GPT_hexagon_largest_angle_l321_32140

theorem hexagon_largest_angle (a : ℚ) 
  (h₁ : (a + 2) + (2 * a - 3) + (3 * a + 1) + 4 * a + (5 * a - 4) + (6 * a + 2) = 720) :
  6 * a + 2 = 4374 / 21 :=
by sorry

end NUMINAMATH_GPT_hexagon_largest_angle_l321_32140


namespace NUMINAMATH_GPT_grade_assignment_ways_l321_32168

theorem grade_assignment_ways : (4^12 = 16777216) := 
by 
  sorry

end NUMINAMATH_GPT_grade_assignment_ways_l321_32168


namespace NUMINAMATH_GPT_find_n_l321_32137

-- Define the vectors \overrightarrow {AB}, \overrightarrow {BC}, and \overrightarrow {AC}
def vectorAB : ℝ × ℝ := (2, 4)
def vectorBC (n : ℝ) : ℝ × ℝ := (-2, 2 * n)
def vectorAC : ℝ × ℝ := (0, 2)

-- State the theorem and prove the value of n
theorem find_n (n : ℝ) (h : vectorAC = (vectorAB.1 + (vectorBC n).1, vectorAB.2 + (vectorBC n).2)) : n = -1 :=
by
  sorry

end NUMINAMATH_GPT_find_n_l321_32137


namespace NUMINAMATH_GPT_min_value_of_abc_l321_32181

noncomputable def minimum_value_abc (a b c : ℝ) : Prop :=
  (a > 0) ∧ (b > 0) ∧ (c > 0) ∧ ((a + c) * (a + b) = 6 - 2 * Real.sqrt 5) → (2 * a + b + c ≥ 2 * Real.sqrt 5 - 2)

theorem min_value_of_abc (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0)
  (h4 : (a + c) * (a + b) = 6 - 2 * Real.sqrt 5) : 
  2 * a + b + c ≥ 2 * Real.sqrt 5 - 2 :=
by 
  sorry

end NUMINAMATH_GPT_min_value_of_abc_l321_32181


namespace NUMINAMATH_GPT_scientific_notation_of_12_06_million_l321_32162

theorem scientific_notation_of_12_06_million :
  12.06 * 10^6 = 1.206 * 10^7 :=
sorry

end NUMINAMATH_GPT_scientific_notation_of_12_06_million_l321_32162


namespace NUMINAMATH_GPT_probability_red_purple_not_same_bed_l321_32139

def colors : Set String := {"red", "yellow", "white", "purple"}

def C (n k : ℕ) : ℕ := Nat.choose n k

theorem probability_red_purple_not_same_bed : 
  let total_ways := C 4 2
  let unwanted_ways := 2
  let desired_ways := total_ways - unwanted_ways
  let probability := (desired_ways : ℚ) / total_ways
  probability = 2 / 3 := by
  sorry

end NUMINAMATH_GPT_probability_red_purple_not_same_bed_l321_32139


namespace NUMINAMATH_GPT_trains_time_to_clear_each_other_l321_32197

noncomputable def relative_speed (v1 v2 : ℝ) : ℝ :=
  v1 + v2

noncomputable def speed_to_m_s (v_kmph : ℝ) : ℝ :=
  v_kmph * 1000 / 3600

noncomputable def total_length (l1 l2 : ℝ) : ℝ :=
  l1 + l2

theorem trains_time_to_clear_each_other :
  ∀ (l1 l2 : ℝ) (v1_kmph v2_kmph : ℝ),
    l1 = 100 → l2 = 280 →
    v1_kmph = 42 → v2_kmph = 30 →
    (total_length l1 l2) / (speed_to_m_s (relative_speed v1_kmph v2_kmph)) = 19 :=
by
  intros l1 l2 v1_kmph v2_kmph h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  sorry

end NUMINAMATH_GPT_trains_time_to_clear_each_other_l321_32197


namespace NUMINAMATH_GPT_f_zero_eq_zero_f_periodic_l321_32188

def odd_function {α : Type*} [AddGroup α] (f : α → α) : Prop :=
∀ x, f (-x) = -f (x)

def symmetric_about (c : ℝ) (f : ℝ → ℝ) : Prop :=
∀ x, f (c + x) = f (c - x)

variable (f : ℝ → ℝ)
variables (h_odd : odd_function f) (h_sym : symmetric_about 1 f)

theorem f_zero_eq_zero : f 0 = 0 :=
sorry

theorem f_periodic : ∀ x, f (x + 4) = f x :=
sorry

end NUMINAMATH_GPT_f_zero_eq_zero_f_periodic_l321_32188


namespace NUMINAMATH_GPT_book_price_increase_l321_32109

theorem book_price_increase (P : ℝ) (x : ℝ) :
  (P * (1 + x / 100)^2 = P * 1.3225) → x = 15 :=
by
  sorry

end NUMINAMATH_GPT_book_price_increase_l321_32109


namespace NUMINAMATH_GPT_problem_l321_32189

theorem problem (a b : ℝ) (h1 : a^2 - b^2 = 10) (h2 : a^4 + b^4 = 228) :
  a * b = 8 :=
sorry

end NUMINAMATH_GPT_problem_l321_32189


namespace NUMINAMATH_GPT_hyperbola_asymptote_b_value_l321_32124

theorem hyperbola_asymptote_b_value (b : ℝ) (hb : b > 0)
  (asymptote : ∀ x y : ℝ, y = 2 * x → x^2 - (y^2 / b^2) = 1) :
  b = 2 :=
sorry

end NUMINAMATH_GPT_hyperbola_asymptote_b_value_l321_32124


namespace NUMINAMATH_GPT_right_triangle_area_l321_32125

theorem right_triangle_area
  (hypotenuse : ℝ) (angle : ℝ) (hyp_eq : hypotenuse = 12) (angle_eq : angle = 30) :
  ∃ area : ℝ, area = 18 * Real.sqrt 3 :=
by
  have side1 := hypotenuse / 2  -- Shorter leg = hypotenuse / 2
  have side2 := side1 * Real.sqrt 3  -- Longer leg = shorter leg * sqrt 3
  let area := (side1 * side2) / 2  -- Area calculation
  use area
  sorry

end NUMINAMATH_GPT_right_triangle_area_l321_32125


namespace NUMINAMATH_GPT_cyrus_written_pages_on_fourth_day_l321_32103

theorem cyrus_written_pages_on_fourth_day :
  ∀ (total_pages first_day second_day third_day fourth_day remaining_pages: ℕ),
  total_pages = 500 →
  first_day = 25 →
  second_day = 2 * first_day →
  third_day = 2 * second_day →
  remaining_pages = total_pages - (first_day + second_day + third_day + fourth_day) →
  remaining_pages = 315 →
  fourth_day = 10 :=
by
  intros total_pages first_day second_day third_day fourth_day remaining_pages
  intros h_total h_first h_second h_third h_remain h_needed
  sorry

end NUMINAMATH_GPT_cyrus_written_pages_on_fourth_day_l321_32103


namespace NUMINAMATH_GPT_division_of_positive_by_negative_l321_32133

theorem division_of_positive_by_negative :
  4 / (-2) = -2 := 
by
  sorry

end NUMINAMATH_GPT_division_of_positive_by_negative_l321_32133


namespace NUMINAMATH_GPT_map_length_representation_l321_32169

variable (x : ℕ)

theorem map_length_representation :
  (12 : ℕ) * x = 17 * (72 : ℕ) / 12
:=
sorry

end NUMINAMATH_GPT_map_length_representation_l321_32169


namespace NUMINAMATH_GPT_angle_SQR_l321_32112

-- Define angles
def PQR : ℝ := 40
def PQS : ℝ := 28

-- State the theorem
theorem angle_SQR : PQR - PQS = 12 := by
  sorry

end NUMINAMATH_GPT_angle_SQR_l321_32112


namespace NUMINAMATH_GPT_bob_total_candies_l321_32170

noncomputable def total_chewing_gums : ℕ := 45
noncomputable def total_chocolate_bars : ℕ := 60
noncomputable def total_assorted_candies : ℕ := 45

def chewing_gum_ratio_sam_bob : ℕ × ℕ := (2, 3)
def chocolate_bar_ratio_sam_bob : ℕ × ℕ := (3, 1)
def assorted_candy_ratio_sam_bob : ℕ × ℕ := (1, 1)

theorem bob_total_candies :
  let bob_chewing_gums := (total_chewing_gums * chewing_gum_ratio_sam_bob.snd) / (chewing_gum_ratio_sam_bob.fst + chewing_gum_ratio_sam_bob.snd)
  let bob_chocolate_bars := (total_chocolate_bars * chocolate_bar_ratio_sam_bob.snd) / (chocolate_bar_ratio_sam_bob.fst + chocolate_bar_ratio_sam_bob.snd)
  let bob_assorted_candies := (total_assorted_candies * assorted_candy_ratio_sam_bob.snd) / (assorted_candy_ratio_sam_bob.fst + assorted_candy_ratio_sam_bob.snd)
  bob_chewing_gums + bob_chocolate_bars + bob_assorted_candies = 64 := by
  sorry

end NUMINAMATH_GPT_bob_total_candies_l321_32170


namespace NUMINAMATH_GPT_school_class_student_count_l321_32195

theorem school_class_student_count
  (num_classes : ℕ) (num_students : ℕ)
  (h_classes : num_classes = 30)
  (h_students : num_students = 1000)
  (h_max_students_per_class : ∀(n : ℕ), n < 30 → ∀(s : ℕ), s ≤ 33 → s ≤ 1000 / 30) :
  ∃ c, c ≤ num_classes ∧ ∃s, s ≥ 34 :=
by
  sorry

end NUMINAMATH_GPT_school_class_student_count_l321_32195


namespace NUMINAMATH_GPT_minimum_value_f_l321_32131

noncomputable def f (x : ℝ) : ℝ := x^2 + 3 * x + 6 / x + 4 / x^2 - 1

theorem minimum_value_f : 
    ∃ (x : ℝ), x > 0 ∧ 
    (∀ (y : ℝ), y > 0 → f y ≥ f x) ∧ 
    f x = 3 - 6 * Real.sqrt 2 :=
sorry

end NUMINAMATH_GPT_minimum_value_f_l321_32131


namespace NUMINAMATH_GPT_double_and_halve_is_sixteen_l321_32127

-- Definition of the initial number
def initial_number : ℕ := 16

-- Doubling the number
def doubled (n : ℕ) : ℕ := n * 2

-- Halving the number
def halved (n : ℕ) : ℕ := n / 2

-- The theorem that needs to be proven
theorem double_and_halve_is_sixteen : halved (doubled initial_number) = 16 :=
by
  /-
  We need to prove that when the number 16 is doubled and then halved, 
  the result is 16.
  -/
  sorry

end NUMINAMATH_GPT_double_and_halve_is_sixteen_l321_32127


namespace NUMINAMATH_GPT_smallest_square_area_l321_32163

variable (M N : ℝ)

/-- Given that the largest square has an area of 1 cm^2, the middle square has an area M cm^2, and the smallest square has a vertex on the side of the middle square, prove that the area of the smallest square N is equal to ((1 - M) / 2)^2. -/
theorem smallest_square_area (h1 : 1 ≥ 0)
  (h2 : 0 ≤ M ∧ M ≤ 1)
  (h3 : 0 ≤ N) :
  N = (1 - M) ^ 2 / 4 := sorry

end NUMINAMATH_GPT_smallest_square_area_l321_32163


namespace NUMINAMATH_GPT_find_other_endpoint_l321_32157

theorem find_other_endpoint (mx my x₁ y₁ x₂ y₂ : ℤ) 
  (h1 : mx = (x₁ + x₂) / 2) 
  (h2 : my = (y₁ + y₂) / 2) 
  (h3 : mx = 3) 
  (h4 : my = 4) 
  (h5 : x₁ = -2) 
  (h6 : y₁ = -5) : 
  x₂ = 8 ∧ y₂ = 13 := 
by
  sorry

end NUMINAMATH_GPT_find_other_endpoint_l321_32157


namespace NUMINAMATH_GPT_evaluate_expression_l321_32146

theorem evaluate_expression : 
  (1 / 2 + ((2 / 3 * (3 / 8)) + 4) - (8 / 16)) = (17 / 4) :=
by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l321_32146


namespace NUMINAMATH_GPT_inequality_solution_l321_32166

theorem inequality_solution (x : ℝ) : (x - 1) / 3 > 2 → x > 7 :=
by
  intros h
  sorry

end NUMINAMATH_GPT_inequality_solution_l321_32166


namespace NUMINAMATH_GPT_find_missing_number_l321_32147

theorem find_missing_number (x : ℕ) : 
  (1 + 22 + 23 + 24 + 25 + 26 + x + 2) / 8 = 20 → x = 37 := by
  sorry

end NUMINAMATH_GPT_find_missing_number_l321_32147


namespace NUMINAMATH_GPT_cell_count_at_end_of_twelvth_day_l321_32114

def initial_cells : ℕ := 5
def days_per_cycle : ℕ := 3
def total_days : ℕ := 12
def dead_cells_on_ninth_day : ℕ := 3
noncomputable def cells_after_twelvth_day : ℕ :=
  let cycles := total_days / days_per_cycle
  let cells_before_death := initial_cells * 2^cycles
  cells_before_death - dead_cells_on_ninth_day

theorem cell_count_at_end_of_twelvth_day : cells_after_twelvth_day = 77 :=
by sorry

end NUMINAMATH_GPT_cell_count_at_end_of_twelvth_day_l321_32114


namespace NUMINAMATH_GPT_full_day_students_count_l321_32150

-- Define the conditions
def total_students : ℕ := 80
def percentage_half_day_students : ℕ := 25

-- Define the statement to prove
theorem full_day_students_count :
  total_students - (total_students * percentage_half_day_students / 100) = 60 :=
by
  sorry

end NUMINAMATH_GPT_full_day_students_count_l321_32150


namespace NUMINAMATH_GPT_sum_of_number_and_square_is_306_l321_32122

theorem sum_of_number_and_square_is_306 : ∃ x : ℤ, x + x^2 = 306 ∧ x = 17 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_number_and_square_is_306_l321_32122


namespace NUMINAMATH_GPT_percent_of_475_25_is_129_89_l321_32113

theorem percent_of_475_25_is_129_89 :
  (129.89 / 475.25) * 100 = 27.33 :=
by
  sorry

end NUMINAMATH_GPT_percent_of_475_25_is_129_89_l321_32113


namespace NUMINAMATH_GPT_problem_1_problem_2_l321_32100

-- Definitions required for the proof
variables {A B C : ℝ} (a b c : ℝ)
variable (cos_A cos_B cos_C : ℝ)
variables (sin_A sin_C : ℝ)

-- Given conditions
axiom given_condition : (cos_A - 2 * cos_C) / cos_B = (2 * c - a) / b
axiom cos_B_eq : cos_B = 1 / 4
axiom b_eq : b = 2

-- First problem: Proving the value of sin_C / sin_A
theorem problem_1 :
  (cos_A - 2 * cos_C) / cos_B = (2 * c - a) / b → (sin_C / sin_A) = 2 :=
by
  intro h
  sorry

-- Second problem: Proving the area of triangle ABC
theorem problem_2 :
  (cos_B = 1 / 4) → (b = 2) → ((cos_A - 2 * cos_C) / cos_B = (2 * c - a) / b) → (1 / 2 * a * c * sin_A) = (Real.sqrt 15) / 4 :=
by
  intros h1 h2 h3
  sorry

end NUMINAMATH_GPT_problem_1_problem_2_l321_32100


namespace NUMINAMATH_GPT_intersection_ellipse_line_range_b_l321_32136

theorem intersection_ellipse_line_range_b (b : ℝ) : 
  (∀ m : ℝ, ∃ x y : ℝ, x^2 + 2*y^2 = 3 ∧ y = m*x + b) ↔ 
  (- (Real.sqrt 6) / 2) ≤ b ∧ b ≤ (Real.sqrt 6) / 2 :=
by {
  sorry
}

end NUMINAMATH_GPT_intersection_ellipse_line_range_b_l321_32136


namespace NUMINAMATH_GPT_missing_angles_sum_l321_32107

theorem missing_angles_sum 
  (calculated_sum : ℕ) 
  (missed_angles_sum : ℕ)
  (total_corrections : ℕ)
  (polygon_angles : ℕ) 
  (h1 : calculated_sum = 2797) 
  (h2 : total_corrections = 2880) 
  (h3 : polygon_angles = total_corrections - calculated_sum) : 
  polygon_angles = 83 := by
  sorry

end NUMINAMATH_GPT_missing_angles_sum_l321_32107


namespace NUMINAMATH_GPT_prove_number_of_cows_l321_32132

-- Define the conditions: Cows, Sheep, Pigs, Total animals
variables (C S P : ℕ)

-- Condition 1: Twice as many sheep as cows
def condition1 : Prop := S = 2 * C

-- Condition 2: Number of Pigs is 3 times the number of sheep
def condition2 : Prop := P = 3 * S

-- Condition 3: Total number of animals is 108
def condition3 : Prop := C + S + P = 108

-- The theorem to prove
theorem prove_number_of_cows (h1 : condition1 C S) (h2 : condition2 S P) (h3 : condition3 C S P) : C = 12 :=
sorry

end NUMINAMATH_GPT_prove_number_of_cows_l321_32132


namespace NUMINAMATH_GPT_base_7_3516_is_1287_l321_32115

-- Definitions based on conditions
def base7_to_base10 (n : Nat) : Nat :=
  match n with
  | 3516 => 3 * 7^3 + 5 * 7^2 + 1 * 7^1 + 6 * 7^0
  | _ => 0

-- Proving the main question
theorem base_7_3516_is_1287 : base7_to_base10 3516 = 1287 := by
  sorry

end NUMINAMATH_GPT_base_7_3516_is_1287_l321_32115


namespace NUMINAMATH_GPT_abc_ineq_l321_32156

theorem abc_ineq (a b c : ℝ) (h1 : a ≥ b) (h2 : b ≥ c) (h3 : c ≥ 0) : 
  a^2 * b * (a - b) + b^2 * c * (b - c) + c^2 * a * (c - a) ≥ 0 := 
by 
  sorry

end NUMINAMATH_GPT_abc_ineq_l321_32156


namespace NUMINAMATH_GPT_exists_six_numbers_multiple_2002_l321_32164

theorem exists_six_numbers_multiple_2002 (a : Fin 41 → ℕ) (h : Function.Injective a) :
  ∃ (i j k l m n : Fin 41),
    i ≠ j ∧ k ≠ l ∧ m ≠ n ∧
    (a i - a j) * (a k - a l) * (a m - a n) % 2002 = 0 := sorry

end NUMINAMATH_GPT_exists_six_numbers_multiple_2002_l321_32164


namespace NUMINAMATH_GPT_model_A_sampling_l321_32187

theorem model_A_sampling (prod_A prod_B prod_C total_prod total_sampled : ℕ)
    (hA : prod_A = 1200) (hB : prod_B = 6000) (hC : prod_C = 2000)
    (htotal : total_prod = prod_A + prod_B + prod_C) (htotal_car : total_prod = 9200)
    (hsampled : total_sampled = 46) :
    (prod_A * total_sampled) / total_prod = 6 := by
  sorry

end NUMINAMATH_GPT_model_A_sampling_l321_32187


namespace NUMINAMATH_GPT_range_of_a_l321_32126

variable (a : ℝ)
def A (a : ℝ) := {x : ℝ | x^2 - 2*x + a > 0}

theorem range_of_a (h : 1 ∉ A a) : a ≤ 1 :=
by {
  sorry
}

end NUMINAMATH_GPT_range_of_a_l321_32126


namespace NUMINAMATH_GPT_share_of_y_is_63_l321_32167

theorem share_of_y_is_63 (x y z : ℝ) (h1 : y = 0.45 * x) (h2 : z = 0.50 * x) (h3 : x + y + z = 273) : y = 63 :=
by
  -- The proof will go here
  sorry

end NUMINAMATH_GPT_share_of_y_is_63_l321_32167


namespace NUMINAMATH_GPT_seq_sum_l321_32165

theorem seq_sum (r : ℚ) (x y : ℚ) (h1 : r = 1 / 4)
    (h2 : 1024 * r = x) (h3 : x * r = y) : 
    x + y = 320 := by
  sorry

end NUMINAMATH_GPT_seq_sum_l321_32165


namespace NUMINAMATH_GPT_average_speed_ratio_l321_32182

def eddy_distance := 450 -- distance from A to B in km
def eddy_time := 3 -- time taken by Eddy in hours
def freddy_distance := 300 -- distance from A to C in km
def freddy_time := 4 -- time taken by Freddy in hours

def avg_speed (distance : ℕ) (time : ℕ) : ℕ := distance / time

def eddy_avg_speed := avg_speed eddy_distance eddy_time
def freddy_avg_speed := avg_speed freddy_distance freddy_time

def speed_ratio (speed1 : ℕ) (speed2 : ℕ) : ℕ × ℕ := (speed1 / (gcd speed1 speed2), speed2 / (gcd speed1 speed2))

theorem average_speed_ratio : speed_ratio eddy_avg_speed freddy_avg_speed = (2, 1) :=
by
  sorry

end NUMINAMATH_GPT_average_speed_ratio_l321_32182


namespace NUMINAMATH_GPT_train_passing_time_l321_32160

theorem train_passing_time (length_of_train : ℝ) (speed_of_train_kmhr : ℝ) :
  length_of_train = 180 → speed_of_train_kmhr = 36 → (length_of_train / (speed_of_train_kmhr * (1000 / 3600))) = 18 :=
by
  intro h1 h2
  sorry

end NUMINAMATH_GPT_train_passing_time_l321_32160


namespace NUMINAMATH_GPT_value_of_expression_l321_32129

variable {x : ℝ}

theorem value_of_expression (h : x^2 - 3 * x = 2) : 3 * x^2 - 9 * x - 7 = -1 := by
  sorry

end NUMINAMATH_GPT_value_of_expression_l321_32129


namespace NUMINAMATH_GPT_net_percentage_change_l321_32155

-- Definitions based on given conditions
variables (P : ℝ) (P_post_decrease : ℝ) (P_post_increase : ℝ)

-- Conditions
def decreased_by_5_percent : Prop := P_post_decrease = P * (1 - 0.05)
def increased_by_10_percent : Prop := P_post_increase = P_post_decrease * (1 + 0.10)

-- Proof problem
theorem net_percentage_change (h1 : decreased_by_5_percent P P_post_decrease) (h2 : increased_by_10_percent P_post_decrease P_post_increase) : 
  ((P_post_increase - P) / P) * 100 = 4.5 :=
by
  -- The proof would go here
  sorry

end NUMINAMATH_GPT_net_percentage_change_l321_32155


namespace NUMINAMATH_GPT_corrected_mean_l321_32153

theorem corrected_mean (n : ℕ) (mean : ℝ) (obs1 obs2 : ℝ) (inc1 inc2 cor1 cor2 : ℝ)
    (h_num_obs : n = 50)
    (h_initial_mean : mean = 36)
    (h_incorrect1 : inc1 = 23) (h_correct1 : cor1 = 34)
    (h_incorrect2 : inc2 = 55) (h_correct2 : cor2 = 45)
    : (mean * n + (cor1 - inc1) + (cor2 - inc2)) / n = 36.02 := 
by 
  -- Insert steps to prove the theorem here
  sorry

end NUMINAMATH_GPT_corrected_mean_l321_32153


namespace NUMINAMATH_GPT_solve_circle_tangent_and_intercept_l321_32176

namespace CircleProblems

-- Condition: Circle equation
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 + 2 * x - 4 * y + 3 = 0

-- Problem 1: Equations of tangent lines with equal intercepts
def tangent_lines_with_equal_intercepts (l : ℝ → ℝ → Prop) : Prop :=
  (∀ x y : ℝ, l x y ↔ x + y + 1 = 0) ∨ (∀ x y : ℝ, l x y ↔ x + y - 3 = 0)

-- Problem 2: Equations of lines passing through origin and intercepted by the circle with a segment length of 2
def lines_intercepted_by_circle (l : ℝ → ℝ → Prop) : Prop :=
  (∀ x y : ℝ, l x y ↔ x = 0) ∨ (∀ x y : ℝ, l x y ↔ y = - (3 / 4) * x)

theorem solve_circle_tangent_and_intercept (l_tangent l_origin : ℝ → ℝ → Prop) :
  (∀ x y : ℝ, circle_eq x y → l_tangent x y) →
  tangent_lines_with_equal_intercepts l_tangent ∧ lines_intercepted_by_circle l_origin :=
by
  sorry

end CircleProblems

end NUMINAMATH_GPT_solve_circle_tangent_and_intercept_l321_32176


namespace NUMINAMATH_GPT_added_classes_l321_32111

def original_classes := 15
def students_per_class := 20
def new_total_students := 400

theorem added_classes : 
  new_total_students = original_classes * students_per_class + 5 * students_per_class :=
by
  sorry

end NUMINAMATH_GPT_added_classes_l321_32111
