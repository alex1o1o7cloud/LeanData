import Mathlib

namespace NUMINAMATH_GPT_cos_double_angle_value_l368_36870

theorem cos_double_angle_value (θ : ℝ) (h : Real.sin (Real.pi / 2 + θ) = 3 / 5) : 
  Real.cos (2 * θ) = -7 / 25 :=
by
  sorry

end NUMINAMATH_GPT_cos_double_angle_value_l368_36870


namespace NUMINAMATH_GPT_movie_sale_price_l368_36806

/-- 
Given the conditions:
- cost of actors: $1200
- number of people: 50
- cost of food per person: $3
- equipment rental costs twice as much as food and actors combined
- profit made: $5950

Prove that the selling price of the movie was $10,000.
-/
theorem movie_sale_price :
  let cost_of_actors := 1200
  let num_people := 50
  let food_cost_per_person := 3
  let total_food_cost := num_people * food_cost_per_person
  let combined_cost := total_food_cost + cost_of_actors
  let equipment_rental_cost := 2 * combined_cost
  let total_cost := cost_of_actors + total_food_cost + equipment_rental_cost
  let profit := 5950
  let sale_price := total_cost + profit
  sale_price = 10000 := 
by
  sorry

end NUMINAMATH_GPT_movie_sale_price_l368_36806


namespace NUMINAMATH_GPT_length_of_train_l368_36822

theorem length_of_train (speed_kmph : ℕ) (bridge_length_m : ℕ) (crossing_time_s : ℕ) 
  (h1 : speed_kmph = 45) (h2 : bridge_length_m = 220) (h3 : crossing_time_s = 30) :
  ∃ train_length_m : ℕ, train_length_m = 155 :=
by
  sorry

end NUMINAMATH_GPT_length_of_train_l368_36822


namespace NUMINAMATH_GPT_part1_part2_l368_36859

/-- Definition of set A as roots of the equation x^2 - 3x + 2 = 0 --/
def set_A : Set ℝ := {x | x ^ 2 - 3 * x + 2 = 0}

/-- Definition of set B as roots of the equation x^2 + (a - 1)x + a^2 - 5 = 0 --/
def set_B (a : ℝ) : Set ℝ := {x | x ^ 2 + (a - 1) * x + a ^ 2 - 5 = 0}

/-- Proof for intersection condition --/
theorem part1 (a : ℝ) : (set_A ∩ set_B a = {2}) → (a = -3 ∨ a = 1) := by
  sorry

/-- Proof for union condition --/
theorem part2 (a : ℝ) : (set_A ∪ set_B a = set_A) → (a ≤ -3 ∨ a > 7 / 3) := by
  sorry

end NUMINAMATH_GPT_part1_part2_l368_36859


namespace NUMINAMATH_GPT_eat_five_pounds_in_46_875_min_l368_36885

theorem eat_five_pounds_in_46_875_min
  (fat_rate : ℝ) (thin_rate : ℝ) (combined_rate : ℝ) (total_fruit : ℝ)
  (hf1 : fat_rate = 1 / 15)
  (hf2 : thin_rate = 1 / 25)
  (h_comb : combined_rate = fat_rate + thin_rate)
  (h_fruit : total_fruit = 5) :
  total_fruit / combined_rate = 46.875 :=
by
  sorry

end NUMINAMATH_GPT_eat_five_pounds_in_46_875_min_l368_36885


namespace NUMINAMATH_GPT_auditorium_rows_l368_36898

theorem auditorium_rows (x : ℕ) (hx : (320 / x + 4) * (x + 1) = 420) : x = 20 :=
by
  sorry

end NUMINAMATH_GPT_auditorium_rows_l368_36898


namespace NUMINAMATH_GPT_five_circles_intersect_l368_36876

-- Assume we have five circles
variables (circle1 circle2 circle3 circle4 circle5 : Set Point)

-- Assume every four of them intersect at a single point
axiom four_intersect (c1 c2 c3 c4 : Set Point) : ∃ p : Point, p ∈ c1 ∧ p ∈ c2 ∧ p ∈ c3 ∧ p ∈ c4

-- The goal is to prove that there exists a point through which all five circles pass.
theorem five_circles_intersect :
  (∃ p : Point, p ∈ circle1 ∧ p ∈ circle2 ∧ p ∈ circle3 ∧ p ∈ circle4 ∧ p ∈ circle5) :=
sorry

end NUMINAMATH_GPT_five_circles_intersect_l368_36876


namespace NUMINAMATH_GPT_geom_seq_sum_relation_l368_36819

variable {a : ℕ → ℝ}
variable {q : ℝ}

def is_geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
∀ n, a (n + 1) = a n * q

theorem geom_seq_sum_relation (h_geom : is_geometric_sequence a q)
  (h_pos : ∀ n, a n > 0) (h_q_ne_one : q ≠ 1) :
  a 1 + a 4 > a 2 + a 3 :=
by
  sorry

end NUMINAMATH_GPT_geom_seq_sum_relation_l368_36819


namespace NUMINAMATH_GPT_complex_number_powers_l368_36894

theorem complex_number_powers (i : ℂ) (hi : i^2 = -1) : i + i^2 + i^3 = -1 :=
sorry

end NUMINAMATH_GPT_complex_number_powers_l368_36894


namespace NUMINAMATH_GPT_employed_population_is_60_percent_l368_36852

def percent_employed (P : ℝ) (E : ℝ) : Prop :=
  ∃ (P_0 : ℝ) (E_male : ℝ) (E_female : ℝ),
    P_0 = P * 0.45 ∧    -- 45 percent of the population are employed males
    E_female = (E * 0.25) * P ∧   -- 25 percent of the employed people are females
    (0.75 * E = 0.45) ∧    -- 75 percent of the employed people are males which equals to 45% of the total population
    E = 0.6            -- 60% of the population are employed

theorem employed_population_is_60_percent (P : ℝ) (E : ℝ):
  percent_employed P E :=
by
  sorry

end NUMINAMATH_GPT_employed_population_is_60_percent_l368_36852


namespace NUMINAMATH_GPT_product_of_squares_l368_36862

theorem product_of_squares (a_1 a_2 a_3 b_1 b_2 b_3 : ℕ) (N : ℕ) (h1 : (a_1 * b_1)^2 = N) (h2 : (a_2 * b_2)^2 = N) (h3 : (a_3 * b_3)^2 = N) 
: (a_1^2 * b_1^2) = 36 ∨  (a_2^2 * b_2^2) = 36 ∨ (a_3^2 * b_3^2) = 36:= 
sorry

end NUMINAMATH_GPT_product_of_squares_l368_36862


namespace NUMINAMATH_GPT_percentage_of_500_l368_36847

theorem percentage_of_500 : (110 * 500) / 100 = 550 :=
by
  sorry

end NUMINAMATH_GPT_percentage_of_500_l368_36847


namespace NUMINAMATH_GPT_isosceles_triangle_perimeter_l368_36807

theorem isosceles_triangle_perimeter 
  (m : ℝ) 
  (h : 2 * m + 1 = 8) : 
  (m - 2) + 2 * 8 = 17.5 := 
by 
  sorry

end NUMINAMATH_GPT_isosceles_triangle_perimeter_l368_36807


namespace NUMINAMATH_GPT_calculate_expression_l368_36887

theorem calculate_expression (p q r s : ℝ)
  (h1 : p + q + r + s = 10)
  (h2 : p^2 + q^2 + r^2 + s^2 = 26) :
  6 * (p^4 + q^4 + r^4 + s^4) - (p^3 + q^3 + r^3 + s^3) =
    6 * ((p-1)^4 + (q-1)^4 + (r-1)^4 + (s-1)^4) - ((p-1)^3 + (q-1)^3 + (r-1)^3 + (s-1)^3) :=
by {
  sorry
}

end NUMINAMATH_GPT_calculate_expression_l368_36887


namespace NUMINAMATH_GPT_find_rabbits_l368_36834

theorem find_rabbits (heads rabbits chickens : ℕ) (h1 : rabbits + chickens = 40) (h2 : 4 * rabbits = 10 * 2 * chickens - 8) : rabbits = 33 :=
by
  -- We skip the proof here
  sorry

end NUMINAMATH_GPT_find_rabbits_l368_36834


namespace NUMINAMATH_GPT_polyhedron_euler_formula_l368_36845

variable (A F S : ℕ)
variable (closed_polyhedron : Prop)

theorem polyhedron_euler_formula (h : closed_polyhedron) : A + 2 = F + S := sorry

end NUMINAMATH_GPT_polyhedron_euler_formula_l368_36845


namespace NUMINAMATH_GPT_betty_oranges_l368_36851

-- Define the givens and result as Lean definitions and theorems
theorem betty_oranges (kg_apples : ℕ) (cost_apples_per_kg cost_oranges_per_kg total_cost_oranges num_oranges : ℕ) 
    (h1 : kg_apples = 3)
    (h2 : cost_apples_per_kg = 2)
    (h3 : cost_apples_per_kg * 2 = cost_oranges_per_kg)
    (h4 : 12 = total_cost_oranges)
    (h5 : total_cost_oranges / cost_oranges_per_kg = num_oranges) :
    num_oranges = 3 :=
sorry

end NUMINAMATH_GPT_betty_oranges_l368_36851


namespace NUMINAMATH_GPT_new_students_count_l368_36872

theorem new_students_count (x : ℕ) (avg_age_group new_avg_age avg_new_students : ℕ)
  (h1 : avg_age_group = 14) (h2 : new_avg_age = 15) (h3 : avg_new_students = 17)
  (initial_students : ℕ) (initial_avg_age : ℕ)
  (h4 : initial_students = 10) (h5 : initial_avg_age = initial_students * avg_age_group)
  (h6 : new_avg_age * (initial_students + x) = initial_avg_age + (x * avg_new_students)) :
  x = 5 := 
by
  sorry

end NUMINAMATH_GPT_new_students_count_l368_36872


namespace NUMINAMATH_GPT_quadratic_eq_real_roots_roots_diff_l368_36871

theorem quadratic_eq_real_roots (m : ℝ) : 
  ∃ x y : ℝ, x ≠ y ∧ 
  (x^2 + (m-2)*x - m = 0) ∧
  (y^2 + (m-2)*y - m = 0) := sorry

theorem roots_diff (m n : ℝ) (hm : m ≠ 0) (hn : n ≠ 0)
  (h_roots : (m^2 + (m-2)*m - m = 0) ∧ (n^2 + (m-2)*n - m = 0)) :
  m - n = 5/2 := sorry

end NUMINAMATH_GPT_quadratic_eq_real_roots_roots_diff_l368_36871


namespace NUMINAMATH_GPT_Isabel_paper_used_l368_36881

theorem Isabel_paper_used
  (initial_pieces : ℕ)
  (remaining_pieces : ℕ)
  (initial_condition : initial_pieces = 900)
  (remaining_condition : remaining_pieces = 744) :
  initial_pieces - remaining_pieces = 156 :=
by 
  -- Admitting the proof for now
  sorry

end NUMINAMATH_GPT_Isabel_paper_used_l368_36881


namespace NUMINAMATH_GPT_intersection_M_N_l368_36854

-- Definitions of the sets M and N based on the conditions
def M (x : ℝ) : Prop := ∃ (y : ℝ), y = Real.log (x^2 - 3*x - 4)
def N (y : ℝ) : Prop := ∃ (x : ℝ), y = 2^(x - 1)

-- The proof statement
theorem intersection_M_N : { x : ℝ | M x } ∩ { x : ℝ | ∃ y : ℝ, N y ∧ y = Real.log (x^2 - 3*x - 4) } = { x : ℝ | x > 4 } :=
by
  sorry

end NUMINAMATH_GPT_intersection_M_N_l368_36854


namespace NUMINAMATH_GPT_polygon_sides_l368_36868

theorem polygon_sides {n : ℕ} (h : (n - 2) * 180 = 1080) : n = 8 :=
by
  sorry

end NUMINAMATH_GPT_polygon_sides_l368_36868


namespace NUMINAMATH_GPT_problem1_problem2_l368_36821

def count_good_subsets (n : ℕ) : ℕ := 
if n % 2 = 1 then 2^(n - 1) 
else 2^(n - 1) - (1 / 2) * Nat.choose n (n / 2)

def sum_f_good_subsets (n : ℕ) : ℕ :=
if n % 2 = 1 then n * (n + 1) * 2^(n - 3) + (n + 1) / 4 * Nat.choose n ((n - 1) / 2)
else n * (n + 1) * 2^(n - 3) - (n / 2) * ((n / 2) + 1) * Nat.choose (n / 2) (n / 2)

theorem problem1 (n : ℕ)  :
  (count_good_subsets n = (if n % 2 = 1 then 2^(n - 1) else 2^(n - 1) - (1 / 2) * Nat.choose n (n / 2))) :=
sorry

theorem problem2 (n : ℕ) :
  (sum_f_good_subsets n = (if n % 2 = 1 then n * (n + 1) * 2^(n - 3) + (n + 1) / 4 * Nat.choose n ((n - 1) / 2)
  else n * (n + 1) * 2^(n - 3) - (n / 2) * ((n / 2) + 1) * Nat.choose (n / 2) (n / 2))) := 
sorry

end NUMINAMATH_GPT_problem1_problem2_l368_36821


namespace NUMINAMATH_GPT_students_in_photo_l368_36800

theorem students_in_photo (m n : ℕ) (h1 : n = m + 5) (h2 : n = m + 5 ∧ m = 3) : 
  m * n = 24 :=
by
  -- h1: n = m + 5    (new row is 4 students fewer)
  -- h2: m = 3        (all rows have the same number of students after rearrangement)
  -- Prove m * n = 24
  sorry

end NUMINAMATH_GPT_students_in_photo_l368_36800


namespace NUMINAMATH_GPT_attended_college_percentage_l368_36815

variable (total_boys : ℕ) (total_girls : ℕ) (percent_not_attend_boys : ℕ) (percent_not_attend_girls : ℕ)

def total_boys_attended_college (total_boys percent_not_attend_boys : ℕ) : ℕ :=
  total_boys - percent_not_attend_boys * total_boys / 100

def total_girls_attended_college (total_girls percent_not_attend_girls : ℕ) : ℕ :=
  total_girls - percent_not_attend_girls * total_girls / 100

noncomputable def total_student_attended_college (total_boys total_girls percent_not_attend_boys percent_not_attend_girls : ℕ) : ℕ :=
  total_boys_attended_college total_boys percent_not_attend_boys +
  total_girls_attended_college total_girls percent_not_attend_girls

noncomputable def percent_class_attended_college (total_boys total_girls percent_not_attend_boys percent_not_attend_girls : ℕ) : ℕ :=
  total_student_attended_college total_boys total_girls percent_not_attend_boys percent_not_attend_girls * 100 /
  (total_boys + total_girls)

theorem attended_college_percentage :
  total_boys = 300 → total_girls = 240 → percent_not_attend_boys = 30 → percent_not_attend_girls = 30 →
  percent_class_attended_college total_boys total_girls percent_not_attend_boys percent_not_attend_girls = 70 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  sorry

end NUMINAMATH_GPT_attended_college_percentage_l368_36815


namespace NUMINAMATH_GPT_least_small_barrels_l368_36866

theorem least_small_barrels (total_oil : ℕ) (large_barrel : ℕ) (small_barrel : ℕ) (L S : ℕ)
  (h1 : total_oil = 745) (h2 : large_barrel = 11) (h3 : small_barrel = 7)
  (h4 : 11 * L + 7 * S = 745) (h5 : total_oil - 11 * L = 7 * S) : S = 1 :=
by
  sorry

end NUMINAMATH_GPT_least_small_barrels_l368_36866


namespace NUMINAMATH_GPT_pen_cost_l368_36838

theorem pen_cost (x : ℝ) (h1 : 5 * x + x = 24) : x = 4 :=
by
  sorry

end NUMINAMATH_GPT_pen_cost_l368_36838


namespace NUMINAMATH_GPT_trigonometric_value_l368_36841

theorem trigonometric_value (α : ℝ) (h : Real.tan α = 2) :
  (2 * Real.sin α ^ 2 + 1) / Real.cos (2 * (α - Real.pi / 4)) = 13 / 4 := 
sorry

end NUMINAMATH_GPT_trigonometric_value_l368_36841


namespace NUMINAMATH_GPT_ceilings_left_correct_l368_36895

def total_ceilings : ℕ := 28
def ceilings_painted_this_week : ℕ := 12
def ceilings_painted_next_week : ℕ := ceilings_painted_this_week / 4
def ceilings_left_to_paint : ℕ := total_ceilings - (ceilings_painted_this_week + ceilings_painted_next_week)

theorem ceilings_left_correct : ceilings_left_to_paint = 13 := by
  sorry

end NUMINAMATH_GPT_ceilings_left_correct_l368_36895


namespace NUMINAMATH_GPT_B_starts_6_hours_after_A_l368_36809

theorem B_starts_6_hours_after_A 
    (A_walk_speed : ℝ) (B_cycle_speed : ℝ) (catch_up_distance : ℝ)
    (hA : A_walk_speed = 10) (hB : B_cycle_speed = 20) (hD : catch_up_distance = 120) :
    ∃ t : ℝ, t = 6 :=
by
  sorry

end NUMINAMATH_GPT_B_starts_6_hours_after_A_l368_36809


namespace NUMINAMATH_GPT_pumpkin_pie_filling_l368_36855

theorem pumpkin_pie_filling (price_per_pumpkin : ℕ) (total_earnings : ℕ) (total_pumpkins : ℕ) (pumpkins_per_can : ℕ) :
  price_per_pumpkin = 3 →
  total_earnings = 96 →
  total_pumpkins = 83 →
  pumpkins_per_can = 3 →
  (total_pumpkins - total_earnings / price_per_pumpkin) / pumpkins_per_can = 17 :=
by
  intros h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_pumpkin_pie_filling_l368_36855


namespace NUMINAMATH_GPT_remainder_div_197_l368_36805

theorem remainder_div_197 (x q : ℕ) (h_pos : 0 < x) (h_div : 100 = q * x + 3) : 197 % x = 3 :=
sorry

end NUMINAMATH_GPT_remainder_div_197_l368_36805


namespace NUMINAMATH_GPT_number_of_green_balls_l368_36820

theorem number_of_green_balls (b g : ℕ) (h1 : b = 9) (h2 : (b : ℚ) / (b + g) = 3 / 10) : g = 21 :=
sorry

end NUMINAMATH_GPT_number_of_green_balls_l368_36820


namespace NUMINAMATH_GPT_verify_min_n_for_coprime_subset_l368_36896

def is_pairwise_coprime (s : Finset ℕ) : Prop :=
  ∀ (a b : ℕ) (ha : a ∈ s) (hb : b ∈ s), a ≠ b → Nat.gcd a b = 1

def contains_4_pairwise_coprime (s : Finset ℕ) : Prop :=
  ∃ t : Finset ℕ, t ⊆ s ∧ t.card = 4 ∧ is_pairwise_coprime t

def min_n_for_coprime_subset : ℕ :=
  111

theorem verify_min_n_for_coprime_subset (S : Finset ℕ) (hS : S = Finset.range 151) :
  ∀ (n : ℕ), (∀ s : Finset ℕ, s ⊆ S ∧ s.card = n → contains_4_pairwise_coprime s) ↔ (n ≥ min_n_for_coprime_subset) :=
sorry

end NUMINAMATH_GPT_verify_min_n_for_coprime_subset_l368_36896


namespace NUMINAMATH_GPT_salary_for_May_l368_36813

variable (J F M A May : ℕ)

axiom condition1 : (J + F + M + A) / 4 = 8000
axiom condition2 : (F + M + A + May) / 4 = 8800
axiom condition3 : J = 3300

theorem salary_for_May : May = 6500 :=
by sorry

end NUMINAMATH_GPT_salary_for_May_l368_36813


namespace NUMINAMATH_GPT_no_triples_of_consecutive_numbers_l368_36865

theorem no_triples_of_consecutive_numbers (n : ℤ) (a : ℕ) (h : 1 ≤ a ∧ a ≤ 9) :
  ¬(3 * n^2 + 2 = 1111 * a) :=
by sorry

end NUMINAMATH_GPT_no_triples_of_consecutive_numbers_l368_36865


namespace NUMINAMATH_GPT_min_value_of_box_l368_36863

theorem min_value_of_box 
  (a b : ℤ) 
  (h_distinct : a ≠ b) 
  (h_eq : (a * x + b) * (b * x + a) = 34 * x^2 + Box * x + 34) 
  (h_prod : a * b = 34) :
  ∃ (Box : ℤ), Box = 293 :=
by
  sorry

end NUMINAMATH_GPT_min_value_of_box_l368_36863


namespace NUMINAMATH_GPT_differences_impossible_l368_36839

def sum_of_digits (n : ℕ) : ℕ :=
  -- A simple definition for the sum of digits function
  n.digits 10 |>.sum

theorem differences_impossible (a : Fin 100 → ℕ) :
    ¬∃ (perm : Fin 100 → Fin 100), 
      (∀ i, a i - sum_of_digits (a (perm (i : ℕ) % 100)) = i + 1) :=
by
  sorry

end NUMINAMATH_GPT_differences_impossible_l368_36839


namespace NUMINAMATH_GPT_perfect_squares_less_than_20000_representable_l368_36897

-- Define what it means for a number to be a perfect square
def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

-- Define the difference of two consecutive perfect squares
def consecutive_difference (b : ℕ) : ℕ :=
  (b + 1) ^ 2 - b ^ 2

-- Define the condition under which the perfect square is less than 20000
def less_than_20000 (n : ℕ) : Prop :=
  n < 20000

-- Define the main problem statement using the above definitions
theorem perfect_squares_less_than_20000_representable :
  ∃ count : ℕ, (∀ n : ℕ, (is_perfect_square n) ∧ (less_than_20000 n) →
  ∃ b : ℕ, n = consecutive_difference b) ∧ count = 69 :=
sorry

end NUMINAMATH_GPT_perfect_squares_less_than_20000_representable_l368_36897


namespace NUMINAMATH_GPT_squares_in_ap_l368_36825

theorem squares_in_ap (a b c : ℝ) (h : (1 / (a + b) + 1 / (b + c)) / 2 = 1 / (a + c)) : 
  a^2 + c^2 = 2 * b^2 :=
by
  sorry

end NUMINAMATH_GPT_squares_in_ap_l368_36825


namespace NUMINAMATH_GPT_sum_a_b_l368_36811

theorem sum_a_b (a b : ℕ) (h_pos_a : a > 0) (h_pos_b : b > 2) (h_bound : a^b < 500)
  (h_max : ∀ a' b', a' > 0 → b' > 2 → a'^b' < 500 → a'^b' ≤ a^b) :
  a + b = 8 :=
by sorry

end NUMINAMATH_GPT_sum_a_b_l368_36811


namespace NUMINAMATH_GPT_problem_1_problem_2_problem_3_l368_36810

theorem problem_1 (x y : ℝ) : x^2 + y^2 + x * y + x + y ≥ -1 / 3 := 
by sorry

theorem problem_2 (x y z : ℝ) : x^2 + y^2 + z^2 + x * y + y * z + z * x + x + y + z ≥ -3 / 8 := 
by sorry

theorem problem_3 (x y z r : ℝ) : x^2 + y^2 + z^2 + r^2 + x * y + x * z + x * r + y * z + y * r + z * r + x + y + z + r ≥ -2 / 5 := 
by sorry

end NUMINAMATH_GPT_problem_1_problem_2_problem_3_l368_36810


namespace NUMINAMATH_GPT_jelly_bean_probabilities_l368_36892

theorem jelly_bean_probabilities :
  let p_red := 0.15
  let p_orange := 0.35
  let p_yellow := 0.2
  let p_green := 0.3
  p_red + p_orange + p_yellow + p_green = 1 :=
by
  sorry

end NUMINAMATH_GPT_jelly_bean_probabilities_l368_36892


namespace NUMINAMATH_GPT_greatest_common_factor_of_two_digit_palindromes_is_11_l368_36814

-- Define a two-digit palindrome
def is_two_digit_palindrome (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100 ∧ (n / 10 = n % 10)

-- Define the GCD of the set of all such numbers
def GCF_two_digit_palindromes : ℕ :=
  gcd (11 * 1) (gcd (11 * 2) (gcd (11 * 3) (gcd (11 * 4)
  (gcd (11 * 5) (gcd (11 * 6) (gcd (11 * 7) (gcd (11 * 8) (11 * 9))))))))

-- The statement to prove
theorem greatest_common_factor_of_two_digit_palindromes_is_11 :
  GCF_two_digit_palindromes = 11 :=
by
  sorry

end NUMINAMATH_GPT_greatest_common_factor_of_two_digit_palindromes_is_11_l368_36814


namespace NUMINAMATH_GPT_cosine_value_l368_36857

def a : ℝ × ℝ := (2, -1)
def b : ℝ × ℝ := (3, 4)

noncomputable def dot_product (x y : ℝ × ℝ) : ℝ :=
  x.1 * y.1 + x.2 * y.2

noncomputable def magnitude (x : ℝ × ℝ) : ℝ :=
  (x.1 ^ 2 + x.2 ^ 2).sqrt

noncomputable def cos_angle (a b : ℝ × ℝ) : ℝ :=
  dot_product a b / (magnitude a * magnitude b)

theorem cosine_value :
  cos_angle a b = 2 * (5:ℝ).sqrt / 25 :=
by
  sorry

end NUMINAMATH_GPT_cosine_value_l368_36857


namespace NUMINAMATH_GPT_find_m_n_l368_36878

theorem find_m_n : ∃ (m n : ℕ), m^m + (m * n)^n = 1984 ∧ m = 4 ∧ n = 3 := by
  sorry

end NUMINAMATH_GPT_find_m_n_l368_36878


namespace NUMINAMATH_GPT_factorable_polynomial_l368_36801

theorem factorable_polynomial (d f e g b : ℤ) (h1 : d * f = 28) (h2 : e * g = 14)
  (h3 : d * g + e * f = b) : b = 42 :=
by sorry

end NUMINAMATH_GPT_factorable_polynomial_l368_36801


namespace NUMINAMATH_GPT_smallest_positive_integer_n_l368_36879

theorem smallest_positive_integer_n (n : ℕ) (h : 5 * n ≡ 1463 [MOD 26]) : n = 23 :=
sorry

end NUMINAMATH_GPT_smallest_positive_integer_n_l368_36879


namespace NUMINAMATH_GPT_range_of_a_l368_36874

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, 1 ≤ x ∧ x ≤ Real.exp 1 → Real.exp (a * x) ≥ 2 * Real.log x + x^2 - a * x) ↔ 0 ≤ a :=
sorry

end NUMINAMATH_GPT_range_of_a_l368_36874


namespace NUMINAMATH_GPT_solve_system_l368_36812

theorem solve_system :
  {p : ℝ × ℝ | p.1^3 + p.2^3 = 19 ∧ p.1^2 + p.2^2 + 5 * p.1 + 5 * p.2 + p.1 * p.2 = 12} = {(3, -2), (-2, 3)} :=
sorry

end NUMINAMATH_GPT_solve_system_l368_36812


namespace NUMINAMATH_GPT_sum_of_repeating_decimals_l368_36843

noncomputable def x : ℚ := 1 / 9
noncomputable def y : ℚ := 2 / 99
noncomputable def z : ℚ := 3 / 999

theorem sum_of_repeating_decimals :
  x + y + z = 134 / 999 := by
  sorry

end NUMINAMATH_GPT_sum_of_repeating_decimals_l368_36843


namespace NUMINAMATH_GPT_circle_equation_l368_36864

theorem circle_equation (C : ℝ → ℝ → Prop)
  (h₁ : C 1 0)
  (h₂ : C 0 (Real.sqrt 3))
  (h₃ : C (-3) 0) :
  ∃ D E F : ℝ, (∀ x y, C x y ↔ x^2 + y^2 + D * x + E * y + F = 0) ∧ D = 2 ∧ E = 0 ∧ F = -3 := 
by
  sorry

end NUMINAMATH_GPT_circle_equation_l368_36864


namespace NUMINAMATH_GPT_find_original_height_l368_36831

noncomputable def original_height : ℝ := by
  let H := 102.19
  sorry

lemma ball_rebound (H : ℝ) : 
  (H + 2 * 0.8 * H + 2 * 0.56 * H + 2 * 0.336 * H + 2 * 0.168 * H + 2 * 0.0672 * H + 2 * 0.02016 * H = 500) :=
by
  sorry

theorem find_original_height : original_height = 102.19 :=
by
  have h := ball_rebound original_height
  sorry

end NUMINAMATH_GPT_find_original_height_l368_36831


namespace NUMINAMATH_GPT_bells_toll_together_l368_36853

noncomputable def LCM (a b : Nat) : Nat := (a * b) / (Nat.gcd a b)

theorem bells_toll_together :
  let intervals := [2, 4, 6, 8, 10, 12]
  let lcm := intervals.foldl LCM 1
  lcm = 120 →
  let duration := 30 * 60 -- 1800 seconds
  let tolls := duration / lcm
  tolls + 1 = 16 :=
by
  sorry

end NUMINAMATH_GPT_bells_toll_together_l368_36853


namespace NUMINAMATH_GPT_part1_part2_l368_36877

def f (x : ℝ) : ℝ := abs (x + 3) + abs (x - 2)

theorem part1 : {x : ℝ | f x > 7} = {x : ℝ | x < -4} ∪ {x : ℝ | x > 3} :=
by
  sorry

theorem part2 (m : ℝ) (h : m > 1) : ∃ x : ℝ, f x = (4 / (m - 1)) + m :=
by
  sorry

end NUMINAMATH_GPT_part1_part2_l368_36877


namespace NUMINAMATH_GPT_motorcyclist_travel_distances_l368_36844

-- Define the total distance traveled in three days
def total_distance : ℕ := 980

-- Define the total distance traveled in the first two days
def first_two_days_distance : ℕ := 725

-- Define the extra distance traveled on the second day compared to the third day
def second_day_extra : ℕ := 123

-- Define the distances traveled on the first, second, and third days respectively
def day_1_distance : ℕ := 347
def day_2_distance : ℕ := 378
def day_3_distance : ℕ := 255

-- Formalize the theorem statement
theorem motorcyclist_travel_distances :
  total_distance = day_1_distance + day_2_distance + day_3_distance ∧
  first_two_days_distance = day_1_distance + day_2_distance ∧
  day_2_distance = day_3_distance + second_day_extra :=
by 
  sorry

end NUMINAMATH_GPT_motorcyclist_travel_distances_l368_36844


namespace NUMINAMATH_GPT_product_of_sums_of_conjugates_l368_36861

theorem product_of_sums_of_conjugates :
  let a := 8 - Real.sqrt 500
  let b := 8 + Real.sqrt 500
  let c := 12 - Real.sqrt 72
  let d := 12 + Real.sqrt 72
  (a + b) * (c + d) = 384 :=
by
  sorry

end NUMINAMATH_GPT_product_of_sums_of_conjugates_l368_36861


namespace NUMINAMATH_GPT_Katie_homework_problems_l368_36849

theorem Katie_homework_problems :
  let finished_problems := 5
  let remaining_problems := 4
  let total_problems := finished_problems + remaining_problems
  total_problems = 9 :=
by
  sorry

end NUMINAMATH_GPT_Katie_homework_problems_l368_36849


namespace NUMINAMATH_GPT_problem1_problem2_problem3_l368_36803

variables (x y a b c : ℚ)

-- Definition of the operation *
def op_star (x y : ℚ) : ℚ := x * y + 1

-- Prove that 2 * 3 = 7 using the operation *
theorem problem1 : op_star 2 3 = 7 :=
by
  sorry

-- Prove that (1 * 4) * (-1/2) = -3/2 using the operation *
theorem problem2 : op_star (op_star 1 4) (-1/2) = -3/2 :=
by
  sorry

-- Prove the relationship a * (b + c) + 1 = a * b + a * c using the operation *
theorem problem3 : op_star a (b + c) + 1 = op_star a b + op_star a c :=
by
  sorry

end NUMINAMATH_GPT_problem1_problem2_problem3_l368_36803


namespace NUMINAMATH_GPT_max_rectangles_1x2_l368_36818

-- Define the problem conditions
def single_cell_squares : Type := sorry
def rectangles_1x2 (figure : single_cell_squares) : Prop := sorry

-- State the maximum number theorem
theorem max_rectangles_1x2 (figure : single_cell_squares) (h : rectangles_1x2 figure) :
  ∃ (n : ℕ), n ≤ 5 ∧ ∀ m : ℕ, rectangles_1x2 figure ∧ m ≤ 5 → m = 5 :=
sorry

end NUMINAMATH_GPT_max_rectangles_1x2_l368_36818


namespace NUMINAMATH_GPT_binary_111_eq_7_l368_36826

theorem binary_111_eq_7 : (1 * 2^0 + 1 * 2^1 + 1 * 2^2) = 7 :=
by
  sorry

end NUMINAMATH_GPT_binary_111_eq_7_l368_36826


namespace NUMINAMATH_GPT_range_of_real_number_m_l368_36833

open Set

variable {m : ℝ}

theorem range_of_real_number_m (U : Set ℝ) (A : Set ℝ) (B : Set ℝ) (h1 : U = univ) (h2 : A = { x | x < 1 }) (h3 : B = { x | x ≥ m }) (h4 : compl A ⊆ B) : m ≤ 1 := by
  sorry

end NUMINAMATH_GPT_range_of_real_number_m_l368_36833


namespace NUMINAMATH_GPT_excircle_diameter_l368_36804

noncomputable def diameter_of_excircle (a b c S : ℝ) (s : ℝ) : ℝ :=
  2 * S / (s - a)

theorem excircle_diameter (a b c S h_A : ℝ) (s : ℝ) (h_v : 2 * ((a + b + c) / 2) = a + b + c) :
    diameter_of_excircle a b c S s = 2 * S / (s - a) :=
by
  sorry

end NUMINAMATH_GPT_excircle_diameter_l368_36804


namespace NUMINAMATH_GPT_radar_coverage_proof_l368_36802

theorem radar_coverage_proof (n : ℕ) (r : ℝ) (w : ℝ) (d : ℝ) (A : ℝ) : 
  n = 9 ∧ r = 37 ∧ w = 24 ∧ d = 35 / Real.sin (Real.pi / 9) ∧
  A = 1680 * Real.pi / Real.tan (Real.pi / 9) → 
  ∃ OB S_ring, OB = d ∧ S_ring = A 
:= by sorry

end NUMINAMATH_GPT_radar_coverage_proof_l368_36802


namespace NUMINAMATH_GPT_C_alone_work_days_l368_36891

theorem C_alone_work_days (A_work_days B_work_days combined_work_days : ℝ) 
  (A_work_rate B_work_rate C_work_rate combined_work_rate : ℝ)
  (hA : A_work_days = 6)
  (hB : B_work_days = 5)
  (hCombined : combined_work_days = 2)
  (hA_work_rate : A_work_rate = 1 / A_work_days)
  (hB_work_rate : B_work_rate = 1 / B_work_days)
  (hCombined_work_rate : combined_work_rate = 1 / combined_work_days)
  (work_rate_eq : A_work_rate + B_work_rate + C_work_rate = combined_work_rate):
  (1 / C_work_rate) = 7.5 :=
by
  sorry

end NUMINAMATH_GPT_C_alone_work_days_l368_36891


namespace NUMINAMATH_GPT_meal_cost_l368_36873

theorem meal_cost:
  ∀ (s c p k : ℝ), 
  (2 * s + 5 * c + 2 * p + 3 * k = 6.30) →
  (3 * s + 8 * c + 2 * p + 4 * k = 8.40) →
  (s + c + p + k = 3.15) :=
by
  intros s c p k h1 h2
  sorry

end NUMINAMATH_GPT_meal_cost_l368_36873


namespace NUMINAMATH_GPT_partition_2004_ways_l368_36829

theorem partition_2004_ways : 
  ∀ n : ℕ, 1 ≤ n ∧ n ≤ 2004 → 
  ∃! (q r : ℕ), 2004 = q * n + r ∧ 0 ≤ r ∧ r < n :=
by
  sorry

end NUMINAMATH_GPT_partition_2004_ways_l368_36829


namespace NUMINAMATH_GPT_kenya_peanuts_correct_l368_36848

def jose_peanuts : ℕ := 85
def kenya_extra_peanuts : ℕ := 48
def kenya_peanuts : ℕ := jose_peanuts + kenya_extra_peanuts

theorem kenya_peanuts_correct : kenya_peanuts = 133 := 
by 
  sorry

end NUMINAMATH_GPT_kenya_peanuts_correct_l368_36848


namespace NUMINAMATH_GPT_expected_number_of_games_l368_36856
noncomputable def probability_of_A_winning (g : ℕ) : ℚ := 2 / 3
noncomputable def probability_of_B_winning (g : ℕ) : ℚ := 1 / 3
noncomputable def expected_games: ℚ := 266 / 81

theorem expected_number_of_games 
  (match_ends : ∀ g : ℕ, (∃ p1 p2 : ℕ, (p1 = g ∧ p2 = 0) ∨ (p1 = 0 ∧ p2 = g))) 
  (independent_outcomes : ∀ g1 g2 : ℕ, g1 ≠ g2 → probability_of_A_winning g1 * probability_of_A_winning g2 = (2 / 3) * (2 / 3) ∧ probability_of_B_winning g1 * probability_of_B_winning g2 = (1 / 3) * (1 / 3)) :
  (expected_games = 266 / 81) := 
sorry

end NUMINAMATH_GPT_expected_number_of_games_l368_36856


namespace NUMINAMATH_GPT_prime_divisor_form_l368_36823

theorem prime_divisor_form {p q : ℕ} (hp : Nat.Prime p) (hpgt2 : p > 2) (hq : Nat.Prime q) (hq_dvd : q ∣ 2^p - 1) : 
  ∃ k : ℕ, q = 2 * k * p + 1 := 
sorry

end NUMINAMATH_GPT_prime_divisor_form_l368_36823


namespace NUMINAMATH_GPT_birthday_cars_equal_12_l368_36840

namespace ToyCars

def initial_cars : Nat := 14
def bought_cars : Nat := 28
def sister_gave : Nat := 8
def friend_gave : Nat := 3
def remaining_cars : Nat := 43

def total_initial_cars := initial_cars + bought_cars
def total_given_away := sister_gave + friend_gave

theorem birthday_cars_equal_12 (B : Nat) (h : total_initial_cars + B - total_given_away = remaining_cars) : B = 12 :=
sorry

end ToyCars

end NUMINAMATH_GPT_birthday_cars_equal_12_l368_36840


namespace NUMINAMATH_GPT_log_expression_equals_eight_l368_36883

theorem log_expression_equals_eight :
  (Real.log 4 / Real.log 10) + 
  2 * (Real.log 5 / Real.log 10) + 
  3 * (Real.log 2 / Real.log 10) + 
  6 * (Real.log 5 / Real.log 10) + 
  (Real.log 8 / Real.log 10) = 8 := 
by 
  sorry

end NUMINAMATH_GPT_log_expression_equals_eight_l368_36883


namespace NUMINAMATH_GPT_certain_event_at_least_one_genuine_l368_36827

def products : Finset (Fin 12) := sorry
def genuine : Finset (Fin 12) := sorry
def defective : Finset (Fin 12) := sorry
noncomputable def draw3 : Finset (Finset (Fin 12)) := sorry

-- Condition: 12 identical products, 10 genuine, 2 defective
axiom products_condition_1 : products.card = 12
axiom products_condition_2 : genuine.card = 10
axiom products_condition_3 : defective.card = 2
axiom products_condition_4 : ∀ x ∈ genuine, x ∈ products
axiom products_condition_5 : ∀ x ∈ defective, x ∈ products
axiom products_condition_6 : genuine ∩ defective = ∅

-- The statement to be proved: when drawing 3 products randomly, it is certain that at least 1 is genuine.
theorem certain_event_at_least_one_genuine :
  ∀ s ∈ draw3, ∃ x ∈ s, x ∈ genuine :=
sorry

end NUMINAMATH_GPT_certain_event_at_least_one_genuine_l368_36827


namespace NUMINAMATH_GPT_determine_remainder_l368_36882

-- Define the sequence and its sum
def geom_series_sum_mod (a r n m : ℕ) : ℕ := 
  ((r^(n+1) - 1) / (r - 1)) % m

-- Define the specific geometric series and modulo
theorem determine_remainder :
  geom_series_sum_mod 1 11 1800 500 = 1 :=
by
  -- Using geom_series_sum_mod to define the series
  let S := geom_series_sum_mod 1 11 1800 500
  -- Remainder when the series is divided by 500
  show S = 1
  sorry

end NUMINAMATH_GPT_determine_remainder_l368_36882


namespace NUMINAMATH_GPT_union_A_B_l368_36893

def A : Set ℝ := {x : ℝ | -2 ≤ x ∧ x ≤ 0}
def B : Set ℝ := {x : ℝ | 0 < x ∧ x ≤ 3}

theorem union_A_B :
  A ∪ B = {x : ℝ | -2 ≤ x ∧ x ≤ 3} :=
sorry

end NUMINAMATH_GPT_union_A_B_l368_36893


namespace NUMINAMATH_GPT_extended_fishing_rod_length_l368_36858

def original_length : ℝ := 48
def increase_factor : ℝ := 1.33
def extended_length (orig_len : ℝ) (factor : ℝ) : ℝ := orig_len * factor

theorem extended_fishing_rod_length : extended_length original_length increase_factor = 63.84 :=
  by
    -- proof goes here
    sorry

end NUMINAMATH_GPT_extended_fishing_rod_length_l368_36858


namespace NUMINAMATH_GPT_pipe_c_empty_time_l368_36808

theorem pipe_c_empty_time :
  (1 / 45 + 1 / 60 - x = 1 / 40) → (1 / x = 72) :=
by
  sorry

end NUMINAMATH_GPT_pipe_c_empty_time_l368_36808


namespace NUMINAMATH_GPT_midpoint_product_l368_36816

theorem midpoint_product (x y : ℝ) (h1 : (4 : ℝ) = (x + 10) / 2) (h2 : (-2 : ℝ) = (-6 + y) / 2) : x * y = -4 := by
  sorry

end NUMINAMATH_GPT_midpoint_product_l368_36816


namespace NUMINAMATH_GPT_coefficient_sum_eq_512_l368_36880

theorem coefficient_sum_eq_512 (a_0 a_1 a_2 a_3 a_4 a_5 a_6 a_7 a_8 a_9 : ℤ) :
  (1 - x) ^ 9 = a_0 + a_1 * x + a_2 * x^2 + a_3 * x^3 + a_4 * x^4 + a_5 * x^5 + 
                a_6 * x^6 + a_7 * x^7 + a_8 * x^8 + a_9 * x^9 →
  |a_0| + |a_1| + |a_2| + |a_3| + |a_4| + |a_5| + |a_6| + |a_7| + |a_8| + |a_9| = 512 :=
sorry

end NUMINAMATH_GPT_coefficient_sum_eq_512_l368_36880


namespace NUMINAMATH_GPT_sum_of_reciprocals_of_factors_of_12_l368_36837

theorem sum_of_reciprocals_of_factors_of_12 : 
  (1 : ℚ) + (1 / 2) + (1 / 3) + (1 / 4) + (1 / 6) + (1 / 12) = 7 / 3 := 
by 
  sorry

end NUMINAMATH_GPT_sum_of_reciprocals_of_factors_of_12_l368_36837


namespace NUMINAMATH_GPT_average_growth_rate_bing_dwen_dwen_l368_36890

noncomputable def sales_growth_rate (v0 v2 : ℕ) (x : ℝ) : Prop :=
  (1 + x) ^ 2 = (v2 : ℝ) / (v0 : ℝ)

theorem average_growth_rate_bing_dwen_dwen :
  ∀ (v0 v2 : ℕ) (x : ℝ),
    v0 = 10000 →
    v2 = 12100 →
    sales_growth_rate v0 v2 x →
    x = 0.1 :=
by
  intros v0 v2 x h₀ h₂ h_growth
  sorry

end NUMINAMATH_GPT_average_growth_rate_bing_dwen_dwen_l368_36890


namespace NUMINAMATH_GPT_range_of_a_if_f_has_three_zeros_l368_36875

def f (a x : ℝ) : ℝ := x^3 + a * x + 2

theorem range_of_a_if_f_has_three_zeros (a : ℝ) : 
  (∃ x1 x2 x3 : ℝ, f a x1 = 0 ∧ f a x2 = 0 ∧ f a x3 = 0 ∧ x1 ≠ x2 ∧ x2 ≠ x3 ∧ x1 ≠ x3) ↔ a < -3 := 
by
  sorry

end NUMINAMATH_GPT_range_of_a_if_f_has_three_zeros_l368_36875


namespace NUMINAMATH_GPT_seats_required_l368_36824

def children := 58
def per_seat := 2
def seats_needed (children : ℕ) (per_seat : ℕ) := children / per_seat

theorem seats_required : seats_needed children per_seat = 29 := 
by
  sorry

end NUMINAMATH_GPT_seats_required_l368_36824


namespace NUMINAMATH_GPT_total_filled_water_balloons_l368_36860

theorem total_filled_water_balloons :
  let max_rate := 2
  let max_time := 30
  let zach_rate := 3
  let zach_time := 40
  let popped_balloons := 10
  let max_balloons := max_rate * max_time
  let zach_balloons := zach_rate * zach_time
  let total_balloons := max_balloons + zach_balloons - popped_balloons
  total_balloons = 170 :=
by
  sorry

end NUMINAMATH_GPT_total_filled_water_balloons_l368_36860


namespace NUMINAMATH_GPT_highest_number_on_dice_l368_36835

theorem highest_number_on_dice (n : ℕ) (h1 : 0 < n)
  (h2 : ∃ p : ℝ, p = 0.1111111111111111) 
  (h3 : 1 / 9 = 4 / (n * n)) 
  : n = 6 :=
sorry

end NUMINAMATH_GPT_highest_number_on_dice_l368_36835


namespace NUMINAMATH_GPT_emergency_vehicle_reachable_area_l368_36869

theorem emergency_vehicle_reachable_area :
  let speed_roads := 60 -- velocity on roads in miles per hour
    let speed_sand := 10 -- velocity on sand in miles per hour
    let time_limit := 5 / 60 -- time limit in hours
    let max_distance_on_roads := speed_roads * time_limit -- max distance on roads
    let radius_sand_circle := (10 / 12) -- radius on the sand
    -- calculate area covered
  (5 * 5 + 4 * (1 / 4 * Real.pi * (radius_sand_circle)^2)) = (25 + (25 * Real.pi) / 36) :=
by
  sorry

end NUMINAMATH_GPT_emergency_vehicle_reachable_area_l368_36869


namespace NUMINAMATH_GPT_distinct_distances_l368_36836

theorem distinct_distances (points : Finset (ℝ × ℝ)) (h : points.card = 2016) :
  ∃ s : Finset ℝ, s.card ≥ 45 ∧ ∀ p ∈ points, ∃ q ∈ points, p ≠ q ∧ 
    (s = (points.image (λ r => dist p r)).filter (λ x => x ≠ 0)) :=
by
  sorry

end NUMINAMATH_GPT_distinct_distances_l368_36836


namespace NUMINAMATH_GPT_count_multiples_of_7_not_14_lt_400_l368_36867

theorem count_multiples_of_7_not_14_lt_400 : 
  ∃ (n : ℕ), n = 29 ∧ ∀ (m : ℕ), (m < 400 ∧ m % 7 = 0 ∧ m % 14 ≠ 0) ↔ (∃ k : ℕ, 1 ≤ k ∧ k ≤ 29 ∧ m = 7 * (2 * k - 1)) :=
by
  sorry

end NUMINAMATH_GPT_count_multiples_of_7_not_14_lt_400_l368_36867


namespace NUMINAMATH_GPT_positive_value_of_X_l368_36846

-- Definition for the problem's conditions
def hash (X Y : ℝ) : ℝ := X^2 + Y^2

-- Statement of the proof problem
theorem positive_value_of_X (X : ℝ) (h : hash X 7 = 170) : X = 11 :=
by
  sorry

end NUMINAMATH_GPT_positive_value_of_X_l368_36846


namespace NUMINAMATH_GPT_cow_manure_growth_percentage_l368_36899

variable (control_height bone_meal_growth_percentage cow_manure_height : ℝ)
variable (bone_meal_height : ℝ := bone_meal_growth_percentage * control_height)
variable (percentage_growth : ℝ := (cow_manure_height / bone_meal_height) * 100)

theorem cow_manure_growth_percentage 
  (h₁ : control_height = 36)
  (h₂ : bone_meal_growth_percentage = 1.25)
  (h₃ : cow_manure_height = 90) :
  percentage_growth = 200 :=
by {
  sorry
}

end NUMINAMATH_GPT_cow_manure_growth_percentage_l368_36899


namespace NUMINAMATH_GPT_exponential_decreasing_l368_36889

theorem exponential_decreasing (a : ℝ) : (∀ x y : ℝ, x < y → (2 * a - 1)^y < (2 * a - 1)^x) ↔ (1 / 2 < a ∧ a < 1) := 
by
    sorry

end NUMINAMATH_GPT_exponential_decreasing_l368_36889


namespace NUMINAMATH_GPT_total_bill_is_correct_l368_36832

-- Define conditions as constant values
def cost_per_scoop : ℕ := 2
def pierre_scoops : ℕ := 3
def mom_scoops : ℕ := 4

-- Define the total bill calculation
def total_bill := (pierre_scoops * cost_per_scoop) + (mom_scoops * cost_per_scoop)

-- State the theorem that the total bill equals 14
theorem total_bill_is_correct : total_bill = 14 := by
  sorry

end NUMINAMATH_GPT_total_bill_is_correct_l368_36832


namespace NUMINAMATH_GPT_factor_expression_l368_36884

variable (x y : ℝ)

theorem factor_expression :
  4 * x ^ 2 - 4 * x - y ^ 2 + 4 * y - 3 = (2 * x + y - 3) * (2 * x - y + 1) := by
  sorry

end NUMINAMATH_GPT_factor_expression_l368_36884


namespace NUMINAMATH_GPT_production_average_l368_36850

-- Define the conditions and question
theorem production_average (n : ℕ) (P : ℕ) (P_new : ℕ) (h1 : P = n * 70) (h2 : P_new = P + 90) (h3 : P_new = (n + 1) * 75) : n = 3 := 
by sorry

end NUMINAMATH_GPT_production_average_l368_36850


namespace NUMINAMATH_GPT_b_investment_correct_l368_36817

-- Constants for shares and investments
def a_investment : ℕ := 11000
def a_share : ℕ := 2431
def b_share : ℕ := 3315
def c_investment : ℕ := 23000

-- Goal: Prove b's investment given the conditions
theorem b_investment_correct (b_investment : ℕ) (h : 2431 * b_investment = 11000 * 3315) :
  b_investment = 15000 := by
  sorry

end NUMINAMATH_GPT_b_investment_correct_l368_36817


namespace NUMINAMATH_GPT_a_9_equals_18_l368_36830

def is_sequence_of_positive_integers (a : ℕ → ℕ) : Prop :=
∀ n : ℕ, 0 < n → 0 < a n

def satisfies_recursive_relation (a : ℕ → ℕ) : Prop :=
∀ p q : ℕ, 0 < p → 0 < q → a (p + q) = a p + a q

theorem a_9_equals_18 (a : ℕ → ℕ)
  (H1 : is_sequence_of_positive_integers a)
  (H2 : satisfies_recursive_relation a)
  (H3 : a 2 = 4) : a 9 = 18 :=
sorry

end NUMINAMATH_GPT_a_9_equals_18_l368_36830


namespace NUMINAMATH_GPT_determine_dimensions_l368_36842

theorem determine_dimensions (a b : ℕ) (h : a < b) 
    (h1 : ∃ (m n : ℕ), 49 * 51 = (m * a) * (n * b))
    (h2 : ∃ (p q : ℕ), 99 * 101 = (p * a) * (q * b)) : 
    a = 1 ∧ b = 3 :=
  by 
  sorry

end NUMINAMATH_GPT_determine_dimensions_l368_36842


namespace NUMINAMATH_GPT_sin_pi_six_minus_two_alpha_l368_36828

theorem sin_pi_six_minus_two_alpha (α : ℝ) (h : Real.sin (π / 3 - α) = 1 / 3) : 
  Real.sin (π / 6 - 2 * α) = - 7 / 9 :=
by
  sorry

end NUMINAMATH_GPT_sin_pi_six_minus_two_alpha_l368_36828


namespace NUMINAMATH_GPT_cookies_taken_in_four_days_l368_36888

def initial_cookies : ℕ := 70
def cookies_left : ℕ := 28
def days_in_week : ℕ := 7
def days_taken : ℕ := 4
def daily_cookies_taken (total_cookies_taken : ℕ) : ℕ := total_cookies_taken / days_in_week
def total_cookies_taken : ℕ := initial_cookies - cookies_left

theorem cookies_taken_in_four_days :
  daily_cookies_taken total_cookies_taken * days_taken = 24 := by
  sorry

end NUMINAMATH_GPT_cookies_taken_in_four_days_l368_36888


namespace NUMINAMATH_GPT_rearrange_possible_l368_36886

theorem rearrange_possible (n : ℕ) (h : n = 25 ∨ n = 1000) :
  ∃ (f : ℕ → ℕ), (∀ i < n, f i + 1 < n → (f (i + 1) - f i = 3 ∨ f (i + 1) - f i = 5)) :=
  sorry

end NUMINAMATH_GPT_rearrange_possible_l368_36886
