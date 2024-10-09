import Mathlib

namespace white_surface_fraction_l1695_169582

-- Definition of the problem conditions
def larger_cube_surface_area : ℕ := 54
def white_cubes : ℕ := 6
def white_surface_area_minimized : ℕ := 5

-- Theorem statement proving the fraction of white surface area
theorem white_surface_fraction : (white_surface_area_minimized / larger_cube_surface_area : ℚ) = 5 / 54 := 
by
  sorry

end white_surface_fraction_l1695_169582


namespace smallest_altitude_le_3_l1695_169598

theorem smallest_altitude_le_3 (a b c h_a h_b h_c : ℝ) (r : ℝ) (h_r : r = 1)
    (h_a_ge_b : a ≥ b) (h_b_ge_c : b ≥ c) 
    (area_eq1 : (a + b + c) / 2 * r = (a * h_a) / 2) 
    (area_eq2 : (a + b + c) / 2 * r = (b * h_b) / 2) 
    (area_eq3 : (a + b + c) / 2 * r = (c * h_c) / 2) : 
    min h_a (min h_b h_c) ≤ 3 := 
by
  sorry

end smallest_altitude_le_3_l1695_169598


namespace study_time_for_average_l1695_169586

theorem study_time_for_average
    (study_time_exam1 score_exam1 : ℕ)
    (study_time_exam2 score_exam2 average_score desired_average : ℝ)
    (relation : score_exam1 = 20 * study_time_exam1)
    (direct_relation : score_exam2 = 20 * study_time_exam2)
    (total_exams : ℕ)
    (average_condition : (score_exam1 + score_exam2) / total_exams = desired_average) :
    study_time_exam2 = 4.5 :=
by
  have : total_exams = 2 := by sorry
  have : score_exam1 = 60 := by sorry
  have : desired_average = 75 := by sorry
  have : score_exam2 = 90 := by sorry
  sorry

end study_time_for_average_l1695_169586


namespace no_integer_roots_l1695_169527

theorem no_integer_roots (a b c : ℤ) (h1 : a ≠ 0) (h2 : a % 2 = 1) (h3 : b % 2 = 1) (h4 : c % 2 = 1) :
  ∀ x : ℤ, a * x^2 + b * x + c ≠ 0 :=
by
  sorry

end no_integer_roots_l1695_169527


namespace sixth_term_sequence_l1695_169577

theorem sixth_term_sequence (a b c d : ℚ)
  (h1 : a = 1/4 * (3 + b))
  (h2 : b = 1/4 * (a + c))
  (h3 : c = 1/4 * (b + 48))
  (h4 : 48 = 1/4 * (c + d)) :
  d = 2001 / 14 :=
sorry

end sixth_term_sequence_l1695_169577


namespace fraction_irreducible_l1695_169565

theorem fraction_irreducible (a b c d : ℤ) (h : a * d - b * c = 1) : ∀ m : ℤ, m > 1 → ¬ (m ∣ (a^2 + b^2) ∧ m ∣ (a * c + b * d)) :=
by sorry

end fraction_irreducible_l1695_169565


namespace find_digit_l1695_169528

theorem find_digit:
  ∃ d: ℕ, d < 1000 ∧ 1995 * d = 610470 :=
  sorry

end find_digit_l1695_169528


namespace sub_seq_arithmetic_l1695_169550

variable (a : ℕ → ℝ) (d : ℝ)

def is_arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + d

def sub_seq (a : ℕ → ℝ) (k : ℕ) : ℝ :=
  a (3 * k - 1)

theorem sub_seq_arithmetic (h : is_arithmetic_sequence a d) : is_arithmetic_sequence (sub_seq a) (3 * d) := 
sorry


end sub_seq_arithmetic_l1695_169550


namespace hash_of_hash_of_hash_of_70_l1695_169538

def hash (N : ℝ) : ℝ := 0.4 * N + 2

theorem hash_of_hash_of_hash_of_70 : hash (hash (hash 70)) = 8 := by
  sorry

end hash_of_hash_of_hash_of_70_l1695_169538


namespace pascal_row_with_ratio_456_exists_at_98_l1695_169580

theorem pascal_row_with_ratio_456_exists_at_98 :
  ∃ n, ∃ r, 0 ≤ r ∧ r + 2 ≤ n ∧ 
  ((Nat.choose n r : ℚ) / Nat.choose n (r + 1) = 4 / 5) ∧
  ((Nat.choose n (r + 1) : ℚ) / Nat.choose n (r + 2) = 5 / 6) ∧ 
  n = 98 := by
  sorry

end pascal_row_with_ratio_456_exists_at_98_l1695_169580


namespace parabola_intersection_l1695_169587

def parabola1 (x : ℝ) : ℝ := 3 * x ^ 2 - 4 * x + 2
def parabola2 (x : ℝ) : ℝ := 9 * x ^ 2 + 6 * x + 2

theorem parabola_intersection :
  ∃ (x y : ℝ), (parabola1 x = y ∧ parabola2 x = y) ∧ 
                ((x = 0 ∧ y = 2) ∨ (x = -5 / 3 ∧ y = 17)) :=
by
  sorry

end parabola_intersection_l1695_169587


namespace mass_percentage_O_in_C6H8O6_l1695_169558

theorem mass_percentage_O_in_C6H8O6 :
  let atomic_mass_C := 12.01
  let atomic_mass_H := 1.01
  let atomic_mass_O := 16.00
  let molar_mass_C6H8O6 := (6 * atomic_mass_C) + (8 * atomic_mass_H) + (6 * atomic_mass_O)
  let mass_of_oxygen := 8 * atomic_mass_O
  let mass_percentage_O := (mass_of_oxygen / molar_mass_C6H8O6) * 100
  mass_percentage_O = 72.67 :=
by
  -- Definitions
  let atomic_mass_C := 12.01
  let atomic_mass_H := 1.01
  let atomic_mass_O := 16.00
  let molar_mass_C6H8O6 := (6 * atomic_mass_C) + (8 * atomic_mass_H) + (6 * atomic_mass_O)
  let mass_of_oxygen := 8 * atomic_mass_O
  let mass_percentage_O := (mass_of_oxygen / molar_mass_C6H8O6) * 100
  -- Proof
  sorry

end mass_percentage_O_in_C6H8O6_l1695_169558


namespace xyz_divisible_by_55_l1695_169571

-- Definitions and conditions from part (a)
variables (x y z a b c : ℤ)
variable (h1 : x^2 + y^2 = a^2)
variable (h2 : y^2 + z^2 = b^2)
variable (h3 : z^2 + x^2 = c^2)

-- The final statement to prove that xyz is divisible by 55
theorem xyz_divisible_by_55 : 55 ∣ x * y * z := 
by sorry

end xyz_divisible_by_55_l1695_169571


namespace largest_sum_of_three_largest_angles_l1695_169561

-- Definitions and main theorem statement
theorem largest_sum_of_three_largest_angles (EFGH : Type*)
    (a b c d : ℝ) 
    (h1 : a + b + c + d = 360)
    (h2 : b = 3 * c)
    (h3 : ∃ (common_diff : ℝ), (c - a = common_diff) ∧ (b - c = common_diff) ∧ (d - b = common_diff))
    (h4 : ∀ (x y z : ℝ), (x = y + z) ↔ (∃ (progression_diff : ℝ), x - y = y - z ∧ y - z = z - x)) :
    (∃ (A B C D : ℝ), A = a ∧ B = b ∧ C = c ∧ D = d ∧ A + B + C + D = 360 ∧ A = max a (max b (max c d)) ∧ B = 2 * D ∧ A + B + C = 330) :=
sorry

end largest_sum_of_three_largest_angles_l1695_169561


namespace smallest_four_digit_equivalent_6_mod_7_l1695_169547

theorem smallest_four_digit_equivalent_6_mod_7 :
  (∃ (n : ℕ), n >= 1000 ∧ n < 10000 ∧ n % 7 = 6 ∧ (∀ (m : ℕ), m >= 1000 ∧ m < 10000 ∧ m % 7 = 6 → m >= n)) ∧ ∃ (n : ℕ), n = 1000 :=
sorry

end smallest_four_digit_equivalent_6_mod_7_l1695_169547


namespace parallel_vectors_l1695_169534

def a : ℝ × ℝ := (1, 2)
def b (m : ℝ) : ℝ × ℝ := (-1, m)

theorem parallel_vectors (m : ℝ) (h : (1 : ℝ) / (-1 : ℝ) = (2 : ℝ) / m) : m = -2 :=
sorry

end parallel_vectors_l1695_169534


namespace minimal_hair_loss_l1695_169513

theorem minimal_hair_loss (cards : Fin 100 → ℕ)
    (sum_sage1 : ℕ)
    (communicate_card_numbers : List ℕ)
    (communicate_sum : ℕ) :
    (∀ i : Fin 100, (communicate_card_numbers.contains (cards i))) →
    communicate_sum = sum_sage1 →
    sum_sage1 = List.sum communicate_card_numbers →
    communicate_card_numbers.length = 100 →
    ∃ (minimal_loss : ℕ), minimal_loss = 101 := by
  sorry

end minimal_hair_loss_l1695_169513


namespace problem_integer_square_l1695_169578

theorem problem_integer_square 
  (a b c d A : ℤ) 
  (H1 : a^2 + A = b^2) 
  (H2 : c^2 + A = d^2) : 
  ∃ (k : ℕ), 2 * (a + b) * (c + d) * (a * c + b * d - A) = k^2 :=
by
  sorry

end problem_integer_square_l1695_169578


namespace minimum_boxes_l1695_169535

theorem minimum_boxes (x y z : ℕ) (h1 : 50 * x = 40 * y) (h2 : 50 * x = 25 * z) :
  x + y + z = 17 :=
by
  -- Prove that given these equations, the minimum total number of boxes (x + y + z) is 17
  sorry

end minimum_boxes_l1695_169535


namespace gain_percent_l1695_169517

variables (MP CP SP : ℝ)

-- problem conditions
axiom h1 : CP = 0.64 * MP
axiom h2 : SP = 0.84 * MP

-- To prove: Gain percent is 31.25%
theorem gain_percent (CP MP SP : ℝ) (h1 : CP = 0.64 * MP) (h2 : SP = 0.84 * MP) :
  ((SP - CP) / CP) * 100 = 31.25 :=
by sorry

end gain_percent_l1695_169517


namespace count_correct_statements_l1695_169595

theorem count_correct_statements :
  ∃ (M: ℚ) (M1: ℚ) (M2: ℚ) (M3: ℚ) (M4: ℚ)
    (a b c d e : ℚ) (hacb : c ≠ 0) (habc: a ≠ 0) (hbcb : b ≠ 0) (hdcb: d ≠ 0) (hec: e ≠ 0),
  M = (ac + bd - ce) / c 
  ∧ M1 = (-bc - ad - ce) / c 
  ∧ M2 = (-dc - ab - ce) / c 
  ∧ M3 = (-dc - ab - de) / d 
  ∧ M4 = (ce - bd - ac) / (-c)
  ∧ M4 = M
  ∧ (M ≠ M3)
  ∧ (∀ M1, M1 = (-bc - ad - ce) / c → ((a = c ∨ b = d) ↔ b = d))
  ∧ (M4 = (ac + bd - ce)/c) :=
sorry

end count_correct_statements_l1695_169595


namespace intersecting_circle_radius_l1695_169532

-- Definitions representing the conditions
def non_intersecting_circles (O_i : Fin 6 → ℕ) (r_i : Fin 6 → ℝ) : Prop :=
  ∀ i j, i ≠ j → dist (O_i i) (O_i j) ≥ r_i i + r_i j

def min_radius_one (O_i : Fin 6 → ℕ) (r_i : Fin 6 → ℝ) := 
  ∀ i, r_i i ≥ 1

-- The main theorem stating the proof goal
theorem intersecting_circle_radius 
  (O_i : Fin 6 → ℕ) (r_i : Fin 6 → ℝ) (O : ℕ) (r : ℝ)
  (h_non_intersecting : non_intersecting_circles O_i r_i)
  (h_min_radius : min_radius_one O_i r_i)
  (h_intersecting : ∀ i, dist O (O_i i) ≤ r + r_i i) :
  r ≥ 1 := 
sorry

end intersecting_circle_radius_l1695_169532


namespace avg_ballpoint_pens_per_day_l1695_169521

theorem avg_ballpoint_pens_per_day (bundles_sold : ℕ) (pens_per_bundle : ℕ) (days : ℕ) (total_pens : ℕ) (avg_per_day : ℕ) 
  (h1 : bundles_sold = 15)
  (h2 : pens_per_bundle = 40)
  (h3 : days = 5)
  (h4 : total_pens = bundles_sold * pens_per_bundle)
  (h5 : avg_per_day = total_pens / days) :
  avg_per_day = 120 :=
by
  -- placeholder proof
  sorry

end avg_ballpoint_pens_per_day_l1695_169521


namespace hyperbola_focus_and_asymptotes_l1695_169588

def is_focus_on_y_axis (a b : ℝ) (eq : ℝ → ℝ → Prop) : Prop :=
∃ c : ℝ, eq (c^2 * a) (c^2 * b)

def are_asymptotes_perpendicular (eq : ℝ → ℝ → Prop) : Prop :=
∃ k1 k2 : ℝ, (k1 != 0 ∧ k2 != 0 ∧ eq k1 k2 ∧ eq (-k1) k2)

theorem hyperbola_focus_and_asymptotes :
  is_focus_on_y_axis 1 (-1) (fun y x => y^2 - x^2 = 4) ∧ are_asymptotes_perpendicular (fun y x => y = x) :=
by
  sorry

end hyperbola_focus_and_asymptotes_l1695_169588


namespace fish_caught_by_dad_l1695_169516

def total_fish_both : ℕ := 23
def fish_caught_morning : ℕ := 8
def fish_thrown_back : ℕ := 3
def fish_caught_afternoon : ℕ := 5
def fish_kept_brendan : ℕ := fish_caught_morning - fish_thrown_back + fish_caught_afternoon

theorem fish_caught_by_dad : total_fish_both - fish_kept_brendan = 13 := by
  sorry

end fish_caught_by_dad_l1695_169516


namespace ocean_depth_l1695_169552

/-
  Problem:
  Determine the depth of the ocean at the current location of the ship.
  
  Given conditions:
  - The signal sent by the echo sounder was received after 5 seconds.
  - The speed of sound in water is 1.5 km/s.

  Correct answer to prove:
  - The depth of the ocean is 3750 meters.
-/

theorem ocean_depth
  (v : ℝ) (t : ℝ) (depth : ℝ) 
  (hv : v = 1500) 
  (ht : t = 5) 
  (hdepth : depth = 3750) :
  depth = (v * t) / 2 :=
sorry

end ocean_depth_l1695_169552


namespace neighbors_have_even_total_bells_not_always_divisible_by_3_l1695_169536

def num_bushes : ℕ := 19

def is_neighbor (circ : ℕ → ℕ) (i j : ℕ) : Prop := 
  if i = num_bushes - 1 then j = 0
  else j = i + 1

-- Part (a)
theorem neighbors_have_even_total_bells (bells : Fin num_bushes → ℕ) :
  ∃ i : Fin num_bushes, (bells i + bells (⟨(i + 1) % num_bushes, sorry⟩ : Fin num_bushes)) % 2 = 0 := sorry

-- Part (b)
theorem not_always_divisible_by_3 (bells : Fin num_bushes → ℕ) :
  ¬ (∀ i : Fin num_bushes, (bells i + bells (⟨(i + 1) % num_bushes, sorry⟩ : Fin num_bushes)) % 3 = 0) := sorry

end neighbors_have_even_total_bells_not_always_divisible_by_3_l1695_169536


namespace tan_add_pi_over_3_l1695_169555

theorem tan_add_pi_over_3 (x : ℝ) (h : Real.tan x = 3) : 
  Real.tan (x + Real.pi / 3) = (3 + Real.sqrt 3) / (1 - 3 * Real.sqrt 3) :=
by
  sorry

end tan_add_pi_over_3_l1695_169555


namespace arc_length_ratio_l1695_169591

theorem arc_length_ratio
  (h_circ : ∀ (x y : ℝ), (x - 1)^2 + y^2 = 1)
  (h_line : ∀ x y : ℝ, x - y = 0) :
  let shorter_arc := (1 / 4) * (2 * Real.pi)
  let longer_arc := 2 * Real.pi - shorter_arc
  shorter_arc / longer_arc = 1 / 3 :=
by
  sorry

end arc_length_ratio_l1695_169591


namespace no_solution_inequality_l1695_169508

theorem no_solution_inequality (m : ℝ) : ¬(∃ x : ℝ, 2 * x - 1 > 1 ∧ x < m) → m ≤ 1 :=
by
  intro h
  sorry

end no_solution_inequality_l1695_169508


namespace tadd_2019th_number_l1695_169554

def next_start_point (n : ℕ) : ℕ := 
    1 + (n * (2 * 3 + (n - 1) * 9)) / 2

def block_size (n : ℕ) : ℕ := 
    1 + 3 * (n - 1)

def nth_number_said_by_tadd (n : ℕ) (k : ℕ) : ℕ :=
    let block_n := next_start_point n
    block_n + k - 1

theorem tadd_2019th_number :
    nth_number_said_by_tadd 37 2019 = 5979 := 
sorry

end tadd_2019th_number_l1695_169554


namespace region_area_proof_l1695_169581

noncomputable def region_area := 
  let region := {p : ℝ × ℝ | abs (p.1 - p.2^2 / 2) + p.1 + p.2^2 / 2 ≤ 2 - p.2}
  2 * (0.5 * (3 * (2 + 0.5)))

theorem region_area_proof : region_area = 15 / 2 :=
by
  sorry

end region_area_proof_l1695_169581


namespace sufficient_but_not_necessary_l1695_169539

theorem sufficient_but_not_necessary (x : ℝ) : (x > 0 → x * (x + 1) > 0) ∧ ¬ (x * (x + 1) > 0 → x > 0) := 
by 
sorry

end sufficient_but_not_necessary_l1695_169539


namespace find_cheese_calories_l1695_169562

noncomputable def lettuce_calories := 50
noncomputable def carrots_calories := 2 * lettuce_calories
noncomputable def dressing_calories := 210

noncomputable def crust_calories := 600
noncomputable def pepperoni_calories := crust_calories / 3

noncomputable def total_salad_calories := lettuce_calories + carrots_calories + dressing_calories
noncomputable def total_pizza_calories (cheese_calories : ℕ) := crust_calories + pepperoni_calories + cheese_calories

theorem find_cheese_calories (consumed_calories : ℕ) (cheese_calories : ℕ) :
  consumed_calories = 330 →
  1/4 * total_salad_calories + 1/5 * total_pizza_calories cheese_calories = consumed_calories →
  cheese_calories = 400 := by
  sorry

end find_cheese_calories_l1695_169562


namespace chantal_gain_l1695_169567

variable (sweaters balls cost_selling cost_yarn total_gain : ℕ)

def chantal_knits_sweaters : Prop :=
  sweaters = 28 ∧
  balls = 4 ∧
  cost_yarn = 6 ∧
  cost_selling = 35 ∧
  total_gain = (sweaters * cost_selling) - (sweaters * balls * cost_yarn)

theorem chantal_gain : chantal_knits_sweaters sweaters balls cost_selling cost_yarn total_gain → total_gain = 308 :=
by sorry

end chantal_gain_l1695_169567


namespace integer_solution_exists_l1695_169548

theorem integer_solution_exists (a : ℤ) : 
  (∃ k : ℤ, 2 * a^2 = 7 * k + 2) ↔ (a % 7 = 1 ∨ a % 7 = 6) :=
by sorry

end integer_solution_exists_l1695_169548


namespace prime_sum_of_primes_l1695_169590

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def is_square (n : ℕ) : Prop :=
  ∃ k : ℕ, k * k = n

def distinct_primes (p q r s : ℕ) : Prop :=
  is_prime p ∧ is_prime q ∧ is_prime r ∧ is_prime s ∧ p ≠ q ∧ p ≠ r ∧ p ≠ s ∧ q ≠ r ∧ q ≠ s ∧ r ≠ s

theorem prime_sum_of_primes (p q r s : ℕ) :
  distinct_primes p q r s →
  is_prime (p + q + r + s) →
  is_square (p^2 + q * s) →
  is_square (p^2 + q * r) →
  (p = 2 ∧ q = 7 ∧ r = 11 ∧ s = 3) ∨ (p = 2 ∧ q = 7 ∧ r = 3 ∧ s = 11) :=
by
  sorry

end prime_sum_of_primes_l1695_169590


namespace sum_of_coordinates_of_A_l1695_169570

theorem sum_of_coordinates_of_A
  (A B C : ℝ × ℝ)
  (AC AB BC : ℝ)
  (h1 : AC / AB = 1 / 3)
  (h2 : BC / AB = 2 / 3)
  (hB : B = (2, 5))
  (hC : C = (5, 8)) :
  (A.1 + A.2) = 16 :=
sorry

end sum_of_coordinates_of_A_l1695_169570


namespace polar_to_cartesian_l1695_169546

theorem polar_to_cartesian (ρ θ : ℝ) (h : ρ = 4 * Real.cos θ) :
  ∃ x y : ℝ, (x = ρ * Real.cos θ ∧ y = ρ * Real.sin θ ∧
  (x - 2)^2 + y^2 = 4) :=
sorry

end polar_to_cartesian_l1695_169546


namespace double_bed_heavier_than_single_bed_l1695_169524

theorem double_bed_heavier_than_single_bed 
  (S D : ℝ) 
  (h1 : 5 * S = 50) 
  (h2 : 2 * S + 4 * D = 100) 
  : D - S = 10 :=
sorry

end double_bed_heavier_than_single_bed_l1695_169524


namespace shaded_area_ratio_l1695_169511

theorem shaded_area_ratio
  (large_square_area : ℕ := 25)
  (grid_dimension : ℕ := 5)
  (shaded_square_area : ℕ := 2)
  (num_squares : ℕ := 25)
  (ratio : ℚ := 2 / 25) :
  (shaded_square_area : ℚ) / large_square_area = ratio := 
by
  sorry

end shaded_area_ratio_l1695_169511


namespace abs_m_minus_n_l1695_169531

theorem abs_m_minus_n (m n : ℝ) (h_avg : (m + n + 9 + 8 + 10) / 5 = 9) (h_var : (1 / 5 * (m^2 + n^2 + 81 + 64 + 100) - 81) = 2) : |m - n| = 4 :=
  sorry

end abs_m_minus_n_l1695_169531


namespace arrows_from_530_to_533_l1695_169525

-- Define what it means for the pattern to be cyclic with period 5
def cycle_period (n m : Nat) : Prop := n % m = 0

-- Define the equivalent points on the circular track
def equiv_point (n : Nat) (m : Nat) : Nat := n % m

-- Given conditions
def arrow_pattern : Prop :=
  ∀ n : Nat, cycle_period n 5 ∧
  (equiv_point 530 5 = 0) ∧ (equiv_point 533 5 = 3)

-- The theorem to be proved
theorem arrows_from_530_to_533 :
  (∃ seq : List (Nat × Nat),
    seq = [(0, 1), (1, 2), (2, 3)]) :=
sorry

end arrows_from_530_to_533_l1695_169525


namespace polynomial_abs_value_at_neg_one_l1695_169543

theorem polynomial_abs_value_at_neg_one:
  ∃ g : Polynomial ℝ, 
  (∀ x ∈ ({0, 1, 2, 4, 5, 6} : Set ℝ), |g.eval x| = 15) → 
  |g.eval (-1)| = 75 :=
by
  sorry

end polynomial_abs_value_at_neg_one_l1695_169543


namespace total_dolls_l1695_169519

def sisters_dolls : ℝ := 8.5

def hannahs_dolls : ℝ := 5.5 * sisters_dolls

theorem total_dolls : hannahs_dolls + sisters_dolls = 55.25 :=
by
  -- Proof is omitted
  sorry

end total_dolls_l1695_169519


namespace sequence_solution_l1695_169573

theorem sequence_solution (n : ℕ) (S : ℕ → ℕ) (a : ℕ → ℕ)
  (h : ∀ n, S n = 2 * a n - 2^n + 1) : a n = n * 2^(n-1) :=
sorry

end sequence_solution_l1695_169573


namespace complex_div_eq_i_l1695_169523

noncomputable def i := Complex.I

theorem complex_div_eq_i : (1 + i) / (1 - i) = i := 
by
  sorry

end complex_div_eq_i_l1695_169523


namespace seashells_total_l1695_169544

def seashells :=
  let sam_seashells := 18
  let mary_seashells := 47
  sam_seashells + mary_seashells

theorem seashells_total : seashells = 65 := by
  sorry

end seashells_total_l1695_169544


namespace percentage_loss_l1695_169592

variable (CP SP : ℝ)
variable (HCP : CP = 1600)
variable (HSP : SP = 1408)

theorem percentage_loss (HCP : CP = 1600) (HSP : SP = 1408) : 
  (CP - SP) / CP * 100 = 12 := by
sorry

end percentage_loss_l1695_169592


namespace price_of_sports_equipment_l1695_169542

theorem price_of_sports_equipment (x y : ℕ) (a b : ℕ) :
  (2 * x + y = 330) → (5 * x + 2 * y = 780) → x = 120 ∧ y = 90 ∧
  (120 * a + 90 * b = 810) → a = 3 ∧ b = 5 :=
by
  intros h1 h2 h3
  sorry

end price_of_sports_equipment_l1695_169542


namespace division_addition_rational_eq_l1695_169557

theorem division_addition_rational_eq :
  (3 / 7 / 4) + (1 / 2) = 17 / 28 :=
by
  sorry

end division_addition_rational_eq_l1695_169557


namespace alice_winning_strategy_l1695_169597

theorem alice_winning_strategy (n : ℕ) (h : n ≥ 2) :
  (∃ strategy : Π (s : ℕ), s < n → (ℕ × ℕ), 
    ∀ (k : ℕ) (hk : k < n), ¬(strategy k hk).fst = (strategy k hk).snd) ↔ (n % 4 = 3) :=
sorry

end alice_winning_strategy_l1695_169597


namespace roots_of_quadratic_l1695_169593

theorem roots_of_quadratic (p q : ℝ) (h1 : 3 * p^2 + 9 * p - 21 = 0) (h2 : 3 * q^2 + 9 * q - 21 = 0) :
  (3 * p - 4) * (6 * q - 8) = 122 := by
  -- We don't need to provide the proof here, only the statement
  sorry

end roots_of_quadratic_l1695_169593


namespace simplify_expression_l1695_169500

theorem simplify_expression (b : ℝ) (h : b ≠ -1) : 
  1 - (1 / (1 - (b / (1 + b)))) = -b :=
by {
  sorry
}

end simplify_expression_l1695_169500


namespace one_cow_one_bag_in_34_days_l1695_169549

-- Definitions: 34 cows eat 34 bags in 34 days, each cow eats one bag in those 34 days.
def cows : Nat := 34
def bags : Nat := 34
def days : Nat := 34

-- Hypothesis: each cow eats one bag in 34 days.
def one_bag_days (c : Nat) (b : Nat) : Nat := days

-- Theorem: One cow will eat one bag of husk in 34 days.
theorem one_cow_one_bag_in_34_days : one_bag_days 1 1 = 34 := sorry

end one_cow_one_bag_in_34_days_l1695_169549


namespace problem_l1695_169520

theorem problem
  (a b : ℚ)
  (h1 : 3 * a + 5 * b = 47)
  (h2 : 7 * a + 2 * b = 52)
  : a + b = 35 / 3 :=
sorry

end problem_l1695_169520


namespace find_y_from_x_squared_l1695_169568

theorem find_y_from_x_squared (x y : ℤ) (h1 : x^2 = y + 7) (h2 : x = 6) : y = 29 :=
by
  sorry

end find_y_from_x_squared_l1695_169568


namespace sin_arithmetic_sequence_l1695_169575

theorem sin_arithmetic_sequence (a : ℝ) (h : 0 < a ∧ a < 2 * Real.pi) : 
  (Real.sin a + Real.sin (3 * a) = 2 * Real.sin (2 * a)) ↔ (a = Real.pi) :=
sorry

end sin_arithmetic_sequence_l1695_169575


namespace trains_cross_time_l1695_169574

theorem trains_cross_time (length : ℝ) (time1 time2 : ℝ) (speed1 speed2 relative_speed : ℝ) 
  (H1 : length = 120) 
  (H2 : time1 = 12) 
  (H3 : time2 = 20) 
  (H4 : speed1 = length / time1) 
  (H5 : speed2 = length / time2) 
  (H6 : relative_speed = speed1 + speed2) 
  (total_distance : ℝ) (H7 : total_distance = length + length) 
  (T : ℝ) (H8 : T = total_distance / relative_speed) :
  T = 15 := 
sorry

end trains_cross_time_l1695_169574


namespace Lennon_total_reimbursement_l1695_169599

def mileage_reimbursement (industrial_weekday: ℕ → ℕ) (commercial_weekday: ℕ → ℕ) (weekend: ℕ → ℕ) : ℕ :=
  let industrial_rate : ℕ := 36
  let commercial_weekday_rate : ℕ := 42
  let weekend_rate : ℕ := 45
  (industrial_weekday 1 * industrial_rate + commercial_weekday 1 * commercial_weekday_rate)    -- Monday
  + (industrial_weekday 2 * industrial_rate + commercial_weekday 2 * commercial_weekday_rate + commercial_weekday 3 * commercial_weekday_rate)  -- Tuesday
  + (industrial_weekday 3 * industrial_rate + commercial_weekday 3 * commercial_weekday_rate)    -- Wednesday
  + (commercial_weekday 4 * commercial_weekday_rate + commercial_weekday 5 * commercial_weekday_rate)  -- Thursday
  + (industrial_weekday 5 * industrial_rate + commercial_weekday 6 * commercial_weekday_rate + industrial_weekday 6 * industrial_rate)    -- Friday
  + (weekend 1 * weekend_rate)                                       -- Saturday

def monday_industrial_miles : ℕ := 10
def monday_commercial_miles : ℕ := 8

def tuesday_industrial_miles : ℕ := 12
def tuesday_commercial_miles_1 : ℕ := 9
def tuesday_commercial_miles_2 : ℕ := 5

def wednesday_industrial_miles : ℕ := 15
def wednesday_commercial_miles : ℕ := 5

def thursday_commercial_miles_1 : ℕ := 10
def thursday_commercial_miles_2 : ℕ := 10

def friday_industrial_miles_1 : ℕ := 5
def friday_commercial_miles : ℕ := 8
def friday_industrial_miles_2 : ℕ := 3

def saturday_commercial_miles : ℕ := 12

def reimbursement_total :=
  mileage_reimbursement
    (fun day => if day = 1 then monday_industrial_miles else if day = 2 then tuesday_industrial_miles else if day = 3 then wednesday_industrial_miles else if day = 5 then friday_industrial_miles_1 + friday_industrial_miles_2 else 0)
    (fun day => if day = 1 then monday_commercial_miles else if day = 2 then tuesday_commercial_miles_1 + tuesday_commercial_miles_2 else if day = 3 then wednesday_commercial_miles else if day = 4 then thursday_commercial_miles_1 + thursday_commercial_miles_2 else if day = 6 then friday_commercial_miles else 0)
    (fun day => if day = 1 then saturday_commercial_miles else 0)

theorem Lennon_total_reimbursement : reimbursement_total = 4470 := 
by sorry

end Lennon_total_reimbursement_l1695_169599


namespace sequence_divisibility_24_l1695_169509

theorem sequence_divisibility_24 :
  ∀ (x : ℕ → ℕ), (x 0 = 2) → (x 1 = 3) →
    (∀ n : ℕ, x (n+2) = 7 * x (n+1) - x n + 280) →
    (∀ n : ℕ, (x n * x (n+1) + x (n+1) * x (n+2) + x (n+2) * x (n+3) + 2018) % 24 = 0) :=
by
  intro x h1 h2 h3
  sorry

end sequence_divisibility_24_l1695_169509


namespace esther_walks_975_yards_l1695_169512

def miles_to_feet (miles : ℕ) : ℕ := miles * 5280
def feet_to_yards (feet : ℕ) : ℕ := feet / 3

variable (lionel_miles : ℕ) (niklaus_feet : ℕ) (total_feet : ℕ) (esther_yards : ℕ)
variable (h_lionel : lionel_miles = 4)
variable (h_niklaus : niklaus_feet = 1287)
variable (h_total : total_feet = 25332)
variable (h_esther : esther_yards = 975)

theorem esther_walks_975_yards :
  let lionel_distance_in_feet := miles_to_feet lionel_miles
  let combined_distance := lionel_distance_in_feet + niklaus_feet
  let esther_distance_in_feet := total_feet - combined_distance
  feet_to_yards esther_distance_in_feet = esther_yards := by {
    sorry
  }

end esther_walks_975_yards_l1695_169512


namespace average_age_is_26_l1695_169510

noncomputable def devin_age : ℕ := 12
noncomputable def eden_age : ℕ := 2 * devin_age
noncomputable def eden_mom_age : ℕ := 2 * eden_age
noncomputable def eden_grandfather_age : ℕ := (devin_age + eden_age + eden_mom_age) / 2
noncomputable def eden_aunt_age : ℕ := eden_mom_age / devin_age

theorem average_age_is_26 : 
  (devin_age + eden_age + eden_mom_age + eden_grandfather_age + eden_aunt_age) / 5 = 26 :=
by {
  sorry
}

end average_age_is_26_l1695_169510


namespace problem_complement_intersection_l1695_169545

open Set

-- Define the universal set U
def U : Set ℕ := {0, 2, 4, 6, 8, 10}

-- Define set A
def A : Set ℕ := {0, 2, 4, 6}

-- Define set B based on A
def B : Set ℕ := {x | x ∈ A ∧ x < 4}

-- Define the complement of set A within U
def complement_A_U : Set ℕ := U \ A

-- Define the complement of set B within U
def complement_B_U : Set ℕ := U \ B

-- Prove the given equations
theorem problem_complement_intersection :
  (complement_A_U = {8, 10}) ∧ (A ∩ complement_B_U = {4, 6}) := 
by
  sorry

end problem_complement_intersection_l1695_169545


namespace election_votes_total_l1695_169579

theorem election_votes_total 
  (winner_votes : ℕ) (opponent1_votes opponent2_votes opponent3_votes : ℕ)
  (excess1 excess2 excess3 : ℕ)
  (h1 : winner_votes = opponent1_votes + excess1)
  (h2 : winner_votes = opponent2_votes + excess2)
  (h3 : winner_votes = opponent3_votes + excess3)
  (votes_winner : winner_votes = 195)
  (votes_opponent1 : opponent1_votes = 142)
  (votes_opponent2 : opponent2_votes = 116)
  (votes_opponent3 : opponent3_votes = 90)
  (he1 : excess1 = 53)
  (he2 : excess2 = 79)
  (he3 : excess3 = 105) :
  winner_votes + opponent1_votes + opponent2_votes + opponent3_votes = 543 :=
by sorry

end election_votes_total_l1695_169579


namespace find_remainder_l1695_169551

-- Given conditions
def dividend : ℕ := 144
def divisor : ℕ := 11
def quotient : ℕ := 13

-- Theorem statement
theorem find_remainder (dividend divisor quotient : ℕ) (h1 : dividend = divisor * quotient + 1):
  ∃ r, r = dividend % divisor := 
by 
  exists 1
  sorry

end find_remainder_l1695_169551


namespace sectionB_seats_correct_l1695_169533

-- Definitions for the number of seats in Section A
def seatsA_subsection1 : Nat := 60
def seatsA_subsection2 : Nat := 3 * 80
def totalSeatsA : Nat := seatsA_subsection1 + seatsA_subsection2

-- Condition for the number of seats in Section B
def seatsB : Nat := 3 * totalSeatsA + 20

-- Theorem statement to prove the number of seats in Section B
theorem sectionB_seats_correct : seatsB = 920 := by
  sorry

end sectionB_seats_correct_l1695_169533


namespace pairs_count_1432_1433_l1695_169589

def PairsCount (n : ℕ) : ℕ :=
  -- The implementation would count the pairs (x, y) such that |x^2 - y^2| = n
  sorry

-- We write down the theorem that expresses what we need to prove
theorem pairs_count_1432_1433 : PairsCount 1432 = 8 ∧ PairsCount 1433 = 4 := by
  sorry

end pairs_count_1432_1433_l1695_169589


namespace design_height_lower_part_l1695_169583

theorem design_height_lower_part (H : ℝ) (H_eq : H = 2) (L : ℝ) 
  (ratio : (H - L) / L = L / H) : L = Real.sqrt 5 - 1 :=
by {
  sorry
}

end design_height_lower_part_l1695_169583


namespace inequality_reciprocal_of_negatives_l1695_169563

theorem inequality_reciprocal_of_negatives (a b : ℝ) (ha : a < b) (hb : b < 0) : (1 / a) > (1 / b) :=
sorry

end inequality_reciprocal_of_negatives_l1695_169563


namespace part1_l1695_169522

   noncomputable def sin_20_deg_sq : ℝ := (Real.sin (20 * Real.pi / 180))^2
   noncomputable def cos_80_deg_sq : ℝ := (Real.sin (10 * Real.pi / 180))^2
   noncomputable def sqrt3_sin20_cos80 : ℝ := Real.sqrt 3 * Real.sin (20 * Real.pi / 180) * Real.sin (10 * Real.pi / 180)
   noncomputable def value : ℝ := sin_20_deg_sq + cos_80_deg_sq + sqrt3_sin20_cos80

   theorem part1 : value = 1 / 4 := by
     sorry
   
end part1_l1695_169522


namespace solve_floor_equation_l1695_169594

theorem solve_floor_equation (x : ℝ) :
  (⌊⌊2 * x⌋ - 1 / 2⌋ = ⌊x + 3⌋) ↔ (3.5 ≤ x ∧ x < 4.5) :=
sorry

end solve_floor_equation_l1695_169594


namespace find_p_l1695_169596

def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem find_p (p : ℕ) (h : is_prime p) (hpgt1 : 1 < p) :
  8 * p^4 - 3003 = 1997 ↔ p = 5 :=
by
  sorry

end find_p_l1695_169596


namespace algebraic_expression_value_l1695_169569

theorem algebraic_expression_value (x y : ℝ) (h : x - 2 * y^2 = 1) : -2 * x + 4 * y^2 + 1 = -1 :=
by
  sorry

end algebraic_expression_value_l1695_169569


namespace simplify_expression_l1695_169576

theorem simplify_expression : ((3 + 4 + 5 + 6) / 3) + ((3 * 6 + 9) / 4) = 12.75 := by
  sorry

end simplify_expression_l1695_169576


namespace total_sections_l1695_169529

theorem total_sections (boys girls gcd sections_boys sections_girls : ℕ) 
  (h_boys : boys = 408) 
  (h_girls : girls = 264) 
  (h_gcd: gcd = Nat.gcd boys girls)
  (h_sections_boys : sections_boys = boys / gcd)
  (h_sections_girls : sections_girls = girls / gcd)
  (h_total_sections : sections_boys + sections_girls = 28)
: sections_boys + sections_girls = 28 := by
  sorry

end total_sections_l1695_169529


namespace simplify_expression_l1695_169540

theorem simplify_expression (x : ℝ) : 7 * x + 8 - 3 * x + 14 = 4 * x + 22 :=
by
  sorry

end simplify_expression_l1695_169540


namespace odd_perfect_prime_form_n_is_seven_l1695_169503

theorem odd_perfect_prime_form (n p s m : ℕ) (h₁ : n % 2 = 1) (h₂ : ∃ k : ℕ, p = 4 * k + 1) (h₃ : ∃ h : ℕ, s = 4 * h + 1) (h₄ : n = p^s * m^2) (h₅ : ¬ p ∣ m) :
  ∃ k h : ℕ, p = 4 * k + 1 ∧ s = 4 * h + 1 :=
sorry

theorem n_is_seven (n : ℕ) (h₁ : n > 1) (h₂ : ∃ k : ℕ, k * k = n -1) (h₃ : ∃ l : ℕ, l * l = (n * (n + 1)) / 2) :
  n = 7 :=
sorry

end odd_perfect_prime_form_n_is_seven_l1695_169503


namespace common_ratio_of_geometric_series_l1695_169506

theorem common_ratio_of_geometric_series (a b : ℚ) (h1 : a = 4 / 7) (h2 : b = 12 / 7) : b / a = 3 := by
  sorry

end common_ratio_of_geometric_series_l1695_169506


namespace percentage_customers_not_pay_tax_l1695_169537

theorem percentage_customers_not_pay_tax
  (daily_shoppers : ℕ)
  (weekly_tax_payers : ℕ)
  (h1 : daily_shoppers = 1000)
  (h2 : weekly_tax_payers = 6580)
  : ((7000 - weekly_tax_payers) / 7000) * 100 = 6 := 
by sorry

end percentage_customers_not_pay_tax_l1695_169537


namespace negation_of_P_l1695_169541

-- Defining the original proposition
def P : Prop := ∃ x₀ : ℝ, x₀^2 = 1

-- The problem is to prove the negation of the proposition
theorem negation_of_P : (¬P) ↔ (∀ x : ℝ, x^2 ≠ 1) :=
  by sorry

end negation_of_P_l1695_169541


namespace tan_identity_at_30_degrees_l1695_169526

theorem tan_identity_at_30_degrees :
  let A := 30
  let B := 30
  let deg_to_rad := pi / 180
  let tan := fun x : ℝ => Real.tan (x * deg_to_rad)
  (1 + tan A) * (1 + tan B) = (4 + 2 * Real.sqrt 3) / 3 := by
  sorry

end tan_identity_at_30_degrees_l1695_169526


namespace workers_work_5_days_a_week_l1695_169530

def total_weekly_toys : ℕ := 5500
def daily_toys : ℕ := 1100
def days_worked : ℕ := total_weekly_toys / daily_toys

theorem workers_work_5_days_a_week : days_worked = 5 := 
by 
  sorry

end workers_work_5_days_a_week_l1695_169530


namespace average_upstream_speed_l1695_169514

/--
There are three boats moving down a river. Boat A moves downstream at a speed of 1 km in 4 minutes 
and upstream at a speed of 1 km in 8 minutes. Boat B moves downstream at a speed of 1 km in 
5 minutes and upstream at a speed of 1 km in 11 minutes. Boat C moves downstream at a speed of 
1 km in 6 minutes and upstream at a speed of 1 km in 10 minutes. Prove that the average speed 
of the boats against the current is 6.32 km/h.
-/
theorem average_upstream_speed :
  let speed_A_upstream := 1 / (8 / 60 : ℝ)
  let speed_B_upstream := 1 / (11 / 60 : ℝ)
  let speed_C_upstream := 1 / (10 / 60 : ℝ)
  let average_speed := (speed_A_upstream + speed_B_upstream + speed_C_upstream) / 3
  average_speed = 6.32 :=
by
  let speed_A_upstream := 1 / (8 / 60 : ℝ)
  let speed_B_upstream := 1 / (11 / 60 : ℝ)
  let speed_C_upstream := 1 / (10 / 60 : ℝ)
  let average_speed := (speed_A_upstream + speed_B_upstream + speed_C_upstream) / 3
  sorry

end average_upstream_speed_l1695_169514


namespace arithmetic_mean_of_4_and_16_l1695_169566

-- Define the arithmetic mean condition
def is_arithmetic_mean (a b x : ℝ) : Prop :=
  x = (a + b) / 2

-- Theorem to prove that x = 10 if it is the mean of 4 and 16
theorem arithmetic_mean_of_4_and_16 (x : ℝ) (h : is_arithmetic_mean 4 16 x) : x = 10 :=
by
  sorry

end arithmetic_mean_of_4_and_16_l1695_169566


namespace best_representation_is_B_l1695_169507

-- Define the conditions
structure Trip :=
  (home_to_diner : ℝ)
  (diner_stop : ℝ)
  (diner_to_highway : ℝ)
  (highway_to_mall : ℝ)
  (mall_stop : ℝ)
  (highway_return : ℝ)
  (construction_zone : ℝ)
  (return_city_traffic : ℝ)

-- Graph description
inductive Graph
| plateau : Graph
| increasing : Graph → Graph
| decreasing : Graph → Graph

-- Condition that describes the pattern of the graph
def correct_graph (trip : Trip) : Prop :=
  let d1 := trip.home_to_diner
  let d2 := trip.diner_stop
  let d3 := trip.diner_to_highway
  let d4 := trip.highway_to_mall
  let d5 := trip.mall_stop
  let d6 := trip.highway_return
  let d7 := trip.construction_zone
  let d8 := trip.return_city_traffic
  d1 > 0 ∧ d2 = 0 ∧ d3 > 0 ∧ d4 > 0 ∧ d5 = 0 ∧ d6 < 0 ∧ d7 < 0 ∧ d8 < 0

-- Theorem statement
theorem best_representation_is_B (trip : Trip) : correct_graph trip :=
by sorry

end best_representation_is_B_l1695_169507


namespace triangle_side_ratio_impossible_triangle_side_ratio_impossible_2_triangle_side_ratio_impossible_3_l1695_169585

theorem triangle_side_ratio_impossible (a b c : ℝ) (h₁ : a = b / 2) (h₂ : a = c / 3) : false :=
by
  sorry

theorem triangle_side_ratio_impossible_2 (a b c : ℝ) (h₁ : b = a / 2) (h₂ : b = c / 3) : false :=
by
  sorry

theorem triangle_side_ratio_impossible_3 (a b c : ℝ) (h₁ : c = a / 2) (h₂ : c = b / 3) : false :=
by
  sorry

end triangle_side_ratio_impossible_triangle_side_ratio_impossible_2_triangle_side_ratio_impossible_3_l1695_169585


namespace scientific_notation_of_coronavirus_diameter_l1695_169501

theorem scientific_notation_of_coronavirus_diameter:
  0.000000907 = 9.07 * 10^(-7) :=
sorry

end scientific_notation_of_coronavirus_diameter_l1695_169501


namespace length_of_BA_is_sqrt_557_l1695_169553

-- Define the given conditions
def AD : ℝ := 6
def DC : ℝ := 11
def CB : ℝ := 6
def AC : ℝ := 14

-- Define the theorem statement
theorem length_of_BA_is_sqrt_557 (x : ℝ) (H1 : AD = 6) (H2 : DC = 11) (H3 : CB = 6) (H4 : AC = 14) :
  x = Real.sqrt 557 :=
  sorry

end length_of_BA_is_sqrt_557_l1695_169553


namespace polynomial_value_at_2_l1695_169556

def f (x : ℤ) : ℤ := 7 * x^7 + 6 * x^6 + 5 * x^5 + 4 * x^4 + 3 * x^3 + 2 * x^2 + x

theorem polynomial_value_at_2:
  f 2 = 1538 := by
  sorry

end polynomial_value_at_2_l1695_169556


namespace add_fractions_result_l1695_169505

noncomputable def add_fractions (a : ℝ) (h : a ≠ 0): ℝ := (3 / a) + (2 / a)

theorem add_fractions_result (a : ℝ) (h : a ≠ 0) : add_fractions a h = 5 / a := by
  sorry

end add_fractions_result_l1695_169505


namespace cost_difference_l1695_169502

-- Define the costs
def cost_chocolate : ℕ := 3
def cost_candy_bar : ℕ := 7

-- Define the difference to be proved
theorem cost_difference :
  cost_candy_bar - cost_chocolate = 4 :=
by
  -- trivial proof steps
  sorry

end cost_difference_l1695_169502


namespace quadratic_roots_sum_product_l1695_169515

theorem quadratic_roots_sum_product {p q : ℝ} 
  (h1 : p / 3 = 10) 
  (h2 : q / 3 = 15) : 
  p + q = 75 := sorry

end quadratic_roots_sum_product_l1695_169515


namespace intersection_M_N_l1695_169559

def M : Set ℕ := { y | y < 6 }
def N : Set ℕ := {2, 3, 6}

theorem intersection_M_N : M ∩ N = {2, 3} := by
  sorry

end intersection_M_N_l1695_169559


namespace expression_value_l1695_169564

theorem expression_value (x y : ℝ) (h : x - 2 * y = 3) : 1 - 2 * x + 4 * y = -5 :=
by
  sorry

end expression_value_l1695_169564


namespace solve_for_x_minus_y_l1695_169518

theorem solve_for_x_minus_y (x y : ℝ) (h1 : x + y = 6) (h2 : x^2 - y^2 = 24) : x - y = 4 := 
by
  sorry

end solve_for_x_minus_y_l1695_169518


namespace expression1_expression2_expression3_expression4_l1695_169572

theorem expression1 : 12 - (-10) + 7 = 29 := 
by
  sorry

theorem expression2 : 1 + (-2) * abs (-2 - 3) - 5 = -14 :=
by
  sorry

theorem expression3 : (-8 * (-1 / 6 + 3 / 4 - 1 / 12)) / (1 / 6) = -24 :=
by
  sorry

theorem expression4 : -1 ^ 2 - (2 - (-2) ^ 3) / (-2 / 5) * (5 / 2) = 123 / 2 := 
by
  sorry

end expression1_expression2_expression3_expression4_l1695_169572


namespace ratio_of_shoppers_l1695_169560

theorem ratio_of_shoppers (boxes ordered_of_yams: ℕ) (packages_per_box shoppers total_shoppers: ℕ)
  (h1 : packages_per_box = 25)
  (h2 : ordered_of_yams = 5)
  (h3 : total_shoppers = 375)
  (h4 : shoppers = ordered_of_yams * packages_per_box):
  (shoppers : ℕ) / total_shoppers = 1 / 3 := 
sorry

end ratio_of_shoppers_l1695_169560


namespace polynomial_division_remainder_zero_l1695_169504

theorem polynomial_division_remainder_zero (x : ℂ) (hx : x^5 + x^4 + x^3 + x^2 + x + 1 = 0)
  : (x^55 + x^44 + x^33 + x^22 + x^11 + 1) % (x^5 + x^4 + x^3 + x^2 + x + 1) = 0 := by
  sorry

end polynomial_division_remainder_zero_l1695_169504


namespace line_equation_l1695_169584

theorem line_equation (b r S : ℝ) (h : ℝ) (m : ℝ) (eq_one : S = 1/2 * b * h) (eq_two : h = 2*S / b) (eq_three : |m| = r / b) 
  (eq_four : m = r / b) : 
  (∀ x y : ℝ, y = m * (x - b) → b > 0 → r > 0 → S > 0 → rx - bry - rb = 0) := 
sorry

end line_equation_l1695_169584
