import Mathlib

namespace NUMINAMATH_GPT_percentage_of_profits_l2028_202821

variable (R P : ℝ) -- Let R be the revenues and P be the profits in the previous year
variable (H1 : (P/R) * 100 = 10) -- The condition we want to prove
variable (H2 : 0.95 * R) -- Revenues in 2009 are 0.95R
variable (H3 : 0.1 * 0.95 * R) -- Profits in 2009 are 0.1 * 0.95R = 0.095R
variable (H4 : 0.095 * R = 0.95 * P) -- The given relation between profits in 2009 and previous year

theorem percentage_of_profits (H1 : (P/R) * 100 = 10) 
  (H2 : ∀ (R : ℝ),  ∃ ρ, ρ = 0.95 * R)
  (H3 : ∀ (R : ℝ),  ∃ π, π = 0.10 * (0.95 * R))
  (H4 : ∀ (R P : ℝ), 0.095 * R = 0.95 * P) :
  ∀ (P R : ℝ), (P/R) * 100 = 10 := 
by
  sorry

end NUMINAMATH_GPT_percentage_of_profits_l2028_202821


namespace NUMINAMATH_GPT_probability_top_red_second_black_l2028_202892

def num_red_cards : ℕ := 39
def num_black_cards : ℕ := 39
def total_cards : ℕ := 78

theorem probability_top_red_second_black :
  (num_red_cards * num_black_cards) / (total_cards * (total_cards - 1)) = 507 / 2002 := 
sorry

end NUMINAMATH_GPT_probability_top_red_second_black_l2028_202892


namespace NUMINAMATH_GPT_find_k_l2028_202891

-- Define the vectors
def e1 : ℝ × ℝ := (1, 0)
def e2 : ℝ × ℝ := (0, 1)

def a : ℝ × ℝ := (e1.1 - 2 * e2.1, e1.2 - 2 * e2.2)
def b (k : ℝ) : ℝ × ℝ := (k * e1.1 + e2.1, k * e1.2 + e2.2)

-- Define the parallel condition
def parallel (u v : ℝ × ℝ) : Prop :=
  u.1 * v.2 = u.2 * v.1

-- Statement of the problem translated to a Lean theorem
theorem find_k (k : ℝ) : 
  parallel a (b k) -> k = -1 / 2 := by
  sorry

end NUMINAMATH_GPT_find_k_l2028_202891


namespace NUMINAMATH_GPT_find_roots_of_polynomial_l2028_202893

theorem find_roots_of_polynomial :
  ∀ x : ℝ, (3 * x ^ 4 - x ^ 3 - 8 * x ^ 2 - x + 3 = 0) →
    (x = 2 ∨ x = 1/3 ∨ x = -1) :=
by
  intros x h
  sorry

end NUMINAMATH_GPT_find_roots_of_polynomial_l2028_202893


namespace NUMINAMATH_GPT_cakes_sold_l2028_202887

theorem cakes_sold (total_made : ℕ) (cakes_left : ℕ) (cakes_sold : ℕ) :
  total_made = 217 ∧ cakes_left = 72 → cakes_sold = 145 :=
by
  -- Assuming total_made is 217 and cakes_left is 72, we need to show cakes_sold = 145
  sorry

end NUMINAMATH_GPT_cakes_sold_l2028_202887


namespace NUMINAMATH_GPT_meaningful_fraction_l2028_202858

theorem meaningful_fraction (x : ℝ) : (∃ y, y = (1 / (x - 2))) ↔ x ≠ 2 :=
by
  sorry

end NUMINAMATH_GPT_meaningful_fraction_l2028_202858


namespace NUMINAMATH_GPT_time_for_b_and_d_together_l2028_202844

theorem time_for_b_and_d_together :
  let A_rate := 1 / 3
  let D_rate := 1 / 4
  (∃ B_rate C_rate : ℚ,
    B_rate + C_rate = 1 / 3 ∧
    A_rate + C_rate = 1 / 2 ∧
    1 / (B_rate + D_rate) = 2.4) :=
  
by
  let A_rate := 1 / 3
  let D_rate := 1 / 4
  use 1 / 6, 1 / 6
  sorry

end NUMINAMATH_GPT_time_for_b_and_d_together_l2028_202844


namespace NUMINAMATH_GPT_camille_total_birds_l2028_202825

theorem camille_total_birds :
  let cardinals := 3
  let robins := 4 * cardinals
  let blue_jays := 2 * cardinals
  let sparrows := 3 * cardinals + 1
  let pigeons := 3 * blue_jays
  let finches := robins / 2
  cardinals + robins + blue_jays + sparrows + pigeons + finches = 55 :=
by
  let cardinals := 3
  let robins := 4 * cardinals
  let blue_jays := 2 * cardinals
  let sparrows := 3 * cardinals + 1
  let pigeons := 3 * blue_jays
  let finches := robins / 2
  show cardinals + robins + blue_jays + sparrows + pigeons + finches = 55
  sorry

end NUMINAMATH_GPT_camille_total_birds_l2028_202825


namespace NUMINAMATH_GPT_election_votes_l2028_202898

theorem election_votes (T : ℝ) (Vf Va Vn : ℝ)
  (h1 : Va = 0.375 * T)
  (h2 : Vn = 0.125 * T)
  (h3 : Vf = Va + 78)
  (h4 : T = Vf + Va + Vn) :
  T = 624 :=
by
  sorry

end NUMINAMATH_GPT_election_votes_l2028_202898


namespace NUMINAMATH_GPT_selling_price_to_achieve_profit_l2028_202836

theorem selling_price_to_achieve_profit :
  ∃ (x : ℝ), let original_price := 210
              let purchase_price := 190
              let avg_sales_initial := 8
              let profit_goal := 280
              (210 - x = 200) ∧
              let profit_per_item := original_price - purchase_price - x
              let avg_sales_quantity := avg_sales_initial + 2 * x
              profit_per_item * avg_sales_quantity = profit_goal := by
  sorry

end NUMINAMATH_GPT_selling_price_to_achieve_profit_l2028_202836


namespace NUMINAMATH_GPT_solve_for_x_l2028_202856

variable {x y : ℝ}

theorem solve_for_x (h1 : y = (x^2 - 9) / (x - 3)) (h2 : y = 3 * x) : x = 3 / 2 := by
  sorry

end NUMINAMATH_GPT_solve_for_x_l2028_202856


namespace NUMINAMATH_GPT_remainder_when_3m_div_by_5_l2028_202857

variable (m k : ℤ)

theorem remainder_when_3m_div_by_5 (h : m % 5 = 2) : (3 * m) % 5 = 1 := by
  sorry

end NUMINAMATH_GPT_remainder_when_3m_div_by_5_l2028_202857


namespace NUMINAMATH_GPT_rhombus_diagonal_length_l2028_202888

theorem rhombus_diagonal_length (d1 d2 : ℝ) (A : ℝ) (h1 : d1 = 25) (h2 : A = 250) (h3 : A = (d1 * d2) / 2) : d2 = 20 := 
by
  rw [h1, h2] at h3
  sorry

end NUMINAMATH_GPT_rhombus_diagonal_length_l2028_202888


namespace NUMINAMATH_GPT_evaluate_expression_l2028_202899

theorem evaluate_expression (a b : ℕ) (h_a : a = 15) (h_b : b = 7) :
  (a + b) ^ 2 - (a ^ 2 + b ^ 2) = 210 :=
by 
  rw [h_a, h_b]
  sorry

end NUMINAMATH_GPT_evaluate_expression_l2028_202899


namespace NUMINAMATH_GPT_find_age_of_mother_l2028_202819

def Grace_age := 60
def ratio_GM_Grace := 3 / 8
def ratio_GM_Mother := 2

theorem find_age_of_mother (G M GM : ℕ) (h1 : G = ratio_GM_Grace * GM) 
                           (h2 : GM = ratio_GM_Mother * M) (h3 : G = Grace_age) : 
  M = 80 :=
by
  sorry

end NUMINAMATH_GPT_find_age_of_mother_l2028_202819


namespace NUMINAMATH_GPT_weight_difference_l2028_202835

open Real

def yellow_weight : ℝ := 0.6
def green_weight : ℝ := 0.4
def red_weight : ℝ := 0.8
def blue_weight : ℝ := 0.5

def weights : List ℝ := [yellow_weight, green_weight, red_weight, blue_weight]

theorem weight_difference : (List.maximum weights).getD 0 - (List.minimum weights).getD 0 = 0.4 :=
by
  sorry

end NUMINAMATH_GPT_weight_difference_l2028_202835


namespace NUMINAMATH_GPT_range_of_m_l2028_202828

noncomputable def f (x m : ℝ) : ℝ :=
if x < 0 then (x - m) ^ 2 - 2 else 2 * x ^ 3 - 3 * x ^ 2

theorem range_of_m (m : ℝ) : (∃ x : ℝ, f x m = -1) ↔ m ≥ 1 :=
by
  sorry

end NUMINAMATH_GPT_range_of_m_l2028_202828


namespace NUMINAMATH_GPT_total_whales_correct_l2028_202829

def first_trip_male_whales : ℕ := 28
def first_trip_female_whales : ℕ := 2 * first_trip_male_whales
def first_trip_total_whales : ℕ := first_trip_male_whales + first_trip_female_whales

def second_trip_baby_whales : ℕ := 8
def second_trip_parent_whales : ℕ := 2 * second_trip_baby_whales
def second_trip_total_whales : ℕ := second_trip_baby_whales + second_trip_parent_whales

def third_trip_male_whales : ℕ := first_trip_male_whales / 2
def third_trip_female_whales : ℕ := first_trip_female_whales
def third_trip_total_whales : ℕ := third_trip_male_whales + third_trip_female_whales

def total_whales_seen : ℕ :=
  first_trip_total_whales + second_trip_total_whales + third_trip_total_whales

theorem total_whales_correct : total_whales_seen = 178 := by
  sorry

end NUMINAMATH_GPT_total_whales_correct_l2028_202829


namespace NUMINAMATH_GPT_int_power_sum_is_integer_l2028_202813

def is_integer (x : ℝ) : Prop := ∃ n : ℤ, x = n

theorem int_power_sum_is_integer {x : ℝ} (h : is_integer (x + 1/x)) (n : ℤ) : is_integer (x^n + 1/x^n) :=
by
  sorry

end NUMINAMATH_GPT_int_power_sum_is_integer_l2028_202813


namespace NUMINAMATH_GPT_solution_set_l2028_202872

def f (x : ℝ) : ℝ := sorry

axiom ax1 : ∀ a b : ℝ, f (a + b) = f a + f b - 1
axiom ax2 : ∀ x : ℝ, x > 0 → f x > 1
axiom ax3 : f 4 = 5

theorem solution_set (x : ℝ) : f (3 * x^2 - x - 2) < 3 ↔ (-1 < x ∧ x < 4 / 3) :=
by
  sorry

end NUMINAMATH_GPT_solution_set_l2028_202872


namespace NUMINAMATH_GPT_largest_side_l2028_202841

-- Definitions of conditions from part (a)
def perimeter_eq (l w : ℝ) : Prop := 2 * l + 2 * w = 240
def area_eq (l w : ℝ) : Prop := l * w = 2880

-- The main proof statement
theorem largest_side (l w : ℝ) (h1 : perimeter_eq l w) (h2 : area_eq l w) : l = 72 ∨ w = 72 :=
by
  sorry

end NUMINAMATH_GPT_largest_side_l2028_202841


namespace NUMINAMATH_GPT_percentage_of_students_owning_only_cats_is_10_percent_l2028_202890

def total_students : ℕ := 500
def cat_owners : ℕ := 75
def dog_owners : ℕ := 150
def both_cat_and_dog_owners : ℕ := 25
def only_cat_owners : ℕ := cat_owners - both_cat_and_dog_owners
def percent_owning_only_cats : ℚ := (only_cat_owners * 100) / total_students

theorem percentage_of_students_owning_only_cats_is_10_percent : percent_owning_only_cats = 10 := by
  sorry

end NUMINAMATH_GPT_percentage_of_students_owning_only_cats_is_10_percent_l2028_202890


namespace NUMINAMATH_GPT_geom_seq_proof_l2028_202805

noncomputable def geom_seq (a q : ℝ) (n : ℕ) : ℝ :=
  a * q^(n - 1)

variables {a q : ℝ}

theorem geom_seq_proof (h1 : geom_seq a q 7 = 4) (h2 : geom_seq a q 5 + geom_seq a q 9 = 10) :
  geom_seq a q 3 + geom_seq a q 11 = 17 :=
by
  sorry

end NUMINAMATH_GPT_geom_seq_proof_l2028_202805


namespace NUMINAMATH_GPT_arithmetic_sequence_third_term_l2028_202820

theorem arithmetic_sequence_third_term (a d : ℤ) (h : a + (a + 4 * d) = 14) : a + 2 * d = 7 := by
  -- We assume the sum of the first and fifth term is 14 and prove that the third term is 7.
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_third_term_l2028_202820


namespace NUMINAMATH_GPT_part1_part2_part3_l2028_202824

section Part1
variables {a b : ℝ}

theorem part1 (h1 : a + b = 3) (h2 : a * b = 2) : a^2 + b^2 = 5 := 
sorry
end Part1

section Part2
variables {a b c : ℝ}

theorem part2 (h1 : a + b + c = 6) (h2 : a * b + b * c + c * a = 11) : a^2 + b^2 + c^2 = 14 := 
sorry
end Part2

section Part3
variables {a b c : ℝ}

theorem part3 (h1 : a + b + c = 0) (h2 : a^2 + b^2 + c^2 = 6) : a^4 + b^4 + c^4 = 18 :=
sorry
end Part3

end NUMINAMATH_GPT_part1_part2_part3_l2028_202824


namespace NUMINAMATH_GPT_number_of_numbers_l2028_202860

theorem number_of_numbers (n : ℕ) (S : ℕ) 
  (h1 : (S + 26) / n = 16) 
  (h2 : (S + 46) / n = 18) : 
  n = 10 := 
by 
  -- placeholder for the proof
  sorry

end NUMINAMATH_GPT_number_of_numbers_l2028_202860


namespace NUMINAMATH_GPT_grid_permutation_exists_l2028_202806

theorem grid_permutation_exists (n : ℕ) (grid : Fin n → Fin n → ℤ) 
  (cond1 : ∀ i : Fin n, ∃ unique j : Fin n, grid i j = 1)
  (cond2 : ∀ i : Fin n, ∃ unique j : Fin n, grid i j = -1)
  (cond3 : ∀ j : Fin n, ∃ unique i : Fin n, grid i j = 1)
  (cond4 : ∀ j : Fin n, ∃ unique i : Fin n, grid i j = -1)
  (cond5 : ∀ i j, grid i j = 0 ∨ grid i j = 1 ∨ grid i j = -1) :
  ∃ (perm_rows perm_cols : Fin n → Fin n),
    (∀ i j, grid (perm_rows i) (perm_cols j) = -grid i j) :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_grid_permutation_exists_l2028_202806


namespace NUMINAMATH_GPT_total_students_l2028_202845

-- Define the conditions
def chocolates_distributed (y z : ℕ) : ℕ :=
  y * y + z * z

-- Define the main theorem to be proved
theorem total_students (y z : ℕ) (h : z = y + 3) (chocolates_left: ℕ) (initial_chocolates: ℕ)
  (h_chocolates: chocolates_distributed y z = initial_chocolates - chocolates_left) : 
  y + z = 33 :=
by
  sorry

end NUMINAMATH_GPT_total_students_l2028_202845


namespace NUMINAMATH_GPT_incircle_intersections_equation_l2028_202878

-- Assume a triangle ABC with the given configuration
variables {A B C D E F M N : Type}

-- Incircle touches sides CA, AB at points E, F respectively
-- Lines BE and CF intersect the incircle again at points M and N respectively

theorem incircle_intersections_equation
  (triangle_ABC : Type)
  (incircle_I : Type)
  (touch_CA : Type)
  (touch_AB : Type)
  (intersect_BE : Type)
  (intersect_CF : Type)
  (E F : triangle_ABC → incircle_I)
  (M N : intersect_BE → intersect_CF)
  : 
  MN * EF = 3 * MF * NE :=
by 
  -- Sorry as the proof is omitted
  sorry

end NUMINAMATH_GPT_incircle_intersections_equation_l2028_202878


namespace NUMINAMATH_GPT_solve_quadratics_l2028_202866

theorem solve_quadratics :
  (∃ x : ℝ, x^2 + 5 * x - 24 = 0) ∧ (∃ y, y^2 + 5 * y - 24 = 0) ∧
  (∃ z : ℝ, 3 * z^2 + 2 * z - 4 = 0) ∧ (∃ w, 3 * w^2 + 2 * w - 4 = 0) :=
by {
  sorry
}

end NUMINAMATH_GPT_solve_quadratics_l2028_202866


namespace NUMINAMATH_GPT_speed_of_man_l2028_202861

-- Define all given conditions and constants

def trainLength : ℝ := 110 -- in meters
def trainSpeed : ℝ := 40 -- in km/hr
def timeToPass : ℝ := 8.799296056315494 -- in seconds

-- We want to prove that the speed of the man is approximately 4.9968 km/hr
theorem speed_of_man :
  let trainSpeedMS := trainSpeed * (1000 / 3600)
  let relativeSpeed := trainLength / timeToPass
  let manSpeedMS := relativeSpeed - trainSpeedMS
  let manSpeedKMH := manSpeedMS * (3600 / 1000)
  abs (manSpeedKMH - 4.9968) < 0.01 := sorry

end NUMINAMATH_GPT_speed_of_man_l2028_202861


namespace NUMINAMATH_GPT_lesser_fraction_l2028_202884

theorem lesser_fraction (x y : ℚ) (h1 : x + y = 10 / 11) (h2 : x * y = 1 / 8) : min x y = (80 - 2 * Real.sqrt 632) / 176 := 
by sorry

end NUMINAMATH_GPT_lesser_fraction_l2028_202884


namespace NUMINAMATH_GPT_survey_pop_and_coke_l2028_202881

theorem survey_pop_and_coke (total_people : ℕ) (angle_pop angle_coke : ℕ) 
  (h_total : total_people = 500) (h_angle_pop : angle_pop = 240) (h_angle_coke : angle_coke = 90) :
  ∃ (pop_people coke_people : ℕ), pop_people = 333 ∧ coke_people = 125 :=
by 
  sorry

end NUMINAMATH_GPT_survey_pop_and_coke_l2028_202881


namespace NUMINAMATH_GPT_sqrt_meaningful_l2028_202800

theorem sqrt_meaningful (x : ℝ) : x + 1 >= 0 ↔ (∃ y : ℝ, y * y = x + 1) := by
  sorry

end NUMINAMATH_GPT_sqrt_meaningful_l2028_202800


namespace NUMINAMATH_GPT_prove_real_roots_and_find_m_l2028_202823

-- Condition: The quadratic equation
def quadratic_eq (m x : ℝ) : Prop := x^2 - (m-1)*x + m-2 = 0

-- Condition: Discriminant
def discriminant (m : ℝ) : ℝ := (m-3)^2

-- Define the problem as a proposition
theorem prove_real_roots_and_find_m (m : ℝ) :
  (discriminant m ≥ 0) ∧ 
  (|3 - m| = 3 → (m = 0 ∨ m = 6)) :=
by
  sorry

end NUMINAMATH_GPT_prove_real_roots_and_find_m_l2028_202823


namespace NUMINAMATH_GPT_each_student_gets_8_pieces_l2028_202886

-- Define the number of pieces of candy
def candy : Nat := 344

-- Define the number of students
def students : Nat := 43

-- Define the number of pieces each student gets, which we need to prove
def pieces_per_student : Nat := candy / students

-- The proof problem statement
theorem each_student_gets_8_pieces : pieces_per_student = 8 :=
by
  -- This proof content is omitted as per instructions
  sorry

end NUMINAMATH_GPT_each_student_gets_8_pieces_l2028_202886


namespace NUMINAMATH_GPT_passengers_landed_in_virginia_l2028_202809

theorem passengers_landed_in_virginia
  (P_start : ℕ) (D_Texas : ℕ) (C_Texas : ℕ) (D_NC : ℕ) (C_NC : ℕ) (C : ℕ)
  (hP_start : P_start = 124)
  (hD_Texas : D_Texas = 58)
  (hC_Texas : C_Texas = 24)
  (hD_NC : D_NC = 47)
  (hC_NC : C_NC = 14)
  (hC : C = 10) :
  P_start - D_Texas + C_Texas - D_NC + C_NC + C = 67 := by
  sorry

end NUMINAMATH_GPT_passengers_landed_in_virginia_l2028_202809


namespace NUMINAMATH_GPT_radius_of_circle_of_roots_l2028_202834

theorem radius_of_circle_of_roots (z : ℂ)
  (h : (z + 2)^6 = 64 * z^6) :
  ∃ r : ℝ, r = 4 / 3 ∧ ∀ z, (z + 2)^6 = 64 * z^6 →
  abs (z + 2) = (4 / 3 : ℝ) * abs z :=
by
  sorry

end NUMINAMATH_GPT_radius_of_circle_of_roots_l2028_202834


namespace NUMINAMATH_GPT_determine_g_l2028_202843

variable (g : ℕ → ℕ)

theorem determine_g (h : ∀ x, g (x + 1) = 2 * x + 3) : ∀ x, g x = 2 * x + 1 :=
by
  sorry

end NUMINAMATH_GPT_determine_g_l2028_202843


namespace NUMINAMATH_GPT_binom_expansion_l2028_202840

/-- Given the binomial expansion of (sqrt(x) + 3x)^n for n < 15, 
    with the binomial coefficients of the 9th, 10th, and 11th terms forming an arithmetic sequence,
    we conclude that n must be 14 and describe all the rational terms in the expansion.
-/
theorem binom_expansion (n : ℕ) (h : n < 15)
  (h_seq : Nat.choose n 8 + Nat.choose n 10 = 2 * Nat.choose n 9) :
  n = 14 ∧
  (∃ (t1 t2 t3 : ℕ), 
    (t1 = 1 ∧ (Nat.choose 14 0 : ℕ) * (x ^ 7 : ℤ) = x ^ 7) ∧
    (t2 = 164 ∧ (Nat.choose 14 6 : ℕ) * (x ^ 6 : ℤ) = 164 * x ^ 6) ∧
    (t3 = 91 ∧ (Nat.choose 14 12 : ℕ) * (x ^ 5 : ℤ) = 91 * x ^ 5)) := 
  sorry

end NUMINAMATH_GPT_binom_expansion_l2028_202840


namespace NUMINAMATH_GPT_distance_origin_to_point_l2028_202833

noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)

theorem distance_origin_to_point :
  distance (0, 0) (-15, 8) = 17 :=
by 
  sorry

end NUMINAMATH_GPT_distance_origin_to_point_l2028_202833


namespace NUMINAMATH_GPT_geometric_sequence_ratio_l2028_202802

variable (a : ℕ → ℝ)
variable (q : ℝ)
variable (n : ℕ)

-- Definitions based on given conditions
def is_geometric_seq (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n * q

-- Main statement
theorem geometric_sequence_ratio :
  is_geometric_seq a q →
  q = -1/3 →
  (a 1 + a 3 + a 5 + a 7) / (a 2 + a 4 + a 6 + a 8) = -3 :=
by
  intros
  sorry

end NUMINAMATH_GPT_geometric_sequence_ratio_l2028_202802


namespace NUMINAMATH_GPT_valid_addends_l2028_202867

noncomputable def is_valid_addend (n : ℕ) : Prop :=
  ∃ (X Y : ℕ), (100 * 9 + 10 * X + 4) = n ∧ (30 + Y) ∈ [36, 30, 20, 10]

theorem valid_addends :
  ∀ (n : ℕ),
  is_valid_addend n ↔ (n = 964 ∨ n = 974 ∨ n = 984 ∨ n = 994) :=
by
  sorry

end NUMINAMATH_GPT_valid_addends_l2028_202867


namespace NUMINAMATH_GPT_simplified_result_l2028_202871

theorem simplified_result (a b M : ℝ) (h1 : (2 * a) / (a ^ 2 - b ^ 2) - 1 / M = 1 / (a - b))
  (h2 : M - (a - b) = 2 * b) : (2 * a) / (a ^ 2 - b ^ 2) - 1 / (a - b) = 1 / (a + b) :=
by
  sorry

end NUMINAMATH_GPT_simplified_result_l2028_202871


namespace NUMINAMATH_GPT_intersection_of_A_and_B_l2028_202846

def I := {x : ℝ | true}
def A := {x : ℝ | x * (x - 1) ≥ 0}
def B := {x : ℝ | x > 1}
def C := {x : ℝ | x > 1}

theorem intersection_of_A_and_B : A ∩ B = C := by
  sorry

end NUMINAMATH_GPT_intersection_of_A_and_B_l2028_202846


namespace NUMINAMATH_GPT_average_speed_is_70_l2028_202879

noncomputable def average_speed (d₁ d₂ d₃ d₄ t₁ t₂ t₃ t₄ : ℝ) : ℝ :=
  (d₁ + d₂ + d₃ + d₄) / (t₁ + t₂ + t₃ + t₄)

theorem average_speed_is_70 :
  let d₁ := 30
  let s₁ := 60
  let t₁ := d₁ / s₁
  let d₂ := 35
  let s₂ := 70
  let t₂ := d₂ / s₂
  let d₃ := 80
  let t₃ := 1
  let s₃ := d₃ / t₃
  let s₄ := 55
  let t₄ := 20/60.0
  let d₄ := s₄ * t₄
  average_speed d₁ d₂ d₃ d₄ t₁ t₂ t₃ t₄ = 70 :=
by
  sorry

end NUMINAMATH_GPT_average_speed_is_70_l2028_202879


namespace NUMINAMATH_GPT_find_n_tan_eq_l2028_202854

theorem find_n_tan_eq (n : ℤ) (h1 : -90 < n ∧ n < 90) (h2 : ∀ k : ℤ, 225 - 180 * k = 45) : n = 45 := by
  sorry

end NUMINAMATH_GPT_find_n_tan_eq_l2028_202854


namespace NUMINAMATH_GPT_hydrogen_to_oxygen_ratio_l2028_202873

theorem hydrogen_to_oxygen_ratio (total_mass_water mass_hydrogen mass_oxygen : ℝ) 
(h1 : total_mass_water = 117)
(h2 : mass_hydrogen = 13)
(h3 : mass_oxygen = total_mass_water - mass_hydrogen) :
(mass_hydrogen / mass_oxygen) = 1 / 8 := 
sorry

end NUMINAMATH_GPT_hydrogen_to_oxygen_ratio_l2028_202873


namespace NUMINAMATH_GPT_cheaper_module_cost_l2028_202869

theorem cheaper_module_cost (x : ℝ) :
  (21 * x + 10 = 62.50) → (x = 2.50) :=
by
  intro h
  sorry

end NUMINAMATH_GPT_cheaper_module_cost_l2028_202869


namespace NUMINAMATH_GPT_amy_total_tickets_l2028_202849

theorem amy_total_tickets (initial_tickets additional_tickets : ℕ) (h_initial : initial_tickets = 33) (h_additional : additional_tickets = 21) : 
  initial_tickets + additional_tickets = 54 := 
by 
  sorry

end NUMINAMATH_GPT_amy_total_tickets_l2028_202849


namespace NUMINAMATH_GPT_smallest_possible_time_for_travel_l2028_202874

theorem smallest_possible_time_for_travel :
  ∃ t : ℝ, (∀ D M P : ℝ, D = 6 → M = 6 → P = 6 → 
    ∀ motorcycle_speed distance : ℝ, motorcycle_speed = 90 → distance = 135 → 
    t < 3.9) :=
  sorry

end NUMINAMATH_GPT_smallest_possible_time_for_travel_l2028_202874


namespace NUMINAMATH_GPT_num_valid_arrangements_without_A_at_start_and_B_at_end_l2028_202864

-- Define a predicate for person A being at the beginning
def A_at_beginning (arrangement : List ℕ) : Prop :=
  arrangement.head! = 1

-- Define a predicate for person B being at the end
def B_at_end (arrangement : List ℕ) : Prop :=
  arrangement.getLast! = 2

-- Main theorem stating the number of valid arrangements
theorem num_valid_arrangements_without_A_at_start_and_B_at_end : ∃ (count : ℕ), count = 78 :=
by
  have total_arrangements := Nat.factorial 5
  have A_at_start_arrangements := Nat.factorial 4
  have B_at_end_arrangements := Nat.factorial 4
  have both_A_and_B_arrangements := Nat.factorial 3
  let valid_arrangements := total_arrangements - 2 * A_at_start_arrangements + both_A_and_B_arrangements
  use valid_arrangements
  sorry

end NUMINAMATH_GPT_num_valid_arrangements_without_A_at_start_and_B_at_end_l2028_202864


namespace NUMINAMATH_GPT_box_count_neither_markers_nor_erasers_l2028_202801

-- Define the conditions as parameters.
def total_boxes : ℕ := 15
def markers_count : ℕ := 10
def erasers_count : ℕ := 5
def both_count : ℕ := 4

-- State the theorem to be proven in Lean 4.
theorem box_count_neither_markers_nor_erasers : 
  total_boxes - (markers_count + erasers_count - both_count) = 4 := 
sorry

end NUMINAMATH_GPT_box_count_neither_markers_nor_erasers_l2028_202801


namespace NUMINAMATH_GPT_starting_lineups_possible_l2028_202852

open Nat

theorem starting_lineups_possible (total_players : ℕ) (all_stars : ℕ) (lineup_size : ℕ) 
  (fixed_in_lineup : ℕ) (choose_size : ℕ) 
  (h_fixed : fixed_in_lineup = all_stars)
  (h_remaining : total_players - fixed_in_lineup = choose_size)
  (h_lineup : lineup_size = all_stars + choose_size) :
  (Nat.choose choose_size 3 = 220) :=
by
  sorry

end NUMINAMATH_GPT_starting_lineups_possible_l2028_202852


namespace NUMINAMATH_GPT_hexagon_diagonals_sum_correct_l2028_202830

noncomputable def hexagon_diagonals_sum : ℝ :=
  let AB := 40
  let S := 100
  let AC := 140
  let AD := 240
  let AE := 340
  AC + AD + AE

theorem hexagon_diagonals_sum_correct : hexagon_diagonals_sum = 720 :=
  by
  show hexagon_diagonals_sum = 720
  sorry

end NUMINAMATH_GPT_hexagon_diagonals_sum_correct_l2028_202830


namespace NUMINAMATH_GPT_find_circle_equation_l2028_202811

-- Define the conditions and problem
def circle_standard_equation (p1 p2 : ℝ × ℝ) (center : ℝ × ℝ) (radius : ℝ) : Prop :=
  let (x1, y1) := p1
  let (x2, y2) := p2
  let (xc, yc) := center
  (x2 - xc)^2 + (y2 - yc)^2 = radius^2

-- Define the conditions as given in the problem
def point_on_circle : Prop := circle_standard_equation (2, 0) (2, 2) (2, 2) 2

-- The main theorem to prove that the standard equation of the circle holds
theorem find_circle_equation : 
  point_on_circle →
  ∃ h k r, h = 2 ∧ k = 2 ∧ r = 2 ∧ (x - h)^2 + (y - k)^2 = r^2 :=
by
  sorry

end NUMINAMATH_GPT_find_circle_equation_l2028_202811


namespace NUMINAMATH_GPT_hyperbola_sum_l2028_202876

theorem hyperbola_sum (h k a b : ℝ) (c : ℝ)
  (h_eq : h = 3)
  (k_eq : k = -5)
  (a_eq : a = 5)
  (c_eq : c = 7)
  (c_squared_eq : c^2 = a^2 + b^2) :
  h + k + a + b = 3 + 2 * Real.sqrt 6 :=
by
  rw [h_eq, k_eq, a_eq, c_eq] at *
  sorry

end NUMINAMATH_GPT_hyperbola_sum_l2028_202876


namespace NUMINAMATH_GPT_kenya_peanuts_correct_l2028_202831

def jose_peanuts : ℕ := 85
def kenya_more_peanuts : ℕ := 48

def kenya_peanuts : ℕ := jose_peanuts + kenya_more_peanuts

theorem kenya_peanuts_correct : kenya_peanuts = 133 := by
  sorry

end NUMINAMATH_GPT_kenya_peanuts_correct_l2028_202831


namespace NUMINAMATH_GPT_find_window_cost_l2028_202889

-- Definitions (conditions)
def total_damages : ℕ := 1450
def cost_of_tire : ℕ := 250
def number_of_tires : ℕ := 3
def cost_of_tires := number_of_tires * cost_of_tire

-- The cost of the window that needs to be proven
def window_cost := total_damages - cost_of_tires

-- We state the theorem that the window costs $700 and provide a sorry as placeholder for its proof
theorem find_window_cost : window_cost = 700 :=
by sorry

end NUMINAMATH_GPT_find_window_cost_l2028_202889


namespace NUMINAMATH_GPT_find_number_of_pourings_l2028_202842

-- Define the sequence of remaining water after each pouring
def remaining_water (n : ℕ) : ℚ :=
  (2 : ℚ) / (n + 2)

-- The main theorem statement
theorem find_number_of_pourings :
  ∃ n : ℕ, remaining_water n = 1 / 8 :=
by
  sorry

end NUMINAMATH_GPT_find_number_of_pourings_l2028_202842


namespace NUMINAMATH_GPT_area_of_triangle_OPF_l2028_202816

theorem area_of_triangle_OPF (O : ℝ × ℝ) (F : ℝ × ℝ) (P : ℝ × ℝ)
  (hO : O = (0, 0)) (hF : F = (1, 0)) (hP_on_parabola : P.2 ^ 2 = 4 * P.1)
  (hPF : dist P F = 3) : Real.sqrt 2 = 1 / 2 * abs (F.1 - O.1) * (2 * Real.sqrt 2) := 
sorry

end NUMINAMATH_GPT_area_of_triangle_OPF_l2028_202816


namespace NUMINAMATH_GPT_sum_of_three_numbers_l2028_202817

theorem sum_of_three_numbers (S F T : ℕ) (h1 : S = 150) (h2 : F = 2 * S) (h3 : T = F / 3) :
  F + S + T = 550 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_three_numbers_l2028_202817


namespace NUMINAMATH_GPT_total_lobster_pounds_l2028_202895

theorem total_lobster_pounds
  (combined_other_harbors : ℕ)
  (hooper_bay : ℕ)
  (H1 : combined_other_harbors = 160)
  (H2 : hooper_bay = 2 * combined_other_harbors) :
  combined_other_harbors + hooper_bay = 480 :=
by
  -- proof goes here
  sorry

end NUMINAMATH_GPT_total_lobster_pounds_l2028_202895


namespace NUMINAMATH_GPT_greatest_area_difference_l2028_202839

theorem greatest_area_difference :
  ∃ (l1 w1 l2 w2 : ℕ), 2 * l1 + 2 * w1 = 200 ∧ 2 * l2 + 2 * w2 = 200 ∧
  (l1 * w1 - l2 * w2 = 2401) :=
by
  sorry

end NUMINAMATH_GPT_greatest_area_difference_l2028_202839


namespace NUMINAMATH_GPT_find_triples_l2028_202850

theorem find_triples (x y z : ℝ) :
  (x + 1)^2 = x + y + 2 ∧
  (y + 1)^2 = y + z + 2 ∧
  (z + 1)^2 = z + x + 2 ↔ (x = 1 ∧ y = 1 ∧ z = 1) ∨ (x = -1 ∧ y = -1 ∧ z = -1) :=
by
  sorry

end NUMINAMATH_GPT_find_triples_l2028_202850


namespace NUMINAMATH_GPT_half_radius_of_circle_y_l2028_202810

theorem half_radius_of_circle_y 
  (r_x r_y : ℝ) 
  (h₁ : π * r_x^2 = π * r_y^2) 
  (h₂ : 2 * π * r_x = 14 * π) :
  r_y / 2 = 3.5 :=
by {
  sorry
}

end NUMINAMATH_GPT_half_radius_of_circle_y_l2028_202810


namespace NUMINAMATH_GPT_find_numbers_l2028_202853

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

end NUMINAMATH_GPT_find_numbers_l2028_202853


namespace NUMINAMATH_GPT_albert_large_pizzas_l2028_202815

-- Define the conditions
def large_pizza_slices : ℕ := 16
def small_pizza_slices : ℕ := 8
def num_small_pizzas : ℕ := 2
def total_slices_eaten : ℕ := 48

-- Define the question and requirement to prove
def number_of_large_pizzas (L : ℕ) : Prop :=
  large_pizza_slices * L + small_pizza_slices * num_small_pizzas = total_slices_eaten

theorem albert_large_pizzas :
  number_of_large_pizzas 2 :=
by
  sorry

end NUMINAMATH_GPT_albert_large_pizzas_l2028_202815


namespace NUMINAMATH_GPT_sandy_phone_bill_expense_l2028_202812

def sandy_age_now (kim_age : ℕ) : ℕ := 3 * (kim_age + 2) - 2

def sandy_phone_bill (sandy_age : ℕ) : ℕ := 10 * sandy_age

theorem sandy_phone_bill_expense
  (kim_age : ℕ)
  (kim_age_condition : kim_age = 10)
  : sandy_phone_bill (sandy_age_now kim_age) = 340 := by
  sorry

end NUMINAMATH_GPT_sandy_phone_bill_expense_l2028_202812


namespace NUMINAMATH_GPT_not_divisible_by_4_8_16_32_l2028_202859

def x := 80 + 112 + 144 + 176 + 304 + 368 + 3248 + 17

theorem not_divisible_by_4_8_16_32 : 
  ¬ (x % 4 = 0) ∧ ¬ (x % 8 = 0) ∧ ¬ (x % 16 = 0) ∧ ¬ (x % 32 = 0) := 
by 
  sorry

end NUMINAMATH_GPT_not_divisible_by_4_8_16_32_l2028_202859


namespace NUMINAMATH_GPT_period_tan_half_l2028_202826

noncomputable def period_of_tan_half : Real :=
  2 * Real.pi

theorem period_tan_half (f : Real → Real) (h : ∀ x, f x = Real.tan (x / 2)) :
  ∀ x, f (x + period_of_tan_half) = f x := 
by 
  sorry

end NUMINAMATH_GPT_period_tan_half_l2028_202826


namespace NUMINAMATH_GPT_arithmetic_sequence_a_value_l2028_202862

theorem arithmetic_sequence_a_value :
  ∀ (a : ℤ), (-7) - a = a - 1 → a = -3 :=
by
  intro a
  intro h
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_a_value_l2028_202862


namespace NUMINAMATH_GPT_children_tickets_sold_l2028_202848

theorem children_tickets_sold {A C : ℕ} (h1 : 6 * A + 4 * C = 104) (h2 : A + C = 21) : C = 11 :=
by
  sorry

end NUMINAMATH_GPT_children_tickets_sold_l2028_202848


namespace NUMINAMATH_GPT_unique_quantities_not_determinable_l2028_202851

noncomputable def impossible_to_determine_unique_quantities 
(x y : ℝ) : Prop :=
  let acid1 := 54 * 0.35
  let acid2 := 48 * 0.25
  ∀ (final_acid : ℝ), ¬(0.35 * x + 0.25 * y = final_acid ∧ final_acid = 0.75 * (x + y))

theorem unique_quantities_not_determinable :
  impossible_to_determine_unique_quantities 54 48 :=
by
  sorry

end NUMINAMATH_GPT_unique_quantities_not_determinable_l2028_202851


namespace NUMINAMATH_GPT_opposite_of_2023_l2028_202847

theorem opposite_of_2023 : -2023 = -2023 :=
by
  sorry

end NUMINAMATH_GPT_opposite_of_2023_l2028_202847


namespace NUMINAMATH_GPT_steve_cookie_boxes_l2028_202863

theorem steve_cookie_boxes (total_spent milk_cost cereal_cost banana_cost apple_cost : ℝ)
  (num_cereals num_bananas num_apples : ℕ) (cookie_cost_multiplier : ℝ) (cookie_cost : ℝ)
  (cookie_boxes : ℕ) :
  total_spent = 25 ∧ milk_cost = 3 ∧ cereal_cost = 3.5 ∧ banana_cost = 0.25 ∧ apple_cost = 0.5 ∧
  cookie_cost_multiplier = 2 ∧ 
  num_cereals = 2 ∧ num_bananas = 4 ∧ num_apples = 4 ∧
  cookie_cost = cookie_cost_multiplier * milk_cost ∧
  total_spent = (milk_cost + num_cereals * cereal_cost + num_bananas * banana_cost + num_apples * apple_cost + cookie_boxes * cookie_cost)
  → cookie_boxes = 2 :=
sorry

end NUMINAMATH_GPT_steve_cookie_boxes_l2028_202863


namespace NUMINAMATH_GPT_total_songs_l2028_202814

open Nat

/-- Define the overall context and setup for the problem --/
def girls : List String := ["Mary", "Alina", "Tina", "Hanna"]

def hanna_songs : ℕ := 7
def mary_songs : ℕ := 4

def alina_songs (a : ℕ) : Prop := a > mary_songs ∧ a < hanna_songs
def tina_songs (t : ℕ) : Prop := t > mary_songs ∧ t < hanna_songs

theorem total_songs (a t : ℕ) (h_alina : alina_songs a) (h_tina : tina_songs t) : 
  (11 + a + t) % 3 = 0 → (7 + 4 + a + t) / 3 = 7 := by
  sorry

end NUMINAMATH_GPT_total_songs_l2028_202814


namespace NUMINAMATH_GPT_number_of_correct_propositions_is_one_l2028_202832

def obtuse_angle_is_second_quadrant (θ : ℝ) : Prop :=
  θ > 90 ∧ θ < 180

def acute_angle (θ : ℝ) : Prop :=
  θ < 90

def first_quadrant_not_negative (θ : ℝ) : Prop :=
  θ > 0 ∧ θ < 90

def second_quadrant_greater_first (θ₁ θ₂ : ℝ) : Prop :=
  (θ₁ > 90 ∧ θ₁ < 180) → (θ₂ > 0 ∧ θ₂ < 90) → θ₁ > θ₂

theorem number_of_correct_propositions_is_one :
  (¬ ∀ θ, obtuse_angle_is_second_quadrant θ) ∧
  (∀ θ, acute_angle θ → θ < 90) ∧
  (¬ ∀ θ, first_quadrant_not_negative θ) ∧
  (¬ ∀ θ₁ θ₂, second_quadrant_greater_first θ₁ θ₂) →
  1 = 1 :=
by
  sorry

end NUMINAMATH_GPT_number_of_correct_propositions_is_one_l2028_202832


namespace NUMINAMATH_GPT_minimum_bailing_rate_l2028_202875

theorem minimum_bailing_rate (distance_to_shore : ℝ) (row_speed : ℝ) (leak_rate : ℝ) (max_water_intake : ℝ)
  (time_to_shore : ℝ := distance_to_shore / row_speed * 60) (total_water_intake : ℝ := time_to_shore * leak_rate) :
  distance_to_shore = 1.5 → row_speed = 3 → leak_rate = 10 → max_water_intake = 40 →
  ∃ (bail_rate : ℝ), bail_rate ≥ 9 :=
by
  sorry

end NUMINAMATH_GPT_minimum_bailing_rate_l2028_202875


namespace NUMINAMATH_GPT_number_of_lockers_l2028_202870

-- Problem Conditions
def locker_numbers_consecutive_from_one := ∀ (n : ℕ), n ≥ 1
def cost_per_digit := 0.02
def total_cost := 137.94

-- Theorem Statement
theorem number_of_lockers (h1 : locker_numbers_consecutive_from_one) (h2 : cost_per_digit = 0.02) (h3 : total_cost = 137.94) : ∃ n : ℕ, n = 2001 :=
sorry

end NUMINAMATH_GPT_number_of_lockers_l2028_202870


namespace NUMINAMATH_GPT_number_of_factors_and_perfect_square_factors_l2028_202807

open Nat

-- Define the number 1320 and its prime factorization.
def n : ℕ := 1320
def prime_factors : List (ℕ × ℕ) := [(2, 2), (3, 1), (5, 1), (11, 1)]

-- Define a function to count factors.
def count_factors (factors : List (ℕ × ℕ)) : ℕ :=
  factors.foldl (λ acc ⟨_, exp⟩ => acc * (exp + 1)) 1

-- Define a function to count perfect square factors.
def count_perfect_square_factors (factors : List (ℕ × ℕ)) : ℕ :=
  factors.foldl (λ acc ⟨prime, exp⟩ => acc * (if exp % 2 == 0 then exp / 2 + 1 else 1)) 1

theorem number_of_factors_and_perfect_square_factors :
  count_factors prime_factors = 24 ∧ count_perfect_square_factors prime_factors = 2 :=
by
  sorry

end NUMINAMATH_GPT_number_of_factors_and_perfect_square_factors_l2028_202807


namespace NUMINAMATH_GPT_inequality_solution_l2028_202808

theorem inequality_solution (x : ℝ) (h : x ≠ 2 ∧ x ≠ -2) :
    (x^2 - 9) / (x^2 - 4) > 0 ↔ (x < -3 ∨ x > 3) := by
  sorry

end NUMINAMATH_GPT_inequality_solution_l2028_202808


namespace NUMINAMATH_GPT_ab_bc_ca_plus_one_pos_l2028_202855

variable (a b c : ℝ)
variable (h₁ : |a| < 1)
variable (h₂ : |b| < 1)
variable (h₃ : |c| < 1)

theorem ab_bc_ca_plus_one_pos :
  ab + bc + ca + 1 > 0 := sorry

end NUMINAMATH_GPT_ab_bc_ca_plus_one_pos_l2028_202855


namespace NUMINAMATH_GPT_geom_seq_common_ratio_q_l2028_202803

-- Define the geometric sequence
def geom_seq (a₁ q : ℝ) (n : ℕ) : ℝ := a₁ * q ^ (n - 1)

-- State the theorem
theorem geom_seq_common_ratio_q {a₁ q : ℝ} :
  (a₁ = 2) → (geom_seq a₁ q 4 = 16) → (q = 2) :=
by
  intros h₁ h₂
  sorry

end NUMINAMATH_GPT_geom_seq_common_ratio_q_l2028_202803


namespace NUMINAMATH_GPT_total_cost_l2028_202896

variable (a b : ℝ)

def tomato_cost (a : ℝ) := 30 * a
def cabbage_cost (b : ℝ) := 50 * b

theorem total_cost (a b : ℝ) : 
  tomato_cost a + cabbage_cost b = 30 * a + 50 * b := 
by 
  unfold tomato_cost cabbage_cost
  sorry

end NUMINAMATH_GPT_total_cost_l2028_202896


namespace NUMINAMATH_GPT_difference_of_squares_144_l2028_202865

theorem difference_of_squares_144 (n : ℕ) (h : 3 * n + 3 < 150) : (n + 2)^2 - n^2 = 144 :=
by
  -- Given the conditions, we need to show this holds.
  sorry

end NUMINAMATH_GPT_difference_of_squares_144_l2028_202865


namespace NUMINAMATH_GPT_jakes_weight_l2028_202804

theorem jakes_weight (J S B : ℝ) 
  (h1 : 0.8 * J = 2 * S)
  (h2 : J + S = 168)
  (h3 : B = 1.25 * (J + S))
  (h4 : J + S + B = 221) : 
  J = 120 :=
by
  sorry

end NUMINAMATH_GPT_jakes_weight_l2028_202804


namespace NUMINAMATH_GPT_rectangle_side_excess_l2028_202885

theorem rectangle_side_excess
  (L W : ℝ)  -- length and width of the rectangle
  (x : ℝ)   -- percentage in excess for the first side
  (h1 : 0.95 * (L * (1 + x / 100) * W) = 1.102 * (L * W)) :
  x = 16 :=
by
  sorry

end NUMINAMATH_GPT_rectangle_side_excess_l2028_202885


namespace NUMINAMATH_GPT_y_share_per_rupee_l2028_202882

theorem y_share_per_rupee (a p : ℝ) (h1 : a * p = 18)
                            (h2 : p + a * p + 0.30 * p = 70) :
    a = 0.45 :=
by 
  sorry

end NUMINAMATH_GPT_y_share_per_rupee_l2028_202882


namespace NUMINAMATH_GPT_tom_split_number_of_apples_l2028_202837

theorem tom_split_number_of_apples
    (S : ℕ)
    (h1 : S = 8 * A)
    (h2 : A * 5 / 8 / 2 = 5) :
    A = 2 :=
by
  sorry

end NUMINAMATH_GPT_tom_split_number_of_apples_l2028_202837


namespace NUMINAMATH_GPT_arithmetic_sequence_sum_range_l2028_202838

noncomputable def S_n (a d : ℝ) (n : ℕ) : ℝ := n * (2 * a + (n - 1) * d) / 2

theorem arithmetic_sequence_sum_range 
  (a d : ℝ)
  (h1 : 1 ≤ a + 3 * d) 
  (h2 : a + 3 * d ≤ 4)
  (h3 : 2 ≤ a + 4 * d)
  (h4 : a + 4 * d ≤ 3) 
  : 0 ≤ S_n a d 6 ∧ S_n a d 6 ≤ 30 := 
sorry

end NUMINAMATH_GPT_arithmetic_sequence_sum_range_l2028_202838


namespace NUMINAMATH_GPT_porch_length_is_6_l2028_202894

-- Define the conditions for the house and porch areas
def house_length : ℝ := 20.5
def house_width : ℝ := 10
def porch_width : ℝ := 4.5
def total_shingle_area : ℝ := 232

-- Define the area calculations
def house_area : ℝ := house_length * house_width
def porch_area : ℝ := total_shingle_area - house_area

-- The theorem to prove
theorem porch_length_is_6 : porch_area / porch_width = 6 := by
  sorry

end NUMINAMATH_GPT_porch_length_is_6_l2028_202894


namespace NUMINAMATH_GPT_new_estimated_y_value_l2028_202883

theorem new_estimated_y_value
  (initial_slope : ℝ) (initial_intercept : ℝ) (avg_x_initial : ℝ)
  (datapoints_removed_low_x : ℝ) (datapoints_removed_high_x : ℝ)
  (datapoints_removed_low_y : ℝ) (datapoints_removed_high_y : ℝ)
  (new_slope : ℝ) 
  (x_value : ℝ)
  (estimated_y_new : ℝ) :
  initial_slope = 1.5 →
  initial_intercept = 1 →
  avg_x_initial = 2 →
  datapoints_removed_low_x = 2.6 →
  datapoints_removed_high_x = 1.4 →
  datapoints_removed_low_y = 2.8 →
  datapoints_removed_high_y = 5.2 →
  new_slope = 1.4 →
  x_value = 6 →
  estimated_y_new = new_slope * x_value + (4 - new_slope * avg_x_initial) →
  estimated_y_new = 9.6 := by
  sorry

end NUMINAMATH_GPT_new_estimated_y_value_l2028_202883


namespace NUMINAMATH_GPT_find_ending_number_l2028_202818

def ending_number (n : ℕ) : Prop :=
  18 < n ∧ n % 7 = 0 ∧ ((21 + n) / 2 : ℝ) = 38.5

theorem find_ending_number : ending_number 56 :=
by
  unfold ending_number
  sorry

end NUMINAMATH_GPT_find_ending_number_l2028_202818


namespace NUMINAMATH_GPT_train_speed_l2028_202822

/-- 
Given:
- Length of train L is 390 meters (0.39 km)
- Speed of man Vm is 2 km/h
- Time to cross man T is 52 seconds

Prove:
- The speed of the train Vt is 25 km/h
--/
theorem train_speed 
  (L : ℝ) (Vm : ℝ) (T : ℝ) (Vt : ℝ)
  (h1 : L = 0.39) 
  (h2 : Vm = 2) 
  (h3 : T = 52 / 3600) 
  (h4 : Vt + Vm = L / T) :
  Vt = 25 :=
by sorry

end NUMINAMATH_GPT_train_speed_l2028_202822


namespace NUMINAMATH_GPT_reciprocal_of_neg_eight_l2028_202827

theorem reciprocal_of_neg_eight : (1 / (-8 : ℝ)) = -1 / 8 := sorry

end NUMINAMATH_GPT_reciprocal_of_neg_eight_l2028_202827


namespace NUMINAMATH_GPT_combined_function_is_linear_l2028_202897

def original_parabola (x : ℝ) : ℝ := 3 * x^2 + 4 * x - 5

def reflected_parabola (x : ℝ) : ℝ := -original_parabola x

def translated_original_parabola (x : ℝ) : ℝ := 3 * (x - 4)^2 + 4 * (x - 4) - 5

def translated_reflected_parabola (x : ℝ) : ℝ := -3 * (x + 6)^2 - 4 * (x + 6) + 5

def combined_function (x : ℝ) : ℝ := translated_original_parabola x + translated_reflected_parabola x

theorem combined_function_is_linear : ∃ (a b : ℝ), ∀ x : ℝ, combined_function x = a * x + b := by
  sorry

end NUMINAMATH_GPT_combined_function_is_linear_l2028_202897


namespace NUMINAMATH_GPT_caterpillar_reaches_top_in_16_days_l2028_202877

-- Define the constants for the problem
def pole_height : ℕ := 20
def daytime_climb : ℕ := 5
def nighttime_slide : ℕ := 4

-- Define the final result we want to prove
theorem caterpillar_reaches_top_in_16_days :
  ∃ days : ℕ, days = 16 ∧ 
  ((20 - 5) / (daytime_climb - nighttime_slide) + 1) = 16 := by
  sorry

end NUMINAMATH_GPT_caterpillar_reaches_top_in_16_days_l2028_202877


namespace NUMINAMATH_GPT_intersection_points_vary_with_a_l2028_202868

-- Define the lines
def line1 (x : ℝ) : ℝ := x + 1
def line2 (a x : ℝ) : ℝ := a * x + 1

-- Prove that the number of intersection points varies with a
theorem intersection_points_vary_with_a (a : ℝ) : 
  (∃ x : ℝ, line1 x = line2 a x) ↔ 
    (if a = 1 then true else true) :=
by 
  sorry

end NUMINAMATH_GPT_intersection_points_vary_with_a_l2028_202868


namespace NUMINAMATH_GPT_part1_part2_l2028_202880

-- Definitions and assumptions based on the problem
def f (x a : ℝ) : ℝ := abs (x - a)

-- Condition (1) with given function and inequality solution set
theorem part1 (a : ℝ) :
  (∀ x, f x a ≤ 3 ↔ -1 ≤ x ∧ x ≤ 5) → a = 2 :=
by
  sorry

-- Condition (2) with the range of m under the previously found value of a
theorem part2 (m : ℝ) :
  (∃ x, f x 2 + f (x + 5) 2 < m) → m > 5 :=
by
  sorry

end NUMINAMATH_GPT_part1_part2_l2028_202880
