import Mathlib

namespace cheapest_pie_cost_is_18_l512_512405

noncomputable def crust_cost : ℝ := 2 + 1 + 1.5
noncomputable def blueberry_container_cost : ℝ := 2.25
noncomputable def blueberry_containers_needed : ℕ := 3 * (16 / 8)
noncomputable def blueberry_filling_cost : ℝ := blueberry_containers_needed * blueberry_container_cost
noncomputable def cherry_filling_cost : ℝ := 14
noncomputable def cheapest_filling_cost : ℝ := min blueberry_filling_cost cherry_filling_cost
noncomputable def total_cheapest_pie_cost : ℝ := crust_cost + cheapest_filling_cost

theorem cheapest_pie_cost_is_18 : total_cheapest_pie_cost = 18 := by
  sorry

end cheapest_pie_cost_is_18_l512_512405


namespace train_probability_half_l512_512377

theorem train_probability_half :
  let train_arrival_range := (0:ℝ, 60)
  let john_arrival_range := (0:ℝ, 60)
  let waiting_time := 30
  let total_area := 60 * 60
  let valid_area := (1/2) * (train_arrival_range.2 + john_arrival_range.2) * train_arrival_range.1 +
                    (1/2) * waiting_time * waiting_time
  let probability := valid_area / total_area
  probability = 1/2 :=
by
  -- definitions
  let train_arrival_range := (0:ℝ, 60)
  let john_arrival_range := (0:ℝ, 60)
  let waiting_time := 30
  let total_area := 60 * 60
  let valid_area := (1/2) * (train_arrival_range.2 + john_arrival_range.2) * train_arrival_range.1 +
                    (1/2) * waiting_time * waiting_time
  let probability := valid_area / total_area

  -- proof
  sorry

end train_probability_half_l512_512377


namespace not_a_fraction_l512_512724

axiom x : ℝ
axiom a : ℝ
axiom b : ℝ

noncomputable def A := 1 / (x^2)
noncomputable def B := (b + 3) / a
noncomputable def C := (x^2 - 1) / (x + 1)
noncomputable def D := (2 / 7) * a

theorem not_a_fraction : ¬ (D = A) ∧ ¬ (D = B) ∧ ¬ (D = C) :=
by 
  sorry

end not_a_fraction_l512_512724


namespace cost_of_song_book_l512_512148

theorem cost_of_song_book 
  (flute_cost : ℝ) 
  (stand_cost : ℝ) 
  (total_cost : ℝ) 
  (h1 : flute_cost = 142.46) 
  (h2 : stand_cost = 8.89) 
  (h3 : total_cost = 158.35) : 
  total_cost - (flute_cost + stand_cost) = 7.00 := 
by 
  sorry

end cost_of_song_book_l512_512148


namespace total_distance_traveled_l512_512802

theorem total_distance_traveled :
  let radius := 50
  let angle := 45
  let num_girls := 8
  let cos_135 := Real.cos (135 * Real.pi / 180)
  let distance_one_way := radius * Real.sqrt (2 * (1 - cos_135))
  let distance_one_girl := 4 * distance_one_way
  let total_distance := num_girls * distance_one_girl
  total_distance = 1600 * Real.sqrt (2 + Real.sqrt 2) :=
by
  let radius := 50
  let angle := 45
  let num_girls := 8
  let cos_135 := Real.cos (135 * Real.pi / 180)
  let distance_one_way := radius * Real.sqrt (2 * (1 - cos_135))
  let distance_one_girl := 4 * distance_one_way
  let total_distance := num_girls * distance_one_girl
  show total_distance = 1600 * Real.sqrt (2 + Real.sqrt 2)
  sorry

end total_distance_traveled_l512_512802


namespace red_pigment_weight_in_brown_paint_l512_512358

theorem red_pigment_weight_in_brown_paint :
  ∀ (M G : ℝ), 
    (M + G = 10) → 
    (0.5 * M + 0.3 * G = 4) →
    0.5 * M = 2.5 :=
by sorry

end red_pigment_weight_in_brown_paint_l512_512358


namespace part_a1_part_a2_part_b1_part_b2_part_b3_part_c_l512_512957

variable {ℝ : Type*} [linear_ordered_field ℝ]
variable (c s : ℝ → ℝ)
variable (n : ℕ)

axiom func_eqn : ∀ (x y : ℝ), x ≠ 0 → y ≠ 0 → c (x / y) = c x * c y - s x * s y
axiom nonconstant : ¬∀ I : set ℝ, I ≠ ∅ → (∃ α β : ℝ, I = Ioc α β → ∀ x ∈ I, c x = c (x + 1) ∧ s x = s (x + 1))

theorem part_a1 (x : ℝ) (hx : x ≠ 0) : c (1 / x) = c x := sorry
theorem part_a2 (x : ℝ) (hx : x ≠ 0) : s (1 / x) = -s x := sorry
theorem part_b1 : c 1 = 1 := sorry
theorem part_b2 : s 1 = s (-1) := sorry
theorem part_b3 : s 1 = 0 := sorry
theorem part_c : (∀ x : ℝ, c x + s x = x ^ n ∧ (∀ x : ℝ, c (-x) = c x) ∨ (∀ x : ℝ, c (-x) = -c x)) := sorry

noncomputable def find_cs : (ℝ → ℝ) × (ℝ → ℝ) :=
(λ x, real.cosh (real.log x), λ x, real.sinh (real.log x))

end part_a1_part_a2_part_b1_part_b2_part_b3_part_c_l512_512957


namespace find_h_formula_l512_512613

noncomputable def f (x : ℝ) : ℝ := 2 ^ (-x)

noncomputable def g (x : ℝ) : ℝ := -Real.log (x) / Real.log 2  -- Equivalent to -log_2(x)

noncomputable def h (x : ℝ) : ℝ := g (x + 1)  -- Shift left by 1 unit

theorem find_h_formula : h(x) = -Real.log (x + 1) / Real.log 2 := 
by
  sorry

end find_h_formula_l512_512613


namespace trigonometric_identity_proof_l512_512785

theorem trigonometric_identity_proof (h1 : tan (real.pi / 4.5) = sin (real.pi / 4.5) / cos (real.pi / 4.5)) :
  (tan (real.pi / 4.5) ^ 2 - sin (real.pi / 4.5) ^ 2) / (tan (real.pi / 4.5) ^ 2 * sin (real.pi / 4.5) ^ 2) = 1 :=
by
  sorry

end trigonometric_identity_proof_l512_512785


namespace domain_f_b_neg_5_b_range_for_fx_gt_gx_l512_512515

noncomputable def f (x : ℝ) (b : ℝ) := real.log (4^x + b * 2^x + 4) / real.log 2
def g (x : ℝ) := x

theorem domain_f_b_neg_5 :
  {x : ℝ | 4^x - 5 * 2^x + 4 > 0} = {x : ℝ | x < 0} ∪ {x : ℝ | x > 2} := by
  sorry

theorem b_range_for_fx_gt_gx :
  ∀ b : ℝ, (∀ x : ℝ, f x b > g x) → b > -3 := by
  sorry

end domain_f_b_neg_5_b_range_for_fx_gt_gx_l512_512515


namespace original_average_age_older_l512_512274

theorem original_average_age_older : 
  ∀ (n : ℕ) (T : ℕ), (T = n * 40) →
  (T + 408) / (n + 12) = 36 →
  40 - 36 = 4 :=
by
  intros n T hT hNewAvg
  sorry

end original_average_age_older_l512_512274


namespace cos_arithmetic_seq_product_l512_512045

theorem cos_arithmetic_seq_product :
  ∃ (a b : ℝ), (∃ (a₁ : ℝ), ∀ n : ℕ, (n > 0) → ∃ m : ℕ, cos (a₁ + (2 * Real.pi / 3) * (n - 1)) = [a, b] ∧ (a * b = -1 / 2)) := 
sorry

end cos_arithmetic_seq_product_l512_512045


namespace tetrahedron_equidistant_faces_geometric_sequence_l512_512385

-- Define the problem relating to the tetrahedron
theorem tetrahedron_equidistant_faces
  (A B C D O M : Type)
  (is_midpoint : is_midpoint M A B C)
  (is_centroid : is_centroid M A B C)
  (is_equidistant : ∀ face : {A, B, C, D}, distance O face = distance_from_all_faces O A B C D) :
  AO = 3 * OM :=
sorry

-- Define the problem relating to geometric sequence
theorem geometric_sequence
  (c : ℕ → ℕ) 
  (is_geometric : ∀ n, c (n + 1) = r * c n)
  (d : ℕ → ℕ) 
  (d_def : ∀ n, d n = (c 1 * c 2 * ... * c n) ^ (1/n)) :
∀ n, d (n + 1) = s * d n :=
sorry

end tetrahedron_equidistant_faces_geometric_sequence_l512_512385


namespace find_a3_l512_512463

-- Define the sequence sum S_n
def S (n : ℕ) : ℚ := (n + 1) / (n + 2)

-- Define the sequence term a_n using S_n
def a (n : ℕ) : ℚ :=
  if h : n = 1 then S 1 else S n - S (n - 1)

-- State the theorem to find the value of a_3
theorem find_a3 : a 3 = 1 / 20 :=
by
  -- The proof is omitted, use sorry to skip it
  sorry

end find_a3_l512_512463


namespace nancy_finish_time_l512_512447

def total_math_problems : ℝ := 17.0
def total_spelling_problems : ℝ := 15.0
def problems_per_hour : ℝ := 8.0
def total_problems : ℝ := total_math_problems + total_spelling_problems
def time_to_finish_all_problems (total_problems problems_per_hour : ℝ) : ℝ :=
  total_problems / problems_per_hour

theorem nancy_finish_time :
  time_to_finish_all_problems total_problems problems_per_hour = 4.0 := by
  sorry

end nancy_finish_time_l512_512447


namespace photovoltaic_profit_problem_l512_512361

/-- A photovoltaic enterprise's net profit calculation and decision problem -/
theorem photovoltaic_profit_problem:
  ∀ (n : ℕ) (h1 : n ≥ 1) (h2 : 2 < n) (h3 : n < 18),

  -- Conditions
  let investment : ℝ := 144,
  let maintenance_cost : ℝ := (4 * n^2 + 20 * n),
  let annual_revenue : ℝ := 1,

  -- Net Profit Function
  let net_profit_fn (n: ℕ) : ℝ := 100 * n - (4 * n^2 + 20 * n) - 144,

  -- Determine if project yields a profit starting from the 3rd year onwards
  net_profit_fn n > 0 ->
  2 < n ∧ n < 18 ->       -- net profit will be positive in this range
  ∃ k : ℕ, k = 3,

  -- Maximize average annual profit
  let avg_annual_profit_fn (n : ℕ) : ℝ := net_profit_fn n / n,
  let max_average_annual_profit : ℝ := 32,
  let option1_profit := 6 * max_average_annual_profit + 72,
  option1_profit = 264,

  -- Maximize net profit
  let max_net_profit : ℝ := 256,
  let option2_profit := max_net_profit + 8,
  option2_profit = 264,

  -- Option 1 is more beneficial
  option1_profit = 264 /\ option2_profit = 264 ->
  ∃ (better_option : string),
    better_option = "option ①" :=
sorry

end photovoltaic_profit_problem_l512_512361


namespace intersection_A_B_range_of_a_l512_512519

def A : Set ℝ := { x | (x - 3) / (2 * x) ≥ 1 }
def B : Set ℝ := { x | (1 : ℝ) / 8 < 2 ^ x ∧ 2 ^ x < 2 }
def C (a : ℝ) : Set ℝ := { x | 2 * a ≤ x ∧ x ≤ a + 1 }

theorem intersection_A_B :
  A = { x | -3 ≤ x ∧ x < 0 } ∧ B = { x | -3 < x ∧ x < 1 } ∧ (A ∩ B) = { x | -3 ≤ x ∧ x < 0 } :=
by
  sorry

theorem range_of_a (a : ℝ) :
  (A ∩ B) ⊇ C a ↔ (-3 / 2 < a ∧ a < -1) ∨ a > 1 :=
by
  sorry

end intersection_A_B_range_of_a_l512_512519


namespace max_ab_l512_512547

theorem max_ab (a b : ℝ) (h_geometric_mean : sqrt 3 = sqrt (3^a * 3^b)) : ab ≤ 1/4 :=
by
  have h_sum : a + b = 1, from sorry
  have h_am_gm : ab ≤ (a + b) / 2 ^ 2, from sorry
  rw [h_sum] at h_am_gm
  norm_num at h_am_gm
  exact h_am_gm

end max_ab_l512_512547


namespace log_seven_cubed_root_49_l512_512803

theorem log_seven_cubed_root_49 :
  log 7 (49 ^ (1 / 3 : ℝ)) = 2 / 3 := by
  sorry

end log_seven_cubed_root_49_l512_512803


namespace third_pipe_empties_in_240_minutes_l512_512340

-- Given conditions
def pipe_A_rate : ℝ := 1 / 60
def pipe_B_rate : ℝ := 1 / 80
def combined_rate_with_all_pipes_open : ℝ := 1 / 40

-- The rate at which the third pipe empties the cistern
def C := pipe_A_rate + pipe_B_rate - combined_rate_with_all_pipes_open

-- Proving that the third pipe alone can empty the cistern in 240 minutes
theorem third_pipe_empties_in_240_minutes : 1 / C = 240 := by
  -- All relevant conditions directly translate from the given problem
  sorry

end third_pipe_empties_in_240_minutes_l512_512340


namespace segment_lengths_l512_512436

noncomputable def segment_length := (x1 x2 : ℝ) => abs (x2 - x1)

theorem segment_lengths : 
  let A := 2
  let B := -5
  let C := -2
  let D := 4
  segment_length B A = 7 ∧ segment_length D C = 6 :=
by
  let A := 2
  let B := -5
  let C := -2
  let D := 4
  have h_AB : segment_length B A = abs (B - A) := rfl
  have h_CD : segment_length D C = abs (D - C) := rfl
  simp [segment_length, abs] at h_AB h_CD
  sorry

end segment_lengths_l512_512436


namespace subtraction_and_rounding_l512_512650

-- Define the constants for the numbers involved
def a : ℝ := 96.865
def b : ℝ := 45.239

-- Define the rounding function to the nearest tenth
def round_nearest_tenth (x : ℝ) : ℝ :=
  (Real.floor (x * 10 + 0.5)) / 10

-- State the theorem
theorem subtraction_and_rounding:
  round_nearest_tenth (a - b) = 51.6 := by
  sorry

end subtraction_and_rounding_l512_512650


namespace find_a_values_l512_512113

def setA : Set ℝ := {-1, 1/2, 1}
def setB (a : ℝ) : Set ℝ := {x | a * x^2 = 1 ∧ a ≥ 0}

def full_food (A B : Set ℝ) : Prop := A ⊆ B ∨ B ⊆ A
def partial_food (A B : Set ℝ) : Prop := (∃ x, x ∈ A ∧ x ∈ B) ∧ ¬(A ⊆ B ∨ B ⊆ A)

theorem find_a_values :
  ∀ a : ℝ, full_food setA (setB a) ∨ partial_food setA (setB a) ↔ a = 0 ∨ a = 1 ∨ a = 4 := 
by
  sorry

end find_a_values_l512_512113


namespace interval_length_l512_512578

theorem interval_length (a b m h : ℝ) (h_eq : h = m / |a - b|) : |a - b| = m / h := 
by 
  sorry

end interval_length_l512_512578


namespace arrange_comics_l512_512191

theorem arrange_comics:
  let spiderman_books := 7
  let archie_books := 6
  let garfield_books := 5
  let total_books := spiderman_books + archie_books + garfield_books
  
  let number_of_arrangements := 
          factorial spiderman_books * 
          (factorial archie_books * factorial garfield_books * 2) 
  in 
  number_of_arrangements = 871219200 :=
by sorry

end arrange_comics_l512_512191


namespace weight_of_7_weights_l512_512722

theorem weight_of_7_weights :
  ∀ (w : ℝ), (16 * w + 0.6 = 17.88) → 7 * w = 7.56 :=
by
  intros w h
  sorry

end weight_of_7_weights_l512_512722


namespace problem_1_problem_2_l512_512306

noncomputable def a_seq : ℕ → ℚ
| 0     := 0
| 1     := 1
| (n+1) := (n+2 : ℚ) / n * S_seq n
  where S_seq : ℕ → ℚ
  | 0     := 0
  | 1     := 1
  | (n+1) := S_seq n + a_seq (n+1)

theorem problem_1 (n : ℕ) (hn : n ≠ 0) :
  let Sn := λ n, ∑ i in finset.range n, a_seq (i+1)
  in (Sn (n+1) / (n+1) = 2 * Sn n / n) :=
sorry

theorem problem_2 (n : ℕ) (hn : n ≠ 0) :
  let Sn := λ n, ∑ i in finset.range n, a_seq (i+1)
  in Sn (n+1) = 4 * a_seq n :=
sorry

end problem_1_problem_2_l512_512306


namespace minimum_at_x2_l512_512902

noncomputable def f : ℝ → ℝ := λ x, x^3 + (-3) * x^2 + 0 * x - 1

theorem minimum_at_x2 (f : ℝ → ℝ) (a b c : ℝ) :
  (∀ x, f x = x^3 + a * x^2 + b * x + c) →
  (∀ x, f' x = 3 * x^2 + 2 * a * x + b) →
  (f' 0 = 0) →
  (f' 2 = 0) →
  (f 2 = -5) →
  a = -3 ∧ b = 0 ∧ (∀ x, f x = x^3 - 3 * x^2 - 1) :=
by
  intros h1 h2 h3 h4 h5
  sorry

end minimum_at_x2_l512_512902


namespace number_of_correct_statements_is_one_l512_512221

noncomputable def sequence (x : ℕ → ℝ) : Prop :=
∀ n > 0, x(n + 1) = 1 / (1 - x n)

theorem number_of_correct_statements_is_one (x : ℕ → ℝ) :
  (sequence x) →
  (x 2 = 5 → x 7 = 4 / 5) →
  (x 1 = 2 → x.sum (range 2022) ≠ 2021 / 2) →
  ((x 1 + 1) * (x 2 + 1) * x 9 = -1 → ¬ (x 1 = real.sqrt 2)) →
  1 = 1 :=
by
  -- Proof omitted, insert full proof here
  sorry

end number_of_correct_statements_is_one_l512_512221


namespace product_of_cosines_of_two_distinct_values_in_S_l512_512021

noncomputable def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + d

theorem product_of_cosines_of_two_distinct_values_in_S
  (a : ℕ → ℝ)
  (d : ℝ)
  (h_sequence: arithmetic_sequence a (2 * Real.pi / 3))
  (hS : ∃ (b c : ℝ), b ≠ c ∧ (∀ n : ℕ, ∃b', b' = cos (a n) ∧ (b' = b ∨ b' = c))) :
  (∃ b c : ℝ, b ≠ c ∧ (∀ n : ℕ, cos (a n) = b ∨ cos (a n) = c) ∧ b * c = -1 / 2) := 
sorry

end product_of_cosines_of_two_distinct_values_in_S_l512_512021


namespace product_of_cosine_values_l512_512040

noncomputable theory
open_locale classical

def arithmetic_sequence (a d : ℝ) (n : ℕ) : ℝ := a + (n - 1) * d

theorem product_of_cosine_values (a b : ℝ) 
    (h : ∃ a1 : ℝ, ∀ n : ℕ+, ∃ a b : ℝ, 
         S = {cos (arithmetic_sequence a1 (2*π/3) n) | n ∈ ℕ*} ∧
         S = {a, b}) : a * b = -1/2 :=
begin
  sorry
end

end product_of_cosine_values_l512_512040


namespace smallest_positive_period_of_f_max_min_values_of_f_l512_512899

noncomputable def f (x : ℝ) : ℝ := (Real.sin x + Real.cos x)^2 + Real.cos (2 * x)

theorem smallest_positive_period_of_f :
  ∃ T > 0, (∀ x, f (x + T) = f x) ∧ (∀ T' > 0, (∀ x, f (x + T') = f x) → T' ≥ T) :=
sorry

theorem max_min_values_of_f :
  (∀ x ∈ Set.Icc 0 (Real.pi / 2), 0 ≤ f x ∧ f x ≤ 1 + Real.sqrt 2) ∧
  (∃ x ∈ Set.Icc 0 (Real.pi / 2), f x = 0) ∧
  (∃ x ∈ Set.Icc 0 (Real.pi / 2), f x = 1 + Real.sqrt 2) :=
sorry

end smallest_positive_period_of_f_max_min_values_of_f_l512_512899


namespace intersection_point_of_lines_l512_512673

theorem intersection_point_of_lines : ∃ (x y : ℝ), x + y = 5 ∧ x - y = 1 ∧ x = 3 ∧ y = 2 :=
by
  sorry

end intersection_point_of_lines_l512_512673


namespace product_of_cosine_elements_l512_512011

-- Definitions for the problem conditions
def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + d

def S (a : ℕ → ℝ) (d : ℝ) : Set ℝ :=
  {x | ∃ n : ℕ, x = Real.cos (a n)}

-- Main theorem statement
theorem product_of_cosine_elements 
  (a : ℕ → ℝ) 
  (d : ℝ) 
  (h_seq : arithmetic_sequence a (2 * Real.pi / 3))
  (h_S_elements : (S a (2 * Real.pi / 3)).card = 2) 
  (h_S_contains : ∃ a b, a ≠ b ∧ S a (2 * Real.pi / 3) = {a, b}) :
  let (a, b) := Classical.choose (h_S_contains) in
  a * b = -1 / 2 :=
by
  sorry

end product_of_cosine_elements_l512_512011


namespace sample_standard_deviation_same_sample_range_same_l512_512857

open Nat

variables {n : ℕ} (x : Fin n → ℝ) (c : ℝ)
hypothesis (h_c : c ≠ 0)

/-- Assertion C: The sample standard deviations of the two sets of sample data are the same. -/
theorem sample_standard_deviation_same :
  (1 / n * ∑ i, (x i - (1 / n * ∑ i, x i))^2).sqrt =
  (1 / n * ∑ i, (x i + c - (1 / n * ∑ i, x i + c))^2).sqrt := sorry

/-- Assertion D: The sample ranges of the two sets of sample data are the same. -/
theorem sample_range_same :
  (Finset.sup Finset.univ x - Finset.inf Finset.univ x) =
  (Finset.sup Finset.univ (fun i => x i + c) - Finset.inf Finset.univ (fun i => x i + c)) := sorry

end sample_standard_deviation_same_sample_range_same_l512_512857


namespace find_pairs_s_t_l512_512958

theorem find_pairs_s_t (n : ℤ) (hn : n > 1) : 
  ∃ s t : ℤ, (
    (∀ x : ℝ, x ^ n + s * x = 2007 ∧ x ^ n + t * x = 2008 → 
     (s, t) = (2006, 2007) ∨ (s, t) = (-2008, -2009) ∨ (s, t) = (-2006, -2007))
  ) :=
sorry

end find_pairs_s_t_l512_512958


namespace ben_bonus_amount_l512_512397

variables (B : ℝ)

-- Conditions
def condition1 := B - (1/22) * B - (1/4) * B - (1/8) * B = 867

-- Theorem statement
theorem ben_bonus_amount (h : condition1 B) : B = 1496.50 := 
sorry

end ben_bonus_amount_l512_512397


namespace concurrency_of_lines_l512_512563

noncomputable def circle (C : Type) := 
  { center : C, radius : ℝ }

structure TangencyCondition (C : Type) :=
  (common_tangent : C → C → set (line C))

-- Given conditions
variables {C : Type} [euclidean_space C]

def circle_c : circle C := ⟨O, r⟩
def circle_c1 : circle C := ⟨O₁, r₁⟩
def circle_c2 : circle C := ⟨O₂, r₂⟩

-- Tangency conditions
variable (A₁ A₂ A : C)

axiom is_internally_tangent_circle_c_c1 : ∀ ⦃x : C⦄, x = A₁ → ∀ {y : C}, y = O₁ → (x ≠ y) ∧ (O + O₁).intersection (O₁ + A₁) = A₁
axiom is_internally_tangent_circle_c_c2 : ∀ ⦃x : C⦄, x = A₂ → ∀ {y : C}, y = O₂ → (x ≠ y) ∧ (O + O₂).intersection (O₂ + A₂) = A₂
axiom is_externally_tangent_circle_c1_c2 : ∀ ⦃x : C⦄, x = A → ∀ {y : C}, y = A → (y ≠ x) ∧ (O₁ + O₂).intersection (O₁ + A) = A

-- Goal to prove: concurrency of lines
theorem concurrency_of_lines :
  ( (line_through O A) ∩ (line_through O₁ A₂) ) ∩ ( (line_through O₂ A₁) ) ≠ ∅ :=
sorry

end concurrency_of_lines_l512_512563


namespace algebraic_expression_value_l512_512835

theorem algebraic_expression_value (x y : ℝ) (h : x - 2 = 3 * y) :
  x^2 - 6 * x * y + 9 * y^2 = 4 :=
sorry

end algebraic_expression_value_l512_512835


namespace cos_set_product_l512_512030

noncomputable def arithmetic_sequence (a1 : ℝ) (d : ℝ) : ℕ → ℝ :=
  λ n, a1 + d * (n - 1)

theorem cos_set_product (a1 : ℝ) (h1 : (arithmetic_sequence a1 (2 * Real.pi / 3) '' {n | n ≥ 1}).to_set.cos.to_finset.card = 2) :
  let S := (arithmetic_sequence a1 (2 * Real.pi / 3) '' {n | n ≥ 1}).to_set.cos.to_finset in 
  (S.to_finset : set ℝ).product = -1 / 2 := sorry

end cos_set_product_l512_512030


namespace compare_expressions_l512_512485

theorem compare_expressions (x y : ℝ) (h1: x * y > 0) (h2: x ≠ y) : 
  x^4 + 6 * x^2 * y^2 + y^4 > 4 * x * y * (x^2 + y^2) :=
by
  sorry

end compare_expressions_l512_512485


namespace find_sum_of_squares_l512_512807

theorem find_sum_of_squares (x y : ℕ) (h1 : x ≠ 0) (h2 : y ≠ 0) (h3 : x * y + x + y = 35) (h4 : x^2 * y + x * y^2 = 210) : x^2 + y^2 = 154 :=
sorry

end find_sum_of_squares_l512_512807


namespace nicholas_crackers_l512_512988

theorem nicholas_crackers (marcus_crackers mona_crackers nicholas_crackers : ℕ) 
  (h1 : marcus_crackers = 3 * mona_crackers)
  (h2 : nicholas_crackers = mona_crackers + 6)
  (h3 : marcus_crackers = 27) : nicholas_crackers = 15 := by
  sorry

end nicholas_crackers_l512_512988


namespace count_valid_functions_l512_512962

def B := {1, 2, 3, 4, 5}

/-- 
Prove that the number of functions f: B → B such that 
1. f(f(x)) is a constant function 
2. f(x) ≠ x for all x ≠ c 
is 205.
-/
theorem count_valid_functions : 
  let B := {1, 2, 3, 4, 5}
  let f : B → B
  let N := 
    ∑ (c : B), 
    ∑ (k in finset.range 5.filter (λ n, n > 0)), 
    (finset.choose 4 k) * (k : nat) ^ (4 - k)
  in 5 * N = 205 :=
by sorry

end count_valid_functions_l512_512962


namespace min_n_to_find_numbers_with_same_digit_sum_l512_512606

-- Define the sum of digits function S(m)
def sum_of_digits (m : ℕ) : ℕ :=
  m.digits.sum

-- State the main theorem for the problem
theorem min_n_to_find_numbers_with_same_digit_sum :
  ∃ n : ℕ, (n = 185) ∧ (∀ (a : finset ℕ) (ha : a.card = n) (h_subset : a ⊆ finset.range 2018),
    ∃ b : finset ℕ, b ⊆ a ∧ b.card = 8 ∧ (∃ s : ℕ, ∀ x ∈ b, sum_of_digits x = s)) :=
sorry

end min_n_to_find_numbers_with_same_digit_sum_l512_512606


namespace min_distance_PQ_curve_l512_512652

/-- The minimum distance between a point P on the curve y = x^2 + 1 (where x ≥ 0) 
    and a point Q on the curve y = √(x - 1) (where x ≥ 1) is 3√2 / 4. -/
theorem min_distance_PQ_curve (x1 x2 : ℝ) (y1 y2 : ℝ)
  (hP : y1 = x1^2 + 1) (hx1 : 0 ≤ x1)
  (hQ : y2 = sqrt (x2 - 1)) (hx2 : 1 ≤ x2) :
  dist (x1, y1) (x2, y2) ≥ (3 * Real.sqrt 2) / 4 :=
sorry

end min_distance_PQ_curve_l512_512652


namespace friendship_chains_l512_512351

   -- Definitions
   def students : Nat := 90
   def friends (s : Finset Nat) : Prop := ∀ t : Finset Nat, (t.card = 10 → ∃ a b ∈ t, a ≠ b ∧ s a b)

   -- Theorem statement (proof omitted)
   theorem friendship_chains (students : Nat) (friends : ∀ t : Finset Nat, t.card = 10 → ∃ a b ∈ t, a ≠ b ∧ friends a b) :
     ∃ g : Finset (Finset Nat), (∀ c ∈ g, ∀ x y ∈ c, x ≠ y → friends x y) ∧ g.card ≤ 9 :=
     sorry
   
end friendship_chains_l512_512351


namespace fx_fixed_point_l512_512348

theorem fx_fixed_point (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  ∃ x y, (x = -1) ∧ (y = 3) ∧ (a * (x + 1) + 2 = y) :=
by
  sorry

end fx_fixed_point_l512_512348


namespace complement_union_A_B_eq_neg2_0_l512_512536

def U : Set ℤ := {-2, -1, 0, 1, 2, 3}
def A : Set ℤ := {-1, 2}
def B : Set ℤ := {x | x^2 - 4 * x + 3 = 0}

theorem complement_union_A_B_eq_neg2_0 :
  U \ (A ∪ B) = {-2, 0} := by
  sorry

end complement_union_A_B_eq_neg2_0_l512_512536


namespace distinct_pairs_count_l512_512818

-- Define the conditions for the problem
def conditions (x y : ℕ) : Prop :=
  (0 < x) ∧ (x < y) ∧ (sqrt 2025 = sqrt x * sqrt y)

-- Define the main theorem to prove
theorem distinct_pairs_count : ∃ n, (∀ x y : ℕ, conditions x y → n = 6) :=
sorry

end distinct_pairs_count_l512_512818


namespace rectangle_area_l512_512735

theorem rectangle_area (L B : ℕ) (h1 : L - B = 23) (h2 : 2 * (L + B) = 266) : L * B = 4290 :=
by
  sorry

end rectangle_area_l512_512735


namespace isosceles_triangle_count_l512_512196

-- Definitions of points and geoboard conditions
structure Point := (x : ℕ) (y : ℕ)

def geoboard (n : ℕ) := List Point

noncomputable def is_on_geoboard (p : Point) (n : ℕ) : Prop := 
  p.x < n ∧ p.y < n

noncomputable def distance (p1 p2 : Point) : ℕ :=
  let dx := (p1.x - p2.x).natAbs
  let dy := (p1.y - p2.y).natAbs
  (dx^2 + dy^2).sqrt.natAbs

noncomputable def is_isosceles (a b c : Point) : Prop :=
  let d_ab := distance a b
  let d_bc := distance b c
  let d_ca := distance c a
  d_ab = d_bc ∨ d_bc = d_ca ∨ d_ca = d_ab

def is_valid_point (p : Point) (excluded : Point) : Prop :=
  p ≠ excluded

theorem isosceles_triangle_count :
  ∃ (board : geoboard 8) (points : List Point),
  ∀ a b : Point, 
    distance a b = 3
    → a ∈ board 
    → b ∈ board 
    → 18 = (List.filter (λ c, is_on_geoboard c 8 ∧ is_valid_point c a ∧ is_valid_point c b ∧ is_isosceles a b c) board).length :=
by 
  sorry

end isosceles_triangle_count_l512_512196


namespace worker_schedule_l512_512771

open Nat

theorem worker_schedule (x : ℕ) :
  24 * 3 + (15 - 3) * x > 408 :=
by
  sorry

end worker_schedule_l512_512771


namespace complement_A_union_B_l512_512534

-- Define the universal set U
def U : Set ℤ := {-2, -1, 0, 1, 2, 3}

-- Define the set A
def A : Set ℤ := {-1, 2}

-- Define the set B using the quadratic equation condition
def B : Set ℤ := {x | x^2 - 4*x + 3 = 0}

-- State the theorem we want to prove
theorem complement_A_union_B : (U \ (A ∪ B)) = {-2, 0} := by
  sorry

end complement_A_union_B_l512_512534


namespace minimum_cards_needed_l512_512693

/-- 
  Given a set of 900 three-digit numbers ranging from 100 to 999, 
  some of which can still be valid three-digit numbers when reversed using digits {0, 1, 6, 8, 9}, 
  prove the minimum number of cards needed to cover all numbers and their reversible counterparts is 46.
-/
theorem minimum_cards_needed : 
  let reversible_digits := {0, 1, 6, 8, 9}
  let valid_three_digit (n : ℕ) := n ≥ 100 ∧ n ≤ 999
  ∃ (n : ℕ), 
    (∀ x ∈ (list.range 900).filter valid_three_digit, (x ∈ reversible_digits ∨ 
    ∃ y ∈ reversible_digits, y = (x / 100) ∧ reversible_digits x % 10 ≠ 0)) → 
    n = 46 :=
sorry -- proof goes here

end minimum_cards_needed_l512_512693


namespace arrange_8_tetrahedra_l512_512782

-- Definitions to represent the conditions
structure Tetrahedra := 
  (vertices : Fin 4 -> ℝ^3)

def touches (T1 T2 : Tetrahedra) : Prop :=
  ∀ i j, i ≠ j → (let face_i := finset.filter (λ x, x ≠ i) finset.univ in
                  let face_j := finset.filter (λ x, x ≠ j) finset.univ in
                  ∃ polygon : set (ℝ^3), polygon.nonempty ∧ 
                  ∀ p ∈ polygon, 
                    (p ∉ convex_hull (face_i.image T1.vertices) ∨ 
                     p ∉ convex_hull (face_i.image T2.vertices)))

-- The main theorem to be proved
theorem arrange_8_tetrahedra : 
  ∃ (Ts : Fin 8 -> Tetrahedra), ∀ i j, i ≠ j → touches (Ts i) (Ts j) :=
sorry

end arrange_8_tetrahedra_l512_512782


namespace arithmetic_sequence_problem_l512_512575

theorem arithmetic_sequence_problem
  (a : ℕ → ℕ)
  (h1 : a 2 + a 3 = 15)
  (h2 : a 3 + a 4 = 20) :
  a 4 + a 5 = 25 :=
sorry

end arithmetic_sequence_problem_l512_512575


namespace area_enclosed_curve_l512_512714

-- The proof statement
theorem area_enclosed_curve (x y : ℝ) : (x^2 + y^2 = 2 * (|x| + |y|)) → 
  (area_of_enclosed_region = 2 * π + 8) :=
sorry

end area_enclosed_curve_l512_512714


namespace sequence_exists_and_unique_l512_512692

theorem sequence_exists_and_unique (a : ℕ → ℕ) :
  a 0 = 11 ∧ a 7 = 12 ∧
  (∀ n : ℕ, n < 6 → a n + a (n + 1) + a (n + 2) = 50) →
  (a 0 = 11 ∧ a 1 = 12 ∧ a 2 = 27 ∧ a 3 = 11 ∧ a 4 = 12 ∧
   a 5 = 27 ∧ a 6 = 11 ∧ a 7 = 12) :=
by
  sorry

end sequence_exists_and_unique_l512_512692


namespace population_net_increase_period_l512_512931

theorem population_net_increase_period :
  (∃ (hours : ℝ), hours = 24) :=
begin
  let birth_rate := 7 / 2,   -- birth rate per second
  let death_rate := 2 / 2,   -- death rate per second
  let net_increase_per_sec := birth_rate - death_rate, -- net increase per second
  let total_increase := 216000,
  let period_seconds := total_increase / net_increase_per_sec,
  let period_hours := period_seconds / 3600,
  use period_hours,
  have : period_hours = 24,
  { calc
    period_hours = (total_increase / net_increase_per_sec) / 3600 : by refl
    ... = (216000 / (5 / 2)) / 3600 : by refl
    ... = (216000 * 2 / 5) / 3600 : by rw div_div_eq_mul_div
    ... = (432000 / 5) / 3600 : by ring
    ... = 86400 / 3600 : by rw div_eq_mul_inv
    ... = 24 : by norm_num },
  exact this,
end

end population_net_increase_period_l512_512931


namespace sequence_primes_l512_512091

theorem sequence_primes (n : ℕ) : (num_primes (list.seq (λ k, 17 * (10^k - 1) / 9) 1 n) = 1) :=
sorry

end sequence_primes_l512_512091


namespace tomatoes_not_sold_l512_512621

theorem tomatoes_not_sold (total_harvested sold_mrs_maxwell sold_mr_wilson : ℝ)
  (h1 : total_harvested = 245.5)
  (h2 : sold_mrs_maxwell = 125.5)
  (h3 : sold_mr_wilson = 78) :
total_harvested - (sold_mrs_maxwell + sold_mr_wilson) = 42 :=
by {
  rw [h1, h2, h3],
  norm_num,
  sorry
}

end tomatoes_not_sold_l512_512621


namespace range_of_a_l512_512083

theorem range_of_a (a : ℝ) : 
  (¬ (∀ x : ℝ, 1 ≤ x ∧ x ≤ 2 → x^2 - a ≥ 0) ∧ (∃ x : ℝ, x^2 + 2 * a * x + 2 - a = 0)) → a > 1 :=
by
  sorry

end range_of_a_l512_512083


namespace sum_inequality_l512_512592

theorem sum_inequality
  (n m : ℕ) (h1 : 1 ≤ m) (h2 : m ≤ n) : 
  (∑ k in Finset.range (n - m + 1) + m, (1 / (k: ℝ)^2 + 1 / (k: ℝ)^3)) 
  ≥ m * (∑ k in Finset.range (n - m + 1) + m, (1 / (k: ℝ)^2)) ^ 2 :=
by
  sorry

end sum_inequality_l512_512592


namespace num_correct_statements_l512_512218

-- Define the sequence
def seq (x : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, n ≥ 1 → x (n + 1) = 1 / (1 - x n)

-- Define the problem statements
def statement_1 (x : ℕ → ℝ) : Prop :=
  x 2 = 5 → x 7 = 4 / 5

def statement_2 (x : ℕ → ℝ) : Prop :=
  x 1 = 2 → (∑ i in Finset.range 2022, x (i + 1)) = 2021 / 2

def statement_3 (x : ℕ → ℝ) : Prop :=
  (x 1 + 1) * (x 2 + 1) * x 9 = -1 → x 1 = Real.sqrt 2

-- Main theorem to prove
theorem num_correct_statements (x : ℕ → ℝ) (h_seq : seq x) :
  (if statement_1 x then 1 else 0) +
  (if statement_2 x then 1 else 0) +
  (if statement_3 x then 1 else 0) = 1 := by
  sorry

end num_correct_statements_l512_512218


namespace xiao_ming_english_score_l512_512630

theorem xiao_ming_english_score :
  let a := 92
  let b := 90
  let c := 95
  let w_a := 3
  let w_b := 3
  let w_c := 4
  let total_weight := (w_a + w_b + w_c)
  let score := (a * w_a + b * w_b + c * w_c) / total_weight
  score = 92.6 :=
by
  sorry

end xiao_ming_english_score_l512_512630


namespace fraction_remaining_after_first_song_is_5_over_8_l512_512625

theorem fraction_remaining_after_first_song_is_5_over_8:
  (let n := 900 in
  let before_start := (3 / 4 : ℚ) * n in
  let did_not_go := 20 in
  let remaining := n - before_start - did_not_go in
  let arrived_during_middle := 80 in
  let arrived_after_first_song := remaining - arrived_during_middle in
  arrived_after_first_song / remaining = 5 / 8) :=
begin
  let n := 900 : ℚ,
  let before_start := (3 / 4 : ℚ) * n,
  let did_not_go := 20 : ℚ,
  let remaining := n - before_start - did_not_go,
  let arrived_during_middle := 80 : ℚ,
  let arrived_after_first_song := remaining - arrived_during_middle,
  have h : arrived_after_first_song / remaining = 5 / 8 := sorry,
  exact h,
end

end fraction_remaining_after_first_song_is_5_over_8_l512_512625


namespace solve_fractional_equation_l512_512258

theorem solve_fractional_equation (x : ℝ) (h₁ : x ≠ 1) (h₂ : x ≠ -1) :
  (3 / (x - 1) = 5 + 3 * x / (1 - x)) → (x = 4) :=
by 
  intro h
  -- proof steps would go here
  sorry

end solve_fractional_equation_l512_512258


namespace charity_fundraising_l512_512728

theorem charity_fundraising (num_people : ℕ) (amount_event1 amount_event2 : ℕ) (total_amount_per_person : ℕ) :
  num_people = 8 →
  amount_event1 = 2000 →
  amount_event2 = 1000 →
  total_amount_per_person = (amount_event1 + amount_event2) / num_people →
  total_amount_per_person = 375 :=
by
  intros h1 h2 h3 h4
  sorry

end charity_fundraising_l512_512728


namespace sequence_bn_general_formula_sum_first_20_terms_l512_512844

theorem sequence_bn_general_formula (a : ℕ → ℕ)
  (h1 : a 1 = 1)
  (h2 : ∀ n, if n % 2 = 1 then a (n + 1) = a n + 1 else a (n + 1) = a n + 2)
  (b : ℕ → ℕ) :
  b 1 = 2 ∧ b 2 = 5 ∧ ∀ n, b n = 3 * n - 1 :=
by
  sorry

theorem sum_first_20_terms (a : ℕ → ℕ)
  (h1 : a 1 = 1)
  (h2 : ∀ n, if n % 2 = 1 then a (n + 1) = a n + 1 else a (n + 1) = a n + 2) :
  (∑ i in finset.range 20, a (i + 1)) = 300 :=
by
  sorry

end sequence_bn_general_formula_sum_first_20_terms_l512_512844


namespace product_of_cosine_values_l512_512038

noncomputable theory
open_locale classical

def arithmetic_sequence (a d : ℝ) (n : ℕ) : ℝ := a + (n - 1) * d

theorem product_of_cosine_values (a b : ℝ) 
    (h : ∃ a1 : ℝ, ∀ n : ℕ+, ∃ a b : ℝ, 
         S = {cos (arithmetic_sequence a1 (2*π/3) n) | n ∈ ℕ*} ∧
         S = {a, b}) : a * b = -1/2 :=
begin
  sorry
end

end product_of_cosine_values_l512_512038


namespace sum_of_digits_of_second_smallest_multiple_l512_512963

theorem sum_of_digits_of_second_smallest_multiple :
  let M := Nat.lcm 1 (Nat.lcm 2 (Nat.lcm 3 (Nat.lcm 4 (Nat.lcm 5 (Nat.lcm 6 (Nat.lcm 7 (Nat.lcm 8 9)))))))
  let second_smallest_multiple := 2 * M
  Nat.digits 10 second_smallest_multiple.sum = 9 :=
by
  let M := Nat.lcm 1 (Nat.lcm 2 (Nat.lcm 3 (Nat.lcm 4 (Nat.lcm 5 (Nat.lcm 6 (Nat.lcm 7 (Nat.lcm 8 9)))))))
  let second_smallest_multiple := 2 * M
  have h : second_smallest_multiple = 5040 := sorry
  have h_sum : Nat.digits 10 second_smallest_multiple.sum = 9 := sorry
  exact h_sum

end sum_of_digits_of_second_smallest_multiple_l512_512963


namespace complement_A_union_B_l512_512530

-- Define the universal set U
def U : Set ℤ := {-2, -1, 0, 1, 2, 3}

-- Define the set A
def A : Set ℤ := {-1, 2}

-- Define the set B using the quadratic equation condition
def B : Set ℤ := {x | x^2 - 4*x + 3 = 0}

-- State the theorem we want to prove
theorem complement_A_union_B : (U \ (A ∪ B)) = {-2, 0} := by
  sorry

end complement_A_union_B_l512_512530


namespace power_of_3_even_tens_digit_l512_512202

theorem power_of_3_even_tens_digit (n : ℤ) (hn : n ≥ 3) : (3^n % 100 / 10) % 2 = 0 :=
sorry

end power_of_3_even_tens_digit_l512_512202


namespace product_of_cosines_of_two_distinct_values_in_S_l512_512022

noncomputable def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + d

theorem product_of_cosines_of_two_distinct_values_in_S
  (a : ℕ → ℝ)
  (d : ℝ)
  (h_sequence: arithmetic_sequence a (2 * Real.pi / 3))
  (hS : ∃ (b c : ℝ), b ≠ c ∧ (∀ n : ℕ, ∃b', b' = cos (a n) ∧ (b' = b ∨ b' = c))) :
  (∃ b c : ℝ, b ≠ c ∧ (∀ n : ℕ, cos (a n) = b ∨ cos (a n) = c) ∧ b * c = -1 / 2) := 
sorry

end product_of_cosines_of_two_distinct_values_in_S_l512_512022


namespace number_of_possible_values_for_n_l512_512294

/-- Given a polynomial \(x^3 - 2015x^2 + mx + n\) with integer coefficients 
  and three distinct positive zeros, exactly one of which is an integer and the average of the other two, 
  there are 254015 possible values for \(n\). --/
theorem number_of_possible_values_for_n : 
  ∃ n : ℤ, ∃ (a r : ℝ), 
  (
    (x : ℝ → ℝ, x^3 - 2015 * x^2 + m * x + n = (x - a) * (x - (a / 2 + r)) * (x - (a / 2 - r))) ∧
    a ∈ ℤ ∧ a > 0 ∧
    r ∉ ℚ ∧ r > 0 ∧
    2 * a = 2015 ∧ 
    (∃ k : ℤ, 1 ≤ k ∧ k ≤ 254015 ∧ r^2 = (1 / 4) * a^2 - k)
  )
sorry

end number_of_possible_values_for_n_l512_512294


namespace center_square_is_nine_l512_512384

theorem center_square_is_nine (grid : Fin 3 → Fin 3 → ℕ) 
  (h1 : ∀ i j, grid i j ∈ Finset.range 1 10)
  (h2 : ∀ i j, (i < 2 → grid (i + 1) j = grid i j + 1) ∨ (j < 2 → grid i (j + 1) = grid i j + 1) ∨ 
               ((i > 0) → grid (i - 1) j = grid i j - 1) ∨ ((j > 0) → grid i (j - 1) = grid i j - 1))
  (h3 : grid 0 0 + grid 0 1 + grid 0 2 = 15) :
  grid 1 1 = 9 := 
sorry

end center_square_is_nine_l512_512384


namespace sin_cos_identity_angle_between_vectors_l512_512480

theorem sin_cos_identity (θ : ℝ)
  (h1 : (Real.cos θ - 2, Real.sin θ) • (Real.cos θ, Real.sin θ - 2) = -1 / 3) :
  Real.sin θ * Real.cos θ = -5 / 18 :=
begin
  sorry,
end

theorem angle_between_vectors (θ : ℝ)
  (h2 : (Real.cos θ + 2)^2 + (Real.sin θ)^2 = 7)
  (h3 : 0 < θ ∧ θ < Real.pi / 2) :
  Real.cos (Real.angle (0, 2) (Real.cos θ, Real.sin θ)) = Real.sqrt 3 / 2 :=
begin
  sorry,
end

end sin_cos_identity_angle_between_vectors_l512_512480


namespace order_of_magnitude_l512_512677

noncomputable def a : Real := 70.3
noncomputable def b : Real := 70.2
noncomputable def c : Real := Real.log 0.3

theorem order_of_magnitude : a > b ∧ b > c :=
by
  sorry

end order_of_magnitude_l512_512677


namespace solve_fractional_equation_l512_512252

theorem solve_fractional_equation (x : ℝ) (h₁ : x ≠ 1) (h₂ : x ≠ -1) :
  (3 / (x - 1) = 5 + 3 * x / (1 - x)) → (x = 4) :=
by 
  intro h
  -- proof steps would go here
  sorry

end solve_fractional_equation_l512_512252


namespace cubical_cake_combined_volume_surface_area_l512_512749

-- Definitions based on the problem conditions
def edge_length := 3
def volume (a : ℕ) := a^3
def triangular_area (b h : ℝ) := 0.5 * b * h
def total_surface_area (t_area r_area n : ℝ) := 2 * t_area + n * r_area

-- Main theorem statement
theorem cubical_cake_combined_volume_surface_area :
  let a := edge_length in
  let c := triangular_area 3 1.5 * a in
  let s := total_surface_area (triangular_area 3 1.5) (3 * 1.5) 3 in
  c + s = 24.75 :=
by
  sorry

end cubical_cake_combined_volume_surface_area_l512_512749


namespace cone_central_angle_lateral_surface_l512_512582

-- Definition of the problem's conditions
def cone_slant_height := 4
def max_cross_section_area := 4 * Real.sqrt 3

-- Angle theta such that the area of the axial cross-section is maximized
def theta : Real := Real.pi / 3

-- Radius of the base of the cone
def base_radius := 2

-- The central angle of the sector in the lateral surface development of the cone
def central_angle (r : Real) : Real := (2 * Real.pi * r) / cone_slant_height

theorem cone_central_angle_lateral_surface :
  central_angle base_radius = Real.pi :=
by
  sorry

end cone_central_angle_lateral_surface_l512_512582


namespace quadratic_coefficients_l512_512449

theorem quadratic_coefficients (a b c : ℝ) (h₀: 0 < a) 
  (h₁: |a + b + c| = 3) 
  (h₂: |4 * a + 2 * b + c| = 3) 
  (h₃: |9 * a + 3 * b + c| = 3) : 
  (a = 6 ∧ b = -24 ∧ c = 21) ∨ (a = 3 ∧ b = -15 ∧ c = 15) ∨ (a = 3 ∧ b = -9 ∧ c = 3) :=
sorry

end quadratic_coefficients_l512_512449


namespace complement_union_A_B_eq_neg2_0_l512_512538

def U : Set ℤ := {-2, -1, 0, 1, 2, 3}
def A : Set ℤ := {-1, 2}
def B : Set ℤ := {x | x^2 - 4 * x + 3 = 0}

theorem complement_union_A_B_eq_neg2_0 :
  U \ (A ∪ B) = {-2, 0} := by
  sorry

end complement_union_A_B_eq_neg2_0_l512_512538


namespace joe_height_l512_512216

theorem joe_height (S J : ℕ) (h1 : S + J = 120) (h2 : J = 2 * S + 6) : J = 82 :=
by
  sorry

end joe_height_l512_512216


namespace find_alpha_l512_512129

-- Defining the rectangle and conditions
variables (α : ℝ)
-- Define angles
variables (ABD ACD : ℝ)
-- Angle sum
def angle_sum (ABD ACD α : ℝ) : Prop := α = ABD + ACD

-- Given conditions
axiom square_side_length (length : ℝ) : length = 1
axiom rectangle_subdivided (ABD ACD : ℝ) : 0 < ABD ∧ ABD < 45 ∧ 0 < ACD ∧ ACD < 45

-- Main theorem statement
theorem find_alpha (ABD ACD : ℝ) (h : angle_sum ABD ACD α) 
  (h_subdivided : rectangle_subdivided ABD ACD) : α = 45 :=
begin
  sorry,
end

end find_alpha_l512_512129


namespace cos_arithmetic_seq_product_l512_512048

theorem cos_arithmetic_seq_product :
  ∃ (a b : ℝ), (∃ (a₁ : ℝ), ∀ n : ℕ, (n > 0) → ∃ m : ℕ, cos (a₁ + (2 * Real.pi / 3) * (n - 1)) = [a, b] ∧ (a * b = -1 / 2)) := 
sorry

end cos_arithmetic_seq_product_l512_512048


namespace John_completes_work_alone_10_days_l512_512152

theorem John_completes_work_alone_10_days
  (R : ℕ)
  (T : ℕ)
  (W : ℕ)
  (H1 : R = 40)
  (H2 : T = 8)
  (H3 : 1/10 = (1/R) + (1/W))
  : W = 10 := sorry

end John_completes_work_alone_10_days_l512_512152


namespace solve_fractional_equation_l512_512229

theorem solve_fractional_equation (x : ℝ) (hx : x ≠ 1 ∧ x ≠ 1) : 3 / (x - 1) = 5 + 3 * x / (1 - x) ↔ x = 4 :=
by 
  split
  -- Forward direction (Given the equation, show x = 4)
  intro h
  -- Multiply out and simplify as shown in the solution steps
  -- Sorry statement skips the detailed proof
  sorry
  -- Converse direction (Substitute x = 4 and verify it satisfies the equation)
  intro hx
  -- Substitution and verification steps
  -- Sorry statement skips the detailed proof
  sorry

end solve_fractional_equation_l512_512229


namespace cos_585_eq_neg_sqrt2_div_2_l512_512780

theorem cos_585_eq_neg_sqrt2_div_2 :
  Real.cos (585 * Real.pi / 180) = -Real.sqrt 2 / 2 :=
by
  sorry

end cos_585_eq_neg_sqrt2_div_2_l512_512780


namespace solve_equation_l512_512239

theorem solve_equation (x : ℝ) (h : x ≠ 1 ∧ x ≠ 1) : (3 / (x - 1) = 5 + 3 * x / (1 - x)) ↔ x = 4 := by
  sorry

end solve_equation_l512_512239


namespace part_a_part_b_l512_512591

-- Define the problem conditions
variables {n z : ℤ}
variables (h1 : n > 1) (h2 : z > 1) (h3 : Int.gcd n z = 1)

-- (a) At least one z_i is divisible by n for 0 ≤ i < n
theorem part_a : 
  ∃ i : ℕ, 0 ≤ i ∧ i < n ∧ n ∣ (1 + ∑ j in (Finset.range i).map Finset.range.map id, z^j) :=
by sorry

-- Additional condition for (b)
variables (h4 : Int.gcd (z - 1) n = 1)

-- (b) At least one z_i is divisible by n given (z-1, n) = 1
theorem part_b : 
  ∃ i : ℕ, 0 ≤ i ∧ i < n ∧ n ∣ (1 + ∑ j in (Finset.range i).map Finset.range.map id, z^j) :=
by sorry

end part_a_part_b_l512_512591


namespace daniel_total_earnings_l512_512423

-- Definitions of conditions
def fabric_delivered_monday : ℕ := 20
def fabric_delivered_tuesday : ℕ := 2 * fabric_delivered_monday
def fabric_delivered_wednesday : ℕ := fabric_delivered_tuesday / 4
def total_fabric_delivered : ℕ := fabric_delivered_monday + fabric_delivered_tuesday + fabric_delivered_wednesday

def cost_per_yard : ℕ := 2
def total_earnings : ℕ := total_fabric_delivered * cost_per_yard

-- Proposition to be proved
theorem daniel_total_earnings : total_earnings = 140 := by
  sorry

end daniel_total_earnings_l512_512423


namespace min_value_of_f_l512_512741

/-- Define the function f(x). -/
def f (x : ℝ) : ℝ :=
  4^x + 4^(-x) - 2^(x+1) - 2^(1-x) + 5

/-- Prove that the minimum value of f(x) is 3. -/
theorem min_value_of_f : ∃ x₀ : ℝ, f x₀ = 3 ∧ ∀ x : ℝ, f x ≥ 3 := 
sorry

end min_value_of_f_l512_512741


namespace count_clicks_for_speed_approximation_l512_512301

theorem count_clicks_for_speed_approximation
  (train_speed : ℝ) -- speed in miles per hour
  (rail_length : ℝ := 50) -- length of each rail in feet
  : (60 * rail_length) / 5280 ≈ 34 := 
begin
  -- We define the conversions used in the problem.
  let feet_per_mile := 5280,
  let minutes_per_hour := 60,
  let clicks_time := 34,

  -- Now we use the given train speed in miles per hour and the length of the rail in feet to show the approximation.
  have speed_in_feet_per_minute : ℝ := (train_speed * feet_per_mile) / minutes_per_hour,
  have clicks_per_minute : ℝ := speed_in_feet_per_minute / rail_length,
  have time_to_count_clicks : ℝ := train_speed / clicks_per_minute,

  -- Simplify the time to count clicks to show it is approximately 34 seconds.
  have time_in_seconds := time_to_count_clicks * 60,
  have approx_time : ℝ := (3000 / 5280) * 60,

  exact calc 
    approx_time 
        ≈ time_in_seconds : by sorry
        ... ≈ clicks_time : by norm_num,
end

end count_clicks_for_speed_approximation_l512_512301


namespace base4_quotient_l512_512806

theorem base4_quotient :
  let n1 := 1 * 4^3 + 2 * 4^2 + 1 * 4^1 + 3 * 4^0,
      n2 := 1 * 4^1 + 3 * 4^0,
      q  := n1 / n2
  in q = 3 * 4^1 + 2 * 4^0 :=
by
  sorry

end base4_quotient_l512_512806


namespace product_of_cosine_values_l512_512023

def arithmetic_seq (a₁ d : ℝ) (n : ℕ) : ℝ := a₁ + d * (n - 1)

theorem product_of_cosine_values (a₁ : ℝ) (h : ∃ (a b : ℝ), S = {a, b} ∧ S = {cos (arithmetic_seq a₁ (2 * π / 3) n) | n ∈ ℕ.succ}) :
  ∃ (a b : ℝ), a * b = -1 / 2 :=
begin
  obtain ⟨a, b, hS₁, hS₂⟩ := h,
  -- the proof will go here
  sorry
end

end product_of_cosine_values_l512_512023


namespace expected_winnings_value_l512_512379

-- Define the conditions
def prob_heads : ℚ := 1/3
def prob_tails : ℚ := 1/2
def prob_edge : ℚ := 1/6

def win_heads : ℚ := 2
def win_tails : ℚ := 4
def loss_edge : ℚ := -6

-- Define the expectation calculation
def expected_winnings : ℚ :=
  (prob_heads * win_heads) + (prob_tails * win_tails) + (prob_edge * loss_edge)

-- Theorem stating the expected winnings
theorem expected_winnings_value : (expected_winnings : ℚ) ≈ 1.67 :=
by
  sorry

end expected_winnings_value_l512_512379


namespace painters_work_days_l512_512948

noncomputable def work_product (n : ℕ) (d : ℚ) : ℚ := n * d

theorem painters_work_days :
  (work_product 5 2 = work_product 4 (2 + 1/2)) :=
by
  sorry

end painters_work_days_l512_512948


namespace sum_segments_l512_512983

-- Define the vertices of a regular octagon inscribed in a unit circle
noncomputable def P := Complex.exp (0 * Complex.I)
noncomputable def Q := Complex.exp (Complex.pi / 4 * Complex.I)
noncomputable def R := Complex.exp (Complex.pi / 2 * Complex.I)
noncomputable def S := Complex.exp (3 * Complex.pi / 4 * Complex.I)
noncomputable def T := Complex.exp (Complex.pi * Complex.I)
noncomputable def U := Complex.exp (5 * Complex.pi / 4 * Complex.I)
noncomputable def V := Complex.exp (3 * Complex.pi / 2 * Complex.I)
noncomputable def W := Complex.exp (7 * Complex.pi / 4 * Complex.I)

-- Define the segments as differences between vertices
noncomputable def A := Q - P
noncomputable def B := R - Q
noncomputable def C := S - R
noncomputable def D := T - S

-- Prove the sums of segments
theorem sum_segments : 
  (A + C = Complex.exp (Complex.pi / 4 * Complex.I) - 1 + Complex.exp (3 * Complex.pi / 4 * Complex.I) - Complex.exp (Complex.pi / 2 * Complex.I) ∧
   B + D = Complex.exp (Complex.pi / 2 * Complex.I) - Complex.exp (Complex.pi / 4 * Complex.I) + Complex.exp (Complex.pi * Complex.I) - Complex.exp (3 * Complex.pi / 4 * Complex.I) ∧
   A + B + C = (Complex.exp (Complex.pi / 4 * Complex.I) - 1 + Complex.exp (3 * Complex.pi / 4 * Complex.I) - Complex.exp (Complex.pi / 2 * Complex.I)) + (Complex.exp (Complex.pi / 2 * Complex.I) - Complex.exp (Complex.pi / 4 * Complex.I)) ∧
   A + B + C + D = (Complex.exp (Complex.pi / 4 * Complex.I) - 1 + Complex.exp (3 * Complex.pi / 4 * Complex.I) - Complex.exp (Complex.pi / 2 * Complex.I)) + (Complex.exp (Complex.pi / 2 * Complex.I) - Complex.exp (Complex.pi / 4 * Complex.I)) + (Complex.exp (Complex.pi * Complex.I) - Complex.exp (3 * Complex.pi / 4 * Complex.I)))
  := sorry

end sum_segments_l512_512983


namespace smallest_n_for_decimal_364_l512_512277

theorem smallest_n_for_decimal_364 (m n : ℕ) (h_coprime : Nat.coprime m n) (h_pos : 0 < m) (h_bound : m < n) (h_decimal : ∃ k, Decimal.ofNat m / Decimal.ofNat n = 0.3 + 6 / 10^1 + 4 / 10^2 + k) : n = 8 :=
sorry

end smallest_n_for_decimal_364_l512_512277


namespace carol_extra_invitations_l512_512404

theorem carol_extra_invitations : 
  let invitations_per_pack := 3
  let packs_bought := 2
  let friends_to_invite := 9
  packs_bought * invitations_per_pack < friends_to_invite → 
  friends_to_invite - (packs_bought * invitations_per_pack) = 3 :=
by 
  intros _  -- Introduce the condition
  exact sorry  -- Placeholder for the proof

end carol_extra_invitations_l512_512404


namespace box_width_l512_512759

theorem box_width (h : ℝ) (d : ℝ) (l : ℝ) (w : ℝ) 
  (h_eq_8 : h = 8)
  (l_eq_2h : l = 2 * h)
  (d_eq_20 : d = 20) :
  w = 4 * Real.sqrt 5 :=
by
  sorry

end box_width_l512_512759


namespace circle_through_four_points_l512_512701

theorem circle_through_four_points
  (n k : ℤ)
  (h1 : n ≥ 2)
  (h2 : k ≥ (5 * n) / 2 - 1) :
  ∀ (points : set (ℤ × ℤ)), points.card = k →
  (∀ (pt : ℤ × ℤ), pt ∈ points → 
    1 ≤ pt.1 ∧ pt.1 ≤ n ∧ 
    1 ≤ pt.2 ∧ pt.2 ≤ n) →
  ∃ (p1 p2 p3 p4 : ℤ × ℤ), 
    {p1, p2, p3, p4} ⊆ points ∧ 
    ∃ (c : (ℝ × ℝ) × ℝ), 
    (c.1.1 - p1.1) ^ 2 + 
    (c.1.2 - p1.2) ^ 2 = c.2 ^ 2 ∧
    (c.1.1 - p2.1) ^ 2 + 
    (c.1.2 - p2.2) ^ 2 = c.2 ^ 2 ∧
    (c.1.1 - p3.1) ^ 2 + 
    (c.1.2 - p3.2) ^ 2 = c.2 ^ 2 ∧
    (c.1.1 - p4.1) ^ 2 + 
    (c.1.2 - p4.2) ^ 2 = c.2 ^ 2 := sorry

end circle_through_four_points_l512_512701


namespace rectangle_length_fraction_l512_512674

theorem rectangle_length_fraction 
  (s r : ℝ) 
  (A b ℓ : ℝ)
  (area_square : s * s = 1600)
  (radius_eq_side : r = s)
  (area_rectangle : A = ℓ * b)
  (breadth_rect : b = 10)
  (area_rect_val : A = 160) :
  ℓ / r = 2 / 5 := 
by
  sorry

end rectangle_length_fraction_l512_512674


namespace set1_p_or_q_set1_p_and_q_set1_not_p_set2_p_or_q_set2_p_and_q_set2_not_p_l512_512797

-- Definitions for Set (1)
def p1 : Prop := ∃ x : ℝ, x^2 + 1 = 0
def q1 : Prop := (1 = -1)

-- Definitions for Set (2)
def p2 : Prop := ∀ (a b : ℝ) (iso : IsoscelesTriangle a b), base_angles_equal iso
def q2 : Prop := ∀ (a b : ℝ) (iso : IsoscelesTriangle a b), acute_triangle iso

-- Proof statements for Set (1)
theorem set1_p_or_q : p1 ∨ q1 := true
theorem set1_p_and_q : p1 ∧ q1 = false := sorry
theorem set1_not_p : ¬p1 := true

-- Proof statements for Set (2)
theorem set2_p_or_q : p2 ∨ q2 := true
theorem set2_p_and_q : p2 ∧ q2 = false := sorry
theorem set2_not_p : ¬p2 = false := sorry

end set1_p_or_q_set1_p_and_q_set1_not_p_set2_p_or_q_set2_p_and_q_set2_not_p_l512_512797


namespace Laura_pays_more_l512_512585

theorem Laura_pays_more 
  (slices : ℕ) 
  (cost_plain : ℝ) 
  (cost_mushrooms : ℝ) 
  (laura_mushroom_slices : ℕ) 
  (laura_plain_slices : ℕ) 
  (jessica_plain_slices: ℕ) :
  slices = 12 →
  cost_plain = 12 →
  cost_mushrooms = 3 →
  laura_mushroom_slices = 4 →
  laura_plain_slices = 2 →
  jessica_plain_slices = 6 →
  15 / 12 * (laura_mushroom_slices + laura_plain_slices) - 
  (cost_plain / 12 * jessica_plain_slices) = 1.5 :=
by
  intro slices_eq
  intro cost_plain_eq
  intro cost_mushrooms_eq
  intro laura_mushroom_slices_eq
  intro laura_plain_slices_eq
  intro jessica_plain_slices_eq
  sorry

end Laura_pays_more_l512_512585


namespace solve_fractional_equation_l512_512256

theorem solve_fractional_equation (x : ℝ) (h₁ : x ≠ 1) (h₂ : x ≠ -1) :
  (3 / (x - 1) = 5 + 3 * x / (1 - x)) → (x = 4) :=
by 
  intro h
  -- proof steps would go here
  sorry

end solve_fractional_equation_l512_512256


namespace projectile_first_reaches_20_l512_512280

-- Given conditions
def height_eq (t : ℝ) : ℝ := -4.2 * t^2 + 18.9 * t

-- Problem statement
theorem projectile_first_reaches_20 (t : ℝ) : height_eq t = 20 -> t ≈ 0.8810 :=
by
  intros h
  sorry -- Proof is omitted, only the problem statement is needed.

end projectile_first_reaches_20_l512_512280


namespace problem1_problem2_problem3_l512_512085

noncomputable def a (x : ℝ) : ℝ × ℝ := (sqrt 3 * cos x, cos x)
noncomputable def b (x : ℝ) : ℝ × ℝ := (0, sin x)
noncomputable def c (x : ℝ) : ℝ × ℝ := (sin x, cos x)
noncomputable def d (x : ℝ) : ℝ × ℝ := (sin x, sin x)

def theta (x : ℝ) : ℝ := 
    let a_dot_b := (sqrt 3 * cos x) * 0 + (cos x) * (sin x)
    let norm_a := sqrt ((sqrt 3 * cos x) ^ 2 + (cos x) ^ 2)
    let norm_b := sqrt (0 ^ 2 + (sin x) ^ 2)
    acos (a_dot_b / (norm_a * norm_b))

theorem problem1 (x : ℝ) : x = π / 4 → theta x = π / 3 :=
by
  intro hx
  rw [hx]
  sorry

theorem problem2 (x k : ℤ) : x = 3 * π / 8 + k * π → (c x) ∙ (d x) = (sqrt 2 + 1) / 2 :=
by
  intro hx
  rw [hx]
  sorry

noncomputable def m (s : ℝ) (t : ℝ) : ℝ × ℝ := (s, t)
noncomputable def norm_m (s : ℝ) (t : ℝ) : ℝ := sqrt (s ^ 2 + t ^ 2)

theorem problem3 : min (norm_m (π / 12 + 0 * π) 1) = (sqrt (π ^ 2 + 144)) / 12 :=
by
  sorry

end problem1_problem2_problem3_l512_512085


namespace probability_f4_positive_l512_512486

theorem probability_f4_positive {f : ℝ → ℝ} (h_odd : ∀ x, f (-x) = -f x)
  (h_fn : ∀ x < 0, f x = a + x + Real.logb 2 (-x)) (h_a : a > -4 ∧ a < 5) :
  (1/3 : ℝ) < (2/3 : ℝ) :=
sorry

end probability_f4_positive_l512_512486


namespace trip_total_time_l512_512355

theorem trip_total_time 
  (x : ℕ) 
  (h1 : 30 * 5 = 150) 
  (h2 : 42 * x + 150 = 38 * (x + 5)) 
  (h3 : 38 = (150 + 42 * x) / (5 + x)) : 
  5 + x = 15 := by
  sorry

end trip_total_time_l512_512355


namespace circle_polar_equation_l512_512074

-- Definitions and conditions
def circle_equation_cartesian (x y : ℝ) : Prop :=
  x^2 + y^2 - 2 * y = 0

def polar_coordinates (ρ θ : ℝ) (x y : ℝ) : Prop :=
  x = ρ * Real.cos θ ∧ y = ρ * Real.sin θ

-- Theorem to be proven
theorem circle_polar_equation (ρ θ : ℝ) :
  (∀ x y : ℝ, circle_equation_cartesian x y → 
  polar_coordinates ρ θ x y) → ρ = 2 * Real.sin θ :=
by
  -- This is a placeholder for the proof
  sorry

end circle_polar_equation_l512_512074


namespace sixteen_ray_partitional_not_twelve_ray_partitional_l512_512172

-- Definitions and conditions
def unit_square_region : Type :=
{ x : ℝ × ℝ // 0 ≤ x.1 ∧ x.1 ≤ 1 ∧ 0 ≤ x.2 ∧ x.2 ≤ 1 }

def n_ray_partitional (n : ℕ) (X : unit_square_region) : Prop :=
∃ rays : Fin n → ℝ × ℝ,
  (∀ i, 0 ≤ rays i.1 ∧ rays i.1 ≤ 1 ∧ 0 ≤ rays i.2 ∧ rays i.2 ≤ 1) ∧
  ∀ i j, i ≠ j → rays i ≠ rays j ∧
  ∀ id : ℕ, 
    (∃ n_triangles : Fin n → set unit_square_region, 
    (∀ t, n_triangles t ⊆ unit_square_region) ∧
    ∀ t₁ t₂, t₁ ≠ t₂ → n_triangles t₁ ∩ n_triangles t₂ = ∅ ∧ 
    (∀ t, ∃ area : ℝ, area (n_triangles t) = 1 / n))

-- The goal
theorem sixteen_ray_partitional_not_twelve_ray_partitional : 
  let P := { X : unit_square_region // n_ray_partitional 16 X ∧ ¬ n_ray_partitional 12 X } in
  ∃ s : Finset unit_square_region, 
  s.card = 16 ∧ (∀ X, X ∈ s → X ∈ P) ∧
  ∀ X, X ∈ P → X ∈ s :=
by
  sorry

end sixteen_ray_partitional_not_twelve_ray_partitional_l512_512172


namespace range_of_square_sum_is_half_to_one_l512_512066

noncomputable def range_of_square_sum (x y : ℝ) (hx : x ≥ 0) (hy : y ≥ 0) (hxy : x + y = 1) : set ℝ :=
{x | ∃ (t : ℝ), t = x^2 + y^2}

theorem range_of_square_sum_is_half_to_one (x y : ℝ) (hx : x ≥ 0) (hy : y ≥ 0) (hxy : x + y = 1) :
  range_of_square_sum x y hx hy hxy = set.Icc (1/2 : ℝ) 1 :=
begin
  sorry
end

end range_of_square_sum_is_half_to_one_l512_512066


namespace find_f_neg1_l512_512651

variable (f : ℝ → ℝ)

-- Conditions
axiom additivity : ∀ x y : ℝ, f(x + y) = f(x) + f(y)
axiom f2_is_4 : f(2) = 4

-- Proposition to prove
theorem find_f_neg1 : f(-1) = -2 :=
  sorry

end find_f_neg1_l512_512651


namespace range_of_f_smallest_positive_period_intervals_of_monotonic_increase_l512_512820

noncomputable def f (x : ℝ) : ℝ := 2 * sin x ^ 2 + 2 * sqrt 3 * sin x * cos x + 1

theorem range_of_f : set.Icc 0 4 = { y : ℝ | ∃ x : ℝ, f x = y } :=
sorry

theorem smallest_positive_period : ∀ (x : ℝ), f (x + π) = f x :=
sorry

theorem intervals_of_monotonic_increase : 
  ∀ k : ℤ, ∀ x : ℝ, 
  (-(π / 6) + (k : ℝ) * π ≤ x ∧ x ≤ (π / 3) + (k : ℝ) * π) →
  monotone_on f (set.Icc (-(π / 6) + (k : ℝ) * π) ((π / 3) + (k : ℝ) * π)) :=
sorry

end range_of_f_smallest_positive_period_intervals_of_monotonic_increase_l512_512820


namespace at_least_two_inequalities_hold_l512_512163

theorem at_least_two_inequalities_hold 
  (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) 
  (h : a + b + c ≥ a * b * c) : 
  (2 / a + 3 / b + 6 / c ≥ 6 ∨ 2 / b + 3 / c + 6 / a ≥ 6) ∨ (2 / b + 3 / c + 6 / a ≥ 6 ∨ 2 / c + 3 / a + 6 / b ≥ 6) ∨ (2 / c + 3 / a + 6 / b ≥ 6 ∨ 2 / a + 3 / b + 6 / c ≥ 6) := 
sorry

end at_least_two_inequalities_hold_l512_512163


namespace product_of_cosine_values_l512_512027

def arithmetic_seq (a₁ d : ℝ) (n : ℕ) : ℝ := a₁ + d * (n - 1)

theorem product_of_cosine_values (a₁ : ℝ) (h : ∃ (a b : ℝ), S = {a, b} ∧ S = {cos (arithmetic_seq a₁ (2 * π / 3) n) | n ∈ ℕ.succ}) :
  ∃ (a b : ℝ), a * b = -1 / 2 :=
begin
  obtain ⟨a, b, hS₁, hS₂⟩ := h,
  -- the proof will go here
  sorry
end

end product_of_cosine_values_l512_512027


namespace solve_fractional_equation_l512_512246

theorem solve_fractional_equation (x : ℝ) (h1 : x ≠ 1) (h2 : x ≠ -1) : 
    (3 / (x - 1) = 5 + 3 * x / (1 - x)) → x = 4 := 
by
  sorry

end solve_fractional_equation_l512_512246


namespace cos_arithmetic_seq_product_l512_512044

theorem cos_arithmetic_seq_product :
  ∃ (a b : ℝ), (∃ (a₁ : ℝ), ∀ n : ℕ, (n > 0) → ∃ m : ℕ, cos (a₁ + (2 * Real.pi / 3) * (n - 1)) = [a, b] ∧ (a * b = -1 / 2)) := 
sorry

end cos_arithmetic_seq_product_l512_512044


namespace union_of_sets_l512_512910

def log_condition (x : ℝ) : Prop := log x / log 2 > 0

def set_A : set ℝ := { x | log_condition x }
def set_B : set ℝ := { x | -1 < x ∧ x < 3 }

theorem union_of_sets : set_A ∪ set_B = { x : ℝ | x > -1 } := 
by sorry

end union_of_sets_l512_512910


namespace distance_between_pathway_lines_is_5_l512_512370

-- Define the conditions
def parallel_lines_distance (distance : ℤ) : Prop :=
  distance = 30

def pathway_length_between_lines (length : ℤ) : Prop :=
  length = 10

def pathway_line_length (length : ℤ) : Prop :=
  length = 60

-- Main proof problem
theorem distance_between_pathway_lines_is_5:
  ∀ (d : ℤ), parallel_lines_distance 30 → 
  pathway_length_between_lines 10 → 
  pathway_line_length 60 → 
  d = 5 := 
by
  sorry

end distance_between_pathway_lines_is_5_l512_512370


namespace same_function_f_g_l512_512330

def f : ℝ → ℝ := λ x, |x|
def g : ℝ → ℝ := λ x, real.sqrt (x ^ 2)

theorem same_function_f_g : ∀ x : ℝ, f x = g x :=
by
sorry

end same_function_f_g_l512_512330


namespace measure_angle_CAB_eq_60_l512_512671

-- Define the geometric setting
variables {A B C K C1 B1 B2 C2 : Type}
variables [Incenter A B C K] [Midpoint A B C1] [Midpoint A C B1]
variables [IntersectionLine C1 K AC B2] [IntersectionLine B1 K AB C2]
variables [EqualAreaTriangles AB2C2 ABC]

-- State the theorem
theorem measure_angle_CAB_eq_60 :
  MeasureAngle CAB = 60 :=
begin
  sorry
end

end measure_angle_CAB_eq_60_l512_512671


namespace mohamed_donation_l512_512589

/-- Leila and Mohamed toy donation problem --/
theorem mohamed_donation :
  ∃ n : ℕ, n = 3 ∧ 
    (∃ l : ℕ, l = 50 ∧
    ∃ m : ℕ, m = l + 7 ∧
    m = n * 19) :=
by
  -- Define Leila's total toy donation
  let l := 2 * 25
  -- Hypothesis that Mohamed donated 7 more toys than Leila
  let m := l + 7
  -- Calculate bags donated by Mohamed
  have h : ∃ n, m = n * 19, from
    ⟨3, by
      rw [m, l]
      norm_num⟩
  exact h

end mohamed_donation_l512_512589


namespace polar_bear_diameter_paradox_l512_512994

section

variable (R_I R_P : ℝ)
variable (floe_mass bear_mass : ℝ)
variable (correct_measurements : R_I = 4.25 ∧ R_P = 4.5 ∧ 0.5 * R_I ≈ (floe_mass / bear_mass) * (R_P - R_I))

theorem polar_bear_diameter_paradox :
  (R_P = 4.5) → (R_I = 4.25) → (correct_measurements ∧ floe_mass > bear_mass) →
  floe_mass / bear_mass ≈ (R_P - R_I) / R_I :=
by
  sorry

end

end polar_bear_diameter_paradox_l512_512994


namespace number_of_valid_strings_15_l512_512601

-- Define the sequences based on initial conditions and recurrence relations
def B_1 : ℕ → ℕ
| 0       := 0
| 1       := 1
| n + 2   := B_1 (n + 1) + B_2 (n + 1) + B_3 (n + 1)

def B_2 : ℕ → ℕ
| 0       := 0
| 1       := 0
| n + 2   := B_1 (n + 1)

def B_3 : ℕ → ℕ
| 0       := 0
| 1       := 0
| n + 2   := B_2 (n + 1)

-- Total number of valid strings of length n
def T (n : ℕ) : ℕ := B_1 n + B_2 n + B_3 n

-- Final proof statement in Lean 4
theorem number_of_valid_strings_15 : T 15 = 2584 :=
sorry

end number_of_valid_strings_15_l512_512601


namespace correct_equation_l512_512273

theorem correct_equation (x : ℝ) (hx : x > 80) : 
  353 / (x - 80) - 353 / x = 5 / 3 :=
sorry

end correct_equation_l512_512273


namespace product_of_cosine_values_l512_512039

noncomputable theory
open_locale classical

def arithmetic_sequence (a d : ℝ) (n : ℕ) : ℝ := a + (n - 1) * d

theorem product_of_cosine_values (a b : ℝ) 
    (h : ∃ a1 : ℝ, ∀ n : ℕ+, ∃ a b : ℝ, 
         S = {cos (arithmetic_sequence a1 (2*π/3) n) | n ∈ ℕ*} ∧
         S = {a, b}) : a * b = -1/2 :=
begin
  sorry
end

end product_of_cosine_values_l512_512039


namespace angle_BAN_is_60_degrees_l512_512468

theorem angle_BAN_is_60_degrees 
  {A B C K N : Type} 
  [IsoscelesTriangle A B C AC] 
  (h1 : Point K BC)
  (h2 : Point N BC)
  (h3 : Between K B N)
  (h4 : KN = AN)
  (h5 : ∠BAK = ∠NAC) 
  : ∠BAN = 60 := 
sorry

end angle_BAN_is_60_degrees_l512_512468


namespace commuter_distance_l512_512334

theorem commuter_distance (east_mov west_mov south_mov north_mov : ℝ) 
    (h_east : east_mov = 21) (h_west : west_mov = 9)
    (h_south : south_mov = 15) (h_north : north_mov = 3) :
    real.sqrt ((east_mov - west_mov) ^ 2 + (south_mov - north_mov) ^ 2) = 12 * real.sqrt 2 :=
by
  sorry

end commuter_distance_l512_512334


namespace abcde_sum_l512_512969

theorem abcde_sum : 
  ∀ (a b c d e : ℝ), 
  a + 1 = b + 2 → 
  b + 2 = c + 3 → 
  c + 3 = d + 4 → 
  d + 4 = e + 5 → 
  e + 5 = a + b + c + d + e + 10 → 
  a + b + c + d + e = -35 / 4 :=
sorry

end abcde_sum_l512_512969


namespace product_of_cosine_values_l512_512042

noncomputable theory
open_locale classical

def arithmetic_sequence (a d : ℝ) (n : ℕ) : ℝ := a + (n - 1) * d

theorem product_of_cosine_values (a b : ℝ) 
    (h : ∃ a1 : ℝ, ∀ n : ℕ+, ∃ a b : ℝ, 
         S = {cos (arithmetic_sequence a1 (2*π/3) n) | n ∈ ℕ*} ∧
         S = {a, b}) : a * b = -1/2 :=
begin
  sorry
end

end product_of_cosine_values_l512_512042


namespace find_angle_phi_l512_512162

noncomputable def triangle_angle_cot (a b c : ℝ) (Δ : ℝ) : ℝ :=
  if Δ ≠ 0 then
    let discriminant := 2 * (a^4 + b^4 + c^4 - b^2 * c^2 - c^2 * a^2 - a^2 * b^2) in
    let numerator := -(a^2 + b^2 + c^2) + discriminant.sqrt in
    let denominator := 12 * Δ in
    (numerator : ℝ) / denominator
  else
    0  -- this will never be used if Δ ≠ 0

theorem find_angle_phi (a b c Δ : ℝ) (h : a ≠ b ∧ b ≠ c ∧ c ≠ a ∧ Δ ≠ 0):
  ∃ φ : ℝ, φ = arccot_tri a b c Δ := sorry

end find_angle_phi_l512_512162


namespace divide_weights_into_equal_piles_l512_512569

theorem divide_weights_into_equal_piles : ∃ (A B : Finset ℕ), 
  (A ∪ B = (Finset.range 2009)) ∧ 
  (A ∩ B = ∅) ∧ 
  (∀ i, 0 ≤ i < 2008 → (abs (w (i+1) - w i) = 1)) ∧ 
  ((∑ i in A, w i) = (∑ i in B, w i)) :=
sorry

variables {w : Fin 2009 → ℕ}

-- Conditions
def weights_condition : Prop :=
  (∀ i, w i ≤ 1000) ∧
  (∀ i, 0 ≤ i < 2008 → abs (w (i+1) - w i) = 1) ∧
  (∑ i in Finset.range 2009, w i) % 2 = 0

variable hw : weights_condition

end divide_weights_into_equal_piles_l512_512569


namespace sum_first_15_arith_prog_l512_512737

theorem sum_first_15_arith_prog (a d : ℤ) 
  (h : (a + 3 * d) + (a + 11 * d) = 12) : 
  let S₁₅ := 15 / 2 * (2 * a + 14 * d)
  in S₁₅ = 90 := 
by {
  sorry
}

end sum_first_15_arith_prog_l512_512737


namespace phi_range_monotonically_increasing_f_l512_512896

open Real

theorem phi_range_monotonically_increasing_f (φ : ℝ) :
  (abs φ < π) →
  (∀ x ∈ (Ioo (π / 5) (5 * π / 8)), (-4 * cos (2 * x + φ) > 0)) →
  (φ > π / 10 ∧ φ < π / 4) :=
by
  intros hφ hfx
  sorry

end phi_range_monotonically_increasing_f_l512_512896


namespace complex_quadrant_l512_512176

theorem complex_quadrant :
  let i := Complex.i in
  let z := (2 * i) / (1 - i) in
  z = -1 + i ∧ -Real.Im(z) > 0 ∧ Real.Re(z) < 0 :=
by
  -- proof steps here
  sorry

end complex_quadrant_l512_512176


namespace split_fraction_l512_512267

theorem split_fraction (n d a b x y : ℤ) (h_d : d = a * b) (h_ad : a.gcd b = 1) (h_frac : (n:ℚ) / (d:ℚ) = 58 / 77) (h_eq : 11 * x + 7 * y = 58) : 
  (58:ℚ) / 77 = (4:ℚ) / 7 + (2:ℚ) / 11 :=
by
  sorry

end split_fraction_l512_512267


namespace asset_percentage_increase_l512_512136

constant scale_factor_year1 : ℝ := 1 + 20 / 100
constant scale_factor_year2 : ℝ := 1 + 30 / 100

theorem asset_percentage_increase :
  (scale_factor_year1 * scale_factor_year2 - 1) * 100 = 56 := by
  sorry

end asset_percentage_increase_l512_512136


namespace intersection_A_B_l512_512593

open Set

namespace Proof

def A := {1, 2, 3, 4, 5} : Set ℝ
def B := {x : ℝ | x^2 ∈ A}

theorem intersection_A_B :
  A ∩ B = {1, 2} :=
by
  sorry

end Proof

end intersection_A_B_l512_512593


namespace clowns_to_guppies_ratio_l512_512627

theorem clowns_to_guppies_ratio
  (C : ℕ)
  (tetra : ℕ)
  (guppies : ℕ)
  (total_animals : ℕ)
  (h1 : tetra = 4 * C)
  (h2 : guppies = 30)
  (h3 : total_animals = 330)
  (h4 : total_animals = tetra + C + guppies) :
  C / guppies = 2 :=
by
  sorry

end clowns_to_guppies_ratio_l512_512627


namespace part_I_part_II_l512_512075

variable (a : ℝ) (x : ℝ)
def f (x : ℝ) : ℝ := |x - 1| + |x + a|
def g (x : ℝ) : ℝ := f x - |3 + a|

theorem part_I : a = 3 → (∀ x, (x < -4 ∨ x > 2) ↔ |x - 1| + |x + 3| > 6) := by
  sorry

theorem part_II : (∀ a, (a ≥ -2) ↔ ∃ x, g x = 0) := by
  sorry

end part_I_part_II_l512_512075


namespace exists_k_for_360_gon_l512_512799

theorem exists_k_for_360_gon (O : Point) (A B : Point) (vertices : Finset Point) :
  vertices.card = 180 →
  (∀ (subset : Finset Point), subset ⊆ vertices → subset.card = 2 → 
    ∃ (k : ℕ), k = 120 ∧ ∃ (A B : Point), A ∈ subset ∧ B ∈ subset ∧ angle O A B = k) :=
begin
  sorry
end

end exists_k_for_360_gon_l512_512799


namespace exists_unique_subset_X_l512_512791

theorem exists_unique_subset_X :
  ∃ (X : Set ℤ), ∀ n : ℤ, ∃! (a b : ℤ), a ∈ X ∧ b ∈ X ∧ a + 2 * b = n :=
sorry

end exists_unique_subset_X_l512_512791


namespace min_value_expression_l512_512440

theorem min_value_expression :
  ∃ θ : ℝ, 0 < θ ∧ θ < π / 2 ∧ (∀ θ' : ℝ, 0 < θ' ∧ θ' < π / 2 → 
    (3 * Real.sin θ' + 4 / Real.cos θ' + 2 * Real.sqrt 3 * Real.tan θ') ≥ 9 * Real.sqrt 3) ∧ 
    (3 * Real.sin θ + 4 / Real.cos θ + 2 * Real.sqrt 3 * Real.tan θ = 9 * Real.sqrt 3) :=
by
  sorry

end min_value_expression_l512_512440


namespace range_of_a_l512_512837

def f : ℝ → ℝ := λ x, (x + 1) * Real.log (x + 1)
def P (a : ℝ) : Prop := ∀ x ≥ 0, f x ≥ a * x

theorem range_of_a :
  {a : ℝ | P a} = Set.Iic 1 :=
by
  sorry

end range_of_a_l512_512837


namespace problem_solution_l512_512598

noncomputable def is_valid_permutation (s : List Char) : Prop :=
  s.length = 16 ∧
  (∀ i, i < 4 → s.nthLe i sorry ≠ 'A') ∧
  (∀ i, 4 ≤ i ∧ i < 9 → s.nthLe i sorry ≠ 'B') ∧
  (∀ i, 9 ≤ i → s.nthLe i sorry ≠ 'C' ∧ s.nthLe i sorry ≠ 'D')

def count_valid_permutations : Nat :=
  List.permutations "AAAABBBBBCCCCDDD".toList.count is_valid_permutation

def answer : Nat :=
  count_valid_permutations % 1000

theorem problem_solution : answer = 62 :=
  sorry

end problem_solution_l512_512598


namespace parallelogram_angle_measures_l512_512571

theorem parallelogram_angle_measures
  (A B C D : ℝ)
  (parallelogram : A + B = 180 ∧ B = D ∧ A = C)
  (hA : A = 125) :
  B = 55 ∧ C = 125 ∧ D = 55 :=
by
  obtain ⟨hAB, hBD, hAC⟩ := parallelogram
  exact ⟨180 - A, A, 180 - A⟩ - sorry

end parallelogram_angle_measures_l512_512571


namespace cheapest_pie_cost_is_18_l512_512406

noncomputable def crust_cost : ℝ := 2 + 1 + 1.5
noncomputable def blueberry_container_cost : ℝ := 2.25
noncomputable def blueberry_containers_needed : ℕ := 3 * (16 / 8)
noncomputable def blueberry_filling_cost : ℝ := blueberry_containers_needed * blueberry_container_cost
noncomputable def cherry_filling_cost : ℝ := 14
noncomputable def cheapest_filling_cost : ℝ := min blueberry_filling_cost cherry_filling_cost
noncomputable def total_cheapest_pie_cost : ℝ := crust_cost + cheapest_filling_cost

theorem cheapest_pie_cost_is_18 : total_cheapest_pie_cost = 18 := by
  sorry

end cheapest_pie_cost_is_18_l512_512406


namespace pseudocode_output_l512_512805

theorem pseudocode_output : 
  let t : ℕ := 1
  let i : ℕ := 2
  let final_t := by
    let mut t := t
    let mut i := i
    while i <= 4 do
      t := t * i
      i := i + 1
    exact t
  final_t = 24 :=
by
  sorry

end pseudocode_output_l512_512805


namespace jackson_email_problem_l512_512583

variables (E_0 E_1 E_2 E_3 X : ℕ)

/-- Jackson's email deletion and receipt problem -/
theorem jackson_email_problem
  (h1 : E_1 = E_0 - 50 + 15)
  (h2 : E_2 = E_1 - X + 5)
  (h3 : E_3 = E_2 + 10)
  (h4 : E_3 = 30) :
  X = 50 :=
sorry

end jackson_email_problem_l512_512583


namespace quadrilateral_is_square_l512_512058

structure Point :=
  (x : ℝ)
  (y : ℝ)

def distance (p1 p2 : Point) : ℝ :=
  (p2.x - p1.x) ^ 2 + (p2.y - p1.y) ^ 2

def dot_product (v1 v2 : Point) : ℝ :=
  v1.x * v2.x + v1.y * v2.y

def vector (p1 p2 : Point) : Point :=
  ⟨p2.x - p1.x, p2.y - p1.y⟩

def is_square (A B C D : Point) : Prop :=
  let AB := vector A B;
  let BC := vector B C;
  let CD := vector C D;
  let DA := vector D A;
  distance A B = distance B C ∧
  distance B C = distance C D ∧
  distance C D = distance D A ∧
  dot_product AB BC = 0 ∧
  dot_product BC CD = 0 ∧
  dot_product CD DA = 0 ∧
  dot_product DA AB = 0

theorem quadrilateral_is_square :
  let A := ⟨-1, 3⟩ in
  let B := ⟨1, -2⟩ in
  let C := ⟨6, 0⟩ in
  let D := ⟨4, 5⟩ in
  is_square A B C D :=
  sorry

end quadrilateral_is_square_l512_512058


namespace centroid_moves_on_circle_l512_512873

-- Define Triangle ABC with some properties
structure Triangle :=
(A B C : Point)
(is_isosceles : A ≠ B ∧ A ≠ C ∧ dist A B = dist A C)
(vertex_fixed : ∀ (P : Point), dist A P = dist A B)

-- Define the median and centroid properties
def midpoint (A B : Point) : Point := Point.mk ((A.x + B.x) / 2) ((A.y + B.y) / 2)

def median (A B C : Point) : Segment :=
Segment.mk A (midpoint B C)

def centroid (A B C : Point) : Point := 
let M := midpoint B C in 
Point.mk (2 / 3 * A.x + 1 / 3 * M.x) (2 / 3 * A.y + 1 / 3 * M.y)

-- Mathematical statement for the proof problem
theorem centroid_moves_on_circle (A B C : Point) (t : Triangle) 
(h_isosceles : t.is_isosceles)
(h_vertex_fixed : t.vertex_fixed)
:
  ∃ (O : Point), ∀ (BC : Segment), centroid A B C = Point.mk (O.x + r * cos (θ)) (O.y + r * sin (θ)) :=
sorry

end centroid_moves_on_circle_l512_512873


namespace graph_passes_through_point_l512_512105

theorem graph_passes_through_point (a : ℝ) (x y : ℝ) (h : a < 0) : (1 - a)^0 - 1 = -1 :=
by
  sorry

end graph_passes_through_point_l512_512105


namespace olivia_race_time_l512_512428

theorem olivia_race_time (total_time : ℕ) (time_difference : ℕ) (olivia_time : ℕ)
  (h1 : total_time = 112) (h2 : time_difference = 4) (h3 : olivia_time + (olivia_time - time_difference) = total_time) :
  olivia_time = 58 :=
by
  sorry

end olivia_race_time_l512_512428


namespace luke_stickers_l512_512984

theorem luke_stickers : 
  let initial := 20 in
  let bought := 12 in
  let birthday := 20 in
  let given_to_sister := 5 in
  let used_for_card := 8 in
  initial + bought + birthday - given_to_sister - used_for_card = 39 := 
  by
  sorry

end luke_stickers_l512_512984


namespace find_unit_vector_l512_512600

noncomputable def unit_vector : Vector ℝ 3 :=
  ⟨0, -2 / Real.sqrt 5, 1 / Real.sqrt 5⟩

theorem find_unit_vector (b : Vector ℝ 3) (v : Vector ℝ 3) :
  b = ⟨0, 1, 2⟩ →
  (b ⬝ v = 0) →
  ∥v∥ = 1 →
  v = unit_vector := by
  intros h_b h_perp h_unit
  rw [←h_b] at h_perp
  sorry

end find_unit_vector_l512_512600


namespace relationship_y1_y2_y3_l512_512472

variable (y1 y2 y3 b : ℝ)
variable (h1 : y1 = 3 * (-3) - b)
variable (h2 : y2 = 3 * 1 - b)
variable (h3 : y3 = 3 * (-1) - b)

theorem relationship_y1_y2_y3 : y1 < y3 ∧ y3 < y2 := by
  sorry

end relationship_y1_y2_y3_l512_512472


namespace max_sqrt_abs_cubed_sum_l512_512920

theorem max_sqrt_abs_cubed_sum (x y : ℝ) (h : x^2 + y^2 = 1) : 
  sqrt (|x|^3 + |y|^3) ≤ (sqrt (2 * sqrt 2 + 1)) / sqrt 3 := 
sorry

end max_sqrt_abs_cubed_sum_l512_512920


namespace angle_bisector_CE_l512_512314

-- Define the conditions using Lean structures
variables {C A B E : Point}
variables (circ1 circ2 : Circle)
variables (tangent_point_C : circ1.TangentPoint C circ2)
variables (chord_AB : segment A B)
variables (tangent_point_E : chord_AB.TangentPoint circ2 E)

-- Define the theorem statement
theorem angle_bisector_CE : is_angle_bisector (angle A C B) (line C E) :=
sorry

end angle_bisector_CE_l512_512314


namespace sum_triangle_areas_l512_512187

-- Definition of the area of the triangle formed by the line kx + (k+1)y - 1 = 0 with the coordinate axes
def triangle_area (k : ℤ) : ℝ := 1 / (2 * |k| * |k + 1|)

-- Sum of areas from k = 1 to k = 2008
def sum_of_areas (n : ℕ) : ℝ := ∑ k in Finset.range n, triangle_area (k + 1)

-- Problem statement: the sum of the areas from k = 1 to k = 2008 equals 1004/2009
theorem sum_triangle_areas : sum_of_areas 2008 = 1004 / 2009 :=
by sorry

end sum_triangle_areas_l512_512187


namespace curve_is_line_l512_512811

theorem curve_is_line (r : ℝ) (θ : ℝ) : θ = real.pi / 4 → 
  ∃ (l : ℝ → ℝ), ∀ (r : ℝ), (r, θ) = (r, real.pi / 4) → (r * real.cos (real.pi / 4), r * real.sin (real.pi / 4)) = (r, l r) := 
by {
  intro h,
  use id, -- The identity function represents the line y = x for the angle π/4.
  intros r hr,
  rw [hr],
  simp [real.sin_pi_div_four, real.cos_pi_div_four],
  sorry
}

end curve_is_line_l512_512811


namespace smallest_positive_integer_m_l512_512966

-- Define the set S
def S := {z : ℂ | ∃ x y : ℝ, z = x + y * complex.I ∧ 1/2 ≤ x ∧ x ≤ real.sqrt 2 / 2}

-- Define the proof statement
theorem smallest_positive_integer_m :
  ∃ m : ℕ, m = 13 ∧ ∀ n : ℕ, n ≥ m → ∃ z : ℂ, z ∈ S ∧ z ^ n = 1 := sorry

end smallest_positive_integer_m_l512_512966


namespace Helen_raisins_l512_512089

/-- Given that Helen baked 19 chocolate chip cookies yesterday, baked some raisin cookies and 237 chocolate chip cookies this morning,
    and baked 25 more chocolate chip cookies than raisin cookies in total,
    prove that the number of raisin cookies (R) she baked is 231. -/
theorem Helen_raisins (R : ℕ) (h1 : 25 + R = 256) : R = 231 :=
by
  sorry

end Helen_raisins_l512_512089


namespace time_difference_l512_512336

theorem time_difference :
  (let
    blocks := 18,
    walk_time_per_block := 1, -- in minutes
    bike_time_per_block := 20 / 60 -- in minutes
  in
    (blocks * walk_time_per_block - blocks * bike_time_per_block) = 12) :=
by
  sorry

end time_difference_l512_512336


namespace problem_l512_512576

-- Defining the numbers and their possible values
def numbers := {1, 2, 4, 5, 6, 8}

-- Conditions
def condition1 (x y : ℕ) : Prop := x - y = 3 ∨ y - x = 3
def condition2 (x y : ℕ) : Prop := x - y ∈ numbers ∧ y - x ∈ numbers

-- Assignments based on conditions
def A := 6
def B := 5
def C := 1
def D := 2
def E := 5
def F := 4

-- Proving that A + C = 14
theorem problem : A + C = 14 :=
by {
  -- assign values
  let A := 6,
  let C := 1,
  
  -- prove
  calc
  A + C = 6 + 1 : by rfl
  ... = 14 : by sorry
}

end problem_l512_512576


namespace value_of_b7b9_l512_512134

-- Define arithmetic sequence and geometric sequence with given conditions
variable (a : ℕ → ℝ) (b : ℕ → ℝ)

-- The given conditions in the problem
def a_seq_arithmetic (a : ℕ → ℝ) := ∀ n, a n = a 1 + (n - 1) • (a 2 - a 1)
def b_seq_geometric (b : ℕ → ℝ) := ∃ r : ℝ, ∀ n, b (n + 1) = r * b n
def given_condition (a : ℕ → ℝ) := 2 * a 5 - (a 8)^2 + 2 * a 11 = 0
def b8_eq_a8 (a b : ℕ → ℝ) := b 8 = a 8

-- The statement to prove
theorem value_of_b7b9 : a_seq_arithmetic a → b_seq_geometric b → given_condition a → b8_eq_a8 a b → b 7 * b 9 = 4 := by
  intros a_arith b_geom cond b8a8
  sorry

end value_of_b7b9_l512_512134


namespace function_characterization_l512_512429

theorem function_characterization :
  (∃ f : ℕ → ℕ, (∀ m n : ℕ, ∃ k : ℕ, f(n) + 2 * m * n + f(m) = k * k) ∧
    (∀ n, f(n) = (n + 2 * 0)^2 - 2 * 0^2)
  :=
begin
  sorry
end

end function_characterization_l512_512429


namespace reflection_correct_l512_512821

-- Define the initial vector and the vector we are reflecting over
def v : ℝ × ℝ := (2, 5)
def u : ℝ × ℝ := (-1, 2)

-- Define the projection of v onto u
def proj (v u : ℝ × ℝ) : ℝ × ℝ :=
  let num := (v.1 * u.1 + v.2 * u.2)
  let denom := (u.1 * u.1 + u.2 * u.2)
  (num / denom * u.1, num / denom * u.2)

-- Define the reflection of v over u
def reflection (v u : ℝ × ℝ) : ℝ × ℝ :=
  let p := proj v u
  (2 * p.1 - v.1, 2 * p.2 - v.2)

-- State the theorem
theorem reflection_correct : reflection v u = (-26/5, 27/5) :=
  sorry

end reflection_correct_l512_512821


namespace knight_tour_proof_l512_512947

-- Define the properties of the chessboard and the knight's movement
def is_black_square (pos : Nat × Nat) : Bool :=
  (pos.1 + pos.2) % 2 = 1

def knight_tour_impossible : Prop :=
  let start_pos := (0, 0)
  let end_pos := (7, 7)
  (is_black_square start_pos) ∧ 
  (is_black_square end_pos) ∧ 
  (¬ ∃ p : (Nat × Nat) → Bool, 
      (∀ i j : Nat, i < 8 ∧ j < 8 → p (i, j) ∨ ¬ p (i, j)) ∧ 
      (∀ k : Nat, k < 64 → exists p' : (Nat × Nat), 
          p' (k, k) = p (k, k))) 

theorem knight_tour_proof : knight_tour_impossible := sorry

end knight_tour_proof_l512_512947


namespace solve_equation_l512_512260

theorem solve_equation : ∀ (x : ℝ), x ≠ 1 → (3 / (x - 1) = 5 + 3 * x / (1 - x)) → x = 4 :=
by
  intros x hx heq
  -- sorry to skip the proof
  sorry

end solve_equation_l512_512260


namespace sum_of_consecutive_integers_is_33_l512_512298

theorem sum_of_consecutive_integers_is_33 :
  ∃ (x : ℕ), x * (x + 1) = 272 ∧ x + (x + 1) = 33 :=
by
  sorry

end sum_of_consecutive_integers_is_33_l512_512298


namespace find_other_number_l512_512658

theorem find_other_number (a b : ℕ) (h1 : (a + b) / 2 = 7) (h2 : a = 5) : b = 9 :=
by
  sorry

end find_other_number_l512_512658


namespace correct_statement_is_C_l512_512725

theorem correct_statement_is_C :
  (let prob_lottery_win := 0.05 in
   let tickets_bought := 20 in
   let prob_not_win := (1 - prob_lottery_win) ^ tickets_bought in
   let statement_A := ¬(prob_not_win > 0) in
   
   let trials := 5 in
   let successful_trials := 3 in
   let empirical_prob := successful_trials / trials in
   let true_prob_dart := sorry in 
   let statement_B := empirical_prob ≠ true_prob_dart in

   let prob_heads := 1 / 2 in
   let statement_C := prob_heads = 1 / 2 in
   
   let people := 400 in
   let statement_D := false in -- Assumption that it's impossible for two people to share a birthday is false
   
   (statement_A → false) ∧
   (statement_B → false) ∧
   (statement_C → true) ∧
   (statement_D → false)) :=
by
  intros prob_lottery_win tickets_bought prob_not_win statement_A
         trials successful_trials empirical_prob true_prob_dart statement_B
         prob_heads statement_C people statement_D;
  split;
  { intro h, sorry },
  { intro h, sorry },
  { intro h, exact h },
  { intro h, sorry }

end correct_statement_is_C_l512_512725


namespace triangle_is_equilateral_l512_512104

-- Given the conditions
variables {A B C : Angle}
variables {a b c : ℝ}
variables (h1 : (a + b + c) * (b + c - a) = 3 * b * c)
variables (h2 : sin A = 2 * sin B * cos C)

-- Proof statement
theorem triangle_is_equilateral (h1 : (a + b + c) * (b + c - a) = 3 * b * c)
                                 (h2 : sin A = 2 * sin B * cos C) : 
                                 ∃ ABC : Triangle ℝ, ABC.is_equilateral :=
sorry

end triangle_is_equilateral_l512_512104


namespace solve_equation_l512_512241

theorem solve_equation (x : ℝ) (h : x ≠ 1 ∧ x ≠ 1) : (3 / (x - 1) = 5 + 3 * x / (1 - x)) ↔ x = 4 := by
  sorry

end solve_equation_l512_512241


namespace tomatoes_harvest_ratio_l512_512271

noncomputable def tomatoes_ratio (w t f : ℕ) (g r : ℕ) : ℕ × ℕ :=
  if (w = 400) ∧ ((w + t + f) = 2000) ∧ ((g = 700) ∧ (r = 700) ∧ ((g + r) = f)) ∧ (t = 200) then 
    (2, 1)
  else 
    sorry

theorem tomatoes_harvest_ratio : 
  ∀ (w t f : ℕ) (g r : ℕ), 
  (w = 400) → 
  (w + t + f = 2000) → 
  (g = 700) → 
  (r = 700) → 
  (g + r = f) → 
  (t = 200) →
  tomatoes_ratio w t f g r = (2, 1) :=
by {
  -- insert proof here
  sorry
}

end tomatoes_harvest_ratio_l512_512271


namespace a10_has_more_than_1000_nines_l512_512683

def sequence : ℕ → ℕ
| 0     := 9
| (n+1) := 3 * (sequence n)^4 + 4 * (sequence n)^3

theorem a10_has_more_than_1000_nines :
  let a_10 := sequence 10 in 
  (string.hash a_10.repr).to_list.count '9' > 1000 := 
sorry

end a10_has_more_than_1000_nines_l512_512683


namespace frac_square_between_half_and_one_l512_512689

theorem frac_square_between_half_and_one :
  let fraction := (11 : ℝ) / 12
  let expr := fraction^2
  (1 / 2) < expr ∧ expr < 1 :=
by
  let fraction := (11 : ℝ) / 12
  let expr := fraction^2
  have h1 : (1 / 2) < expr := sorry
  have h2 : expr < 1 := sorry
  exact ⟨h1, h2⟩

end frac_square_between_half_and_one_l512_512689


namespace points_set_cardinality_l512_512291

-- Define the set of points under the given conditions
def points_set : set (ℝ × ℝ) :=
  { p | ∃ (x y : ℝ), p = (x, y) ∧ x > 0 ∧ y > 0 ∧ real.log10 (x^3 + (1 / 3) * y^3 + (1 / 9)) = real.log10 x + real.log10 y }

-- Statement to prove that the number of elements in the set is 1.
theorem points_set_cardinality : set.card points_set = 1 :=
by
  sorry

end points_set_cardinality_l512_512291


namespace nearest_lateral_surface_area_l512_512687

noncomputable def pi := Real.pi

def radius : ℝ := 4
def slant_height : ℝ := 5

def lateral_surface_area (r l : ℝ) : ℝ := pi * r * l

theorem nearest_lateral_surface_area :
  abs (lateral_surface_area radius slant_height - 62.8) < 
  min (abs (lateral_surface_area radius slant_height - 24.0))
  (min (abs (lateral_surface_area radius slant_height - 74.2))
   (abs (lateral_surface_area radius slant_height - 113.0))) :=
sorry

end nearest_lateral_surface_area_l512_512687


namespace sum_ineq_l512_512067

theorem sum_ineq (n : ℕ) (a b : Fin n → ℝ)
  (h₀ : ∀ i, 0 < a i)
  (h₁ : ∀ i, 0 < b i)
  (h₂ : ∑ i, a i = ∑ i, b i) :
  ∑ i, a i ^ 2 / (a i + b i) ≥ 1 / 2 * ∑ i, a i := by
  sorry

end sum_ineq_l512_512067


namespace pavan_total_distance_l512_512199

theorem pavan_total_distance:
  ∀ (D : ℝ),
  (∃ Time1 Time2,
    Time1 = (D / 2) / 30 ∧
    Time2 = (D / 2) / 25 ∧
    Time1 + Time2 = 11)
  → D = 150 :=
by
  intros D h
  sorry

end pavan_total_distance_l512_512199


namespace rightmost_three_digits_of_5_pow_1994_l512_512315

theorem rightmost_three_digits_of_5_pow_1994 : (5 ^ 1994) % 1000 = 625 :=
by
  sorry

end rightmost_three_digits_of_5_pow_1994_l512_512315


namespace sin_alpha_value_l512_512112

theorem sin_alpha_value (α : ℝ) (h₁ : 0 < α ∧ α < π / 2)
  (h₂ : sin (α + π / 2) = 3 / 5) : sin α = 4 / 5 :=
sorry

end sin_alpha_value_l512_512112


namespace solve_equation_l512_512263

theorem solve_equation : ∀ (x : ℝ), x ≠ 1 → (3 / (x - 1) = 5 + 3 * x / (1 - x)) → x = 4 :=
by
  intros x hx heq
  -- sorry to skip the proof
  sorry

end solve_equation_l512_512263


namespace wire_triangle_arrangement_l512_512743

theorem wire_triangle_arrangement (k p : Nat) :
  (∀ (t1 t2 : Fin k), t1 ≠ t2 → ∃ v : Fin (3 * k), v ∈ t1 ∧ v ∈ t2 ∧ ∀ v', v' ∈ t1 ∧ v' ∈ t2 → v' = v) →
  (∀ v : Fin (3 * k), ∃ t : Fin k, v ∈ t ∧ t.card = p) →
  (k = 1 ∧ p = 1) ∨ (k = 4 ∧ p = 2) ∨ (k = 7 ∧ p = 3) :=
sorry

end wire_triangle_arrangement_l512_512743


namespace prob_two_correct_A_B_C_prob_dist_and_expectation_l512_512272

noncomputable def P_A := 2/3
noncomputable def P_B := 1/3
noncomputable def P_C := 1/3

-- Question 1: Prove the probability of exactly two correct answers is 1/3
theorem prob_two_correct_A_B_C : 
  let p_a := P_A 
  let p_b := P_B 
  let p_c := P_C 
  p_a * p_b * (1 - p_c) + p_a * (1 - p_b) * p_c + (1 - p_a) * p_b * p_c = 1/3 := by
  sorry

-- Question 2: Prove the probability distribution and expectation
theorem prob_dist_and_expectation :
  let p_a := P_A 
  let p_b := P_B 
  let p_c := P_C 
  let P_X := fun x =>  
    match x with
    | 0 => (1 - p_a) * (1 - p_b) * (1 - p_c)
    | 1 => p_a * (1 - p_b) * (1 - p_c)
    | 2 => (1 - p_a) * p_b * (1 - p_c) + (1 - p_a) * (1 - p_b) * p_c
    | 3 => p_a * p_b * (1 - p_c) + p_a * (1 - p_b) * p_c
    | 4 => (1 - p_a) * p_b * p_c
    | 5 => p_a * p_b * p_c
    | _ => 0
  let E_X := 0 * (1 - p_a) * (1 - p_b) * (1 - p_c) + 
            1 * p_a * (1 - p_b) * (1 - p_c) + 
            2 * ((1 - p_a) * p_b * (1 - p_c) + (1 - p_a) * (1 - p_b) * p_c) + 
            3 * (p_a * p_b * (1 - p_c) + p_a * (1 - p_b) * p_c) + 
            4 * (1 - p_a) * p_b * p_c + 
            5 * p_a * p_b * p_c
  (P_X 0 = 4/27) ∧ (P_X 1 = 8/27) ∧ (P_X 2 = 4/27) ∧ (P_X 3 = 8/27) ∧ (P_X 4 = 1/27) ∧ (P_X 5 = 2/27) ∧ (E_X = 2) := by
  sorry

end prob_two_correct_A_B_C_prob_dist_and_expectation_l512_512272


namespace b_n_general_term_T_n_upper_bound_l512_512068

noncomputable def a_n : ℕ → ℝ :=
  λ n, 3 * n - 1

def b_n (n : ℕ) : ℝ := 1/3^n

def c_n (n : ℕ) : ℝ := (3 * n - 1) * b_n n

def S_n (n : ℕ) : ℝ := ∑ k in Finset.range n, b_n (k + 1)

def T_n (n : ℕ) : ℝ := ∑ k in Finset.range n, c_n (k + 1)

theorem b_n_general_term (n : ℕ) : b_n n = (1 : ℝ) / (3^n) :=
  by sorry

theorem T_n_upper_bound (n : ℕ) : T_n n < 7 / 4 :=
  by sorry

end b_n_general_term_T_n_upper_bound_l512_512068


namespace luke_stickers_l512_512985

theorem luke_stickers : 
  let initial := 20 in
  let bought := 12 in
  let birthday := 20 in
  let given_to_sister := 5 in
  let used_for_card := 8 in
  initial + bought + birthday - given_to_sister - used_for_card = 39 := 
  by
  sorry

end luke_stickers_l512_512985


namespace at_least_one_blue_card_drawn_sum_equals_10_different_arrangements_l512_512349

-- Definitions for card labels
inductive Color
| red : Color
| blue : Color

structure Card where
  number : Nat
  color : Color

def red_card_1 : Card := ⟨1, Color.red⟩
def red_card_2 : Card := ⟨2, Color.red⟩
def red_card_3 : Card := ⟨3, Color.red⟩
def red_card_4 : Card := ⟨4, Color.red⟩
def blue_card_1 : Card := ⟨1, Color.blue⟩
def blue_card_2 : Card := ⟨2, Color.blue⟩

def all_cards : List Card :=
  [red_card_1, red_card_2, red_card_3, red_card_4, blue_card_1, blue_card_2]

-- Theorem 1: At least one blue card is drawn
theorem at_least_one_blue_card_drawn :
  ∑ (c1 c2 c3 c4 : Card) ∈ all_cards.choose 4, 
    (c1.color = Color.blue ∨ c2.color = Color.blue ∨ c3.color = Color.blue ∨ c4.color = Color.blue) =
    14 :=
by sorry

-- Theorem 2: Arrangements where sum of card numbers equals 10
theorem sum_equals_10_different_arrangements :
  ∑ (c1 c2 c3 c4 : Card) ∈ all_cards.choose 4, 
    (c1.number + c2.number + c3.number + c4.number = 10) ∧ 
    is_permutation (List.permutations [c1, c2, c3, c4]) = 
    48 :=
by sorry

end at_least_one_blue_card_drawn_sum_equals_10_different_arrangements_l512_512349


namespace cheapest_pie_cost_l512_512413

def crust_cost : ℝ := 2 + 1 + 1.5

def blueberry_pie_cost : ℝ :=
  let blueberries_needed := 3 * 16
  let containers_required := blueberries_needed / 8
  let blueberries_cost := containers_required * 2.25
  crust_cost + blueberries_cost

def cherry_pie_cost : ℝ := crust_cost + 14

theorem cheapest_pie_cost : blueberry_pie_cost = 18 :=
by sorry

end cheapest_pie_cost_l512_512413


namespace amanda_pay_if_not_finished_l512_512156

-- Define Amanda's hourly rate and daily work hours.
def amanda_hourly_rate : ℝ := 50
def amanda_daily_hours : ℝ := 10

-- Define the percentage of pay Jose will withhold.
def withholding_percentage : ℝ := 0.20

-- Define Amanda's total pay if she finishes the sales report.
def amanda_total_pay : ℝ := amanda_hourly_rate * amanda_daily_hours

-- Define the amount withheld if she does not finish the sales report.
def withheld_amount : ℝ := amanda_total_pay * withholding_percentage

-- Define the amount Amanda will receive if she does not finish the sales report.
def amanda_final_pay_not_finished : ℝ := amanda_total_pay - withheld_amount

-- The theorem to prove:
theorem amanda_pay_if_not_finished : amanda_final_pay_not_finished = 400 := by
  sorry

end amanda_pay_if_not_finished_l512_512156


namespace solve_equation_l512_512234

theorem solve_equation : (x : ℝ) (hx : x ≠ 1) (h : 3 / (x - 1) = 5 + 3 * x / (1 - x)) : x = 4 :=
sorry

end solve_equation_l512_512234


namespace product_of_complex_numbers_l512_512117

def z1 : ℂ := 2 + I
def z2 : ℂ := 1 - I

theorem product_of_complex_numbers : z1 * z2 = 3 - I := 
by 
  sorry

end product_of_complex_numbers_l512_512117


namespace geometry_problem_l512_512839

noncomputable def problem_statement (A B C K L : Point) (triangle_ABC : Triangle A B C)
  (right_angle_C : Angle A C B = 90) (bisector_BK : LineOfBisector B K A)
  (circumcircle_AKB : Circumcircle AKB) : Prop :=
  IsRightTriangle triangle_ABC ∧
  Circumscribed_circle A K B L ∧
  OnSameLine K A → 
  PointOnCircle L circumcircle_AKB →
  CB + CL = AB

theorem geometry_problem {A B C K L : Point} 
  (triangle_ABC : Triangle A B C)
  (right_angle_C : Angle A C B = 90)
  (bisector_BK : LineOfBisector B K A)
  (circumcircle_AKB : Circumcircle AKB)
  (L_on_circumcircle : PointOnCircle L circumcircle_AKB) 
  : CB + CL = AB := 
sorry

end geometry_problem_l512_512839


namespace area_ratio_of_multiplied_square_l512_512736

theorem area_ratio_of_multiplied_square (s : ℝ) : 
  let A_original := s ^ 2 in
  let A_resultant := (s * Real.sqrt 5) ^ 2 in
  A_original / A_resultant = 1 / 5 :=
by
  sorry

end area_ratio_of_multiplied_square_l512_512736


namespace a_2009_eq_1_a_2014_eq_0_l512_512842

section
variable (a : ℕ → ℕ)
variable (n : ℕ)

-- Condition 1: a_{4n-3} = 1
axiom cond1 : ∀ n : ℕ, a (4 * n - 3) = 1

-- Condition 2: a_{4n-1} = 0
axiom cond2 : ∀ n : ℕ, a (4 * n - 1) = 0

-- Condition 3: a_{2n} = a_n
axiom cond3 : ∀ n : ℕ, a (2 * n) = a n

-- Theorem: a_{2009} = 1
theorem a_2009_eq_1 : a 2009 = 1 := by
  sorry

-- Theorem: a_{2014} = 0
theorem a_2014_eq_0 : a 2014 = 0 := by
  sorry

end

end a_2009_eq_1_a_2014_eq_0_l512_512842


namespace intersection_A_B_l512_512594

open Set

namespace Proof

def A := {1, 2, 3, 4, 5} : Set ℝ
def B := {x : ℝ | x^2 ∈ A}

theorem intersection_A_B :
  A ∩ B = {1, 2} :=
by
  sorry

end Proof

end intersection_A_B_l512_512594


namespace polar_bear_paradox_l512_512999

theorem polar_bear_paradox
  (d_earth : ℝ) (d_floe : ℝ)
  (h1 : d_earth = 8.5) (h2 : d_floe = 9)
  (h3 : ∀ (measurements_correct : Prop), measurements_correct) :
  ∃ (mass_ratio : ℝ), mass_ratio > 1 ∧ 
                       abs (d_floe / 2 - d_earth / 2) = 0.25 :=
by
  use 10 -- Assume an example mass ratio of 10.
  split
  . exact zero_lt_ten
  . calc
    abs (4.5 - 4.25) = abs (0.25) : by norm_num
                ... = 0.25 : abs_of_nonneg (by norm_num)

end polar_bear_paradox_l512_512999


namespace mass_of_actual_car_l512_512745

-- Define the conditions and the main theorem to prove the mass of the actual car.
variable (mass_model : ℕ) (scale_ratio : ℕ)

theorem mass_of_actual_car (h_mass_model : mass_model = 2) (h_scale_ratio : scale_ratio = 8) : 
  let volume_scale := scale_ratio ^ 3 in
  let actual_mass := volume_scale * mass_model in
  actual_mass = 1024 := 
by
  sorry

end mass_of_actual_car_l512_512745


namespace gcd_5800_14025_l512_512435

theorem gcd_5800_14025 : Int.gcd 5800 14025 = 25 := by
  sorry

end gcd_5800_14025_l512_512435


namespace length_GH_eq_17_l512_512639

variable (A B C D E F G : Point)
variable (AD BE CF GH : ℝ)
variable (PT : Line)

-- Given conditions
axiom AD_equal_15 : AD = 15
axiom BE_equal_9 : BE = 9
axiom CF_equal_27 : CF = 27
axiom triangle_ABC_right_at_C : is_right_triangle ABC C

-- The problem statement to prove
theorem length_GH_eq_17 
  (h1 : AD = 15) 
  (h2 : BE = 9)
  (h3 : CF = 27)
  (h4 : is_right_triangle ABC C) :
  (∃ G, is_centroid G ABC ∧ perpendicular GH PT ∧ length GH = 17) :=
sorry

end length_GH_eq_17_l512_512639


namespace inverse_tangent_line_l512_512922

theorem inverse_tangent_line
  (f : ℝ → ℝ)
  (hf₁ : ∃ g : ℝ → ℝ, ∀ x, g (f x) = x ∧ f (g x) = x) 
  (hf₂ : ∀ x, deriv f x ≠ 0)
  (h_tangent : ∀ x₀, (2 * x₀ - f x₀ + 3) = 0) :
  ∀ x₀, (x₀ - 2 * f x₀ - 3) = 0 :=
by
  sorry

end inverse_tangent_line_l512_512922


namespace planting_methods_count_l512_512827

theorem planting_methods_count:
  let varieties := {1, 2, 3, 4, 5, 6},
      plots := {A, B, C, D},
      no1 := 1,
      no2 := 2,
      cannot_be_planted_on := λ v, v = no1 ∨ v = no2 → ∀ p, p = A,
      count := 
        (if (¬no1 ∈ varieties ∧ ¬no2 ∈ varieties) then
          (4.choose 4) * (4.perm 4)
        else 
          if (no1 ∈ varieties ∧ no2 ∈ varieties) then 
            (4.choose 2) * (2.choose 1) * (3.perm 3)
          else
            2 * (4.choose 3) * (3.choose 1) * (3.perm 3)),

  count = 240 :=
by
  sorry

end planting_methods_count_l512_512827


namespace right_triangle_hypotenuse_l512_512840

theorem right_triangle_hypotenuse (a b : ℕ) (h₁ : a = 3) (h₂ : b = 5) : 
  ∃ h : ℝ, h = Real.sqrt (a^2 + b^2) ∧ h = Real.sqrt 34 := 
by
  sorry

end right_triangle_hypotenuse_l512_512840


namespace no_coloring_scheme_l512_512561

theorem no_coloring_scheme (c : ℕ → ℕ → bool) :
    let n := 2010 in
    (∀ i j, c i j = tt ∨ c i j = ff) ∧
    (∀ i, ∑ j in finset.range n, if c i j = tt then 1 else 0 = n / 2) ∧
    (∀ j, ∑ i in finset.range n, if c i j = tt then 1 else 0 = n / 2) ∧
    (∀ i j, c i j ≠ c (n - 1 - i) (n - 1 - j))
    → false := by
  sorry

end no_coloring_scheme_l512_512561


namespace truncatable_coin_sequence_l512_512125

theorem truncatable_coin_sequence (n : ℕ → ℕ) (h_increasing : ∀ k l, k < l → n k < n l) :
  ∃ N, ∀ m, ∃ a b, (
    (∀ i, a i < N → 0 ≤ i) ∧ 
    sum (λ i, a i • n i) = sum (λ j, b j • n j) ) :=
sorry

end truncatable_coin_sequence_l512_512125


namespace largest_divisor_540_180_under_60_l512_512716

theorem largest_divisor_540_180_under_60 : 
  ∃ d, d ∣ 540 ∧ d ∣ 180 ∧ d < 60 ∧ (∀ e, e ∣ 540 ∧ e ∣ 180 ∧ e < 60 → e ≤ d) ∧ d = 45 :=
begin
  sorry
end

end largest_divisor_540_180_under_60_l512_512716


namespace original_price_l512_512679

theorem original_price (q r : ℝ) :
  let x := 2 / (1 + (q - r) / 100 - (q * r) / 10000) in
  x = 20000 / (10000 + 100 * (q - r) - q * r) := 
by 
  sorry

end original_price_l512_512679


namespace solution_of_system_l512_512646

def system_of_equations (x y : ℝ) : Prop :=
  (x^2 * y - x * y^2 - 3 * x + 3 * y + 1 = 0) ∧ 
  (x^3 * y - x * y^3 - 3 * x^2 + 3 * y^2 + 3 = 0)

theorem solution_of_system : system_of_equations 2 1 :=
by
  unfold system_of_equations
  split
  simp
  sorry

end solution_of_system_l512_512646


namespace part1_b1_b2_general_formula_part2_sum_first_20_terms_l512_512846

/-- Define the sequence a_n recursively --/
def a : ℕ → ℕ
| 0 := 1
| (n+1) := if n % 2 = 0 then a n + 1 else a n + 2

/-- Define the sequence b_n such that b_n = a_{2n} --/
def b (n : ℕ) : ℕ := a (2 * (n + 1))

/-- State the properties to be proved --/
theorem part1_b1_b2_general_formula :
  b 0 = 2 ∧
  b 1 = 5 ∧
  (∀ n : ℕ, b (n + 1) = 3 * (n + 1) - 1) :=
by
  sorry

theorem part2_sum_first_20_terms :
  (∑ i in finset.range 20, a i) = 300 :=
by
  sorry

end part1_b1_b2_general_formula_part2_sum_first_20_terms_l512_512846


namespace sample_std_dev_range_same_l512_512854

noncomputable def sample_std_dev (data : List ℝ) : ℝ := sorry
noncomputable def sample_range (data : List ℝ) : ℝ := sorry

theorem sample_std_dev_range_same (n : ℕ) (c : ℝ) (Hc : c ≠ 0) (x : Fin n → ℝ) :
  sample_std_dev (List.ofFn (λ i => x i)) = sample_std_dev (List.ofFn (λ i => x i + c)) ∧
  sample_range (List.ofFn (λ i => x i)) = sample_range (List.ofFn (λ i => x i + c)) :=
by
  sorry

end sample_std_dev_range_same_l512_512854


namespace sum_of_underlined_numbers_positive_l512_512763

-- Define a type for the sequence of numbers
variable {α : Type*} [LinearOrderedField α]

/-- Given a sequence of numbers satisfying the underlining conditions, 
    prove the sum of the underlined numbers is positive -/
theorem sum_of_underlined_numbers_positive 
  (a : ℕ → α) -- a sequence of real numbers
  (n : ℕ) -- length of the sequence
  (h1 : ∀ i, 0 ≤ i ∧ i < n → a i > 0 → a i underlined) -- each positive number is underlined
  (h2 : ∀ i k, 0 ≤ i ∧ i + k < n → (a i + a (i + 1) + ... + a (i + k)) > 0 → a i underlined) -- sum condition for underlining
  : ∑ i in (finset.range n).filter (λ i, is_underlined a i), a i > 0 
  := 
sorry

end sum_of_underlined_numbers_positive_l512_512763


namespace find_shorter_side_of_rectangle_l512_512354

theorem find_shorter_side_of_rectangle:
  ∃ x : ℝ, 
    let y := 9 in 
    let d := Real.sqrt (x^2 + y^2) in
    (x + y) - d = 3 ∧ y = 9 ∧ x = 3.75 :=
by
  sorry

end find_shorter_side_of_rectangle_l512_512354


namespace number_of_planes_l512_512697

def intersect_at_single_point (l₁ l₂ l₃ : Line) : Prop := ∃ p : Point, (p ∈ l₁) ∧ (p ∈ l₂) ∧ (p ∈ l₃)

def pairwise_intersecting_lines (l₁ l₂ l₃ : Line) : Prop :=
  (∃ p₁ : Point, p₁ ∈ l₁ ∧ p₁ ∈ l₂) ∧
  (∃ p₂ : Point, p₂ ∈ l₂ ∧ p₂ ∈ l₃) ∧
  (∃ p₃ : Point, p₃ ∈ l₃ ∧ p₃ ∈ l₁)

theorem number_of_planes (l₁ l₂ l₃ : Line) (h : pairwise_intersecting_lines l₁ l₂ l₃) :
  (intersect_at_single_point l₁ l₂ l₃ ∨ ¬ intersect_at_single_point l₁ l₂ l₃) →
  ∃ n : ℕ, n = 1 ∨ n = 3 :=
sorry

end number_of_planes_l512_512697


namespace cosine_of_angle_between_vectors_l512_512540

variables {ℝ : Type*} [inner_product_space ℝ E]
variables (a b : E)

theorem cosine_of_angle_between_vectors 
  (h₁ : 2 * ∥a∥ = 3 * ∥b∥) 
  (h₂ : ∥a - 2 • b∥ = ∥a + b∥) :
  real.cos (inner_product_space.angle a b) = 1 / 3 :=
by
  sorry

end cosine_of_angle_between_vectors_l512_512540


namespace sin_300_eq_neg_half_l512_512415

theorem sin_300_eq_neg_half :
  real.sin (300 * real.pi / 180) = -1/2 :=
by
  sorry

end sin_300_eq_neg_half_l512_512415


namespace net_profit_expr_l512_512360

def investmentCost := 144
def maintenanceCost (n : ℕ) := 4 * n^2 + 20 * n
def revenue (n : ℕ) := n

def netProfit (n : ℕ) := revenue n - maintenanceCost n - investmentCost

theorem net_profit_expr (n : ℕ) (h1 : n ≥ 3) :
  netProfit n = -4 * n^2 + 80 * n - 144 ∧ n - 4 * n^2 + 80 * n - 144 > 0 :=
by
  sorry

end net_profit_expr_l512_512360


namespace common_ratio_of_geometric_series_l512_512388

theorem common_ratio_of_geometric_series (a S r : ℝ) (h₁ : a = 400) (h₂ : S = 2500) :
  S = a / (1 - r) → r = 21 / 25 :=
by
  intros h₃
  rw [h₁, h₂] at h₃
  sorry

end common_ratio_of_geometric_series_l512_512388


namespace normal_dist_transformation_l512_512891

open MeasureTheory

variables (ξ : ℝ → Prop) (mean variance : ℝ)

-- Assume that ξ follows a normal distribution N(3, 4)
def normal_dist (ξ : ℝ → Prop) : Prop :=
  ∃ (μ σ : ℝ), μ = 3 ∧ σ = 2 ∧ (∀ (x : ℝ), ξ x = normalPDF μ σ x)

-- Define the expected value (mean)
def E (f : ℝ → ℝ) : ℝ :=
  ∫ x, f x ∂measure_space.volume

-- Define the variance
def D (f : ℝ → ℝ) : ℝ :=
  E (λ x, (f x - E f) ^ 2)

-- Define the specific random variable and its transformation
def ξ_transformed (x : ℝ) : ℝ :=
  2 * x + 1

-- The statement to be proved
theorem normal_dist_transformation :
  normal_dist ξ →
  E ξ = 3 →
  D ξ = 4 →
  E ξ_transformed = 7 ∧ D ξ_transformed = 16 := by
  intros h_norm h_Eξ h_Dξ
  sorry

end normal_dist_transformation_l512_512891


namespace quentavious_gum_count_l512_512207

def initial_nickels : Nat := 5
def remaining_nickels : Nat := 2
def gum_per_nickel : Nat := 2
def traded_nickels (initial remaining : Nat) : Nat := initial - remaining
def total_gum (trade_n gum_per_n : Nat) : Nat := trade_n * gum_per_n

theorem quentavious_gum_count : total_gum (traded_nickels initial_nickels remaining_nickels) gum_per_nickel = 6 := by
  sorry

end quentavious_gum_count_l512_512207


namespace find_chord_eq_l512_512869

-- Given conditions 
def ellipse_eq (x y : ℝ) : Prop := 4 * x^2 + 9 * y^2 = 144
def point_p : (ℝ × ℝ) := (3, 2)
def midpoint_chord (p1 p2 p : (ℝ × ℝ)) : Prop := p.fst = (p1.fst + p2.fst) / 2 ∧ p.snd = (p1.snd + p2.snd) / 2

-- Conditions in Lean definition
def conditions (x1 y1 x2 y2 : ℝ) : Prop :=
  ellipse_eq x1 y1 ∧ ellipse_eq x2 y2 ∧ midpoint_chord (x1,y1) (x2,y2) point_p

-- The statement to prove
theorem find_chord_eq (x1 y1 x2 y2 : ℝ) (h : conditions x1 y1 x2 y2) :
  ∃ m b : ℝ, (m = -2 / 3) ∧ b = 2 - m * 3 ∧ (∀ x y : ℝ, y = m * x + b → 2 * x + 3 * y - 12 = 0) :=
by {
  sorry
}

end find_chord_eq_l512_512869


namespace midpoints_on_single_line_l512_512345

-- Given an isosceles trapezoid ABCD with AB parallel to CD
variable (A B C D Q P : Type*)
variable [HasDistance A B C D Q P]

def isosceles_trapezoid (A B C D : Type*)
  {AB CD : A → D → Prop} 
  (h_parallel : ∀ (a b c d : A), AB a b = AB c d ∧ CD a b = CD c d)
  (h_isosceles : ∀ (a b c d : A), AB a b = AB c d) 
  : Prop := sorry

-- Points Q and P on lateral sides AB and CD respectively
variable (lateral_sides: AB Q ∧ CD P)

-- Condition: CP = AQ
variable (condition : distance C P = distance A Q)

-- Midpoint function
variable (midpoint : (A × A) → A)

-- Proof statement: The midpoints of all such segments PQ lie on a single line, specifically the midline MN
theorem midpoints_on_single_line :
  ∀ (A B C D Q P : Type*)
    (isosceles_trapezoid : isosceles_trapezoid A B C D)
    (lateral_sides: AB Q ∧ CD P)
    (condition : distance C P = distance A Q),
    ∃ MN : Type*, ∀ (PQ : A × A), midpoint PQ ∈ MN :=
sorry

end midpoints_on_single_line_l512_512345


namespace andy_late_minutes_l512_512393

theorem andy_late_minutes (school_starts_at : Nat) (normal_travel_time : Nat) 
  (stop_per_light : Nat) (red_lights : Nat) (construction_wait : Nat) 
  (left_house_at : Nat) : 
  let total_delay := (stop_per_light * red_lights) + construction_wait
  let total_travel_time := normal_travel_time + total_delay
  let arrive_time := left_house_at + total_travel_time
  let late_time := arrive_time - school_starts_at
  late_time = 7 :=
by
  sorry

end andy_late_minutes_l512_512393


namespace total_surface_area_pyramid_l512_512684

noncomputable def pyramid_base_side (a : ℝ) : Type := ℝ
noncomputable def ratio_m (m : ℝ) : Type := ℝ
noncomputable def ratio_n (n : ℝ) : Type := ℝ

theorem total_surface_area_pyramid (a m n : ℝ) (h_a_pos : 0 < a) (h_m_pos : 0 < m) (h_n_pos : 0 < n) :
  total_surface_area a m n = (a^2 * real.sqrt 3) / 4 * (1 + real.sqrt ((3 * (m + 2 * n)) / m)) :=
sorry

end total_surface_area_pyramid_l512_512684


namespace find_b1_l512_512303

noncomputable def sequence (b : ℕ → ℝ) : Prop :=
∀ n, n ≥ 2 → ∑ i in range n, b (i + 1) = n^3 * b n

def b25_val (b : ℕ → ℝ) : Prop :=
b 25 = 2

theorem find_b1 (b : ℕ → ℝ) [seq : sequence b] [b25 : b25_val b] : b 1 = 15624 := sorry

end find_b1_l512_512303


namespace find_xyz_value_l512_512179

noncomputable theory
variables {x y z : ℂ}

-- Conditions 
def condition1 (x y z : ℂ) := x * y + 6 * y = -24
def condition2 (x y z : ℂ) := y * z + 6 * z = -24
def condition3 (x y z : ℂ) := z * x + 6 * x = -24

-- Theorem statement proving the value of xyz
theorem find_xyz_value (h1 : condition1 x y z) 
                       (h2 : condition2 x y z) 
                       (h3 : condition3 x y z) : x * y * z = 120 :=
sorry

end find_xyz_value_l512_512179


namespace amount_after_a_year_l512_512655

def initial_amount : ℝ := 90
def interest_rate : ℝ := 0.10

theorem amount_after_a_year : initial_amount * (1 + interest_rate) = 99 := 
by
  -- Here 'sorry' indicates that the proof is not provided.
  sorry

end amount_after_a_year_l512_512655


namespace cosine_of_angle_between_OA_OB_find_k_perpendicular_l512_512139

noncomputable def dot_product (v1 v2 : ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

noncomputable def magnitude (v : ℝ × ℝ) : ℝ :=
  Real.sqrt (v.1 ^ 2 + v.2 ^ 2)

theorem cosine_of_angle_between_OA_OB :
  let OA := (1, 2)
  let OB := (2, -4)
  (dot_product OA OB) / ((magnitude OA) * (magnitude OB)) = -3 / 5 :=
by
  sorry

theorem find_k_perpendicular :
  let OA := (1, 2)
  let OB := (2, -4)
  let OP := (1.5, -1)
  let k := (1 : ℝ) / 14
  dot_product OP (OA + k * OB) = 0 :=
by
  sorry

end cosine_of_angle_between_OA_OB_find_k_perpendicular_l512_512139


namespace sine_addition_example_cosine_addition_identity_max_cosine_product_l512_512636

-- Definitions based on identified conditions
def sine_sum_formula (α β : ℝ) : ℝ :=
  sin (α + β) = sin α * cos β + cos α * sin β

def sine_diff_formula (α β : ℝ) : ℝ :=
  sin (α - β) = sin α * cos β - cos α * sin β

-- Problem (1)
theorem sine_addition_example : sin 15 + sin 75 = (sqrt 6) / 2 :=
by
  sorry

-- Definitions based on cosine identities
def cosine_sum_formula (α β : ℝ) : ℝ :=
  cos (α + β) = cos α * cos β - sin α * sin β

def cosine_diff_formula (α β : ℝ) : ℝ :=
  cos (α - β) = cos α * cos β + sin α * sin β

-- Problem (2)
theorem cosine_addition_identity (A B : ℝ) : 
  cos A + cos B = 2 * cos ((A + B) / 2) * cos ((A - B) / 2) :=
by
  sorry

-- Problem (3)
theorem max_cosine_product (x : ℝ) (h : x ∈ set.Icc 0 (π / 4)) :
  cos (2 * x) * cos (2 * x + π / 6) ≤ sqrt 3 / 2 :=
by
  sorry

end sine_addition_example_cosine_addition_identity_max_cosine_product_l512_512636


namespace reduce_consumption_percentage_l512_512734

theorem reduce_consumption_percentage :
  ∀ (current_rate old_rate : ℝ), 
  current_rate = 20 → 
  old_rate = 16 → 
  ((current_rate - old_rate) / old_rate * 100) = 25 :=
by
  intros current_rate old_rate h_current h_old
  sorry

end reduce_consumption_percentage_l512_512734


namespace equidistant_planes_l512_512313

noncomputable def bisector_planes (a1 b1 c1 d1 a2 b2 c2 d2 : ℝ) : 
  (ℝ × ℝ × ℝ × ℝ) × (ℝ × ℝ × ℝ × ℝ) :=
  let B1 := (a1 - a2, b1 - b2, c1 - c2, d1 - d2)
  let B2 := (a1 + a2, b1 + b2, c1 + c2, d1 + d2)
  (B1, B2)

theorem equidistant_planes (a1 b1 c1 d1 a2 b2 c2 d2 : ℝ) :
  let B1 := (a1 - a2, b1 - b2, c1 - c2, d1 - d2)
  let B2 := (a1 + a2, b1 + b2, c1 + c2, d1 + d2)
  bisector_planes a1 b1 c1 d1 a2 b2 c2 d2 = (B1, B2) :=
by 
  intros
  refl

end equidistant_planes_l512_512313


namespace product_of_cosine_elements_l512_512009

-- Definitions for the problem conditions
def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + d

def S (a : ℕ → ℝ) (d : ℝ) : Set ℝ :=
  {x | ∃ n : ℕ, x = Real.cos (a n)}

-- Main theorem statement
theorem product_of_cosine_elements 
  (a : ℕ → ℝ) 
  (d : ℝ) 
  (h_seq : arithmetic_sequence a (2 * Real.pi / 3))
  (h_S_elements : (S a (2 * Real.pi / 3)).card = 2) 
  (h_S_contains : ∃ a b, a ≠ b ∧ S a (2 * Real.pi / 3) = {a, b}) :
  let (a, b) := Classical.choose (h_S_contains) in
  a * b = -1 / 2 :=
by
  sorry

end product_of_cosine_elements_l512_512009


namespace sample_standard_deviation_same_sample_range_same_l512_512855

open Nat

variables {n : ℕ} (x : Fin n → ℝ) (c : ℝ)
hypothesis (h_c : c ≠ 0)

/-- Assertion C: The sample standard deviations of the two sets of sample data are the same. -/
theorem sample_standard_deviation_same :
  (1 / n * ∑ i, (x i - (1 / n * ∑ i, x i))^2).sqrt =
  (1 / n * ∑ i, (x i + c - (1 / n * ∑ i, x i + c))^2).sqrt := sorry

/-- Assertion D: The sample ranges of the two sets of sample data are the same. -/
theorem sample_range_same :
  (Finset.sup Finset.univ x - Finset.inf Finset.univ x) =
  (Finset.sup Finset.univ (fun i => x i + c) - Finset.inf Finset.univ (fun i => x i + c)) := sorry

end sample_standard_deviation_same_sample_range_same_l512_512855


namespace max_value_m_l512_512268

open Real

noncomputable def f (x : ℝ) : ℝ := sorry

theorem max_value_m :
  (∀ x : ℝ, f (-x) = -f x) ∧
  (∀ x : ℝ, f (x + 5) = f x) ∧
  (f 2 ≥ 2) ∧
  (∃ m : ℝ, f 3 = (2^(m + 1) - 3) / (2^m + 1)) →
  ∃ m : ℝ, m = -2 :=
by
  intros h
  obtain ⟨m, h_m⟩ := h.4
  have h1 : f 3 = f (-2), from (by calc
    f 3 = f (3 - 5) : by rw [h.2]
        ... = f (-2) : by norm_num
  ),
  have h2 : f (-2) = -f 2, from h.1 2,
  have h3 : -f 2 ≤ -2, by exact neg_le_neg h.3,
  have : (2^(m + 1) - 3) / (2^m + 1) ≤ -2, from h1.symm ▸ h_m ▸ h3,
  sorry

end max_value_m_l512_512268


namespace min_value_a_l512_512101

theorem min_value_a (a : ℝ) : 
  (∀ x : ℝ, x^2 - x - 6 > 0 → x > a) ∧
  ¬ (∀ x : ℝ, x > a → x^2 - x - 6 > 0) ↔ a = 3 :=
sorry

end min_value_a_l512_512101


namespace inscribed_sphere_radius_of_regular_triangular_prism_l512_512467

theorem inscribed_sphere_radius_of_regular_triangular_prism
  (P A B C : ℝ)
  (h_base: A = 1)
  (h_height: C = sqrt 2) :
  r = (sqrt 2 / 6) :=
by
  sorry

end inscribed_sphere_radius_of_regular_triangular_prism_l512_512467


namespace chess_bishop_knight_expected_moves_equal_prob_l512_512570

theorem chess_bishop_knight_expected_moves_equal_prob:
  ∃ p : ℚ, let (a, b) := (p.num, p.den) in
  (8 * (1 - p) = 4 * (1 - p) / p) →
  (100 * a + b = 102) :=
begin
  sorry
end

end chess_bishop_knight_expected_moves_equal_prob_l512_512570


namespace range_of_x_l512_512082

variable {x : ℝ}

def p := 6 - 3 * x ≥ 0
def q := (1 / (x + 1)) < 0

theorem range_of_x (h : p ∧ ¬ q) : -1 ≤ x ∧ x ≤ 2 :=
by
  sorry

end range_of_x_l512_512082


namespace solve_fractional_equation_l512_512227

theorem solve_fractional_equation (x : ℝ) (hx : x ≠ 1 ∧ x ≠ 1) : 3 / (x - 1) = 5 + 3 * x / (1 - x) ↔ x = 4 :=
by 
  split
  -- Forward direction (Given the equation, show x = 4)
  intro h
  -- Multiply out and simplify as shown in the solution steps
  -- Sorry statement skips the detailed proof
  sorry
  -- Converse direction (Substitute x = 4 and verify it satisfies the equation)
  intro hx
  -- Substitution and verification steps
  -- Sorry statement skips the detailed proof
  sorry

end solve_fractional_equation_l512_512227


namespace sample_stats_equal_l512_512860

/-- Let x be a data set of n samples and y be another data set of n samples such that 
    ∀ i, y_i = x_i + c where c is a non-zero constant.
    Prove that the sample standard deviations and the ranges of x and y are the same. -/
theorem sample_stats_equal (n : ℕ) (x y : Fin n → ℝ) (c : ℝ) (h : c ≠ 0)
    (h_y : ∀ i : Fin n, y i = x i + c) :
    (stddev x = stddev y) ∧ (range x = range y) := 
sorry

end sample_stats_equal_l512_512860


namespace twin_ages_l512_512774

theorem twin_ages (x : ℕ) (h : (x + 1) ^ 2 = x ^ 2 + 15) : x = 7 :=
sorry

end twin_ages_l512_512774


namespace solve_fractional_equation_l512_512249

theorem solve_fractional_equation (x : ℝ) (h1 : x ≠ 1) (h2 : x ≠ -1) : 
    (3 / (x - 1) = 5 + 3 * x / (1 - x)) → x = 4 := 
by
  sorry

end solve_fractional_equation_l512_512249


namespace max_lines_between_points_l512_512690

noncomputable def maxLines (points : Nat) := 
  let deg := [1, 2, 3, 4, 5]
  (1 * (points - 1) + 2 * (points - 2) + 3 * (points - 3) + 4 * (points - 4) + 5 * (points - 5)) / 2

theorem max_lines_between_points :
  ∀ (n : Nat), n = 15 → maxLines n = 85 :=
by
  intros n hn
  sorry

end max_lines_between_points_l512_512690


namespace square_tablecloth_side_length_l512_512765

theorem square_tablecloth_side_length (area : ℝ) (h : area = 5) : ∃ a : ℝ, a > 0 ∧ a * a = 5 := 
by
  use Real.sqrt 5
  constructor
  · apply Real.sqrt_pos.2; linarith
  · exact Real.mul_self_sqrt (by linarith [h])

end square_tablecloth_side_length_l512_512765


namespace ratio_of_AB_and_CD_lengths_of_AB_and_CD_l512_512003

variable {α : Type*} [LinearOrderedField α]

noncomputable def AB_CD_ratio (AM DM BN CN : α) : α :=
  (AM / DM) * (BN / CN)

theorem ratio_of_AB_and_CD
  {AM DM BN CN : α}
  (h_AM : AM = 9 / 7)
  (h_DM : DM = 12 / 7)
  (h_BN : BN = 20 / 9)
  (h_CN : CN = 25 / 9) :
  AB_CD_ratio AM DM BN CN =
  3 / 5 :=
by
  simp only [AB_CD_ratio, h_AM, h_DM, h_BN, h_CN]
  norm_num

theorem lengths_of_AB_and_CD
  {AB CD : α}
  (h_ABCD_ratio : AB / CD = 3 / 5)
  (h_inscribed_circles_tangent : true) : -- Implicit condition not formulaic
  AB = 3 ∧ CD = 5 :=
by
  have h1 : AB = 3 * x for some x := sorry -- Placeholder logic to be derived based on actual proof details
  have h2 : CD = 5 * x for some x := sorry -- Placeholder logic to be derived based on actual proof details
  split
    -- Here we would show the steps to solve for AB and CD
    exact sorry -- Placeholder
    exact sorry -- Placeholder

end ratio_of_AB_and_CD_lengths_of_AB_and_CD_l512_512003


namespace T_shape_exists_in_remaining_grid_l512_512828

def grid : Type := fin 100 × fin 100

def rectangle := fin 2 × fin 1 ⊕ fin 1 × fin 2

noncomputable def initial_grid (n : ℕ) : set grid := {p | true}

def cut_rectangles (g : set grid) (rects : list rectangle) : set grid :=
by sorry

def T_shape : set grid := { (0,0), (1,0), (2,0), (1,1) }

theorem T_shape_exists_in_remaining_grid :
  ∀ rects : list rectangle, rects.length = 1950 →
    (∀ r ∈ rects, (∃ a b, r = sum.inl (a,b) ∨ r = sum.inr (a,b))) →
    ∃ g', g' = cut_rectangles initial_grid rects ∧ (T_shape ⊆ g') :=
by sorry

end T_shape_exists_in_remaining_grid_l512_512828


namespace combined_tax_rate_correct_l512_512150

noncomputable def combined_tax_rate (t_j t_i : ℚ) (I_j I_i : ℕ) : ℚ :=
  let tax_j := t_j * I_j
  let tax_i := t_i * I_i
  let total_tax := tax_j + tax_i
  let total_income := I_j + I_i
  (total_tax / total_income) * 100

theorem combined_tax_rate_correct :
  combined_tax_rate 0.30 0.40 56000 74000 ≈ 35.69 := sorry

end combined_tax_rate_correct_l512_512150


namespace sum_seq_formula_l512_512006

open Nat

-- Definitions based on the given conditions
def seq (n : ℕ) : ℕ → ℝ := λ n, match n with
  | 0   => 1
  | 1   => 1/3
  | 2   => 1/6
  | _   => 0 -- This will need to be extended for a general proof
  
def sum_seq : ℕ → ℝ := λ n, (2 * n) / (n + 1)

-- The theorem to prove, using the given conditions
theorem sum_seq_formula (n : ℕ) (h_pos : n > 0) 
  (h_init : seq 1 = 1) 
  (h_rec : ∀ m : ℕ, m ≥ 1 → sum_seq (m - 1) + seq m = m^2 * seq m) : 
  sum_seq n = (2 * n) / (n + 1) :=
by
  sorry

end sum_seq_formula_l512_512006


namespace sample_stats_equal_l512_512862

/-- Let x be a data set of n samples and y be another data set of n samples such that 
    ∀ i, y_i = x_i + c where c is a non-zero constant.
    Prove that the sample standard deviations and the ranges of x and y are the same. -/
theorem sample_stats_equal (n : ℕ) (x y : Fin n → ℝ) (c : ℝ) (h : c ≠ 0)
    (h_y : ∀ i : Fin n, y i = x i + c) :
    (stddev x = stddev y) ∧ (range x = range y) := 
sorry

end sample_stats_equal_l512_512862


namespace complement_union_eq_l512_512526

def U : Set ℤ := {-2, -1, 0, 1, 2, 3}
def A : Set ℤ := {-1, 2}
def B : Set ℤ := {x | x ^ 2 - 4 * x + 3 = 0}

theorem complement_union_eq :
  (U \ (A ∪ B)) = {-2, 0} :=
by
  sorry

end complement_union_eq_l512_512526


namespace sin_alpha_value_l512_512063

theorem sin_alpha_value (α : ℝ) (h1 : 0 < α ∧ α < π / 2) (h2 : Real.tan (π - α) + 3 = 0) : 
  Real.sin α = 3 * Real.sqrt 10 / 10 := 
by
  sorry

end sin_alpha_value_l512_512063


namespace part1_b1_b2_general_formula_part2_sum_first_20_terms_l512_512848

/-- Define the sequence a_n recursively --/
def a : ℕ → ℕ
| 0 := 1
| (n+1) := if n % 2 = 0 then a n + 1 else a n + 2

/-- Define the sequence b_n such that b_n = a_{2n} --/
def b (n : ℕ) : ℕ := a (2 * (n + 1))

/-- State the properties to be proved --/
theorem part1_b1_b2_general_formula :
  b 0 = 2 ∧
  b 1 = 5 ∧
  (∀ n : ℕ, b (n + 1) = 3 * (n + 1) - 1) :=
by
  sorry

theorem part2_sum_first_20_terms :
  (∑ i in finset.range 20, a i) = 300 :=
by
  sorry

end part1_b1_b2_general_formula_part2_sum_first_20_terms_l512_512848


namespace sum_of_k_selections_l512_512394

theorem sum_of_k_selections (k : ℕ) :
  (∑ i in finset.range k, (k * i + (i + 1))) = (k * (k^2 + 1)) / 2 :=
begin
  sorry
end

end sum_of_k_selections_l512_512394


namespace part1_part2_l512_512510

def f (x a : ℝ) := |x - a| + 2 * |x + 1|

-- Part 1: Solve the inequality f(x) > 4 when a = 2
theorem part1 (x : ℝ) : f x 2 > 4 ↔ (x < -4/3 ∨ x > 0) := by
  sorry

-- Part 2: If the solution set of the inequality f(x) < 3x + 4 is {x | x > 2}, find the value of a.
theorem part2 (a : ℝ) : (∀ x : ℝ, (f x a < 3 * x + 4 ↔ x > 2)) → a = 6 := by
  sorry

end part1_part2_l512_512510


namespace probability_zero_probability_correct_l512_512178

noncomputable def probability_lfloor_sqrt : ℝ :=
  let x := (random.uniform 50 150 : ℝ)
  if 100 <= x ∧ x < 121 then
    if 90 <= x ∧ x < 96.1 then 0
    else 0
  else 0

theorem probability_zero :
  ∀ (x : ℝ), 100 <= x ∧ x < 121 → ¬ (90 <= x ∧ x < 96.1) :=
by
  intros x h
  have : ¬ (100 <= x ∧ x < 121 ∧ 90 <= x ∧ x < 96.1) := by
    intros h'
    cases h' with h₁ h₂
    cases h₁ with h₃ h₄
    cases h₂ with h₅ h₆
    exact (not_le_of_lt h₆) h₃
  exact this

theorem probability_correct :
  probability_lfloor_sqrt = 0 := 
by
  sorry

end probability_zero_probability_correct_l512_512178


namespace cheapest_pie_cost_is_18_l512_512407

noncomputable def crust_cost : ℝ := 2 + 1 + 1.5
noncomputable def blueberry_container_cost : ℝ := 2.25
noncomputable def blueberry_containers_needed : ℕ := 3 * (16 / 8)
noncomputable def blueberry_filling_cost : ℝ := blueberry_containers_needed * blueberry_container_cost
noncomputable def cherry_filling_cost : ℝ := 14
noncomputable def cheapest_filling_cost : ℝ := min blueberry_filling_cost cherry_filling_cost
noncomputable def total_cheapest_pie_cost : ℝ := crust_cost + cheapest_filling_cost

theorem cheapest_pie_cost_is_18 : total_cheapest_pie_cost = 18 := by
  sorry

end cheapest_pie_cost_is_18_l512_512407


namespace find_p4_q4_l512_512283

-- Definitions
def p (x : ℝ) : ℝ := 3 * (x - 6) * (x - 2)
def q (x : ℝ) : ℝ := (x - 6) * (x + 3)

-- Statement to prove
theorem find_p4_q4 : (p 4) / (q 4) = 6 / 7 :=
by
  sorry

end find_p4_q4_l512_512283


namespace find_dihedral_angle_l512_512276

noncomputable def dihedral_angle_problem :
  Type := sorry  -- We may need custom types for geometric objects

def is_isosceles_right_triangle (ABC : Triangle ℝ) (AB : ℝ) : Prop :=
  (ABC.angleA = ABC.angleB = π/4) ∧ (ABC.hypotenuse = AB)

def is_midpoint (D H : Point ℝ) (A C : Line ℝ) : Prop :=
  H = (A + C) / 2 ∧ D.rectilinear_above H

def height (ABCD : Pyramid ℝ) : ℝ :=
  2

theorem find_dihedral_angle (ABCD : Pyramid ℝ)
  (hypotenuse : ℝ) (height : ℝ) (mid_ac : Point ℝ) :
  ∀ {a b c d: Point ℝ}, 
  is_isosceles_right_triangle a b c 
  → is_midpoint d a c 
  → dihedral_angle a b d = arcsin (sqrt (3 / 5)) :=
sorry

end find_dihedral_angle_l512_512276


namespace range_of_a_l512_512448

variable (a : ℝ) (x : ℝ)

theorem range_of_a
  (h1 : 2 * x < 3 * (x - 3) + 1)
  (h2 : (3 * x + 2) / 4 > x + a) :
  -11 / 4 ≤ a ∧ a < -5 / 2 :=
sorry

end range_of_a_l512_512448


namespace calculate_subtraction_l512_512698

theorem calculate_subtraction :
  ∀ (x : ℕ), (49 = 50 - 1) → (49^2 = 50^2 - 99)
  := by
  intros x h
  sorry

end calculate_subtraction_l512_512698


namespace vlad_gosha_knight_king_impossible_l512_512704

theorem vlad_gosha_knight_king_impossible :
  (∀ (x y : ℕ), (1 ≤ x ∧ x ≤ 64) ∧ (1 ≤ y ∧ y ≤ 64) →
    (vlad_knight_move x y ↔ gosha_king_move x y)) →
  false :=
by
  sorry

-- Definitions for moves can be declared (not part of the problem initially, so they serve for compiling)
def vlad_knight_move (x y : ℕ) : Prop := sorry
def gosha_king_move (x y : ℕ) : Prop := sorry

end vlad_gosha_knight_king_impossible_l512_512704


namespace total_copies_l512_512733

-- Conditions: Defining the rates of two copy machines and the time duration
def rate1 : ℕ := 35 -- rate in copies per minute for the first machine
def rate2 : ℕ := 65 -- rate in copies per minute for the second machine
def time : ℕ := 30 -- time in minutes

-- The theorem stating that the total number of copies made by both machines in 30 minutes is 3000
theorem total_copies : rate1 * time + rate2 * time = 3000 := by
  sorry

end total_copies_l512_512733


namespace equal_valerian_drops_l512_512392

def is_sunny (d : ℕ) : Prop := sorry -- Replace with the actual condition for a sunny day
def is_cloudy (d : ℕ) : Prop := sorry -- Replace with the actual condition for a cloudy day
def andrey_stepanovich_drops (d : ℕ) : ℕ :=
  if is_sunny d then sorry else d -- Replace sorry with the actual drop count function for Andrey Stepanovich
def ivan_petrovich_drops (d : ℕ) : ℕ :=
  if is_cloudy d then d else 0

theorem equal_valerian_drops (April_days : Fin 30) :
  let AS_total := (∑ d in Finset.range 30, andrey_stepanovich_drops d)
  let IP_total := (∑ d in Finset.range 30, ivan_petrovich_drops d)
  AS_total = IP_total := sorry

end equal_valerian_drops_l512_512392


namespace complement_A_union_B_l512_512533

-- Define the universal set U
def U : Set ℤ := {-2, -1, 0, 1, 2, 3}

-- Define the set A
def A : Set ℤ := {-1, 2}

-- Define the set B using the quadratic equation condition
def B : Set ℤ := {x | x^2 - 4*x + 3 = 0}

-- State the theorem we want to prove
theorem complement_A_union_B : (U \ (A ∪ B)) = {-2, 0} := by
  sorry

end complement_A_union_B_l512_512533


namespace solve_equation_l512_512237

theorem solve_equation : (x : ℝ) (hx : x ≠ 1) (h : 3 / (x - 1) = 5 + 3 * x / (1 - x)) : x = 4 :=
sorry

end solve_equation_l512_512237


namespace tangle_language_num_valid_words_l512_512672

noncomputable def num_valid_words : ℕ :=
  let total_words_of_length :=
    λ n, (25:ℕ)^n
  let words_without_B_of_length :=
    λ n, (24:ℕ)^n
  let words_with_B_of_length :=
    λ n, total_words_of_length n - words_without_B_of_length n
  
  let valid_one_letter := words_with_B_of_length 1
  let valid_two_letter := words_with_B_of_length 2
  let valid_three_letter := words_with_B_of_length 3
  let valid_four_letter := words_with_B_of_length 4
  let valid_five_letter := words_with_B_of_length 5

  valid_one_letter + valid_two_letter + valid_three_letter + valid_four_letter + valid_five_letter

theorem tangle_language_num_valid_words : num_valid_words = 1855701 := sorry

end tangle_language_num_valid_words_l512_512672


namespace sum_of_underlined_is_positive_l512_512760

-- Definitions
def is_positive (x : ℤ) : Prop := x > 0

def is_underlined (a : ℕ → ℤ) (i : ℕ) : Prop :=
  is_positive (a i) ∨ ∃ k : ℕ, k ≤ (n - 1 - i) ∧ (∑ j in finset.range (k + 1), a (i + j)) > 0

-- Main statement
theorem sum_of_underlined_is_positive (a : ℕ → ℤ) (n : ℕ) (h : ∀ i : ℕ, i < n → is_underlined a i) :
  ∑ i in finset.range n, if is_underlined a i then a i else 0 > 0 :=
sorry

end sum_of_underlined_is_positive_l512_512760


namespace imaginary_part_of_conjugate_eq_two_l512_512498

open Complex

theorem imaginary_part_of_conjugate_eq_two :
  let z := (2 - 2 * I) / (1 + I)
  im (conj z) = 2 := 
by 
  sorry

end imaginary_part_of_conjugate_eq_two_l512_512498


namespace solve_for_Z_l512_512980

def i : ℂ := complex.I  -- Define i as the imaginary unit

theorem solve_for_Z (Z : ℂ) (h : (1 + complex.I) * Z = 2) : Z = 1 - complex.I :=
by
  sorry

end solve_for_Z_l512_512980


namespace ordered_pairs_count_l512_512544

theorem ordered_pairs_count :
  ∃! (xy: ℝ × ℝ), xy.1 + 4 * xy.2 = 4 ∧ (| |xy.1| - |xy.2| |) = 2 :=
sorry

end ordered_pairs_count_l512_512544


namespace binomial_expansion_problem_l512_512893

theorem binomial_expansion_problem :
  (∃ n : ℕ, 
    (binom n (n-2) + binom n (n-1) + binom n n = 121) ∧
      (∀ r : ℕ, (r = 11 -> (3^r * binom 15 r * (x^r)) = (T 12 = binom 15 11 * (3^11) * (x^11))
      ∧ (r = 12 -> (3^r * binom 15 r * (x^r)) = (T 13 = binom 15 12 * (3^12) * (x^12)) ∨ (r = 11 ∨ r = 12)))
    
    (T 12 = binom 15 11 * (3^11) * (x^11)) 
      ∧ (T 13 = binom 15 12 * (3^12) * (x^12))): sorry

end binomial_expansion_problem_l512_512893


namespace solve_equation_l512_512265

theorem solve_equation : ∀ (x : ℝ), x ≠ 1 → (3 / (x - 1) = 5 + 3 * x / (1 - x)) → x = 4 :=
by
  intros x hx heq
  -- sorry to skip the proof
  sorry

end solve_equation_l512_512265


namespace tomatoes_not_sold_l512_512622

theorem tomatoes_not_sold (total_harvested sold_mrs_maxwell sold_mr_wilson : ℝ)
  (h1 : total_harvested = 245.5)
  (h2 : sold_mrs_maxwell = 125.5)
  (h3 : sold_mr_wilson = 78) :
total_harvested - (sold_mrs_maxwell + sold_mr_wilson) = 42 :=
by {
  rw [h1, h2, h3],
  norm_num,
  sorry
}

end tomatoes_not_sold_l512_512622


namespace second_player_guarantee_symmetry_l512_512702

theorem second_player_guarantee_symmetry (n : ℕ) (hn : n = 1999) :
  ∃ (f : ℕ → ℕ) (hf : ∀ i j : ℕ, i ≠ j → f i ≠ f j),
    (∀ (a : ℕ → ℕ), (∀ i : ℕ, i < n → a i = 0 ∨ a i = 1) →
      ∃ (b : ℕ → ℕ), (∀ i : ℕ, i < n → b i = 0 ∨ b i = 1) ∧
        (∀ i : ℕ, i < (n / 2) → b i = b (n - i - 1))) :=
by
  intros,
  sorry

end second_player_guarantee_symmetry_l512_512702


namespace joe_height_is_82_l512_512215

-- Given the conditions:
def Sara_height (x : ℝ) : Prop := true

def Joe_height (j : ℝ) (x : ℝ) : Prop := j = 6 + 2 * x

def combined_height (j : ℝ) (x : ℝ) : Prop := j + x = 120

-- We need to prove:
theorem joe_height_is_82 (x j : ℝ) 
  (h1 : combined_height j x)
  (h2 : Joe_height j x) :
  j = 82 := 
by 
  sorry

end joe_height_is_82_l512_512215


namespace pure_imaginary_number_solution_l512_512116

-- Definition of the problem
theorem pure_imaginary_number_solution (a : ℝ) (h1 : a^2 - 4 = 0) (h2 : a^2 - 3 * a + 2 ≠ 0) : a = -2 :=
sorry

end pure_imaginary_number_solution_l512_512116


namespace ways_to_pick_four_shoes_l512_512829

theorem ways_to_pick_four_shoes (pairs : Finset (Fin 5)) :
  ∃ (n : ℕ), n = 120 ∧ 
  (∃ p ∈ pairs, ∃ (remaining_shoes : Finset (Fin (2 * 4))), 
     ∃ (chosen_pairs : Finset (Fin 4)), chosen_pairs.card = 2 ∧ 
     ∃ (shoes_from_pairs : list (Fin 2)), 
       shoes_from_pairs.length = 2 ∧ 
       ∀ s ∈ shoes_from_pairs, s ∈ remaining_shoes) :=
by {
  sorry
}

end ways_to_pick_four_shoes_l512_512829


namespace cheapest_pie_cost_l512_512412

def crust_cost : ℝ := 2 + 1 + 1.5

def blueberry_pie_cost : ℝ :=
  let blueberries_needed := 3 * 16
  let containers_required := blueberries_needed / 8
  let blueberries_cost := containers_required * 2.25
  crust_cost + blueberries_cost

def cherry_pie_cost : ℝ := crust_cost + 14

theorem cheapest_pie_cost : blueberry_pie_cost = 18 :=
by sorry

end cheapest_pie_cost_l512_512412


namespace find_z_l512_512973

variable (x y : ℝ)
variable (h_pos : x > 0)
variable (h_z_square : (x + complex.I * y) ^ 2 = 18 * complex.I)

theorem find_z :
  (∃ x y, x > 0 ∧ (x + complex.I * y) ^ 2 = 18 * complex.I ∧ (x + complex.I * y) = 3 + 3 * complex.I) :=
  sorry

end find_z_l512_512973


namespace total_marbles_l512_512933

variable (r b g : ℝ)
variable (h1 : r = 1.3 * b)
variable (h2 : g = 1.7 * r)

theorem total_marbles (r b g : ℝ) (h1 : r = 1.3 * b) (h2 : g = 1.7 * r) :
  r + b + g = 3.469 * r :=
by
  sorry

end total_marbles_l512_512933


namespace Lisa_spent_l512_512615

-- Defining the number of DVDs and the cost per DVD.
def number_of_DVDs := 4
def cost_per_DVD := 1.20

-- The statement we need to prove
theorem Lisa_spent : (number_of_DVDs * cost_per_DVD) = 4.80 := by
  sorry

end Lisa_spent_l512_512615


namespace domain_of_f_l512_512278

def domain_f (x : ℝ) : Prop :=
  2 * x - 3 ≥ 0 ∧ x ≠ 3

def domain_set : Set ℝ :=
  { x | (3 / 2) ≤ x ∧ x < 3 ∨ 3 < x }

theorem domain_of_f :
  { x : ℝ | domain_f x } = domain_set := by
  sorry

end domain_of_f_l512_512278


namespace polar_to_cartesian_correct_l512_512081

noncomputable def polar_to_cartesian (r θ : ℝ) : ℝ × ℝ :=
  (r * Real.cos θ, r * Real.sin θ)

theorem polar_to_cartesian_correct : polar_to_cartesian 2 (5 * Real.pi / 6) = (-Real.sqrt 3, 1) :=
by
  sorry -- We are not required to provide the proof here

end polar_to_cartesian_correct_l512_512081


namespace solve_fractional_equation_l512_512230

theorem solve_fractional_equation (x : ℝ) (hx : x ≠ 1 ∧ x ≠ 1) : 3 / (x - 1) = 5 + 3 * x / (1 - x) ↔ x = 4 :=
by 
  split
  -- Forward direction (Given the equation, show x = 4)
  intro h
  -- Multiply out and simplify as shown in the solution steps
  -- Sorry statement skips the detailed proof
  sorry
  -- Converse direction (Substitute x = 4 and verify it satisfies the equation)
  intro hx
  -- Substitution and verification steps
  -- Sorry statement skips the detailed proof
  sorry

end solve_fractional_equation_l512_512230


namespace minimal_circle_intersect_l512_512883

noncomputable def circle_eq := 
  ∀ (x y : ℝ), 
    (x^2 + y^2 + 4 * x + y + 1 = 0) ∧
    (x^2 + y^2 + 2 * x + 2 * y + 1 = 0) → 
    (x^2 + y^2 + (6/5) * x + (3/5) * y + 1 = 0)

theorem minimal_circle_intersect :
  circle_eq :=
by
  sorry

end minimal_circle_intersect_l512_512883


namespace problem1_problem2_l512_512909

section Problems

variables (a x : ℝ)

def A := { x | -3 ≤ x ∧ x ≤ 6 }
def B := { x | 2 * a - 1 ≤ x ∧ x ≤ a + 1 }

-- 1. If a = -2, prove A ∪ B = { x | -5 ≤ x ∧ x ≤ 6 }
theorem problem1 (h : a = -2) : {x | (-3 ≤ x ∧ x ≤ 6) ∨ (-5 ≤ x ∧ x ≤ -1)} = {x | -5 ≤ x ∧ x ≤ 6} :=
by sorry

-- 2. If A ∩ B = B, prove a ≥ -1
theorem problem2 (h : ∀ x, (x ∈ A ∧ x ∈ B ↔ x ∈ B)) : a ≥ -1 :=
by sorry

end Problems

end problem1_problem2_l512_512909


namespace net_profit_expr_l512_512359

def investmentCost := 144
def maintenanceCost (n : ℕ) := 4 * n^2 + 20 * n
def revenue (n : ℕ) := n

def netProfit (n : ℕ) := revenue n - maintenanceCost n - investmentCost

theorem net_profit_expr (n : ℕ) (h1 : n ≥ 3) :
  netProfit n = -4 * n^2 + 80 * n - 144 ∧ n - 4 * n^2 + 80 * n - 144 > 0 :=
by
  sorry

end net_profit_expr_l512_512359


namespace max_value_of_sum_l512_512341

theorem max_value_of_sum (x y z : ℝ) (h : 0 ≤ x ∧ 0 ≤ y ∧ 0 ≤ z) 
  (eq : x^2 + y^2 + z^2 + x + 2*y + 3*z = (13 : ℝ) / 4) : x + y + z ≤ 3 / 2 :=
sorry

end max_value_of_sum_l512_512341


namespace plane_divides_tetrahedron_l512_512198

-- Defining points on the tetrahedron edges with given ratios
def tetrahedron (A B C D K N M : Point) : Prop :=
  collinear A B K ∧ collinear B C N ∧ collinear A D M ∧
  divides A B K (2 / 3) ∧ divides B C N (2 / 3) ∧ divides A D M (3 / 4)

-- Proving the ratio on edge CD
theorem plane_divides_tetrahedron (A B C D K N M : Point) (h : tetrahedron A B C D K N M) :
  ∃ F : Point, collinear C D F ∧ divides C D F (4 / 7) :=
sorry

end plane_divides_tetrahedron_l512_512198


namespace existence_of_abc_l512_512607

theorem existence_of_abc (n : ℕ) (hn : 0 < n) (A : Set ℕ) (hA_subset : A ⊆ Finset.range (5 ^ n + 1)) (hA_card : A.card = 4 * n + 2) :
  ∃ a b c ∈ A, a < b ∧ b < c ∧ c + 2 * a > 3 * b :=
sorry

end existence_of_abc_l512_512607


namespace area_bounded_by_curve_and_line_l512_512778

theorem area_bounded_by_curve_and_line :
  let curve_x (t : ℝ) := 10 * (t - Real.sin t)
  let curve_y (t : ℝ) := 10 * (1 - Real.cos t)
  let y_line := 15
  (∫ t in (2/3) * Real.pi..(4/3) * Real.pi, 100 * (1 - Real.cos t)^2) = 100 * Real.pi + 200 * Real.sqrt 3 :=
by
  sorry

end area_bounded_by_curve_and_line_l512_512778


namespace rectangle_area_l512_512372

theorem rectangle_area (x : ℕ) (hx : x > 0)
  (h₁ : (x + 5) * 2 * (x + 10) = 3 * x * (x + 10))
  (h₂ : (x - 10) = x + 10 - 10) :
  x * (x + 10) = 200 :=
by {
  sorry
}

end rectangle_area_l512_512372


namespace intersection_M_N_l512_512964

noncomputable def M : Set ℝ := {y | ∃ x ∈ Icc 0 5, y = 2 * Real.cos x}
noncomputable def N : Set ℝ := {x | ∃ y, y = Real.log (x - 1) / Real.log 2}

theorem intersection_M_N: M ∩ N = {x | 1 < x ∧ x ≤ 2} :=
by
  sorry

end intersection_M_N_l512_512964


namespace sum_of_undefined_expression_sum_of_undefined_values_l512_512322

theorem sum_of_undefined_expression (y : ℝ) :
  (y^2 - 7*y + 10 = 0) → (y = 5 ∨ y = 2) :=
begin
  intro h,
  have factored : (y - 5) * (y - 2) = y^2 - 7*y + 10,
  { ring },
  rw ← factored at h,
  apply eq_zero_or_eq_zero_of_mul_eq_zero h,
end

theorem sum_of_undefined_values :
  ∑ (y ∈ {5, 2}), y = 7 := 
by { simp, norm_num, }

end

end sum_of_undefined_expression_sum_of_undefined_values_l512_512322


namespace locus_of_points_sin_eq_zero_l512_512437

theorem locus_of_points_sin_eq_zero (x y : ℝ) : 
  (∃ k : ℤ, sin (x + y) = 0 ↔ x + y = k * real.pi) :=
sorry

end locus_of_points_sin_eq_zero_l512_512437


namespace part_i_error_part_ii_correction_l512_512328

-- Part I: Proof of incorrect equality
theorem part_i_error (a b : ℝ) :
    (4 - (9/2) = 5 - (9/2)) → (4 = 5) := sorry

-- Part II: Proof of correct inequality involving logarithms
theorem part_ii_correction (a b : ℝ) (x : ℝ) (h1 : 3 > 2) (h2 : 0 < x) :
    (3 * log x > 2 * log x) → (x^3 > x^2) :=
by {
    assume : 3 * log x > 2 * log x,
    have : log x < 0, from sorry, -- x < 1 -> log x < 0
    exact lt_trans (pow_lt_pow_of_lt_left this zero_lt_two h1) sorry,
}

end part_i_error_part_ii_correction_l512_512328


namespace ball_distribution_impossible_l512_512145

theorem ball_distribution_impossible (balls piles : ℕ) (distinct_balls : list ℕ) :
  balls = 44 ∧ piles = 9 ∧ distinct_balls.length = piles ∧ 
  (∀ i j : ℕ, i < j ∧ i < piles ∧ j < piles → distinct_balls.nth i ≠ distinct_balls.nth j) → 
  ∀ sum_distinct_balls ≥ (∑ k in finset.Icc 1 piles, k), balls < sum_distinct_balls :=
by
  intros h
  cases h with h_balls h'
  cases h' with h_piles h''
  cases h'' with h_length h_distinct
  have : (∑ k in finset.Icc 1 piles, k) = 45 := by sorry
  show balls < 45 from by sorry
  exact this h_h

end ball_distribution_impossible_l512_512145


namespace mandy_cinnamon_nutmeg_difference_l512_512616

theorem mandy_cinnamon_nutmeg_difference :
  0.67 - 0.5 = 0.17 :=
by
  sorry

end mandy_cinnamon_nutmeg_difference_l512_512616


namespace angle_sum_unique_l512_512917

theorem angle_sum_unique (α β : ℝ) (h1 : α ∈ Set.Ioo (π / 2) π) (h2 : β ∈ Set.Ioo (π / 2) π) 
  (h3 : Real.tan α + Real.tan β - Real.tan α * Real.tan β + 1 = 0) : 
  α + β = 7 * π / 4 :=
sorry

end angle_sum_unique_l512_512917


namespace probability_of_same_color_correct_l512_512352

/-- Define events and their probabilities based on the given conditions --/
def probability_of_two_black_stones : ℚ := 1 / 7
def probability_of_two_white_stones : ℚ := 12 / 35

/-- Define the probability of drawing two stones of the same color --/
def probability_of_two_same_color_stones : ℚ :=
  probability_of_two_black_stones + probability_of_two_white_stones

theorem probability_of_same_color_correct :
  probability_of_two_same_color_stones = 17 / 35 :=
by
  -- We only set up the theorem, the proof is not considered here
  sorry

end probability_of_same_color_correct_l512_512352


namespace possible_value_of_xyz_l512_512972

-- Definitions of the conditions
variables (x y z : ℂ)

def condition1 : Prop := 2 * x * y + 5 * y = -20
def condition2 : Prop := 2 * y * z + 5 * z = -20
def condition3 : Prop := 2 * z * x + 5 * x = -20

-- Main statement to prove
theorem possible_value_of_xyz (h1 : condition1 x y z) (h2 : condition2 x y z) (h3 : condition3 x y z) : x * y * z = 25 :=
by sorry

end possible_value_of_xyz_l512_512972


namespace limit_derivative_f_at_1_l512_512508

noncomputable def f : ℝ → ℝ := fun x => x^2

theorem limit_derivative_f_at_1 : (Real.limit (fun Δx => (f 1 + Δx - f 1) / Δx) 0 = f' 1) :=
by
  sorry

end limit_derivative_f_at_1_l512_512508


namespace luke_stickers_l512_512986

theorem luke_stickers : 
  let initial_stickers := 20 in
  let bought_stickers := 12 in
  let birthday_stickers := 20 in
  let given_to_sister := 5 in
  let used_for_card := 8 in
  (initial_stickers + bought_stickers + birthday_stickers - given_to_sister - used_for_card) = 39 :=
by
  sorry

end luke_stickers_l512_512986


namespace cheapest_pie_l512_512408

def cost_flour : ℝ := 2
def cost_sugar : ℝ := 1
def cost_eggs_butter : ℝ := 1.5
def cost_crust : ℝ := cost_flour + cost_sugar + cost_eggs_butter

def weight_blueberries : ℝ := 3
def container_weight : ℝ := 0.5 -- 8 oz in pounds
def price_per_blueberry_container : ℝ := 2.25
def cost_blueberries (weight: ℝ) (container_weight: ℝ) (price_per_container: ℝ) : ℝ :=
  (weight / container_weight) * price_per_container

def weight_cherries : ℝ := 4
def price_cherry_bag : ℝ := 14

def cost_blueberry_pie : ℝ := cost_crust + cost_blueberries weight_blueberries container_weight price_per_blueberry_container
def cost_cherry_pie : ℝ := cost_crust + price_cherry_bag

theorem cheapest_pie : min cost_blueberry_pie cost_cherry_pie = 18 := by
  sorry

end cheapest_pie_l512_512408


namespace leaves_blew_away_l512_512993

theorem leaves_blew_away :
  let initial_leaves := 5678
  let leaves_left := 1432
  initial_leaves - leaves_left = 4246 := by
  let initial_leaves := 5678
  let leaves_left := 1432
  show initial_leaves - leaves_left = 4246
  calc
    initial_leaves - leaves_left = 5678 - 1432 : by rw [initial_leaves, leaves_left]
    ... = 4246 : by norm_num

end leaves_blew_away_l512_512993


namespace div_fraction_fraction_division_eq_l512_512707

theorem div_fraction (a b : ℕ) (h : b ≠ 0) : (a : ℚ) / b = (a : ℚ) * (1 / (b : ℚ)) := 
by sorry

theorem fraction_division_eq : (3 : ℚ) / 7 / 4 = 3 / 28 := 
by 
  calc
    (3 : ℚ) / 7 / 4 = (3 / 7) * (1 / 4) : by rw [div_fraction] 
                ... = 3 / 28            : by normalization_tactic -- Use appropriate tactic for simplification
                ... = 3 / 28            : by rfl

end div_fraction_fraction_division_eq_l512_512707


namespace find_fixed_point_of_g_l512_512787

open Complex

noncomputable def g (z : ℂ) : ℂ :=
  ((-2 + 2 * I * real.sqrt 3) * z + (-3 * real.sqrt 3 - 27 * I)) / 3

theorem find_fixed_point_of_g :
  let d : ℂ := (-69 * real.sqrt 3 / 37) - (141 * I / 37) in
  g d = d :=
by
  intro d
  have d_def : d = (-69 * real.sqrt 3 / 37) - (141 * I / 37) := rfl
  sorry

end find_fixed_point_of_g_l512_512787


namespace cone_volume_div_pi_l512_512363

noncomputable def sector_arc_length (r : ℝ) (θ : ℝ) : ℝ := (θ / 360) * (2 * ℝ.pi * r)

noncomputable def cone_base_radius (arc_length : ℝ) : ℝ := arc_length / (2 * ℝ.pi)

noncomputable def cone_height (r : ℝ) (base_radius : ℝ) : ℝ := 
  real.sqrt (r^2 - base_radius^2)

noncomputable def cone_volume (base_radius : ℝ) (height : ℝ) : ℝ := 
  (1 / 3) * ℝ.pi * base_radius^2 * height

theorem cone_volume_div_pi (base_radius : ℝ := 32 / 3) (height : ℝ := 8 * real.sqrt 10 / 3) :
  (cone_volume base_radius height) / ℝ.pi = 8192 * real.sqrt 10 / 81 :=
by 
  sorry

end cone_volume_div_pi_l512_512363


namespace find_other_number_l512_512287

theorem find_other_number (LCM HCF number1 number2 : ℕ) 
  (hLCM : LCM = 7700) 
  (hHCF : HCF = 11) 
  (hNumber1 : number1 = 308)
  (hProductEquality : number1 * number2 = LCM * HCF) :
  number2 = 275 :=
by
  -- proof omitted
  sorry

end find_other_number_l512_512287


namespace sequence_limit_l512_512786

noncomputable def c : ℕ → ℝ
| 1     := 0
| (n+1) := ((n / (n + 1)) ^ 2 * c (n + 1)) + (6 * n / ((n + 1) ^ 2))

theorem sequence_limit (c : ℕ → ℝ)
  (h₁ : c 1 = 0)
  (h𝑛 : ∀ n ≥ 1, c (n + 1) = ((n / (n + 1)) ^ 2 * c n) + (6 * n / ((n + 1) ^ 2))) :
  filter.tendsto c filter.at_top (nhds 3) := sorry

end sequence_limit_l512_512786


namespace solve_equation_l512_512259

theorem solve_equation : ∀ (x : ℝ), x ≠ 1 → (3 / (x - 1) = 5 + 3 * x / (1 - x)) → x = 4 :=
by
  intros x hx heq
  -- sorry to skip the proof
  sorry

end solve_equation_l512_512259


namespace sequence_bn_general_formula_sum_first_20_terms_l512_512845

theorem sequence_bn_general_formula (a : ℕ → ℕ)
  (h1 : a 1 = 1)
  (h2 : ∀ n, if n % 2 = 1 then a (n + 1) = a n + 1 else a (n + 1) = a n + 2)
  (b : ℕ → ℕ) :
  b 1 = 2 ∧ b 2 = 5 ∧ ∀ n, b n = 3 * n - 1 :=
by
  sorry

theorem sum_first_20_terms (a : ℕ → ℕ)
  (h1 : a 1 = 1)
  (h2 : ∀ n, if n % 2 = 1 then a (n + 1) = a n + 1 else a (n + 1) = a n + 2) :
  (∑ i in finset.range 20, a (i + 1)) = 300 :=
by
  sorry

end sequence_bn_general_formula_sum_first_20_terms_l512_512845


namespace circumcircle_intersection_point_l512_512868

-- Definitions based on the conditions given
variable {A B C M N O R : Type}
variable [Triangle ABC] [Acute △ ABC] [IsDiameter BC (circle_MN : Circle BC)]
variable {M : Point AB} [OnCircle M circle_MN]
variable {N : Point AC} [OnCircle N circle_MN]
variable {O : Point BC} [Midpoint O B C]
variable {R : Point (triangle_intersection A Bisector)] [Intersection R (angle_bisector BAC) (angle_bisector MON)]

-- Statement to prove
theorem circumcircle_intersection_point (h : IsAcuteTriange ABC ∧ AB ≠ AC ∧ IsDiameter BC (circle_MN : Circle BC),
                                         hM : OnCircle M circle_MN, hN : OnCircle N circle_MN,
                                         hO : Midpoint O B C,
                                         hR : Intersection R (angle_bisector BAC) (angle_bisector MON)) :
  ∃ P : Point BC, OnCircle P (Circumcircle B M R) ∧ OnCircle P (Circumcircle C N R) :=
sorry

end circumcircle_intersection_point_l512_512868


namespace sum_b_p_eq_22032_l512_512603

def b (p : ℕ) : ℕ :=
  if h : (∃ k : ℕ, |k - Nat.sqrt p| * |k - Nat.sqrt p| < 1) then
    Classical.choose h
  else 0  -- This case won't occur due to the condition in the problem statement

theorem sum_b_p_eq_22032 : ∑ p in Finset.range 1000, b (p + 1) = 22032 := by
  sorry

end sum_b_p_eq_22032_l512_512603


namespace quadratic_expression_and_intersections_l512_512887

noncomputable def quadratic_eq_expression (a b c : ℝ) : Prop :=
  ∃ a b c : ℝ, (a * (1:ℝ) ^ 2 + b * (1:ℝ) + c = -3) ∧ (4 * a + 2 * b + c = - 5 / 2) ∧ (b = -2 * a) ∧ (c = -5 / 2) ∧ (a = 1 / 2)

noncomputable def find_m (a b c : ℝ) : Prop :=
  ∀ x m : ℝ, (a * (-2:ℝ)^2 + b * (-2:ℝ) + c = m) → (a * (4:ℝ) + b * (4:ℝ) + c = m) → (6:ℝ) = abs (x - (-2:ℝ)) → m = 3 / 2

noncomputable def y_range (a b c : ℝ) : Prop :=
  ∀ x y : ℝ, 
  (x^2 * a + x * b + c >= -3) ∧ 
  (x^2 * a + x * b + c < 5) ↔ (-3 < x ∧ x < 3)

theorem quadratic_expression_and_intersections 
  (a b c : ℝ) (h1 : quadratic_eq_expression a b c) (h2 : find_m a b c) : y_range a b c :=
  sorry

end quadratic_expression_and_intersections_l512_512887


namespace solve_fractional_equation_l512_512255

theorem solve_fractional_equation (x : ℝ) (h₁ : x ≠ 1) (h₂ : x ≠ -1) :
  (3 / (x - 1) = 5 + 3 * x / (1 - x)) → (x = 4) :=
by 
  intro h
  -- proof steps would go here
  sorry

end solve_fractional_equation_l512_512255


namespace shaded_region_area_l512_512699

noncomputable def radius1 : ℝ := 4
noncomputable def radius2 : ℝ := 5
noncomputable def distance_between_centers : ℝ := 6
noncomputable def large_circle_radius := radius1 + radius2 + distance_between_centers
noncomputable def area_large_circle := π * large_circle_radius^2
noncomputable def area_small_circle1 := π * radius1^2
noncomputable def area_small_circle2 := π * radius2^2
noncomputable def total_area_small_circles := area_small_circle1 + area_small_circle2

theorem shaded_region_area : area_large_circle - total_area_small_circles = 184 * π :=
by
  sorry

end shaded_region_area_l512_512699


namespace union_of_sets_l512_512876

-- Define the sets and conditions
variables (a b : ℝ)
variables (A : Set ℝ) (B : Set ℝ)
variables (log2 : ℝ → ℝ)

-- State the assumptions and final proof goal
theorem union_of_sets (h_inter : A ∩ B = {2}) 
                      (h_A : A = {3, log2 a}) 
                      (h_B : B = {a, b}) 
                      (h_log2 : log2 4 = 2) :
  A ∪ B = {2, 3, 4} :=
by {
    sorry
}

end union_of_sets_l512_512876


namespace tan_of_sin_l512_512545

variable (a b x : ℝ)

theorem tan_of_sin (h1 : a > b) (h2 : b > 0) (h3 : 0 < x) (h4 : x < π / 2) 
  (h5 : sin x = 2 * a * b / sqrt (4 * a ^ 2 * b ^ 2 + (a ^ 2 + b ^ 2) ^ 2)) :
  tan x = 2 * a * b / (a ^ 2 + b ^ 2) :=
sorry

end tan_of_sin_l512_512545


namespace product_of_cosine_elements_l512_512013

-- Definitions for the problem conditions
def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + d

def S (a : ℕ → ℝ) (d : ℝ) : Set ℝ :=
  {x | ∃ n : ℕ, x = Real.cos (a n)}

-- Main theorem statement
theorem product_of_cosine_elements 
  (a : ℕ → ℝ) 
  (d : ℝ) 
  (h_seq : arithmetic_sequence a (2 * Real.pi / 3))
  (h_S_elements : (S a (2 * Real.pi / 3)).card = 2) 
  (h_S_contains : ∃ a b, a ≠ b ∧ S a (2 * Real.pi / 3) = {a, b}) :
  let (a, b) := Classical.choose (h_S_contains) in
  a * b = -1 / 2 :=
by
  sorry

end product_of_cosine_elements_l512_512013


namespace square_sandbox_length_width_l512_512588

theorem square_sandbox_length_width (cost_per_bag : ℝ) (bags_coverage : ℝ) (total_cost : ℝ) (h1 : cost_per_bag = 4) (h2 : bags_coverage = 3) (h3 : total_cost = 12) :
  let num_bags := total_cost / cost_per_bag in
  let total_coverage := num_bags * bags_coverage in
  let side_length := real.sqrt total_coverage in
  side_length = 3 :=
by {
  sorry
}

end square_sandbox_length_width_l512_512588


namespace find_p_q_d_l512_512660

noncomputable def cubic_polynomial_real_root (p q d : ℕ) (x : ℝ) : Prop :=
  27 * x^3 - 12 * x^2 - 4 * x - 1 = 0 ∧ x = (p^(1/3) + q^(1/3) + 1) / d ∧
  p > 0 ∧ q > 0 ∧ d > 0

theorem find_p_q_d :
  ∃ (p q d : ℕ), cubic_polynomial_real_root p q d 1 ∧ p + q + d = 3 :=
by
  sorry

end find_p_q_d_l512_512660


namespace eccentricity_difference_l512_512757

variables {a b : ℝ} (h₁ : a > 0) (h₂ : b > 0)

-- Define the hyperbola equation C
def hyperbola := {p : ℝ × ℝ | (p.1^2 / a^2) - (p.2^2 / b^2) = 1}

-- Define the eccentricity function f
def eccentricity (θ : ℝ) : ℝ :=
  -- Placeholder definition, should be replaced with the actual function
  sorry

-- Given values for theta, compute the respective eccentricities
def f_theta_2π3 : ℝ := eccentricity (2 * π / 3)
def f_theta_π3 : ℝ := eccentricity (π / 3)

-- Prove the required expression
theorem eccentricity_difference : f_theta_2π3 - f_theta_π3 = 2 * sqrt 3 / 3 :=
by {
  -- Placeholder for the actual proof
  sorry
}

end eccentricity_difference_l512_512757


namespace smallest_sum_four_consecutive_primes_div_by_5_l512_512442

def is_prime (n : ℕ) : Prop := nat.prime n

def consecutive_primes (a b c d : ℕ) : Prop :=
  is_prime a ∧ is_prime b ∧ is_prime c ∧ is_prime d ∧ (b = a + 1) ∧ (c = b + 1) ∧ (d = c + 1)

theorem smallest_sum_four_consecutive_primes_div_by_5 :
  ∃ a b c d : ℕ, consecutive_primes a b c d ∧ (a + b + c + d) % 5 = 0 ∧ (a + b + c + d = 60) := 
sorry

end smallest_sum_four_consecutive_primes_div_by_5_l512_512442


namespace ad_equals_two_l512_512483

noncomputable def geometric_sequence (a b c d : ℝ) : Prop :=
  (b / a = c / b) ∧ (c / b = d / c)

theorem ad_equals_two (a b c d : ℝ) 
  (h1 : geometric_sequence a b c d) 
  (h2 : ∃ (b c : ℝ), (1, 2) = (b, c) ∧ b = 1 ∧ c = 2) :
  a * d = 2 :=
by
  sorry

end ad_equals_two_l512_512483


namespace sum_of_underlined_numbers_positive_l512_512762

-- Define a type for the sequence of numbers
variable {α : Type*} [LinearOrderedField α]

/-- Given a sequence of numbers satisfying the underlining conditions, 
    prove the sum of the underlined numbers is positive -/
theorem sum_of_underlined_numbers_positive 
  (a : ℕ → α) -- a sequence of real numbers
  (n : ℕ) -- length of the sequence
  (h1 : ∀ i, 0 ≤ i ∧ i < n → a i > 0 → a i underlined) -- each positive number is underlined
  (h2 : ∀ i k, 0 ≤ i ∧ i + k < n → (a i + a (i + 1) + ... + a (i + k)) > 0 → a i underlined) -- sum condition for underlining
  : ∑ i in (finset.range n).filter (λ i, is_underlined a i), a i > 0 
  := 
sorry

end sum_of_underlined_numbers_positive_l512_512762


namespace sequence_bn_general_formula_sum_first_20_terms_l512_512843

theorem sequence_bn_general_formula (a : ℕ → ℕ)
  (h1 : a 1 = 1)
  (h2 : ∀ n, if n % 2 = 1 then a (n + 1) = a n + 1 else a (n + 1) = a n + 2)
  (b : ℕ → ℕ) :
  b 1 = 2 ∧ b 2 = 5 ∧ ∀ n, b n = 3 * n - 1 :=
by
  sorry

theorem sum_first_20_terms (a : ℕ → ℕ)
  (h1 : a 1 = 1)
  (h2 : ∀ n, if n % 2 = 1 then a (n + 1) = a n + 1 else a (n + 1) = a n + 2) :
  (∑ i in finset.range 20, a (i + 1)) = 300 :=
by
  sorry

end sequence_bn_general_formula_sum_first_20_terms_l512_512843


namespace problem_statement_l512_512115

noncomputable def a : ℝ := Real.log 45 / Real.log 3
noncomputable def b : ℝ := Real.log 45 / Real.log 5

theorem problem_statement : 3 ^ a = 45 ∧ 5 ^ b = 45 → (2 / a + 1 / b = 1) :=
by
  sorry

end problem_statement_l512_512115


namespace solve_fractional_equation_l512_512250

theorem solve_fractional_equation (x : ℝ) (h1 : x ≠ 1) (h2 : x ≠ -1) : 
    (3 / (x - 1) = 5 + 3 * x / (1 - x)) → x = 4 := 
by
  sorry

end solve_fractional_equation_l512_512250


namespace sum_odd_greater_than_even_by_45_l512_512329

def two_digit_numbers : List ℕ := List.range' 10 90

def is_odd (n : ℕ) : Prop := n % 2 = 1

def is_even (n : ℕ) : Prop := n % 2 = 0

def sum_of_odd_numbers (l : List ℕ) : ℕ := l.filter is_odd |>.sum

def sum_of_even_numbers (l : List ℕ) : ℕ := l.filter is_even |>.sum

theorem sum_odd_greater_than_even_by_45 :
  sum_of_odd_numbers two_digit_numbers - sum_of_even_numbers two_digit_numbers = 45 :=
sorry

end sum_odd_greater_than_even_by_45_l512_512329


namespace sample_stats_equal_l512_512861

/-- Let x be a data set of n samples and y be another data set of n samples such that 
    ∀ i, y_i = x_i + c where c is a non-zero constant.
    Prove that the sample standard deviations and the ranges of x and y are the same. -/
theorem sample_stats_equal (n : ℕ) (x y : Fin n → ℝ) (c : ℝ) (h : c ≠ 0)
    (h_y : ∀ i : Fin n, y i = x i + c) :
    (stddev x = stddev y) ∧ (range x = range y) := 
sorry

end sample_stats_equal_l512_512861


namespace cos_C_value_l512_512944

-- Define the main proof problem

theorem cos_C_value (A B C a b c : ℝ) (h₁ : sin A / sin B = 3 / 4) (h₂ : sin B / sin C = 2 / 3) 
  (h₃ : c = 2 * a) (h₄ : b = 4 / 3 * a) : cos C = -11 / 24 := 
by
  sorry

end cos_C_value_l512_512944


namespace floor_100x_l512_512729

noncomputable def x : ℝ :=
  (∑ n in Finset.range 44, Real.cos (n+1) * Real.pi / 180) / 
  (∑ n in Finset.range 44, Real.sin (n+1) * Real.pi / 180)

theorem floor_100x : ⌊100 * x⌋ = 241 :=
by
  sorry

end floor_100x_l512_512729


namespace average_of_first_50_multiples_of_13_l512_512715

def first_term : ℕ := 13
def common_difference : ℕ := 13
def num_terms : ℕ := 50

theorem average_of_first_50_multiples_of_13 :
  (∑ i in Finset.range num_terms, (first_term + common_difference * i)) / num_terms = 331.5 :=
by
  sorry

end average_of_first_50_multiples_of_13_l512_512715


namespace WangGangsSeatLocation_l512_512087

theorem WangGangsSeatLocation :
  ∀ (rows cols : ℕ) (ZY_row ZY_col WG_row WG_col : ℕ),
    rows = 7 →
    cols = 8 →
    (ZY_row, ZY_col) = (2, 4) →
    (WG_row, WG_col) = (5, 8) →
    WG_row = 5 ∧ WG_col = 8 :=
by
  intros rows cols ZY_row ZY_col WG_row WG_col
  intro h_rows h_cols h_ZY h_WG
  rw [h_WG]
  exact ⟨rfl, rfl⟩

end WangGangsSeatLocation_l512_512087


namespace find_b_l512_512889

-- Definitions used in the conditions
def l (k b x : ℝ) : ℝ := k * x + b
def f (a x : ℝ) : ℝ := a * x ^ 2
def g (x : ℝ) : ℝ := Real.exp x
def tangent_point (a : ℝ) : ℕ := 1
def f_at_tangent (a : ℝ) : ℝ := f a 1
def slope_f_at_tangent (a : ℝ) : ℝ := 2 * a

-- Given condition assumptions
variable (a : ℝ) (ha : a > 0)
variable (k b : ℝ)
variable (htangent_f : k = slope_f_at_tangent a)
variable (hl : b = f_at_tangent a - k * tangent_point a)

-- Condition for the line being tangent to g(x)
variable (x0 : ℝ)
variable (htangent_g : k = g x0)
variable (hl_tangent : b = g x0 - k * x0)

-- We want to prove the value of b
theorem find_b : b = - (1 / 2) * Real.exp (3 / 2) := sorry

end find_b_l512_512889


namespace f_at_8_l512_512070

noncomputable def f (x : ℝ) : ℝ := sorry

axiom odd_function (f : ℝ → ℝ) : ∀ x, f(-x) = -f(x)
axiom periodic_function (f : ℝ → ℝ) : ∀ x, f(x+2) = -f(x)

theorem f_at_8 (f : ℝ → ℝ) [odd_function f] [periodic_function f] : f 8 = 0 := 
by
  sorry

end f_at_8_l512_512070


namespace expected_num_games_correct_l512_512200

-- Define the conditions as per the problem statement
def prob_A_win := 2 / 3
def prob_B_win := 1 / 3

-- Define the event of the game stopping conditions
def stop_condition (n : ℕ) (A_wins B_wins : ℕ) : Prop :=
  (A_wins ≥ B_wins + 2 ∨ B_wins ≥ A_wins + 2 ∨ n = 6)

-- Define the expected number of games until the stop condition
noncomputable def expected_number_of_games : ℚ :=
  ∑ n in {2, 4, 6}, n * P(ξ = n)

theorem expected_num_games_correct :
  expected_number_of_games = 266 / 81 :=
sorry

end expected_num_games_correct_l512_512200


namespace cosine_theta_max_val_f_l512_512496

theorem cosine_theta_max_val_f :
  (∃ θ, (∀ x, sin x - 2 * cos x ≤ sin θ - 2 * cos θ) ∧ cos θ = -2 * real.sqrt 5 / 5) :=
sorry

end cosine_theta_max_val_f_l512_512496


namespace cheapest_pie_cost_l512_512411

def crust_cost : ℝ := 2 + 1 + 1.5

def blueberry_pie_cost : ℝ :=
  let blueberries_needed := 3 * 16
  let containers_required := blueberries_needed / 8
  let blueberries_cost := containers_required * 2.25
  crust_cost + blueberries_cost

def cherry_pie_cost : ℝ := crust_cost + 14

theorem cheapest_pie_cost : blueberry_pie_cost = 18 :=
by sorry

end cheapest_pie_cost_l512_512411


namespace squares_different_areas_l512_512333

theorem squares_different_areas :
    ∃ (s1 s2 : ℝ), s1 ≠ s2 ∧ s1 > 0 ∧ s2 > 0 ∧ 
    (area (square s1) ≠ area (square s2)) := 
by
  sorry

end squares_different_areas_l512_512333


namespace original_price_is_100_l512_512586

variable (P : ℝ) -- Declare the original price P as a real number
variable (h : 0.10 * P = 10) -- The condition given in the problem

theorem original_price_is_100 (P : ℝ) (h : 0.10 * P = 10) : P = 100 := by
  sorry

end original_price_is_100_l512_512586


namespace remainder_of_N_l512_512180

theorem remainder_of_N (N : ℕ)
  (h₁ : ∀ x : ℂ, (Polynomial.div (Polynomial.C 166 - ∑ d in (Finset.filter (λ d, d > 0) (Finset.divisors N)), Polynomial.X^d) (Polynomial.X^2 + Polynomial.X + 1)).is_zero)
  : N % 1000 = 672 := 
sorry

end remainder_of_N_l512_512180


namespace hundredth_integer_is_388_l512_512649

def next_int (board : Set ℤ) : ℤ :=
  let possible_sums := { x + y | x ∈ board, y ∈ board, x ≠ y }
  let largest := board.max' ⟨1, begin use 1, simp, end⟩
  let n := (Set.univ.diff (board.union possible_sums)).min' (Set.exists_mem_of_ne_empty (Set.univ.diff (board.union possible_sums)))
  in n

noncomputable def integer_sequence : ℕ → Set ℤ
| 0     := {1, 2, 4, 6}
| (n+1) := let b := integer_sequence n in b ∪ {next_int b}

theorem hundredth_integer_is_388 : (integer_sequence 99).max' ⟨388, begin sorry end⟩ = 388 :=
sorry

end hundredth_integer_is_388_l512_512649


namespace cost_of_fencing_l512_512304

theorem cost_of_fencing
  (length width : ℕ)
  (ratio : 3 * width = 2 * length ∧ length * width = 5766)
  (cost_per_meter_in_paise : ℕ := 50)
  : (cost_per_meter_in_paise / 100 : ℝ) * 2 * (length + width) = 155 := 
by
  -- definitions
  sorry

end cost_of_fencing_l512_512304


namespace abi_suji_age_ratio_l512_512295

theorem abi_suji_age_ratio (A S : ℕ) (h1 : S = 24) 
  (h2 : (A + 3) / (S + 3) = 11 / 9) : A / S = 5 / 4 := 
by 
  sorry

end abi_suji_age_ratio_l512_512295


namespace probability_weight_range_l512_512004

open ProbabilityTheory

variable {X : Type}
variable [MeasureSpace X]

-- Given normal distribution with μ = 50 and σ = 0.1
def normal_distribution : Measure X := sorry

noncomputable def P_49_9_50_1 : ℝ := 0.6826
noncomputable def P_49_8_50_2 : ℝ := 0.9544

theorem probability_weight_range :
  (μ = 50) → (σ = 0.1) → 
  (P(49.9 < X ∧ X < 50.1) = 0.6826) → 
  (P(49.8 < X ∧ X < 50.2) = 0.9544) →
  P(49.8 < X ∧ X < 50.1) = 0.8185 :=
by
  intros
  sorry

end probability_weight_range_l512_512004


namespace minimum_sum_of_squares_l512_512604

/-- Given positive real numbers x1, x2, x3 such that x1 + 2*x2 + 3*x3 = 60,
    prove that the minimum value of x1^2 + x2^2 + x3^2 is 1800 / 7.
-/
theorem minimum_sum_of_squares (x1 x2 x3 : ℝ) (h1 : x1 > 0) (h2 : x2 > 0) (h3 : x3 > 0) 
  (hsum : x1 + 2 * x2 + 3 * x3 = 60) : 
  x1^2 + x2^2 + x3^2 ≥ 1800 / 7 := 
begin
    sorry -- Proof not required as per instructions.
end

end minimum_sum_of_squares_l512_512604


namespace max_value_of_A_l512_512053

theorem max_value_of_A (n : ℕ) (x : Fin n → ℝ) (hx : ∀ i, 0 < x i ∧ x i < 1) :
  let A := (∑ i, (1 - x i) ^ (1 / 4)) / (∑ i, 1 / (x i ^ (1 / 4))) in
  A ≤ (Real.sqrt 2) / 2 :=
by
  sorry

end max_value_of_A_l512_512053


namespace arithmetic_sequence_solution_l512_512102

theorem arithmetic_sequence_solution
  (a b c : ℤ)
  (h1 : a + 1 = b - a)
  (h2 : b - a = c - b)
  (h3 : c - b = -9 - c) :
  b = -5 ∧ a * c = 21 :=
by sorry

end arithmetic_sequence_solution_l512_512102


namespace circle_area_l512_512572

theorem circle_area (x y : ℝ) :
  2 * x ^ 2 + 2 * y ^ 2 + 10 * x - 6 * y - 18 = 0 →
  ∃ (r : ℝ), r ^ 2 = 35 / 2 ∧ real.pi * r ^ 2 = (35 / 2) * real.pi :=
by sorry

end circle_area_l512_512572


namespace intersection_A_B_l512_512595

def A : Set ℕ := {1, 2, 3, 4, 5}
def B : Set ℤ := {x | ∃ (n : ℕ), n ∈ A ∧ x * x = n}

theorem intersection_A_B : A ∩ B = {1, 2} :=
by
  sorry

end intersection_A_B_l512_512595


namespace range_of_m_l512_512514

noncomputable def f (x : ℝ) : ℝ := |x - 3| - 2
noncomputable def g (x : ℝ) : ℝ := -|x + 1| + 4

theorem range_of_m :
  (∀ x : ℝ, f x - g x ≥ m + 1) ↔ m ≤ -3 :=
by sorry

end range_of_m_l512_512514


namespace range_of_a_l512_512185

theorem range_of_a (a : ℝ) (h1 : 0 < a) :
  (∀ x : ℝ, x^2 - 4*a*x + 3*a^2 ≤ 0 → x^2 - x - 6 ≤ 0) ∧
  (¬ (∀ x : ℝ, x^2 - x - 6 ≤ 0 → x^2 - 4*a*x + 3*a^2 ≤ 0)) →
  0 < a ∧ a ≤ 1 :=
sorry

end range_of_a_l512_512185


namespace sum_nine_terms_of_arithmetic_sequence_l512_512071

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

def sum_of_first_n_terms (a : ℕ → ℝ) (S : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, S n = (n * (a 0 + a (n - 1))) / 2

theorem sum_nine_terms_of_arithmetic_sequence
  (a : ℕ → ℝ) (S : ℕ → ℝ)
  (h1 : arithmetic_sequence a)
  (h2 : sum_of_first_n_terms a S)
  (h3 : a 5 = 7) :
  S 9 = 63 := by
  sorry

end sum_nine_terms_of_arithmetic_sequence_l512_512071


namespace number_of_points_in_second_quadrant_l512_512108

theorem number_of_points_in_second_quadrant :
  let x_range := (-10 : ℤ) :: List.range 21,
      y_range := (-10 : ℤ) :: List.range 21,
      points := (x_range.product y_range).filter (λ p, p.1 < 0 ∧ p.2 > 0) in
  points.length = 100 :=
by
  let x_range := (-10 : ℤ) :: List.range 21;
  let y_range := (-10 : ℤ) :: List.range 21;
  let points := (x_range.product y_range).filter (λ p, p.1 < 0 ∧ p.2 > 0);
  exact Iff.mp list.length_eq_of_permutations_permutation sorry

end number_of_points_in_second_quadrant_l512_512108


namespace center_of_hyperbola_l512_512809

theorem center_of_hyperbola : 
  ∃ (h k : ℚ), 
  (h = 4/3) ∧ (k = 1) ∧ 
  ((∃ y x : ℚ, (2*y - 2)^2 / 5^2 - (3*x - 4)^2 / 4^2 = 1) ↔ 
  ((y - k)^2 / (5/2)^2 - (x - h)^2 / (4/3)^2 = 1)) :=
by
  use 4/3, 1
  split;
  intros;
  sorry

end center_of_hyperbola_l512_512809


namespace determine_a_l512_512550

theorem determine_a (a x : ℝ) (h : x = 1) (h_eq : a * x + 2 * x = 3) : a = 1 :=
by
  subst h
  simp at h_eq
  linarith

end determine_a_l512_512550


namespace cost_per_person_is_correct_l512_512281

-- Define the given conditions
def fee_per_30_minutes : ℕ := 4000
def bikes : ℕ := 4
def hours : ℕ := 3
def people : ℕ := 6

-- Calculate the correct answer based on the given conditions
noncomputable def cost_per_person : ℕ :=
  let fee_per_hour := 2 * fee_per_30_minutes
  let fee_per_3_hours := hours * fee_per_hour
  let total_cost := bikes * fee_per_3_hours
  total_cost / people

-- The theorem to be proved
theorem cost_per_person_is_correct : cost_per_person = 16000 := sorry

end cost_per_person_is_correct_l512_512281


namespace cosA_plus_sinC_range_l512_512131

theorem cosA_plus_sinC_range (A B C : ℝ) 
  (hA : 0 < A ∧ A < π / 2) 
  (hB : B = π / 6) 
  (hC : 0 < C ∧ C < π / 2) 
  (hABC : A + B + C = π) : 
  (√3 / 2 < cos A + sin C ∧ cos A + sin C < 3 / 2) :=
sorry

end cosA_plus_sinC_range_l512_512131


namespace sample_standard_deviation_same_sample_ranges_same_l512_512864

variables {n : ℕ} (x y : Fin n → ℝ) (c : ℝ)
  (h_y : ∀ i, y i = x i + c)
  (h_c_ne_zero : c ≠ 0)

-- Statement for standard deviations being the same
theorem sample_standard_deviation_same :
  let mean (s : Fin n → ℝ) := (∑ i, s i) / n
  in let stddev (s : Fin n → ℝ) := sqrt ((∑ i, (s i - mean s) ^ 2) / n)
  in stddev x = stddev y := 
sorry

-- Statement for ranges being the same
theorem sample_ranges_same :
  let range (s : Fin n → ℝ) := (Finset.univ.sup s) - (Finset.univ.inf s)
  in range x = range y :=
sorry

end sample_standard_deviation_same_sample_ranges_same_l512_512864


namespace product_of_cosine_values_l512_512043

noncomputable theory
open_locale classical

def arithmetic_sequence (a d : ℝ) (n : ℕ) : ℝ := a + (n - 1) * d

theorem product_of_cosine_values (a b : ℝ) 
    (h : ∃ a1 : ℝ, ∀ n : ℕ+, ∃ a b : ℝ, 
         S = {cos (arithmetic_sequence a1 (2*π/3) n) | n ∈ ℕ*} ∧
         S = {a, b}) : a * b = -1/2 :=
begin
  sorry
end

end product_of_cosine_values_l512_512043


namespace cube_volume_edge_length_range_l512_512927

theorem cube_volume_edge_length_range (a : ℝ) (h : a^3 = 9) : 2 < a ∧ a < 2.5 :=
by {
    -- proof will go here
    sorry
}

end cube_volume_edge_length_range_l512_512927


namespace eccentricity_of_hyperbola_l512_512905

theorem eccentricity_of_hyperbola 
  (a b : ℝ) 
  (ha : a > 0) 
  (hb : b > 0) 
  (h_asymptote : (b / a = 5 / 3)) : 
  ∃ e : ℝ, e = (√34) / 3 :=
by
  sorry

end eccentricity_of_hyperbola_l512_512905


namespace amount_after_a_year_l512_512656

def initial_amount : ℝ := 90
def interest_rate : ℝ := 0.10

theorem amount_after_a_year : initial_amount * (1 + interest_rate) = 99 := 
by
  -- Here 'sorry' indicates that the proof is not provided.
  sorry

end amount_after_a_year_l512_512656


namespace sides_of_triangle_AED_l512_512665

theorem sides_of_triangle_AED (AB BC CD AD : ℝ) (h1 : AB = 3) (h2 : BC = 10) (h3 : CD = 4) (h4 : AD = 12) : 
  (∃ BE CE : ℝ, BE = 2.5 ∧ CE = 3.33) :=
by
  -- Conditions given
  have h_AB := h1
  have h_BC := h2
  have h_CD := h3
  have h_AD := h4
  -- We skip the detailed steps and just state the final result
  existsi (5/6 * 3)
  existsi (5/6 * 4)
  split
  · exact 2.5
  · exact 3.33
  sorry

end sides_of_triangle_AED_l512_512665


namespace lunchroom_total_people_l512_512138

theorem lunchroom_total_people (n a1 d teachers : ℕ) 
  (h_n : n = 34) 
  (h_a1 : a1 = 6) 
  (h_d : d = 1) 
  (h_teachers : teachers = 5) : 
  let a_n := a1 + (n - 1) * d in
  let students := n * (a1 + a_n) / 2 in
  students + teachers = 770 :=
by
  -- Placeholder for the proof logic
  sorry

end lunchroom_total_people_l512_512138


namespace hyperbola_eccentricity_proof_l512_512877

-- Define the hyperbola equation and necessary conditions
def hyperbola (a b : ℝ) (x y : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ (x^2 / a^2 - y^2 / b^2 = 1)

-- Define the conditions |PF1| + |PF2| = 3b and |PF1| * |PF2| = 9/4 * ab
def hyperbola_conditions (a b : ℝ) (PF1 PF2 : ℝ) : Prop :=
  |PF1| + |PF2| = 3 * b ∧ |PF1| * |PF2| = (9 / 4) * a * b

-- Define the eccentricity of the hyperbola
def hyperbola_eccentricity (a b : ℝ) : ℝ :=
  let c := real.sqrt (a^2 + b^2) in c / a

-- The main theorem to prove the eccentricity is 5/3 given the conditions
theorem hyperbola_eccentricity_proof (a b PF1 PF2 : ℝ) (h1 : hyperbola a b PF1 b) (h2 : hyperbola_conditions a b PF1 PF2) :
  hyperbola_eccentricity a b = 5 / 3 :=
sorry

end hyperbola_eccentricity_proof_l512_512877


namespace find_remainder_l512_512326

variable (x y remainder : ℕ)
variable (h1 : x = 7 * y + 3)
variable (h2 : 2 * x = 18 * y + remainder)
variable (h3 : 11 * y - x = 1)

theorem find_remainder : remainder = 2 := 
by
  sorry

end find_remainder_l512_512326


namespace johns_equation_l512_512587

theorem johns_equation (a b c d e : ℤ) (ha : a = 2) (hb : b = 3) 
  (hc : c = 4) (hd : d = 5) : 
  a - (b - (c * (d - e))) = a - b - c * d + e ↔ e = 8 := 
by
  sorry

end johns_equation_l512_512587


namespace part_one_solution_set_l512_512903

def f (x : ℝ) (λ : ℝ) : ℝ := |x + 2| + λ * |x - 2|

theorem part_one_solution_set (x : ℝ) : f x 3 > 6 ↔ x ∈ (set.Iio 1 ∪ set.Ioi (5 / 2)) :=
by sorry

end part_one_solution_set_l512_512903


namespace votes_cast_l512_512732

theorem votes_cast (V : ℝ) (h1 : ∃ V, (0.65 * V) = (0.35 * V + 2340)) : V = 7800 :=
by
  sorry

end votes_cast_l512_512732


namespace calc_x_equals_condition_l512_512400

theorem calc_x_equals_condition (m n p q x : ℝ) :
  x^2 + (2 * m * p + 2 * n * q) ^ 2 + (2 * m * q - 2 * n * p) ^ 2 = (m ^ 2 + n ^ 2 + p ^ 2 + q ^ 2) ^ 2 →
  x = m ^ 2 + n ^ 2 - p ^ 2 - q ^ 2 ∨ x = - m ^ 2 - n ^ 2 + p ^ 2 + q ^ 2 :=
by
  sorry

end calc_x_equals_condition_l512_512400


namespace find_cost_price_l512_512380

noncomputable def cost_price (C : ℝ) : Prop :=
  let SP1 := 0.90 * C in
  let SP2 := 1.03 * C in
  SP2 = SP1 + 140

theorem find_cost_price : ∃ C : ℝ, cost_price C ∧ C = 1076.92 :=
begin
  use 1076.92,
  unfold cost_price,
  simp,
  linarith,
end

end find_cost_price_l512_512380


namespace find_function_l512_512666

/-- A function f satisfies the equation f(x) + (x + 1/2) * f(1 - x) = 1. -/
def satisfies_equation (f : ℝ → ℝ) : Prop :=
∀ x : ℝ, f x + (x + 1 / 2) * f (1 - x) = 1

/-- We want to prove two things:
 1) f(0) = 2 and f(1) = -2
 2) f(x) =  2 / (1 - 2x) for x ≠ 1/2
 -/
theorem find_function (f : ℝ → ℝ) (h : satisfies_equation f) :
  (f 0 = 2 ∧ f 1 = -2) ∧ (∀ x : ℝ, x ≠ 1 / 2 → f x = 2 / (1 - 2 * x)) ∧ (f (1 / 2) = 1 / 2) :=
by
  sorry

end find_function_l512_512666


namespace Tara_savings_after_one_year_l512_512654

theorem Tara_savings_after_one_year :
  ∀ (initial_amount : ℝ) (interest_rate : ℝ),
    initial_amount = 90 → interest_rate = 0.1 →
    let interest_earned := initial_amount * interest_rate in
    let total_amount := initial_amount + interest_earned in
    total_amount = 99 :=
by
  intros initial_amount interest_rate h_initial_amount h_interest_rate
  rw [h_initial_amount, h_interest_rate]
  let interest_earned := initial_amount * interest_rate
  let total_amount := initial_amount + interest_earned
  rw [h_initial_amount, h_interest_rate]
  sorry

end Tara_savings_after_one_year_l512_512654


namespace problem_EF_7sqrt2_l512_512161

theorem problem_EF_7sqrt2 (A B C D E F : Point) (AB : Segment A B) (AC : Segment A C) (AD : Segment A D) 
  (EC : Segment E C) (FD : Segment F D) (hAB_len : AB.length = 26) (hAC_len : AC.length = 1) 
  (hAD_len : AD.length = 8) (hEE_circle : ∀ X, X ∈ Circle A B → ¬X ∈ AB) 
  (hEC_perp : EC ⊥ AB) (hFD_perp : FD ⊥ AB) : 
  EF.length = 7 * Real.sqrt 2 := 
sorry

end problem_EF_7sqrt2_l512_512161


namespace intersection_A_B_l512_512875

-- Definition of sets A and B
def A : Set ℝ := { x | x > 1 }
def B : Set ℝ := { y | y > 0 }

-- The proof goal
theorem intersection_A_B : A ∩ B = { x | x > 1 } :=
by sorry

end intersection_A_B_l512_512875


namespace find_vector_l512_512445

noncomputable section

open_locale real_inner_product_space

variables {v : EuclideanSpace ℝ (Fin 2)}

theorem find_vector :
    let v := ![13, -6]
    (v - ![9, 3]) ⬝ ![3, 1] = 0 ∧
    (v - ![8, 4]) ⬝ ![4, 2] = 0 :=
by
    let v := ![13, -6]
    split
    exact sorry
    exact sorry

end find_vector_l512_512445


namespace variance_calculation_l512_512895

noncomputable def arithmetic_seq (a1 : ℤ) (d : ℤ) (n : ℕ) : ℤ :=
  a1 + (n - 1) * d

noncomputable def random_variable (a1 : ℤ) (d : ℤ) : ℕ → ℤ
  | 1 => a1
  | n + 1 => random_variable a1 d n + d

theorem variance_calculation (a1 : ℤ) :
  let seq := λ n, arithmetic_seq a1 3 (n + 1)
  let ξ := random_variable a1 3
  (ξ 0 = a1) →
  (ξ 1 = a1 + 3) →
  (ξ 2 = a1 + 6) →
  (ξ 3 = a1 + 9) →
  (ξ 4 = a1 + 12) →
  (ξ 5 = a1 + 15) →
  (ξ 6 = a1 + 18) →
  (ξ 7 = a1 + 21) →
  (ξ 8 = a1 + 24) →
  ( 1 / (9 : ℝ) * 
    ((0 - 12)^2 + (3 - 12)^2 + (6 - 12)^2 + (9 - 12)^2 +
     (12 - 12)^2 + (15 - 12)^2 + (18 - 12)^2 + (21 - 12)^2 + 
     (24 - 12)^2) = 60) :=
by
  sorry

end variance_calculation_l512_512895


namespace arithmetic_sequence_middle_term_is_average_l512_512938

theorem arithmetic_sequence_middle_term_is_average :
  ∀ (a b : ℤ), 
  let e := (a + b) / 2 in
  a = 23 → b = 53 →  e = 38 :=
by
  intros a b e h₁ h₂
  rw [h₁, h₂]
  have : e = (23 + 53) / 2 := rfl
  rw this
  simp
  sorry

end arithmetic_sequence_middle_term_is_average_l512_512938


namespace n_P_deg_leq_2_l512_512181

noncomputable def n_P (P : Polynomial ℤ) : ℕ :=
  P.roots.count 
    (λ k, P.eval k = 1 ∨ P.eval k = -1)

theorem n_P_deg_leq_2 (P : Polynomial ℤ) (h : P.degree > 0) : 
  n_P P - P.degree.natDegree ≤ 2 :=
sorry

end n_P_deg_leq_2_l512_512181


namespace general_equations_and_min_area_l512_512487

noncomputable def circle_parametric (θ : ℝ) : ℝ × ℝ := (3 + 2 * Real.cos θ, -4 + 2 * Real.sin θ)
def line_polar (ρ θ : ℝ) : Prop := ρ * Real.cos (θ - π/4) = Real.sqrt 2

theorem general_equations_and_min_area :
  (forall θ, let (x, y) := circle_parametric θ in (x - 3)^2 + (y + 4)^2 = 4) ∧
  (forall (x y : ℝ), (∃ ρ θ, line_polar ρ θ ∧ ρ * Real.cos θ = x ∧ ρ * Real.sin θ = y) ↔ (x + y = 2)) ∧
  (∃ M : ℝ × ℝ, M.1 + M.2 = 2 ∧ 
                 let C := (3, -4) in 
                 let d := 3 * Real.sqrt 2 / 2 in 
                 dist M C ≥ d ∧ 
                 2 * Real.sqrt (dist M C ^ 2 - 4) = Real.sqrt 2) :=
by
  sorry

end general_equations_and_min_area_l512_512487


namespace validArrangements_l512_512124

def countArrangements (A B : ℕ) (team : Finset ℕ) := (team.card = 6) → 
  ∃ ways : ℕ, ways = 252 ∧ 
  all_elements_distinct team ∧
  (∀ l1 l2 l3 l4,  
   {l1, l2, l3, l4} ⊆ team →
   l1 ≠ A →
   l4 ≠ B →
   ((countOrigins team) - (countOrigins {A} ∪ countOrigins {B})) = 4 )
-- The set {l1, l2, l3, l4} represents the selected legs

theorem validArrangements :
  countArrangements 0 1 {0, 1, 2, 3, 4, 5} :=
begin
  sorry
end

end validArrangements_l512_512124


namespace intersection_A_B_l512_512596

def A : Set ℕ := {1, 2, 3, 4, 5}
def B : Set ℤ := {x | ∃ (n : ℕ), n ∈ A ∧ x * x = n}

theorem intersection_A_B : A ∩ B = {1, 2} :=
by
  sorry

end intersection_A_B_l512_512596


namespace product_of_cosine_elements_l512_512014

-- Definitions for the problem conditions
def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + d

def S (a : ℕ → ℝ) (d : ℝ) : Set ℝ :=
  {x | ∃ n : ℕ, x = Real.cos (a n)}

-- Main theorem statement
theorem product_of_cosine_elements 
  (a : ℕ → ℝ) 
  (d : ℝ) 
  (h_seq : arithmetic_sequence a (2 * Real.pi / 3))
  (h_S_elements : (S a (2 * Real.pi / 3)).card = 2) 
  (h_S_contains : ∃ a b, a ≠ b ∧ S a (2 * Real.pi / 3) = {a, b}) :
  let (a, b) := Classical.choose (h_S_contains) in
  a * b = -1 / 2 :=
by
  sorry

end product_of_cosine_elements_l512_512014


namespace tangent_line_at_P_on_circle_l512_512663

noncomputable def tangent_line_of_circle_through_point (x y : ℝ) (h1 : (x - 1) ^ 2 + y ^ 2 = 5) : Prop :=
  x + 2 * y - 6 = 0

theorem tangent_line_at_P_on_circle :
  tangent_line_of_circle_through_point 2 2 ((2 - 1) ^ 2 + 2 ^ 2 = 5) :=
by
  -- The proof should go here
  sorry

end tangent_line_at_P_on_circle_l512_512663


namespace tangent_line_at_0_g_nonpositive_f_relationship_l512_512079

noncomputable def f (x : ℝ) : ℝ := Real.exp x * Real.cos x - x - 1
noncomputable def g (x : ℝ) : ℝ := (fun x => Real.exp x * (Real.cos x - Real.sin x) - 1) x

theorem tangent_line_at_0 : (∀ x, f x - 0 * x = 0) := sorry
theorem g_nonpositive (x : ℝ) (hx : 0 ≤ x ∧ x < Real.pi) : g x ≤ 0 := sorry
theorem f_relationship (m n : ℝ) (hm : 0 < m ∧ m < Real.pi / 2) (hn : 0 < n ∧ n < Real.pi / 2) : 
  f (m + n) - f m < f n := sorry

end tangent_line_at_0_g_nonpositive_f_relationship_l512_512079


namespace cos_set_product_l512_512034

noncomputable def arithmetic_sequence (a1 : ℝ) (d : ℝ) : ℕ → ℝ :=
  λ n, a1 + d * (n - 1)

theorem cos_set_product (a1 : ℝ) (h1 : (arithmetic_sequence a1 (2 * Real.pi / 3) '' {n | n ≥ 1}).to_set.cos.to_finset.card = 2) :
  let S := (arithmetic_sequence a1 (2 * Real.pi / 3) '' {n | n ≥ 1}).to_set.cos.to_finset in 
  (S.to_finset : set ℝ).product = -1 / 2 := sorry

end cos_set_product_l512_512034


namespace propositions_truth_l512_512543

theorem propositions_truth:
  (∀ (a b : ℝ), (∃ 𝜆 : ℝ, b = 𝜆 * a) ↔ (∃! 𝜆 : ℝ, b = 𝜆 * a))
  ∧ (∀ (a b c : ℝ), a * b * c ≠ 0 → (b^2 = a * c ↔ (a = c ∨ a ≠ c)))
  ∧ (∀ (A B : Prop), (A ∧ B = false) ↔ (A ↔ ¬B))
  ∧ (∀ (b c : ℝ), (∀ x : ℝ, (x^2 + b * x + c) = (x^2 + c)) ↔ (b = 0)) :=
sorry

end propositions_truth_l512_512543


namespace indefinite_integral_l512_512343

theorem indefinite_integral :
  ∃ C : ℝ, ∀ x : ℝ,
  ∫ (2 - 4 * x) * sin (2 * x) dx = (2 * x - 1) * cos (2 * x) - sin (2 * x) + C := 
sorry

end indefinite_integral_l512_512343


namespace coefficient_x3_l512_512454

theorem coefficient_x3 (a : ℝ) (h : (Expand (2 - a * x)^6) x^3 = -160) : a = 1 := sorry

end coefficient_x3_l512_512454


namespace product_of_cosine_elements_l512_512010

-- Definitions for the problem conditions
def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + d

def S (a : ℕ → ℝ) (d : ℝ) : Set ℝ :=
  {x | ∃ n : ℕ, x = Real.cos (a n)}

-- Main theorem statement
theorem product_of_cosine_elements 
  (a : ℕ → ℝ) 
  (d : ℝ) 
  (h_seq : arithmetic_sequence a (2 * Real.pi / 3))
  (h_S_elements : (S a (2 * Real.pi / 3)).card = 2) 
  (h_S_contains : ∃ a b, a ≠ b ∧ S a (2 * Real.pi / 3) = {a, b}) :
  let (a, b) := Classical.choose (h_S_contains) in
  a * b = -1 / 2 :=
by
  sorry

end product_of_cosine_elements_l512_512010


namespace commission_calculation_l512_512387

theorem commission_calculation (total_sales : ℝ) (commission_rate : ℝ) : 
  total_sales = 720 →
  commission_rate = 0.025 → 
  (total_sales * commission_rate) = 18 := 
by
  intros h_total_sales h_commission_rate
  rw [h_total_sales, h_commission_rate]
  norm_num
  sorry

end commission_calculation_l512_512387


namespace rebecca_perms_count_l512_512209

/-
Rebecca runs a hair salon. She charges $30 for haircuts, $40 for perms, and $60 for dye jobs, but she has to buy a box of hair dye for $10 to dye every head of hair. 
Today, she has four haircuts, some perms, and two dye jobs scheduled. If she makes $50 in tips, she will have $310 at the end of the day.
Prove that the number of perms she has scheduled equals 1.
-/

theorem rebecca_perms_count : 
  let price_haircut := 30
  let price_perm := 40
  let price_dye_job := 60
  let cost_hair_dye := 10
  let num_haircuts := 4
  let num_dye_jobs := 2
  let tips := 50
  let total_end_day := 310
  (num_haircuts * price_haircut) + (num_dye_jobs * price_dye_job) - (num_dye_jobs * cost_hair_dye) + tips + (perm_count * price_perm) = total_end_day →  
  perm_count = 1 :=
by 
  intros price_haircut price_perm price_dye_job cost_hair_dye num_haircuts num_dye_jobs tips total_end_day perm_count h
  sorry

end rebecca_perms_count_l512_512209


namespace number_of_ways_to_100_in_4_seconds_l512_512335

-- Define the factorial function recursively
def factorial : ℕ → ℕ
| 0     := 1
| (n + 1) := (n + 1) * factorial n

-- Define the condition that checks if a given sequence of numbers achieves 100 in exactly 4 seconds
def reaches_target (steps: List ℕ) (target : ℕ) : Prop :=
  steps.foldr (λ (x: ℕ) (acc: ℕ), acc + x) 0 = target

-- Define the problem statement
theorem number_of_ways_to_100_in_4_seconds :
  ∃ (ways : Finset (List ℤ)),  ways.card = 36 ∧
  ∀ (step_seq : List ℤ), step_seq ∈ ways → 
  reaches_target step_seq 100 ∧ step_seq.length = 4 :=
sorry

end number_of_ways_to_100_in_4_seconds_l512_512335


namespace product_of_roots_unity_sum_of_lin_comb_l512_512469

noncomputable def roots_of_unity (n : ℕ) : ℕ → ℂ 
| 0 => 1
| k => complex.exp (2 * k * real.pi * complex.I / n)

theorem product_of_roots_unity (n : ℕ) (hn : 0 < n) : 
    (∏ k in finset.range (n - 1), complex.abs (1 - roots_of_unity n (k + 1))) = n - 1 := 
sorry

theorem sum_of_lin_comb (n : ℕ) (hn : 1 < n) (λ : fin n → ℝ):
    (∀ k, 0 ≤ k < n → (∑ i in finset.range (n - 1), λ (i + 1) * (roots_of_unity n k ^ (i + 1)).re) ≥ -1) →
    (∑ i in finset.range (n - 1), λ (i + 1)) ≤ n - 1 :=
sorry

end product_of_roots_unity_sum_of_lin_comb_l512_512469


namespace cube_contains_two_non_overlapping_tetrahedra_l512_512203

theorem cube_contains_two_non_overlapping_tetrahedra (a : ℝ) (hcube : a > 0) :
  ∃ (T1 T2 : set (ℝ × ℝ × ℝ)),
    (∀ x1 x2, x1 ∈ T1 → x2 ∈ T2 → x1 ≠ x2) ∧
    (∀ x, x ∈ T1 → ∃ v w u z, T1 = regular_tetrahedron a v w u z) ∧
    (∀ x, x ∈ T2 → ∃ v w u z, T2 = regular_tetrahedron a v w u z) :=
sorry

def regular_tetrahedron (a : ℝ) (v w u z : ℝ × ℝ × ℝ) : set (ℝ × ℝ × ℝ) :=
-- Definition of a regular tetrahedron with edge length a and vertices v, w, u, z
sorry

end cube_contains_two_non_overlapping_tetrahedra_l512_512203


namespace range_of_x_l512_512879

theorem range_of_x (f : ℝ → ℝ) (h_increasing : ∀ x y, x ≤ y → f x ≤ f y) (h_defined : ∀ x, -1 ≤ x ∧ x ≤ 1)
  (h_condition : ∀ x, f (x-2) < f (1-x)) : ∀ x, 1 ≤ x ∧ x < 3/2 :=
by
  sorry

end range_of_x_l512_512879


namespace angle_BDA_measure_l512_512817

theorem angle_BDA_measure :
  ∀ (ABC BAD BDA ABD : ℝ), 
    ABC = 120 ∧ BAD = 32 
    ∧ (BAD + ABD + BDA = 180)
    ∧ ABD = 180 - ABC
    → BDA = 88 :=
by 
  intros ABC BAD BDA ABD h
  obtain ⟨h₁, h₂, h₃, h₄⟩ := h
  rw [h₄, h₁] at *
  linarith

end angle_BDA_measure_l512_512817


namespace magnitude_of_a_l512_512482

theorem magnitude_of_a (n : ℝ) (h1 : (1:ℝ, n, 1/2) = (1, (-1/2 : ℝ), 1/2)) : 
  ∥(1, n, 1/2 : ℝ × ℝ × ℝ)∥ = sqrt(6) / 2 :=
by
  sorry

end magnitude_of_a_l512_512482


namespace solve_fractional_equation_l512_512253

theorem solve_fractional_equation (x : ℝ) (h₁ : x ≠ 1) (h₂ : x ≠ -1) :
  (3 / (x - 1) = 5 + 3 * x / (1 - x)) → (x = 4) :=
by 
  intro h
  -- proof steps would go here
  sorry

end solve_fractional_equation_l512_512253


namespace ratio_b_a_l512_512579

variable {A B C a b c : ℝ}

-- Given conditions in the problem
axiom angle_sum : A + B + C = Real.pi
axiom arithmetic_sequence : 2 * b = a + c
axiom angle_C : C = 2 * (A + B)
axiom cosine_rule : Real.cos C = (a^2 + b^2 - c^2) / (2 * a * b)

theorem ratio_b_a (h1 : angle_sum) (h2 : arithmetic_sequence) (h3 : angle_C) (h4 : Real.cos C = -1/2) :
  b / a = 5 / 3 :=
by
  sorry

end ratio_b_a_l512_512579


namespace simplify_expression_l512_512642

theorem simplify_expression : 
  (real.sqrt (64 ^ (1/6)) - real.sqrt ((17/2)))^2 = (50 - 8 * real.sqrt 34) / 4 :=
by
  sorry

end simplify_expression_l512_512642


namespace part1_b1_b2_general_formula_part2_sum_first_20_terms_l512_512847

/-- Define the sequence a_n recursively --/
def a : ℕ → ℕ
| 0 := 1
| (n+1) := if n % 2 = 0 then a n + 1 else a n + 2

/-- Define the sequence b_n such that b_n = a_{2n} --/
def b (n : ℕ) : ℕ := a (2 * (n + 1))

/-- State the properties to be proved --/
theorem part1_b1_b2_general_formula :
  b 0 = 2 ∧
  b 1 = 5 ∧
  (∀ n : ℕ, b (n + 1) = 3 * (n + 1) - 1) :=
by
  sorry

theorem part2_sum_first_20_terms :
  (∑ i in finset.range 20, a i) = 300 :=
by
  sorry

end part1_b1_b2_general_formula_part2_sum_first_20_terms_l512_512847


namespace equation_of_line_AB_l512_512201

-- Define point P
structure Point where
  x : ℝ
  y : ℝ

def P : Point := ⟨2, -1⟩

-- Define the circle equation
def circle_eq (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 25

-- Define the center C
def C : Point := ⟨1, 0⟩

-- The equation of line AB we want to verify
def line_AB (P : Point) := P.x - P.y - 3 = 0

-- The theorem to prove
theorem equation_of_line_AB :
  (circle_eq P.x P.y ∧ P = ⟨2, -1⟩ ∧ C = ⟨1, 0⟩) → line_AB P :=
by
  sorry

end equation_of_line_AB_l512_512201


namespace calculate_volume_of_solid_of_revolution_l512_512628

noncomputable def volume_of_solid_of_revolution (b c b' c' : ℝ) : ℝ :=
  (π / 3) * (b' + c') * (b * c' - b' * c)

theorem calculate_volume_of_solid_of_revolution (b c b' c' : ℝ) :
  volume_of_solid_of_revolution b c b' c' =
    (π / 3) * (b' + c') * (b * c' - b' * c) :=
by
  sorry

end calculate_volume_of_solid_of_revolution_l512_512628


namespace external_tangent_length_l512_512060

-- Statement of the problem in Lean 4
theorem external_tangent_length {
  A B C D P Q : Type
} (hABC : RightTriangle A B C) 
  (hCD_alt : Altitude C D A B)
  (hO1_incircle : Incircle O1 A C D)
  (hO2_incircle : Incircle O2 B C D)
  (hP_intersect_BC : ExternalTangentIntersects P O1 O2 B C)
  (hQ_intersect_AC : ExternalTangentIntersects Q O1 O2 A C) :
  PQ_length P Q = AC_length A C + BC_length B C - AB_length A B := 
sorry

end external_tangent_length_l512_512060


namespace spherical_to_rectangular_conversion_l512_512790

/-- Convert a point in spherical coordinates to rectangular coordinates given specific angles and distance -/
theorem spherical_to_rectangular_conversion :
  ∀ (ρ θ φ : ℝ) (x y z : ℝ), 
  ρ = 15 → θ = 225 * (Real.pi / 180) → φ = 45 * (Real.pi / 180) →
  x = ρ * Real.sin φ * Real.cos θ → y = ρ * Real.sin φ * Real.sin θ → z = ρ * Real.cos φ →
  x = -15 / 2 ∧ y = -15 / 2 ∧ z = 15 * Real.sqrt 2 / 2 := by
  sorry

end spherical_to_rectangular_conversion_l512_512790


namespace perpendicular_lines_l512_512974

open EuclideanGeometry

variable {A B C O S : Point}

noncomputable def acute_triangle (ABC : Triangle) : Prop :=
  ∀ (a b c : ℝ), a + b + c = 180 ∧ a > 0 ∧ b > 0 ∧ c > 0 ∧ a < 90 ∧ b < 90 ∧ c < 90

noncomputable def circumcenter (O : Point) (ABC : Triangle) : Prop :=
  ∃ (R : ℝ), circle O R = circumcircle ABC

noncomputable def intersects_second_time 
  (AC : Line) (circumcircle_ABO : Circle) (S : Point) : Prop :=
  S ∈ circumcircle_ABO ∧ S ≠ AC.first_intersection

theorem perpendicular_lines
  (ABC : Triangle) (O : Point) (AC : Line) (circumcircle_ABO : Circle) (S : Point)
  (h1 : acute_triangle ABC)
  (h2 : circumcenter O ABC)
  (h3 : intersects_second_time AC circumcircle_ABO S) :
  is_perpendicular (Line.mk O S) (Line.mk (triangle.vertex_B ABC) (triangle.vertex_C ABC)) :=
sorry

end perpendicular_lines_l512_512974


namespace cost_of_song_book_l512_512146

-- Definitions of the constants:
def cost_of_flute : ℝ := 142.46
def cost_of_music_stand : ℝ := 8.89
def total_spent : ℝ := 158.35

-- Definition of the combined cost of the flute and music stand:
def combined_cost := cost_of_flute + cost_of_music_stand

-- The final theorem to prove that the cost of the song book is $7.00:
theorem cost_of_song_book : total_spent - combined_cost = 7.00 := by
  sorry

end cost_of_song_book_l512_512146


namespace ratio_diagonals_of_squares_l512_512320

variable (d₁ d₂ : ℝ)

theorem ratio_diagonals_of_squares (h : ∃ k : ℝ, d₂ = k * d₁) (h₁ : 1 < k ∧ k < 9) : 
  (∃ k : ℝ, 4 * (d₂ / Real.sqrt 2) = k * 4 * (d₁ / Real.sqrt 2)) → k = 5 := by
  sorry

end ratio_diagonals_of_squares_l512_512320


namespace sum_of_consecutive_integers_with_product_272_l512_512297

theorem sum_of_consecutive_integers_with_product_272 :
    ∃ (x y : ℕ), x * y = 272 ∧ y = x + 1 ∧ x + y = 33 :=
by
  sorry

end sum_of_consecutive_integers_with_product_272_l512_512297


namespace like_terms_l512_512916

theorem like_terms (x y : ℕ) (h1 : x + 1 = 2) (h2 : x + y = 2) : x = 1 ∧ y = 1 :=
by
  sorry

end like_terms_l512_512916


namespace solve_equation_l512_512244

theorem solve_equation (x : ℝ) (h : x ≠ 1 ∧ x ≠ 1) : (3 / (x - 1) = 5 + 3 * x / (1 - x)) ↔ x = 4 := by
  sorry

end solve_equation_l512_512244


namespace common_ratio_of_geometric_series_l512_512391

theorem common_ratio_of_geometric_series (a S r : ℚ) (ha : a = 400) (hS : S = 2500) (hS_eq : S = a / (1 - r)) : r = 21 / 25 :=
by {
  rw [ha, hS] at hS_eq,
  sorry
}

end common_ratio_of_geometric_series_l512_512391


namespace algebraic_expression_value_l512_512644

variable (a : ℝ)

theorem algebraic_expression_value (h : a = Real.sqrt 2) :
  (a / (a - 1)^2) / (1 + 1 / (a - 1)) = Real.sqrt 2 + 1 :=
by
  sorry

end algebraic_expression_value_l512_512644


namespace prime_number_identity_l512_512431

theorem prime_number_identity (p m : ℕ) (h1 : Nat.Prime p) (h2 : m > 0) (h3 : 2 * p^2 + p + 9 = m^2) :
  p = 5 ∧ m = 8 :=
sorry

end prime_number_identity_l512_512431


namespace circle_center_radius_l512_512659

theorem circle_center_radius :
  ∃ (h : ℝ × ℝ) (r : ℝ),
    (h = (1, -3)) ∧ (r = 2) ∧ ∀ x y : ℝ, 
    (x - h.1)^2 + (y - h.2)^2 = 4 → x^2 + y^2 - 2*x + 6*y + 6 = 0 :=
sorry

end circle_center_radius_l512_512659


namespace count_two_digit_multiples_of_nine_l512_512092

-- Define the problem conditions
def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99
def is_multiple_of_nine (n : ℕ) : Prop := n % 9 = 0

-- State the theorem that counts the number of two-digit multiples of 9
theorem count_two_digit_multiples_of_nine : 
  (finset.card (finset.filter (λ n, is_multiple_of_nine n) (finset.range (99 + 1)))).filter is_two_digit = 10 :=
sorry

end count_two_digit_multiples_of_nine_l512_512092


namespace log_sum_log5_50_plus_log5_125_l512_512777

theorem log_sum :
  ∀ (a b : ℝ), log 5 a + log 5 b = log 5 (a * b) :=
begin
  intros a b,
  apply log_mul,
  { exact five_gt_one },
  { exact a },
  { exact b },
end

theorem log5_50_plus_log5_125 :
  log 5 50 + log 5 125 = 7 + log 5 2 :=
by
  have h1 : log 5 50 + log 5 125 = log 5 (50 * 125) := log_sum 50 125
  have h2 : 50 * 125 = 6250 := by norm_num
  rw [h2, log_mul],
  sorry

end log_sum_log5_50_plus_log5_125_l512_512777


namespace sum_of_consecutive_integers_with_product_272_l512_512296

theorem sum_of_consecutive_integers_with_product_272 :
    ∃ (x y : ℕ), x * y = 272 ∧ y = x + 1 ∧ x + y = 33 :=
by
  sorry

end sum_of_consecutive_integers_with_product_272_l512_512296


namespace first_problem_l512_512849

-- Definitions for the first problem
variable {a : ℕ → ℝ}
variable {S : ℕ → ℝ}
variable (h_pos : ∀ n, a n > 0)
variable (h_seq : ∀ n, (a n + 1)^2 = 4 * (S n + 1))

-- Theorem statement for the first problem
theorem first_problem (h_pos : ∀ n, a n > 0) (h_seq : ∀ n, (a n + 1)^2 = 4 * (S n + 1)) :
  ∃ d, ∀ n, a (n + 1) - a n = d := sorry

end first_problem_l512_512849


namespace inequality_solution_l512_512432

theorem inequality_solution (x : ℝ) :
  (7 : ℝ) / 30 + abs (x - 7 / 60) < 11 / 20 ↔ -1 / 5 < x ∧ x < 13 / 30 :=
by
  sorry

end inequality_solution_l512_512432


namespace people_per_bus_l512_512367

def num_vans : ℝ := 6.0
def num_buses : ℝ := 8.0
def people_per_van : ℝ := 6.0
def extra_people : ℝ := 108.0

theorem people_per_bus :
  let people_vans := num_vans * people_per_van
  let people_buses := people_vans + extra_people
  let people_per_bus := people_buses / num_buses
  people_per_bus = 18.0 :=
by 
  sorry

end people_per_bus_l512_512367


namespace max_distance_between_P_and_Q_l512_512489

-- Definitions of the circle and ellipse
def is_on_circle (P : ℝ × ℝ) : Prop := P.1^2 + (P.2 - 6)^2 = 2
def is_on_ellipse (Q : ℝ × ℝ) : Prop := (Q.1^2) / 10 + Q.2^2 = 1

-- The maximum distance between any point on the circle and any point on the ellipse
theorem max_distance_between_P_and_Q :
  ∃ P Q : ℝ × ℝ, is_on_circle P ∧ is_on_ellipse Q ∧ dist P Q = 6 * Real.sqrt 2 :=
sorry

end max_distance_between_P_and_Q_l512_512489


namespace largest_inscribed_square_side_length_ABC_l512_512574

def is_acute (A B C : ℝ) : Prop := A < 90 ∧ B < 90 ∧ C < 90 

def inscribed_square_side_length (a b c : ℝ) (h_a h_b h_c : ℝ) [h_nonneg : a > 0 ∧ b > 0 ∧ c > 0 ∧ h_a > 0 ∧ h_b > 0 ∧ h_c > 0] : ℝ :=
  let x := a * h_a / (h_a + a)
  let y := b * h_b / (h_b + b)
  let z := c * h_c / (h_c + c)
  max x (max y z)

theorem largest_inscribed_square_side_length_ABC :
  ∀ {A B C : ℝ} {a b c : ℝ} (h_A : A > B) (h_B : B > C) (h_sum : A + C = 2 * B) 
    (h_ab : a = 4) (h_bc : b = 5) (h_abc : is_acute A B C) (h_area : a * b * real.sin 120 = 10 * real.sqrt 3),
    inscribed_square_side_length a b c (2 * real.sqrt 3) (10 * real.sqrt 7 / 7) (5 * real.sqrt 3 / 2) = (300 - 160 * real.sqrt 3) / 11 :=
by
  intros
  sorry

end largest_inscribed_square_side_length_ABC_l512_512574


namespace plane_symmetric_points_l512_512784

-- We introduce points D and B.
structure Point :=
  (x : ℝ) (y : ℝ) (z : ℝ)

def D : Point := ⟨-5, 6, -1⟩
def B : Point := ⟨3, 4, 1⟩

-- We denote the equation of the plane as a function: 4x - y + z + 9 = 0 
-- This is expressed as: plane_eqn: 4 * x - y + z + 9 == 0.

theorem plane_symmetric_points :
  ∀ (p : Point),
    (p = D ∨ p = B) ->
    4 * p.x - p.y + p.z + 9 = 0 := 
by
  intro p h
  cases h
  -- We prove the cases when p is equal to D and B respectively.
  rw h
  -- Proof for point D:
  unfold D
  norm_num
  -- Proof for point B:
  rw h
  unfold B
  norm_num
  -- Skip the formal proof details using sorry.
  sorry

end plane_symmetric_points_l512_512784


namespace quadratic_has_two_real_roots_find_k_when_roots_are_integers_l512_512086

theorem quadratic_has_two_real_roots (k : ℝ) (h : k ≠ 0) :
    ∃ x1 x2 : ℝ, k * x1^2 + (2*k + 1) * x1 + 2 = 0 ∧ k * x2^2 + (2*k + 1) * x2 + 2 = 0 :=
by
    have Δ : (2*k + 1)^2 - 4 * k * 2 := (2*k - 1)^2
    have Δ_nonneg : (2*k - 1)^2 ≥ 0 := by apply pow_two_nonneg
    sorry

theorem find_k_when_roots_are_integers (k : ℕ) (hk : k ≠ 0) (hpos : 0 < k)
    (hint : ∃ x1 x2 : ℤ, k * x1^2 + (2*k + 1) * x1 + 2 = 0 ∧ k * x2^2 + (2*k + 1) * x2 + 2 = 0) : k = 1 :=
by
    sorry

end quadratic_has_two_real_roots_find_k_when_roots_are_integers_l512_512086


namespace Cameron_books_proof_l512_512776

noncomputable def Cameron_initial_books :=
  let B : ℕ := 24
  let B_donated := B / 4
  let B_left := B - B_donated
  let C_donated (C : ℕ) := C / 3
  let C_left (C : ℕ) := C - C_donated C
  ∃ C : ℕ, B_left + C_left C = 38 ∧ C = 30

-- Note that we use sorry to indicate the proof is omitted.
theorem Cameron_books_proof : Cameron_initial_books :=
by {
  sorry
}

end Cameron_books_proof_l512_512776


namespace relationship_between_y_coordinates_l512_512477

theorem relationship_between_y_coordinates (b y1 y2 y3 : ℝ)
  (h1 : y1 = 3 * (-3) - b)
  (h2 : y2 = 3 * 1 - b)
  (h3 : y3 = 3 * (-1) - b) :
  y1 < y3 ∧ y3 < y2 := 
sorry

end relationship_between_y_coordinates_l512_512477


namespace sample_std_dev_range_same_l512_512851

noncomputable def sample_std_dev (data : List ℝ) : ℝ := sorry
noncomputable def sample_range (data : List ℝ) : ℝ := sorry

theorem sample_std_dev_range_same (n : ℕ) (c : ℝ) (Hc : c ≠ 0) (x : Fin n → ℝ) :
  sample_std_dev (List.ofFn (λ i => x i)) = sample_std_dev (List.ofFn (λ i => x i + c)) ∧
  sample_range (List.ofFn (λ i => x i)) = sample_range (List.ofFn (λ i => x i + c)) :=
by
  sorry

end sample_std_dev_range_same_l512_512851


namespace product_of_cosine_values_l512_512028

def arithmetic_seq (a₁ d : ℝ) (n : ℕ) : ℝ := a₁ + d * (n - 1)

theorem product_of_cosine_values (a₁ : ℝ) (h : ∃ (a b : ℝ), S = {a, b} ∧ S = {cos (arithmetic_seq a₁ (2 * π / 3) n) | n ∈ ℕ.succ}) :
  ∃ (a b : ℝ), a * b = -1 / 2 :=
begin
  obtain ⟨a, b, hS₁, hS₂⟩ := h,
  -- the proof will go here
  sorry
end

end product_of_cosine_values_l512_512028


namespace product_of_cosine_values_l512_512029

def arithmetic_seq (a₁ d : ℝ) (n : ℕ) : ℝ := a₁ + d * (n - 1)

theorem product_of_cosine_values (a₁ : ℝ) (h : ∃ (a b : ℝ), S = {a, b} ∧ S = {cos (arithmetic_seq a₁ (2 * π / 3) n) | n ∈ ℕ.succ}) :
  ∃ (a b : ℝ), a * b = -1 / 2 :=
begin
  obtain ⟨a, b, hS₁, hS₂⟩ := h,
  -- the proof will go here
  sorry
end

end product_of_cosine_values_l512_512029


namespace cindi_spent_30_dollars_l512_512800

variable (C : ℕ) -- the number of pencils Cindi bought
variable (MarciaBought : ℕ := 2 * C)
variable (DonnaBought : ℕ := 3 * MarciaBought)
variable (totalBought : ℕ := DonnaBought + MarciaBought)

theorem cindi_spent_30_dollars
    (h1 : DonnaBought = 3 * MarciaBought)
    (h2 : MarciaBought = 2 * C)
    (h3 : totalBought = 480) :
    0.50 * C = 30 := by
  sorry

end cindi_spent_30_dollars_l512_512800


namespace total_weight_of_fruits_l512_512551

/-- Define the given conditions in Lean -/
def weight_of_orange_bags (n : ℕ) : ℝ :=
  if n = 12 then 24 else 0

def weight_of_apple_bags (n : ℕ) : ℝ :=
  if n = 8 then 30 else 0

/-- Prove that the total weight of 5 bags of oranges and 4 bags of apples is 25 pounds given the conditions -/
theorem total_weight_of_fruits :
  weight_of_orange_bags 12 / 12 * 5 + weight_of_apple_bags 8 / 8 * 4 = 25 :=
by sorry

end total_weight_of_fruits_l512_512551


namespace profit_percentage_calculation_l512_512118

noncomputable def profit_percentage (SP CP : ℝ) : ℝ := ((SP - CP) / CP) * 100

theorem profit_percentage_calculation (SP : ℝ) (h : CP = 0.92 * SP) : |profit_percentage SP (0.92 * SP) - 8.70| < 0.01 :=
by
  sorry

end profit_percentage_calculation_l512_512118


namespace solve_fractional_equation_l512_512224

theorem solve_fractional_equation (x : ℝ) (hx : x ≠ 1 ∧ x ≠ 1) : 3 / (x - 1) = 5 + 3 * x / (1 - x) ↔ x = 4 :=
by 
  split
  -- Forward direction (Given the equation, show x = 4)
  intro h
  -- Multiply out and simplify as shown in the solution steps
  -- Sorry statement skips the detailed proof
  sorry
  -- Converse direction (Substitute x = 4 and verify it satisfies the equation)
  intro hx
  -- Substitution and verification steps
  -- Sorry statement skips the detailed proof
  sorry

end solve_fractional_equation_l512_512224


namespace parametrized_curve_area_l512_512573

noncomputable def region_area := 
  ∫ t in 0..2 * Real.pi, (t * Real.sin t * (-2) * Real.sin (2 * t)) dt

theorem parametrized_curve_area : region_area = 32 / 9 :=
sorry

end parametrized_curve_area_l512_512573


namespace find_points_of_tangency_l512_512822

variable {A B C D D' : Type}

def point_of_tangency (A B D' : Type) : Set Type :=
  { x | ∃ (circle S : Type), 
    tangent_circle_to_side x A ∧
    tangent_circle_to_side x B ∧
    passes_through x A B D' }

axiom tangent_circle_to_side (x side : Type) : Prop
axiom passes_through (x A B D' : Type) : Prop

theorem find_points_of_tangency (A B C D' : Type) :
  point_of_tangency A B D' = sorry :=
sorry

end find_points_of_tangency_l512_512822


namespace g_has_two_zeros_l512_512501

noncomputable def f (x : ℝ) : ℝ :=
if x > 0 then x - 2 else -x^2 + (1/2) * x + 1

def g (x : ℝ) : ℝ := f(x) + x

theorem g_has_two_zeros :
  ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ g(x₁) = 0 ∧ g(x₂) = 0 :=
sorry

end g_has_two_zeros_l512_512501


namespace increasing_interval_of_even_function_l512_512548

def is_even (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f x = f (-x)

theorem increasing_interval_of_even_function
  (a : ℝ) (f : ℝ → ℝ) (h1 : f = λ x, (a-1) * x^2 + a * x + 3)
  (h2 : is_even f) :
  ∃ I : set ℝ, I = set.Iic 0 ∧ ∀ x y : ℝ, x ∈ I → y ∈ I → x ≤ y → f x ≤ f y :=
sorry

end increasing_interval_of_even_function_l512_512548


namespace kabadi_players_l512_512647

-- Definitions from the conditions
def Kho_only : Nat := 30
def Both : Nat := 5
def Total : Nat := 40

-- Question translated to Lean statement
theorem kabadi_players :
  ∃ K : Nat, K = 10 ∧ (Total = (K - Both) + Kho_only + Both) :=
sorry

end kabadi_players_l512_512647


namespace complement_union_A_B_in_U_l512_512523

def U : Set ℤ := {-2, -1, 0, 1, 2, 3}

def A : Set ℤ := {-1, 2}

def B : Set ℤ := {x | x^2 - 4 * x + 3 = 0}

theorem complement_union_A_B_in_U :
  (U \ (A ∪ B)) = {-2, 0} := by
  sorry

end complement_union_A_B_in_U_l512_512523


namespace basketball_tournament_rankings_l512_512130

-- Conditions definitions
def preliminary_matches : Type := (E F G H I J K : Type) → Prop
def semi_final_pairings : Type := (winner_fst_prelim winner_sec_prelim winner_third_prelim : Type) → Prop
def ranking_sequences := 16

-- Lean statement 
theorem basketball_tournament_rankings : 
  ∀ (teams : Type) (E F G H I J K : teams) (preliminary_matches : preliminary_matches E F G H I J K) (semi_final_pairings : semi_final_pairings (winner_fst_prelim teams) (winner_sec_prelim teams) (winner_third_prelim teams)),
  ranking_sequences = 16 :=
by
  sorry

end basketball_tournament_rankings_l512_512130


namespace fraction_division_l512_512712

theorem fraction_division (a b c d : ℚ) (h : b ≠ 0 ∧ d ≠ 0) : (a / b) / c = a / (b * c) := sorry

example : (3 / 7) / 4 = 3 / 28 := by
  apply fraction_division
  exact ⟨by norm_num, by norm_num⟩

end fraction_division_l512_512712


namespace perp_bisector_through_fixed_point_l512_512458

-- Definitions of points and conditions.
variables {A B C M E F K L D : Type}

-- A fixed circle with fixed points A and B.
axiom circle_fixed : circle A B

-- A variable point C such that △ABC is acute.
axiom triangle_acute : acute_triangle A B C

-- M is the midpoint of AB.
axiom midpoint_M : midpoint A B M

-- E and F are the projections of M onto AC and BC respectively.
axiom projections_EF : projections M A C E ∧ projections M B C F

-- The goal is to prove that the perpendicular bisector of EF passes through a specific point.
theorem perp_bisector_through_fixed_point {A B C M E F : Type} (circle_fixed : circle A B)
  (triangle_acute : acute_triangle A B C)
  (midpoint_M : midpoint A B M)
  (projections_EF : projections M A C E ∧ projections M B C F) :
  ∃ P, perpendicular_bisector_through_point E F P :=
sorry

end perp_bisector_through_fixed_point_l512_512458


namespace Tara_savings_after_one_year_l512_512653

theorem Tara_savings_after_one_year :
  ∀ (initial_amount : ℝ) (interest_rate : ℝ),
    initial_amount = 90 → interest_rate = 0.1 →
    let interest_earned := initial_amount * interest_rate in
    let total_amount := initial_amount + interest_earned in
    total_amount = 99 :=
by
  intros initial_amount interest_rate h_initial_amount h_interest_rate
  rw [h_initial_amount, h_interest_rate]
  let interest_earned := initial_amount * interest_rate
  let total_amount := initial_amount + interest_earned
  rw [h_initial_amount, h_interest_rate]
  sorry

end Tara_savings_after_one_year_l512_512653


namespace meeting_point_l512_512088

theorem meeting_point (H S : ℝ × ℝ) (hH : H = (7, -3)) (hS : S = (3, 5)) :
  let M := ((H.1 + S.1) / 2, (H.2 + S.2) / 2) in M = (5, 1) := 
by
  sorry

end meeting_point_l512_512088


namespace calculation_correct_l512_512402

def expression : ℝ := 200 * 375 * 0.0375 * 5

theorem calculation_correct : expression = 14062.5 := 
by
  sorry

end calculation_correct_l512_512402


namespace percentage_value_l512_512103

theorem percentage_value (M : ℝ) (h : (25 / 100) * M = (55 / 100) * 1500) : M = 3300 :=
by
  sorry

end percentage_value_l512_512103


namespace probability_four_is_largest_l512_512747

/-- The probability that 4 is the largest value selected when three cards are drawn from a set of 5 cards numbered 1 to 5 is 3/10. -/
theorem probability_four_is_largest :
  let cards := {1, 2, 3, 4, 5}
  in probability {s | s.cardinality = 3 ∧ 4 ∈ s ∧ ∀ x ∈ s, x ≤ 4} = 3 / 10 := sorry

end probability_four_is_largest_l512_512747


namespace frog_escape_probability_l512_512929

noncomputable theory
open_locale classical

-- Define the probability function P based on the given conditions and recursive relation
def P : ℕ → ℝ
| 0       := 0
| 7       := 1
| 14      := 1
| (N+1) := if N < 13 then (N.succ : ℝ) /14 * P (N) + (1 - (N.succ : ℝ) / 14) * P (N + 2) else 0

theorem frog_escape_probability : P 3 = 98 / 197 :=
sorry

end frog_escape_probability_l512_512929


namespace trigonometric_identity_l512_512513

theorem trigonometric_identity
  (θ φ : ℝ)
  (f : ℝ → ℝ)
  (h1 : ∀ x, f(x) = 3 * Real.sin(x) + 4 * Real.cos(x))
  (h2 : (∃ k : ℤ, sin(θ + φ) = 1 ∨ sin(θ + φ) = -1))
  (h3 : sin(φ) = 4 / 5)
  (h4 : cos(φ) = 3 / 5) :
  cos(2 * θ) + sin(θ) * cos(θ) = 19 / 25 := sorry

end trigonometric_identity_l512_512513


namespace f_f_3_equals_13_over_9_l512_512751

-- Define the piecewise function f as given in the problem
def f (x : ℝ) : ℝ :=
  if x ≤ 1 then x^2 + 1 else 2 / x

-- State the problem we need to prove: f(f(3)) = 13/9
theorem f_f_3_equals_13_over_9 : f (f 3) = 13 / 9 := sorry

end f_f_3_equals_13_over_9_l512_512751


namespace complement_union_eq_l512_512525

def U : Set ℤ := {-2, -1, 0, 1, 2, 3}
def A : Set ℤ := {-1, 2}
def B : Set ℤ := {x | x ^ 2 - 4 * x + 3 = 0}

theorem complement_union_eq :
  (U \ (A ∪ B)) = {-2, 0} :=
by
  sorry

end complement_union_eq_l512_512525


namespace partition_set_l512_512459

theorem partition_set (n : ℕ) (h : n ≥ 3) :
  ∃ (S T : set ℕ), 
    (S ∪ T = {x : ℕ | 1 ≤ x ∧ x ≤ n^2 - n}) ∧ (S ∩ T = ∅) ∧ (S ≠ ∅) ∧ (T ≠ ∅) ∧
    (∀ (a : list ℕ), a.sorted (<) → a.length = n → (set_of (λ k, 2 ≤ k ∧ k ≤ n - 1) (∑ k in set_of (λ k, 2 ≤ k ∧ k ≤ n - 1), 
    a.nth_le (k - 2) k )) = ∅) :=
sorry

end partition_set_l512_512459


namespace option1_cost_option2_cost_cost_effectiveness_x_50_alternative_strategy_l512_512356

-- Define the constants and conditions
def price_dispenser : ℕ := 350
def price_barrel : ℕ := 50
def discount : ℝ := 0.9
def num_dispensers : ℕ := 20

-- Define cost calculations for Option 1
def cost_option1 (x : ℕ) : ℕ :=
  let dispensers_cost := num_dispensers * price_dispenser
  let barrels_cost := (x - num_dispensers) * price_barrel
  dispensers_cost + barrels_cost

-- Define cost calculations for Option 2
def cost_option2 (x : ℕ) : ℕ :=
  let discounted_dispenser := (price_dispenser : ℝ) * discount
  let discounted_barrel := (price_barrel : ℝ) * discount
  let dispensers_cost := discounted_dispenser * (num_dispensers : ℝ)
  let barrels_cost := discounted_barrel * (x : ℝ)
  (dispensers_cost + barrels_cost : ℕ)

-- Proving cost calculations
theorem option1_cost (x : ℕ) : cost_option1 x = 50 * x + 6000 :=
  by sorry

theorem option2_cost (x : ℕ) : cost_option2 x = 45 * x + 6300 :=
  by sorry

-- Proving cost effectiveness for x = 50
theorem cost_effectiveness_x_50 : cost_option1 50 < cost_option2 50 :=
  by sorry

-- Proving alternative purchasing strategy for x = 50
theorem alternative_strategy : 20 * price_dispenser + 30 * (price_barrel * discount : ℝ : ℕ) < 8500 :=
  by sorry

end option1_cost_option2_cost_cost_effectiveness_x_50_alternative_strategy_l512_512356


namespace polar_bear_diameter_paradox_l512_512996

section

variable (R_I R_P : ℝ)
variable (floe_mass bear_mass : ℝ)
variable (correct_measurements : R_I = 4.25 ∧ R_P = 4.5 ∧ 0.5 * R_I ≈ (floe_mass / bear_mass) * (R_P - R_I))

theorem polar_bear_diameter_paradox :
  (R_P = 4.5) → (R_I = 4.25) → (correct_measurements ∧ floe_mass > bear_mass) →
  floe_mass / bear_mass ≈ (R_P - R_I) / R_I :=
by
  sorry

end

end polar_bear_diameter_paradox_l512_512996


namespace equation_of_directrix_of_parabola_l512_512662

-- Defining the standard form of the parabola and its properties
def parabola_standard_form (x y : ℝ) (p : ℝ) : Prop := x^2 = 4 * p * y

-- Given conditions
def given_parabola : Prop := parabola_standard_form x y (2)

-- Prove:
theorem equation_of_directrix_of_parabola :
  given_parabola → ∃ y0 : ℝ, y0 = -2 :=
by
  intro h
  use -2
  sorry

end equation_of_directrix_of_parabola_l512_512662


namespace bridget_apples_l512_512398

theorem bridget_apples :
  ∃ x : ℕ, (x - x / 3 - 4) = 6 :=
by
  sorry

end bridget_apples_l512_512398


namespace second_number_percentage_less_than_third_l512_512369

theorem second_number_percentage_less_than_third (T : ℝ) :
  let first_number := 0.75 * T,
      second_number := first_number - 0.06 * first_number in
  second_number = 0.705 * T → 
  (T - second_number) / T * 100 = 29.5 :=
by
  intros
  sorry

end second_number_percentage_less_than_third_l512_512369


namespace angle_ADO_eq_angle_HAN_l512_512961

-- Let ABC be an acute triangle with all angles acute, inscribed in a circle with center O.
-- H is the orthocenter of triangle ABC.
-- M is the midpoint of BC.
-- D is the foot of the angle bisector from A.
-- The circumcircle of triangle BHC intersects segment OM at point N.
-- We need to show that angle ADO is equal to angle HAN.

variables {A B C O H M D N : Point}

-- Assuming acute triangle ABC with circumcenter O.
axiom ABC_inscribed_circle : Circle
axiom acute_triangle_ABC : ∀ (X : Point), X ∈ {A, B, C} → acute_angle X
axiom center_O : O = circumcenter ABC_inscribed_circle
axiom point_on_circumcircle_ABC : ∀ X ∈ {A, B, C}, X ∈ circle_points ABC_inscribed_circle
axiom orthocenter_ABC : H = orthocenter A B C

-- M is the midpoint of BC.
axiom midpoint_M : midpoint B C M

-- D is the foot of the angle bisector from A.
axiom foot_angle_bisector_D : foot bisector A (angle B A C) D

-- Circumcircle of triangle BHC intersects OM at point N.
axiom circumcircle_BHC : Circle
axiom circumcircle_BHC_intersects_OM : ∀ P, P ∈ segment O M → P ∈ circle_points circumcircle_BHC → P = N

-- We need to show that angle ADO is equal to angle HAN.
theorem angle_ADO_eq_angle_HAN :
  ∀ (ADO HAN : Angle), ADO = ∠ A D O → HAN = ∠ H A N → ADO = HAN :=
by
  sorry


end angle_ADO_eq_angle_HAN_l512_512961


namespace basis_with_p_q_l512_512064

variables {V : Type*} [add_comm_group V] [vector_space ℝ V] 
variables (a b c p q : V)
hypothesis (h1 : is_basis ℝ ![a, b, c])
hypothesis (h2 : p = a + b)
hypothesis (h3 : q = a - b)

theorem basis_with_p_q :
  is_basis ℝ ![p, q, a + 2 • c] :=
sorry

end basis_with_p_q_l512_512064


namespace ab_c_sum_geq_expr_ab_c_sum_eq_iff_l512_512978

theorem ab_c_sum_geq_expr (a b c : ℝ) (α : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) : 
  a * b * c * (a^α + b^α + c^α) ≥ a^(α+2) * (-a + b + c) + b^(α+2) * (a - b + c) + c^(α+2) * (a + b - c) :=
sorry

theorem ab_c_sum_eq_iff (a b c : ℝ) (α : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  a * b * c * (a^α + b^α + c^α) = a^(α+2) * (-a + b + c) + b^(α+2) * (a - b + c) + c^(α+2) * (a + b - c) ↔ a = b ∧ b = c :=
sorry

end ab_c_sum_geq_expr_ab_c_sum_eq_iff_l512_512978


namespace reciprocal_of_x2_minus_x_l512_512111

-- Definitions based on conditions
def x : ℂ := (1 - complex.I * real.sqrt 3) / 2
def i : ℂ := complex.I

-- The theorem to be proved
theorem reciprocal_of_x2_minus_x : (1 / (x ^ 2 - x)) = -1 :=
by
  rw [x]
  sorry

end reciprocal_of_x2_minus_x_l512_512111


namespace problem_statement_l512_512062

variable (a : ℕ → ℚ) (S : ℕ → ℚ)
variable (d : ℚ)

-- Definition of the arithmetic sequence sum
def arithSeqSum (n : ℕ) : ℚ := n * (2 * a 1 + (n - 1) * d) / 2

-- Conditions
axiom h1 : S 5 = arithSeqSum 5
axiom h2 : S 6 = arithSeqSum 6
axiom h3 : S 4 = arithSeqSum 4
axiom h4 : S 5 > S 6
axiom h5 : S 6 > S 4

-- The proof problems
theorem problem_statement : (d < 0) ∧ (S 10 > 0) ∧ (S 11 < 0) := by
  sorry

end problem_statement_l512_512062


namespace product_of_cosines_of_two_distinct_values_in_S_l512_512019

noncomputable def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + d

theorem product_of_cosines_of_two_distinct_values_in_S
  (a : ℕ → ℝ)
  (d : ℝ)
  (h_sequence: arithmetic_sequence a (2 * Real.pi / 3))
  (hS : ∃ (b c : ℝ), b ≠ c ∧ (∀ n : ℕ, ∃b', b' = cos (a n) ∧ (b' = b ∨ b' = c))) :
  (∃ b c : ℝ, b ≠ c ∧ (∀ n : ℕ, cos (a n) = b ∨ cos (a n) = c) ∧ b * c = -1 / 2) := 
sorry

end product_of_cosines_of_two_distinct_values_in_S_l512_512019


namespace complement_A_union_B_l512_512531

-- Define the universal set U
def U : Set ℤ := {-2, -1, 0, 1, 2, 3}

-- Define the set A
def A : Set ℤ := {-1, 2}

-- Define the set B using the quadratic equation condition
def B : Set ℤ := {x | x^2 - 4*x + 3 = 0}

-- State the theorem we want to prove
theorem complement_A_union_B : (U \ (A ∪ B)) = {-2, 0} := by
  sorry

end complement_A_union_B_l512_512531


namespace cos_set_product_l512_512036

noncomputable def arithmetic_sequence (a1 : ℝ) (d : ℝ) : ℕ → ℝ :=
  λ n, a1 + d * (n - 1)

theorem cos_set_product (a1 : ℝ) (h1 : (arithmetic_sequence a1 (2 * Real.pi / 3) '' {n | n ≥ 1}).to_set.cos.to_finset.card = 2) :
  let S := (arithmetic_sequence a1 (2 * Real.pi / 3) '' {n | n ≥ 1}).to_set.cos.to_finset in 
  (S.to_finset : set ℝ).product = -1 / 2 := sorry

end cos_set_product_l512_512036


namespace triangle_PQR_proof_l512_512142

-- Definitions from the conditions
variables 
  (P Q R T : Point)  -- Points in the plane representing the vertices and point T on PR
  (circumradius : ℝ) -- Radius of the circumcircle of triangle PQT
  (PT TR PR QR : ℝ)  -- Lengths of the sides and segments

-- Conditions
def conditions (P Q R T : Point) (circumradius PT TR QR PR : ℝ) : Prop :=
  (PT = 8) ∧ (TR = 1) ∧ (circumradius = 3 * sqrt 3) ∧ 
  (PR = PT + TR) ∧ (QR = sqrt(PT * TR))

-- Proving QR is 3 units and angle QRP is (π / 3 ± acos (5 / 6))
theorem triangle_PQR_proof
  (h_conditions : conditions P Q R T circumradius PT TR QR PR) :
  QR = 3 ∧ 
  ∃ θ : ℝ, θ = (π / 3 ± acos (5 / 6)) :=
begin
  sorry
end

end triangle_PQR_proof_l512_512142


namespace complement_union_A_B_eq_neg2_0_l512_512537

def U : Set ℤ := {-2, -1, 0, 1, 2, 3}
def A : Set ℤ := {-1, 2}
def B : Set ℤ := {x | x^2 - 4 * x + 3 = 0}

theorem complement_union_A_B_eq_neg2_0 :
  U \ (A ∪ B) = {-2, 0} := by
  sorry

end complement_union_A_B_eq_neg2_0_l512_512537


namespace unsold_tomatoes_l512_512619

theorem unsold_tomatoes (total_harvest sold_maxwell sold_wilson : ℝ) 
(h_total_harvest : total_harvest = 245.5)
(h_sold_maxwell : sold_maxwell = 125.5)
(h_sold_wilson : sold_wilson = 78) :
(total_harvest - (sold_maxwell + sold_wilson) = 42) :=
by {
  sorry
}

end unsold_tomatoes_l512_512619


namespace total_length_of_segments_l512_512939

theorem total_length_of_segments
  (l1 l2 l3 l4 l5 l6 : ℕ) 
  (hl1 : l1 = 5) 
  (hl2 : l2 = 1) 
  (hl3 : l3 = 4) 
  (hl4 : l4 = 2) 
  (hl5 : l5 = 3) 
  (hl6 : l6 = 3) : 
  l1 + l2 + l3 + l4 + l5 + l6 = 18 := 
by 
  sorry

end total_length_of_segments_l512_512939


namespace points_cyclic_l512_512955

-- Given Definitions
variable (A B C A1 B1 C1 H A2 C2 P T : Type)
variable (is_height : Triangle A B C → Line A1 B1 C1)
variable (is_orthocenter : Orthocenter H A B C)
variable (line_parallel : Line A2 C2 ∥ Line A C)
variable (intersect_AA1_A2 : Line A1 ∩ Line A2)
variable (intersect_CC1_C2 : Line C1 ∩ Line C2)
variable (B1_outside_circumcircle : ∀ (circumcircle : Circumcircle A2 H C2), ¬ (IsIn B1 circumcircle))
variable (tangent_to_circumcircle : ∀ (circumcircle : Circumcircle A2 H C2), TangentLine B1 P circumcircle ∧ TangentLine B1 T circumcircle)

-- To prove
theorem points_cyclic (A1 C1 P T : Point) (circ_cyclic :  Cyclic A1 C1 P T) : Cyclic A1 C1 P T :=
by
  sorry

end points_cyclic_l512_512955


namespace manicure_cost_l512_512953

noncomputable def cost_of_manicure : ℝ := 30

theorem manicure_cost
    (cost_hair_updo : ℝ)
    (total_cost_with_tips : ℝ)
    (tip_rate : ℝ)
    (M : ℝ) :
  cost_hair_updo = 50 →
  total_cost_with_tips = 96 →
  tip_rate = 0.20 →
  (cost_hair_updo + M + tip_rate * cost_hair_updo + tip_rate * M = total_cost_with_tips) →
  M = cost_of_manicure :=
by
  intros h1 h2 h3 h4
  sorry

end manicure_cost_l512_512953


namespace parabola_y_axis_symmetry_l512_512293

theorem parabola_y_axis_symmetry (a b c d : ℝ) (r : ℝ) :
  (2019^2 + 2019 * a + b = 0) ∧ (2019^2 + 2019 * c + d = 0) ∧
  (a = -(2019 + r)) ∧ (c = -(2019 - r)) →
  b = -d :=
by
  sorry

end parabola_y_axis_symmetry_l512_512293


namespace cut_parallelogram_into_triangles_l512_512422

-- Define the vertices of a parallelogram
variables {A B C D M : Point}

-- Define the structure of the parallelogram
structure Parallelogram (A B C D : Point) :=
(le_side : Line (B ↔ C) ∧ Line (A ↔ D))
(mid_point : (M : Point) → M ∈ IntSeg (B ↔ C) ∧ MidPoint B C M)

-- State the theorem to prove
theorem cut_parallelogram_into_triangles (parallelogram : Parallelogram A B C D) :
  ∃ (T1 T2: Triangle), 
    ParallelogramRegion A B C D cut_from_vertex := T1 ∧
    ParallelogramRegion A B C D cut_from_vertex := T2 :=
sorry

end cut_parallelogram_into_triangles_l512_512422


namespace system_solution_unique_l512_512946

theorem system_solution_unique (a x y : ℝ) :
  ((x - a) ^ 2 = 16 * (y - x + a - 3)) ∧ (log (x / 3) (y / 3) = 1) ↔
  a = 19 := by
  sorry

end system_solution_unique_l512_512946


namespace simultaneous_equations_solution_l512_512266

theorem simultaneous_equations_solution (x y : ℚ) (h1 : 3 * x - 4 * y = 11) (h2 : 9 * x + 6 * y = 33) : 
  x = 11 / 3 ∧ y = 0 :=
by {
  sorry
}

end simultaneous_equations_solution_l512_512266


namespace lowest_fraction_combine_two_slowest_l512_512801

def rate_a (hours : ℕ) : ℚ := 1 / 4
def rate_b (hours : ℕ) : ℚ := 1 / 5
def rate_c (hours : ℕ) : ℚ := 1 / 8

theorem lowest_fraction_combine_two_slowest : 
  (rate_b 1 + rate_c 1) = 13 / 40 :=
by sorry

end lowest_fraction_combine_two_slowest_l512_512801


namespace solution_to_inequality_l512_512685

noncomputable def inequality_solution_set : Set ℝ :=
  (-∞, -1) ∪ (2, +∞)

theorem solution_to_inequality (a b x : ℝ) (h₁ : a > 0) (h₂ : b = a) :
  (ax + b) / (x - 2) > 0 ↔ x ∈ inequality_solution_set :=
by sorry

end solution_to_inequality_l512_512685


namespace perimeter_of_triangle_min_cos_A_l512_512602

noncomputable theory

variables {A B C : ℝ}
variables {a b c : ℝ}
variables {triangle_ABC : a = 2 ∧ sin B * (1 + cos A) = sin A * (2 - cos B)}

theorem perimeter_of_triangle
  (h : a = 2 ∧ sin B * (1 + cos A) = sin A * (2 - cos B))
  (h1 : a = 2)
  (h2 : sin B * (1 + cos A) = sin A * (2 - cos B)) :
  a + b + c = 6 :=
sorry

theorem min_cos_A
  (h : a = 2 ∧ sin B * (1 + cos A) = sin A * (2 - cos B))
  (h1 : a = 2)
  (h2 : sin B * (1 + cos A) = sin A * (2 - cos B)) :
  ∃ c, cos A = 1/2 :=
sorry

end perimeter_of_triangle_min_cos_A_l512_512602


namespace verify_solution_l512_512403

noncomputable def problem_statement : ℤ :=
  2^4 * 3^2 * 5^3 * 7^2

def solution : ℤ := 80_182

theorem verify_solution : problem_statement / 11 = solution := by
  sorry

end verify_solution_l512_512403


namespace melissa_oranges_l512_512188

theorem melissa_oranges :
  let initial_oranges := 70
  let john_taken := 19
  let melissa_after_john := initial_oranges - john_taken
  let sarah_fraction := 3 / 7
  let sarah_taken := Int.floor (sarah_fraction * melissa_after_john)
  let melissa_after_sarah := melissa_after_john - sarah_taken
  let michael_fraction := 0.15
  let michael_taken := Int.floor (michael_fraction * melissa_after_sarah)
  let final_oranges := melissa_after_sarah - michael_taken
  final_oranges = 26 :=
by
  let initial_oranges := 70
  let john_taken := 19
  let melissa_after_john := initial_oranges - john_taken

  let sarah_fraction := 3 / 7
  let sarah_taken := Int.floor (sarah_fraction * melissa_after_john)
  let melissa_after_sarah := melissa_after_john - sarah_taken

  let michael_fraction := 0.15
  let michael_taken := Int.floor (michael_fraction * melissa_after_sarah)
  let final_oranges := melissa_after_sarah - michael_taken

  show final_oranges = 26, 
  sorry

end melissa_oranges_l512_512188


namespace simplify_q_l512_512184

variables (d e f : ℝ) (h_de : d ≠ e) (h_df : d ≠ f) (h_ef : e ≠ f)

def q (x : ℝ) : ℝ := (x + d)^4 / ((d - e) * (d - f)) + (x + e)^4 / ((e - d) * (e - f)) + (x + f)^4 / ((f - d) * (f - e))

theorem simplify_q (x : ℝ) : q d e f x = d + e + f + 4 * x :=
by {
  -- Skipping the proof as specified
  sorry
}

end simplify_q_l512_512184


namespace polar_to_rectangular_l512_512789

theorem polar_to_rectangular (r θ : ℝ) (h_r : r = 7) (h_θ : θ = π / 4) : 
  (r * Real.cos θ, r * Real.sin θ) = (7 * Real.sqrt 2 / 2, 7 * Real.sqrt 2 / 2) :=
by 
  -- proof goes here
  sorry

end polar_to_rectangular_l512_512789


namespace jony_stops_walking_l512_512951

def block_distance : ℕ := 40
def jony_speed : ℕ := 100
def start_time := (7, 0)  -- Representing time as (hours, minutes)

def walk_time (start_block : ℕ) (end_block : ℕ) : ℕ :=
  (end_block - start_block) * block_distance

theorem jony_stops_walking :
  let total_distance := (walk_time 10 90) + (walk_time 90 70),
      total_time_minutes := total_distance / jony_speed,
      hours := start_time.1,
      minutes := start_time.2 + total_time_minutes
  in (hours, minutes) = (7, 40) :=
by
  sorry

end jony_stops_walking_l512_512951


namespace max_min_values_of_y_l512_512815

def y (x : Real) : Real := 3 - 4 * Real.sin x - (Real.cos x) ^ 2

theorem max_min_values_of_y :
  (∃ x : Real, y x = 7) ∧ (∃ x : Real, y x = -1) ∧
  (∀ x : Real, y x ≤ 7) ∧ (∀ x : Real, y x ≥ -1) := by
  sorry

end max_min_values_of_y_l512_512815


namespace sale_in_fifth_month_l512_512366

theorem sale_in_fifth_month (avg_sale_per_month : ℕ) (num_months : ℕ) (sales1 sales2 sales3 sales4 sales6 : ℕ) :
  avg_sale_per_month = 6000 →
  num_months = 5 →
  sales1 = 5420 →
  sales2 = 5660 →
  sales3 = 6200 →
  sales4 = 6350 →
  sales6 = 5870 →
  ∃ sale5,
    sale5 = avg_sale_per_month * num_months - (sales1 + sales2 + sales3 + sales4) ∧
    sale5 = 6370 :=
by
  intros h_avg h_months h_sale1 h_sale2 h_sale3 h_sale4 h_sale6
  use avg_sale_per_month * num_months - (sales1 + sales2 + sales3 + sales4)
  simp [h_avg, h_months, h_sale1, h_sale2, h_sale3, h_sale4]
  exact ⟨rfl, rfl⟩

end sale_in_fifth_month_l512_512366


namespace no_obtuse_equilateral_triangle_exists_l512_512427

theorem no_obtuse_equilateral_triangle_exists :
  ¬(∃ (a b c : ℝ), a = b ∧ b = c ∧ a + b + c = π ∧ a > π/2 ∧ b > π/2 ∧ c > π/2) :=
sorry

end no_obtuse_equilateral_triangle_exists_l512_512427


namespace heartsuit_ratio_l512_512107

def heartsuit (n m : ℕ) : ℕ := n^4 * m^3

theorem heartsuit_ratio :
  (heartsuit 2 4) / (heartsuit 4 2) = 1 / 2 := by
  sorry

end heartsuit_ratio_l512_512107


namespace max_profit_achieved_l512_512623

theorem max_profit_achieved :
  ∃ x : ℤ, 
    (x = 21) ∧ 
    (21 + 14 = 35) ∧ 
    (30 - 21 = 9) ∧ 
    (21 - 5 = 16) ∧
    (-x + 1965 = 1944) :=
by
  sorry

end max_profit_achieved_l512_512623


namespace power_function_value_at_4_l512_512460

theorem power_function_value_at_4
  (f : ℝ → ℝ)
  (α : ℝ)
  (h1 : ∀ x, f(x) = x^α)
  (h2 : f(2) = Real.sqrt 2) :
  f(4) = 2 :=
  sorry

end power_function_value_at_4_l512_512460


namespace relationship_of_y_values_l512_512475

theorem relationship_of_y_values (b y1 y2 y3 : ℝ) (h1 : y1 = 3 * (-3) - b)
                                (h2 : y2 = 3 * 1 - b)
                                (h3 : y3 = 3 * (-1) - b) :
  y1 < y3 ∧ y3 < y2 :=
by
  sorry

end relationship_of_y_values_l512_512475


namespace num_correct_statements_l512_512219

-- Define the sequence
def seq (x : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, n ≥ 1 → x (n + 1) = 1 / (1 - x n)

-- Define the problem statements
def statement_1 (x : ℕ → ℝ) : Prop :=
  x 2 = 5 → x 7 = 4 / 5

def statement_2 (x : ℕ → ℝ) : Prop :=
  x 1 = 2 → (∑ i in Finset.range 2022, x (i + 1)) = 2021 / 2

def statement_3 (x : ℕ → ℝ) : Prop :=
  (x 1 + 1) * (x 2 + 1) * x 9 = -1 → x 1 = Real.sqrt 2

-- Main theorem to prove
theorem num_correct_statements (x : ℕ → ℝ) (h_seq : seq x) :
  (if statement_1 x then 1 else 0) +
  (if statement_2 x then 1 else 0) +
  (if statement_3 x then 1 else 0) = 1 := by
  sorry

end num_correct_statements_l512_512219


namespace find_range_m_l512_512874

-- Definitions of the conditions
def p (m : ℝ) : Prop := ∃ x y : ℝ, (x + y - m = 0) ∧ ((x - 1)^2 + y^2 = 1)
def q (m : ℝ) : Prop := ∃ x : ℝ, (x^2 - x + m - 4 = 0) ∧ x ≠ 0 ∧ ∀ y : ℝ, (y^2 - y + m - 4 = 0) → x * y < 0

theorem find_range_m (m : ℝ) : (p m ∨ q m) ∧ ¬p m → (m ≤ 1 - Real.sqrt 2 ∨ 1 + Real.sqrt 2 ≤ m ∧ m < 4) :=
by
  sorry

end find_range_m_l512_512874


namespace problem_l512_512892

namespace arithmetic_sequence

def is_arithmetic_sequence (a : ℕ → ℚ) := ∃ d : ℚ, ∀ n : ℕ, a (n + 1) = a n + d

theorem problem 
  (a : ℕ → ℚ) 
  (h_arith : is_arithmetic_sequence a) 
  (h_cond : a 1 + a 7 + a 13 = 4) : a 2 + a 12 = 8 / 3 :=
sorry

end arithmetic_sequence

end problem_l512_512892


namespace tan_theta_l512_512599

variable (k θ : ℝ)
variable (D : Matrix (Fin 2) (Fin 2) ℝ) (R : Matrix (Fin 2) (Fin 2) ℝ)

-- Conditions
def D_condition : D = ![![k, 0], ![0, k]] := sorry
def R_condition : R = ![![Real.cos θ, -Real.sin θ], ![Real.sin θ, Real.cos θ]] := sorry
def RD_condition : R ⬝ D = ![![12, -5], ![5, 12]] := sorry
def k_positive : k > 0 := sorry

-- Theorem statement
theorem tan_theta : Real.tan θ = 5 / 12 := sorry

end tan_theta_l512_512599


namespace birds_landing_l512_512694

theorem birds_landing (initial_birds total_birds birds_landed : ℤ) 
  (h_initial : initial_birds = 12) 
  (h_total : total_birds = 20) :
  birds_landed = total_birds - initial_birds :=
by
  sorry

end birds_landing_l512_512694


namespace train_cross_time_approx_l512_512581

-- Definitions based on conditions
def train_length : ℝ := 120 -- in meters
def train_speed_kmh : ℝ := 160 -- in km/hr

-- Conversion factor from km/hr to m/s
def kmh_to_mps_factor : ℝ := 5 / 18

-- Speed in meters per second
def train_speed_mps : ℝ := train_speed_kmh * kmh_to_mps_factor

-- Calculate time to cross the electric pole
def time_to_cross_pole : ℝ := train_length / train_speed_mps

-- Statement to prove
theorem train_cross_time_approx : abs (time_to_cross_pole - 2.7) < 0.01 :=
by
  sorry

end train_cross_time_approx_l512_512581


namespace sum_p_k_lt_sqrt_2a_n_plus_1_l512_512904

noncomputable def f (x : ℝ) : ℝ := Real.log (1 + x) - x

def b_n (n : ℕ) : ℝ := Real.log (1 + n) - n

def a_n (n : ℕ) : ℝ := Real.log (1 + n) - (b_n n)

def p_k (k : ℕ) : ℝ := (finset.range (2 <| k+1)).odd.prod id / (finset.range (2 <| k+1)).even.prod id

theorem sum_p_k_lt_sqrt_2a_n_plus_1 (n : ℕ) (h : 0 < n) : 
  ∑ k in finset.range n, p_k (k+1) < Real.sqrt (2 * a_n n + 1) - 1 :=
sorry

end sum_p_k_lt_sqrt_2a_n_plus_1_l512_512904


namespace complement_union_A_B_eq_neg2_0_l512_512535

def U : Set ℤ := {-2, -1, 0, 1, 2, 3}
def A : Set ℤ := {-1, 2}
def B : Set ℤ := {x | x^2 - 4 * x + 3 = 0}

theorem complement_union_A_B_eq_neg2_0 :
  U \ (A ∪ B) = {-2, 0} := by
  sorry

end complement_union_A_B_eq_neg2_0_l512_512535


namespace common_solutions_for_y_l512_512798

theorem common_solutions_for_y (x y : ℝ) :
  (x^2 + y^2 = 16) ∧ (x^2 - 3 * y = 12) ↔ (y = -4 ∨ y = 1) :=
by
  sorry

end common_solutions_for_y_l512_512798


namespace part1_part2_l512_512982

def f (x k : ℝ) : ℝ := log x + k / x

theorem part1 (k : ℝ) :
  (∀ x : ℝ, x > 0 → (f x k)' = 1 / x - k / x^2) ∧ 
  ((f e k)' = 0 → k = e) ∧ 
  ((∀ x, 0 < x ∧ x < e → (f x e)' < 0) ∧ (∀ x, x > e → (f x e)' > 0) ∧ f e e = 2) :=
sorry

theorem part2 :
  (∀ x1 x2 : ℝ, 0 < x2 ∧ x2 < x1 → f x1 k - x1 < f x2 k - x2) → k ∈ Ici (1/4) :=
sorry

end part1_part2_l512_512982


namespace range_for_a_l512_512077

def f (a : ℝ) (x : ℝ) := 2 * x^3 - a * x^2 + 1

def two_zeros_in_interval (a : ℝ) : Prop :=
  ∃ x1 x2 : ℝ, (1/2 ≤ x1 ∧ x1 ≤ 2) ∧ (1/2 ≤ x2 ∧ x2 ≤ 2) ∧ x1 ≠ x2 ∧ f a x1 = 0 ∧ f a x2 = 0

theorem range_for_a {a : ℝ} : (3/2 : ℝ) < a ∧ a ≤ (17/4 : ℝ) ↔ two_zeros_in_interval a :=
by sorry

end range_for_a_l512_512077


namespace emily_needs_2_over_3_liters_of_b_l512_512935

-- Define the initial volumes of chemical B and water
def volume_b_initial : ℝ := 0.08
def volume_water_initial : ℝ := 0.04
def total_initial_solution : ℝ := volume_b_initial + volume_water_initial

-- The fraction of chemical B in the initial solution
def fraction_b : ℝ := volume_b_initial / total_initial_solution

-- Final volume of the solution needed
def final_volume_solution : ℝ := 1

-- The amount of chemical B needed for the final solution
def volume_b_needed : ℝ := final_volume_solution * fraction_b

theorem emily_needs_2_over_3_liters_of_b :
  volume_b_needed = 2 / 3 :=
by
  sorry

end emily_needs_2_over_3_liters_of_b_l512_512935


namespace log_sum_l512_512399

-- Define the common logarithm function using Lean's natural logarithm with a change of base
noncomputable def log_base_10 (x : ℝ) : ℝ := Real.log x / Real.log 10

theorem log_sum : log_base_10 5 + log_base_10 0.2 = 0 :=
by
  -- Placeholder for the proof to be completed
  sorry

end log_sum_l512_512399


namespace f_increasing_on_zero_to_infty_l512_512204

def f (x : ℝ) : ℝ := Real.exp x + 1 / Real.exp x

theorem f_increasing_on_zero_to_infty :
  ∀ x y : ℝ, 0 < x → x < y → f x < f y :=
by
  sorry

end f_increasing_on_zero_to_infty_l512_512204


namespace rectangle_inequality_l512_512631

open Real

-- Define a rectangle with vertices A, B, C, D
structure Rectangle (A B C D : Point) : Prop :=
  (midpoint_BC : is_midpoint E B C)
  (midpoint_CD : is_midpoint F C D)
  (is_rectangle : is_rectangle A B C D)

-- Define is_midpoint property
def is_midpoint (M A B : Point) : Prop :=
  M = (A + B) / 2

-- Define is_rectangle property
def is_rectangle (A B C D : Point) : Prop :=
  dist A B = dist C D ∧ dist B C = dist D A ∧
  dist A C = dist B D ∧ dist C A = dist D B

-- Main theorem statement
theorem rectangle_inequality (A B C D E F : Point) 
  (h1 : is_midpoint E B C) 
  (h2 : is_midpoint F C D) 
  (h3 : is_rectangle A B C D) : 
  dist A E < 2 * dist E F :=
sorry

end rectangle_inequality_l512_512631


namespace permutations_consecutive_sums_l512_512424

theorem permutations_consecutive_sums (n : ℕ) (h : n ≥ 2) :
  (∀ a b : Fin n → ℕ, (Perm.uncurry (Fin n)).permutes (λ i, a i + b i) [1, 2, ..., n] →
    ∃ k : ℕ, (k ∈ (Fin n) → (a k + b k) = k + 1) ∨
              (k + i ≤ succ (Fin n) ∧ (a (k+i) + b (k+i-1)) = k + i)) ↔
  Odd n :=
sorry

end permutations_consecutive_sums_l512_512424


namespace chickadees_on_one_tree_l512_512307

-- Define the total number of chickadees and trees
def total_chickadees : ℕ := 2018
def total_trees : ℕ := 120

-- Define the condition where a chickadee moves after a shot
def move_condition (initial final : list ℕ) : Prop :=
  ∃ i j : ℕ, (i ≠ j ∧ initial[i] > 0 ∧ final[i] = initial[i] - 1 ∧ final[j] = initial[j] + 1 ∧ 
               ∀ k : ℕ, k ≠ i → k ≠ j → final[k] = initial[k] ∧ initial[j] ≥ initial[i])

-- Prove that after a finite number of moves, all chickadees are on one tree
theorem chickadees_on_one_tree :
  ∀ (initial : list ℕ), (list.sum initial = total_chickadees ∧ list.length initial = total_trees) →
  ∃ final : list ℕ, (∃ n : ℕ, (∀ i : ℕ, i < n → move_condition (nth config  i) (nth config (i + 1))) ∧ is_single_tree final) :=
sorry

-- A helper function to check if all elements except one are zero
def is_single_tree (l : list ℕ) : Prop :=
  ∃ k : ℕ, (l = [0, ..., 0, 2018, 0, ..., 0] ∧ k = 2018)


end chickadees_on_one_tree_l512_512307


namespace relationship_y1_y2_y3_l512_512473

variable (y1 y2 y3 b : ℝ)
variable (h1 : y1 = 3 * (-3) - b)
variable (h2 : y2 = 3 * 1 - b)
variable (h3 : y3 = 3 * (-1) - b)

theorem relationship_y1_y2_y3 : y1 < y3 ∧ y3 < y2 := by
  sorry

end relationship_y1_y2_y3_l512_512473


namespace minimum_marked_points_in_convex_ngon_l512_512321

theorem minimum_marked_points_in_convex_ngon (n : ℕ) (h : n ≥ 3) : 
  ∃ M : Finset (Point2D ℝ), (M.card = n - 2) ∧ ∀ (i j k : ℕ) (hperm : Set.UnivP {i, j, k} (Finset.range n)) 
  (hi : i < j) (hj : j < k), ∃ m ∈ M, 
  PointInTriangle (A_i, A_j, A_k) m := 
sorry

end minimum_marked_points_in_convex_ngon_l512_512321


namespace sample_stats_equal_l512_512859

/-- Let x be a data set of n samples and y be another data set of n samples such that 
    ∀ i, y_i = x_i + c where c is a non-zero constant.
    Prove that the sample standard deviations and the ranges of x and y are the same. -/
theorem sample_stats_equal (n : ℕ) (x y : Fin n → ℝ) (c : ℝ) (h : c ≠ 0)
    (h_y : ∀ i : Fin n, y i = x i + c) :
    (stddev x = stddev y) ∧ (range x = range y) := 
sorry

end sample_stats_equal_l512_512859


namespace proof_of_true_proposition_l512_512059

-- Definitions based on conditions
def p : Prop := ∀ x y: ℝ, x > y → (1/x < 1/y)
def q : Prop := ∀ x: ℝ, cos(x + Real.pi / 2) = -sin(x)

-- True proposition according to the problem solution
def true_proposition : Prop := ¬p ∧ q

-- The theorem statement
theorem proof_of_true_proposition : true_proposition :=
by sorry

end proof_of_true_proposition_l512_512059


namespace largest_t_solution_l512_512814

theorem largest_t_solution :
  ∃ t : ℚ, (t = 7 / 4) ∧ (∀ t' : ℚ, 
      (\frac{16 * t'^2 - 40 * t' + 15}{4 * t' - 3} + 7 * t' = 5 * t' + 2) → t' ≤ t) :=
sorry

end largest_t_solution_l512_512814


namespace quadrant_angle_l512_512918

theorem quadrant_angle (θ : ℝ) (k : ℤ) (h_theta : 0 < θ ∧ θ < 90) : 
  ((180 * k + θ) % 360 < 90) ∨ (180 * k + θ) % 360 ≥ 180 ∧ (180 * k + θ) % 360 < 270 :=
sorry

end quadrant_angle_l512_512918


namespace nonneg_int_solutions_count_l512_512676

theorem nonneg_int_solutions_count :
  {p : ℕ × ℕ | 3 * p.1^2 + p.2^2 = 3 * p.1 - 2 * p.2}.to_finset.card = 2 := 
by {
  sorry
}

end nonneg_int_solutions_count_l512_512676


namespace correct_minutes_added_l512_512770

theorem correct_minutes_added :
  let time_lost_per_day : ℚ := 3 + 1/4
  let start_time := 1 -- in P.M. on March 15
  let end_time := 3 -- in P.M. on March 22
  let total_days := 7 -- days from March 15 to March 22
  let extra_hours := 2 -- hours on March 22 from 1 P.M. to 3 P.M.
  let total_hours := (total_days * 24) + extra_hours
  let time_lost_per_minute := time_lost_per_day / (24 * 60)
  let total_time_lost := total_hours * time_lost_per_minute
  let total_time_lost_minutes := total_time_lost * 60
  n = total_time_lost_minutes 
→ n = 221 / 96 := 
sorry

end correct_minutes_added_l512_512770


namespace johns_age_fraction_l512_512154

theorem johns_age_fraction (F M J : ℕ) 
  (hF : F = 40) 
  (hFM : F = M + 4) 
  (hJM : J = M - 16) : 
  J / F = 1 / 2 := 
by
  -- We don't need to fill in the proof, adding sorry to skip it
  sorry

end johns_age_fraction_l512_512154


namespace product_of_cosine_values_l512_512041

noncomputable theory
open_locale classical

def arithmetic_sequence (a d : ℝ) (n : ℕ) : ℝ := a + (n - 1) * d

theorem product_of_cosine_values (a b : ℝ) 
    (h : ∃ a1 : ℝ, ∀ n : ℕ+, ∃ a b : ℝ, 
         S = {cos (arithmetic_sequence a1 (2*π/3) n) | n ∈ ℕ*} ∧
         S = {a, b}) : a * b = -1/2 :=
begin
  sorry
end

end product_of_cosine_values_l512_512041


namespace expressions_negative_l512_512289

variables (P Q R S T : ℝ)

axiom hP : P ≈ -4.2
axiom hQ : Q ≈ -2.3
axiom hR : R ≈ 0
axiom hS : S ≈ 1.1
axiom hT : T ≈ 2.7

theorem expressions_negative : (P - Q < 0) ∧ (P + T < 0) := 
by 
  sorry

end expressions_negative_l512_512289


namespace prob_one_infected_exactly_eq_three_over_eight_prob_distribution_and_expected_value_X_l512_512126

variable (B_infected_by_A : Prop)
variable (C_infected_by_A : Prop)
variable (D_infected_by_A : Prop)
variable (prob_B_infected_by_A : ℚ := 1 / 2)
variable (prob_C_infected_by_A : ℚ := 1 / 2)
variable (prob_D_infected_by_A : ℚ := 1 / 2)
variable (prob_C_infected_by_A_or_B : ℚ := 1 / 2)
variable (prob_D_infected_by_A_or_B_or_C : ℚ := 1 / 3)

noncomputable def prob_exactly_one_infected (B C D : Prop) : ℚ :=
  3 * (1 / 2) * (1 - 1 / 2) ^ 2

theorem prob_one_infected_exactly_eq_three_over_eight :
  prob_exactly_one_infected B_infected_by_A C_infected_by_A D_infected_by_A = 3 / 8 :=
sorry

open Prob

structure ProbDistribution (X : Type) (P : X → ℚ) :=
(support : Set X)
(prob : ∀ x ∈ support, 0 ≤ P x ∧ P x ≤ 1)
(sum_one : ∑ x in support, P x = 1)

def X_probabilities : ℚ :=
  (1 / 3, 1 / 2, 1 / 6)

noncomputable def expected_value_X (X : Type) (P : X → ℚ) [Fintype X] : ℚ :=
  ∑ x, x * P x

theorem prob_distribution_and_expected_value_X :
  let X := {1, 2, 3}
  let P := fun x : ℚ => if x = 1 then 1 / 3 else if x = 2 then 1 / 2 else 1 / 6
  ProbDistribution X P ∧ expected_value_X X P = 11 / 6 :=
sorry

end prob_one_infected_exactly_eq_three_over_eight_prob_distribution_and_expected_value_X_l512_512126


namespace range_of_function_l512_512558

theorem range_of_function 
  (f : ℝ → ℝ) 
  (h : f 3 = 2) 
  (eq_f : ∀ x, f x = logBase 5 (3^x + (-2))) : 
  set.range (λ (x : ℝ), -x^(-2/3)) = set.Iio 0 := by
sory

end range_of_function_l512_512558


namespace product_of_cosines_of_two_distinct_values_in_S_l512_512016

noncomputable def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + d

theorem product_of_cosines_of_two_distinct_values_in_S
  (a : ℕ → ℝ)
  (d : ℝ)
  (h_sequence: arithmetic_sequence a (2 * Real.pi / 3))
  (hS : ∃ (b c : ℝ), b ≠ c ∧ (∀ n : ℕ, ∃b', b' = cos (a n) ∧ (b' = b ∨ b' = c))) :
  (∃ b c : ℝ, b ≠ c ∧ (∀ n : ℕ, cos (a n) = b ∨ cos (a n) = c) ∧ b * c = -1 / 2) := 
sorry

end product_of_cosines_of_two_distinct_values_in_S_l512_512016


namespace proof_l512_512611

variables {A B C D E F O : Type} 
variables [EuclideanGeometry A B C D E F O]
variables {R : ℝ} -- circumradius
variables (AD BE CF : ℝ)

noncomputable def problem_statement : Prop :=
  (O.is_circumcenter (triangle A B C)) ∧
  AO.extends_to D ∧ BO.extends_to E ∧ CO.extends_to F ∧
  (1 / AD + 1 / BE + 1 / CF) = 2 / R

theorem proof : problem_statement A B C D E F O R AD BE CF :=
  sorry

end proof_l512_512611


namespace video_minutes_per_week_l512_512155

theorem video_minutes_per_week
  (daily_videos : ℕ := 3)
  (short_video_length : ℕ := 2)
  (long_video_multiplier : ℕ := 6)
  (days_in_week : ℕ := 7) :
  (2 * short_video_length + long_video_multiplier * short_video_length) * days_in_week = 112 := 
by 
  -- conditions
  let short_videos_per_day := 2
  let long_video_length := long_video_multiplier * short_video_length
  let daily_total := short_videos_per_day * short_video_length + long_video_length
  let weekly_total := daily_total * days_in_week
  -- proof
  sorry

end video_minutes_per_week_l512_512155


namespace number_of_men_is_56_l512_512337

-- Conditions: Definitions as given in the problem
def men_can_do_work_in_days (M : ℕ) (days : ℕ) : Prop :=
  M * days

-- Theorem to prove the number of men is 56
theorem number_of_men_is_56 (M : ℕ) (W : ℕ) :
  men_can_do_work_in_days M 60 = W →
  men_can_do_work_in_days (M - 8) 70 = W →
  M = 56 :=
by
  intro h1 h2
  sorry

end number_of_men_is_56_l512_512337


namespace div_fraction_fraction_division_eq_l512_512708

theorem div_fraction (a b : ℕ) (h : b ≠ 0) : (a : ℚ) / b = (a : ℚ) * (1 / (b : ℚ)) := 
by sorry

theorem fraction_division_eq : (3 : ℚ) / 7 / 4 = 3 / 28 := 
by 
  calc
    (3 : ℚ) / 7 / 4 = (3 / 7) * (1 / 4) : by rw [div_fraction] 
                ... = 3 / 28            : by normalization_tactic -- Use appropriate tactic for simplification
                ... = 3 / 28            : by rfl

end div_fraction_fraction_division_eq_l512_512708


namespace total_cost_is_72_l512_512347

-- Definitions based on conditions
def adults (total_people : ℕ) (kids : ℕ) : ℕ := total_people - kids
def cost_per_adult_meal (cost_per_meal : ℕ) (adults : ℕ) : ℕ := cost_per_meal * adults
def total_cost (total_people : ℕ) (kids : ℕ) (cost_per_meal : ℕ) : ℕ := 
  cost_per_adult_meal cost_per_meal (adults total_people kids)

-- Given values
def total_people := 11
def kids := 2
def cost_per_meal := 8

-- Theorem statement
theorem total_cost_is_72 : total_cost total_people kids cost_per_meal = 72 := by
  sorry

end total_cost_is_72_l512_512347


namespace roger_allowance_spend_l512_512637

variable (A m s : ℝ)

-- Conditions from the problem
def condition1 : Prop := m = 0.25 * (A - 2 * s)
def condition2 : Prop := s = 0.10 * (A - 0.5 * m)
def goal : Prop := m + s = 0.59 * A

theorem roger_allowance_spend (h1 : condition1 A m s) (h2 : condition2 A m s) : goal A m s :=
  sorry

end roger_allowance_spend_l512_512637


namespace solve_equation_l512_512243

theorem solve_equation (x : ℝ) (h : x ≠ 1 ∧ x ≠ 1) : (3 / (x - 1) = 5 + 3 * x / (1 - x)) ↔ x = 4 := by
  sorry

end solve_equation_l512_512243


namespace Mark_time_spent_l512_512989

theorem Mark_time_spent :
  let parking_time := 5
  let walking_time := 3
  let long_wait_time := 30
  let short_wait_time := 10
  let long_wait_days := 2
  let short_wait_days := 3
  let work_days := 5
  (parking_time + walking_time) * work_days + 
    long_wait_time * long_wait_days + 
    short_wait_time * short_wait_days = 130 :=
by
  sorry

end Mark_time_spent_l512_512989


namespace product_of_cosine_elements_l512_512012

-- Definitions for the problem conditions
def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + d

def S (a : ℕ → ℝ) (d : ℝ) : Set ℝ :=
  {x | ∃ n : ℕ, x = Real.cos (a n)}

-- Main theorem statement
theorem product_of_cosine_elements 
  (a : ℕ → ℝ) 
  (d : ℝ) 
  (h_seq : arithmetic_sequence a (2 * Real.pi / 3))
  (h_S_elements : (S a (2 * Real.pi / 3)).card = 2) 
  (h_S_contains : ∃ a b, a ≠ b ∧ S a (2 * Real.pi / 3) = {a, b}) :
  let (a, b) := Classical.choose (h_S_contains) in
  a * b = -1 / 2 :=
by
  sorry

end product_of_cosine_elements_l512_512012


namespace solve_ordered_pair_l512_512819

theorem solve_ordered_pair (x y : ℝ) 
  (h1 : x + y = (7 - x) + (7 - y))
  (h2 : x^2 - y = (x - 2) + (y - 2)) :
  (x = -5 ∧ y = 12) ∨ (x = 2 ∧ y = 5) :=
  sorry

end solve_ordered_pair_l512_512819


namespace transformation_matrix_l512_512438

-- Definitions for the conditions.
def dilation_matrix (s : ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
  !![s, 0; 0, s]

def reflection_x_matrix : Matrix (Fin 2) (Fin 2) ℝ :=
  !![1, 0; 0, -1]

-- Statement to prove the resulting transformation matrix.
theorem transformation_matrix :
  dilation_matrix 5 ⬝ reflection_x_matrix = !![5, 0; 0, -5] :=
by sorry

end transformation_matrix_l512_512438


namespace polynomial_factorization_l512_512516

variable (a b c : ℝ)

theorem polynomial_factorization :
  2 * a * (b - c)^3 + 3 * b * (c - a)^3 + 2 * c * (a - b)^3 =
  (a - b) * (b - c) * (c - a) * (5 * b - c) :=
by sorry

end polynomial_factorization_l512_512516


namespace problem1_problem2_l512_512832

noncomputable def f (x : ℝ) : ℝ := Real.exp (x / 2) - x / 4
noncomputable def g (x : ℝ) : ℝ := (x + 1) * (1 / 2 * Real.exp (x / 2) - 1 / 4)
noncomputable def F (x : ℝ) (a : ℝ) : ℝ := Real.log (x + 1) - a * f(x) + 4

-- Proof problem 1: g(x) is monotonically increasing on (-1, +∞)
theorem problem1 : ∀ x : ℝ, -1 < x → 0 ≤ deriv g x :=
sorry

-- Proof problem 2: Determine range of a such that F(x) has no zeros
theorem problem2 (a : ℝ) : (∀ x : ℝ, F x a ≠ 0) ↔ 4 < a :=
sorry

end problem1_problem2_l512_512832


namespace hyperbola_focus_proof_l512_512433

noncomputable def hyperbola_focus : ℝ × ℝ :=
  (-3, 2.5 + 2 * Real.sqrt 3)

theorem hyperbola_focus_proof :
  ∃ x y : ℝ, 
  -2 * x^2 + 4 * y^2 - 12 * x - 20 * y + 5 = 0 
  → (x = -3) ∧ (y = 2.5 + 2 * Real.sqrt 3) := 
by 
  sorry

end hyperbola_focus_proof_l512_512433


namespace triangle_median_perpendicular_l512_512123

/-- In triangle DEF, if the median from D to EF is perpendicular to the median from E to DF,
    and EF = 8 and DF = 10, then the length of DE is sqrt(41). -/
theorem triangle_median_perpendicular :
  ∀ (D E F : ℝ) (distance : ℝ × ℝ → ℝ × ℝ → ℝ) (M N : ℝ × ℝ),
    distance D F = 10 →
    distance E F = 8 →
    let M = midpoint E F in
    let N = midpoint D F in
    ∀ (medD_EF_perpendicular_medE_DF : angle_between D M E + angle_between E N D = π / 2),
    distance D E = √41 := 
sorry

end triangle_median_perpendicular_l512_512123


namespace angle_between_vector_and_linear_combination_is_90_degrees_l512_512175

noncomputable theory

def a : ℝ × ℝ × ℝ := (2, -3, -4)
def b : ℝ × ℝ × ℝ := (5, 7, -2)
def c : ℝ × ℝ × ℝ := (8, -1, 9)

def dot_product (u v : ℝ × ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2 + u.3 * v.3

def scalar_mul (k : ℝ) (v : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (k * v.1, k * v.2, k * v.3)

def vector_sub (u v : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (u.1 - v.1, u.2 - v.2, u.3 - v.3)

def angle_is_90_degrees (a b c : ℝ × ℝ × ℝ) : Prop :=
  let ac := dot_product a c
  let ab := dot_product a b
  let v := vector_sub (scalar_mul ac b) (scalar_mul ab c)
  dot_product a v = 0

theorem angle_between_vector_and_linear_combination_is_90_degrees :
  angle_is_90_degrees a b c := 
sorry

end angle_between_vector_and_linear_combination_is_90_degrees_l512_512175


namespace license_plate_combinations_l512_512396

theorem license_plate_combinations :
  let letters := 26
  let first_repeat_choices := letters
  let second_repeat_choices := letters - 1
  let unique_letters_choose := Nat.choose (letters - 2) 2
  let positions_first_repeat := Nat.choose 6 2
  let positions_second_repeat := Nat.choose 4 2
  let remaining_positions_arrangements := 2!
  let digits := 10
  let digit_choices := digits * (digits - 1) * (digits - 2)
  (first_repeat_choices * second_repeat_choices * unique_letters_choose *
    (positions_first_repeat * positions_second_repeat * remaining_positions_arrangements) *
    digit_choices) = 241164000 := by
  let letters := 26
  let first_repeat_choices := letters
  let second_repeat_choices := letters - 1
  let unique_letters_choose := Nat.choose (letters - 2) 2
  let positions_first_repeat := Nat.choose 6 2
  let positions_second_repeat := Nat.choose 4 2
  let remaining_positions_arrangements := 2!
  let digits := 10
  let digit_choices := digits * (digits - 1) * (digits - 2)
  have : first_repeat_choices * second_repeat_choices * unique_letters_choose = 650 * 276 := by sorry
  have : positions_first_repeat * positions_second_repeat * remaining_positions_arrangements = 15 * 6 * 2 := by sorry
  have : digits * (digits - 1) * (digits - 2) = 720 := by sorry
  exact (650 * 276 * (15 * 6 * 2) * 720),
  sorry

end license_plate_combinations_l512_512396


namespace find_f_g_intervals_l512_512493

variable {φ ω x: ℝ}
variable (f : ℝ → ℝ)
variable (g : ℝ → ℝ)
variable (k : ℤ)

-- Given conditions as definitions
def is_even_function := f x = 2*sin (ω * x + φ) ∧ (0 < φ ∧ φ < π) ∧ (ω > 0) ∧ (f x = f (-x))
def symmetric_distance := (∃ d : ℝ, d = π/2 ∧ ∀ x, f (x + d) = f x)

-- Define f(x) and g(x)
def f_def := f x = 2*cos (2*x)
def g_def := g x = 2*cos (x/2 - π/3)

-- Interval where g(x) is monotonically decreasing
def decreasing_interval := ∀ k : ℤ, 4*k*π + 2*π/3 ≤ x ∧ x ≤ 4*k*π + 8*π/3 

theorem find_f_g_intervals:
    is_even_function f ∧ symmetric_distance f →
    (f_def f) ∧ (decreasing_interval g x)
:= sorry

end find_f_g_intervals_l512_512493


namespace two_pow_pos_contradiction_l512_512310

theorem two_pow_pos_contradiction : (∃ x : ℝ, 2 ^ x ≤ 0) → False := 
by
  sorry

end two_pow_pos_contradiction_l512_512310


namespace complement_union_A_B_in_U_l512_512524

def U : Set ℤ := {-2, -1, 0, 1, 2, 3}

def A : Set ℤ := {-1, 2}

def B : Set ℤ := {x | x^2 - 4 * x + 3 = 0}

theorem complement_union_A_B_in_U :
  (U \ (A ∪ B)) = {-2, 0} := by
  sorry

end complement_union_A_B_in_U_l512_512524


namespace coeff_e_cannot_be_zero_l512_512284

noncomputable def polynomial_coeff_cannot_be_zero (a b c d e f : ℂ) (p q r s : ℝ) : Prop :=
  ∀ Q : ℂ[X],
    (Q = X ^ 6 + a * X ^ 5 + b * X ^ 4 + c * X ^ 3 + d * X ^ 2 + e * X) ∧
    (Q.coeff 0 = 0) ∧
    (Q.has_root 0) ∧
    (Q.has_root (complex.i)) ∧
    (Q.has_root (-complex.i)) ∧
    (Q.has_root p) ∧
    (Q.has_root q) ∧
    (Q.has_root r) ∧
    (Q.has_root s) →
    e ≠ 0

theorem coeff_e_cannot_be_zero (a b c d e f : ℂ) (p q r s : ℝ) : polynomial_coeff_cannot_be_zero a b c d e f p q r s :=
sorry

end coeff_e_cannot_be_zero_l512_512284


namespace break_even_price_correct_l512_512192

-- Conditions
def variable_cost_per_handle : ℝ := 0.60
def fixed_cost_per_week : ℝ := 7640
def handles_per_week : ℝ := 1910

-- Define the correct answer for the price per handle to break even
def break_even_price_per_handle : ℝ := 4.60

-- The statement to prove
theorem break_even_price_correct :
  fixed_cost_per_week + (variable_cost_per_handle * handles_per_week) / handles_per_week = break_even_price_per_handle :=
by
  -- The proof is omitted
  sorry

end break_even_price_correct_l512_512192


namespace find_x_logarithm_l512_512878

theorem find_x_logarithm :
  ∀ (x : ℝ), 
    (log 2 = 0.3010) →
    (log 3 = 0.4771) →
    (2 : ℝ) ^ (x + 2) = 72 →
    x = 2.17 := 
by
  intro x hlog2 hlog3 hpow
  -- assume 2^{x+2} = 72, log(2) = 0.3010, log(3) = 0.4771, prove x = 2.17
  sorry

end find_x_logarithm_l512_512878


namespace area_bounded_by_curves_is_4pi_l512_512740

noncomputable def parametric_x (t : ℝ) : ℝ := 16 * (Real.cos t)^3
noncomputable def parametric_y (t : ℝ) : ℝ := 2 * (Real.sin t)^3

theorem area_bounded_by_curves_is_4pi : (∫ t in -Real.pi / 3..Real.pi / 3, parametric_y t * deriv parametric_x t) = 4 * Real.pi :=
by
  sorry

end area_bounded_by_curves_is_4pi_l512_512740


namespace convex_body_with_circular_sections_is_sphere_l512_512936

-- Definitions of convex body, closed body, and properties of cross-sections
def is_convex_body (B : Set Point) : Prop := sorry -- Assume conditions that ensure B is convex
def is_closed (B : Set Point) : Prop := sorry   -- Assume conditions that ensure B is closed
def is_circle_or_empty_section (B : Set Point) : Prop :=
  ∀ (P : Plane), let S := {x | P.contains x ∧ B.contains x} in S = ∅ ∨ (∃ C : Circle, S = C.points)

-- Main theorem to prove
theorem convex_body_with_circular_sections_is_sphere (B : Set Point) :
  is_convex_body B → is_closed B → is_circle_or_empty_section B → is_sphere B :=
by
  sorry

end convex_body_with_circular_sections_is_sphere_l512_512936


namespace complement_union_A_B_in_U_l512_512521

def U : Set ℤ := {-2, -1, 0, 1, 2, 3}

def A : Set ℤ := {-1, 2}

def B : Set ℤ := {x | x^2 - 4 * x + 3 = 0}

theorem complement_union_A_B_in_U :
  (U \ (A ∪ B)) = {-2, 0} := by
  sorry

end complement_union_A_B_in_U_l512_512521


namespace race_distance_l512_512128

theorem race_distance
  (x y z d : ℝ) 
  (h1 : d / x = (d - 25) / y)
  (h2 : d / y = (d - 15) / z)
  (h3 : d / x = (d - 35) / z) : 
  d = 75 := 
sorry

end race_distance_l512_512128


namespace unique_construction_of_triangle_possible_two_constructions_of_triangle_l512_512420

-- Part (a)
theorem unique_construction_of_triangle
  (c a b : ℝ) (h : a > b) (angle_C : ℝ) :
  ∃! (ABC : Type) [triangle ABC], (side_length ABC "AB" = c) ∧ 
    (side_length ABC "AC" - side_length ABC "BC" = a - b) ∧
    (angle ABC "C" = angle_C) :=
sorry

-- Part (b)
theorem possible_two_constructions_of_triangle
  (c a b : ℝ) (angle_C : ℝ) :
  ∃ (ABC₁ ABC₂ : Type) [triangle ABC₁] [triangle ABC₂], 
    (ABC₁ ≠ ABC₂) ∧ 
    (side_length ABC₁ "AB" = c) ∧ (side_length ABC₂ "AB" = c) ∧
    (side_length ABC₁ "AC" + side_length ABC₁ "BC" = a + b) ∧
    (side_length ABC₂ "AC" + side_length ABC₂ "BC" = a + b) ∧
    (angle ABC₁ "C" = angle_C) ∧ (angle ABC₂ "C" = angle_C) :=
sorry

end unique_construction_of_triangle_possible_two_constructions_of_triangle_l512_512420


namespace binary_to_decimal_l512_512788

theorem binary_to_decimal (b : string) (h : b = "11110") : 
  ∑ i, (b.get! i).digitCharVal!.iget * 2^i = 30 := sorry

end binary_to_decimal_l512_512788


namespace calculate_x_plus_y_l512_512419

-- Let AB and A'B' be segments with lengths 3 and 5 units respectively
def AB_length : ℝ := 3
def A'B'_length : ℝ := 5

-- Let x be the distance from P to the midpoint of AB, and y the distance from P' to the midpoint of A'B'
def x : ℝ := 2 
def ratio : ℝ := 5 / 3

-- Calculate y using the given ratio y/x = 5/3
def y : ℝ := ratio * x

-- Prove that x + y = 16/3
theorem calculate_x_plus_y : x + y = 16 / 3 := by
  sorry

end calculate_x_plus_y_l512_512419


namespace hyperbola_equation_l512_512418

theorem hyperbola_equation (a b : ℝ) (c : ℝ) (h1 : c = 4)
  (h2 : ∀ (x y : ℝ), (x - c)^2 + y^2 = 16 → (x, y) = (a, b) ∨ (x, y) = (0, 0))
  (h3 : a^2 + b^2 = c^2)
  (h4 : ∀ (x y : ℝ), x = a ∨ y = b → (x = 0 ∨ y = 0) ∨ (x, y) = (a, b)) :
  ∃ a b : ℝ, (a = 2 ∧ b = real.sqrt(12)) ∧ (∃ x y : ℝ, (x^2 / 4 - y^2 / 12 = 1)) := 
by
  sorry

end hyperbola_equation_l512_512418


namespace circle_passes_fixed_points_minimal_area_circle_l512_512500

noncomputable def circle_eq (a : ℝ) : ℝ × ℝ → Prop :=
λ (p : ℝ × ℝ), let x := p.fst, y := p.snd in
  x^2 + y^2 - 2*a*x + (2 - 4*a)*y + 4*a - 4 = 0

def pointA := (2 : ℝ, 0 : ℝ)
def pointB := (-2/5 : ℝ, 6/5 : ℝ)

theorem circle_passes_fixed_points (a : ℝ) :
  circle_eq a pointA ∧ circle_eq a pointB :=
by {
  sorry
}

def minimal_area_circle_eq : ℝ × ℝ → Prop :=
λ (p : ℝ × ℝ), let x := p.fst, y := p.snd in
  (x - (4/5))^2 + (y - (3/5))^2 = 9/5

theorem minimal_area_circle :
  ∃ (a : ℝ), ∀ (p : ℝ × ℝ), circle_eq a p → minimal_area_circle_eq p :=
by {
  sorry
}

end circle_passes_fixed_points_minimal_area_circle_l512_512500


namespace smallest_positive_period_of_f_minimum_value_of_f_on_interval_l512_512900

def f (x : Real) : Real := Real.sin x - 2 * Real.sqrt 3 * Real.sin (x / 2) ^ 2

theorem smallest_positive_period_of_f :
  ∀ x, f (x + 2 * Real.pi) = f x :=
sorry

theorem minimum_value_of_f_on_interval :
  ∃ x ∈ Set.Icc 0 (2 * Real.pi / 3), f x = - Real.sqrt 3 :=
sorry

end smallest_positive_period_of_f_minimum_value_of_f_on_interval_l512_512900


namespace value_of_C_l512_512773

theorem value_of_C (k : ℝ) (C : ℝ) (h : k = 0.4444444444444444) :
  (2 * k * 0 ^ 2 + 6 * k * 0 + C = 0) ↔ C = 2 :=
by {
  sorry
}

end value_of_C_l512_512773


namespace total_distance_of_relay_race_l512_512941

theorem total_distance_of_relay_race 
    (fraction_siwon : ℝ := 3/10) 
    (fraction_dawon : ℝ := 4/10) 
    (distance_together : ℝ := 140) :
    (fraction_siwon + fraction_dawon) * 200 = distance_together :=
by
    sorry

end total_distance_of_relay_race_l512_512941


namespace triangle_area_144_l512_512945

-- Assuming the existence of a triangle XYZ with vertices and notation consistent with the problem
variables {Point : Type*} [metric_space Point]

structure Triangle (Point : Type*) :=
  (X Y Z : Point)

variables (t : Triangle Point) 
  (XM YN : Point → Point) -- medians as functions
  (perpendicular : ∀ (A B : Point), Prop)
  (length : Point → ℝ)

-- Condition: triangle XYZ
def is_triangle (t : Triangle Point) : Prop := true -- placeholder, triangle property needed

-- Given conditions in the problem
def median_XM := length (XM t.X) = 12
def median_YN := length (YN t.Y) = 18
def medians_perpendicular := perpendicular (XM t.X) (YN t.Y)

noncomputable def area_XYZ : ℝ := 
  -- Formalize the area calculation, this is a placeholder
  144

-- The final theorem statement: prove the area given the conditions is 144
theorem triangle_area_144 :
  is_triangle t →
  median_XM →
  median_YN →
  medians_perpendicular →
  area_XYZ = 144 :=
by sorry

end triangle_area_144_l512_512945


namespace robotics_club_non_participants_l512_512568

theorem robotics_club_non_participants (club_students electronics_students programming_students both_students : ℕ) 
  (h1 : club_students = 80) 
  (h2 : electronics_students = 45) 
  (h3 : programming_students = 50) 
  (h4 : both_students = 30) : 
  club_students - (electronics_students - both_students + programming_students - both_students + both_students) = 15 :=
by
  -- The proof would be here
  sorry

end robotics_club_non_participants_l512_512568


namespace complex_sum_to_zero_l512_512605

noncomputable def z : ℂ := sorry

theorem complex_sum_to_zero 
  (h₁ : z ^ 3 = 1) 
  (h₂ : z ≠ 1) : 
  z ^ 103 + z ^ 104 + z ^ 105 + z ^ 106 + z ^ 107 + z ^ 108 = 0 :=
sorry

end complex_sum_to_zero_l512_512605


namespace annulus_area_l512_512312

theorem annulus_area (r_inner r_outer : ℝ) (h_inner : r_inner = 8) (h_outer : r_outer = 2 * r_inner) :
  π * r_outer ^ 2 - π * r_inner ^ 2 = 192 * π :=
by
  sorry

end annulus_area_l512_512312


namespace angle_B_equals_pi_over_3_max_area_of_triangle_l512_512928

variable {A B C : ℝ} -- Angles in triangle
variable {a b c : ℝ} -- Sides opposite to angles A, B, C
variable (b_eq_2 : b = 2)
variable (h : b * sin A = sqrt 3 * a * cos B)

-- Proof problem (1): Find the angle B
theorem angle_B_equals_pi_over_3
  (h : b * sin A = sqrt 3 * a * cos B) : B = π / 3 :=
sorry

-- Proof problem (2): Find the maximum area of triangle ABC when b = 2
theorem max_area_of_triangle 
  (h : b * sin A = sqrt 3 * a * cos B) 
  (b_eq_2 : b = 2) : S = sqrt 3 :=
sorry

end angle_B_equals_pi_over_3_max_area_of_triangle_l512_512928


namespace sum_angles_of_two_triangles_l512_512565

theorem sum_angles_of_two_triangles (a1 a3 a5 a2 a4 a6 : ℝ) 
  (hABC : a1 + a3 + a5 = 180) (hDEF : a2 + a4 + a6 = 180) : 
  a1 + a2 + a3 + a4 + a5 + a6 = 360 :=
by
  sorry

end sum_angles_of_two_triangles_l512_512565


namespace complement_union_A_B_in_U_l512_512520

def U : Set ℤ := {-2, -1, 0, 1, 2, 3}

def A : Set ℤ := {-1, 2}

def B : Set ℤ := {x | x^2 - 4 * x + 3 = 0}

theorem complement_union_A_B_in_U :
  (U \ (A ∪ B)) = {-2, 0} := by
  sorry

end complement_union_A_B_in_U_l512_512520


namespace jane_dress_shop_total_dresses_l512_512584

theorem jane_dress_shop_total_dresses : 
  let red_dresses := 83 in
  let blue_dresses := red_dresses + 34 in
  red_dresses + blue_dresses = 200 :=
by 
  -- prove the statement
  sorry

end jane_dress_shop_total_dresses_l512_512584


namespace cos_arithmetic_seq_product_l512_512047

theorem cos_arithmetic_seq_product :
  ∃ (a b : ℝ), (∃ (a₁ : ℝ), ∀ n : ℕ, (n > 0) → ∃ m : ℕ, cos (a₁ + (2 * Real.pi / 3) * (n - 1)) = [a, b] ∧ (a * b = -1 / 2)) := 
sorry

end cos_arithmetic_seq_product_l512_512047


namespace point_on_x_axis_equidistant_from_A_and_B_is_M_l512_512943

theorem point_on_x_axis_equidistant_from_A_and_B_is_M :
  ∃ M : ℝ × ℝ × ℝ, (M = (-3 / 2, 0, 0)) ∧ 
  (dist M (1, -3, 1) = dist M (2, 0, 2)) := by
  sorry

end point_on_x_axis_equidistant_from_A_and_B_is_M_l512_512943


namespace find_f_116_5_l512_512664

def f (x : ℝ) : ℝ := 
  if -3 ≤ x ∧ x ≤ -2 then 2 * x else 0

lemma f_is_even : ∀ x, f x = f (-x) :=
by sorry

lemma f_shift : ∀ x, f (x + 3) = -f (x) :=
by sorry

theorem find_f_116_5 : f 116.5 = -5 :=
by 
  sorry

end find_f_116_5_l512_512664


namespace solve_equation_l512_512261

theorem solve_equation : ∀ (x : ℝ), x ≠ 1 → (3 / (x - 1) = 5 + 3 * x / (1 - x)) → x = 4 :=
by
  intros x hx heq
  -- sorry to skip the proof
  sorry

end solve_equation_l512_512261


namespace find_equation_of_line_l512_512755

theorem find_equation_of_line :
  (∃ l : ℝ → ℝ, 
      (∀ x, l x = -5/12 * (x + 4)) ∨ 
      (∀ x, x = -4)) ↔
  (∃ A B : ℝ × ℝ,
    (A.1 + 1)^2 + (A.2 - 2)^2 = 25 ∧    -- A lies on the circle
    (B.1 + 1)^2 + (B.2 - 2)^2 = 25 ∧    -- B lies on the circle
    dist A B = 8 ∧                     -- Distance |AB| = 8
    (A.1 = -4 ∨ B.1 = -4))             -- Line passes through (-4,0)
  (5 * (A.1) * .+ 12 * (A.2) + 20 = 0 ∨ A.1 = -4) := 
begin
  sorry
end

end find_equation_of_line_l512_512755


namespace right_prism_diagonal_l512_512331

theorem right_prism_diagonal (a b c : ℝ) : ¬(∃ a b c : ℝ, {√(a^2 + b^2), √(b^2 + c^2), √(a^2 + c^2)} = {5, 6, 8}) :=
sorry

end right_prism_diagonal_l512_512331


namespace minimum_obtuse_angles_in_octagon_l512_512934

theorem minimum_obtuse_angles_in_octagon :
  ∃ x : ℕ, x ≥ 5 ∧
    (∀ angles : Fin 8 → ℝ,
      (∀ i, angles i ≥ 90) →
      (∑ i, angles i = 1080) →
      (∃ n_right_or_acute : ℕ, n_right_or_acute <= 8 - x ∧
       ∀ i, angles i ≠ 90 ∨ angles i ≤ 90) →
      x) := sorry

end minimum_obtuse_angles_in_octagon_l512_512934


namespace significant_improvement_l512_512357

def old_device_data : List Float := [9.8, 10.3, 10.0, 10.2, 9.9, 9.8, 10.0, 10.1, 10.2, 9.7]
def new_device_data : List Float := [10.1, 10.4, 10.1, 10.0, 10.1, 10.3, 10.6, 10.5, 10.4, 10.5]

def mean (data : List Float) : Float :=
  data.sum / (data.length : Float)

def variance (data : List Float) (μ : Float) : Float :=
  (data.map (λ x => (x - μ) ^ 2)).sum / (data.length : Float)

def old_mean : Float := mean old_device_data
def new_mean : Float := mean new_device_data

def old_variance : Float := variance old_device_data old_mean
def new_variance : Float := variance new_device_data new_mean

theorem significant_improvement :
  new_mean - old_mean ≥ 2 * Float.sqrt ((old_variance + new_variance) / 10) :=
by 
  -- Proof omitted
  sorry

end significant_improvement_l512_512357


namespace necessary_and_sufficient_condition_l512_512455

variable (a : ℝ)

-- p represents the condition: -1 < a < 1
def p : Prop := (-1 < a) ∧ (a < 1)

-- q represents the condition: one root of the quadratic equation is greater than zero and the other is less than zero
def q : Prop :=
  let Δ := (a + 1)^2 - 4 * (a - 2) in -- discriminant of the quadratic formula
  (Δ > 0) ∧ (a - 2 < 0)

theorem necessary_and_sufficient_condition (a : ℝ) : p a ↔ q a :=
by sorry

end necessary_and_sufficient_condition_l512_512455


namespace yellow_ball_range_l512_512562

-- Definitions
def probability_condition (x : ℕ) : Prop :=
  (20 / 100 : ℝ) ≤ (4 * x / ((x + 2) * (x + 1))) ∧ (4 * x / ((x + 2) * (x + 1))) ≤ (33 / 100)

theorem yellow_ball_range (x : ℕ) : probability_condition x ↔ 9 ≤ x ∧ x ≤ 16 := 
by
  sorry

end yellow_ball_range_l512_512562


namespace smallest_constant_C_l512_512182

theorem smallest_constant_C (n : ℕ) (hn : n ≥ 2) (x : ℕ → ℝ) (hx : ∀ i, 0 ≤ x i) :
  ∃ C, (∀ x1 : ℕ → ℝ, ∑ i in finset.range n, ∑ j in finset.range n, (x1 i * x1 j * (x1 i ^ 3 + x1 j ^ 3)) ≤ C * ((∑ i in finset.range n, x1 i) ^ 5)) ∧ C = 1 :=
sorry

end smallest_constant_C_l512_512182


namespace max_x_plus_2y_l512_512456

theorem max_x_plus_2y (x y : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : 4 * x^2 + 9 * y^2 = 36) :
  ∃ θ ∈ (0 : ℝ, π / 2 : ℝ), x = 3 * Real.cos θ ∧ y = 2 * Real.sin θ ∧ x + 2 * y = 5 := 
begin
  -- Statement includes conditions as per the problem description
  sorry
end

end max_x_plus_2y_l512_512456


namespace solve_equation_l512_512236

theorem solve_equation : (x : ℝ) (hx : x ≠ 1) (h : 3 / (x - 1) = 5 + 3 * x / (1 - x)) : x = 4 :=
sorry

end solve_equation_l512_512236


namespace median_is_in_category_3_l512_512841

def median_category_is_3 (n f1 f2 f3 f4 f5 : ℕ) : Prop :=
  (n = 60) →
  (f1 = 9) →
  (f2 = 12) →
  (f3 = 10) →
  (f5 = 12) →
  (f4 = 60 - f1 - f2 - f3 - f5) →
  (let frequencies := [f1, f2, f3, f4, f5] in 
   let cumulative_frequencies := list.scanl (+) 0 frequencies in
   ∃ k, k < 5 ∧ cumulative_frequencies.get! k < 60 / 2 ∧ 60 / 2 < cumulative_frequencies.get! (k+1) ∧
       k = 2)

theorem median_is_in_category_3 : median_category_is_3 60 9 12 10 17 12 :=
by
  sorry

end median_is_in_category_3_l512_512841


namespace cost_to_replace_is_800_l512_512153

-- Definitions based on conditions
def trade_in_value (num_movies : ℕ) (trade_in_price : ℕ) : ℕ :=
  num_movies * trade_in_price

def dvd_cost (num_movies : ℕ) (dvd_price : ℕ) : ℕ :=
  num_movies * dvd_price

def replacement_cost (num_movies : ℕ) (trade_in_price : ℕ) (dvd_price : ℕ) : ℕ :=
  dvd_cost num_movies dvd_price - trade_in_value num_movies trade_in_price

-- Problem statement: it costs John $800 to replace his movies
theorem cost_to_replace_is_800 (num_movies trade_in_price dvd_price : ℕ)
  (h1 : num_movies = 100) (h2 : trade_in_price = 2) (h3 : dvd_price = 10) :
  replacement_cost num_movies trade_in_price dvd_price = 800 :=
by
  -- Proof would go here
  sorry

end cost_to_replace_is_800_l512_512153


namespace julie_net_monthly_income_is_l512_512952

section JulieIncome

def starting_pay : ℝ := 5.00
def additional_experience_pay_per_year : ℝ := 0.50
def years_of_experience : ℕ := 3
def work_hours_per_day : ℕ := 8
def work_days_per_week : ℕ := 6
def bi_weekly_bonus : ℝ := 50.00
def tax_rate : ℝ := 0.12
def insurance_premium_per_month : ℝ := 40.00
def missed_days : ℕ := 1

-- Calculate Julie's net monthly income
def net_monthly_income : ℝ :=
    let hourly_wage := starting_pay + additional_experience_pay_per_year * years_of_experience
    let daily_earnings := hourly_wage * work_hours_per_day
    let weekly_earnings := daily_earnings * (work_days_per_week - missed_days)
    let bi_weekly_earnings := weekly_earnings * 2
    let gross_monthly_income := bi_weekly_earnings * 2 + bi_weekly_bonus * 2
    let tax_deduction := gross_monthly_income * tax_rate
    let total_deductions := tax_deduction + insurance_premium_per_month
    gross_monthly_income - total_deductions

theorem julie_net_monthly_income_is : net_monthly_income = 963.20 :=
    sorry

end JulieIncome

end julie_net_monthly_income_is_l512_512952


namespace integer_sequence_existence_l512_512850

theorem integer_sequence_existence
  (n : ℕ) (a : ℕ → ℤ) (A B C : ℤ) 
  (h1 : (a 1 < A ∧ A < B ∧ B < a n) ∨ (a 1 > A ∧ A > B ∧ B > a n))
  (h2 : ∀ i, 1 ≤ i ∧ i ≤ n - 1 → (a (i + 1) - a i ≤ 1 ∨ a (i + 1) - a i ≥ -1))
  (h3 : A ≤ C ∧ C ≤ B ∨ A ≥ C ∧ C ≥ B) :
  ∃ i, 1 < i ∧ i < n ∧ a i = C := sorry

end integer_sequence_existence_l512_512850


namespace subset_sum_divisible_by_n_l512_512177

theorem subset_sum_divisible_by_n (n : ℤ) (a : ℤ → ℤ) (h : ∀ i, 1 ≤ i ∧ i ≤ n → a i ∈ ℤ) :
  ∃ (I : Finset ℤ), I.nonempty ∧ (∑ i in I, a i) % n = 0 := sorry

end subset_sum_divisible_by_n_l512_512177


namespace constant_term_binomial_expansion_l512_512808

open BigOperators
open Nat

-- Define the binomial coefficient function
def binom (n k : ℕ) : ℕ := choose n k

-- Statement: Prove that the constant term in the expansion of (sqrt(x) - 1/x)^6 is 15
theorem constant_term_binomial_expansion (x : ℝ) :
  (6.choose 2 : ℤ) = 15 := by
  sorry

end constant_term_binomial_expansion_l512_512808


namespace number_of_real_solutions_l512_512793

theorem number_of_real_solutions :
  (∀ x : ℝ, 2^(2*x + 1) - 2^(x + 2) - 2^x + 2 = 0) → {x : ℝ | 2^(2*x + 1) - 2^(x + 2) - 2^x + 2 = 0}.card = 2 :=
by
  sorry

end number_of_real_solutions_l512_512793


namespace problem1_problem2_l512_512007

noncomputable def a_n (n : ℕ) : ℕ :=
  if n = 0 then 1 else 2^(n-1)

noncomputable def b_n (n : ℕ) : ℚ :=
  n / (a_n n)

noncomputable def T_n (n : ℕ) : ℚ :=
  ∑ i in Finset.range n, b_n (i + 1)

theorem problem1 (n : ℕ) (h : 0 < n) : a_n n = 2^(n-1) := by
  sorry

theorem problem2 (n : ℕ) : T_n n = 4 - (n + 2) * (1/2)^(n-1) := by
  sorry

end problem1_problem2_l512_512007


namespace angle_A_area_triangle_l512_512872

-- Define the given conditions
variables (A B C : ℝ) (a b c R : ℝ) 

-- Triangle's properties: sides opposite to angles A, B, and C are a, b, and c respectively.
-- Given conditions
assume h_b2 : b^2 = a^2 + c^2 - (real.sqrt 3) * a * c
assume h_c : c = (real.sqrt 3) * b
assume h_R : R = 2

/-- Problem Statements --/
theorem angle_A (h_b2 : b^2 = a^2 + c^2 - (real.sqrt 3) * a * c) (h_c : c = (real.sqrt 3) * b) : 
  A = π / 2 ∨ A = π / 6 :=
sorry

theorem area_triangle (h_b2 : b^2 = a^2 + c^2 - (real.sqrt 3) * a * c) (h_c : c = (real.sqrt 3) * b) (h_R : R = 2) : 
  (A = π / 2 → (1/2) * b * c * (real.sin A) = 2 * real.sqrt 3) ∧ 
  (A = π / 6 → (1/2) * b * c * (real.sin A) = real.sqrt 3) :=
sorry

end angle_A_area_triangle_l512_512872


namespace gcd_gt_one_l512_512000

-- Defining the given conditions and the statement to prove
theorem gcd_gt_one (a b x y : ℕ) (h : (a^2 + b^2) ∣ (a * x + b * y)) : 
  Nat.gcd (x^2 + y^2) (a^2 + b^2) > 1 := 
sorry

end gcd_gt_one_l512_512000


namespace cheapest_pie_l512_512409

def cost_flour : ℝ := 2
def cost_sugar : ℝ := 1
def cost_eggs_butter : ℝ := 1.5
def cost_crust : ℝ := cost_flour + cost_sugar + cost_eggs_butter

def weight_blueberries : ℝ := 3
def container_weight : ℝ := 0.5 -- 8 oz in pounds
def price_per_blueberry_container : ℝ := 2.25
def cost_blueberries (weight: ℝ) (container_weight: ℝ) (price_per_container: ℝ) : ℝ :=
  (weight / container_weight) * price_per_container

def weight_cherries : ℝ := 4
def price_cherry_bag : ℝ := 14

def cost_blueberry_pie : ℝ := cost_crust + cost_blueberries weight_blueberries container_weight price_per_blueberry_container
def cost_cherry_pie : ℝ := cost_crust + price_cherry_bag

theorem cheapest_pie : min cost_blueberry_pie cost_cherry_pie = 18 := by
  sorry

end cheapest_pie_l512_512409


namespace invalid_votes_percentage_is_correct_l512_512132

-- Definitions based on conditions
def total_votes : ℕ := 5500
def other_candidate_votes : ℕ := 1980
def valid_votes_percentage_other : ℚ := 0.45

-- Derived values
def valid_votes : ℚ := other_candidate_votes / valid_votes_percentage_other
def invalid_votes : ℚ := total_votes - valid_votes
def invalid_votes_percentage : ℚ := (invalid_votes / total_votes) * 100

-- Proof statement
theorem invalid_votes_percentage_is_correct :
  invalid_votes_percentage = 20 := sorry

end invalid_votes_percentage_is_correct_l512_512132


namespace num_factors_144_multiple_of_6_l512_512795

theorem num_factors_144_multiple_of_6 : 
  let n := 144
  let factors := (List.range (n + 1)).filter (λ d, n % d = 0)
  let multiples_of_6 := factors.filter (λ d, d % 6 = 0)
  multiples_of_6.length = 7 := by
  sorry

end num_factors_144_multiple_of_6_l512_512795


namespace roses_to_sister_l512_512099

theorem roses_to_sister (total_roses roses_to_mother roses_to_grandmother roses_kept : ℕ) 
  (h1 : total_roses = 20)
  (h2 : roses_to_mother = 6)
  (h3 : roses_to_grandmother = 9)
  (h4 : roses_kept = 1) : 
  total_roses - (roses_to_mother + roses_to_grandmother + roses_kept) = 4 :=
by
  sorry

end roses_to_sister_l512_512099


namespace total_toys_l512_512766

theorem total_toys (T : ℕ) (h1 : 0.7 * T = 140) : T = 200 :=
sorry

end total_toys_l512_512766


namespace relationship_of_y_values_l512_512474

theorem relationship_of_y_values (b y1 y2 y3 : ℝ) (h1 : y1 = 3 * (-3) - b)
                                (h2 : y2 = 3 * 1 - b)
                                (h3 : y3 = 3 * (-1) - b) :
  y1 < y3 ∧ y3 < y2 :=
by
  sorry

end relationship_of_y_values_l512_512474


namespace log_x_32_l512_512552

theorem log_x_32 (x : ℝ) (h : log 8 (5 * x) = 3) : log x 32 = 0.75 :=
sorry

end log_x_32_l512_512552


namespace cos_arithmetic_seq_product_l512_512046

theorem cos_arithmetic_seq_product :
  ∃ (a b : ℝ), (∃ (a₁ : ℝ), ∀ n : ℕ, (n > 0) → ∃ m : ℕ, cos (a₁ + (2 * Real.pi / 3) * (n - 1)) = [a, b] ∧ (a * b = -1 / 2)) := 
sorry

end cos_arithmetic_seq_product_l512_512046


namespace fraction_exponent_product_l512_512317

theorem fraction_exponent_product :
  (∀ (a : ℚ) (m : ℤ), a ≠ 0 → a^m * a^(-m) = a^(m - m)) →
  (∀ (a : ℚ), a ≠ 0 → a^0 = 1) →
  let a := (5 / 9 : ℚ)
  let m := (4 : ℤ)
  a^m * a^(-m) = 1 :=
by 
  intros H1 H2 a_ne_zero a m
  sorry

end fraction_exponent_product_l512_512317


namespace value_of_a_l512_512897

variable (x : ℝ) -- x is a real number
variables (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) -- a > 0 and a ≠ 1

theorem value_of_a :
  (∀ x, (a^ (x / 2) = 2^(- (x / 2)))) → a = 1/2 :=
by
  intros h
  -- Proof is omitted
  sorry

end value_of_a_l512_512897


namespace complement_union_A_B_in_U_l512_512522

def U : Set ℤ := {-2, -1, 0, 1, 2, 3}

def A : Set ℤ := {-1, 2}

def B : Set ℤ := {x | x^2 - 4 * x + 3 = 0}

theorem complement_union_A_B_in_U :
  (U \ (A ∪ B)) = {-2, 0} := by
  sorry

end complement_union_A_B_in_U_l512_512522


namespace cos_set_product_l512_512035

noncomputable def arithmetic_sequence (a1 : ℝ) (d : ℝ) : ℕ → ℝ :=
  λ n, a1 + d * (n - 1)

theorem cos_set_product (a1 : ℝ) (h1 : (arithmetic_sequence a1 (2 * Real.pi / 3) '' {n | n ≥ 1}).to_set.cos.to_finset.card = 2) :
  let S := (arithmetic_sequence a1 (2 * Real.pi / 3) '' {n | n ≥ 1}).to_set.cos.to_finset in 
  (S.to_finset : set ℝ).product = -1 / 2 := sorry

end cos_set_product_l512_512035


namespace necessary_not_sufficient_condition_l512_512497

theorem necessary_not_sufficient_condition (x : ℝ) : (x < 2) → (x^2 - x - 2 < 0) :=
by {
  sorry
}

end necessary_not_sufficient_condition_l512_512497


namespace greatest_consecutive_integers_sum_36_l512_512318

theorem greatest_consecutive_integers_sum_36 : ∀ (x : ℤ), (x + (x + 1) + (x + 2) = 36) → (x + 2 = 13) :=
by
  sorry

end greatest_consecutive_integers_sum_36_l512_512318


namespace complement_A_union_B_l512_512532

-- Define the universal set U
def U : Set ℤ := {-2, -1, 0, 1, 2, 3}

-- Define the set A
def A : Set ℤ := {-1, 2}

-- Define the set B using the quadratic equation condition
def B : Set ℤ := {x | x^2 - 4*x + 3 = 0}

-- State the theorem we want to prove
theorem complement_A_union_B : (U \ (A ∪ B)) = {-2, 0} := by
  sorry

end complement_A_union_B_l512_512532


namespace proof_problem_l512_512971

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := Real.log x - (1 / 2) * a * x^2
noncomputable def g (x : ℝ) (b : ℝ) : ℝ := Real.exp x - b * x

-- Define the tangent condition
def tangent_perpendicular_condition (a : ℝ) : Prop :=
  let slope_tangent := (1 - a)
  slope_tangent = -1

-- Define the monotonic intervals condition
def monotonic_intervals_condition (b : ℝ) : Prop :=
  (b ≤ 0 ∧ ∀ x, 0 < x → Real.exp x - b > 0) ∨
  (b > 0 ∧ ∃ ln_b, ln_b = Real.log b ∧ (∀ x, x < ln_b → Real.exp x - b < 0) ∧ (∀ x, x > ln_b → Real.exp x - b > 0))

-- Define the inequality condition
def inequality_condition (a : ℝ) (b : ℝ) : Prop :=
  ∀ x, 0 < x → b * f x a + b * x ≤ x * g x b

-- The final theorem
theorem proof_problem (a b : ℝ) : tangent_perpendicular_condition a →
                                   monotonic_intervals_condition b →
                                   inequality_condition a b → 
                                   a = 2 ∧ (b ∈ Set.Icc 0 Real.exp 1)
  := sorry

end proof_problem_l512_512971


namespace minimal_area_l512_512608

variable (t : ℝ)

def AB := 5
def BC := 4
def CA := 3

def ants_positions (t : ℝ) : Prop :=
  0 < t ∧ t < 3 ∧ 
  A(t) := (47 / 24)

theorem minimal_area (h₀ : 0 < t) (h₁ : t < 3) : t = 47 / 24 :=
  sorry

end minimal_area_l512_512608


namespace ellipse_properties_l512_512072

theorem ellipse_properties
  (a b c : ℝ) (h1 : a > b) (h2 : b > 0)
  (h_minor_axis : 2 * b = 2)
  (h_eccentricity : c / a = sqrt 6 / 3)
  (h2_eq_c2 : c^2 = a^2 - b^2)
  :
  (a = 3) ∧ (b = 1) ∧ (AOB_ctry : ∀ (k : ℝ), (k^2 > 1) → (k^2 < 13 / 3) → k ∈ (-sqrt 39 / 3, -1) ∪ (1, sqrt 39 / 3)) :=
sorry

end ellipse_properties_l512_512072


namespace max_value_of_A_l512_512052

theorem max_value_of_A (n : ℕ) (x : Fin n → ℝ) (hx : ∀ i, 0 < x i ∧ x i < 1) :
  let A := (∑ i, (1 - x i) ^ (1 / 4)) / (∑ i, 1 / (x i ^ (1 / 4))) in
  A ≤ (Real.sqrt 2) / 2 :=
by
  sorry

end max_value_of_A_l512_512052


namespace divide_fractions_l512_512705

def fraction := ℚ

theorem divide_fractions :
  (3 : fraction) / 4 / ((7 : fraction) / 8) = (6 / 7 : fraction) :=
  sorry

end divide_fractions_l512_512705


namespace fraction_division_l512_512713

theorem fraction_division (a b c d : ℚ) (h : b ≠ 0 ∧ d ≠ 0) : (a / b) / c = a / (b * c) := sorry

example : (3 / 7) / 4 = 3 / 28 := by
  apply fraction_division
  exact ⟨by norm_num, by norm_num⟩

end fraction_division_l512_512713


namespace area_of_stripe_correct_l512_512353

def diameter := 20
def height := 50
def stripe_width := 2
def revolutions := 3

noncomputable def area_of_stripe : ℝ :=
  let circumference := Real.pi * diameter
  let stripe_length := revolutions * circumference
  stripe_width * stripe_length

theorem area_of_stripe_correct : area_of_stripe = 240 * Real.pi :=
by
  sorry

end area_of_stripe_correct_l512_512353


namespace simplify_and_evaluate_expression_l512_512643

theorem simplify_and_evaluate_expression (x : ℤ) (hx : x = 3) : 
  (1 - (x / (x + 1))) / ((x^2 - 2 * x + 1) / (x^2 - 1)) = 1 / 2 := by
  rw [hx]
  -- Here we perform the necessary rewrites and simplifications as shown in the steps
  sorry

end simplify_and_evaluate_expression_l512_512643


namespace tower_height_l512_512661

theorem tower_height (a b c : ℕ) (H : ℝ) 
  (ha : a = 800) (hb : b = 700) (hc : c = 500) 
  (h_sum_angles : ∃ α β γ : ℝ, α + β + γ = 90 ∧ tan α = H / a ∧ tan β = H / b ∧ tan γ = H / c) :
  H = 100 * Real.sqrt 14 ∧ Real.round (H) = 374 :=
by 
  sorry

end tower_height_l512_512661


namespace common_internal_tangent_length_l512_512311

/-- Given two circles with specific center distances and radii, prove the length of the common internal tangent. -/
theorem common_internal_tangent_length
  (A B : Point)
  (rA rB : ℝ)
  (distance : ℝ) :
  ∥A - B∥ = distance →
  rA = 10 → rB = 7 →
  distance = 50 →
  ∃ CD : ℝ, CD = sqrt (distance^2 - (rA + rB)^2) ∧ CD = sqrt 2211 :=
by
  sorry

end common_internal_tangent_length_l512_512311


namespace tangent_lines_and_extreme_values_l512_512507

-- Definition of the function f
def f (x : ℝ) : ℝ := x^3 + 3*x^2 - 9*x - 3

-- Definition of the derivative of f
def f' (x : ℝ) : ℝ := 3*x^2 + 6*x - 9

-- Given line equation
def line (x y : ℝ) : Prop := x - 9*y + 1 = 0

theorem tangent_lines_and_extreme_values :
  (∀ x₀, f' x₀ = -9 → (
     let y₀ := f x₀ in 
     x₀ = 0 → y₀ = -3 ∧ tangent_line_eq f x₀ y₀ = "y = -9x - 3"
     ∧ 
     x₀ = -2 → y₀ = 19 ∧ tangent_line_eq f x₀ y₀ = "y = -9x + 19"
  ))
  ∧
  (∃ x₁, f' x₁ = 0 → (
    (x₁ = -3 ∧ f x₁ = 24) ∨ (x₁ = 1 ∧ f x₁ = -8)
  )) :=
sorry

-- Define tangent_line_eq function to match tangent line format
def tangent_line_eq (f : ℝ → ℝ) (x₀ y₀ : ℝ) : String := sorry


end tangent_lines_and_extreme_values_l512_512507


namespace triangle_XPY_area_result_valid_l512_512143

noncomputable def area_XPY (XM YN XY : ℝ) (a b : ℕ) : ℝ :=
  let XZ := 20 * Real.sqrt 2
  let area := 50
  if h : a * Real.sqrt b = area then a + b else 0

theorem triangle_XPY_area_result_valid :
  ∀ (XM YN XY : ℝ) (a b : ℕ),
  XM = 15 ∧ YN = 20 ∧ XY = 20 ∧ b = 1 →
  area_XPY XM YN XY a b = 51 :=
by
  intros XM YN XY a b h
  cases h with hXM h1
  cases h1 with hYN h2
  cases h2 with hXY hb
  simp [area_XPY, hXM, hYN, hXY, hb]
  sorry

end triangle_XPY_area_result_valid_l512_512143


namespace problem_statement_l512_512110

def B : ℂ := 5 - 2 * complex.I
def N : ℂ := -5 + 2 * complex.I
def T : ℂ := 0 + 2 * complex.I
def Q : ℂ := 3

theorem problem_statement :
  B - N + T - Q = 7 - 2 * complex.I :=
by
  sorry

end problem_statement_l512_512110


namespace fraction_exponent_product_l512_512316

theorem fraction_exponent_product :
  (∀ (a : ℚ) (m : ℤ), a ≠ 0 → a^m * a^(-m) = a^(m - m)) →
  (∀ (a : ℚ), a ≠ 0 → a^0 = 1) →
  let a := (5 / 9 : ℚ)
  let m := (4 : ℤ)
  a^m * a^(-m) = 1 :=
by 
  intros H1 H2 a_ne_zero a m
  sorry

end fraction_exponent_product_l512_512316


namespace cos_arithmetic_seq_product_l512_512050

theorem cos_arithmetic_seq_product :
  ∃ (a b : ℝ), (∃ (a₁ : ℝ), ∀ n : ℕ, (n > 0) → ∃ m : ℕ, cos (a₁ + (2 * Real.pi / 3) * (n - 1)) = [a, b] ∧ (a * b = -1 / 2)) := 
sorry

end cos_arithmetic_seq_product_l512_512050


namespace dogs_in_pet_shop_l512_512339

variable (D C B x : ℕ)

theorem dogs_in_pet_shop 
  (h1 : D = 7 * x) 
  (h2 : B = 8 * x)
  (h3 : D + B = 330) : 
  D = 154 :=
by
  sorry

end dogs_in_pet_shop_l512_512339


namespace contemporaries_probability_l512_512700

theorem contemporaries_probability:
  (∀ (x y : ℝ),
    0 ≤ x ∧ x ≤ 400 ∧
    0 ≤ y ∧ y ≤ 400 ∧
    (x < y + 80) ∧ (y < x + 80)) →
    (∃ p : ℝ, p = 9 / 25) :=
by sorry

end contemporaries_probability_l512_512700


namespace odd_function_zeros_l512_512492

noncomputable def f (x : ℝ) : ℝ :=
if x < 0 then x - 3 / x + 2 
else if x = 0 then 0 
else x - 3 / x - 2

theorem odd_function_zeros :
  (∀ x : ℝ, f (-x) = -f(x)) ∧
  (f (-3) = 0 ∧ f 0 = 0 ∧ f 3 = 0) :=
by
  -- Proof omitted
  sorry

end odd_function_zeros_l512_512492


namespace tangent_line_derivative_positive_l512_512921

theorem tangent_line_derivative_positive (f : ℝ → ℝ) (x : ℝ) (h : ∀ y, y = 2*x - 1 → tangent_line f (x, f x) y) : f' x > 0 :=
sorry

end tangent_line_derivative_positive_l512_512921


namespace percentage_relationships_l512_512979

variable (a b c d e f g : ℝ)

theorem percentage_relationships (h1 : d = 0.22 * b) (h2 : d = 0.35 * f)
                                 (h3 : e = 0.27 * a) (h4 : e = 0.60 * f)
                                 (h5 : c = 0.14 * a) (h6 : c = 0.40 * b)
                                 (h7 : d = 2 * c) (h8 : g = 3 * e):
    b = 0.7 * a ∧ f = 0.45 * a ∧ g = 0.81 * a :=
sorry

end percentage_relationships_l512_512979


namespace product_of_cosines_of_two_distinct_values_in_S_l512_512018

noncomputable def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + d

theorem product_of_cosines_of_two_distinct_values_in_S
  (a : ℕ → ℝ)
  (d : ℝ)
  (h_sequence: arithmetic_sequence a (2 * Real.pi / 3))
  (hS : ∃ (b c : ℝ), b ≠ c ∧ (∀ n : ℕ, ∃b', b' = cos (a n) ∧ (b' = b ∨ b' = c))) :
  (∃ b c : ℝ, b ≠ c ∧ (∀ n : ℕ, cos (a n) = b ∨ cos (a n) = c) ∧ b * c = -1 / 2) := 
sorry

end product_of_cosines_of_two_distinct_values_in_S_l512_512018


namespace tangent_lines_to_circle_l512_512002

noncomputable theory
open Real

def circle_eqn (x y k : ℝ) : ℝ := x^2 + y^2 + k * x + 2 * y + k^2 - 15

theorem tangent_lines_to_circle :
  {k : ℝ | ∃ (x y : ℝ), circle_eqn x y k = 0} ∩
  {k : ℝ | ∀ (x y : ℝ), circle_eqn x 2 k > 0 → x ≠ 1 ∨ y ≠ 2} = 
  {k : ℝ | - (8 * sqrt 3 / 3) < k ∧ k < -3} ∪ {k : ℝ | 2 < k ∧ k < 8 * sqrt 3 / 3} :=
sorry

end tangent_lines_to_circle_l512_512002


namespace vector_projection_and_magnitude_l512_512541

variables {x : ℝ}
def a := (1 : ℝ, -1 : ℝ)
def b := (2 : ℝ, x : ℝ)

def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2
def magnitude (v : ℝ × ℝ) : ℝ := Real.sqrt (v.1^2 + v.2^2)

theorem vector_projection_and_magnitude (h_proj : (dot_product a b) / magnitude a = -Real.sqrt 2) :
  x = 4 ∧ magnitude b = 2 * Real.sqrt 5 :=
by
  sorry

end vector_projection_and_magnitude_l512_512541


namespace Barbara_wins_1000_coins_Jenna_wins_1001_coins_l512_512932

-- Definitions based on the conditions
def canRemoveCoins (player: String) (n: Nat) : Bool :=
  match player with
  | "Barbara" => n > 0 ∧ (n = 1 ∨ n = 5)
  | "Jenna"   => n > 0 ∧ (n = 2 ∨ n = 4)
  | _         => false

def winningPosition (n: Nat) : Bool :=
  match n % 6 with
  | 1 | 2 | 4 | 5 => true
  | _             => false

-- Theorem to state that Barbara wins with 1000 coins
theorem Barbara_wins_1000_coins : (n: Nat) (h : n = 1000) → (canRemoveCoins "Barbara" n) → winningPosition n →
  winningPosition n :=
by
  intros
  sorry

-- Theorem to state that Jenna wins with 1001 coins
theorem Jenna_wins_1001_coins : (n: Nat) (h : n = 1001) → (canRemoveCoins "Jenna" n) → winningPosition n →
  winningPosition n :=
by
  intros
  sorry

end Barbara_wins_1000_coins_Jenna_wins_1001_coins_l512_512932


namespace zeta_sum_diverges_to_infinity_l512_512825

noncomputable theory

open Real

def riemann_zeta (x : ℝ) : ℝ := ∑' n : ℕ, 1 / (n + 1 : ℝ)^x

theorem zeta_sum_diverges_to_infinity : (∑ k in (filter (λ k, even k) (Icc 2 ∞)), ⌊riemann_zeta (2 * k)⌋) = ⊤ :=
by
  sorry

end zeta_sum_diverges_to_infinity_l512_512825


namespace cyclic_intersections_of_circles_on_quadrilateral_l512_512633

theorem cyclic_intersections_of_circles_on_quadrilateral
  (A B C D K L M N : Point)
  (h1 : InscribedQuadrilateral A B C D)
  (h2 : OnCircle K A B)
  (h3 : OnCircle L B C)
  (h4 : OnCircle M C D)
  (h5 : OnCircle N A D)
  (hKL_not_Corners : K ≠ A ∧ K ≠ B ∧ K ≠ C ∧ K ≠ D)
  (hLM_not_Corners : L ≠ A ∧ L ≠ B ∧ L ≠ C ∧ L ≠ D)
  (hMN_not_Corners : M ≠ A ∧ M ≠ B ∧ M ≠ C ∧ M ≠ D)
  (hNK_not_Corners : N ≠ A ∧ N ≠ B ∧ N ≠ C ∧ N ≠ D) :
  CyclicQuadrilateral K L M N :=
sorry

end cyclic_intersections_of_circles_on_quadrilateral_l512_512633


namespace problem_solution_l512_512881

noncomputable def eq_roots_and_mn : ℂ :=
  let z := -1 + complex.I in
  let m := 0 in
  let n := 1 in
  have h1 : z + 1 = complex.I := by sorry,
  have h2 : (complex.I : ℂ) + (-complex.I) = -0 := by simp,
  have h3 : -complex.I^2 = 1 := by simp,
  ⟨m, n, h1, h2, h3⟩ 

theorem problem_solution :
  ∃ m n : ℝ, ∀ z : ℂ, z = -1 + complex.I → z + 1 = complex.I → z + 1 ≠ -complex.I → 
    m = 0 ∧ n = 1 :=
by
  use 0, 1
  intros z hz1 hz2 hz3
  split
  . exact rfl
  . exact rfl

  sorry

end problem_solution_l512_512881


namespace div_fraction_fraction_division_eq_l512_512706

theorem div_fraction (a b : ℕ) (h : b ≠ 0) : (a : ℚ) / b = (a : ℚ) * (1 / (b : ℚ)) := 
by sorry

theorem fraction_division_eq : (3 : ℚ) / 7 / 4 = 3 / 28 := 
by 
  calc
    (3 : ℚ) / 7 / 4 = (3 / 7) * (1 / 4) : by rw [div_fraction] 
                ... = 3 / 28            : by normalization_tactic -- Use appropriate tactic for simplification
                ... = 3 / 28            : by rfl

end div_fraction_fraction_division_eq_l512_512706


namespace remaining_meal_for_children_l512_512753

theorem remaining_meal_for_children 
  (total_adults : ℕ) (total_children : ℕ) 
  (meal_for_adults : ℕ) (meal_for_children : ℕ) 
  (adults_had_meal : ℕ) :
  total_adults = 55 ∧ total_children = 70 ∧
  meal_for_adults = 70 ∧ meal_for_children = 90 ∧
  adults_had_meal = 7 →
  let children_per_adult_meal := meal_for_children / meal_for_adults in
  let equivalent_children_for_adults := adults_had_meal * children_per_adult_meal in
  let remaining_children_meals := meal_for_children - equivalent_children_for_adults in
  remaining_children_meals = 81 := sorry

end remaining_meal_for_children_l512_512753


namespace four_digit_odd_numbers_count_l512_512703

/-- The number of four-digit odd numbers that can be formed
using the digits 0, 1, 2, 3, 4, 5 without repeating any digit
is equal to 144. -/
theorem four_digit_odd_numbers_count : ∃ n, n = 144 ∧
  ∀ d1 d2 d3 d4 ∈ {0, 1, 2, 3, 4, 5},
  (d1 ≠ d2 ∧ d1 ≠ d3 ∧ d1 ≠ d4 ∧ d2 ≠ d3 ∧ d2 ≠ d4 ∧ d3 ≠ d4) →
  (d1 ≠ 0) →
  (d4 = 1 ∨ d4 = 3 ∨ d4 = 5) →
  d1 * 1000 + d2 * 100 + d3 * 10 + d4 = n :=
sorry

end four_digit_odd_numbers_count_l512_512703


namespace part1_i_part1_ii_part2_l512_512937

variables {x y t : ℝ} {P : ℝ × ℝ} {A B M Q R N : ℝ × ℝ}
def circle (O : ℝ × ℝ) (r : ℝ) : set (ℝ × ℝ) := {P | (P.1 - O.1)^2 + (P.2 - O.2)^2 = r^2}
def line (k b : ℝ) : set (ℝ × ℝ) := {P | P.2 = k * P.1 + b}
def line_t (t : ℝ) : set (ℝ × ℝ) := {P | P.1 = t}
def distance (P Q : ℝ × ℝ) := sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)

theorem part1_i :
  let P := (4/3, 1) in
  P ∈ circle (0, 0) 1 →
  distance (0, 0) P = 5 / 3 →
  ∃ k : ℝ, -1 / k ∈ [0, 24 / 7] ∧ (∀ {x y : ℝ},
    P ∈ line k (1 - k * P.1)) :=
sorry

theorem part1_ii :
  let P := (4/3, y) in
  P ∈ line_t t →
  (1 < t ∧ t < 2) →
  ∃ (A B : ℝ × ℝ),
    A ∈ circle (0, 0) 1 ∧ B ∈ circle (0, 0) 1
    ∧ B = midpoint A P ∧
    (B.1 + A.1) / 2 = P.1 →
  abs(y) ≤ sqrt 65 / 3 :=
sorry

theorem part2 :
  let M := (t, 0), Q := (t / 2, 0) in
  (1 < t ∧ t < 2) →
  R ∈ circle (0, 0) 1 →
  distance R M = 1 →
  ∃ N : ℝ × ℝ,
    N ∈ circle (0, 0) 1 ∧
    N ≠ R ∧
    distance N Q = sqrt (14) / 8 :=
sorry

end part1_i_part1_ii_part2_l512_512937


namespace area_triangle_BQW_eq_96_l512_512940

-- Definitions based on conditions
def Rectangle (A B C D Z W Q : Type) :=
  (ABCD : Rectangle) (AZ = 8) (WC = 8) (AB = 16)
-- we will assume the function Area exists though it might need defining in a real environment.
def Trapezoid (ZWCD : Trapezoid) :=
  area(ZWCD) = 192

-- Main theorem statement
theorem area_triangle_BQW_eq_96 (A B C D Z W Q : Type )
  [Rectangle ABCD : Rectangle := sorry]
  [Trapezoid ZWCD : Trapezoid := sorry]
  (midpoint(ZW, Q) : bool := sorry ) :
  ∃ B Q W t : Type, area(BQW) = 96 :=
begin
  exact sorry
end

end area_triangle_BQW_eq_96_l512_512940


namespace seller_is_cheating_l512_512302

theorem seller_is_cheating (L1 L2 : ℝ) (h : L1 ≠ L2) (product_weight : ℝ) : 
  (0 < product_weight) → ((product_weight + (product_weight * L2 / L1)) / 2 > product_weight) := 
by
  intro h1,
  sorry

end seller_is_cheating_l512_512302


namespace fraction_BC_AE_l512_512197

noncomputable theory

variables (A B C D E : Type) [linear_ordered_field A] 
variables (distance : A → A → A)
variables (AB BD CD AD DE AE BC : A)
variables (x y : A)

-- Given conditions
def cond1 := AB = 3 * BD
def cond2 := AD = 5 * CD
def cond3 := DE = AD
def cond4 := AD = 4 * BD

-- Conjecture to prove
theorem fraction_BC_AE (cond1 : cond1) (cond2 : cond2) (cond3 : cond3) (AD_eq_4BD : cond4) :
  BC = (1/40) * AE :=
sorry  -- proof not provided

end fraction_BC_AE_l512_512197


namespace ab_cd_value_l512_512106

variables (a b c d : ℝ)

theorem ab_cd_value 
  (h1 : a + b + c = 5) 
  (h2 : a + b + d = 9) 
  (h3 : a + c + d = 20) 
  (h4 : b + c + d = 13) : 
  ab + cd = 72 :=
by
  -- We define intermediate results that we would have derived
  let a_val := (47 / 3) - 13
  let b_val := (47 / 3) - 20
  let c_val := (47 / 3) - 9
  let d_val := (47 / 3) - 5

  -- We assert these intermediate results
  have ha : a = a_val := sorry
  have hb : b = b_val := sorry
  have hc : c = c_val := sorry
  have hd : d = d_val := sorry

  -- Substitute these into ab + cd to get the correct answer
  calc
  ab + cd 
      = (a_val * b_val) + (c_val * d_val) : sorry
  ... = 72 : sorry

end ab_cd_value_l512_512106


namespace bad_mood_dwarves_l512_512350

inductive Dwarf
| Весельчак | Ворчун | Простачок | Скромник | Соня | Умник | Чихун

open Dwarf

variables (mood : Dwarf → Prop)
variables (hatsOff : Dwarf → Prop)

-- Conditions
axiom initial_condition : ∀ d, hatsOff d
axiom second_condition : hatsOff Простачок ∧ (∀ d ≠ Простачок, ¬ hatsOff d)

-- Answer
theorem bad_mood_dwarves : ∃ (bad : Dwarf → Prop), bad Скромник ∧ bad Соня ∧ bad Чихун :=
sorry

end bad_mood_dwarves_l512_512350


namespace sample_std_dev_range_same_l512_512853

noncomputable def sample_std_dev (data : List ℝ) : ℝ := sorry
noncomputable def sample_range (data : List ℝ) : ℝ := sorry

theorem sample_std_dev_range_same (n : ℕ) (c : ℝ) (Hc : c ≠ 0) (x : Fin n → ℝ) :
  sample_std_dev (List.ofFn (λ i => x i)) = sample_std_dev (List.ofFn (λ i => x i + c)) ∧
  sample_range (List.ofFn (λ i => x i)) = sample_range (List.ofFn (λ i => x i + c)) :=
by
  sorry

end sample_std_dev_range_same_l512_512853


namespace triangles_incongruent_with_two_sides_and_non_included_angle_l512_512726

theorem triangles_incongruent_with_two_sides_and_non_included_angle : 
  ∀ (Δ1 Δ2 : Triangle) (a1 a2 b1 b2 : Side) (α1 α2 : Angle), 
  Δ1.side_equal a1 a2 → Δ1.side_equal b1 b2 → Δ1.angle_equal α1 α2 → 
  ¬ (Δ1.congruent Δ2) :=
by sorry

end triangles_incongruent_with_two_sides_and_non_included_angle_l512_512726


namespace find_phi_l512_512078

noncomputable def f (x : ℝ) (φ : ℝ) : ℝ := Real.tan (3 * x + φ)

-- Conditions
def symmetric_about (x y : ℝ) (f : ℝ → ℝ) : Prop :=
  ∀ t : ℝ, f (x + t) = 2 * y - f (x - t)

def condition1  := ∀ (φ : ℝ), |φ| ≤ Real.pi / 4
def condition2 (φ : ℝ) := symmetric_about (-Real.pi / 9) 0 (f φ)

-- The Lean statement to be proven
theorem find_phi : ∃ (φ : ℝ), condition1 φ ∧ condition2 φ ∧ φ = -Real.pi / 6 :=
by {
  sorry
}

end find_phi_l512_512078


namespace eccentricity_proof_l512_512657

noncomputable def eccentricity_of_hyperbola 
  (a b : ℝ) (h₁ : a > 0) (h₂ : b > 0)
  (h_asymptotes : ∀ x y : ℝ, 4 * a * x + b * y = 0 ∨ 4 * a * x - b * y = 0) : ℝ :=
  real.sqrt (1 + (b / a)^2)

theorem eccentricity_proof 
  (a b : ℝ) (h₁ : a > 0) (h₂ : b > 0)
  (h_asymptotes : ∀ x y : ℝ, 4 * a * x + b * y = 0 ∨ 4 * a * x - b * y = 0) :
  eccentricity_of_hyperbola a b h₁ h₂ h_asymptotes = real.sqrt 5 :=
sorry

end eccentricity_proof_l512_512657


namespace parabola_AB_length_l512_512490

noncomputable theory

open Real

def parabola_focus (p : ℝ) : Prop :=
  (p / 2 = 2)

def parabola_equation (x y p : ℝ) : Prop :=
  y^2 = 2 * p * x

def intersection_y_coordinates (x y1 y2 : ℝ) : Prop :=
  (y1^2 = 8 * x) ∧ (y2^2 = 8 * x) ∧ (y1 > 0) ∧ (y2 < 0)

def length_AB (y1 y2 : ℝ) : ℝ :=
  abs (y1 - y2)

theorem parabola_AB_length :
  ∃ (p x y1 y2 : ℝ), 
    parabola_focus p ∧
    parabola_equation 4 y1 p ∧
    parabola_equation 4 y2 p ∧
    intersection_y_coordinates 4 y1 y2 ∧
    length_AB y1 y2 = 8 * sqrt 2 :=
by
  sorry

end parabola_AB_length_l512_512490


namespace angle_BCG_eq_angle_BCF_l512_512166

variables (A B C D E F G : Type) [geometry A B C D E F G]

-- Let ABC be a triangle
variable (triangle_ABC : triangle A B C)

-- Point D on segment AC such that BD = CD
variable (D_on_AC : point_on_segment D A C)
variable (isosceles_triangle_BDC : isosceles_triangle B D C)

-- A line parallel to BD intersects segment BC at E and intersects line AB at F
variable (line_parallel_BD : parallel_line BD E F)
variable (E_on_BC : point_on_segment E B C)
variable (F_on_AB : point_on_line F A B)

-- The intersection point of lines AE and BD is G
variable (G_intersection_AE_BD : intersection_point G AE BD)

-- Show that ∠BCG = ∠BCF
theorem angle_BCG_eq_angle_BCF : angle B C G = angle B C F :=
sorry

end angle_BCG_eq_angle_BCF_l512_512166


namespace find_r_l512_512930

-- Define the problem setup
variables {A B C D E F M N : ℂ} -- vertices of the regular hexagon and points M, N
variables (r : ℝ) (hexagon : (A B C D E F : ℂ))
noncomputable def divides (x y : ℂ) (k : ℝ) : Prop := x = (1 - k) * A + k * C

-- Given conditions
axiom regular_hexagon : regular_hexagon ABCDEF
axiom divides_M : divides M A C r
axiom divides_N : divides N C E r
axiom collinear_B_M_N : collinear (B, M, N)

-- The theorem to be proved
theorem find_r : r = Nat.sqrt 3 / 3 :=
  sorry

end find_r_l512_512930


namespace fixed_point_theorem_l512_512008

noncomputable def S_n (n : ℕ) : set ℕ := {i | 1 ≤ i ∧ i ≤ n}

def permutation (P : ℕ → ℕ) (n : ℕ) : Prop := 
  ∀ (x : ℕ), x ∈ S_n n → P x ∈ S_n n

def fixed_point (P : ℕ → ℕ) (j : ℕ) (n : ℕ) : Prop := 
  P j = j ∧ j ∈ S_n n

def f_n (n : ℕ) : ℕ :=
  sorry -- The definition of f_n which counts permutations of S_n with no fixed points

def g_n (n : ℕ) : ℕ :=
  sorry -- The definition of g_n which counts permutations of S_n with exactly one fixed point

theorem fixed_point_theorem (n : ℕ) (P : ℕ → ℕ) : 
  permutation P n → |f_n n - g_n n| = 1 :=
sorry

end fixed_point_theorem_l512_512008


namespace C_should_pay_72_l512_512381

noncomputable def C_share_of_rent (oxenA, monthsA, oxenB, monthsB, oxenC, monthsC, total_rent) :=
  let oxen_months_A := oxenA * monthsA in
  let oxen_months_B := oxenB * monthsB in
  let oxen_months_C := oxenC * monthsC in
  let total_oxen_months := oxen_months_A + oxen_months_B + oxen_months_C in
  let C_fraction := oxen_months_C.to_rat / total_oxen_months.to_rat in
  (total_rent.to_rat * C_fraction).to_int

theorem C_should_pay_72 :
  C_share_of_rent 10 7 12 5 15 3 280 = 72 :=
begin
  sorry
end

end C_should_pay_72_l512_512381


namespace proof_of_problem_l512_512165

noncomputable def problem_statement (a b c x y z : ℝ) : Prop :=
  23 * x + b * y + c * z = 0 ∧
  a * x + 33 * y + c * z = 0 ∧
  a * x + b * y + 52 * z = 0 ∧
  a ≠ 23 ∧
  x ≠ 0 →
  (a + 10) / (a + 10 - 23) + b / (b - 33) + c / (c - 52) = 1

theorem proof_of_problem (a b c x y z : ℝ) (h : problem_statement a b c x y z) : 
  (a + 10) / (a + 10 - 23) + b / (b - 33) + c / (c - 52) = 1 :=
sorry

end proof_of_problem_l512_512165


namespace cos_set_product_l512_512033

noncomputable def arithmetic_sequence (a1 : ℝ) (d : ℝ) : ℕ → ℝ :=
  λ n, a1 + d * (n - 1)

theorem cos_set_product (a1 : ℝ) (h1 : (arithmetic_sequence a1 (2 * Real.pi / 3) '' {n | n ≥ 1}).to_set.cos.to_finset.card = 2) :
  let S := (arithmetic_sequence a1 (2 * Real.pi / 3) '' {n | n ≥ 1}).to_set.cos.to_finset in 
  (S.to_finset : set ℝ).product = -1 / 2 := sorry

end cos_set_product_l512_512033


namespace part1_solution_part2_solution_l512_512742

-- Definitions for conditions
def mixed_number_1 := (25 : ℝ) / 9
def mixed_number_2 := (64 : ℝ) / 27
def pi_term := 1 -- Because π^0 = 1

-- Theorem statement for part 1
theorem part1_solution :
  (mixed_number_1^0.5 + 0.1^(-2) + mixed_number_2^(-2 / 3) - 3 * pi_term + 37 / 48) = 100 := sorry

-- Definitions for polynomial roots based on Vieta's formulas from conditions
variables (a b : ℝ)

-- conditions to check the inequality and root equation
axiom root_eq : a^2 - 6 * a + 4 = 0
axiom root_eq_other : b^2 - 6 * b + 4 = 0
axiom roots_inequality : a > b ∧ b > 0

-- Theorem statement for part 2
theorem part2_solution (h1 : a^2 - 6 * a + 4 = 0) (h2 : b^2 - 6 * b + 4 = 0) (h3: a > b ∧ b > 0):
  (sqrt a - sqrt b) / (sqrt a + sqrt b) = sqrt 2 / 2 := sorry

end part1_solution_part2_solution_l512_512742


namespace volume_behavior_l512_512488

noncomputable def F (x : ℝ) : ℝ := (1/12) * Real.sqrt(3 * x^2 - x^4)

theorem volume_behavior :
  ∃ x_max, (∀ x ≥ 0, F x ≤ F x_max) ∧ (¬ ∀ x₁ x₂, x₁ < x₂ → F x₁ < F x₂) :=
by
  -- Place the definitions of F, conditions, and conclusion here
  sorry

end volume_behavior_l512_512488


namespace price_of_mixture_l512_512580

theorem price_of_mixture (p1 p2 : ℝ) (r : ℝ) (h1 : p1 = 62) (h2 : p2 = 72) (h3 : r = 1) : (p1 + p2) / 2 = 67 :=
by
  rw [h1, h2, h3]
  sorry

end price_of_mixture_l512_512580


namespace polar_bear_paradox_l512_512997

theorem polar_bear_paradox
  (d_earth : ℝ) (d_floe : ℝ)
  (h1 : d_earth = 8.5) (h2 : d_floe = 9)
  (h3 : ∀ (measurements_correct : Prop), measurements_correct) :
  ∃ (mass_ratio : ℝ), mass_ratio > 1 ∧ 
                       abs (d_floe / 2 - d_earth / 2) = 0.25 :=
by
  use 10 -- Assume an example mass ratio of 10.
  split
  . exact zero_lt_ten
  . calc
    abs (4.5 - 4.25) = abs (0.25) : by norm_num
                ... = 0.25 : abs_of_nonneg (by norm_num)

end polar_bear_paradox_l512_512997


namespace alex_friends_invite_l512_512382

theorem alex_friends_invite (burger_buns_per_pack : ℕ)
                            (packs_of_buns : ℕ)
                            (buns_needed_by_each_guest : ℕ)
                            (total_buns : ℕ)
                            (friends_who_dont_eat_buns : ℕ)
                            (friends_who_dont_eat_meat : ℕ)
                            (total_friends_invited : ℕ) 
                            (h1 : burger_buns_per_pack = 8)
                            (h2 : packs_of_buns = 3)
                            (h3 : buns_needed_by_each_guest = 3)
                            (h4 : total_buns = packs_of_buns * burger_buns_per_pack)
                            (h5 : friends_who_dont_eat_buns = 1)
                            (h6 : friends_who_dont_eat_meat = 1)
                            (h7 : total_friends_invited = (total_buns / buns_needed_by_each_guest) + friends_who_dont_eat_buns) :
  total_friends_invited = 9 :=
by sorry

end alex_friends_invite_l512_512382


namespace problem_1_parity_of_f_when_a_is_zero_problem_2_increasing_f_problem_3_range_of_t_l512_512080

noncomputable def f (a x : ℝ) := x * |2 * a - x| + 2 * x

theorem problem_1_parity_of_f_when_a_is_zero :
  ∀ x : ℝ, f 0 x = - f 0 (-x) := sorry

theorem problem_2_increasing_f :
  ∀ a : ℝ, (-1 : ℝ) ≤ a ∧ a ≤ 1 ↔ ∀ x y : ℝ, x ≤ y → f a x ≤ f a y := sorry

theorem problem_3_range_of_t :
  ∃ a : ℝ, (-2 : ℝ) ≤ a ∧ a ≤ 2 ∧ (∃ x1 x2 x3 : ℝ, 
    x1 ≠ x2 ∧ x2 ≠ x3 ∧ x1 ≠ x3 ∧ 
    (f a x1 - t * f (2 * a) = 0) ∧ (f a x2 - t * f (2 * a) = 0) ∧ 
    (f a x3 - t * f (2 * a) = 0)) → 
  (1 : ℝ) < t ∧ t < (9/8 : ℝ) := sorry

end problem_1_parity_of_f_when_a_is_zero_problem_2_increasing_f_problem_3_range_of_t_l512_512080


namespace unit_squares_line_division_l512_512223

theorem unit_squares_line_division (d : ℝ) (h : ∃ (d : ℝ), line_eq d ∧ 8 - 2 * d = 3) : d = 2.5 :=
sorry

-- Definitions to support the theorem
def line_eq (d : ℝ) : Prop := ∀ x y : ℝ, (4 - d) * y = 4 * (x - d)

end unit_squares_line_division_l512_512223


namespace download_time_correct_l512_512991

-- Define the given conditions
def total_size : ℕ := 880
def downloaded : ℕ := 310
def speed : ℕ := 3

-- Calculate the remaining time to download
def time_remaining : ℕ := (total_size - downloaded) / speed

-- Theorem statement that needs to be proven
theorem download_time_correct : time_remaining = 190 := by
  -- Proof goes here
  sorry

end download_time_correct_l512_512991


namespace car_trader_profit_l512_512767

theorem car_trader_profit (P : ℝ) : 
  let purchase_price := 0.80 * P
  let selling_price := 1.28000000000000004 * P
  let profit := selling_price - purchase_price
  let percentage_increase := (profit / purchase_price) * 100
  percentage_increase = 60 := 
by
  sorry

end car_trader_profit_l512_512767


namespace cyclic_tangential_quadrilateral_l512_512205

open Real

variables {a b c d : ℝ} (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d)

-- Define the semi-perimeter
noncomputable def s := (a + b + c + d) / 2

-- Define the area
noncomputable def t := sqrt (a * b * c * d)

-- Define the tangent of half-angle
noncomputable def tan_half_angle := sqrt (c * d / (a * b))

-- Define the radius of the inscribed circle
noncomputable def r := sqrt (a * b * c * d) / (a + c)

theorem cyclic_tangential_quadrilateral
  (h_cyclic : false)  -- cyclic condition not used explicitly, marked as false for now
  (h_tangential : s - a = s - b ∧ s - c = s - d) :
  t = sqrt (a * b * c * d) ∧
  tan_half_angle = sqrt (c * d / (a * b)) ∧
  r = sqrt (a * b * c * d) / (a + c) := by
  sorry

end cyclic_tangential_quadrilateral_l512_512205


namespace suzanna_bike_distance_l512_512285

variable (constant_rate : ℝ) (time_minutes : ℝ) (interval : ℝ) (distance_per_interval : ℝ)

theorem suzanna_bike_distance :
  (constant_rate = 1 / interval) ∧ (interval = 5) ∧ (distance_per_interval = constant_rate * interval) ∧ (time_minutes = 30) →
  ((time_minutes / interval) * distance_per_interval = 6) :=
by
  intros
  sorry

end suzanna_bike_distance_l512_512285


namespace infinite_composite_numbers_divisibility_l512_512640

theorem infinite_composite_numbers_divisibility (a b : ℕ) (p : ℕ) (hp : prime p) (hab_coprime : coprime a b) (ha_gt_b : a > b) :
  ∃ (n : ℕ), ∃ (m : ℕ), (3^(n-1) - 2^(n-1)) % n = 0 ∧ composite n :=
sorry

end infinite_composite_numbers_divisibility_l512_512640


namespace minimum_value_expression_l512_512794

theorem minimum_value_expression : ∀ x y : ℝ, ∃ (mx my : ℝ), (x = mx → y = my → x^2 + y^2 - 8*x + 6*y + 25 = 0) :=
begin
  sorry
end

end minimum_value_expression_l512_512794


namespace cos_set_product_l512_512031

noncomputable def arithmetic_sequence (a1 : ℝ) (d : ℝ) : ℕ → ℝ :=
  λ n, a1 + d * (n - 1)

theorem cos_set_product (a1 : ℝ) (h1 : (arithmetic_sequence a1 (2 * Real.pi / 3) '' {n | n ≥ 1}).to_set.cos.to_finset.card = 2) :
  let S := (arithmetic_sequence a1 (2 * Real.pi / 3) '' {n | n ≥ 1}).to_set.cos.to_finset in 
  (S.to_finset : set ℝ).product = -1 / 2 := sorry

end cos_set_product_l512_512031


namespace sum_is_square_l512_512100

theorem sum_is_square (a b c : ℕ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : c ≠ 0) 
  (h4 : Nat.gcd a b = 1) (h5 : Nat.gcd b c = 1) (h6 : Nat.gcd c a = 1) 
  (h7 : (1:ℚ)/a + (1:ℚ)/b = (1:ℚ)/c) : ∃ k : ℕ, a + b = k ^ 2 := 
by 
  sorry

end sum_is_square_l512_512100


namespace solve_equation_l512_512238

theorem solve_equation (x : ℝ) (h : x ≠ 1 ∧ x ≠ 1) : (3 / (x - 1) = 5 + 3 * x / (1 - x)) ↔ x = 4 := by
  sorry

end solve_equation_l512_512238


namespace pond_tadpoles_fish_difference_l512_512696

theorem pond_tadpoles_fish_difference :
  ∀ (initial_fish : ℕ) (fish_caught : ℕ) (tadpoles_ratio : ℕ) (tadpoles_develop: ℕ),
    initial_fish = 50 →
    fish_caught = 7 →
    tadpoles_ratio = 3 →
    tadpoles_develop = 2 →

    let remaining_fish := initial_fish - fish_caught in
    let initial_tadpoles := initial_fish * tadpoles_ratio in
    let remaining_tadpoles := initial_tadpoles / tadpoles_develop in

    remaining_tadpoles - remaining_fish = 32 :=
by
  intros initial_fish fish_caught tadpoles_ratio tadpoles_develop
    initial_fish_eq fish_caught_eq tadpoles_ratio_eq tadpoles_develop_eq
  simp [initial_fish_eq, fish_caught_eq, tadpoles_ratio_eq, tadpoles_develop_eq]
  simp
  sorry

end pond_tadpoles_fish_difference_l512_512696


namespace provisions_last_days_l512_512566

theorem provisions_last_days (D : ℕ) 
  (soldiers_initial : ℕ := 1200)
  (consumption_initial : ℕ := 3)
  (soldiers_added : ℕ := 528)
  (consumption_new : ℕ := 2.5)
  (days_new_soldiers : ℕ := 25) :
  (soldiers_initial * consumption_initial * D = (soldiers_initial + soldiers_added) * consumption_new * days_new_soldiers) →
  D = 300 :=
by
  intros h
  sorry

end provisions_last_days_l512_512566


namespace angle_MHC_l512_512559

-- Define the given conditions
def angle_A := 100 -- degrees
def angle_B := 50 -- degrees
def angle_C := 30 -- degrees

noncomputable def is_altitude (AH : Line) (A B C H : Point) : Prop := 
  ⟦ some definition that AH is an altitude from A to BC ⟧

noncomputable def is_median (BM : Line) (B M A C : Point) : Prop := 
  ⟦ some definition that BM is a median from B to AC ⟧

-- The main statement to be proven
theorem angle_MHC (A B C H M : Point) (AH BM : Line) 
  (h1 : angle A = 100) (h2 : angle B = 50) (h3 : angle C = 30) 
  (h4 : is_altitude AH A B C H) (h5 : is_median BM B M A C) : 
  angle MHC = 30 :=
sorry

end angle_MHC_l512_512559


namespace MichelangeloCeilingPainting_l512_512190

theorem MichelangeloCeilingPainting (total_ceiling week1_ceiling next_week_fraction : ℕ) 
  (a1 : total_ceiling = 28) 
  (a2 : week1_ceiling = 12) 
  (a3 : total_ceiling - (week1_ceiling + next_week_fraction * week1_ceiling) = 13) : 
  next_week_fraction = 1 / 4 := 
by 
  sorry

end MichelangeloCeilingPainting_l512_512190


namespace sample_standard_deviation_same_sample_range_same_l512_512858

open Nat

variables {n : ℕ} (x : Fin n → ℝ) (c : ℝ)
hypothesis (h_c : c ≠ 0)

/-- Assertion C: The sample standard deviations of the two sets of sample data are the same. -/
theorem sample_standard_deviation_same :
  (1 / n * ∑ i, (x i - (1 / n * ∑ i, x i))^2).sqrt =
  (1 / n * ∑ i, (x i + c - (1 / n * ∑ i, x i + c))^2).sqrt := sorry

/-- Assertion D: The sample ranges of the two sets of sample data are the same. -/
theorem sample_range_same :
  (Finset.sup Finset.univ x - Finset.inf Finset.univ x) =
  (Finset.sup Finset.univ (fun i => x i + c) - Finset.inf Finset.univ (fun i => x i + c)) := sorry

end sample_standard_deviation_same_sample_range_same_l512_512858


namespace cost_of_song_book_l512_512147

-- Definitions of the constants:
def cost_of_flute : ℝ := 142.46
def cost_of_music_stand : ℝ := 8.89
def total_spent : ℝ := 158.35

-- Definition of the combined cost of the flute and music stand:
def combined_cost := cost_of_flute + cost_of_music_stand

-- The final theorem to prove that the cost of the song book is $7.00:
theorem cost_of_song_book : total_spent - combined_cost = 7.00 := by
  sorry

end cost_of_song_book_l512_512147


namespace line_PR_passes_through_fixed_point_l512_512775

def cubic_parabola (a₁ a₂ a₃ : ℝ) (x : ℝ) : ℝ := x^3 + a₁ * x^2 + a₂ * x + a₃

def collinear (x₁ y₁ x₂ y₂ x₃ y₃ : ℝ) : Prop :=
  (y₂ - y₁) * (x₃ - x₁) = (y₃ - y₁) * (x₂ - x₁)

theorem line_PR_passes_through_fixed_point (a₁ a₂ a₃ x₁ x₂ x₃ : ℝ)
  (h₁ : x₁ + x₂ + x₃ = -a₁) :
  let y₁ := cubic_parabola a₁ a₂ a₃ x₁
  let y₂ := cubic_parabola a₁ a₂ a₃ x₂
  let y₃ := cubic_parabola a₁ a₂ a₃ x₃
  in collinear x₁ y₁ x₂ y₂ x₃ y₃ :=
by sorry

end line_PR_passes_through_fixed_point_l512_512775


namespace proof_S5_l512_512838

noncomputable def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q a1, ∀ n, a (n + 1) = a1 * q ^ (n + 1)

theorem proof_S5 (a : ℕ → ℝ) (S : ℕ → ℝ) (q a1 : ℝ) : 
  (geometric_sequence a) → 
  (a 2 * a 5 = 2 * a 3) → 
  ((a 4 + 2 * a 7) / 2 = 5 / 4) → 
  (S 5 = a1 * (1 - (1 / 2) ^ 5) / (1 - 1 / 2)) → 
  S 5 = 31 := 
by sorry

end proof_S5_l512_512838


namespace tyler_people_count_l512_512682

axiom (four_person_eggs : Nat) : four_person_eggs = 2
axiom (tyler_has_eggs : Nat) : tyler_has_eggs = 3
axiom (tyler_needs_more_eggs : Nat) : tyler_needs_more_eggs = 1
axiom (total_eggs_needed : Nat) : total_eggs_needed = tyler_has_eggs + tyler_needs_more_eggs

theorem tyler_people_count (four_person_eggs : Nat) (tyler_has_eggs : Nat) (tyler_needs_more_eggs : Nat) (total_eggs_needed : Nat) :
  total_eggs_needed = 4 →
  tyler_has_eggs + tyler_needs_more_eggs = 4 →
  total_eggs_needed = 2 * four_person_eggs →
  (4 / four_person_eggs) * 4 = 8 :=
by
  intros h1 h2 h3
  rw [h2] at h3
  rw [←h3] at h1
  exact h1

end tyler_people_count_l512_512682


namespace empty_subset_A_l512_512517

open Set

theorem empty_subset_A : let A := {x : ℝ | x^2 - x = 0} in ∅ ⊆ A :=
by
  let A := {x : ℝ | x^2 - x = 0}
  exact empty_subset A

end empty_subset_A_l512_512517


namespace increasing_sequence_range_of_a_l512_512186

theorem increasing_sequence_range_of_a 
    (f : ℕ+ → ℝ)
    (h₁ : ∀ x : ℕ+, f x = if (x : ℕ) ≤ 7 then (3 - a) * x - 3 else a^(x - 6)) 
    (h₂ : ∀ n m : ℕ+, n < m → f n < f m) 
    : 2 < a ∧ a < 3 :=
sorry

end increasing_sequence_range_of_a_l512_512186


namespace max_xy_l512_512065

theorem max_xy (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x + y = 18) : xy ≤ 81 :=
by sorry

end max_xy_l512_512065


namespace find_point_B_find_range_of_k_l512_512056

section
  variables (A : ℝ × ℝ) (a : ℝ × ℝ)
  def is_same_direction (v1 v2 : ℝ × ℝ) := ∃ (k : ℝ), k ≠ 0 ∧ v1 = (k * v2.1, k * v2.2)
  def length (v : ℝ × ℝ) := real.sqrt (v.1 ^ 2 + v.2 ^ 2)

  -- Question 1
  theorem find_point_B (B : ℝ × ℝ) (h1 : A = (1, -2)) (h2 : a = (2, 3))
    (h3 : is_same_direction (B.1 - A.1, B.2 - A.2) a)
    (h4 : length (B.1 - A.1, B.2 - A.2) = 2 * real.sqrt 13) :
    B = (5, 4) :=
  sorry

  -- Question 2
  theorem find_range_of_k (b : ℝ × ℝ) (k : ℝ) (h1 : a = (2, 3)) (h2 : b = (-3, k))
    (h3 : 0 > ((a.1 * b.1) + (a.2 * b.2))) : 
    k < 2 ∧ ¬ (k = - 9 / 2) :=
  sorry
end

end find_point_B_find_range_of_k_l512_512056


namespace number_of_valid_subsets_l512_512173

def is_isolated (A : set ℕ) (x : ℕ) : Prop :=
  x ∈ A ∧ x - 1 ∉ A ∧ x + 1 ∉ A

def no_isolated_elements (A : set ℕ) : Prop :=
  ∀ x ∈ A, ¬ is_isolated A x

def valid_subsets (S : set ℕ) : set (set ℕ) :=
  { A | A ⊆ S ∧ no_isolated_elements A ∧ A.card = 4 }

def S : set ℕ := {0, 1, 2, 3, 4, 5}

noncomputable def count_valid_subsets : ℕ :=
  (valid_subsets S).to_finset.card

theorem number_of_valid_subsets : count_valid_subsets = 6 := by
  sorry

end number_of_valid_subsets_l512_512173


namespace number_of_k_values_l512_512292

-- Definitions and conditions
def k_condition (k a b : ℕ) : Prop :=
  a > 0 ∧ b > 0 ∧ k > 0 ∧ (k * (a + b) = 2015 * Nat.lcm a b)

-- Main theorem statement
theorem number_of_k_values : 
  (∃ (k : ℕ), ∃ (a b : ℕ), k_condition k a b) → ∃ t, t = 1007 := 
begin
  sorry
end

end number_of_k_values_l512_512292


namespace extreme_values_f_range_of_a_l512_512512

noncomputable def f (x a : ℝ) : ℝ := x^3 - x^2 - x - a
noncomputable def df (x : ℝ) : ℝ := 3 * x^2 - 2 * x - 1

theorem extreme_values_f (a : ℝ) :
  ∃ (x₁ x₂ : ℝ), df x₁ = 0 ∧ df x₂ = 0 ∧ f x₁ a = (5 / 27) - a ∧ f x₂ a = -1 - a :=
sorry

theorem range_of_a (a : ℝ) :
  (∃ (a : ℝ), f (-1/3) a < 0 ∧ f 1 a > 0) ↔ (a < -1 ∨ a > 5 / 27) :=
sorry

end extreme_values_f_range_of_a_l512_512512


namespace CylindricalVesselFillsInTime_l512_512208

theorem CylindricalVesselFillsInTime
  (v : ℝ := 2) 
  (m_drop : ℝ := 5 * 10^(-6)) 
  (n : ℝ := 400) 
  (rho_B : ℝ := 1000) 
  (h : ℝ := 0.2) 
  (A : ℝ := 10^(-3)) :
  (time_to_fill : ℝ) :=
  let m_water : ℝ := n * m_drop
  let m_water_per_sec_per_m_sq : ℝ := m_water * v
  let m_water_per_sec_per_10_cm_sq : ℝ := m_water_per_sec_per_m_sq * A
  let V : ℝ := A * h
  let m_required : ℝ := rho_B * V
  let t : ℝ := m_required / m_water_per_sec_per_10_cm_sq in
  round (t / 3600) = 14 := by
  sorry

end CylindricalVesselFillsInTime_l512_512208


namespace range_of_a_l512_512073

theorem range_of_a (a : ℝ) : 
  (∃ x y : ℝ, x^2 + y^2 + 2 * x - 4 * y + a = 0) → a < 5 := 
by sorry

end range_of_a_l512_512073


namespace find_c_l512_512721

theorem find_c (c : ℝ) :
  (∀ x : ℝ, -2 * x^2 + c * x - 8 < 0 ↔ x ∈ Set.Iio 2 ∪ Set.Ioi 6) → c = 16 :=
by
  intros h
  sorry

end find_c_l512_512721


namespace melissa_points_per_game_l512_512189

theorem melissa_points_per_game (total_points : ℕ) (games_played : ℕ) (h1 : total_points = 1200) (h2 : games_played = 10) : (total_points / games_played) = 120 := 
by
  -- Here we would insert the proof steps, but we use sorry to represent the omission
  sorry

end melissa_points_per_game_l512_512189


namespace player_can_zero_entries_l512_512001

theorem player_can_zero_entries
  (n : ℕ)
  (A : array (fin n) (array (fin n) ℝ))
  (h_nonneg : ∀ i j, 0 ≤ A[i][j])
  (h_row_sum_eq : ∀ i, (finset.univ.sum (λ j, A[i][j])) = (finset.univ.sum (λ k, A[0][k])))
  (h_col_sum_eq : ∀ j, (finset.univ.sum (λ i, A[i][j])) = (finset.univ.sum (λ k, A[k][0]))):
  ∃ B : array (fin n) (array (fin n) ℝ), (∀ i j, B[i][j] = 0) := 
sorry

end player_can_zero_entries_l512_512001


namespace product_of_cosines_of_two_distinct_values_in_S_l512_512017

noncomputable def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + d

theorem product_of_cosines_of_two_distinct_values_in_S
  (a : ℕ → ℝ)
  (d : ℝ)
  (h_sequence: arithmetic_sequence a (2 * Real.pi / 3))
  (hS : ∃ (b c : ℝ), b ≠ c ∧ (∀ n : ℕ, ∃b', b' = cos (a n) ∧ (b' = b ∨ b' = c))) :
  (∃ b c : ℝ, b ≠ c ∧ (∀ n : ℕ, cos (a n) = b ∨ cos (a n) = c) ∧ b * c = -1 / 2) := 
sorry

end product_of_cosines_of_two_distinct_values_in_S_l512_512017


namespace ratio_of_areas_l512_512764

theorem ratio_of_areas (s : ℝ) : 
  let original_area := s^2 in
  let new_area := (3 * s)^2 in
  original_area / new_area = 1 / 9 :=
by
  sorry

end ratio_of_areas_l512_512764


namespace base7_sum_of_digits_of_product_l512_512779

-- Define the given numbers in base 7
def num1_base7 := "35_7"
def num2_base7 := "12_7"
def num3_base7 := "16_7"

-- Function to convert base 7 to base 10
noncomputable def base7_to_base10 (s : string) : ℕ := sorry

-- Function to convert base 10 to base 7
noncomputable def base10_to_base7 (n : ℕ) : string := sorry

-- Function to compute the sum of digits in a base 7 number
noncomputable def sum_of_digits_base7 (s : string) : ℕ := sorry

-- Translate the conditions into base 10
noncomputable def num1_base10 := base7_to_base10 num1_base7
noncomputable def num2_base10 := base7_to_base10 num2_base7
noncomputable def num3_base10 := base7_to_base10 num3_base7

-- Sum the second and third numbers in base 10, then convert back to base 7
noncomputable def sum_base10 := num2_base10 + num3_base10
noncomputable def sum_base7 := base10_to_base7 sum_base10

-- Convert the first number to base 10, multiply, and convert back to base 7
noncomputable def product_base10 := num1_base10 * base7_to_base10 sum_base7
noncomputable def product_base7 := base10_to_base7 product_base10

-- Statement of the theorem
theorem base7_sum_of_digits_of_product : 
  sum_of_digits_base7 product_base7 = 7 := 
sorry

end base7_sum_of_digits_of_product_l512_512779


namespace mark_total_eggs_in_a_week_l512_512618

-- Define the given conditions
def first_store_eggs_per_day := 5 * 12 -- 5 dozen eggs per day
def second_store_eggs_per_day := 30
def third_store_eggs_per_odd_day := 25 * 12 -- 25 dozen eggs per odd day
def third_store_eggs_per_even_day := 15 * 12 -- 15 dozen eggs per even day
def days_per_week := 7
def odd_days_per_week := 4
def even_days_per_week := 3

-- Lean theorem statement to prove the total eggs supplied in a week
theorem mark_total_eggs_in_a_week : 
    first_store_eggs_per_day * days_per_week + 
    second_store_eggs_per_day * days_per_week + 
    third_store_eggs_per_odd_day * odd_days_per_week + 
    third_store_eggs_per_even_day * even_days_per_week =
    2370 := 
    sorry  -- Placeholder for the actual proof

end mark_total_eggs_in_a_week_l512_512618


namespace log_expression_eval_l512_512781

theorem log_expression_eval :
  2 * log 5 10 + log 5 0.25 = 2 :=
by
  sorry

end log_expression_eval_l512_512781


namespace solve_equation_l512_512242

theorem solve_equation (x : ℝ) (h : x ≠ 1 ∧ x ≠ 1) : (3 / (x - 1) = 5 + 3 * x / (1 - x)) ↔ x = 4 := by
  sorry

end solve_equation_l512_512242


namespace mike_office_visits_per_day_l512_512992

-- Define the constants from the conditions
def pull_ups_per_visit : ℕ := 2
def total_pull_ups_per_week : ℕ := 70
def days_per_week : ℕ := 7

-- Calculate total office visits per week
def office_visits_per_week : ℕ := total_pull_ups_per_week / pull_ups_per_visit

-- Lean statement that states Mike goes into his office 5 times a day
theorem mike_office_visits_per_day : office_visits_per_week / days_per_week = 5 := by
  sorry

end mike_office_visits_per_day_l512_512992


namespace relationship_between_y_coordinates_l512_512478

theorem relationship_between_y_coordinates (b y1 y2 y3 : ℝ)
  (h1 : y1 = 3 * (-3) - b)
  (h2 : y2 = 3 * 1 - b)
  (h3 : y3 = 3 * (-1) - b) :
  y1 < y3 ∧ y3 < y2 := 
sorry

end relationship_between_y_coordinates_l512_512478


namespace fruit_basket_count_l512_512097

-- Define the number of apples and oranges
def apples := 7
def oranges := 12

-- Condition: A fruit basket must contain at least two pieces of fruit
def min_pieces_of_fruit := 2

-- Problem: Prove that there are 101 different fruit baskets containing at least two pieces of fruit
theorem fruit_basket_count (n_apples n_oranges n_min_pieces : Nat) (h_apples : n_apples = apples) (h_oranges : n_oranges = oranges) (h_min_pieces : n_min_pieces = min_pieces_of_fruit) :
  (n_apples = 7) ∧ (n_oranges = 12) ∧ (n_min_pieces = 2) → (104 - 3 = 101) :=
by
  sorry

end fruit_basket_count_l512_512097


namespace polar_bear_diameter_paradox_l512_512995

section

variable (R_I R_P : ℝ)
variable (floe_mass bear_mass : ℝ)
variable (correct_measurements : R_I = 4.25 ∧ R_P = 4.5 ∧ 0.5 * R_I ≈ (floe_mass / bear_mass) * (R_P - R_I))

theorem polar_bear_diameter_paradox :
  (R_P = 4.5) → (R_I = 4.25) → (correct_measurements ∧ floe_mass > bear_mass) →
  floe_mass / bear_mass ≈ (R_P - R_I) / R_I :=
by
  sorry

end

end polar_bear_diameter_paradox_l512_512995


namespace solve_fractional_equation_l512_512248

theorem solve_fractional_equation (x : ℝ) (h1 : x ≠ 1) (h2 : x ≠ -1) : 
    (3 / (x - 1) = 5 + 3 * x / (1 - x)) → x = 4 := 
by
  sorry

end solve_fractional_equation_l512_512248


namespace DivisibleByThreeCount_l512_512417

-- Define the set of all 100-digit numbers using only digits 1 and 2
def valid_numbers (n : ℕ) : set ℕ :=
  {x : ℕ | (nat.digits 10 x).length = n ∧ ∀ d ∈ nat.digits 10 x, d = 1 ∨ d = 2}

-- Define the count of numbers divisible by 3
def count_divisible_by_3 (s : set ℕ) : ℕ :=
  s.count (λ x, x % 3 = 0)

-- State that A_100 is the count of 100-digit numbers divisible by 3
def A_100 : ℕ := count_divisible_by_3 (valid_numbers 100)

-- Theorem stating the required proof
theorem DivisibleByThreeCount (A_100 : ℕ) : A_100 = (4^50 + 2) / 3 :=
by sorry

end DivisibleByThreeCount_l512_512417


namespace factor_polynomial_l512_512452

theorem factor_polynomial : ∀ y : ℝ, 3 * y^2 - 27 = 3 * (y + 3) * (y - 3) :=
by
  intros y
  sorry

end factor_polynomial_l512_512452


namespace linear_regression_passes_through_centroid_l512_512464

def set_of_points : List (ℝ × ℝ) := [(0, 1), (1, 3), (2, 5), (3, 7)]

def centroid (points : List (ℝ × ℝ)) : ℝ × ℝ :=
  let (sum_x, sum_y) := points.foldr (λ (p : ℝ × ℝ) (acc : ℝ × ℝ) => (acc.1 + p.1, acc.2 + p.2)) (0, 0)
  let n := points.length
  (sum_x / n, sum_y / n)

theorem linear_regression_passes_through_centroid :
  let c := centroid set_of_points in
  c = (1.5, 4) :=
by
  let points := set_of_points
  have : centroid points = (1.5, 4) := sorry
  exact this

end linear_regression_passes_through_centroid_l512_512464


namespace fraction_division_l512_512711

theorem fraction_division (a b c d : ℚ) (h : b ≠ 0 ∧ d ≠ 0) : (a / b) / c = a / (b * c) := sorry

example : (3 / 7) / 4 = 3 / 28 := by
  apply fraction_division
  exact ⟨by norm_num, by norm_num⟩

end fraction_division_l512_512711


namespace sample_standard_deviation_same_sample_ranges_same_l512_512866

variables {n : ℕ} (x y : Fin n → ℝ) (c : ℝ)
  (h_y : ∀ i, y i = x i + c)
  (h_c_ne_zero : c ≠ 0)

-- Statement for standard deviations being the same
theorem sample_standard_deviation_same :
  let mean (s : Fin n → ℝ) := (∑ i, s i) / n
  in let stddev (s : Fin n → ℝ) := sqrt ((∑ i, (s i - mean s) ^ 2) / n)
  in stddev x = stddev y := 
sorry

-- Statement for ranges being the same
theorem sample_ranges_same :
  let range (s : Fin n → ℝ) := (Finset.univ.sup s) - (Finset.univ.inf s)
  in range x = range y :=
sorry

end sample_standard_deviation_same_sample_ranges_same_l512_512866


namespace minimum_value_of_function_l512_512834

theorem minimum_value_of_function (x : ℝ) (hx : x > -1) : 
  ∃ (y : ℝ), y = x + 1 / (x + 1) ∧ y = 1 :=
begin
  -- Proof can be filled here
  sorry
end

end minimum_value_of_function_l512_512834


namespace lines_parallel_l512_512911

-- Define the lines L1 and L2
def L1 (a : ℝ) : (ℝ × ℝ) → ℝ :=
  λ (p : ℝ × ℝ), p.1 + a * p.2 + 6

def L2 (a : ℝ) : (ℝ × ℝ) → ℝ :=
  λ (p : ℝ × ℝ), (a - 2) * p.1 + 3 * p.2 + 2 * a

-- Parallelism condition between L1 and L2
def parallel_condition (a : ℝ) : Prop :=
  (1 / (a - 2) = a / 3)

-- Ratio condition for parallelism but not coincident
def ratio_condition (a : ℝ) : Prop :=
  (1 / (a - 2)) ≠ 6 / (2 * a)

-- The main theorem to prove
theorem lines_parallel (a : ℝ) (h1 : parallel_condition a) (h2 : ratio_condition a) : a = -1 :=
by
  sorry

end lines_parallel_l512_512911


namespace trajectory_midpoint_l512_512416

theorem trajectory_midpoint (P M D : ℝ × ℝ) (hP : P.1 ^ 2 + P.2 ^ 2 = 16) (hD : D = (P.1, 0)) (hM : M = ((P.1 + D.1)/2, (P.2 + D.2)/2)) :
  (M.1 ^ 2) / 4 + (M.2 ^ 2) / 16 = 1 :=
by
  sorry

end trajectory_midpoint_l512_512416


namespace circle_radius_five_eq_neg_eight_l512_512826

theorem circle_radius_five_eq_neg_eight (c : ℝ) :
  (∃ x y : ℝ, x^2 + 8*x + y^2 + 2*y + c = 0 ∧ (x + 4)^2 + (y + 1)^2 = 25) → c = -8 :=
by
  sorry

end circle_radius_five_eq_neg_eight_l512_512826


namespace sample_std_dev_range_same_l512_512852

noncomputable def sample_std_dev (data : List ℝ) : ℝ := sorry
noncomputable def sample_range (data : List ℝ) : ℝ := sorry

theorem sample_std_dev_range_same (n : ℕ) (c : ℝ) (Hc : c ≠ 0) (x : Fin n → ℝ) :
  sample_std_dev (List.ofFn (λ i => x i)) = sample_std_dev (List.ofFn (λ i => x i + c)) ∧
  sample_range (List.ofFn (λ i => x i)) = sample_range (List.ofFn (λ i => x i + c)) :=
by
  sorry

end sample_std_dev_range_same_l512_512852


namespace simplify_product_l512_512222

theorem simplify_product : (∏ n in Finset.range 200, (5 * n + 5) / (5 * n)) = (404 / 201) :=
by
  -- Definitions used in the problem
  sorry

end simplify_product_l512_512222


namespace polar_bear_paradox_l512_512998

theorem polar_bear_paradox
  (d_earth : ℝ) (d_floe : ℝ)
  (h1 : d_earth = 8.5) (h2 : d_floe = 9)
  (h3 : ∀ (measurements_correct : Prop), measurements_correct) :
  ∃ (mass_ratio : ℝ), mass_ratio > 1 ∧ 
                       abs (d_floe / 2 - d_earth / 2) = 0.25 :=
by
  use 10 -- Assume an example mass ratio of 10.
  split
  . exact zero_lt_ten
  . calc
    abs (4.5 - 4.25) = abs (0.25) : by norm_num
                ... = 0.25 : abs_of_nonneg (by norm_num)

end polar_bear_paradox_l512_512998


namespace cost_per_person_correct_l512_512151

noncomputable def cost_paid_by_each_person : ℝ := 
let chocolate_cupcakes := 3.5 * 1.50 in
let almond_pastries := 2.25 * 2.75 in
let raspberry_muffins := 5.0 * 2.10 in
let subtotal := chocolate_cupcakes + almond_pastries + raspberry_muffins in
let sales_tax := subtotal * 0.07 in
let total_cost_before_coupon := subtotal + sales_tax in
let discount := total_cost_before_coupon * 0.20 in
let total_cost_after_coupon := total_cost_before_coupon - discount in
total_cost_after_coupon / 2

theorem cost_per_person_correct : cost_paid_by_each_person = 9.38925 := 
begin
  -- Detailed proof steps omitted
  sorry
end

end cost_per_person_correct_l512_512151


namespace angle_between_PQ_and_RS_half_angle_XOZ_l512_512170

noncomputable theory

open EuclideanGeometry

variables {A X Y Z B P Q R S O : Point}

-- Conditions
variable (h1 : convex_pentagon_inscribed_in_semicircle A X Y Z B)
variable (h2 : perpendicular_from_point_onto_line P Y A X)
variable (h3 : perpendicular_from_point_onto_line Q Y B X)
variable (h4 : perpendicular_from_point_onto_line R Y A Z)
variable (h5 : perpendicular_from_point_onto_line S Y B Z)
variable (hO : midpoint O A B)
-- Question to be proved
theorem angle_between_PQ_and_RS_half_angle_XOZ :
  acute_angle (line_through P Q) (line_through R S) = (angle X O Z) / 2 :=
sorry

end angle_between_PQ_and_RS_half_angle_XOZ_l512_512170


namespace weight_of_fish_in_barrel_l512_512327

/-- 
Given a barrel with an initial weight of 54 kg when full of fish,
and a weight of 29 kg after removing half of the fish,
prove that the initial weight of the fish in the barrel was 50 kg.
-/
theorem weight_of_fish_in_barrel (B F : ℝ)
  (h1: B + F = 54)
  (h2: B + F / 2 = 29) : F = 50 := 
sorry

end weight_of_fish_in_barrel_l512_512327


namespace relationship_of_y_values_l512_512476

theorem relationship_of_y_values (b y1 y2 y3 : ℝ) (h1 : y1 = 3 * (-3) - b)
                                (h2 : y2 = 3 * 1 - b)
                                (h3 : y3 = 3 * (-1) - b) :
  y1 < y3 ∧ y3 < y2 :=
by
  sorry

end relationship_of_y_values_l512_512476


namespace pyramid_max_volume_height_l512_512494

-- Define the conditions and the theorem
theorem pyramid_max_volume_height
  (a h V : ℝ)
  (SA : ℝ := 2 * Real.sqrt 3)
  (h_eq : h = Real.sqrt (SA^2 - (Real.sqrt 2 * a / 2)^2))
  (V_eq : V = (1 / 3) * a^2 * h)
  (derivative_at_max : ∀ a, (48 * a^3 - 3 * a^5 = 0) → (a = 0 ∨ a = 4))
  (max_a_value : a = 4):
  h = 2 :=
by
  sorry

end pyramid_max_volume_height_l512_512494


namespace selection_combinations_l512_512133

noncomputable def num_possible_selections (boys : ℕ) (girls : ℕ) (include_kimi : Bool) (stone_goes : Bool) : ℕ :=
  if boys = 3 ∧ girls = 2 then
    if include_kimi ∧ ¬ stone_goes then
      3 -- Combinations when Kimi goes and Stone doesn't
    else if ¬ include_kimi ∧ stone_goes then
      3 -- Combinations when Kimi doesn't go and Stone goes
    else
      6 -- Remaining combinations without specific constraints on Kimi and Stone
  else
    0

theorem selection_combinations : num_possible_selections 3 2 true false + num_possible_selections 3 2 false true + num_possible_selections 3 2 false false = 12 :=
by
  sorry

end selection_combinations_l512_512133


namespace students_taking_statistics_l512_512127

theorem students_taking_statistics (H S total : ℕ)
  (H_hist : H = 36)
  (H_union_S : total = 57)
  (H_hist_not_stat : H - S = 25)
  (total_students : total = H ∪ S . H ∩ S : ℕ := {total_students ~total_students}) 
  : 32 = S := sorry

end students_taking_statistics_l512_512127


namespace average_stamps_per_day_l512_512617

theorem average_stamps_per_day :
  let a1 := 8
  let d := 8
  let n := 6
  let stamps_collected : Fin n → ℕ := λ i => a1 + i * d
  -- sum the stamps collected over six days
  let S := List.sum (List.ofFn stamps_collected)
  -- calculate average
  let average := S / n
  average = 28 :=
by sorry

end average_stamps_per_day_l512_512617


namespace triangle_area_correct_l512_512768

-- Define the coordinates of the triangle vertices
def A : (ℝ × ℝ) := (2, 3)
def B : (ℝ × ℝ) := (7, 3)
def C : (ℝ × ℝ) := (2, -4)

-- Define the lengths of the sides using Euclidean distance
def length_AB : ℝ := real.dist A B
def length_AC : ℝ := real.dist A C

-- Definition of the area of a right triangle
def area_right_triangle (base height : ℝ) : ℝ := 0.5 * base * height

-- Calculate the expected area given the vertices coordinates
def triangle_area : ℝ := area_right_triangle length_AB length_AC

theorem triangle_area_correct : triangle_area = 17.5 :=
by
  -- Prove that 'length_AB' and 'length_AC' are 5 and 7 respectively
  have hab : length_AB = 5 := real.dist_eq (2, 3) (7, 3) -- horizontal distance
  have hac : length_AC = 7 := real.dist_eq (2, 3) (2, -4) -- vertical distance
  -- Use these to show the area is indeed 17.5
  show area_right_triangle hab hac = 17.5
  sorry

end triangle_area_correct_l512_512768


namespace product_of_cosine_values_l512_512025

def arithmetic_seq (a₁ d : ℝ) (n : ℕ) : ℝ := a₁ + d * (n - 1)

theorem product_of_cosine_values (a₁ : ℝ) (h : ∃ (a b : ℝ), S = {a, b} ∧ S = {cos (arithmetic_seq a₁ (2 * π / 3) n) | n ∈ ℕ.succ}) :
  ∃ (a b : ℝ), a * b = -1 / 2 :=
begin
  obtain ⟨a, b, hS₁, hS₂⟩ := h,
  -- the proof will go here
  sorry
end

end product_of_cosine_values_l512_512025


namespace α_plus_2β_eq_pi_div_2_l512_512882

open Real

noncomputable def α : ℝ := sorry
noncomputable def β : ℝ := sorry

axiom h1 : 0 < α ∧ α < π / 2
axiom h2 : 0 < β ∧ β < π / 2
axiom h3 : 3 * sin α ^ 2 + 2 * sin β ^ 2 = 1
axiom h4 : 3 * sin (2 * α) - 2 * sin (2 * β) = 0

theorem α_plus_2β_eq_pi_div_2 : α + 2 * β = π / 2 :=
by
  sorry

end α_plus_2β_eq_pi_div_2_l512_512882


namespace carol_weight_l512_512686

variable (a c : ℝ)

theorem carol_weight (h1 : a + c = 240) (h2 : c - a = (2 / 3) * c) : c = 180 :=
by
  sorry

end carol_weight_l512_512686


namespace sample_standard_deviation_same_sample_range_same_l512_512856

open Nat

variables {n : ℕ} (x : Fin n → ℝ) (c : ℝ)
hypothesis (h_c : c ≠ 0)

/-- Assertion C: The sample standard deviations of the two sets of sample data are the same. -/
theorem sample_standard_deviation_same :
  (1 / n * ∑ i, (x i - (1 / n * ∑ i, x i))^2).sqrt =
  (1 / n * ∑ i, (x i + c - (1 / n * ∑ i, x i + c))^2).sqrt := sorry

/-- Assertion D: The sample ranges of the two sets of sample data are the same. -/
theorem sample_range_same :
  (Finset.sup Finset.univ x - Finset.inf Finset.univ x) =
  (Finset.sup Finset.univ (fun i => x i + c) - Finset.inf Finset.univ (fun i => x i + c)) := sorry

end sample_standard_deviation_same_sample_range_same_l512_512856


namespace oliver_workout_ratio_l512_512194

theorem oliver_workout_ratio :
  let Monday := 4 in
  let Tuesday := Monday - 2 in
  let Thursday := 2 * Tuesday in
  let TotalHours := 18 in
  let Wednesday := TotalHours - (Monday + Tuesday + Thursday) in
  (Wednesday / Monday) = 2 :=
by {
  let Monday := 4,
  let Tuesday := 4 - 2,
  let Thursday := 2 * (4 - 2),
  let TotalHours := 18,
  let Wednesday := TotalHours - (Monday + Tuesday + Thursday),
  show (Wednesday / Monday) = 2,
  sorry
}

end oliver_workout_ratio_l512_512194


namespace jason_flames_per_minute_l512_512324

theorem jason_flames_per_minute :
  (∀ (t : ℕ), t % 15 = 0 -> (5 * (t / 15) = 20)) :=
sorry

end jason_flames_per_minute_l512_512324


namespace q_investment_l512_512738

theorem q_investment (p_investment : ℝ) (profit_ratio_p : ℝ) (profit_ratio_q : ℝ) (q_investment : ℝ) 
  (h1 : p_investment = 40000) 
  (h2 : profit_ratio_p / profit_ratio_q = 2 / 3) 
  : q_investment = 60000 := 
sorry

end q_investment_l512_512738


namespace intercept_condition_slope_condition_l512_512614

theorem intercept_condition (m : ℚ) (h : m ≠ -1) : 
  (m^2 - 2 * m - 3) * -3 + (2 * m^2 + m - 1) * 0 + (-2 * m + 6) = 0 → 
  m = -5 / 3 := 
  sorry

theorem slope_condition (m : ℚ) (h : m ≠ -1) : 
  (m^2 + 2 * m - 4) = 0 → 
  m = 4 / 3 := 
  sorry

end intercept_condition_slope_condition_l512_512614


namespace find_a_b_l512_512901

open Function

noncomputable def f (a b x : ℝ) : ℝ := a ^ x + b

theorem find_a_b (a b : ℝ) (h1 : a > 0) (h2 : a ≠ 1)
    (h3 : ∀ x, -1 ≤ x ∧ x ≤ 0 → f a b x ∈ set.Icc (-1) 0) :
    a + b = -3 / 2 :=
by
  sorry

end find_a_b_l512_512901


namespace point_in_third_quadrant_l512_512114

theorem point_in_third_quadrant (m n : ℝ) (h1 : m < 0) (h2 : n > 0) : -m^2 < 0 ∧ -n < 0 :=
by
  sorry

end point_in_third_quadrant_l512_512114


namespace necessary_and_sufficient_condition_l512_512051

-- Define the arithmetic sequence
def arithmetic_seq (a_1 : ℤ) (d : ℤ) (n : ℤ) : ℤ :=
  a_1 + (n - 1) * d

-- Define the sum of the first k terms of an arithmetic sequence
def sum_arithmetic_seq (a_1 : ℤ) (d : ℤ) (k : ℤ) : ℤ :=
  (k * (2 * a_1 + (k - 1) * d)) / 2

-- Prove that d > 0 is a necessary and sufficient condition for S_3n - S_2n > S_2n - S_n
/-- Necessary and sufficient condition for the inequality S_{3n} - S_{2n} > S_{2n} - S_n -/
theorem necessary_and_sufficient_condition {a_1 d n : ℤ} :
  d > 0 ↔ sum_arithmetic_seq a_1 d (3 * n) - sum_arithmetic_seq a_1 d (2 * n) > 
             sum_arithmetic_seq a_1 d (2 * n) - sum_arithmetic_seq a_1 d n :=
by sorry

end necessary_and_sufficient_condition_l512_512051


namespace tangent_line_curve_l512_512923

theorem tangent_line_curve (a : ℝ) :
  (∃ x_0 : ℝ, x_0 - 3 = Real.exp (x_0 + a) ∧ Real.exp (x_0 + a)) = 1) → a = -4 := by
  sorry

end tangent_line_curve_l512_512923


namespace projection_scalar_mul_projection_add_l512_512309

-- Define the vector in terms of its components in unit vectors i and j
structure Vector2D (α : Type) [Add α] [Mul α] :=
  (x : α)
  (y : α)

variables {α : Type} [Add α] [Mul α] [AddCommMonoid α]

-- Define vector addition
instance : Add (Vector2D α) :=
  ⟨λ v w, Vector2D.mk (v.x + w.x) (v.y + w.y)⟩

-- Define scalar multiplication
instance : MulAction α (Vector2D α) :=
  ⟨λ k v, Vector2D.mk (k * v.x) (k * v.y), sorry⟩

theorem projection_scalar_mul (v : Vector2D α) (k : α) :
  k • v = Vector2D.mk (k * v.x) (k * v.y) := by sorry

theorem projection_add (v w : Vector2D α) :
  v + w = Vector2D.mk (v.x + w.x) (v.y + w.y) := by sorry

end projection_scalar_mul_projection_add_l512_512309


namespace triangle_perimeter_l512_512678

theorem triangle_perimeter (r A p : ℝ) (h_r : r = 4.5) (h_A : A = 78.75) :
  p = 35 :=
by
  have h_formula : A = r * (p / 2),
    sorry -- This is where you would use the given conditions to show this equivalence.
  rw [h_r, h_A] at h_formula,
  -- The rest of the proof steps
  sorry

end triangle_perimeter_l512_512678


namespace alyssa_cut_11_roses_l512_512695

theorem alyssa_cut_11_roses (initial_roses cut_roses final_roses : ℕ) 
  (h1 : initial_roses = 3) 
  (h2 : final_roses = 14) 
  (h3 : initial_roses + cut_roses = final_roses) : 
  cut_roses = 11 :=
by
  rw [h1, h2] at h3
  sorry

end alyssa_cut_11_roses_l512_512695


namespace coloring_of_1989_equal_circles_l512_512346

theorem coloring_of_1989_equal_circles :
  ∀ (G : SimpleGraph ℝ), G.is_planar → G.card_vertex = 1989 → ∃ (c : ℕ), c ≤ 4 ∧ G.chromatic_number = c :=
by
  sorry

end coloring_of_1989_equal_circles_l512_512346


namespace determine_t_range_l512_512612

noncomputable def f (x : ℝ) (t : ℝ) : ℝ := log (2^x + t) / log 2
def shrinking_function (f : ℝ → ℝ) := ∃ a b : ℝ, a < b ∧ ∀ x ∈ set.Icc a b, f x ∈ set.Icc (a / 2) (b / 2)
def t_range (t : ℝ) : Prop := 0 < t ∧ t < 1/4

theorem determine_t_range (t : ℝ) :
  shrinking_function (λ x, f x t) → t_range t :=
sorry

end determine_t_range_l512_512612


namespace jesses_room_total_area_l512_512949

-- Define the dimensions of the first rectangular part
def length1 : ℕ := 12
def width1 : ℕ := 8

-- Define the dimensions of the second rectangular part
def length2 : ℕ := 6
def width2 : ℕ := 4

-- Define the areas of both parts
def area1 : ℕ := length1 * width1
def area2 : ℕ := length2 * width2

-- Define the total area
def total_area : ℕ := area1 + area2

-- Statement of the theorem we want to prove
theorem jesses_room_total_area : total_area = 120 :=
by
  -- We would provide the proof here
  sorry

end jesses_room_total_area_l512_512949


namespace tetrahedron_distance_sum_eq_l512_512632

variable {Point : Type} [MetricSpace Point]

variables (A B C D T M : Point)

/-- M is the centroid of the tetrahedron DABC if the following holds. -/
def is_centroid (A B C D M : Point) : Prop :=
  let V := [A, B, C, D]
  V.sum / V.length = M

/-- The main theorem. -/
theorem tetrahedron_distance_sum_eq (hM : is_centroid A B C D M) :
  dist T A ^ 2 + dist T B ^ 2 + dist T C ^ 2 + dist T D ^ 2 = 
  dist M A ^ 2 + dist M B ^ 2 + dist M C ^ 2 + dist M D ^ 2 + 4 * dist T M ^ 2 :=
sorry

end tetrahedron_distance_sum_eq_l512_512632


namespace largest_number_in_set_l512_512970

theorem largest_number_in_set (a : ℤ) (h : a = 3) : 
  let s := {-5 * a, 3 * a ^ 2, 48 / a, 2 * a - 1, 2}
  in s.max = 3 * a ^ 2 := 
by
  have ha : a = 3 := h
  let s := {-5 * a, 3 * a ^ 2, 48 / a, 2 * a - 1, 2}
  sorry

end largest_number_in_set_l512_512970


namespace no_integer_solution_for_triples_l512_512913

theorem no_integer_solution_for_triples :
  ∀ (x y z : ℤ),
    x^2 - 2*x*y + 3*y^2 - z^2 = 17 →
    -x^2 + 4*y*z + z^2 = 28 →
    x^2 + 2*x*y + 5*z^2 = 42 →
    false :=
by
  intros x y z h1 h2 h3
  sorry

end no_integer_solution_for_triples_l512_512913


namespace number_of_parallel_pairs_l512_512093

-- Definitions related to octahedron geometry
structure RegularOctahedron :=
  (faces : Finset (Finset ℕ))
  (vertices : Finset ℕ)
  (edges : Finset (Finset ℕ))

-- Given conditions
def octahedron_faces : Finset (Finset ℕ) := 
  { {1, 2, 3}, {1, 3, 4}, {1, 4, 5}, {1, 5, 2}, {6, 2, 3}, {6, 3, 4}, {6, 4, 5}, {6, 5, 2} }

def octahedron_vertices : Finset ℕ := {1, 2, 3, 4, 5, 6} 

def octahedron_edges : Finset (Finset ℕ) := 
  { {1, 2}, {1, 3}, {1, 4}, {1, 5}, {2, 3}, {3, 4}, {4, 5}, {5, 2}, {6, 2}, {6, 3}, {6, 4}, {6, 5} }

def regular_octahedron : RegularOctahedron :=
  { faces := octahedron_faces,
    vertices := octahedron_vertices,
    edges := octahedron_edges }

-- Predicates for parallel edges
def edges_parallel (e1 e2 : Finset ℕ) : Prop := 
  ¬(∃ v, v ∈ e1 ∧ v ∈ e2) 

-- Main theorem
theorem number_of_parallel_pairs : 
  ∃ (pairs : Finset (Finset (Finset ℕ))), 
    (∀ (p : Finset (Finset ℕ)), p ∈ pairs → ∃ e1 e2, p = {e1, e2} ∧ e1 ∈ octahedron_edges ∧ e2 ∈ octahedron_edges ∧ edges_parallel e1 e2) 
    ∧ pairs.card = 6 :=
sorry

end number_of_parallel_pairs_l512_512093


namespace fraction_division_l512_512710

theorem fraction_division (a b c d : ℚ) (h : b ≠ 0 ∧ d ≠ 0) : (a / b) / c = a / (b * c) := sorry

example : (3 / 7) / 4 = 3 / 28 := by
  apply fraction_division
  exact ⟨by norm_num, by norm_num⟩

end fraction_division_l512_512710


namespace solve_equation_l512_512264

theorem solve_equation : ∀ (x : ℝ), x ≠ 1 → (3 / (x - 1) = 5 + 3 * x / (1 - x)) → x = 4 :=
by
  intros x hx heq
  -- sorry to skip the proof
  sorry

end solve_equation_l512_512264


namespace problem_equiv_l512_512503

open Real

noncomputable def f (x : ℝ) : ℝ := (sqrt (x^2 - x^4)) / (|x - 1| - 1)

theorem problem_equiv : 
  (∀ x, f x → ((x ∈ Icc (-1 : ℝ) (0 : ℝ) ∨ x ∈ Icc (0 : ℝ) (1 : ℝ))
  ∧ (f x ∈ Ioo (-1 : ℝ) (1 : ℝ))
  ∧ (¬ ∀ y ∈ Icc (-1 : ℝ) 1, f y ≤ f x ∧ f y ≥ f x)
  ∧ (∀ x, f x = -f (-x)))) :=
by
  sorry -- Proof will be provided separately

end problem_equiv_l512_512503


namespace simplify_fraction_l512_512645

theorem simplify_fraction :
  (3 * (Real.sqrt 3 + Real.sqrt 8)) / (2 * Real.sqrt (3 + Real.sqrt 5)) = 
  (297 - 99 * Real.sqrt 5 + 108 * Real.sqrt 6 - 36 * Real.sqrt 30) / 16 := by
  sorry

end simplify_fraction_l512_512645


namespace find_sphere_radius_l512_512461

-- Define the conditions
variables (A B C D E O: Type) (BC : ℝ) (r : ℝ)
-- Given conditions in the problem
axiom h_cone : right_circular_cone A B C D
axiom h_sphere : inscribed_in_sphere O A B D C
axiom h_midpoint : is_midpoint E A B
axiom h_perpendicular : is_perpendicular AC DE

-- Define the main problem statement we want to prove
theorem find_sphere_radius (h1 : h_cone) (h2 : h_sphere) 
(h3 : h_midpoint) (h4 : h_perpendicular) (BC_eq : BC = 1) : 
    r = (sqrt 6) / 4 :=
sorry

end find_sphere_radius_l512_512461


namespace count_whole_integers_between_cubed_roots_l512_512095

theorem count_whole_integers_between_cubed_roots :
  let s1 := Real.cbrt 30
      s2 := Real.cbrt 1000
  in (3 < s1 ∧ s1 < 4 ∧ s2 = 10) → (∃ n : ℕ, n = 6 ∧ ∀ m ∈ ({4, 5, 6, 7, 8, 9} : Finset ℕ), 4 ≤ m ∧ m ≤ 9) :=
by
  intros s1 s2 h
  have h_eq_s1 : s1 = Real.cbrt 30 := rfl
  have h_eq_s2 : s2 = Real.cbrt 1000 := rfl
  use 6
  split
  . reflexivity
  . intros m hm
    finish


end count_whole_integers_between_cubed_roots_l512_512095


namespace solve_fractional_equation_l512_512251

theorem solve_fractional_equation (x : ℝ) (h1 : x ≠ 1) (h2 : x ≠ -1) : 
    (3 / (x - 1) = 5 + 3 * x / (1 - x)) → x = 4 := 
by
  sorry

end solve_fractional_equation_l512_512251


namespace polynomial_coefficients_sum_even_odd_coefficients_difference_square_l512_512968

theorem polynomial_coefficients_sum (a : Fin 8 → ℝ):
  (a 1) + (a 2) + (a 3) + (a 4) + (a 5) + (a 6) + (a 7) = 3^7 - 1 :=
by
  -- assume the polynomial (1 + 2x)^7 has coefficients a 0, a 1, ..., a 7
  -- such that (1 + 2x)^7 = a 0 + a 1 * x + a 2 * x^2 + ... + a 7 * x^7
  sorry

theorem even_odd_coefficients_difference_square (a : Fin 8 → ℝ):
  (a 0 + a 2 + a 4 + a 6)^2 - (a 1 + a 3 + a 5 + a 7)^2 = -3^7 :=
by
  -- assume the polynomial (1 + 2x)^7 has coefficients a 0, a 1, ..., a 7
  -- such that (1 + 2x)^7 = a 0 + a 1 * x + a 2 * x^2 + ... + a 7 * x^7
  sorry

end polynomial_coefficients_sum_even_odd_coefficients_difference_square_l512_512968


namespace circumcircle_intersect_at_BC_l512_512465

variables (A B C M N O R K : Type*) [Geometry A] [Geometry B] [Geometry C]
variables [Geometry M] [Geometry N] [Geometry O] [Geometry R] [Geometry K]
variables (triangle_ABC : Triangle A B C)

-- Given conditions
def acute_triangle (A B C : Triangle.t) : Prop :=
A.B.angle < 90 ∧ C.B.angle < 90 ∧ A.C.angle < 90

lemma AB_neq_AC (triangle_ABC : Triangle A B C) : A.toPoint ≠ A.toPoint := sorry

-- The circle with diameter BC
def circle_with_diameter (B C : Type*) : Circle.t := sorry

lemma BC_circle_intersects_sides (triangle_ABC : Triangle A B C) (circle_BC : Circle.t) :
  (circle_BC ∩ triangle_ABC.edges) = {M.toPoint, N.toPoint} := sorry

def midpoint_of_BC (B C O : Type*) : O.toPoint = midpoint B.toPoint C.toPoint := sorry

-- The angle bisectors intersection point
def angle_bisectors_intersect (triangle_ABC : Triangle A B C) (M N O : Type*) : R := sorry

-- The main theorem to be proved
theorem circumcircle_intersect_at_BC (triangle_ABC : Triangle A B C) (circle_BC : Circle.t) (R : Type*) :
  circumcircle B M R ∩ circumcircle C N R = {intersection_point_on_BC} := sorry

end circumcircle_intersect_at_BC_l512_512465


namespace digits_in_Q_l512_512171

noncomputable def number_of_digits (n : ℕ) : ℕ :=
  n.toString.length

theorem digits_in_Q :
  let a := 48_769_231_456_789_325_678_912
  let b := 348_973_489_379_256_789
  let c := 3.25
  let Q := a * (Real.ofInt b * c)
  number_of_digits Q = 45 :=
by
  let a := 48_769_231_456_789_325_678_912
  let b := 348_973_489_379_256_789
  let c := 3.25
  let Q := a * (Real.ofInt b * c)
  sorry

end digits_in_Q_l512_512171


namespace cos_theta_value_find_k_l512_512912

variables (a b : ℝ × ℝ)

def vector_a : ℝ × ℝ := (1, 2)
def vector_b : ℝ × ℝ := (1, -1)

def cos_theta (a b : ℝ × ℝ) : ℝ :=
(a.1 * b.1 + a.2 * b.2) / (real.sqrt (a.1^2 + a.2^2) * real.sqrt (b.1^2 + b.2^2))

def dot_product (u v : ℝ × ℝ) : ℝ := u.1 * v.1 + u.2 * v.2

theorem cos_theta_value : cos_theta vector_a vector_b = -real.sqrt 10 / 10 :=
by sorry

theorem find_k (k : ℝ) : dot_product (2 • vector_a + vector_b) (k • vector_a - vector_b) = 0 → k = 0 :=
by sorry

end cos_theta_value_find_k_l512_512912


namespace quadratic_roots_range_l512_512499

theorem quadratic_roots_range (m : ℝ) : 
  (2 * x^2 - (m + 1) * x + m = 0) → 
  (m^2 - 6 * m + 1 > 0) → 
  (0 < m) → 
  (0 < m ∧ m < 3 - 2 * Real.sqrt 2 ∨ m > 3 + 2 * Real.sqrt 2) :=
by
  sorry

end quadratic_roots_range_l512_512499


namespace range_of_omega_correct_l512_512502

noncomputable def range_of_omega (f : ℝ → ℝ) (ω : ℝ) (ϕ : ℝ) : Prop :=
  (∃ ω > 0, -π ≤ ϕ ∧ ϕ ≤ 0) ∧
  (∀ x : ℝ, f x = cos (ω * x + ϕ)) ∧
  (∀ x y : ℝ, -π/4 ≤ x ∧ x ≤ 3π/16 ∧ -π/4 ≤ y ∧ y ≤ 3π/16 → f x ≤ f y) →
  (∀ ω, 0 < ω ∧ ω ≤ 2)

theorem range_of_omega_correct :
  ∃ f : ℝ → ℝ, ∃ ω ϕ : ℝ, range_of_omega f ω ϕ := sorry

end range_of_omega_correct_l512_512502


namespace new_quadratic_equation_l512_512680

-- Define the quadratic equation and its roots
theorem new_quadratic_equation (a b c : ℝ) (x1 x2 : ℝ) (h : a ≠ 0)
  (h_roots : a * x1^2 + b * x1 + c = 0 ∧ a * x2^2 + b * x2 + c = 0 ∧ x1 ≥ x2) :
  ∃ y1 y2 : ℝ, (y1 = x1 - 1 ∧ y2 = x2 + 1) ∧ 
  (a * y1^2 + b * y1 + (c - a + real.sqrt (b^2 - 4 * a * c)) = 0 ∧ 
   a * y2^2 + b * y2 + (c - a + real.sqrt (b^2 - 4 * a * c)) = 0) :=
sorry

end new_quadratic_equation_l512_512680


namespace solve_fractional_equation_l512_512254

theorem solve_fractional_equation (x : ℝ) (h₁ : x ≠ 1) (h₂ : x ≠ -1) :
  (3 / (x - 1) = 5 + 3 * x / (1 - x)) → (x = 4) :=
by 
  intro h
  -- proof steps would go here
  sorry

end solve_fractional_equation_l512_512254


namespace luke_stickers_l512_512987

theorem luke_stickers : 
  let initial_stickers := 20 in
  let bought_stickers := 12 in
  let birthday_stickers := 20 in
  let given_to_sister := 5 in
  let used_for_card := 8 in
  (initial_stickers + bought_stickers + birthday_stickers - given_to_sister - used_for_card) = 39 :=
by
  sorry

end luke_stickers_l512_512987


namespace monotonically_increasing_k_range_l512_512069

theorem monotonically_increasing_k_range :
  ∀ (k : ℝ), (∀ x : ℝ, 5 ≤ x → ∀ y, 5 ≤ y → f x ≤ f y) ↔ k ≤ 40 :=
by
  let f := λ x : ℝ, 4 * x^2 - k * x - 8
  sorry

end monotonically_increasing_k_range_l512_512069


namespace square_and_semicircle_side_relationship_l512_512752

def π : ℝ := Real.pi

theorem square_and_semicircle_side_relationship
  (s : ℝ) (d : ℝ)
  (h1 : s = 8) 
  (h2 : 4 * s - s + (1/2) * π * d = 36.56637061435917) : 
  d = s :=
by
  sorry

end square_and_semicircle_side_relationship_l512_512752


namespace part1_l512_512506

theorem part1 (a : ℝ) (f : ℝ → ℝ) (h : ∀ x, x > 0 → f x < 0) :
  a > 1 :=
sorry

end part1_l512_512506


namespace halfway_fraction_between_is_one_fourth_l512_512813

theorem halfway_fraction_between_is_one_fourth : 
  let f1 := (1 / 4 : ℚ)
  let f2 := (1 / 6 : ℚ)
  let f3 := (1 / 3 : ℚ)
  ((f1 + f2 + f3) / 3) = (1 / 4) := 
by
  let f1 := (1 / 4 : ℚ)
  let f2 := (1 / 6 : ℚ)
  let f3 := (1 / 3 : ℚ)
  sorry

end halfway_fraction_between_is_one_fourth_l512_512813


namespace area_rectangle_ABCD_l512_512577

-- Declare variables and their types
variable (A B C D E F G O H : Type)
variable (A E F G : Point)
variable (BC AE : Segment)
variable (area_ABCD area_AOF area_GOH area_BCFG area_CEH : ℝ)

-- Define midpoints
def is_midpoint (P Q R: Point) : Prop := dist P R = 2 * dist P Q

-- Given conditions in Lean code
variable h1 : is_midpoint B E C
variable h2 : is_midpoint A O E
variable h3 : area_AOF = 2 ∧ area_BCFG = 11
variable h4 : area_CEH = 7

-- Proof goal: Area of rectangle ABCD is 28 square centimeters
theorem area_rectangle_ABCD : area_ABCD = 28 :=
by
  sorry

end area_rectangle_ABCD_l512_512577


namespace range_of_m_l512_512505

theorem range_of_m (f : ℝ → ℝ) (a : ℝ) (m : ℝ) : 
  (∀ x, f x = log x + x^2 - 2 * a * x + 1) →
  (∀ a, a ∈ Icc (-2 : ℝ) (0 : ℝ)) →
  (∃ (x0 : ℝ), x0 ∈ Ioc 0 1 ∧ (∀ a ∈ Icc (-2 : ℝ) (0 : ℝ), 2 * m * exp a * (a + 1) + f x0 > a^2 + 2 * a + 4)) ↔ (1 < m ∧ m ≤ exp 2) :=
by
  sorry

end range_of_m_l512_512505


namespace common_ratio_of_geometric_series_l512_512389

theorem common_ratio_of_geometric_series (a S r : ℝ) (h₁ : a = 400) (h₂ : S = 2500) :
  S = a / (1 - r) → r = 21 / 25 :=
by
  intros h₃
  rw [h₁, h₂] at h₃
  sorry

end common_ratio_of_geometric_series_l512_512389


namespace product_of_cosines_of_two_distinct_values_in_S_l512_512020

noncomputable def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + d

theorem product_of_cosines_of_two_distinct_values_in_S
  (a : ℕ → ℝ)
  (d : ℝ)
  (h_sequence: arithmetic_sequence a (2 * Real.pi / 3))
  (hS : ∃ (b c : ℝ), b ≠ c ∧ (∀ n : ℕ, ∃b', b' = cos (a n) ∧ (b' = b ∨ b' = c))) :
  (∃ b c : ℝ, b ≠ c ∧ (∀ n : ℕ, cos (a n) = b ∨ cos (a n) = c) ∧ b * c = -1 / 2) := 
sorry

end product_of_cosines_of_two_distinct_values_in_S_l512_512020


namespace general_term_sum_bn_l512_512894

noncomputable def S (n : ℕ) : ℕ := 2 * n^2 + 2 * n
noncomputable def a (n : ℕ) : ℕ := 4 * n
noncomputable def b (n : ℕ) : ℕ := 2 ^ (4 * n)
noncomputable def T (n : ℕ) : ℝ := (16 / 15) * (16^n - 1)

theorem general_term (n : ℕ) (h1 : S n = 2 * n^2 + 2 * n) 
    (h2 : S (n-1) = 2 * (n-1)^2 + 2 * (n-1))
    (h3 : n ≥ 1) : a n = 4 * n :=
by sorry

theorem sum_bn (n : ℕ) (h : ∀ n, (b n, a n) = ((2 ^ (4 * n)), 4 * n)) : 
    T n = (16 / 15) * (16^n - 1) :=
by sorry

end general_term_sum_bn_l512_512894


namespace margin_in_terms_of_selling_price_l512_512924

noncomputable theory

variables (S C M : ℝ) (n : ℝ)

-- Conditions
def cost_relation : Prop := S - M = C
def margin_relation : Prop := M = (2 / n) * C

-- Theorem statement
theorem margin_in_terms_of_selling_price (h1 : cost_relation S C M) (h2 : margin_relation M n C) : 
  M = (2 * S) / (n + 2) := 
sorry

end margin_in_terms_of_selling_price_l512_512924


namespace distance_between_points_is_sqrt_5_l512_512140

noncomputable def distance_between_polar_points : ℝ :=
  let xA := 1 * Real.cos (3/4 * Real.pi)
  let yA := 1 * Real.sin (3/4 * Real.pi)
  let xB := 2 * Real.cos (Real.pi / 4)
  let yB := 2 * Real.sin (Real.pi / 4)
  Real.sqrt ((xB - xA)^2 + (yB - yA)^2)

theorem distance_between_points_is_sqrt_5 :
  distance_between_polar_points = Real.sqrt 5 :=
by
  sorry

end distance_between_points_is_sqrt_5_l512_512140


namespace find_z_coordinate_l512_512756

theorem find_z_coordinate :
  ∀ (t : ℝ), ∃ (z : ℝ), z = 2 - 3 * t ∧ (1 + 3 * t = 7) :=
by
  intro t
  use -4
  constructor
  sorry


end find_z_coordinate_l512_512756


namespace regina_earnings_l512_512210

def num_cows : ℕ := 20

def num_pigs (num_cows : ℕ) : ℕ := 4 * num_cows

def price_per_pig : ℕ := 400
def price_per_cow : ℕ := 800

def earnings (num_cows num_pigs price_per_cow price_per_pig : ℕ) : ℕ :=
  num_cows * price_per_cow + num_pigs * price_per_pig

theorem regina_earnings :
  earnings num_cows (num_pigs num_cows) price_per_cow price_per_pig = 48000 :=
by
  -- proof omitted
  sorry

end regina_earnings_l512_512210


namespace speed_of_second_train_l512_512376

noncomputable def speed_of_first_train_kmph := 60 -- km/h
noncomputable def speed_of_first_train_mps := (speed_of_first_train_kmph * 1000) / 3600 -- m/s
noncomputable def length_of_first_train := 145 -- m
noncomputable def length_of_second_train := 165 -- m
noncomputable def time_to_cross := 8 -- seconds
noncomputable def total_distance := length_of_first_train + length_of_second_train -- m
noncomputable def relative_speed := total_distance / time_to_cross -- m/s

theorem speed_of_second_train (V : ℝ) :
  V * 1000 / 3600 + 60 * 1000 / 3600 = 38.75 →
  V = 79.5 := by {
  sorry
}

end speed_of_second_train_l512_512376


namespace complement_union_eq_l512_512529

def U : Set ℤ := {-2, -1, 0, 1, 2, 3}
def A : Set ℤ := {-1, 2}
def B : Set ℤ := {x | x ^ 2 - 4 * x + 3 = 0}

theorem complement_union_eq :
  (U \ (A ∪ B)) = {-2, 0} :=
by
  sorry

end complement_union_eq_l512_512529


namespace amanda_pay_if_not_finished_l512_512157

-- Define Amanda's hourly rate and daily work hours.
def amanda_hourly_rate : ℝ := 50
def amanda_daily_hours : ℝ := 10

-- Define the percentage of pay Jose will withhold.
def withholding_percentage : ℝ := 0.20

-- Define Amanda's total pay if she finishes the sales report.
def amanda_total_pay : ℝ := amanda_hourly_rate * amanda_daily_hours

-- Define the amount withheld if she does not finish the sales report.
def withheld_amount : ℝ := amanda_total_pay * withholding_percentage

-- Define the amount Amanda will receive if she does not finish the sales report.
def amanda_final_pay_not_finished : ℝ := amanda_total_pay - withheld_amount

-- The theorem to prove:
theorem amanda_pay_if_not_finished : amanda_final_pay_not_finished = 400 := by
  sorry

end amanda_pay_if_not_finished_l512_512157


namespace solve_fractional_equation_l512_512247

theorem solve_fractional_equation (x : ℝ) (h1 : x ≠ 1) (h2 : x ≠ -1) : 
    (3 / (x - 1) = 5 + 3 * x / (1 - x)) → x = 4 := 
by
  sorry

end solve_fractional_equation_l512_512247


namespace cost_of_ice_cream_l512_512772

/-- Alok ordered 16 chapatis, 5 plates of rice, 7 plates of mixed vegetable, and 6 ice-cream cups. 
    The cost of each chapati is Rs. 6, that of each plate of rice is Rs. 45, and that of mixed 
    vegetable is Rs. 70. Alok paid the cashier Rs. 931. Prove the cost of each ice-cream cup is Rs. 20. -/
theorem cost_of_ice_cream (n_chapatis n_rice n_vegetable n_ice_cream : ℕ) 
    (cost_chapati cost_rice cost_vegetable total_paid : ℕ)
    (h_chapatis : n_chapatis = 16) 
    (h_rice : n_rice = 5)
    (h_vegetable : n_vegetable = 7)
    (h_ice_cream : n_ice_cream = 6)
    (h_cost_chapati : cost_chapati = 6)
    (h_cost_rice : cost_rice = 45)
    (h_cost_vegetable : cost_vegetable = 70)
    (h_total_paid : total_paid = 931) :
    (total_paid - (n_chapatis * cost_chapati + n_rice * cost_rice + n_vegetable * cost_vegetable)) / n_ice_cream = 20 := 
by
  sorry

end cost_of_ice_cream_l512_512772


namespace problem_solution_l512_512430

theorem problem_solution (n : ℕ) (h1 : n ≥ 1) (m : ℕ) (h2 : ∃ (m≥1), (2^n - 1) % 3 = 0 ∧ (4 * m^2 + 1) % ((2^n - 1) / 3) = 0) :
  ∃ (r : ℕ), n = 2^r :=
by
  sorry

end problem_solution_l512_512430


namespace solve_fractional_equation_l512_512225

theorem solve_fractional_equation (x : ℝ) (hx : x ≠ 1 ∧ x ≠ 1) : 3 / (x - 1) = 5 + 3 * x / (1 - x) ↔ x = 4 :=
by 
  split
  -- Forward direction (Given the equation, show x = 4)
  intro h
  -- Multiply out and simplify as shown in the solution steps
  -- Sorry statement skips the detailed proof
  sorry
  -- Converse direction (Substitute x = 4 and verify it satisfies the equation)
  intro hx
  -- Substitution and verification steps
  -- Sorry statement skips the detailed proof
  sorry

end solve_fractional_equation_l512_512225


namespace common_ratio_of_geometric_series_l512_512390

theorem common_ratio_of_geometric_series (a S r : ℚ) (ha : a = 400) (hS : S = 2500) (hS_eq : S = a / (1 - r)) : r = 21 / 25 :=
by {
  rw [ha, hS] at hS_eq,
  sorry
}

end common_ratio_of_geometric_series_l512_512390


namespace max_value_expression_l512_512054

theorem max_value_expression (n : ℕ) (x : Fin n → ℝ) 
  (h : ∀ i, 0 < x i ∧ x i < 1) :
  (∃ x_i, i ∈ Finset.range n → ∃ (hxi : x_i ∈ set.Ioo 0 1), 
  A = (∑ i in Finset.range n, Real.root 4 (1 - x i)) / (∑ i in Finset.range n, 1 / Real.root 4 (x i))) :=
  sorry

end max_value_expression_l512_512054


namespace weight_of_B_l512_512275

variable (W_A W_B W_C W_D : ℝ)

theorem weight_of_B (h1 : (W_A + W_B + W_C + W_D) / 4 = 60)
                    (h2 : (W_A + W_B) / 2 = 55)
                    (h3 : (W_B + W_C) / 2 = 50)
                    (h4 : (W_C + W_D) / 2 = 65) :
                    W_B = 50 :=
by sorry

end weight_of_B_l512_512275


namespace quadratic_function_negative_values_l512_512549

theorem quadratic_function_negative_values (a : ℝ) : 
  (∃ x : ℝ, (x^2 - a*x + 1) < 0) ↔ (a > 2 ∨ a < -2) :=
by
  sorry

end quadratic_function_negative_values_l512_512549


namespace unique_root_ln_eqn_l512_512450

/-- For what values of the parameter \(a\) does the equation
   \(\ln(x - 2a) - 3(x - 2a)^2 + 2a = 0\) have a unique root? -/
theorem unique_root_ln_eqn (a : ℝ) :
  ∃! x : ℝ, (Real.log (x - 2 * a) - 3 * (x - 2 * a) ^ 2 + 2 * a = 0) ↔
  a = (1 + Real.log 6) / 4 :=
sorry

end unique_root_ln_eqn_l512_512450


namespace product_of_cosine_elements_l512_512015

-- Definitions for the problem conditions
def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + d

def S (a : ℕ → ℝ) (d : ℝ) : Set ℝ :=
  {x | ∃ n : ℕ, x = Real.cos (a n)}

-- Main theorem statement
theorem product_of_cosine_elements 
  (a : ℕ → ℝ) 
  (d : ℝ) 
  (h_seq : arithmetic_sequence a (2 * Real.pi / 3))
  (h_S_elements : (S a (2 * Real.pi / 3)).card = 2) 
  (h_S_contains : ∃ a b, a ≠ b ∧ S a (2 * Real.pi / 3) = {a, b}) :
  let (a, b) := Classical.choose (h_S_contains) in
  a * b = -1 / 2 :=
by
  sorry

end product_of_cosine_elements_l512_512015


namespace least_multiple_of_13_gt_450_l512_512319

theorem least_multiple_of_13_gt_450 : ∃ (n : ℕ), (455 = 13 * n) ∧ 455 > 450 ∧ ∀ m : ℕ, (13 * m > 450) → 455 ≤ 13 * m :=
by
  sorry

end least_multiple_of_13_gt_450_l512_512319


namespace final_position_correct_l512_512754

structure Position :=
(base : ℝ × ℝ)
(stem : ℝ × ℝ)

def initial_position : Position :=
{ base := (0, -1),
  stem := (1, 0) }

def reflect_x (p : Position) : Position :=
{ base := (p.base.1, -p.base.2),
  stem := (p.stem.1, -p.stem.2) }

def rotate_90_ccw (p : Position) : Position :=
{ base := (-p.base.2, p.base.1),
  stem := (-p.stem.2, p.stem.1) }

def half_turn (p : Position) : Position :=
{ base := (-p.base.1, -p.base.2),
  stem := (-p.stem.1, -p.stem.2) }

def reflect_y (p : Position) : Position :=
{ base := (-p.base.1, p.base.2),
  stem := (-p.stem.1, p.stem.2) }

def final_position : Position :=
reflect_y (half_turn (rotate_90_ccw (reflect_x initial_position)))

theorem final_position_correct : final_position = { base := (1, 0), stem := (0, 1) } :=
sorry

end final_position_correct_l512_512754


namespace cos_set_product_l512_512032

noncomputable def arithmetic_sequence (a1 : ℝ) (d : ℝ) : ℕ → ℝ :=
  λ n, a1 + d * (n - 1)

theorem cos_set_product (a1 : ℝ) (h1 : (arithmetic_sequence a1 (2 * Real.pi / 3) '' {n | n ≥ 1}).to_set.cos.to_finset.card = 2) :
  let S := (arithmetic_sequence a1 (2 * Real.pi / 3) '' {n | n ≥ 1}).to_set.cos.to_finset in 
  (S.to_finset : set ℝ).product = -1 / 2 := sorry

end cos_set_product_l512_512032


namespace distinct_sums_for_large_n_counterexample_for_equal_n_l512_512871

open Nat

/-- Given \( k > 2 \) and \( n > \binom{k}{3} \) (where \(\binom{a}{b}\) is the usual binomial coefficient),
    show that there must be at least \( k+1 \) distinct numbers in the set
    \( \{a_i + b_i, b_i + c_i, c_i + a_i \mid i = 1, 2, \dots, n\} \),
    where \( a_i, b_i, c_i \) (for \( i = 1, 2, \dots, n \)) are 3n distinct real numbers. -/
theorem distinct_sums_for_large_n (k : ℕ) (n : ℕ) (ai bi ci : ℕ → ℝ) 
  (hk : k > 2) 
  (hn : n > Nat.choose k 3) 
  (h_distinct : ∀ i j, i ≠ j → ai i ≠ ai j ∧ bi i ≠ bi j ∧ ci i ≠ ci j ∧ ai i ≠ bi j ∧ bi i ≠ ci j ∧ ci i ≠ ai j) :
  ∃ k1 (S : set ℝ), S.card ≥ k + 1 ∧ (∀ i, (ai i + bi i) ∈ S ∧ (bi i + ci i) ∈ S ∧ (ci i + ai i) ∈ S) :=
sorry

/-- Show that the statement is not always true for \( n = \binom{k}{3} \). -/
theorem counterexample_for_equal_n (k : ℕ) (n : ℕ) (ai bi ci : ℕ → ℝ) 
  (hk : k > 2) 
  (hn : n = Nat.choose k 3) 
  (h_distinct : ∀ i j, i ≠ j → ai i ≠ ai j ∧ bi i ≠ bi j ∧ ci i ≠ ci j ∧ ai i ≠ bi j ∧ bi i ≠ ci j ∧ ci i ≠ ai j) :
  ¬ (∃ k1 (S : set ℝ), S.card ≥ k + 1 ∧ 
    (∀ i, (ai i + bi i) ∈ S ∧ (bi i + ci i) ∈ S ∧ (ci i + ai i) ∈ S)) :=
sorry

end distinct_sums_for_large_n_counterexample_for_equal_n_l512_512871


namespace solve_equation_l512_512240

theorem solve_equation (x : ℝ) (h : x ≠ 1 ∧ x ≠ 1) : (3 / (x - 1) = 5 + 3 * x / (1 - x)) ↔ x = 4 := by
  sorry

end solve_equation_l512_512240


namespace problem_statement_l512_512466

-- Definitions for the ellipse
def ellipse (x y a b : ℝ) : Prop := (x^2)/(a^2) + (y^2)/(b^2) = 1

-- Given conditions
variables (e : ℝ) (h1 : e = sqrt 3 / 2) (a b c : ℝ) (ha : a > b) (hb : b > 0) (minor_axis : 2 * b = 2)
include h1 ha hb minor_axis

-- Definition for the line passing through point (0, 2)
def line (l : ℝ → ℝ → Prop) : Prop := ∃ k : ℝ, l = (λ x y, y = k * x + 2)

-- Equation to check the condition for OP ⋅ OQ = 0
def orthogonal_condition (x1 y1 x2 y2 : ℝ) : Prop := x1 * x2 + (k * x1 + 2) * (k * x2 + 2) = 0

-- The main theorem
theorem problem_statement : 
  (∃ a, ∃ b, ∃c, a > b ∧ b > 0 ∧ e = sqrt 3 / 2 ∧ 2 * b = 2 ∧ (b = 1) ∧ (a^2 = b^2 + c^2) ∧ (a = 2)) ∧ 
  (ellipse (x y : ℝ) 2 1) ∧ 
  (∃ k : ℝ, line (λ x y, y = k * x + 2)) ∧ 
  (∀ P Q : ℝ × ℝ, line (λ x y, y = k * x + 2) → orthogonal_condition (P.1) (P.2) (Q.1) (Q.2)) :=
sorry

end problem_statement_l512_512466


namespace intervals_of_monotonicity_a_eq_1_max_value_implies_a_half_l512_512504

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := Real.log x + Real.log (2 - x) + a * x

theorem intervals_of_monotonicity_a_eq_1 : 
  ∀ x : ℝ, (0 < x ∧ x < Real.sqrt 2) → 
  f x 1 < f (Real.sqrt 2) 1 ∧ 
  ∀ x : ℝ, (Real.sqrt 2 < x ∧ x < 2) → 
  f x 1 > f (Real.sqrt 2) 1 := 
sorry

theorem max_value_implies_a_half : 
  ∀ x : ℝ, (0 < x ∧ x ≤ 1) ∧ f 1 a = 1/2 → a = 1/2 := 
sorry

end intervals_of_monotonicity_a_eq_1_max_value_implies_a_half_l512_512504


namespace snail_reaches_top_l512_512378

-- Given conditions
def tree_height : ℕ := 24
def day_climb : ℕ := 6
def night_slide : ℕ := 4
def net_progress : ℕ := day_climb - night_slide
def last_climb : ℕ := day_climb
def days_to_top : ℕ := 10

-- Proof statement
theorem snail_reaches_top :
    let net_progress := day_climb - night_slide in
    let remaining_climb := tree_height - last_climb in
    let required_days := remaining_climb / net_progress in
    required_days + 1 = days_to_top :=
    by 
    -- Proof steps would go here
    sorry

end snail_reaches_top_l512_512378


namespace crease_points_ellipse_l512_512195

theorem crease_points_ellipse (R a : ℝ) (x y : ℝ) (h1 : 0 < R) (h2 : 0 < a) (h3 : a < R) : 
  (x - a / 2) ^ 2 / (R / 2) ^ 2 + y ^ 2 / ((R / 2) ^ 2 - (a / 2) ^ 2) ≥ 1 :=
by
  -- Omitted detailed proof steps
  sorry

end crease_points_ellipse_l512_512195


namespace probability_product_divisible_by_8_l512_512638

theorem probability_product_divisible_by_8 :
  let dice := fin 8;
  let roll_outcomes := fin 6;
  probability (∃ n ∈ roll_outcomes, n = 8) = 65 / 72 :=
sorry

end probability_product_divisible_by_8_l512_512638


namespace solve_fractional_equation_l512_512257

theorem solve_fractional_equation (x : ℝ) (h₁ : x ≠ 1) (h₂ : x ≠ -1) :
  (3 / (x - 1) = 5 + 3 * x / (1 - x)) → (x = 4) :=
by 
  intro h
  -- proof steps would go here
  sorry

end solve_fractional_equation_l512_512257


namespace sequence_exists_and_unique_l512_512691

theorem sequence_exists_and_unique (a : ℕ → ℕ) :
  a 0 = 11 ∧ a 7 = 12 ∧
  (∀ n : ℕ, n < 6 → a n + a (n + 1) + a (n + 2) = 50) →
  (a 0 = 11 ∧ a 1 = 12 ∧ a 2 = 27 ∧ a 3 = 11 ∧ a 4 = 12 ∧
   a 5 = 27 ∧ a 6 = 11 ∧ a 7 = 12) :=
by
  sorry

end sequence_exists_and_unique_l512_512691


namespace proposition_truth_values_l512_512557

-- The original proposition: If z1 and z2 are conjugate complex numbers, then |z1| = |z2|
def original_proposition (z1 z2 : ℂ) : Prop :=
  (complex.conj z1 = z2) → (complex.abs z1 = complex.abs z2)

-- Definition of the inverse proposition: If |z1| = |z2|, then z1 and z2 are conjugate complex numbers
def inverse_proposition (z1 z2 : ℂ) : Prop :=
  (complex.abs z1 = complex.abs z2) → (complex.conj z1 = z2)

-- Definition of the negation of the original proposition
def negation_proposition (z1 z2 : ℂ) : Prop :=
  ¬((complex.conj z1 = z2) → (complex.abs z1 = complex.abs z2))

-- Definition of the contrapositive: If |z1| ≠ |z2|, then z1 and z2 are not conjugate complex numbers
def contrapositive_proposition (z1 z2 : ℂ) : Prop :=
  (complex.abs z1 ≠ complex.abs z2) → (complex.conj z1 ≠ z2)

-- Lean 4 statement to prove the truth values of the inverse proposition, the negation, and the contrapositive
theorem proposition_truth_values (z1 z2 : ℂ) :
  original_proposition z1 z2 →
  ¬inverse_proposition z1 z2 ∧
  ¬negation_proposition z1 z2 ∧
  contrapositive_proposition z1 z2 :=
begin
  -- proof would go here, but we are skipping it as per the instructions
  sorry
end

end proposition_truth_values_l512_512557


namespace midpoint_condition_l512_512960

theorem midpoint_condition (a b : ℝ) (P : ℝ × ℝ) (AP PB AB s : ℝ) (C : ℝ × ℝ)
  (h1 : AB = sqrt (a^2 + b^2))
  (h2 : AP^2 + PB^2 = s)
  (CP : ℝ)
  (h3 : CP^2 = a^2)
  (h4 : s = 2 * CP^2) :
  (2 * a^2 = 2 * (1/2 * sqrt (a^2 + b^2))^2 - 2 * (1/2 * sqrt (a^2 + b^2)) * sqrt (a^2 + b^2) + a^2 + b^2) :=
sorry

end midpoint_condition_l512_512960


namespace solution_set_x_l512_512792

theorem solution_set_x (x : ℝ) :
  abs (x + 1) * abs (x - 2) * abs (x + 3) * abs (x - 4) = 
  abs (x - 1) * abs (x + 2) * abs (x - 3) * abs (x + 4) ↔
  x = 0 ∨ x = sqrt 7 ∨ x = -sqrt 7 ∨ 
  x = sqrt ((13 + sqrt 73) / 2) ∨ x = -sqrt ((13 + sqrt 73) / 2) ∨ 
  x = sqrt ((13 - sqrt 73) / 2) ∨ x = -sqrt ((13 - sqrt 73) / 2) :=
  sorry

end solution_set_x_l512_512792


namespace angle_between_PQ_and_RS_half_angle_XOZ_l512_512169

noncomputable theory

open EuclideanGeometry

variables {A X Y Z B P Q R S O : Point}

-- Conditions
variable (h1 : convex_pentagon_inscribed_in_semicircle A X Y Z B)
variable (h2 : perpendicular_from_point_onto_line P Y A X)
variable (h3 : perpendicular_from_point_onto_line Q Y B X)
variable (h4 : perpendicular_from_point_onto_line R Y A Z)
variable (h5 : perpendicular_from_point_onto_line S Y B Z)
variable (hO : midpoint O A B)
-- Question to be proved
theorem angle_between_PQ_and_RS_half_angle_XOZ :
  acute_angle (line_through P Q) (line_through R S) = (angle X O Z) / 2 :=
sorry

end angle_between_PQ_and_RS_half_angle_XOZ_l512_512169


namespace initial_carrots_count_l512_512624

theorem initial_carrots_count (x : ℕ) (h1 : x - 2 + 21 = 31) : x = 12 := by
  sorry

end initial_carrots_count_l512_512624


namespace find_product_l512_512731

theorem find_product (a b c d : ℚ) 
  (h₁ : 2 * a + 4 * b + 6 * c + 8 * d = 48)
  (h₂ : 4 * (d + c) = b)
  (h₃ : 4 * b + 2 * c = a)
  (h₄ : c + 1 = d) :
  a * b * c * d = -319603200 / 10503489 := sorry

end find_product_l512_512731


namespace remainder_of_349_by_17_is_9_l512_512325

theorem remainder_of_349_by_17_is_9 :
  349 % 17 = 9 :=
sorry

end remainder_of_349_by_17_is_9_l512_512325


namespace complex_modulus_z_l512_512610

-- Define the complex number z
def z : ℂ := (1 - 2 * complex.I) / (2 + complex.I)

-- State the theorem to prove
theorem complex_modulus_z : complex.abs z = 1 :=
sorry

end complex_modulus_z_l512_512610


namespace find_probability_of_B_l512_512567

-- Define the conditions and the problem
def system_A_malfunction_prob := 1 / 10
def at_least_one_not_malfunction_prob := 49 / 50

/-- The probability that System B malfunctions given that 
  the probability of at least one system not malfunctioning 
  is 49/50 and the probability of System A malfunctioning is 1/10 -/
theorem find_probability_of_B (p : ℝ) 
  (h1 : system_A_malfunction_prob = 1 / 10) 
  (h2 : at_least_one_not_malfunction_prob = 49 / 50) 
  (h3 : 1 - (system_A_malfunction_prob * p) = at_least_one_not_malfunction_prob) : 
  p = 1 / 5 :=
sorry

end find_probability_of_B_l512_512567


namespace relationship_y1_y2_y3_l512_512471

variable (y1 y2 y3 b : ℝ)
variable (h1 : y1 = 3 * (-3) - b)
variable (h2 : y2 = 3 * 1 - b)
variable (h3 : y3 = 3 * (-1) - b)

theorem relationship_y1_y2_y3 : y1 < y3 ∧ y3 < y2 := by
  sorry

end relationship_y1_y2_y3_l512_512471


namespace unsold_tomatoes_l512_512620

theorem unsold_tomatoes (total_harvest sold_maxwell sold_wilson : ℝ) 
(h_total_harvest : total_harvest = 245.5)
(h_sold_maxwell : sold_maxwell = 125.5)
(h_sold_wilson : sold_wilson = 78) :
(total_harvest - (sold_maxwell + sold_wilson) = 42) :=
by {
  sorry
}

end unsold_tomatoes_l512_512620


namespace distance_GH_pq_l512_512597

-- Definitions of the given conditions
variables {A B C D G H : Type} [OrderedField A B C D G H]

-- Conditions about the geometrical properties
def is_isosceles_trapezoid (A B C D : Type) : Prop := (Parallel A D B C) ∧ (Isosceles A B C D)
def angle_cond (A D : Type) : Prop := (angle A D = π / 4)
def diagonal_length (A C B D : Type) : Prop := (diagonal A C = 12 * sqrt 13) ∧ (diagonal B D = 12 * sqrt 13)
def distance_GA_12sqrt3 (G A : Type) : Prop := (distance G A = 12 * sqrt 3)
def distance_GD_36sqrt3 (G D : Type) : Prop := (distance G D = 36 * sqrt 3)
def altitude_H_foot (C : Type) (AD : Type) : Prop := (foot_linked C AD _ H)

-- Question statement in Lean: Find the expression p*sqrt q such that GH distance is obtained and p + q equals 39
theorem distance_GH_pq (A B C D G H : Type) [is_isosceles_trapezoid A B C D] [angle_cond A D]
  [diagonal_length A B C D] [distance_GA_12sqrt3 G A] [distance_GD_36sqrt3 G D] [altitude_H_foot C _] :
  ∃ p q : ℕ, (distance G H = p * sqrt q) ∧ (nat.gcd q 1 = 1) ∧ (p + q = 39) :=
  sorry -- Proof omitted

end distance_GH_pq_l512_512597


namespace cos_arithmetic_seq_product_l512_512049

theorem cos_arithmetic_seq_product :
  ∃ (a b : ℝ), (∃ (a₁ : ℝ), ∀ n : ℕ, (n > 0) → ∃ m : ℕ, cos (a₁ + (2 * Real.pi / 3) * (n - 1)) = [a, b] ∧ (a * b = -1 / 2)) := 
sorry

end cos_arithmetic_seq_product_l512_512049


namespace probability_transform_in_S_l512_512374

open Complex

noncomputable def S : Set ℂ :=
  { z : ℂ | let x := z.re in let y := z.im in -2 ≤ x ∧ x ≤ 2 ∧ -2 ≤ y ∧ y ≤ 2 }

theorem probability_transform_in_S (z : ℂ) (hz : z ∈ S) :
  (1/2 + Complex.I/2) * z ∈ S :=
  sorry

end probability_transform_in_S_l512_512374


namespace meet_second_time_4_5_minutes_l512_512453

-- Define the initial conditions
def opposite_ends := true      -- George and Henry start from opposite ends
def pass_in_center := 1.5      -- They pass each other in the center after 1.5 minutes
def no_time_lost := true       -- No time lost in turning
def constant_speeds := true    -- They maintain their respective speeds

-- Prove that they pass each other the second time after 4.5 minutes
theorem meet_second_time_4_5_minutes :
  opposite_ends ∧ pass_in_center = 1.5 ∧ no_time_lost ∧ constant_speeds → 
  ∃ t : ℝ, t = 4.5 := by
  sorry

end meet_second_time_4_5_minutes_l512_512453


namespace count_repeating_primes_correct_l512_512446

def is_prime (n : ℕ) : Prop := Nat.Prime n

def is_repeating_decimal (n : ℕ) : Prop :=
  let k := n + 1
  ¬ (k % 2 = 0) ∧ ¬ (k % 5 = 0)

def filtered_primes := (List.range' 2 (200 - 2 + 1)).filter is_prime

def count_repeating_primes : ℕ :=
  (filtered_primes.filter is_repeating_decimal).length

theorem count_repeating_primes_correct :
  count_repeating_primes = 35 :=
by
  sorry

end count_repeating_primes_correct_l512_512446


namespace proof_A_union_C_U_B_l512_512518

-- Definitions based on conditions
def U := {x : ℤ | abs x < 3}
def A := {1, 2}
def B := {-2, -1, 2}

-- Definition of complement of B in U
def C_U_B := U \ B

-- Theorem statement equivalent to the given problem
theorem proof_A_union_C_U_B : A ∪ (C_U_B) = {0, 1, 2} := 
by sorry

end proof_A_union_C_U_B_l512_512518


namespace trigonometric_identity_l512_512688

theorem trigonometric_identity : 
  cos (15 * (Real.pi / 180)) * sin (30 * (Real.pi / 180)) * cos (75 * (Real.pi / 180)) * sin (150 * (Real.pi / 180)) = 1 / 16 := 
by
  sorry

end trigonometric_identity_l512_512688


namespace probability_third_term_three_a_plus_b_l512_512174

open Finset

def T : Finset (Equiv.Perm (Fin 6)) :=
  univ.filter (λ σ, σ 0 ≠ 1)

def favorable_permutations : Nat :=
  (T.filter (λ σ, σ 2 = 3)).card

def total_permutations : Nat :=
  T.card

def probability : Rat :=
  favorable_permutations / total_permutations

theorem probability_third_term_three :
  probability = 4 / 25 :=
by
  sorry

theorem a_plus_b :
  let ⟨a, b, h⟩ := probability.num_denom in
  a + b = 29 :=
by
  sorry

end probability_third_term_three_a_plus_b_l512_512174


namespace length_of_uncovered_side_l512_512373

variables (L W : ℝ)

-- Conditions
def area_eq_680 := (L * W = 680)
def fence_eq_178 := (2 * W + L = 178)

-- Theorem statement to prove the length of the uncovered side
theorem length_of_uncovered_side (h1 : area_eq_680 L W) (h2 : fence_eq_178 L W) : L = 170 := 
sorry

end length_of_uncovered_side_l512_512373


namespace ratio_of_areas_l512_512648

theorem ratio_of_areas (y : ℝ) (h₁ : y > 0) :
  let A_C := y^2,
      A_D := (4 * y)^2 in
  A_C / A_D = 1 / 16 := by
  sorry

end ratio_of_areas_l512_512648


namespace domain_of_sqrt_tan_sub_one_eq_l512_512279

noncomputable def domain_of_function : Set ℝ := 
  {x : ℝ | ∃ k : ℤ, (π / 4 + k * π) ≤ x ∧ x < (π / 2 + k * π)}

theorem domain_of_sqrt_tan_sub_one_eq : 
  (∀ x, (∃ x', x = x' → (∃ k : ℤ, (π / 4 + k * π) ≤ x' ∧ x' < (π / 2 + k * π)) 
  ↔ (∃ k : ℤ, (π / 4 + k * π) ≤ x ∧ x < (π / 2 + k * π)))) :=
by
  sorry

end domain_of_sqrt_tan_sub_one_eq_l512_512279


namespace range_of_b_for_no_common_points_l512_512555

def no_intersection (b : ℝ) : Prop :=
  ¬ (∃ x : ℝ, b = 2^x + 1) ∧ ¬ (∃ x : ℝ, b = - (2^x + 1))

theorem range_of_b_for_no_common_points :
  { b : ℝ | no_intersection b } = set.Ioo (-1) 1 :=
by
  sorry

end range_of_b_for_no_common_points_l512_512555


namespace find_f_of_f2_l512_512076

def f (x : ℝ) : ℝ :=
  if x ≥ 1 then x ^ 2 - 3 * x + 1 else (1 / 2) ^ x + 1 / 2

theorem find_f_of_f2 : f(f(2)) = 5 / 2 :=
by
  sorry

end find_f_of_f2_l512_512076


namespace monotonically_decreasing_interval_range_of_a_l512_512833

noncomputable def f (x : ℝ) : ℝ := (1 / 3) * x^3 - x^2 - 3 * x + 2

theorem monotonically_decreasing_interval :
  ∃ m n : ℝ, m < n ∧ ∀ x, m ≤ x ∧ x ≤ n → (f' x ≤ 0) :=
sorry

theorem range_of_a :
  ∀ (a : ℝ),  (∀ (x : ℝ), (-1 ≤ x ∧ x ≤ 3) → (2 * a - 3 ≤ x ∧ x ≤ a + 3)) ↔ (0 ≤ a ∧ a ≤ 1) :=
sorry

end monotonically_decreasing_interval_range_of_a_l512_512833


namespace solution_set_of_inequality_l512_512830

noncomputable def f (x : ℝ) : ℝ :=
if 0 < x ∧ x < 1 then -2 else if x ≥ 1 then 1 else 0

theorem solution_set_of_inequality :
    { x : ℝ | log 2 x - (log (1/4) (4 * x) - 1) * f (log 3 x + 1) ≤ 5 }
    = { x : ℝ | 1/3 < x ∧ x ≤ 4 } :=
by 
  sorry

end solution_set_of_inequality_l512_512830


namespace min_value_quadratic_expr_l512_512718

-- Define the quadratic function
def quadratic_expr (x : ℝ) : ℝ := 8 * x^2 - 24 * x + 1729

-- State the theorem to prove the minimum value
theorem min_value_quadratic_expr : (∃ x : ℝ, ∀ y : ℝ, quadratic_expr y ≥ quadratic_expr x) ∧ ∃ x : ℝ, quadratic_expr x = 1711 :=
by
  -- The proof will go here
  sorry

end min_value_quadratic_expr_l512_512718


namespace y_relationship_l512_512057

-- Define the inverse proportion function
def inverse_proportion_function (x : ℝ) : ℝ := -2 / x

-- Define the conditions for the points A, B, and C
variables (x₁ x₂ x₃ y₁ y₂ y₃ : ℝ)

-- Given x₁ < x₂ < 0 < x₃
axiom x1_lt_x2 : x₁ < x₂
axiom x2_lt_0 : x₂ < 0
axiom x3_gt_0 : 0 < x₃

-- Given the points A, B, and C lie on the inverse proportion function y = -2 / x
axiom A_on_curve : y₁ = inverse_proportion_function x₁
axiom B_on_curve : y₂ = inverse_proportion_function x₂
axiom C_on_curve : y₃ = inverse_proportion_function x₃

theorem y_relationship : y₃ < y₁ ∧ y₁ < y₂ :=
by
  sorry

end y_relationship_l512_512057


namespace part_1_part_2_l512_512462

noncomputable def S (n : ℕ) : ℚ
| 0     := 0
| (n+1) := S n + a (n+1)

noncomputable def a (n : ℕ) : ℚ :=
if n = 1 then 
  (1 / 2)
else 
  -(1 / (2 * n * (n-1)))

theorem part_1 (n : ℕ) (hn : n ≥ 2) : 
  ∃ d : ℚ, (∀ m : ℕ, m ≥ n → (1 / S m) - (1 / S (m - 1)) = d) :=
sorry

theorem part_2 (n : ℕ) : 
  a n = if n = 1 then 
    (1 / 2) 
  else 
    -(1 / (2 * n * (n - 1))) :=
sorry

end part_1_part_2_l512_512462


namespace area_of_triangle_TAD_l512_512183

noncomputable def triangle_area (A T D : Point) : ℝ := sorry

def area_pentagon (ABTCD: Polygon) : ℝ := 22

def equal_segments (AB CD : Segment) : Prop := AB.length = CD.length

def circumcircle_tangent (TAB TCD : Triangle) : Prop := sorry -- requires a complex geometric definition

def angle_ATD_90 (A T D : Point) : Prop := ∡ A T D = 90

def angle_BTC_120 (B T C : Point) : Prop := ∡ B T C = 120

def segment_BT_4 (B T : Point) : Prop := dist B T = 4

def segment_CT_5 (C T : Point) : Prop := dist C T = 5

theorem area_of_triangle_TAD (A B C D T : Point) 
(h1 : equal_segments (segment A B) (segment C D))
(h2 : circumcircle_tangent (triangle T A B) (triangle T C D))
(h3 : angle_ATD_90 A T D)
(h4 : angle_BTC_120 B T C)
(h5 : segment_BT_4 B T)
(h6 : segment_CT_5 C T)
(h7 : area_pentagon (polygon A B T C D) = 22) : 
triangle_area A T D = 64 * (2 - Real.sqrt 3) :=
sorry

end area_of_triangle_TAD_l512_512183


namespace determine_range_of_b_l512_512836

noncomputable def range_of_b (b : ℝ) : Prop :=
  let d := abs b / Real.sqrt 2 in
  1 < d ∧ d < 2

theorem determine_range_of_b (b : ℝ) :
  range_of_b b ↔ b < -Real.sqrt 2 ∨ Real.sqrt 2 < b :=
by
  sorry

end determine_range_of_b_l512_512836


namespace product_of_cosine_values_l512_512037

noncomputable theory
open_locale classical

def arithmetic_sequence (a d : ℝ) (n : ℕ) : ℝ := a + (n - 1) * d

theorem product_of_cosine_values (a b : ℝ) 
    (h : ∃ a1 : ℝ, ∀ n : ℕ+, ∃ a b : ℝ, 
         S = {cos (arithmetic_sequence a1 (2*π/3) n) | n ∈ ℕ*} ∧
         S = {a, b}) : a * b = -1/2 :=
begin
  sorry
end

end product_of_cosine_values_l512_512037


namespace ellipse_equation_l512_512495

theorem ellipse_equation (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h_inequality : a > b)
  (h_focus : a^2 - b^2 = 9) (h_point : (0:ℝ, -3) ∈ {p : ℝ × ℝ | p.1^2 / a^2 + p.2^2 / b^2 = 1}) :
  (a = real.sqrt 18 ∧ b = real.sqrt 9) ∧ ∀ x y : ℝ, x^2 / 18 + y^2 / 9 = 1 :=
sorry

end ellipse_equation_l512_512495


namespace photovoltaic_profit_problem_l512_512362

/-- A photovoltaic enterprise's net profit calculation and decision problem -/
theorem photovoltaic_profit_problem:
  ∀ (n : ℕ) (h1 : n ≥ 1) (h2 : 2 < n) (h3 : n < 18),

  -- Conditions
  let investment : ℝ := 144,
  let maintenance_cost : ℝ := (4 * n^2 + 20 * n),
  let annual_revenue : ℝ := 1,

  -- Net Profit Function
  let net_profit_fn (n: ℕ) : ℝ := 100 * n - (4 * n^2 + 20 * n) - 144,

  -- Determine if project yields a profit starting from the 3rd year onwards
  net_profit_fn n > 0 ->
  2 < n ∧ n < 18 ->       -- net profit will be positive in this range
  ∃ k : ℕ, k = 3,

  -- Maximize average annual profit
  let avg_annual_profit_fn (n : ℕ) : ℝ := net_profit_fn n / n,
  let max_average_annual_profit : ℝ := 32,
  let option1_profit := 6 * max_average_annual_profit + 72,
  option1_profit = 264,

  -- Maximize net profit
  let max_net_profit : ℝ := 256,
  let option2_profit := max_net_profit + 8,
  option2_profit = 264,

  -- Option 1 is more beneficial
  option1_profit = 264 /\ option2_profit = 264 ->
  ∃ (better_option : string),
    better_option = "option ①" :=
sorry

end photovoltaic_profit_problem_l512_512362


namespace smallest_n_for_modulo_eq_l512_512796

theorem smallest_n_for_modulo_eq :
  ∃ (n : ℕ), (3^n % 4 = n^3 % 4) ∧ (∀ m : ℕ, m < n → 3^m % 4 ≠ m^3 % 4) ∧ n = 7 :=
by
  sorry

end smallest_n_for_modulo_eq_l512_512796


namespace solve_equation_l512_512262

theorem solve_equation : ∀ (x : ℝ), x ≠ 1 → (3 / (x - 1) = 5 + 3 * x / (1 - x)) → x = 4 :=
by
  intros x hx heq
  -- sorry to skip the proof
  sorry

end solve_equation_l512_512262


namespace hyperbola_focal_length_l512_512401

-- Define a hyperbola with the given equation
def hyperbola (x y : ℝ) : Prop := (x^2 / 10) - (y^2 / 2) = 1

-- Define the semi-major axis and semi-minor axis.
def a_sq : ℝ := 10
def b_sq : ℝ := 2

-- Define the focal distance
def c := Real.sqrt (a_sq + b_sq)

-- Define the focal length
def focal_length := 2 * c

-- State the theorem
theorem hyperbola_focal_length : focal_length = 4 * Real.sqrt 3 := 
by 
  -- The proof is omitted as per the instructions
  sorry

end hyperbola_focal_length_l512_512401


namespace angle_half_size_l512_512168

noncomputable def convex_pentagon : Type := sorry
noncomputable def semicircle : convex_pentagon → Type := sorry
noncomputable def inscribed (p : convex_pentagon) : Prop := sorry
noncomputable def diameter (p : convex_pentagon) : Type := sorry
noncomputable def midpoint (a b : Type) : Type := sorry
noncomputable def perpendicular (p q : Type) : Prop := sorry
noncomputable def angle (a b : Type) : ℝ := sorry
noncomputable def acute_angle (a b : Type) : Type := sorry

variable {AXYZB : convex_pentagon}
variable {AB : Type}

theorem angle_half_size (h1 : inscribed AXYZB)
  (h2 : diameter AXYZB AB)
  (O : Type)
  (h3 : midpoint AB O)
  (P Q R S : Type)
  (h4 : perpendicular Y AX P)
  (h5 : perpendicular Y BX Q)
  (h6 : perpendicular Y AZ R)
  (h7 : perpendicular Y BZ S) :
  acute_angle PQ RS = angle XOZ / 2 := 
sorry

end angle_half_size_l512_512168


namespace triangle_formation_probability_l512_512546

/-- Given 3 segments chosen from the 5 given segments, 
can they form a triangle with the probability of \(\frac{3}{10}\)? -/
theorem triangle_formation_probability :
  let segments := [2, 4, 6, 8, 10]
  ∃ (valid_combinations : List (List ℕ)), 
    valid_combinations.length = 3 ∧
    ∀ (comb : List ℕ), comb ∈ valid_combinations → 
      comb.length = 3 ∧ 
      comb.sum > 2 * (comb.max') 
  → (valid_combinations.length : ℚ) / 10 = 3 / 10 :=
by
  sorry

end triangle_formation_probability_l512_512546


namespace angle_APD_correct_l512_512956

namespace ConvexQuadrilateral

variables {P : Type} [TopologicalSpace P] [NormedAddCommGroup P] [NormedSpace ℝ P]
variables (A B C D P : P)

-- Definitions for the angles in degrees
def angle_DAC : ℝ := 36
def angle_BDC : ℝ := 36
def angle_CBD : ℝ := 18
def angle_BAC : ℝ := 72

-- Angle measure context: assume these angles are correctly computed/converted
noncomputable def measure_angle_APD : ℝ := 108

-- Assumptions as conditions
axiom DAC_eq_36 : ∠DAC = angle_DAC
axiom BDC_eq_36 : ∠BDC = angle_BDC
axiom CBD_eq_18 : ∠CBD = angle_CBD
axiom BAC_eq_72 : ∠BAC = angle_BAC
axiom intersection_P : P = intersect AC BD

-- Goal
theorem angle_APD_correct : ∠APD = measure_angle_APD := by
  sorry

end ConvexQuadrilateral

end angle_APD_correct_l512_512956


namespace largest_angle_right_triangle_l512_512816

theorem largest_angle_right_triangle (u : ℝ) (h_pos : 0 < u) : 
  let a := Real.sqrt (8 * u - 4),
      b := Real.sqrt (8 * u + 4),
      c := 4 * Real.sqrt u in
  a^2 + b^2 = c^2 ∧ angle a b c = π / 2 :=
by
  simp [a, b, c]
  sorry

end largest_angle_right_triangle_l512_512816


namespace tangent_line_equation_l512_512812

def f (x : ℝ) : ℝ := x * Real.exp x

def tangent_line_at_neg_one : Prop :=
  ∃ (m b : ℝ), (∀ x, f x = m * x + b) ∧ b = -Real.exp (-1)

theorem tangent_line_equation :
  ∃ (m : ℝ), ∀ x : ℝ, f x = m * x + (-Real.exp (-1)) :=
begin
  sorry
end

end tangent_line_equation_l512_512812


namespace complex_conjugate_product_l512_512981

namespace ComplexProblem

open Complex

theorem complex_conjugate_product (z : ℂ) (hz : z = ⟨0, 1⟩ * (1 - ⟨0, 1⟩)) : z * conj(z) = 2 := by
  sorry

end ComplexProblem

end complex_conjugate_product_l512_512981


namespace number_of_palindromes_on_24_hour_clock_l512_512744

-- Define the concept of a palindrome in the context of a 24-hour digital clock.
def is_palindrome (s : String) : Prop :=
  s = s.reverse

-- Define the format of the time (hh:mm) and specify the range for hours and minutes.
def valid_time (hour min : ℕ) : Prop :=
  (0 ≤ hour ∧ hour < 24) ∧ (0 ≤ min ∧ min < 60)

-- Count all valid palindromes given the time constraints in a 24-hour clock.
def count_palindromes : ℕ :=
  (List.range 24).sum (λ hour =>
    (List.range 60).countp (λ min =>
      is_palindrome (hour.repr ++ ":" ++ if min < 10 then "0" ++ min.repr else min.repr)))

theorem number_of_palindromes_on_24_hour_clock : count_palindromes = 64 := 
by
  -- Skip the proof; the theorem states that the number of palindromes is 64
  sorry

end number_of_palindromes_on_24_hour_clock_l512_512744


namespace equivalent_ann_rate_to_nearest_hundredth_l512_512269

-- Definitions: interest compounding quarterly and equivalent annual rate
def quarterly_rate (annual_rate : ℝ) : ℝ := annual_rate / 4
def equivalent_annual_rate (quarterly_rate : ℝ) : ℝ := (1 + quarterly_rate / 100) ^ 4 - 1

-- The main theorem statement: proving the equivalent annual interest rate
theorem equivalent_ann_rate_to_nearest_hundredth :
  let annual_rate := 8 in
  let q_rate := quarterly_rate annual_rate in
  let r := equivalent_annual_rate q_rate in
  Float.ofScientific 0 2 "* 100 <= 0.08243216 * 100 and 0.08243216 * 100 <= 2 * f_ofInt 0 ≤ 4 * 8 point_f=2 
  round (r * 100) = 8.24 := 
begin
  -- introduction of rate variables (step 1)
  let annual_rate := 8,
  let q_rate := quarterly_rate annual_rate,

  -- calculate quarterly rate (step 1)
  have q_rate_eq : quarterly_rate annual_rate = 2 := by
    unfold quarterly_rate,
    norm_num,

  -- calculate equivalent annual rate (step 2, 3)
  let r : ℝ := equivalent_annual_rate q_rate,
  have r_def : equivalent_annual_rate q_rate = (1 + 2 / 100) ^ 4 - 1 := by
    unfold equivalent_annual_rate,

  -- calculate the resulting annual rate (step 3)
  have r_eval : r = 1.08243216 - 1 := by
    rw [r_def],
    norm_num,

  -- convert to percentage and round (step 4)
  have r_percent : r * 100 = 8.243216 := 
    by unfold r_eval,

  have r_rounded : round (r * 100) = 8.24 := by
    norm_num,

  show round (r * 100) = 8.24 from r_rounded,
    sorry
)

end equivalent_ann_rate_to_nearest_hundredth_l512_512269


namespace simplify_expression_l512_512641

variable (a b : Real)

theorem simplify_expression (a b : Real) : 
    3 * b * (3 * b ^ 2 + 2 * b) - b ^ 2 + 2 * a * (2 * a ^ 2 - 3 * a) - 4 * a * b = 
    9 * b ^ 3 + 5 * b ^ 2 + 4 * a ^ 3 - 6 * a ^ 2 - 4 * a * b := by
  sorry

end simplify_expression_l512_512641


namespace quad_roots_sum_l512_512120

theorem quad_roots_sum {x₁ x₂ : ℝ} (h1 : x₁ + x₂ = 5) (h2 : x₁ * x₂ = -6) :
  1 / x₁ + 1 / x₂ = -5 / 6 :=
by
  sorry

end quad_roots_sum_l512_512120


namespace max_dist_vec_sum_l512_512867

open EuclideanGeometry

variables {A B C P : Point}
variables (ABC : Triangle A B C)

def is_right_triangle (T : Triangle A B C) : Prop :=
  ∃ (α : ℝ), α = 90 ∧ angle A B C = α

def is_isosceles_right_triangle (T : Triangle A B C) : Prop :=
  is_right_triangle T ∧ dist A B = 1 ∧ dist A C = 1

def point_on_segment (P : Point) (B C : Point) : Prop :=
  ∃ (t : ℝ), 0 ≤ t ∧ t ≤ 1 ∧ P = (1 - t) • B + t • C

def vec_PB (P B : Point) : Vector := P -ᵥ B
def vec_PC (P C : Point) : Vector := P -ᵥ C

def dist_vec_sum (P B C : Point) : ℝ :=
  ∥vec_PB P B + 2 • vec_PC P C∥

theorem max_dist_vec_sum (h1 : is_isosceles_right_triangle ABC)
                           (h2 : point_on_segment P B C) :
  ∃ (M : ℝ), M = 2 * sqrt 2 ∧ ∀ (Q : Point), point_on_segment Q B C →
    dist_vec_sum Q B C ≤ M :=
sorry

end max_dist_vec_sum_l512_512867


namespace ram_balance_speed_l512_512635

theorem ram_balance_speed
  (part_speed : ℝ)
  (balance_distance : ℝ)
  (total_distance : ℝ)
  (total_time : ℝ)
  (part_time : ℝ)
  (balance_speed : ℝ)
  (h1 : part_speed = 20)
  (h2 : total_distance = 400)
  (h3 : total_time = 8)
  (h4 : part_time = 3.2)
  (h5 : balance_distance = total_distance - part_speed * part_time)
  (h6 : balance_speed = balance_distance / (total_time - part_time)) :
  balance_speed = 70 :=
by
  simp [h1, h2, h3, h4, h5, h6]
  sorry

end ram_balance_speed_l512_512635


namespace sample_standard_deviation_same_sample_ranges_same_l512_512863

variables {n : ℕ} (x y : Fin n → ℝ) (c : ℝ)
  (h_y : ∀ i, y i = x i + c)
  (h_c_ne_zero : c ≠ 0)

-- Statement for standard deviations being the same
theorem sample_standard_deviation_same :
  let mean (s : Fin n → ℝ) := (∑ i, s i) / n
  in let stddev (s : Fin n → ℝ) := sqrt ((∑ i, (s i - mean s) ^ 2) / n)
  in stddev x = stddev y := 
sorry

-- Statement for ranges being the same
theorem sample_ranges_same :
  let range (s : Fin n → ℝ) := (Finset.univ.sup s) - (Finset.univ.inf s)
  in range x = range y :=
sorry

end sample_standard_deviation_same_sample_ranges_same_l512_512863


namespace existence_of_schedule_for_n_3_n_is_odd_l512_512308

def summer_camp (n : ℕ) : Prop :=
  ∃ (schedule : list (fin (3 * n) × fin (3 * n) × fin (3 * n))),
    (∀ (i j : fin (3 * n)), 
      (i ≠ j) → (∃ (day : fin (#schedule)), 
        (i, j) ∈ {(x.1, x.2) | x ∈ (schedule.nth day).to_list} ∨
        (i, j) ∈ {(x.1, x.3) | x ∈ (schedule.nth day).to_list} ∨
        (i, j) ∈ {(x.2, x.3) | x ∈ (schedule.nth day).to_list})
    )

theorem existence_of_schedule_for_n_3 : summer_camp 3 :=
by
  sorry

theorem n_is_odd (n : ℕ) (h : summer_camp n) : n % 2 = 1 :=
by
  sorry

end existence_of_schedule_for_n_3_n_is_odd_l512_512308


namespace molecular_weight_CaH2_correct_l512_512719

-- Define the atomic weights
def atomic_weight_Ca : ℝ := 40.08
def atomic_weight_H : ℝ := 1.008

-- Define the formula to compute the molecular weight
def molecular_weight_CaH2 (atomic_weight_Ca : ℝ) (atomic_weight_H : ℝ) : ℝ :=
  (1 * atomic_weight_Ca) + (2 * atomic_weight_H)

-- Theorem stating that the molecular weight of CaH2 is 42.096 g/mol
theorem molecular_weight_CaH2_correct : molecular_weight_CaH2 atomic_weight_Ca atomic_weight_H = 42.096 := 
by 
  sorry

end molecular_weight_CaH2_correct_l512_512719


namespace ordering_of_f_values_l512_512888

theorem ordering_of_f_values
  {f : ℝ → ℝ}
  (hf_even : ∀ x, f x = f (-x))
  (hf_periodic : ∀ x, f (x + 1) = 1 / f x)
  (hf_decreasing : ∀ x y, -1 ≤ x ∧ x < y ∧ y ≤ 0 → f y < f x) :
  let a := f (Real.log 2 / Real.log 0.5)
  let b := f (Real.log 4 / Real.log 2)
  let c := f (Real.sqrt 2)
  in a > c ∧ c > b := 
sorry

end ordering_of_f_values_l512_512888


namespace asymptote_hyperbola_l512_512906

theorem asymptote_hyperbola (m : ℝ) (x y : ℝ) :
  (abs m > 0) → (
  (x^2) / (abs m) - (y^2) / (abs m + 3) = 1) →
  (2 * sqrt 5)^2 = (abs m) + (abs m + 3) →
  y = 2 * x :=
by
  intros
  sorry

end asymptote_hyperbola_l512_512906


namespace eight_pow_three_eq_two_pow_nine_l512_512919

theorem eight_pow_three_eq_two_pow_nine : 8^3 = 2^9 := by
  sorry -- Proof is skipped

end eight_pow_three_eq_two_pow_nine_l512_512919


namespace sum_of_first_seven_distinct_positive_integer_multiples_of_5_that_are_squares_is_3500_l512_512720

theorem sum_of_first_seven_distinct_positive_integer_multiples_of_5_that_are_squares_is_3500 :
  let squares := {n : ℕ | ∃ k : ℕ, n = k^2 ∧ n % 5 = 0} in
  let first_seven := [25, 100, 225, 400, 625, 900, 1225] in
  (∀ n, n ∈ first_seven → n ∈ squares) →
  (List.sum first_seven = 3500) :=
by
  sorry

end sum_of_first_seven_distinct_positive_integer_multiples_of_5_that_are_squares_is_3500_l512_512720


namespace no_such_function_exists_l512_512609

noncomputable theory

open Classical

theorem no_such_function_exists :
  ∀ (f : ℝ → ℝ),
    (∃ M : ℝ, M > 0 ∧ ∀ x : ℝ, -M ≤ f x ∧ f x ≤ M) →
    f 1 = 1 →
    (∀ x : ℝ, x ≠ 0 → f (x + 1 / x^2) = f x + (f (1 / x))^2) →
    False :=
by
  intro f h1 h2 h3
  sorry

end no_such_function_exists_l512_512609


namespace Amanda_income_if_report_not_finished_l512_512158

def hourly_pay : ℝ := 50.0
def hours_worked_per_day : ℝ := 10.0
def percent_withheld : ℝ := 0.2

theorem Amanda_income_if_report_not_finished : 
  let total_daily_income := hourly_pay * hours_worked_per_day in
  let amount_withheld := percent_withheld * total_daily_income in
  let amount_received := total_daily_income - amount_withheld in
  amount_received = 400 :=
by
  sorry

end Amanda_income_if_report_not_finished_l512_512158


namespace intersection_proof_l512_512810

-- Definitions based on conditions
def circle1 (x y : ℝ) : Prop := (x - 2) ^ 2 + (y - 10) ^ 2 = 50
def circle2 (x y : ℝ) : Prop := x ^ 2 + y ^ 2 + 2 * (x - y) - 18 = 0

-- Correct answer tuple
def intersection_points : (ℝ × ℝ) × (ℝ × ℝ) := ((3, 3), (-3, 5))

-- The goal statement to prove
theorem intersection_proof :
  (circle1 3 3 ∧ circle2 3 3) ∧ (circle1 (-3) 5 ∧ circle2 (-3) 5) :=
by
  sorry

end intersection_proof_l512_512810


namespace rectangle_inscribed_circle_circumference_l512_512758

theorem rectangle_inscribed_circle_circumference :
  ∀ (a b : ℝ), (a = 9) ∧ (b = 12) → (∀ d, d = Real.sqrt (a^2 + b^2) ↔ d = 15) →
  ∀ C, C = Real.pi * 15 ↔ C = 15 * Real.pi :=
by
  intros a b h1 h2 C
  cases h1 with ha hb
  rw [ha, hb] at h2
  have d_eq : d = 15 := by 
    rw h2
    sorry
  rw [h2, d_eq, mul_comm]
  exact Iff.rfl

end rectangle_inscribed_circle_circumference_l512_512758


namespace div_fraction_fraction_division_eq_l512_512709

theorem div_fraction (a b : ℕ) (h : b ≠ 0) : (a : ℚ) / b = (a : ℚ) * (1 / (b : ℚ)) := 
by sorry

theorem fraction_division_eq : (3 : ℚ) / 7 / 4 = 3 / 28 := 
by 
  calc
    (3 : ℚ) / 7 / 4 = (3 / 7) * (1 / 4) : by rw [div_fraction] 
                ... = 3 / 28            : by normalization_tactic -- Use appropriate tactic for simplification
                ... = 3 / 28            : by rfl

end div_fraction_fraction_division_eq_l512_512709


namespace cucumbers_left_over_l512_512096

theorem cucumbers_left_over :
  ∀ (boxes cucumbers_per_box rotten cucumbers_per_bag bags : ℕ),
  boxes = 7 →
  cucumbers_per_box = 16 →
  rotten = 13 →
  bags = 8 →
  cucumbers_per_bag = (boxes * cucumbers_per_box - rotten) / bags →
  (boxes * cucumbers_per_box - rotten) % bags = 3 :=
by
  intros boxes cucumbers_per_box rotten cucumbers_per_bag bags
  intros h_boxes h_cucumbers_per_box h_rotten h_bags h_cucumbers_per_bag
  rw [h_boxes, h_cucumbers_per_box, h_rotten, h_bags]
  sorry

end cucumbers_left_over_l512_512096


namespace find_S_l512_512975

section
variable (f : ℕ → ℕ) 
variable (h : ∀ n, f n = (n-1) * f (n-1))
variable (h₁ : f 1 ≠ 0)

theorem find_S : S = 72 :=
by sorry

def S := (f 11) / ((11-1) * (f (11-3)))

#eval S
end

end find_S_l512_512975


namespace ratio_area_triangle_circumscribed_circle_l512_512634

-- Given triangle ABC with sides a, b, c and angles α, β, γ
-- and R as the radius of the circumscribed circle

variables {a b c R : ℝ} {α β γ : ℝ}

-- Conditions
def area_triangle (b c sinα : ℝ) : ℝ := 1 / 2 * b * c * sinα

def sine_rule (a b c sinα sinβ sinγ : ℝ) (R : ℝ) : Prop :=
  a / sinα = 2 * R ∧ b / sinβ = 2 * R ∧ c / sinγ = 2 * R

-- Statement to prove
theorem ratio_area_triangle_circumscribed_circle
  (b c : ℝ) (α : ℝ) (sinα sinβ sinγ : ℝ) (R : ℝ)
  (h1 : sine_rule (2 * R * sinα) b c sinα sinβ sinγ R) :
  (area_triangle b c sinα) / (π * R^2) < 2 / 3 :=
sorry

end ratio_area_triangle_circumscribed_circle_l512_512634


namespace hyperbola_equation_l512_512885

theorem hyperbola_equation :
  ∀ (c b a : ℝ), 
  (c = 2) →
  (b = real.sqrt 3) →
  (c^2 = a^2 + b^2) →
  (a = 1) →
  (∀ x y : ℝ, x^2 - (y^2 / 3) = 1) := 
by
  intros c b a hc hb h_rel ha x y
  sorry

end hyperbola_equation_l512_512885


namespace area_of_triangle_is_2_sqrt_3_l512_512870

theorem area_of_triangle_is_2_sqrt_3
  (a b : ℝ) (a_pos : a > 0) (b_pos : b > 0) (h_ab : a > b)
  (e : ℝ) (h_e : e = Real.sqrt 7 / 3)
  (F A B O : EuclideanSpace ℝ (Fin 2))
  (AF BF : ℝ) (h_AF : AF = 2) (h_BF : BF = 4)
  (ellipse_eq : ∀ (x y : ℝ), (x^2 / a^2) + (y^2 / b^2) = 1)
  (line_through_O : Line ℝ) (h_line : line_through_O.Contains O)
  (intersects_at : line_through_O.Intersects C A ∧ line_through_O.Intersects C B) :
  Area (Triangle.mk A F B) = 2 * Real.sqrt 3 :=
sorry

end area_of_triangle_is_2_sqrt_3_l512_512870


namespace line_y_intercept_l512_512890

theorem line_y_intercept (t : ℝ) (h : ∃ (t : ℝ), ∀ (x y : ℝ), x - 2 * y + t = 0 → (x = 2 ∧ y = -1)) :
  ∃ y : ℝ, (0 - 2 * y + t = 0) ∧ y = -2 :=
by
  sorry

end line_y_intercept_l512_512890


namespace geometric_sequence_common_ratio_l512_512365

theorem geometric_sequence_common_ratio :
  ∃ r : ℚ, 
    (r = (-48 : ℚ) / 32) ∧
    (r = 72 / (-48 : ℚ)) ∧
    (r = -3 / 2) :=
by
  use (-3 / 2)
  split
  {
    calc (-48 : ℚ) / (32 : ℚ) = -3 / 2 : by norm_num
  }
  split
  {
    calc (72 : ℚ) / (-48 : ℚ) = -3 / 2 : by norm_num
  }
  {
    exact rfl
  }

end geometric_sequence_common_ratio_l512_512365


namespace percent_of_y_l512_512323

theorem percent_of_y (y : ℝ) : 0.30 * (0.80 * y) = 0.24 * y :=
by sorry

end percent_of_y_l512_512323


namespace measure_angle_A_value_b_sin_B_div_c_l512_512122

variables {a b c : ℝ} (A B C : ℝ)
-- Conditions
noncomputable def is_geom_progression (a b c : ℝ) : Prop :=
  b^2 = a * c

noncomputable def given_equation (a b c : ℝ) : Prop :=
  a^2 - c^2 = a * c - b * c

-- Prove that given the conditions, ∠A = 60°
theorem measure_angle_A (h1 : is_geom_progression a b c) (h2 : given_equation a b c) : A = 60 :=
by
  sorry

-- Prove that given the conditions, the value of (b * sin B) / c is √3/2
theorem value_b_sin_B_div_c (h1 : is_geom_progression a b c) (h2 : given_equation a b c) (h3 : A = 60) : (b * sin B) / c = sqrt 3 / 2 :=
by
  sorry

end measure_angle_A_value_b_sin_B_div_c_l512_512122


namespace regina_earnings_l512_512212

-- Definitions based on conditions
def num_cows := 20
def num_pigs := 4 * num_cows
def price_per_pig := 400
def price_per_cow := 800

-- Total earnings calculation based on definitions
def earnings_from_cows := num_cows * price_per_cow
def earnings_from_pigs := num_pigs * price_per_pig
def total_earnings := earnings_from_cows + earnings_from_pigs

-- Proof statement
theorem regina_earnings : total_earnings = 48000 := by
  sorry

end regina_earnings_l512_512212


namespace sqrt_n_is_D_sequence_two_pow_n_is_not_D_sequence_exists_bn_less_than_one_exists_sum_greater_than_2024_l512_512556

open Nat

-- Definition of a D sequence
def is_D_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) ^ 2 - a n ^ 2 = d

-- Problem 1
theorem sqrt_n_is_D_sequence : 
  is_D_sequence (λ n, real.sqrt (n : ℝ)) 1 := 
sorry

-- Problem 2
theorem two_pow_n_is_not_D_sequence : 
  ¬ is_D_sequence (λ n, 2 ^ (n : ℝ)) ∃ d ≠ 0 := 
sorry

-- Given a D sequence, show there exists an element in {bn} that is less than 1
theorem exists_bn_less_than_one (a : ℕ → ℝ) (d : ℝ) (h : is_D_sequence a d) :
  ∃ n : ℕ, (a (n + 1) - a n) < 1 := 
sorry

-- Given a D sequence, show there exists a positive integer n such that the sum exceeds 2024
theorem exists_sum_greater_than_2024 (a : ℕ → ℝ) (d : ℝ) (h : is_D_sequence a d) :
  ∃ n : ℕ, (∑ i in range (n + 1), 1 / (a i)) > 2024 := 
sorry

end sqrt_n_is_D_sequence_two_pow_n_is_not_D_sequence_exists_bn_less_than_one_exists_sum_greater_than_2024_l512_512556


namespace solve_fractional_equation_l512_512228

theorem solve_fractional_equation (x : ℝ) (hx : x ≠ 1 ∧ x ≠ 1) : 3 / (x - 1) = 5 + 3 * x / (1 - x) ↔ x = 4 :=
by 
  split
  -- Forward direction (Given the equation, show x = 4)
  intro h
  -- Multiply out and simplify as shown in the solution steps
  -- Sorry statement skips the detailed proof
  sorry
  -- Converse direction (Substitute x = 4 and verify it satisfies the equation)
  intro hx
  -- Substitution and verification steps
  -- Sorry statement skips the detailed proof
  sorry

end solve_fractional_equation_l512_512228


namespace T_2016_plus_fraction_l512_512884

-- Definitions of sequences and conditions.

def a_n (n : ℕ) : ℤ := n - 2

def b_n (n : ℕ) : ℝ := 
  let cos_term := (a_n n + 2) * Real.cos ((a_n n + 2) * Real.pi / 2)
  let frac_term := 1 / (a_n (2 * n - 1) * a_n (2 * n + 1))
  cos_term + frac_term

def T_n (n : ℕ) : ℝ :=
  b_n n + T_n (n - 1)

-- Provided conditions.
def S_2 : ℤ := -1
def S_5 : ℤ := 5

-- Main statement to prove.
theorem T_2016_plus_fraction : T_2016 + (2016 / 4031) = 1008 := sorry

end T_2016_plus_fraction_l512_512884


namespace right_triangle_legs_l512_512668

noncomputable def triangle_legs_length (a b : ℝ) : ℝ × ℝ :=
(sqrt(a * (a + b)), sqrt(b * (a + b)))

theorem right_triangle_legs (a b : ℝ) (h : a > 0 ∧ b > 0) :
  ∃ AC BC : ℝ, AC = sqrt(a * (a + b)) ∧ BC = sqrt(b * (a + b)) :=
begin
  use (sqrt(a * (a + b)), sqrt(b * (a + b))),
  split;
  refl,
end

end right_triangle_legs_l512_512668


namespace sum_of_roots_tan_equation_l512_512444

theorem sum_of_roots_tan_equation : 
  let f : ℝ → ℝ := λ x, tan x in
  (∀ x ∈ set.Icc 0 (2 * real.pi), f^2 x - 8 * f x + real.sqrt 2 = 0) → (sum (roots f 0 2*π) = 4 * real.pi) := sorry

end sum_of_roots_tan_equation_l512_512444


namespace find_b_l512_512491

noncomputable def f (x : ℝ) : ℝ := x^2 - 2 * x + 2

theorem find_b :
  ∃ b : ℝ, (∀ x ∈ set.Icc (1 : ℝ) b, f x ∈ set.Icc (1 : ℝ) b) → b = 2 :=
by {
  sorry
}

end find_b_l512_512491


namespace value_of_a8_l512_512141

def sequence (a : ℕ → ℤ) : Prop :=
  a 1 = -1 ∧ ∀ n, a (n + 1) = a n - 3

theorem value_of_a8 :
  ∃ a : ℕ → ℤ, sequence a ∧ a 8 = -22 :=
begin
  -- proof goes here
  sorry
end

end value_of_a8_l512_512141


namespace interval_of_increase_monotone_increasing_monotonically_increasing_decreasing_l512_512831

noncomputable def f (x a : ℝ) : ℝ := Real.exp x - a * x - 1

theorem interval_of_increase (a : ℝ) : 
  (∀ x : ℝ, 0 < a → (Real.exp x - a ≥ 0 ↔ x ≥ Real.log a)) ∧ 
  (∀ x : ℝ, a ≤ 0 → (Real.exp x - a ≥ 0)) :=
by sorry

theorem monotone_increasing (a : ℝ) (h : ∀ x : ℝ, Real.exp x - a ≥ 0) : 
  a ≤ 0 :=
by sorry

theorem monotonically_increasing_decreasing : 
  ∃ a : ℝ, (∀ x ≤ 0, Real.exp x - a ≤ 0) ∧ 
           (∀ x ≥ 0, Real.exp x - a ≥ 0) ↔ a = 1 :=
by sorry

end interval_of_increase_monotone_increasing_monotonically_increasing_decreasing_l512_512831


namespace only_coprime_n_is_one_l512_512344

def isCoprime (a b : ℕ) : Prop := gcd a b = 1

def specialForm (m : ℕ) : ℕ := 2 ^ m + 3 ^ m + 6 ^ m - 1

theorem only_coprime_n_is_one :
  ∀ n ∈ ℕ, (∀ m ∈ ℕ, isCoprime n (specialForm m)) → n = 1 :=
by
  intro n h1,
  sorry

end only_coprime_n_is_one_l512_512344


namespace download_time_correct_l512_512990

-- Define the given conditions
def total_size : ℕ := 880
def downloaded : ℕ := 310
def speed : ℕ := 3

-- Calculate the remaining time to download
def time_remaining : ℕ := (total_size - downloaded) / speed

-- Theorem statement that needs to be proven
theorem download_time_correct : time_remaining = 190 := by
  -- Proof goes here
  sorry

end download_time_correct_l512_512990


namespace evaluate_f_f_one_fourth_l512_512898

def f (x : ℝ) : ℝ :=
if x > 0 then log x / log 2 else 3 ^ x

theorem evaluate_f_f_one_fourth :
  f (f (1/4)) = 1/9 :=
by
  sorry

end evaluate_f_f_one_fourth_l512_512898


namespace number_of_common_tangents_l512_512426

theorem number_of_common_tangents (x y : ℝ) :
  let circle1 := ∀ x y : ℝ, x^2 + y^2 = 9 in
  let circle2 := ∀ x y : ℝ, x^2 + y^2 - 8x + 6y + 9 = 0 in
  let center1 := (0, 0) in
  let radius1 := 3 in
  let center2 := (4, -3) in
  let radius2 := 4 in
  |(center2.1 - center1.1)^2 + (center2.2 - center1.2)^2| = 5 in
  1 + ∣radius1 - radius2∣ < 5 ∧ 5 < radius1 + radius2 :=
  2 :=
sorry

end number_of_common_tangents_l512_512426


namespace reflection_matrix_over_vector_l512_512439

theorem reflection_matrix_over_vector :
  let v := (Matrix 2 1 ℚ) (fun i j => if (i, j) = (0, 0) then 5 else if (i, j) = (1, 0) then 1 else 0)
  in (Matrix 2 2 ℚ) (fun i j => 
    match (i, j) with
    | (0, 0) => 12/13
    | (0, 1) => 5/13
    | (1, 0) => 5/13
    | (1, 1) => -12/13) = 
  let a := 5
  let b := 1
  let norm_sq := (a * a + b * b)
  let p := fun x y => (a * x + b * y) / norm_sq
  let matrix_elem i j := match (i, j) with
      | (0, 0) => 2 * a * a / norm_sq - 1
      | (0, 1) => 2 * a * b / norm_sq
      | (1, 0) => 2 * a * b / norm_sq
      | (1, 1) => 2 * b * b / norm_sq - 1
  in Matrix 2 2 ℚ matrix_elem := sorry

end reflection_matrix_over_vector_l512_512439


namespace necessary_conditions_l512_512425

theorem necessary_conditions (x y : ℝ) (h : |x - real.log y / real.log 3| = x + real.log y / real.log 3) : x * (y - 1) = 0 :=
sorry

end necessary_conditions_l512_512425


namespace incorrect_option_B_l512_512542

variable {α : Type*} [InnerProductSpace ℝ α]

-- Definitions based on the conditions
def vec_norm (v : α) : ℝ := ∥v∥
def dot_product (v w : α) : ℝ := ⟪v, w⟫
def scalar_mul (λ : ℝ) (v : α) : α := λ • v

-- Incorrectness assertion
theorem incorrect_option_B (a b : α) : ¬(abs (dot_product a b) = vec_norm a * vec_norm b) := 
by sorry

end incorrect_option_B_l512_512542


namespace cubic_difference_l512_512109

theorem cubic_difference (x : ℝ) (h : (x + 16) ^ (1/3) - (x - 16) ^ (1/3) = 4) : 
  235 < x^2 ∧ x^2 < 240 := 
sorry

end cubic_difference_l512_512109


namespace central_angle_of_cone_development_diagram_l512_512300

-- Given conditions: radius of the base of the cone and slant height
def radius_base := 1
def slant_height := 3

-- Target theorem: prove the central angle of the lateral surface development diagram is 120 degrees
theorem central_angle_of_cone_development_diagram : 
  ∃ n : ℝ, (2 * π) = (n * π * slant_height) / 180 ∧ n = 120 :=
by
  use 120
  sorry

end central_angle_of_cone_development_diagram_l512_512300


namespace correct_option_is_B_l512_512332

def satisfy_triangle_inequality (a b c : ℕ) : Prop :=
  (a + b > c) ∧ (a + c > b) ∧ (b + c > a)

theorem correct_option_is_B :
  satisfy_triangle_inequality 3 4 5 ∧
  ¬ satisfy_triangle_inequality 1 1 2 ∧
  ¬ satisfy_triangle_inequality 1 4 6 ∧
  ¬ satisfy_triangle_inequality 2 3 7 :=
by
  sorry

end correct_option_is_B_l512_512332


namespace angle_between_vectors_l512_512121

variables (a b : ℝ^3)
def magnitude (v : ℝ^3) : ℝ := real.sqrt (v.dot v)

theorem angle_between_vectors 
  (h1 : magnitude a = real.sqrt 2)
  (h2 : magnitude b = 2)
  (h3 : a.dot (a - b) = 0) :
  real.arccos (a.dot b / (magnitude a * magnitude b)) = π / 4 :=
sorry

end angle_between_vectors_l512_512121


namespace solve_fractional_equation_l512_512245

theorem solve_fractional_equation (x : ℝ) (h1 : x ≠ 1) (h2 : x ≠ -1) : 
    (3 / (x - 1) = 5 + 3 * x / (1 - x)) → x = 4 := 
by
  sorry

end solve_fractional_equation_l512_512245


namespace volume_of_tetrahedron_l512_512395

theorem volume_of_tetrahedron 
  (A B C D E : ℝ)
  (AB AD AE: ℝ)
  (h_AB : AB = 3)
  (h_AD : AD = 4)
  (h_AE : AE = 1)
  (V : ℝ) :
  (V = (4 * Real.sqrt 3) / 3) :=
sorry

end volume_of_tetrahedron_l512_512395


namespace find_asymptote_slope_l512_512670

theorem find_asymptote_slope (x y : ℝ) (h : (y^2) / 9 - (x^2) / 4 = 1) : y = 3 / 2 * x :=
sorry

end find_asymptote_slope_l512_512670


namespace sum_of_two_squares_l512_512160

theorem sum_of_two_squares (u : ℤ) (hu : u % 2 = 1) : ∃ a b : ℤ, (a^2 + b^2 = (3^u + 3^(2 * u) + 1)) := sorry

end sum_of_two_squares_l512_512160


namespace measure_of_angle_C_in_triangle_l512_512560

theorem measure_of_angle_C_in_triangle
  (a b c : ℝ)
  (A B C : ℝ)
  (p q : ℝ × ℝ)
  (h1 : p = (1, -real.sqrt 3))
  (h2 : q = (real.cos B, real.sin B))
  (h3 : p.1 * q.2 - p.2 * q.1 = 0)
  (h4 : b * real.cos C + c * real.cos B = 2 * a * real.sin A)
  (h5 : A + B + C = real.pi) :
  C = real.pi / 6 :=
by sorry

end measure_of_angle_C_in_triangle_l512_512560


namespace initial_number_of_quarters_l512_512750

theorem initial_number_of_quarters 
  (pennies : ℕ) (nickels : ℕ) (dimes : ℕ) (half_dollars : ℕ) (dollar_coins : ℕ) 
  (two_dollar_coins : ℕ) (quarters : ℕ)
  (cost_per_sundae : ℝ) 
  (special_topping_cost : ℝ)
  (featured_flavor_discount : ℝ)
  (members_with_special_topping : ℕ)
  (members_with_featured_flavor : ℕ)
  (left_over : ℝ)
  (expected_quarters : ℕ) :
  pennies = 123 ∧
  nickels = 85 ∧
  dimes = 35 ∧
  half_dollars = 15 ∧
  dollar_coins = 5 ∧
  quarters = expected_quarters ∧
  two_dollar_coins = 4 ∧
  cost_per_sundae = 5.25 ∧
  special_topping_cost = 0.50 ∧
  featured_flavor_discount = 0.25 ∧
  members_with_special_topping = 3 ∧
  members_with_featured_flavor = 5 ∧
  left_over = 0.97 →
  expected_quarters = 54 :=
  by
  sorry

end initial_number_of_quarters_l512_512750


namespace right_triangle_cos_sin_relation_l512_512005

theorem right_triangle_cos_sin_relation 
  (A B C : ℝ) 
  (h : ∠C = π / 2) 
  (h1 : cos A ^ 2 + cos B ^ 2 + 2 * sin A * sin B * cos C = 3 / 2) 
  (h2 : cos B ^ 2 + 2 * sin B * cos A = 5 / 3)
  (h3 : cos C = 0) : 
  cos A ^ 2 + 2 * sin A * cos B = 4 / 3 := 
by 
  sorry

end right_triangle_cos_sin_relation_l512_512005


namespace polar_coordinates_l512_512886

def cartesian_to_polar (x y : ℝ) : ℝ × ℝ :=
  let ρ := real.sqrt (x^2 + y^2)
  let θ := real.atan2 y x
  (ρ, θ)

theorem polar_coordinates (P : ℂ) (x y : ℝ) (hP : P = complex.mk x y)
  (h_coord : (x, y) = (-3, 3))
  : cartesian_to_polar x y = (3 * real.sqrt 2, 3 / 4 * real.pi) :=
by {
  -- The proof will be here
  sorry
}

end polar_coordinates_l512_512886


namespace extreme_point_inequality_l512_512511

noncomputable def f (x : ℝ) (a : ℝ) : ℝ :=
  x - a / x - 2 * Real.log x

theorem extreme_point_inequality (x₁ x₂ a : ℝ) (h1 : x₁ < x₂) (h2 : f x₁ a = 0) (h3 : f x₂ a = 0) 
(h_a_range : 0 < a) (h_a_lt_1 : a < 1) :
  f x₂ a < x₂ - 1 :=
sorry

end extreme_point_inequality_l512_512511


namespace number_of_valid_functions_l512_512441

theorem number_of_valid_functions : 
  ∃ (functions : Finset (Polynomial ℚ)), 
  (∀ (g : Polynomial ℚ), g ∈ functions → 
    ∃ (a b c d : ℚ), 
      (g = a * X ^ 3 + b * X ^ 2 + c * X + d) ∧ 
      (g * (Polynomial.eval (ScalarRing.equiv_neg) g) = Polynomial.eval (⇑(Polynomial.map_degree_type 3) g)) ∧ 
      (a = 0 ∨ a = 1) ∧ (b = 0 ∨ b = 1) ∧ (c = 0 ∨ c = 1) ∧ (d = 0 ∨ d = 1)) ∧
  card functions = 4 :=
begin
  sorry -- Proof goes here
end

end number_of_valid_functions_l512_512441


namespace matching_shoes_probability_l512_512746

theorem matching_shoes_probability : 
  let total_pairs := 9
  let total_shoes := 2 * total_pairs
  let matching_pairs := total_pairs
  let total_selections := @Nat.choose total_shoes 2
  let probability := matching_pairs / total_selections
  probability = 1 / 17 :=
by
  -- Conditions
  let total_pairs := 9
  let total_shoes := 2 * total_pairs
  let matching_pairs := total_pairs
  let total_selections := @Nat.choose total_shoes 2
  
  -- Correct answer
  have h_prob : probability = 1 / 17,
  {
    /- Proof omitted -/
    sorry,
  }
  exact h_prob

end matching_shoes_probability_l512_512746


namespace solve_equation_l512_512232

theorem solve_equation : (x : ℝ) (hx : x ≠ 1) (h : 3 / (x - 1) = 5 + 3 * x / (1 - x)) : x = 4 :=
sorry

end solve_equation_l512_512232


namespace part1_part2_l512_512509

def f (x : ℝ) : ℝ := |x - 1| + |x + 1| - 2

theorem part1 (x : ℝ) : f x ≥ 1 ↔ (x ≤ -5/2 ∨ x ≥ 3/2) :=
sorry

theorem part2 (a : ℝ) : (∀ x : ℝ, f x ≥ a^2 - a - 2) ↔ (-1 ≤ a ∧ a ≤ 2) :=
sorry

end part1_part2_l512_512509


namespace cone_lateral_surface_area_l512_512119

noncomputable def lateral_surface_area_cone (r h : ℝ) : ℝ :=
  let l := Real.sqrt (r^2 + h^2)
  π * r * l

theorem cone_lateral_surface_area :
  lateral_surface_area_cone 2 (√5) = 6 * π :=
by
  sorry

end cone_lateral_surface_area_l512_512119


namespace sum_x_bounds_l512_512270

noncomputable def min_sum_x (n : ℕ) (x : Fin n → ℝ) : ℝ := 1

noncomputable def max_sum_x (n : ℕ) : ℝ :=
  (∑ k in Finset.range n, (Real.sqrt k - Real.sqrt (k - 1)) ^ 2) ^ 0.5

theorem sum_x_bounds (n : ℕ) (x : Fin n → ℝ) 
  (h1 : ∀ i, 0 ≤ x i) 
  (h2 : (∑ i in Finset.range n, x i ^ 2) + 2 * ∑ k in Finset.range n, ∑ j in Finset.Ico (k + 1) n, Real.sqrt (k / j) * x k * x j = 1) :
  (∑ i in Finset.range n, x i) = min_sum_x n x ∨ (∑ i in Finset.range n, x i) = max_sum_x n :=
by
  sorry

end sum_x_bounds_l512_512270


namespace product_of_cosine_values_l512_512024

def arithmetic_seq (a₁ d : ℝ) (n : ℕ) : ℝ := a₁ + d * (n - 1)

theorem product_of_cosine_values (a₁ : ℝ) (h : ∃ (a b : ℝ), S = {a, b} ∧ S = {cos (arithmetic_seq a₁ (2 * π / 3) n) | n ∈ ℕ.succ}) :
  ∃ (a b : ℝ), a * b = -1 / 2 :=
begin
  obtain ⟨a, b, hS₁, hS₂⟩ := h,
  -- the proof will go here
  sorry
end

end product_of_cosine_values_l512_512024


namespace solve_equation_l512_512233

theorem solve_equation : (x : ℝ) (hx : x ≠ 1) (h : 3 / (x - 1) = 5 + 3 * x / (1 - x)) : x = 4 :=
sorry

end solve_equation_l512_512233


namespace distinct_arrangements_of_BALL_l512_512914

theorem distinct_arrangements_of_BALL : 
  (∃ (word : List Char), word = ['B', 'A', 'L', 'L'] 
    ∧ (∃ n_L n_A n_B : ℕ, 
          n_L = 2 ∧ n_A = 1 ∧ n_B = 1 
          ∧ nat.factorial 4 / (nat.factorial n_L * nat.factorial n_A * nat.factorial n_B) = 12)) :=
sorry

end distinct_arrangements_of_BALL_l512_512914


namespace equality_holds_iff_l512_512144

theorem equality_holds_iff (k t x y z : ℤ) (h_arith_prog : x + z = 2 * y) :
  (k * y^3 = x^3 + z^3) ↔ (k = 2 * (3 * t^2 + 1)) := by
  sorry

end equality_holds_iff_l512_512144


namespace no_a_for_empty_intersection_a_in_range_for_subset_union_l512_512084

open Set

def A (a : ℝ) : Set ℝ := {x | (x - a) * (x - (a + 3)) ≤ 0}
def B : Set ℝ := {x | x^2 - 4 * x - 5 > 0}

-- Problem 1: There is no a such that A ∩ B = ∅
theorem no_a_for_empty_intersection : ∀ a : ℝ, A a ∩ B = ∅ → False := by
  sorry

-- Problem 2: If A ∪ B = B, then a ∈ (-∞, -4) ∪ (5, ∞)
theorem a_in_range_for_subset_union (a : ℝ) : A a ∪ B = B → a ∈ Iio (-4) ∪ Ioi 5 := by
  sorry

end no_a_for_empty_intersection_a_in_range_for_subset_union_l512_512084


namespace inequality_l512_512730

noncomputable theory

def a (x : ℕ) : ℝ := x^(1 / 12 : ℝ)
def b (x : ℕ) : ℝ := x^(1 / 4 : ℝ)
def c (x : ℕ) : ℝ := x^(1 / 6 : ℝ)

theorem inequality (x : ℕ) (hx : 0 < x) : 2^a x + 2^b x ≥ 2^(1 + c x) := 
sorry

end inequality_l512_512730


namespace starting_lineups_count_l512_512783

theorem starting_lineups_count :
  let total_players := 15
  let lineup_size := 6
  let all_stars := 3
  let remaining_players := total_players - all_stars
  lineup_size - all_stars ≤ remaining_players →
  ∃ (lineups : ℕ), lineups = Nat.choose remaining_players (lineup_size - all_stars) ∧ lineups = 220 :=
by
  intros h
  use Nat.choose 12 3
  split
  exact rfl
  sorry

end starting_lineups_count_l512_512783


namespace playground_length_l512_512925

theorem playground_length
  (P : ℕ)
  (B : ℕ)
  (h1 : P = 1200)
  (h2 : B = 500)
  (h3 : P = 2 * (100 + B)) :
  100 = 100 :=
 by sorry

end playground_length_l512_512925


namespace students_taking_neither_mat_phy_cs_l512_512193

theorem students_taking_neither_mat_phy_cs 
  (total_students : ℕ) (math_students : ℕ) (physics_students : ℕ) 
  (cs_students : ℕ) (math_and_physics : ℕ) (math_and_cs : ℕ) 
  (physics_and_cs : ℕ) (all_three : ℕ)
  (h1 : total_students = 60)
  (h2 : math_students = 42)
  (h3 : physics_students = 35)
  (h4 : cs_students = 15)
  (h5 : math_and_physics = 25)
  (h6 : math_and_cs = 10)
  (h7 : physics_and_cs = 5)
  (h8 : all_three = 4) : 
  total_students - (math_students - math_and_physics - math_and_cs + all_three +
                   physics_students - math_and_physics - physics_and_cs + all_three +
                   cs_students - math_and_cs - physics_and_cs + all_three -
                   all_three) = 0 :=
by {
  rw [h1, h2, h3, h4, h5, h6, h7, h8],
  sorry
}

end students_taking_neither_mat_phy_cs_l512_512193


namespace contact_alignment_possible_l512_512769

/-- A vacuum tube has seven contacts arranged in a circle and is inserted into a socket that has seven holes.
Prove that it is possible to number the tube's contacts and the socket's holes in such a way that:
in any insertion of the tube, at least one contact will align with its corresponding hole (i.e., the hole with the same number). -/
theorem contact_alignment_possible : ∃ (f : Fin 7 → Fin 7), ∀ (rotation : Fin 7 → Fin 7), ∃ k : Fin 7, f k = rotation k := 
sorry

end contact_alignment_possible_l512_512769


namespace successive_percentage_reduction_l512_512739

variable (a b : ℝ) 

theorem successive_percentage_reduction :
  a = 25 →
  b = 30 →
  (a + b - (a * b / 100)) = 47.5 :=
by
  intros h₁ h₂
  rw [h₁, h₂]
  norm_num
  sorry

end successive_percentage_reduction_l512_512739


namespace connected_components_after_optimal_play_l512_512383

open Finset Nat

def is_black_connected_component (board : ℕ × ℕ → Prop) (comp : Finset (ℕ × ℕ)) : Prop :=
  ∀ p ∈ comp, ∀ q ∈ comp, p ≠ q → (∃ path : list (ℕ × ℕ), list.chain' (λ a b, (abs (a.1 - b.1) + abs (a.2 - b.2) = 1) ∧ board a ∧ board b) (p :: path ++ [q]))

def board : ℕ × ℕ → Prop := sorry -- condition to determine if a square is black

theorem connected_components_after_optimal_play : 
  ∃ comps : Finset (Finset (ℕ × ℕ)), 
    (∀ c ∈ comps, is_black_connected_component board c) → 
    (comps.card = 16) :=
by
  sorry

end connected_components_after_optimal_play_l512_512383


namespace find_floor_length_l512_512288

noncomputable def floor_breadth (b : ℝ) : ℝ := b
noncomputable def floor_length (b : ℝ) : ℝ := 3 * b

def area (b : ℝ) : ℝ := (floor_length b) * (floor_breadth b)

def paint_cost_per_sqm : ℝ := 3.00001
def total_paint_cost_floor : ℝ := 361
def total_cost : ℝ := 500
def cost_per_meter_border : ℝ := 15

theorem find_floor_length (b : ℝ) (h1 : paint_cost_per_sqm * area b = total_paint_cost_floor)
  (h2 : total_cost - total_paint_cost_floor = cost_per_meter_border * 2 * (floor_length b + floor_breadth b)) :
  floor_length b = 18.99 :=
sorry

end find_floor_length_l512_512288


namespace cars_without_features_l512_512626

theorem cars_without_features (total_cars cars_with_air_bags cars_with_power_windows cars_with_sunroofs 
                               cars_with_air_bags_and_power_windows cars_with_air_bags_and_sunroofs 
                               cars_with_power_windows_and_sunroofs cars_with_all_features: ℕ)
                               (h1 : total_cars = 80)
                               (h2 : cars_with_air_bags = 45)
                               (h3 : cars_with_power_windows = 40)
                               (h4 : cars_with_sunroofs = 25)
                               (h5 : cars_with_air_bags_and_power_windows = 20)
                               (h6 : cars_with_air_bags_and_sunroofs = 15)
                               (h7 : cars_with_power_windows_and_sunroofs = 10)
                               (h8 : cars_with_all_features = 8) : 
    total_cars - (cars_with_air_bags + cars_with_power_windows + cars_with_sunroofs 
                 - cars_with_air_bags_and_power_windows - cars_with_air_bags_and_sunroofs 
                 - cars_with_power_windows_and_sunroofs + cars_with_all_features) = 7 :=
by sorry

end cars_without_features_l512_512626


namespace cost_of_song_book_l512_512149

theorem cost_of_song_book 
  (flute_cost : ℝ) 
  (stand_cost : ℝ) 
  (total_cost : ℝ) 
  (h1 : flute_cost = 142.46) 
  (h2 : stand_cost = 8.89) 
  (h3 : total_cost = 158.35) : 
  total_cost - (flute_cost + stand_cost) = 7.00 := 
by 
  sorry

end cost_of_song_book_l512_512149


namespace equation_of_line_l_l512_512470

-- Define the point P
def P : ℝ × ℝ := (1, -2)

-- Define the circle C with center (2, -3) and radius 3
def C (x y : ℝ) : Prop := (x - 2)^2 + (y + 3)^2 = 9

-- Define the center of the circle
def center_C : ℝ × ℝ := (2, -3)

-- Define the equation of line l when angle ∠ACB is minimized
def line_l (x y : ℝ) : Prop := x - y - 3 = 0

-- Theorem to prove the equation of line l
theorem equation_of_line_l : 
  ∃ l, (∀ x y, l x y ↔ line_l x y) ∧ 
  ∃ A B : ℝ × ℝ, 
    (C A.1 A.2) ∧ (C B.1 B.2) ∧ 
    (A ≠ B) ∧ 
    (l A.1 A.2) ∧ (l B.1 B.2) ∧
    ∀ x y, line_l x y :=
by
  sorry

end equation_of_line_l_l512_512470


namespace total_surface_area_of_cone_l512_512305

-- Define the given essential properties and geometry of the cone.
def slant_height_of_cone : ℝ := 4
def area_of_cross_section_of_cone : ℝ := π

-- State the goal as a Lean theorem
theorem total_surface_area_of_cone :
  (let r := 2 in
   let circumference_of_base := 2 * π * r in
   let lateral_surface_area := (1 / 2) * circumference_of_base * slant_height_of_cone in
   let area_of_base := π * r ^ 2 in
   let total_surface_area := lateral_surface_area + area_of_base in
   area_of_cross_section_of_cone = π → total_surface_area = 12 * π) :=
begin
  sorry
end

end total_surface_area_of_cone_l512_512305


namespace find_departure_time_l512_512727

theorem find_departure_time (h : Time) (H1 : 4 * 60 < h) (H2 : h < 5 * 60) (r : Time) (H3 : 5 * 60 < r) (H4 : r < 6 * 60) (H5 : swapped_clock_time h r) : 
  h = 4 * 60 + 26 + (122 / 143) :=
sorry

end find_departure_time_l512_512727


namespace solve_equation_l512_512231

theorem solve_equation : (x : ℝ) (hx : x ≠ 1) (h : 3 / (x - 1) = 5 + 3 * x / (1 - x)) : x = 4 :=
sorry

end solve_equation_l512_512231


namespace joe_height_is_82_l512_512214

-- Given the conditions:
def Sara_height (x : ℝ) : Prop := true

def Joe_height (j : ℝ) (x : ℝ) : Prop := j = 6 + 2 * x

def combined_height (j : ℝ) (x : ℝ) : Prop := j + x = 120

-- We need to prove:
theorem joe_height_is_82 (x j : ℝ) 
  (h1 : combined_height j x)
  (h2 : Joe_height j x) :
  j = 82 := 
by 
  sorry

end joe_height_is_82_l512_512214


namespace projection_onto_plane_l512_512965

-- Definition of a plane passing through the origin
def plane_passes_through_origin (n : ℝ × ℝ × ℝ) (p : ℝ × ℝ × ℝ) : Prop :=
  let (nx, ny, nz) := n in
  let (px, py, pz) := p in
  nx * px + ny * py + nz * pz = 0

-- Definition of vector projection
def projection (v n : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  let (vx, vy, vz) := v in
  let (nx, ny, nz) := n in
  let dot_vn := vx * nx + vy * ny + vz * nz in
  let dot_nn := nx * nx + ny * ny + nz * nz in
  (vx - (dot_vn / dot_nn) * nx,
   vy - (dot_vn / dot_nn) * ny,
   vz - (dot_vn / dot_nn) * nz)

-- Given conditions
def Q_normal := (2, 1, 1)
def u := (4, 5, 1)
def proj_u := (0, 3, -1)
def v := (2, 1, 8)

-- Statement of the proof problem
theorem projection_onto_plane :
  plane_passes_through_origin Q_normal u ∧
  projection u Q_normal = proj_u →
  projection v Q_normal = (-4, -2, 5) :=
sorry

end projection_onto_plane_l512_512965


namespace consuela_total_amount_l512_512421

noncomputable def round_to_nearest_dollar (x : ℝ) : ℤ :=
  if x - Real.floor x < 0.5 then Real.floor x else Real.ceil x

theorem consuela_total_amount :
  let c1 := 2.95
  let c2 := 6.48
  let c3 := 8.99
  round_to_nearest_dollar (c1) + round_to_nearest_dollar (c2) + round_to_nearest_dollar (c3) = 18 :=
by
  let c1 := 2.95
  let c2 := 6.48
  let c3 := 8.99
  sorry

end consuela_total_amount_l512_512421


namespace abs_a_gt_neg_b_l512_512457

variable {a b : ℝ}

theorem abs_a_gt_neg_b (h : a < b ∧ b < 0) : |a| > -b :=
by
  sorry

end abs_a_gt_neg_b_l512_512457


namespace complement_union_A_B_eq_neg2_0_l512_512539

def U : Set ℤ := {-2, -1, 0, 1, 2, 3}
def A : Set ℤ := {-1, 2}
def B : Set ℤ := {x | x^2 - 4 * x + 3 = 0}

theorem complement_union_A_B_eq_neg2_0 :
  U \ (A ∪ B) = {-2, 0} := by
  sorry

end complement_union_A_B_eq_neg2_0_l512_512539


namespace primes_sq_sum_divisible_by_24_l512_512976

theorem primes_sq_sum_divisible_by_24 
 (p : ℕ → ℕ) 
 (h_prime: ∀ i, i < 24 → Nat.Prime (p i)) 
 (h_ge_5: ∀ i, i < 24 → p i ≥ 5) :
  (∑ i in Finset.range 24, (p i)^2) % 24 = 0 :=
by
  sorry

end primes_sq_sum_divisible_by_24_l512_512976


namespace part1_99_9_certainty_part2_probability_l512_512669

-- Definitions for given values from contingency table
def a := 60
def b := 40
def c := 30
def d := 70
def n := a + b + c + d

-- K^2 calculation from contingency table values
def K_squared : ℝ :=
  let numerator := n * ((a * d - b * c) ^ 2)
  let denominator := (a + b) * (c + d) * (a + c) * (b + d)
  numerator / denominator

-- Lean statement for proving K^2 value and its comparison
theorem part1_99_9_certainty : K_squared ≥ 10.828 := by
  sorry

-- Definitions for probability calculation
def total_ways := Nat.choose 6 2  -- binomial coefficient "6 choose 2"
def favorable_ways := 8

-- Probability of selecting exactly 1 from City A and 1 from City B
def probability : ℚ := favorable_ways / total_ways

-- Lean statement for proving the probability of specific selection
theorem part2_probability : probability = 8 / 15 := by
  sorry

end part1_99_9_certainty_part2_probability_l512_512669


namespace circle_line_bisect_l512_512554

theorem circle_line_bisect (a : ℝ) :
    (∀ x y : ℝ, (x - (-1))^2 + (y - 2)^2 = 5 → 3 * x + y + a = 0) → a = 1 :=
sorry

end circle_line_bisect_l512_512554


namespace verify_f_l512_512386

def functional_eq (f : ℝ → ℝ) : Prop :=
  ∀ (x y : ℝ), f ((x + y) / 2) = (f x + f y) / 2

theorem verify_f {f : ℝ → ℝ} (h : f = (λ x, 3 * x)) :
  functional_eq f :=
by 
  rw [h]
  sorry

end verify_f_l512_512386


namespace count_perfect_squares_l512_512342

noncomputable def sequence (n : ℕ) : ℕ :=
  Nat.recOn n 1 (λ n a, a + (Nat.sqrt a))

theorem count_perfect_squares :
  (Finset.filter (λ x, ∃ k : ℕ, k * k = x) (Finset.range 1000001)).card = 10 := 
  sorry

end count_perfect_squares_l512_512342


namespace investment_period_p_l512_512681

variables {x t : ℕ}

def ratio_of_investments (p q : ℕ) : Prop := p = 7 * x ∧ q = 5 * x
def ratio_of_profits (profit_p profit_q : ℕ) : Prop := profit_p / profit_q = 7 / 13
def proportional_profit (money time profit : ℕ) : Prop := profit = money * time

theorem investment_period_p (p q profit_p profit_q : ℕ)
  (h1 : ratio_of_investments p q)
  (h2 : ratio_of_profits profit_p profit_q)
  (h3 : proportional_profit p t profit_p)
  (h4 : proportional_profit q 13 profit_q) :
  t = 7 :=
by
  sorry

end investment_period_p_l512_512681


namespace solve_fractional_equation_l512_512226

theorem solve_fractional_equation (x : ℝ) (hx : x ≠ 1 ∧ x ≠ 1) : 3 / (x - 1) = 5 + 3 * x / (1 - x) ↔ x = 4 :=
by 
  split
  -- Forward direction (Given the equation, show x = 4)
  intro h
  -- Multiply out and simplify as shown in the solution steps
  -- Sorry statement skips the detailed proof
  sorry
  -- Converse direction (Substitute x = 4 and verify it satisfies the equation)
  intro hx
  -- Substitution and verification steps
  -- Sorry statement skips the detailed proof
  sorry

end solve_fractional_equation_l512_512226


namespace product_of_cosine_values_l512_512026

def arithmetic_seq (a₁ d : ℝ) (n : ℕ) : ℝ := a₁ + d * (n - 1)

theorem product_of_cosine_values (a₁ : ℝ) (h : ∃ (a b : ℝ), S = {a, b} ∧ S = {cos (arithmetic_seq a₁ (2 * π / 3) n) | n ∈ ℕ.succ}) :
  ∃ (a b : ℝ), a * b = -1 / 2 :=
begin
  obtain ⟨a, b, hS₁, hS₂⟩ := h,
  -- the proof will go here
  sorry
end

end product_of_cosine_values_l512_512026


namespace max_knights_is_seven_l512_512368

-- Definitions of conditions
def students : ℕ := 11
def total_statements : ℕ := students * (students - 1)
def liar_statements : ℕ := 56

-- Definition translating the problem statement
theorem max_knights_is_seven : ∃ (k li : ℕ), 
  (k + li = students) ∧ 
  (k * li = liar_statements) ∧ 
  (k = 7) := 
by
  sorry

end max_knights_is_seven_l512_512368


namespace polar_to_cartesian_and_OP_values_l512_512908

-- Definitions and conditions based on the problem
def polar_eq (ρ θ : ℝ) : Prop :=
  ρ^2 - 4 * real.sqrt 2 * ρ * real.cos (θ - real.pi / 4) + 6 = 0

def cartesian_eq (x y : ℝ) : Prop :=
  x^2 + y^2 - 4 * x - 4 * y + 6 = 0

-- The main statement
theorem polar_to_cartesian_and_OP_values:
  (∀ (ρ θ : ℝ), polar_eq ρ θ ↔ ∃ (x y : ℝ), x = ρ * real.cos θ ∧ y = ρ * real.sin θ ∧ cartesian_eq x y) ∧
  (∀ (x y : ℝ), cartesian_eq x y → ((2 - real.sqrt 2) ≤ real.sqrt(x^2 + y^2) ∧ real.sqrt(x^2 + y^2) ≤ 2 + real.sqrt 2)) :=
by
  -- Proof is non-trivial and will be provided as needed
  sorry

end polar_to_cartesian_and_OP_values_l512_512908


namespace max_value_expression_l512_512055

theorem max_value_expression (n : ℕ) (x : Fin n → ℝ) 
  (h : ∀ i, 0 < x i ∧ x i < 1) :
  (∃ x_i, i ∈ Finset.range n → ∃ (hxi : x_i ∈ set.Ioo 0 1), 
  A = (∑ i in Finset.range n, Real.root 4 (1 - x i)) / (∑ i in Finset.range n, 1 / Real.root 4 (x i))) :=
  sorry

end max_value_expression_l512_512055


namespace intersection_point_l512_512590

noncomputable def f (x : ℝ) : ℝ := (x^2 - 4 * x + 3) / (2 * x - 6)
noncomputable def g (a b c x : ℝ) : ℝ := (a * x^2 + b * x + c) / (x - 3)

theorem intersection_point (a b c : ℝ) (h_asymp : ¬(2 = 0) ∧ (a ≠ 0 ∨ b ≠ 0)) (h_perpendicular : True) (h_y_intersect : g a b c 0 = 1) (h_intersects : f (-1) = g a b c (-1)):
  f 1 = 0 :=
by
  dsimp [f, g] at *
  sorry

end intersection_point_l512_512590


namespace arrangements_7_people_no_A_at_head_no_B_in_middle_l512_512290

theorem arrangements_7_people_no_A_at_head_no_B_in_middle :
  let n := 7
  let total_arrangements := Nat.factorial n
  let A_at_head := Nat.factorial (n - 1)
  let B_in_middle := A_at_head
  let overlap := Nat.factorial (n - 2)
  total_arrangements - 2 * A_at_head + overlap = 3720 :=
by
  let n := 7
  let total_arrangements := Nat.factorial n
  let A_at_head := Nat.factorial (n - 1)
  let B_in_middle := A_at_head
  let overlap := Nat.factorial (n - 2)
  show total_arrangements - 2 * A_at_head + overlap = 3720
  sorry

end arrangements_7_people_no_A_at_head_no_B_in_middle_l512_512290


namespace number_of_volunteers_with_protein_not_less_than_20_l512_512748

noncomputable def normal_distribution (mean : ℝ) (variance : ℝ) : ℝ → ℝ :=
  λ x, (Real.exp (- (x - mean) ^ 2 / (2 * variance)) / ((2 * Real.pi * variance).sqrt))

variables (σ : ℝ)
variables (n_volunteers : ℕ)
variables (immune_response : ℝ → ℝ)

def immune_response_distribution (x : ℝ) : ℝ :=
  normal_distribution 15 σ x

def proportion_in_interval : ℝ :=
  19 / 25

def total_volunteers : ℕ :=
  500

theorem number_of_volunteers_with_protein_not_less_than_20 :
  let num_outside_interval := (1 - proportion_in_interval) * total_volunteers,
      num_not_less_than_20 := num_outside_interval / 2
  in num_not_less_than_20 = 60 :=
sorry

end number_of_volunteers_with_protein_not_less_than_20_l512_512748


namespace relationship_between_y_coordinates_l512_512479

theorem relationship_between_y_coordinates (b y1 y2 y3 : ℝ)
  (h1 : y1 = 3 * (-3) - b)
  (h2 : y2 = 3 * 1 - b)
  (h3 : y3 = 3 * (-1) - b) :
  y1 < y3 ∧ y3 < y2 := 
sorry

end relationship_between_y_coordinates_l512_512479


namespace regina_earnings_l512_512211

def num_cows : ℕ := 20

def num_pigs (num_cows : ℕ) : ℕ := 4 * num_cows

def price_per_pig : ℕ := 400
def price_per_cow : ℕ := 800

def earnings (num_cows num_pigs price_per_cow price_per_pig : ℕ) : ℕ :=
  num_cows * price_per_cow + num_pigs * price_per_pig

theorem regina_earnings :
  earnings num_cows (num_pigs num_cows) price_per_cow price_per_pig = 48000 :=
by
  -- proof omitted
  sorry

end regina_earnings_l512_512211


namespace postage_problem_l512_512443

theorem postage_problem (n : ℕ) (h_positive : n > 0) (h_postage : ∀ k, k ∈ List.range 121 → ∃ a b c : ℕ, 6 * a + n * b + (n + 2) * c = k) :
  6 * n * (n + 2) - (6 + n + (n + 2)) = 120 → n = 8 := 
by
  sorry

end postage_problem_l512_512443


namespace unordered_pairs_of_edges_determining_plane_l512_512094

-- Given definitions
inductive Edge
| mk : ℕ → Edge

def regular_tetrahedron_edges : set Edge :=
  {Edge.mk 1, Edge.mk 2, Edge.mk 3, Edge.mk 4, Edge.mk 5, Edge.mk 6}

def shares_common_vertex (e1 e2 : Edge) : Prop := sorry -- Define as per problem conditions
def determines_plane (e1 e2 : Edge) : Prop :=
  shares_common_vertex e1 e2

-- Main theorem
theorem unordered_pairs_of_edges_determining_plane (edges : set Edge) (h : edges = regular_tetrahedron_edges) :
  {pair : set (Edge × Edge) | ∃ e1 e2, e1 ∈ edges ∧ e2 ∈ edges ∧ e1 ≠ e2 ∧ determines_plane e1 e2}.card = 10 :=
by sorry

end unordered_pairs_of_edges_determining_plane_l512_512094


namespace q_p_sum_is_neg13_l512_512282

noncomputable def p (x : ℝ) : ℝ := x^2 - 4
noncomputable def q (x : ℝ) : ℝ := -|x| + 1

def q_p_sum : ℝ :=
  let xs := [-3, -2, -1, 0, 1, 2, 3]
  (xs.map (λ x => q (p x))).sum

theorem q_p_sum_is_neg13 : q_p_sum = -13 := by
  sorry

end q_p_sum_is_neg13_l512_512282


namespace desired_chord_length_squared_l512_512675

theorem desired_chord_length_squared (R : ℝ) (x : ℝ) :
  R = 10 →
  x = 5 →
  (∀ (chord_len₁ chord_len₂ : ℝ), chord_len₁ = 10 → chord_len₂ = 14 → (dist : ℝ), dist = 6 → 
    chord_len₁^2 / 4 + x^2 = R^2 ∧ chord_len₂^2 / 4 + (dist - x)^2 = R^2) →
  (∀ (d_avg : ℝ), d_avg = (1 + 5) / 2 → (chord_mid_len : ℝ), chord_mid_len^2 / 4 + d_avg^2 = R^2 → 
    chord_mid_len = real.sqrt 164) :=
begin
  assume diameter,
  assume x_fixed,
  assume chord_length_equations,
  assume average_distance,
  sorry
end

end desired_chord_length_squared_l512_512675


namespace complement_union_eq_l512_512528

def U : Set ℤ := {-2, -1, 0, 1, 2, 3}
def A : Set ℤ := {-1, 2}
def B : Set ℤ := {x | x ^ 2 - 4 * x + 3 = 0}

theorem complement_union_eq :
  (U \ (A ∪ B)) = {-2, 0} :=
by
  sorry

end complement_union_eq_l512_512528


namespace largest_integer_solution_l512_512717

theorem largest_integer_solution :
  ∃ (x : ℤ), (x < 4) ∧ (4 * x + 7 < 24) :=
by
  existsi (3 : ℤ)
  split
  · norm_num
  · norm_num

end largest_integer_solution_l512_512717


namespace algebra_1_algebra_2_l512_512484

variable (x1 x2 : ℝ)
variable (h_root1 : x1^2 - 2*x1 - 1 = 0)
variable (h_root2 : x2^2 - 2*x2 - 1 = 0)
variable (h_sum : x1 + x2 = 2)
variable (h_prod : x1 * x2 = -1)

theorem algebra_1 : (x1 + x2) * (x1 * x2) = -2 := by
  -- Proof here
  sorry

theorem algebra_2 : (x1 - x2)^2 = 8 := by
  -- Proof here
  sorry

end algebra_1_algebra_2_l512_512484


namespace range_of_m_l512_512880

variable {f : ℝ → ℝ}

-- Definition: f is an odd function
def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f (x)

-- Definition: f is a decreasing function on the interval (-2, 2)
def is_decreasing_on_interval (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a < x → x < y → y < b → f y ≤ f x

-- Given conditions
variables (m : ℝ)
variable odd_f : is_odd f
variable decreasing_f : is_decreasing_on_interval f (-2) 2
variable h : f (m - 1) + f (2 * m - 1) > 0

theorem range_of_m :
  -1 / 2 < m ∧ m < 2 / 3 :=
sorry

end range_of_m_l512_512880


namespace find_equation_of_perpendicular_line_l512_512434

noncomputable def line_through_point_perpendicular
    (A : ℝ × ℝ) (a b c : ℝ) (hA : A = (2, 3)) (hLine : a = 2 ∧ b = 1 ∧ c = -5) :
    Prop :=
  ∃ (m : ℝ) (b1 : ℝ), (m = (1 / 2)) ∧
    (b1 = 3 - m * 2) ∧
    (∀ (x y : ℝ), y = m * (x - 2) + 3 → a * x + b * y + c = 0 → x - 2 * y + 4 = 0)

theorem find_equation_of_perpendicular_line :
  line_through_point_perpendicular (2, 3) 2 1 (-5) rfl ⟨rfl, rfl, rfl⟩ :=
sorry

end find_equation_of_perpendicular_line_l512_512434


namespace complement_union_eq_l512_512527

def U : Set ℤ := {-2, -1, 0, 1, 2, 3}
def A : Set ℤ := {-1, 2}
def B : Set ℤ := {x | x ^ 2 - 4 * x + 3 = 0}

theorem complement_union_eq :
  (U \ (A ∪ B)) = {-2, 0} :=
by
  sorry

end complement_union_eq_l512_512527


namespace sum_of_digits_in_product_is_fourteen_l512_512824

def first_number : ℕ := -- Define the 101-digit number 141,414,141,...,414,141
  141 * 10^98 + 141 * 10^95 + 141 * 10^92 -- continue this pattern...

def second_number : ℕ := -- Define the 101-digit number 707,070,707,...,070,707
  707 * 10^98 + 707 * 10^95 + 707 * 10^92 -- continue this pattern...

def units_digit (n : ℕ) : ℕ := n % 10
def ten_thousands_digit (n : ℕ) : ℕ := (n / 10000) % 10

theorem sum_of_digits_in_product_is_fourteen :
  units_digit (first_number * second_number) + ten_thousands_digit (first_number * second_number) = 14 :=
sorry

end sum_of_digits_in_product_is_fourteen_l512_512824


namespace angle_half_size_l512_512167

noncomputable def convex_pentagon : Type := sorry
noncomputable def semicircle : convex_pentagon → Type := sorry
noncomputable def inscribed (p : convex_pentagon) : Prop := sorry
noncomputable def diameter (p : convex_pentagon) : Type := sorry
noncomputable def midpoint (a b : Type) : Type := sorry
noncomputable def perpendicular (p q : Type) : Prop := sorry
noncomputable def angle (a b : Type) : ℝ := sorry
noncomputable def acute_angle (a b : Type) : Type := sorry

variable {AXYZB : convex_pentagon}
variable {AB : Type}

theorem angle_half_size (h1 : inscribed AXYZB)
  (h2 : diameter AXYZB AB)
  (O : Type)
  (h3 : midpoint AB O)
  (P Q R S : Type)
  (h4 : perpendicular Y AX P)
  (h5 : perpendicular Y BX Q)
  (h6 : perpendicular Y AZ R)
  (h7 : perpendicular Y BZ S) :
  acute_angle PQ RS = angle XOZ / 2 := 
sorry

end angle_half_size_l512_512167


namespace seat_notation_l512_512926

theorem seat_notation (row1 col1 row2 col2 : ℕ) (h : (row1, col1) = (5, 2)) : (row2, col2) = (7, 3) :=
 by
  sorry

end seat_notation_l512_512926


namespace num_integers_satisfying_inequality_l512_512090

theorem num_integers_satisfying_inequality : 
  {x : ℤ | |6 * x - 4| ≤ 8}.toFinset.card = 4 := 
by 
  sorry

end num_integers_satisfying_inequality_l512_512090


namespace binomial_expansion_coeff_eq_l512_512915

theorem binomial_expansion_coeff_eq (n : ℕ) :
  (1 + Polynomial.C (1 : ℝ))^n = Polynomial.C (1 : ℝ) + 
       Polynomial.C (6 : ℝ) * Polynomial.X + 
       Polynomial.C (15 : ℝ) * Polynomial.X^2 + 
       Polynomial.C (20 : ℝ) * Polynomial.X^3 + 
       Polynomial.C (15 : ℝ) * Polynomial.X^4 + 
       Polynomial.C (6 : ℝ) * Polynomial.X^5 + 
       Polynomial.C (1 : ℝ) * Polynomial.X^6 
  → n = 6 :=
by
  sorry

end binomial_expansion_coeff_eq_l512_512915


namespace identify_coin_weights_l512_512451

theorem identify_coin_weights :
  ∃ (weigh : list (ℕ × ℕ) → ℕ → ℕ) (pan : list ℕ → ℕ)
    (results : list (list ℕ)) (coins : list ℕ),
    (coins = [1, 2, 3, 4]) ∧
    (length results ≤ 4) ∧
    ∀ result ∈ results, ∃ a b, weigh [(a, b)] coins = pan result :=
sorry

end identify_coin_weights_l512_512451


namespace cheapest_pie_l512_512410

def cost_flour : ℝ := 2
def cost_sugar : ℝ := 1
def cost_eggs_butter : ℝ := 1.5
def cost_crust : ℝ := cost_flour + cost_sugar + cost_eggs_butter

def weight_blueberries : ℝ := 3
def container_weight : ℝ := 0.5 -- 8 oz in pounds
def price_per_blueberry_container : ℝ := 2.25
def cost_blueberries (weight: ℝ) (container_weight: ℝ) (price_per_container: ℝ) : ℝ :=
  (weight / container_weight) * price_per_container

def weight_cherries : ℝ := 4
def price_cherry_bag : ℝ := 14

def cost_blueberry_pie : ℝ := cost_crust + cost_blueberries weight_blueberries container_weight price_per_blueberry_container
def cost_cherry_pie : ℝ := cost_crust + price_cherry_bag

theorem cheapest_pie : min cost_blueberry_pie cost_cherry_pie = 18 := by
  sorry

end cheapest_pie_l512_512410


namespace find_rate_l512_512338

variable (P : ℝ) (r : ℝ)
def T := 10
def SI := (P * r * T) / 100

theorem find_rate (h₁ : SI = (6 / 5) * P) : r = 12 := by
  sorry

end find_rate_l512_512338


namespace sample_standard_deviation_same_sample_ranges_same_l512_512865

variables {n : ℕ} (x y : Fin n → ℝ) (c : ℝ)
  (h_y : ∀ i, y i = x i + c)
  (h_c_ne_zero : c ≠ 0)

-- Statement for standard deviations being the same
theorem sample_standard_deviation_same :
  let mean (s : Fin n → ℝ) := (∑ i, s i) / n
  in let stddev (s : Fin n → ℝ) := sqrt ((∑ i, (s i - mean s) ^ 2) / n)
  in stddev x = stddev y := 
sorry

-- Statement for ranges being the same
theorem sample_ranges_same :
  let range (s : Fin n → ℝ) := (Finset.univ.sup s) - (Finset.univ.inf s)
  in range x = range y :=
sorry

end sample_standard_deviation_same_sample_ranges_same_l512_512865


namespace smallest_n_for_pencil_purchase_l512_512098

theorem smallest_n_for_pencil_purchase (a b c d n : ℕ)
  (h1 : 6 * a + 10 * b = n)
  (h2 : 6 * c + 10 * d = n + 2)
  (h3 : 7 * a + 12 * b > 7 * c + 12 * d)
  (h4 : 3 * (c - a) + 5 * (d - b) = 1)
  (h5 : d - b > 0) :
  n = 100 :=
by
  sorry

end smallest_n_for_pencil_purchase_l512_512098


namespace intersection_of_function_and_inverse_l512_512286

theorem intersection_of_function_and_inverse (b a : Int) 
  (h₁ : a = 2 * (-4) + b) 
  (h₂ : a = (-4 - b) / 2) 
  : a = -4 :=
by
  sorry

end intersection_of_function_and_inverse_l512_512286


namespace largest_prime_factor_101_l512_512364

theorem largest_prime_factor_101 
  (seq : List (Fin 10000))
  (h1 : ∀ i, i < seq.length → (seq.get i).div 100 % 100 = (seq.get ((i + 1) % seq.length)).div 10000 ∧ (seq.get i).div 10 % 10 = (seq.get ((i + 1) % seq.length)).div 1000 % 10)
  (h2 : ∀ i, i < seq.length - 1 → (seq.get i).div 100 % 100 = (seq.get ((i + 1) % seq.length)).div 10000 ∧ (seq.get (seq.length - 1)).div 100 % 100 = (seq.get 0).div 10000 ∧ (seq.get (seq.length - 1)).div 10 % 10 = (seq.get 0).div 1000 % 10):
let S := seq.foldl (λ acc x, acc + x) 0 in 
∃ (k : ℕ), S = 1111 * k ∧ Nat.Prime 101 ∧ 101 ∣ S := sorry

end largest_prime_factor_101_l512_512364


namespace johns_remaining_money_l512_512950

/-- John had $20. He spent 1/5 of his money on snacks, 
3/4 of the remaining money on necessities, 
and 25% of what was left on a gift for a friend. 
How much is left of John's money? -/
theorem johns_remaining_money : 
  let initial_money := 20
  let spent_on_snacks := (1 / 5) * initial_money
  let remaining_after_snacks := initial_money - spent_on_snacks
  let spent_on_necessities := (3 / 4) * remaining_after_snacks
  let remaining_after_necessities := remaining_after_snacks - spent_on_necessities
  let spent_on_gift := (25 / 100) * remaining_after_necessities
  let final_remaining := remaining_after_necessities - spent_on_gift
  in final_remaining = 3 := 
by 
  let initial_money := 20
  let spent_on_snacks := (1 / 5) * initial_money
  let remaining_after_snacks := initial_money - spent_on_snacks
  let spent_on_necessities := (3 / 4) * remaining_after_snacks
  let remaining_after_necessities := remaining_after_snacks - spent_on_necessities
  let spent_on_gift := (25 / 100) * remaining_after_necessities
  let final_remaining := remaining_after_necessities - spent_on_gift
  show final_remaining = 3
  from sorry

end johns_remaining_money_l512_512950


namespace evaluate_expression_l512_512804

theorem evaluate_expression : 
  (4 + 5) * (4^2 + 5^2) * (4^4 + 5^4) * (4^8 + 5^8) * (4^16 + 5^16) * (4^32 + 5^32) * (4^64 + 5^64) = 5^128 - 4^128 :=
by sorry

end evaluate_expression_l512_512804


namespace rotate_60_deg_l512_512135

theorem rotate_60_deg (z : ℂ) (h : z = complex.mk (Real.sqrt 3) 1) : 
  z * (complex.mk (1/2) (Real.sqrt 3 / 2)) = complex.I * 2 :=
by 
  rw [h, complex.mul_re, complex.mul_im, complex.add_re, complex.add_im, complex.mul_re, complex.mul_im]
  -- definitions for complex arithmetic can be instantiated here
  exact sorry

end rotate_60_deg_l512_512135


namespace smallest_n_divided_into_smaller_cubes_l512_512823

theorem smallest_n_divided_into_smaller_cubes :
  ∃ (n : ℕ), (n > 0) ∧ (∃ (smaller_side : ℕ), smaller_side > 0 ∧ (\(k \in ℕ), (k * smaller_side) = 1996 ∧ k <= n^3)) ∧ n = 13 :=
by
  sorry

end smallest_n_divided_into_smaller_cubes_l512_512823


namespace sum_of_underlined_is_positive_l512_512761

-- Definitions
def is_positive (x : ℤ) : Prop := x > 0

def is_underlined (a : ℕ → ℤ) (i : ℕ) : Prop :=
  is_positive (a i) ∨ ∃ k : ℕ, k ≤ (n - 1 - i) ∧ (∑ j in finset.range (k + 1), a (i + j)) > 0

-- Main statement
theorem sum_of_underlined_is_positive (a : ℕ → ℤ) (n : ℕ) (h : ∀ i : ℕ, i < n → is_underlined a i) :
  ∑ i in finset.range n, if is_underlined a i then a i else 0 > 0 :=
sorry

end sum_of_underlined_is_positive_l512_512761


namespace inequality_holds_l512_512164

variable {p q r : ℝ} {n : ℕ}

-- Assumptions
axiom h1 : p > 0 ∧ q > 0 ∧ r > 0
axiom h2 : p * q * r = 1

-- Theorem Statement
theorem inequality_holds : 
  (∀ (n : ℕ), 1 / (p^n + q^n + 1) + 1 / (q^n + r^n + 1) + 1 / (r^n + p^n + 1) ≤ 1) :=
by
  assume n
  sorry

end inequality_holds_l512_512164


namespace negation_of_p_is_neg_p_l512_512481

-- Define the proposition p
def p : Prop := ∃ m : ℝ, m > 0 ∧ ∃ x : ℝ, m * x^2 + x - 2 * m = 0

-- Define the negation of p
def neg_p : Prop := ∀ m : ℝ, m > 0 → ¬ ∃ x : ℝ, m * x^2 + x - 2 * m = 0

-- The theorem statement
theorem negation_of_p_is_neg_p : (¬ p) = neg_p := 
by
  sorry

end negation_of_p_is_neg_p_l512_512481


namespace hcf_of_three_numbers_l512_512553

theorem hcf_of_three_numbers (a b c : ℕ) (h1 : Nat.lcm a (Nat.lcm b c) = 45600) (h2 : a * b * c = 109183500000) :
  Nat.gcd a (Nat.gcd b c) = 2393750 := by
  sorry

end hcf_of_three_numbers_l512_512553


namespace Amanda_income_if_report_not_finished_l512_512159

def hourly_pay : ℝ := 50.0
def hours_worked_per_day : ℝ := 10.0
def percent_withheld : ℝ := 0.2

theorem Amanda_income_if_report_not_finished : 
  let total_daily_income := hourly_pay * hours_worked_per_day in
  let amount_withheld := percent_withheld * total_daily_income in
  let amount_received := total_daily_income - amount_withheld in
  amount_received = 400 :=
by
  sorry

end Amanda_income_if_report_not_finished_l512_512159


namespace problem_statement_l512_512959

theorem problem_statement (p : ℕ) (x : ℤ) (hp : p.prime) (hodd : p % 2 = 1) (hdiv : p ∣ x^3 - 1) (hne : ¬ p ∣ x - 1) :
  p ∣ (p - 1)! * (x - (x^2) / 2 + (x^3) / 3 - ... - (x^(p - 1)) / (p - 1)) := 
sorry

end problem_statement_l512_512959


namespace sum_of_consecutive_integers_is_33_l512_512299

theorem sum_of_consecutive_integers_is_33 :
  ∃ (x : ℕ), x * (x + 1) = 272 ∧ x + (x + 1) = 33 :=
by
  sorry

end sum_of_consecutive_integers_is_33_l512_512299


namespace smallest_positive_integer_for_z_in_T_l512_512967

def T (x y : ℝ) : Prop := (1 / 2 ≤ x ∧ x ≤ Real.sqrt 2 / 2)

theorem smallest_positive_integer_for_z_in_T (p : ℕ) (n : ℕ) :
  p = 12 → n ≥ p → (∃ x y : ℝ, T x y ∧ Complex.ofReal x + Complex.i * y = 1) :=
by
  intros h₁ h₂
  sorry

end smallest_positive_integer_for_z_in_T_l512_512967


namespace number_of_correct_statements_is_one_l512_512220

noncomputable def sequence (x : ℕ → ℝ) : Prop :=
∀ n > 0, x(n + 1) = 1 / (1 - x n)

theorem number_of_correct_statements_is_one (x : ℕ → ℝ) :
  (sequence x) →
  (x 2 = 5 → x 7 = 4 / 5) →
  (x 1 = 2 → x.sum (range 2022) ≠ 2021 / 2) →
  ((x 1 + 1) * (x 2 + 1) * x 9 = -1 → ¬ (x 1 = real.sqrt 2)) →
  1 = 1 :=
by
  -- Proof omitted, insert full proof here
  sorry

end number_of_correct_statements_is_one_l512_512220


namespace regina_earnings_l512_512213

-- Definitions based on conditions
def num_cows := 20
def num_pigs := 4 * num_cows
def price_per_pig := 400
def price_per_cow := 800

-- Total earnings calculation based on definitions
def earnings_from_cows := num_cows * price_per_cow
def earnings_from_pigs := num_pigs * price_per_pig
def total_earnings := earnings_from_cows + earnings_from_pigs

-- Proof statement
theorem regina_earnings : total_earnings = 48000 := by
  sorry

end regina_earnings_l512_512213


namespace find_eccentricity_l512_512061

-- Definitions of the given conditions
variables {a b c : ℝ} (F₁ F₂ A P : ℝ × ℝ)
hypothesis (ha : a > 0) (hb : b > 0) (hab : a > b) 
hypothesis (ell_def : ∀ (x y : ℝ), (x, y) ∈ set_of (λ p : ℝ × ℝ, (p.1 ^ 2) / (a ^ 2) + (p.2 ^ 2) / (b ^ 2) = 1))
hypothesis (F₁_def : F₁ = (-c, 0)) (F₂_def : F₂ = (c, 0))
hypothesis (A_def : A = (-a, 0))
hypothesis (P_def : ∃ k: ℝ, P = (k, (sqrt 3 / 6) * (k + a)))
hypothesis (isosceles_triangle : dist P F₂ = dist F₁ F₂ ∧ ∠ F₁ F₂ P = real.pi / 3)

-- Definition of eccentricity
def eccentricity (a c : ℝ) := c / a

-- Statement we want to prove
theorem find_eccentricity : eccentricity a c = 1 / 4 :=
sorry

end find_eccentricity_l512_512061


namespace problem_solution_l512_512942

-- Definition of vectors
def A : ℝ × ℝ × ℝ := (0, 2, 1)
def B : ℝ × ℝ × ℝ := (0, -1, 1)
def C : ℝ × ℝ × ℝ := (-1, 2, -1)

noncomputable def vec_sub (P Q : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (P.1 - Q.1, P.2 - Q.2, P.3 - Q.3)

noncomputable def dot_product (u v : ℝ × ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2 + u.3 * v.3

noncomputable def norm (u : ℝ × ℝ × ℝ) : ℝ :=
  real.sqrt (u.1^2 + u.2^2 + u.3^2)

theorem problem_solution
  (A B C : ℝ × ℝ × ℝ)
  (AB BC : ℝ × ℝ × ℝ := vec_sub B A, vec_sub C B)
  (n : ℝ × ℝ × ℝ := (2, 0, -1)):
  -- Conditions based on the correct answers
  
  (∃ (OB : ℝ × ℝ × ℝ := B), sin (real.arccos (dot_product OB n / (norm OB * norm n))) = real.sqrt 10 / 10) ∧
  (∃ (O : ℝ × ℝ × ℝ := (0, 0, 0)), distance O ABC = real.sqrt 5 / 5) ∧
  (∃ (OA BC : ℝ × ℝ × ℝ := vec_sub A O, vec_sub C B), cos (real.arccos (dot_product OA BC / (norm OA * norm BC))) = 2 * real.sqrt 70 / 35) ∧
  (∃ (d : ℝ := distance A OB), d = 3*real.sqrt 2 / 2) :=
begin
  sorry
end

end problem_solution_l512_512942


namespace cyclic_quadrilateral_triangle_ABC_angles_l512_512954

-- Definitions as per the conditions stated in the problem
variables (A B C B1 C1 A1 : Type) 
           [Inhabited A] [Inhabited B] [Inhabited C] 
           [Inhabited B1] [Inhabited C1] [Inhabited A1]

-- Assuming angle bisectors and other properties
noncomputable def angle_bisector (A1 A B1 : Type) [Inhabited A1] [Inhabited A] [Inhabited B1] := sorry 

-- Part (a)
theorem cyclic_quadrilateral (h1 : angle_bisector A1 A B1)
                             (h2 : angle_bisector B1 B C1)
                             (h3 : angle_bisector C1 C A1)
                             (h4 : AC = BC)
                             (h5 : A1C1 ≠ B1C1) : 
            (∃ (circ : Type) [Inhabited circ], cyclic quadrilateral A B C C1 circ) :=
sorry

-- Part (b)
theorem triangle_ABC_angles (h : ∠BAC1 = π / 6)
                            (h1 : cyclic_quadrilateral_proof):
            (angles_triangle_ABC (∠ABCDE, ∠ABCD, ∠ACBD) = (π/6, 5π/12, 5π/12)) :=
sorry

end cyclic_quadrilateral_triangle_ABC_angles_l512_512954


namespace smoking_related_chronic_bronchitis_l512_512723

theorem smoking_related_chronic_bronchitis :
  (confidence_more_than_99_pct : ∀ (S : Set Person), S ≠ ∅ → smokingRelatedToChronicBronchitisWithConfidence S 0.99) →
  (∃ (smokers : Set Person), Set.count smokers 100) →
  (∃ (smokersWithoutChronicBronchitis : Set Person), 
    smokersWithoutChronicBronchitis ⊆ smokers ∧ Set.count smokersWithoutChronicBronchitis 100) :=
by 
  sorry


end smoking_related_chronic_bronchitis_l512_512723


namespace stratified_sampling_grade12_l512_512375

noncomputable def grade12_ratio : ℚ := 3 / 10

def total_sample_size : ℕ := 200

def expected_grade12_students (ratio : ℚ) (total : ℕ) : ℕ :=
  (ratio * total).to_nat

theorem stratified_sampling_grade12 : expected_grade12_students grade12_ratio total_sample_size = 60 :=
  sorry

end stratified_sampling_grade12_l512_512375


namespace center_of_circle_C_min_tangent_length_l512_512907

noncomputable def line_parametric (t : ℝ) : ℝ × ℝ := 
  ( √2 / 2 * t, √2 / 2 * t + 4 * √2 )

noncomputable def circle_polar (θ : ℝ) : ℝ := 
  2 * cos (θ + π / 4)

theorem center_of_circle_C :
  let C_center := ( √2 / 2, -√2 / 2 )
  true := by
  -- proof to be filled later
  sorry

theorem min_tangent_length :
  let C_center := ( √2 / 2, -√2 / 2 )
  let l_equation := λ (x y : ℝ) => x - y + 4 * √2 = 0
  let distance := (|√2 / 2 + √2 / 2 + 4 * √2|) / √2
  let radius := 1
  (distance - radius) ^ 2 = (2 * √6) ^ 2 := by
  -- proof to be filled later
  sorry

end center_of_circle_C_min_tangent_length_l512_512907


namespace quentavious_gum_count_l512_512206

def initial_nickels : Nat := 5
def remaining_nickels : Nat := 2
def gum_per_nickel : Nat := 2
def traded_nickels (initial remaining : Nat) : Nat := initial - remaining
def total_gum (trade_n gum_per_n : Nat) : Nat := trade_n * gum_per_n

theorem quentavious_gum_count : total_gum (traded_nickels initial_nickels remaining_nickels) gum_per_nickel = 6 := by
  sorry

end quentavious_gum_count_l512_512206


namespace least_upper_bound_IO_l512_512977

variable {ABC : Type} [Triangle ABC] -- Define the triangle ABC
variable (I O : ABC) -- Define incenter I and circumcenter O
variable (R : ℝ) -- Define the circumradius R of the triangle

-- Hypotheses: 
-- 1. I is an incenter
noncomputable def isIncenter (I : ABC) : Prop := sorry 
-- 2. O is a circumcenter
noncomputable def isCircumcenter (O : ABC) : Prop := sorry 
-- 3. Circumradius is R
noncomputable def circumradius (R : ℝ) : Prop := sorry 

-- Theorem: The least upper bound of all possible values of IO is R
theorem least_upper_bound_IO :
  isIncenter I → isCircumcenter O → circumradius R → ∀ I O, IO < R :=
by
  sorry

end least_upper_bound_IO_l512_512977


namespace monochromatic_possible_l512_512629

def bead (color : Type) : Type := color

inductive Color
| Red : Color
| Blue : Color

def adjacent_same_color (necklace : List Color) (i : ℕ) : Prop :=
  (necklace[i] = necklace[(i + 1) % necklace.length]) ∧ (necklace[(i + 1) % necklace.length] = necklace[(i + 2) % necklace.length])

def can_repaint (necklace : List Color) (i : ℕ) : List Color :=
  if adjacent_same_color necklace i then
    necklace.set i (if necklace[i] = Color.Red then Color.Blue else Color.Red)
  else
    necklace

noncomputable def always_mono_bead (n : ℕ) : Prop :=
  ∀ necklace : List Color, necklace.length = n → ∃ m : ℕ, m > 0 ∧
    ∀ i : ℕ, 
      let final_necklace := (List.range m).foldl (λ acc _, can_repaint acc i) necklace
      in final_necklace.all (λ x, x = final_necklace.head!)

theorem monochromatic_possible (n : ℕ) : n > 3 → (n % 2 = 1) → always_mono_bead n := 
by
  sorry

end monochromatic_possible_l512_512629


namespace not_center_of_symmetry_neg_pi_over_18_l512_512667

def is_center_of_symmetry (x : ℝ) : Prop :=
  ∃ k : ℤ, x = (k * π) / 6 - π / 9

theorem not_center_of_symmetry_neg_pi_over_18 :
  ¬ is_center_of_symmetry (-π / 18) := 
sorry

end not_center_of_symmetry_neg_pi_over_18_l512_512667


namespace function_range_l512_512137

theorem function_range (x : ℝ) :
  (sqrt x) / (x - 1) ∈ set.range (λ y, sqrt y / (y - 1)) → (x ≥ 0 ∧ x ≠ 1) := 
sorry

end function_range_l512_512137


namespace pirate_prob_l512_512371

def probability_treasure_no_traps := 1 / 3
def probability_traps_no_treasure := 1 / 6
def probability_neither := 1 / 2

theorem pirate_prob : (70 : ℝ) * ((1 / 3)^4 * (1 / 2)^4) = 35 / 648 := by
  sorry

end pirate_prob_l512_512371


namespace joe_height_l512_512217

theorem joe_height (S J : ℕ) (h1 : S + J = 120) (h2 : J = 2 * S + 6) : J = 82 :=
by
  sorry

end joe_height_l512_512217


namespace expression_equals_l512_512414

theorem expression_equals (numerator: ℕ → ℕ) (denominator: ℕ → ℕ) (n : ℕ) (m : ℕ) (hmn : n > m):
  (∀ i, i > 0 → numerator i = 1 + m + (i - 1)) →
  (∀ j, j > 0 → j ≤ m → denominator j = 1 + n + (j - 1)) →
   (finset.prod (finset.range (m + 1)) numerator) /
   (finset.prod (finset.range (n - m + 1)) denominator) =
    (nat.fact n / (nat.fact m * nat.fact (n - m))) ^ 2 :=
by
  sorry

end expression_equals_l512_512414


namespace solve_equation_l512_512235

theorem solve_equation : (x : ℝ) (hx : x ≠ 1) (h : 3 / (x - 1) = 5 + 3 * x / (1 - x)) : x = 4 :=
sorry

end solve_equation_l512_512235


namespace cricket_game_runs_l512_512564

theorem cricket_game_runs (x : ℕ) (h₁ : 3.2 * x + 250 = 282) : x = 10 :=
by
  sorry

end cricket_game_runs_l512_512564
