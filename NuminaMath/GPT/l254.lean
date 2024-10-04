import Mathlib

namespace combined_weight_of_daughter_and_child_l254_254882

theorem combined_weight_of_daughter_and_child 
  (G D C : ℝ)
  (h1 : G + D + C = 110)
  (h2 : C = 1/5 * G)
  (h3 : D = 50) :
  D + C = 60 :=
sorry

end combined_weight_of_daughter_and_child_l254_254882


namespace spadesuit_problem_l254_254618

def spadesuit (x y : ℝ) := (x + y) * (x - y)

theorem spadesuit_problem : spadesuit 5 (spadesuit 3 2) = 0 := by
  sorry

end spadesuit_problem_l254_254618


namespace count_three_digit_prime_integers_l254_254221

def prime_digits : List ℕ := [2, 3, 5, 7]

def is_three_digit_prime_integer (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000 ∧ (∀ d ∈ List.ofDigits 10 (Nat.digits 10 n), d ∈ prime_digits)

theorem count_three_digit_prime_integers : ∃! n, n = 64 ∧
  (∃ f : Fin 3 → ℕ, ∀ i : Fin 3, f i ∈ prime_digits ∧
  Nat.ofDigits 10 (List.map f ([2, 1, 0].map (Nat.pow 10))) = n) :=
begin
  sorry
end

end count_three_digit_prime_integers_l254_254221


namespace problem_final_value_l254_254507

theorem problem_final_value (x y z : ℝ) (hz : z ≠ 0) 
  (h1 : 3 * x - 2 * y - 2 * z = 0) 
  (h2 : x - 4 * y + 8 * z = 0) :
  (3 * x^2 - 2 * x * y) / (y^2 + 4 * z^2) = 120 / 269 := 
by 
  sorry

end problem_final_value_l254_254507


namespace evaluate_expression_l254_254484

-- Define the expression and the expected result
def expression := -(14 / 2 * 9 - 60 + 3 * 9)
def expectedResult := -30

-- The theorem that states the equivalence
theorem evaluate_expression : expression = expectedResult := by
  sorry

end evaluate_expression_l254_254484


namespace genetic_recombination_does_not_occur_during_dna_replication_l254_254127

-- Definitions based on conditions
def dna_replication_spermatogonial_cells : Prop := 
  ∃ dna_interphase: Prop, ∃ dna_unwinding: Prop, 
    ∃ gene_mutation: Prop, ∃ protein_synthesis: Prop,
      dna_interphase ∧ dna_unwinding ∧ gene_mutation ∧ protein_synthesis

def genetic_recombination_not_occur : Prop :=
  ¬ ∃ genetic_recombination: Prop, genetic_recombination

-- Proof problem statement
theorem genetic_recombination_does_not_occur_during_dna_replication : 
  dna_replication_spermatogonial_cells → genetic_recombination_not_occur :=
by sorry

end genetic_recombination_does_not_occur_during_dna_replication_l254_254127


namespace smallest_of_powers_l254_254295

theorem smallest_of_powers :
  (2:ℤ)^(55) < (3:ℤ)^(44) ∧ (2:ℤ)^(55) < (5:ℤ)^(33) ∧ (2:ℤ)^(55) < (6:ℤ)^(22) := by
  sorry

end smallest_of_powers_l254_254295


namespace inequality_three_variables_l254_254660

theorem inequality_three_variables (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (h : x + y + z = 1) : 
  (1/x) + (1/y) + (1/z) ≥ 9 := 
by 
  sorry

end inequality_three_variables_l254_254660


namespace mike_total_work_time_l254_254394

theorem mike_total_work_time :
  let wash_time := 10
  let oil_change_time := 15
  let tire_change_time := 30
  let paint_time := 45
  let engine_service_time := 60

  let num_wash := 9
  let num_oil_change := 6
  let num_tire_change := 2
  let num_paint := 4
  let num_engine_service := 3
  
  let total_minutes := 
        num_wash * wash_time +
        num_oil_change * oil_change_time +
        num_tire_change * tire_change_time +
        num_paint * paint_time +
        num_engine_service * engine_service_time

  let total_hours := total_minutes / 60

  total_hours = 10 :=
  by
    -- Definitions of times per task
    let wash_time := 10
    let oil_change_time := 15
    let tire_change_time := 30
    let paint_time := 45
    let engine_service_time := 60

    -- Definitions of number of tasks performed
    let num_wash := 9
    let num_oil_change := 6
    let num_tire_change := 2
    let num_paint := 4
    let num_engine_service := 3

    -- Calculate total minutes
    let total_minutes := 
      num_wash * wash_time +
      num_oil_change * oil_change_time +
      num_tire_change * tire_change_time +
      num_paint * paint_time +
      num_engine_service * engine_service_time
    
    -- Calculate total hours
    let total_hours := total_minutes / 60

    -- Required equality to prove
    have : total_hours = 10 := sorry
    exact this

end mike_total_work_time_l254_254394


namespace xiao_zhao_physical_education_grade_l254_254709

def classPerformanceScore : ℝ := 40
def midtermExamScore : ℝ := 50
def finalExamScore : ℝ := 45

def classPerformanceWeight : ℝ := 0.3
def midtermExamWeight : ℝ := 0.2
def finalExamWeight : ℝ := 0.5

def overallGrade : ℝ :=
  (classPerformanceScore * classPerformanceWeight) +
  (midtermExamScore * midtermExamWeight) +
  (finalExamScore * finalExamWeight)

theorem xiao_zhao_physical_education_grade : overallGrade = 44.5 := by
  sorry

end xiao_zhao_physical_education_grade_l254_254709


namespace max_x_plus_y_l254_254366

theorem max_x_plus_y (x y : ℝ) (h : x^2 + y^2 + x * y = 1) : x + y ≤ 2 * Real.sqrt (3) / 3 :=
sorry

end max_x_plus_y_l254_254366


namespace moles_of_NaCl_formed_l254_254029

-- Given conditions
def sodium_bisulfite_moles : ℕ := 2
def hydrochloric_acid_moles : ℕ := 2
def balanced_reaction : Prop :=
  ∀ (NaHSO3 HCl NaCl H2O SO2 : ℕ), 
    NaHSO3 + HCl = NaCl + H2O + SO2

-- Target to prove:
theorem moles_of_NaCl_formed :
  balanced_reaction → sodium_bisulfite_moles = hydrochloric_acid_moles → 
  sodium_bisulfite_moles = 2 := 
sorry

end moles_of_NaCl_formed_l254_254029


namespace period_of_trig_sum_l254_254452

theorem period_of_trig_sum : ∀ x : ℝ, 2 * Real.sin x + 3 * Real.cos x = 2 * Real.sin (x + 2 * Real.pi) + 3 * Real.cos (x + 2 * Real.pi) := 
sorry

end period_of_trig_sum_l254_254452


namespace cell_cycle_correct_statement_l254_254155

theorem cell_cycle_correct_statement :
  ∃ (correct_statement : String), correct_statement = "In the cell cycle, chromatin DNA is easier to replicate than chromosome DNA" :=
by
  let A := "The separation of alleles occurs during the interphase of the cell cycle"
  let B := "In the cell cycle of plant cells, spindle fibers appear during the interphase"
  let C := "In the cell cycle, chromatin DNA is easier to replicate than chromosome DNA"
  let D := "In the cell cycle of liver cells, chromosomes exist for a longer time than chromatin"
  existsi C
  sorry

end cell_cycle_correct_statement_l254_254155


namespace cost_of_gravelling_path_eq_630_l254_254001

-- Define the dimensions of the grassy plot.
def length_grassy_plot : ℝ := 110
def width_grassy_plot : ℝ := 65

-- Define the width of the gravel path.
def width_gravel_path : ℝ := 2.5

-- Define the cost of gravelling per square meter in INR.
def cost_per_sqm : ℝ := 0.70

-- Compute the dimensions of the plot including the gravel path.
def length_including_path := length_grassy_plot + 2 * width_gravel_path
def width_including_path := width_grassy_plot + 2 * width_gravel_path

-- Compute the area of the plot including the gravel path.
def area_including_path := length_including_path * width_including_path

-- Compute the area of the grassy plot without the gravel path.
def area_grassy_plot := length_grassy_plot * width_grassy_plot

-- Compute the area of the gravel path alone.
def area_gravel_path := area_including_path - area_grassy_plot

-- Compute the total cost of gravelling the path.
def total_cost := area_gravel_path * cost_per_sqm

-- The theorem stating the cost of gravelling the path.
theorem cost_of_gravelling_path_eq_630 : total_cost = 630 := by
  -- Proof goes here
  sorry

end cost_of_gravelling_path_eq_630_l254_254001


namespace M_subset_N_l254_254792

def M : Set ℝ := {x | ∃ k : ℤ, x = (k / 2) * 180 + 45}
def N : Set ℝ := {x | ∃ k : ℤ, x = (k / 4) * 180 + 45}

theorem M_subset_N : M ⊆ N :=
sorry

end M_subset_N_l254_254792


namespace soccer_team_wins_l254_254267

theorem soccer_team_wins :
  ∃ W D : ℕ, 
    (W + 2 + D = 20) ∧  -- total games
    (3 * W + D = 46) ∧  -- total points
    (W = 14) :=         -- correct answer
by
  sorry

end soccer_team_wins_l254_254267


namespace negation_proposition_l254_254275

theorem negation_proposition (l : ℝ) (h : l = 1) : 
  (¬ ∃ x : ℝ, x + l ≥ 0) = (∀ x : ℝ, x + l < 0) := by 
  sorry

end negation_proposition_l254_254275


namespace johns_average_speed_l254_254237

def continuous_driving_duration (start_time end_time : ℝ) (distance : ℝ) : Prop :=
start_time = 10.5 ∧ end_time = 14.75 ∧ distance = 190

theorem johns_average_speed
  (start_time end_time : ℝ) 
  (distance : ℝ)
  (h : continuous_driving_duration start_time end_time distance) :
  (distance / (end_time - start_time) = 44.7) :=
by
  sorry

end johns_average_speed_l254_254237


namespace distinct_real_solutions_exist_l254_254031

theorem distinct_real_solutions_exist (a : ℝ) (h : a > 3 / 4) : 
  ∃ (x y : ℝ), x ≠ y ∧ x = a - y^2 ∧ y = a - x^2 := 
sorry

end distinct_real_solutions_exist_l254_254031


namespace mary_talking_ratio_l254_254663

theorem mary_talking_ratio:
  let mac_download_time := 10
  let windows_download_time := 3 * mac_download_time
  let audio_glitch_time := 2 * 4
  let video_glitch_time := 6
  let total_glitch_time := audio_glitch_time + video_glitch_time
  let total_download_time := mac_download_time + windows_download_time
  let total_time := 82
  let talking_time := total_time - total_download_time
  let talking_time_without_glitch := talking_time - total_glitch_time
  talking_time_without_glitch / total_glitch_time = 2 :=
by
  sorry

end mary_talking_ratio_l254_254663


namespace solve_for_x_l254_254550

-- Define the equation
def equation (x : ℝ) : Prop := (x^2 + 3 * x + 4) / (x + 5) = x + 6

-- Prove that x = -13 / 4 satisfies the equation
theorem solve_for_x : ∃ x : ℝ, equation x ∧ x = -13 / 4 :=
by
  sorry

end solve_for_x_l254_254550


namespace javier_savings_l254_254310

theorem javier_savings (regular_price : ℕ) (discount1 : ℕ) (discount2 : ℕ) : 
  (regular_price = 50) 
  ∧ (discount1 = 40)
  ∧ (discount2 = 50) 
  → (30 = (100 * (regular_price * 3 - (regular_price + (regular_price * (100 - discount1) / 100) + regular_price / 2)) / (regular_price * 3))) :=
by
  intros h
  sorry

end javier_savings_l254_254310


namespace abs_neg_three_l254_254836

noncomputable def abs_val (a : ℤ) : ℤ :=
  if a < 0 then -a else a

theorem abs_neg_three : abs_val (-3) = 3 :=
by
  sorry

end abs_neg_three_l254_254836


namespace quotient_real_iff_quotient_purely_imaginary_iff_l254_254274

variables {a b c d : ℝ} -- Declare real number variables

-- Problem 1: Proving the necessary and sufficient condition for the quotient to be a real number
theorem quotient_real_iff (a b c d : ℝ) : 
  (c ≠ 0 ∨ d ≠ 0) → 
  (∀ i : ℝ, ∃ r : ℝ, a/c = r ∧ b/d = 0) ↔ (a * d - b * c = 0) := 
by sorry -- Proof to be filled in

-- Problem 2: Proving the necessary and sufficient condition for the quotient to be a purely imaginary number
theorem quotient_purely_imaginary_iff (a b c d : ℝ) : 
  (c ≠ 0 ∨ d ≠ 0) → 
  (∀ r : ℝ, ∃ i : ℝ, a/c = 0 ∧ b/d = i) ↔ (a * c + b * d = 0) := 
by sorry -- Proof to be filled in

end quotient_real_iff_quotient_purely_imaginary_iff_l254_254274


namespace almond_butter_servings_l254_254712

noncomputable def servings_in_container (total_tbsps : ℚ) (serving_size : ℚ) : ℚ :=
  total_tbsps / serving_size

theorem almond_butter_servings :
  servings_in_container (34 + 3/5) (5 + 1/2) = 6 + 21/55 :=
by
  sorry

end almond_butter_servings_l254_254712


namespace line_up_ways_l254_254382

theorem line_up_ways (n : ℕ) (h : n = 5) :
  let categories := ((range n).filter (λ x, x ≠ 0 ∧ x ≠ (n - 1))) in
  categories.length * fact (n - 1) = 72 :=
by
  rw h
  let categories := ((range 5).filter (λ x, x ≠ 0 ∧ x ≠ (5 - 1)))
  have h_cat_len : categories.length = 3 := by decide
  rw [h_cat_len, fact]
  norm_num
  sorry

end line_up_ways_l254_254382


namespace functional_equation_solution_l254_254017

noncomputable def f : ℝ → ℝ := sorry

theorem functional_equation_solution :
  (∀ x y : ℝ, f (x + y) * f (x - y) = (f x + f y)^2 - 4 * x * y * f y) →
  (∀ x : ℝ, f x = 0 ∨ f x = x^2) := by
  intro h
  sorry

end functional_equation_solution_l254_254017


namespace sum_of_repeating_decimals_l254_254751

/-- Definitions of the repeating decimals as real numbers. --/
def x : ℝ := 0.3 -- This actually represents 0.\overline{3} in Lean
def y : ℝ := 0.04 -- This actually represents 0.\overline{04} in Lean
def z : ℝ := 0.005 -- This actually represents 0.\overline{005} in Lean

/-- The theorem stating that the sum of these repeating decimals is a specific fraction. --/
theorem sum_of_repeating_decimals : x + y + z = (14 : ℝ) / 37 := 
by 
  sorry -- Placeholder for the proof

end sum_of_repeating_decimals_l254_254751


namespace area_of_trapezium_is_105_l254_254489

-- Define points in the coordinate plane
structure Point :=
  (x : ℝ)
  (y : ℝ)

def A : Point := ⟨2, 3⟩
def B : Point := ⟨14, 3⟩
def C : Point := ⟨18, 10⟩
def D : Point := ⟨0, 10⟩

noncomputable def length (p1 p2 : Point) : ℝ := abs (p2.x - p1.x)
noncomputable def height (p1 p2 : Point) : ℝ := abs (p2.y - p1.y)

-- Calculate lengths of parallel sides AB and CD, and height
noncomputable def AB := length A B
noncomputable def CD := length C D
noncomputable def heightAC := height A C

-- Define the area of trapezium
noncomputable def area_trapezium (AB CD height : ℝ) : ℝ := (1/2) * (AB + CD) * height

-- The proof problem statement
theorem area_of_trapezium_is_105 :
  area_trapezium AB CD heightAC = 105 := by
  sorry

end area_of_trapezium_is_105_l254_254489


namespace individual_weight_l254_254804

def total_students : ℕ := 1500
def sampled_students : ℕ := 100

def individual := "the weight of each student"

theorem individual_weight :
  (total_students = 1500) →
  (sampled_students = 100) →
  individual = "the weight of each student" :=
by
  intros h1 h2
  sorry

end individual_weight_l254_254804


namespace final_result_l254_254450

-- define the initial matrix M
def M : Matrix (Fin 3) (Fin 3) ℤ :=
  ![
    ![53, 158, 53],
    ![23, 93, 53],
    ![50, 170, 53]
  ]

-- define the equivalence conditions
def cond_1 : 53 % 2 = 1 := by norm_num
def cond_2 : 53 % 3 = 2 := by norm_num
def cond_3 : 53 % 5 = 3 := by norm_num
def cond_4 : 53 % 7 = 4 := by norm_num

-- define the equivalence of Z_210 with Z_2 × Z_3 × Z_5 × Z_7
def Z210_eq : ZMod 210 ≃ ZMod 2 × ZMod 3 × ZMod 5 × ZMod 7 := sorry

-- Combining modulo operations.
def M_mod_2 : Matrix (Fin 3) (Fin 3) (ZMod 2) := M.map (fun x => x % 2)
def M_mod_3 : Matrix (Fin 3) (Fin 3) (ZMod 3) := M.map (fun x => x % 3)
def M_mod_5 : Matrix (Fin 3) (Fin 3) (ZMod 5) := M.map (fun x => x % 5)
def M_mod_7 : Matrix (Fin 3) (Fin 3) (ZMod 7) := M.map (fun x => x % 7)

-- prove that the final result is 1234
theorem final_result : (M_mod_2, M_mod_3, M_mod_5, M_mod_7) = 1234 := sorry

end final_result_l254_254450


namespace total_jelly_beans_l254_254532

theorem total_jelly_beans (vanilla_jelly_beans : ℕ) (h1 : vanilla_jelly_beans = 120) 
                           (grape_jelly_beans : ℕ) (h2 : grape_jelly_beans = 5 * vanilla_jelly_beans + 50) :
                           vanilla_jelly_beans + grape_jelly_beans = 770 :=
by
  sorry

end total_jelly_beans_l254_254532


namespace combination_property_l254_254970

theorem combination_property (x : ℕ) (hx : 2 * x - 1 ≤ 11 ∧ x ≤ 11) :
  (Nat.choose 11 (2 * x - 1) = Nat.choose 11 x) → (x = 1 ∨ x = 4) :=
by
  sorry

end combination_property_l254_254970


namespace toothpicks_needed_base_1001_l254_254513

-- Define the number of small triangles at the base of the larger triangle
def base_triangle_count := 1001

-- Define the total number of small triangles using the sum of the first 'n' natural numbers
def total_small_triangles (n : ℕ) : ℕ :=
  (n * (n + 1)) / 2

-- Calculate the total number of sides for all triangles if there was no sharing
def total_sides (n : ℕ) : ℕ :=
  3 * total_small_triangles n

-- Calculate the number of shared toothpicks
def shared_toothpicks (n : ℕ) : ℕ :=
  total_sides n / 2

-- Calculate the number of unshared perimeter toothpicks
def unshared_perimeter_toothpicks (n : ℕ) : ℕ :=
  3 * n

-- Calculate the total number of toothpicks required
def total_toothpicks (n : ℕ) : ℕ :=
  shared_toothpicks n + unshared_perimeter_toothpicks n

-- Prove that the total toothpicks required for the base of 1001 small triangles is 755255
theorem toothpicks_needed_base_1001 : total_toothpicks base_triangle_count = 755255 :=
by {
  sorry
}

end toothpicks_needed_base_1001_l254_254513


namespace arctan_sum_pi_div_two_l254_254900

theorem arctan_sum_pi_div_two :
  let α := Real.arctan (3 / 4)
  let β := Real.arctan (4 / 3)
  α + β = Real.pi / 2 := by
  sorry

end arctan_sum_pi_div_two_l254_254900


namespace score_stability_l254_254646

theorem score_stability (mean_A mean_B : ℝ) (h_mean_eq : mean_A = mean_B)
  (variance_A variance_B : ℝ) (h_variance_A : variance_A = 0.06) (h_variance_B : variance_B = 0.35) :
  variance_A < variance_B :=
by
  -- Theorem statement and conditions sufficient to build successfully
  sorry

end score_stability_l254_254646


namespace b_2023_equals_one_fifth_l254_254997

theorem b_2023_equals_one_fifth (b : ℕ → ℚ) (h1 : b 1 = 4) (h2 : b 2 = 5)
    (h_rec : ∀ (n : ℕ), n ≥ 3 → b n = b (n - 1) / b (n - 2)) :
    b 2023 = 1 / 5 := by
  sorry

end b_2023_equals_one_fifth_l254_254997


namespace cubic_inequality_l254_254129

theorem cubic_inequality 
  (x y z : ℝ) 
  (hx : 0 ≤ x ∧ x ≤ 1) 
  (hy : 0 ≤ y ∧ y ≤ 1) 
  (hz : 0 ≤ z ∧ z ≤ 1) :
  2 * (x^3 + y^3 + z^3) - (x^2 * y + y^2 * z + z^2 * x) ≤ 3 :=
by
  sorry

end cubic_inequality_l254_254129


namespace prime_factorization_count_l254_254633

theorem prime_factorization_count :
  (∃ (S : Finset ℕ), S = {97, 101, 2, 13, 107, 109} ∧ S.card = 6) :=
by
  sorry

end prime_factorization_count_l254_254633


namespace scientific_notation_example_l254_254215

theorem scientific_notation_example : 0.00001 = 1 * 10^(-5) :=
sorry

end scientific_notation_example_l254_254215


namespace sum_of_repeating_decimals_l254_254753

/-- Definitions of the repeating decimals as real numbers. --/
def x : ℝ := 0.3 -- This actually represents 0.\overline{3} in Lean
def y : ℝ := 0.04 -- This actually represents 0.\overline{04} in Lean
def z : ℝ := 0.005 -- This actually represents 0.\overline{005} in Lean

/-- The theorem stating that the sum of these repeating decimals is a specific fraction. --/
theorem sum_of_repeating_decimals : x + y + z = (14 : ℝ) / 37 := 
by 
  sorry -- Placeholder for the proof

end sum_of_repeating_decimals_l254_254753


namespace value_of_g_g_2_l254_254357

def g (x : ℝ) : ℝ := 4 * x^2 + 3

theorem value_of_g_g_2 : g (g 2) = 1447 := by
  sorry

end value_of_g_g_2_l254_254357


namespace triangle_area_example_l254_254598

def Point := (ℝ × ℝ)

def triangle_area (A B C : Point) : ℝ :=
  0.5 * |A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2)|

theorem triangle_area_example :
  triangle_area (-2, 3) (7, -1) (4, 6) = 25.5 :=
by
  -- Proof will be here
  sorry

end triangle_area_example_l254_254598


namespace even_function_maximum_value_l254_254422

noncomputable def f (x : ℝ) : ℝ := cos x - cos (2 * x)

/-- f(x) is an even function. -/
theorem even_function : ∀ x : ℝ, f (-x) = f x :=
by
  intro x
  have h1 : cos (-x) = cos x := cos_neg x
  have h2 : cos (-2 * x) = cos (2 * x) := by rw [neg_mul, cos_neg]
  rw [f, f, h1, h2]

/-- The maximum value of f(x) is 9/8. -/
theorem maximum_value : ∃ x : ℝ, f x = 9 / 8 :=
by
  use real.acos (1 / 4)
  sorry -- detailed proof of maximum value is beyond the scope of this translation

end even_function_maximum_value_l254_254422


namespace functional_equation_g_l254_254104

variable (g : ℝ → ℝ)
variable (f : ℝ)
variable (h : ℝ)

theorem functional_equation_g (H1 : ∀ x y : ℝ, g (x + y) = g x * g y)
                            (H2 : g 3 = 4) :
                            g 6 = 16 := 
by
  sorry

end functional_equation_g_l254_254104


namespace rectangle_area_l254_254362

theorem rectangle_area (l w : ℕ) (h_diagonal : l^2 + w^2 = 17^2) (h_perimeter : l + w = 23) : l * w = 120 :=
by
  sorry

end rectangle_area_l254_254362


namespace upstream_speed_is_8_l254_254149

-- Definitions of given conditions
def downstream_speed : ℝ := 13
def stream_speed : ℝ := 2.5
def man's_upstream_speed : ℝ := downstream_speed - 2 * stream_speed

-- Theorem to prove
theorem upstream_speed_is_8 : man's_upstream_speed = 8 :=
by
  rw [man's_upstream_speed, downstream_speed, stream_speed]
  sorry

end upstream_speed_is_8_l254_254149


namespace projection_areas_are_correct_l254_254983

noncomputable def S1 := 1/2 * 2 * 2
noncomputable def S2 := 1/2 * 2 * Real.sqrt 2
noncomputable def S3 := 1/2 * 2 * Real.sqrt 2

theorem projection_areas_are_correct :
  S3 = S2 ∧ S3 ≠ S1 :=
by
  sorry

end projection_areas_are_correct_l254_254983


namespace radius_of_large_circle_l254_254328

/-- Five circles are described with the given properties. -/
def small_circle_radius : ℝ := 2

/-- The angle between any centers of the small circles is 72 degrees due to equal spacing. -/
def angle_between_centers : ℝ := 72

/-- The final theorem states that the radius of the larger circle is as follows. -/
theorem radius_of_large_circle (number_of_circles : ℕ)
        (radius_small : ℝ)
        (angle : ℝ)
        (internally_tangent : ∀ (i : ℕ), i < number_of_circles → Prop)
        (externally_tangent : ∀ (i j : ℕ), i ≠ j → i < number_of_circles → j < number_of_circles → Prop) :
  number_of_circles = 5 →
  radius_small = small_circle_radius →
  angle = angle_between_centers →
  (∃ R : ℝ, R = 4 * Real.sqrt 5 - 2) 
:= by
  -- mathematical proof goes here
  sorry

end radius_of_large_circle_l254_254328


namespace functional_equation_unique_solution_l254_254026

theorem functional_equation_unique_solution (f : ℝ → ℝ) :
  (∀ a b c : ℝ, a + f b + f (f c) = 0 → f a ^ 3 + b * f b ^ 2 + c ^ 2 * f c = 3 * a * b * c) →
  (∀ x : ℝ, f x = x ∨ f x = -x ∨ f x = 0) :=
by
  sorry

end functional_equation_unique_solution_l254_254026


namespace least_possible_xy_l254_254777

theorem least_possible_xy (x y : ℕ) (hx : 0 < x) (hy : 0 < y)
  (h : 1 / x + 1 / (3 * y) = 1 / 6) : x * y = 48 :=
by
  sorry

end least_possible_xy_l254_254777


namespace Eldora_total_cost_l254_254482

-- Conditions
def paper_clip_cost : ℝ := 1.85
def index_card_cost : ℝ := 3.95 -- from Finn's purchase calculation
def total_cost (clips : ℝ) (cards : ℝ) (clip_price : ℝ) (card_price : ℝ) : ℝ :=
  (clips * clip_price) + (cards * card_price)

theorem Eldora_total_cost :
  total_cost 15 7 paper_clip_cost index_card_cost = 55.40 :=
by
  sorry

end Eldora_total_cost_l254_254482


namespace interval_intersection_l254_254175

/--
  This statement asserts that the intersection of the intervals (2/4, 3/4) and (2/5, 3/5)
  results in the interval (1/2, 0.6), which is the solution to the problem.
-/
theorem interval_intersection :
  { x : ℝ | 2 < 4 * x ∧ 4 * x < 3 ∧ 2 < 5 * x ∧ 5 * x < 3 } = { x : ℝ | 0.5 < x ∧ x < 0.6 } :=
by
  sorry

end interval_intersection_l254_254175


namespace lemons_for_lemonade_l254_254464

theorem lemons_for_lemonade (lemons_gallons_ratio : 30 / 25 = x / 10) : x = 12 :=
by
  sorry

end lemons_for_lemonade_l254_254464


namespace f_g_g_f_l254_254496

noncomputable def f (x: ℝ) := 1 - 2 * x
noncomputable def g (x: ℝ) := x^2 + 3

theorem f_g (x : ℝ) : f (g x) = -2 * x^2 - 5 :=
by
  sorry

theorem g_f (x : ℝ) : g (f x) = 4 * x^2 - 4 * x + 4 :=
by
  sorry

end f_g_g_f_l254_254496


namespace smaller_group_men_l254_254972

theorem smaller_group_men (M : ℕ) (h1 : 36 * 25 = M * 90) : M = 10 :=
by
  -- Here we would provide the proof. Unfortunately, proving this in Lean 4 requires knowledge of algebra.
  sorry

end smaller_group_men_l254_254972


namespace xy_relationship_l254_254317

theorem xy_relationship : 
  (∀ x y, (x = 1 ∧ y = 1) ∨ (x = 2 ∧ y = 4) ∨ (x = 3 ∧ y = 9) ∨ (x = 4 ∧ y = 16) ∨ (x = 5 ∧ y = 25) 
  → y = x * x) :=
by {
  sorry
}

end xy_relationship_l254_254317


namespace unique_solution_mod_37_system_l254_254030

theorem unique_solution_mod_37_system :
  ∃! (a b c d : ℤ), 
  (a^2 + b * c ≡ a [ZMOD 37]) ∧
  (b * (a + d) ≡ b [ZMOD 37]) ∧
  (c * (a + d) ≡ c [ZMOD 37]) ∧
  (b * c + d^2 ≡ d [ZMOD 37]) ∧
  (a * d - b * c ≡ 1 [ZMOD 37]) :=
sorry

end unique_solution_mod_37_system_l254_254030


namespace find_m_l254_254861

theorem find_m (m : ℤ) (y : ℤ) : 
  (y^2 + m * y + 2) % (y - 1) = (m + 3) ∧ 
  (y^2 + m * y + 2) % (y + 1) = (3 - m) ∧
  (m + 3 = 3 - m) → m = 0 :=
sorry

end find_m_l254_254861


namespace prime_ge_5_divisible_by_12_l254_254245

theorem prime_ge_5_divisible_by_12 (p : ℕ) (hp1 : p ≥ 5) (hp2 : Nat.Prime p) : 12 ∣ p^2 - 1 :=
by
  sorry

end prime_ge_5_divisible_by_12_l254_254245


namespace remainder_of_sum_divided_by_14_l254_254119

def consecutive_odds : List ℤ := [12157, 12159, 12161, 12163, 12165, 12167, 12169]

def sum_of_consecutive_odds := consecutive_odds.sum

theorem remainder_of_sum_divided_by_14 :
  (sum_of_consecutive_odds % 14) = 7 := by
  sorry

end remainder_of_sum_divided_by_14_l254_254119


namespace person_b_days_work_alone_l254_254257

theorem person_b_days_work_alone (B : ℕ) (h1 : (1 : ℚ) / 40 + 1 / B = 1 / 24) : B = 60 := 
by
  sorry

end person_b_days_work_alone_l254_254257


namespace repeating_decimals_product_fraction_l254_254024

theorem repeating_decimals_product_fraction : 
  let x := 1 / 33
  let y := 9 / 11
  x * y = 9 / 363 := 
by
  sorry

end repeating_decimals_product_fraction_l254_254024


namespace find_x_value_l254_254388

theorem find_x_value (x : ℝ) (h : 150 + 90 + x + 90 = 360) : x = 30 := by
  sorry

end find_x_value_l254_254388


namespace smallest_positive_integer_remainder_conditions_l254_254123

theorem smallest_positive_integer_remainder_conditions :
  ∃ b : ℕ, (b % 4 = 3 ∧ b % 6 = 5) ∧ ∀ n : ℕ, (n % 4 = 3 ∧ n % 6 = 5) → n ≥ b := 
by
  have b := 23
  use b
  sorry

end smallest_positive_integer_remainder_conditions_l254_254123


namespace range_of_x_l254_254736

noncomputable def integerPart (x : ℝ) : ℤ := Int.floor x

theorem range_of_x (x : ℝ) (h : integerPart ((1 - 3 * x) / 2) = -1) : (1 / 3) < x ∧ x ≤ 1 :=
by
  sorry

end range_of_x_l254_254736


namespace math_problem_l254_254971

theorem math_problem (a b : ℝ) (h : |a + 1| + (b - 2)^2 = 0) : (a + b)^9 + a^6 = 2 :=
sorry

end math_problem_l254_254971


namespace hexagon_perimeter_is_24_l254_254389

-- Conditions given in the problem
def AB : ℝ := 3
def EF : ℝ := 3
def BE : ℝ := 4
def AF : ℝ := 4
def CD : ℝ := 5
def DF : ℝ := 5

-- Statement to show that the perimeter is 24 units
theorem hexagon_perimeter_is_24 :
  AB + BE + CD + DF + EF + AF = 24 :=
by
  sorry

end hexagon_perimeter_is_24_l254_254389


namespace wheel_sum_even_and_greater_than_10_l254_254693

-- Definitions based on conditions
def prob_even_A : ℚ := 3 / 8
def prob_odd_A : ℚ := 5 / 8
def prob_even_B : ℚ := 1 / 4
def prob_odd_B : ℚ := 3 / 4

-- Event probabilities from solution steps
def prob_both_even : ℚ := prob_even_A * prob_even_B
def prob_both_odd : ℚ := prob_odd_A * prob_odd_B
def prob_even_sum : ℚ := prob_both_even + prob_both_odd
def prob_even_sum_greater_10 : ℚ := 1 / 3

-- Compute final probability
def final_probability : ℚ := prob_even_sum * prob_even_sum_greater_10

-- The statement that needs proving
theorem wheel_sum_even_and_greater_than_10 : final_probability = 3 / 16 := by
  sorry

end wheel_sum_even_and_greater_than_10_l254_254693


namespace part1_part2_l254_254042

noncomputable def f (x : ℝ) : ℝ := (2 * x - 1) / (x + 1)

theorem part1 : ∀ x1 x2 : ℝ, 0 ≤ x1 → 0 ≤ x2 → x1 < x2 → f x1 < f x2 := by
  sorry

theorem part2 (m : ℝ) : 1 ≤ m →
  (∀ x : ℝ, 1 ≤ x → x ≤ m → f x ≤ f m) ∧ 
  (∀ x : ℝ, 1 ≤ x → x ≤ m → f 1 ≤ f x) →
  f m - f 1 = 1 / 2 →
  m = 2 := by
  sorry

end part1_part2_l254_254042


namespace geese_count_l254_254687

theorem geese_count (initial : ℕ) (flown_away : ℕ) (left : ℕ) 
  (h₁ : initial = 51) (h₂ : flown_away = 28) : 
  left = initial - flown_away → left = 23 := 
by
  sorry

end geese_count_l254_254687


namespace combined_gross_profit_correct_l254_254880

def calculate_final_selling_price (initial_price : ℝ) (markup : ℝ) (discounts : List ℝ) : ℝ :=
  let marked_up_price := initial_price * (1 + markup)
  let final_price := List.foldl (λ price discount => price * (1 - discount)) marked_up_price discounts
  final_price

def calculate_gross_profit (initial_price : ℝ) (markup : ℝ) (discounts : List ℝ) : ℝ :=
  calculate_final_selling_price initial_price markup discounts - initial_price

noncomputable def combined_gross_profit : ℝ :=
  let earrings_gross_profit := calculate_gross_profit 240 0.25 [0.15]
  let bracelet_gross_profit := calculate_gross_profit 360 0.30 [0.10, 0.05]
  let necklace_gross_profit := calculate_gross_profit 480 0.40 [0.20, 0.05]
  let ring_gross_profit := calculate_gross_profit 600 0.35 [0.10, 0.05, 0.02]
  let pendant_gross_profit := calculate_gross_profit 720 0.50 [0.20, 0.03, 0.07]
  earrings_gross_profit + bracelet_gross_profit + necklace_gross_profit + ring_gross_profit + pendant_gross_profit

theorem combined_gross_profit_correct : combined_gross_profit = 224.97 :=
  by
  sorry

end combined_gross_profit_correct_l254_254880


namespace arctan_sum_pi_div_two_l254_254903

theorem arctan_sum_pi_div_two :
  let α := Real.arctan (3 / 4)
  let β := Real.arctan (4 / 3)
  α + β = Real.pi / 2 := by
  sorry

end arctan_sum_pi_div_two_l254_254903


namespace b3_b7_equals_16_l254_254625

variable {a b : ℕ → ℝ}
variable {d : ℝ}

-- Conditions: a is an arithmetic sequence with common difference d
def is_arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + d

-- Condition: b is a geometric sequence
def is_geometric_sequence (b : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, b (n + 1) = b n * r

-- Given condition on the arithmetic sequence a
def condition_on_a (a : ℕ → ℝ) (d : ℝ) : Prop :=
  2 * a 2 - (a 5) ^ 2 + 2 * a 8 = 0

-- Define the specific arithmetic sequence in terms of d and a5
noncomputable def a_seq (a5 d : ℝ) : ℕ → ℝ
| 0 => a5 - 5 * d
| 1 => a5 - 4 * d
| 2 => a5 - 3 * d
| 3 => a5 - 2 * d
| 4 => a5 - d
| 5 => a5
| 6 => a5 + d
| 7 => a5 + 2 * d
| 8 => a5 + 3 * d
| 9 => a5 + 4 * d
| n => 0 -- extending for unspecified

-- Condition: b_5 = a_5
def b_equals_a (a b : ℕ → ℝ) : Prop :=
  b 5 = a 5

-- Theorem: Given the conditions, prove b_3 * b_7 = 16
theorem b3_b7_equals_16 (a b : ℕ → ℝ) (d : ℝ)
  (ha_seq : is_arithmetic_sequence a d)
  (hb_seq : is_geometric_sequence b)
  (h_cond_a : condition_on_a a d)
  (h_b_equals_a : b_equals_a a b) : b 3 * b 7 = 16 :=
by
  sorry

end b3_b7_equals_16_l254_254625


namespace ceil_sqrt_180_eq_14_l254_254935

theorem ceil_sqrt_180_eq_14
  (h : 13 < Real.sqrt 180 ∧ Real.sqrt 180 < 14) :
  Int.ceil (Real.sqrt 180) = 14 :=
  sorry

end ceil_sqrt_180_eq_14_l254_254935


namespace extreme_values_a_4_find_a_minimum_minus_5_l254_254352

noncomputable def f (x a : ℝ) : ℝ := 2 * x^2 - a * x + 5

theorem extreme_values_a_4 :
  (∀ x, x ∈ Set.Icc (-1:ℝ) 2 -> f x 4 ≤ 11) ∧ (∃ x, x ∈ Set.Icc (-1:ℝ) 2 ∧ f x 4 = 11) ∧
  (∀ x, x ∈ Set.Icc (-1:ℝ) 2 -> f x 4 ≥ 3) ∧ (∃ x, x ∈ Set.Icc (-1:ℝ) 2 ∧ f x 4 = 3) :=
  sorry

theorem find_a_minimum_minus_5 :
  ∀ (a : ℝ), (∃ x, x ∈ Set.Icc (-1:ℝ) 2 ∧ f x a = -5) -> (a = -12 ∨ a = 9) :=
  sorry

end extreme_values_a_4_find_a_minimum_minus_5_l254_254352


namespace most_cost_effective_payment_l254_254584

theorem most_cost_effective_payment :
  let worker_days := 5 * 10
  let hourly_rate_per_worker := 8 * 10 * 4
  let paint_cost := 4800
  let area_painted := 150
  let cost_option_1 := worker_days * 30
  let cost_option_2 := paint_cost * 0.30
  let cost_option_3 := area_painted * 12
  let cost_option_4 := 5 * hourly_rate_per_worker
  (cost_option_2 < cost_option_1) ∧ (cost_option_2 < cost_option_3) ∧ (cost_option_2 < cost_option_4) :=
by
  sorry

end most_cost_effective_payment_l254_254584


namespace circle_standard_equation_l254_254493

theorem circle_standard_equation {a : ℝ} :
  (∃ a : ℝ, a ≠ 0 ∧ (a = 2 * a - 3 ∨ a = 3 - 2 * a) ∧ 
  (((x - a)^2 + (y - (2 * a - 3))^2 = a^2) ∧ 
   ((x - 3)^2 + (y - 3)^2 = 9 ∨ (x - 1)^2 + (y + 1)^2 = 1))) :=
sorry

end circle_standard_equation_l254_254493


namespace length_more_than_breadth_by_10_l254_254845

-- Definitions based on conditions
def length : ℕ := 55
def cost_per_meter : ℚ := 26.5
def total_fencing_cost : ℚ := 5300
def perimeter : ℚ := total_fencing_cost / cost_per_meter

-- Calculate breadth (b) and difference (x)
def breadth := 45 -- This is inferred manually from the solution for completeness
def difference (b : ℚ) := length - b

-- The statement we need to prove
theorem length_more_than_breadth_by_10 :
  difference 45 = 10 :=
by
  sorry

end length_more_than_breadth_by_10_l254_254845


namespace chocolate_bar_cost_l254_254710

variable (cost_per_bar num_bars : ℝ)

theorem chocolate_bar_cost (num_scouts smores_per_scout smores_per_bar : ℕ) (total_cost : ℝ)
  (h1 : num_scouts = 15)
  (h2 : smores_per_scout = 2)
  (h3 : smores_per_bar = 3)
  (h4 : total_cost = 15)
  (h5 : num_bars = (num_scouts * smores_per_scout) / smores_per_bar)
  (h6 : total_cost = cost_per_bar * num_bars) :
  cost_per_bar = 1.50 :=
by
  sorry

end chocolate_bar_cost_l254_254710


namespace chess_group_players_l254_254444

theorem chess_group_players (n : ℕ) (h : n * (n - 1) / 2 = 1225) : n = 50 :=
sorry

end chess_group_players_l254_254444


namespace two_point_distribution_properties_l254_254511

open ProbabilityTheory

noncomputable def X : ℕ → ℝ := λ n, if n = 0 then 0 else if n = 1 then 1 else 0

theorem two_point_distribution_properties :
  (PMF.toOuterMeasure (PMF.ofMultiset [0, 1])).volume {1} = (1 / 2) ∧
  PMF.mean (PMF.ofMultiset [0, 1]) = (1 / 2) ∧
  PMF.mean (PMF.map (λ x, 2 * x) (PMF.ofMultiset [0, 1])) ≠ (1 / 2) ∧
  PMF.variance (PMF.ofMultiset [0, 1]) = (1 / 4) :=
by
  sorry

end two_point_distribution_properties_l254_254511


namespace problem_one_problem_two_l254_254898

noncomputable def prob_one_from_each (p : ℝ) (n : ℕ) : ℝ :=
(choose n 1 * p * (1 - p)) * (choose n 1 * p * (1 - p))

noncomputable def prob_at_least_one (p : ℝ) (n : ℕ) : ℝ :=
1 - (1 - p)^n

theorem problem_one:
  prob_one_from_each 0.6 2 = 0.2304 := 
by sorry

theorem problem_two:
  prob_at_least_one 0.6 4 = 0.9744 := 
by sorry

end problem_one_problem_two_l254_254898


namespace find_y_l254_254405

theorem find_y (a b : ℝ) (y : ℝ) (h0 : b ≠ 0) (h1 : (3 * a)^(2 * b) = a^b * y^b) : y = 9 * a := by
  sorry

end find_y_l254_254405


namespace sphere_cylinder_surface_area_difference_l254_254638

theorem sphere_cylinder_surface_area_difference (R : ℝ) :
  let S_sphere := 4 * Real.pi * R^2
  let S_lateral := 4 * Real.pi * R^2
  S_sphere - S_lateral = 0 :=
by
  sorry

end sphere_cylinder_surface_area_difference_l254_254638


namespace samia_walking_distance_l254_254830

theorem samia_walking_distance
  (speed_bike : ℝ)
  (speed_walk : ℝ)
  (total_time : ℝ) 
  (fraction_bike : ℝ) 
  (d : ℝ)
  (walking_distance : ℝ) :
  speed_bike = 15 ∧ 
  speed_walk = 4 ∧ 
  total_time = 1 ∧ 
  fraction_bike = 2/3 ∧ 
  walking_distance = (1/3) * d ∧ 
  (53 * d / 180 = total_time) → 
  walking_distance = 1.1 := 
by 
  sorry

end samia_walking_distance_l254_254830


namespace book_page_count_l254_254728

theorem book_page_count:
  (∃ (total_pages : ℕ), 
    (∃ (days_read : ℕ) (pages_per_day : ℕ), 
      days_read = 12 ∧ 
      pages_per_day = 8 ∧ 
      (days_read * pages_per_day) = 2 * (total_pages / 3)) 
  ↔ total_pages = 144) :=
by 
  sorry

end book_page_count_l254_254728


namespace xiaoli_time_l254_254683

variable {t : ℕ} -- Assuming t is a natural number (time in seconds)

theorem xiaoli_time (record_time : ℕ) (t_non_break : t ≥ record_time) (h : record_time = 14) : t ≥ 14 :=
by
  rw [h] at t_non_break
  exact t_non_break

end xiaoli_time_l254_254683


namespace dave_walk_probability_l254_254928

/-- Prove that with 15 gates arranged 100 feet apart, if a departure gate is 
    initially assigned at random and changed to another gate at random, then
    the probability that the walking distance is 300 feet or less simplifies 
    to the fraction 37/105. Consequently, m + n = 142. --/
theorem dave_walk_probability : 
  ∃ (m n : ℕ), (m + n = 142) ∧ ((74 : ℚ) / 210).num = m ∧ ((74 : ℚ) / 210).denom = n := 
by
  sorry

end dave_walk_probability_l254_254928


namespace Mary_younger_by_14_l254_254309

variable (Betty_age : ℕ) (Albert_age : ℕ) (Mary_age : ℕ)

theorem Mary_younger_by_14 :
  (Betty_age = 7) →
  (Albert_age = 4 * Betty_age) →
  (Albert_age = 2 * Mary_age) →
  (Albert_age - Mary_age = 14) :=
by
  intros
  sorry

end Mary_younger_by_14_l254_254309


namespace single_elimination_games_l254_254643

theorem single_elimination_games (n : ℕ) (h : n = 512) : 
  ∃ g : ℕ, g = n - 1 ∧ g = 511 := 
by
  use n - 1
  sorry

end single_elimination_games_l254_254643


namespace interval_of_x_l254_254168

theorem interval_of_x (x : ℝ) : 
  (2 < 4 * x ∧ 4 * x < 3) ∧ (2 < 5 * x ∧ 5 * x < 3) ↔ (1 / 2 < x ∧ x < 3 / 5) :=
by
  sorry

end interval_of_x_l254_254168


namespace min_large_trucks_needed_l254_254664

-- Define the parameters for the problem
def total_fruit : ℕ := 134
def load_large_truck : ℕ := 15
def load_small_truck : ℕ := 7

-- Define the main theorem to be proved
theorem min_large_trucks_needed :
  ∃ (n : ℕ), n = 8 ∧ (total_fruit = n * load_large_truck + 2 * load_small_truck) :=
by sorry

end min_large_trucks_needed_l254_254664


namespace daily_profit_35_selling_price_for_600_profit_no_900_profit_possible_l254_254877

-- Definitions based on conditions
def purchase_price : ℝ := 30
def max_selling_price : ℝ := 55
def linear_relationship (x : ℝ) : ℝ := -2 * x + 140
def profit (x : ℝ) : ℝ := (x - purchase_price) * linear_relationship x

-- Part 1: Daily profit when selling price is 35 yuan
theorem daily_profit_35 : profit 35 = 350 :=
  sorry

-- Part 2: Selling price for a daily profit of 600 yuan
theorem selling_price_for_600_profit (x : ℝ) (h1 : 30 ≤ x) (h2 : x ≤ 55) : profit x = 600 → x = 40 :=
  sorry

-- Part 3: Possibility of daily profit of 900 yuan
theorem no_900_profit_possible (h1 : ∀ x, 30 ≤ x ∧ x ≤ 55 → profit x ≠ 900) : ¬ ∃ x, 30 ≤ x ∧ x ≤ 55 ∧ profit x = 900 :=
  sorry

end daily_profit_35_selling_price_for_600_profit_no_900_profit_possible_l254_254877


namespace Martha_knitting_grandchildren_l254_254662

theorem Martha_knitting_grandchildren (T_hat T_scarf T_mittens T_socks T_sweater T_total : ℕ)
  (h_hat : T_hat = 2) (h_scarf : T_scarf = 3) (h_mittens : T_mittens = 2)
  (h_socks : T_socks = 3) (h_sweater : T_sweater = 6) (h_total : T_total = 48) :
  (T_total / (T_hat + T_scarf + T_mittens + T_socks + T_sweater)) = 3 := by
  sorry

end Martha_knitting_grandchildren_l254_254662


namespace alpha_value_l254_254772

theorem alpha_value (α : ℝ) (h : 0 ≤ α ∧ α ≤ 2 * Real.pi 
    ∧ ∃β : ℝ, β = 2 * Real.pi / 3 ∧ (Real.sin β, Real.cos β) = (Real.sin α, Real.cos α)) : 
    α = 5 * Real.pi / 3 := 
  by
    sorry

end alpha_value_l254_254772


namespace john_minimum_pizzas_l254_254649

theorem john_minimum_pizzas (car_cost bag_cost earnings_per_pizza gas_cost p : ℕ) 
  (h_car : car_cost = 6000)
  (h_bag : bag_cost = 200)
  (h_earnings : earnings_per_pizza = 12)
  (h_gas : gas_cost = 4)
  (h_p : 8 * p >= car_cost + bag_cost) : p >= 775 := 
sorry

end john_minimum_pizzas_l254_254649


namespace four_units_away_l254_254255

theorem four_units_away (x : ℤ) (h : abs (x + 2) = 4) : x = 2 ∨ x = -6 :=
by
  sorry

end four_units_away_l254_254255


namespace sequence_general_formula_l254_254961

theorem sequence_general_formula (a : ℕ → ℝ) (h₁ : a 1 = 3) 
    (h₂ : ∀ n : ℕ, 1 < n → a n = (n / (n - 1)) * a (n - 1)) : 
    ∀ n : ℕ, 1 ≤ n → a n = 3 * n :=
by
  -- Proof description here
  sorry

end sequence_general_formula_l254_254961


namespace grid_covering_impossible_l254_254202

theorem grid_covering_impossible :
  ∀ (x y : ℕ), x + y = 19 → 6 * x + 7 * y = 132 → False :=
by
  intros x y h1 h2
  -- Proof would go here.
  sorry

end grid_covering_impossible_l254_254202


namespace smallest_positive_integer_remainder_l254_254126

theorem smallest_positive_integer_remainder
  (b : ℕ) (h1 : b % 4 = 3) (h2 : b % 6 = 5) :
  b = 11 := by
  sorry

end smallest_positive_integer_remainder_l254_254126


namespace time_period_is_12_hours_l254_254805

-- Define the conditions in the problem
def birth_rate := 8 / 2 -- people per second
def death_rate := 6 / 2 -- people per second
def net_increase := 86400 -- people

-- Define the net increase per second
def net_increase_per_second := birth_rate - death_rate

-- Total time period in seconds
def time_period_seconds := net_increase / net_increase_per_second

-- Convert the time period to hours
def time_period_hours := time_period_seconds / 3600

-- The theorem we want to state and prove
theorem time_period_is_12_hours : time_period_hours = 12 :=
by
  -- Proof goes here
  sorry

end time_period_is_12_hours_l254_254805


namespace problem1_problem2_problem3_problem4_l254_254262

-- Problem 1: Prove X = 93 given X - 12 = 81
theorem problem1 (X : ℝ) (h : X - 12 = 81) : X = 93 :=
by
  sorry

-- Problem 2: Prove X = 5.4 given 5.1 + X = 10.5
theorem problem2 (X : ℝ) (h : 5.1 + X = 10.5) : X = 5.4 :=
by
  sorry

-- Problem 3: Prove X = 0.7 given 6X = 4.2
theorem problem3 (X : ℝ) (h : 6 * X = 4.2) : X = 0.7 :=
by
  sorry

-- Problem 4: Prove X = 5 given X ÷ 0.4 = 12.5
theorem problem4 (X : ℝ) (h : X / 0.4 = 12.5) : X = 5 :=
by
  sorry

end problem1_problem2_problem3_problem4_l254_254262


namespace repeating_decimals_sum_l254_254748

theorem repeating_decimals_sum : 
  (0.3333333333333333 : ℝ) + (0.0404040404040404 : ℝ) + (0.005005005005005 : ℝ) = (14 / 37 : ℝ) :=
by {
  sorry
}

end repeating_decimals_sum_l254_254748


namespace total_jelly_beans_l254_254534

theorem total_jelly_beans (vanilla_jelly_beans : ℕ) (h1 : vanilla_jelly_beans = 120) 
                           (grape_jelly_beans : ℕ) (h2 : grape_jelly_beans = 5 * vanilla_jelly_beans + 50) :
                           vanilla_jelly_beans + grape_jelly_beans = 770 :=
by
  sorry

end total_jelly_beans_l254_254534


namespace distance_between_points_on_line_l254_254062

theorem distance_between_points_on_line (a b c d m k : ℝ) 
  (hab : b = m * a + k) (hcd : d = m * c + k) :
  dist (a, b) (c, d) = |a - c| * Real.sqrt (1 + m^2) :=
by
  sorry

end distance_between_points_on_line_l254_254062


namespace quadratic_roots_expression_l254_254242

theorem quadratic_roots_expression (x1 x2 : ℝ) (h1 : x1^2 + x1 - 2023 = 0) (h2 : x2^2 + x2 - 2023 = 0) :
  x1^2 + 2*x1 + x2 = 2022 :=
by
  sorry

end quadratic_roots_expression_l254_254242


namespace find_median_of_first_twelve_positive_integers_l254_254578

def median_of_first_twelve_positive_integers : ℚ :=
  let A := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
  (A[5] + A[6]) / 2

theorem find_median_of_first_twelve_positive_integers :
  median_of_first_twelve_positive_integers = 6.5 :=
by
  sorry

end find_median_of_first_twelve_positive_integers_l254_254578


namespace worst_player_is_niece_l254_254890

structure Player where
  name : String
  sex : String
  generation : Nat

def grandmother := Player.mk "Grandmother" "Female" 1
def niece := Player.mk "Niece" "Female" 2
def grandson := Player.mk "Grandson" "Male" 3
def son_in_law := Player.mk "Son-in-law" "Male" 2

def worst_player : Player := niece
def best_player : Player := grandmother

-- Conditions
def cousin_check : worst_player ≠ best_player ∧
                   worst_player.generation ≠ best_player.generation ∧ 
                   worst_player.sex ≠ best_player.sex := 
  by sorry

-- Prove that the worst player is the niece
theorem worst_player_is_niece : worst_player = niece :=
  by sorry

end worst_player_is_niece_l254_254890


namespace polynomial_identity_l254_254820

theorem polynomial_identity :
  (3 * x ^ 2 - 4 * y ^ 3) * (9 * x ^ 4 + 12 * x ^ 2 * y ^ 3 + 16 * y ^ 6) = 27 * x ^ 6 - 64 * y ^ 9 :=
by
  sorry

end polynomial_identity_l254_254820


namespace soccer_league_games_l254_254570

theorem soccer_league_games (n_teams games_played : ℕ) (h1 : n_teams = 10) (h2 : games_played = 45) :
  ∃ k : ℕ, (n_teams * (n_teams - 1)) / 2 = games_played ∧ k = 1 :=
by
  sorry

end soccer_league_games_l254_254570


namespace find_naturals_divisibility_l254_254943

theorem find_naturals_divisibility :
  {n : ℕ | (2^n + n) ∣ (8^n + n)} = {1, 2, 4, 6} :=
by sorry

end find_naturals_divisibility_l254_254943


namespace bill_drew_12_triangles_l254_254311

theorem bill_drew_12_triangles 
  (T : ℕ)
  (total_lines : T * 3 + 8 * 4 + 4 * 5 = 88) : 
  T = 12 :=
sorry

end bill_drew_12_triangles_l254_254311


namespace range_of_x_l254_254738

noncomputable def integer_part (x : ℝ) : ℤ := ⌊x⌋

theorem range_of_x (x : ℝ) (h : integer_part ((1 - 3*x) / 2) = -1) : (1 / 3) < x ∧ x ≤ 1 :=
by sorry

end range_of_x_l254_254738


namespace arctan_triangle_complementary_l254_254904

theorem arctan_triangle_complementary :
  (Real.arctan (3 / 4) + Real.arctan (4 / 3) = Real.pi / 2) :=
begin
  sorry
end

end arctan_triangle_complementary_l254_254904


namespace provisions_last_for_more_days_l254_254145

def initial_men : ℕ := 2000
def initial_days : ℕ := 65
def additional_men : ℕ := 3000
def days_used : ℕ := 15
def remaining_provisions :=
  initial_men * initial_days - initial_men * days_used
def total_men_after_reinforcement := initial_men + additional_men
def remaining_days := remaining_provisions / total_men_after_reinforcement

theorem provisions_last_for_more_days :
  remaining_days = 20 := by
  sorry

end provisions_last_for_more_days_l254_254145


namespace Sarah_is_26_l254_254260

noncomputable def Sarah_age (mark_age billy_age ana_age : ℕ): ℕ :=
  3 * mark_age - 4

def Mark_age (billy_age : ℕ): ℕ :=
  billy_age + 4

def Billy_age (ana_age : ℕ): ℕ :=
  ana_age / 2

def Ana_age : ℕ := 15 - 3

theorem Sarah_is_26 : Sarah_age (Mark_age (Billy_age Ana_age)) (Billy_age Ana_age) Ana_age = 26 := 
by
  sorry

end Sarah_is_26_l254_254260


namespace smallest_integer_for_inequality_l254_254454

theorem smallest_integer_for_inequality :
  ∃ x : ℤ, x^2 < 2 * x + 1 ∧ ∀ y : ℤ, y^2 < 2 * y + 1 → x ≤ y := sorry

end smallest_integer_for_inequality_l254_254454


namespace find_number_l254_254132

variable (N : ℝ)

theorem find_number (h : (5 / 6) * N = (5 / 16) * N + 50) : N = 96 := 
by 
  sorry

end find_number_l254_254132


namespace day_100_M_minus_1_is_Tuesday_l254_254392

variable {M : ℕ}

-- Given conditions
def day_200_M_is_Monday (M : ℕ) : Prop :=
  ((200 % 7) = 6)

def day_300_M_plus_2_is_Monday (M : ℕ) : Prop :=
  ((300 % 7) = 6)

-- Statement to prove
theorem day_100_M_minus_1_is_Tuesday (M : ℕ) 
  (h1 : day_200_M_is_Monday M) 
  (h2 : day_300_M_plus_2_is_Monday M) 
  : (((100 + (365 - 200)) % 7 + 7 - 1) % 7 = 2) :=
sorry

end day_100_M_minus_1_is_Tuesday_l254_254392


namespace identify_jars_l254_254966

namespace JarIdentification

/-- Definitions of Jar labels -/
inductive JarLabel
| Nickels
| Dimes
| Nickels_and_Dimes

open JarLabel

/-- Mislabeling conditions for each jar -/
def mislabeled (jarA : JarLabel) (jarB : JarLabel) (jarC : JarLabel) : Prop :=
  ((jarA ≠ Nickels) ∧ (jarB ≠ Dimes) ∧ (jarC ≠ Nickels_and_Dimes)) ∧
  ((jarC = Nickels ∨ jarC = Dimes))

/-- Given the result of a coin draw from the jar labeled "Nickels and Dimes" -/
def jarIdentity (jarA jarB jarC : JarLabel) (drawFromC : String) : Prop :=
  if drawFromC = "Nickel" then
    jarC = Nickels ∧ jarA = Nickels_and_Dimes ∧ jarB = Dimes
  else if drawFromC = "Dime" then
    jarC = Dimes ∧ jarB = Nickels_and_Dimes ∧ jarA = Nickels
  else 
    false

/-- Main theorem to prove the identification of jars -/
theorem identify_jars (jarA jarB jarC : JarLabel) (draw : String)
  (h1 : mislabeled jarA jarB jarC) :
  jarIdentity jarA jarB jarC draw :=
by
  sorry

end JarIdentification

end identify_jars_l254_254966


namespace PQ_is_10_5_l254_254440

noncomputable def PQ_length_proof_problem : Prop := 
  ∃ (PQ : ℝ),
    PQ = 10.5 ∧ 
    ∃ (ST : ℝ) (SU : ℝ),
      ST = 4.5 ∧ SU = 7.5 ∧ 
      ∃ (QR : ℝ) (PR : ℝ),
        QR = 21 ∧ PR = 15 ∧ 
        ∃ (angle_PQR angle_STU : ℝ),
          angle_PQR = 120 ∧ angle_STU = 120 ∧ 
          PQ / ST = PR / SU

theorem PQ_is_10_5 :
  PQ_length_proof_problem := sorry

end PQ_is_10_5_l254_254440


namespace angles_cosine_sum_l254_254266

theorem angles_cosine_sum (A B : ℝ) 
  (h1 : Real.sin A + Real.sin B = 1)
  (h2 : Real.cos A + Real.cos B = 0) :
  12 * Real.cos (2 * A) + 4 * Real.cos (2 * B) = 8 :=
sorry

end angles_cosine_sum_l254_254266


namespace spherical_to_rectangular_example_l254_254605

noncomputable def spherical_to_rectangular (ρ θ φ : ℝ) : ℝ × ℝ × ℝ :=
  (ρ * Real.sin φ * Real.cos θ,
   ρ * Real.sin φ * Real.sin θ,
   ρ * Real.cos φ)

theorem spherical_to_rectangular_example :
  spherical_to_rectangular 5 (3 * Real.pi / 2) (Real.pi / 3) = (0, -5 * Real.sqrt 3 / 2, 5 / 2) :=
by
  simp [spherical_to_rectangular, Real.sin, Real.cos]
  sorry

end spherical_to_rectangular_example_l254_254605


namespace calculate_expression_l254_254158

-- Define the expression x + x * (factorial x)^x
def expression (x : ℕ) : ℕ :=
  x + x * (Nat.factorial x) ^ x

-- Set the value of x
def x_value : ℕ := 3

-- State the proposition
theorem calculate_expression : expression x_value = 651 := 
by 
  -- By substitution and calculation, the proof follows.
  sorry

end calculate_expression_l254_254158


namespace overall_percentage_l254_254597

theorem overall_percentage (s1 s2 s3 : ℝ) (h1 : s1 = 60) (h2 : s2 = 80) (h3 : s3 = 85) :
  (s1 + s2 + s3) / 3 = 75 := by
  sorry

end overall_percentage_l254_254597


namespace scientific_notation_of_0_0000025_l254_254465

theorem scientific_notation_of_0_0000025 :
  0.0000025 = 2.5 * 10^(-6) :=
by
  sorry

end scientific_notation_of_0_0000025_l254_254465


namespace sector_area_l254_254350

-- Given conditions
variables {l r : ℝ}

-- Definitions (conditions from the problem)
def arc_length (l : ℝ) := l
def radius (r : ℝ) := r

-- Problem statement
theorem sector_area (l r : ℝ) : 
    (1 / 2) * l * r = (1 / 2) * l * r :=
by
  sorry

end sector_area_l254_254350


namespace find_first_5digits_of_M_l254_254109

def last6digits (n : ℕ) : ℕ := n % 1000000

def first5digits (n : ℕ) : ℕ := n / 10

theorem find_first_5digits_of_M (M : ℕ) (h1 : last6digits M = last6digits (M^2)) (h2 : M > 999999) : first5digits M = 60937 := 
by sorry

end find_first_5digits_of_M_l254_254109


namespace intervals_equinumerous_l254_254829

-- Definitions and statements
theorem intervals_equinumerous (a : ℝ) (h : 0 < a) : 
  ∃ (f : Set.Icc 0 1 → Set.Icc 0 a), Function.Bijective f :=
by
  sorry

end intervals_equinumerous_l254_254829


namespace ab_plus_2_l254_254996

theorem ab_plus_2 (a b : ℝ) (h : ∀ x : ℝ, (x - 3) * (3 * x + 7) = x^2 - 12 * x + 27 → x = a ∨ x = b) (ha : a ≠ b) :
  (a + 2) * (b + 2) = -30 :=
sorry

end ab_plus_2_l254_254996


namespace find_a6_l254_254349

noncomputable def a (n : ℕ) : ℝ := sorry

axiom geom_seq_inc :
  ∀ n : ℕ, a n < a (n + 1)

axiom root_eqn_a2_a4 :
  ∃ a2 a4 : ℝ, (a 2 = a2) ∧ (a 4 = a4) ∧ (a2^2 - 6 * a2 + 5 = 0) ∧ (a4^2 - 6 * a4 + 5 = 0)

theorem find_a6 : a 6 = 25 := 
sorry

end find_a6_l254_254349


namespace interval_of_x_l254_254181

theorem interval_of_x (x : ℝ) :
  (2 < 4 * x ∧ 4 * x < 3) ∧ (2 < 5 * x ∧ 5 * x < 3) ↔ (1 / 2 < x ∧ x < 3 / 5) := by
  sorry

end interval_of_x_l254_254181


namespace three_digit_number_divisible_by_8_and_even_tens_digit_l254_254455

theorem three_digit_number_divisible_by_8_and_even_tens_digit (d : ℕ) (hd : d % 2 = 0) (hdiv : (100 * 5 + 10 * d + 4) % 8 = 0) :
  100 * 5 + 10 * d + 4 = 544 :=
by
  sorry

end three_digit_number_divisible_by_8_and_even_tens_digit_l254_254455


namespace arctan_triangle_complementary_l254_254908

theorem arctan_triangle_complementary :
  (Real.arctan (3 / 4) + Real.arctan (4 / 3) = Real.pi / 2) :=
begin
  sorry
end

end arctan_triangle_complementary_l254_254908


namespace age_of_b_l254_254298

theorem age_of_b (a b c : ℕ) (h1 : a = b + 2) (h2 : b = 2 * c) (h3 : a + b + c = 32) : b = 12 :=
by sorry

end age_of_b_l254_254298


namespace average_first_15_nat_l254_254586

-- Define the sequence and necessary conditions
def sum_first_n_nat (n : ℕ) : ℕ := n * (n + 1) / 2

theorem average_first_15_nat : (sum_first_n_nat 15) / 15 = 8 := 
by 
  -- Here we shall place the proof to show the above statement holds true
  sorry

end average_first_15_nat_l254_254586


namespace infinite_series_sum_eq_seven_l254_254611

noncomputable def infinite_series_sum : ℝ :=
  ∑' k : ℕ, (1 + k)^2 / 3^(1 + k)

theorem infinite_series_sum_eq_seven : infinite_series_sum = 7 :=
sorry

end infinite_series_sum_eq_seven_l254_254611


namespace degree_of_each_exterior_angle_of_regular_octagon_l254_254677
-- We import the necessary Lean libraries

-- Define the degrees of each exterior angle of a regular octagon
theorem degree_of_each_exterior_angle_of_regular_octagon : 
  (∑ (i : Fin 8), (360 / 8) = 360 → (360 / 8) = 45) :=
by
  sorry

end degree_of_each_exterior_angle_of_regular_octagon_l254_254677


namespace athlete_speed_l254_254131

theorem athlete_speed (distance time : ℝ) (h1 : distance = 200) (h2 : time = 25) :
  (distance / time) = 8 := by
  sorry

end athlete_speed_l254_254131


namespace least_xy_value_l254_254784

theorem least_xy_value (x y : ℕ) (hx : 0 < x) (hy : 0 < y) (h : 1 / x + 1 / (3 * y) = 1 / 6) : x * y = 90 :=
by
  sorry

end least_xy_value_l254_254784


namespace length_more_than_breadth_by_10_l254_254844

-- Definitions based on conditions
def length : ℕ := 55
def cost_per_meter : ℚ := 26.5
def total_fencing_cost : ℚ := 5300
def perimeter : ℚ := total_fencing_cost / cost_per_meter

-- Calculate breadth (b) and difference (x)
def breadth := 45 -- This is inferred manually from the solution for completeness
def difference (b : ℚ) := length - b

-- The statement we need to prove
theorem length_more_than_breadth_by_10 :
  difference 45 = 10 :=
by
  sorry

end length_more_than_breadth_by_10_l254_254844


namespace part1_part2_l254_254351

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := 5 * Real.log x + a * x^2 - 6 * x
noncomputable def f_prime (x : ℝ) (a : ℝ) : ℝ := 5 / x + 2 * a * x - 6

theorem part1 (a : ℝ) (h_tangent : f_prime 1 a = 0) : a = 1 / 2 :=
by {
  sorry
}

theorem part2 (a : ℝ) (h_a : a = 1/2) :
  (∀ x, 0 < x → x < 1 → f_prime x a > 0) ∧
  (∀ x, 5 < x → f_prime x a > 0) ∧
  (∀ x, 1 < x → x < 5 → f_prime x a < 0) :=
by {
  sorry
}

end part1_part2_l254_254351


namespace range_of_a_l254_254368

noncomputable def equation_has_two_roots (a m : ℝ) : Prop :=
  ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ 
    x₁ + a * (2 * x₁ + 2 * m - 4 * Real.exp 1 * x₁) * (Real.log (x₁ + m) - Real.log x₁) = 0 ∧ 
    x₂ + a * (2 * x₂ + 2 * m - 4 * Real.exp 1 * x₂) * (Real.log (x₂ + m) - Real.log x₂) = 0

theorem range_of_a (m : ℝ) (hm : 0 < m) : 
  (∃ a, equation_has_two_roots a m) ↔ (a < 0 ∨ a > 1 / (2 * Real.exp 1)) := 
sorry

end range_of_a_l254_254368


namespace find_x_squared_plus_inverse_squared_l254_254206

theorem find_x_squared_plus_inverse_squared (x : ℝ) 
(h : x^4 + (1 / x^4) = 2398) : 
  x^2 + (1 / x^2) = 20 * Real.sqrt 6 :=
sorry

end find_x_squared_plus_inverse_squared_l254_254206


namespace lollipops_per_day_l254_254632

variable (Alison_lollipops : ℕ) (Henry_lollipops : ℕ) (Diane_lollipops : ℕ) (Total_lollipops : ℕ) (Days : ℕ)

-- Conditions given in the problem
axiom condition1 : Alison_lollipops = 60
axiom condition2 : Henry_lollipops = Alison_lollipops + 30
axiom condition3 : Alison_lollipops = Diane_lollipops / 2
axiom condition4 : Total_lollipops = Alison_lollipops + Henry_lollipops + Diane_lollipops
axiom condition5 : Days = 6

-- Question to prove
theorem lollipops_per_day : (Total_lollipops / Days) = 45 := sorry

end lollipops_per_day_l254_254632


namespace birthday_cars_equal_12_l254_254985

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

end birthday_cars_equal_12_l254_254985


namespace total_time_in_pool_is_29_minutes_l254_254724

noncomputable def calculate_total_time_in_pool : ℝ :=
  let jerry := 3             -- Jerry's time in minutes
  let elaine := 2 * jerry    -- Elaine's time in minutes
  let george := elaine / 3    -- George's time in minutes
  let susan := 150 / 60      -- Susan's time in minutes
  let puddy := elaine / 2    -- Puddy's time in minutes
  let frank := elaine / 2    -- Frank's time in minutes
  let estelle := 0.1 * 60    -- Estelle's time in minutes
  let total_excluding_newman := jerry + elaine + george + susan + puddy + frank + estelle
  let newman := total_excluding_newman / 7   -- Newman's average time
  total_excluding_newman + newman

theorem total_time_in_pool_is_29_minutes : 
  calculate_total_time_in_pool = 29 :=
by
  sorry

end total_time_in_pool_is_29_minutes_l254_254724


namespace function_decreasing_iff_l254_254041

theorem function_decreasing_iff (a : ℝ) :
  (0 < a ∧ a < 1) ∧ a ≤ 1/4 ↔ (0 < a ∧ a ≤ 1/4) :=
by
  sorry

end function_decreasing_iff_l254_254041


namespace part1_part2_l254_254346

variable {a b c : ℚ}

theorem part1 (ha : a < 0) : (a / |a|) = -1 :=
sorry

theorem part2 (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) : 
  min (a * b / |a * b| + |b * c| / (b * c) + a * c / |a * c| + |a * b * c| / (a * b * c)) (-2) = -2 :=
sorry

end part1_part2_l254_254346


namespace minimum_omega_l254_254430

theorem minimum_omega (ω : ℝ) (hω_pos : ω > 0)
  (f : ℝ → ℝ) (hf : ∀ x, f x = Real.sin (ω * x + Real.pi / 3))
  (C : ℝ → ℝ) (hC : ∀ x, C x = Real.sin (ω * (x + Real.pi / 2) + Real.pi / 3)) :
  (∀ x, C x = C (-x)) ↔ ω = 1 / 3 := by
sorry

end minimum_omega_l254_254430


namespace negation_of_exists_l254_254589

theorem negation_of_exists : (¬ ∃ x_0 : ℝ, x_0 < 0 ∧ x_0^2 > 0) ↔ ∀ x : ℝ, x < 0 → x^2 ≤ 0 :=
sorry

end negation_of_exists_l254_254589


namespace smallest_positive_integer_remainder_conditions_l254_254122

theorem smallest_positive_integer_remainder_conditions :
  ∃ b : ℕ, (b % 4 = 3 ∧ b % 6 = 5) ∧ ∀ n : ℕ, (n % 4 = 3 ∧ n % 6 = 5) → n ≥ b := 
by
  have b := 23
  use b
  sorry

end smallest_positive_integer_remainder_conditions_l254_254122


namespace smallest_N_for_percentages_l254_254067

theorem smallest_N_for_percentages 
  (N : ℕ) 
  (h1 : ∃ N, ∀ f ∈ [1/10, 2/5, 1/5, 3/10], ∃ k : ℕ, N * f = k) :
  N = 10 := 
by
  sorry

end smallest_N_for_percentages_l254_254067


namespace arctan_sum_pi_div_two_l254_254899

theorem arctan_sum_pi_div_two :
  let α := Real.arctan (3 / 4)
  let β := Real.arctan (4 / 3)
  α + β = Real.pi / 2 := by
  sorry

end arctan_sum_pi_div_two_l254_254899


namespace find_ck_l254_254278

def arithmetic_seq (d : ℕ) (n : ℕ) : ℕ := 1 + (n - 1) * d
def geometric_seq (r : ℕ) (n : ℕ) : ℕ := r^(n - 1)
def c_seq (a_seq : ℕ → ℕ) (b_seq : ℕ → ℕ) (n : ℕ) := a_seq n + b_seq n

theorem find_ck (d r k : ℕ) (a_seq := arithmetic_seq d) (b_seq := geometric_seq r) :
  c_seq a_seq b_seq (k - 1) = 200 →
  c_seq a_seq b_seq (k + 1) = 400 →
  c_seq a_seq b_seq k = 322 :=
by
  sorry

end find_ck_l254_254278


namespace find_coefficients_l254_254166

theorem find_coefficients (A B : ℝ) (h_roots : (x^2 + A * x + B = 0 ∧ (x = A ∨ x = B))) :
  (A = 0 ∧ B = 0) ∨ (A = 1 ∧ B = -2) :=
by sorry

end find_coefficients_l254_254166


namespace phil_baseball_cards_left_l254_254827

-- Step a): Define the conditions
def packs_week := 20
def weeks_year := 52
def lost_factor := 1 / 2

-- Step c): Establish the theorem statement
theorem phil_baseball_cards_left : 
  (packs_week * weeks_year * (1 - lost_factor) = 520) := 
  by
    -- proof steps will come here
    sorry

end phil_baseball_cards_left_l254_254827


namespace hyperbola_sqrt3_eccentricity_l254_254210

noncomputable def hyperbola_eccentricity (m : ℝ) : ℝ :=
  let a := 2
  let b := m
  let c := Real.sqrt (a^2 + b^2)
  c / a

theorem hyperbola_sqrt3_eccentricity (m : ℝ) (h_m_pos : 0 < m) (h_slope : m = 2 * Real.sqrt 2) :
  hyperbola_eccentricity m = Real.sqrt 3 :=
by
  unfold hyperbola_eccentricity
  rw [h_slope]
  simp
  sorry

end hyperbola_sqrt3_eccentricity_l254_254210


namespace ratio_expression_x_2y_l254_254766

theorem ratio_expression_x_2y :
  ∀ (x y : ℝ), x / (2 * y) = 27 → (7 * x + 6 * y) / (x - 2 * y) = 96 / 13 :=
by
  intros x y h
  sorry

end ratio_expression_x_2y_l254_254766


namespace solve_quadratic_abs_l254_254568

theorem solve_quadratic_abs (x : ℝ) :
  x^2 - |x| - 1 = 0 ↔ x = (1 + Real.sqrt 5) / 2 ∨ x = (1 - Real.sqrt 5) / 2 ∨ 
                   x = (-1 + Real.sqrt 5) / 2 ∨ x = (-1 - Real.sqrt 5) / 2 := 
sorry

end solve_quadratic_abs_l254_254568


namespace find_p_q_of_divisibility_l254_254359

theorem find_p_q_of_divisibility 
  (p q : ℤ) 
  (h1 : (x + 3) ∣ (x^5 - 2*x^4 + 3*x^3 - p*x^2 + q*x - 6)) 
  (h2 : (x - 2) ∣ (x^5 - 2*x^4 + 3*x^3 - p*x^2 + q*x - 6)) 
  : p = -31 ∧ q = -71 :=
by
  sorry

end find_p_q_of_divisibility_l254_254359


namespace cost_of_pencil_and_pens_l254_254841

variable (p q : ℝ)

def equation1 := 3 * p + 4 * q = 3.20
def equation2 := 2 * p + 3 * q = 2.50

theorem cost_of_pencil_and_pens (h1 : equation1 p q) (h2 : equation2 p q) : p + 2 * q = 1.80 := 
by 
  sorry

end cost_of_pencil_and_pens_l254_254841


namespace trapezoid_area_l254_254563

theorem trapezoid_area 
  (diagonals_perpendicular : ∀ A B C D : ℝ, (A ≠ B → C ≠ D → A * C + B * D = 0)) 
  (diagonal_length : ∀ B D : ℝ, B ≠ D → (B - D) = 17) 
  (height_of_trapezoid : ∀ (height : ℝ), height = 15) : 
  ∃ (area : ℝ), area = 4335 / 16 := 
sorry

end trapezoid_area_l254_254563


namespace min_omega_symmetry_l254_254431

theorem min_omega_symmetry :
  ∃ ω > 0, (∀ x : ℝ, sin (ω * x + ω * (π / 2) + π / 3) = sin ((-ω) * x + ω * (π / 2) + π / 3)) →
  ω = 1 / 3 :=
by {
  sorry
}

end min_omega_symmetry_l254_254431


namespace solution_exists_for_any_y_l254_254332

theorem solution_exists_for_any_y (z : ℝ) : (∀ y : ℝ, ∃ x : ℝ, x^2 + y^2 + 4*z^2 + 2*x*y*z - 9 = 0) ↔ |z| ≤ 3 / 2 := 
sorry

end solution_exists_for_any_y_l254_254332


namespace average_monthly_income_l254_254676

theorem average_monthly_income (P Q R : ℝ) (h1 : (P + Q) / 2 = 5050)
  (h2 : (Q + R) / 2 = 6250) (h3 : P = 4000) : (P + R) / 2 = 5200 := by
  sorry

end average_monthly_income_l254_254676


namespace count_lineup_excluding_youngest_l254_254377

theorem count_lineup_excluding_youngest 
  (n : ℕ) (h_n : n = 5) (youngest_position : Fin n → Prop) 
  (h_youngest_position : ∀ (pos : Fin n), youngest_position pos → pos ≠ 0 ∧ pos ≠ (n - 1)) :
  (∃ (count : ℕ), count = (4 * 3 * 3 * 2) ∧ count = 216) := 
sorry

end count_lineup_excluding_youngest_l254_254377


namespace retain_exactly_five_coins_l254_254494

-- Define the problem setup
structure GameSetup :=
  (players : Finset (String)) -- Five friends: "Abby", "Bernardo", "Carl", "Debra", "Elina"
  (initial_coins : ℕ := 5)    -- Each player starts with 5 coins
  (rounds : ℕ := 3)           -- Number of rounds
  (urn : Finset (String) := {"green", "red", "white", "white", "white"}) -- Balls in the urn

-- The game conditions
def game_conditions (setup : GameSetup) : Prop :=
  setup.players = {"Abby", "Bernardo", "Carl", "Debra", "Elina"} ∧
  setup.initial_coins = 5 ∧
  setup.rounds = 3 ∧
  setup.urn = {"green", "red", "white", "white", "white"}

-- The statement to prove: Probability that everyone still has exactly 5 coins at the end of 3 rounds
theorem retain_exactly_five_coins (setup : GameSetup) (h : game_conditions setup) :
  (1 / 10 : ℚ) ^ 3 = 1 / 1000 :=
by
  sorry

end retain_exactly_five_coins_l254_254494


namespace problem_1_problem_2_l254_254956

theorem problem_1 (a b c : ℝ) (h₀ : a > 0) (h₁ : b > 0) (h₂ : c > 0) (h₃ : ∀ x, |x + a| + |x - b| + c ≥ 4) : 
  a + b + c = 4 :=
sorry

theorem problem_2 (a b c : ℝ) (h₀ : a > 0) (h₁ : b > 0) (h₂ : c > 0) (h₃ : a + b + c = 4) : 
  (1/4) * a^2 + (1/9) * b^2 + c^2 = 8 / 7 :=
sorry

end problem_1_problem_2_l254_254956


namespace intersection_of_M_and_N_l254_254495

noncomputable def M : Set ℝ := { y : ℝ | ∃ x : ℝ, y = x^2 }
noncomputable def N : Set ℝ := { y : ℝ | ∃ x : ℝ, x^2 + y^2 = 1 }

theorem intersection_of_M_and_N : M ∩ N = { y : ℝ | 0 ≤ y ∧ y ≤ 1 } :=
by
  sorry

end intersection_of_M_and_N_l254_254495


namespace negation_of_proposition_l254_254276
open Real

theorem negation_of_proposition :
  ¬ (∃ x₀ : ℝ, (2/x₀) + log x₀ ≤ 0) ↔ ∀ x : ℝ, (2/x) + log x > 0 :=
by
  sorry

end negation_of_proposition_l254_254276


namespace product_mod_7_l254_254013

theorem product_mod_7 :
  (2009 % 7 = 4) ∧ (2010 % 7 = 5) ∧ (2011 % 7 = 6) ∧ (2012 % 7 = 0) →
  (2009 * 2010 * 2011 * 2012) % 7 = 0 :=
by
  sorry

end product_mod_7_l254_254013


namespace coloring_circle_impossible_l254_254235

theorem coloring_circle_impossible (n : ℕ) (h : n = 2022) : 
  ¬ (∃ (coloring : ℕ → ℕ), (∀ i, 0 ≤ coloring i ∧ coloring i < 3) ∧ (∀ i, coloring ((i + 1) % n) ≠ coloring i)) :=
sorry

end coloring_circle_impossible_l254_254235


namespace second_number_is_11_l254_254141

-- Define the conditions
variables (x : ℕ) (h1 : 5 * x = 55)

-- The theorem we want to prove
theorem second_number_is_11 : x = 11 :=
sorry

end second_number_is_11_l254_254141


namespace cost_of_four_pencils_and_four_pens_l254_254840

def pencil_cost : ℝ := sorry
def pen_cost : ℝ := sorry

axiom h1 : 8 * pencil_cost + 3 * pen_cost = 5.10
axiom h2 : 3 * pencil_cost + 5 * pen_cost = 4.95

theorem cost_of_four_pencils_and_four_pens : 4 * pencil_cost + 4 * pen_cost = 4.488 :=
by
  sorry

end cost_of_four_pencils_and_four_pens_l254_254840


namespace operation_evaluation_l254_254453

theorem operation_evaluation : 65 + 5 * 12 / (180 / 3) = 66 :=
by
  -- Parentheses
  have h1 : 180 / 3 = 60 := by sorry
  -- Multiplication and Division
  have h2 : 5 * 12 = 60 := by sorry
  have h3 : 60 / 60 = 1 := by sorry
  -- Addition
  exact sorry

end operation_evaluation_l254_254453


namespace radius_of_circle_with_center_on_line_and_passing_through_points_l254_254764

theorem radius_of_circle_with_center_on_line_and_passing_through_points : 
  (∃ a b : ℝ, 2 * a + b = 0 ∧ 
              (a - 1) ^ 2 + (b - 3) ^ 2 = r ^ 2 ∧ 
              (a - 4) ^ 2 + (b - 2) ^ 2 = r ^ 2 
              → r = 5) := 
by 
  sorry

end radius_of_circle_with_center_on_line_and_passing_through_points_l254_254764


namespace problem1_problem2_problem3_l254_254212

noncomputable def U : Set ℝ := {x | x ≤ 1 ∨ x ≥ 2}
noncomputable def A : Set ℝ := {x | x < 1 ∨ x > 3}
noncomputable def B : Set ℝ := {x | x < 1 ∨ x > 2}

theorem problem1 : A ∩ B = {x | x < 1 ∨ x > 3} := 
  sorry

theorem problem2 : A ∩ (U \ B) = ∅ := 
  sorry

theorem problem3 : U \ (A ∪ B) = {1, 2} := 
  sorry

end problem1_problem2_problem3_l254_254212


namespace arctan_3_4_add_arctan_4_3_is_pi_div_2_l254_254918

noncomputable def arctan_add (a b : ℝ) : ℝ :=
  Real.arctan a + Real.arctan b

theorem arctan_3_4_add_arctan_4_3_is_pi_div_2 :
  arctan_add (3 / 4) (4 / 3) = Real.pi / 2 :=
sorry

end arctan_3_4_add_arctan_4_3_is_pi_div_2_l254_254918


namespace problem1_problem2_l254_254954

noncomputable def cos_alpha (α : ℝ) : ℝ := (Real.sqrt 2 + 4) / 6
noncomputable def cos_alpha_plus_half_beta (α β : ℝ) : ℝ := 5 * Real.sqrt 3 / 9

theorem problem1 {α : ℝ} (hα1 : 0 < α) (hα2 : α < Real.pi / 2) 
                 (h1 : Real.cos (Real.pi / 4 + α) = 1 / 3) :
  Real.cos α = cos_alpha α :=
sorry

theorem problem2 {α β : ℝ} (hα1 : 0 < α) (hα2 : α < Real.pi / 2) 
                 (hβ1 : -Real.pi / 2 < β) (hβ2 : β < 0) 
                 (h1 : Real.cos (Real.pi / 4 + α) = 1 / 3) 
                 (h2 : Real.cos (Real.pi / 4 - β / 2) = Real.sqrt 3 / 3) :
  Real.cos (α + β / 2) = cos_alpha_plus_half_beta α β :=
sorry

end problem1_problem2_l254_254954


namespace sally_cut_red_orchids_l254_254572

-- Definitions and conditions
def initial_red_orchids := 9
def orchids_in_vase_after_cutting := 15

-- Problem statement
theorem sally_cut_red_orchids : (orchids_in_vase_after_cutting - initial_red_orchids) = 6 := by
  sorry

end sally_cut_red_orchids_l254_254572


namespace shirt_pants_outfits_l254_254457

theorem shirt_pants_outfits
  (num_shirts : ℕ) (num_pants : ℕ) (num_formal_pants : ℕ) (num_casual_pants : ℕ) (num_assignee_shirts : ℕ) :
  num_shirts = 5 →
  num_pants = 6 →
  num_formal_pants = 3 →
  num_casual_pants = 3 →
  num_assignee_shirts = 3 →
  (num_casual_pants * num_shirts) + (num_formal_pants * num_assignee_shirts) = 24 :=
by
  intros h_shirts h_pants h_formal h_casual h_assignee
  sorry

end shirt_pants_outfits_l254_254457


namespace sum_of_repeating_decimals_l254_254752

/-- Definitions of the repeating decimals as real numbers. --/
def x : ℝ := 0.3 -- This actually represents 0.\overline{3} in Lean
def y : ℝ := 0.04 -- This actually represents 0.\overline{04} in Lean
def z : ℝ := 0.005 -- This actually represents 0.\overline{005} in Lean

/-- The theorem stating that the sum of these repeating decimals is a specific fraction. --/
theorem sum_of_repeating_decimals : x + y + z = (14 : ℝ) / 37 := 
by 
  sorry -- Placeholder for the proof

end sum_of_repeating_decimals_l254_254752


namespace sum_of_sides_eq_13_or_15_l254_254855

noncomputable def squares_side_lengths (b d : ℕ) : Prop :=
  15^2 = b^2 + 10^2 + d^2

theorem sum_of_sides_eq_13_or_15 :
  ∃ b d : ℕ, squares_side_lengths b d ∧ (b + d = 13 ∨ b + d = 15) :=
sorry

end sum_of_sides_eq_13_or_15_l254_254855


namespace func_identity_equiv_l254_254136

theorem func_identity_equiv (f : ℝ → ℝ) :
  (∀ x y : ℝ, f (x + y) = f (x) + f (y)) ↔ (∀ x y : ℝ, f (xy + x + y) = f (xy) + f (x) + f (y)) :=
by
  sorry

end func_identity_equiv_l254_254136


namespace GE_eq_GH_l254_254870

variables (A B C D E F G H : Type) [Inhabited A] [Inhabited B] [Inhabited C] [Inhabited D]
          [Inhabited E] [Inhabited F] [Inhabited G] [Inhabited H]
          
variables (AC : Line A C) (AB : Line A B) (BE : Line B E) (DE : Line D E)
          (BG : Line B G) (AF : Line A F) (DE' : Line D E') (angleC : Angle C = 90)

variables (circB : Circle B BC) (tangentDE : Tangent DE circB E) (perpAB : Perpendicular AC AB)
          (intersectionF : Intersect (PerpendicularLine C AB) BE F)
          (intersectionG : Intersect AF DE G) (intersectionH : Intersect (ParallelLine A BG) DE H)

theorem GE_eq_GH : GE = GH := sorry

end GE_eq_GH_l254_254870


namespace arctan_sum_eq_pi_div_two_l254_254922

theorem arctan_sum_eq_pi_div_two : Real.arctan (3 / 4) + Real.arctan (4 / 3) = Real.pi / 2 :=
by
  sorry

end arctan_sum_eq_pi_div_two_l254_254922


namespace abs_neg_three_l254_254835

noncomputable def abs_val (a : ℤ) : ℤ :=
  if a < 0 then -a else a

theorem abs_neg_three : abs_val (-3) = 3 :=
by
  sorry

end abs_neg_three_l254_254835


namespace total_pages_in_book_l254_254726

/-- Bill started reading a book on the first day of April. 
    He read 8 pages every day and by the 12th of April, he 
    had covered two-thirds of the book. Prove that the 
    total number of pages in the book is 144. --/
theorem total_pages_in_book 
  (pages_per_day : ℕ)
  (days_till_april_12 : ℕ)
  (total_pages_read : ℕ)
  (fraction_of_book_read : ℚ)
  (total_pages : ℕ)
  (h1 : pages_per_day = 8)
  (h2 : days_till_april_12 = 12)
  (h3 : total_pages_read = pages_per_day * days_till_april_12)
  (h4 : fraction_of_book_read = 2/3)
  (h5 : total_pages_read = (fraction_of_book_read * total_pages)) :
  total_pages = 144 := by
  sorry

end total_pages_in_book_l254_254726


namespace find_a9_l254_254807

-- Define the arithmetic sequence
def arithmetic_seq (a : ℕ → ℝ) : Prop :=
  ∃ d, ∀ n, a (n + 1) = a n + d

-- Given conditions
def a_n : ℕ → ℝ := sorry   -- The sequence itself is unknown initially.

axiom a3 : a_n 3 = 5
axiom a4_a8 : a_n 4 + a_n 8 = 22

theorem find_a9 : a_n 9 = 41 :=
by
  sorry

end find_a9_l254_254807


namespace unique_ordered_pairs_satisfying_equation_l254_254930

theorem unique_ordered_pairs_satisfying_equation :
  ∃ x y : ℕ, 0 < x ∧ 0 < y ∧ x^6 * y^6 - 19 * x^3 * y^3 + 18 = 0 ↔ (x, y) = (1, 1) ∧
  (∀ x y : ℕ, 0 < x ∧ 0 < y ∧ x^6 * y^6 - 19 * x^3 * y^3 + 18 = 0 → (x, y) = (1, 1)) :=
by
  sorry

end unique_ordered_pairs_satisfying_equation_l254_254930


namespace sum_not_prime_l254_254247

-- Definitions based on conditions:
variables {a b c d : ℕ}

-- Conditions:
axiom h_ab_eq_cd : a * b = c * d

-- Statement to prove:
theorem sum_not_prime (a b c d : ℕ) (h : a * b = c * d) : ¬Nat.Prime (a + b + c + d) :=
sorry

end sum_not_prime_l254_254247


namespace root_of_polynomial_l254_254658

theorem root_of_polynomial (a b : ℝ) (h₁ : a^4 + a^3 - 1 = 0) (h₂ : b^4 + b^3 - 1 = 0) : 
  (ab : ℝ) → ab * ab * ab * ab * ab * ab + ab * ab * ab * ab + ab * ab * ab - ab * ab - 1 = 0 :=
sorry

end root_of_polynomial_l254_254658


namespace max_clouds_through_planes_l254_254644

-- Define the problem parameters and conditions
def max_clouds (n : ℕ) : ℕ :=
  n + 1

-- Mathematically equivalent proof problem statement in Lean 4
theorem max_clouds_through_planes : max_clouds 10 = 11 :=
  by
    sorry  -- Proof skipped as required

end max_clouds_through_planes_l254_254644


namespace part_a_part_b_part_c_l254_254666

theorem part_a (p q : ℝ) : q < p^2 → ∃ r1 r2 : ℝ, r1 ≠ r2 ∧ (r1 + r2 = 2 * p) ∧ (r1 * r2 = q) :=
by
  sorry

theorem part_b (p q : ℝ) : q = 4 * p - 4 → (2^2 - 2 * p * 2 + q = 0) :=
by
  sorry

theorem part_c (p q : ℝ) : q = p^2 ∧ q = 4 * p - 4 → (p = 2 ∧ q = 4) :=
by
  sorry

end part_a_part_b_part_c_l254_254666


namespace least_possible_xy_l254_254776

theorem least_possible_xy (x y : ℕ) (hx : 0 < x) (hy : 0 < y)
  (h : 1 / x + 1 / (3 * y) = 1 / 6) : x * y = 48 :=
by
  sorry

end least_possible_xy_l254_254776


namespace arctan_sum_pi_div_two_l254_254913

noncomputable def arctan_sum : Real :=
  Real.arctan (3 / 4) + Real.arctan (4 / 3)

theorem arctan_sum_pi_div_two : arctan_sum = Real.pi / 2 := 
by sorry

end arctan_sum_pi_div_two_l254_254913


namespace omega_min_value_l254_254423

theorem omega_min_value (ω : ℝ) (hω : ω > 0)
    (hSymmetry : ∀ x : ℝ, sin (ω * x + ω * π / 2 + π / 3) = sin (ω * -x + ω * π / 2 + π / 3)) :
    ω = 1 / 3 :=
begin
  sorry
end

end omega_min_value_l254_254423


namespace beast_of_war_running_time_correct_l254_254852

def running_time_millennium : ℕ := 120

def running_time_alpha_epsilon (rt_millennium : ℕ) : ℕ := rt_millennium - 30

def running_time_beast_of_war (rt_alpha_epsilon : ℕ) : ℕ := rt_alpha_epsilon + 10

theorem beast_of_war_running_time_correct :
  running_time_beast_of_war (running_time_alpha_epsilon running_time_millennium) = 100 := by sorry

end beast_of_war_running_time_correct_l254_254852


namespace repeatingDecimals_fraction_eq_l254_254755

noncomputable def repeatingDecimalsSum : ℚ :=
  let x : ℚ := 1 / 3
  let y : ℚ := 4 / 99
  let z : ℚ := 5 / 999
  x + y + z

theorem repeatingDecimals_fraction_eq : repeatingDecimalsSum = 42 / 111 :=
  sorry

end repeatingDecimals_fraction_eq_l254_254755


namespace earnings_per_widget_l254_254297

theorem earnings_per_widget (W_h : ℝ) (H_w : ℕ) (W_t : ℕ) (E_w : ℝ) (E : ℝ) :
  W_h = 12.50 ∧ H_w = 40 ∧ W_t = 1000 ∧ E_w = 660 →
  E = 0.16 :=
by
  sorry

end earnings_per_widget_l254_254297


namespace total_distance_craig_walked_l254_254606

theorem total_distance_craig_walked :
  0.2 + 0.7 = 0.9 :=
by sorry

end total_distance_craig_walked_l254_254606


namespace calculate_rent_is_correct_l254_254887

noncomputable def requiredMonthlyRent 
  (purchase_cost : ℝ) 
  (monthly_set_aside_percent : ℝ)
  (annual_property_tax : ℝ)
  (annual_insurance : ℝ)
  (annual_return_percent : ℝ) : ℝ :=
  let annual_return := annual_return_percent * purchase_cost
  let total_yearly_expenses := annual_return + annual_property_tax + annual_insurance
  let monthly_expenses := total_yearly_expenses / 12
  let retention_rate := 1 - monthly_set_aside_percent
  monthly_expenses / retention_rate

theorem calculate_rent_is_correct 
  (purchase_cost : ℝ := 200000)
  (monthly_set_aside_percent : ℝ := 0.2)
  (annual_property_tax : ℝ := 5000)
  (annual_insurance : ℝ := 2400)
  (annual_return_percent : ℝ := 0.08) :
  requiredMonthlyRent purchase_cost monthly_set_aside_percent annual_property_tax annual_insurance annual_return_percent = 2437.50 :=
by
  sorry

end calculate_rent_is_correct_l254_254887


namespace exist_a_b_if_and_only_if_n_prime_divisor_1_mod_4_l254_254659

theorem exist_a_b_if_and_only_if_n_prime_divisor_1_mod_4
  (n : ℕ) (hn₁ : Odd n) (hn₂ : 0 < n) :
  (∃ (a b : ℕ), 0 < a ∧ 0 < b ∧ (4 : ℚ) / n = 1 / a + 1 / b) ↔
  ∃ p, p ∣ n ∧ Prime p ∧ p % 4 = 1 :=
by
  sorry

end exist_a_b_if_and_only_if_n_prime_divisor_1_mod_4_l254_254659


namespace geom_sequence_product_l254_254621

noncomputable def geom_seq (a : ℕ → ℝ) :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

theorem geom_sequence_product (a : ℕ → ℝ) (h1 : geom_seq a) (h2 : a 0 * a 4 = 4) :
  a 0 * a 1 * a 2 * a 3 * a 4 = 32 ∨ a 0 * a 1 * a 2 * a 3 * a 4 = -32 :=
by
  sorry

end geom_sequence_product_l254_254621


namespace new_mean_after_adding_14_to_each_of_15_numbers_l254_254133

theorem new_mean_after_adding_14_to_each_of_15_numbers (avg : ℕ) (n : ℕ) (n_sum : ℕ) (new_sum : ℕ) :
  avg = 40 →
  n = 15 →
  n_sum = n * avg →
  new_sum = n_sum + n * 14 →
  new_sum / n = 54 :=
by
  intros h_avg h_n h_n_sum h_new_sum
  sorry

end new_mean_after_adding_14_to_each_of_15_numbers_l254_254133


namespace tom_age_ratio_l254_254447

theorem tom_age_ratio (T N : ℝ) (h1 : T - N = 3 * (T - 4 * N)) : T / N = 5.5 :=
by
  sorry

end tom_age_ratio_l254_254447


namespace min_value_expression_l254_254079

theorem min_value_expression (α β : ℝ) : 
  ∃ a b : ℝ, 
    ((2 * Real.cos α + 5 * Real.sin β - 8) ^ 2 + 
    (2 * Real.sin α + 5 * Real.cos β - 15) ^ 2  = 100) :=
sorry

end min_value_expression_l254_254079


namespace net_investment_change_l254_254303

def initial_investment : ℝ := 100
def first_year_increase (init : ℝ) : ℝ := init * 1.50
def second_year_decrease (value : ℝ) : ℝ := value * 0.70

theorem net_investment_change :
  second_year_decrease (first_year_increase initial_investment) - initial_investment = 5 :=
by
  -- This will be placeholder proof
  sorry

end net_investment_change_l254_254303


namespace intersection_of_A_and_B_l254_254953

open Set

-- Definitions of sets A and B as per conditions in the problem
def A := {x : ℝ | -1 < x ∧ x < 2}
def B := {x : ℝ | -3 < x ∧ x ≤ 1}

-- The proof statement that A ∩ B = {x | -1 < x ∧ x ≤ 1}
theorem intersection_of_A_and_B : A ∩ B = {x : ℝ | -1 < x ∧ x ≤ 1} := by
  sorry

end intersection_of_A_and_B_l254_254953


namespace total_goals_in_league_l254_254847

variables (g1 g2 T : ℕ)

-- Conditions
def equal_goals : Prop := g1 = g2
def players_goals : Prop := g1 = 30
def total_goals_percentage : Prop := (g1 + g2) * 5 = T

-- Theorem to prove: Given the conditions, the total number of goals T should be 300
theorem total_goals_in_league (h1 : equal_goals g1 g2) (h2 : players_goals g1) (h3 : total_goals_percentage g1 g2 T) : T = 300 :=
sorry

end total_goals_in_league_l254_254847


namespace expression_evaluation_l254_254968

theorem expression_evaluation (m n : ℤ) (h : m * n = m + 3) : 2 * m * n + 3 * m - 5 * m * n - 10 = -19 := 
by 
  sorry

end expression_evaluation_l254_254968


namespace solve_equation_l254_254094

theorem solve_equation :
  ∃ x : Real, (x = 2 ∨ x = (-(1:Real) - Real.sqrt 17) / 2) ∧ (x^2 - |x - 1| - 3 = 0) :=
by
  sorry

end solve_equation_l254_254094


namespace perimeter_paper_count_l254_254801

theorem perimeter_paper_count (n : Nat) (h : n = 10) : 
  let top_side := n
  let right_side := n - 1
  let bottom_side := n - 1
  let left_side := n - 2
  top_side + right_side + bottom_side + left_side = 36 :=
by
  sorry

end perimeter_paper_count_l254_254801


namespace minimum_omega_l254_254434

theorem minimum_omega {ω : ℝ} (hω : ω > 0)
    (symmetry : ∃ k : ℤ, ∀ x : ℝ, 
      (sin (ω * x + ω * π / 2 + π / 3) = sin (-ω * x + ω * π / 2 + π / 3))) 
    : ω = 1 / 3 :=
by
  sorry

end minimum_omega_l254_254434


namespace distinct_real_roots_l254_254564

theorem distinct_real_roots (p : ℝ) : (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ (x1^2 - 2 * |x1| - p = 0) ∧ (x2^2 - 2 * |x2| - p = 0)) → p > -1 :=
by
  intro h
  sorry

end distinct_real_roots_l254_254564


namespace inequality_one_inequality_two_l254_254097

theorem inequality_one (x : ℝ) : 7 * x - 2 < 3 * (x + 2) → x < 2 :=
by
  sorry

theorem inequality_two (x : ℝ) : (x - 1) / 3 ≥ (x - 3) / 12 + 1 → x ≥ 13 / 3 :=
by
  sorry

end inequality_one_inequality_two_l254_254097


namespace infinite_product_value_l254_254936

noncomputable def infinite_product : ℝ :=
  ∏' (n : ℕ), 3^(n/(2^n * n))

theorem infinite_product_value :
  infinite_product = 15.5884572681 :=
sorry

#eval infinite_product -- should evaluate to 15.5884572681 (approximately)

end infinite_product_value_l254_254936


namespace smallest_k_l254_254982

theorem smallest_k (s : Finset ℕ) (h₁ : ∀ x ∈ s, 1 ≤ x ∧ x ≤ 99) (h₂ : s.card = 7) :
  ∃ a b ∈ s, a ≠ b ∧ (1 / 2 : ℝ) ≤ (b : ℝ) / (a : ℝ) ∧ (b : ℝ) / (a : ℝ) ≤ 2 := 
sorry

end smallest_k_l254_254982


namespace num_entrees_ordered_l254_254099

-- Define the conditions
def appetizer_cost: ℝ := 10
def entree_cost: ℝ := 20
def tip_rate: ℝ := 0.20
def total_spent: ℝ := 108

-- Define the theorem to prove the number of entrees ordered
theorem num_entrees_ordered : ∃ E : ℝ, (entree_cost * E) + appetizer_cost + (tip_rate * ((entree_cost * E) + appetizer_cost)) = total_spent ∧ E = 4 := 
by
  sorry

end num_entrees_ordered_l254_254099


namespace bruces_son_age_l254_254477

variable (Bruce_age : ℕ) (son_age : ℕ)
variable (h1 : Bruce_age = 36)
variable (h2 : Bruce_age + 6 = 3 * (son_age + 6))

theorem bruces_son_age :
  son_age = 8 :=
by {
  sorry
}

end bruces_son_age_l254_254477


namespace find_k_l254_254037

theorem find_k (k : ℝ) (h : ∃ x : ℝ, x^2 - 2 * x + 2 * k = 0 ∧ x = 1) : k = 1 / 2 :=
by {
  sorry 
}

end find_k_l254_254037


namespace rope_length_in_cm_l254_254717

-- Define the given conditions
def num_equal_pieces : ℕ := 150
def length_equal_piece_mm : ℕ := 75
def num_remaining_pieces : ℕ := 4
def length_remaining_piece_mm : ℕ := 100

-- Prove that the total length of the rope in centimeters is 1165
theorem rope_length_in_cm : (num_equal_pieces * length_equal_piece_mm + num_remaining_pieces * length_remaining_piece_mm) / 10 = 1165 :=
by
  sorry

end rope_length_in_cm_l254_254717


namespace hyperbola_condition_l254_254134

theorem hyperbola_condition (k : ℝ) : 
  (0 < k ∧ k < 1) → ¬((k > 1 ∨ k < -2) ↔ (0 < k ∧ k < 1)) :=
by
  intro hk
  sorry

end hyperbola_condition_l254_254134


namespace min_value_x2y3z_l254_254246

theorem min_value_x2y3z (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (h : 2 / x + 3 / y + 1 / z = 12) :
  x^2 * y^3 * z ≥ 1 / 64 :=
by
  sorry

end min_value_x2y3z_l254_254246


namespace find_wrong_quotient_l254_254227

-- Define the conditions
def correct_divisor : Nat := 21
def correct_quotient : Nat := 24
def mistaken_divisor : Nat := 12
def dividend : Nat := correct_divisor * correct_quotient

-- State the theorem to prove the wrong quotient
theorem find_wrong_quotient : (dividend / mistaken_divisor) = 42 := by
  sorry

end find_wrong_quotient_l254_254227


namespace geometric_progression_x_unique_l254_254615

theorem geometric_progression_x_unique (x : ℝ) :
  (70+x)^2 = (30+x)*(150+x) ↔ x = 10 := by
  sorry

end geometric_progression_x_unique_l254_254615


namespace remainder_of_n_mod_9_eq_5_l254_254103

-- Definitions of the variables and conditions
variables (a b c n : ℕ)

-- The given conditions as assumptions
def conditions : Prop :=
  a + b + c = 63 ∧
  a = c + 22 ∧
  n = 2 * a + 3 * b + 4 * c

-- The proof statement that needs to be proven
theorem remainder_of_n_mod_9_eq_5 (h : conditions a b c n) : n % 9 = 5 := 
  sorry

end remainder_of_n_mod_9_eq_5_l254_254103


namespace printing_machine_completion_time_l254_254885

-- Definitions of times in hours
def start_time : ℕ := 9 -- 9:00 AM
def half_job_time : ℕ := 12 -- 12:00 PM
def completion_time : ℕ := 15 -- 3:00 PM

-- Time taken to complete half the job
def half_job_duration : ℕ := half_job_time - start_time

-- Total time to complete the entire job
def total_job_duration : ℕ := 2 * half_job_duration

-- Proof that the machine will complete the job at 3:00 PM
theorem printing_machine_completion_time : 
    start_time + total_job_duration = completion_time :=
sorry

end printing_machine_completion_time_l254_254885


namespace exists_polynomials_Q_R_l254_254238

noncomputable def polynomial_with_integer_coeff (P : Polynomial ℤ) : Prop :=
  true

theorem exists_polynomials_Q_R (P : Polynomial ℤ) (hP : polynomial_with_integer_coeff P) :
  ∃ (Q R : Polynomial ℤ), 
    (∃ g : Polynomial ℤ, P * Q = Polynomial.comp g (Polynomial.X ^ 2)) ∧ 
    (∃ h : Polynomial ℤ, P * R = Polynomial.comp h (Polynomial.X ^ 3)) :=
by
  sorry

end exists_polynomials_Q_R_l254_254238


namespace jessica_total_monthly_payment_l254_254395

-- Definitions for the conditions
def basicCableCost : ℕ := 15
def movieChannelsCost : ℕ := 12
def sportsChannelsCost : ℕ := movieChannelsCost - 3

-- The statement to be proven
theorem jessica_total_monthly_payment :
  basicCableCost + movieChannelsCost + sportsChannelsCost = 36 := 
by
  sorry

end jessica_total_monthly_payment_l254_254395


namespace solution_set_ineq1_solution_set_ineq2_l254_254871

theorem solution_set_ineq1 (x : ℝ) : 
  (-3 * x ^ 2 + x + 1 > 0) ↔ (x ∈ Set.Ioo ((1 - Real.sqrt 13) / 6) ((1 + Real.sqrt 13) / 6)) := 
sorry

theorem solution_set_ineq2 (x : ℝ) : 
  (x ^ 2 - 2 * x + 1 ≤ 0) ↔ (x = 1) := 
sorry

end solution_set_ineq1_solution_set_ineq2_l254_254871


namespace freshman_to_sophomore_ratio_l254_254723

variable (f s : ℕ)

-- Define the participants from freshmen and sophomores
def freshmen_participants : ℕ := (3 * f) / 7
def sophomores_participants : ℕ := (2 * s) / 3

-- Theorem: There are 14/9 times as many freshmen as sophomores
theorem freshman_to_sophomore_ratio (h : freshmen_participants f = sophomores_participants s) : 
  9 * f = 14 * s :=
by
  sorry

end freshman_to_sophomore_ratio_l254_254723


namespace no_minimum_value_l254_254657

noncomputable def f (x : ℝ) : ℝ :=
  (1 + 1 / Real.log (Real.sqrt (x^2 + 10) - x)) *
  (1 + 2 / Real.log (Real.sqrt (x^2 + 10) - x))

theorem no_minimum_value : ¬ ∃ x, (0 < x ∧ x < 4.5) ∧ (∀ y, (0 < y ∧ y < 4.5) → f x ≤ f y) :=
sorry

end no_minimum_value_l254_254657


namespace solution_set_of_inequality_l254_254856

theorem solution_set_of_inequality :
  { x : ℝ | (x - 2) / (x + 3) ≥ 0 } = { x : ℝ | x < -3 } ∪ { x : ℝ | x ≥ 2 } := 
sorry

end solution_set_of_inequality_l254_254856


namespace minimum_value_of_f_l254_254653

open Real

noncomputable def f (x : ℝ) : ℝ := (2*x - 1) * exp x / (x - 1)

theorem minimum_value_of_f : f (3 / 2) = 4 * exp (3 / 2) :=
by
  -- Proof required here
  sorry

end minimum_value_of_f_l254_254653


namespace interval_intersection_l254_254177

/--
  This statement asserts that the intersection of the intervals (2/4, 3/4) and (2/5, 3/5)
  results in the interval (1/2, 0.6), which is the solution to the problem.
-/
theorem interval_intersection :
  { x : ℝ | 2 < 4 * x ∧ 4 * x < 3 ∧ 2 < 5 * x ∧ 5 * x < 3 } = { x : ℝ | 0.5 < x ∧ x < 0.6 } :=
by
  sorry

end interval_intersection_l254_254177


namespace remainder_division_x_squared_minus_one_l254_254498

variable (f g h : ℝ → ℝ)

noncomputable def remainder_when_divided_by_x_squared_minus_one (x : ℝ) : ℝ :=
-7 * x - 9

theorem remainder_division_x_squared_minus_one (h1 : ∀ x, f x = g x * (x - 1) + 8) (h2 : ∀ x, f x = h x * (x + 1) + 1) :
  ∀ x, f x % (x^2 - 1) = -7 * x - 9 :=
sorry

end remainder_division_x_squared_minus_one_l254_254498


namespace common_ratio_l254_254994

-- Problem Statement Definitions
variable (a1 q : ℝ)

-- Given Conditions
def a3 := a1 * q^2
def S3 := a1 * (1 + q + q^2)

-- Proof Statement
theorem common_ratio (h1 : a3 = 3/2) (h2 : S3 = 9/2) : q = 1 ∨ q = -1/2 := by
  sorry

end common_ratio_l254_254994


namespace sin_eq_cos_is_necessary_but_not_sufficient_for_alpha_eq_l254_254442

open Real

theorem sin_eq_cos_is_necessary_but_not_sufficient_for_alpha_eq :
  (∀ α : ℝ, sin α = cos α → ∃ k : ℤ, α = (k : ℝ) * π + π / 4) ∧
  (¬ ∀ k : ℤ, ∀ α : ℝ, α = (k : ℝ) * π + π / 4 → sin α = cos α) :=
by
  sorry

end sin_eq_cos_is_necessary_but_not_sufficient_for_alpha_eq_l254_254442


namespace triangle_inequality_l254_254068

theorem triangle_inequality (a : ℝ) (h₁ : a > 5) (h₂ : a < 19) : 5 < a ∧ a < 19 :=
by
  exact ⟨h₁, h₂⟩

end triangle_inequality_l254_254068


namespace number_of_females_l254_254163

theorem number_of_females 
  (total_students : ℕ) 
  (sampled_students : ℕ) 
  (sampled_female_less_than_male : ℕ) 
  (h_total : total_students = 1600)
  (h_sample : sampled_students = 200)
  (h_diff : sampled_female_less_than_male = 20) : 
  ∃ F M : ℕ, F + M = total_students ∧ (F / M : ℝ) = 9 / 11 ∧ F = 720 :=
by
  sorry

end number_of_females_l254_254163


namespace find_P_at_1_l254_254265

noncomputable def P (x : ℝ) : ℝ := x ^ 2 + x + 1008

theorem find_P_at_1 :
  (∀ x : ℝ, P (P x) - (P x) ^ 2 = x ^ 2 + x + 2016) →
  P 1 = 1010 := by
  intros H
  sorry

end find_P_at_1_l254_254265


namespace exterior_angle_of_regular_octagon_l254_254678

theorem exterior_angle_of_regular_octagon (sum_of_exterior_angles : ℝ) (n_sides : ℕ) (is_regular : n_sides = 8 ∧ sum_of_exterior_angles = 360) :
  sum_of_exterior_angles / n_sides = 45 := by
  sorry

end exterior_angle_of_regular_octagon_l254_254678


namespace correct_statement_A_l254_254569

-- Declare Avogadro's constant
def Avogadro_constant : ℝ := 6.022e23

-- Given conditions
def gas_mass_ethene : ℝ := 5.6 -- grams of ethylene
def gas_mass_cyclopropane : ℝ := 5.6 -- grams of cyclopropane
def gas_combined_carbon_atoms : ℝ := 0.4 * Avogadro_constant

-- Assertion to prove
theorem correct_statement_A :
    gas_combined_carbon_atoms = 0.4 * Avogadro_constant :=
by
  sorry

end correct_statement_A_l254_254569


namespace count_valid_arrangements_l254_254383

theorem count_valid_arrangements : 
  ∃ n : ℕ, (n = 5!) ∧
        (∃ z : ℕ, z = 4! ∧
        n = 120 ∧
        z = 24 ∧
        ∀ invalid_arrangements : ℕ, invalid_arrangements = 2 * z
        ∧ invalid_arrangements = 48
        ∧ (valid_arrangements = n - invalid_arrangements ∧ valid_arrangements = 72)) := 
sorry

end count_valid_arrangements_l254_254383


namespace solution_set_of_inequality_l254_254441

theorem solution_set_of_inequality (x : ℝ) :
  2 * |x - 1| - 1 < 0 ↔ (1 / 2) < x ∧ x < (3 / 2) :=
  sorry

end solution_set_of_inequality_l254_254441


namespace impossible_to_fill_grid_l254_254010

def is_impossible : Prop :=
  ∀ (grid : Fin 3 → Fin 3 → ℕ), 
  (∀ i j, grid i j ≠ grid i (j + 1) ∧ grid i j ≠ grid (i + 1) j) →
  (∀ i, (grid i 0) * (grid i 1) * (grid i 2) = 2005) →
  (∀ j, (grid 0 j) * (grid 1 j) * (grid 2 j) = 2005) →
  (grid 0 0) * (grid 1 1) * (grid 2 2) = 2005 →
  (grid 0 2) * (grid 1 1) * (grid 2 0) = 2005 →
  False

theorem impossible_to_fill_grid : is_impossible :=
  sorry

end impossible_to_fill_grid_l254_254010


namespace solution_set_of_bx2_ax_c_lt_zero_l254_254627

theorem solution_set_of_bx2_ax_c_lt_zero (a b c : ℝ) (h1 : a > 0) (h2 : b = a) (h3 : c = -6 * a) (h4 : ∀ x, ax^2 - bx + c < 0 ↔ -2 < x ∧ x < 3) :
  ∀ x, bx^2 + ax + c < 0 ↔ -3 < x ∧ x < 2 :=
by
  sorry

end solution_set_of_bx2_ax_c_lt_zero_l254_254627


namespace cubic_inequality_solution_l254_254331

theorem cubic_inequality_solution :
  ∀ x : ℝ, (x + 1) * (x + 2)^2 ≤ 0 ↔ -2 ≤ x ∧ x ≤ -1 := 
by 
  sorry

end cubic_inequality_solution_l254_254331


namespace arctan_sum_pi_div_two_l254_254901

theorem arctan_sum_pi_div_two :
  let α := Real.arctan (3 / 4)
  let β := Real.arctan (4 / 3)
  α + β = Real.pi / 2 := by
  sorry

end arctan_sum_pi_div_two_l254_254901


namespace printer_z_time_l254_254414

theorem printer_z_time (t_z : ℝ)
  (hx : (∀ (p : ℝ), p = 16))
  (hy : (∀ (q : ℝ), q = 12))
  (ratio : (16 / (1 /  ((1 / 12) + (1 / t_z)))) = 10 / 3) :
  t_z = 8 := by
  sorry

end printer_z_time_l254_254414


namespace solve_equation_l254_254551

theorem solve_equation (x : ℝ) (h : ((x^2 + 3*x + 4) / (x + 5)) = x + 6) : x = -13 / 4 :=
by sorry

end solve_equation_l254_254551


namespace pool_buckets_l254_254334

theorem pool_buckets (buckets_george_per_round buckets_harry_per_round rounds : ℕ) 
  (h_george : buckets_george_per_round = 2) 
  (h_harry : buckets_harry_per_round = 3) 
  (h_rounds : rounds = 22) : 
  buckets_george_per_round + buckets_harry_per_round * rounds = 110 := 
by 
  sorry

end pool_buckets_l254_254334


namespace geometric_sum_eight_terms_l254_254229

noncomputable def geometric_series_sum_8 (a r : ℝ) : ℝ :=
  a * (1 - r^8) / (1 - r)

theorem geometric_sum_eight_terms
  (a r : ℝ) (h_geom_pos : r > 0)
  (h_sum_two : a + a * r = 2)
  (h_sum_eight : a * r^2 + a * r^3 = 8) :
  geometric_series_sum_8 a r = 170 := 
sorry

end geometric_sum_eight_terms_l254_254229


namespace sum_geometric_series_l254_254039

-- Given the conditions
def q : ℕ := 2
def a3 : ℕ := 16
def n : ℕ := 2017
def a1 : ℕ := 4

-- Define the sum of the first n terms of a geometric series
noncomputable def geometricSeriesSum (a1 q n : ℕ) : ℕ :=
  a1 * (1 - q^n) / (1 - q)

-- State the problem
theorem sum_geometric_series :
  geometricSeriesSum a1 q n = 2^2019 - 4 :=
sorry

end sum_geometric_series_l254_254039


namespace solve_system_l254_254098

noncomputable def system_solutions : Set (ℝ × ℝ) :=
  {p | 
    let (x, y) := p in 
    ( ( (y^5 / x) ^ log 10 x = y ^ (2 * log 10 (x * y)) ) ∧ 
      ( x^2 - 2 * x * y - 4 * x - 3 * y^2 + 12 * y = 0 ) )}

theorem solve_system :
  system_solutions = { (2, 2), (9, 3), ( (9 - Real.sqrt 17) / 2, (Real.sqrt 17 - 1) / 2 ) } :=
sorry

end solve_system_l254_254098


namespace median_of_first_twelve_positive_integers_l254_254577

theorem median_of_first_twelve_positive_integers :
  let S := (set.range 12).image (λ x, x + 1) in  -- Set of first twelve positive integers
  median S = 6.5 :=
by
  sorry

end median_of_first_twelve_positive_integers_l254_254577


namespace enclosing_sphere_radius_and_area_l254_254020

theorem enclosing_sphere_radius_and_area :
  let r : ℝ := 2
  let centers := { (a, b, c) : ℝ × ℝ × ℝ | abs a = r ∧ abs b = r ∧ abs c = r }
  ∀ p ∈ centers, dist (0, 0, 0) p = r + ((2 - 1) * 4) :=
  r + sqrt 3 * 2 :=
  let enclosing_r := 2 * sqrt 3 + (2 : ℝ)
  let surface_area := 4 * π * (enclosing_r ^ 2)
  enclosing_r = 2 * sqrt 3 + 2 ∧ surface_area = 4 * π * (2 * sqrt 3 + 2) ^ 2
by
  sorry

end enclosing_sphere_radius_and_area_l254_254020


namespace interval_of_x_l254_254187

theorem interval_of_x (x : ℝ) : 
  (2 < 4 * x ∧ 4 * x < 3) → (2 < 5 * x ∧ 5 * x < 3) → (1 / 2 < x ∧ x < 3 / 5) :=
by
  sorry

end interval_of_x_l254_254187


namespace rectangular_to_cylindrical_4_neg4_6_l254_254162

theorem rectangular_to_cylindrical_4_neg4_6 :
  let x := 4
  let y := -4
  let z := 6
  let r := 4 * Real.sqrt 2
  let theta := (7 * Real.pi) / 4
  (r = Real.sqrt (x^2 + y^2)) ∧
  (Real.cos theta = x / r) ∧
  (Real.sin theta = y / r) ∧
  0 ≤ theta ∧ theta < 2 * Real.pi ∧
  z = 6 → 
  (r, theta, z) = (4 * Real.sqrt 2, (7 * Real.pi) / 4, 6) :=
by
  sorry

end rectangular_to_cylindrical_4_neg4_6_l254_254162


namespace leftover_stickers_l254_254085

-- Definitions for each person's stickers
def ninaStickers : ℕ := 53
def oliverStickers : ℕ := 68
def pattyStickers : ℕ := 29

-- The number of stickers in a package
def packageSize : ℕ := 18

-- The total number of stickers
def totalStickers : ℕ := ninaStickers + oliverStickers + pattyStickers

-- Proof that the number of leftover stickers is 6 when all stickers are divided into packages of 18
theorem leftover_stickers : totalStickers % packageSize = 6 := by
  sorry

end leftover_stickers_l254_254085


namespace sally_cost_is_42000_l254_254249

-- Definitions for conditions
def lightningCost : ℕ := 140000
def materCost : ℕ := (10 * lightningCost) / 100
def sallyCost : ℕ := 3 * materCost

-- Theorem statement
theorem sally_cost_is_42000 : sallyCost = 42000 := by
  sorry

end sally_cost_is_42000_l254_254249


namespace smallest_four_digit_divisible_by_3_5_7_11_l254_254188

theorem smallest_four_digit_divisible_by_3_5_7_11 : 
  ∃ n : ℕ, n >= 1000 ∧ n < 10000 ∧ 
          n % 3 = 0 ∧ n % 5 = 0 ∧ n % 7 = 0 ∧ n % 11 = 0 ∧ n = 1155 :=
by
  sorry

end smallest_four_digit_divisible_by_3_5_7_11_l254_254188


namespace find_number_l254_254360

theorem find_number (a : ℕ) (h : a = 105) : 
  a^3 / (49 * 45 * 25) = 21 :=
by
  sorry

end find_number_l254_254360


namespace third_number_in_pascals_triangle_row_51_l254_254288

theorem third_number_in_pascals_triangle_row_51 :
  let n := 51 in 
  ∃ result, result = (n * (n - 1)) / 2 ∧ result = 1275 :=
by
  let n := 51
  use (n * (n - 1)) / 2
  split
  . rfl
  . exact Nat.div_eq_of_eq_mul_left (by norm_num) (by norm_num; ring)
  sorry -- This 'sorry' is provided to formally conclude the directive


end third_number_in_pascals_triangle_row_51_l254_254288


namespace problem_l254_254828

theorem problem (a : ℤ) (n : ℕ) : (a + 1) ^ (2 * n + 1) + a ^ (n + 2) ∣ a ^ 2 + a + 1 :=
sorry

end problem_l254_254828


namespace intersection_or_parallel_lines_l254_254445

structure Triangle (Point : Type) :=
  (A B C : Point)

structure Plane (Point : Type) :=
  (P1 P2 P3 P4 : Point)

variables {Point : Type}
variables (triABC triA1B1C1 : Triangle Point)
variables (plane1 plane2 plane3 : Plane Point)

-- Intersection conditions
variable (AB_intersects_A1B1 : (triABC.A, triABC.B) = (triA1B1C1.A, triA1B1C1.B))
variable (BC_intersects_B1C1 : (triABC.B, triABC.C) = (triA1B1C1.B, triA1B1C1.C))
variable (CA_intersects_C1A1 : (triABC.C, triABC.A) = (triA1B1C1.C, triA1B1C1.A))

theorem intersection_or_parallel_lines :
  ∃ P : Point, (
    (∃ A1 : Point, (triABC.A, A1) = (P, P)) ∧
    (∃ B1 : Point, (triABC.B, B1) = (P, P)) ∧
    (∃ C1 : Point, (triABC.C, C1) = (P, P))
  ) ∨ (
    (∃ d1 d2 d3 : Point, 
      (∀ A1 B1 C1 : Point,
        (triABC.A, A1) = (d1, d1) ∧ 
        (triABC.B, B1) = (d2, d2) ∧ 
        (triABC.C, C1) = (d3, d3)
      )
    )
  ) := by
  sorry

end intersection_or_parallel_lines_l254_254445


namespace interval_intersection_l254_254174

/--
  This statement asserts that the intersection of the intervals (2/4, 3/4) and (2/5, 3/5)
  results in the interval (1/2, 0.6), which is the solution to the problem.
-/
theorem interval_intersection :
  { x : ℝ | 2 < 4 * x ∧ 4 * x < 3 ∧ 2 < 5 * x ∧ 5 * x < 3 } = { x : ℝ | 0.5 < x ∧ x < 0.6 } :=
by
  sorry

end interval_intersection_l254_254174


namespace total_boys_in_class_l254_254599

/-- 
  Given 
    - n + 1 positions in a circle, where n is the number of boys and 1 position for the teacher.
    - The boy at the 6th position is exactly opposite to the boy at the 16th position.
  Prove that the total number of boys in the class is 20.
-/
theorem total_boys_in_class (n : ℕ) (h1 : n + 1 > 16) (h2 : (6 + 10) % (n + 1) = 16):
  n = 20 := 
by 
  sorry

end total_boys_in_class_l254_254599


namespace divisor_of_7_l254_254408

theorem divisor_of_7 (a n : ℤ) (h1 : a ≥ 1) (h2 : a ∣ (n + 2)) (h3 : a ∣ (n^2 + n + 5)) : a = 1 ∨ a = 7 :=
by
  sorry

end divisor_of_7_l254_254408


namespace integer_roots_condition_l254_254613

theorem integer_roots_condition (a : ℝ) (h_pos : 0 < a) :
  (∀ x y : ℤ, (a ^ 2 * x ^ 2 + a * x + 1 - 13 * a ^ 2 = 0) ∧ (a ^ 2 * y ^ 2 + a * y + 1 - 13 * a ^ 2 = 0)) ↔
  (a = 1 ∨ a = 1/3 ∨ a = 1/4) :=
by sorry

end integer_roots_condition_l254_254613


namespace annuity_payment_l254_254285

variable (P : ℝ) (A : ℝ) (i : ℝ) (n1 n2 : ℕ)

-- Condition: Principal amount
axiom principal_amount : P = 24000

-- Condition: Annual installment for the first 5 years
axiom annual_installment : A = 1500 

-- Condition: Annual interest rate
axiom interest_rate : i = 0.045 

-- Condition: Years before equal annual installments
axiom years_before_installment : n1 = 5 

-- Condition: Years for repayment after the first 5 years
axiom repayment_years : n2 = 7 

-- Remaining debt after n1 years
noncomputable def remaining_debt_after_n1 : ℝ :=
  P * (1 + i) ^ n1 - A * ((1 + i) ^ n1 - 1) / i

-- Annual payment for n2 years to repay the remaining debt
noncomputable def annual_payment (D : ℝ) : ℝ :=
  D * (1 + i) ^ n2 / (((1 + i) ^ n2 - 1) / i)

axiom remaining_debt_amount : remaining_debt_after_n1 P A i n1 = 21698.685 

theorem annuity_payment : annual_payment (remaining_debt_after_n1 P A i n1) = 3582 := by
  sorry

end annuity_payment_l254_254285


namespace number_of_n_for_prime_l254_254768

theorem number_of_n_for_prime (n : ℕ) : (n > 0) → ∃! n, Nat.Prime (n * (n + 2)) :=
by 
  sorry

end number_of_n_for_prime_l254_254768


namespace blueberries_per_basket_l254_254253

-- Definitions based on the conditions
def total_blueberries : ℕ := 200
def total_baskets : ℕ := 10

-- Statement to be proven
theorem blueberries_per_basket : total_blueberries / total_baskets = 20 := 
by
  sorry

end blueberries_per_basket_l254_254253


namespace median_of_first_twelve_positive_integers_l254_254576

theorem median_of_first_twelve_positive_integers : 
  let first_twelve_positive_integers := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12] in
  (first_twelve_positive_integers.length = 12) →
  (first_twelve_positive_integers.nth (first_twelve_positive_integers.length / 2 - 1) = some 6) →
  (first_twelve_positive_integers.nth (first_twelve_positive_integers.length / 2) = some 7) →
  (6 + 7) / 2 = 6.5 :=
by
  sorry

end median_of_first_twelve_positive_integers_l254_254576


namespace repeatingDecimals_fraction_eq_l254_254758

noncomputable def repeatingDecimalsSum : ℚ :=
  let x : ℚ := 1 / 3
  let y : ℚ := 4 / 99
  let z : ℚ := 5 / 999
  x + y + z

theorem repeatingDecimals_fraction_eq : repeatingDecimalsSum = 42 / 111 :=
  sorry

end repeatingDecimals_fraction_eq_l254_254758


namespace find_divisor_nearest_to_3105_l254_254490

def nearest_divisible_number (n : ℕ) (d : ℕ) : ℕ :=
  if n % d = 0 then n else n + d - (n % d)

theorem find_divisor_nearest_to_3105 (d : ℕ) (h : nearest_divisible_number 3105 d = 3108) : d = 3 :=
by
  sorry

end find_divisor_nearest_to_3105_l254_254490


namespace longest_boat_length_l254_254002

theorem longest_boat_length (a : ℝ) (c : ℝ) 
  (parallel_banks : ∀ x y : ℝ, (x = y) ∨ (x = -y)) 
  (right_angle_bend : ∃ b : ℝ, b = a) :
  c = 2 * a * Real.sqrt 2 := by
  sorry

end longest_boat_length_l254_254002


namespace min_value_eq_144_l254_254656

noncomputable def min_value (x y z w : ℝ) (h_pos_x : x > 0) (h_pos_y : y > 0) (h_pos_z : z > 0) (h_pos_w : w > 0) (h_sum : x + y + z + w = 1) : ℝ :=
  if x <= 0 ∨ y <= 0 ∨ z <= 0 ∨ w <= 0 then 0 else (x + y + z) / (x * y * z * w)

theorem min_value_eq_144 (x y z w : ℝ) (h_pos_x : x > 0) (h_pos_y : y > 0) (h_pos_z : z > 0) (h_pos_w : w > 0) (h_sum : x + y + z + w = 1) :
  min_value x y z w h_pos_x h_pos_y h_pos_z h_pos_w h_sum = 144 :=
sorry

end min_value_eq_144_l254_254656


namespace percent_calculation_l254_254590

-- Given conditions
def part : ℝ := 120.5
def whole : ℝ := 80.75

-- Theorem statement
theorem percent_calculation : (part / whole) * 100 = 149.26 := 
sorry

end percent_calculation_l254_254590


namespace prime_digit_three_digit_numbers_l254_254217

theorem prime_digit_three_digit_numbers : 
  let primes := {2, 3, 5, 7}
  in (⌊3⌋ : fin 10 → ℕ) * |primes| = 64 := 
by {
  let primes := {2, 3, 5, 7}
  calc (4 : ℝ)^3 
  : sorry
}

end prime_digit_three_digit_numbers_l254_254217


namespace g_diff_l254_254924

def g (x : ℝ) : ℝ := 2 * x^3 + 5 * x^2 - 2 * x - 1

theorem g_diff (x h : ℝ) : g (x + h) - g x = h * (6 * x^2 + 6 * x * h + 2 * h^2 + 10 * x + 5 * h - 2) := 
by
  sorry

end g_diff_l254_254924


namespace intersection_of_A_and_B_l254_254045

noncomputable def A : Set ℕ := {x | 2 ≤ x ∧ x ≤ 4}
def B : Set ℕ := {x | x ≤ 3}

theorem intersection_of_A_and_B : A ∩ B = {2, 3} :=
by
  sorry

end intersection_of_A_and_B_l254_254045


namespace probability_factor_less_than_eight_l254_254287

theorem probability_factor_less_than_eight (n : ℕ) (h72 : n = 72) :
  (∃ k < 8, k ∣ n) →
  (∃ p q, p/q = 5/12) :=
by
  sorry

end probability_factor_less_than_eight_l254_254287


namespace combin_sum_l254_254478

def combin (n m : ℕ) : ℕ := Nat.factorial n / (Nat.factorial m * Nat.factorial (n - m))

theorem combin_sum (n : ℕ) (h₁ : n = 99) : combin n 2 + combin n 3 = 161700 := by
  sorry

end combin_sum_l254_254478


namespace sum_of_100_and_98_consecutive_diff_digits_l254_254732

def S100 (n : ℕ) : ℕ := 50 * (2 * n + 99)
def S98 (n : ℕ) : ℕ := 49 * (2 * n + 297)

theorem sum_of_100_and_98_consecutive_diff_digits (n : ℕ) :
  ¬ (S100 n % 10 = S98 n % 10) :=
sorry

end sum_of_100_and_98_consecutive_diff_digits_l254_254732


namespace number_of_ways_to_fill_grid_l254_254224

open Finset

noncomputable def count_grid_filling : ℕ :=
  (factorial 3) ^ 3

theorem number_of_ways_to_fill_grid : count_grid_filling = 216 :=
sorry

end number_of_ways_to_fill_grid_l254_254224


namespace repeated_pair_exists_l254_254402

theorem repeated_pair_exists (a : Fin 99 → Fin 10)
  (h1 : ∀ n : Fin 98, a n = 1 → a (n + 1) ≠ 2)
  (h2 : ∀ n : Fin 98, a n = 3 → a (n + 1) ≠ 4) :
  ∃ k l : Fin 98, k ≠ l ∧ a k = a l ∧ a (k + 1) = a (l + 1) :=
sorry

end repeated_pair_exists_l254_254402


namespace number_of_yellow_highlighters_l254_254803

-- Definitions based on the given conditions
def total_highlighters : Nat := 12
def pink_highlighters : Nat := 6
def blue_highlighters : Nat := 4

-- Statement to prove the question equals the correct answer given the conditions
theorem number_of_yellow_highlighters : 
  ∃ y : Nat, y = total_highlighters - (pink_highlighters + blue_highlighters) := 
by
  -- TODO: The proof will be filled in here
  sorry

end number_of_yellow_highlighters_l254_254803


namespace no_egg_arrangements_possible_l254_254084

noncomputable def num_egg_arrangements 
  (total_eggs : ℕ) 
  (type_A_eggs : ℕ) 
  (type_B_eggs : ℕ)
  (type_C_eggs : ℕ)
  (groups : ℕ)
  (ratio_A : ℕ) 
  (ratio_B : ℕ) 
  (ratio_C : ℕ) : ℕ :=
if (total_eggs = type_A_eggs + type_B_eggs + type_C_eggs) ∧ 
   (type_A_eggs / groups = ratio_A) ∧ 
   (type_B_eggs / groups = ratio_B) ∧ 
   (type_C_eggs / groups = ratio_C) then 0 else 0

theorem no_egg_arrangements_possible :
  num_egg_arrangements 35 15 12 8 5 2 3 1 = 0 := 
by sorry

end no_egg_arrangements_possible_l254_254084


namespace student_count_l254_254064

theorem student_count 
( M S N : ℕ ) 
(h1 : N - M = 10) 
(h2 : N - S = 15) 
(h3 : N - (M + S - 7) = 2) : 
N = 34 :=
by
  sorry

end student_count_l254_254064


namespace right_triangle_area_l254_254111

theorem right_triangle_area (a b c : ℕ) (h1 : a = 16) (h2 : b = 30) (h3 : c = 34) 
(h4 : a^2 + b^2 = c^2) : 
   1 / 2 * a * b = 240 :=
by 
  sorry

end right_triangle_area_l254_254111


namespace repeatingDecimals_fraction_eq_l254_254756

noncomputable def repeatingDecimalsSum : ℚ :=
  let x : ℚ := 1 / 3
  let y : ℚ := 4 / 99
  let z : ℚ := 5 / 999
  x + y + z

theorem repeatingDecimals_fraction_eq : repeatingDecimalsSum = 42 / 111 :=
  sorry

end repeatingDecimals_fraction_eq_l254_254756


namespace find_b9_l254_254681

theorem find_b9 {b : ℕ → ℕ} 
  (h1 : ∀ n, b (n + 2) = b (n + 1) + b n)
  (h2 : b 8 = 100) :
  b 9 = 194 :=
sorry

end find_b9_l254_254681


namespace Sarah_is_26_l254_254261

noncomputable def Sarah_age (mark_age billy_age ana_age : ℕ): ℕ :=
  3 * mark_age - 4

def Mark_age (billy_age : ℕ): ℕ :=
  billy_age + 4

def Billy_age (ana_age : ℕ): ℕ :=
  ana_age / 2

def Ana_age : ℕ := 15 - 3

theorem Sarah_is_26 : Sarah_age (Mark_age (Billy_age Ana_age)) (Billy_age Ana_age) Ana_age = 26 := 
by
  sorry

end Sarah_is_26_l254_254261


namespace equal_remainders_prime_condition_l254_254340

theorem equal_remainders_prime_condition {p x : ℕ} (hp : Nat.Prime p) (hx_pos : 0 < x) 
  (h1 : ∃ r, x % p = r ∧ p^2 % x = r) :
  ∃ r, r = 0 ∨ r = 1 where
    r = x % p :=
by
  sorry

end equal_remainders_prime_condition_l254_254340


namespace min_time_to_shoe_horses_l254_254872

-- Definitions based on the conditions
def n_blacksmiths : ℕ := 48
def n_horses : ℕ := 60
def t_hoof : ℕ := 5 -- minutes per hoof
def n_hooves : ℕ := n_horses * 4
def total_time : ℕ := n_hooves * t_hoof
def t_min : ℕ := total_time / n_blacksmiths

-- The theorem states that the minimal time required is 25 minutes
theorem min_time_to_shoe_horses : t_min = 25 := by
  sorry

end min_time_to_shoe_horses_l254_254872


namespace square_division_rectangles_l254_254718

theorem square_division_rectangles (k l : ℕ) (h_square : exists s : ℝ, 0 < s) 
(segment_division : ∀ (p q : ℝ), exists r : ℕ, r = s * k ∧ r = s * l) :
  ∃ n : ℕ, n = k * l :=
sorry

end square_division_rectangles_l254_254718


namespace E_72_eq_9_l254_254403

def E (n : ℕ) : ℕ :=
  -- Assume a function definition counting representations
  -- (this function body is a placeholder, as the exact implementation
  -- is not part of the problem statement)
  sorry

theorem E_72_eq_9 :
  E 72 = 9 :=
sorry

end E_72_eq_9_l254_254403


namespace halfway_between_fractions_l254_254741

-- Definitions used in the conditions
def one_eighth := (1 : ℚ) / 8
def three_tenths := (3 : ℚ) / 10

-- The mathematical assertion to prove
theorem halfway_between_fractions : (one_eighth + three_tenths) / 2 = 17 / 80 := by
  sorry

end halfway_between_fractions_l254_254741


namespace geometric_sequence_third_term_l254_254304

theorem geometric_sequence_third_term :
  ∀ (a r : ℕ), a = 2 ∧ a * r ^ 3 = 162 → a * r ^ 2 = 18 :=
by
  intros a r
  intro h
  have ha : a = 2 := h.1
  have h_fourth_term : a * r ^ 3 = 162 := h.2
  sorry

end geometric_sequence_third_term_l254_254304


namespace milan_billed_minutes_l254_254191

-- Variables corresponding to the conditions
variables (f r b : ℝ) (m : ℕ)

-- The conditions of the problem
def conditions : Prop :=
  f = 2 ∧ r = 0.12 ∧ b = 23.36 ∧ b = f + r * m

-- The theorem based on given conditions and aiming to prove that m = 178
theorem milan_billed_minutes (h : conditions f r b m) : m = 178 :=
sorry

end milan_billed_minutes_l254_254191


namespace intersection_is_correct_l254_254248

def setA : Set ℕ := {0, 1, 2}
def setB : Set ℕ := {1, 2, 3}

theorem intersection_is_correct : setA ∩ setB = {1, 2} := by
  sorry

end intersection_is_correct_l254_254248


namespace company_a_taxis_l254_254019

variable (a b : ℕ)

theorem company_a_taxis
  (h1 : 5 * a < 56)
  (h2 : 6 * a > 56)
  (h3 : 4 * b < 56)
  (h4 : 5 * b > 56)
  (h5 : b = a + 3) :
  a = 10 := by
  sorry

end company_a_taxis_l254_254019


namespace find_x_condition_l254_254946

theorem find_x_condition (x : ℚ) :
  (∀ y : ℚ, 10 * x * y - 15 * y + 3 * x - 9 / 2 = 0) → x = 3 / 2 :=
begin
  sorry
end

end find_x_condition_l254_254946


namespace smallest_of_powers_l254_254294

theorem smallest_of_powers :
  min (2^55) (min (3^44) (min (5^33) (6^22))) = 2^55 :=
by
  sorry

end smallest_of_powers_l254_254294


namespace no_groups_of_six_l254_254473

theorem no_groups_of_six (x y z : ℕ) 
  (h1 : (2 * x + 6 * y + 10 * z) / (x + y + z) = 5)
  (h2 : (2 * x + 30 * y + 90 * z) / (2 * x + 6 * y + 10 * z) = 7) : 
  y = 0 := 
sorry

end no_groups_of_six_l254_254473


namespace maximize_revenue_l254_254708

theorem maximize_revenue (p : ℝ) (h₁ : p ≤ 30) (h₂ : p = 18.75) : 
  ∃(R : ℝ), R = p * (150 - 4 * p) :=
by
  sorry

end maximize_revenue_l254_254708


namespace pascal_triangle_third_number_l254_254291

theorem pascal_triangle_third_number (n : ℕ) (h : n + 1 = 52) : (nat.choose n 2) = 1275 := by
  have h_n : n = 51 := by
    linarith
  rw [h_n]
  norm_num

end pascal_triangle_third_number_l254_254291


namespace smallest_natur_number_with_units_digit_6_and_transf_is_four_times_l254_254616

open Nat

theorem smallest_natur_number_with_units_digit_6_and_transf_is_four_times (n : ℕ) :
  (n % 10 = 6 ∧ ∃ m, 6 * 10 ^ (m - 1) + n / 10 = 4 * n) → n = 153846 :=
by 
  sorry

end smallest_natur_number_with_units_digit_6_and_transf_is_four_times_l254_254616


namespace mean_of_five_numbers_is_correct_l254_254279

-- Define the sum of the five numbers
def sum_of_five_numbers : ℚ := 3 / 4

-- Define the number of numbers
def number_of_numbers : ℚ := 5

-- Define the mean
def mean_of_five_numbers := sum_of_five_numbers / number_of_numbers

-- State the theorem
theorem mean_of_five_numbers_is_correct : mean_of_five_numbers = 3 / 20 :=
by
  -- The proof is omitted, use sorry to indicate this.
  sorry

end mean_of_five_numbers_is_correct_l254_254279


namespace total_jelly_beans_l254_254529

-- Given conditions:
def vanilla_jelly_beans : ℕ := 120
def grape_jelly_beans := 5 * vanilla_jelly_beans + 50

-- The proof problem:
theorem total_jelly_beans :
  vanilla_jelly_beans + grape_jelly_beans = 770 :=
by
  -- Proof steps would go here
  sorry

end total_jelly_beans_l254_254529


namespace interval_of_x_l254_254180

theorem interval_of_x (x : ℝ) :
  (2 < 4 * x ∧ 4 * x < 3) ∧ (2 < 5 * x ∧ 5 * x < 3) ↔ (1 / 2 < x ∧ x < 3 / 5) := by
  sorry

end interval_of_x_l254_254180


namespace solve_y_equation_l254_254554

theorem solve_y_equation :
  ∃ y : ℚ, 4 * (5 * y + 3) - 3 = -3 * (2 - 8 * y) ∧ y = 15 / 4 :=
by
  sorry

end solve_y_equation_l254_254554


namespace chess_club_girls_l254_254878

theorem chess_club_girls (B G : ℕ) (h1 : B + G = 32) (h2 : (1 / 2 : ℝ) * G + B = 20) : G = 24 :=
by
  -- proof
  sorry

end chess_club_girls_l254_254878


namespace minimum_omega_for_symmetric_curve_l254_254425

theorem minimum_omega_for_symmetric_curve (ω : ℝ) (hω : ω > 0) :
  (∀ x : ℝ, sin (ω * (x + π / 2) + π / 3) = sin (-ω * (x + π / 2) + π / 3)) ↔ ω = 1 / 3 :=
by
  sorry

end minimum_omega_for_symmetric_curve_l254_254425


namespace complement_of_A_relative_to_U_l254_254652

def U : Set ℕ := {1, 2, 3, 4, 5, 6}
def A : Set ℕ := {x | 1 ≤ x ∧ x ≤ 3}

theorem complement_of_A_relative_to_U : (U \ A) = {4, 5, 6} := 
by
  sorry

end complement_of_A_relative_to_U_l254_254652


namespace rajas_income_l254_254302

theorem rajas_income (I : ℝ) 
  (h1 : 0.60 * I + 0.10 * I + 0.10 * I + 5000 = I) : I = 25000 :=
by
  sorry

end rajas_income_l254_254302


namespace number_of_three_digit_prime_digits_l254_254216

theorem number_of_three_digit_prime_digits : 
  let primes := {2, 3, 5, 7} in
  ∃ n : ℕ, n = (primes.toFinset.card) ^ 3 ∧ n = 64 :=
by
  -- let primes be the set of prime digits 2, 3, 5, 7
  let primes := {2, 3, 5, 7}
  -- assert the cardinality of primes is 4
  have h_primes_card : primes.toFinset.card = 4 := by sorry
  -- assert the number of three-digit integers with each digit being prime is 4^3
  let n := (primes.toFinset.card) ^ 3
  -- assert n is equal to 64
  have h_n_64 : n = 64 := by sorry
  -- hence conclude the proof
  exact ⟨n, rfl, h_n_64⟩

end number_of_three_digit_prime_digits_l254_254216


namespace conditional_probabilities_l254_254449

def PA : ℝ := 0.20
def PB : ℝ := 0.18
def PAB : ℝ := 0.12

theorem conditional_probabilities :
  PAB / PB = 2 / 3 ∧ PAB / PA = 3 / 5 := by
  sorry

end conditional_probabilities_l254_254449


namespace radius_for_visibility_l254_254144

theorem radius_for_visibility (r : ℝ) (h₁ : r > 0)
  (h₂ : ∃ o : ℝ, ∀ (s : ℝ), s = 3 → o = 0):
  (∃ p : ℝ, p = 1/3) ∧ (r = 3.6) :=
sorry

end radius_for_visibility_l254_254144


namespace intersection_correct_l254_254630

-- Define sets M and N
def M := {x : ℝ | x < 1}
def N := {x : ℝ | Real.log (2 * x + 1) > 0}

-- Define the intersection of M and N
def M_intersect_N := {x : ℝ | 0 < x ∧ x < 1}

-- Prove that M_intersect_N is the correct intersection
theorem intersection_correct : M ∩ N = M_intersect_N :=
by
  sorry

end intersection_correct_l254_254630


namespace inequality_abc_l254_254093

theorem inequality_abc (a b c : ℝ) : a^2 + 4 * b^2 + 8 * c^2 ≥ 3 * a * b + 4 * b * c + 2 * c * a :=
by
  sorry

end inequality_abc_l254_254093


namespace function_d_has_no_boundary_point_l254_254329

def is_boundary_point (f : ℝ → ℝ) (x₀ : ℝ) : Prop :=
  (∃ x₁ < x₀, f x₁ = 0) ∧ (∃ x₂ > x₀, f x₂ = 0)

def f_a (b : ℝ) (x : ℝ) : ℝ := x^2 + b * x - 2
def f_b (x : ℝ) : ℝ := abs (x^2 - 3)
def f_c (x : ℝ) : ℝ := 1 - abs (x - 2)
def f_d (x : ℝ) : ℝ := x^3 + x

theorem function_d_has_no_boundary_point :
  ¬ ∃ x₀ : ℝ, is_boundary_point f_d x₀ :=
sorry

end function_d_has_no_boundary_point_l254_254329


namespace repeating_sum_to_fraction_l254_254744

theorem repeating_sum_to_fraction :
  (0.333333333333333 ~ 1/3) ∧ 
  (0.0404040404040401 ~ 4/99) ∧ 
  (0.005005005005001 ~ 5/999) →
  (0.333333333333333 + 0.0404040404040401 + 0.005005005005001) = (112386 / 296703) := 
by
  repeat { sorry }

end repeating_sum_to_fraction_l254_254744


namespace a_squared_plus_b_squared_equals_61_l254_254799

theorem a_squared_plus_b_squared_equals_61 (a b : ℝ) (h1 : a + b = -9) (h2 : a = 30 / b) : a^2 + b^2 = 61 :=
sorry

end a_squared_plus_b_squared_equals_61_l254_254799


namespace pistachio_shells_percentage_l254_254467

theorem pistachio_shells_percentage (total_pistachios : ℕ) (opened_shelled_pistachios : ℕ) (P : ℝ) :
  total_pistachios = 80 →
  opened_shelled_pistachios = 57 →
  (0.75 : ℝ) * (P / 100) * (total_pistachios : ℝ) = (opened_shelled_pistachios : ℝ) →
  P = 95 :=
by
  intros h_total h_opened h_equation
  sorry

end pistachio_shells_percentage_l254_254467


namespace negation_exists_equiv_forall_l254_254846

theorem negation_exists_equiv_forall :
  (¬ (∃ x : ℤ, x^2 + 2*x - 1 < 0)) ↔ (∀ x : ℤ, x^2 + 2*x - 1 ≥ 0) :=
by
  sorry

end negation_exists_equiv_forall_l254_254846


namespace lcm_of_three_numbers_l254_254273

theorem lcm_of_three_numbers :
  ∀ (a b c : ℕ) (hcf : ℕ), hcf = Nat.gcd (Nat.gcd a b) c → a = 136 → b = 144 → c = 168 → hcf = 8 →
  Nat.lcm (Nat.lcm a b) c = 411264 :=
by
  intros a b c hcf h1 h2 h3 h4
  rw [h2, h3, h4]
  sorry

end lcm_of_three_numbers_l254_254273


namespace right_triangle_shorter_leg_l254_254230
-- Import all necessary libraries

-- Define the problem
theorem right_triangle_shorter_leg (a b c : ℕ) (h1 : a < b) (h2 : b < c) (h3 : c = 65) (h4 : a^2 + b^2 = c^2) :
  a = 25 :=
sorry

end right_triangle_shorter_leg_l254_254230


namespace total_exterior_angles_l254_254065

-- Define that the sum of the exterior angles of any convex polygon is 360 degrees
def sum_exterior_angles (n : ℕ) : ℝ := 360

-- Given four polygons: a triangle, a quadrilateral, a pentagon, and a hexagon
def triangle_exterior_sum := sum_exterior_angles 3
def quadrilateral_exterior_sum := sum_exterior_angles 4
def pentagon_exterior_sum := sum_exterior_angles 5
def hexagon_exterior_sum := sum_exterior_angles 6

-- The total sum of the exterior angles of these four polygons combined
def total_exterior_angle_sum := 
  triangle_exterior_sum + 
  quadrilateral_exterior_sum + 
  pentagon_exterior_sum + 
  hexagon_exterior_sum

-- The final proof statement
theorem total_exterior_angles : total_exterior_angle_sum = 1440 := by
  sorry

end total_exterior_angles_l254_254065


namespace sufficient_not_necessary_condition_l254_254338

variable (a : ℝ)

def M := {1, a}
def N := {-1, 0, 1}

theorem sufficient_not_necessary_condition : 
  (M ⊆ N ↔ (a = 0 ∨ a = -1)) → (M ⊆ N) ∧ (a = 0 → M ⊆ N) ∧ ¬(a = 0 → ¬M ⊆ N) :=
by 
sorry

end sufficient_not_necessary_condition_l254_254338


namespace problem1_solution_problem2_solution_l254_254009

-- Proof for Problem 1
theorem problem1_solution (x y : ℝ) 
(h1 : x - y - 1 = 4)
(h2 : 4 * (x - y) - y = 5) : 
x = 20 ∧ y = 15 := sorry

-- Proof for Problem 2
theorem problem2_solution (x : ℝ) 
(h1 : 4 * x - 1 ≥ x + 1)
(h2 : (1 - x) / 2 < x) : 
x ≥ 2 / 3 := sorry

end problem1_solution_problem2_solution_l254_254009


namespace third_restaurant_meals_per_day_l254_254214

-- Define the daily meals served by the first two restaurants
def meals_first_restaurant_per_day : ℕ := 20
def meals_second_restaurant_per_day : ℕ := 40

-- Define the total meals served by all three restaurants per week
def total_meals_per_week : ℕ := 770

-- Define the weekly meals served by the first two restaurants
def meals_first_restaurant_per_week : ℕ := meals_first_restaurant_per_day * 7
def meals_second_restaurant_per_week : ℕ := meals_second_restaurant_per_day * 7

-- Total weekly meals served by the first two restaurants
def total_meals_first_two_restaurants_per_week : ℕ := meals_first_restaurant_per_week + meals_second_restaurant_per_week

-- Weekly meals served by the third restaurant
def meals_third_restaurant_per_week : ℕ := total_meals_per_week - total_meals_first_two_restaurants_per_week

-- Convert weekly meals served by the third restaurant to daily meals
def meals_third_restaurant_per_day : ℕ := meals_third_restaurant_per_week / 7

-- Goal: Prove the third restaurant serves 50 meals per day
theorem third_restaurant_meals_per_day : meals_third_restaurant_per_day = 50 := by
  -- proof skipped
  sorry

end third_restaurant_meals_per_day_l254_254214


namespace parsley_rows_l254_254989

-- Define the conditions laid out in the problem
def garden_rows : ℕ := 20
def plants_per_row : ℕ := 10
def rosemary_rows : ℕ := 2
def chives_planted : ℕ := 150

-- Define the target statement to prove
theorem parsley_rows :
  let total_plants := garden_rows * plants_per_row
  let remaining_rows := garden_rows - rosemary_rows
  let chives_rows := chives_planted / plants_per_row
  let parsley_rows := remaining_rows - chives_rows
  parsley_rows = 3 :=
by
  sorry

end parsley_rows_l254_254989


namespace operation_star_correct_l254_254926

def op_table (i j : ℕ) : ℕ :=
  if i = 1 then
    if j = 1 then 4 else if j = 2 then 1 else if j = 3 then 2 else if j = 4 then 3 else 0
  else if i = 2 then
    if j = 1 then 1 else if j = 2 then 3 else if j = 3 then 4 else if j = 4 then 2 else 0
  else if i = 3 then
    if j = 1 then 2 else if j = 2 then 4 else if j = 3 then 1 else if j = 4 then 3 else 0
  else if i = 4 then
    if j = 1 then 3 else if j = 2 then 2 else if j = 3 then 3 else if j = 4 then 4 else 0
  else 0

theorem operation_star_correct : op_table (op_table 3 1) (op_table 4 2) = 3 :=
  by sorry

end operation_star_correct_l254_254926


namespace remainder_identity_l254_254668

variable {n : ℕ}

theorem remainder_identity
  (a b a_1 b_1 a_2 b_2 : ℕ)
  (ha : a = a_1 + a_2 * n)
  (hb : b = b_1 + b_2 * n) :
  (((a + b) % n = (a_1 + b_1) % n) ∧ ((a - b) % n = (a_1 - b_1) % n)) ∧ ((a * b) % n = (a_1 * b_1) % n) := by
  sorry

end remainder_identity_l254_254668


namespace derivative_quadrant_l254_254636

theorem derivative_quadrant (b c : ℝ) (H_b : b = -4) : ¬ ∃ x y : ℝ, x < 0 ∧ y > 0 ∧ 2*x + b = y := by
  sorry

end derivative_quadrant_l254_254636


namespace union_of_A_and_B_l254_254706

open Set

-- Define the sets A and B based on given conditions
def A (x : ℤ) : Set ℤ := {y | y = x^2 ∨ y = 2 * x - 1 ∨ y = -4}
def B (x : ℤ) : Set ℤ := {y | y = x - 5 ∨ y = 1 - x ∨ y = 9}

-- Specific condition given in the problem
def A_intersect_B_condition (x : ℤ) : Prop :=
  A x ∩ B x = {9}

-- Prove problem statement that describes the union of A and B
theorem union_of_A_and_B (x : ℤ) (h : A_intersect_B_condition x) : A x ∪ B x = {-8, -7, -4, 4, 9} :=
sorry

end union_of_A_and_B_l254_254706


namespace triangle_angle_C_l254_254984

theorem triangle_angle_C (A B C : ℝ) (h1 : A = 86) (h2 : B = 3 * C + 22) (h3 : A + B + C = 180) : C = 18 :=
by
  sorry

end triangle_angle_C_l254_254984


namespace monotonically_increasing_range_of_a_l254_254509

noncomputable def f (a x : ℝ) : ℝ :=
  x - (1 / 3) * Real.sin (2 * x) + a * Real.sin x

theorem monotonically_increasing_range_of_a :
  (∀ x y : ℝ, x ≤ y → f a x ≤ f a y) ↔ (-1 / 3 : ℝ) ≤ a ∧ a ≤ (1 / 3 : ℝ) :=
sorry

end monotonically_increasing_range_of_a_l254_254509


namespace find_S20_l254_254032

theorem find_S20 (a : ℕ → ℕ) (S : ℕ → ℕ) 
  (h1 : ∀ n, n ≥ 2 → S n = 1 + 2 * a n)
  (h2 : a 1 = 2) : 
  S 20 = 2^19 + 1 := 
sorry

end find_S20_l254_254032


namespace jelly_beans_total_l254_254535

-- Definitions from the conditions
def vanilla : Nat := 120
def grape : Nat := 5 * vanilla + 50
def total : Nat := vanilla + grape

-- Statement to prove
theorem jelly_beans_total :
  total = 770 := 
by 
  sorry

end jelly_beans_total_l254_254535


namespace gilbert_parsley_count_l254_254192

variable (basil mint parsley : ℕ)
variable (initial_basil : ℕ := 3)
variable (extra_basil : ℕ := 1)
variable (initial_mint : ℕ := 2)
variable (herb_total : ℕ := 5)

def initial_parsley := herb_total - (initial_basil + extra_basil)

theorem gilbert_parsley_count : initial_parsley = 1 := by
  -- basil = initial_basil + extra_basil
  -- mint = 0 (since all mint plants eaten)
  -- herb_total = basil + parsley
  -- 5 = 4 + parsley
  -- parsley = 1
  sorry

end gilbert_parsley_count_l254_254192


namespace expression_for_f_l254_254950

noncomputable def f (x : ℝ) : ℝ := sorry

theorem expression_for_f (x : ℝ) :
  (∀ x, f (x - 1) = x^2) → f x = x^2 + 2 * x + 1 :=
by
  intro h
  sorry

end expression_for_f_l254_254950


namespace determine_a_from_root_l254_254199

noncomputable def quadratic_eq (x a : ℝ) : Prop := x^2 - a = 0

theorem determine_a_from_root :
  (∃ a : ℝ, quadratic_eq 2 a) → (∃ a : ℝ, a = 4) :=
by
  intro h
  obtain ⟨a, ha⟩ := h
  use a
  have h_eq : 2^2 - a = 0 := ha
  linarith

end determine_a_from_root_l254_254199


namespace maximum_k_value_l254_254194

noncomputable def f (x : ℝ) : ℝ := (1 + Real.log x) / (x - 1)
noncomputable def g (x : ℝ) (k : ℕ) : ℝ := k / x

theorem maximum_k_value (c : ℝ) (h_c : c > 1) : 
  (∃ a b : ℝ, 0 < a ∧ a < b ∧ b < c ∧ f c = f a ∧ f a = g b 3) ∧ 
  (∀ k : ℕ, k > 3 → ¬ ∃ a b : ℝ, 0 < a ∧ a < b ∧ b < c ∧ f c = f a ∧ f a = g b k) :=
sorry

end maximum_k_value_l254_254194


namespace Moscow_1975_p_q_r_equal_primes_l254_254081

theorem Moscow_1975_p_q_r_equal_primes (a b c : ℕ) (p q r : ℕ) 
  (hp : p = b^c + a) 
  (hq : q = a^b + c) 
  (hr : r = c^a + b) 
  (prime_p : Prime p) 
  (prime_q : Prime q) 
  (prime_r : Prime r) : 
  q = r :=
sorry

end Moscow_1975_p_q_r_equal_primes_l254_254081


namespace linear_eq_represents_plane_l254_254980

theorem linear_eq_represents_plane (A B C : ℝ) (h : ¬ (A = 0 ∧ B = 0 ∧ C = 0)) :
  ∃ (P : ℝ × ℝ × ℝ → Prop), (∀ (x y z : ℝ), P (x, y, z) ↔ A * x + B * y + C * z = 0) ∧ 
  (P (0, 0, 0)) :=
by
  -- To be filled in with the proof steps
  sorry

end linear_eq_represents_plane_l254_254980


namespace distance_preserving_l254_254651

variables {Point : Type} {d : Point → Point → ℕ} {f : Point → Point}

axiom distance_one (A B : Point) : d A B = 1 → d (f A) (f B) = 1

theorem distance_preserving :
  ∀ (A B : Point) (n : ℕ), n > 0 → d A B = n → d (f A) (f B) = n :=
by
  sorry

end distance_preserving_l254_254651


namespace radius_of_smaller_molds_l254_254147

noncomputable def volumeOfHemisphere (r : ℝ) : ℝ := (2 / 3) * Real.pi * r^3

theorem radius_of_smaller_molds (r : ℝ) :
  volumeOfHemisphere 2 = 64 * volumeOfHemisphere r → r = 1 / 2 :=
by
  intro h
  sorry

end radius_of_smaller_molds_l254_254147


namespace log_order_l254_254949

theorem log_order (a b c : ℝ) (h_a : a = Real.log 6 / Real.log 2) 
  (h_b : b = Real.log 15 / Real.log 5) (h_c : c = Real.log 21 / Real.log 7) : 
  a > b ∧ b > c := by sorry

end log_order_l254_254949


namespace maple_trees_remaining_l254_254571

-- Define the initial number of maple trees in the park
def initial_maple_trees : ℝ := 9.0

-- Define the number of maple trees that will be cut down
def cut_down_maple_trees : ℝ := 2.0

-- Define the expected number of maple trees left after cutting down
def remaining_maple_trees : ℝ := 7.0

-- Theorem to prove the remaining number of maple trees is correct
theorem maple_trees_remaining :
  initial_maple_trees - cut_down_maple_trees = remaining_maple_trees := by
  admit -- sorry can be used alternatively

end maple_trees_remaining_l254_254571


namespace andre_total_payment_l254_254822

def treadmill_initial_price : ℝ := 1350
def treadmill_discount : ℝ := 0.30
def plate_initial_price : ℝ := 60
def plate_discount : ℝ := 0.15
def plate_quantity : ℝ := 2

theorem andre_total_payment :
  let treadmill_discounted_price := treadmill_initial_price * (1 - treadmill_discount)
  let plates_total_initial_price := plate_quantity * plate_initial_price
  let plates_discounted_price := plates_total_initial_price * (1 - plate_discount)
  treadmill_discounted_price + plates_discounted_price = 1047 := 
by
  sorry

end andre_total_payment_l254_254822


namespace find_alpha_plus_beta_l254_254203

theorem find_alpha_plus_beta (α β : ℝ)
  (h : ∀ x : ℝ, x ≠ 45 → (x - α) / (x + β) = (x^2 - 90 * x + 1981) / (x^2 + 63 * x - 3420)) :
  α + β = 113 :=
by
  sorry

end find_alpha_plus_beta_l254_254203


namespace pedoe_inequality_l254_254074

variables {a b c a' b' c' Δ Δ' : ℝ} {A A' : ℝ}

theorem pedoe_inequality :
  a' ^ 2 * (-a ^ 2 + b ^ 2 + c ^ 2) +
  b' ^ 2 * (a ^ 2 - b ^ 2 + c ^ 2) +
  c' ^ 2 * (a ^ 2 + b ^ 2 - c ^ 2) -
  16 * Δ * Δ' =
  2 * (b * c' - b' * c) ^ 2 +
  8 * b * b' * c * c' * (Real.sin ((A - A') / 2)) ^ 2 := sorry

end pedoe_inequality_l254_254074


namespace largest_divisor_of_difference_of_squares_l254_254815

theorem largest_divisor_of_difference_of_squares (m n : ℤ) (hm : m % 2 = 1) (hn : n % 2 = 1) (h : n < m) :
  ∃ k, (∀ m n : ℤ, m % 2 = 1 → n % 2 = 1 → n < m → k ∣ (m^2 - n^2)) ∧ (∀ j : ℤ, (∀ m n : ℤ, m % 2 = 1 → n % 2 = 1 → n < m → j ∣ (m^2 - n^2)) → j ≤ k) ∧ k = 8 :=
sorry

end largest_divisor_of_difference_of_squares_l254_254815


namespace parallel_vectors_perpendicular_vectors_l254_254503

/-- Given vectors a and b where a = (1, 2) and b = (x, 1),
    let u = a + b and v = a - b.
    Prove that if u is parallel to v, then x = 1/2. 
    Also, prove that if u is perpendicular to v, then x = 2 or x = -2. --/

noncomputable def vector_a : ℝ × ℝ := (1, 2)
noncomputable def vector_b (x : ℝ) : ℝ × ℝ := (x, 1)
noncomputable def vector_u (x : ℝ) : ℝ × ℝ := (1 + x, 3)
noncomputable def vector_v (x : ℝ) : ℝ × ℝ := (1 - x, 1)

theorem parallel_vectors (x : ℝ) :
  (vector_u x).fst / (vector_v x).fst = (vector_u x).snd / (vector_v x).snd ↔ x = 1 / 2 :=
by
  sorry

theorem perpendicular_vectors (x : ℝ) :
  (vector_u x).fst * (vector_v x).fst + (vector_u x).snd * (vector_v x).snd = 0 ↔ x = 2 ∨ x = -2 :=
by
  sorry

end parallel_vectors_perpendicular_vectors_l254_254503


namespace beast_of_war_running_time_correct_l254_254851

def running_time_millennium : ℕ := 120

def running_time_alpha_epsilon (rt_millennium : ℕ) : ℕ := rt_millennium - 30

def running_time_beast_of_war (rt_alpha_epsilon : ℕ) : ℕ := rt_alpha_epsilon + 10

theorem beast_of_war_running_time_correct :
  running_time_beast_of_war (running_time_alpha_epsilon running_time_millennium) = 100 := by sorry

end beast_of_war_running_time_correct_l254_254851


namespace repeating_sum_to_fraction_l254_254746

theorem repeating_sum_to_fraction :
  (0.333333333333333 ~ 1/3) ∧ 
  (0.0404040404040401 ~ 4/99) ∧ 
  (0.005005005005001 ~ 5/999) →
  (0.333333333333333 + 0.0404040404040401 + 0.005005005005001) = (112386 / 296703) := 
by
  repeat { sorry }

end repeating_sum_to_fraction_l254_254746


namespace koala_fiber_intake_l254_254400

theorem koala_fiber_intake (r a : ℝ) (hr : r = 0.20) (ha : a = 8) : (a / r) = 40 :=
by
  sorry

end koala_fiber_intake_l254_254400


namespace coin_toss_sequences_l254_254371

theorem coin_toss_sequences :
  ∃ S : Finset (List Bool), 
    (∀ s ∈ S, s.length = 18 ∧ 
      (count_subsequence s [tt, tt] = 6) ∧ 
      (count_subsequence s [tt, ff] = 5) ∧ 
      (count_subsequence s [ff, tt] = 3) ∧ 
      (count_subsequence s [ff, ff] = 3)) ∧ 
    S.card = 840 := 
sorry

end coin_toss_sequences_l254_254371


namespace sandy_siding_cost_l254_254673

theorem sandy_siding_cost
  (wall_length wall_height roof_base roof_height : ℝ)
  (siding_length siding_height siding_cost : ℝ)
  (num_walls num_roof_faces num_siding_sections : ℝ)
  (total_cost : ℝ)
  (h_wall_length : wall_length = 10)
  (h_wall_height : wall_height = 7)
  (h_roof_base : roof_base = 10)
  (h_roof_height : roof_height = 6)
  (h_siding_length : siding_length = 10)
  (h_siding_height : siding_height = 15)
  (h_siding_cost : siding_cost = 35)
  (h_num_walls : num_walls = 2)
  (h_num_roof_faces : num_roof_faces = 1)
  (h_num_siding_sections : num_siding_sections = 2)
  (h_total_cost : total_cost = 70) :
  (siding_cost * num_siding_sections) = total_cost := 
by
  sorry

end sandy_siding_cost_l254_254673


namespace roots_transformation_l254_254080

noncomputable def poly_with_roots (r₁ r₂ r₃ : ℝ) : Polynomial ℝ :=
  Polynomial.X ^ 3 - 5 * Polynomial.X ^ 2 + 10

noncomputable def transformed_poly_with_roots (r₁ r₂ r₃ : ℝ) : Polynomial ℝ :=
  Polynomial.X ^ 3 - 15 * Polynomial.X ^ 2 + 270

theorem roots_transformation (r₁ r₂ r₃ : ℝ) (h : poly_with_roots r₁ r₂ r₃ = 0) :
  transformed_poly_with_roots (3 * r₁) (3 * r₂) (3 * r₃) = Polynomial.X ^ 3 - 15 * Polynomial.X ^ 2 + 270 :=
by
  sorry

end roots_transformation_l254_254080


namespace min_transfers_to_uniform_cards_l254_254869

theorem min_transfers_to_uniform_cards (n : ℕ) (h : n = 101) (s : Fin n) :
  ∃ k : ℕ, (∀ s1 s2 : Fin n → ℕ, 
    (∀ i, s1 i = i + 1) ∧ (∀ j, s2 j = 51) → -- Initial and final conditions
    k ≤ 42925) := 
sorry

end min_transfers_to_uniform_cards_l254_254869


namespace area_of_PQRSUV_proof_l254_254488

noncomputable def PQRSW_area (PQ QR RS SW : ℝ) : ℝ :=
  (1 / 2) * PQ * QR + (1 / 2) * (RS + SW) * 5

noncomputable def WUV_area (WU UV : ℝ) : ℝ :=
  WU * UV

theorem area_of_PQRSUV_proof 
  (PQ QR RS SW WU UV : ℝ)
  (hPQ : PQ = 8) (hQR : QR = 5) (hRS : RS = 7) (hSW : SW = 10)
  (hWU : WU = 6) (hUV : UV = 7) :
  PQRSW_area PQ QR RS SW + WUV_area WU UV = 147 :=
by
  simp only [PQRSW_area, WUV_area, hPQ, hQR, hRS, hSW, hWU, hUV]
  norm_num
  sorry

end area_of_PQRSUV_proof_l254_254488


namespace proof_problem_l254_254816

theorem proof_problem (a b c : ℕ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0)
  (h4 : a * b ∣ c * (c ^ 2 - c + 1))
  (h5 : (c ^ 2 + 1) ∣ (a + b)) :
  (a = c ∧ b = c ^ 2 - c + 1) ∨ (a = c ^ 2 - c + 1 ∧ b = c) :=
sorry

end proof_problem_l254_254816


namespace number_of_paths_passing_through_C_from_A_to_B_l254_254601

theorem number_of_paths_passing_through_C_from_A_to_B :
  let total_paths_from_A_to_B := (4 + 2).choose 2,
      paths_from_A_to_C := (2 + 1).choose 1,
      paths_from_C_to_B := (2 + 1).choose 1 in
  total_paths_from_A_to_B = 15 ∧
  paths_from_A_to_C = 3 ∧
  paths_from_C_to_B = 3 ∧
  (paths_from_A_to_C * paths_from_C_to_B) = 42 :=
by
    -- Conditions and calculations:
    have total_paths := nat.choose 6 2,
    have paths_A_C := nat.choose 3 1,
    have paths_C_B := nat.choose 3 1,
    trivial

end number_of_paths_passing_through_C_from_A_to_B_l254_254601


namespace intersection_point_on_semicircle_l254_254204

theorem intersection_point_on_semicircle {O A B C D E F G H1 H2 : Point} 
  (h1 : points_lie_on_semicircle [C, D] O A B) 
  (h2 : intersect_chord AD BC E) 
  (h3 : points_on_extensions F G AC BD AF BG) 
  (h4 : AF * BG = AE * BE) 
  (h5 : orthocenters H1 H2 AEF BEG) : 
  ∃ K : Point, 
  (K_lies_on_semicircle K O A B) ∧ 
  (intersection_point AH1 BH2 K) ∧
  collinear [F, K, G] :=
by
  sorry

end intersection_point_on_semicircle_l254_254204


namespace chips_calories_l254_254575

-- Define the conditions
def calories_from_breakfast : ℕ := 560
def calories_from_lunch : ℕ := 780
def calories_from_cake : ℕ := 110
def calories_from_coke : ℕ := 215
def daily_calorie_limit : ℕ := 2500
def remaining_calories : ℕ := 525

-- Define the total calories consumed so far
def total_consumed : ℕ := calories_from_breakfast + calories_from_lunch + calories_from_cake + calories_from_coke

-- Define the total allowable calories without exceeding the limit
def total_allowed : ℕ := daily_calorie_limit - remaining_calories

-- Define the calories in the chips
def calories_in_chips : ℕ := total_allowed - total_consumed

-- Prove that the number of calories in the chips is 310
theorem chips_calories :
  calories_in_chips = 310 :=
by
  sorry

end chips_calories_l254_254575


namespace multiplication_problem_l254_254539

noncomputable def problem_statement (x : ℂ) : Prop :=
  (x^4 + 30 * x^2 + 225) * (x^2 - 15) = x^6 - 3375

theorem multiplication_problem (x : ℂ) : 
  problem_statement x :=
sorry

end multiplication_problem_l254_254539


namespace unique_integer_solution_l254_254929

theorem unique_integer_solution (n : ℤ) :
  (⌊n^2 / 4 + n⌋ - ⌊n / 2⌋^2 = 5) ↔ (n = 10) :=
by sorry

end unique_integer_solution_l254_254929


namespace complement_union_eq_l254_254046

def M : Set ℝ := {x | (x + 3) * (x - 1) < 0}
def N : Set ℝ := {x | x ≤ -3}
def complement (A : Set ℝ) : Set ℝ := {x | x ∉ A}

theorem complement_union_eq :
  complement (M ∪ N) = {x | x ≥ 1} :=
sorry

end complement_union_eq_l254_254046


namespace fraction_of_money_left_l254_254895

theorem fraction_of_money_left (m c : ℝ) 
   (h1 : (1/5) * m = (1/3) * c) :
   (m - ((3/5) * m) = (2/5) * m) := by
  sorry

end fraction_of_money_left_l254_254895


namespace find_smallest_d_l254_254492

-- Given conditions: The known digits sum to 26
def sum_known_digits : ℕ := 5 + 2 + 4 + 7 + 8 

-- Define the smallest digit d such that 52,d47,8 is divisible by 9
def smallest_d (d : ℕ) (sum_digits_with_d : ℕ) : Prop :=
  sum_digits_with_d = sum_known_digits + d ∧ (sum_digits_with_d % 9 = 0)

theorem find_smallest_d : ∃ d : ℕ, smallest_d d 27 :=
sorry

end find_smallest_d_l254_254492


namespace equation_of_line_m_l254_254932

-- Given conditions
def point (α : Type*) := α × α

def l_eq (p : point ℝ) : Prop := p.1 + 3 * p.2 = 7 -- Equation of line l
def m_intercept : point ℝ := (1, 2) -- Intersection point of l and m
def q : point ℝ := (2, 5) -- Point Q
def q'' : point ℝ := (5, 0) -- Point Q''

-- Proving the equation of line m
theorem equation_of_line_m (m_eq : point ℝ → Prop) :
  (∀ P : point ℝ, m_eq P ↔ P.2 = 2 * P.1 - 2) ↔
  (∃ P : point ℝ, m_eq P ∧ P = (5, 0)) :=
sorry

end equation_of_line_m_l254_254932


namespace find_f6_l254_254040

-- Define the function f and the necessary properties
variable (f : ℕ+ → ℕ+)
variable (h1 : ∀ n : ℕ+, f n + f (n + 1) + f (f n) = 3 * n + 1)
variable (h2 : f 1 ≠ 1)

-- State the theorem to prove that f(6) = 5
theorem find_f6 : f 6 = 5 :=
sorry

end find_f6_l254_254040


namespace jessica_total_monthly_payment_l254_254396

-- Definitions for the conditions
def basicCableCost : ℕ := 15
def movieChannelsCost : ℕ := 12
def sportsChannelsCost : ℕ := movieChannelsCost - 3

-- The statement to be proven
theorem jessica_total_monthly_payment :
  basicCableCost + movieChannelsCost + sportsChannelsCost = 36 := 
by
  sorry

end jessica_total_monthly_payment_l254_254396


namespace product_of_intersection_coordinates_l254_254579

noncomputable def circle1 := {P : ℝ×ℝ | (P.1^2 - 4*P.1 + P.2^2 - 8*P.2 + 20) = 0}
noncomputable def circle2 := {P : ℝ×ℝ | (P.1^2 - 6*P.1 + P.2^2 - 8*P.2 + 25) = 0}

theorem product_of_intersection_coordinates :
  ∀ P ∈ circle1 ∩ circle2, P = (2, 4) → (P.1 * P.2) = 8 :=
by
  sorry

end product_of_intersection_coordinates_l254_254579


namespace log_base_4_of_8_l254_254190

noncomputable def log_base_change (b a c : ℝ) : ℝ :=
  Real.log a / Real.log b

theorem log_base_4_of_8 : log_base_change 4 8 10 = 3 / 2 :=
by
  have h1 : Real.log 8 = 3 * Real.log 2 := by
    sorry  -- Use properties of logarithms: 8 = 2^3
  have h2 : Real.log 4 = 2 * Real.log 2 := by
    sorry  -- Use properties of logarithms: 4 = 2^2
  have h3 : log_base_change 4 8 10 = (3 * Real.log 2) / (2 * Real.log 2) := by
    rw [log_base_change, h1, h2]
  have h4 : (3 * Real.log 2) / (2 * Real.log 2) = 3 / 2 := by
    sorry  -- Simplify the fraction
  rw [h3, h4]

end log_base_4_of_8_l254_254190


namespace factorize_expression1_factorize_expression2_l254_254940

section
variable (x y : ℝ)

theorem factorize_expression1 : (x^2 + y^2)^2 - 4 * x^2 * y^2 = (x + y)^2 * (x - y)^2 :=
sorry

theorem factorize_expression2 : 3 * x^3 - 12 * x^2 * y + 12 * x * y^2 = 3 * x * (x - 2 * y)^2 :=
sorry
end

end factorize_expression1_factorize_expression2_l254_254940


namespace count_lineup_excluding_youngest_l254_254378

theorem count_lineup_excluding_youngest 
  (n : ℕ) (h_n : n = 5) (youngest_position : Fin n → Prop) 
  (h_youngest_position : ∀ (pos : Fin n), youngest_position pos → pos ≠ 0 ∧ pos ≠ (n - 1)) :
  (∃ (count : ℕ), count = (4 * 3 * 3 * 2) ∧ count = 216) := 
sorry

end count_lineup_excluding_youngest_l254_254378


namespace smallest_positive_integer_remainder_l254_254125

theorem smallest_positive_integer_remainder
  (b : ℕ) (h1 : b % 4 = 3) (h2 : b % 6 = 5) :
  b = 11 := by
  sorry

end smallest_positive_integer_remainder_l254_254125


namespace fish_served_l254_254148

theorem fish_served (H E P : ℕ) 
  (h1 : H = E) (h2 : E = P) 
  (fat_herring fat_eel fat_pike total_fat : ℕ) 
  (herring_fat : fat_herring = 40) 
  (eel_fat : fat_eel = 20)
  (pike_fat : fat_pike = 30)
  (total_fat_served : total_fat = 3600) 
  (fat_eq : 40 * H + 20 * E + 30 * P = 3600) : 
  H = 40 ∧ E = 40 ∧ P = 40 := by
  sorry

end fish_served_l254_254148


namespace average_apples_per_hour_l254_254254

theorem average_apples_per_hour :
  (5.0 / 3.0) = 1.67 := 
sorry

end average_apples_per_hour_l254_254254


namespace find_a_l254_254788

theorem find_a (a : ℝ) : (∃ x y : ℝ, 3 * x + a * y - 5 = 0 ∧ x = 1 ∧ y = 2) → a = 1 :=
by
  intro h
  match h with
  | ⟨x, y, hx, hx1, hy2⟩ => 
    have h1 : x = 1 := hx1
    have h2 : y = 2 := hy2
    rw [h1, h2] at hx
    sorry

end find_a_l254_254788


namespace base3_to_base5_conversion_l254_254927

-- Define the conversion from base 3 to decimal
def base3_to_decimal (n : ℕ) : ℕ :=
  n % 10 * 1 + (n / 10 % 10) * 3 + (n / 100 % 10) * 9 + (n / 1000 % 10) * 27 + (n / 10000 % 10) * 81

-- Define the conversion from decimal to base 5
def decimal_to_base5 (n : ℕ) : ℕ :=
  n % 5 + (n / 5 % 5) * 10 + (n / 25 % 5) * 100

-- The initial number in base 3
def initial_number_base3 : ℕ := 10121

-- The final number in base 5
def final_number_base5 : ℕ := 342

-- The theorem that states the conversion result
theorem base3_to_base5_conversion :
  decimal_to_base5 (base3_to_decimal initial_number_base3) = final_number_base5 :=
by
  sorry

end base3_to_base5_conversion_l254_254927


namespace solve_for_x_l254_254553

theorem solve_for_x (x : ℝ) (h : 6 * x ^ (1 / 3) - 3 * (x / x ^ (2 / 3)) = -1 + 2 * x ^ (1 / 3) + 4) :
  x = 27 :=
by 
  sorry

end solve_for_x_l254_254553


namespace complex_in_third_quadrant_l254_254499

open Complex

noncomputable def quadrant (z : ℂ) : ℕ :=
  if z.re > 0 ∧ z.im > 0 then 1
  else if z.re < 0 ∧ z.im > 0 then 2
  else if z.re < 0 ∧ z.im < 0 then 3
  else 4

theorem complex_in_third_quadrant (z : ℂ) (h : (2 + I) * z = -I) : quadrant z = 3 := by
  sorry

end complex_in_third_quadrant_l254_254499


namespace polynomial_satisfies_conditions_l254_254409

noncomputable def f (x y z : ℝ) : ℝ := (x^2 - y^3) * (y^3 - z^6) * (z^6 - x^2)

theorem polynomial_satisfies_conditions :
  (∀ x y z : ℝ, f x (z^2) y + f x (y^2) z = 0) ∧ 
  (∀ x y z : ℝ, f (z^3) y x + f (x^3) y z = 0) :=
by
  sorry

end polynomial_satisfies_conditions_l254_254409


namespace radius_of_circle_l254_254143

theorem radius_of_circle (A C : ℝ) (h1 : A = π * (r : ℝ)^2) (h2 : C = 2 * π * r) (h3 : A / C = 10) :
  r = 20 :=
by
  sorry

end radius_of_circle_l254_254143


namespace probability_contemporaries_correct_l254_254283

def alice_lifespan : ℝ := 150
def bob_lifespan : ℝ := 150
def total_years : ℝ := 800

noncomputable def probability_contemporaries : ℝ :=
  let unshaded_tri_area := (650 * 150) / 2
  let unshaded_area := 2 * unshaded_tri_area
  let total_area := total_years * total_years
  let shaded_area := total_area - unshaded_area
  shaded_area / total_area

theorem probability_contemporaries_correct : 
  probability_contemporaries = 27125 / 32000 :=
by
  sorry

end probability_contemporaries_correct_l254_254283


namespace cards_distribution_l254_254798

theorem cards_distribution (total_cards : ℕ) (total_people : ℕ) (cards_per_person : ℕ) (extra_cards : ℕ) (people_with_extra_cards : ℕ) (people_with_fewer_cards : ℕ) :
  total_cards = 100 →
  total_people = 15 →
  total_cards / total_people = cards_per_person →
  total_cards % total_people = extra_cards →
  people_with_extra_cards = extra_cards →
  people_with_fewer_cards = total_people - people_with_extra_cards →
  people_with_fewer_cards = 5 :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end cards_distribution_l254_254798


namespace arctan_sum_eq_pi_div_two_l254_254923

theorem arctan_sum_eq_pi_div_two : Real.arctan (3 / 4) + Real.arctan (4 / 3) = Real.pi / 2 :=
by
  sorry

end arctan_sum_eq_pi_div_two_l254_254923


namespace max_height_l254_254593

noncomputable def ball_height (t : ℝ) : ℝ :=
  -4.9 * t^2 + 50 * t + 15

theorem max_height : ∃ t : ℝ, t < 50 / 4.9 ∧ ball_height t = 142.65 :=
sorry

end max_height_l254_254593


namespace g_extreme_points_product_inequality_l254_254043

noncomputable def f (a x : ℝ) : ℝ := (-x^2 + a * x - a) / Real.exp x

noncomputable def f' (a x : ℝ) : ℝ := (x^2 - (a + 2) * x + 2 * a) / Real.exp x

noncomputable def g (a x : ℝ) : ℝ := (f a x + f' a x) / (x - 1)

theorem g_extreme_points_product_inequality {a x1 x2 : ℝ} 
  (h_cond1 : a > 2)
  (h_cond2 : x1 + x2 = (a + 2) / 2)
  (h_cond3 : x1 * x2 = 1)
  (h_cond4 : x1 ≠ 1 ∧ x2 ≠ 1)
  (h_x1 : x1 ∈ (Set.Ioo 0 1 ∪ Set.Ioi 1))
  (h_x2 : x2 ∈ (Set.Ioo 0 1 ∪ Set.Ioi 1)) :
  g a x1 * g a x2 < 4 / Real.exp 2 :=
sorry

end g_extreme_points_product_inequality_l254_254043


namespace tan_alpha_eq_neg_one_l254_254955

theorem tan_alpha_eq_neg_one (α : ℝ) (h : Real.sin (π / 6 - α) = Real.cos (π / 6 + α)) : Real.tan α = -1 :=
  sorry

end tan_alpha_eq_neg_one_l254_254955


namespace total_cookies_dropped_throughout_entire_baking_process_l254_254542

def initially_baked_by_alice := 74 + 45 + 15
def initially_baked_by_bob := 7 + 32 + 18

def initially_dropped_by_alice := 5 + 8
def initially_dropped_by_bob := 10 + 6

def additional_baked_by_alice := 5 + 4 + 12
def additional_baked_by_bob := 22 + 36 + 14

def edible_cookies := 145

theorem total_cookies_dropped_throughout_entire_baking_process :
  initially_baked_by_alice + initially_baked_by_bob +
  additional_baked_by_alice + additional_baked_by_bob -
  edible_cookies = 139 := by
  sorry

end total_cookies_dropped_throughout_entire_baking_process_l254_254542


namespace sum_coefficients_l254_254061

theorem sum_coefficients (a1 a2 a3 a4 a5 : ℤ) (h : ∀ x : ℕ, a1 * (x - 1) ^ 4 + a2 * (x - 1) ^ 3 + a3 * (x - 1) ^ 2 + a4 * (x - 1) + a5 = x ^ 4) :
  a2 + a3 + a4 = 14 :=
  sorry

end sum_coefficients_l254_254061


namespace dante_initially_has_8_jelly_beans_l254_254112

-- Conditions
def aaron_jelly_beans : ℕ := 5
def bianca_jelly_beans : ℕ := 7
def callie_jelly_beans : ℕ := 8
def dante_jelly_beans_initially (D : ℕ) : Prop := 
  ∀ (D : ℕ), (6 ≤ D - 1 ∧ D - 1 ≤ callie_jelly_beans - 1)

-- Theorem
theorem dante_initially_has_8_jelly_beans :
  ∃ (D : ℕ), (aaron_jelly_beans + 1 = 6) →
             (callie_jelly_beans = 8) →
             dante_jelly_beans_initially D →
             D = 8 := 
by
  sorry

end dante_initially_has_8_jelly_beans_l254_254112


namespace basketball_game_score_l254_254707

theorem basketball_game_score 
  (a r b d : ℕ)
  (H1 : a = b)
  (H2 : a + a * r + a * r^2 = b + (b + d) + (b + 2 * d))
  (H3 : a * (1 + r + r^2 + r^3) = 4 * b + 6 * d + 3)
  (H4 : r = 3)
  (H5 : a = 3)
  (H6 : d = 10)
  (H7 : a * (1 + r) = 12)
  (H8 : b * (1 + 3 + (b + d)) = 16) :
  a + a * r + b + (b + d) = 28 :=
by simp [H4, H5, H6, H7, H8]; linarith

end basketball_game_score_l254_254707


namespace expression_parity_l254_254508

theorem expression_parity (a b c : ℕ) (h_a_pos : a > 0) (h_b_pos : b > 0) (h_a_odd : a % 2 = 1) (h_b_odd : b % 2 = 1) : (3^a + (b + 1)^2 * c) % 2 = 1 :=
by sorry

end expression_parity_l254_254508


namespace find_u_l254_254558

theorem find_u 
    (a b c p q u : ℝ) 
    (H₁: (∀ x, x^3 + 2*x^2 + 5*x - 8 = 0 → x = a ∨ x = b ∨ x = c))
    (H₂: (∀ x, x^3 + p*x^2 + q*x + u = 0 → x = a+b ∨ x = b+c ∨ x = c+a)) :
    u = 18 :=
by 
    sorry

end find_u_l254_254558


namespace triangle_angle_sum_l254_254233

theorem triangle_angle_sum (angle_Q R P : ℝ)
  (h1 : R = 3 * angle_Q)
  (h2 : angle_Q = 30)
  (h3 : P + angle_Q + R = 180) :
    P = 60 :=
by
  sorry

end triangle_angle_sum_l254_254233


namespace negation_P_eq_Q_l254_254436

-- Define the proposition P: For any x ∈ ℝ, x^2 - 2x - 3 ≤ 0
def P : Prop := ∀ x : ℝ, x^2 - 2*x - 3 ≤ 0

-- Define its negation which is the proposition Q
def Q : Prop := ∃ x : ℝ, x^2 - 2*x - 3 > 0

-- Prove that the negation of P is equivalent to Q
theorem negation_P_eq_Q : ¬P = Q :=
  by
  sorry

end negation_P_eq_Q_l254_254436


namespace evaluate_expression_at_y_minus3_l254_254022

theorem evaluate_expression_at_y_minus3 :
  let y := -3
  (5 + y * (2 + y) - 4^2) / (y - 4 + y^2 - y) = -8 / 5 :=
by
  let y := -3
  sorry

end evaluate_expression_at_y_minus3_l254_254022


namespace expand_binomials_l254_254937

variable (x y : ℝ)

theorem expand_binomials: (2 * x - 5) * (3 * y + 15) = 6 * x * y + 30 * x - 15 * y - 75 :=
by sorry

end expand_binomials_l254_254937


namespace find_d_l254_254110

theorem find_d (a b c d : ℝ) (h : a^2 + b^2 + c^2 + 4 = d + Real.sqrt (a + b + c - d + 3)) : 
  d = 13 / 4 :=
sorry

end find_d_l254_254110


namespace sum_base6_l254_254314

theorem sum_base6 : 
  ∀ (a b : ℕ) (h1 : a = 4532) (h2 : b = 3412),
  (a + b = 10414) :=
by
  intros a b h1 h2
  rw [h1, h2]
  sorry

end sum_base6_l254_254314


namespace alice_favorite_number_l254_254888

def is_multiple (x y : ℕ) : Prop := ∃ k : ℕ, k * y = x
def digit_sum (n : ℕ) : ℕ := n.digits 10 |>.sum

theorem alice_favorite_number 
  (n : ℕ) 
  (h1 : 90 ≤ n ∧ n ≤ 150) 
  (h2 : is_multiple n 13) 
  (h3 : ¬ is_multiple n 4) 
  (h4 : is_multiple (digit_sum n) 4) : 
  n = 143 := 
by 
  sorry

end alice_favorite_number_l254_254888


namespace cone_lateral_area_l254_254682

-- Definitions from the conditions
def radius_base : ℝ := 1 -- in cm
def slant_height : ℝ := 2 -- in cm

-- Statement to be proved: The lateral area of the cone is 2π cm²
theorem cone_lateral_area : 
  1/2 * (2 * π * radius_base) * slant_height = 2 * π :=
by
  sorry

end cone_lateral_area_l254_254682


namespace carl_max_value_carry_l254_254159

variables (rock_weight_3_pound : ℕ := 3) (rock_value_3_pound : ℕ := 9)
          (rock_weight_6_pound : ℕ := 6) (rock_value_6_pound : ℕ := 20)
          (rock_weight_2_pound : ℕ := 2) (rock_value_2_pound : ℕ := 5)
          (weight_limit : ℕ := 20)
          (max_six_pound_rocks : ℕ := 2)

noncomputable def max_value_carry : ℕ :=
  max (2 * rock_value_6_pound + 2 * rock_value_3_pound) 
      (4 * rock_value_3_pound + 4 * rock_value_2_pound)

theorem carl_max_value_carry : max_value_carry = 58 :=
by sorry

end carl_max_value_carry_l254_254159


namespace pages_read_over_weekend_l254_254892

-- Define the given conditions
def total_pages : ℕ := 408
def days_left : ℕ := 5
def pages_per_day : ℕ := 59

-- Define the calculated pages to be read over the remaining days
def pages_remaining := days_left * pages_per_day

-- Define the pages read over the weekend
def pages_over_weekend := total_pages - pages_remaining

-- Prove that Bekah read 113 pages over the weekend
theorem pages_read_over_weekend : pages_over_weekend = 113 :=
by {
  -- proof should be here, but we place sorry since proof is not required
  sorry
}

end pages_read_over_weekend_l254_254892


namespace interval_of_x_l254_254179

theorem interval_of_x (x : ℝ) :
  (2 < 4 * x ∧ 4 * x < 3) ∧ (2 < 5 * x ∧ 5 * x < 3) ↔ (1 / 2 < x ∧ x < 3 / 5) := by
  sorry

end interval_of_x_l254_254179


namespace betty_eggs_used_l254_254862

-- Conditions as definitions
def ratio_sugar_cream_cheese (sugar cream_cheese : ℚ) : Prop :=
  sugar / cream_cheese = 1 / 4

def ratio_vanilla_cream_cheese (vanilla cream_cheese : ℚ) : Prop :=
  vanilla / cream_cheese = 1 / 2

def ratio_eggs_vanilla (eggs vanilla : ℚ) : Prop :=
  eggs / vanilla = 2

-- Given conditions
def sugar_used : ℚ := 2 -- cups of sugar

-- The statement to prove
theorem betty_eggs_used (cream_cheese vanilla eggs : ℚ) 
  (h1 : ratio_sugar_cream_cheese sugar_used cream_cheese)
  (h2 : ratio_vanilla_cream_cheese vanilla cream_cheese)
  (h3 : ratio_eggs_vanilla eggs vanilla) :
  eggs = 8 :=
sorry

end betty_eggs_used_l254_254862


namespace max_volume_rectangular_frame_l254_254152

theorem max_volume_rectangular_frame (L W H : ℝ) (h1 : 2 * W = L) (h2 : 4 * (L + W) + 4 * H = 18) :
  volume = (2 * 1 * 1.5 : ℝ) := 
sorry

end max_volume_rectangular_frame_l254_254152


namespace solve_inequality_l254_254481

open Real

theorem solve_inequality (x : ℝ) : (x ≠ 3) ∧ (x * (x + 1) / (x - 3) ^ 2 ≥ 9) ↔ (2.13696 ≤ x ∧ x < 3) ∨ (3 < x ∧ x ≤ 4.73804) :=
by
  sorry

end solve_inequality_l254_254481


namespace evaluate_expression_l254_254609

theorem evaluate_expression (b : ℝ) (hb : b = -3) : 
  (3 * b⁻¹ + b⁻¹ / 3) / b = 10 / 27 :=
by 
  -- Lean code typically begins the proof block here
  sorry  -- The proof itself is omitted

end evaluate_expression_l254_254609


namespace distribution_of_earnings_l254_254833

theorem distribution_of_earnings :
  let payments := [10, 15, 20, 25, 30, 50]
  let total_earnings := payments.sum 
  let equal_share := total_earnings / 6
  50 - equal_share = 25 := by
  sorry

end distribution_of_earnings_l254_254833


namespace inverse_proportion_comparison_l254_254957

theorem inverse_proportion_comparison (y1 y2 : ℝ) 
  (h1 : y1 = - 6 / 2)
  (h2 : y2 = - 6 / -1) : 
  y1 < y2 :=
by
  sorry

end inverse_proportion_comparison_l254_254957


namespace converse_proposition_converse_proposition_true_l254_254421

theorem converse_proposition (x : ℝ) (h : x > 0) : x^2 - 1 > 0 :=
by sorry

theorem converse_proposition_true (x : ℝ) (h : x^2 - 1 > 0) : x > 0 :=
by sorry

end converse_proposition_converse_proposition_true_l254_254421


namespace focus_with_greatest_y_coordinate_l254_254893

-- Define the conditions as hypotheses
def ellipse_major_axis : (ℝ × ℝ) := (0, 3)
def ellipse_minor_axis : (ℝ × ℝ) := (2, 0)
def ellipse_semi_major_axis : ℝ := 3
def ellipse_semi_minor_axis : ℝ := 2

-- Define the theorem to compute the coordinates of the focus with the greater y-coordinate
theorem focus_with_greatest_y_coordinate :
  let a := ellipse_semi_major_axis
  let b := ellipse_semi_minor_axis
  let c := Real.sqrt (a^2 - b^2)
  (0, c) = (0, (Real.sqrt 5) / 2) :=
by
  -- skipped proof
  sorry

end focus_with_greatest_y_coordinate_l254_254893


namespace sum_of_coefficients_l254_254795

theorem sum_of_coefficients (a a1 a2 a3 a4 a5 a6 a7 : ℤ) (a_eq : (1 - 2 * (0:ℤ)) ^ 7 = a)
  (hx_eq : ∀ (x : ℤ), (1 - 2 * x) ^ 7 = a + a1 * x + a2 * x^2 + a3 * x^3 + a4 * x^4 + a5 * x^5 + a6 * x^6 + a7 * x^7) :
  a1 + a2 + a3 + a4 + a5 + a6 + a7 = -2 :=
by
  sorry

end sum_of_coefficients_l254_254795


namespace crackers_per_person_l254_254410

theorem crackers_per_person:
  ∀ (total_crackers friends : ℕ), total_crackers = 36 → friends = 18 → total_crackers / friends = 2 :=
by
  intros total_crackers friends h1 h2
  sorry

end crackers_per_person_l254_254410


namespace average_speed_round_trip_l254_254299

theorem average_speed_round_trip :
  ∀ (D : ℝ), 
  D > 0 → 
  let upstream_speed := 6 
  let downstream_speed := 5 
  (2 * D) / ((D / upstream_speed) + (D / downstream_speed)) = 60 / 11 :=
by
  intro D hD
  let upstream_speed := 6
  let downstream_speed := 5
  have h : (2 * D) / ((D / upstream_speed) + (D / downstream_speed)) = 60 / 11 := sorry
  exact h

end average_speed_round_trip_l254_254299


namespace sum_of_four_smallest_divisors_l254_254596

-- Define a natural number n and divisors d1, d2, d3, d4
def is_divisor (d n : ℕ) : Prop := ∃ k : ℕ, n = k * d

-- Primary problem condition (sum of four divisors equals 2n)
def sum_of_divisors_eq (n d1 d2 d3 d4 : ℕ) : Prop := d1 + d2 + d3 + d4 = 2 * n

-- Assume the four divisors of n are distinct
def distinct (d1 d2 d3 d4 : ℕ) : Prop := d1 ≠ d2 ∧ d1 ≠ d3 ∧ d1 ≠ d4 ∧ d2 ≠ d3 ∧ d2 ≠ d4 ∧ d3 ≠ d4

-- State the Lean proof problem
theorem sum_of_four_smallest_divisors (n d1 d2 d3 d4 : ℕ) (h1 : d1 < d2) (h2 : d2 < d3) (h3 : d3 < d4) 
    (h_div1 : is_divisor d1 n) (h_div2 : is_divisor d2 n) (h_div3 : is_divisor d3 n) (h_div4 : is_divisor d4 n)
    (h_sum : sum_of_divisors_eq n d1 d2 d3 d4) (h_distinct : distinct d1 d2 d3 d4) : 
    (d1 + d2 + d3 + d4 = 10 ∨ d1 + d2 + d3 + d4 = 11 ∨ d1 + d2 + d3 + d4 = 12) := 
sorry

end sum_of_four_smallest_divisors_l254_254596


namespace fuelA_amount_l254_254600

def tankCapacity : ℝ := 200
def ethanolInFuelA : ℝ := 0.12
def ethanolInFuelB : ℝ := 0.16
def totalEthanol : ℝ := 30
def limitedFuelA : ℝ := 100
def limitedFuelB : ℝ := 150

theorem fuelA_amount : ∃ (x : ℝ), 
  (x ≤ limitedFuelA ∧ x ≥ 0) ∧ 
  ((tankCapacity - x) ≤ limitedFuelB ∧ (tankCapacity - x) ≥ 0) ∧ 
  (ethanolInFuelA * x + ethanolInFuelB * (tankCapacity - x)) = totalEthanol ∧ 
  x = 50 := 
by
  sorry

end fuelA_amount_l254_254600


namespace masha_final_number_stabilizes_masha_smallest_initial_number_ends_with_09_l254_254252

/-- 
Part (a): Define the problem statement where, given the iterative process on a number,
it stabilizes at 17.
-/
theorem masha_final_number_stabilizes (x y : ℕ) (n : ℕ) (h_stable : ∀ x y, 10 * x + y = 3 * x + 2 * y) :
  n = 17 :=
by
  sorry

/--
Part (b): Define the problem statement to find the smallest 2015-digit number ending with the
digits 09 that eventually stabilizes to 17.
-/
theorem masha_smallest_initial_number_ends_with_09 :
  ∃ (n : ℕ), n ≥ 10^2014 ∧ n % 100 = 9 ∧ (∃ k : ℕ, 10^2014 + k = n ∧ (10 * ((n - k) / 10) + (n % 10)) = 17) :=
by
  sorry

end masha_final_number_stabilizes_masha_smallest_initial_number_ends_with_09_l254_254252


namespace batch_of_pizza_dough_makes_three_pizzas_l254_254516

theorem batch_of_pizza_dough_makes_three_pizzas
  (pizza_dough_time : ℕ)
  (baking_time : ℕ)
  (total_time_minutes : ℕ)
  (oven_capacity : ℕ)
  (total_pizzas : ℕ) 
  (number_of_batches : ℕ)
  (one_batch_pizzas : ℕ) :
  pizza_dough_time = 30 →
  baking_time = 30 →
  total_time_minutes = 300 →
  oven_capacity = 2 →
  total_pizzas = 12 →
  total_time_minutes = total_pizzas / oven_capacity * baking_time + number_of_batches * pizza_dough_time →
  number_of_batches = total_time_minutes / 30 →
  one_batch_pizzas = total_pizzas / number_of_batches →
  one_batch_pizzas = 3 :=
by
  intros
  sorry

end batch_of_pizza_dough_makes_three_pizzas_l254_254516


namespace gilda_stickers_left_l254_254619

variable (S : ℝ) (hS : S > 0)

def remaining_after_olga : ℝ := 0.70 * S
def remaining_after_sam : ℝ := 0.80 * remaining_after_olga S
def remaining_after_max : ℝ := 0.70 * remaining_after_sam S
def remaining_after_charity : ℝ := 0.90 * remaining_after_max S

theorem gilda_stickers_left :
  remaining_after_charity S / S * 100 = 35.28 := by
  sorry

end gilda_stickers_left_l254_254619


namespace exists_zero_in_interval_l254_254363

open Set Real

theorem exists_zero_in_interval (f : ℝ → ℝ) (a b : ℝ) (h_cont : ContinuousOn f (Icc a b)) 
  (h_pos : f a * f b > 0) : ∃ c ∈ Ioo a b, f c = 0 := sorry

end exists_zero_in_interval_l254_254363


namespace probability_factor_of_120_l254_254884

open Finset

theorem probability_factor_of_120 : 
  let s := range 37
  let five_factorial := 5!
  let factors_of_120 := {n ∈ s | five_factorial % n = 0}
  (finsupp.card factors_of_120 : ℚ) / (finsupp.card s : ℚ) = 4 / 9 :=
by sorry

end probability_factor_of_120_l254_254884


namespace perimeter_of_Triangle_PXY_l254_254448

open Triangle

noncomputable def TrianglePXYPerimeter : ℝ :=
  let P := (0, 0)
  let Q := (13, 0)
  let R := (21, 20)
  let I := incenter (triangle.mk P Q R)
  let X := intersection_point (line_parallel_to I P Q) (line_through P R)
  let Y := intersection_point (line_parallel_to I P Q) (line_through Q R)
  length P X + length X Y + length Y P

theorem perimeter_of_Triangle_PXY :
  let P := (0, 0)
  let Q := (13, 0)
  let R := (21, 20)
  let I := incenter (triangle.mk P Q R)
  let X := intersection_point (line_parallel_to I P Q) (line_through P R)
  let Y := intersection_point (line_parallel_to I P Q) (line_through Q R)
  TrianglePXYPerimeter (triangle.mk P X Y) = 34 := by
  sorry

end perimeter_of_Triangle_PXY_l254_254448


namespace book_page_count_l254_254727

theorem book_page_count:
  (∃ (total_pages : ℕ), 
    (∃ (days_read : ℕ) (pages_per_day : ℕ), 
      days_read = 12 ∧ 
      pages_per_day = 8 ∧ 
      (days_read * pages_per_day) = 2 * (total_pages / 3)) 
  ↔ total_pages = 144) :=
by 
  sorry

end book_page_count_l254_254727


namespace fraction_addition_l254_254313

theorem fraction_addition (x : ℝ) (h : x + 1 ≠ 0) : (x / (x + 1) + 1 / (x + 1) = 1) :=
sorry

end fraction_addition_l254_254313


namespace count_restricted_arrangements_l254_254374

theorem count_restricted_arrangements (n : ℕ) (hn : n = 5) : 
  (n.factorial - 2 * (n - 1).factorial) = 72 := 
by 
  sorry

end count_restricted_arrangements_l254_254374


namespace max_elephants_l254_254543

def union_members : ℕ := 28
def non_union_members : ℕ := 37

/-- Given 28 union members and 37 non-union members, where elephants are distributed equally among
each group and each person initially receives at least one elephant, and considering 
the unique distribution constraint, the maximum number of elephants is 2072. -/
theorem max_elephants (n : ℕ) 
  (h1 : n % union_members = 0)
  (h2 : n % non_union_members = 0)
  (h3 : n ≥ union_members * non_union_members) :
  n = 2072 :=
by sorry

end max_elephants_l254_254543


namespace profit_percentage_l254_254587

theorem profit_percentage (SP : ℝ) (h1 : SP > 0) (h2 : CP = 0.99 * SP) : (SP - CP) / CP * 100 = 1.01 :=
by
  sorry

end profit_percentage_l254_254587


namespace ratio_playground_landscape_l254_254102

-- Defining the conditions
def breadth := 420
def length := breadth / 6
def playground_area := 4200
def landscape_area := length * breadth

-- Stating the theorem to prove the ratio is 1:7
theorem ratio_playground_landscape :
  (playground_area.toFloat / landscape_area.toFloat) = (1.0 / 7.0) :=
by
  sorry

end ratio_playground_landscape_l254_254102


namespace interval_of_x_l254_254171

theorem interval_of_x (x : ℝ) : 
  (2 < 4 * x ∧ 4 * x < 3) ∧ (2 < 5 * x ∧ 5 * x < 3) ↔ (1 / 2 < x ∧ x < 3 / 5) :=
by
  sorry

end interval_of_x_l254_254171


namespace product_calc_l254_254007

theorem product_calc : (16 * 0.5 * 4 * 0.125 = 4) :=
by
  sorry

end product_calc_l254_254007


namespace B_work_days_l254_254138

theorem B_work_days (x : ℝ) :
  (1 / 3 + 1 / x = 1 / 2) → x = 6 := by
  sorry

end B_work_days_l254_254138


namespace Caitlin_age_l254_254602

theorem Caitlin_age (Aunt_Anna_age : ℕ) (h1 : Aunt_Anna_age = 54) (Brianna_age : ℕ) (h2 : Brianna_age = (2 * Aunt_Anna_age) / 3) (Caitlin_age : ℕ) (h3 : Caitlin_age = Brianna_age - 7) : 
  Caitlin_age = 29 := 
  sorry

end Caitlin_age_l254_254602


namespace num_factors_48_l254_254054

theorem num_factors_48 : 
  let n := 48 in
  ∃ num_factors, num_factors = 10 ∧ 
  (∀ p k, prime p → (n = p ^ k → 1 + k)) := 
sorry

end num_factors_48_l254_254054


namespace find_a_l254_254631

variable (a b : ℝ × ℝ) 

axiom b_eq : b = (2, -1)
axiom length_eq_one : ‖a + b‖ = 1
axiom parallel_x_axis : (a + b).snd = 0

theorem find_a : a = (-1, 1) ∨ a = (-3, 1) := by
  sorry

end find_a_l254_254631


namespace original_rectangle_area_l254_254715

theorem original_rectangle_area : 
  ∃ (a b : ℤ), (a + b = 20) ∧ (a * b = 96) := by
  sorry

end original_rectangle_area_l254_254715


namespace binomial_problem_l254_254316

-- Definition of the binomial coefficient
def binomial (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

-- The problem statement: prove that binomial(13, 11) * 2 = 156
theorem binomial_problem : binomial 13 11 * 2 = 156 := by
  sorry

end binomial_problem_l254_254316


namespace largest_possible_s_l254_254992

theorem largest_possible_s (r s: ℕ) (h1: r ≥ s) (h2: s ≥ 3)
  (h3: (59 : ℚ) / 58 * (180 * (s - 2) / s) = (180 * (r - 2) / r)) : s = 117 :=
sorry

end largest_possible_s_l254_254992


namespace distinguishable_rearrangements_of_contest_with_vowels_first_l254_254793

open Finset

theorem distinguishable_rearrangements_of_contest_with_vowels_first :
  let vowels := ['O', 'E'],
      consonants := ['C', 'N', 'T', 'S', 'T'],
      vowel_arrangements := 2!, -- 2! arrangements of vowels
      consonant_arrangements := 5! / 2! -- 5! arrangements of consonants accounted for repetition of 'T'
  in
  vowel_arrangements * consonant_arrangements = 120 := by
  let vowels := ['O', 'E'],
      consonants := ['C', 'N', 'T', 'S', 'T'],
      vowel_arrangements := Nat.factorial 2, -- 2! arrangements of vowels
      consonant_arrangements := Nat.factorial 5 / Nat.factorial 2 -- 5! arrangements of consonants accounted for repetition of 'T'
  show vowel_arrangements * consonant_arrangements = 120
  from calc
  2 * 60 = 120 : by sorry

#print distinguishable_rearrangements_of_contest_with_vowels_first

-- sorry statement to skip the proof.

end distinguishable_rearrangements_of_contest_with_vowels_first_l254_254793


namespace trig_identity_eq_one_l254_254865

theorem trig_identity_eq_one :
  (Real.sin (160 * Real.pi / 180) + Real.sin (40 * Real.pi / 180)) *
  (Real.sin (140 * Real.pi / 180) + Real.sin (20 * Real.pi / 180)) +
  (Real.sin (50 * Real.pi / 180) - Real.sin (70 * Real.pi / 180)) *
  (Real.sin (130 * Real.pi / 180) - Real.sin (110 * Real.pi / 180)) =
  1 :=
sorry

end trig_identity_eq_one_l254_254865


namespace division_multiplication_example_l254_254694

theorem division_multiplication_example : 120 / 4 / 2 * 3 = 45 := by
  sorry

end division_multiplication_example_l254_254694


namespace choose_3_of_9_colors_l254_254372

-- Define the combination function
noncomputable def combination (n k : ℕ) := n.choose k

-- Noncomputable because factorial and combination require division.
noncomputable def combination_9_3 := combination 9 3

-- State the theorem we are proving
theorem choose_3_of_9_colors : combination_9_3 = 84 :=
by
  -- Proof skipped
  sorry

end choose_3_of_9_colors_l254_254372


namespace adult_tickets_count_l254_254645

theorem adult_tickets_count (A C : ℕ) (h1 : A + C = 7) (h2 : 21 * A + 14 * C = 119) : A = 3 :=
sorry

end adult_tickets_count_l254_254645


namespace base_k_to_decimal_l254_254196

theorem base_k_to_decimal (k : ℕ) (h : 1 * k^2 + 3 * k + 2 = 30) : k = 4 :=
  sorry

end base_k_to_decimal_l254_254196


namespace volcano_ash_height_l254_254886

theorem volcano_ash_height (r d : ℝ) (h : r = 2700) (h₁ : 2 * r = 18 * d) : d = 300 :=
by
  sorry

end volcano_ash_height_l254_254886


namespace emily_dice_probability_l254_254934

theorem emily_dice_probability :
  let prob_first_die := 1 / 2
  let prob_second_die := 3 / 8
  prob_first_die * prob_second_die = 3 / 16 :=
by
  sorry

end emily_dice_probability_l254_254934


namespace interval_of_x_l254_254172

theorem interval_of_x (x : ℝ) : 
  (2 < 4 * x ∧ 4 * x < 3) ∧ (2 < 5 * x ∧ 5 * x < 3) ↔ (1 / 2 < x ∧ x < 3 / 5) :=
by
  sorry

end interval_of_x_l254_254172


namespace no_integers_solution_l254_254547

theorem no_integers_solution (k : ℕ) (x y z : ℤ) (hx1 : 0 < x) (hx2 : x < k) (hy1 : 0 < y) (hy2 : y < k) (hz : z > 0) :
  x^k + y^k ≠ z^k :=
sorry

end no_integers_solution_l254_254547


namespace isosceles_triangle_perimeter_l254_254072

def is_isosceles (a b c : ℝ) : Prop :=
  (a = b ∨ b = c ∨ c = a)

theorem isosceles_triangle_perimeter 
  (a b c : ℝ) 
  (h_iso : is_isosceles a b c) 
  (h1 : a = 2 ∨ a = 4) 
  (h2 : b = 2 ∨ b = 4) 
  (h3 : c = 2 ∨ c = 4) :
  a + b + c = 10 :=
  sorry

end isosceles_triangle_perimeter_l254_254072


namespace range_a_A_intersect_B_empty_range_a_A_union_B_eq_B_l254_254791

-- Definition of the sets A and B
def A (a : ℝ) (x : ℝ) : Prop := a - 1 < x ∧ x < 2 * a + 1
def B (x : ℝ) : Prop := 0 < x ∧ x < 1

-- Proving range of a for A ∩ B = ∅
theorem range_a_A_intersect_B_empty (a : ℝ) :
  (¬ ∃ x : ℝ, A a x ∧ B x) ↔ (a ≤ -2 ∨ a ≥ 2 ∨ (-2 < a ∧ a ≤ -1/2)) := sorry

-- Proving range of a for A ∪ B = B
theorem range_a_A_union_B_eq_B (a : ℝ) :
  (∀ x : ℝ, A a x ∨ B x → B x) ↔ (a ≤ -2) := sorry

end range_a_A_intersect_B_empty_range_a_A_union_B_eq_B_l254_254791


namespace total_pieces_of_mail_l254_254077

-- Definitions based on given conditions
def pieces_each_friend_delivers : ℕ := 41
def pieces_johann_delivers : ℕ := 98
def number_of_friends : ℕ := 2

-- Theorem statement to prove the total number of pieces of mail delivered
theorem total_pieces_of_mail :
  (number_of_friends * pieces_each_friend_delivers) + pieces_johann_delivers = 180 := 
by
  -- proof would go here
  sorry

end total_pieces_of_mail_l254_254077


namespace smallest_y_in_arithmetic_series_l254_254998

theorem smallest_y_in_arithmetic_series (x y z : ℝ) (h1 : x < y) (h2 : y < z) (h3 : (x * y * z) = 216) : y = 6 :=
by 
  sorry

end smallest_y_in_arithmetic_series_l254_254998


namespace necessary_but_not_sufficient_l254_254883

-- Definitions
def quadratic_eq (a b c x : ℝ) := a * x^2 + b * x + c

-- The condition we are given
axiom m : ℝ

-- The quadratic equation specific condition
axiom quadratic_condition : quadratic_eq 1 2 m = 0

-- The necessary but not sufficient condition for real solutions
theorem necessary_but_not_sufficient (h : m < 2) : 
  ∃ x : ℝ, quadratic_eq 1 2 m x = 0 ∧ quadratic_eq 1 2 m x = 0 → m ≤ 1 ∨ m > 1 :=
sorry

end necessary_but_not_sufficient_l254_254883


namespace bucket_full_weight_l254_254876

variable (p q : ℝ)

theorem bucket_full_weight (p q : ℝ) (x y: ℝ) (h1 : x + 3/4 * y = p) (h2 : x + 1/3 * y = q) :
  x + y = (8 * p - 7 * q) / 5 :=
by
  sorry

end bucket_full_weight_l254_254876


namespace find_a_plus_b_l254_254925

theorem find_a_plus_b (a b : ℝ) 
  (h1 : 2 = a - b / 2) 
  (h2 : 6 = a - b / 3) : 
  a + b = 38 := by
  sorry

end find_a_plus_b_l254_254925


namespace percentage_less_than_l254_254639

theorem percentage_less_than (x y : ℝ) (h1 : y = x * 1.8181818181818181) : (∃ P : ℝ, P = 45) :=
by
  sorry

end percentage_less_than_l254_254639


namespace pascals_triangle_third_number_l254_254290

theorem pascals_triangle_third_number (n : ℕ) (k : ℕ) (hnk : n = 51) (hk : k = 2) :
  (nat.choose n k) = 1275 :=
by {
  subst hnk,
  subst hk,
  sorry
}

end pascals_triangle_third_number_l254_254290


namespace value_expression_at_5_l254_254358

theorem value_expression_at_5 (x : ℕ) (hx : x = 5) : 2 * x^2 + 4 = 54 :=
by
  -- Adding sorry to skip the proof.
  sorry

end value_expression_at_5_l254_254358


namespace arithmetic_seq_sixth_term_l254_254268

theorem arithmetic_seq_sixth_term
  (a d : ℤ)
  (h1 : a + d = 14)
  (h2 : a + 3 * d = 32) : a + 5 * d = 50 := 
by
  sorry

end arithmetic_seq_sixth_term_l254_254268


namespace evaluate_expression_l254_254812

-- Defining the conditions and constants as per the problem statement
def factor_power_of_2 (n : ℕ) : ℕ :=
  if n % 8 = 0 then 3 else 0 -- Greatest power of 2 in 360
  
def factor_power_of_5 (n : ℕ) : ℕ :=
  if n % 5 = 0 then 1 else 0 -- Greatest power of 5 in 360

def expression (b a : ℕ) : ℚ := (2 / 3)^(b - a)

noncomputable def target_value : ℚ := 9 / 4

theorem evaluate_expression : expression (factor_power_of_5 360) (factor_power_of_2 360) = target_value := 
  by
    sorry

end evaluate_expression_l254_254812


namespace base_6_units_digit_l254_254839

def num1 : ℕ := 217
def num2 : ℕ := 45
def base : ℕ := 6

theorem base_6_units_digit :
  (num1 % base) * (num2 % base) % base = (num1 * num2) % base :=
by
  sorry

end base_6_units_digit_l254_254839


namespace nonagon_diagonals_l254_254051

-- Define the number of sides for a nonagon.
def n : ℕ := 9

-- Define the formula for the number of diagonals in a polygon.
def D (n : ℕ) : ℕ := n * (n - 3) / 2

-- State the theorem to prove that the number of diagonals in a nonagon is 27.
theorem nonagon_diagonals : D n = 27 := by
  sorry

end nonagon_diagonals_l254_254051


namespace f_800_value_l254_254813

theorem f_800_value (f : ℝ → ℝ) (f_condition : ∀ x y : ℝ, 0 < x → 0 < y → f (x * y) = f x / y) (f_400 : f 400 = 4) : f 800 = 2 :=
  sorry

end f_800_value_l254_254813


namespace unique_solution_l254_254759

noncomputable def check_triplet (a b c : ℕ) : Prop :=
  5^a + 3^b - 2^c = 32

theorem unique_solution : ∀ (a b c : ℕ), a > 0 ∧ b > 0 ∧ c > 0 ∧ check_triplet a b c ↔ (a = 2 ∧ b = 2 ∧ c = 1) :=
  by sorry

end unique_solution_l254_254759


namespace length_of_bridge_l254_254567

theorem length_of_bridge
  (length_of_train : ℕ)
  (speed_km_hr : ℝ)
  (time_sec : ℝ)
  (h_train_length : length_of_train = 155)
  (h_train_speed : speed_km_hr = 45)
  (h_time : time_sec = 30) :
  ∃ (length_of_bridge : ℝ),
    length_of_bridge = 220 :=
by
  sorry

end length_of_bridge_l254_254567


namespace supplement_of_angle_l254_254975

theorem supplement_of_angle (complement_of_angle : ℝ) (h1 : complement_of_angle = 30) :
  ∃ (angle supplement_angle : ℝ), angle + complement_of_angle = 90 ∧ angle + supplement_angle = 180 ∧ supplement_angle = 120 :=
by
  sorry

end supplement_of_angle_l254_254975


namespace jar_last_days_l254_254517

theorem jar_last_days :
  let serving_size := 0.5 -- each serving is 0.5 ounces
  let daily_servings := 3  -- James uses 3 servings every day
  let quart_ounces := 32   -- 1 quart = 32 ounces
  let jar_size := quart_ounces - 2 -- container is 2 ounces less than 1 quart
  let daily_consumption := daily_servings * serving_size
  let number_of_days := jar_size / daily_consumption
  number_of_days = 20 := by
  sorry

end jar_last_days_l254_254517


namespace sequence_formula_l254_254341

theorem sequence_formula (a : ℕ → ℝ)
  (h1 : ∀ n : ℕ, a n ≠ 0)
  (h2 : a 1 = 1)
  (h3 : ∀ n : ℕ, n > 0 → a (n + 1) = 1 / (n + 1 + 1 / (a n))) :
  ∀ n : ℕ, n > 0 → a n = 2 / ((n : ℝ) ^ 2 - n + 2) :=
by
  sorry

end sequence_formula_l254_254341


namespace average_marks_l254_254868

-- Define the conditions
variables (M P C : ℝ)
variables (h1 : M + P = 60) (h2 : C = P + 10)

-- Define the theorem statement
theorem average_marks : (M + C) / 2 = 35 :=
by {
  sorry -- Placeholder for the proof.
}

end average_marks_l254_254868


namespace arctan_sum_pi_div_two_l254_254909

noncomputable def arctan_sum : Real :=
  Real.arctan (3 / 4) + Real.arctan (4 / 3)

theorem arctan_sum_pi_div_two : arctan_sum = Real.pi / 2 := 
by sorry

end arctan_sum_pi_div_two_l254_254909


namespace find_t_l254_254200

noncomputable def a_sequence (a : ℕ → ℤ) : Prop :=
  a 1 = 5 ∧ ∀ n : ℕ, n ≥ 2 → a (n + 1) = 3 * a n + 3 ^ n

noncomputable def b_sequence (a : ℕ → ℤ) (b : ℕ → ℤ) (t : ℤ) : Prop :=
  ∀ n : ℕ, b n = (a (n + 1) + t) / 3^(n + 1)

theorem find_t (a : ℕ → ℤ) (b : ℕ → ℤ) (t : ℤ) :
  a_sequence a →
  b_sequence a b t →
  (∀ n : ℕ, (b (n + 1) - b n) = (b 1 - b 0)) →
  t = -1 / 2 :=
by
  sorry

end find_t_l254_254200


namespace prob_x_lt_y_is_correct_l254_254714

open Set

noncomputable def prob_x_lt_y : ℝ :=
  let rectangle := Icc (0: ℝ) 4 ×ˢ Icc (0: ℝ) 3
  let area_rectangle := 4 * 3
  let triangle := {p : ℝ × ℝ | p.1 ∈ Icc (0: ℝ) 3 ∧ p.2 ∈ Icc (0: ℝ) 3 ∧ p.1 < p.2}
  let area_triangle := 1 / 2 * 3 * 3
  let probability := area_triangle / area_rectangle
  probability

-- To state as a theorem using Lean's notation
theorem prob_x_lt_y_is_correct : prob_x_lt_y = 3 / 8 := sorry

end prob_x_lt_y_is_correct_l254_254714


namespace simplify_fraction_l254_254832

theorem simplify_fraction :
  (1 / ((1 / (Real.sqrt 2 + 1)) + (2 / (Real.sqrt 3 - 1)) + (3 / (Real.sqrt 5 + 2)))) =
  (1 / (Real.sqrt 2 + 2 * Real.sqrt 3 + 3 * Real.sqrt 5 - 5)) :=
by
  sorry

end simplify_fraction_l254_254832


namespace base_seven_sum_l254_254761

def base_seven_sum_of_product (n m : ℕ) : ℕ :=
  let product := n * m
  let digits := product.digits 7
  digits.sum

theorem base_seven_sum (k l : ℕ) (hk : k = 5 * 7 + 3) (hl : l = 343) :
  base_seven_sum_of_product k l = 11 := by
  sorry

end base_seven_sum_l254_254761


namespace steaks_from_15_pounds_of_beef_l254_254819

-- Definitions for conditions
def pounds_to_ounces (pounds : ℕ) : ℕ := pounds * 16

def steaks_count (total_ounces : ℕ) (ounces_per_steak : ℕ) : ℕ := total_ounces / ounces_per_steak

-- Translate the problem to Lean statement
theorem steaks_from_15_pounds_of_beef : 
  steaks_count (pounds_to_ounces 15) 12 = 20 :=
by
  sorry

end steaks_from_15_pounds_of_beef_l254_254819


namespace fraction_reduction_l254_254973

theorem fraction_reduction (x y : ℝ) : 
  (4 * x - 4 * y) / (4 * x * 4 * y) = (1 / 4) * ((x - y) / (x * y)) := 
by 
  sorry

end fraction_reduction_l254_254973


namespace planted_field_fraction_is_correct_l254_254025

noncomputable def planted_fraction : ℚ :=
  let a := 5
  let b := 12
  let hypotenuse := Real.sqrt (a^2 + b^2)
  let triangle_area := (1 / 2 : ℚ) * a * b
  
  -- Side length of the square
  let x := (3 : ℚ) * (7 : ℚ)^(-1)
  let square_area := x^2

  -- Planted area
  let planted_area := triangle_area - square_area
  
  -- Calculated planted fraction
  planted_area / triangle_area

theorem planted_field_fraction_is_correct :
  planted_fraction = 1461 / 1470 :=
sorry

end planted_field_fraction_is_correct_l254_254025


namespace gcd_38_23_is_1_l254_254312

theorem gcd_38_23_is_1 : Nat.gcd 38 23 = 1 := by
  sorry

end gcd_38_23_is_1_l254_254312


namespace problem_statement_l254_254336

noncomputable def theta (h1 : 2 * Real.cos θ + Real.sin θ = 0) (h2 : 0 < θ ∧ θ < Real.pi) : Real :=
θ

noncomputable def varphi (h4 : Real.pi / 2 < φ ∧ φ < Real.pi) : Real :=
φ

theorem problem_statement
  (θ : Real) (φ : Real)
  (h1 : 2 * Real.cos θ + Real.sin θ = 0)
  (h2 : 0 < θ ∧ θ < Real.pi)
  (h3 : Real.sin (θ - φ) = Real.sqrt 10 / 10)
  (h4 : Real.pi / 2 < φ ∧ φ < Real.pi) :
  Real.tan θ = -2 ∧
  Real.sin θ = (2 * Real.sqrt 5) / 5 ∧
  Real.cos θ = -Real.sqrt 5 / 5 ∧
  Real.cos φ = -Real.sqrt 2 / 10 :=
by
  sorry

end problem_statement_l254_254336


namespace true_proposition_among_options_l254_254005

theorem true_proposition_among_options :
  (∀ (x y : ℝ), (x > |y|) → (x > y)) ∧
  (¬ (∀ (x : ℝ), (x > 1) → (x^2 > 1))) ∧
  (¬ (∀ (x : ℤ), (x = 1) → (x^2 + x - 2 = 0))) ∧
  (¬ (∀ (x : ℝ), (x^2 > 0) → (x > 1))) :=
by
  sorry

end true_proposition_among_options_l254_254005


namespace unique_solution_of_functional_equation_l254_254323

theorem unique_solution_of_functional_equation
  (f : ℝ → ℝ)
  (h : ∀ x y : ℝ, f (f (x + y)) = f x + y) :
  ∀ x : ℝ, f x = x := 
sorry

end unique_solution_of_functional_equation_l254_254323


namespace solution_of_system_l254_254263

variable (x y : ℝ) 

def equation1 (x y : ℝ) : Prop := 3 * |x| + 5 * y + 9 = 0
def equation2 (x y : ℝ) : Prop := 2 * x - |y| - 7 = 0

theorem solution_of_system : ∃ y : ℝ, equation1 0 y ∧ equation2 0 y := by
  sorry

end solution_of_system_l254_254263


namespace sum_of_integers_l254_254271

theorem sum_of_integers (x y : ℕ) (h1 : x - y = 6) (h2 : x * y = 112) (h3 : x > y) : x + y = 22 :=
sorry

end sum_of_integers_l254_254271


namespace interval_of_x_l254_254186

theorem interval_of_x (x : ℝ) : 
  (2 < 4 * x ∧ 4 * x < 3) → (2 < 5 * x ∧ 5 * x < 3) → (1 / 2 < x ∧ x < 3 / 5) :=
by
  sorry

end interval_of_x_l254_254186


namespace repeating_decimals_sum_l254_254749

theorem repeating_decimals_sum : 
  (0.3333333333333333 : ℝ) + (0.0404040404040404 : ℝ) + (0.005005005005005 : ℝ) = (14 / 37 : ℝ) :=
by {
  sorry
}

end repeating_decimals_sum_l254_254749


namespace pascal_triangle_third_number_l254_254289

theorem pascal_triangle_third_number {n k : ℕ} (h : n = 51) (hk : k = 2) : Nat.choose n k = 1275 :=
by
  rw [h, hk]
  norm_num

#check pascal_triangle_third_number

end pascal_triangle_third_number_l254_254289


namespace problem_inequality_solution_l254_254502

noncomputable def find_b_and_c (x : ℝ) (b c : ℝ) : Prop :=
  ∀ x, (x > 2 ∨ x < 1) ↔ x^2 + b*x + c > 0

theorem problem_inequality_solution (x : ℝ) :
  find_b_and_c x (-3) 2 ∧ (2*x^2 - 3*x + 1 ≤ 0 ↔ 1/2 ≤ x ∧ x ≤ 1) :=
by
  sorry

end problem_inequality_solution_l254_254502


namespace total_jelly_beans_l254_254533

theorem total_jelly_beans (vanilla_jelly_beans : ℕ) (h1 : vanilla_jelly_beans = 120) 
                           (grape_jelly_beans : ℕ) (h2 : grape_jelly_beans = 5 * vanilla_jelly_beans + 50) :
                           vanilla_jelly_beans + grape_jelly_beans = 770 :=
by
  sorry

end total_jelly_beans_l254_254533


namespace number_of_prime_digit_numbers_l254_254220

-- Define the set of prime digits
def prime_digits : Set ℕ := {2, 3, 5, 7}

-- Define the predicate for a three-digit number with each digit being a prime
def is_prime_digit_number (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000 ∧ 
  (prime_digits.contains ((n / 100) % 10)) ∧ 
  (prime_digits.contains ((n / 10) % 10)) ∧ 
  (prime_digits.contains (n % 10))

-- The proof problem statement
theorem number_of_prime_digit_numbers : 
  (Finset.univ.filter (λ n : ℕ, is_prime_digit_number n)).card = 64 :=
sorry

end number_of_prime_digit_numbers_l254_254220


namespace allison_greater_probability_l254_254889

-- Definitions and conditions for the problem
def faceRollAllison : Nat := 6
def facesBrian : List Nat := [1, 3, 3, 5, 5, 6]
def facesNoah : List Nat := [4, 4, 4, 4, 5, 5]

-- Function to calculate probability
def probability_less_than (faces : List Nat) (value : Nat) : ℚ :=
  (faces.filter (fun x => x < value)).length / faces.length

-- Main theorem statement
theorem allison_greater_probability :
  probability_less_than facesBrian 6 * probability_less_than facesNoah 6 = 5 / 6 := by
  sorry

end allison_greater_probability_l254_254889


namespace four_nonzero_complex_numbers_form_square_l254_254057

open Complex

theorem four_nonzero_complex_numbers_form_square :
  ∃ (S : Finset ℂ), S.card = 4 ∧ (∀ z ∈ S, z ≠ 0) ∧ (∀ z ∈ S, ∃ (θ : ℝ), z = exp (θ * I) ∧ (exp (4 * θ * I) - z).re = 0 ∧ (exp (4 * θ * I) - z).im = cos (π / 2)) := 
sorry

end four_nonzero_complex_numbers_form_square_l254_254057


namespace number_of_space_diagonals_l254_254713

theorem number_of_space_diagonals (V E F tF qF : ℕ)
    (hV : V = 30) (hE : E = 72) (hF : F = 44) (htF : tF = 34) (hqF : qF = 10) : 
    V * (V - 1) / 2 - E - qF * 2 = 343 :=
by
  sorry

end number_of_space_diagonals_l254_254713


namespace apples_to_mangos_equivalent_l254_254419

-- Definitions and conditions
def apples_worth_mangos (a b : ℝ) : Prop := (5 / 4) * 16 * a = 10 * b

-- Theorem statement
theorem apples_to_mangos_equivalent : 
  ∀ (a b : ℝ), apples_worth_mangos a b → (3 / 4) * 12 * a = 4.5 * b :=
by
  intro a b
  intro h
  sorry

end apples_to_mangos_equivalent_l254_254419


namespace contrapositive_proof_l254_254270

variable {p q : Prop}

theorem contrapositive_proof : (p → q) ↔ (¬q → ¬p) :=
  by sorry

end contrapositive_proof_l254_254270


namespace hyperbola_eccentricity_l254_254500

theorem hyperbola_eccentricity (a b c : ℝ) (h_a_pos : 0 < a) (h_b_pos : 0 < b) 
  (h_eq : b ^ 2 = (5 / 4) * a ^ 2) 
  (h_c : c ^ 2 = a ^ 2 + b ^ 2) : 
  (3 / 2) = c / a :=
by sorry

end hyperbola_eccentricity_l254_254500


namespace range_of_a_l254_254367

theorem range_of_a (a : ℝ) :
  (∀ p : ℝ × ℝ, (p.1 - 2 * a) ^ 2 + (p.2 - (a + 3)) ^ 2 = 4 → p.1 ^ 2 + p.2 ^ 2 = 1) →
  -1 < a ∧ a < 0 := 
sorry

end range_of_a_l254_254367


namespace square_nonneg_of_nonneg_l254_254704

theorem square_nonneg_of_nonneg (x : ℝ) (hx : 0 ≤ x) : 0 ≤ x^2 :=
sorry

end square_nonneg_of_nonneg_l254_254704


namespace five_people_lineup_count_l254_254380

theorem five_people_lineup_count :
  let youngest := "Y"
  let people := ["A", "B", "C", "D", youngest]
  (people' : list string) (yield_positions : list string),
  (yield_positions.all_different ∧ youngest ∉ yield_positions.take 1 ++ yield_positions.drop 4) ∧ 
  yield_positions.permutations.count = 72 :=
by {
  let youngest := "Y"
  let people := ["A", "B", "C", "D", youngest]
  let valid_positions := [[a , b , c, d , youngest], [a, youngest , c , d , youngest], any_order]
  have h : valid_positions.length = 72,
  sorry
}

end five_people_lineup_count_l254_254380


namespace exists_coprime_linear_combination_l254_254592

theorem exists_coprime_linear_combination (a b p : ℤ) :
  ∃ k l : ℤ, Int.gcd k l = 1 ∧ p ∣ (a * k + b * l) :=
  sorry

end exists_coprime_linear_combination_l254_254592


namespace omega_min_value_l254_254424

theorem omega_min_value (ω : ℝ) (hω : ω > 0)
    (hSymmetry : ∀ x : ℝ, sin (ω * x + ω * π / 2 + π / 3) = sin (ω * -x + ω * π / 2 + π / 3)) :
    ω = 1 / 3 :=
begin
  sorry
end

end omega_min_value_l254_254424


namespace min_value_xy_l254_254781

theorem min_value_xy (x y : ℕ) (h : 0 < x ∧ 0 < y) (cond : (1 : ℚ) / x + (1 : ℚ) /(3 * y) = 1 / 6) : 
  xy = 192 :=
sorry

end min_value_xy_l254_254781


namespace quadratic_roots_expression_l254_254243

theorem quadratic_roots_expression (x1 x2 : ℝ) (h1 : x1^2 + x1 - 2023 = 0) (h2 : x2^2 + x2 - 2023 = 0) :
  x1^2 + 2*x1 + x2 = 2022 :=
by
  sorry

end quadratic_roots_expression_l254_254243


namespace value_of_sum_l254_254626

theorem value_of_sum (a b c d : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hd : d ≠ 0)
  (hc_solution : c^2 + a * c + b = 0) (hd_solution : d^2 + a * d + b = 0)
  (ha_solution : a^2 + c * a + d = 0) (hb_solution : b^2 + c * b + d = 0)
: a + b + c + d = -2 := sorry -- The proof is omitted as requested

end value_of_sum_l254_254626


namespace range_of_x_l254_254073

noncomputable def function_y (x : ℝ) : ℝ := 2 / (Real.sqrt (x + 4))

theorem range_of_x : ∀ x : ℝ, (∃ y : ℝ, y = function_y x) → x > -4 :=
by
  intro x h
  sorry

end range_of_x_l254_254073


namespace increase_in_volume_eq_l254_254269

theorem increase_in_volume_eq (x : ℝ) (l w h : ℝ) (h₀ : l = 6) (h₁ : w = 4) (h₂ : h = 5) :
  (6 + x) * 4 * 5 = 6 * 4 * (5 + x) :=
by
  sorry

end increase_in_volume_eq_l254_254269


namespace jorge_acres_l254_254991

theorem jorge_acres (A : ℕ) (H1 : A = 60) 
    (H2 : ∀ acres, acres / 3 = 60 / 3 ∧ 2 * (acres / 3) = 2 * (60 / 3)) 
    (H3 : ∀ good_yield_per_acre, good_yield_per_acre = 400) 
    (H4 : ∀ clay_yield_per_acre, clay_yield_per_acre = 200) 
    (H5 : ∀ total_yield, total_yield = (2 * (A / 3) * 400 + (A / 3) * 200)) 
    : total_yield = 20000 :=
by 
  sorry

end jorge_acres_l254_254991


namespace problem1_problem2_l254_254497

-- We define a point P(x, y) on the circle x^2 + y^2 = 2y.
variables {x y a : ℝ}

-- Condition for the point P to be on the circle
def on_circle (x y : ℝ) : Prop := x^2 + y^2 = 2 * y

-- Definition for 2x + y range
def range_2x_plus_y (x y : ℝ) : Prop := - Real.sqrt 5 + 1 ≤ 2 * x + y ∧ 2 * x + y ≤ Real.sqrt 5 + 1

-- Definition for the range of a given x + y + a ≥ 0
def range_a (x y a : ℝ) : Prop := x + y + a ≥ 0 → a ≥ Real.sqrt 2 - 1

-- Main statements to prove
theorem problem1 (hx : on_circle x y) : range_2x_plus_y x y := sorry

theorem problem2 (hx : on_circle x y) (h : ∀ θ, x = Real.cos θ ∧ y = 1 + Real.sin θ) : range_a x y a := sorry

end problem1_problem2_l254_254497


namespace arctan_sum_pi_div_two_l254_254912

noncomputable def arctan_sum : Real :=
  Real.arctan (3 / 4) + Real.arctan (4 / 3)

theorem arctan_sum_pi_div_two : arctan_sum = Real.pi / 2 := 
by sorry

end arctan_sum_pi_div_two_l254_254912


namespace find_angle_A_l254_254369

theorem find_angle_A (a b c : ℝ) (A : ℝ) (h : a^2 = b^2 - b * c + c^2) : A = 60 :=
sorry

end find_angle_A_l254_254369


namespace budget_percentage_l254_254401

-- Define the given conditions
def basic_salary_per_hour : ℝ := 7.50
def commission_rate : ℝ := 0.16
def hours_worked : ℝ := 160
def total_sales : ℝ := 25000
def amount_for_insurance : ℝ := 260

-- Define the basic salary, commission, and total earnings
def basic_salary : ℝ := basic_salary_per_hour * hours_worked
def commission : ℝ := commission_rate * total_sales
def total_earnings : ℝ := basic_salary + commission
def amount_for_budget : ℝ := total_earnings - amount_for_insurance

-- Define the proof problem
theorem budget_percentage : (amount_for_budget / total_earnings) * 100 = 95 := by
  simp [basic_salary, commission, total_earnings, amount_for_budget]
  sorry

end budget_percentage_l254_254401


namespace arctan_3_4_add_arctan_4_3_is_pi_div_2_l254_254916

noncomputable def arctan_add (a b : ℝ) : ℝ :=
  Real.arctan a + Real.arctan b

theorem arctan_3_4_add_arctan_4_3_is_pi_div_2 :
  arctan_add (3 / 4) (4 / 3) = Real.pi / 2 :=
sorry

end arctan_3_4_add_arctan_4_3_is_pi_div_2_l254_254916


namespace parameter_a_values_l254_254324

theorem parameter_a_values (a : ℝ) :
  (∃ x y : ℝ, |x + y + 8| + |x - y + 8| = 16 ∧ ((|x| - 8)^2 + (|y| - 15)^2 = a) ∧
    (∀ x₁ y₁ x₂ y₂ : ℝ, |x₁ + y₁ + 8| + |x₁ - y₁ + 8| = 16 →
      (|x₁| - 8)^2 + (|y₁| - 15)^2 = a →
      |x₂ + y₂ + 8| + |x₂ - y₂ + 8| = 16 →
      (|x₂| - 8)^2 + (|y₂| - 15)^2 = a →
      (x₁, y₁) = (x₂, y₂) ∨ (x₁, y₁) = (y₂, x₂))) ↔ a = 49 ∨ a = 289 :=
by sorry

end parameter_a_values_l254_254324


namespace interval_of_x_l254_254183

theorem interval_of_x (x : ℝ) : 
  (2 < 4 * x ∧ 4 * x < 3) → (2 < 5 * x ∧ 5 * x < 3) → (1 / 2 < x ∧ x < 3 / 5) :=
by
  sorry

end interval_of_x_l254_254183


namespace bruce_bank_savings_l254_254731

def aunt_gift : ℕ := 75
def grandfather_gift : ℕ := 150
def total_gift : ℕ := aunt_gift + grandfather_gift
def fraction_saved : ℚ := 1/5
def amount_saved : ℚ := total_gift * fraction_saved

theorem bruce_bank_savings : amount_saved = 45 := by
  sorry

end bruce_bank_savings_l254_254731


namespace find_x_if_alpha_beta_eq_4_l254_254524

def alpha (x : ℝ) : ℝ := 4 * x + 9
def beta (x : ℝ) : ℝ := 9 * x + 6

theorem find_x_if_alpha_beta_eq_4 :
  (∃ x : ℝ, alpha (beta x) = 4 ∧ x = -29 / 36) :=
by
  sorry

end find_x_if_alpha_beta_eq_4_l254_254524


namespace smaller_mold_radius_l254_254146

theorem smaller_mold_radius :
  (∀ (R : ℝ) (n : ℕ), 
    R = 2 ∧ n = 64 →
    let V_large := (2 / 3) * Real.pi * R^3 in
    let V_smalls := (2 / 3) * Real.pi * (R / 2 ^ (2 / 3))^3 * n in
    V_large = V_smalls
  ) := 
by {
  intros R n,
  intro h,
  simp at *,
  let V_large := (2/3) * Real.pi * (2:ℝ)^3,
  let V_smalls := (2/3) * Real.pi * (2 / (2 * 2 ^ (1 / 3)))^3 * 64,
  sorry
}

end smaller_mold_radius_l254_254146


namespace find_first_number_l254_254510

theorem find_first_number
  (x y : ℝ)
  (h1 : y = 3.0)
  (h2 : x * y + 4 = 19) : x = 5 := by
  sorry

end find_first_number_l254_254510


namespace interval_of_x_l254_254170

theorem interval_of_x (x : ℝ) : 
  (2 < 4 * x ∧ 4 * x < 3) ∧ (2 < 5 * x ∧ 5 * x < 3) ↔ (1 / 2 < x ∧ x < 3 / 5) :=
by
  sorry

end interval_of_x_l254_254170


namespace can_cut_rectangle_with_area_300_cannot_cut_rectangle_with_ratio_3_2_l254_254527

-- Question and conditions
def side_length_of_square (A : ℝ) := A = 400
def area_of_rect (A : ℝ) := A = 300
def ratio_of_rect (length width : ℝ) := 3 * width = 2 * length

-- Prove that Li can cut a rectangle with area 300 from the square with area 400
theorem can_cut_rectangle_with_area_300 
  (a : ℝ) (h1 : side_length_of_square a)
  (length width : ℝ)
  (ha : a ^ 2 = 400) (har : length * width = 300) :
  length ≤ a ∧ width ≤ a :=
by
  sorry

-- Prove that Li cannot cut a rectangle with ratio 3:2 from the square
theorem cannot_cut_rectangle_with_ratio_3_2 (a : ℝ)
  (h1 : side_length_of_square a)
  (length width : ℝ)
  (har : area_of_rect (length * width))
  (hratio : ratio_of_rect length width)
  (ha : a ^ 2 = 400) :
  ¬(length ≤ a ∧ width ≤ a) :=
by
  sorry

end can_cut_rectangle_with_area_300_cannot_cut_rectangle_with_ratio_3_2_l254_254527


namespace monotonic_decreasing_interval_of_f_l254_254106

noncomputable def f (x : ℝ) : ℝ := x^3 - 3 * x

theorem monotonic_decreasing_interval_of_f :
  ∀ x : ℝ, -1 < x ∧ x < 1 → deriv f x < 0 :=
by
  sorry

end monotonic_decreasing_interval_of_f_l254_254106


namespace b_gets_more_than_c_l254_254459

-- Define A, B, and C as real numbers
variables (A B C : ℝ)

theorem b_gets_more_than_c 
  (h1 : A = 3 * B)
  (h2 : B = C + 25)
  (h3 : A + B + C = 645)
  (h4 : B = 134) : 
  B - C = 25 :=
by
  -- Using the conditions from the problem
  sorry

end b_gets_more_than_c_l254_254459


namespace ginger_distance_l254_254193

theorem ginger_distance : 
  ∀ (d : ℝ), (d / 4 - d / 6 = 1 / 16) → (d = 3 / 4) := 
by 
  intro d h
  sorry

end ginger_distance_l254_254193


namespace circle_radius_l254_254711

theorem circle_radius (M N r : ℝ) (h1 : M = π * r^2) (h2 : N = 2 * π * r) (h3 : M / N = 10) : r = 20 :=
by
  sorry

end circle_radius_l254_254711


namespace expression_simplification_l254_254969

open Real

theorem expression_simplification (x y : ℝ) (h1 : x ≠ 0) (h2 : y ≠ 0) (h3 : 3*x + y / 3 ≠ 0) :
  (3*x + y/3)⁻¹ * ((3*x)⁻¹ + (y/3)⁻¹) = 1 / (3 * (x * y)) :=
by
  -- proof steps would go here
  sorry

end expression_simplification_l254_254969


namespace oak_trees_in_park_l254_254443

theorem oak_trees_in_park (planting_today : ℕ) (total_trees : ℕ) 
  (h1 : planting_today = 4) (h2 : total_trees = 9) : 
  total_trees - planting_today = 5 :=
by
  -- proof goes here
  sorry

end oak_trees_in_park_l254_254443


namespace unique_solution_l254_254486

def is_solution (f : ℝ → ℝ) : Prop :=
  ∀ (x : ℝ) (hx : 0 < x), 
    ∃! (y : ℝ) (hy : 0 < y), 
      x * f y + y * f x ≤ 2

theorem unique_solution : ∀ (f : ℝ → ℝ), 
  is_solution f ↔ (∀ x, 0 < x → f x = 1 / x) :=
by
  intros
  sorry

end unique_solution_l254_254486


namespace expand_product_l254_254939

-- Define the expressions (x + 3)(x + 8) and x^2 + 11x + 24
def expr1 (x : ℝ) : ℝ := (x + 3) * (x + 8)
def expr2 (x : ℝ) : ℝ := x^2 + 11 * x + 24

-- Prove that the two expressions are equal
theorem expand_product (x : ℝ) : expr1 x = expr2 x := by
  sorry

end expand_product_l254_254939


namespace simplified_fraction_l254_254674

noncomputable def simplify_and_rationalize (a b c d e f : ℝ) : ℝ :=
  (Real.sqrt a / Real.sqrt b) * (Real.sqrt c / Real.sqrt d) * (Real.sqrt e / Real.sqrt f)

theorem simplified_fraction :
  simplify_and_rationalize 3 7 5 9 6 8 = Real.sqrt 35 / 14 :=
by
  sorry

end simplified_fraction_l254_254674


namespace base_height_l254_254011

-- Define the height of the sculpture and the combined height.
def sculpture_height : ℚ := 2 + 10 / 12
def total_height : ℚ := 3 + 2 / 3

-- We want to prove that the base height is 5/6 feet.
theorem base_height :
  total_height - sculpture_height = 5 / 6 :=
by
  sorry

end base_height_l254_254011


namespace print_time_nearest_whole_l254_254151

theorem print_time_nearest_whole 
  (pages_per_minute : ℕ) (total_pages : ℕ) (expected_time : ℕ)
  (h1 : pages_per_minute = 25) (h2 : total_pages = 575) : 
  expected_time = 23 :=
by
  sorry

end print_time_nearest_whole_l254_254151


namespace number_of_whole_numbers_between_roots_l254_254356

theorem number_of_whole_numbers_between_roots :
  let sqrt_18 := Real.sqrt 18
  let sqrt_98 := Real.sqrt 98
  Nat.card { x : ℕ | sqrt_18 < x ∧ x < sqrt_98 } = 5 := 
by
  sorry

end number_of_whole_numbers_between_roots_l254_254356


namespace kids_go_to_camp_l254_254742

theorem kids_go_to_camp (total_kids: Nat) (kids_stay_home: Nat) 
  (h1: total_kids = 1363293) (h2: kids_stay_home = 907611) : total_kids - kids_stay_home = 455682 :=
by
  have h_total : total_kids = 1363293 := h1
  have h_stay_home : kids_stay_home = 907611 := h2
  sorry

end kids_go_to_camp_l254_254742


namespace repeating_decimals_sum_l254_254750

theorem repeating_decimals_sum : 
  (0.3333333333333333 : ℝ) + (0.0404040404040404 : ℝ) + (0.005005005005005 : ℝ) = (14 / 37 : ℝ) :=
by {
  sorry
}

end repeating_decimals_sum_l254_254750


namespace profit_calculation_l254_254117

-- Definitions from conditions
def initial_shares := 20
def cost_per_share := 3
def sold_shares := 10
def sale_price_per_share := 4
def remaining_shares_value_multiplier := 2

-- Calculations based on conditions
def initial_cost := initial_shares * cost_per_share
def revenue_from_sold_shares := sold_shares * sale_price_per_share
def remaining_shares := initial_shares - sold_shares
def value_of_remaining_shares := remaining_shares * (cost_per_share * remaining_shares_value_multiplier)
def total_value := revenue_from_sold_shares + value_of_remaining_shares
def expected_profit := total_value - initial_cost

-- The problem statement to be proven
theorem profit_calculation : expected_profit = 40 := by
  -- Proof steps go here
  sorry

end profit_calculation_l254_254117


namespace mixture_alcohol_quantity_l254_254462

theorem mixture_alcohol_quantity:
  ∀ (A W : ℝ), 
    A / W = 4 / 3 ∧ A / (W + 7) = 4 / 5 → A = 14 :=
by
  intros A W h
  sorry

end mixture_alcohol_quantity_l254_254462


namespace sum_of_terms_l254_254641

noncomputable def u1 := 8
noncomputable def r := 2

def first_geometric (u2 u3 : ℝ) (u1 r : ℝ) : Prop := 
  u2 = r * u1 ∧ u3 = r^2 * u1

def last_arithmetic (u2 u3 u4 : ℝ) : Prop := 
  u3 - u2 = u4 - u3

def terms (u1 u2 u3 u4 : ℝ) (r : ℝ) : Prop :=
  first_geometric u2 u3 u1 r ∧
  last_arithmetic u2 u3 u4 ∧
  u4 = u1 + 40

theorem sum_of_terms (u1 u2 u3 u4 : ℝ)
  (h : terms u1 u2 u3 u4 r) : u1 + u2 + u3 + u4 = 104 :=
by {
  sorry
}

end sum_of_terms_l254_254641


namespace solution_is_three_l254_254417

def equation (x : ℝ) : Prop := 
  Real.sqrt (4 - 3 * Real.sqrt (10 - 3 * x)) = x - 2

theorem solution_is_three : equation 3 :=
by sorry

end solution_is_three_l254_254417


namespace contractor_fine_amount_l254_254469

def total_days := 30
def daily_earning := 25
def total_earnings := 360
def days_absent := 12
def days_worked := total_days - days_absent
def fine_per_absent_day (x : ℝ) : Prop :=
  (daily_earning * days_worked) - (x * days_absent) = total_earnings

theorem contractor_fine_amount : ∃ x : ℝ, fine_per_absent_day x := by
  use 7.5
  sorry

end contractor_fine_amount_l254_254469


namespace shorter_base_of_isosceles_trapezoid_l254_254763

theorem shorter_base_of_isosceles_trapezoid
  (a b : ℝ)
  (h : a > b)
  (h_division : (a + b) / 2 = (a - b) / 2 + 10) :
  b = 10 :=
by
  sorry

end shorter_base_of_isosceles_trapezoid_l254_254763


namespace abs_five_minus_sqrt_pi_l254_254897

theorem abs_five_minus_sqrt_pi : |5 - Real.sqrt Real.pi| = 3.22755 := by
  sorry

end abs_five_minus_sqrt_pi_l254_254897


namespace simple_interest_rate_l254_254476

theorem simple_interest_rate (P : ℝ) (R : ℝ) (T : ℝ) 
  (hT : T = 10) (hSI : (P * R * T) / 100 = (1 / 5) * P) : R = 2 :=
by
  sorry

end simple_interest_rate_l254_254476


namespace Tammy_second_day_speed_l254_254834

variable (v t : ℝ)

/-- This statement represents Tammy's climbing situation -/
theorem Tammy_second_day_speed:
  (t + (t - 2) = 14) ∧
  (v * t + (v + 0.5) * (t - 2) = 52) →
  (v + 0.5 = 4) :=
by
  sorry

end Tammy_second_day_speed_l254_254834


namespace number_of_red_pencils_l254_254281

theorem number_of_red_pencils (B R G : ℕ) (h1 : B + R + G = 20) (h2 : B = 6 * G) (h3 : R < B) : R = 6 :=
by
  sorry

end number_of_red_pencils_l254_254281


namespace sum_factors_of_18_l254_254580

theorem sum_factors_of_18 : (1 + 18 + 2 + 9 + 3 + 6) = 39 := by
  sorry

end sum_factors_of_18_l254_254580


namespace arrangements_no_adjacent_dances_arrangements_alternating_order_l254_254722

-- Part (1)
theorem arrangements_no_adjacent_dances (singing_programs dance_programs : ℕ) (h_s : singing_programs = 5) (h_d : dance_programs = 4) :
  ∃ n, n = 43200 := 
by sorry

-- Part (2)
theorem arrangements_alternating_order (singing_programs dance_programs : ℕ) (h_s : singing_programs = 5) (h_d : dance_programs = 4) :
  ∃ n, n = 2880 := 
by sorry

end arrangements_no_adjacent_dances_arrangements_alternating_order_l254_254722


namespace round_2748397_542_nearest_integer_l254_254259

theorem round_2748397_542_nearest_integer :
  let n := 2748397.542
  let int_part := 2748397
  let decimal_part := 0.542
  (n.round = 2748398) :=
by
  sorry

end round_2748397_542_nearest_integer_l254_254259


namespace tangent_line_at_zero_l254_254679

noncomputable def f (x : ℝ) : ℝ := Real.exp x * Real.sin x

theorem tangent_line_at_zero : ∀ x : ℝ, x = 0 → Real.exp x * Real.sin x = 0 ∧ (Real.exp x * (Real.sin x + Real.cos x)) = 1 → (∀ y, y = x) :=
  by
    sorry

end tangent_line_at_zero_l254_254679


namespace average_speed_is_five_l254_254004

-- Define the speeds for each segment
def swimming_speed : ℝ := 2 -- km/h
def biking_speed : ℝ := 15 -- km/h
def running_speed : ℝ := 9 -- km/h
def kayaking_speed : ℝ := 6 -- km/h

-- Define the problem to prove the average speed
theorem average_speed_is_five :
  let segments := [swimming_speed, biking_speed, running_speed, kayaking_speed]
  let harmonic_mean (speeds : List ℝ) : ℝ :=
    let n := speeds.length
    n / (speeds.foldl (fun acc s => acc + 1 / s) 0)
  harmonic_mean segments = 5 := by
  sorry

end average_speed_is_five_l254_254004


namespace sarah_dimes_l254_254096

theorem sarah_dimes (d n : ℕ) (h1 : d + n = 50) (h2 : 10 * d + 5 * n = 200) : d = 10 :=
sorry

end sarah_dimes_l254_254096


namespace pipe_q_fill_time_l254_254574

theorem pipe_q_fill_time :
  ∀ (T : ℝ), (2 * (1 / 10 + 1 / T) + 10 * (1 / T) = 1) → T = 15 :=
by
  intro T
  intro h
  sorry

end pipe_q_fill_time_l254_254574


namespace stacy_history_paper_length_l254_254264

theorem stacy_history_paper_length
  (days : ℕ)
  (pages_per_day : ℕ)
  (h_days : days = 6)
  (h_pages_per_day : pages_per_day = 11) :
  (days * pages_per_day) = 66 :=
by {
  sorry -- Proof goes here
}

end stacy_history_paper_length_l254_254264


namespace phil_cards_left_l254_254825

-- Conditions
def cards_per_week : ℕ := 20
def weeks_per_year : ℕ := 52

-- Total number of cards in a year
def total_cards (cards_per_week weeks_per_year : ℕ) : ℕ := cards_per_week * weeks_per_year

-- Number of cards left after losing half in fire
def cards_left (total_cards : ℕ) : ℕ := total_cards / 2

-- Theorem to prove
theorem phil_cards_left (cards_per_week weeks_per_year : ℕ) :
  cards_left (total_cards cards_per_week weeks_per_year) = 520 :=
by
  sorry

end phil_cards_left_l254_254825


namespace total_games_played_l254_254137

theorem total_games_played (n : ℕ) (h1 : n = 9) (h2 : ∀ i j, i ≠ j → 4 * (Nat.choose n 2) = 144) : 
  4 * (Nat.choose 9 2) = 144 := by
  sorry

end total_games_played_l254_254137


namespace arctan_triangle_complementary_l254_254905

theorem arctan_triangle_complementary :
  (Real.arctan (3 / 4) + Real.arctan (4 / 3) = Real.pi / 2) :=
begin
  sorry
end

end arctan_triangle_complementary_l254_254905


namespace repeating_sum_to_fraction_l254_254743

theorem repeating_sum_to_fraction :
  (0.333333333333333 ~ 1/3) ∧ 
  (0.0404040404040401 ~ 4/99) ∧ 
  (0.005005005005001 ~ 5/999) →
  (0.333333333333333 + 0.0404040404040401 + 0.005005005005001) = (112386 / 296703) := 
by
  repeat { sorry }

end repeating_sum_to_fraction_l254_254743


namespace P_is_in_third_quadrant_l254_254087

noncomputable def point : Type := (ℝ × ℝ)

def P : point := (-3, -4)

def is_in_third_quadrant (p : point) : Prop :=
  p.1 < 0 ∧ p.2 < 0

theorem P_is_in_third_quadrant : is_in_third_quadrant P :=
by {
  -- Prove that P is in the third quadrant
  sorry
}

end P_is_in_third_quadrant_l254_254087


namespace area_T_is_34_l254_254472

/-- Define the dimensions of the large rectangle -/
def width_rect : ℕ := 10
def height_rect : ℕ := 4
/-- Define the dimensions of the removed section -/
def width_removed : ℕ := 6
def height_removed : ℕ := 1

/-- Calculate the area of the large rectangle -/
def area_rect : ℕ := width_rect * height_rect

/-- Calculate the area of the removed section -/
def area_removed : ℕ := width_removed * height_removed

/-- Calculate the area of the "T" shape -/
def area_T : ℕ := area_rect - area_removed

/-- To prove that the area of the T-shape is 34 square units -/
theorem area_T_is_34 : area_T = 34 := 
by {
  sorry
}

end area_T_is_34_l254_254472


namespace solve_system_of_equations_l254_254418

/-- Definition representing our system of linear equations. --/
def system_of_equations (x1 x2 : ℚ) : Prop :=
  (3 * x1 - 5 * x2 = 2) ∧ (2 * x1 + 4 * x2 = 5)

/-- The main theorem stating the solution to our system of equations. --/
theorem solve_system_of_equations : 
  ∃ x1 x2 : ℚ, system_of_equations x1 x2 ∧ x1 = 3/2 ∧ x2 = 1/2 :=
by
  sorry

end solve_system_of_equations_l254_254418


namespace coaches_needed_l254_254595

theorem coaches_needed (x : ℕ) : 44 * x + 64 = 328 := by
  sorry

end coaches_needed_l254_254595


namespace jelly_beans_total_l254_254537

-- Definitions from the conditions
def vanilla : Nat := 120
def grape : Nat := 5 * vanilla + 50
def total : Nat := vanilla + grape

-- Statement to prove
theorem jelly_beans_total :
  total = 770 := 
by 
  sorry

end jelly_beans_total_l254_254537


namespace distinct_positive_factors_48_l254_254055

theorem distinct_positive_factors_48 : 
  ∀ (n : ℕ), n = 48 → ∀ (p q : ℕ), p = 2 ∧ q = 3 → (∃ a b : ℕ, 48 = p^a * q^b ∧ (a + 1) * (b + 1) = 10) :=
by
  intros n hn p q hpq
  have h_48 : 48 = 2^4 * 3^1 := by norm_num
  use 4, 1
  split
  · exact h_48
  · norm_num
  sorry

end distinct_positive_factors_48_l254_254055


namespace quadratic_rational_solutions_product_l254_254607

theorem quadratic_rational_solutions_product :
  ∃ (c₁ c₂ : ℕ), (7 * x^2 + 15 * x + c₁ = 0 ∧ 225 - 28 * c₁ = k^2 ∧ ∃ k : ℤ, k^2 = 225 - 28 * c₁) ∧
                 (7 * x^2 + 15 * x + c₂ = 0 ∧ 225 - 28 * c₂ = k^2 ∧ ∃ k : ℤ, k^2 = 225 - 28 * c₂) ∧
                 (c₁ = 1) ∧ (c₂ = 8) ∧ (c₁ * c₂ = 8) :=
by
  sorry

end quadratic_rational_solutions_product_l254_254607


namespace intersecting_lines_b_plus_m_l254_254118

theorem intersecting_lines_b_plus_m :
  ∃ (m b : ℚ), (∀ x y : ℚ, y = m * x + 5 → y = 4 * x + b → (x, y) = (8, 14)) →
               b + m = -63 / 4 :=
by
  sorry

end intersecting_lines_b_plus_m_l254_254118


namespace expression_evaluation_l254_254581

theorem expression_evaluation : (50 + 12) ^ 2 - (12 ^ 2 + 50 ^ 2) = 1200 := 
by
  sorry

end expression_evaluation_l254_254581


namespace smallest_sum_l254_254205

theorem smallest_sum (x y : ℕ) (h_pos_x : x > 0) (h_pos_y : y > 0) (h_neq : x ≠ y)
  (h_eq : (1 : ℚ) / x + (1 : ℚ) / y = 1 / 18) : x + y = 75 :=
by
  sorry

end smallest_sum_l254_254205


namespace students_didnt_make_cut_l254_254113

theorem students_didnt_make_cut (g b c : ℕ) (hg : g = 15) (hb : b = 25) (hc : c = 7) : g + b - c = 33 := by
  sorry

end students_didnt_make_cut_l254_254113


namespace number_of_valid_permutations_l254_254375

theorem number_of_valid_permutations : 
  let n := 5 in 
  let total_permutations := n! in 
  let restricted_permutations := 2 * (n - 1)! in 
  total_permutations - restricted_permutations = 72 := 
by 
  sorry

end number_of_valid_permutations_l254_254375


namespace absent_children_l254_254821

theorem absent_children (A : ℕ) (h1 : 2 * 610 = (610 - A) * 4) : A = 305 := 
by sorry

end absent_children_l254_254821


namespace polar_coordinates_of_point_l254_254160

theorem polar_coordinates_of_point :
  ∃ (r θ : ℝ), r = 2 ∧ θ = (2 * Real.pi) / 3 ∧
  (r > 0) ∧ (0 ≤ θ) ∧ (θ < 2 * Real.pi) ∧
  (-1, Real.sqrt 3) = (r * Real.cos θ, r * Real.sin θ) :=
by 
  sorry

end polar_coordinates_of_point_l254_254160


namespace interval_intersection_l254_254176

/--
  This statement asserts that the intersection of the intervals (2/4, 3/4) and (2/5, 3/5)
  results in the interval (1/2, 0.6), which is the solution to the problem.
-/
theorem interval_intersection :
  { x : ℝ | 2 < 4 * x ∧ 4 * x < 3 ∧ 2 < 5 * x ∧ 5 * x < 3 } = { x : ℝ | 0.5 < x ∧ x < 0.6 } :=
by
  sorry

end interval_intersection_l254_254176


namespace arctan_3_4_add_arctan_4_3_is_pi_div_2_l254_254915

noncomputable def arctan_add (a b : ℝ) : ℝ :=
  Real.arctan a + Real.arctan b

theorem arctan_3_4_add_arctan_4_3_is_pi_div_2 :
  arctan_add (3 / 4) (4 / 3) = Real.pi / 2 :=
sorry

end arctan_3_4_add_arctan_4_3_is_pi_div_2_l254_254915


namespace josef_timothy_game_l254_254988

theorem josef_timothy_game (n : ℕ) (h₁ : 1 ≤ n) (h₂ : n ≤ 1000) : 
  (∃ k : ℕ, 1000 = k * n) → ∃ d : ℕ, d = 16 :=
by
  sorry

end josef_timothy_game_l254_254988


namespace walnuts_amount_l254_254399

theorem walnuts_amount (w : ℝ) (total_nuts : ℝ) (almonds : ℝ) (h1 : total_nuts = 0.5) (h2 : almonds = 0.25) (h3 : w + almonds = total_nuts) : w = 0.25 :=
by
  sorry

end walnuts_amount_l254_254399


namespace probability_of_isosceles_triangle_l254_254231

open ProbabilityTheory

def balls (in_bag out_bag : Finset ℕ) := out_bag = {3, 6} ∧ in_bag = {3, 4, 5, 6}
def isosceles_triangle (x y z : ℕ) := (x = y ∨ y = z ∨ x = z) ∧ (x + y > z) ∧ (x + z > y) ∧ (y + z > x)
def successful_outcomes (in_bag : Finset ℕ) (out_bag : Finset ℕ) := 
  {x ∈ in_bag | isosceles_triangle x 3 6}

noncomputable def probability_isosceles (in_bag out_bag : Finset ℕ) (h : balls in_bag out_bag) : ℚ := 
  (((successful_outcomes in_bag out_bag).card : ℚ) / in_bag.card)

theorem probability_of_isosceles_triangle : 
  ∀ (in_bag out_bag : Finset ℕ), balls in_bag out_bag → probability_isosceles in_bag out_bag = 1 / 4 := by
  intros in_bag out_bag h
  sorry

end probability_of_isosceles_triangle_l254_254231


namespace arctan_3_4_add_arctan_4_3_is_pi_div_2_l254_254914

noncomputable def arctan_add (a b : ℝ) : ℝ :=
  Real.arctan a + Real.arctan b

theorem arctan_3_4_add_arctan_4_3_is_pi_div_2 :
  arctan_add (3 / 4) (4 / 3) = Real.pi / 2 :=
sorry

end arctan_3_4_add_arctan_4_3_is_pi_div_2_l254_254914


namespace correct_choice_l254_254345

def proposition_p : Prop := ∀ (x : ℝ), 2^x > x^2
def proposition_q : Prop := ∃ (x_0 : ℝ), x_0 - 2 > 0

theorem correct_choice : ¬proposition_p ∧ proposition_q :=
by
  sorry

end correct_choice_l254_254345


namespace prove_market_demand_prove_tax_revenue_prove_per_unit_tax_rate_prove_tax_revenue_specified_l254_254463

noncomputable def market_supply_function (P : ℝ) : ℝ := 6 * P - 312

noncomputable def market_demand_function (a b P : ℝ) : ℝ := a - b * P

noncomputable def price_elasticity_supply (P_e Q_e : ℝ) : ℝ := 6 * (P_e / Q_e)

noncomputable def price_elasticity_demand (b P_e Q_e : ℝ) : ℝ := -b * (P_e / Q_e)

noncomputable def tax_rate := 30

noncomputable def consumer_price_after_tax := 118

theorem prove_market_demand (a P_e Q_e : ℝ) :
  1.5 * |price_elasticity_demand 4 P_e Q_e| = price_elasticity_supply P_e Q_e →
  market_demand_function a 4 P_e = a - 4 * P_e := sorry

theorem prove_tax_revenue (Q_d : ℝ) :
  Q_d = 216 →
  Q_d * tax_rate = 6480 := sorry

theorem prove_per_unit_tax_rate (t : ℝ) :
  t = 60 → 4 * t = 240 := sorry

theorem prove_tax_revenue_specified (t : ℝ) :
  t = 60 →
  (288 * t - 2.4 * t^2) = 8640 := sorry

end prove_market_demand_prove_tax_revenue_prove_per_unit_tax_rate_prove_tax_revenue_specified_l254_254463


namespace max_investment_at_7_percent_l254_254721

variables (x y : ℝ)

theorem max_investment_at_7_percent 
  (h1 : x + y = 25000)
  (h2 : 0.07 * x + 0.12 * y ≥ 2450) : 
  x ≤ 11000 :=
sorry

end max_investment_at_7_percent_l254_254721


namespace molecular_weight_of_one_mole_l254_254286

-- Definitions derived from the conditions in the problem:

def molecular_weight_nine_moles (w : ℕ) : ℕ :=
  2664

def molecular_weight_one_mole (w : ℕ) : ℕ :=
  w / 9

-- The theorem to prove, based on the above definitions and conditions:
theorem molecular_weight_of_one_mole (w : ℕ) (hw : molecular_weight_nine_moles w = 2664) :
  molecular_weight_one_mole w = 296 :=
sorry

end molecular_weight_of_one_mole_l254_254286


namespace kids_left_playing_l254_254686

-- Define the conditions
def initial_kids : ℝ := 22.0
def kids_went_home : ℝ := 14.0

-- Theorem statement: Prove that the number of kids left playing is 8.0
theorem kids_left_playing : initial_kids - kids_went_home = 8.0 :=
by
  sorry -- Proof is left as an exercise

end kids_left_playing_l254_254686


namespace choosing_officers_l254_254344

theorem choosing_officers :
  let members := {Alice, Bob, Carol, Dave}
  let qualifications (m : member) := (m = Dave) -> true 
  let eligible_for_president (m : member) := (m = Dave)
  let officers := {president, secretary, treasurer} 
  let ways_to_assign_roles := 1 * (Nat.choose 3 2) * 2 
  ways_to_assign_roles = 6 :=
by
  sorry

end choosing_officers_l254_254344


namespace arctan_sum_pi_div_two_l254_254911

noncomputable def arctan_sum : Real :=
  Real.arctan (3 / 4) + Real.arctan (4 / 3)

theorem arctan_sum_pi_div_two : arctan_sum = Real.pi / 2 := 
by sorry

end arctan_sum_pi_div_two_l254_254911


namespace arctan_triangle_complementary_l254_254907

theorem arctan_triangle_complementary :
  (Real.arctan (3 / 4) + Real.arctan (4 / 3) = Real.pi / 2) :=
begin
  sorry
end

end arctan_triangle_complementary_l254_254907


namespace angle_C_is_sixty_l254_254234

variable {A B C D E : Type}
variable {AD BE BC AC : ℝ}
variable {triangle : A ≠ B ∧ B ≠ C ∧ C ≠ A} 
variable (angle_C : ℝ)

-- Given conditions
variable (h_eq : AD * BC = BE * AC)
variable (h_ineq : AC ≠ BC)

-- To prove
theorem angle_C_is_sixty (h_eq : AD * BC = BE * AC) (h_ineq : AC ≠ BC) : angle_C = 60 :=
by
  sorry

end angle_C_is_sixty_l254_254234


namespace calculate_expression_l254_254157

theorem calculate_expression :
  3 ^ 3 * 2 ^ 2 * 7 ^ 2 * 11 = 58212 :=
by
  sorry

end calculate_expression_l254_254157


namespace tan_arctan_five_twelfths_l254_254603

theorem tan_arctan_five_twelfths : Real.tan (Real.arctan (5 / 12)) = 5 / 12 :=
by
  sorry

end tan_arctan_five_twelfths_l254_254603


namespace probability_of_exactly_one_head_l254_254690

theorem probability_of_exactly_one_head (h1 h2 : Bool) :
  let outcomes := [(true, true), (true, false), (false, true), (false, false)] in
  let favorable := [(true, false), (false, true)] in
  2 / 4 = 1 / 2 :=
by
  sorry

end probability_of_exactly_one_head_l254_254690


namespace period_of_y_l254_254451

def y (x : ℝ) : ℝ := 2 * Real.sin x + 3 * Real.cos x

theorem period_of_y : ∀ x, y(x + 2 * Real.pi) = y(x) := sorry

end period_of_y_l254_254451


namespace isosceles_triangle_perimeter_l254_254070

-- Define the lengths of the sides
def side1 := 2 -- 2 cm
def side2 := 4 -- 4 cm

-- Define the condition of being isosceles
def is_isosceles (a b c : ℝ) : Prop := (a = b) ∨ (a = c) ∨ (b = c)

-- Define the triangle inequality
def triangle_inequality (a b c : ℝ) : Prop :=
  (a + b > c) ∧ (a + c > b) ∧ (b + c > a)

-- Define the triangle perimeter
def perimeter (a b c : ℝ) : ℝ := a + b + c

-- Define the main theorem to prove
theorem isosceles_triangle_perimeter {a b : ℝ} (ha : a = side1) (hb : b = side2)
    (h1 : is_isosceles a b c) (h2 : triangle_inequality a b c) : perimeter a b c = 10 :=
sorry

end isosceles_triangle_perimeter_l254_254070


namespace sum_of_repeating_decimals_l254_254754

/-- Definitions of the repeating decimals as real numbers. --/
def x : ℝ := 0.3 -- This actually represents 0.\overline{3} in Lean
def y : ℝ := 0.04 -- This actually represents 0.\overline{04} in Lean
def z : ℝ := 0.005 -- This actually represents 0.\overline{005} in Lean

/-- The theorem stating that the sum of these repeating decimals is a specific fraction. --/
theorem sum_of_repeating_decimals : x + y + z = (14 : ℝ) / 37 := 
by 
  sorry -- Placeholder for the proof

end sum_of_repeating_decimals_l254_254754


namespace solve_a_plus_b_l254_254967

theorem solve_a_plus_b (a b : ℕ) (h_pos_a : 0 < a) (h_pos_b : 0 < b)
  (h_eq : 143 * a + 500 * b = 2001) : a + b = 9 :=
by
  -- Add proof here
  sorry

end solve_a_plus_b_l254_254967


namespace circle_chord_length_equal_l254_254808

def equation_of_circle (D E F : ℝ) : Prop :=
  ∀ x y : ℝ, x^2 + y^2 + D * x + E * y + F = 0

def distances_equal (D E F : ℝ) : Prop :=
  (D^2 ≠ E^2 ∧ E^2 > 4 * F) → 
  (∀ x y : ℝ, (x^2 + y^2 + D * x + E * y + F = 0) → (x = -D/2) ∧ (y = -E/2) → (abs x = abs y))

theorem circle_chord_length_equal (D E F : ℝ) (h : D^2 ≠ E^2 ∧ E^2 > 4 * F) :
  distances_equal D E F :=
by
  sorry

end circle_chord_length_equal_l254_254808


namespace geometric_sequence_n_l254_254981

theorem geometric_sequence_n (a1 an q : ℚ) (n : ℕ) (h1 : a1 = 9 / 8) (h2 : an = 1 / 3) (h3 : q = 2 / 3) : n = 4 :=
by
  sorry

end geometric_sequence_n_l254_254981


namespace correct_factorization_l254_254583

theorem correct_factorization (x m n a : ℝ) : 
  (¬ (x^2 + 2 * x + 1 = x * (x + 2) + 1)) ∧
  (¬ (m^2 - 2 * m * n + n^2 = (m + n)^2)) ∧
  (¬ (-a^4 + 16 = -(a^2 + 4) * (a^2 - 4))) ∧
  (x^3 - 4 * x = x * (x + 2) * (x - 2)) :=
by
  sorry

end correct_factorization_l254_254583


namespace solution_set_M_inequality_ab_l254_254629

def f (x : ℝ) : ℝ := |x - 1| + |x + 3|

theorem solution_set_M :
  {x | -3 ≤ x ∧ x ≤ 1} = { x : ℝ | f x ≤ 4 } :=
sorry

theorem inequality_ab
  (a b : ℝ) (h1 : -3 ≤ a ∧ a ≤ 1) (h2 : -3 ≤ b ∧ b ≤ 1) :
  (a^2 + 2 * a - 3) * (b^2 + 2 * b - 3) ≥ 0 :=
sorry

end solution_set_M_inequality_ab_l254_254629


namespace tan_identity_find_sum_l254_254213

-- Given conditions
def is_geometric_sequence (a b c : ℝ) : Prop := b^2 = a * c

-- Specific problem statements
theorem tan_identity (a b c : ℝ) (A B C : ℝ)
  (h_geometric : is_geometric_sequence a b c)
  (h_cosB : Real.cos B = 3 / 4) :
  1 / Real.tan A + 1 / Real.tan C = 4 / Real.sqrt 7 :=
sorry

theorem find_sum (a b c : ℝ)
  (h_dot_product : a * c * 3 / 4 = 3 / 2) :
  a + c = 3 :=
sorry

end tan_identity_find_sum_l254_254213


namespace arctan_sum_eq_pi_div_two_l254_254921

theorem arctan_sum_eq_pi_div_two : Real.arctan (3 / 4) + Real.arctan (4 / 3) = Real.pi / 2 :=
by
  sorry

end arctan_sum_eq_pi_div_two_l254_254921


namespace ice_cream_total_volume_l254_254435

/-- 
  The interior of a right, circular cone is 12 inches tall with a 3-inch radius at the opening.
  The interior of the cone is filled with ice cream.
  The cone has a hemisphere of ice cream exactly covering the opening of the cone.
  On top of this hemisphere, there is a cylindrical layer of ice cream of height 2 inches 
  and the same radius as the hemisphere (3 inches).
  Prove that the total volume of ice cream is 72π cubic inches.
-/
theorem ice_cream_total_volume :
  let r := 3
  let h_cone := 12
  let h_cylinder := 2
  let V_cone := 1/3 * Real.pi * r^2 * h_cone
  let V_hemisphere := 2/3 * Real.pi * r^3
  let V_cylinder := Real.pi * r^2 * h_cylinder
  V_cone + V_hemisphere + V_cylinder = 72 * Real.pi :=
by {
  let r := 3
  let h_cone := 12
  let h_cylinder := 2
  let V_cone := 1/3 * Real.pi * r^2 * h_cone
  let V_hemisphere := 2/3 * Real.pi * r^3
  let V_cylinder := Real.pi * r^2 * h_cylinder
  sorry
}

end ice_cream_total_volume_l254_254435


namespace high_temp_three_years_same_l254_254823

theorem high_temp_three_years_same
  (T : ℝ)                               -- The high temperature for the three years with the same temperature
  (temp2017 : ℝ := 79)                   -- The high temperature for 2017
  (temp2016 : ℝ := 71)                   -- The high temperature for 2016
  (average_temp : ℝ := 84)               -- The average high temperature for 5 years
  (num_years : ℕ := 5)                   -- The number of years to consider
  (years_with_same_temp : ℕ := 3)        -- The number of years with the same high temperature
  (total_temp : ℝ := average_temp * num_years) -- The sum of the high temperatures for the 5 years
  (total_known_temp : ℝ := temp2017 + temp2016) -- The known high temperatures for 2016 and 2017
  (total_for_three_years : ℝ := total_temp - total_known_temp) -- Total high temperatures for the three years
  (high_temp_per_year : ℝ := total_for_three_years / years_with_same_temp) -- High temperature per year for three years
  :
  T = 90 :=
sorry

end high_temp_three_years_same_l254_254823


namespace arctan_sum_pi_div_two_l254_254902

theorem arctan_sum_pi_div_two :
  let α := Real.arctan (3 / 4)
  let β := Real.arctan (4 / 3)
  α + β = Real.pi / 2 := by
  sorry

end arctan_sum_pi_div_two_l254_254902


namespace work_completion_together_l254_254881

theorem work_completion_together (man_days : ℕ) (son_days : ℕ) (together_days : ℕ) 
  (h_man : man_days = 10) (h_son : son_days = 10) : together_days = 5 :=
by sorry

end work_completion_together_l254_254881


namespace correct_system_of_equations_l254_254705

theorem correct_system_of_equations (x y : ℝ) :
  (5 * x + 6 * y = 16) ∧ (4 * x + y = x + 5 * y) :=
sorry

end correct_system_of_equations_l254_254705


namespace inequality_solution_empty_solution_set_l254_254588

-- Problem 1: Prove the inequality and the solution range
theorem inequality_solution (x : ℝ) : (-7 < x ∧ x < 3) ↔ ( (x - 3)/(x + 7) < 0 ) :=
sorry

-- Problem 2: Prove the conditions for empty solution set
theorem empty_solution_set (a : ℝ) : (a > 0) ↔ ∀ x : ℝ, ¬ (x^2 - 4*a*x + 4*a^2 + a ≤ 0) :=
sorry

end inequality_solution_empty_solution_set_l254_254588


namespace total_profit_is_8800_l254_254308

variable (A B C : Type) [CommRing A] [CommRing B] [CommRing C]

variable (investment_A investment_B investment_C : ℝ)
variable (total_profit : ℝ)

-- Conditions
def A_investment_three_times_B (investment_A investment_B : ℝ) : Prop :=
  investment_A = 3 * investment_B

def B_invest_two_thirds_C (investment_B investment_C : ℝ) : Prop :=
  investment_B = 2 / 3 * investment_C

def B_share_is_1600 (investment_B total_profit : ℝ) : Prop :=
  1600 = (2 / 11) * total_profit

theorem total_profit_is_8800 :
  A_investment_three_times_B investment_A investment_B →
  B_invest_two_thirds_C investment_B investment_C →
  B_share_is_1600 investment_B total_profit →
  total_profit = 8800 :=
by
  intros
  sorry

end total_profit_is_8800_l254_254308


namespace roots_n_not_divisible_by_5_for_any_n_l254_254258

theorem roots_n_not_divisible_by_5_for_any_n (x1 x2 : ℝ) (n : ℕ)
  (hx : x1^2 - 6 * x1 + 1 = 0)
  (hy : x2^2 - 6 * x2 + 1 = 0)
  : ¬(∃ (k : ℕ), (x1^k + x2^k) % 5 = 0) :=
sorry

end roots_n_not_divisible_by_5_for_any_n_l254_254258


namespace steve_annual_salary_l254_254100

variable (S : ℝ)

theorem steve_annual_salary :
  (0.70 * S - 800 = 27200) → (S = 40000) :=
by
  intro h
  sorry

end steve_annual_salary_l254_254100


namespace smallest_integer_n_condition_l254_254120

theorem smallest_integer_n_condition :
  (∃ n : ℕ, n > 0 ∧ (∀ (m : ℤ), (1 ≤ m ∧ m ≤ 1992) → (∃ (k : ℤ), (m : ℚ) / 1993 < k / n ∧ k / n < (m + 1 : ℚ) / 1994))) ↔ n = 3987 :=
sorry

end smallest_integer_n_condition_l254_254120


namespace num_factors_48_l254_254056

theorem num_factors_48 : 
  ∀ (n : ℕ), n = 48 → (∃ k : ℕ, k = 10 ∧ ∀ d : ℕ, d ∣ n → (d = 1 ∨ d = 2 ∨ d = 3 ∨ d = 4 ∨ d = 6 ∨ d = 8 ∨ d = 12 ∨ d = 16 ∨ d = 24 ∨ d = 48)) :=
  by
    intros n h
    sorry

end num_factors_48_l254_254056


namespace interval_of_x_l254_254169

theorem interval_of_x (x : ℝ) : 
  (2 < 4 * x ∧ 4 * x < 3) ∧ (2 < 5 * x ∧ 5 * x < 3) ↔ (1 / 2 < x ∧ x < 3 / 5) :=
by
  sorry

end interval_of_x_l254_254169


namespace jump_rope_total_l254_254078

theorem jump_rope_total :
  (56 * 3) + (35 * 4) = 308 :=
by
  sorry

end jump_rope_total_l254_254078


namespace negation_of_proposition_l254_254790

-- Definitions using the conditions stated
def p (x : ℝ) : Prop := x^2 - x + 1/4 ≥ 0

-- The statement to prove
theorem negation_of_proposition :
  (¬ (∀ x : ℝ, p x)) = (∃ x : ℝ, ¬ p x) :=
by
  -- Proof will go here; replaced by sorry as per instruction
  sorry

end negation_of_proposition_l254_254790


namespace product_mod_7_l254_254014

theorem product_mod_7 :
  (2009 % 7 = 4) ∧ (2010 % 7 = 5) ∧ (2011 % 7 = 6) ∧ (2012 % 7 = 0) →
  (2009 * 2010 * 2011 * 2012) % 7 = 0 :=
by
  sorry

end product_mod_7_l254_254014


namespace weight_of_3_moles_HBrO3_is_386_73_l254_254696

noncomputable def H_weight : ℝ := 1.01
noncomputable def Br_weight : ℝ := 79.90
noncomputable def O_weight : ℝ := 16.00
noncomputable def HBrO3_weight : ℝ := H_weight + Br_weight + 3 * O_weight
noncomputable def weight_of_3_moles_of_HBrO3 : ℝ := 3 * HBrO3_weight

theorem weight_of_3_moles_HBrO3_is_386_73 : weight_of_3_moles_of_HBrO3 = 386.73 := by
  sorry

end weight_of_3_moles_HBrO3_is_386_73_l254_254696


namespace num_people_in_group_l254_254665

-- Define constants and conditions
def cost_per_set : ℕ := 3  -- $3 to make 4 S'mores
def smores_per_set : ℕ := 4
def total_cost : ℕ := 18   -- $18 total cost
def smores_per_person : ℕ := 3

-- Calculate total S'mores that can be made
def total_sets : ℕ := total_cost / cost_per_set
def total_smores : ℕ := total_sets * smores_per_set

-- Proof problem statement
theorem num_people_in_group : (total_smores / smores_per_person) = 8 :=
by
  sorry

end num_people_in_group_l254_254665


namespace calculate_drift_l254_254875

def width_of_river : ℕ := 400
def speed_of_boat : ℕ := 10
def time_to_cross : ℕ := 50
def actual_distance_traveled := speed_of_boat * time_to_cross

theorem calculate_drift : actual_distance_traveled - width_of_river = 100 :=
by
  -- width_of_river = 400
  -- speed_of_boat = 10
  -- time_to_cross = 50
  -- actual_distance_traveled = 10 * 50 = 500
  -- expected drift = 500 - 400 = 100
  sorry

end calculate_drift_l254_254875


namespace balls_distribution_l254_254223

def balls_into_boxes : Nat := 6
def boxes : Nat := 3
def at_least_one_in_first (n m : Nat) : ℕ := sorry -- Use a function with appropriate constraints to ensure at least 1 ball is in the first box

theorem balls_distribution (n m : Nat) (h: n = 6) (h2: m = 3) :
  at_least_one_in_first n m = 665 :=
by
  sorry

end balls_distribution_l254_254223


namespace nonagon_diagonals_l254_254048

-- Define the number of sides of the polygon (nonagon)
def num_sides : ℕ := 9

-- Define the formula for the number of diagonals in a convex n-sided polygon
def number_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

-- State the theorem
theorem nonagon_diagonals : number_diagonals num_sides = 27 := 
by
--placeholder for the proof
sorry

end nonagon_diagonals_l254_254048


namespace min_value_xy_l254_254780

theorem min_value_xy (x y : ℕ) (h : 0 < x ∧ 0 < y) (cond : (1 : ℚ) / x + (1 : ℚ) /(3 * y) = 1 / 6) : 
  xy = 192 :=
sorry

end min_value_xy_l254_254780


namespace isosceles_triangle_perimeter_l254_254071

def is_isosceles (a b c : ℝ) : Prop :=
  (a = b ∨ b = c ∨ c = a)

theorem isosceles_triangle_perimeter 
  (a b c : ℝ) 
  (h_iso : is_isosceles a b c) 
  (h1 : a = 2 ∨ a = 4) 
  (h2 : b = 2 ∨ b = 4) 
  (h3 : c = 2 ∨ c = 4) :
  a + b + c = 10 :=
  sorry

end isosceles_triangle_perimeter_l254_254071


namespace prop_neg_or_not_l254_254364

theorem prop_neg_or_not (p q : Prop) (h : ¬(p ∨ ¬ q)) : ¬ p ∧ q :=
by
  sorry

end prop_neg_or_not_l254_254364


namespace group_size_l254_254370

def total_people (I N B Ne : ℕ) : ℕ := I + N - B + B + Ne

theorem group_size :
  let I := 55
  let N := 43
  let B := 61
  let Ne := 63
  total_people I N B Ne = 161 :=
by
  sorry

end group_size_l254_254370


namespace valid_triangles_pentadecagon_l254_254965

-- Definitions of the problem
def vertices : ℕ := 15

def total_triangles (n : ℕ) : ℕ := (finset.range (n).choose 3).card

def invalid_triangles (n : ℕ) : ℕ := n

def valid_triangles (n : ℕ) : ℕ := total_triangles n - invalid_triangles n

-- The theorem stating the number of valid triangles
theorem valid_triangles_pentadecagon : valid_triangles vertices = 440 :=
by sorry

end valid_triangles_pentadecagon_l254_254965


namespace marks_in_chemistry_l254_254735

-- Define the given conditions
def marks_english := 76
def marks_math := 65
def marks_physics := 82
def marks_biology := 85
def average_marks := 75
def number_subjects := 5

-- Define the theorem statement to prove David's marks in Chemistry
theorem marks_in_chemistry :
  let total_marks := marks_english + marks_math + marks_physics + marks_biology
  let total_marks_all_subjects := average_marks * number_subjects
  let marks_chemistry := total_marks_all_subjects - total_marks
  marks_chemistry = 67 :=
sorry

end marks_in_chemistry_l254_254735


namespace cos_nx_minus_sin_nx_eq_one_l254_254416

theorem cos_nx_minus_sin_nx_eq_one (n : ℕ) (x : ℝ) :
  (∃ k : ℤ, x = 2 * k * Real.pi) ∨ (∃ k : ℤ, n % 2 = 0 ∧ x = (2 * k + 1) * Real.pi) ↔ cos x ^ n - sin x ^ n = 1 :=
sorry

end cos_nx_minus_sin_nx_eq_one_l254_254416


namespace true_proposition_l254_254785

-- Define proposition p
def p : Prop := ∀ x : ℝ, Real.log (x^2 + 4) / Real.log 2 ≥ 2

-- Define proposition q
def q : Prop := ∀ x : ℝ, x ≥ 0 → x^(1/2) ≤ x^(1/2)

-- Theorem: true proposition is p ∨ ¬q
theorem true_proposition : p ∨ ¬q :=
by
  sorry

end true_proposition_l254_254785


namespace modulus_product_l254_254483

open Complex -- to open the complex namespace

-- Define the complex numbers
def z1 : ℂ := 10 - 5 * Complex.I
def z2 : ℂ := 7 + 24 * Complex.I

-- State the theorem to prove
theorem modulus_product : abs (z1 * z2) = 125 * Real.sqrt 5 := by
  sorry

end modulus_product_l254_254483


namespace find_angle_A_l254_254952

theorem find_angle_A (a b : ℝ) (A B : ℝ) 
  (ha : a = Real.sqrt 2) (hb : b = Real.sqrt 3) (hB : B = Real.pi / 3) :
  A = Real.pi / 4 :=
by
  -- This is a placeholder for the proof
  sorry

end find_angle_A_l254_254952


namespace find_duplicated_page_number_l254_254848

noncomputable def duplicated_page_number (n : ℕ) (incorrect_sum : ℕ) : ℕ :=
  incorrect_sum - n * (n + 1) / 2

theorem find_duplicated_page_number :
  ∃ n k, (1 <= k ∧ k <= n) ∧ ( ∃ n, (1 <= n) ∧ ( n * (n + 1) / 2 + k = 2550) )
  ∧ duplicated_page_number 70 2550 = 65 :=
by
  sorry

end find_duplicated_page_number_l254_254848


namespace recipe_butter_per_cup_l254_254000

theorem recipe_butter_per_cup (coconut_oil_to_butter_substitution : ℝ)
  (remaining_butter : ℝ)
  (planned_baking_mix : ℝ)
  (used_coconut_oil : ℝ)
  (butter_per_cup : ℝ)
  (h1 : coconut_oil_to_butter_substitution = 1)
  (h2 : remaining_butter = 4)
  (h3 : planned_baking_mix = 6)
  (h4 : used_coconut_oil = 8) :
  butter_per_cup = 4 / 3 := 
by 
  sorry

end recipe_butter_per_cup_l254_254000


namespace interval_of_x_l254_254182

theorem interval_of_x (x : ℝ) :
  (2 < 4 * x ∧ 4 * x < 3) ∧ (2 < 5 * x ∧ 5 * x < 3) ↔ (1 / 2 < x ∧ x < 3 / 5) := by
  sorry

end interval_of_x_l254_254182


namespace count_valid_arrangements_l254_254384

theorem count_valid_arrangements : 
  ∃ n : ℕ, (n = 5!) ∧
        (∃ z : ℕ, z = 4! ∧
        n = 120 ∧
        z = 24 ∧
        ∀ invalid_arrangements : ℕ, invalid_arrangements = 2 * z
        ∧ invalid_arrangements = 48
        ∧ (valid_arrangements = n - invalid_arrangements ∧ valid_arrangements = 72)) := 
sorry

end count_valid_arrangements_l254_254384


namespace limit_a_n_l254_254088

open Nat Real

noncomputable def a_n (n : ℕ) : ℝ := (7 * n - 1) / (n + 1)

theorem limit_a_n : ∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N, |a_n n - 7| < ε := 
by {
  -- The proof would go here.
  sorry
}

end limit_a_n_l254_254088


namespace measure_of_angle_Q_l254_254514

theorem measure_of_angle_Q (Q R : ℝ) 
  (h1 : Q = 2 * R)
  (h2 : 130 + 90 + 110 + 115 + Q + R = 540) :
  Q = 63.33 :=
by
  sorry

end measure_of_angle_Q_l254_254514


namespace kelsey_videos_watched_l254_254858

-- Definitions of conditions
def total_videos : ℕ := 411
def ekon_less : ℕ := 17
def kelsey_more : ℕ := 43

-- Variables representing videos watched by Uma, Ekon, and Kelsey
variables (U E K : ℕ)
hypothesis total_watched : U + E + K = total_videos
hypothesis ekon_watch : E = U - ekon_less
hypothesis kelsey_watch : K = E + kelsey_more

-- The statement to be proved
theorem kelsey_videos_watched : K = 160 :=
by sorry

end kelsey_videos_watched_l254_254858


namespace count_positive_bases_for_log_1024_l254_254506

-- Define the conditions 
def is_positive_integer_log_base (b n : ℕ) : Prop := b^n = 1024 ∧ n > 0

-- State that there are exactly 4 positive integers b that satisfy the condition
theorem count_positive_bases_for_log_1024 :
  (∃ b1 b2 b3 b4 : ℕ, b1 ≠ b2 ∧ b1 ≠ b3 ∧ b1 ≠ b4 ∧ b2 ≠ b3 ∧ b2 ≠ b4 ∧ b3 ≠ b4 ∧
    (∀ b, is_positive_integer_log_base b 1 ∨ is_positive_integer_log_base b 2 ∨ is_positive_integer_log_base b 5 ∨ is_positive_integer_log_base b 10) ∧
    (is_positive_integer_log_base b1 1 ∨ is_positive_integer_log_base b1 2 ∨ is_positive_integer_log_base b1 5 ∨ is_positive_integer_log_base b1 10) ∧
    (is_positive_integer_log_base b2 1 ∨ is_positive_integer_log_base b2 2 ∨ is_positive_integer_log_base b2 5 ∨ is_positive_integer_log_base b2 10) ∧
    (is_positive_integer_log_base b3 1 ∨ is_positive_integer_log_base b3 2 ∨ is_positive_integer_log_base b3 5 ∨ is_positive_integer_log_base b3 10) ∧
    (is_positive_integer_log_base b4 1 ∨ is_positive_integer_log_base b4 2 ∨ is_positive_integer_log_base b4 5 ∨ is_positive_integer_log_base b4 10)) :=
sorry

end count_positive_bases_for_log_1024_l254_254506


namespace perpendicular_bisector_of_circles_l254_254565

theorem perpendicular_bisector_of_circles
  (circle1 : ∀ x y : ℝ, x^2 + y^2 - 4 * x + 6 * y = 0)
  (circle2 : ∀ x y : ℝ, x^2 + y^2 - 6 * x = 0) :
  ∃ x y : ℝ, (3 * x - y - 9 = 0) :=
by
  sorry

end perpendicular_bisector_of_circles_l254_254565


namespace D_300_l254_254240

def D (n : ℕ) : ℕ :=
sorry

theorem D_300 : D 300 = 73 := 
by 
sorry

end D_300_l254_254240


namespace factors_of_48_l254_254052

theorem factors_of_48 : ∃ n, n = 48 → number_of_distinct_positive_factors n = 10 :=
sorry

-- Auxiliary function definitions to support the main theorem
def number_of_distinct_positive_factors (n : ℕ) : ℕ := 
sorry

end factors_of_48_l254_254052


namespace James_total_water_capacity_l254_254986

theorem James_total_water_capacity : 
  let cask_capacity := 20 -- capacity of a cask in gallons
  let barrel_capacity := 2 * cask_capacity + 3 -- capacity of a barrel in gallons
  let total_capacity := 4 * barrel_capacity + cask_capacity -- total water storage capacity
  total_capacity = 192 := by
    let cask_capacity := 20
    let barrel_capacity := 2 * cask_capacity + 3
    let total_capacity := 4 * barrel_capacity + cask_capacity
    have h : total_capacity = 192 := by sorry
    exact h

end James_total_water_capacity_l254_254986


namespace diff_of_cubes_is_sum_of_squares_l254_254546

theorem diff_of_cubes_is_sum_of_squares (n : ℤ) : 
  (n+2)^3 - n^3 = n^2 + (n+2)^2 + (2*n+2)^2 := 
by sorry

end diff_of_cubes_is_sum_of_squares_l254_254546


namespace part1_monotonicity_part2_inequality_l254_254789

noncomputable def f (a x : ℝ) : ℝ := a * (Real.exp x + a) - x

theorem part1_monotonicity (a : ℝ) :
  (∀ x : ℝ, a ≤ 0 → f a x < f a (x + 1)) ∧
  (a > 0 → ∀ x : ℝ, (x < Real.log (1 / a) → f a x > f a (x + 1)) ∧
  (x > Real.log (1 / a) → f a x < f a (x + 1))) := sorry

theorem part2_inequality (a : ℝ) (ha : a > 0) :
  ∀ x : ℝ, f a x > 2 * Real.log a + (3 / 2) := sorry

end part1_monotonicity_part2_inequality_l254_254789


namespace empty_vessel_percentage_l254_254154

theorem empty_vessel_percentage
  (P : ℝ) -- weight of the paint that completely fills the vessel
  (E : ℝ) -- weight of the empty vessel
  (h1 : 0.5 * (E + P) = E + 0.42857142857142855 * P)
  (h2 : 0.07142857142857145 * P = 0.5 * E):
  (E / (E + P) * 100) = 12.5 :=
by
  sorry

end empty_vessel_percentage_l254_254154


namespace sin_A_plus_B_eq_max_area_eq_l254_254802

-- Conditions for problem 1 and 2
variables (A B C a b c : ℝ)
variable (h_A_B_C : A + B + C = Real.pi)
variable (h_sin_C_div_2 : Real.sin (C / 2) = 2 * Real.sqrt 2 / 3)

noncomputable def sin_A_plus_B := Real.sin (A + B)

-- Problem 1: Prove that sin(A + B) = 4 * sqrt 2 / 9
theorem sin_A_plus_B_eq : sin_A_plus_B A B = 4 * Real.sqrt 2 / 9 :=
by sorry

-- Adding additional conditions for problem 2
variable (h_a_b_sum : a + b = 2 * Real.sqrt 2)

noncomputable def area (a b C : ℝ) := (1 / 2) * a * b * (2 * Real.sin (C / 2) * (Real.cos (C / 2)))

-- Problem 2: Prove that the maximum value of the area S of triangle ABC is 4 * sqrt 2 / 9
theorem max_area_eq : ∃ S, S = area a b C ∧ S ≤ 4 * Real.sqrt 2 / 9 :=
by sorry

end sin_A_plus_B_eq_max_area_eq_l254_254802


namespace smallest_positive_integer_remainder_l254_254124

theorem smallest_positive_integer_remainder
  (b : ℕ) (h1 : b % 4 = 3) (h2 : b % 6 = 5) :
  b = 11 := by
  sorry

end smallest_positive_integer_remainder_l254_254124


namespace smallest_of_powers_l254_254296

theorem smallest_of_powers :
  (2:ℤ)^(55) < (3:ℤ)^(44) ∧ (2:ℤ)^(55) < (5:ℤ)^(33) ∧ (2:ℤ)^(55) < (6:ℤ)^(22) := by
  sorry

end smallest_of_powers_l254_254296


namespace find_x_l254_254947

theorem find_x (x : ℚ) (h : ∀ (y : ℚ), 10 * x * y - 15 * y + 3 * x - 9 / 2 = 0) : x = 3 / 2 :=
sorry

end find_x_l254_254947


namespace intersection_A_B_l254_254635

def setA : Set ℤ := { x | x < -3 }
def setB : Set ℤ := {-5, -4, -3, 1}

theorem intersection_A_B : setA ∩ setB = {-5, -4} := by
  sorry

end intersection_A_B_l254_254635


namespace probability_equal_dice_show_numbers_5_dice_l254_254873

noncomputable def probability_equal_dice_show_numbers (n : ℕ) (k : ℕ) : ℚ :=
  (nat.choose n k) * (1/2)^n

theorem probability_equal_dice_show_numbers_5_dice :
  probability_equal_dice_show_numbers 5 2 + probability_equal_dice_show_numbers 5 3 = 5 / 8 := by
  sorry

end probability_equal_dice_show_numbers_5_dice_l254_254873


namespace exists_f_gcd_form_l254_254817

noncomputable def f : ℤ → ℕ := sorry

theorem exists_f_gcd_form :
  (∀ x y : ℤ, Nat.gcd (f x) (f y) = Nat.gcd (f x) (Int.natAbs (x - y))) →
  ∃ m n : ℕ, (0 < m ∧ 0 < n) ∧ (∀ x : ℤ, f x = Nat.gcd (m + Int.natAbs x) n) :=
sorry

end exists_f_gcd_form_l254_254817


namespace bus_ride_cost_l254_254461

variable (cost_bus cost_train : ℝ)

-- Condition 1: cost_train = cost_bus + 2.35
#check (cost_train = cost_bus + 2.35)

-- Condition 2: cost_bus + cost_train = 9.85
#check (cost_bus + cost_train = 9.85)

theorem bus_ride_cost :
  (∃ (cost_bus cost_train : ℝ),
    cost_train = cost_bus + 2.35 ∧
    cost_bus + cost_train = 9.85) →
  cost_bus = 3.75 :=
sorry

end bus_ride_cost_l254_254461


namespace interval_of_x_l254_254178

theorem interval_of_x (x : ℝ) :
  (2 < 4 * x ∧ 4 * x < 3) ∧ (2 < 5 * x ∧ 5 * x < 3) ↔ (1 / 2 < x ∧ x < 3 / 5) := by
  sorry

end interval_of_x_l254_254178


namespace triangle_third_side_range_l254_254201

variable (a b c : ℝ)

theorem triangle_third_side_range 
  (h₁ : |a + b - 4| + (a - b + 2)^2 = 0)
  (h₂ : a + b > c)
  (h₃ : a + c > b)
  (h₄ : b + c > a) : 2 < c ∧ c < 4 := 
sorry

end triangle_third_side_range_l254_254201


namespace polygon_sides_l254_254849

theorem polygon_sides (n : ℕ) (h : 3 * n * (n * (n - 3)) = 300) : n = 10 :=
sorry

end polygon_sides_l254_254849


namespace ratio_of_voters_l254_254979

theorem ratio_of_voters (V_X V_Y : ℝ) 
  (h1 : 0.62 * V_X + 0.38 * V_Y = 0.54 * (V_X + V_Y)) : V_X / V_Y = 2 :=
by
  sorry

end ratio_of_voters_l254_254979


namespace find_pairs_nat_numbers_l254_254027

theorem find_pairs_nat_numbers (a b : ℕ) :
  (a^3 * b - 1) % (a + 1) = 0 ∧ (a * b^3 + 1) % (b - 1) = 0 ↔ 
  (a = 2 ∧ b = 2) ∨ (a = 1 ∧ b = 3) ∨ (a = 3 ∧ b = 3) :=
by
  sorry

end find_pairs_nat_numbers_l254_254027


namespace kelsey_video_count_l254_254857

variable (E U K : ℕ)

noncomputable def total_videos : ℕ := 411
noncomputable def ekon_videos : ℕ := E
noncomputable def uma_videos : ℕ := E + 17
noncomputable def kelsey_videos : ℕ := E + 43

theorem kelsey_video_count (E U K : ℕ) 
  (h1 : total_videos = ekon_videos + uma_videos + kelsey_videos)
  (h2 : uma_videos = ekon_videos + 17)
  (h3 : kelsey_videos = ekon_videos + 43)
  : kelsey_videos = 160 := 
sorry

end kelsey_video_count_l254_254857


namespace sticker_distribution_unique_arrangements_l254_254047

theorem sticker_distribution_unique_arrangements :
  let stickers := 10
  let sheets := 5
  let colors := 2
  (Nat.choose (stickers + sheets - 1) stickers) * (colors ^ sheets) = 32032 :=
by
  sorry

end sticker_distribution_unique_arrangements_l254_254047


namespace no_real_solution_l254_254487

theorem no_real_solution (x : ℝ) : x + 64 / (x + 3) ≠ -13 :=
by {
  -- Proof is not required, so we mark it as sorry.
  sorry
}

end no_real_solution_l254_254487


namespace p_nonnegative_iff_equal_l254_254545

def p (a b c x : ℝ) : ℝ := (x - a) * (x - b) + (x - b) * (x - c) + (x - c) * (x - a)

theorem p_nonnegative_iff_equal (a b c : ℝ) : (∀ x : ℝ, p a b c x ≥ 0) ↔ a = b ∧ b = c :=
by
  sorry

end p_nonnegative_iff_equal_l254_254545


namespace line_up_ways_l254_254381

theorem line_up_ways (n : ℕ) (h : n = 5) :
  let categories := ((range n).filter (λ x, x ≠ 0 ∧ x ≠ (n - 1))) in
  categories.length * fact (n - 1) = 72 :=
by
  rw h
  let categories := ((range 5).filter (λ x, x ≠ 0 ∧ x ≠ (5 - 1)))
  have h_cat_len : categories.length = 3 := by decide
  rw [h_cat_len, fact]
  norm_num
  sorry

end line_up_ways_l254_254381


namespace final_price_percentage_l254_254300

theorem final_price_percentage (original_price sale_price final_price : ℝ) (h1 : sale_price = 0.9 * original_price) 
(h2 : final_price = sale_price - 0.1 * sale_price) : final_price / original_price = 0.81 :=
by
  sorry

end final_price_percentage_l254_254300


namespace mashas_end_number_is_17_smallest_starting_number_ends_with_09_l254_254251

def mashas_operation (n : ℕ) : ℕ :=
  let y := n % 10
  let x := n / 10
  3 * x + 2 * y

def mashas_stable_result (n : ℕ) : Prop :=
  mashas_operation n = n

theorem mashas_end_number_is_17 :
  ∃ n : ℕ, mashas_stable_result n ∧ n = 17 :=
sorry

def is_smallest_starting_number (n : ℕ) : Prop :=
  (nat.gcd n 17 = 17) ∧ (nat.log 10 n = 2014) ∧ (n % 100 = 9)

theorem smallest_starting_number_ends_with_09 :
  ∃ n : ℕ, is_smallest_starting_number n :=
sorry

end mashas_end_number_is_17_smallest_starting_number_ends_with_09_l254_254251


namespace mult_63_37_l254_254015

theorem mult_63_37 : 63 * 37 = 2331 :=
by {
  sorry
}

end mult_63_37_l254_254015


namespace polynomial_positive_for_all_reals_l254_254667

theorem polynomial_positive_for_all_reals (m : ℝ) : m^6 - m^5 + m^4 + m^2 - m + 1 > 0 :=
by
  sorry

end polynomial_positive_for_all_reals_l254_254667


namespace prime_divisors_6270_l254_254794

theorem prime_divisors_6270 : 
  ∃ (p1 p2 p3 p4 p5 : ℕ), 
  p1 = 2 ∧ p2 = 3 ∧ p3 = 5 ∧ p4 = 11 ∧ p5 = 19 ∧ 
  (p1 * p2 * p3 * p4 * p5 = 6270) ∧ 
  (Nat.Prime p1 ∧ Nat.Prime p2 ∧ Nat.Prime p3 ∧ Nat.Prime p4 ∧ Nat.Prime p5) ∧ 
  (∀ q, Nat.Prime q ∧ q ∣ 6270 → (q = p1 ∨ q = p2 ∨ q = p3 ∨ q = p4 ∨ q = p5)) := 
by 
  sorry

end prime_divisors_6270_l254_254794


namespace paint_cans_used_l254_254544

theorem paint_cans_used (initial_rooms : ℕ) (lost_cans : ℕ) (remaining_rooms : ℕ) 
    (h1 : initial_rooms = 50) (h2 : lost_cans = 5) (h3 : remaining_rooms = 40) : 
    (remaining_rooms / (initial_rooms - remaining_rooms) / lost_cans) = 20 :=
by
  sorry

end paint_cans_used_l254_254544


namespace reciprocal_sum_is_1_implies_at_least_one_is_2_l254_254770

-- Lean statement for the problem
theorem reciprocal_sum_is_1_implies_at_least_one_is_2 (a b c d : ℕ) (h_distinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d)
  (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) (h_pos_d : 0 < d) 
  (h_sum : (1 : ℚ) / a + (1 : ℚ) / b + (1 : ℚ) / c + (1 : ℚ) / d = 1) : 
  a = 2 ∨ b = 2 ∨ c = 2 ∨ d = 2 := 
sorry

end reciprocal_sum_is_1_implies_at_least_one_is_2_l254_254770


namespace find_x_and_C_l254_254504

def A (x : ℝ) : Set ℝ := {1, 3, x^2}
def B (x : ℝ) : Set ℝ := {1, 2 - x}

theorem find_x_and_C (x : ℝ) (C : Set ℝ) :
  B x ⊆ A x → B (-2) ∪ C = A (-2) → x = -2 ∧ C = {3} :=
by
  sorry

end find_x_and_C_l254_254504


namespace abs_eq_1_solution_set_l254_254232

theorem abs_eq_1_solution_set (x : ℝ) : (|x| + |x + 1| = 1) ↔ (x ∈ Set.Icc (-1 : ℝ) 0) := by
  sorry

end abs_eq_1_solution_set_l254_254232


namespace Beast_of_War_running_time_l254_254854

theorem Beast_of_War_running_time 
  (M : ℕ) 
  (AE : ℕ) 
  (BoWAC : ℕ)
  (h1 : M = 120)
  (h2 : AE = M - 30)
  (h3 : BoWAC = AE + 10) : 
  BoWAC = 100 
  := 
sorry

end Beast_of_War_running_time_l254_254854


namespace salary_percentage_l254_254691

theorem salary_percentage (m n : ℝ) (P : ℝ) (h1 : m + n = 572) (h2 : n = 260) (h3 : m = (P / 100) * n) : P = 120 := 
by
  sorry

end salary_percentage_l254_254691


namespace difference_in_cents_l254_254413

theorem difference_in_cents (pennies dimes : ℕ) (h : pennies + dimes = 5050) (hpennies : 1 ≤ pennies) (hdimes : 1 ≤ dimes) : 
  let total_value := pennies + 10 * dimes
  let max_value := 50500 - 9 * 1
  let min_value := 50500 - 9 * 5049
  max_value - min_value = 45432 := 
by 
  -- proof goes here
  sorry

end difference_in_cents_l254_254413


namespace binary_sum_l254_254008

-- Define the binary representations in terms of their base 10 equivalent.
def binary_111111111 := 511
def binary_1111111 := 127

-- State the proof problem.
theorem binary_sum : binary_111111111 + binary_1111111 = 638 :=
by {
  -- placeholder for proof
  sorry
}

end binary_sum_l254_254008


namespace sara_lunch_total_l254_254095

theorem sara_lunch_total :
  let hotdog := 5.36
  let salad := 5.10
  hotdog + salad = 10.46 :=
by
  let hotdog := 5.36
  let salad := 5.10
  sorry

end sara_lunch_total_l254_254095


namespace unit_prices_min_total_cost_l254_254333

-- Part (1): Proving the unit prices of ingredients A and B.
theorem unit_prices (x y : ℝ)
    (h₁ : x + y = 68)
    (h₂ : 5 * x + 3 * y = 280) :
    x = 38 ∧ y = 30 :=
by
  -- Sorry, proof not provided
  sorry

-- Part (2): Proving the minimum cost calculation.
theorem min_total_cost (m : ℝ)
    (h₁ : m + (36 - m) = 36)
    (h₂ : m ≥ 2 * (36 - m)) :
    (38 * m + 30 * (36 - m)) = 1272 :=
by
  -- Sorry, proof not provided
  sorry

end unit_prices_min_total_cost_l254_254333


namespace problem_statement_l254_254337

open Real

noncomputable def log4 (x : ℝ) : ℝ := log x / log 4

noncomputable def a : ℝ := log4 (sqrt 5)
noncomputable def b : ℝ := log 2 / log 5
noncomputable def c : ℝ := log4 5

theorem problem_statement : b < a ∧ a < c :=
by
  sorry

end problem_statement_l254_254337


namespace determine_range_of_a_l254_254365

theorem determine_range_of_a (a : ℝ) (h : ∃ x y : ℝ, x ≠ y ∧ a * x^2 - x + 2 = 0 ∧ a * y^2 - y + 2 = 0) : 
  a < 1 / 8 ∧ a ≠ 0 :=
sorry

end determine_range_of_a_l254_254365


namespace distances_equal_l254_254976

noncomputable def distance_from_point_to_line (x y m : ℝ) : ℝ :=
  |m * x + y + 3| / Real.sqrt (m^2 + 1)

theorem distances_equal (m : ℝ) :
  distance_from_point_to_line 3 2 m = distance_from_point_to_line (-1) 4 m ↔
  (m = 1 / 2 ∨ m = -6) := 
sorry

end distances_equal_l254_254976


namespace compatibility_condition_l254_254415

theorem compatibility_condition (a b c d x : ℝ) 
  (h1 : a * x + b = 0) (h2 : c * x + d = 0) : a * d - b * c = 0 :=
sorry

end compatibility_condition_l254_254415


namespace total_weight_kg_l254_254475

def envelope_weight_grams : ℝ := 8.5
def num_envelopes : ℝ := 800

theorem total_weight_kg : (envelope_weight_grams * num_envelopes) / 1000 = 6.8 :=
by
  sorry

end total_weight_kg_l254_254475


namespace discount_difference_l254_254006

theorem discount_difference (x : ℝ) (h1 : x = 8000) : 
  (x * 0.7) - ((x * 0.8) * 0.9) = 160 :=
by
  rw [h1]
  sorry

end discount_difference_l254_254006


namespace arctan_3_4_add_arctan_4_3_is_pi_div_2_l254_254917

noncomputable def arctan_add (a b : ℝ) : ℝ :=
  Real.arctan a + Real.arctan b

theorem arctan_3_4_add_arctan_4_3_is_pi_div_2 :
  arctan_add (3 / 4) (4 / 3) = Real.pi / 2 :=
sorry

end arctan_3_4_add_arctan_4_3_is_pi_div_2_l254_254917


namespace jessica_total_payment_l254_254397

-- Definitions based on the conditions
def basic_cable_cost : Nat := 15
def movie_channels_cost : Nat := 12
def sports_channels_cost : Nat := movie_channels_cost - 3

-- Definition of the total monthly payment given Jessica adds both movie and sports channels
def total_monthly_payment : Nat :=
  basic_cable_cost + (movie_channels_cost + sports_channels_cost)

-- The proof statement
theorem jessica_total_payment : total_monthly_payment = 36 :=
by
  -- skip the proof
  sorry

end jessica_total_payment_l254_254397


namespace prime_three_digit_integers_count_l254_254218

theorem prime_three_digit_integers_count :
  let primes := [2, 3, 5, 7]
  in (finset.card (finset.pi_finset (finset.singleton 1) (λ _, finset.inj_on primes _))) ^ 3 = 64 :=
by
  let primes := [2, 3, 5, 7]
  sorry

end prime_three_digit_integers_count_l254_254218


namespace phil_cards_left_l254_254824

-- Conditions
def cards_per_week : ℕ := 20
def weeks_per_year : ℕ := 52

-- Total number of cards in a year
def total_cards (cards_per_week weeks_per_year : ℕ) : ℕ := cards_per_week * weeks_per_year

-- Number of cards left after losing half in fire
def cards_left (total_cards : ℕ) : ℕ := total_cards / 2

-- Theorem to prove
theorem phil_cards_left (cards_per_week weeks_per_year : ℕ) :
  cards_left (total_cards cards_per_week weeks_per_year) = 520 :=
by
  sorry

end phil_cards_left_l254_254824


namespace brianna_remaining_money_l254_254896

variable (m c n : ℕ)

theorem brianna_remaining_money (h : (1 / 5 : ℝ) * m = (1 / 3 : ℝ) * n * c) : (m - n * c) / m = 2 / 5 :=
by
  have hnc : n * c = (3 / 5 : ℝ) * m := by
    rw ← mul_assoc
    rw ← (div_eq_mul_one_div _ _).symm
    rw h
    ring

  have h1 : (m - n * c) = m - (3 / 5 : ℝ) * m := by
    rw hnc

  have h2 : 1 = 5 / 5 := by norm_num

  have h3 : (5 / 5) * m = m := by rw [h2, mul_one]

  have h4 : (m - (3 / 5) * m) = (2 / 5) * m := by
    rw [← sub_mul, h3]
    norm_num

  rw div_eq_mul_inv
  rw ← h4
  norm_num

  sorry

end brianna_remaining_money_l254_254896


namespace abs_neg_three_eq_three_l254_254837

theorem abs_neg_three_eq_three : abs (-3) = 3 := 
by 
  sorry

end abs_neg_three_eq_three_l254_254837


namespace intersection_A_B_l254_254239

def A : Set ℕ := {1, 3, 5, 7, 9}
def B : Set ℕ := { x | 2 ≤ x ∧ x ≤ 5 }

theorem intersection_A_B : A ∩ B = {3, 5} :=
  sorry

end intersection_A_B_l254_254239


namespace distinct_constructions_l254_254470

def num_cube_constructions (white_cubes : Nat) (blue_cubes : Nat) : Nat :=
  if white_cubes = 5 ∧ blue_cubes = 3 then 5 else 0

theorem distinct_constructions : num_cube_constructions 5 3 = 5 :=
by
  sorry

end distinct_constructions_l254_254470


namespace total_pages_in_book_l254_254725

/-- Bill started reading a book on the first day of April. 
    He read 8 pages every day and by the 12th of April, he 
    had covered two-thirds of the book. Prove that the 
    total number of pages in the book is 144. --/
theorem total_pages_in_book 
  (pages_per_day : ℕ)
  (days_till_april_12 : ℕ)
  (total_pages_read : ℕ)
  (fraction_of_book_read : ℚ)
  (total_pages : ℕ)
  (h1 : pages_per_day = 8)
  (h2 : days_till_april_12 = 12)
  (h3 : total_pages_read = pages_per_day * days_till_april_12)
  (h4 : fraction_of_book_read = 2/3)
  (h5 : total_pages_read = (fraction_of_book_read * total_pages)) :
  total_pages = 144 := by
  sorry

end total_pages_in_book_l254_254725


namespace no_member_of_T_divisible_by_9_but_some_member_divisible_by_4_l254_254995

def sum_of_squares_of_four_consecutive_integers (n : ℤ) : ℤ :=
  (n - 2) ^ 2 + (n - 1) ^ 2 + n ^ 2 + (n + 1) ^ 2

def is_divisible_by (a b : ℤ) : Prop := b ≠ 0 ∧ a % b = 0

theorem no_member_of_T_divisible_by_9_but_some_member_divisible_by_4 :
  ¬ (∃ n : ℤ, is_divisible_by (sum_of_squares_of_four_consecutive_integers n) 9) ∧
  (∃ n : ℤ, is_divisible_by (sum_of_squares_of_four_consecutive_integers n) 4) :=
by 
  sorry

end no_member_of_T_divisible_by_9_but_some_member_divisible_by_4_l254_254995


namespace max_value_theorem_l254_254655

open Real

noncomputable def max_value (x y : ℝ) : ℝ :=
  x * y * (75 - 5 * x - 3 * y)

theorem max_value_theorem :
  ∃ x y : ℝ, x > 0 ∧ y > 0 ∧ 5 * x + 3 * y < 75 ∧ max_value x y = 3125 / 3 := by
  sorry

end max_value_theorem_l254_254655


namespace movie_theater_attendance_l254_254685

theorem movie_theater_attendance : 
  let total_seats := 750
  let empty_seats := 218
  let people := total_seats - empty_seats
  people = 532 :=
by
  sorry

end movie_theater_attendance_l254_254685


namespace range_of_m_l254_254347

theorem range_of_m (x y m : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : 2 / x + 1 / y = 1) (h4 : x + 2 * y > m^2 + 2 * m) :
  -4 < m ∧ m < 2 :=
sorry

end range_of_m_l254_254347


namespace trig_identity_solution_l254_254135

open Real

theorem trig_identity_solution :
  sin (15 * (π / 180)) * cos (45 * (π / 180)) + sin (105 * (π / 180)) * sin (135 * (π / 180)) = sqrt 3 / 2 :=
by
  -- Placeholder for the proof
  sorry

end trig_identity_solution_l254_254135


namespace multiple_of_second_lock_time_l254_254650

def first_lock_time := 5
def second_lock_time := 3 * first_lock_time - 3
def combined_lock_time := 60

theorem multiple_of_second_lock_time : combined_lock_time / second_lock_time = 5 := by
  -- Adding a proof placeholder using sorry
  sorry

end multiple_of_second_lock_time_l254_254650


namespace find_minimum_abs_sum_l254_254525

noncomputable def minimum_abs_sum (α β γ : ℝ) : ℝ :=
|α| + |β| + |γ|

theorem find_minimum_abs_sum :
  ∃ α β γ : ℝ, α + β + γ = 2 ∧ α * β * γ = 4 ∧
  minimum_abs_sum α β γ = 6 := by
  sorry

end find_minimum_abs_sum_l254_254525


namespace unique_solution_pair_l254_254318

theorem unique_solution_pair (x y : ℝ) :
  (4 * x ^ 2 + 6 * x + 4) * (4 * y ^ 2 - 12 * y + 25) = 28 →
  (x, y) = (-3 / 4, 3 / 2) := by
  intro h
  sorry

end unique_solution_pair_l254_254318


namespace degrees_to_radians_l254_254591

theorem degrees_to_radians (deg : ℝ) (rad : ℝ) (h1 : 1 = π / 180) (h2 : deg = 60) : rad = deg * (π / 180) :=
by
  sorry

end degrees_to_radians_l254_254591


namespace complex_expression_calculation_l254_254562

noncomputable def complex_i := Complex.I -- Define the imaginary unit i

theorem complex_expression_calculation : complex_i * (1 - complex_i)^2 = 2 := by
  sorry

end complex_expression_calculation_l254_254562


namespace smallest_positive_integer_remainder_conditions_l254_254121

theorem smallest_positive_integer_remainder_conditions :
  ∃ b : ℕ, (b % 4 = 3 ∧ b % 6 = 5) ∧ ∀ n : ℕ, (n % 4 = 3 ∧ n % 6 = 5) → n ≥ b := 
by
  have b := 23
  use b
  sorry

end smallest_positive_integer_remainder_conditions_l254_254121


namespace number_of_unique_outfits_l254_254557

-- Define the given conditions
def num_shirts : ℕ := 8
def num_ties : ℕ := 6
def special_shirt_ties : ℕ := 3
def remaining_shirts := num_shirts - 1
def remaining_ties := num_ties

-- Define the proof problem
theorem number_of_unique_outfits : num_shirts * num_ties - remaining_shirts * remaining_ties + special_shirt_ties = 45 :=
by
  sorry

end number_of_unique_outfits_l254_254557


namespace no_corner_cut_possible_l254_254393

-- Define the cube and the triangle sides
def cube_edge_length : ℝ := 15
def triangle_side1 : ℝ := 5
def triangle_side2 : ℝ := 6
def triangle_side3 : ℝ := 8

-- Main statement: Prove that it's not possible to cut off a corner of the cube to form the given triangle
theorem no_corner_cut_possible :
  ¬ (∃ (a b c : ℝ),
    a^2 + b^2 = triangle_side1^2 ∧
    b^2 + c^2 = triangle_side2^2 ∧
    c^2 + a^2 = triangle_side3^2 ∧
    a^2 + b^2 + c^2 = 62.5) :=
sorry

end no_corner_cut_possible_l254_254393


namespace lcm_inequality_l254_254479

open Nat

theorem lcm_inequality (n : ℕ) (h : n ≥ 5 ∧ Odd n) :
  Nat.lcm n (n + 1) * (n + 2) > Nat.lcm (n + 1) (n + 2) * (n + 3) := by
  sorry

end lcm_inequality_l254_254479


namespace total_wall_area_l254_254460

variable (L W : ℝ) -- Length and width of the regular tile
variable (R : ℕ) -- Number of regular tiles

-- Conditions:
-- 1. The area covered by regular tiles is 70 square feet.
axiom regular_tiles_cover_area : R * (L * W) = 70

-- 2. Jumbo tiles make up 1/3 of the total tiles, and each jumbo tile has an area three times that of a regular tile.
axiom length_ratio : ∀ jumbo_tiles, 3 * (jumbo_tiles * (L * W)) = 105

theorem total_wall_area (L W : ℝ) (R : ℕ) 
  (regular_tiles_cover_area : R * (L * W) = 70) 
  (length_ratio : ∀ jumbo_tiles, 3 * (jumbo_tiles * (L * W)) = 105) : 
  (R * (L * W)) + (3 * (R / 2) * (L * W)) = 175 :=
by
  sorry

end total_wall_area_l254_254460


namespace bruce_bank_savings_l254_254730

def aunt_gift : ℕ := 75
def grandfather_gift : ℕ := 150
def total_gift : ℕ := aunt_gift + grandfather_gift
def fraction_saved : ℚ := 1/5
def amount_saved : ℚ := total_gift * fraction_saved

theorem bruce_bank_savings : amount_saved = 45 := by
  sorry

end bruce_bank_savings_l254_254730


namespace circumference_of_circle_inscribing_rectangle_l254_254716

theorem circumference_of_circle_inscribing_rectangle (a b : ℝ) (h₁ : a = 9) (h₂ : b = 12) :
  ∃ C : ℝ, C = 15 * Real.pi := by
  sorry

end circumference_of_circle_inscribing_rectangle_l254_254716


namespace solve_for_x_l254_254549

-- Define the equation
def equation (x : ℝ) : Prop := (x^2 + 3 * x + 4) / (x + 5) = x + 6

-- Prove that x = -13 / 4 satisfies the equation
theorem solve_for_x : ∃ x : ℝ, equation x ∧ x = -13 / 4 :=
by
  sorry

end solve_for_x_l254_254549


namespace laptop_price_l254_254661

theorem laptop_price (x : ℝ) : 
  (0.8 * x - 120) = 0.9 * x - 64 → x = 560 :=
by
  sorry

end laptop_price_l254_254661


namespace zoe_total_cost_correct_l254_254458

theorem zoe_total_cost_correct :
  (6 * 0.5) + (6 * (1 + 2 * 0.75)) + (6 * 2 * 3) = 54 :=
by
  sorry

end zoe_total_cost_correct_l254_254458


namespace train_speed_kmph_l254_254306

theorem train_speed_kmph (length : ℝ) (time : ℝ) (speed_conversion : ℝ) (speed_kmph : ℝ) :
  length = 100.008 → time = 4 → speed_conversion = 3.6 →
  speed_kmph = (length / time) * speed_conversion → speed_kmph = 90.0072 :=
by
  sorry

end train_speed_kmph_l254_254306


namespace largest_n_l254_254018

theorem largest_n (x y z : ℕ) (hx : x > 0) (hy : y > 0) (hz : z > 0) : 
  ∃ n : ℕ, n > 0 ∧ n = 10 ∧ n^2 = x^2 + y^2 + z^2 + 2 * x * y + 2 * y * z + 2 * z * x + 5 * x + 5 * y + 5 * z - 12 := 
sorry

end largest_n_l254_254018


namespace total_cost_correct_l254_254806

def shirt_price : ℕ := 5
def hat_price : ℕ := 4
def jeans_price : ℕ := 10
def jacket_price : ℕ := 20
def shoes_price : ℕ := 15

def num_shirts : ℕ := 4
def num_jeans : ℕ := 3
def num_hats : ℕ := 4
def num_jackets : ℕ := 3
def num_shoes : ℕ := 2

def third_jacket_discount : ℕ := jacket_price / 2
def discount_per_two_shirts : ℕ := 2
def free_hat : ℕ := if num_jeans ≥ 3 then 1 else 0
def shoes_discount : ℕ := (num_shirts / 2) * discount_per_two_shirts

def total_cost : ℕ :=
  (num_shirts * shirt_price) +
  (num_jeans * jeans_price) +
  ((num_hats - free_hat) * hat_price) +
  ((num_jackets - 1) * jacket_price + third_jacket_discount) +
  (num_shoes * shoes_price - shoes_discount)

theorem total_cost_correct : total_cost = 138 := by
  sorry

end total_cost_correct_l254_254806


namespace impossible_sum_of_two_smaller_angles_l254_254978

theorem impossible_sum_of_two_smaller_angles
  {α β γ : ℝ}
  (h1 : α + β + γ = 180)
  (h2 : 0 < α + β ∧ α + β < 180) :
  α + β ≠ 130 :=
sorry

end impossible_sum_of_two_smaller_angles_l254_254978


namespace lines_intersect_at_same_point_l254_254608

theorem lines_intersect_at_same_point (m k : ℝ) :
  (∃ x y : ℝ, y = 3 * x + 5 ∧ y = -4 * x + m ∧ y = 2 * x + k) ↔ k = (m + 30) / 7 :=
by {
  sorry -- proof not required, only statement.
}

end lines_intersect_at_same_point_l254_254608


namespace min_n_Sn_l254_254623

/--
Given an arithmetic sequence {a_n}, let S_n denote the sum of its first n terms.
If S_4 = -2, S_5 = 0, and S_6 = 3, then the minimum value of n * S_n is -9.
-/
theorem min_n_Sn (a : ℕ → ℤ) (S : ℕ → ℤ)
  (h₁ : S 4 = -2)
  (h₂ : S 5 = 0)
  (h₃ : S 6 = 3)
  (h₄ : ∀ n, S n = (n * (2 * a 1 + (n - 1) * (a 2 - a 1))) / 2)
  : ∃ n : ℕ, n * S n = -9 := 
sorry

end min_n_Sn_l254_254623


namespace set_intersection_l254_254036

def A : Set ℝ := {1, 2, 3, 4, 5}
def B : Set ℝ := {x | x * (4 - x) < 0}
def C_R_B : Set ℝ := {x | 0 ≤ x ∧ x ≤ 4}

theorem set_intersection :
  A ∩ C_R_B = {1, 2, 3, 4} :=
by
  -- Proof goes here
  sorry

end set_intersection_l254_254036


namespace blake_spent_60_on_mangoes_l254_254729

def spent_on_oranges : ℕ := 40
def spent_on_apples : ℕ := 50
def initial_amount : ℕ := 300
def change : ℕ := 150
def total_spent := initial_amount - change
def total_spent_on_fruits := spent_on_oranges + spent_on_apples
def spending_on_mangoes := total_spent - total_spent_on_fruits

theorem blake_spent_60_on_mangoes : spending_on_mangoes = 60 := 
by
  -- The proof will go here
  sorry

end blake_spent_60_on_mangoes_l254_254729


namespace factor_tree_value_l254_254228

theorem factor_tree_value :
  let F := 7 * (2 * 2)
  let H := 11 * 2
  let G := 11 * H
  let X := F * G
  X = 6776 :=
by
  sorry

end factor_tree_value_l254_254228


namespace nonagon_diagonals_l254_254050

-- Define the number of sides for a nonagon.
def n : ℕ := 9

-- Define the formula for the number of diagonals in a polygon.
def D (n : ℕ) : ℕ := n * (n - 3) / 2

-- State the theorem to prove that the number of diagonals in a nonagon is 27.
theorem nonagon_diagonals : D n = 27 := by
  sorry

end nonagon_diagonals_l254_254050


namespace candy_bar_cost_l254_254894

def cost_soft_drink : ℕ := 2
def num_candy_bars : ℕ := 5
def total_spent : ℕ := 27
def cost_per_candy_bar (C : ℕ) : Prop := cost_soft_drink + num_candy_bars * C = total_spent

-- The theorem we want to prove
theorem candy_bar_cost (C : ℕ) (h : cost_per_candy_bar C) : C = 5 :=
by sorry

end candy_bar_cost_l254_254894


namespace construction_rates_construction_cost_l254_254688

-- Defining the conditions as Lean hypotheses

def length := 1650
def diff_rate := 30
def time_ratio := 3/2

-- Daily construction rates (questions answered as hypotheses as well)
def daily_rate_A := 60
def daily_rate_B := 90

-- Additional conditions for cost calculations
def cost_A_per_day := 90000
def cost_B_per_day := 120000
def total_days := 14
def alone_days_A := 5

-- Problem stated as proofs to be completed
theorem construction_rates :
  (∀ (x : ℕ), x = daily_rate_A ∧ (x + diff_rate) = daily_rate_B ∧ 
  (1650 / (x + diff_rate)) * (3/2) = (1650 / x) → 
  60 = daily_rate_A ∧ (60 + 30) = daily_rate_B ) :=
by sorry

theorem construction_cost :
  (∀ (m : ℕ), m = alone_days_A ∧ 
  (cost_A_per_day * total_days + cost_B_per_day * (total_days - alone_days_A)) / 1000 = 2340) :=
by sorry

end construction_rates_construction_cost_l254_254688


namespace base_conversion_min_sum_l254_254107

theorem base_conversion_min_sum (c d : ℕ) (h : 5 * c + 8 = 8 * d + 5) : c + d = 15 := by
  sorry

end base_conversion_min_sum_l254_254107


namespace square_difference_l254_254211

theorem square_difference (x : ℤ) (h : x^2 = 1444) : (x + 1) * (x - 1) = 1443 := 
by 
  sorry

end square_difference_l254_254211


namespace find_m_l254_254515

-- Definitions based on conditions
def is_isosceles_triangle (a b c : ℝ) : Prop :=
  a = b ∨ b = c ∨ c = a

def are_roots_of_quadratic (b c m : ℝ) : Prop :=
  b * c = 6 - m ∧ b + c = -(m + 2)

-- The theorem statement
theorem find_m {a b c m : ℝ} (h₁ : a = 5) (h₂ : is_isosceles_triangle a b c) (h₃ : are_roots_of_quadratic b c m) : m = -10 :=
sorry

end find_m_l254_254515


namespace phil_baseball_cards_left_l254_254826

-- Step a): Define the conditions
def packs_week := 20
def weeks_year := 52
def lost_factor := 1 / 2

-- Step c): Establish the theorem statement
theorem phil_baseball_cards_left : 
  (packs_week * weeks_year * (1 - lost_factor) = 520) := 
  by
    -- proof steps will come here
    sorry

end phil_baseball_cards_left_l254_254826


namespace numberOfCubesWithNoMoreThanFourNeighbors_l254_254771

def unitCubesWithAtMostFourNeighbors (a b c : ℕ) (h1 : a > 4) (h2 : b > 4) (h3 : c > 4) 
(h4 : (a - 2) * (b - 2) * (c - 2) = 836) : ℕ := 
  4 * (a - 2 + b - 2 + c - 2) + 8

theorem numberOfCubesWithNoMoreThanFourNeighbors (a b c : ℕ) 
(h1 : a > 4) (h2 : b > 4) (h3 : c > 4)
(h4 : (a - 2) * (b - 2) * (c - 2) = 836) :
  unitCubesWithAtMostFourNeighbors a b c h1 h2 h3 h4 = 144 :=
sorry

end numberOfCubesWithNoMoreThanFourNeighbors_l254_254771


namespace five_people_lineup_count_l254_254379

theorem five_people_lineup_count :
  let youngest := "Y"
  let people := ["A", "B", "C", "D", youngest]
  (people' : list string) (yield_positions : list string),
  (yield_positions.all_different ∧ youngest ∉ yield_positions.take 1 ++ yield_positions.drop 4) ∧ 
  yield_positions.permutations.count = 72 :=
by {
  let youngest := "Y"
  let people := ["A", "B", "C", "D", youngest]
  let valid_positions := [[a , b , c, d , youngest], [a, youngest , c , d , youngest], any_order]
  have h : valid_positions.length = 72,
  sorry
}

end five_people_lineup_count_l254_254379


namespace no_statement_implies_neg_p_or_q_l254_254480

def statement1 (p q : Prop) : Prop := p ∨ q
def statement2 (p q : Prop) : Prop := p ∨ ¬ q
def statement3 (p q : Prop) : Prop := ¬ p ∨ q
def statement4 (p q : Prop) : Prop := ¬ p ∧ q
def neg_p_or_q (p q : Prop) : Prop := ¬ (p ∨ q)

theorem no_statement_implies_neg_p_or_q (p q : Prop) :
  ¬ (statement1 p q → neg_p_or_q p q) ∧
  ¬ (statement2 p q → neg_p_or_q p q) ∧
  ¬ (statement3 p q → neg_p_or_q p q) ∧
  ¬ (statement4 p q → neg_p_or_q p q)
:= by
  sorry

end no_statement_implies_neg_p_or_q_l254_254480


namespace factor_expression_l254_254012

variable (x : ℝ)

theorem factor_expression :
  (18 * x ^ 6 + 50 * x ^ 4 - 8) - (2 * x ^ 6 - 6 * x ^ 4 - 8) = 8 * x ^ 4 * (2 * x ^ 2 + 7) :=
by
  sorry

end factor_expression_l254_254012


namespace distribution_problem_distribution_problem_variable_distribution_problem_equal_l254_254684

def books_distribution_fixed (n : ℕ) (a b c : ℕ) : ℕ :=
  if h : a + b + c = n then n.factorial / (a.factorial * b.factorial * c.factorial) else 0

theorem distribution_problem (n a b c : ℕ) (h : a + b + c = n) : 
  books_distribution_fixed n a b c = 1260 :=
sorry

def books_distribution_variable (n : ℕ) (a b c : ℕ) : ℕ :=
  if h : a + b + c = n then (n.factorial / (a.factorial * b.factorial * c.factorial)) * 6 else 0

theorem distribution_problem_variable (n a b c : ℕ) (h : a + b + c = n) : 
  books_distribution_variable n a b c = 7560 :=
sorry

def books_distribution_equal (n : ℕ) (k : ℕ) : ℕ :=
  if h : 3 * k = n then n.factorial / (k.factorial * k.factorial * k.factorial) else 0

theorem distribution_problem_equal (n k : ℕ) (h : 3 * k = n) : 
  books_distribution_equal n k = 1680 :=
sorry

end distribution_problem_distribution_problem_variable_distribution_problem_equal_l254_254684


namespace fraction_of_teeth_removed_l254_254769

theorem fraction_of_teeth_removed
  (total_teeth : ℕ)
  (initial_teeth : ℕ)
  (second_fraction : ℚ)
  (third_fraction : ℚ)
  (second_removed : ℕ)
  (third_removed : ℕ)
  (fourth_removed : ℕ)
  (total_removed : ℕ)
  (first_removed : ℕ)
  (fraction_first_removed : ℚ) :
  total_teeth = 32 →
  initial_teeth = 32 →
  second_fraction = 3 / 8 →
  third_fraction = 1 / 2 →
  second_removed = 12 →
  third_removed = 16 →
  fourth_removed = 4 →
  total_removed = 40 →
  first_removed + second_removed + third_removed + fourth_removed = total_removed →
  first_removed = 8 →
  fraction_first_removed = first_removed / initial_teeth →
  fraction_first_removed = 1 / 4 :=
by
  intros h1 h2 h3 h4 h5 h6 h7 h8 h9 h10 h11
  sorry

end fraction_of_teeth_removed_l254_254769


namespace gcd_8fact_11fact_9square_l254_254695

theorem gcd_8fact_11fact_9square : Nat.gcd (Nat.factorial 8) ((Nat.factorial 11) * 9^2) = 40320 := 
sorry

end gcd_8fact_11fact_9square_l254_254695


namespace mass_percentage_correct_l254_254028

noncomputable def mass_percentage_C_H_N_O_in_C20H25N3O 
  (m_C : ℚ) (m_H : ℚ) (m_N : ℚ) (m_O : ℚ) 
  (atoms_C : ℚ) (atoms_H : ℚ) (atoms_N : ℚ) (atoms_O : ℚ)
  (total_mass : ℚ)
  (percentage_C : ℚ) (percentage_H : ℚ) (percentage_N : ℚ) (percentage_O : ℚ) :=
  atoms_C = 20 ∧ atoms_H = 25 ∧ atoms_N = 3 ∧ atoms_O = 1 ∧ 
  m_C = 12.01 ∧ m_H = 1.008 ∧ m_N = 14.01 ∧ m_O = 16 ∧ 
  total_mass = (atoms_C * m_C) + (atoms_H * m_H) + (atoms_N * m_N) + (atoms_O * m_O) ∧ 
  percentage_C = (atoms_C * m_C / total_mass) * 100 ∧ 
  percentage_H = (atoms_H * m_H / total_mass) * 100 ∧ 
  percentage_N = (atoms_N * m_N / total_mass) * 100 ∧ 
  percentage_O = (atoms_O * m_O / total_mass) * 100 

theorem mass_percentage_correct : 
  mass_percentage_C_H_N_O_in_C20H25N3O 12.01 1.008 14.01 16 20 25 3 1 323.43 74.27 7.79 12.99 4.95 :=
by {
  sorry
}

end mass_percentage_correct_l254_254028


namespace problem_statement_l254_254680

noncomputable def given_function (x : ℝ) : ℝ := Real.sin (Real.pi / 2 - 2 * x)

def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f x = f (-x)

def smallest_positive_period (f : ℝ → ℝ) (T : ℝ) : Prop :=
  T > 0 ∧ ∀ x : ℝ, f (x + T) = f x

theorem problem_statement :
  is_even_function given_function ∧ smallest_positive_period given_function Real.pi :=
by
  sorry

end problem_statement_l254_254680


namespace sin_eq_sin_sinx_l254_254058

noncomputable def S (x : ℝ) := Real.sin x - x

theorem sin_eq_sin_sinx (x : ℝ) (h : 0 ≤ x ∧ x ≤ Real.arcsin 742) :
  ∃! x, Real.sin x = Real.sin (Real.sin x) :=
by
  sorry

end sin_eq_sin_sinx_l254_254058


namespace find_remainder_l254_254491

noncomputable def remainder_expr_division (β : ℂ) (hβ : β^4 + β^3 + β^2 + β + 1 = 0) : ℂ :=
  1 - β

theorem find_remainder (β : ℂ) (hβ : β^4 + β^3 + β^2 + β + 1 = 0) :
  ∃ r, (x^45 + x^34 + x^23 + x^12 + 1) % (x^4 + x^3 + x^2 + x + 1) = r ∧ r = remainder_expr_division β hβ :=
sorry

end find_remainder_l254_254491


namespace man_l254_254866

theorem man's_rate_in_still_water 
  (V_s V_m : ℝ)
  (with_stream : V_m + V_s = 24)  -- Condition 1
  (against_stream : V_m - V_s = 10) -- Condition 2
  : V_m = 17 := 
by
  sorry

end man_l254_254866


namespace mom_approach_is_sampling_survey_l254_254582

def is_sampling_survey (action : String) : Prop :=
  action = "tasting a little bit"

def is_census (action : String) : Prop :=
  action = "tasting the entire dish"

theorem mom_approach_is_sampling_survey :
  is_sampling_survey "tasting a little bit" :=
by {
  -- This follows from the given conditions directly.
  sorry
}

end mom_approach_is_sampling_survey_l254_254582


namespace term_300_is_neg_8_l254_254390

noncomputable def geom_seq (a r : ℤ) : ℕ → ℤ
| 0       => a
| (n + 1) => r * geom_seq a r n

-- First term and second term are given as conditions.
def a1 : ℤ := 8
def a2 : ℤ := -8

-- Define the common ratio based on the conditions
def r : ℤ := a2 / a1

-- Theorem stating the 300th term is -8
theorem term_300_is_neg_8 : geom_seq a1 r 299 = -8 :=
by
  have h_r : r = -1 := by
    rw [r, a2, a1]
    norm_num
  rw [h_r]
  sorry

end term_300_is_neg_8_l254_254390


namespace general_formula_minimum_n_exists_l254_254197

noncomputable def a_n (n : ℕ) : ℝ := 3 * (-2)^(n-1)
noncomputable def S_n (n : ℕ) : ℝ := 1 - (-2)^n

theorem general_formula (n : ℕ) : a_n n = 3 * (-2)^(n-1) :=
by sorry

theorem minimum_n_exists :
  (∃ n : ℕ, S_n n > 2016) ∧ (∀ m : ℕ, S_n m > 2016 → 11 ≤ m) :=
by sorry

end general_formula_minimum_n_exists_l254_254197


namespace intersection_A_B_l254_254361

def A (x : ℝ) : Prop := (2 * x - 1 > 0)
def B (x : ℝ) : Prop := (x * (x - 2) < 0)

theorem intersection_A_B :
  {x : ℝ | A x ∧ B x} = {x : ℝ | 1 / 2 < x ∧ x < 2} :=
by
  sorry

end intersection_A_B_l254_254361


namespace distinct_positive_factors_of_48_l254_254053

theorem distinct_positive_factors_of_48 : 
  let n := 48 in
  let factors := (2^4) * (3^1) in
  ∀ n : ℕ, n = factors → 
  (let num_factors := (4 + 1) * (1 + 1)
  in num_factors = 10) :=
by 
  let n := 48
  let factors := (2^4) * (3^1)
  assume h : n = factors
  let num_factors := (4 + 1) * (1 + 1)
  show num_factors = 10 from sorry

end distinct_positive_factors_of_48_l254_254053


namespace number_of_valid_permutations_l254_254376

theorem number_of_valid_permutations : 
  let n := 5 in 
  let total_permutations := n! in 
  let restricted_permutations := 2 * (n - 1)! in 
  total_permutations - restricted_permutations = 72 := 
by 
  sorry

end number_of_valid_permutations_l254_254376


namespace length_of_rectangle_l254_254800

theorem length_of_rectangle (P L B : ℕ) (h₁ : P = 800) (h₂ : B = 300) (h₃ : P = 2 * (L + B)) : L = 100 := by
  sorry

end length_of_rectangle_l254_254800


namespace no_real_solutions_l254_254222

theorem no_real_solutions :
  ∀ z : ℝ, ¬ ((-6 * z + 27) ^ 2 + 4 = -2 * |z|) :=
by
  sorry

end no_real_solutions_l254_254222


namespace limit_of_sequence_N_of_epsilon_l254_254091

theorem limit_of_sequence (a_n : ℕ → ℝ) (a : ℝ) (h : ∀ n, a_n n = (7 * n - 1) / (n + 1)) :
  (∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N, |a_n n - a| < ε) ↔ a = 7 := sorry

theorem N_of_epsilon (ε : ℝ) (hε : ε > 0) :
  ∃ N : ℕ, N = ⌈8 / ε⌉ := sorry

end limit_of_sequence_N_of_epsilon_l254_254091


namespace f_800_value_l254_254814

variable (f : ℝ → ℝ)
variable (f_prop : ∀ (x y : ℝ), 0 < x → 0 < y → f(x * y) = f(x) / y)
variable (f_400 : f 400 = 4)

theorem f_800_value : f 800 = 2 :=
by {
  sorry
}

end f_800_value_l254_254814


namespace number_of_ants_in_section_correct_l254_254474

noncomputable def ants_in_section := 
  let width_feet : ℝ := 600
  let length_feet : ℝ := 800
  let ants_per_square_inch : ℝ := 5
  let side_feet : ℝ := 200
  let feet_to_inches : ℝ := 12
  let side_inches := side_feet * feet_to_inches
  let area_section_square_inches := side_inches^2
  ants_per_square_inch * area_section_square_inches

theorem number_of_ants_in_section_correct :
  ants_in_section = 28800000 := 
by 
  unfold ants_in_section 
  sorry

end number_of_ants_in_section_correct_l254_254474


namespace limit_a_n_l254_254089

open Nat Real

noncomputable def a_n (n : ℕ) : ℝ := (7 * n - 1) / (n + 1)

theorem limit_a_n : ∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N, |a_n n - 7| < ε := 
by {
  -- The proof would go here.
  sorry
}

end limit_a_n_l254_254089


namespace least_values_3198_l254_254698

theorem least_values_3198 (x y : ℕ) (hX : ∃ n : ℕ, 3198 + n * 9 = 27)
                         (hY : ∃ m : ℕ, 3198 + m * 11 = 11) :
  x = 6 ∧ y = 8 :=
by
  sorry

end least_values_3198_l254_254698


namespace sheets_in_stack_l254_254003

theorem sheets_in_stack (h : 200 * t = 2.5) (h_pos : t > 0) : (5 / t) = 400 :=
by
  sorry

end sheets_in_stack_l254_254003


namespace average_age_of_all_l254_254420

theorem average_age_of_all (students parents : ℕ) (student_avg parent_avg : ℚ) 
  (h_students: students = 40) 
  (h_student_avg: student_avg = 12) 
  (h_parents: parents = 60) 
  (h_parent_avg: parent_avg = 36)
  : (students * student_avg + parents * parent_avg) / (students + parents) = 26.4 :=
by
  sorry

end average_age_of_all_l254_254420


namespace find_max_m_l254_254209

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := (1/2) * Real.exp (2 * x) - a * x

noncomputable def g (x : ℝ) (m : ℝ) : ℝ := (x - m) * f x 1 - (1/4) * Real.exp (2 * x) + x^2 + x

theorem find_max_m (h_inc : ∀ x > 0, g x m ≥ g x m) : m ≤ 1 :=
by
  sorry

end find_max_m_l254_254209


namespace repeating_decimals_sum_l254_254747

theorem repeating_decimals_sum : 
  (0.3333333333333333 : ℝ) + (0.0404040404040404 : ℝ) + (0.005005005005005 : ℝ) = (14 / 37 : ℝ) :=
by {
  sorry
}

end repeating_decimals_sum_l254_254747


namespace pens_exceed_500_on_saturday_l254_254082

theorem pens_exceed_500_on_saturday :
  ∃ k : ℕ, (5 * 3 ^ k > 500) ∧ k = 6 :=
by 
  sorry   -- Skipping the actual proof here

end pens_exceed_500_on_saturday_l254_254082


namespace number_of_SUVs_washed_l254_254520

theorem number_of_SUVs_washed (charge_car charge_truck charge_SUV total_raised : ℕ) (num_trucks num_cars S : ℕ) :
  charge_car = 5 →
  charge_truck = 6 →
  charge_SUV = 7 →
  total_raised = 100 →
  num_trucks = 5 →
  num_cars = 7 →
  total_raised = num_cars * charge_car + num_trucks * charge_truck + S * charge_SUV →
  S = 5 :=
by
  intros h1 h2 h3 h4 h5 h6 h7
  sorry

end number_of_SUVs_washed_l254_254520


namespace Katya_possible_numbers_l254_254703

def divisible_by (n m : ℕ) : Prop := m % n = 0

def possible_numbers (n : ℕ) : Prop :=
  let condition1 := divisible_by 7 n  -- Alyona's condition
  let condition2 := divisible_by 5 n  -- Lena's condition
  let condition3 := n < 9             -- Rita's condition
  (condition1 ∨ condition2) ∧ condition3 ∧ 
  ((condition1 ∧ condition3 ∧ ¬condition2) ∨ (condition2 ∧ condition3 ∧ ¬condition1))

theorem Katya_possible_numbers :
  ∀ n : ℕ, 
    (possible_numbers n) ↔ (n = 5 ∨ n = 7) :=
sorry

end Katya_possible_numbers_l254_254703


namespace sequence_infinite_coprime_l254_254523

theorem sequence_infinite_coprime (a : ℤ) (h : a > 1) :
  ∃ (S : ℕ → ℕ), (∀ n m : ℕ, n ≠ m → Int.gcd (a^(S n + 1) + a^S n - 1) (a^(S m + 1) + a^S m - 1) = 1) :=
sorry

end sequence_infinite_coprime_l254_254523


namespace arctan_sum_pi_div_two_l254_254910

noncomputable def arctan_sum : Real :=
  Real.arctan (3 / 4) + Real.arctan (4 / 3)

theorem arctan_sum_pi_div_two : arctan_sum = Real.pi / 2 := 
by sorry

end arctan_sum_pi_div_two_l254_254910


namespace fish_lifespan_l254_254115

theorem fish_lifespan (H : ℝ) (D : ℝ) (F : ℝ) 
  (h_hamster : H = 2.5)
  (h_dog : D = 4 * H)
  (h_fish : F = D + 2) : 
  F = 12 :=
by
  rw [h_hamster, h_dog] at h_fish
  simp at h_fish
  exact h_fish

end fish_lifespan_l254_254115


namespace range_exp3_eq_l254_254850

noncomputable def exp3 (x : ℝ) : ℝ := 3^x

theorem range_exp3_eq (x : ℝ) : Set.range (exp3) = Set.Ioi 0 :=
sorry

end range_exp3_eq_l254_254850


namespace sum_of_arithmetic_sequence_l254_254818

def f (x : ℝ) : ℝ := (x - 3)^3 + x - 1

def is_arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + d

theorem sum_of_arithmetic_sequence (a : ℕ → ℝ) (d : ℝ)
  (h_arith : is_arithmetic_sequence a d)
  (h_nonzero : d ≠ 0)
  (h_sum_f : f (a 1) + f (a 2) + f (a 3) + f (a 4) + f (a 5) + f (a 6) + f (a 7) = 14) :
  a 1 + a 2 + a 3 + a 4 + a 5 + a 6 + a 7 = 21 :=
by 
  sorry

end sum_of_arithmetic_sequence_l254_254818


namespace initial_goldfish_eq_15_l254_254086

-- Let's define our setup as per the conditions provided
def fourGoldfishLeft := 4
def elevenGoldfishDisappeared := 11

-- Our main statement that we need to prove
theorem initial_goldfish_eq_15 : fourGoldfishLeft + elevenGoldfishDisappeared = 15 := by
  sorry

end initial_goldfish_eq_15_l254_254086


namespace third_offense_fraction_l254_254335

-- Define the conditions
def sentence_assault : ℕ := 3
def sentence_poisoning : ℕ := 24
def total_sentence : ℕ := 36

-- The main theorem to prove
theorem third_offense_fraction :
  (total_sentence - (sentence_assault + sentence_poisoning)) / (sentence_assault + sentence_poisoning) = 1 / 3 := by
  sorry

end third_offense_fraction_l254_254335


namespace smallest_possible_value_l254_254867

theorem smallest_possible_value (a b c d : ℤ) 
  (h1 : a + b + c + d < 25) 
  (h2 : a > 8) 
  (h3 : b < 5) 
  (h4 : c % 2 = 1) 
  (h5 : d % 2 = 0) : 
  ∃ a' b' c' d' : ℤ, a' > 8 ∧ b' < 5 ∧ c' % 2 = 1 ∧ d' % 2 = 0 ∧ a' + b' + c' + d' < 25 ∧ (a' - b' + c' - d' = -4) := 
by 
  use 9, 4, 1, 10
  sorry

end smallest_possible_value_l254_254867


namespace arctan_triangle_complementary_l254_254906

theorem arctan_triangle_complementary :
  (Real.arctan (3 / 4) + Real.arctan (4 / 3) = Real.pi / 2) :=
begin
  sorry
end

end arctan_triangle_complementary_l254_254906


namespace solve_fra_eq_l254_254556

theorem solve_fra_eq : ∀ x : ℝ, (x - 2) / (x + 2) + 4 / (x^2 - 4) = 1 → x = 3 :=
by 
  -- Proof steps go here
  sorry

end solve_fra_eq_l254_254556


namespace total_jelly_beans_l254_254530

-- Given conditions:
def vanilla_jelly_beans : ℕ := 120
def grape_jelly_beans := 5 * vanilla_jelly_beans + 50

-- The proof problem:
theorem total_jelly_beans :
  vanilla_jelly_beans + grape_jelly_beans = 770 :=
by
  -- Proof steps would go here
  sorry

end total_jelly_beans_l254_254530


namespace distance_halfway_along_orbit_l254_254108

variable {Zeta : Type}  -- Zeta is a type representing the planet
variable (distance_from_focus : Zeta → ℝ)  -- Function representing the distance from the sun (focus)

-- Conditions
variable (perigee_distance : ℝ := 3)
variable (apogee_distance : ℝ := 15)
variable (a : ℝ := (perigee_distance + apogee_distance) / 2)  -- semi-major axis

theorem distance_halfway_along_orbit (z : Zeta) (h1 : distance_from_focus z = perigee_distance) (h2 : distance_from_focus z = apogee_distance) :
  distance_from_focus z = a :=
sorry

end distance_halfway_along_orbit_l254_254108


namespace cedar_vs_pine_height_cedar_vs_birch_height_l254_254063

-- Define the heights as rational numbers
def pine_tree_height := 14 + 1/4
def birch_tree_height := 18 + 1/2
def cedar_tree_height := 20 + 5/8

-- Theorem to prove the height differences
theorem cedar_vs_pine_height :
  cedar_tree_height - pine_tree_height = 6 + 3/8 :=
by
  sorry

theorem cedar_vs_birch_height :
  cedar_tree_height - birch_tree_height = 2 + 1/8 :=
by
  sorry

end cedar_vs_pine_height_cedar_vs_birch_height_l254_254063


namespace geometric_sequence_k_value_l254_254342

theorem geometric_sequence_k_value (S : ℕ → ℝ) (a : ℕ → ℝ) (k : ℝ)
  (hS : ∀ n, S n = k + 3^n)
  (h_geom : ∀ n, a (n+1) = S (n+1) - S n)
  (h_geo_seq : ∀ n, a (n+2) / a (n+1) = a (n+1) / a n) :
  k = -1 := by
  sorry

end geometric_sequence_k_value_l254_254342


namespace smallest_term_abs_l254_254343

noncomputable def arithmetic_sequence (a : ℕ → ℝ) :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem smallest_term_abs {a : ℕ → ℝ}
  (h_arith : arithmetic_sequence a)
  (h1 : a 1 > 0)
  (hS12 : (12 / 2) * (2 * a 1 + 11 * (a 2 - a 1)) > 0)
  (hS13 : (13 / 2) * (2 * a 1 + 12 * (a 2 - a 1)) < 0) :
  ∀ (n : ℕ), 1 ≤ n ∧ n ≤ 13 → n ≠ 7 → abs (a 6) > abs (a 1 + 6 * (a 2 - a 1)) :=
sorry

end smallest_term_abs_l254_254343


namespace perimeter_not_55_l254_254284

def is_valid_perimeter (a b p : ℕ) : Prop :=
  ∃ x : ℕ, a + b > x ∧ a + x > b ∧ b + x > a ∧ p = a + b + x

theorem perimeter_not_55 (a b : ℕ) (h1 : a = 18) (h2 : b = 10) : ¬ is_valid_perimeter a b 55 :=
by
  rw [h1, h2]
  sorry

end perimeter_not_55_l254_254284


namespace sum_of_digits_B_l254_254353

noncomputable def digit_sum (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem sum_of_digits_B (n : ℕ) (h : n = 4444^4444) : digit_sum (digit_sum (digit_sum n)) = 7 :=
by
  sorry

end sum_of_digits_B_l254_254353


namespace probability_of_Z_l254_254307

/-
  Given: 
  - P(W) = 3 / 8
  - P(X) = 1 / 4
  - P(Y) = 1 / 8

  Prove: 
  - P(Z) = 1 / 4 when P(Z) = 1 - (P(W) + P(X) + P(Y))
-/

theorem probability_of_Z (P_W P_X P_Y P_Z : ℚ) (h_W : P_W = 3 / 8) (h_X : P_X = 1 / 4) (h_Y : P_Y = 1 / 8) (h_Z : P_Z = 1 - (P_W + P_X + P_Y)) : 
  P_Z = 1 / 4 :=
by
  -- We can write the whole Lean Math proof here. However, per the instructions, we'll conclude with sorry.
  sorry

end probability_of_Z_l254_254307


namespace circumscribed_circle_diameter_l254_254974

theorem circumscribed_circle_diameter (a : ℝ) (A : ℝ) (h_a : a = 16) (h_A : A = 30) :
    let D := a / Real.sin (A * Real.pi / 180)
    D = 32 := by
  sorry

end circumscribed_circle_diameter_l254_254974


namespace quadratic_root_form_eq_l254_254931

theorem quadratic_root_form_eq (c : ℚ) : 
  (∀ x : ℚ, x^2 - 7 * x + c = 0 → x = (7 + Real.sqrt (9 * c)) / 2 ∨ x = (7 - Real.sqrt (9 * c)) / 2) →
  c = 49 / 13 := 
by
  sorry

end quadratic_root_form_eq_l254_254931


namespace original_price_of_color_TV_l254_254305

theorem original_price_of_color_TV
  (x : ℝ)  -- Let the variable x represent the original price
  (h1 : x * 1.4 * 0.8 - x = 144)  -- Condition as equation
  : x = 1200 := 
sorry  -- Proof to be filled in later

end original_price_of_color_TV_l254_254305


namespace smallest_four_digit_multiple_of_primes_l254_254189

theorem smallest_four_digit_multiple_of_primes : 
  let lcm_3_5_7_11 := Nat.lcm (Nat.lcm (Nat.lcm 3 5) 7) 11 in 
  ∀ n, 1000 <= n * lcm_3_5_7_11 → 1155 <= n * lcm_3_5_7_11 :=
by
  sorry

end smallest_four_digit_multiple_of_primes_l254_254189


namespace azure_valley_skirts_l254_254548

variables (P S A : ℕ)

theorem azure_valley_skirts (h1 : P = 10) 
                           (h2 : P = S / 4) 
                           (h3 : S = 2 * A / 3) : 
  A = 60 :=
by sorry

end azure_valley_skirts_l254_254548


namespace sum_S17_l254_254775

-- Definitions of the required arithmetic sequence elements.
variable (a1 d : ℤ)

-- Definition of the arithmetic sequence
def aₙ (n : ℤ) : ℤ := a1 + (n - 1) * d
def Sₙ (n : ℤ) : ℤ := n * a1 + (n * (n - 1) / 2) * d

-- Theorem for the problem statement
theorem sum_S17 : (aₙ a1 d 7 + aₙ a1 d 5) = (3 + aₙ a1 d 5) → (a1 + 8 * d = 3) → Sₙ a1 d 17 = 51 :=
by
  intros h1 h2
  sorry

end sum_S17_l254_254775


namespace count_valid_N_l254_254672

theorem count_valid_N : ∃ (count : ℕ), count = 10 ∧ 
    (∀ N : ℕ, (10 ≤ N ∧ N < 100) → 
        (∃ a b c d : ℕ, 
            a < 3 ∧ b < 3 ∧ c < 3 ∧ d < 4 ∧
            N = 3 * a + b ∧ N = 4 * c + d ∧
            2 * N % 50 = ((9 * a + b) + (8 * c + d)) % 50)) :=
sorry

end count_valid_N_l254_254672


namespace interval_of_x_l254_254185

theorem interval_of_x (x : ℝ) : 
  (2 < 4 * x ∧ 4 * x < 3) → (2 < 5 * x ∧ 5 * x < 3) → (1 / 2 < x ∧ x < 3 / 5) :=
by
  sorry

end interval_of_x_l254_254185


namespace limit_of_sequence_N_of_epsilon_l254_254090

theorem limit_of_sequence (a_n : ℕ → ℝ) (a : ℝ) (h : ∀ n, a_n n = (7 * n - 1) / (n + 1)) :
  (∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N, |a_n n - a| < ε) ↔ a = 7 := sorry

theorem N_of_epsilon (ε : ℝ) (hε : ε > 0) :
  ∃ N : ℕ, N = ⌈8 / ε⌉ := sorry

end limit_of_sequence_N_of_epsilon_l254_254090


namespace quarters_initial_l254_254671

-- Define the given conditions
def candies_cost_dimes : Nat := 4 * 3
def candies_cost_cents : Nat := candies_cost_dimes * 10
def lollipop_cost_quarters : Nat := 1
def lollipop_cost_cents : Nat := lollipop_cost_quarters * 25
def total_spent_cents : Nat := candies_cost_cents + lollipop_cost_cents
def money_left_cents : Nat := 195
def total_initial_money_cents : Nat := money_left_cents + total_spent_cents
def dimes_count : Nat := 19
def dimes_value_cents : Nat := dimes_count * 10

-- Prove that the number of quarters initially is 6
theorem quarters_initial (quarters_count : Nat) (h : quarters_count * 25 = total_initial_money_cents - dimes_value_cents) : quarters_count = 6 :=
by
  sorry

end quarters_initial_l254_254671


namespace annual_cost_l254_254573

def monday_miles : ℕ := 50
def wednesday_miles : ℕ := 50
def friday_miles : ℕ := 50
def sunday_miles : ℕ := 50

def tuesday_miles : ℕ := 100
def thursday_miles : ℕ := 100
def saturday_miles : ℕ := 100

def cost_per_mile : ℝ := 0.1
def weekly_fee : ℝ := 100
def weeks_in_year : ℕ := 52

noncomputable def total_weekly_miles : ℕ := 
  (monday_miles + wednesday_miles + friday_miles + sunday_miles) * 1 +
  (tuesday_miles + thursday_miles + saturday_miles) * 1

noncomputable def weekly_mileage_cost : ℝ := total_weekly_miles * cost_per_mile

noncomputable def weekly_total_cost : ℝ := weekly_fee + weekly_mileage_cost

noncomputable def annual_total_cost : ℝ := weekly_total_cost * weeks_in_year

theorem annual_cost (monday_miles wednesday_miles friday_miles sunday_miles
                     tuesday_miles thursday_miles saturday_miles : ℕ)
                     (cost_per_mile weekly_fee : ℝ) 
                     (weeks_in_year : ℕ) :
  monday_miles = 50 → wednesday_miles = 50 → friday_miles = 50 → sunday_miles = 50 →
  tuesday_miles = 100 → thursday_miles = 100 → saturday_miles = 100 →
  cost_per_mile = 0.1 → weekly_fee = 100 → weeks_in_year = 52 →
  annual_total_cost = 7800 :=
by
  intros
  sorry

end annual_cost_l254_254573


namespace number_of_three_digit_prime_integers_l254_254219

def prime_digits : Set Nat := {2, 3, 5, 7}

theorem number_of_three_digit_prime_integers : 
  (∃ count, count = 4 * 4 * 4 ∧ count = 64) :=
by
  sorry

end number_of_three_digit_prime_integers_l254_254219


namespace cody_discount_l254_254315

theorem cody_discount (initial_cost tax_rate cody_paid total_paid price_before_discount discount: ℝ) 
  (h1 : initial_cost = 40)
  (h2 : tax_rate = 0.05)
  (h3 : cody_paid = 17)
  (h4 : total_paid = 2 * cody_paid)
  (h5 : price_before_discount = initial_cost * (1 + tax_rate))
  (h6 : discount = price_before_discount - total_paid) :
  discount = 8 := by
  sorry

end cody_discount_l254_254315


namespace min_value_x_plus_reciprocal_min_value_x_plus_reciprocal_equality_at_one_l254_254797

theorem min_value_x_plus_reciprocal (x : ℝ) (h : x > 0) : x + 1 / x ≥ 2 :=
by
  sorry

theorem min_value_x_plus_reciprocal_equality_at_one : (1 : ℝ) + 1 / 1 = 2 :=
by
  norm_num

end min_value_x_plus_reciprocal_min_value_x_plus_reciprocal_equality_at_one_l254_254797


namespace inequality_holds_for_triangle_sides_l254_254614

theorem inequality_holds_for_triangle_sides (a : ℝ) : 
  (∀ (x y z : ℕ), x + y > z ∧ y + z > x ∧ z + x > y → (x^2 + y^2 + z^2 ≤ a * (x * y + y * z + z * x))) ↔ (1 ≤ a ∧ a ≤ 6 / 5) :=
by sorry

end inequality_holds_for_triangle_sides_l254_254614


namespace Masc_age_difference_l254_254250

theorem Masc_age_difference (masc_age sam_age : ℕ) (h1 : masc_age + sam_age = 27) (h2 : masc_age = 17) (h3 : sam_age = 10) : masc_age - sam_age = 7 :=
by {
  -- Proof would go here, but it's omitted as per instructions
  sorry
}

end Masc_age_difference_l254_254250


namespace max_dominoes_in_grid_l254_254226

-- Definitions representing the conditions
def total_squares (rows cols : ℕ) : ℕ := rows * cols
def domino_squares : ℕ := 3
def max_dominoes (total domino : ℕ) : ℕ := total / domino

-- Statement of the problem
theorem max_dominoes_in_grid : max_dominoes (total_squares 20 19) domino_squares = 126 :=
by
  -- placeholders for the actual proof
  sorry

end max_dominoes_in_grid_l254_254226


namespace min_val_f_l254_254620

noncomputable def f (x : ℝ) : ℝ :=
  4 / (x - 2) + x

theorem min_val_f (x : ℝ) (h : x > 2) : ∃ y, y = f x ∧ y ≥ 6 :=
by {
  sorry
}

end min_val_f_l254_254620


namespace factorial_floor_expression_l254_254733

open Nat

theorem factorial_floor_expression :
  ⇑floor ((2012! + 2008!) / (2011! + 2010!)) = 1 :=
sorry

end factorial_floor_expression_l254_254733


namespace art_collection_total_cost_l254_254648

theorem art_collection_total_cost 
  (price_first_three : ℕ)
  (price_fourth : ℕ)
  (total_first_three : price_first_three * 3 = 45000)
  (price_fourth_cond : price_fourth = price_first_three + (price_first_three / 2)) :
  3 * price_first_three + price_fourth = 67500 :=
by
  sorry

end art_collection_total_cost_l254_254648


namespace factorize_expression_l254_254485

variable (x y : ℝ)

theorem factorize_expression : 
  (y - 2 * x * y + x^2 * y) = y * (1 - x)^2 := 
by
  sorry

end factorize_expression_l254_254485


namespace symmetry_graph_l254_254960

theorem symmetry_graph (θ:ℝ) (hθ: θ > 0):
  (∀ k: ℤ, 2 * (3 * Real.pi / 4) + (Real.pi / 3) - 2 * θ = k * Real.pi + Real.pi / 2) 
  → θ = Real.pi / 6 :=
by 
  sorry

end symmetry_graph_l254_254960


namespace pages_called_this_week_l254_254810

-- Definitions as per conditions
def pages_called_last_week := 10.2
def total_pages_called := 18.8

-- Theorem to prove the solution
theorem pages_called_this_week :
  total_pages_called - pages_called_last_week = 8.6 :=
by
  sorry

end pages_called_this_week_l254_254810


namespace cost_of_each_gumdrop_l254_254059

theorem cost_of_each_gumdrop (cents : ℕ) (gumdrops : ℕ) (cost_per_gumdrop : ℕ) : 
  cents = 224 → gumdrops = 28 → cost_per_gumdrop = cents / gumdrops → cost_per_gumdrop = 8 :=
by
  intros h_cents h_gumdrops h_cost
  sorry

end cost_of_each_gumdrop_l254_254059


namespace gcd_n_four_plus_sixteen_and_n_plus_three_l254_254767

theorem gcd_n_four_plus_sixteen_and_n_plus_three (n : ℕ) (hn1 : n > 9) (hn2 : n ≠ 94) :
  Nat.gcd (n^4 + 16) (n + 3) = 1 :=
by
  sorry

end gcd_n_four_plus_sixteen_and_n_plus_three_l254_254767


namespace pie_not_crust_percentage_l254_254150

theorem pie_not_crust_percentage (total_weight crust_weight : ℝ) 
  (h1 : total_weight = 200) (h2 : crust_weight = 50) : 
  (total_weight - crust_weight) / total_weight * 100 = 75 :=
by
  sorry

end pie_not_crust_percentage_l254_254150


namespace chocolate_cost_l254_254594

def cost_of_chocolates (candies_per_box : ℕ) (cost_per_box : ℕ) (total_candies : ℕ) : ℕ :=
  (total_candies / candies_per_box) * cost_per_box

theorem chocolate_cost : cost_of_chocolates 30 8 450 = 120 :=
by
  -- The proof is not needed per the instructions
  sorry

end chocolate_cost_l254_254594


namespace driver_speed_l254_254471

theorem driver_speed (v : ℝ) : 
  (∀ t > 0, ∀ d > 0, (v + 10) * (t * (3 / 4)) = d → v * t = d) → 
  v = 30 := 
by
  intro h
  have eq1 : (3 / 4) * (v + 10) = v by sorry
  exact sorry

end driver_speed_l254_254471


namespace length_more_than_breadth_l254_254842

theorem length_more_than_breadth (b : ℝ) (x : ℝ) 
  (h1 : b + x = 55) 
  (h2 : 4 * b + 2 * x = 200) 
  (h3 : (5300 : ℝ) / 26.5 = 200)
  : x = 10 := 
by
  sorry

end length_more_than_breadth_l254_254842


namespace arctan_sum_eq_pi_div_two_l254_254920

theorem arctan_sum_eq_pi_div_two : Real.arctan (3 / 4) + Real.arctan (4 / 3) = Real.pi / 2 :=
by
  sorry

end arctan_sum_eq_pi_div_two_l254_254920


namespace third_team_cups_l254_254831

theorem third_team_cups (required_cups : ℕ) (first_team : ℕ) (second_team : ℕ) (third_team : ℕ) :
  required_cups = 280 ∧ first_team = 90 ∧ second_team = 120 →
  third_team = required_cups - (first_team + second_team) :=
by
  intro h
  rcases h with ⟨h1, h2, h3⟩
  sorry

end third_team_cups_l254_254831


namespace interval_intersection_l254_254173

/--
  This statement asserts that the intersection of the intervals (2/4, 3/4) and (2/5, 3/5)
  results in the interval (1/2, 0.6), which is the solution to the problem.
-/
theorem interval_intersection :
  { x : ℝ | 2 < 4 * x ∧ 4 * x < 3 ∧ 2 < 5 * x ∧ 5 * x < 3 } = { x : ℝ | 0.5 < x ∧ x < 0.6 } :=
by
  sorry

end interval_intersection_l254_254173


namespace cost_of_marker_l254_254114

theorem cost_of_marker (n m : ℝ) (h1 : 3 * n + 2 * m = 7.45) (h2 : 4 * n + 3 * m = 10.40) : m = 1.40 :=
  sorry

end cost_of_marker_l254_254114


namespace minimum_omega_l254_254429

theorem minimum_omega (ω : ℝ) (hω_pos : ω > 0)
  (f : ℝ → ℝ) (hf : ∀ x, f x = Real.sin (ω * x + Real.pi / 3))
  (C : ℝ → ℝ) (hC : ∀ x, C x = Real.sin (ω * (x + Real.pi / 2) + Real.pi / 3)) :
  (∀ x, C x = C (-x)) ↔ ω = 1 / 3 := by
sorry

end minimum_omega_l254_254429


namespace minimum_omega_l254_254433

theorem minimum_omega {ω : ℝ} (hω : ω > 0)
    (symmetry : ∃ k : ℤ, ∀ x : ℝ, 
      (sin (ω * x + ω * π / 2 + π / 3) = sin (-ω * x + ω * π / 2 + π / 3))) 
    : ω = 1 / 3 :=
by
  sorry

end minimum_omega_l254_254433


namespace find_first_offset_l254_254944

variable (d y A x : ℝ)

theorem find_first_offset (h_d : d = 40) (h_y : y = 6) (h_A : A = 300) :
    x = 9 :=
by
  sorry

end find_first_offset_l254_254944


namespace line_equation_l254_254167

-- Define the point A(2, 1)
def A : ℝ × ℝ := (2, 1)

-- Define the notion of a line with equal intercepts on the coordinates
def line_has_equal_intercepts (c : ℝ) : Prop :=
  ∃ (m b : ℝ), ∀ (x y : ℝ), y = m * x + b ↔ x = y ∧ y = c

-- Define the condition that the line passes through point A
def line_passes_through_A (m b : ℝ) : Prop :=
  A.2 = m * A.1 + b

-- Define the two possible equations for the line
def line_eq1 (x y : ℝ) : Prop :=
  x + y - 3 = 0

def line_eq2 (x y : ℝ) : Prop :=
  2 * x - y = 0

-- Combined conditions in a single theorem
theorem line_equation (m b c x y : ℝ) (h_pass : line_passes_through_A m b) (h_int : line_has_equal_intercepts c) :
  (line_eq1 x y ∨ line_eq2 x y) :=
sorry

end line_equation_l254_254167


namespace dot_product_parallel_vectors_l254_254505

variable (x : ℝ)
def a : ℝ × ℝ := (x, x - 1)
def b : ℝ × ℝ := (1, 2)
def are_parallel (a b : ℝ × ℝ) : Prop :=
  a.1 / b.1 = a.2 / b.2

theorem dot_product_parallel_vectors
  (h_parallel : are_parallel (a x) b)
  (h_x : x = -1) :
  (a x).1 * (b).1 + (a x).2 * (b).2 = -5 :=
by
  sorry

end dot_product_parallel_vectors_l254_254505


namespace power_C_50_l254_254811

def matrixC : Matrix (Fin 2) (Fin 2) ℤ :=
  ![![2, 1], ![-4, -1]]

theorem power_C_50 :
  matrixC ^ 50 = ![![4^49 + 1, 4^49], ![-4^50, -2 * 4^49 + 1]] :=
by
  sorry

end power_C_50_l254_254811


namespace min_value_xy_l254_254779

theorem min_value_xy (x y : ℕ) (h : 0 < x ∧ 0 < y) (cond : (1 : ℚ) / x + (1 : ℚ) /(3 * y) = 1 / 6) : 
  xy = 192 :=
sorry

end min_value_xy_l254_254779


namespace actual_cost_of_article_l254_254130

-- Define the basic conditions of the problem
variable (x : ℝ)
variable (h : x - 0.24 * x = 1064)

-- The theorem we need to prove
theorem actual_cost_of_article : x = 1400 :=
by
  -- since we are not proving anything here, we skip the proof
  sorry

end actual_cost_of_article_l254_254130


namespace max_isosceles_tris_2017_gon_l254_254622

theorem max_isosceles_tris_2017_gon :
  ∀ (n : ℕ), n = 2017 →
  ∃ (t : ℕ), (∃ (d : ℕ), d = 2014 ∧ 2015 = (n - 2)) →
  t = 2010 :=
by
  sorry

end max_isosceles_tris_2017_gon_l254_254622


namespace symmetry_axis_of_transformed_function_l254_254044

theorem symmetry_axis_of_transformed_function :
  let initial_func (x : ℝ) := Real.sin (4 * x - π / 6)
  let stretched_func (x : ℝ) := Real.sin (8 * x - π / 3)
  let transformed_func (x : ℝ) := Real.sin (8 * (x + π / 4) - π / 3)
  let ω := 8
  let φ := 5 * π / 3
  x = π / 12 :=
  sorry

end symmetry_axis_of_transformed_function_l254_254044


namespace rational_point_partition_exists_l254_254092

open Set

-- Define rational numbers
noncomputable def Q : Set ℚ :=
  {x | True}

-- Define the set of rational points in the plane
def I : Set (ℚ × ℚ) := 
  {p | p.1 ∈ Q ∧ p.2 ∈ Q}

-- Statement of the theorem
theorem rational_point_partition_exists :
  ∃ (A B : Set (ℚ × ℚ)),
    (∀ (y : ℚ), {p ∈ A | p.1 = y}.Finite) ∧
    (∀ (x : ℚ), {p ∈ B | p.2 = x}.Finite) ∧
    (A ∪ B = I) ∧
    (A ∩ B = ∅) :=
sorry

end rational_point_partition_exists_l254_254092


namespace jane_total_drying_time_l254_254076

theorem jane_total_drying_time :
  let base_coat := 4
  let color_coat_1 := 5
  let color_coat_2 := 6
  let color_coat_3 := 7
  let nail_art_1 := 8
  let nail_art_2 := 10
  let top_coat := 9
  base_coat + color_coat_1 + color_coat_2 + color_coat_3 + nail_art_1 + nail_art_2 + top_coat = 49 :=
by 
  sorry

end jane_total_drying_time_l254_254076


namespace relationship_x_y_l254_254339

variable (a b x y : ℝ)

theorem relationship_x_y (h1: 0 < a) (h2: a < b)
  (hx : x = (Real.sqrt (a + b) - Real.sqrt b))
  (hy : y = (Real.sqrt b - Real.sqrt (b - a))) :
  x < y :=
  sorry

end relationship_x_y_l254_254339


namespace repeatingDecimals_fraction_eq_l254_254757

noncomputable def repeatingDecimalsSum : ℚ :=
  let x : ℚ := 1 / 3
  let y : ℚ := 4 / 99
  let z : ℚ := 5 / 999
  x + y + z

theorem repeatingDecimals_fraction_eq : repeatingDecimalsSum = 42 / 111 :=
  sorry

end repeatingDecimals_fraction_eq_l254_254757


namespace union_of_A_B_l254_254034

def A : Set ℝ := {x | -1 ≤ x ∧ x ≤ 2}
def B : Set ℝ := {x | x > 0}

theorem union_of_A_B :
  A ∪ B = {x | x ≥ -1} := by
  sorry

end union_of_A_B_l254_254034


namespace no_four_nat_numbers_sum_2_pow_100_prod_17_pow_100_l254_254319

theorem no_four_nat_numbers_sum_2_pow_100_prod_17_pow_100 :
  ¬ ∃ (a b c d : ℕ), a + b + c + d = 2^100 ∧ a * b * c * d = 17^100 :=
by
  sorry

end no_four_nat_numbers_sum_2_pow_100_prod_17_pow_100_l254_254319


namespace find_n_l254_254153

-- Definitions and conditions
def painted_total_faces (n : ℕ) : ℕ := 6 * n^2
def total_faces_of_unit_cubes (n : ℕ) : ℕ := 6 * n^3
def fraction_of_red_faces (n : ℕ) : ℚ := (painted_total_faces n : ℚ) / (total_faces_of_unit_cubes n : ℚ)

-- Statement to be proven
theorem find_n (n : ℕ) (h : fraction_of_red_faces n = 1 / 4) : n = 4 :=
by
  sorry

end find_n_l254_254153


namespace probability_at_least_one_blown_l254_254541

theorem probability_at_least_one_blown (P_A P_B P_AB : ℝ)  
  (hP_A : P_A = 0.085) 
  (hP_B : P_B = 0.074) 
  (hP_AB : P_AB = 0.063) : 
  P_A + P_B - P_AB = 0.096 :=
by
  sorry

end probability_at_least_one_blown_l254_254541


namespace common_point_graphs_l254_254962

theorem common_point_graphs 
  (a b c d : ℝ)
  (h1 : ∃ x : ℝ, 2*a + (1 / (x - b)) = 2*c + (1 / (x - d))) :
  ∃ x : ℝ, 2*b + (1 / (x - a)) = 2*d + (1 / (x - c)) :=
by
  sorry

end common_point_graphs_l254_254962


namespace emily_catch_catfish_l254_254021

-- Definitions based on given conditions
def num_trout : ℕ := 4
def num_bluegills : ℕ := 5
def weight_trout : ℕ := 2
def weight_catfish : ℚ := 1.5
def weight_bluegill : ℚ := 2.5
def total_fish_weight : ℚ := 25

-- Lean statement to prove the number of catfish
theorem emily_catch_catfish : ∃ (num_catfish : ℕ), 
  num_catfish * weight_catfish = total_fish_weight - (num_trout * weight_trout + num_bluegills * weight_bluegill) ∧
  num_catfish = 3 := by
  sorry

end emily_catch_catfish_l254_254021


namespace definite_integral_l254_254604

open Real

theorem definite_integral : ∫ x in (0 : ℝ)..(π / 2), (x + sin x) = π^2 / 8 + 1 :=
by
  sorry

end definite_integral_l254_254604


namespace perp_vectors_dot_product_eq_zero_l254_254963

noncomputable def vector_a : ℝ × ℝ := (1, 2)
noncomputable def vector_b (x : ℝ) : ℝ × ℝ := (x, 4)

def dot_product (v1 v2 : ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

theorem perp_vectors_dot_product_eq_zero (x : ℝ) (h : dot_product vector_a (vector_b x) = 0) : x = -8 :=
  by sorry

end perp_vectors_dot_product_eq_zero_l254_254963


namespace value_does_not_appear_l254_254327

theorem value_does_not_appear : 
  let f : ℕ → ℕ := fun x => 5*x^5 + 4*x^4 + 3*x^3 + 2*x^2 + x + 1
  let x := 2
  let values := [14, 31, 64, 129, 259]
  127 ∉ values :=
by
  sorry

end value_does_not_appear_l254_254327


namespace min_value_a_p_a_q_l254_254198

theorem min_value_a_p_a_q (a : ℕ → ℕ) (p q : ℕ) (h_arith_geom : ∀ n, a (n + 2) = a (n + 1) + a n * 2)
(h_a9 : a 9 = a 8 + 2 * a 7)
(h_ap_aq : a p * a q = 8 * a 1 ^ 2) :
    (1 / p : ℝ) + (4 / q : ℝ) = 9 / 5 := by
    sorry

end min_value_a_p_a_q_l254_254198


namespace parallel_lines_from_perpendicularity_l254_254634

variables (a b : Type) (α β : Type)

-- Define the necessary conditions
def is_line (l : Type) : Prop := sorry
def is_plane (p : Type) : Prop := sorry
def perpendicular (l : Type) (p : Type) : Prop := sorry
def parallel (l1 l2 : Type) : Prop := sorry

axiom line_a : is_line a
axiom line_b : is_line b
axiom plane_alpha : is_plane α
axiom plane_beta : is_plane β
axiom a_perp_alpha : perpendicular a α
axiom b_perp_alpha : perpendicular b α

-- State the theorem
theorem parallel_lines_from_perpendicularity : parallel a b :=
  sorry

end parallel_lines_from_perpendicularity_l254_254634


namespace solve_system_l254_254326

theorem solve_system :
  ∃ (x y : ℚ), (4 * x - 35 * y = -1) ∧ (3 * y - x = 5) ∧ (x = -172 / 23) ∧ (y = -19 / 23) :=
by
  sorry

end solve_system_l254_254326


namespace count_restricted_arrangements_l254_254373

theorem count_restricted_arrangements (n : ℕ) (hn : n = 5) : 
  (n.factorial - 2 * (n - 1).factorial) = 72 := 
by 
  sorry

end count_restricted_arrangements_l254_254373


namespace x_squared_plus_y_squared_l254_254035

theorem x_squared_plus_y_squared (x y : ℝ) 
  (h1 : (1/x) + (1/y) = 5) 
  (h2 : x * y + x + y = 11) : 
  x^2 + y^2 = 2893 / 36 := 
by 
  sorry

end x_squared_plus_y_squared_l254_254035


namespace interval_of_x_l254_254184

theorem interval_of_x (x : ℝ) : 
  (2 < 4 * x ∧ 4 * x < 3) → (2 < 5 * x ∧ 5 * x < 3) → (1 / 2 < x ∧ x < 3 / 5) :=
by
  sorry

end interval_of_x_l254_254184


namespace part_1_part_2_l254_254128

noncomputable def prob_pass_no_fee : ℚ :=
  (3 / 4) * (2 / 3) +
  (1 / 4) * (3 / 4) * (2 / 3) +
  (3 / 4) * (1 / 3) * (2 / 3) +
  (1 / 4) * (3 / 4) * (1 / 3) * (2 / 3)

noncomputable def prob_pass_200_fee : ℚ :=
  (1 / 4) * (1 / 4) * (3 / 4) * ((2 / 3) + (1 / 3) * (2 / 3)) +
  (1 / 3) * (1 / 3) * (2 / 3) * ((3 / 4) + (1 / 4) * (3 / 4))

theorem part_1 : prob_pass_no_fee = 5 / 6 := by
  sorry

theorem part_2 : prob_pass_200_fee = 1 / 9 := by
  sorry

end part_1_part_2_l254_254128


namespace area_of_ground_l254_254642

def height_of_rain : ℝ := 0.05
def volume_of_water : ℝ := 750

theorem area_of_ground : ∃ A : ℝ, A = (volume_of_water / height_of_rain) ∧ A = 15000 := by
  sorry

end area_of_ground_l254_254642


namespace range_of_m_for_distance_l254_254391

def distance (x1 y1 x2 y2 : ℝ) : ℝ :=
  (|x1 - x2|) + 2 * (|y1 - y2|)

theorem range_of_m_for_distance (m : ℝ) : 
  distance 2 1 (-1) m ≤ 5 ↔ 0 ≤ m ∧ m ≤ 2 :=
by
  sorry

end range_of_m_for_distance_l254_254391


namespace evaluate_expr_l254_254023

theorem evaluate_expr : 3 * (3 * (3 * (3 * (3 * (3 * 2 * 2) * 2) * 2) * 2) * 2) * 2 = 1458 := by
  sorry

end evaluate_expr_l254_254023


namespace general_term_arithmetic_sequence_sum_terms_sequence_l254_254774

noncomputable def a_n (n : ℕ) : ℤ := 
  2 * (n : ℤ) - 1

theorem general_term_arithmetic_sequence :
  ∀ n : ℕ, a_n n = 2 * (n : ℤ) - 1 :=
by sorry

noncomputable def c (n : ℕ) : ℚ := 
  1 / ((2 * (n : ℤ) - 1) * (2 * (n + 1) - 1))

noncomputable def T_n (n : ℕ) : ℚ :=
  (1 / 2 : ℚ) * (1 - (1 / (2 * (n : ℤ) + 1)))

theorem sum_terms_sequence :
  ∀ n : ℕ, T_n n = (n : ℚ) / (2 * (n : ℤ) + 1) :=
by sorry

end general_term_arithmetic_sequence_sum_terms_sequence_l254_254774


namespace robert_ate_more_l254_254669

variable (robert_chocolates : ℕ) (nickel_chocolates : ℕ)
variable (robert_ate_9 : robert_chocolates = 9) (nickel_ate_2 : nickel_chocolates = 2)

theorem robert_ate_more : robert_chocolates - nickel_chocolates = 7 :=
  by
    sorry

end robert_ate_more_l254_254669


namespace solution_set_inequality_l254_254208

theorem solution_set_inequality (m : ℝ) (f : ℝ → ℝ) (h1 : ∀ x, f x = Real.exp x + Real.exp (-x))
  (h2 : ∀ x, f (-x) = f x) (h3 : ∀ x, 0 ≤ x → ∀ y, 0 ≤ y → x ≤ y → f x ≤ f y) :
  (f (2 * m) > f (m - 2)) ↔ (m > (2 / 3) ∨ m < -2) :=
  sorry

end solution_set_inequality_l254_254208


namespace set_nonempty_iff_nonneg_l254_254060

theorem set_nonempty_iff_nonneg (a : ℝ) :
  (∃ x : ℝ, x^2 ≤ a) ↔ a ≥ 0 :=
sorry

end set_nonempty_iff_nonneg_l254_254060


namespace range_of_a_l254_254348

theorem range_of_a (a : ℝ) (h : ∀ x, a ≤ x ∧ x ≤ a + 2 → |x + a| ≥ 2 * |x|) : a ≤ -3 / 2 := 
by
  sorry

end range_of_a_l254_254348


namespace bus_speed_excluding_stoppages_l254_254322

theorem bus_speed_excluding_stoppages (S : ℝ) (h₀ : 0 < S) (h₁ : 36 = (2/3) * S) : S = 54 :=
by 
  sorry

end bus_speed_excluding_stoppages_l254_254322


namespace y_coord_vertex_of_parabola_l254_254760

-- Define the quadratic equation of the parabola
def parabola (x : ℝ) : ℝ := 2 * x^2 + 16 * x + 29

-- Statement to prove
theorem y_coord_vertex_of_parabola : ∃ (x : ℝ), parabola x = 2 * (x + 4)^2 - 3 := sorry

end y_coord_vertex_of_parabola_l254_254760


namespace solve_equation_l254_254555

theorem solve_equation (x : ℝ) (h1 : x ≠ 1) (h2 : x ≠ -6) :
  (3 * x + 6) / (x^2 + 5 * x - 6) = (3 - x) / (x - 1) ↔ x = -4 ∨ x = -2 :=
by
  sorry

end solve_equation_l254_254555


namespace avg_weight_BC_l254_254560

variable (A B C : ℝ)

def totalWeight_ABC := 3 * 45
def totalWeight_AB := 2 * 40
def weight_B := 31

theorem avg_weight_BC : ((B + C) / 2) = 43 :=
  by
    have totalWeight_ABC_eq : A + B + C = totalWeight_ABC := by sorry
    have totalWeight_AB_eq : A + B = totalWeight_AB := by sorry
    have weight_B_eq : B = weight_B := by sorry
    sorry

end avg_weight_BC_l254_254560


namespace david_marks_in_english_l254_254016

theorem david_marks_in_english 
  (math : ℤ) (phys : ℤ) (chem : ℤ) (bio : ℤ) (avg : ℤ) 
  (marks_per_math : math = 85) 
  (marks_per_phys : phys = 92) 
  (marks_per_chem : chem = 87) 
  (marks_per_bio : bio = 95) 
  (avg_marks : avg = 89) 
  (num_subjects : ℤ := 5) :
  ∃ (eng : ℤ), eng + 85 + 92 + 87 + 95 = 89 * 5 ∧ eng = 86 :=
by
  sorry

end david_marks_in_english_l254_254016


namespace total_gallons_l254_254075

-- Definitions from conditions
def num_vans : ℕ := 6
def standard_capacity : ℕ := 8000
def reduced_capacity : ℕ := standard_capacity - (30 * standard_capacity / 100)
def increased_capacity : ℕ := standard_capacity + (50 * standard_capacity / 100)

-- Total number of specific types of vans
def num_standard_vans : ℕ := 2
def num_reduced_vans : ℕ := 1
def num_increased_vans : ℕ := num_vans - num_standard_vans - num_reduced_vans

-- The proof goal
theorem total_gallons : 
  (num_standard_vans * standard_capacity) + 
  (num_reduced_vans * reduced_capacity) + 
  (num_increased_vans * increased_capacity) = 
  57600 := 
by
  -- The necessary proof can be filled here
  sorry

end total_gallons_l254_254075


namespace x_and_y_complete_work_in_12_days_l254_254702

noncomputable def work_rate_x : ℚ := 1 / 24
noncomputable def work_rate_y : ℚ := 1 / 24
noncomputable def combined_work_rate : ℚ := work_rate_x + work_rate_y

theorem x_and_y_complete_work_in_12_days : (1 / combined_work_rate) = 12 :=
by
  sorry

end x_and_y_complete_work_in_12_days_l254_254702


namespace evaluate_powers_l254_254320

theorem evaluate_powers : (81^(1/2:ℝ) * 64^(-1/3:ℝ) * 49^(1/4:ℝ) = 9 * (1/4) * Real.sqrt 7) :=
by
  sorry

end evaluate_powers_l254_254320


namespace abs_neg_three_eq_three_l254_254838

theorem abs_neg_three_eq_three : abs (-3) = 3 := 
by 
  sorry

end abs_neg_three_eq_three_l254_254838


namespace range_of_x_l254_254739

noncomputable def integer_part (x : ℝ) : ℤ := ⌊x⌋

theorem range_of_x (x : ℝ) (h : integer_part ((1 - 3*x) / 2) = -1) : (1 / 3) < x ∧ x ≤ 1 :=
by sorry

end range_of_x_l254_254739


namespace ilya_defeats_dragon_l254_254512

noncomputable def prob_defeat : ℝ := 1 / 4 * 2 + 1 / 3 * 1 + 5 / 12 * 0

theorem ilya_defeats_dragon : prob_defeat = 1 := sorry

end ilya_defeats_dragon_l254_254512


namespace intersection_A_B_when_m_eq_2_range_of_m_for_p_implies_q_l254_254195

noncomputable def A := {x : ℝ | -4 < x ∧ x < 2}
noncomputable def B (m : ℝ) := {x : ℝ | 1 - m ≤ x ∧ x ≤ 1 + m}

theorem intersection_A_B_when_m_eq_2 : (A ∩ B 2) = {x : ℝ | -1 ≤ x ∧ x < 2} :=
by
  sorry

theorem range_of_m_for_p_implies_q : {m : ℝ | m ≥ 5} = {m : ℝ | ∀ x, ((x^2 + 2 * x - 8 < 0) → ((x - 1 + m) * (x - 1 - m) ≤ 0)) ∧ ¬((x - 1 + m) * (x - 1 - m) ≤ 0 → (x^2 + 2 * x - 8 < 0))} :=
by
  sorry

end intersection_A_B_when_m_eq_2_range_of_m_for_p_implies_q_l254_254195


namespace part3_conclusion_l254_254386

-- Definitions and conditions for the problem
def quadratic_function (a x : ℝ) : ℝ := (x - a)^2 + a - 1

-- Part 1: Given condition that (1, 2) lies on the graph of the quadratic function
def part1_condition (a : ℝ) := (quadratic_function a 1) = 2

-- Part 2: Given condition that the function has a minimum value of 2 for 1 ≤ x ≤ 4
def part2_condition (a : ℝ) := ∀ x, 1 ≤ x ∧ x ≤ 4 → quadratic_function a x ≥ 2

-- Part 3: Given condition (m, n) on the graph where m > 0 and m > 2a
def part3_condition (a m n : ℝ) := m > 0 ∧ m > 2 * a ∧ quadratic_function a m = n

-- Conclusion for Part 3: Prove that n > -5/4
theorem part3_conclusion (a m n : ℝ) (h : part3_condition a m n) : n > -5/4 := 
sorry  -- Proof required here

end part3_conclusion_l254_254386


namespace parabola_axis_of_symmetry_range_l254_254387

theorem parabola_axis_of_symmetry_range
  (a b c m n t : ℝ)
  (h₀ : 0 < a)
  (h₁ : m = a * 1^2 + b * 1 + c)
  (h₂ : n = a * 3^2 + b * 3 + c)
  (h₃ : m < n)
  (h₄ : n < c)
  (h_t : t = -b / (2 * a)) :
  (3 / 2) < t ∧ t < 2 :=
sorry

end parabola_axis_of_symmetry_range_l254_254387


namespace max_a4_l254_254951

theorem max_a4 (a1 d a4 : ℝ) 
  (h1 : 2 * a1 + 3 * d ≥ 5) 
  (h2 : a1 + 2 * d ≤ 3) 
  (ha4 : a4 = a1 + 3 * d) : 
  a4 ≤ 4 := 
by 
  sorry

end max_a4_l254_254951


namespace find_f_neg12_add_f_14_l254_254773

noncomputable def f (x : ℝ) : ℝ := 1 + Real.log (Real.sqrt (x^2 - 2*x + 2) - x + 1)

theorem find_f_neg12_add_f_14 : f (-12) + f 14 = 2 :=
by
  -- The hard part, the actual proof, is left as sorry.
  sorry

end find_f_neg12_add_f_14_l254_254773


namespace cistern_filling_time_l254_254859

open Real

theorem cistern_filling_time :
  let rate1 := 1 / 10
  let rate2 := 1 / 12
  let rate3 := -1 / 25
  let rate4 := 1 / 15
  let rate5 := -1 / 30
  let combined_rate := rate1 + rate2 + rate4 + rate3 + rate5
  (300 / combined_rate) = (300 / 53) := by
  let rate1 := 1 / 10
  let rate2 := 1 / 12
  let rate3 := -1 / 25
  let rate4 := 1 / 15
  let rate5 := -1 / 30
  let combined_rate := rate1 + rate2 + rate4 + rate3 + rate5
  sorry

end cistern_filling_time_l254_254859


namespace find_x_plus_y_l254_254786

theorem find_x_plus_y (x y : ℝ) (h1 : x + Real.cos y = 2023) 
                           (h2 : x + 2023 * Real.sin y = 2022) 
                           (h3 : (Real.pi / 2) ≤ y ∧ y ≤ Real.pi) : 
  x + y = 2023 + Real.pi / 2 :=
sorry

end find_x_plus_y_l254_254786


namespace Zhenya_Venya_are_truth_tellers_l254_254165

-- Definitions
def is_truth_teller(dwarf : String) (truth_teller : String → Bool) : Prop :=
  truth_teller dwarf = true

def is_liar(dwarf : String) (truth_teller : String → Bool) : Prop :=
  truth_teller dwarf = false

noncomputable def BenyaStatement := "V is a liar"
noncomputable def ZhenyaStatement := "B is a liar"
noncomputable def SenyaStatement1 := "B and V are liars"
noncomputable def SenyaStatement2 := "Zh is a liar"

-- Conditions and proving the statement
theorem Zhenya_Venya_are_truth_tellers (truth_teller : String → Bool) :
  (∀ dwarf, truth_teller dwarf = true ∨ truth_teller dwarf = false) →
  (is_truth_teller "Benya" truth_teller → is_liar "Venya" truth_teller) →
  (is_truth_teller "Zhenya" truth_teller → is_liar "Benya" truth_teller) →
  (is_truth_teller "Senya" truth_teller → 
    is_liar "Benya" truth_teller ∧ is_liar "Venya" truth_teller ∧ is_liar "Zhenya" truth_teller) →
  is_truth_teller "Zhenya" truth_teller ∧ is_truth_teller "Venya" truth_teller :=
by
  sorry

end Zhenya_Venya_are_truth_tellers_l254_254165


namespace convert_to_polar_coordinates_l254_254161

theorem convert_to_polar_coordinates :
  ∀ (x y : ℝ), 
  x = -1 → 
  y = √3 → 
  (∃ r θ : ℝ, r = sqrt (x^2 + y^2) ∧ θ = real.arctan2 y x ∧ r = 2 ∧ θ = 2 * π / 3) :=
by
  intros x y hx hy
  use (sqrt (x^2 + y^2)), (real.arctan2 y x)
  constructor
  { sorry }, -- Proof that r = sqrt (x^2 + y^2)
  constructor
  { 
    sorry 
  }, -- Proof that θ = real.arctan2 y x
  constructor
  { sorry }, -- Proof that r = 2
  { sorry } -- Proof that θ = 2 * π / 3

end convert_to_polar_coordinates_l254_254161


namespace disjoint_sets_exist_l254_254796

open Finset

theorem disjoint_sets_exist {n k m : ℕ} (S : Finset ℕ) (A : Finset ℕ) 
  (hA : A ⊆ S) (hA_card : A.card = k) (hS : S = range (n + 1)) 
  (hn : n > (m - 1) * (Nat.choose k 2 + 1)) : 
  ∃ (t : Fin ℕ → ℕ) (H : ∀ j, j < m → t j ∈ S), 
  ∀ i j, i ≠ j → ((A.map (Function.add (t i))).disjoint (A.map (Function.add (t j)))) := 
by
  sorry

end disjoint_sets_exist_l254_254796


namespace solve_y_l254_254945

theorem solve_y (y : ℝ) (h1 : y > 0) (h2 : (y - 6) / 16 = 6 / (y - 16)) : y = 22 :=
by
  sorry

end solve_y_l254_254945


namespace find_n_value_l254_254654

theorem find_n_value (x y : ℕ) : x = 3 → y = 1 → n = x - y^(x - y) → x > y → n + x * y = 5 := by sorry

end find_n_value_l254_254654


namespace incorrect_statement_is_C_l254_254540

theorem incorrect_statement_is_C (b h s a x : ℝ) (hb : b > 0) (hh : h > 0) (hs : s > 0) (hx : x < 0) :
  ¬ (9 * s^2 = 4 * (3 * s)^2) :=
by
  sorry

end incorrect_statement_is_C_l254_254540


namespace seashells_unbroken_l254_254411

theorem seashells_unbroken (total_seashells broken_seashells unbroken_seashells : ℕ) 
  (h1 : total_seashells = 6) 
  (h2 : broken_seashells = 4) 
  (h3 : unbroken_seashells = total_seashells - broken_seashells) :
  unbroken_seashells = 2 :=
by
  sorry

end seashells_unbroken_l254_254411


namespace max_possible_value_l254_254720

theorem max_possible_value (P Q : ℤ) (hP : P * P ≤ 729 ∧ 729 ≤ -P * P * P)
  (hQ : Q * Q ≤ 729 ∧ 729 ≤ -Q * Q * Q) :
  10 * (P - Q) = 180 :=
by
  sorry

end max_possible_value_l254_254720


namespace arithmetic_sum_ratio_l254_254033

variable (a_n : ℕ → ℤ) -- the arithmetic sequence
variable (S : ℕ → ℤ) -- sum of the first n terms of the sequence
variable (d : ℤ) (a₁ : ℤ) -- common difference and first term of the sequence

-- Definition of the sum of the first n terms in an arithmetic sequence
def arithmetic_sum (n : ℕ) : ℤ :=
  (n * (2 * a₁ + (n - 1) * d)) / 2

-- Given condition
axiom h1 : (S 6) / (S 3) = 3

-- Definition of S_n in terms of the given formula
axiom S_def : ∀ n, S n = arithmetic_sum n

-- The main goal to prove
theorem arithmetic_sum_ratio : S 12 / S 9 = 5 / 3 := by
  sorry

end arithmetic_sum_ratio_l254_254033


namespace nonagon_diagonals_l254_254049

-- Define the number of sides of the polygon (nonagon)
def num_sides : ℕ := 9

-- Define the formula for the number of diagonals in a convex n-sided polygon
def number_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

-- State the theorem
theorem nonagon_diagonals : number_diagonals num_sides = 27 := 
by
--placeholder for the proof
sorry

end nonagon_diagonals_l254_254049


namespace smallest_positive_x_l254_254697

def is_palindrome (n : ℕ) : Prop :=
  let digits := n.digits 10
  digits = digits.reverse

theorem smallest_positive_x : ∃ x : ℕ, x > 0 ∧ is_palindrome (x + 6789) ∧ x = 218 := by
  sorry

end smallest_positive_x_l254_254697


namespace evaluate_expression_l254_254321

open Nat

theorem evaluate_expression : 
  (3 * 4 * 5 * 6) * (1 / 3 + 1 / 4 + 1 / 5 + 1 / 6) = 342 := by
  sorry

end evaluate_expression_l254_254321


namespace base_radius_of_cone_l254_254412

-- Definitions of the conditions
def R1 : ℕ := 5
def R2 : ℕ := 4
def R3 : ℕ := 4
def height_radius_ratio := 4 / 3

-- Main theorem statement
theorem base_radius_of_cone : 
  (R1 = 5) → (R2 = 4) → (R3 = 4) → (height_radius_ratio = 4 / 3) → 
  ∃ r : ℚ, r = 169 / 60 :=
by 
  intros hR1 hR2 hR3 hRatio
  sorry

end base_radius_of_cone_l254_254412


namespace jessica_total_payment_l254_254398

-- Definitions based on the conditions
def basic_cable_cost : Nat := 15
def movie_channels_cost : Nat := 12
def sports_channels_cost : Nat := movie_channels_cost - 3

-- Definition of the total monthly payment given Jessica adds both movie and sports channels
def total_monthly_payment : Nat :=
  basic_cable_cost + (movie_channels_cost + sports_channels_cost)

-- The proof statement
theorem jessica_total_payment : total_monthly_payment = 36 :=
by
  -- skip the proof
  sorry

end jessica_total_payment_l254_254398


namespace solutionSet_l254_254272

def passesThroughQuadrants (a b : ℝ) : Prop :=
  a > 0

def intersectsXAxisAt (a b : ℝ) : Prop :=
  b = 2 * a

theorem solutionSet (a b x : ℝ) (hq : passesThroughQuadrants a b) (hi : intersectsXAxisAt a b) :
  (a * x > b) ↔ (x > 2) :=
by
  sorry

end solutionSet_l254_254272


namespace length_more_than_breadth_l254_254843

theorem length_more_than_breadth (b : ℝ) (x : ℝ) 
  (h1 : b + x = 55) 
  (h2 : 4 * b + 2 * x = 200) 
  (h3 : (5300 : ℝ) / 26.5 = 200)
  : x = 10 := 
by
  sorry

end length_more_than_breadth_l254_254843


namespace hypotenuse_of_triangle_PQR_l254_254977

theorem hypotenuse_of_triangle_PQR (PA PB PC QR : ℝ) (h1: PA = 2) (h2: PB = 3) (h3: PC = 2)
  (h4: PA + PB + PC = QR) (h5: QR = PA + 3 + 2 * PA): QR = 5 * Real.sqrt 2 := 
sorry

end hypotenuse_of_triangle_PQR_l254_254977


namespace part1_part2_l254_254142

-- Define the cost price, current selling price, sales per week, and change in sales per reduction in price.
def cost_price : ℝ := 50
def current_price : ℝ := 80
def current_sales : ℝ := 200
def sales_increase_per_yuan : ℝ := 20

-- Define the weekly profit calculation.
def weekly_profit (price : ℝ) : ℝ :=
(price - cost_price) * (current_sales + sales_increase_per_yuan * (current_price - price))

-- Part 1: Selling price for a weekly profit of 7500 yuan while maximizing customer benefits.
theorem part1 (price : ℝ) : 
  (weekly_profit price = 7500) →  -- Given condition for weekly profit
  (price = 65) := sorry  -- Conclude that the price must be 65 yuan for maximizing customer benefits

-- Part 2: Selling price to maximize the weekly profit and the maximum profit
theorem part2 : 
  ∃ price : ℝ, (price = 70 ∧ weekly_profit price = 8000) := sorry  -- Conclude that the price is 70 yuan and max profit is 8000 yuan

end part1_part2_l254_254142


namespace irrigation_canal_construction_l254_254689

theorem irrigation_canal_construction (x : ℕ) (m : ℕ) :
  (∀ y : ℕ, 1650 = 3 * 1650 / 2 / (y + 30) → y = 60) →
  (∀ n : ℕ, 14 * 90 + (90 + 120) * (14 - n) = 1650 → n = 5) →
  x = 60 ∧ (x + 30) = 90 ∧ 90 * 5 + (90 + 120) * 9 = 2340 :=
begin
  intros H1 H2,
  split,
  { exact H1 x },
  {
    split,
    { exact (H1 x) + 30 },
    { exact H2 5 },
  }
end

end irrigation_canal_construction_l254_254689


namespace strudel_price_l254_254164

def initial_price := 80
def first_increment (P0 : ℕ) := P0 * 3 / 2
def second_increment (P1 : ℕ) := P1 * 3 / 2
def final_price (P2 : ℕ) := P2 / 2

theorem strudel_price (P0 : ℕ) (P1 : ℕ) (P2 : ℕ) (Pf : ℕ)
  (h0 : P0 = initial_price)
  (h1 : P1 = first_increment P0)
  (h2 : P2 = second_increment P1)
  (hf : Pf = final_price P2) :
  Pf = 90 :=
sorry

end strudel_price_l254_254164


namespace right_triangle_min_hypotenuse_right_triangle_min_hypotenuse_achieved_l254_254719

theorem right_triangle_min_hypotenuse (a b c : ℝ) (h_right : (a^2 + b^2 = c^2)) (h_perimeter : (a + b + c = 8)) : c ≥ 4 * Real.sqrt 2 := by
  sorry

theorem right_triangle_min_hypotenuse_achieved (a b c : ℝ) (h_right : (a^2 + b^2 = c^2)) (h_perimeter : (a + b + c = 8)) (h_isosceles : a = b) : c = 4 * Real.sqrt 2 := by
  sorry

end right_triangle_min_hypotenuse_right_triangle_min_hypotenuse_achieved_l254_254719


namespace work_together_l254_254700

theorem work_together (A_rate B_rate : ℝ) (hA : A_rate = 1 / 9) (hB : B_rate = 1 / 18) : (1 / (A_rate + B_rate) = 6) :=
by
  -- we only need to write the statement, proof is not required.
  sorry

end work_together_l254_254700


namespace gain_percent_of_articles_l254_254637

theorem gain_percent_of_articles (C S : ℝ) (h : 50 * C = 15 * S) : (S - C) / C * 100 = 233.33 :=
by
  sorry

end gain_percent_of_articles_l254_254637


namespace polynomial_positive_values_l254_254942

noncomputable def P (x y : ℝ) : ℝ := x^2 + (x*y + 1)^2

theorem polynomial_positive_values :
  ∀ (z : ℝ), (∃ (x y : ℝ), P x y = z) ↔ z > 0 :=
by
  sorry

end polynomial_positive_values_l254_254942


namespace janet_has_five_dimes_l254_254809

theorem janet_has_five_dimes (n d q : ℕ) 
    (h1 : n + d + q = 10) 
    (h2 : d + q = 7) 
    (h3 : n + d = 8) : 
    d = 5 :=
by
  -- Proof omitted
  sorry

end janet_has_five_dimes_l254_254809


namespace abs_diff_squares_110_108_l254_254860

theorem abs_diff_squares_110_108 : abs ((110 : ℤ)^2 - (108 : ℤ)^2) = 436 := by
  sorry

end abs_diff_squares_110_108_l254_254860


namespace range_of_m_l254_254501

theorem range_of_m (x m : ℝ) (h1 : (x ≥ 0) ∧ (x ≠ 1) ∧ (x = (6 - m) / 4)) :
    m ≤ 6 ∧ m ≠ 2 :=
by
  sorry

end range_of_m_l254_254501


namespace M_inter_N_is_empty_l254_254628

-- Definition conditions
def U : Set ℝ := Set.univ
def M : Set ℝ := {x | x^2 - x > 0}
def N : Set ℝ := {x | (x - 1) / x < 0}

-- Theorem statement
theorem M_inter_N_is_empty : M ∩ N = ∅ := by
  sorry

end M_inter_N_is_empty_l254_254628


namespace least_xy_value_l254_254782

theorem least_xy_value (x y : ℕ) (hx : 0 < x) (hy : 0 < y) (h : 1 / x + 1 / (3 * y) = 1 / 6) : x * y = 90 :=
by
  sorry

end least_xy_value_l254_254782


namespace largest_solution_achieves_largest_solution_l254_254762

theorem largest_solution (x : ℝ) (hx : ⌊x⌋ = 5 + 100 * (x - ⌊x⌋)) : x ≤ 104.99 :=
by
  -- Placeholder for the proof
  sorry

theorem achieves_largest_solution : ∃ (x : ℝ), ⌊x⌋ = 5 + 100 * (x - ⌊x⌋) ∧ x = 104.99 :=
by
  -- Placeholder for the proof
  sorry

end largest_solution_achieves_largest_solution_l254_254762


namespace tony_rope_length_l254_254282

-- Define the lengths of the individual ropes.
def rope_lengths : List ℝ := [8, 20, 2, 2, 2, 7]

-- Define the total number of ropes Tony has.
def num_ropes : ℕ := rope_lengths.length

-- Calculate the total length of ropes before tying them together.
def total_length_before_tying : ℝ := rope_lengths.sum

-- Define the length lost per knot.
def length_lost_per_knot : ℝ := 1.2

-- Calculate the total number of knots needed.
def num_knots : ℕ := num_ropes - 1

-- Calculate the total length lost due to knots.
def total_length_lost : ℝ := num_knots * length_lost_per_knot

-- Calculate the total length of the rope after tying them all together.
def total_length_after_tying : ℝ := total_length_before_tying - total_length_lost

-- The theorem we want to prove.
theorem tony_rope_length : total_length_after_tying = 35 :=
by sorry

end tony_rope_length_l254_254282


namespace total_jumps_is_400_l254_254670

-- Define the variables according to the conditions 
def Ronald_jumps := 157
def Rupert_jumps := Ronald_jumps + 86

-- Prove the total jumps
theorem total_jumps_is_400 : Ronald_jumps + Rupert_jumps = 400 := by
  sorry

end total_jumps_is_400_l254_254670


namespace additional_days_use_l254_254879

variable (m a : ℝ)

theorem additional_days_use (hm : m > 0) (ha : a > 1) : 
  (m / (a - 1) - m / a) = m / (a * (a - 1)) :=
sorry

end additional_days_use_l254_254879


namespace geometric_sequence_fraction_l254_254958

noncomputable def a_n : ℕ → ℝ := sorry -- geometric sequence {a_n}
noncomputable def S : ℕ → ℝ := sorry   -- sequence sum S_n
def q : ℝ := sorry                     -- common ratio

theorem geometric_sequence_fraction (h_sequence: ∀ n, 2 * S (n - 1) = S n + S (n + 1))
  (h_q: ∀ n, a_n (n + 1) = q * a_n n)
  (h_q_neg2: q = -2) :
  (a_n 5 + a_n 7) / (a_n 3 + a_n 5) = 4 :=
by 
  sorry

end geometric_sequence_fraction_l254_254958


namespace minimum_omega_l254_254427

theorem minimum_omega (ω : ℝ) (h_omega_pos : ω > 0) :
    (∃ y : ℝ → ℝ, (∀ x, y x = sin (ω * x + ω * (π / 2) + (π / 3))) ∧ 
    (∀ x, y x = y (-x))) →
    (ω = 1 / 3) :=
sorry

end minimum_omega_l254_254427


namespace value_of_X_l254_254225

noncomputable def M : ℕ := 3009 / 3
noncomputable def N : ℕ := (2 * M) / 3
noncomputable def X : ℕ := M - N

theorem value_of_X : X = 335 := by
  sorry

end value_of_X_l254_254225


namespace smallest_of_powers_l254_254293

theorem smallest_of_powers :
  min (2^55) (min (3^44) (min (5^33) (6^22))) = 2^55 :=
by
  sorry

end smallest_of_powers_l254_254293


namespace minimum_omega_for_symmetric_curve_l254_254426

theorem minimum_omega_for_symmetric_curve (ω : ℝ) (hω : ω > 0) :
  (∀ x : ℝ, sin (ω * (x + π / 2) + π / 3) = sin (-ω * (x + π / 2) + π / 3)) ↔ ω = 1 / 3 :=
by
  sorry

end minimum_omega_for_symmetric_curve_l254_254426


namespace fraction_of_4_is_8_l254_254640

theorem fraction_of_4_is_8 (fraction : ℝ) (h : fraction * 4 = 8) : fraction = 8 := 
sorry

end fraction_of_4_is_8_l254_254640


namespace repeating_sum_to_fraction_l254_254745

theorem repeating_sum_to_fraction :
  (0.333333333333333 ~ 1/3) ∧ 
  (0.0404040404040401 ~ 4/99) ∧ 
  (0.005005005005001 ~ 5/999) →
  (0.333333333333333 + 0.0404040404040401 + 0.005005005005001) = (112386 / 296703) := 
by
  repeat { sorry }

end repeating_sum_to_fraction_l254_254745


namespace arctan_sum_eq_pi_div_two_l254_254919

theorem arctan_sum_eq_pi_div_two : Real.arctan (3 / 4) + Real.arctan (4 / 3) = Real.pi / 2 :=
by
  sorry

end arctan_sum_eq_pi_div_two_l254_254919


namespace problem_statement_l254_254863

theorem problem_statement :
  (∀ x : ℝ, |x| < 2 → x < 3) ∧
  (∀ x : ℝ, ¬ (∃ x : ℝ, x^2 + x + 1 < 0) ↔ ∀ x : ℝ, x^2 + x + 1 ≥ 0) ∧
  (-1 < m ∧ m < 0 → ∀ a b : ℝ, a ≠ b → (a * b > 0)) :=
by
  sorry

end problem_statement_l254_254863


namespace reservoir_shortage_l254_254891

noncomputable def reservoir_information := 
  let current_level := 14 -- million gallons
  let normal_level_due_to_yield := current_level / 2
  let percentage_of_capacity := 0.70
  let evaporation_factor := 0.90
  let total_capacity := current_level / percentage_of_capacity
  let normal_level_after_evaporation := normal_level_due_to_yield * evaporation_factor
  let shortage := total_capacity - normal_level_after_evaporation
  shortage

theorem reservoir_shortage :
  reservoir_information = 13.7 := 
by
  sorry

end reservoir_shortage_l254_254891


namespace tshirt_costs_more_than_jersey_l254_254559

-- Definitions based on the conditions
def cost_of_tshirt : ℕ := 192
def cost_of_jersey : ℕ := 34

-- Theorem statement
theorem tshirt_costs_more_than_jersey : cost_of_tshirt - cost_of_jersey = 158 := by
  sorry

end tshirt_costs_more_than_jersey_l254_254559


namespace largest_possible_value_of_s_l254_254522

theorem largest_possible_value_of_s (p q r s : ℝ)
  (h₁ : p + q + r + s = 12)
  (h₂ : pq + pr + ps + qr + qs + rs = 24) : 
  s ≤ 3 + 3 * Real.sqrt 5 :=
sorry

end largest_possible_value_of_s_l254_254522


namespace savings_of_person_l254_254301

-- Definitions as given in the problem
def income := 18000
def ratio_income_expenditure := 5 / 4

-- Implied definitions based on the conditions and problem context
noncomputable def expenditure := income * (4/5)
noncomputable def savings := income - expenditure

-- Theorem statement
theorem savings_of_person : savings = 3600 :=
by
  -- Placeholder for proof
  sorry

end savings_of_person_l254_254301


namespace range_of_x_l254_254737

noncomputable def integerPart (x : ℝ) : ℤ := Int.floor x

theorem range_of_x (x : ℝ) (h : integerPart ((1 - 3 * x) / 2) = -1) : (1 / 3) < x ∧ x ≤ 1 :=
by
  sorry

end range_of_x_l254_254737


namespace Beast_of_War_running_time_l254_254853

theorem Beast_of_War_running_time 
  (M : ℕ) 
  (AE : ℕ) 
  (BoWAC : ℕ)
  (h1 : M = 120)
  (h2 : AE = M - 30)
  (h3 : BoWAC = AE + 10) : 
  BoWAC = 100 
  := 
sorry

end Beast_of_War_running_time_l254_254853


namespace average_weight_BC_l254_254561

-- Define the weights as variables
variables (A B C : ℝ)

-- Define the conditions
def condition1 : Prop := (A + B + C) / 3 = 45
def condition2 : Prop := (A + B) / 2 = 40
def condition3 : Prop := B = 31

-- The theorem to prove
theorem average_weight_BC (h1 : condition1) (h2 : condition2) (h3 : condition3) : (B + C) / 2 = 43 :=
sorry

end average_weight_BC_l254_254561


namespace evaluate_expression_l254_254456

theorem evaluate_expression (m n : ℝ) (h : 4 * m - 4 + n = 2) : 
  (m * (-2)^2 - 2 * (-2) + n = 10) :=
by
  sorry

end evaluate_expression_l254_254456


namespace second_alloy_amount_l254_254385

theorem second_alloy_amount (x : ℝ) :
  (0.12 * 15 + 0.08 * x = 0.092 * (15 + x)) → x = 35 :=
by
  sorry

end second_alloy_amount_l254_254385


namespace joe_money_left_l254_254987

theorem joe_money_left
  (initial_money : ℕ) (notebook_cost : ℕ) (notebooks : ℕ)
  (book_cost : ℕ) (books : ℕ) (pen_cost : ℕ) (pens : ℕ)
  (sticker_pack_cost : ℕ) (sticker_packs : ℕ) (charity : ℕ)
  (remaining_money : ℕ) :
  initial_money = 150 →
  notebook_cost = 4 →
  notebooks = 7 →
  book_cost = 12 →
  books = 2 →
  pen_cost = 2 →
  pens = 5 →
  sticker_pack_cost = 6 →
  sticker_packs = 3 →
  charity = 10 →
  remaining_money = 60 →
  remaining_money = 
    initial_money - 
    ((notebooks * notebook_cost) + 
     (books * book_cost) + 
     (pens * pen_cost) + 
     (sticker_packs * sticker_pack_cost) + 
     charity) := 
by
  intros; sorry

end joe_money_left_l254_254987


namespace least_xy_value_l254_254783

theorem least_xy_value (x y : ℕ) (hx : 0 < x) (hy : 0 < y) (h : 1 / x + 1 / (3 * y) = 1 / 6) : x * y = 90 :=
by
  sorry

end least_xy_value_l254_254783


namespace jar_lasts_20_days_l254_254518

def serving_size : ℝ := 0.5
def daily_servings : ℕ := 3
def container_size : ℝ := 32 - 2

def daily_usage (serving_size : ℝ) (daily_servings : ℕ) : ℝ :=
  serving_size * daily_servings

def days_to_finish (container_size daily_usage : ℝ) : ℝ :=
  container_size / daily_usage

theorem jar_lasts_20_days :
  days_to_finish container_size (daily_usage serving_size daily_servings) = 20 :=
by
  sorry

end jar_lasts_20_days_l254_254518


namespace number_of_boys_l254_254101

theorem number_of_boys (n : ℕ)
  (initial_avg_height : ℕ)
  (incorrect_height : ℕ)
  (correct_height : ℕ)
  (actual_avg_height : ℕ)
  (h1 : initial_avg_height = 184)
  (h2 : incorrect_height = 166)
  (h3 : correct_height = 106)
  (h4 : actual_avg_height = 182)
  (h5 : initial_avg_height * n - (incorrect_height - correct_height) = actual_avg_height * n) :
  n = 30 :=
sorry

end number_of_boys_l254_254101


namespace spring_extension_l254_254292

theorem spring_extension (A1 A2 : ℝ) (x1 x2 : ℝ) (hA1 : A1 = 29.43) (hx1 : x1 = 0.05) (hA2 : A2 = 9.81) : x2 = 0.029 :=
by 
  sorry

end spring_extension_l254_254292


namespace cleaner_needed_l254_254083

def cleaner_per_dog := 6
def cleaner_per_cat := 4
def cleaner_per_rabbit := 1

def num_dogs := 6
def num_cats := 3
def num_rabbits := 1

def total_cleaner_for_dogs := cleaner_per_dog * num_dogs
def total_cleaner_for_cats := cleaner_per_cat * num_cats
def total_cleaner_for_rabbits := cleaner_per_rabbit * num_rabbits

def total_cleaner := total_cleaner_for_dogs + total_cleaner_for_cats + total_cleaner_for_rabbits

theorem cleaner_needed : total_cleaner = 49 :=
by
  unfold total_cleaner total_cleaner_for_dogs total_cleaner_for_cats total_cleaner_for_rabbits cleaner_per_dog cleaner_per_cat cleaner_per_rabbit num_dogs num_cats num_rabbits
  rw [cleaner_per_dog, cleaner_per_cat, cleaner_per_rabbit]
  rw [num_dogs, num_cats, num_rabbits]
  simp
  sorry -- The proof needs to end with a correct justification which is omitted here

end cleaner_needed_l254_254083


namespace pencils_per_box_l254_254528

theorem pencils_per_box:
  ∀ (red_pencils blue_pencils yellow_pencils green_pencils total_pencils num_boxes : ℕ),
  red_pencils = 20 →
  blue_pencils = 2 * red_pencils →
  yellow_pencils = 40 →
  green_pencils = red_pencils + blue_pencils →
  total_pencils = red_pencils + blue_pencils + yellow_pencils + green_pencils →
  num_boxes = 8 →
  total_pencils / num_boxes = 20 :=
by
  intros red_pencils blue_pencils yellow_pencils green_pencils total_pencils num_boxes
  intros h1 h2 h3 h4 h5 h6
  sorry

end pencils_per_box_l254_254528


namespace smallest_possible_value_of_N_l254_254404

theorem smallest_possible_value_of_N :
  ∀ (a b c d e f : ℕ), a + b + c + d + e + f = 3015 → (0 < a) → (0 < b) → (0 < c) → (0 < d) → (0 < e) → (0 < f) →
  (∃ N : ℕ, N = max (max (max (max (a + b) (b + c)) (c + d)) (d + e)) (e + f) ∧ N = 604) := 
by
  sorry

end smallest_possible_value_of_N_l254_254404


namespace expansion_correct_l254_254938

variable (x y : ℝ)

theorem expansion_correct : 
  (3 * x - 15) * (4 * y + 20) = 12 * x * y + 60 * x - 60 * y - 300 :=
by
  sorry

end expansion_correct_l254_254938


namespace only_one_of_A_B_qualifies_at_least_one_qualifies_l254_254140

-- Define the probabilities
def P_A_written : ℚ := 2/3
def P_B_written : ℚ := 1/2
def P_C_written : ℚ := 3/4

def P_A_interview : ℚ := 1/2
def P_B_interview : ℚ := 2/3
def P_C_interview : ℚ := 1/3

-- Calculate the overall probabilities for each student qualifying
def P_A_qualifies : ℚ := P_A_written * P_A_interview
def P_B_qualifies : ℚ := P_B_written * P_B_interview
def P_C_qualifies : ℚ := P_C_written * P_C_interview

-- Part 1: Probability that only one of A or B qualifies
theorem only_one_of_A_B_qualifies :
  P_A_qualifies * (1 - P_B_qualifies) + (1 - P_A_qualifies) * P_B_qualifies = 4/9 :=
by sorry

-- Part 2: Probability that at least one of A, B, or C qualifies
theorem at_least_one_qualifies :
  1 - (1 - P_A_qualifies) * (1 - P_B_qualifies) * (1 - P_C_qualifies) = 2/3 :=
by sorry

end only_one_of_A_B_qualifies_at_least_one_qualifies_l254_254140


namespace largest_possible_b_b_eq_4_of_largest_l254_254241

theorem largest_possible_b (b : ℚ) (h : (3*b + 4)*(b - 2) = 9*b) : b ≤ 4 := by
  sorry

theorem b_eq_4_of_largest (b : ℚ) (h : (3*b + 4)*(b - 2) = 9*b) (hb : b = 4) : True := by
  sorry

end largest_possible_b_b_eq_4_of_largest_l254_254241


namespace sky_color_change_l254_254612

theorem sky_color_change (hours: ℕ) (colors: ℕ) (minutes_per_hour: ℕ) 
                          (H1: hours = 2) 
                          (H2: colors = 12) 
                          (H3: minutes_per_hour = 60) : 
                          (hours * minutes_per_hour) / colors = 10 := 
by
  sorry

end sky_color_change_l254_254612


namespace min_value_of_y_l254_254207

theorem min_value_of_y {y : ℤ} (h : ∃ x : ℤ, y^2 = (0 ^ 2 + 1 ^ 2 + 2 ^ 2 + 3 ^ 2 + 4 ^ 2 + 5 ^ 2 + (-1) ^ 2 + (-2) ^ 2 + (-3) ^ 2 + (-4) ^ 2 + (-5) ^ 2)) :
  y = -11 :=
by sorry

end min_value_of_y_l254_254207


namespace hyperbola_k_range_l254_254959

theorem hyperbola_k_range {k : ℝ} 
  (h : ∀ x y : ℝ, x^2 + (k-1)*y^2 = k+1 → (k > -1 ∧ k < 1)) : 
  -1 < k ∧ k < 1 :=
by 
  sorry

end hyperbola_k_range_l254_254959


namespace work_days_B_l254_254468

theorem work_days_B (A_days B_days : ℕ) (hA : A_days = 12) (hTogether : (1/12 + 1/A_days) = (1/8)) : B_days = 24 := 
by
  revert hTogether -- reversing to tackle proof
  sorry

end work_days_B_l254_254468


namespace mary_remaining_money_l254_254538

variable (p : ℝ) -- p is the price per drink in dollars

def drinks_cost : ℝ := 3 * p
def medium_pizzas_cost : ℝ := 2 * (2 * p)
def large_pizza_cost : ℝ := 3 * p

def total_cost : ℝ := drinks_cost p + medium_pizzas_cost p + large_pizza_cost p

theorem mary_remaining_money : 
  30 - total_cost p = 30 - 10 * p := 
by
  sorry

end mary_remaining_money_l254_254538


namespace car_speed_l254_254139

theorem car_speed (v : ℝ) : 
  (4 + (1 / (80 / 3600))) = (1 / (v / 3600)) → v = 3600 / 49 :=
sorry

end car_speed_l254_254139


namespace paint_time_l254_254948

theorem paint_time (n1 t1 n2 : ℕ) (k : ℕ) (h : n1 * t1 = k) (h1 : 5 * 4 = k) (h2 : n2 = 6) : (k / n2) = 10 / 3 :=
by {
  -- Proof would go here
  sorry
}

end paint_time_l254_254948


namespace circle_center_radius_proof_l254_254740

noncomputable def circle_center_radius (x y : ℝ) :=
  x^2 + y^2 - 4*x + 2*y + 2 = 0

theorem circle_center_radius_proof :
  ∀ x y : ℝ, circle_center_radius x y ↔ ((x - 2)^2 + (y + 1)^2 = 3) :=
by
  sorry

end circle_center_radius_proof_l254_254740


namespace find_a_l254_254038

theorem find_a (a : ℝ) 
  (h1 : a < 0)
  (h2 : a < 1/3)
  (h3 : -2 * a + (1 - 3 * a) = 6) : 
  a = -1 := 
by 
  sorry

end find_a_l254_254038


namespace exists_infinite_solutions_l254_254933

noncomputable def infinite_solutions_exist (m : ℕ) : Prop := 
  ∃ (a b c : ℕ), 0 < a ∧ 0 < b ∧ 0 < c ∧  (1 / a + 1 / b + 1 / c + 1 / (a * b * c) = m / (a + b + c))

theorem exists_infinite_solutions : infinite_solutions_exist 12 :=
  sorry

end exists_infinite_solutions_l254_254933


namespace total_sharks_l254_254734

-- Define the number of sharks at each beach.
def N : ℕ := 22
def D : ℕ := 4 * N
def H : ℕ := D / 2

-- Proof that the total number of sharks on the three beaches is 154.
theorem total_sharks : N + D + H = 154 := by
  sorry

end total_sharks_l254_254734


namespace proper_subset_count_of_set_l254_254156

theorem proper_subset_count_of_set (s : Finset ℕ) (h : s = {1, 2, 3}) : s.powerset.card - 1 = 7 := by
  sorry

end proper_subset_count_of_set_l254_254156


namespace calculation_correct_l254_254617

theorem calculation_correct (x y : ℝ) (hx0 : x ≠ 0) (hy0 : y ≠ 0) (hxy : x = 2 * y) : 
  (x - 2 / x) * (y + 2 / y) = 1 / 2 * (x^2 - 2 * x + 8 - 16 / x) := 
by 
  sorry

end calculation_correct_l254_254617


namespace solve_system_l254_254526

variable {R : Type*} [CommRing R] {a b c x y z : R}

theorem solve_system (h : a ≠ b ∧ b ≠ c ∧ c ≠ a) 
  (h₁ : z + a*y + a^2*x + a^3 = 0) 
  (h₂ : z + b*y + b^2*x + b^3 = 0) 
  (h₃ : z + c*y + c^2*x + c^3 = 0) :
  x = -(a + b + c) ∧ y = (a * b + a * c + b * c) ∧ z = -(a * b * c) := 
sorry

end solve_system_l254_254526


namespace math_problem_real_solution_l254_254406

theorem math_problem_real_solution (x y : ℝ) (h : x^2 * y^2 - x * y - x / y - y / x = 4) : 
  (x - 2) * (y - 2) = 3 - 2 * Real.sqrt 2 :=
sorry

end math_problem_real_solution_l254_254406


namespace range_of_h_l254_254521

def f (x : ℝ) : ℝ := 4 * x - 3
def h (x : ℝ) : ℝ := f (f (f x))

theorem range_of_h : 
  (∀ x, -1 ≤ x ∧ x ≤ 3 → -127 ≤ h x ∧ h x ≤ 129) :=
by
  sorry

end range_of_h_l254_254521


namespace rate_per_sq_meter_is_900_l254_254566

/-- The length of the room L is 7 (meters). -/
def L : ℝ := 7

/-- The width of the room W is 4.75 (meters). -/
def W : ℝ := 4.75

/-- The total cost of paving the floor is Rs. 29,925. -/
def total_cost : ℝ := 29925

/-- The rate per square meter for the slabs is Rs. 900. -/
theorem rate_per_sq_meter_is_900 :
  total_cost / (L * W) = 900 :=
by
  sorry

end rate_per_sq_meter_is_900_l254_254566


namespace first_lock_stall_time_eq_21_l254_254990

-- Definitions of time taken by locks
def firstLockTime : ℕ := 21 -- This will be proven at the end

variables {x : ℕ} -- time for the first lock
variables (secondLockTime : ℕ) (bothLocksTime : ℕ)

-- Conditions given in the problem
axiom lock_relation : secondLockTime = 3 * x - 3
axiom second_lock_time : secondLockTime = 60
axiom combined_locks_time : bothLocksTime = 300

-- Question: Prove that the first lock time is 21 minutes
theorem first_lock_stall_time_eq_21 :
  (bothLocksTime = 5 * secondLockTime) ∧ (secondLockTime = 60) ∧ (bothLocksTime = 300) → x = 21 :=
sorry

end first_lock_stall_time_eq_21_l254_254990


namespace evaluate_series_l254_254610

noncomputable def infinite_series := ∑ k in (Finset.range ∞), (k + 1)^2 / 3^(k + 1)

theorem evaluate_series : infinite_series = 1 / 2 := sorry

end evaluate_series_l254_254610


namespace hyperbola_equation_l254_254787

theorem hyperbola_equation (h1 : ∀ x y : ℝ, (x = 0 ∧ y = 0)) 
                           (h2 : ∀ a : ℝ, (2 * a = 4)) 
                           (h3 : ∀ c : ℝ, (c = 3)) : 
  ∃ b : ℝ, (b^2 = 5) ∧ (∀ x y : ℝ, (y^2 / 4) - (x^2 / b^2) = 1) :=
sorry

end hyperbola_equation_l254_254787


namespace expression_change_l254_254407

variable (x b : ℝ)

-- The conditions
def expression (x : ℝ) : ℝ := x^3 - 5 * x + 1
def expr_change_plus (x b : ℝ) : ℝ := (x + b)^3 - 5 * (x + b) + 1
def expr_change_minus (x b : ℝ) : ℝ := (x - b)^3 - 5 * (x - b) + 1

-- The Lean statement to prove
theorem expression_change (h_b_pos : 0 < b) :
  expr_change_plus x b - expression x = 3 * b * x^2 + 3 * b^2 * x + b^3 - 5 * b ∨ 
  expr_change_minus x b - expression x = -3 * b * x^2 + 3 * b^2 * x - b^3 + 5 * b := 
by
  sorry

end expression_change_l254_254407


namespace cole_drive_time_l254_254585

theorem cole_drive_time (D : ℝ) (T_work T_home : ℝ) 
  (h1 : T_work = D / 75) 
  (h2 : T_home = D / 105)
  (h3 : T_work + T_home = 4) : 
  T_work * 60 = 140 := 
by sorry

end cole_drive_time_l254_254585


namespace min_omega_symmetry_l254_254432

theorem min_omega_symmetry :
  ∃ ω > 0, (∀ x : ℝ, sin (ω * x + ω * (π / 2) + π / 3) = sin ((-ω) * x + ω * (π / 2) + π / 3)) →
  ω = 1 / 3 :=
by {
  sorry
}

end min_omega_symmetry_l254_254432


namespace smallest_sum_arith_geo_seq_l254_254437

theorem smallest_sum_arith_geo_seq (A B C D : ℕ) 
  (h1 : A + B + C + D > 0)
  (h2 : 2 * B = A + C)
  (h3 : 16 * C = 7 * B)
  (h4 : 16 * D = 49 * B) :
  A + B + C + D = 97 :=
sorry

end smallest_sum_arith_geo_seq_l254_254437


namespace train_speed_ratio_l254_254701

variable (V1 V2 : ℝ)

theorem train_speed_ratio (H1 : V1 * 4 = D1) (H2 : V2 * 36 = D2) (H3 : D1 / D2 = 1 / 9) :
  V1 / V2 = 1 := 
by
  sorry

end train_speed_ratio_l254_254701


namespace selection_ways_l254_254280

namespace CulturalPerformance

-- Define basic conditions
def num_students : ℕ := 6
def can_sing : ℕ := 3
def can_dance : ℕ := 2
def both_sing_and_dance : ℕ := 1

-- Define the proof statement
theorem selection_ways :
  ∃ (ways : ℕ), ways = 15 := by
  sorry

end CulturalPerformance

end selection_ways_l254_254280


namespace solve_equation_l254_254552

theorem solve_equation (x : ℝ) (h : ((x^2 + 3*x + 4) / (x + 5)) = x + 6) : x = -13 / 4 :=
by sorry

end solve_equation_l254_254552


namespace max_value_expr_l254_254999

open Real

theorem max_value_expr {x y z : ℝ} (hx : 0 < x) (hy : 0 < y) (hz : 0 < z)
  (h : (x / y + y / z + z / x) + (y / x + z / y + x / z) = 9) : 
  (x / y + y / z + z / x) * (y / x + z / y + x / z) = 81 / 4 :=
sorry

end max_value_expr_l254_254999


namespace find_b3_b17_l254_254624

variable {a : ℕ → ℤ} -- Arithmetic sequence
variable {b : ℕ → ℤ} -- Geometric sequence

axiom arith_seq {a : ℕ → ℤ} (d : ℤ) : ∀ (n : ℕ), a (n + 1) = a n + d
axiom geom_seq {b : ℕ → ℤ} (r : ℤ) : ∀ (n : ℕ), b (n + 1) = b n * r

theorem find_b3_b17 
  (h_arith : ∃ d, ∀ n, a (n + 1) = a n + d) 
  (h_geom : ∃ r, ∀ n, b (n + 1) = b n * r)
  (h_cond1 : 3 * a 1 - (a 8)^2 + 3 * a 15 = 0)
  (h_cond2 : a 8 = b 10) :
  b 3 * b 17 = 36 := 
sorry

end find_b3_b17_l254_254624


namespace rental_plans_count_l254_254439

-- Define the number of large buses, medium buses, and the total number of people.
def num_large_buses := 42
def num_medium_buses := 25
def total_people := 1511

-- State the theorem to prove that there are exactly 2 valid rental plans.
theorem rental_plans_count (x y : ℕ) :
  (num_large_buses * x + num_medium_buses * y = total_people) →
  (∃! (x y : ℕ), num_large_buses * x + num_medium_buses * y = total_people) :=
by
  sorry

end rental_plans_count_l254_254439


namespace solve_for_x_l254_254765

theorem solve_for_x : ∃ x : ℤ, x + 1 = 5 ∧ x = 4 :=
by
  sorry

end solve_for_x_l254_254765


namespace maximize_profit_at_six_l254_254277

-- Defining the functions (conditions)
def y1 (x : ℝ) : ℝ := 17 * x^2
def y2 (x : ℝ) : ℝ := 2 * x^3 - x^2
def profit (x : ℝ) : ℝ := y1 x - y2 x

-- The condition x > 0
def x_pos (x : ℝ) : Prop := x > 0

-- Proving the maximum profit is achieved at x = 6 (question == answer)
theorem maximize_profit_at_six : ∀ x > 0, (∀ y > 0, y = profit x → x = 6) :=
by 
  intros x hx y hy
  sorry

end maximize_profit_at_six_l254_254277


namespace kanul_cash_percentage_l254_254519

theorem kanul_cash_percentage (raw_materials : ℕ) (machinery : ℕ) (total_amount : ℕ) (cash_percentage : ℕ)
  (H1 : raw_materials = 80000)
  (H2 : machinery = 30000)
  (H3 : total_amount = 137500)
  (H4 : cash_percentage = 20) :
  ((total_amount - (raw_materials + machinery)) * 100 / total_amount) = cash_percentage := by
    sorry

end kanul_cash_percentage_l254_254519


namespace sculpture_cost_in_cny_l254_254256

-- Define the equivalence rates
def usd_to_nad : ℝ := 8
def usd_to_cny : ℝ := 8

-- Define the cost of the sculpture in Namibian dollars
def sculpture_cost_nad : ℝ := 160

-- Theorem: Given the conversion rates, the sculpture cost in Chinese yuan is 160
theorem sculpture_cost_in_cny : (sculpture_cost_nad / usd_to_nad) * usd_to_cny = 160 :=
by sorry

end sculpture_cost_in_cny_l254_254256


namespace sum_of_values_of_N_l254_254438

theorem sum_of_values_of_N (N : ℂ) : (N * (N - 8) = 12) → (∃ x y : ℂ, N = x ∨ N = y ∧ x + y = 8) :=
by
  sorry

end sum_of_values_of_N_l254_254438


namespace jelly_beans_total_l254_254536

-- Definitions from the conditions
def vanilla : Nat := 120
def grape : Nat := 5 * vanilla + 50
def total : Nat := vanilla + grape

-- Statement to prove
theorem jelly_beans_total :
  total = 770 := 
by 
  sorry

end jelly_beans_total_l254_254536


namespace find_a_l254_254993

noncomputable def A := {x : ℝ | x^2 - 8 * x + 15 = 0}
noncomputable def B (a : ℝ) := {x : ℝ | a * x - 1 = 0}

theorem find_a (a : ℝ) : (A ∩ B a = B a) ↔ (a = 0 ∨ a = 1/3 ∨ a = 1/5) :=
by
  sorry

end find_a_l254_254993


namespace tangent_line_at_one_l254_254325

open Real

noncomputable def f (x : ℝ) : ℝ := x * log x

theorem tangent_line_at_one : ∀ x y, (x = 1 ∧ y = 0) → (x - y - 1 = 0) :=
by 
  intro x y h
  sorry

end tangent_line_at_one_l254_254325


namespace population_correct_individual_correct_sample_correct_sample_size_correct_l254_254446

-- Definitions based on the problem conditions
def Population : Type := {s : String // s = "all seventh-grade students in the city"}
def Individual : Type := {s : String // s = "each seventh-grade student in the city"}
def Sample : Type := {s : String // s = "the 500 students that were drawn"}
def SampleSize : ℕ := 500

-- Prove given conditions
theorem population_correct (p : Population) : p.1 = "all seventh-grade students in the city" :=
by sorry

theorem individual_correct (i : Individual) : i.1 = "each seventh-grade student in the city" :=
by sorry

theorem sample_correct (s : Sample) : s.1 = "the 500 students that were drawn" :=
by sorry

theorem sample_size_correct : SampleSize = 500 :=
by sorry

end population_correct_individual_correct_sample_correct_sample_size_correct_l254_254446


namespace minimum_omega_l254_254428

theorem minimum_omega (ω : ℝ) (h_omega_pos : ω > 0) :
    (∃ y : ℝ → ℝ, (∀ x, y x = sin (ω * x + ω * (π / 2) + (π / 3))) ∧ 
    (∀ x, y x = y (-x))) →
    (ω = 1 / 3) :=
sorry

end minimum_omega_l254_254428


namespace arithmetic_progression_primes_l254_254466

theorem arithmetic_progression_primes (p₁ p₂ p₃ : ℕ) (d : ℕ) 
  (hp₁ : Prime p₁) (hp₁_cond : 3 < p₁) 
  (hp₂ : Prime p₂) (hp₂_cond : 3 < p₂) 
  (hp₃ : Prime p₃) (hp₃_cond : 3 < p₃) 
  (h_prog_1 : p₂ = p₁ + d) (h_prog_2 : p₃ = p₁ + 2 * d) : 
  d % 6 = 0 :=
sorry

end arithmetic_progression_primes_l254_254466


namespace number_property_l254_254941

theorem number_property (n : ℕ) (h : n = 7101449275362318840579) :
  n / 7 = 101449275362318840579 :=
sorry

end number_property_l254_254941


namespace jake_pure_alcohol_l254_254236

theorem jake_pure_alcohol (total_shots : ℕ) (shots_per_split : ℕ) (ounces_per_shot : ℚ) (purity : ℚ) :
  total_shots = 8 →
  shots_per_split = 2 →
  ounces_per_shot = 1.5 →
  purity = 0.5 →
  (total_shots / shots_per_split) * ounces_per_shot * purity = 3 := 
by
  sorry

end jake_pure_alcohol_l254_254236


namespace lines_intersect_l254_254692

theorem lines_intersect (m b : ℝ) (h1 : 17 = 2 * m * 4 + 5) (h2 : 17 = 4 * 4 + b) : b + m = 2.5 :=
by {
    sorry
}

end lines_intersect_l254_254692


namespace probability_two_heads_and_die_three_l254_254864

-- Define the events
def coin_flip_outcomes := {flip₁ flip₂ flip₃ | flip₁ ∈ {0, 1} ∧ flip₂ ∈ {0, 1} ∧ flip₃ ∈ {0, 1}}
def exactly_two_heads (flips : list bool) : Prop := flips.count (· = true) = 2
def die_roll_outcomes := { 1, 2, 3, 4, 5, 6 }

-- Define probability space for coin flips and die rolls combined
def combined_outcomes := {c | c.flip₁ ∈ coin_flip_outcomes ∧ c.die_roll ∈ die_roll_outcomes}

theorem probability_two_heads_and_die_three :
  (finset.filter (λ o, exactly_two_heads o.flip₁ ∧ o.die_roll = 3) combined_outcomes).card / combined_outcomes.card = 1 / 16 := by
  sorry

end probability_two_heads_and_die_three_l254_254864


namespace number_of_comic_books_l254_254105

def fairy_tale_books := 305
def science_and_technology_books := fairy_tale_books + 115
def total_books := fairy_tale_books + science_and_technology_books
def comic_books := total_books * 4

theorem number_of_comic_books : comic_books = 2900 := by
  sorry

end number_of_comic_books_l254_254105


namespace number_of_factors_27648_l254_254355

-- Define the number in question
def n : ℕ := 27648

-- State the prime factorization
def n_prime_factors : Nat := 2^10 * 3^3

-- State the theorem to be proven
theorem number_of_factors_27648 : 
  ∃ (f : ℕ), 
  (f = (10+1) * (3+1)) ∧ (f = 44) :=
by
  -- Placeholder for the proof
  sorry

end number_of_factors_27648_l254_254355


namespace count_congruent_to_3_mod_8_l254_254964

theorem count_congruent_to_3_mod_8 : 
  ∃ (count : ℕ), count = 31 ∧ ∀ (x : ℕ), 1 ≤ x ∧ x ≤ 250 → x % 8 = 3 → x = 8 * ((x - 3) / 8) + 3 := sorry

end count_congruent_to_3_mod_8_l254_254964


namespace bisection_next_interval_l254_254699

-- Define the function f(x) = x^3 - 2x - 1
def f (x : ℝ) : ℝ := x^3 - 2*x - 1

-- Define the intervals (1, 2) and (1.5, 2)
def interval_initial : Set ℝ := {x | 1 < x ∧ x < 2}
def interval_next : Set ℝ := {x | 1.5 < x ∧ x < 2}

-- State the theorem, with conditions
theorem bisection_next_interval 
  (root_in_interval_initial : ∃ x, f x = 0 ∧ x ∈ interval_initial)
  (f_1_negative : f 1 < 0)
  (f_2_positive : f 2 > 0)
  : ∃ x, f x = 0 ∧ x ∈ interval_next :=
sorry

end bisection_next_interval_l254_254699


namespace seq_b_arithmetic_diff_seq_a_general_term_l254_254354

variable {n : ℕ}

def seq_a (a : ℕ → ℝ) : Prop :=
  a 1 = 1 ∧ ∀ n, a (n + 1) = 2 * a n / (a n + 2)

def seq_b (a : ℕ → ℝ) (b : ℕ → ℝ) : Prop :=
  ∀ n, b n = 1 / a n

theorem seq_b_arithmetic_diff (a : ℕ → ℝ) (b : ℕ → ℝ) 
  (h_a : seq_a a) (h_b : seq_b a b) :
  ∀ n, b (n + 1) - b n = 1 / 2 :=
by
  sorry

theorem seq_a_general_term (a : ℕ → ℝ) (h_a : seq_a a) :
  ∀ n, a n = 2 / (n + 1) :=
by
  sorry

end seq_b_arithmetic_diff_seq_a_general_term_l254_254354


namespace Ivan_bought_10_cards_l254_254066

-- Define variables and conditions
variables (x : ℕ) -- Number of Uno Giant Family Cards bought
def original_price : ℕ := 12
def discount_per_card : ℕ := 2
def discounted_price := original_price - discount_per_card
def total_paid : ℕ := 100

-- Lean 4 theorem statement
theorem Ivan_bought_10_cards (h : discounted_price * x = total_paid) : x = 10 := by
  -- proof goes here
  sorry

end Ivan_bought_10_cards_l254_254066


namespace probability_of_not_red_l254_254874

-- Definitions based on conditions
def total_number_of_jelly_beans : ℕ := 7 + 9 + 10 + 12 + 5
def number_of_non_red_jelly_beans : ℕ := 9 + 10 + 12 + 5

-- Proving the probability
theorem probability_of_not_red : 
  (number_of_non_red_jelly_beans : ℚ) / total_number_of_jelly_beans = 36 / 43 :=
by sorry

end probability_of_not_red_l254_254874


namespace fish_lifespan_is_12_l254_254116

def hamster_lifespan : ℝ := 2.5
def dog_lifespan : ℝ := 4 * hamster_lifespan
def fish_lifespan : ℝ := dog_lifespan + 2

theorem fish_lifespan_is_12 : fish_lifespan = 12 := by
  sorry

end fish_lifespan_is_12_l254_254116


namespace a_value_for_even_function_l254_254675

def f (x a : ℝ) := (x + 1) * (x + a)

theorem a_value_for_even_function (a : ℝ) (h : ∀ x, f x a = f (-x) a) : a = -1 :=
by
  sorry

end a_value_for_even_function_l254_254675


namespace least_possible_xy_l254_254778

theorem least_possible_xy (x y : ℕ) (hx : 0 < x) (hy : 0 < y)
  (h : 1 / x + 1 / (3 * y) = 1 / 6) : x * y = 48 :=
by
  sorry

end least_possible_xy_l254_254778


namespace math_problem_l254_254244

theorem math_problem (a b : ℝ) (h : a / (1 + a) + b / (1 + b) = 1) : 
  a / (1 + b^2) - b / (1 + a^2) = a - b := 
sorry

end math_problem_l254_254244


namespace isosceles_triangle_perimeter_l254_254069

-- Define the lengths of the sides
def side1 := 2 -- 2 cm
def side2 := 4 -- 4 cm

-- Define the condition of being isosceles
def is_isosceles (a b c : ℝ) : Prop := (a = b) ∨ (a = c) ∨ (b = c)

-- Define the triangle inequality
def triangle_inequality (a b c : ℝ) : Prop :=
  (a + b > c) ∧ (a + c > b) ∧ (b + c > a)

-- Define the triangle perimeter
def perimeter (a b c : ℝ) : ℝ := a + b + c

-- Define the main theorem to prove
theorem isosceles_triangle_perimeter {a b : ℝ} (ha : a = side1) (hb : b = side2)
    (h1 : is_isosceles a b c) (h2 : triangle_inequality a b c) : perimeter a b c = 10 :=
sorry

end isosceles_triangle_perimeter_l254_254069


namespace joe_cut_kids_hair_l254_254647

theorem joe_cut_kids_hair
  (time_women minutes_women count_women : ℕ)
  (time_men minutes_men count_men : ℕ)
  (time_kid minutes_kid : ℕ)
  (total_minutes: ℕ) : 
  minutes_women = 50 → 
  minutes_men = 15 →
  minutes_kid = 25 →
  count_women = 3 →
  count_men = 2 →
  total_minutes = 255 →
  (count_women * minutes_women + count_men * minutes_men + time_kid * minutes_kid) = total_minutes →
  time_kid = 3 :=
by
  intros h1 h2 h3 h4 h5 h6 h7
  -- Proof is not provided, hence stating sorry.
  sorry

end joe_cut_kids_hair_l254_254647


namespace remainder_when_150_divided_by_k_is_2_l254_254330

theorem remainder_when_150_divided_by_k_is_2
  (k : ℕ) (q : ℤ)
  (hk_pos : k > 0)
  (hk_condition : 120 = q * k^2 + 8) :
  150 % k = 2 :=
sorry

end remainder_when_150_divided_by_k_is_2_l254_254330


namespace total_jelly_beans_l254_254531

-- Given conditions:
def vanilla_jelly_beans : ℕ := 120
def grape_jelly_beans := 5 * vanilla_jelly_beans + 50

-- The proof problem:
theorem total_jelly_beans :
  vanilla_jelly_beans + grape_jelly_beans = 770 :=
by
  -- Proof steps would go here
  sorry

end total_jelly_beans_l254_254531
