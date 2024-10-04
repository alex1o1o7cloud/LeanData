import Mathlib

namespace probability_A_and_B_selected_l478_478518

theorem probability_A_and_B_selected :
  let n := 5
  let r := 3
  let total_ways := Nat.choose n r
  let favorable_ways := Nat.choose 3 1
  let probability := favorable_ways / total_ways
  probability = (3 / 10) := 
by
  sorry

end probability_A_and_B_selected_l478_478518


namespace prob_select_A_and_B_l478_478616

theorem prob_select_A_and_B : 
  let total_students := 5
  let to_select := 3
  let favorable_outcomes := Nat.div_fact (Nat.factorial 3) (Nat.factorial 1)
  let total_outcomes := Nat.div_fact (Nat.factorial 5) (Nat.factorial 3 * Nat.factorial (5 - 3))
  (favorable_outcomes : ℚ) / total_outcomes = 3 / 10 :=
by
  sorry

end prob_select_A_and_B_l478_478616


namespace probability_both_A_and_B_selected_l478_478399

-- The universe of students we consider
inductive Student
| A : Student
| B : Student
| C : Student
| D : Student
| E : Student

open Student

-- Function to count combinations
def combinations (n k : ℕ) : ℕ := (Nat.factorial n) / ((Nat.factorial k) * (Nat.factorial (n - k)))

-- The set of all students
def students : List Student := [A, B, C, D, E]

-- The total number of ways to select 3 students from 5
def total_selections := combinations 5 3

-- The number of ways A and B are both selected by selecting 1 more from the remaining 3 students
def selections_AB := combinations 3 1

-- The probability that both A and B are selected
def probability_AB := (selections_AB : ℚ) / (total_selections : ℚ)

-- The proof statement
theorem probability_both_A_and_B_selected :
  probability_AB = 3 / 10 :=
sorry

end probability_both_A_and_B_selected_l478_478399


namespace factorize_quadratic_l478_478176

theorem factorize_quadratic (x : ℝ) : 
  16 * x^2 - 40 * x + 25 = (4 * x - 5)^2 :=
sorry

end factorize_quadratic_l478_478176


namespace acute_angle_MN_PA_l478_478804

-- Define the parameters of the triangle.
variables (P A T U G M N : Type)
variables (PA PU AG : ℝ)
variables (angle_P angle_A angle_T : ℝ)

-- Initial conditions
axiom angle_P_def : angle_P = 36
axiom angle_A_def : angle_A = 56
axiom PA_def : PA = 10
axiom PU_def : PU = 1
axiom AG_def : AG = 1
axiom M_midpoint : M = midpoint P A
axiom N_midpoint : N = midpoint U G

-- Target acute angle between MN and PA is 80 degrees
theorem acute_angle_MN_PA : 
  acute_angle (angle_between MN PA) = 80 := 
sory

end acute_angle_MN_PA_l478_478804


namespace total_matches_played_l478_478947

theorem total_matches_played :
  ∀ (runs1 runs2 runs_all avg1 avg2 avg_all matches1 matches2 total_matches : ℕ),
  avg1 = 50 → matches1 = 30 →
  avg2 = 26 → matches2 = 15 →
  avg_all = 42 →
  runs1 = avg1 * matches1 →
  runs2 = avg2 * matches2 →
  runs_all = runs1 + runs2 →
  total_matches = runs_all / avg_all →
  total_matches = 45 :=
by
  intros runs1 runs2 runs_all avg1 avg2 avg_all matches1 matches2 total_matches
  assume h_avg1 h_matches1 h_avg2 h_matches2 h_avg_all h_runs1 h_runs2 h_runs_all h_total_matches
  rw [h_avg1, h_matches1, h_avg2, h_matches2, h_avg_all, h_runs1, h_runs2, h_runs_all] at h_total_matches
  sorry

end total_matches_played_l478_478947


namespace bob_shucking_rate_l478_478117

theorem bob_shucking_rate (H1 : 10 / 5 = 2) (H2 : 240 = (2 * 60) * 2) :
  240 / 120 = 2 := by
  rw [←H2] at *
  rw [H1]
  linarith

end bob_shucking_rate_l478_478117


namespace probability_of_selecting_A_and_B_l478_478670

open_locale classical

noncomputable def select_probability (n k : ℕ) : ℚ :=
  (nat.choose n k)⁻¹

theorem probability_of_selecting_A_and_B :
  let total_students := 5,
      students_to_select := 3,
      remaining_students := 3 in
  select_probability total_students students_to_select * nat.choose remaining_students 1 = 3 / 10 :=
by
  sorry

end probability_of_selecting_A_and_B_l478_478670


namespace largest_circle_radius_in_square_outside_semicircle_l478_478812

theorem largest_circle_radius_in_square_outside_semicircle 
  (ABCD_is_square : ∀ (A B C D : ℝ → ℝ), is_square A B C D 2)
  (semicircle_diameter_CD : ∃ (P : ℝ × ℝ), is_semicircle_diameter (P.1, P.2) C D) :
  exists (r : ℝ), r = 4 - 2 * real.sqrt 3 ∧ 
    (∀ (O : ℝ × ℝ), is_circle_inside_square_outside_semicircle O r A B C D P) :=
sorry

end largest_circle_radius_in_square_outside_semicircle_l478_478812


namespace blue_tshirt_count_per_pack_l478_478138

theorem blue_tshirt_count_per_pack :
  ∀ (total_tshirts white_packs blue_packs tshirts_per_white_pack tshirts_per_blue_pack : ℕ), 
    white_packs = 3 →
    blue_packs = 2 → 
    tshirts_per_white_pack = 6 → 
    total_tshirts = 26 →
    total_tshirts = white_packs * tshirts_per_white_pack + blue_packs * tshirts_per_blue_pack →
  tshirts_per_blue_pack = 4 :=
by
  intros total_tshirts white_packs blue_packs tshirts_per_white_pack tshirts_per_blue_pack
  intros h1 h2 h3 h4 h5
  sorry

end blue_tshirt_count_per_pack_l478_478138


namespace nonzero_terms_in_expansion_l478_478124

-- Define the polynomials
def poly1 : Polynomial ℚ := 2 * Polynomial.X + 3
def poly2 : Polynomial ℚ := Polynomial.X^2 + 4 * Polynomial.X + 5
def poly3 : Polynomial ℚ := Polynomial.X^3 - Polynomial.X^2 + 2 * Polynomial.X + 1

-- Define the expression
def expression : Polynomial ℚ := poly1 * poly2 - 4 * poly3

-- Count nonzero terms
def num_nonzero_terms (p : Polynomial ℚ) : ℕ :=
  p.coeffs.filter (λ c => c ≠ 0).length

-- State the proof problem
theorem nonzero_terms_in_expansion : num_nonzero_terms expression = 4 := by
  sorry

end nonzero_terms_in_expansion_l478_478124


namespace probability_A_and_B_selected_l478_478377

theorem probability_A_and_B_selected 
  (students : Finset ℕ) (A B : ℕ) 
  (h_students_size : students.card = 5) 
  (h_A_in_students : A ∈ students) 
  (h_B_in_students : B ∈ students)
  : (↑(Nat.choose 3) : ℚ) / (↑(Nat.choose 5) : ℚ) = 3 / 10 := 
by sorry

end probability_A_and_B_selected_l478_478377


namespace domain_of_f_f_of_f_of_1_l478_478755

def f (x : ℝ) : ℝ := (1 - x^2) / (1 + x^2)

theorem domain_of_f : ∀ x : ℝ, (1 + x^2) ≠ 0 := by
  intro x
  apply ne_of_gt
  apply add_pos_of_pos_of_nonneg
  exact one_pos
  apply pow_two_nonneg

theorem f_of_f_of_1 : f (f 1) = 1 := by
  calc
    f 1 = (1 - 1^2) / (1 + 1^2) := rfl
    ... = 0 : by norm_num
    f 0 = (1 - 0^2) / (1 + 0^2) := rfl
    ... = 1 : by norm_num

end domain_of_f_f_of_f_of_1_l478_478755


namespace probability_A_and_B_selected_l478_478533

theorem probability_A_and_B_selected :
  let n := 5
  let r := 3
  let total_ways := Nat.choose n r
  let favorable_ways := Nat.choose 3 1
  let probability := favorable_ways / total_ways
  probability = (3 / 10) := 
by
  sorry

end probability_A_and_B_selected_l478_478533


namespace prob_select_A_and_B_l478_478654

theorem prob_select_A_and_B (students : Finset ℕ) (A B : ℕ) (hA : A ∈ students) (hB : B ∈ students) (h_card : students.card = 5) :
  (students.choose 3).count {comb | A ∈ comb ∧ B ∈ comb} / (students.choose 3).card = 3 / 10 := sorry

end prob_select_A_and_B_l478_478654


namespace Julie_monthly_salary_l478_478807

theorem Julie_monthly_salary 
(hourly_rate : ℕ) (hours_per_day : ℕ) (days_per_week : ℕ) (weeks_per_month : ℕ) (missed_days : ℕ) 
(h1 : hourly_rate = 5) (h2 : hours_per_day = 8) 
(h3 : days_per_week = 6) (h4 : weeks_per_month = 4) 
(h5 : missed_days = 1) : 
hourly_rate * hours_per_day * days_per_week * weeks_per_month - hourly_rate * hours_per_day * missed_days = 920 :=
by sorry

end Julie_monthly_salary_l478_478807


namespace incenter_inequality_l478_478816

/-- Let ABC be a triangle with incenter I. If P is a point inside the triangle such that
  ∠PBA + ∠PCA = ∠PBC + ∠PCB, then AP ≥ AI, and equality holds if and only if P = I. -/
theorem incenter_inequality (A B C I P : Point) (h_incenter : Incenter I A B C)
 (h_point : Inside P (Triangle A B C))
 (h_angles : angle P B A + angle P C A = angle P B C + angle P C B) :
 AP ≥ AI ∧ (AP = AI ↔ P = I) :=
sorry

end incenter_inequality_l478_478816


namespace relationship_between_y1_y2_l478_478778

theorem relationship_between_y1_y2 (y1 y2 : ℝ) :
    (y1 = -3 * 2 + 4 ∧ y2 = -3 * (-1) + 4) → y1 < y2 :=
by
  sorry

end relationship_between_y1_y2_l478_478778


namespace probability_of_selecting_A_and_B_l478_478259

-- Define the combination function
def combination (n r : ℕ) : ℕ := Nat.factorial n / (Nat.factorial r * Nat.factorial (n - r))

-- Define the conditions
def students : ℕ := 5
def to_select : ℕ := 3

-- Define the favorable outcomes where A and B are selected
def favorable_outcomes : ℕ := combination 3 1

-- Define the total outcomes of selecting 3 out of 5 students
def total_outcomes : ℕ := combination 5 3

-- Define the probability calculation
def probability_of_A_and_B : ℚ := favorable_outcomes / total_outcomes

-- Proposition to prove
theorem probability_of_selecting_A_and_B : probability_of_A_and_B = 3 / 10 :=
by
  sorry

end probability_of_selecting_A_and_B_l478_478259


namespace sqrt_b2_sub_ac_lt_sqrt3a_l478_478897

theorem sqrt_b2_sub_ac_lt_sqrt3a
  (a b c : ℝ)
  (h1 : a > b)
  (h2 : b > c)
  (h3 : a + b + c = 0) :
  sqrt (b^2 - a * c) < sqrt (3) * a :=
sorry

end sqrt_b2_sub_ac_lt_sqrt3a_l478_478897


namespace probability_of_A_and_B_selected_l478_478475

variable {α : Type}

/-- From 5 students, we randomly select 3 students to participate in community service work. 
  Let A and B be two specific students among these five. The probability that 
  both A and B are selected is 3 / 10. -/
theorem probability_of_A_and_B_selected (A B : α) (s : Finset α) (h : s.card = 5) :
  (∃ t : Finset α, t ⊆ s ∧ A ∈ t ∧ B ∈ t ∧ (t.card = 3) ∧ (t.card = 3) →
   (@Finset.exists_subset_card_eq α t s 3) →
   (@Finset.exists_subset_card_eq α t (Finset.erase s A) 2) ∧
   (@Finset.exists_subset_card_eq α t (Finset.erase (Finset.erase s A) B) 1)) →
  (3 : ℚ) / 10 :=
by
  sorry -- Proof omitted as per instructions.

end probability_of_A_and_B_selected_l478_478475


namespace probability_of_selecting_A_and_B_l478_478677

open_locale classical

noncomputable def select_probability (n k : ℕ) : ℚ :=
  (nat.choose n k)⁻¹

theorem probability_of_selecting_A_and_B :
  let total_students := 5,
      students_to_select := 3,
      remaining_students := 3 in
  select_probability total_students students_to_select * nat.choose remaining_students 1 = 3 / 10 :=
by
  sorry

end probability_of_selecting_A_and_B_l478_478677


namespace max_average_speed_interval_l478_478017

theorem max_average_speed_interval :
  let distance : ℕ → ℕ := λ t,
    match t with
    | 0 => 0
    | 1 => 50
    | 2 => 120
    | 3 => 210
    | 4 => 320
    | _ => 0
    in
  let average_speed : ℕ → ℕ := λ t, distance t - distance (t - 1) 
    in
  ∀ t : ℕ, t ≥ 1 → t ≤ 4 → ((5 - (average_speed 4) < (average_speed t))  ) = (t = 4)
    := sorry

end max_average_speed_interval_l478_478017


namespace fence_poles_needed_l478_478956

theorem fence_poles_needed 
    (side1 side2 side3 side4 : ℕ)
    (distance1 distance2 : ℕ)
    (h1 : side1 = 100) (h2 : side2 = 80) (h3 : side3 = 90) (h4 : side4 = 70)
    (h_dist1 : distance1 = 15) (h_dist2 : distance2 = 10) : 
    (let segments_needed (side length : ℕ) : ℕ := (side + length - 1) / length + 1 in
    segments_needed side1 distance1 + 
    segments_needed side2 distance1 + 
    segments_needed side3 distance2 + 
    segments_needed side4 distance2) - 3 = 33 :=
by {
    -- skip proof for now
    sorry
}

end fence_poles_needed_l478_478956


namespace probability_of_selecting_A_and_B_l478_478269

-- Define the combination function
def combination (n r : ℕ) : ℕ := Nat.factorial n / (Nat.factorial r * Nat.factorial (n - r))

-- Define the conditions
def students : ℕ := 5
def to_select : ℕ := 3

-- Define the favorable outcomes where A and B are selected
def favorable_outcomes : ℕ := combination 3 1

-- Define the total outcomes of selecting 3 out of 5 students
def total_outcomes : ℕ := combination 5 3

-- Define the probability calculation
def probability_of_A_and_B : ℚ := favorable_outcomes / total_outcomes

-- Proposition to prove
theorem probability_of_selecting_A_and_B : probability_of_A_and_B = 3 / 10 :=
by
  sorry

end probability_of_selecting_A_and_B_l478_478269


namespace minimum_flights_are_five_l478_478864

-- Define a function that constructs the initial one-way flights circle.
def initial_circle_flights : set (ℕ × ℕ) := 
  { (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 1) }

-- Define the cities
def cities : set ℕ := {1, 2, 3, 4, 5, 6, 7}

-- Define the function to add the minimum number of flights.
def add_minimum_flights (flights : set (ℕ × ℕ)) : Prop :=
  ∃ additional_flights : set (ℕ × ℕ), 
      additional_flights.card = 5 ∧ 
      ∀ x y ∈ cities, 
        x ≠ y → 
        (∃ path : list ℕ, 
          path.head = x ∧ path.ilast = y ∧ path.length ≤ 3 ∧ 
          all_edges_in path (flights ∪ additional_flights))

-- Define the edges predicate for checking if all the edges in the path exist
def all_edges_in (path : list ℕ) (edges : set (ℕ × ℕ)) : Prop :=
  ∀ i ∈ finset.range (path.length - 1), 
    (path.nth_le i sorry, path.nth_le (i + 1) sorry) ∈ edges

-- Prove the minimum flights needed.
theorem minimum_flights_are_five : add_minimum_flights initial_circle_flights := 
sorry

end minimum_flights_are_five_l478_478864


namespace Miss_Darlington_total_blueberries_l478_478841

-- Conditions
def initial_basket := 20
def additional_baskets := 9

-- Definition and statement to be proved
theorem Miss_Darlington_total_blueberries :
  initial_basket + additional_baskets * initial_basket = 200 :=
by
  sorry

end Miss_Darlington_total_blueberries_l478_478841


namespace prob_select_A_and_B_l478_478631

theorem prob_select_A_and_B : 
  let total_students := 5
  let to_select := 3
  let favorable_outcomes := Nat.div_fact (Nat.factorial 3) (Nat.factorial 1)
  let total_outcomes := Nat.div_fact (Nat.factorial 5) (Nat.factorial 3 * Nat.factorial (5 - 3))
  (favorable_outcomes : ℚ) / total_outcomes = 3 / 10 :=
by
  sorry

end prob_select_A_and_B_l478_478631


namespace probability_A_and_B_selected_l478_478385

theorem probability_A_and_B_selected 
  (students : Finset ℕ) (A B : ℕ) 
  (h_students_size : students.card = 5) 
  (h_A_in_students : A ∈ students) 
  (h_B_in_students : B ∈ students)
  : (↑(Nat.choose 3) : ℚ) / (↑(Nat.choose 5) : ℚ) = 3 / 10 := 
by sorry

end probability_A_and_B_selected_l478_478385


namespace positive_number_from_square_roots_l478_478784

theorem positive_number_from_square_roots (x n : ℕ) (h1 : (x + 1) = real.sqrt n) (h2 : (4 - 2 * x) = real.sqrt n) : n = 36 := by
    sorry

end positive_number_from_square_roots_l478_478784


namespace probability_of_selecting_A_and_B_l478_478661

open_locale classical

noncomputable def select_probability (n k : ℕ) : ℚ :=
  (nat.choose n k)⁻¹

theorem probability_of_selecting_A_and_B :
  let total_students := 5,
      students_to_select := 3,
      remaining_students := 3 in
  select_probability total_students students_to_select * nat.choose remaining_students 1 = 3 / 10 :=
by
  sorry

end probability_of_selecting_A_and_B_l478_478661


namespace probability_of_selecting_A_and_B_l478_478544

theorem probability_of_selecting_A_and_B (students : Finset ℕ) (A B : ℕ)
  (h_size : students.card = 5) (h_in : A ∈ students ∧ B ∈ students) : 
  (Finset.card {s : Finset ℕ // s ⊆ students ∧ s.card = 3 ∧ A ∈ s ∧ B ∈ s}.to_set) / 
  (Finset.card {s : Finset ℕ // s ⊆ students ∧ s.card = 3}.to_set) = 3 / 10 :=
sorry

end probability_of_selecting_A_and_B_l478_478544


namespace repeated_operations_l478_478948

theorem repeated_operations (x : ℝ) (n : ℕ) (hx : x ≠ 0) : 
  let y := (iterate (λ z, 1 / ((z^2)^3)) n x) in 
  y = x ^ (-6)^n :=
by {
  sorry
}

end repeated_operations_l478_478948


namespace find_initial_pineapples_l478_478910

-- Definitions for the conditions stated in step a)
def sold_pineapples : ℕ := 48
def rotten_pineapples : ℕ := 9
def fresh_pineapples_left : ℕ := 29

-- The number of initially present pineapples to be proven
def initial_pineapples (P : ℕ) :=
  P = sold_pineapples + (fresh_pineapples_left + rotten_pineapples)

-- The statement to prove
theorem find_initial_pineapples : ∃ P : ℕ, initial_pineapples P = 86 :=
by 
  sorry

end find_initial_pineapples_l478_478910


namespace disproved_option_a_disproved_option_b_disproved_option_c_proved_option_d_l478_478933

theorem disproved_option_a : ¬ ∀ m n : ℤ, abs m = abs n → m = n := 
by 
  intro h
  have h1 : abs (-3) = abs 3 := by norm_num
  have h2 : h (-3) 3 h1
  contradiction

theorem disproved_option_b : ¬ ∀ m n : ℤ, m > n → abs m > abs n := 
by 
  intro h
  have h1 : 1 > -3 := by norm_num
  have h2 : abs 1 = 1 := by norm_num
  have h3 : abs (-3) = 3 := by norm_num
  have h4 : h 1 (-3) h1
  contradiction

theorem disproved_option_c : ¬ ∀ m n : ℤ, abs m > abs n → m > n :=
by 
  intro h
  have h1 : abs (-3) > abs 1 := by norm_num
  have h2 : (-3) < 1 := by norm_num
  exact not_lt_of_gt h2 (h (-3) 1 h1)

theorem proved_option_d : ∀ m n : ℤ, m < n → n < 0 → abs m > abs n := 
by 
  intro m n hmn hn0
  simp only [abs_lt, int.lt_neg, int.abs]
  have h1 : -n < -m := by exact neg_lt_neg hmn
  exact_mod_cast h1

end disproved_option_a_disproved_option_b_disproved_option_c_proved_option_d_l478_478933


namespace exchange_candies_l478_478846

-- Define the problem conditions and calculate the required values
def chocolates := 7
def caramels := 9
def exchange := 5

-- Combinatorial function to calculate binomial coefficient
def binomial (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem exchange_candies (h1 : chocolates = 7) (h2 : caramels = 9) (h3 : exchange = 5) :
  binomial chocolates exchange * binomial caramels exchange = 2646 := by
  sorry

end exchange_candies_l478_478846


namespace quadratic_points_relation_l478_478729

theorem quadratic_points_relation (c y1 y2 y3 : ℝ) :
  (A : ℝ × ℝ → Prop) (B : ℝ × ℝ → Prop) (C : ℝ × ℝ → Prop)
  (hA : A (-1, y1))
  (hB : B (1, y2))
  (hC : C (4, y3)) 
  (A_def : A = λ (p : ℝ × ℝ), p.2 = p.1^2 - 6 * p.1 + c)
  (B_def : B = λ (p : ℝ × ℝ), p.2 = p.1^2 - 6 * p.1 + c)
  (C_def : C = λ (p : ℝ × ℝ), p.2 = p.1^2 - 6 * p.1 + c) :
  y1 > y2 ∧ y2 > y3 :=
by
  sorry

end quadratic_points_relation_l478_478729


namespace probability_of_selecting_A_and_B_l478_478263

-- Define the combination function
def combination (n r : ℕ) : ℕ := Nat.factorial n / (Nat.factorial r * Nat.factorial (n - r))

-- Define the conditions
def students : ℕ := 5
def to_select : ℕ := 3

-- Define the favorable outcomes where A and B are selected
def favorable_outcomes : ℕ := combination 3 1

-- Define the total outcomes of selecting 3 out of 5 students
def total_outcomes : ℕ := combination 5 3

-- Define the probability calculation
def probability_of_A_and_B : ℚ := favorable_outcomes / total_outcomes

-- Proposition to prove
theorem probability_of_selecting_A_and_B : probability_of_A_and_B = 3 / 10 :=
by
  sorry

end probability_of_selecting_A_and_B_l478_478263


namespace probability_A_and_B_selected_l478_478687

/-- There are 5 students including A and B. The probability of selecting both A and B 
    when 3 students are selected randomly from 5 is 3/10. -/
theorem probability_A_and_B_selected :
  (∃ (students : List String) (h : students.length = 5),
    ∃ (selected : List String) (h' : selected.length = 3),
    "A" ∈ selected ∧ "B" ∈ selected) →
  ∃ (P : ℚ), P = 3 / 10 :=
by
  sorry

end probability_A_and_B_selected_l478_478687


namespace probability_A_B_l478_478454

open Nat

-- Define a function for combination
def combination (n r : Nat) : Nat :=
  factorial n / (factorial r * factorial (n - r))

-- Define the set of students and the selection conditions
def num_students : Nat := 5
def num_selected : Nat := 3

-- Define A and B as selected students
def selected_A_B : Nat := combination 3 1

-- Define the probability calculation
def probability_A_B_selected : Rat :=
  (selected_A_B : Rat) / (combination num_students num_selected : Rat)

-- The statement of the problem
theorem probability_A_B : probability_A_B_selected = 3 / 10 := by
  sorry

end probability_A_B_l478_478454


namespace sqrt_500_simplified_l478_478871

theorem sqrt_500_simplified : sqrt 500 = 10 * sqrt 5 :=
by
sorry

end sqrt_500_simplified_l478_478871


namespace probability_A_and_B_selected_l478_478230

theorem probability_A_and_B_selected (students : Finset ℕ) (A B : ℕ) (h : A ∈ students ∧ B ∈ students ∧ students.card = 5) : 
  (students.choose 3).count (λ s, {A, B} ⊆ s) / (students.choose 3).card = 3 / 10 := by
  sorry

end probability_A_and_B_selected_l478_478230


namespace smallest_four_digit_number_with_unique_digits_and_second_digit_six_l478_478845

theorem smallest_four_digit_number_with_unique_digits_and_second_digit_six : 
  ∃ (n : ℕ), nat.digits 10 n = [1, 6, 0, 2] ∧ 
             1000 ≤ n ∧ 
             n ≤ 9999 ∧ 
             (nat.digits 10 n).nodup ∧
             (nat.digits 10 n).nth 1 = some 6 :=
begin
  sorry
end

end smallest_four_digit_number_with_unique_digits_and_second_digit_six_l478_478845


namespace probability_A_and_B_selected_l478_478241

theorem probability_A_and_B_selected (students : Finset ℕ) (A B : ℕ) (h : A ∈ students ∧ B ∈ students ∧ students.card = 5) : 
  (students.choose 3).count (λ s, {A, B} ⊆ s) / (students.choose 3).card = 3 / 10 := by
  sorry

end probability_A_and_B_selected_l478_478241


namespace prob_select_A_and_B_l478_478630

theorem prob_select_A_and_B : 
  let total_students := 5
  let to_select := 3
  let favorable_outcomes := Nat.div_fact (Nat.factorial 3) (Nat.factorial 1)
  let total_outcomes := Nat.div_fact (Nat.factorial 5) (Nat.factorial 3 * Nat.factorial (5 - 3))
  (favorable_outcomes : ℚ) / total_outcomes = 3 / 10 :=
by
  sorry

end prob_select_A_and_B_l478_478630


namespace a_minus_b_a_squared_lt_zero_sufficient_not_necessary_for_a_lt_b_l478_478831

theorem a_minus_b_a_squared_lt_zero_sufficient_not_necessary_for_a_lt_b
  (a b : ℝ) :
  (∀ a b : ℝ, (a - b) * a ^ 2 < 0 → a < b) ∧ 
  (¬∀ a b : ℝ, a < b → (a - b) * a ^ 2 < 0) :=
sorry

end a_minus_b_a_squared_lt_zero_sufficient_not_necessary_for_a_lt_b_l478_478831


namespace probability_of_selecting_A_and_B_l478_478300

open_locale classical

-- Define the number of students
def num_students := 5

-- Define the number of students to be selected
def students_to_select := 3

def combination (n k : ℕ) : ℕ := nat.choose n k

-- Define the correct answer
def correct_probability := 3 / 10

-- Define the probability calculation
def calc_probability := (combination 3 1) / (combination 5 3)

-- Statement of the theorem to be proved
theorem probability_of_selecting_A_and_B :
  calc_probability = correct_probability :=
by sorry

end probability_of_selecting_A_and_B_l478_478300


namespace student_A_incorrect_l478_478797

def is_on_circle (center : ℝ × ℝ) (radius : ℝ) (point : ℝ × ℝ) : Prop :=
  let (cx, cy) := center
  let (px, py) := point
  (px - cx)^2 + (py - cy)^2 = radius^2

def center : ℝ × ℝ := (2, -3)
def radius : ℝ := 5
def point_A : ℝ × ℝ := (-2, -1)
def point_D : ℝ × ℝ := (5, 1)

theorem student_A_incorrect :
  ¬ is_on_circle center radius point_A ∧ is_on_circle center radius point_D :=
by
  sorry

end student_A_incorrect_l478_478797


namespace shaded_area_fraction_l478_478994

theorem shaded_area_fraction :
  let A := (0, 0)
  let B := (4, 0)
  let C := (4, 4)
  let D := (0, 4)
  let P := ((A.1 + B.1) / 2, (A.2 + B.2) / 2)
  let Q := ((B.1 + C.1) / 2, (B.2 + C.2) / 2)
  let R := ((C.1 + D.1) / 2, (C.2 + D.2) / 2)
  let S := ((D.1 + A.1) / 2, (D.2 + A.2) / 2)
  let area_triangle := 1 / 2 * 2 * 2
  let shaded_area := 2 * area_triangle
  let total_area := 4 * 4
  shaded_area / total_area = 1 / 4 :=
by
  sorry

end shaded_area_fraction_l478_478994


namespace weightlifter_one_hand_l478_478973

theorem weightlifter_one_hand (total_weight : ℕ) (h : total_weight = 20) (even_distribution : total_weight % 2 = 0) : total_weight / 2 = 10 :=
by
  sorry

end weightlifter_one_hand_l478_478973


namespace sum_of_altitudes_eq_20_27_l478_478997

noncomputable def line : ℝ → ℝ → Prop := λ x y, 8 * x + 7 * y = 56

theorem sum_of_altitudes_eq_20_27 :
  ∃ (a b c : ℝ), line a 0 ∧ line 0 b ∧ (1 / 2) * a * b = 28 ∧ 
  sqrt (a^2 + b^2) * c = 56 ∧ (a + b + c = 20.27) :=
sorry

end sum_of_altitudes_eq_20_27_l478_478997


namespace probability_A_and_B_selected_l478_478375

theorem probability_A_and_B_selected 
  (students : Finset ℕ) (A B : ℕ) 
  (h_students_size : students.card = 5) 
  (h_A_in_students : A ∈ students) 
  (h_B_in_students : B ∈ students)
  : (↑(Nat.choose 3) : ℚ) / (↑(Nat.choose 5) : ℚ) = 3 / 10 := 
by sorry

end probability_A_and_B_selected_l478_478375


namespace probability_of_selecting_A_and_B_l478_478282

open_locale classical

-- Define the number of students
def num_students := 5

-- Define the number of students to be selected
def students_to_select := 3

def combination (n k : ℕ) : ℕ := nat.choose n k

-- Define the correct answer
def correct_probability := 3 / 10

-- Define the probability calculation
def calc_probability := (combination 3 1) / (combination 5 3)

-- Statement of the theorem to be proved
theorem probability_of_selecting_A_and_B :
  calc_probability = correct_probability :=
by sorry

end probability_of_selecting_A_and_B_l478_478282


namespace rubiks_cube_repetition_l478_478995

/-- Define the parameters of the problem:
  - x_move: the move X which rotates a layer 90 degrees clockwise around the X axis.
  - y_move: the move Y which rotates a layer 90 degrees clockwise around the Y axis.
  - move M is performing X followed by Y.
  - We define the repetition property of move M.
  - We want to show that the smallest number of repetitions of M to return the cube to the original state is 28.
--/
variables (X Y : ℕ) (M : ℕ)

theorem rubiks_cube_repetition :
  is_move_X X ∧ is_move_Y Y ∧ is_move_M M ∧ M = X + Y →
  (∀ n, rotate_cubies X Y n ≃ initial_state ↔ n = 28) := sorry

end rubiks_cube_repetition_l478_478995


namespace probability_of_selecting_A_and_B_l478_478281

open_locale classical

-- Define the number of students
def num_students := 5

-- Define the number of students to be selected
def students_to_select := 3

def combination (n k : ℕ) : ℕ := nat.choose n k

-- Define the correct answer
def correct_probability := 3 / 10

-- Define the probability calculation
def calc_probability := (combination 3 1) / (combination 5 3)

-- Statement of the theorem to be proved
theorem probability_of_selecting_A_and_B :
  calc_probability = correct_probability :=
by sorry

end probability_of_selecting_A_and_B_l478_478281


namespace probability_A_and_B_selected_l478_478577

/-- Let there be a group of 5 students including A and B. 
    If 3 students are selected randomly from these 5, the probability of selecting both A and B is 3/10. -/
theorem probability_A_and_B_selected : 
  let students := 5 in
  let selected := 3 in
  let combinations := students.choose selected in
  let favorable_combinations := (students - 2).choose (selected - 2) in
  let probability := (favorable_combinations : ℚ) / (combinations : ℚ) in
  probability = 3 / 10 := 
by sorry

end probability_A_and_B_selected_l478_478577


namespace transport_load_with_trucks_l478_478089

theorem transport_load_with_trucks
  (total_weight : ℕ)
  (box_max_weight : ℕ)
  (truck_capacity : ℕ)
  (num_trucks : ℕ)
  (H_weight : total_weight = 13500)
  (H_box : box_max_weight = 350)
  (H_truck : truck_capacity = 1500)
  (H_num_trucks : num_trucks = 11) :
  ∃ (boxes : ℕ), boxes * box_max_weight >= total_weight ∧ num_trucks * truck_capacity >= total_weight := 
sorry

end transport_load_with_trucks_l478_478089


namespace probability_of_selecting_A_and_B_l478_478662

open_locale classical

noncomputable def select_probability (n k : ℕ) : ℚ :=
  (nat.choose n k)⁻¹

theorem probability_of_selecting_A_and_B :
  let total_students := 5,
      students_to_select := 3,
      remaining_students := 3 in
  select_probability total_students students_to_select * nat.choose remaining_students 1 = 3 / 10 :=
by
  sorry

end probability_of_selecting_A_and_B_l478_478662


namespace num_sequences_equality_l478_478764

theorem num_sequences_equality : 
  let letters := ['E', 'Q', 'U', 'A', 'L', 'I']
  ∀ sequences : List Char, sequences.head = 'Y' ∧ sequences.last = 'T' ∧ NoDup sequences ∧ sequences.length = 6 
  → sequences = 6 * 5 * 4 * 3 := 
by
  sorry

end num_sequences_equality_l478_478764


namespace probability_A_and_B_selected_l478_478522

theorem probability_A_and_B_selected :
  let n := 5
  let r := 3
  let total_ways := Nat.choose n r
  let favorable_ways := Nat.choose 3 1
  let probability := favorable_ways / total_ways
  probability = (3 / 10) := 
by
  sorry

end probability_A_and_B_selected_l478_478522


namespace probability_A_and_B_selected_l478_478341

theorem probability_A_and_B_selected (A B : Type) (S : Finset (Type)) (h : S.card = 5) (hA : A ∈ S) (hB : B ∈ S) (T : Finset (Type)) (hT : T.card = 3) :
  (∃ T ⊆ S, A ∈ T ∧ B ∈ T ∧ T.card = 3) → 
  ∃ T ⊆ S, T.card = 3 ∧ (finset.card (T ∩ {A, B}) = 2) → 
  (finset.card (T ∩ {A, B}) = 2) = (3 / 10 : Real) := sorry

end probability_A_and_B_selected_l478_478341


namespace circle_equation_line_equation_l478_478720

theorem circle_equation
  (C : ℝ → ℝ → Prop)
  (A B : ℝ × ℝ)
  (hA : A = (1, 3))
  (hB : B = (-1, 1))
  (center_line : ℝ → ℝ)
  (h_center_line : ∀ x, center_line x = 2 * x - 1)
  (center : ℝ × ℝ)
  (h_center : ∃ a, center = (a, center_line a))
  (passes_through : (ℝ × ℝ) → Prop)
  (h_passes_through : ∀ p, passes_through p ↔ C p.fst p.snd)
  (hC_A : passes_through A)
  (hC_B : passes_through B) :
  ∃ r, ∀ x y, C x y ↔ (x - 1) ^ 2 + (y - 1) ^ 2 = r^2 :=
sorry

theorem line_equation
  (C : ℝ → ℝ → Prop)
  (A B : ℝ × ℝ)
  (hA : A = (1, 3))
  (hB : B = (-1, 1))
  (center_line : ℝ → ℝ)
  (h_center_line : ∀ x, center_line x = 2 * x - 1)
  (center : ℝ × ℝ)
  (h_center : ∃ a, center = (a, center_line a))
  (passes_through : (ℝ × ℝ) → Prop)
  (h_passes_through : ∀ p, passes_through p ↔ C p.fst p.snd)
  (hC_A : passes_through A)
  (hC_B : passes_through B)
  (point : ℝ × ℝ)
  (h_point : point = (2, 2))
  (chord_length : ℝ)
  (h_chord : chord_length = 2 * (√3)) :
  ∃ l, (∀ x y, l x y ↔ y = 2) ∨ (∀ x y, l x y ↔ x = 2) :=
sorry

end circle_equation_line_equation_l478_478720


namespace probability_both_A_B_selected_l478_478211

-- Define the problem with the necessary conditions.
def probability_AB_selected (total_students : ℕ) (select_students : ℕ) (A B : ℕ) : ℚ :=
  if A < total_students ∧ B < total_students ∧ total_students = 5 ∧ select_students = 3 then
    let total_ways := (Nat.factorial total_students) / ((Nat.factorial select_students) * (Nat.factorial (total_students - select_students)))
    let favorable_ways := (Nat.factorial (total_students - 2)) / ((Nat.factorial (select_students - 2)) * (Nat.factorial ((total_students - 2) - (select_students - 2))))
    favorable_ways / total_ways
  else 0

theorem probability_both_A_B_selected 
  (total_students : ℕ) (select_students : ℕ) (A B : ℕ) 
  (h1 : total_students = 5) (h2 : select_students = 3) 
  (h3 : A < total_students) (h4 : B < total_students) 
  : probability_AB_selected total_students select_students A B = 3 / 10 := 
by {
  -- Insert the delta and logic to prove the theorem here.
  sorry
}

end probability_both_A_B_selected_l478_478211


namespace probability_A_and_B_selected_l478_478689

/-- There are 5 students including A and B. The probability of selecting both A and B 
    when 3 students are selected randomly from 5 is 3/10. -/
theorem probability_A_and_B_selected :
  (∃ (students : List String) (h : students.length = 5),
    ∃ (selected : List String) (h' : selected.length = 3),
    "A" ∈ selected ∧ "B" ∈ selected) →
  ∃ (P : ℚ), P = 3 / 10 :=
by
  sorry

end probability_A_and_B_selected_l478_478689


namespace george_speed_to_school_l478_478710

theorem george_speed_to_school :
  ∀ (d1 d2 v1 v2 v_arrive : ℝ), 
  d1 = 1.0 → d2 = 0.5 → v1 = 3.0 → v2 * (d1 / v1 + d2 / v2) = (d1 + d2) / 4.0 → v_arrive = 12.0 :=
by sorry

end george_speed_to_school_l478_478710


namespace probability_AB_selected_l478_478512

def factorial (n : Nat) : Nat := if n = 0 then 1 else n * factorial (n - 1)

def choose (n k : Nat) : Nat := factorial n / (factorial k * factorial (n - k))

theorem probability_AB_selected (h1 : 5 = 5) (h2 : 3 = 3) : (choose 3 1) / (choose 5 3) = 3 / 10 := by
  sorry

end probability_AB_selected_l478_478512


namespace probability_both_A_B_selected_l478_478215

-- Define the problem with the necessary conditions.
def probability_AB_selected (total_students : ℕ) (select_students : ℕ) (A B : ℕ) : ℚ :=
  if A < total_students ∧ B < total_students ∧ total_students = 5 ∧ select_students = 3 then
    let total_ways := (Nat.factorial total_students) / ((Nat.factorial select_students) * (Nat.factorial (total_students - select_students)))
    let favorable_ways := (Nat.factorial (total_students - 2)) / ((Nat.factorial (select_students - 2)) * (Nat.factorial ((total_students - 2) - (select_students - 2))))
    favorable_ways / total_ways
  else 0

theorem probability_both_A_B_selected 
  (total_students : ℕ) (select_students : ℕ) (A B : ℕ) 
  (h1 : total_students = 5) (h2 : select_students = 3) 
  (h3 : A < total_students) (h4 : B < total_students) 
  : probability_AB_selected total_students select_students A B = 3 / 10 := 
by {
  -- Insert the delta and logic to prove the theorem here.
  sorry
}

end probability_both_A_B_selected_l478_478215


namespace probability_A_B_l478_478446

open Nat

-- Define a function for combination
def combination (n r : Nat) : Nat :=
  factorial n / (factorial r * factorial (n - r))

-- Define the set of students and the selection conditions
def num_students : Nat := 5
def num_selected : Nat := 3

-- Define A and B as selected students
def selected_A_B : Nat := combination 3 1

-- Define the probability calculation
def probability_A_B_selected : Rat :=
  (selected_A_B : Rat) / (combination num_students num_selected : Rat)

-- The statement of the problem
theorem probability_A_B : probability_A_B_selected = 3 / 10 := by
  sorry

end probability_A_B_l478_478446


namespace select_3_from_5_prob_l478_478430

theorem select_3_from_5_prob {students : Finset ℕ} 
  (h : students.card = 5) 
  {A B : ℕ} 
  (hA : A ∈ students) 
  (hB : B ∈ students) 
  (H : 3 ≤ students.card) : 
  (finset.card (students.filter (λ x, x ≠ A ∧ x ≠ B)) = 3) → 
  (finset.card ((students.filter (λ x, x ≠ A ∧ x ≠ B)).subtype (λ x, True)) = 3) → 
  ((finset.card (finset.filter (λ (students : Finset ℕ) A ∈ students ∧ B ∈ students ↔ 3 = students.card) 5)) = ↑10) → 
  ∃ students', (students'.card = 3 ∧ A ∈ students' ∧ B ∈ students') → 
  ∃ (p : ℚ), p = 3 / 10 ∧ students.filter (λ x, x = A ∨ x = B) = 2 :=
  sorry

end select_3_from_5_prob_l478_478430


namespace positive_number_is_36_l478_478783

theorem positive_number_is_36
  (x : ℝ) 
  (h : (x + 1) = (4 - 2x)) 
  (y : ℝ) 
  (hx1 : y = x + 1) 
  (hx2 : y = 4 - 2x) : 
  y^2 = 36 :=
by 
  -- Proof omitted
  sorry

end positive_number_is_36_l478_478783


namespace probability_of_selecting_A_and_B_l478_478284

open_locale classical

-- Define the number of students
def num_students := 5

-- Define the number of students to be selected
def students_to_select := 3

def combination (n k : ℕ) : ℕ := nat.choose n k

-- Define the correct answer
def correct_probability := 3 / 10

-- Define the probability calculation
def calc_probability := (combination 3 1) / (combination 5 3)

-- Statement of the theorem to be proved
theorem probability_of_selecting_A_and_B :
  calc_probability = correct_probability :=
by sorry

end probability_of_selecting_A_and_B_l478_478284


namespace probability_A_and_B_selected_l478_478231

theorem probability_A_and_B_selected (students : Finset ℕ) (A B : ℕ) (h : A ∈ students ∧ B ∈ students ∧ students.card = 5) : 
  (students.choose 3).count (λ s, {A, B} ⊆ s) / (students.choose 3).card = 3 / 10 := by
  sorry

end probability_A_and_B_selected_l478_478231


namespace probability_A_B_l478_478451

open Nat

-- Define a function for combination
def combination (n r : Nat) : Nat :=
  factorial n / (factorial r * factorial (n - r))

-- Define the set of students and the selection conditions
def num_students : Nat := 5
def num_selected : Nat := 3

-- Define A and B as selected students
def selected_A_B : Nat := combination 3 1

-- Define the probability calculation
def probability_A_B_selected : Rat :=
  (selected_A_B : Rat) / (combination num_students num_selected : Rat)

-- The statement of the problem
theorem probability_A_B : probability_A_B_selected = 3 / 10 := by
  sorry

end probability_A_B_l478_478451


namespace probability_A_and_B_selected_l478_478390

theorem probability_A_and_B_selected 
  (students : Finset ℕ) (A B : ℕ) 
  (h_students_size : students.card = 5) 
  (h_A_in_students : A ∈ students) 
  (h_B_in_students : B ∈ students)
  : (↑(Nat.choose 3) : ℚ) / (↑(Nat.choose 5) : ℚ) = 3 / 10 := 
by sorry

end probability_A_and_B_selected_l478_478390


namespace factorization_of_polynomial_l478_478186

theorem factorization_of_polynomial (x : ℝ) : 16 * x ^ 2 - 40 * x + 25 = (4 * x - 5) ^ 2 := by 
  sorry

end factorization_of_polynomial_l478_478186


namespace probability_A_and_B_selected_l478_478345

theorem probability_A_and_B_selected (A B : Type) (S : Finset (Type)) (h : S.card = 5) (hA : A ∈ S) (hB : B ∈ S) (T : Finset (Type)) (hT : T.card = 3) :
  (∃ T ⊆ S, A ∈ T ∧ B ∈ T ∧ T.card = 3) → 
  ∃ T ⊆ S, T.card = 3 ∧ (finset.card (T ∩ {A, B}) = 2) → 
  (finset.card (T ∩ {A, B}) = 2) = (3 / 10 : Real) := sorry

end probability_A_and_B_selected_l478_478345


namespace probability_A_B_selected_l478_478350

noncomputable def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

def combination (n r : ℕ) : ℕ :=
  factorial n / (factorial r * factorial (n - r))

theorem probability_A_B_selected :
  let total_ways := combination 5 3 in
  let ways_A_B := combination 3 1 in
  (ways_A_B : ℚ) / total_ways = 3 / 10 :=
by
  sorry

end probability_A_B_selected_l478_478350


namespace find_c_l478_478715

theorem find_c (a b c : ℝ) (h1 : a + b = 5) (h2 : c^2 = a * b + b - 9) : c = 0 :=
by
  sorry

end find_c_l478_478715


namespace sqrt_500_eq_10_sqrt_5_l478_478866

theorem sqrt_500_eq_10_sqrt_5 : sqrt 500 = 10 * sqrt 5 :=
  sorry

end sqrt_500_eq_10_sqrt_5_l478_478866


namespace prob_select_A_and_B_l478_478649

theorem prob_select_A_and_B (students : Finset ℕ) (A B : ℕ) (hA : A ∈ students) (hB : B ∈ students) (h_card : students.card = 5) :
  (students.choose 3).count {comb | A ∈ comb ∧ B ∈ comb} / (students.choose 3).card = 3 / 10 := sorry

end prob_select_A_and_B_l478_478649


namespace prob_select_A_and_B_l478_478659

theorem prob_select_A_and_B (students : Finset ℕ) (A B : ℕ) (hA : A ∈ students) (hB : B ∈ students) (h_card : students.card = 5) :
  (students.choose 3).count {comb | A ∈ comb ∧ B ∈ comb} / (students.choose 3).card = 3 / 10 := sorry

end prob_select_A_and_B_l478_478659


namespace total_distinct_paths_l478_478114

-- Definitions
def grid := fin 3 × fin 3
def start_point := (0 : fin 3, 0 : fin 3)

def valid_path (path : list grid) : Prop :=
  path.head = start_point ∧ path.last = start_point ∧
  (∀ point ∈ path, point ∈ finset.univ (fin 3 × fin 3)) ∧
  path.nodup' ∧ path.length = 17

theorem total_distinct_paths : 
  { path : list grid // valid_path path }.card = 12 :=
by
  sorry

end total_distinct_paths_l478_478114


namespace probability_AB_selected_l478_478497

def factorial (n : Nat) : Nat := if n = 0 then 1 else n * factorial (n - 1)

def choose (n k : Nat) : Nat := factorial n / (factorial k * factorial (n - k))

theorem probability_AB_selected (h1 : 5 = 5) (h2 : 3 = 3) : (choose 3 1) / (choose 5 3) = 3 / 10 := by
  sorry

end probability_AB_selected_l478_478497


namespace probability_of_selecting_A_and_B_l478_478297

open_locale classical

-- Define the number of students
def num_students := 5

-- Define the number of students to be selected
def students_to_select := 3

def combination (n k : ℕ) : ℕ := nat.choose n k

-- Define the correct answer
def correct_probability := 3 / 10

-- Define the probability calculation
def calc_probability := (combination 3 1) / (combination 5 3)

-- Statement of the theorem to be proved
theorem probability_of_selecting_A_and_B :
  calc_probability = correct_probability :=
by sorry

end probability_of_selecting_A_and_B_l478_478297


namespace math_proof_problem_l478_478813

noncomputable def f (a b : ℚ) : ℝ := sorry

axiom f_cond1 (a b c : ℚ) : f (a * b) c = f a c * f b c ∧ f c (a * b) = f c a * f c b
axiom f_cond2 (a : ℚ) : f a (1 - a) = 1

theorem math_proof_problem (a b : ℚ) (ha : a ≠ 0) (hb : b ≠ 0) : 
  (f a a = 1) ∧ 
  (f a (-a) = 1) ∧
  (f a b * f b a = 1) := 
by 
  sorry

end math_proof_problem_l478_478813


namespace greatest_integer_le_624_l478_478050

theorem greatest_integer_le_624 :
  let expr := (4^101 + 5^101) / (4^97 + 5^97)
  floor expr = 624 :=
by
  let expr := (4^101 + 5^101) / (4^97 + 5^97)
  sorry

end greatest_integer_le_624_l478_478050


namespace probability_of_selecting_A_and_B_is_three_tenths_l478_478598

-- Define the set of students
def students : Finset ℕ := {0, 1, 2, 3, 4}

-- Define subsets A and B
def student_A : ℕ := 0
def student_B : ℕ := 1

-- Total number of ways to select 3 students out of 5
def total_ways : ℕ := students.card.choose 3

-- Number of ways to select 3 students including A and B
def favorable_ways : ℕ := {2, 3, 4}.card.choose 1

-- The probability
def probability_of_selecting_A_and_B : ℚ := (favorable_ways : ℚ) / (total_ways : ℚ)

theorem probability_of_selecting_A_and_B_is_three_tenths :
  probability_of_selecting_A_and_B = 3 / 10 :=
by
  sorry

end probability_of_selecting_A_and_B_is_three_tenths_l478_478598


namespace probability_A_and_B_selected_l478_478697

/-- There are 5 students including A and B. The probability of selecting both A and B 
    when 3 students are selected randomly from 5 is 3/10. -/
theorem probability_A_and_B_selected :
  (∃ (students : List String) (h : students.length = 5),
    ∃ (selected : List String) (h' : selected.length = 3),
    "A" ∈ selected ∧ "B" ∈ selected) →
  ∃ (P : ℚ), P = 3 / 10 :=
by
  sorry

end probability_A_and_B_selected_l478_478697


namespace prob_select_A_and_B_l478_478618

theorem prob_select_A_and_B : 
  let total_students := 5
  let to_select := 3
  let favorable_outcomes := Nat.div_fact (Nat.factorial 3) (Nat.factorial 1)
  let total_outcomes := Nat.div_fact (Nat.factorial 5) (Nat.factorial 3 * Nat.factorial (5 - 3))
  (favorable_outcomes : ℚ) / total_outcomes = 3 / 10 :=
by
  sorry

end prob_select_A_and_B_l478_478618


namespace factor_quadratic_l478_478167

theorem factor_quadratic : ∀ (x : ℝ), 16*x^2 - 40*x + 25 = (4*x - 5)^2 := by
  intro x
  sorry

end factor_quadratic_l478_478167


namespace probability_AB_selected_l478_478493

def factorial (n : Nat) : Nat := if n = 0 then 1 else n * factorial (n - 1)

def choose (n k : Nat) : Nat := factorial n / (factorial k * factorial (n - k))

theorem probability_AB_selected (h1 : 5 = 5) (h2 : 3 = 3) : (choose 3 1) / (choose 5 3) = 3 / 10 := by
  sorry

end probability_AB_selected_l478_478493


namespace Kiley_ate_two_slices_l478_478156

-- Definitions of conditions
def calories_per_slice := 350
def total_calories := 2800
def percentage_eaten := 0.25

-- Statement of the problem
theorem Kiley_ate_two_slices (h1 : total_calories = 2800)
                            (h2 : calories_per_slice = 350)
                            (h3 : percentage_eaten = 0.25) :
  (total_calories / calories_per_slice * percentage_eaten) = 2 := 
sorry

end Kiley_ate_two_slices_l478_478156


namespace symmetric_points_sum_l478_478743

theorem symmetric_points_sum (a b : ℝ) (h₁ : a = -2) (h₂ : b = -4) : a + b = -6 :=
by
  rw [h₁, h₂]
  exact eq.refl (-6)

end symmetric_points_sum_l478_478743


namespace prob_select_A_and_B_l478_478626

theorem prob_select_A_and_B : 
  let total_students := 5
  let to_select := 3
  let favorable_outcomes := Nat.div_fact (Nat.factorial 3) (Nat.factorial 1)
  let total_outcomes := Nat.div_fact (Nat.factorial 5) (Nat.factorial 3 * Nat.factorial (5 - 3))
  (favorable_outcomes : ℚ) / total_outcomes = 3 / 10 :=
by
  sorry

end prob_select_A_and_B_l478_478626


namespace probability_A_and_B_selected_l478_478582

/-- Let there be a group of 5 students including A and B. 
    If 3 students are selected randomly from these 5, the probability of selecting both A and B is 3/10. -/
theorem probability_A_and_B_selected : 
  let students := 5 in
  let selected := 3 in
  let combinations := students.choose selected in
  let favorable_combinations := (students - 2).choose (selected - 2) in
  let probability := (favorable_combinations : ℚ) / (combinations : ℚ) in
  probability = 3 / 10 := 
by sorry

end probability_A_and_B_selected_l478_478582


namespace probability_both_A_B_selected_l478_478206

-- Define the problem with the necessary conditions.
def probability_AB_selected (total_students : ℕ) (select_students : ℕ) (A B : ℕ) : ℚ :=
  if A < total_students ∧ B < total_students ∧ total_students = 5 ∧ select_students = 3 then
    let total_ways := (Nat.factorial total_students) / ((Nat.factorial select_students) * (Nat.factorial (total_students - select_students)))
    let favorable_ways := (Nat.factorial (total_students - 2)) / ((Nat.factorial (select_students - 2)) * (Nat.factorial ((total_students - 2) - (select_students - 2))))
    favorable_ways / total_ways
  else 0

theorem probability_both_A_B_selected 
  (total_students : ℕ) (select_students : ℕ) (A B : ℕ) 
  (h1 : total_students = 5) (h2 : select_students = 3) 
  (h3 : A < total_students) (h4 : B < total_students) 
  : probability_AB_selected total_students select_students A B = 3 / 10 := 
by {
  -- Insert the delta and logic to prove the theorem here.
  sorry
}

end probability_both_A_B_selected_l478_478206


namespace visiting_all_cubes_impossible_l478_478954

def Vec3 := (ℕ × ℕ × ℕ)

-- Defining the adjacency relation
def adjacent (a b : Vec3) : Prop :=
  (abs (a.1 - b.1) = 1 ∧ a.2 = b.2 ∧ a.3 = b.3) ∨
  (a.1 = b.1 ∧ abs (a.2 - b.2) = 1 ∧ a.3 = b.3) ∨
  (a.1 = b.1 ∧ a.2 = b.2 ∧ abs (a.3 - b.3) = 1)

-- Condition that prohibits moving in the same direction twice in a row
inductive Direction
| x_pos | x_neg | y_pos | y_neg | z_pos | z_neg

def valid_move (prev_move curr_move : Direction) : Prop :=
  prev_move ≠ curr_move

-- The main theorem to prove
theorem visiting_all_cubes_impossible :
  ¬ ∃ (path : List Vec3),
    (∀ v ∈ path, v.1 < 3 ∧ v.2 < 3 ∧ v.3 < 3) ∧
    (∀ i < path.length - 1, adjacent (path.nth_le i sorry) (path.nth_le (i + 1) sorry)) ∧
    (∀ i < path.length - 2, valid_move (direction_of (path.nth_le i sorry) (path.nth_le (i + 1) sorry)) 
                                       (direction_of (path.nth_le (i + 1) sorry) (path.nth_le (i + 2) sorry))) ∧
    path.length = 27 := sorry

end visiting_all_cubes_impossible_l478_478954


namespace probability_of_selecting_A_and_B_l478_478548

theorem probability_of_selecting_A_and_B (students : Finset ℕ) (A B : ℕ)
  (h_size : students.card = 5) (h_in : A ∈ students ∧ B ∈ students) : 
  (Finset.card {s : Finset ℕ // s ⊆ students ∧ s.card = 3 ∧ A ∈ s ∧ B ∈ s}.to_set) / 
  (Finset.card {s : Finset ℕ // s ⊆ students ∧ s.card = 3}.to_set) = 3 / 10 :=
sorry

end probability_of_selecting_A_and_B_l478_478548


namespace probability_of_selecting_A_and_B_is_three_tenths_l478_478599

-- Define the set of students
def students : Finset ℕ := {0, 1, 2, 3, 4}

-- Define subsets A and B
def student_A : ℕ := 0
def student_B : ℕ := 1

-- Total number of ways to select 3 students out of 5
def total_ways : ℕ := students.card.choose 3

-- Number of ways to select 3 students including A and B
def favorable_ways : ℕ := {2, 3, 4}.card.choose 1

-- The probability
def probability_of_selecting_A_and_B : ℚ := (favorable_ways : ℚ) / (total_ways : ℚ)

theorem probability_of_selecting_A_and_B_is_three_tenths :
  probability_of_selecting_A_and_B = 3 / 10 :=
by
  sorry

end probability_of_selecting_A_and_B_is_three_tenths_l478_478599


namespace remainder_poly_correct_l478_478194

/-- the remainder of dividing the polynomial p by q, r -/
def remainder : Polynomial ℤ → Polynomial ℤ → Polynomial ℤ
| p q =>
  let (d, r) := p.divMod q in r

open Polynomial

theorem remainder_poly_correct :
  let p := X^6 + C 2 * X^5 - C 3 * X^4 - C 4 * X^3 + C 5 * X^2 - C 6 * X + C 7
  let q := (X^2 - C 1) * (X - C 2)
  remainder p q = (-C 3 * X^2 - C 8 * X + C 13) :=
by
  sorry

end remainder_poly_correct_l478_478194


namespace probability_A_and_B_selected_l478_478704

/-- There are 5 students including A and B. The probability of selecting both A and B 
    when 3 students are selected randomly from 5 is 3/10. -/
theorem probability_A_and_B_selected :
  (∃ (students : List String) (h : students.length = 5),
    ∃ (selected : List String) (h' : selected.length = 3),
    "A" ∈ selected ∧ "B" ∈ selected) →
  ∃ (P : ℚ), P = 3 / 10 :=
by
  sorry

end probability_A_and_B_selected_l478_478704


namespace number_of_male_workers_l478_478081

def num_female_workers : ℕ := 15
def num_child_workers : ℕ := 5
def wage_per_male_worker : ℕ := 25
def wage_per_female_worker : ℕ := 20
def wage_per_child_worker : ℕ := 8
def average_wage : ℕ := 21

theorem number_of_male_workers (num_male_workers : ℕ) :
  let total_daily_wage := num_male_workers * wage_per_male_worker 
                        + num_female_workers * wage_per_female_worker 
                        + num_child_workers * wage_per_child_worker,
      total_workers := num_male_workers + num_female_workers + num_child_workers in
  (total_daily_wage / total_workers = average_wage) → num_male_workers = 20 :=
by
  sorry

end number_of_male_workers_l478_478081


namespace common_multiples_count_l478_478819

def multiple_set (k n : ℕ) : Set ℕ := {m | ∃ i : ℕ, i < n ∧ m = k * (i + 1)}

theorem common_multiples_count :
  let S := multiple_set 5 1500
  let T := multiple_set 10 1500
  (Set.card (S ∩ T) = 750) :=
by
  sorry

end common_multiples_count_l478_478819


namespace probability_A_B_l478_478456

open Nat

-- Define a function for combination
def combination (n r : Nat) : Nat :=
  factorial n / (factorial r * factorial (n - r))

-- Define the set of students and the selection conditions
def num_students : Nat := 5
def num_selected : Nat := 3

-- Define A and B as selected students
def selected_A_B : Nat := combination 3 1

-- Define the probability calculation
def probability_A_B_selected : Rat :=
  (selected_A_B : Rat) / (combination num_students num_selected : Rat)

-- The statement of the problem
theorem probability_A_B : probability_A_B_selected = 3 / 10 := by
  sorry

end probability_A_B_l478_478456


namespace negation_example_l478_478758

variable {I : Set ℝ}

theorem negation_example (h : ∀ x ∈ I, x^3 - x^2 + 1 ≤ 0) : ¬(∀ x ∈ I, x^3 - x^2 + 1 ≤ 0) ↔ ∃ x ∈ I, x^3 - x^2 + 1 > 0 :=
by
  sorry

end negation_example_l478_478758


namespace percentage_girls_l478_478908

theorem percentage_girls (initial_boys : ℕ) (initial_girls : ℕ) (added_boys : ℕ) :
  initial_boys = 11 → initial_girls = 13 → added_boys = 1 → 
  100 * initial_girls / (initial_boys + added_boys + initial_girls) = 52 :=
by
  intros h_boys h_girls h_added
  sorry

end percentage_girls_l478_478908


namespace julie_monthly_salary_l478_478809

def hourly_rate : ℕ := 5
def hours_per_day : ℕ := 8
def days_per_week : ℕ := 6
def missed_days : ℕ := 1
def weeks_per_month : ℕ := 4

theorem julie_monthly_salary :
  (hourly_rate * hours_per_day * (days_per_week - missed_days) * weeks_per_month) = 920 :=
by
  sorry

end julie_monthly_salary_l478_478809


namespace probability_A_and_B_selected_l478_478519

theorem probability_A_and_B_selected :
  let n := 5
  let r := 3
  let total_ways := Nat.choose n r
  let favorable_ways := Nat.choose 3 1
  let probability := favorable_ways / total_ways
  probability = (3 / 10) := 
by
  sorry

end probability_A_and_B_selected_l478_478519


namespace retirement_pension_l478_478094

theorem retirement_pension (k x : ℝ)
  (h1 : k * (x + 4)^2 = k * x^2 + 144)
  (h2 : k * (x + 9)^2 = k * x^2 + 324) :
  k * x^2 = (real.sqrt 171 / 5) * x^2 :=
by sorry

end retirement_pension_l478_478094


namespace probability_both_A_and_B_selected_l478_478400

-- The universe of students we consider
inductive Student
| A : Student
| B : Student
| C : Student
| D : Student
| E : Student

open Student

-- Function to count combinations
def combinations (n k : ℕ) : ℕ := (Nat.factorial n) / ((Nat.factorial k) * (Nat.factorial (n - k)))

-- The set of all students
def students : List Student := [A, B, C, D, E]

-- The total number of ways to select 3 students from 5
def total_selections := combinations 5 3

-- The number of ways A and B are both selected by selecting 1 more from the remaining 3 students
def selections_AB := combinations 3 1

-- The probability that both A and B are selected
def probability_AB := (selections_AB : ℚ) / (total_selections : ℚ)

-- The proof statement
theorem probability_both_A_and_B_selected :
  probability_AB = 3 / 10 :=
sorry

end probability_both_A_and_B_selected_l478_478400


namespace probability_of_A_and_B_selected_l478_478473

variable {α : Type}

/-- From 5 students, we randomly select 3 students to participate in community service work. 
  Let A and B be two specific students among these five. The probability that 
  both A and B are selected is 3 / 10. -/
theorem probability_of_A_and_B_selected (A B : α) (s : Finset α) (h : s.card = 5) :
  (∃ t : Finset α, t ⊆ s ∧ A ∈ t ∧ B ∈ t ∧ (t.card = 3) ∧ (t.card = 3) →
   (@Finset.exists_subset_card_eq α t s 3) →
   (@Finset.exists_subset_card_eq α t (Finset.erase s A) 2) ∧
   (@Finset.exists_subset_card_eq α t (Finset.erase (Finset.erase s A) B) 1)) →
  (3 : ℚ) / 10 :=
by
  sorry -- Proof omitted as per instructions.

end probability_of_A_and_B_selected_l478_478473


namespace count_perfect_cubes_l478_478771

theorem count_perfect_cubes (a b : ℕ) (h1 : a = 200) (h2 : b = 1200) :
  ∃ n, n = 5 ∧ ∀ x, (x^3 > a) ∧ (x^3 < b) → (x = 6 ∨ x = 7 ∨ x = 8 ∨ x = 9 ∨ x = 10) := 
sorry

end count_perfect_cubes_l478_478771


namespace quadratic_transformation_l478_478022

theorem quadratic_transformation (x d e : ℝ) (h : x^2 - 24*x + 45 = (x+d)^2 + e) : d + e = -111 :=
sorry

end quadratic_transformation_l478_478022


namespace probability_of_greater_area_l478_478964

noncomputable def isosceles_right_triangle_area : ℝ :=
  let AB := 1
  let AC := 1
  1 / 2

theorem probability_of_greater_area (P : Point) :
  P ∈ interior_of_triangle ABC → 
  (isosceles_right_triangle ABC (∠ BAC = 90°) → real.is_probability (area_of_t_triangle_ABP_greater_than_ACP_has_a_greater_area_than)) = 1/2 :=
begin
  sorry
end

end probability_of_greater_area_l478_478964


namespace perfectCubesBetween200and1200_l478_478768

theorem perfectCubesBetween200and1200 : ∃ n m : ℕ, (n = 6) ∧ (m = 10) ∧ (m - n + 1 = 5) ∧ (n^3 ≥ 200) ∧ (m^3 ≤ 1200) := 
by
  have h1 : 6^3 ≥ 200 := by norm_num
  have h2 : 10^3 ≤ 1200 := by norm_num
  use [6, 10]
  constructor; {refl} -- n = 6
  constructor; {refl} -- m = 10
  constructor;
  { norm_num },
  constructor; 
  { exact h1 },
  { exact h2 }
  sorry

end perfectCubesBetween200and1200_l478_478768


namespace buckets_required_l478_478916

theorem buckets_required (C : ℝ) (N : ℝ):
  (62.5 * (2 / 5) * C = N * C) → N = 25 :=
by
  sorry

end buckets_required_l478_478916


namespace projection_matrix_3_4_l478_478193

def projection_matrix (v : ℝ × ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
  let ⟨vx, vy⟩ := v in
  let denom := vx * vx + vy * vy in
  Matrix.of (λ i j, ([9, 12, 12, 16][(2 * i.val + j.val)] / denom))

theorem projection_matrix_3_4 :
  projection_matrix (3, 4) = Matrix.of (λ i j, [9/25, 12/25, 12/25, 16/25][2 * i.val + j.val]) :=
sorry

end projection_matrix_3_4_l478_478193


namespace walking_time_estimate_l478_478059

-- Define constants for distance, speed, and time conversion factor
def distance : ℝ := 1000
def speed : ℝ := 4000
def time_conversion : ℝ := 60

-- Define the expected time to walk from home to school in minutes
def expected_time : ℝ := 15

-- Prove the time calculation
theorem walking_time_estimate : (distance / speed) * time_conversion = expected_time :=
by
  sorry

end walking_time_estimate_l478_478059


namespace probability_A_and_B_selected_l478_478387

theorem probability_A_and_B_selected 
  (students : Finset ℕ) (A B : ℕ) 
  (h_students_size : students.card = 5) 
  (h_A_in_students : A ∈ students) 
  (h_B_in_students : B ∈ students)
  : (↑(Nat.choose 3) : ℚ) / (↑(Nat.choose 5) : ℚ) = 3 / 10 := 
by sorry

end probability_A_and_B_selected_l478_478387


namespace find_ab_and_monotonicity_l478_478757

noncomputable def f (x : ℝ) (a b : ℝ) := a * x^2 + b * Real.log x

theorem find_ab_and_monotonicity :
  ∃ a b : ℝ, (f 1 a b = 1 / 2 ∧ (2 * a + b = 0)) ∧ ((f' := λ x, deriv (f x a b)) 1 = 0 ∧ a = 1 / 2 ∧ b = -1) ∧
  ((∀ x : ℝ, 0 < x ∧ x < 1 → f' x < 0) ∧ (∀ x : ℝ, 1 < x → f' x > 0)) :=
by
  sorry

end find_ab_and_monotonicity_l478_478757


namespace scale_length_l478_478096

theorem scale_length (length_of_part : ℕ) (number_of_parts : ℕ) (h1 : number_of_parts = 2) (h2 : length_of_part = 40) :
  number_of_parts * length_of_part = 80 := 
by
  sorry

end scale_length_l478_478096


namespace probability_both_A_and_B_selected_l478_478412

-- The universe of students we consider
inductive Student
| A : Student
| B : Student
| C : Student
| D : Student
| E : Student

open Student

-- Function to count combinations
def combinations (n k : ℕ) : ℕ := (Nat.factorial n) / ((Nat.factorial k) * (Nat.factorial (n - k)))

-- The set of all students
def students : List Student := [A, B, C, D, E]

-- The total number of ways to select 3 students from 5
def total_selections := combinations 5 3

-- The number of ways A and B are both selected by selecting 1 more from the remaining 3 students
def selections_AB := combinations 3 1

-- The probability that both A and B are selected
def probability_AB := (selections_AB : ℚ) / (total_selections : ℚ)

-- The proof statement
theorem probability_both_A_and_B_selected :
  probability_AB = 3 / 10 :=
sorry

end probability_both_A_and_B_selected_l478_478412


namespace wedge_volume_l478_478955

noncomputable def volume_of_cylinder_wedge (r h : ℝ) (angle : ℝ) : ℝ :=
  let full_volume := π * r^2 * h
  in full_volume * (angle / (2 * π))

theorem wedge_volume (r h : ℝ) (angle : ℝ) (π_pos : 0 < π) (r_eq_5 : r = 5) (h_eq_10 : h = 10) (angle_eq_pi_over_3 : angle = π / 3) :
  volume_of_cylinder_wedge r h angle = 250 * π / 6 :=
by
  -- Given conditions are r = 5, h = 10, and the angle is π/3 (60 degrees)
  rw [r_eq_5, h_eq_10, angle_eq_pi_over_3]
  -- Calculate the full volume
  have full_volume := π * (5:ℝ)^2 * (10:ℝ)
  -- Simplify the volume calculation
  rw [←mul_assoc, pow_two, show (5:ℝ)^2 = 25, by norm_num]
  rw [←mul_assoc, show (π * 25 * 10) = 250 * π, by norm_num]
  -- Calculate the wedge volume (one-sixth of the full volume)
  have wedge_volume := (250 * π) * (1 / 6)
  -- Show that it simplifies to the expected result
  rw [←mul_assoc, show (250:ℝ) * (1 / 6) = 250 / 6, by norm_num]
  refl

end wedge_volume_l478_478955


namespace probability_A_and_B_selected_l478_478535

theorem probability_A_and_B_selected :
  let n := 5
  let r := 3
  let total_ways := Nat.choose n r
  let favorable_ways := Nat.choose 3 1
  let probability := favorable_ways / total_ways
  probability = (3 / 10) := 
by
  sorry

end probability_A_and_B_selected_l478_478535


namespace cone_area_example_l478_478023

def cone_lateral_surface_area (r l : ℝ) : ℝ :=
  1 / 2 * (2 * real.pi * r) * l

theorem cone_area_example : cone_lateral_surface_area 3 5 = 15 * real.pi :=
by
  sorry

end cone_area_example_l478_478023


namespace purely_imaginary_z_l478_478748

theorem purely_imaginary_z (a : ℝ) :
  (a^2 - 2 * a = 0) ∧ (a - 2 ≠ 0) → a = 0 :=
by {
  intro h,
  cases h with h1 h2,
  -- We are supposed to prove this statement, but we will leave the proof incomplete here
  sorry
}

end purely_imaginary_z_l478_478748


namespace select_3_from_5_prob_l478_478439

theorem select_3_from_5_prob {students : Finset ℕ} 
  (h : students.card = 5) 
  {A B : ℕ} 
  (hA : A ∈ students) 
  (hB : B ∈ students) 
  (H : 3 ≤ students.card) : 
  (finset.card (students.filter (λ x, x ≠ A ∧ x ≠ B)) = 3) → 
  (finset.card ((students.filter (λ x, x ≠ A ∧ x ≠ B)).subtype (λ x, True)) = 3) → 
  ((finset.card (finset.filter (λ (students : Finset ℕ) A ∈ students ∧ B ∈ students ↔ 3 = students.card) 5)) = ↑10) → 
  ∃ students', (students'.card = 3 ∧ A ∈ students' ∧ B ∈ students') → 
  ∃ (p : ℚ), p = 3 / 10 ∧ students.filter (λ x, x = A ∨ x = B) = 2 :=
  sorry

end select_3_from_5_prob_l478_478439


namespace prob_select_A_and_B_l478_478652

theorem prob_select_A_and_B (students : Finset ℕ) (A B : ℕ) (hA : A ∈ students) (hB : B ∈ students) (h_card : students.card = 5) :
  (students.choose 3).count {comb | A ∈ comb ∧ B ∈ comb} / (students.choose 3).card = 3 / 10 := sorry

end prob_select_A_and_B_l478_478652


namespace inequality_proof_l478_478716

noncomputable def a : ℝ := 2 ^ (Real.log2 3)
noncomputable def b : ℝ := Real.log (3 / Real.exp 2)
noncomputable def c : ℝ := Real.pi ^ (-Real.e)

theorem inequality_proof : a > c ∧ c > b := by
  sorry

end inequality_proof_l478_478716


namespace even_integers_from_set_count_l478_478765

theorem even_integers_from_set_count :
  let S := {1, 3, 4, 6, 7, 9}
  ∃ count, ∀ n, (n >= 200 ∧ n < 900 ∧ even n ∧ ∀ i j, i ≠ j → (i ∈ digits 10 n) → (j ∈ digits 10 n) → i ≠ j ∧ ∀ d ∈ digits 10 n, d ∈ S) → count = 16 :=
sorry  -- Proof is omitted

end even_integers_from_set_count_l478_478765


namespace probability_of_selecting_A_and_B_l478_478270

-- Define the combination function
def combination (n r : ℕ) : ℕ := Nat.factorial n / (Nat.factorial r * Nat.factorial (n - r))

-- Define the conditions
def students : ℕ := 5
def to_select : ℕ := 3

-- Define the favorable outcomes where A and B are selected
def favorable_outcomes : ℕ := combination 3 1

-- Define the total outcomes of selecting 3 out of 5 students
def total_outcomes : ℕ := combination 5 3

-- Define the probability calculation
def probability_of_A_and_B : ℚ := favorable_outcomes / total_outcomes

-- Proposition to prove
theorem probability_of_selecting_A_and_B : probability_of_A_and_B = 3 / 10 :=
by
  sorry

end probability_of_selecting_A_and_B_l478_478270


namespace select_3_from_5_prob_l478_478435

theorem select_3_from_5_prob {students : Finset ℕ} 
  (h : students.card = 5) 
  {A B : ℕ} 
  (hA : A ∈ students) 
  (hB : B ∈ students) 
  (H : 3 ≤ students.card) : 
  (finset.card (students.filter (λ x, x ≠ A ∧ x ≠ B)) = 3) → 
  (finset.card ((students.filter (λ x, x ≠ A ∧ x ≠ B)).subtype (λ x, True)) = 3) → 
  ((finset.card (finset.filter (λ (students : Finset ℕ) A ∈ students ∧ B ∈ students ↔ 3 = students.card) 5)) = ↑10) → 
  ∃ students', (students'.card = 3 ∧ A ∈ students' ∧ B ∈ students') → 
  ∃ (p : ℚ), p = 3 / 10 ∧ students.filter (λ x, x = A ∨ x = B) = 2 :=
  sorry

end select_3_from_5_prob_l478_478435


namespace unique_triple_of_real_numbers_l478_478100

def untranslatable (s : String) : Prop := 
  ¬s.contains "AUG"

def a_n : ℕ → ℕ := sorry -- Definition of a_n should observe the problem's conditions

theorem unique_triple_of_real_numbers :
  ∃! (x y z : ℝ), x = 4 ∧ y = 0 ∧ z = -1 ∧ (∀ n ≥ 100, a_n n = x * a_n (n-1) + y * a_n (n-2) + z * a_n (n-3)) :=
sorry

end unique_triple_of_real_numbers_l478_478100


namespace exists_disjoint_subsets_with_equal_sum_l478_478728

theorem exists_disjoint_subsets_with_equal_sum
  (S : Finset ℕ)
  (h_size_S : S.card = 10)
  (h_range : ∀ x ∈ S, 10 ≤ x ∧ x ≤ 99) :
  ∃ (A B : Finset ℕ), A ⊆ S ∧ B ⊆ S ∧ A ∩ B = ∅ ∧ A.sum (λ x, x) = B.sum (λ x, x) :=
by
  sorry

end exists_disjoint_subsets_with_equal_sum_l478_478728


namespace monsieur_dupond_l478_478843

def reachable (start_point target_point : ℤ × ℤ) : Prop := 
  let moves := [(λ (p : ℤ × ℤ), (p.2, p.1)), 
                (λ (p : ℤ × ℤ), (3 * p.1, -2 * p.2)),
                (λ (p : ℤ × ℤ), (-2 * p.1, 3 * p.2)),
                (λ (p : ℤ × ℤ), (p.1 + 1, p.2 + 4)),
                (λ (p : ℤ × ℤ), (p.1 - 1, p.2 - 4))]
  (start_point = target_point) ∨ 
  (∃ seq, ∀ i, i < seq.length → 
    let p := seq.foldl (λ p f, f p) start_point in 
    (∃ m ∈ moves, f = m) ∧ (p = target_point))

theorem monsieur_dupond :
  ¬ reachable (0, 1) (0, 0) :=
by
  sorry

end monsieur_dupond_l478_478843


namespace probability_of_selecting_A_and_B_is_three_tenths_l478_478606

-- Define the set of students
def students : Finset ℕ := {0, 1, 2, 3, 4}

-- Define subsets A and B
def student_A : ℕ := 0
def student_B : ℕ := 1

-- Total number of ways to select 3 students out of 5
def total_ways : ℕ := students.card.choose 3

-- Number of ways to select 3 students including A and B
def favorable_ways : ℕ := {2, 3, 4}.card.choose 1

-- The probability
def probability_of_selecting_A_and_B : ℚ := (favorable_ways : ℚ) / (total_ways : ℚ)

theorem probability_of_selecting_A_and_B_is_three_tenths :
  probability_of_selecting_A_and_B = 3 / 10 :=
by
  sorry

end probability_of_selecting_A_and_B_is_three_tenths_l478_478606


namespace prob_select_A_and_B_l478_478660

theorem prob_select_A_and_B (students : Finset ℕ) (A B : ℕ) (hA : A ∈ students) (hB : B ∈ students) (h_card : students.card = 5) :
  (students.choose 3).count {comb | A ∈ comb ∧ B ∈ comb} / (students.choose 3).card = 3 / 10 := sorry

end prob_select_A_and_B_l478_478660


namespace probability_A_and_B_selected_l478_478382

theorem probability_A_and_B_selected 
  (students : Finset ℕ) (A B : ℕ) 
  (h_students_size : students.card = 5) 
  (h_A_in_students : A ∈ students) 
  (h_B_in_students : B ∈ students)
  : (↑(Nat.choose 3) : ℚ) / (↑(Nat.choose 5) : ℚ) = 3 / 10 := 
by sorry

end probability_A_and_B_selected_l478_478382


namespace company_employee_count_l478_478952

/-- 
 Given the employees are divided into three age groups: A, B, and C, with a ratio of 5:4:1,
 a stratified sampling method is used to draw a sample of size 20 from the population,
 and the probability of selecting both person A and person B from group C is 1/45.
 Prove the total number of employees in the company is 100.
-/
theorem company_employee_count :
  ∃ (total_employees : ℕ),
    (∃ (ratio_A : ℕ) (ratio_B : ℕ) (ratio_C : ℕ),
      ratio_A = 5 ∧ 
      ratio_B = 4 ∧ 
      ratio_C = 1 ∧
      ∃ (sample_size : ℕ), 
        sample_size = 20 ∧
        ∃ (prob_selecting_two_from_C : ℚ),
          prob_selecting_two_from_C = 1 / 45 ∧
          total_employees = 100) :=
sorry

end company_employee_count_l478_478952


namespace felix_distance_l478_478885

theorem felix_distance : 
  ∀ (avg_speed: ℕ) (time: ℕ) (factor: ℕ), 
  avg_speed = 66 → factor = 2 → time = 4 → (factor * avg_speed * time = 528) := 
by
  intros avg_speed time factor h_avg_speed h_factor h_time
  rw [h_avg_speed, h_factor, h_time]
  norm_num
  sorry

end felix_distance_l478_478885


namespace probability_A_B_selected_l478_478359

noncomputable def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

def combination (n r : ℕ) : ℕ :=
  factorial n / (factorial r * factorial (n - r))

theorem probability_A_B_selected :
  let total_ways := combination 5 3 in
  let ways_A_B := combination 3 1 in
  (ways_A_B : ℚ) / total_ways = 3 / 10 :=
by
  sorry

end probability_A_B_selected_l478_478359


namespace triangle_angle_A_cos_B_minus_C_l478_478790

theorem triangle_angle_A (AB AC BC : ℝ) (hAB : AB = 2) (hAC : AC = 3) (hBC : BC = sqrt 7) : 
  let cosA := (AB^2 + AC^2 - BC^2) / (2 * AB * AC) in 
  cosA = 1/2 → 
  ∃ A : ℝ, A = π / 3 :=
by
  sorry

theorem cos_B_minus_C (AB AC BC A cosA : ℝ) 
  (hAB : AB = 2) (hAC : AC = 3) (hBC : BC = sqrt 7) (hA : A = π / 3) : 
  let cosC := sqrt (1 - (AB * sqrt 3 / BC) ^ 2) in 
  let B := 2 * π / 3 - acos cosC in 
  cos (B - acos cosC) = 11/14 :=
by
  sorry

end triangle_angle_A_cos_B_minus_C_l478_478790


namespace net_profit_example_l478_478062

theorem net_profit_example :
  let CP := 100 in
  let MP := CP + 0.2 * CP in
  let SP := MP - 0.15 * MP in
  SP - CP = 2 :=
by
  sorry

end net_profit_example_l478_478062


namespace find_actual_average_height_l478_478881

noncomputable def actualAverageHeight (avg_height : ℕ) (num_boys : ℕ) (wrong_height : ℕ) (actual_height : ℕ) : Float :=
  let incorrect_total := avg_height * num_boys
  let difference := wrong_height - actual_height
  let correct_total := incorrect_total - difference
  (Float.ofInt correct_total) / (Float.ofNat num_boys)

theorem find_actual_average_height (avg_height num_boys wrong_height actual_height : ℕ) :
  avg_height = 185 ∧ num_boys = 35 ∧ wrong_height = 166 ∧ actual_height = 106 →
  actualAverageHeight avg_height num_boys wrong_height actual_height = 183.29 := by
  intros h
  have h_avg := h.1
  have h_num := h.2.1
  have h_wrong := h.2.2.1
  have h_actual := h.2.2.2
  rw [h_avg, h_num, h_wrong, h_actual]
  sorry

end find_actual_average_height_l478_478881


namespace probability_of_selecting_A_and_B_l478_478663

open_locale classical

noncomputable def select_probability (n k : ℕ) : ℚ :=
  (nat.choose n k)⁻¹

theorem probability_of_selecting_A_and_B :
  let total_students := 5,
      students_to_select := 3,
      remaining_students := 3 in
  select_probability total_students students_to_select * nat.choose remaining_students 1 = 3 / 10 :=
by
  sorry

end probability_of_selecting_A_and_B_l478_478663


namespace parallelogram_area_l478_478123

variables {ℝ : Type} [inner_product_space ℝ ℝ]

-- Conditions
def p : ℝ := 2
def q : ℝ := 1
def angle_pq : ℝ := π / 3

def a := 2 * p + 3 * q
def b := p - 2 * q

-- Theorem to prove
theorem parallelogram_area :
  ∥a × b∥ = 7 * sqrt 3 :=
sorry

end parallelogram_area_l478_478123


namespace marathon_distance_m_l478_478962

theorem marathon_distance_m (k m : ℕ) (h_marathon_km : 42) (h_marathon_m : 195) (h_km_to_m : 1000) 
  (h_john_marathons : 15) (h_total_distance : k * 1000 + m = 15 * (42 * 1000 + 195)) (h_m_range : 0 ≤ m ∧ m < 1000) :
  m = 925 := 
by 
  sorry

end marathon_distance_m_l478_478962


namespace class_percentage_of_girls_l478_478906

/-
Given:
- Initial number of boys in the class: 11
- Number of girls in the class: 13
- 1 boy is added to the class, resulting in the new total number of boys being 12

Prove:
- The percentage of the class that are girls is 52%.
-/
theorem class_percentage_of_girls (initial_boys : ℕ) (girls : ℕ) (added_boy : ℕ)
  (new_boy_total : ℕ) (total_students : ℕ) (percent_girls : ℕ) (h1 : initial_boys = 11) 
  (h2 : girls = 13) (h3 : added_boy = 1) (h4 : new_boy_total = initial_boys + added_boy) 
  (h5 : total_students = new_boy_total + girls) 
  (h6 : percent_girls = (girls * 100) / total_students) : percent_girls = 52 :=
sorry

end class_percentage_of_girls_l478_478906


namespace probability_A_and_B_selected_l478_478237

theorem probability_A_and_B_selected (students : Finset ℕ) (A B : ℕ) (h : A ∈ students ∧ B ∈ students ∧ students.card = 5) : 
  (students.choose 3).count (λ s, {A, B} ⊆ s) / (students.choose 3).card = 3 / 10 := by
  sorry

end probability_A_and_B_selected_l478_478237


namespace min_jumps_to_visit_all_points_and_return_l478_478028

theorem min_jumps_to_visit_all_points_and_return (n : ℕ) (h : n = 2016) :
  ∃ (min_jumps : ℕ), min_jumps = 2017 ∧
    ∀ (j : ℕ), (j = 2 ∨ j = 3) →
    (let P := list.range n in 
     ∃ seq : list ℕ,
       list.perm (list.fin_range n) seq ∧ 
       seq.head! = 0 ∧ 
       seq.last' = some 0 ∧
       ∀ i ∈ list.init seq, (seq.get i + j) % n = some (seq.get (i + 1))) :=
sorry

end min_jumps_to_visit_all_points_and_return_l478_478028


namespace probability_A_B_selected_l478_478355

noncomputable def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

def combination (n r : ℕ) : ℕ :=
  factorial n / (factorial r * factorial (n - r))

theorem probability_A_B_selected :
  let total_ways := combination 5 3 in
  let ways_A_B := combination 3 1 in
  (ways_A_B : ℚ) / total_ways = 3 / 10 :=
by
  sorry

end probability_A_B_selected_l478_478355


namespace probability_A_B_l478_478463

open Nat

-- Define a function for combination
def combination (n r : Nat) : Nat :=
  factorial n / (factorial r * factorial (n - r))

-- Define the set of students and the selection conditions
def num_students : Nat := 5
def num_selected : Nat := 3

-- Define A and B as selected students
def selected_A_B : Nat := combination 3 1

-- Define the probability calculation
def probability_A_B_selected : Rat :=
  (selected_A_B : Rat) / (combination num_students num_selected : Rat)

-- The statement of the problem
theorem probability_A_B : probability_A_B_selected = 3 / 10 := by
  sorry

end probability_A_B_l478_478463


namespace solution_set_of_f_prime_gt_zero_l478_478775

noncomputable def f (x : ℝ) : ℝ := x^2 - 2*x - 4 * Real.log x

theorem solution_set_of_f_prime_gt_zero :
  {x : ℝ | 0 < x ∧ 2*x - 2 - (4 / x) > 0} = {x : ℝ | 2 < x} :=
by
  sorry

end solution_set_of_f_prime_gt_zero_l478_478775


namespace comb_13_eq_comb_7_implies_comb_2_eq_190_l478_478731

theorem comb_13_eq_comb_7_implies_comb_2_eq_190
  (n : ℕ)
  (h : nat.choose n 13 = nat.choose n 7) :
  nat.choose n 2 = 190 :=
by
  sorry

end comb_13_eq_comb_7_implies_comb_2_eq_190_l478_478731


namespace probability_both_A_B_selected_l478_478223

-- Define the problem with the necessary conditions.
def probability_AB_selected (total_students : ℕ) (select_students : ℕ) (A B : ℕ) : ℚ :=
  if A < total_students ∧ B < total_students ∧ total_students = 5 ∧ select_students = 3 then
    let total_ways := (Nat.factorial total_students) / ((Nat.factorial select_students) * (Nat.factorial (total_students - select_students)))
    let favorable_ways := (Nat.factorial (total_students - 2)) / ((Nat.factorial (select_students - 2)) * (Nat.factorial ((total_students - 2) - (select_students - 2))))
    favorable_ways / total_ways
  else 0

theorem probability_both_A_B_selected 
  (total_students : ℕ) (select_students : ℕ) (A B : ℕ) 
  (h1 : total_students = 5) (h2 : select_students = 3) 
  (h3 : A < total_students) (h4 : B < total_students) 
  : probability_AB_selected total_students select_students A B = 3 / 10 := 
by {
  -- Insert the delta and logic to prove the theorem here.
  sorry
}

end probability_both_A_B_selected_l478_478223


namespace friends_in_group_l478_478087

theorem friends_in_group (cost_per_costume : ℕ) (total_spent : ℕ) 
  (h1 : cost_per_costume = 5) (h2 : total_spent = 40) : 
  total_spent / cost_per_costume = 8 :=
by {
  rw [h1, h2],
  calc 40 / 5 = 8 : by norm_num,
}

end friends_in_group_l478_478087


namespace probability_of_selecting_A_and_B_l478_478298

open_locale classical

-- Define the number of students
def num_students := 5

-- Define the number of students to be selected
def students_to_select := 3

def combination (n k : ℕ) : ℕ := nat.choose n k

-- Define the correct answer
def correct_probability := 3 / 10

-- Define the probability calculation
def calc_probability := (combination 3 1) / (combination 5 3)

-- Statement of the theorem to be proved
theorem probability_of_selecting_A_and_B :
  calc_probability = correct_probability :=
by sorry

end probability_of_selecting_A_and_B_l478_478298


namespace probability_both_A_and_B_selected_l478_478409

-- The universe of students we consider
inductive Student
| A : Student
| B : Student
| C : Student
| D : Student
| E : Student

open Student

-- Function to count combinations
def combinations (n k : ℕ) : ℕ := (Nat.factorial n) / ((Nat.factorial k) * (Nat.factorial (n - k)))

-- The set of all students
def students : List Student := [A, B, C, D, E]

-- The total number of ways to select 3 students from 5
def total_selections := combinations 5 3

-- The number of ways A and B are both selected by selecting 1 more from the remaining 3 students
def selections_AB := combinations 3 1

-- The probability that both A and B are selected
def probability_AB := (selections_AB : ℚ) / (total_selections : ℚ)

-- The proof statement
theorem probability_both_A_and_B_selected :
  probability_AB = 3 / 10 :=
sorry

end probability_both_A_and_B_selected_l478_478409


namespace probability_A_and_B_selected_l478_478333

theorem probability_A_and_B_selected (A B : Type) (S : Finset (Type)) (h : S.card = 5) (hA : A ∈ S) (hB : B ∈ S) (T : Finset (Type)) (hT : T.card = 3) :
  (∃ T ⊆ S, A ∈ T ∧ B ∈ T ∧ T.card = 3) → 
  ∃ T ⊆ S, T.card = 3 ∧ (finset.card (T ∩ {A, B}) = 2) → 
  (finset.card (T ∩ {A, B}) = 2) = (3 / 10 : Real) := sorry

end probability_A_and_B_selected_l478_478333


namespace water_added_l478_478079

theorem water_added (initial_fullness : ℝ) (fullness_after : ℝ) (capacity : ℝ) 
  (h_initial : initial_fullness = 0.30) (h_after : fullness_after = 3/4) (h_capacity : capacity = 100) : 
  fullness_after * capacity - initial_fullness * capacity = 45 := 
by 
  sorry

end water_added_l478_478079


namespace melany_initial_money_l478_478840

theorem melany_initial_money (total_perimeter cost_per_foot money_missing : ℕ)
    (h1 : total_perimeter = 5000)
    (h2 : cost_per_foot = 30)
    (h3 : money_missing = 1000 * cost_per_foot) :
    initially_money = 120000 :=
begin
    sorry
end

end melany_initial_money_l478_478840


namespace probability_A_B_selected_l478_478366

noncomputable def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

def combination (n r : ℕ) : ℕ :=
  factorial n / (factorial r * factorial (n - r))

theorem probability_A_B_selected :
  let total_ways := combination 5 3 in
  let ways_A_B := combination 3 1 in
  (ways_A_B : ℚ) / total_ways = 3 / 10 :=
by
  sorry

end probability_A_B_selected_l478_478366


namespace probability_A_and_B_selected_l478_478310

open Finset

noncomputable def total_ways : ℕ := (choose 5 3)

noncomputable def favorable_ways : ℕ := (choose 3 1)

noncomputable def probability (total favorable : ℕ) : ℚ := favorable / total

theorem probability_A_and_B_selected :
  probability total_ways favorable_ways = 3 / 10 :=
by
  sorry

end probability_A_and_B_selected_l478_478310


namespace find_supplementary_angle_l478_478903

noncomputable def degree (x : ℝ) : ℝ := x
noncomputable def complementary_angle (x : ℝ) : ℝ := 90 - x
noncomputable def supplementary_angle (x : ℝ) : ℝ := 180 - x

theorem find_supplementary_angle
  (x : ℝ)
  (h1 : degree x / complementary_angle x = 1 / 8) :
  supplementary_angle x = 170 :=
by
  sorry

end find_supplementary_angle_l478_478903


namespace probability_A_B_l478_478449

open Nat

-- Define a function for combination
def combination (n r : Nat) : Nat :=
  factorial n / (factorial r * factorial (n - r))

-- Define the set of students and the selection conditions
def num_students : Nat := 5
def num_selected : Nat := 3

-- Define A and B as selected students
def selected_A_B : Nat := combination 3 1

-- Define the probability calculation
def probability_A_B_selected : Rat :=
  (selected_A_B : Rat) / (combination num_students num_selected : Rat)

-- The statement of the problem
theorem probability_A_B : probability_A_B_selected = 3 / 10 := by
  sorry

end probability_A_B_l478_478449


namespace tangent_line_at_point_l478_478016

noncomputable def curve : ℝ → ℝ := λ x, exp (-5 * x) + 2

theorem tangent_line_at_point :
  ∃ b : ℝ, ∀ x : ℝ, curve x = -5 * x + b ∧ curve 0 = 3 ∧ b = 3 :=
begin
  sorry
end

end tangent_line_at_point_l478_478016


namespace total_trench_length_l478_478943

-- Given definitions
def num_workers : ℕ := 4
def length_dug_by_fourth_worker : ℕ := 50
def additional_length_per_worker (L : ℕ) : ℕ := (3 * L) / 4

-- Proof statement
theorem total_trench_length : 
  let total_length := length_dug_by_fourth_worker + num_workers.pred * additional_length_per_worker length_dug_by_fourth_worker in
  total_length = 200 := by
  sorry

end total_trench_length_l478_478943


namespace tensor_problem_l478_478898

namespace MathProof

def tensor (a b : ℚ) : ℚ := b^2 + 1

theorem tensor_problem (m : ℚ) : tensor m (tensor m 3) = 101 := by
  -- problem statement, proof not included
  sorry

end MathProof

end tensor_problem_l478_478898


namespace prob_select_A_and_B_l478_478638

theorem prob_select_A_and_B (students : Finset ℕ) (A B : ℕ) (hA : A ∈ students) (hB : B ∈ students) (h_card : students.card = 5) :
  (students.choose 3).count {comb | A ∈ comb ∧ B ∈ comb} / (students.choose 3).card = 3 / 10 := sorry

end prob_select_A_and_B_l478_478638


namespace chord_intercept_length_l478_478888

noncomputable def focus_of_parabola (x : ℝ) (y : ℝ) : Prop :=
x = 0 ∧ y = 1

noncomputable def directrix_of_parabola (y : ℝ) : Prop :=
y = -1

noncomputable def circle (x : ℝ) (y : ℝ) : Prop :=
x^2 + (y - 1)^2 = 4

theorem chord_intercept_length :
  let focus_x := 0
  let focus_y := 1
  let directrix_y := -1
  let radius := 2
  let center_x := focus_x
  let center_y := focus_y
  let circle_eq := circle x y
  -- Find y-intercepts
  let y_top := 3
  let y_bottom := -1
  -- Compute chord length
  y_top - y_bottom = 4 :=
by sorry

end chord_intercept_length_l478_478888


namespace log3_cubicroot_of_3_l478_478158

noncomputable def log_base_3 (x : ℝ) : ℝ := Real.log x / Real.log 3

theorem log3_cubicroot_of_3 :
  log_base_3 (3 ^ (1/3 : ℝ)) = 1 / 3 :=
by
  sorry

end log3_cubicroot_of_3_l478_478158


namespace line_intersects_circle_l478_478074

theorem line_intersects_circle {O : Point} {l : Line} (r : ℝ) (d : ℝ) (hr : r = 6) (hd : d = 5) : 
  line_circle_relationship O l r d = "intersect" := 
by
  sorry

def line_circle_relationship (O : Point) (l : Line) (r : ℝ) (d : ℝ) : String := 
  if r > d then "intersect" else if r = d then "tangent" else "disjoint"

end line_intersects_circle_l478_478074


namespace select_3_from_5_prob_l478_478443

theorem select_3_from_5_prob {students : Finset ℕ} 
  (h : students.card = 5) 
  {A B : ℕ} 
  (hA : A ∈ students) 
  (hB : B ∈ students) 
  (H : 3 ≤ students.card) : 
  (finset.card (students.filter (λ x, x ≠ A ∧ x ≠ B)) = 3) → 
  (finset.card ((students.filter (λ x, x ≠ A ∧ x ≠ B)).subtype (λ x, True)) = 3) → 
  ((finset.card (finset.filter (λ (students : Finset ℕ) A ∈ students ∧ B ∈ students ↔ 3 = students.card) 5)) = ↑10) → 
  ∃ students', (students'.card = 3 ∧ A ∈ students' ∧ B ∈ students') → 
  ∃ (p : ℚ), p = 3 / 10 ∧ students.filter (λ x, x = A ∨ x = B) = 2 :=
  sorry

end select_3_from_5_prob_l478_478443


namespace triangle_inequality_circumradius_l478_478719

theorem triangle_inequality_circumradius (a b c R : ℝ) 
  (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h_triangle : a + b > c ∧ a + c > b ∧ b + c > a) 
  (circumradius_def : R = (a * b * c) / (4 * (Real.sqrt ((a + b + c) * (a + b - c) * (a - b + c) * (-a + b + c))))) :
  (1 / (a * b)) + (1 / (b * c)) + (1 / (c * a)) ≥ (1 / (R ^ 2)) :=
sorry

end triangle_inequality_circumradius_l478_478719


namespace prob_select_A_and_B_l478_478627

theorem prob_select_A_and_B : 
  let total_students := 5
  let to_select := 3
  let favorable_outcomes := Nat.div_fact (Nat.factorial 3) (Nat.factorial 1)
  let total_outcomes := Nat.div_fact (Nat.factorial 5) (Nat.factorial 3 * Nat.factorial (5 - 3))
  (favorable_outcomes : ℚ) / total_outcomes = 3 / 10 :=
by
  sorry

end prob_select_A_and_B_l478_478627


namespace complex_exp_sum_l478_478827

def w : ℂ := sorry  -- We define w as a complex number, satisfying the given condition.

theorem complex_exp_sum (h : w^2 - w + 1 = 0) : 
  w^97 + w^98 + w^99 + w^100 + w^101 + w^102 = -2 + 2 * w :=
by
  sorry

end complex_exp_sum_l478_478827


namespace probability_both_A_and_B_selected_l478_478418

-- The universe of students we consider
inductive Student
| A : Student
| B : Student
| C : Student
| D : Student
| E : Student

open Student

-- Function to count combinations
def combinations (n k : ℕ) : ℕ := (Nat.factorial n) / ((Nat.factorial k) * (Nat.factorial (n - k)))

-- The set of all students
def students : List Student := [A, B, C, D, E]

-- The total number of ways to select 3 students from 5
def total_selections := combinations 5 3

-- The number of ways A and B are both selected by selecting 1 more from the remaining 3 students
def selections_AB := combinations 3 1

-- The probability that both A and B are selected
def probability_AB := (selections_AB : ℚ) / (total_selections : ℚ)

-- The proof statement
theorem probability_both_A_and_B_selected :
  probability_AB = 3 / 10 :=
sorry

end probability_both_A_and_B_selected_l478_478418


namespace problem1_problem2_l478_478988

theorem problem1 : (sqrt 48 / sqrt 3 - 2 * sqrt (1/5) * sqrt 30 + sqrt 24 = 4) := by
  sorry

theorem problem2 : ((2 * sqrt 3 - 1) ^ 2 + (sqrt 3 + 2) * (sqrt 3 - 2) = 12 - 4 * sqrt 3) := by
  sorry

end problem1_problem2_l478_478988


namespace ratio_of_areas_l478_478037

-- Conditions
def sides_PQR : ℝ × ℝ × ℝ := (6, 8, 10)
def sides_XYZ : ℝ × ℝ × ℝ := (7, 24, 25)

-- Theorem statement
theorem ratio_of_areas (hPQR : sides_PQR = (6, 8, 10)) (hXYZ : sides_XYZ = (7, 24, 25)) : 
  let area_PQR := (1 / 2) * 6 * 8,
      area_XYZ := (1 / 2) * 7 * 24
  in area_PQR / area_XYZ = 2 / 7 := 
  by sorry

end ratio_of_areas_l478_478037


namespace basketball_preference_related_to_gender_basketball_preference_ratio_proof_l478_478879

theorem basketball_preference_related_to_gender :
  let (a, b, c, d) := (30, 10, 20, 40)
  let n := a + b + c + d
  let chi_squared := n * (a * d - b * c) ^ 2 / ((a + b) * (c + d) * (a + c) * (b + d))
  chi_squared > 10.828 :=
by
  -- Data from the contingency table
  have h_a : a = 30 := by rfl
  have h_b : b = 10 := by rfl
  have h_c : c = 20 := by rfl
  have h_d : d = 40 := by rfl
  have h_n : n = 100 := by rfl
  have h_chi_squared : chi_squared = 100 * (30 * 40 - 10 * 20) ^ 2 / (40 * 60 * 50 * 50) := by sorry
  have h_chi_squared_value : chi_squared = 16.667 := by sorry
  show 16.667 > 10.828
  -- Here the explicit calculation would go to prove the statement
  sorry

theorem basketball_preference_ratio_proof : 
  let P (A B : Prop) : ℚ := sorry
  let P_cond (A B : Prop) : ℚ := sorry
  let R := (P ⟨¬B⟩ / P ⟨B⟩) * (P ⟨A⟩ / P ⟨¬A⟩)
  let P_B_A := 1/3
  let P_A_B := 3/4
  R = (P_B_A / (1-P_B_A)) * ((1-P_A_B) / P_A_B) :=
by
  -- Using provided data and definitions
  have h_P_B_A : P ⟨¬B⟩ = 1/3 := by rfl
  have h_P_A_B : P ⟨A⟩ = 3/4 := by rfl
  have h_R_value : R = (1/3) / (2/3) * (1/4) / (3/4) := by sorry
  have h_R_final : R = 1/6 := by sorry
  show
    (P ⟨¬B⟩ / (1 - P ⟨¬B⟩)) * ((1 - P ⟨A⟩) / P ⟨A⟩) = 1/6
  -- Here the explicit calculation would go to prove the statement
  sorry

end basketball_preference_related_to_gender_basketball_preference_ratio_proof_l478_478879


namespace probability_A_and_B_selected_l478_478523

theorem probability_A_and_B_selected :
  let n := 5
  let r := 3
  let total_ways := Nat.choose n r
  let favorable_ways := Nat.choose 3 1
  let probability := favorable_ways / total_ways
  probability = (3 / 10) := 
by
  sorry

end probability_A_and_B_selected_l478_478523


namespace trajectory_of_point_P_l478_478723

-- Define the structure of a regular triangular pyramid.
structure RegularTriangularPyramid (S A B C P : Type) :=
  (inside_base : P → Bool)
  (distance_to_face_SAB : P → ℝ)
  (distance_to_face_SBC : P → ℝ)
  (distance_to_face_SAC : P → ℝ)
  (arithmetic_progression : P → Prop)
  (trajectory_is_straight_line : Prop)

-- Define the conditions for a regular triangular pyramid S-ABC and point P.
def conditions (S A B C P : Type) [RegularTriangularPyramid S A B C P] : Prop :=
  ∀ (p : P), 
    RegularTriangularPyramid.inside_base p ⇒ 
    RegularTriangularPyramid.arithmetic_progression p

-- Define the theorem statement.
theorem trajectory_of_point_P (S A B C P : Type) [RegularTriangularPyramid S A B C P] 
  (h : conditions S A B C P) :
  ∀ (p : P), 
    RegularTriangularPyramid.inside_base p → 
    RegularTriangularPyramid.arithmetic_progression p → 
    RegularTriangularPyramid.trajectory_is_straight_line :=
sorry

end trajectory_of_point_P_l478_478723


namespace probability_A_B_selected_l478_478364

noncomputable def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

def combination (n r : ℕ) : ℕ :=
  factorial n / (factorial r * factorial (n - r))

theorem probability_A_B_selected :
  let total_ways := combination 5 3 in
  let ways_A_B := combination 3 1 in
  (ways_A_B : ℚ) / total_ways = 3 / 10 :=
by
  sorry

end probability_A_B_selected_l478_478364


namespace probability_both_A_B_selected_l478_478213

-- Define the problem with the necessary conditions.
def probability_AB_selected (total_students : ℕ) (select_students : ℕ) (A B : ℕ) : ℚ :=
  if A < total_students ∧ B < total_students ∧ total_students = 5 ∧ select_students = 3 then
    let total_ways := (Nat.factorial total_students) / ((Nat.factorial select_students) * (Nat.factorial (total_students - select_students)))
    let favorable_ways := (Nat.factorial (total_students - 2)) / ((Nat.factorial (select_students - 2)) * (Nat.factorial ((total_students - 2) - (select_students - 2))))
    favorable_ways / total_ways
  else 0

theorem probability_both_A_B_selected 
  (total_students : ℕ) (select_students : ℕ) (A B : ℕ) 
  (h1 : total_students = 5) (h2 : select_students = 3) 
  (h3 : A < total_students) (h4 : B < total_students) 
  : probability_AB_selected total_students select_students A B = 3 / 10 := 
by {
  -- Insert the delta and logic to prove the theorem here.
  sorry
}

end probability_both_A_B_selected_l478_478213


namespace probability_of_selecting_A_and_B_l478_478678

open_locale classical

noncomputable def select_probability (n k : ℕ) : ℚ :=
  (nat.choose n k)⁻¹

theorem probability_of_selecting_A_and_B :
  let total_students := 5,
      students_to_select := 3,
      remaining_students := 3 in
  select_probability total_students students_to_select * nat.choose remaining_students 1 = 3 / 10 :=
by
  sorry

end probability_of_selecting_A_and_B_l478_478678


namespace prob_select_A_and_B_l478_478655

theorem prob_select_A_and_B (students : Finset ℕ) (A B : ℕ) (hA : A ∈ students) (hB : B ∈ students) (h_card : students.card = 5) :
  (students.choose 3).count {comb | A ∈ comb ∧ B ∈ comb} / (students.choose 3).card = 3 / 10 := sorry

end prob_select_A_and_B_l478_478655


namespace probability_A_and_B_selected_l478_478314

open Finset

noncomputable def total_ways : ℕ := (choose 5 3)

noncomputable def favorable_ways : ℕ := (choose 3 1)

noncomputable def probability (total favorable : ℕ) : ℚ := favorable / total

theorem probability_A_and_B_selected :
  probability total_ways favorable_ways = 3 / 10 :=
by
  sorry

end probability_A_and_B_selected_l478_478314


namespace min_n_value_l478_478741

theorem min_n_value (n : ℕ) (h1 : n > 0) (h2 : ∃ k : ℕ, k * k = 12 * n) : n ≥ 3 :=
by
  sorry

example : min_n_value 3 (by norm_num) (by norm_num; use 6; norm_num) := 
by
  sorry

end min_n_value_l478_478741


namespace paper_roll_length_l478_478971

/-- Data:
  w: width of the paper in cm = 3 cm
  d_initial: initial diameter of the rod = 1 cm
  n: number of wraps = 900
  d_final: final diameter of the roll = 13 cm
We need to prove that the length of the paper is approximately 27 * π meters.
-/
theorem paper_roll_length (w d_initial: ℝ) (n: ℕ) (d_final: ℝ) :
  (w = 3) → (d_initial = 1) → (n = 900) → (d_final = 13) →
  ∃ L : ℝ, L = 27 * Real.pi :=
by
  intro h_w h_initial h_n h_final
  use 27 * Real.pi
  sorry

end paper_roll_length_l478_478971


namespace quadratic_completing_square_l478_478902

theorem quadratic_completing_square :
  ∃ a b c : ℚ,
    -3 * a = -3 ∧
    27 * b = 27 ∧
    -153 = a ((- (b / 2)) ^ 2 - (b / 2) ^ 2) + c ∧
    a + b + c = -99.75 :=
by
  sorry

end quadratic_completing_square_l478_478902


namespace probability_of_A_and_B_selected_l478_478480

variable {α : Type}

/-- From 5 students, we randomly select 3 students to participate in community service work. 
  Let A and B be two specific students among these five. The probability that 
  both A and B are selected is 3 / 10. -/
theorem probability_of_A_and_B_selected (A B : α) (s : Finset α) (h : s.card = 5) :
  (∃ t : Finset α, t ⊆ s ∧ A ∈ t ∧ B ∈ t ∧ (t.card = 3) ∧ (t.card = 3) →
   (@Finset.exists_subset_card_eq α t s 3) →
   (@Finset.exists_subset_card_eq α t (Finset.erase s A) 2) ∧
   (@Finset.exists_subset_card_eq α t (Finset.erase (Finset.erase s A) B) 1)) →
  (3 : ℚ) / 10 :=
by
  sorry -- Proof omitted as per instructions.

end probability_of_A_and_B_selected_l478_478480


namespace determine_b_l478_478091

theorem determine_b (b n : ℝ) (hn : n^2 + 1/20 = 1/5) (hb : b = 2 * n) :
  b = -real.sqrt (3/5) :=
by
  sorry

end determine_b_l478_478091


namespace distinct_prime_factors_count_l478_478145

theorem distinct_prime_factors_count :
  ∀ (a b c d : ℕ),
  (a = 79) → (b = 3^4) → (c = 5 * 17) → (d = 3 * 29) →
  (∃ s : Finset ℕ, ∀ n ∈ s, Nat.Prime n ∧ 79 * 81 * 85 * 87 = s.prod id) :=
sorry

end distinct_prime_factors_count_l478_478145


namespace smallest_k_repr_19_pow_n_sub_5_pow_m_exists_l478_478196

theorem smallest_k_repr_19_pow_n_sub_5_pow_m_exists :
  ∃ (k n m : ℕ), k > 0 ∧ n > 0 ∧ m > 0 ∧ k = 19 ^ n - 5 ^ m ∧ k = 14 :=
by
  sorry

end smallest_k_repr_19_pow_n_sub_5_pow_m_exists_l478_478196


namespace probability_of_selecting_A_and_B_l478_478274

-- Define the combination function
def combination (n r : ℕ) : ℕ := Nat.factorial n / (Nat.factorial r * Nat.factorial (n - r))

-- Define the conditions
def students : ℕ := 5
def to_select : ℕ := 3

-- Define the favorable outcomes where A and B are selected
def favorable_outcomes : ℕ := combination 3 1

-- Define the total outcomes of selecting 3 out of 5 students
def total_outcomes : ℕ := combination 5 3

-- Define the probability calculation
def probability_of_A_and_B : ℚ := favorable_outcomes / total_outcomes

-- Proposition to prove
theorem probability_of_selecting_A_and_B : probability_of_A_and_B = 3 / 10 :=
by
  sorry

end probability_of_selecting_A_and_B_l478_478274


namespace probability_both_A_B_selected_l478_478205

-- Define the problem with the necessary conditions.
def probability_AB_selected (total_students : ℕ) (select_students : ℕ) (A B : ℕ) : ℚ :=
  if A < total_students ∧ B < total_students ∧ total_students = 5 ∧ select_students = 3 then
    let total_ways := (Nat.factorial total_students) / ((Nat.factorial select_students) * (Nat.factorial (total_students - select_students)))
    let favorable_ways := (Nat.factorial (total_students - 2)) / ((Nat.factorial (select_students - 2)) * (Nat.factorial ((total_students - 2) - (select_students - 2))))
    favorable_ways / total_ways
  else 0

theorem probability_both_A_B_selected 
  (total_students : ℕ) (select_students : ℕ) (A B : ℕ) 
  (h1 : total_students = 5) (h2 : select_students = 3) 
  (h3 : A < total_students) (h4 : B < total_students) 
  : probability_AB_selected total_students select_students A B = 3 / 10 := 
by {
  -- Insert the delta and logic to prove the theorem here.
  sorry
}

end probability_both_A_B_selected_l478_478205


namespace probability_of_selecting_A_and_B_l478_478675

open_locale classical

noncomputable def select_probability (n k : ℕ) : ℚ :=
  (nat.choose n k)⁻¹

theorem probability_of_selecting_A_and_B :
  let total_students := 5,
      students_to_select := 3,
      remaining_students := 3 in
  select_probability total_students students_to_select * nat.choose remaining_students 1 = 3 / 10 :=
by
  sorry

end probability_of_selecting_A_and_B_l478_478675


namespace cyclist_wait_time_l478_478061

-- Given conditions
def hiker_speed := 4  -- in km/h
def cyclist_speed := 18  -- in km/h
def wait_time := 5 / 60  -- in hours (5 minutes)

-- Distance the cyclist travels in the wait time
def distance_cyclist := cyclist_speed * wait_time

-- Distance the hiker travels in the wait time
def distance_hiker := hiker_speed * wait_time

-- The difference in distance between cyclist and hiker after 5 minutes
def distance_difference := distance_cyclist - distance_hiker

-- Time needed for hiker to catch up in hours
def time_hiker := distance_difference / hiker_speed

-- Time needed for hiker to catch up in minutes
def time_hiker_minutes := time_hiker * 60

theorem cyclist_wait_time:
  time_hiker_minutes = 17.5 :=
sorry

end cyclist_wait_time_l478_478061


namespace probability_A_and_B_selected_l478_478246

theorem probability_A_and_B_selected (students : Finset ℕ) (A B : ℕ) (h : A ∈ students ∧ B ∈ students ∧ students.card = 5) : 
  (students.choose 3).count (λ s, {A, B} ⊆ s) / (students.choose 3).card = 3 / 10 := by
  sorry

end probability_A_and_B_selected_l478_478246


namespace max_cookies_ben_could_have_eaten_l478_478040

theorem max_cookies_ben_could_have_eaten (c : ℕ) (h_total : c = 36)
  (h_beth : ∃ n: ℕ, (n = 2 ∨ n = 3) ∧ c = (n + 1) * ben)
  (h_max : ∀ n, (n = 2 ∨ n = 3) → n * 12 ≤ n * ben)
  : ben = 12 := 
sorry

end max_cookies_ben_could_have_eaten_l478_478040


namespace first_day_of_month_l478_478877

theorem first_day_of_month 
  (d_24: ℕ) (mod_7: d_24 % 7 = 6) : 
  (d_24 - 23) % 7 = 4 :=
by 
  -- denotes the 24th day is a Saturday (Saturday is the 6th day in a 0-6 index)
  -- hence mod_7: d_24 % 7 = 6 means d_24 falls on a Saturday
  sorry

end first_day_of_month_l478_478877


namespace probability_both_A_and_B_selected_l478_478419

-- The universe of students we consider
inductive Student
| A : Student
| B : Student
| C : Student
| D : Student
| E : Student

open Student

-- Function to count combinations
def combinations (n k : ℕ) : ℕ := (Nat.factorial n) / ((Nat.factorial k) * (Nat.factorial (n - k)))

-- The set of all students
def students : List Student := [A, B, C, D, E]

-- The total number of ways to select 3 students from 5
def total_selections := combinations 5 3

-- The number of ways A and B are both selected by selecting 1 more from the remaining 3 students
def selections_AB := combinations 3 1

-- The probability that both A and B are selected
def probability_AB := (selections_AB : ℚ) / (total_selections : ℚ)

-- The proof statement
theorem probability_both_A_and_B_selected :
  probability_AB = 3 / 10 :=
sorry

end probability_both_A_and_B_selected_l478_478419


namespace probability_A_and_B_selected_l478_478313

open Finset

noncomputable def total_ways : ℕ := (choose 5 3)

noncomputable def favorable_ways : ℕ := (choose 3 1)

noncomputable def probability (total favorable : ℕ) : ℚ := favorable / total

theorem probability_A_and_B_selected :
  probability total_ways favorable_ways = 3 / 10 :=
by
  sorry

end probability_A_and_B_selected_l478_478313


namespace probability_both_A_B_selected_l478_478216

-- Define the problem with the necessary conditions.
def probability_AB_selected (total_students : ℕ) (select_students : ℕ) (A B : ℕ) : ℚ :=
  if A < total_students ∧ B < total_students ∧ total_students = 5 ∧ select_students = 3 then
    let total_ways := (Nat.factorial total_students) / ((Nat.factorial select_students) * (Nat.factorial (total_students - select_students)))
    let favorable_ways := (Nat.factorial (total_students - 2)) / ((Nat.factorial (select_students - 2)) * (Nat.factorial ((total_students - 2) - (select_students - 2))))
    favorable_ways / total_ways
  else 0

theorem probability_both_A_B_selected 
  (total_students : ℕ) (select_students : ℕ) (A B : ℕ) 
  (h1 : total_students = 5) (h2 : select_students = 3) 
  (h3 : A < total_students) (h4 : B < total_students) 
  : probability_AB_selected total_students select_students A B = 3 / 10 := 
by {
  -- Insert the delta and logic to prove the theorem here.
  sorry
}

end probability_both_A_B_selected_l478_478216


namespace sqrt_500_eq_10_sqrt_5_l478_478867

theorem sqrt_500_eq_10_sqrt_5 : sqrt 500 = 10 * sqrt 5 :=
  sorry

end sqrt_500_eq_10_sqrt_5_l478_478867


namespace fraction_equality_l478_478987

noncomputable def x := (4 : ℚ) / 6
noncomputable def y := (8 : ℚ) / 12

theorem fraction_equality : (6 * x + 8 * y) / (48 * x * y) = (7 : ℚ) / 16 := 
by 
  sorry

end fraction_equality_l478_478987


namespace floor_of_factorial_fraction_l478_478944

theorem floor_of_factorial_fraction :
  (Int.floor $ (2007.factorial + 2004.factorial : ℚ) / (2006.factorial + 2005.factorial : ℚ)) = 2006 := 
sorry

end floor_of_factorial_fraction_l478_478944


namespace probability_A_and_B_selected_l478_478701

/-- There are 5 students including A and B. The probability of selecting both A and B 
    when 3 students are selected randomly from 5 is 3/10. -/
theorem probability_A_and_B_selected :
  (∃ (students : List String) (h : students.length = 5),
    ∃ (selected : List String) (h' : selected.length = 3),
    "A" ∈ selected ∧ "B" ∈ selected) →
  ∃ (P : ℚ), P = 3 / 10 :=
by
  sorry

end probability_A_and_B_selected_l478_478701


namespace carlos_blocks_after_exchanges_l478_478989

def initial_blocks : ℕ := 58
def fraction_given_to_Rachel : ℚ := 2/5
def fraction_exchanged_with_Nicole : ℚ := 1/2

theorem carlos_blocks_after_exchanges :
  let blocks_given_to_Rachel := (initial_blocks : ℚ) * fraction_given_to_Rachel,
      blocks_left_after_Rachel := initial_blocks - blocks_given_to_Rachel.nat_floor,
      blocks_exchanged_with_Nicole := (blocks_left_after_Rachel : ℚ) * fraction_exchanged_with_Nicole,
      blocks_left_after_Nicole := blocks_left_after_Rachel - blocks_exchanged_with_Nicole.nat_floor
  in blocks_left_after_Nicole = 18 := 
by
  sorry

end carlos_blocks_after_exchanges_l478_478989


namespace brenda_peaches_left_brenda_peaches_left_correct_l478_478121

theorem brenda_peaches_left (total_peaches : ℕ) (fresh_pct : ℝ) (too_small_peaches : ℕ)
  (h1 : total_peaches = 250)
  (h2 : fresh_pct = 0.60)
  (h3 : too_small_peaches = 15) : ℕ :=
sorry

theorem brenda_peaches_left_correct : brenda_peaches_left 250 0.60 15 = 135 :=
by {
  rw brenda_peaches_left,
  exact sorry
}

end brenda_peaches_left_brenda_peaches_left_correct_l478_478121


namespace probability_of_selecting_A_and_B_l478_478258

-- Define the combination function
def combination (n r : ℕ) : ℕ := Nat.factorial n / (Nat.factorial r * Nat.factorial (n - r))

-- Define the conditions
def students : ℕ := 5
def to_select : ℕ := 3

-- Define the favorable outcomes where A and B are selected
def favorable_outcomes : ℕ := combination 3 1

-- Define the total outcomes of selecting 3 out of 5 students
def total_outcomes : ℕ := combination 5 3

-- Define the probability calculation
def probability_of_A_and_B : ℚ := favorable_outcomes / total_outcomes

-- Proposition to prove
theorem probability_of_selecting_A_and_B : probability_of_A_and_B = 3 / 10 :=
by
  sorry

end probability_of_selecting_A_and_B_l478_478258


namespace number_to_add_l478_478057

theorem number_to_add (n : ℕ) (k : ℕ) (d : ℕ) (h : n % k = 0) : d = 0 :=
by
  assume n k d h
  let number_needed_to_add := k - (n % k)
  have h1 : d = number_needed_to_add,
  from sorry

example : ∃ d, (1782452 + d) % 167 = 0 :=
begin
  use 0,
  exact number_to_add 1782452 167 0 (by norm_num),
end

end number_to_add_l478_478057


namespace bus_interval_l478_478979

theorem bus_interval (total_hours_per_day : ℕ) (total_days : ℕ) (total_buses : ℕ) 
  (H1 : total_hours_per_day = 12)
  (H2 : total_days = 5)
  (H3 : total_buses = 120) : 
  let total_time_in_minutes := total_hours_per_day * 60 in
  let buses_per_day := total_buses / total_days in
  let interval := total_time_in_minutes / buses_per_day in
  interval = 30 :=
by
  sorry

end bus_interval_l478_478979


namespace tetrahedron_inscribed_sphere_property_l478_478854

theorem tetrahedron_inscribed_sphere_property 
  (G O A B C D A' B' C' D' : Type) 
  [regular_tetrahedron ABCD] 
  (inside_tetrahedron O : point_in_tetrahedron O ABCD) 
  (line_OG : line O G)
  (intersections : line_intersects_faces OG A' B' C' D') 
  (G_center : center_of_inscribed_sphere G ABCD) 
  :
  (\frac{dist O A'}{dist G A'} + \frac{dist O B'}{dist G B'} + \frac{dist O C'}{dist G C'} + \frac{dist O D'}{dist G D'}) = 4 :=
sorry

end tetrahedron_inscribed_sphere_property_l478_478854


namespace probability_AB_selected_l478_478508

def factorial (n : Nat) : Nat := if n = 0 then 1 else n * factorial (n - 1)

def choose (n k : Nat) : Nat := factorial n / (factorial k * factorial (n - k))

theorem probability_AB_selected (h1 : 5 = 5) (h2 : 3 = 3) : (choose 3 1) / (choose 5 3) = 3 / 10 := by
  sorry

end probability_AB_selected_l478_478508


namespace probability_of_selecting_A_and_B_l478_478542

theorem probability_of_selecting_A_and_B (students : Finset ℕ) (A B : ℕ)
  (h_size : students.card = 5) (h_in : A ∈ students ∧ B ∈ students) : 
  (Finset.card {s : Finset ℕ // s ⊆ students ∧ s.card = 3 ∧ A ∈ s ∧ B ∈ s}.to_set) / 
  (Finset.card {s : Finset ℕ // s ⊆ students ∧ s.card = 3}.to_set) = 3 / 10 :=
sorry

end probability_of_selecting_A_and_B_l478_478542


namespace probability_of_selecting_A_and_B_is_three_tenths_l478_478602

-- Define the set of students
def students : Finset ℕ := {0, 1, 2, 3, 4}

-- Define subsets A and B
def student_A : ℕ := 0
def student_B : ℕ := 1

-- Total number of ways to select 3 students out of 5
def total_ways : ℕ := students.card.choose 3

-- Number of ways to select 3 students including A and B
def favorable_ways : ℕ := {2, 3, 4}.card.choose 1

-- The probability
def probability_of_selecting_A_and_B : ℚ := (favorable_ways : ℚ) / (total_ways : ℚ)

theorem probability_of_selecting_A_and_B_is_three_tenths :
  probability_of_selecting_A_and_B = 3 / 10 :=
by
  sorry

end probability_of_selecting_A_and_B_is_three_tenths_l478_478602


namespace sqrt_500_simplified_l478_478870

theorem sqrt_500_simplified : sqrt 500 = 10 * sqrt 5 :=
by
sorry

end sqrt_500_simplified_l478_478870


namespace kiley_slices_eaten_l478_478155

def slices_of_cheesecake (total_calories_per_cheesecake calories_per_slice : ℕ) : ℕ :=
  total_calories_per_cheesecake / calories_per_slice

def slices_eaten (total_slices percentage_ate : ℚ) : ℚ :=
  total_slices * percentage_ate

theorem kiley_slices_eaten :
  ∀ (total_calories_per_cheesecake calories_per_slice : ℕ) (percentage_ate : ℚ),
  total_calories_per_cheesecake = 2800 →
  calories_per_slice = 350 →
  percentage_ate = (25 / 100 : ℚ) →
  slices_eaten (slices_of_cheesecake total_calories_per_cheesecake calories_per_slice) percentage_ate = 2 :=
by
  intros total_calories_per_cheesecake calories_per_slice percentage_ate h1 h2 h3
  rw [h1, h2, h3]
  sorry

end kiley_slices_eaten_l478_478155


namespace can_place_circles_l478_478938

theorem can_place_circles (r: ℝ) (h: r = 2008) :
  ∃ (n: ℕ), (n > 4016) ∧ ((n: ℝ) / 2 > r) :=
by 
  sorry

end can_place_circles_l478_478938


namespace median_of_list_1_to_5000_and_squares_1_to_200_l478_478051

theorem median_of_list_1_to_5000_and_squares_1_to_200 :
  let nums := (list.range 5000).map (λ x, x + 1) ++ (list.range 200).map (λ y, (y + 1) ^ 2)
  in (list_sorted_median nums = 2600.5) :=
by
  sorry

end median_of_list_1_to_5000_and_squares_1_to_200_l478_478051


namespace integral_equation_solution_l478_478001

noncomputable def phi (x : ℝ) : ℝ := x

theorem integral_equation_solution :
  ∀ x : ℝ, (φ x) = x → (φ x) = ∫ t in 0..x, (1 + (φ t)^2) / (1 + t^2) :=
by
  intros x H
  sorry

end integral_equation_solution_l478_478001


namespace max_hua_bei_jue_sai_l478_478798

theorem max_hua_bei_jue_sai : ∃ N : ℕ, N = 2011 - 100 - 10 ∧ N = 1901 := 
by
  use 1901
  -- Setup the conditions based on the problem
  have two_digit := 10
  have three_digit := 100
  -- Use the conditions to state the problem
  have maximum_value := 2011 - three_digit - two_digit
  -- Show that with these values, the maximum possible value is 1901
  rw [maximum_value]
  -- Ensure we use the given conditions accurately
  exact 2011 - 100 - 10 = 1901
  sorry

end max_hua_bei_jue_sai_l478_478798


namespace probability_of_selecting_A_and_B_l478_478681

open_locale classical

noncomputable def select_probability (n k : ℕ) : ℚ :=
  (nat.choose n k)⁻¹

theorem probability_of_selecting_A_and_B :
  let total_students := 5,
      students_to_select := 3,
      remaining_students := 3 in
  select_probability total_students students_to_select * nat.choose remaining_students 1 = 3 / 10 :=
by
  sorry

end probability_of_selecting_A_and_B_l478_478681


namespace factorize_quadratic_l478_478177

theorem factorize_quadratic (x : ℝ) : 
  16 * x^2 - 40 * x + 25 = (4 * x - 5)^2 :=
sorry

end factorize_quadratic_l478_478177


namespace parabola_condition_lambda_condition_l478_478722

noncomputable def parabola_equation (p : ℝ) (h : p > 0) : Prop :=
∀ (x y : ℝ), y^2 = 2 * p * x

noncomputable def point_on_line (m : ℝ) (x y x₀ y₀ : ℝ) : Prop :=
y = m * (x - x₀)

noncomputable def distance (A B : ℝ × ℝ) := 
real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)

theorem parabola_condition (p : ℝ) (h : p > 0)
  (x1 x2 y1 y2 : ℝ) 
  (hx1 : x1 < x2) 
  (h_intersect : point_on_line (2 * real.sqrt 2) x1 y1 (p / 2) 0 ∧ point_on_line (2 * real.sqrt 2) x2 y2 (p / 2) 0)
  (h_dist : distance (x1, y1) (x2, y2) = 9) :
  (∀ (x y : ℝ), y^2 = 8 * x) :=
sorry

theorem lambda_condition (p : ℝ) (h : p = 4) 
  (x1 y1 x2 y2 : ℝ)
  (h_A : (x1, y1) = (1, -2 * real.sqrt 2))
  (h_B : (x2, y2) = (4, 4 * real.sqrt 2))
  (λ : ℝ)
  (h_oc : ∀ (x3 y3 : ℝ), (x3, y3) = (x1 + λ * x2, y1 + λ * y2) → y3^2 = 8 * x3) :
  λ = 0 ∨ λ = 2 :=
sorry

end parabola_condition_lambda_condition_l478_478722


namespace paint_sectors_l478_478152

-- Define the problem conditions
def sectors (n : ℕ) (h : n ≥ 2) : Type := Fin n

-- Define the painting problem
def painting (n : ℕ) (h : n ≥ 2) (c : sectors n h → Prop) : Prop :=
  ∀ (i : sectors n h),
    (c i = red ∨ c i = white ∨ c i = blue) ∧
    ∀ (j : sectors n h), (i ≠ j → adjacent i j → c i ≠ c j)

-- Problem statement in Lean
theorem paint_sectors (n : ℕ) (h : n ≥ 2) :
  ∃ a_n : ℕ, 
    (a_n = 2 * (-1)^n + 2^n) ∧
    (a_2 = 6) ∧
    (∀ m : ℕ, m ≥ 3 → a_m + a_{m-1} = 3 * 2^(m-1)) :=
begin
  sorry
end

end paint_sectors_l478_478152


namespace teacherB_teaches_C_l478_478876

variables (Teacher : Type) (Subject : Type) (Place : Type)

variables (A B C : Teacher) (S_A S_B S_C : Subject) (H Cc Sh : Place)

-- Definitions based on the conditions
axiom teacherA_not_in_Harbin : ¬ (A = H)
axiom teacherB_not_in_Changchun : ¬ (B = Cc)
axiom teacher_in_Harbin_not_teach_C : ∀ (t : Teacher), t = H → ¬ (t = C)
axiom teacher_in_Changchun_teach_A : ∀ (t : Teacher), t = Cc → t = A
axiom teacherB_not_teach_B : ¬ (S_B = B)

-- Proof statement
theorem teacherB_teaches_C : S_B = C :=
sorry

end teacherB_teaches_C_l478_478876


namespace factorization_identity_l478_478170

theorem factorization_identity (x : ℝ) : 
  16 * x^2 - 40 * x + 25 = (4 * x - 5)^2 :=
  sorry

end factorization_identity_l478_478170


namespace problem_solution_l478_478817

theorem problem_solution :
  let A := List.foldl (*) 1 (List.prod (List.divisors 60)) in
  (nat.distinct_prime_factors A = 3) ∧ (nat.num_divisors A = 637) :=
by
  sorry

end problem_solution_l478_478817


namespace probability_of_A_and_B_selected_l478_478476

variable {α : Type}

/-- From 5 students, we randomly select 3 students to participate in community service work. 
  Let A and B be two specific students among these five. The probability that 
  both A and B are selected is 3 / 10. -/
theorem probability_of_A_and_B_selected (A B : α) (s : Finset α) (h : s.card = 5) :
  (∃ t : Finset α, t ⊆ s ∧ A ∈ t ∧ B ∈ t ∧ (t.card = 3) ∧ (t.card = 3) →
   (@Finset.exists_subset_card_eq α t s 3) →
   (@Finset.exists_subset_card_eq α t (Finset.erase s A) 2) ∧
   (@Finset.exists_subset_card_eq α t (Finset.erase (Finset.erase s A) B) 1)) →
  (3 : ℚ) / 10 :=
by
  sorry -- Proof omitted as per instructions.

end probability_of_A_and_B_selected_l478_478476


namespace min_distance_origin_to_line_l478_478852

theorem min_distance_origin_to_line : 
  let line_eq : ℝ × ℝ → Prop := λ P, P.1 + P.2 - 4 = 0 in
  let dist_to_origin : ℝ → Prop := λ d, ∃ P : ℝ × ℝ, line_eq P ∧ d = real.sqrt ((P.1)^2 + (P.2)^2) in
  ∀ d, dist_to_origin d → d = 2 * real.sqrt 2 :=
sorry

end min_distance_origin_to_line_l478_478852


namespace area_of_triangle_l478_478927

def line1 (x : ℝ) : ℝ := 3 * x + 6
def line2 (x : ℝ) : ℝ := -2 * x + 10

theorem area_of_triangle : 
  let inter_x := (10 - 6) / (3 + 2)
  let inter_y := line1 inter_x
  let base := (10 - 6 : ℝ)
  let height := inter_x
  base * height / 2 = 8 / 5 := 
by
  sorry

end area_of_triangle_l478_478927


namespace cole_avg_speed_to_work_l478_478130

variable (v1 : ℝ) -- Cole's average speed driving to work
variable (v2 : ℝ := 105) -- Cole's average speed driving back home
variable (t_total : ℝ := 4) -- Total round trip time in hours
variable (t_work_min : ℕ := 140) -- Time to drive to work in minutes
variable (minutes_in_hour : ℝ := 60) -- Conversion factor from minutes to hours

-- Convert time to drive to work to hours
def t_work_hours : ℝ := t_work_min / minutes_in_hour

-- Total distance for the round trip
def d_work : ℝ := t_work_hours * v1

-- Total time driving back home (in hours)
def t_home_hours : ℝ := t_total - t_work_hours

-- Distance = Speed × Time for the trip back home
def d_home : ℝ := t_home_hours * v2

-- Assertion: The distance to work and back should be equal
def distance_equality : Prop := d_work = d_home

-- Prove that Cole's average speed driving to work was approximately 75 km/h
theorem cole_avg_speed_to_work : v1 ≈ 75 :=
by 
  let h1 : t_work_hours = 140 / 60 := rfl
  let h2 : t_home_hours = t_total - t_work_hours := rfl
  let h3 : d_work = t_work_hours * v1 := rfl
  let h4 : d_home = t_home_hours * v2 := rfl
  have : distance_equality := sorry

end cole_avg_speed_to_work_l478_478130


namespace probability_of_selecting_A_and_B_l478_478666

open_locale classical

noncomputable def select_probability (n k : ℕ) : ℚ :=
  (nat.choose n k)⁻¹

theorem probability_of_selecting_A_and_B :
  let total_students := 5,
      students_to_select := 3,
      remaining_students := 3 in
  select_probability total_students students_to_select * nat.choose remaining_students 1 = 3 / 10 :=
by
  sorry

end probability_of_selecting_A_and_B_l478_478666


namespace quadrilateral_rhombus_l478_478800

variables {V : Type*} [inner_product_space ℝ V] [decidable_eq V]
variables {A B C D : V}

theorem quadrilateral_rhombus (h1 : A - B = D - C) (h2 : ⟪A - C, B - D⟫ = 0) : 
  let AB := A - B,
      AC := A - C,
      BD := B - D,
      CD := C - D in
  AB = CD ∧ ⟪AC, BD⟫ = 0 → parallelogram V A B C D ∧ ⟪AC, BD⟫ = 0 → rhombus V A B C D :=
sorry

end quadrilateral_rhombus_l478_478800


namespace circumradius_of_right_triangle_l478_478083

theorem circumradius_of_right_triangle (a b c : ℝ) (h : a^2 + b^2 = c^2) : 
  ∃ R : ℝ, R = c / 2 :=
begin
  use c / 2,
  exact sorry
end

noncomputable def triangle_example := (8, 15, 17)

example : ∃ R : ℝ, R = 17 / 2 :=
begin
  let ⟨a, b, c⟩ := triangle_example,
  have h : a^2 + b^2 = c^2 := by norm_num, -- 8^2 + 15^2 = 17^2
  apply circumradius_of_right_triangle a b c h
end

end circumradius_of_right_triangle_l478_478083


namespace probability_A_and_B_selected_l478_478302

open Finset

noncomputable def total_ways : ℕ := (choose 5 3)

noncomputable def favorable_ways : ℕ := (choose 3 1)

noncomputable def probability (total favorable : ℕ) : ℚ := favorable / total

theorem probability_A_and_B_selected :
  probability total_ways favorable_ways = 3 / 10 :=
by
  sorry

end probability_A_and_B_selected_l478_478302


namespace select_3_from_5_prob_l478_478437

theorem select_3_from_5_prob {students : Finset ℕ} 
  (h : students.card = 5) 
  {A B : ℕ} 
  (hA : A ∈ students) 
  (hB : B ∈ students) 
  (H : 3 ≤ students.card) : 
  (finset.card (students.filter (λ x, x ≠ A ∧ x ≠ B)) = 3) → 
  (finset.card ((students.filter (λ x, x ≠ A ∧ x ≠ B)).subtype (λ x, True)) = 3) → 
  ((finset.card (finset.filter (λ (students : Finset ℕ) A ∈ students ∧ B ∈ students ↔ 3 = students.card) 5)) = ↑10) → 
  ∃ students', (students'.card = 3 ∧ A ∈ students' ∧ B ∈ students') → 
  ∃ (p : ℚ), p = 3 / 10 ∧ students.filter (λ x, x = A ∨ x = B) = 2 :=
  sorry

end select_3_from_5_prob_l478_478437


namespace andre_flowers_given_l478_478862

variable (initialFlowers totalFlowers flowersGiven : ℕ)

theorem andre_flowers_given (h1 : initialFlowers = 67) (h2 : totalFlowers = 90) :
  flowersGiven = totalFlowers - initialFlowers → flowersGiven = 23 :=
by
  intro h3
  rw [h1, h2] at h3
  simp at h3
  exact h3

end andre_flowers_given_l478_478862


namespace repeating_decimal_eq_fraction_l478_478162

noncomputable def repeating_decimal_to_fraction (x : ℝ) : ℝ :=
  let x : ℝ := 4.5656565656 -- * 0.5656... repeating
  (100*x - x) / (100 - 1)

-- Define the theorem we want to prove
theorem repeating_decimal_eq_fraction : 
  ∀ x : ℝ, x = 4.565656 -> x = (452 : ℝ) / (99 : ℝ) :=
by
  intro x h
  -- here we would provide the proof steps, but since it's omitted
  -- we'll use sorry to skip it.
  sorry

end repeating_decimal_eq_fraction_l478_478162


namespace result_number_of_edges_l478_478850

-- Define the conditions
def hexagon (side_length : ℕ) : Prop := side_length = 1 ∧ (∃ edges, edges = 6 ∧ edges = 6 * side_length)
def triangle (side_length : ℕ) : Prop := side_length = 1 ∧ (∃ edges, edges = 3 ∧ edges = 3 * side_length)

-- State the theorem
theorem result_number_of_edges (side_length_hex : ℕ) (side_length_tri : ℕ)
  (h_h : hexagon side_length_hex) (h_t : triangle side_length_tri)
  (aligned_edge_to_edge : side_length_hex = side_length_tri ∧ side_length_hex = 1 ∧ side_length_tri = 1) :
  ∃ edges, edges = 5 :=
by
  -- Proof is not provided, it is marked with sorry
  sorry

end result_number_of_edges_l478_478850


namespace probability_A_B_l478_478466

open Nat

-- Define a function for combination
def combination (n r : Nat) : Nat :=
  factorial n / (factorial r * factorial (n - r))

-- Define the set of students and the selection conditions
def num_students : Nat := 5
def num_selected : Nat := 3

-- Define A and B as selected students
def selected_A_B : Nat := combination 3 1

-- Define the probability calculation
def probability_A_B_selected : Rat :=
  (selected_A_B : Rat) / (combination num_students num_selected : Rat)

-- The statement of the problem
theorem probability_A_B : probability_A_B_selected = 3 / 10 := by
  sorry

end probability_A_B_l478_478466


namespace handshakes_at_conference_l478_478981

-- Define the number of gremlins and imps
def num_gremlins : ℕ := 25
def num_imps : ℕ := 10

-- Define the number of handshakes among gremlins using the combination formula
def gremlin_gremlin_handshakes : ℕ := num_gremlins * (num_gremlins - 1) / 2

-- Define the number of handshakes between gremlins and imps
def gremlin_imp_handshakes : ℕ := num_imps * num_gremlins

-- Define the total number of handshakes
def total_handshakes : ℕ := gremlin_gremlin_handshakes + gremlin_imp_handshakes

-- The theorem that the total number of handshakes is 550
theorem handshakes_at_conference (hg : num_gremlins = 25) (hi : num_imps = 10) : total_handshakes = 550 :=
by
  rw [hg, hi]
  unfold total_handshakes gremlin_gremlin_handshakes gremlin_imp_handshakes
  norm_num
  sorry

end handshakes_at_conference_l478_478981


namespace find_integer_solutions_l478_478190

theorem find_integer_solutions :
  {p : ℤ × ℤ × ℤ // 0 < p.1 ∧ 0 < p.2.1 ∧ 0 < p.2.2 
    ∧ p.1 ≤ p.2.1 ∧ p.2.1 ≤ p.2.2 
    ∧ (p.1^2 + p.2.1^2 + p.2.2^2 = 2005)} =
   {⟨4, 33, 30⟩, ⟨32, 9, 30⟩, ⟨40, 9, 18⟩, ⟨12, 31, 30⟩, ⟨24, 23, 30⟩, ⟨4, 15, 22⟩, ⟨36, 15, 42⟩} :=
by
  -- Proof steps to be added
  sorry

end find_integer_solutions_l478_478190


namespace water_content_in_boxes_l478_478077

noncomputable def totalWaterInBoxes (num_boxes : ℕ) (bottles_per_box : ℕ) (capacity_per_bottle : ℚ) (fill_fraction : ℚ) : ℚ :=
  num_boxes * bottles_per_box * capacity_per_bottle * fill_fraction

theorem water_content_in_boxes :
  totalWaterInBoxes 10 50 12 (3 / 4) = 4500 := 
by
  sorry

end water_content_in_boxes_l478_478077


namespace probability_A_and_B_selected_l478_478566

/-- Let there be a group of 5 students including A and B. 
    If 3 students are selected randomly from these 5, the probability of selecting both A and B is 3/10. -/
theorem probability_A_and_B_selected : 
  let students := 5 in
  let selected := 3 in
  let combinations := students.choose selected in
  let favorable_combinations := (students - 2).choose (selected - 2) in
  let probability := (favorable_combinations : ℚ) / (combinations : ℚ) in
  probability = 3 / 10 := 
by sorry

end probability_A_and_B_selected_l478_478566


namespace find_a_of_parabola_directrix_l478_478014

theorem find_a_of_parabola_directrix (a : ℝ) :
  (∃ d : ℝ, d = 1 ∧ is_directrix y=ax^2 d) → a = -1/4 :=
by
  sorry

end find_a_of_parabola_directrix_l478_478014


namespace probability_A_and_B_selected_l478_478380

theorem probability_A_and_B_selected 
  (students : Finset ℕ) (A B : ℕ) 
  (h_students_size : students.card = 5) 
  (h_A_in_students : A ∈ students) 
  (h_B_in_students : B ∈ students)
  : (↑(Nat.choose 3) : ℚ) / (↑(Nat.choose 5) : ℚ) = 3 / 10 := 
by sorry

end probability_A_and_B_selected_l478_478380


namespace probability_A_B_selected_l478_478352

noncomputable def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

def combination (n r : ℕ) : ℕ :=
  factorial n / (factorial r * factorial (n - r))

theorem probability_A_B_selected :
  let total_ways := combination 5 3 in
  let ways_A_B := combination 3 1 in
  (ways_A_B : ℚ) / total_ways = 3 / 10 :=
by
  sorry

end probability_A_B_selected_l478_478352


namespace select_3_from_5_prob_l478_478422

theorem select_3_from_5_prob {students : Finset ℕ} 
  (h : students.card = 5) 
  {A B : ℕ} 
  (hA : A ∈ students) 
  (hB : B ∈ students) 
  (H : 3 ≤ students.card) : 
  (finset.card (students.filter (λ x, x ≠ A ∧ x ≠ B)) = 3) → 
  (finset.card ((students.filter (λ x, x ≠ A ∧ x ≠ B)).subtype (λ x, True)) = 3) → 
  ((finset.card (finset.filter (λ (students : Finset ℕ) A ∈ students ∧ B ∈ students ↔ 3 = students.card) 5)) = ↑10) → 
  ∃ students', (students'.card = 3 ∧ A ∈ students' ∧ B ∈ students') → 
  ∃ (p : ℚ), p = 3 / 10 ∧ students.filter (λ x, x = A ∨ x = B) = 2 :=
  sorry

end select_3_from_5_prob_l478_478422


namespace probability_different_colors_l478_478030

theorem probability_different_colors
  (red_chips green_chips : ℕ)
  (total_chips : red_chips + green_chips = 10)
  (prob_red : ℚ := red_chips / 10)
  (prob_green : ℚ := green_chips / 10) :
  ((prob_red * prob_green) + (prob_green * prob_red) = 12 / 25) := by
sorry

end probability_different_colors_l478_478030


namespace log_a_eq_implies_eq_l478_478199

theorem log_a_eq_implies_eq (a M N : ℝ) (h₀ : 0 < a) (h₁ : a ≠ 1) (h₂ : log a M = log a N) : M = N :=
sorry

end log_a_eq_implies_eq_l478_478199


namespace probability_AB_selected_l478_478511

def factorial (n : Nat) : Nat := if n = 0 then 1 else n * factorial (n - 1)

def choose (n k : Nat) : Nat := factorial n / (factorial k * factorial (n - k))

theorem probability_AB_selected (h1 : 5 = 5) (h2 : 3 = 3) : (choose 3 1) / (choose 5 3) = 3 / 10 := by
  sorry

end probability_AB_selected_l478_478511


namespace prob_select_A_and_B_l478_478615

theorem prob_select_A_and_B : 
  let total_students := 5
  let to_select := 3
  let favorable_outcomes := Nat.div_fact (Nat.factorial 3) (Nat.factorial 1)
  let total_outcomes := Nat.div_fact (Nat.factorial 5) (Nat.factorial 3 * Nat.factorial (5 - 3))
  (favorable_outcomes : ℚ) / total_outcomes = 3 / 10 :=
by
  sorry

end prob_select_A_and_B_l478_478615


namespace problem_statement_l478_478821

noncomputable def nonreal_omega_root (ω : ℂ) : Prop :=
  ω^3 = 1 ∧ ω^2 + ω + 1 = 0

theorem problem_statement (ω : ℂ) (h : nonreal_omega_root ω) :
  (1 - 2 * ω + ω^2)^6 + (1 + 2 * ω - ω^2)^6 = 1458 :=
sorry

end problem_statement_l478_478821


namespace tommy_saw_13_cars_l478_478036

-- Defining the constants and variables based on given conditions
def trucks : ℕ := 12
def vehicles := ∀ x : ℕ, x = 4
def total_wheels : ℕ := 100

-- Theorem stating Tommy saw 13 cars
theorem tommy_saw_13_cars (trucks_have_4_wheels : vehicles trucks)
  (total_wheels_condition : total_wheels = 100) : 
  let truck_wheels := trucks * 4 in
  let remaining_wheels := total_wheels - truck_wheels in
  let cars := remaining_wheels / 4 in
  cars = 13 := 
by
  -- Skipping the proof details
  sorry

end tommy_saw_13_cars_l478_478036


namespace least_positive_n_l478_478814

-- Define the game and its conditions
def game_box (n : ℕ) : ℕ := n -- Function to represent the number of coins in box n

-- Prove that the least positive integer n such that the game can be played indefinitely is 2^k + k - 1
theorem least_positive_n {k : ℕ} (hk : k > 0) : ∃ n, n = 2^k + k - 1 ∧ n ≥ k + 1 ∧ (∀ (boxes : fin n → ℕ), (∀ i, boxes i = i) → 
  (∃ (continue_game_indefinitely : Prop), continue_game_indefinitely)) :=
sorry

end least_positive_n_l478_478814


namespace probability_of_selecting_A_and_B_l478_478669

open_locale classical

noncomputable def select_probability (n k : ℕ) : ℚ :=
  (nat.choose n k)⁻¹

theorem probability_of_selecting_A_and_B :
  let total_students := 5,
      students_to_select := 3,
      remaining_students := 3 in
  select_probability total_students students_to_select * nat.choose remaining_students 1 = 3 / 10 :=
by
  sorry

end probability_of_selecting_A_and_B_l478_478669


namespace new_area_is_1_12_original_area_l478_478958

variable (L W : ℝ)

def A_original : ℝ := L * W
def L_new : ℝ := 1.60 * L
def W_new : ℝ := 0.70 * W
def A_new : ℝ := L_new * W_new

theorem new_area_is_1_12_original_area : A_new = 1.12 * A_original := by
  sorry

end new_area_is_1_12_original_area_l478_478958


namespace probability_A_and_B_selected_l478_478585

/-- Let there be a group of 5 students including A and B. 
    If 3 students are selected randomly from these 5, the probability of selecting both A and B is 3/10. -/
theorem probability_A_and_B_selected : 
  let students := 5 in
  let selected := 3 in
  let combinations := students.choose selected in
  let favorable_combinations := (students - 2).choose (selected - 2) in
  let probability := (favorable_combinations : ℚ) / (combinations : ℚ) in
  probability = 3 / 10 := 
by sorry

end probability_A_and_B_selected_l478_478585


namespace AM_GM_inequality_l478_478742

theorem AM_GM_inequality (n : ℕ) (x : Fin n → ℝ) (h : ∀ i, 0 < x i) :
  (∑ i, x i ^ 2 / x ((i + 1) % n)) ≥ (∑ i, x i) :=
sorry

end AM_GM_inequality_l478_478742


namespace probability_A_and_B_selected_l478_478576

/-- Let there be a group of 5 students including A and B. 
    If 3 students are selected randomly from these 5, the probability of selecting both A and B is 3/10. -/
theorem probability_A_and_B_selected : 
  let students := 5 in
  let selected := 3 in
  let combinations := students.choose selected in
  let favorable_combinations := (students - 2).choose (selected - 2) in
  let probability := (favorable_combinations : ℚ) / (combinations : ℚ) in
  probability = 3 / 10 := 
by sorry

end probability_A_and_B_selected_l478_478576


namespace multiplication_problem_l478_478799

-- Definitions for different digits A, B, C, D
def is_digit (n : ℕ) := n < 10

theorem multiplication_problem 
  (A B C D : ℕ) 
  (hA : is_digit A) 
  (hB : is_digit B) 
  (hC : is_digit C) 
  (hD : is_digit D) 
  (h_diff : ∀ x y : ℕ, x ≠ y → is_digit x → is_digit y → x ≠ A → y ≠ B → x ≠ C → y ≠ D)
  (hD1 : D = 1)
  (h_mult : A * D = A) 
  (hC_eq : C = A + B) :
  A + C = 5 := sorry

end multiplication_problem_l478_478799


namespace probability_of_selecting_A_and_B_l478_478280

open_locale classical

-- Define the number of students
def num_students := 5

-- Define the number of students to be selected
def students_to_select := 3

def combination (n k : ℕ) : ℕ := nat.choose n k

-- Define the correct answer
def correct_probability := 3 / 10

-- Define the probability calculation
def calc_probability := (combination 3 1) / (combination 5 3)

-- Statement of the theorem to be proved
theorem probability_of_selecting_A_and_B :
  calc_probability = correct_probability :=
by sorry

end probability_of_selecting_A_and_B_l478_478280


namespace factor_quadratic_l478_478168

theorem factor_quadratic : ∀ (x : ℝ), 16*x^2 - 40*x + 25 = (4*x - 5)^2 := by
  intro x
  sorry

end factor_quadratic_l478_478168


namespace smallest_fourth_number_l478_478963

theorem smallest_fourth_number :
  ∃ (a b : ℕ), 145 + 10 * a + b = 4 * (28 + a + b) ∧ 10 * a + b = 35 :=
by
  sorry

end smallest_fourth_number_l478_478963


namespace right_angled_triangle_hypotenuse_and_altitude_relation_l478_478137

variables (a b c m : ℝ)

theorem right_angled_triangle_hypotenuse_and_altitude_relation
  (h1 : b^2 + c^2 = a^2)
  (h2 : m^2 = (b - c)^2)
  (h3 : b * c = a * m) :
  m = (a * (Real.sqrt 5 - 1)) / 2 := 
sorry

end right_angled_triangle_hypotenuse_and_altitude_relation_l478_478137


namespace count_perfect_cubes_l478_478770

theorem count_perfect_cubes (a b : ℕ) (h1 : a = 200) (h2 : b = 1200) :
  ∃ n, n = 5 ∧ ∀ x, (x^3 > a) ∧ (x^3 < b) → (x = 6 ∨ x = 7 ∨ x = 8 ∨ x = 9 ∨ x = 10) := 
sorry

end count_perfect_cubes_l478_478770


namespace six_hundred_billion_in_scientific_notation_l478_478106

theorem six_hundred_billion_in_scientific_notation (billion : ℕ) (h_billion : billion = 10^9) : 
  600 * billion = 6 * 10^11 :=
by
  rw [h_billion]
  sorry

end six_hundred_billion_in_scientific_notation_l478_478106


namespace probability_A_and_B_selected_l478_478247

theorem probability_A_and_B_selected (students : Finset ℕ) (A B : ℕ) (h : A ∈ students ∧ B ∈ students ∧ students.card = 5) : 
  (students.choose 3).count (λ s, {A, B} ⊆ s) / (students.choose 3).card = 3 / 10 := by
  sorry

end probability_A_and_B_selected_l478_478247


namespace probability_both_A_and_B_selected_l478_478410

-- The universe of students we consider
inductive Student
| A : Student
| B : Student
| C : Student
| D : Student
| E : Student

open Student

-- Function to count combinations
def combinations (n k : ℕ) : ℕ := (Nat.factorial n) / ((Nat.factorial k) * (Nat.factorial (n - k)))

-- The set of all students
def students : List Student := [A, B, C, D, E]

-- The total number of ways to select 3 students from 5
def total_selections := combinations 5 3

-- The number of ways A and B are both selected by selecting 1 more from the remaining 3 students
def selections_AB := combinations 3 1

-- The probability that both A and B are selected
def probability_AB := (selections_AB : ℚ) / (total_selections : ℚ)

-- The proof statement
theorem probability_both_A_and_B_selected :
  probability_AB = 3 / 10 :=
sorry

end probability_both_A_and_B_selected_l478_478410


namespace compare_logs_and_exponent_l478_478717

theorem compare_logs_and_exponent (a b c : ℝ) (h1 : a = Real.log 6 / Real.log 2) (h2 : b = Real.log 12 / Real.log 3) (h3 : c = 2 ^ 0.6) :
  c < b ∧ b < a :=
by
  sorry

end compare_logs_and_exponent_l478_478717


namespace prob_select_A_and_B_l478_478619

theorem prob_select_A_and_B : 
  let total_students := 5
  let to_select := 3
  let favorable_outcomes := Nat.div_fact (Nat.factorial 3) (Nat.factorial 1)
  let total_outcomes := Nat.div_fact (Nat.factorial 5) (Nat.factorial 3 * Nat.factorial (5 - 3))
  (favorable_outcomes : ℚ) / total_outcomes = 3 / 10 :=
by
  sorry

end prob_select_A_and_B_l478_478619


namespace probability_both_A_and_B_selected_l478_478414

-- The universe of students we consider
inductive Student
| A : Student
| B : Student
| C : Student
| D : Student
| E : Student

open Student

-- Function to count combinations
def combinations (n k : ℕ) : ℕ := (Nat.factorial n) / ((Nat.factorial k) * (Nat.factorial (n - k)))

-- The set of all students
def students : List Student := [A, B, C, D, E]

-- The total number of ways to select 3 students from 5
def total_selections := combinations 5 3

-- The number of ways A and B are both selected by selecting 1 more from the remaining 3 students
def selections_AB := combinations 3 1

-- The probability that both A and B are selected
def probability_AB := (selections_AB : ℚ) / (total_selections : ℚ)

-- The proof statement
theorem probability_both_A_and_B_selected :
  probability_AB = 3 / 10 :=
sorry

end probability_both_A_and_B_selected_l478_478414


namespace factorization_identity_l478_478174

theorem factorization_identity (x : ℝ) : 
  16 * x^2 - 40 * x + 25 = (4 * x - 5)^2 :=
  sorry

end factorization_identity_l478_478174


namespace rectangle_perimeter_l478_478972

-- The conditions are directly brought in as assumptions and definitions.
def triangleDEF_right : Prop := (7^2 + 24^2 = 25^2)

def area_triangleDEF := (1 / 2 : ℝ) * 7 * 24

def width_rectangle := 7

def length_rectangle := area_triangleDEF / width_rectangle

def perimeter_rectangle := 2 * (length_rectangle + width_rectangle)

theorem rectangle_perimeter : triangleDEF_right → area_triangleDEF = 84 → perimeter_rectangle = 38 :=
by
  intros
  rw [triangleDEF_right] at a
  rw [area_triangleDEF] at b
  rw [length_rectangle, perimeter_rectangle]
  sorry

end rectangle_perimeter_l478_478972


namespace sum_of_reciprocals_eq_reciprocal_l478_478961

-- Conditions: Definitions of x1, x2, x3
variables (x1 x2 x3 : ℝ)

-- Hypotheses: The line intersects the graph of y = x² at points with abscissas x1, x2, and intersects the x-axis at abscissa x3.
hypothesis (h1 : ∃ k: ℝ, ∀ x, (x = x1 ∨ x = x2) → (x^2 = k * (x - x3)))

-- To Prove: 1/x1 + 1/x2 = 1/x3
theorem sum_of_reciprocals_eq_reciprocal (h_vieta1 : x1 + x2 = x1 * x2 * x3) : 
  1 / x1 + 1 / x2 = 1 / x3 :=
sorry

end sum_of_reciprocals_eq_reciprocal_l478_478961


namespace probability_A_and_B_selected_l478_478344

theorem probability_A_and_B_selected (A B : Type) (S : Finset (Type)) (h : S.card = 5) (hA : A ∈ S) (hB : B ∈ S) (T : Finset (Type)) (hT : T.card = 3) :
  (∃ T ⊆ S, A ∈ T ∧ B ∈ T ∧ T.card = 3) → 
  ∃ T ⊆ S, T.card = 3 ∧ (finset.card (T ∩ {A, B}) = 2) → 
  (finset.card (T ∩ {A, B}) = 2) = (3 / 10 : Real) := sorry

end probability_A_and_B_selected_l478_478344


namespace sqrt_500_eq_10_sqrt_5_l478_478868

theorem sqrt_500_eq_10_sqrt_5 : sqrt 500 = 10 * sqrt 5 :=
  sorry

end sqrt_500_eq_10_sqrt_5_l478_478868


namespace exists_points_E_F_l478_478021

variables {A B C D E F : Point} [triangle ABC]

def seg_midpoint (P Q : Point) : Point :=
(midpoint P Q)

theorem exists_points_E_F
  (hD_on_AB : D ∈ segment A B)
  (hE_on_CA : E ∈ segment C A)
  (hF_on_AB : F ∈ segment A B)
  (h1 : collinear B (seg_midpoint D E) (seg_midpoint D F))
  (h2 : collinear C (seg_midpoint D E) (seg_midpoint E F)) :
  ∃ (E F : Point),
    (E ∈ segment C A) ∧
    (F ∈ segment A B) ∧
    collinear B (seg_midpoint D E) (seg_midpoint D F) ∧
    collinear C (seg_midpoint D E) (seg_midpoint E F) :=
sorry

end exists_points_E_F_l478_478021


namespace prob_select_A_and_B_l478_478657

theorem prob_select_A_and_B (students : Finset ℕ) (A B : ℕ) (hA : A ∈ students) (hB : B ∈ students) (h_card : students.card = 5) :
  (students.choose 3).count {comb | A ∈ comb ∧ B ∈ comb} / (students.choose 3).card = 3 / 10 := sorry

end prob_select_A_and_B_l478_478657


namespace probability_of_selecting_A_and_B_l478_478564

theorem probability_of_selecting_A_and_B (students : Finset ℕ) (A B : ℕ)
  (h_size : students.card = 5) (h_in : A ∈ students ∧ B ∈ students) : 
  (Finset.card {s : Finset ℕ // s ⊆ students ∧ s.card = 3 ∧ A ∈ s ∧ B ∈ s}.to_set) / 
  (Finset.card {s : Finset ℕ // s ⊆ students ∧ s.card = 3}.to_set) = 3 / 10 :=
sorry

end probability_of_selecting_A_and_B_l478_478564


namespace probability_of_selecting_A_and_B_l478_478546

theorem probability_of_selecting_A_and_B (students : Finset ℕ) (A B : ℕ)
  (h_size : students.card = 5) (h_in : A ∈ students ∧ B ∈ students) : 
  (Finset.card {s : Finset ℕ // s ⊆ students ∧ s.card = 3 ∧ A ∈ s ∧ B ∈ s}.to_set) / 
  (Finset.card {s : Finset ℕ // s ⊆ students ∧ s.card = 3}.to_set) = 3 / 10 :=
sorry

end probability_of_selecting_A_and_B_l478_478546


namespace probability_A_B_selected_l478_478349

noncomputable def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

def combination (n r : ℕ) : ℕ :=
  factorial n / (factorial r * factorial (n - r))

theorem probability_A_B_selected :
  let total_ways := combination 5 3 in
  let ways_A_B := combination 3 1 in
  (ways_A_B : ℚ) / total_ways = 3 / 10 :=
by
  sorry

end probability_A_B_selected_l478_478349


namespace total_students_approx_661_l478_478942

variable (N A B S T : ℕ)
variable (percent_T : ℝ)

-- Given conditions
def condition_1 := N = 240
def condition_2 := A = 423
def condition_3 := B = 134
def condition_4 := percent_T = 0.80
def condition_5 := S = N + A - B
def condition_6 := S = percent_T * T

-- Prove T is approximately 661
theorem total_students_approx_661 (h1 : condition_1) (h2 : condition_2) (h3 : condition_3) 
    (h4 : condition_4) (h5 : condition_5) (h6 : condition_6) : 
    T ≈ 661 := by
  sorry

end total_students_approx_661_l478_478942


namespace probability_of_selecting_A_and_B_l478_478552

theorem probability_of_selecting_A_and_B (students : Finset ℕ) (A B : ℕ)
  (h_size : students.card = 5) (h_in : A ∈ students ∧ B ∈ students) : 
  (Finset.card {s : Finset ℕ // s ⊆ students ∧ s.card = 3 ∧ A ∈ s ∧ B ∈ s}.to_set) / 
  (Finset.card {s : Finset ℕ // s ⊆ students ∧ s.card = 3}.to_set) = 3 / 10 :=
sorry

end probability_of_selecting_A_and_B_l478_478552


namespace cos_sum_sin_double_sum_cos_double_sum_l478_478042

noncomputable def triangle := Type

variables (A B C I K H O : triangle)
variables (r R IK OH S : ℝ)

-- Some preconditions akin to geometric properties and relevant triangle identities
axiom cos_sum_identity (A B C : triangle) (r : ℝ) (IK : ℝ) : 
  cos A + cos B + cos C = 3/2 - IK^2 / (2 * r^2)

axiom sin_double_sum_identity (A B C : triangle) (R : ℝ) (S : ℝ) :
  sin (2 * A) + sin (2 * B) + sin (2 * C) = 2 * S / (R^2)

axiom cos_double_sum_identity (A B C : triangle) (R : ℝ) (OH : ℝ) :
  cos (2 * A) + cos (2 * B) + cos (2 * C) = OH^2 / (2 * R^2) - 3 / 2

-- Proof problem statements
theorem cos_sum (A B C I : triangle) (r : ℝ) (IK : ℝ) :
  cos A + cos B + cos C = 3/2 - IK^2 / (2 * r^2) :=
by
  apply cos_sum_identity
  -- provide necessary steps if needed
  sorry

theorem sin_double_sum (A B C O : triangle) (R : ℝ) (S : ℝ) :
  sin (2 * A) + sin (2 * B) + sin (2 * C) = 2 * S / (R^2) :=
by
  apply sin_double_sum_identity
  -- provide necessary steps if needed
  sorry

theorem cos_double_sum (A B C O H : triangle) (R : ℝ) (OH : ℝ) :
  cos (2 * A) + cos (2 * B) + cos (2 * C) = OH^2 / (2 * R^2) - 3 / 2 :=
by
  apply cos_double_sum_identity
  -- provide necessary steps if needed
  sorry

end cos_sum_sin_double_sum_cos_double_sum_l478_478042


namespace probability_A_and_B_selected_l478_478581

/-- Let there be a group of 5 students including A and B. 
    If 3 students are selected randomly from these 5, the probability of selecting both A and B is 3/10. -/
theorem probability_A_and_B_selected : 
  let students := 5 in
  let selected := 3 in
  let combinations := students.choose selected in
  let favorable_combinations := (students - 2).choose (selected - 2) in
  let probability := (favorable_combinations : ℚ) / (combinations : ℚ) in
  probability = 3 / 10 := 
by sorry

end probability_A_and_B_selected_l478_478581


namespace probability_of_selecting_A_and_B_l478_478265

-- Define the combination function
def combination (n r : ℕ) : ℕ := Nat.factorial n / (Nat.factorial r * Nat.factorial (n - r))

-- Define the conditions
def students : ℕ := 5
def to_select : ℕ := 3

-- Define the favorable outcomes where A and B are selected
def favorable_outcomes : ℕ := combination 3 1

-- Define the total outcomes of selecting 3 out of 5 students
def total_outcomes : ℕ := combination 5 3

-- Define the probability calculation
def probability_of_A_and_B : ℚ := favorable_outcomes / total_outcomes

-- Proposition to prove
theorem probability_of_selecting_A_and_B : probability_of_A_and_B = 3 / 10 :=
by
  sorry

end probability_of_selecting_A_and_B_l478_478265


namespace probability_A_and_B_selected_l478_478238

theorem probability_A_and_B_selected (students : Finset ℕ) (A B : ℕ) (h : A ∈ students ∧ B ∈ students ∧ students.card = 5) : 
  (students.choose 3).count (λ s, {A, B} ⊆ s) / (students.choose 3).card = 3 / 10 := by
  sorry

end probability_A_and_B_selected_l478_478238


namespace probability_A_and_B_selected_l478_478381

theorem probability_A_and_B_selected 
  (students : Finset ℕ) (A B : ℕ) 
  (h_students_size : students.card = 5) 
  (h_A_in_students : A ∈ students) 
  (h_B_in_students : B ∈ students)
  : (↑(Nat.choose 3) : ℚ) / (↑(Nat.choose 5) : ℚ) = 3 / 10 := 
by sorry

end probability_A_and_B_selected_l478_478381


namespace probability_A_and_B_selected_l478_478308

open Finset

noncomputable def total_ways : ℕ := (choose 5 3)

noncomputable def favorable_ways : ℕ := (choose 3 1)

noncomputable def probability (total favorable : ℕ) : ℚ := favorable / total

theorem probability_A_and_B_selected :
  probability total_ways favorable_ways = 3 / 10 :=
by
  sorry

end probability_A_and_B_selected_l478_478308


namespace factorization_identity_l478_478175

theorem factorization_identity (x : ℝ) : 
  16 * x^2 - 40 * x + 25 = (4 * x - 5)^2 :=
  sorry

end factorization_identity_l478_478175


namespace probability_of_selecting_A_and_B_l478_478562

theorem probability_of_selecting_A_and_B (students : Finset ℕ) (A B : ℕ)
  (h_size : students.card = 5) (h_in : A ∈ students ∧ B ∈ students) : 
  (Finset.card {s : Finset ℕ // s ⊆ students ∧ s.card = 3 ∧ A ∈ s ∧ B ∈ s}.to_set) / 
  (Finset.card {s : Finset ℕ // s ⊆ students ∧ s.card = 3}.to_set) = 3 / 10 :=
sorry

end probability_of_selecting_A_and_B_l478_478562


namespace problem_statement_l478_478996

-- Defining the set K
def K := {x : ℝ | (0 ≤ x ∧ x ≤ 1) ∧ ∃ (d : ℕ → ℕ), (∀ n, d n ∈ {0, 2}) ∧ x = ∑' (n : ℕ), (d n) * 3⁻¹^(n + 1)}

-- Defining the set S
def S := {z : ℝ | ∃ x y ∈ K, z = x + y}

-- The theorem to be proved
theorem problem_statement : S = {z : ℝ | 0 ≤ z ∧ z ≤ 2} := 
sorry

end problem_statement_l478_478996


namespace total_configurations_l478_478080

-- Conditions
def num_fiction_books : ℕ := 3
def num_non_fiction_books : ℕ := 3

-- Factorials
def factorial (n : ℕ) : ℕ := if n = 0 then 1 else n * factorial (n - 1)

-- Prove the total configurations
theorem total_configurations : factorial num_fiction_books * factorial num_non_fiction_books = 36 :=
by
  -- Using the supplied factorial function and conditions, we can verify the result.
  have fiction_factorial : factorial 3 = 6 := rfl
  have non_fiction_factorial : factorial 3 = 6 := rfl
  rw [fiction_factorial, non_fiction_factorial]
  exact rfl

end total_configurations_l478_478080


namespace prob_select_A_and_B_l478_478639

theorem prob_select_A_and_B (students : Finset ℕ) (A B : ℕ) (hA : A ∈ students) (hB : B ∈ students) (h_card : students.card = 5) :
  (students.choose 3).count {comb | A ∈ comb ∧ B ∈ comb} / (students.choose 3).card = 3 / 10 := sorry

end prob_select_A_and_B_l478_478639


namespace number_of_pens_l478_478011

theorem number_of_pens (C S : ℝ) (n : ℝ) (h1 : n * C = 5 * S) (h2 : S = 1.5 * C) : n = 8 :=
by {
  -- Assume h1 and h2 as conditions
  -- From h2, we have S = 1.5 * C
  have h3 : 5 * S = 5 * (1.5 * C), from congr_arg (λ x, 5 * x) h2,
  -- Simplify
  rw [mul_assoc, show 5 * 1.5 = 7.5, by norm_num] at h3,
  -- n * C = 7.5 * C
  rw h3 at h1,
  -- Simplify
  have h4 : n * C = 7.5 * C, from h1,
  -- Divide both sides by C
  have h5 : n = 7.5, from eq_of_mul_eq_mul_left (by linarith) h4,
  -- Nearest whole number to 7.5 is 8
  have h6 : 8 = 8, from rfl,
  sorry
}

end number_of_pens_l478_478011


namespace initial_fruits_count_l478_478974

variables (oranges limes fruit_initial fruit_remaining : ℕ)

-- Conditions
def martin_has_50_oranges : oranges = 50 :=
rfl

def oranges_twice_limes : oranges = 2 * limes :=
by sorry

def martin_ate_half : fruit_initial = 2 * fruit_remaining :=
by sorry

-- Problem statement
theorem initial_fruits_count :
  oranges = 50 →
  oranges = 2 * limes →
  fruit_remaining = oranges + limes →
  fruit_initial = 2 * fruit_remaining →
  fruit_initial = 150 :=
by sorry

end initial_fruits_count_l478_478974


namespace probability_A_and_B_selected_l478_478536

theorem probability_A_and_B_selected :
  let n := 5
  let r := 3
  let total_ways := Nat.choose n r
  let favorable_ways := Nat.choose 3 1
  let probability := favorable_ways / total_ways
  probability = (3 / 10) := 
by
  sorry

end probability_A_and_B_selected_l478_478536


namespace repeating_decimal_eq_fraction_l478_478163

noncomputable def repeating_decimal_to_fraction (x : ℝ) : ℝ :=
  let x : ℝ := 4.5656565656 -- * 0.5656... repeating
  (100*x - x) / (100 - 1)

-- Define the theorem we want to prove
theorem repeating_decimal_eq_fraction : 
  ∀ x : ℝ, x = 4.565656 -> x = (452 : ℝ) / (99 : ℝ) :=
by
  intro x h
  -- here we would provide the proof steps, but since it's omitted
  -- we'll use sorry to skip it.
  sorry

end repeating_decimal_eq_fraction_l478_478163


namespace probability_of_selecting_A_and_B_is_three_tenths_l478_478590

-- Define the set of students
def students : Finset ℕ := {0, 1, 2, 3, 4}

-- Define subsets A and B
def student_A : ℕ := 0
def student_B : ℕ := 1

-- Total number of ways to select 3 students out of 5
def total_ways : ℕ := students.card.choose 3

-- Number of ways to select 3 students including A and B
def favorable_ways : ℕ := {2, 3, 4}.card.choose 1

-- The probability
def probability_of_selecting_A_and_B : ℚ := (favorable_ways : ℚ) / (total_ways : ℚ)

theorem probability_of_selecting_A_and_B_is_three_tenths :
  probability_of_selecting_A_and_B = 3 / 10 :=
by
  sorry

end probability_of_selecting_A_and_B_is_three_tenths_l478_478590


namespace probability_A_B_l478_478459

open Nat

-- Define a function for combination
def combination (n r : Nat) : Nat :=
  factorial n / (factorial r * factorial (n - r))

-- Define the set of students and the selection conditions
def num_students : Nat := 5
def num_selected : Nat := 3

-- Define A and B as selected students
def selected_A_B : Nat := combination 3 1

-- Define the probability calculation
def probability_A_B_selected : Rat :=
  (selected_A_B : Rat) / (combination num_students num_selected : Rat)

-- The statement of the problem
theorem probability_A_B : probability_A_B_selected = 3 / 10 := by
  sorry

end probability_A_B_l478_478459


namespace probability_of_selecting_A_and_B_is_three_tenths_l478_478594

-- Define the set of students
def students : Finset ℕ := {0, 1, 2, 3, 4}

-- Define subsets A and B
def student_A : ℕ := 0
def student_B : ℕ := 1

-- Total number of ways to select 3 students out of 5
def total_ways : ℕ := students.card.choose 3

-- Number of ways to select 3 students including A and B
def favorable_ways : ℕ := {2, 3, 4}.card.choose 1

-- The probability
def probability_of_selecting_A_and_B : ℚ := (favorable_ways : ℚ) / (total_ways : ℚ)

theorem probability_of_selecting_A_and_B_is_three_tenths :
  probability_of_selecting_A_and_B = 3 / 10 :=
by
  sorry

end probability_of_selecting_A_and_B_is_three_tenths_l478_478594


namespace probability_of_selecting_A_and_B_is_three_tenths_l478_478595

-- Define the set of students
def students : Finset ℕ := {0, 1, 2, 3, 4}

-- Define subsets A and B
def student_A : ℕ := 0
def student_B : ℕ := 1

-- Total number of ways to select 3 students out of 5
def total_ways : ℕ := students.card.choose 3

-- Number of ways to select 3 students including A and B
def favorable_ways : ℕ := {2, 3, 4}.card.choose 1

-- The probability
def probability_of_selecting_A_and_B : ℚ := (favorable_ways : ℚ) / (total_ways : ℚ)

theorem probability_of_selecting_A_and_B_is_three_tenths :
  probability_of_selecting_A_and_B = 3 / 10 :=
by
  sorry

end probability_of_selecting_A_and_B_is_three_tenths_l478_478595


namespace probability_A_and_B_selected_l478_478698

/-- There are 5 students including A and B. The probability of selecting both A and B 
    when 3 students are selected randomly from 5 is 3/10. -/
theorem probability_A_and_B_selected :
  (∃ (students : List String) (h : students.length = 5),
    ∃ (selected : List String) (h' : selected.length = 3),
    "A" ∈ selected ∧ "B" ∈ selected) →
  ∃ (P : ℚ), P = 3 / 10 :=
by
  sorry

end probability_A_and_B_selected_l478_478698


namespace cos_C_in_triangle_l478_478803

theorem cos_C_in_triangle (a b c : ℝ) (A B C : ℝ) [triangle ABC] 
  (h_b : b = 2 * a) 
  (h_sin : b * sin A = c * sin C) : 
  cos C = 3 / 4 := 
sorry

end cos_C_in_triangle_l478_478803


namespace curve_C1_cartesian_eq_and_PM_PN_range_l478_478795

theorem curve_C1_cartesian_eq_and_PM_PN_range :
  (∀ θ ρ, ρ * (Real.cos (θ - π / 3)) = 1 ↔ ∀ (x y : ℝ), x + (Real.sqrt 3) * y - 2 = 0) ∧
  (∀ α : ℝ, 
    let P := (2, 0) : (ℝ × ℝ),
    let M := (Real.cos α, -2 + Real.sin α),
    let N := (Real.cos (α + π / 2), -2 + Real.sin (α + π / 2))
    in ∃ (s e : ℝ), 
        s = 10 ∧ e = 26 ∧ 
        (s <= (Real.norm (P.1 - M.1) ^ 2 + Real.norm (P.2 - M.2) ^ 2 + Real.norm (P.1 - N.1) ^ 2 + Real.norm (P.2 - N.2) ^ 2) ∧ 
        (Real.norm (P.1 - M.1) ^ 2 + Real.norm (P.2 - M.2) ^ 2 + Real.norm (P.1 - N.1) ^ 2 + Real.norm (P.2 - N.2) ^ 2) <= e)) :=
begin
  sorry
end

end curve_C1_cartesian_eq_and_PM_PN_range_l478_478795


namespace probability_A_and_B_selected_l478_478705

/-- There are 5 students including A and B. The probability of selecting both A and B 
    when 3 students are selected randomly from 5 is 3/10. -/
theorem probability_A_and_B_selected :
  (∃ (students : List String) (h : students.length = 5),
    ∃ (selected : List String) (h' : selected.length = 3),
    "A" ∈ selected ∧ "B" ∈ selected) →
  ∃ (P : ℚ), P = 3 / 10 :=
by
  sorry

end probability_A_and_B_selected_l478_478705


namespace pyramid_height_l478_478887

variable (a b l : ℝ)

theorem pyramid_height (h₁ : a > 0) (h₂ : b > 0) (h₃ : l > 0) :
  ∃ (SO : ℝ), SO = (1 / 2) * real.sqrt (4 * l^2 - a^2 - b^2) :=
by
  use (1 / 2) * real.sqrt (4 * l^2 - a^2 - b^2)
  sorry

end pyramid_height_l478_478887


namespace extreme_value_at_neg3_nth_equation_l478_478076

-- Problem 1
theorem extreme_value_at_neg3 (a : ℝ) :
  (∃ x : ℝ, x = -3 ∧ f'(x) = 0) ↔ a = 5 :=
by
-- Definition of the function f and its derivative f'
let f := λ x : ℝ, x^3 + a * x^2 + 3 * x - 9
let f' := λ x : ℝ, 3 * x^2 + 2 * a * x + 3

sorry

-- Problem 2
theorem nth_equation (n : ℕ) :
  (∑ k in finset.range (n + 1), (-1 : ℤ)^(k+1) * k^2 : ℤ) = 
  (-1)^(n+1) * ∑ k in finset.range (n + 1), k :=
by
-- The pattern deduced in the problem
sorry

end extreme_value_at_neg3_nth_equation_l478_478076


namespace probability_A_B_selected_l478_478365

noncomputable def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

def combination (n r : ℕ) : ℕ :=
  factorial n / (factorial r * factorial (n - r))

theorem probability_A_B_selected :
  let total_ways := combination 5 3 in
  let ways_A_B := combination 3 1 in
  (ways_A_B : ℚ) / total_ways = 3 / 10 :=
by
  sorry

end probability_A_B_selected_l478_478365


namespace probability_A_and_B_selected_l478_478386

theorem probability_A_and_B_selected 
  (students : Finset ℕ) (A B : ℕ) 
  (h_students_size : students.card = 5) 
  (h_A_in_students : A ∈ students) 
  (h_B_in_students : B ∈ students)
  : (↑(Nat.choose 3) : ℚ) / (↑(Nat.choose 5) : ℚ) = 3 / 10 := 
by sorry

end probability_A_and_B_selected_l478_478386


namespace probability_A_and_B_selected_l478_478240

theorem probability_A_and_B_selected (students : Finset ℕ) (A B : ℕ) (h : A ∈ students ∧ B ∈ students ∧ students.card = 5) : 
  (students.choose 3).count (λ s, {A, B} ⊆ s) / (students.choose 3).card = 3 / 10 := by
  sorry

end probability_A_and_B_selected_l478_478240


namespace probability_A_and_B_selected_l478_478347

theorem probability_A_and_B_selected (A B : Type) (S : Finset (Type)) (h : S.card = 5) (hA : A ∈ S) (hB : B ∈ S) (T : Finset (Type)) (hT : T.card = 3) :
  (∃ T ⊆ S, A ∈ T ∧ B ∈ T ∧ T.card = 3) → 
  ∃ T ⊆ S, T.card = 3 ∧ (finset.card (T ∩ {A, B}) = 2) → 
  (finset.card (T ∩ {A, B}) = 2) = (3 / 10 : Real) := sorry

end probability_A_and_B_selected_l478_478347


namespace combined_area_correct_l478_478098

noncomputable def combined_area_of_rectangle_and_equilateral_triangle : ℝ :=
  let side_of_square := real.sqrt 16
  let radius_of_circle := side_of_square
  let length_of_rectangle := 5 * radius_of_circle
  let breadth_of_rectangle := 11
  let area_of_rectangle := length_of_rectangle * breadth_of_rectangle
  let height_of_equilateral_triangle := breadth_of_rectangle
  let side_of_equilateral_triangle := (22 : ℝ) / real.sqrt 3
  let area_of_equilateral_triangle := (real.sqrt 3 / 4) * side_of_equilateral_triangle ^ 2
  area_of_rectangle + area_of_equilateral_triangle

theorem combined_area_correct : combined_area_of_rectangle_and_equilateral_triangle = 289.282 :=
by
  -- Proof steps are omitted, so we use sorry to skip the proof
  sorry

end combined_area_correct_l478_478098


namespace select_3_from_5_prob_l478_478441

theorem select_3_from_5_prob {students : Finset ℕ} 
  (h : students.card = 5) 
  {A B : ℕ} 
  (hA : A ∈ students) 
  (hB : B ∈ students) 
  (H : 3 ≤ students.card) : 
  (finset.card (students.filter (λ x, x ≠ A ∧ x ≠ B)) = 3) → 
  (finset.card ((students.filter (λ x, x ≠ A ∧ x ≠ B)).subtype (λ x, True)) = 3) → 
  ((finset.card (finset.filter (λ (students : Finset ℕ) A ∈ students ∧ B ∈ students ↔ 3 = students.card) 5)) = ↑10) → 
  ∃ students', (students'.card = 3 ∧ A ∈ students' ∧ B ∈ students') → 
  ∃ (p : ℚ), p = 3 / 10 ∧ students.filter (λ x, x = A ∨ x = B) = 2 :=
  sorry

end select_3_from_5_prob_l478_478441


namespace find_constant_a_l478_478818

variable (A B C D : ℝ)
variable (p q r : ℝ)

-- Conditions based on Vieta's formulas for cubic equation
def sum_roots_cubic : Prop := p + q + r = -B / A
def sum_product_roots_cubic : Prop := p*q + p*r + q*r = C / A
def product_roots_cubic : Prop := p * q * r = -D / A

-- Main theorem to prove
theorem find_constant_a (h1 : sum_roots_cubic A B p q r) 
                        (h2 : sum_product_roots_cubic A B C p q r) 
                        (h3 : product_roots_cubic A D p q r) : 
  let sum_new_roots := p^2 + q + q^2 + r + r^2 + p
  let a := -(sum_new_roots) 
  a = (AB + 2AC - B^2) / A^2 := 
begin
  sorry
end

end find_constant_a_l478_478818


namespace perpendicular_MD_ME_find_line_l_l478_478750

-- Definitions of curves and points
def curve1 (x y : ℝ) : Prop := (x^2/4) + y^2 = 1
def curve2 (x y : ℝ) : Prop := y = x^2 - 1
def pointM : (ℝ × ℝ) := (0, -1)

-- Definitions of intersections and slopes
def line_through_origin (k : ℝ) : ℝ → ℝ := λ x , k * x

def point_on_curve (f : ℝ → ℝ) (c : (ℝ → ℝ → Prop)) (x : ℝ) : Prop :=
  c x (f x)

def slope (p1 p2 : ℝ × ℝ) : ℝ := 
  if (p1.1 = p2.1) then 0 else (p2.2 - p1.2) / (p2.1 - p1.1)

-- Define the perpendicular proof statement
theorem perpendicular_MD_ME:
  ∀ (A B D E : ℝ × ℝ) (k : ℝ),
  point_on_curve (line_through_origin k) curve2 A.1 ->
  point_on_curve (line_through_origin k) curve2 B.1 ->
  curve1 D.1 D.2 ->
  curve1 E.1 E.2 ->
  slope (pointM, D) * slope (pointM, E) = -1 :=
sorry

-- Define the ratios and areas of triangles
def area_triangle (p1 p2 p3 : ℝ × ℝ) : ℝ :=
  0.5 * abs((p1.1 * (p2.2 - p3.2)) + (p2.1 * (p3.2 - p1.2)) + (p3.1 * (p1.2 - p2.2)))

theorem find_line_l (l_slope) :
  ∀ (A B D E : ℝ × ℝ) (k : ℝ),
  point_on_curve (line_through_origin k) curve2 A.1 ->
  point_on_curve (line_through_origin k) curve2 B.1 ->
  curve1 D.1 D.2 ->
  curve1 E.1 E.2 ->
  let S1 := area_triangle pointM A B in
  let S2 := area_triangle pointM D E in
  S1 / S2 = 17/32 ->
  (∃ l : ℝ, l = k ∧ curve2(0, 0)) :=
sorry

end perpendicular_MD_ME_find_line_l_l478_478750


namespace solve_for_m_l478_478149

noncomputable def equation := 62519 * 9999 ^ 2 / 314 * (314 - (m : ℤ)) = 547864

theorem solve_for_m (m : ℤ) (h : 62519 * 9999 ^ 2 / 314 * (314 - m) = 547864) : m = -547550 := sorry

end solve_for_m_l478_478149


namespace probability_A_and_B_selected_l478_478250

theorem probability_A_and_B_selected (students : Finset ℕ) (A B : ℕ) (h : A ∈ students ∧ B ∈ students ∧ students.card = 5) : 
  (students.choose 3).count (λ s, {A, B} ⊆ s) / (students.choose 3).card = 3 / 10 := by
  sorry

end probability_A_and_B_selected_l478_478250


namespace probability_AB_selected_l478_478494

def factorial (n : Nat) : Nat := if n = 0 then 1 else n * factorial (n - 1)

def choose (n k : Nat) : Nat := factorial n / (factorial k * factorial (n - k))

theorem probability_AB_selected (h1 : 5 = 5) (h2 : 3 = 3) : (choose 3 1) / (choose 5 3) = 3 / 10 := by
  sorry

end probability_AB_selected_l478_478494


namespace probability_A_B_l478_478467

open Nat

-- Define a function for combination
def combination (n r : Nat) : Nat :=
  factorial n / (factorial r * factorial (n - r))

-- Define the set of students and the selection conditions
def num_students : Nat := 5
def num_selected : Nat := 3

-- Define A and B as selected students
def selected_A_B : Nat := combination 3 1

-- Define the probability calculation
def probability_A_B_selected : Rat :=
  (selected_A_B : Rat) / (combination num_students num_selected : Rat)

-- The statement of the problem
theorem probability_A_B : probability_A_B_selected = 3 / 10 := by
  sorry

end probability_A_B_l478_478467


namespace prob_select_A_and_B_l478_478614

theorem prob_select_A_and_B : 
  let total_students := 5
  let to_select := 3
  let favorable_outcomes := Nat.div_fact (Nat.factorial 3) (Nat.factorial 1)
  let total_outcomes := Nat.div_fact (Nat.factorial 5) (Nat.factorial 3 * Nat.factorial (5 - 3))
  (favorable_outcomes : ℚ) / total_outcomes = 3 / 10 :=
by
  sorry

end prob_select_A_and_B_l478_478614


namespace probability_A_and_B_selected_l478_478251

theorem probability_A_and_B_selected (students : Finset ℕ) (A B : ℕ) (h : A ∈ students ∧ B ∈ students ∧ students.card = 5) : 
  (students.choose 3).count (λ s, {A, B} ⊆ s) / (students.choose 3).card = 3 / 10 := by
  sorry

end probability_A_and_B_selected_l478_478251


namespace probability_A_and_B_selected_l478_478521

theorem probability_A_and_B_selected :
  let n := 5
  let r := 3
  let total_ways := Nat.choose n r
  let favorable_ways := Nat.choose 3 1
  let probability := favorable_ways / total_ways
  probability = (3 / 10) := 
by
  sorry

end probability_A_and_B_selected_l478_478521


namespace probability_A_B_selected_l478_478369

noncomputable def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

def combination (n r : ℕ) : ℕ :=
  factorial n / (factorial r * factorial (n - r))

theorem probability_A_B_selected :
  let total_ways := combination 5 3 in
  let ways_A_B := combination 3 1 in
  (ways_A_B : ℚ) / total_ways = 3 / 10 :=
by
  sorry

end probability_A_B_selected_l478_478369


namespace lily_pads_half_lake_l478_478066

theorem lily_pads_half_lake (n : ℕ) (h : n = 39) :
  (n - 1) = 38 :=
by
  sorry

end lily_pads_half_lake_l478_478066


namespace sqrt_500_eq_10_sqrt_5_l478_478865

theorem sqrt_500_eq_10_sqrt_5 : sqrt 500 = 10 * sqrt 5 :=
  sorry

end sqrt_500_eq_10_sqrt_5_l478_478865


namespace slope_of_tangent_line_l478_478904

open Real

theorem slope_of_tangent_line (f : ℝ → ℝ) (x a : ℝ) (y : ℝ) (h : f = λ x, x^2 + 3*x) (ha : x = 2) (hy : y = 10) :
  deriv f x = 7 := 
by
  have h_deriv : deriv f x = 2*x + 3,
  { 
    rw h,
    exact deriv_add (deriv_pow' 2 x) (deriv_const_mul (3 : ℝ) x (deriv_id' x))
  },
  rw [h_deriv, ha],
  norm_num

end slope_of_tangent_line_l478_478904


namespace prob_select_A_and_B_l478_478650

theorem prob_select_A_and_B (students : Finset ℕ) (A B : ℕ) (hA : A ∈ students) (hB : B ∈ students) (h_card : students.card = 5) :
  (students.choose 3).count {comb | A ∈ comb ∧ B ∈ comb} / (students.choose 3).card = 3 / 10 := sorry

end prob_select_A_and_B_l478_478650


namespace probability_A_and_B_selected_l478_478695

/-- There are 5 students including A and B. The probability of selecting both A and B 
    when 3 students are selected randomly from 5 is 3/10. -/
theorem probability_A_and_B_selected :
  (∃ (students : List String) (h : students.length = 5),
    ∃ (selected : List String) (h' : selected.length = 3),
    "A" ∈ selected ∧ "B" ∈ selected) →
  ∃ (P : ℚ), P = 3 / 10 :=
by
  sorry

end probability_A_and_B_selected_l478_478695


namespace all_positive_integers_appear_l478_478018

-- The sequence definition
def sequence (n : ℕ) : ℕ
| 0 := 1
| (n + 1) := @well_founded.fix (ℕ) has_well_founded.wf (λ k, k) (λ m IH, 
    if H : ∃ k : ℕ, ¬((k + ∃ i : fin (n + 1), IH i).nat_sqrt = 0)
    then classical.some H
    else 1) n

-- Proving all positive integers appear in the sequence
theorem all_positive_integers_appear : ∀ m : ℕ, ∃ n : ℕ, sequence n = m :=
by sorry

end all_positive_integers_appear_l478_478018


namespace exists_permutation_divisible_by_7_l478_478891

-- Definition: a number formed by the digits of 1137
def digits_1137 := [1, 1, 3, 7]

-- Problem Statement: There exists a permutation of the digits that is divisible by 7
theorem exists_permutation_divisible_by_7 (n : ℕ) (h : ∃ (l : list ℕ), l.perm digits_1137 ∧ n = list.to_nat l) :
  ∃ m, m.perm digits_1137 ∧ m % 7 = 0 :=
sorry

end exists_permutation_divisible_by_7_l478_478891


namespace factor_quadratic_l478_478164

theorem factor_quadratic : ∀ (x : ℝ), 16*x^2 - 40*x + 25 = (4*x - 5)^2 := by
  intro x
  sorry

end factor_quadratic_l478_478164


namespace geometric_arithmetic_harmonic_solution_l478_478892

theorem geometric_arithmetic_harmonic_solution
  (x y : ℝ)
  (h1 : real.sqrt (x * y) = 600)
  (h2 : (x + y) / 2 - 49 = 2 * (x * y) / (x + y)) :
  (x = 800 ∧ y = 450) ∨ (x = 450 ∧ y = 800) :=
by
  -- Proof need to be appended
  sorry

end geometric_arithmetic_harmonic_solution_l478_478892


namespace probability_A_and_B_selected_l478_478311

open Finset

noncomputable def total_ways : ℕ := (choose 5 3)

noncomputable def favorable_ways : ℕ := (choose 3 1)

noncomputable def probability (total favorable : ℕ) : ℚ := favorable / total

theorem probability_A_and_B_selected :
  probability total_ways favorable_ways = 3 / 10 :=
by
  sorry

end probability_A_and_B_selected_l478_478311


namespace probability_of_selecting_A_and_B_l478_478680

open_locale classical

noncomputable def select_probability (n k : ℕ) : ℚ :=
  (nat.choose n k)⁻¹

theorem probability_of_selecting_A_and_B :
  let total_students := 5,
      students_to_select := 3,
      remaining_students := 3 in
  select_probability total_students students_to_select * nat.choose remaining_students 1 = 3 / 10 :=
by
  sorry

end probability_of_selecting_A_and_B_l478_478680


namespace biker_bob_east_distance_l478_478983

noncomputable def distance_between_towns : ℝ := 28.30194339616981
noncomputable def distance_west : ℝ := 30
noncomputable def distance_north_1 : ℝ := 6
noncomputable def distance_north_2 : ℝ := 18
noncomputable def total_distance_north : ℝ := distance_north_1 + distance_north_2
noncomputable def unknown_distance_east : ℝ := 45.0317 -- Expected distance east

theorem biker_bob_east_distance :
  ∃ (E : ℝ), (total_distance_north ^ 2 + (-distance_west + E) ^ 2 = distance_between_towns ^ 2) ∧ E = unknown_distance_east :=
by 
  sorry

end biker_bob_east_distance_l478_478983


namespace complement_of_A_is_correct_l478_478760

open Set

variable (U A : Set ℕ)

def universal_set := {1, 2, 3, 4, 5}
def set_A := {2, 3, 4}
def complement_A := {1, 5}

theorem complement_of_A_is_correct : (U = universal_set) → (A = set_A) → (U \ A = complement_A) :=
by 
  intros hU hA
  rw [hU, hA]
  exact rfl

end complement_of_A_is_correct_l478_478760


namespace ratio_height_radius_l478_478721

variable (V r h : ℝ)

theorem ratio_height_radius (h_eq_2r : h = 2 * r) (volume_eq : π * r^2 * h = V) : h / r = 2 :=
by
  sorry

end ratio_height_radius_l478_478721


namespace kiley_slices_eaten_l478_478154

def slices_of_cheesecake (total_calories_per_cheesecake calories_per_slice : ℕ) : ℕ :=
  total_calories_per_cheesecake / calories_per_slice

def slices_eaten (total_slices percentage_ate : ℚ) : ℚ :=
  total_slices * percentage_ate

theorem kiley_slices_eaten :
  ∀ (total_calories_per_cheesecake calories_per_slice : ℕ) (percentage_ate : ℚ),
  total_calories_per_cheesecake = 2800 →
  calories_per_slice = 350 →
  percentage_ate = (25 / 100 : ℚ) →
  slices_eaten (slices_of_cheesecake total_calories_per_cheesecake calories_per_slice) percentage_ate = 2 :=
by
  intros total_calories_per_cheesecake calories_per_slice percentage_ate h1 h2 h3
  rw [h1, h2, h3]
  sorry

end kiley_slices_eaten_l478_478154


namespace probability_A_and_B_selected_l478_478394

theorem probability_A_and_B_selected 
  (students : Finset ℕ) (A B : ℕ) 
  (h_students_size : students.card = 5) 
  (h_A_in_students : A ∈ students) 
  (h_B_in_students : B ∈ students)
  : (↑(Nat.choose 3) : ℚ) / (↑(Nat.choose 5) : ℚ) = 3 / 10 := 
by sorry

end probability_A_and_B_selected_l478_478394


namespace probability_of_selecting_A_and_B_l478_478679

open_locale classical

noncomputable def select_probability (n k : ℕ) : ℚ :=
  (nat.choose n k)⁻¹

theorem probability_of_selecting_A_and_B :
  let total_students := 5,
      students_to_select := 3,
      remaining_students := 3 in
  select_probability total_students students_to_select * nat.choose remaining_students 1 = 3 / 10 :=
by
  sorry

end probability_of_selecting_A_and_B_l478_478679


namespace probability_A_and_B_selected_l478_478339

theorem probability_A_and_B_selected (A B : Type) (S : Finset (Type)) (h : S.card = 5) (hA : A ∈ S) (hB : B ∈ S) (T : Finset (Type)) (hT : T.card = 3) :
  (∃ T ⊆ S, A ∈ T ∧ B ∈ T ∧ T.card = 3) → 
  ∃ T ⊆ S, T.card = 3 ∧ (finset.card (T ∩ {A, B}) = 2) → 
  (finset.card (T ∩ {A, B}) = 2) = (3 / 10 : Real) := sorry

end probability_A_and_B_selected_l478_478339


namespace intersection_points_count_l478_478772

noncomputable def f1 (x : ℝ) : ℝ := abs (3 * x - 2)
noncomputable def f2 (x : ℝ) : ℝ := -abs (2 * x + 5)

theorem intersection_points_count : 
  (∃ (x1 x2 : ℝ), x1 ≠ x2 ∧ f1 x1 = f2 x1 ∧ f1 x2 = f2 x2 ∧ 
    (∀ x : ℝ, f1 x = f2 x → x = x1 ∨ x = x2)) :=
sorry

end intersection_points_count_l478_478772


namespace probability_of_A_and_B_selected_l478_478470

variable {α : Type}

/-- From 5 students, we randomly select 3 students to participate in community service work. 
  Let A and B be two specific students among these five. The probability that 
  both A and B are selected is 3 / 10. -/
theorem probability_of_A_and_B_selected (A B : α) (s : Finset α) (h : s.card = 5) :
  (∃ t : Finset α, t ⊆ s ∧ A ∈ t ∧ B ∈ t ∧ (t.card = 3) ∧ (t.card = 3) →
   (@Finset.exists_subset_card_eq α t s 3) →
   (@Finset.exists_subset_card_eq α t (Finset.erase s A) 2) ∧
   (@Finset.exists_subset_card_eq α t (Finset.erase (Finset.erase s A) B) 1)) →
  (3 : ℚ) / 10 :=
by
  sorry -- Proof omitted as per instructions.

end probability_of_A_and_B_selected_l478_478470


namespace probability_of_selecting_A_and_B_is_three_tenths_l478_478592

-- Define the set of students
def students : Finset ℕ := {0, 1, 2, 3, 4}

-- Define subsets A and B
def student_A : ℕ := 0
def student_B : ℕ := 1

-- Total number of ways to select 3 students out of 5
def total_ways : ℕ := students.card.choose 3

-- Number of ways to select 3 students including A and B
def favorable_ways : ℕ := {2, 3, 4}.card.choose 1

-- The probability
def probability_of_selecting_A_and_B : ℚ := (favorable_ways : ℚ) / (total_ways : ℚ)

theorem probability_of_selecting_A_and_B_is_three_tenths :
  probability_of_selecting_A_and_B = 3 / 10 :=
by
  sorry

end probability_of_selecting_A_and_B_is_three_tenths_l478_478592


namespace probability_A_and_B_selected_l478_478686

/-- There are 5 students including A and B. The probability of selecting both A and B 
    when 3 students are selected randomly from 5 is 3/10. -/
theorem probability_A_and_B_selected :
  (∃ (students : List String) (h : students.length = 5),
    ∃ (selected : List String) (h' : selected.length = 3),
    "A" ∈ selected ∧ "B" ∈ selected) →
  ∃ (P : ℚ), P = 3 / 10 :=
by
  sorry

end probability_A_and_B_selected_l478_478686


namespace probability_AB_selected_l478_478506

def factorial (n : Nat) : Nat := if n = 0 then 1 else n * factorial (n - 1)

def choose (n k : Nat) : Nat := factorial n / (factorial k * factorial (n - k))

theorem probability_AB_selected (h1 : 5 = 5) (h2 : 3 = 3) : (choose 3 1) / (choose 5 3) = 3 / 10 := by
  sorry

end probability_AB_selected_l478_478506


namespace find_base_s_l478_478796

-- Definitions based on the conditions.
def five_hundred_thirty_base (s : ℕ) : ℕ := 5 * s^2 + 3 * s
def four_hundred_fifty_base (s : ℕ) : ℕ := 4 * s^2 + 5 * s
def one_thousand_one_hundred_base (s : ℕ) : ℕ := s^3 + s^2

-- The theorem to prove.
theorem find_base_s : (∃ s : ℕ, five_hundred_thirty_base s + four_hundred_fifty_base s = one_thousand_one_hundred_base s) → s = 8 :=
by
  sorry

end find_base_s_l478_478796


namespace find_angle_between_vectors_l478_478736

noncomputable def angle_between_vectors (a b : ℝ) (theta : ℝ) : Prop :=
  let ab_cos_theta := (a^2 + b^2) / (3.33 * a * b) in
  theta = Real.arccos ab_cos_theta

theorem find_angle_between_vectors (a b : ℝ) : 
  (a ≠ 0) → (b ≠ 0) → 
  (‖a + b‖ = 2 * ‖a - b‖) →
  ∃ θ, angle_between_vectors a b θ :=
by
  intros h_a_nonzero h_b_nonzero h_norm
  use Real.arccos ((a^2 + b^2) / (3.33 * a * b))
  exact sorry

end find_angle_between_vectors_l478_478736


namespace probability_of_A_and_B_selected_l478_478479

variable {α : Type}

/-- From 5 students, we randomly select 3 students to participate in community service work. 
  Let A and B be two specific students among these five. The probability that 
  both A and B are selected is 3 / 10. -/
theorem probability_of_A_and_B_selected (A B : α) (s : Finset α) (h : s.card = 5) :
  (∃ t : Finset α, t ⊆ s ∧ A ∈ t ∧ B ∈ t ∧ (t.card = 3) ∧ (t.card = 3) →
   (@Finset.exists_subset_card_eq α t s 3) →
   (@Finset.exists_subset_card_eq α t (Finset.erase s A) 2) ∧
   (@Finset.exists_subset_card_eq α t (Finset.erase (Finset.erase s A) B) 1)) →
  (3 : ℚ) / 10 :=
by
  sorry -- Proof omitted as per instructions.

end probability_of_A_and_B_selected_l478_478479


namespace aron_daily_dusting_time_l478_478977

def cleaning_data : Type := 
  { total_cleaning_time_per_week : ℕ,
    vacuuming_time_per_day : ℕ,
    vacuuming_days_per_week : ℕ,
    dusting_days_per_week : ℕ }

-- Given conditions
def aron_cleaning : cleaning_data := 
  { total_cleaning_time_per_week := 130,
    vacuuming_time_per_day := 30,
    vacuuming_days_per_week := 3,
    dusting_days_per_week := 2 }

theorem aron_daily_dusting_time (data : cleaning_data) : 
  data.total_cleaning_time_per_week = 130 → 
  data.vacuuming_time_per_day = 30 → 
  data.vacuuming_days_per_week = 3 → 
  data.dusting_days_per_week = 2 → 
  (data.total_cleaning_time_per_week - data.vacuuming_time_per_day * data.vacuuming_days_per_week) / data.dusting_days_per_week = 20 :=
by {
  sorry
}

end aron_daily_dusting_time_l478_478977


namespace tan_half_alpha_eq_one_third_l478_478711

open Real

theorem tan_half_alpha_eq_one_third (α : ℝ) (h1 : 5 * sin (2 * α) = 6 * cos α) (h2 : 0 < α ∧ α < π / 2) :
  tan (α / 2) = 1 / 3 :=
by
  sorry

end tan_half_alpha_eq_one_third_l478_478711


namespace probability_A_B_l478_478447

open Nat

-- Define a function for combination
def combination (n r : Nat) : Nat :=
  factorial n / (factorial r * factorial (n - r))

-- Define the set of students and the selection conditions
def num_students : Nat := 5
def num_selected : Nat := 3

-- Define A and B as selected students
def selected_A_B : Nat := combination 3 1

-- Define the probability calculation
def probability_A_B_selected : Rat :=
  (selected_A_B : Rat) / (combination num_students num_selected : Rat)

-- The statement of the problem
theorem probability_A_B : probability_A_B_selected = 3 / 10 := by
  sorry

end probability_A_B_l478_478447


namespace probability_of_selecting_A_and_B_l478_478554

theorem probability_of_selecting_A_and_B (students : Finset ℕ) (A B : ℕ)
  (h_size : students.card = 5) (h_in : A ∈ students ∧ B ∈ students) : 
  (Finset.card {s : Finset ℕ // s ⊆ students ∧ s.card = 3 ∧ A ∈ s ∧ B ∈ s}.to_set) / 
  (Finset.card {s : Finset ℕ // s ⊆ students ∧ s.card = 3}.to_set) = 3 / 10 :=
sorry

end probability_of_selecting_A_and_B_l478_478554


namespace prob_select_A_and_B_l478_478640

theorem prob_select_A_and_B (students : Finset ℕ) (A B : ℕ) (hA : A ∈ students) (hB : B ∈ students) (h_card : students.card = 5) :
  (students.choose 3).count {comb | A ∈ comb ∧ B ∈ comb} / (students.choose 3).card = 3 / 10 := sorry

end prob_select_A_and_B_l478_478640


namespace hyperbola_equation_l478_478745

theorem hyperbola_equation (a b k : ℝ) (p : ℝ × ℝ) (h_asymptotes : b = 3 * a)
  (h_hyperbola_passes_point : p = (2, -3 * Real.sqrt 3)) (h_hyperbola : ∀ x y, x^2 - (y^2 / (3 * a)^2) = k) :
  ∃ k, k = 1 :=
by
  -- Given the point p and asymptotes, we should prove k = 1.
  sorry

end hyperbola_equation_l478_478745


namespace probability_both_A_B_selected_l478_478226

-- Define the problem with the necessary conditions.
def probability_AB_selected (total_students : ℕ) (select_students : ℕ) (A B : ℕ) : ℚ :=
  if A < total_students ∧ B < total_students ∧ total_students = 5 ∧ select_students = 3 then
    let total_ways := (Nat.factorial total_students) / ((Nat.factorial select_students) * (Nat.factorial (total_students - select_students)))
    let favorable_ways := (Nat.factorial (total_students - 2)) / ((Nat.factorial (select_students - 2)) * (Nat.factorial ((total_students - 2) - (select_students - 2))))
    favorable_ways / total_ways
  else 0

theorem probability_both_A_B_selected 
  (total_students : ℕ) (select_students : ℕ) (A B : ℕ) 
  (h1 : total_students = 5) (h2 : select_students = 3) 
  (h3 : A < total_students) (h4 : B < total_students) 
  : probability_AB_selected total_students select_students A B = 3 / 10 := 
by {
  -- Insert the delta and logic to prove the theorem here.
  sorry
}

end probability_both_A_B_selected_l478_478226


namespace buckets_with_original_size_l478_478914

variable (C : ℝ) (N : ℝ)

theorem buckets_with_original_size :
  (C : ℝ) > 0 → (62.5 * (2 / 5) * C = N * C) → N = 25 :=
by
  intros hC hV
  have : C ≠ 0 := by linarith
  have h : (2 / 5) * 62.5 = 25 := by norm_num
  field_simp at hV
  rw h at hV
  assumption

end buckets_with_original_size_l478_478914


namespace probability_A_and_B_selected_l478_478304

open Finset

noncomputable def total_ways : ℕ := (choose 5 3)

noncomputable def favorable_ways : ℕ := (choose 3 1)

noncomputable def probability (total favorable : ℕ) : ℚ := favorable / total

theorem probability_A_and_B_selected :
  probability total_ways favorable_ways = 3 / 10 :=
by
  sorry

end probability_A_and_B_selected_l478_478304


namespace projection_matrix_3_4_l478_478192

def projection_matrix (v : ℝ × ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
  let ⟨vx, vy⟩ := v in
  let denom := vx * vx + vy * vy in
  Matrix.of (λ i j, ([9, 12, 12, 16][(2 * i.val + j.val)] / denom))

theorem projection_matrix_3_4 :
  projection_matrix (3, 4) = Matrix.of (λ i j, [9/25, 12/25, 12/25, 16/25][2 * i.val + j.val]) :=
sorry

end projection_matrix_3_4_l478_478192


namespace compute_b_l478_478732

noncomputable def rational_coefficients (a b : ℚ) :=
∃ x : ℚ, (x^3 + a * x^2 + b * x + 15 = 0)

theorem compute_b (a b : ℚ) (h1 : (3 + Real.sqrt 5)∈{root : ℝ | root^3 + a * root^2 + b * root + 15 = 0}) 
(h2 : rational_coefficients a b) : b = -18.5 :=
by
  sorry

end compute_b_l478_478732


namespace probability_of_selecting_A_and_B_l478_478254

-- Define the combination function
def combination (n r : ℕ) : ℕ := Nat.factorial n / (Nat.factorial r * Nat.factorial (n - r))

-- Define the conditions
def students : ℕ := 5
def to_select : ℕ := 3

-- Define the favorable outcomes where A and B are selected
def favorable_outcomes : ℕ := combination 3 1

-- Define the total outcomes of selecting 3 out of 5 students
def total_outcomes : ℕ := combination 5 3

-- Define the probability calculation
def probability_of_A_and_B : ℚ := favorable_outcomes / total_outcomes

-- Proposition to prove
theorem probability_of_selecting_A_and_B : probability_of_A_and_B = 3 / 10 :=
by
  sorry

end probability_of_selecting_A_and_B_l478_478254


namespace probability_of_selecting_A_and_B_l478_478273

-- Define the combination function
def combination (n r : ℕ) : ℕ := Nat.factorial n / (Nat.factorial r * Nat.factorial (n - r))

-- Define the conditions
def students : ℕ := 5
def to_select : ℕ := 3

-- Define the favorable outcomes where A and B are selected
def favorable_outcomes : ℕ := combination 3 1

-- Define the total outcomes of selecting 3 out of 5 students
def total_outcomes : ℕ := combination 5 3

-- Define the probability calculation
def probability_of_A_and_B : ℚ := favorable_outcomes / total_outcomes

-- Proposition to prove
theorem probability_of_selecting_A_and_B : probability_of_A_and_B = 3 / 10 :=
by
  sorry

end probability_of_selecting_A_and_B_l478_478273


namespace probability_of_selecting_A_and_B_l478_478290

open_locale classical

-- Define the number of students
def num_students := 5

-- Define the number of students to be selected
def students_to_select := 3

def combination (n k : ℕ) : ℕ := nat.choose n k

-- Define the correct answer
def correct_probability := 3 / 10

-- Define the probability calculation
def calc_probability := (combination 3 1) / (combination 5 3)

-- Statement of the theorem to be proved
theorem probability_of_selecting_A_and_B :
  calc_probability = correct_probability :=
by sorry

end probability_of_selecting_A_and_B_l478_478290


namespace sin_cos_sum_value_l478_478713

noncomputable def sin_cos_sum (α : ℝ) : ℝ :=
  if 0 < α ∧ α < π / 2 ∧ sin α * cos α = 1 / 8 then sin α + cos α else 0

theorem sin_cos_sum_value (α : ℝ) (h₁ : 0 < α) (h₂ : α < π / 2) (h₃ : sin α * cos α = 1 / 8) :
  sin_cos_sum α = sqrt 5 / 2 :=
by
  dsimp [sin_cos_sum]
  rw [if_pos]
  · split_ifs
    · exact sqrt_div_four_eq_sqrt_five_over_two h₁ h₂ h₃
  sorry

end sin_cos_sum_value_l478_478713


namespace probability_of_selecting_A_and_B_l478_478288

open_locale classical

-- Define the number of students
def num_students := 5

-- Define the number of students to be selected
def students_to_select := 3

def combination (n k : ℕ) : ℕ := nat.choose n k

-- Define the correct answer
def correct_probability := 3 / 10

-- Define the probability calculation
def calc_probability := (combination 3 1) / (combination 5 3)

-- Statement of the theorem to be proved
theorem probability_of_selecting_A_and_B :
  calc_probability = correct_probability :=
by sorry

end probability_of_selecting_A_and_B_l478_478288


namespace symmetry_of_transformed_graphs_l478_478203

theorem symmetry_of_transformed_graphs {f : ℝ → ℝ} :
  ∀ x : ℝ, f(x+1) = f(-(x + 1)) → ∀ x₀ : ℝ, f(x₀) = f(-x₀ - 2) := 
sorry

end symmetry_of_transformed_graphs_l478_478203


namespace probability_of_A_and_B_selected_l478_478489

variable {α : Type}

/-- From 5 students, we randomly select 3 students to participate in community service work. 
  Let A and B be two specific students among these five. The probability that 
  both A and B are selected is 3 / 10. -/
theorem probability_of_A_and_B_selected (A B : α) (s : Finset α) (h : s.card = 5) :
  (∃ t : Finset α, t ⊆ s ∧ A ∈ t ∧ B ∈ t ∧ (t.card = 3) ∧ (t.card = 3) →
   (@Finset.exists_subset_card_eq α t s 3) →
   (@Finset.exists_subset_card_eq α t (Finset.erase s A) 2) ∧
   (@Finset.exists_subset_card_eq α t (Finset.erase (Finset.erase s A) B) 1)) →
  (3 : ℚ) / 10 :=
by
  sorry -- Proof omitted as per instructions.

end probability_of_A_and_B_selected_l478_478489


namespace probability_of_selecting_A_and_B_l478_478256

-- Define the combination function
def combination (n r : ℕ) : ℕ := Nat.factorial n / (Nat.factorial r * Nat.factorial (n - r))

-- Define the conditions
def students : ℕ := 5
def to_select : ℕ := 3

-- Define the favorable outcomes where A and B are selected
def favorable_outcomes : ℕ := combination 3 1

-- Define the total outcomes of selecting 3 out of 5 students
def total_outcomes : ℕ := combination 5 3

-- Define the probability calculation
def probability_of_A_and_B : ℚ := favorable_outcomes / total_outcomes

-- Proposition to prove
theorem probability_of_selecting_A_and_B : probability_of_A_and_B = 3 / 10 :=
by
  sorry

end probability_of_selecting_A_and_B_l478_478256


namespace conic_section_is_ellipse_l478_478937

/-- Given two fixed points (0, 2) and (4, -1) and the equation 
    sqrt(x^2 + (y - 2)^2) + sqrt((x - 4)^2 + (y + 1)^2) = 12, 
    prove that the conic section is an ellipse. -/
theorem conic_section_is_ellipse 
  (x y : ℝ)
  (h : Real.sqrt (x^2 + (y - 2)^2) + Real.sqrt ((x - 4)^2 + (y + 1)^2) = 12) :
  ∃ (F1 F2 : ℝ × ℝ), 
    F1 = (0, 2) ∧ 
    F2 = (4, -1) ∧ 
    ∀ (P : ℝ × ℝ), P = (x, y) → 
      Real.sqrt ((P.1 - F1.1)^2 + (P.2 - F1.2)^2) + 
      Real.sqrt ((P.1 - F2.1)^2 + (P.2 - F2.2)^2) = 12 := 
sorry

end conic_section_is_ellipse_l478_478937


namespace probability_of_selecting_A_and_B_is_three_tenths_l478_478600

-- Define the set of students
def students : Finset ℕ := {0, 1, 2, 3, 4}

-- Define subsets A and B
def student_A : ℕ := 0
def student_B : ℕ := 1

-- Total number of ways to select 3 students out of 5
def total_ways : ℕ := students.card.choose 3

-- Number of ways to select 3 students including A and B
def favorable_ways : ℕ := {2, 3, 4}.card.choose 1

-- The probability
def probability_of_selecting_A_and_B : ℚ := (favorable_ways : ℚ) / (total_ways : ℚ)

theorem probability_of_selecting_A_and_B_is_three_tenths :
  probability_of_selecting_A_and_B = 3 / 10 :=
by
  sorry

end probability_of_selecting_A_and_B_is_three_tenths_l478_478600


namespace total_fencing_cost_is_correct_l478_478777

-- Defining the lengths of each side
def length1 : ℝ := 50
def length2 : ℝ := 75
def length3 : ℝ := 60
def length4 : ℝ := 80
def length5 : ℝ := 65

-- Defining the cost per unit length for each side
def cost_per_meter1 : ℝ := 2
def cost_per_meter2 : ℝ := 3
def cost_per_meter3 : ℝ := 4
def cost_per_meter4 : ℝ := 3.5
def cost_per_meter5 : ℝ := 5

-- Calculating the total cost for each side
def cost1 : ℝ := length1 * cost_per_meter1
def cost2 : ℝ := length2 * cost_per_meter2
def cost3 : ℝ := length3 * cost_per_meter3
def cost4 : ℝ := length4 * cost_per_meter4
def cost5 : ℝ := length5 * cost_per_meter5

-- Summing up the total cost for all sides
def total_cost : ℝ := cost1 + cost2 + cost3 + cost4 + cost5

-- The theorem to be proven
theorem total_fencing_cost_is_correct : total_cost = 1170 := by
  sorry

end total_fencing_cost_is_correct_l478_478777


namespace probability_both_A_and_B_selected_l478_478397

-- The universe of students we consider
inductive Student
| A : Student
| B : Student
| C : Student
| D : Student
| E : Student

open Student

-- Function to count combinations
def combinations (n k : ℕ) : ℕ := (Nat.factorial n) / ((Nat.factorial k) * (Nat.factorial (n - k)))

-- The set of all students
def students : List Student := [A, B, C, D, E]

-- The total number of ways to select 3 students from 5
def total_selections := combinations 5 3

-- The number of ways A and B are both selected by selecting 1 more from the remaining 3 students
def selections_AB := combinations 3 1

-- The probability that both A and B are selected
def probability_AB := (selections_AB : ℚ) / (total_selections : ℚ)

-- The proof statement
theorem probability_both_A_and_B_selected :
  probability_AB = 3 / 10 :=
sorry

end probability_both_A_and_B_selected_l478_478397


namespace probability_A_and_B_selected_l478_478319

open Finset

noncomputable def total_ways : ℕ := (choose 5 3)

noncomputable def favorable_ways : ℕ := (choose 3 1)

noncomputable def probability (total favorable : ℕ) : ℚ := favorable / total

theorem probability_A_and_B_selected :
  probability total_ways favorable_ways = 3 / 10 :=
by
  sorry

end probability_A_and_B_selected_l478_478319


namespace modulus_Z_when_m_zero_Z_purely_imaginary_iff_Z_in_second_quadrant_iff_l478_478832

open Complex Real

-- Definition of the complex number Z
def Z (m : ℝ) := (m^2 + 3 * m - 4 : ℂ) + (m^2 - 10 * m + 9 : ℂ) * Complex.i

-- Problem 1: Proving the modulus of Z when m = 0
theorem modulus_Z_when_m_zero : ∀ m : ℝ, m = 0 → Complex.abs (Z m) = Real.sqrt 97 := by
  sorry

-- Problem 2: Proving Z is purely imaginary if and only if m = -4
theorem Z_purely_imaginary_iff : ∀ m : ℝ, (Z m).re = 0 ∧ (Z m).im ≠ 0 ↔ m = -4 := by
  sorry

-- Problem 3: Proving Z lies in the second quadrant if and only if -4 < m < 1
theorem Z_in_second_quadrant_iff : ∀ m : ℝ, (Z m).re < 0 ∧ (Z m).im > 0 ↔ -4 < m ∧ m < 1 := by
  sorry

end modulus_Z_when_m_zero_Z_purely_imaginary_iff_Z_in_second_quadrant_iff_l478_478832


namespace probability_obtuse_angle_AQB_l478_478853

-- Define the vertices of the pentagon
structure Point where
  x : ℝ
  y : ℝ

def A : Point := { x := 0, y := 3 }
def B : Point := { x := 5, y := 0 }
def C : Point := { x := 2 * Real.pi, y := 0 }
def D : Point := { x := 2 * Real.pi, y := 5 }
def E : Point := { x := 0, y := 5 }

-- Define the interior of the pentagon (a more formal definition would be needed)
def Pentagon : Set Point := {p | /* formal definition of interior of A-B-C-D-E */}

-- Define a random point Q from the interior of the pentagon
axiom Q : Point
axiom Q_in_pentagon : Q ∈ Pentagon

-- Define a predicate for an obtuse angle AQB
def obtuse_angle (A Q B : Point) : Prop := sorry  -- Need formal definition

-- Main statement to prove
theorem probability_obtuse_angle_AQB :
  (measure {Q : Point | Q ∈ Pentagon ∧ obtuse_angle A Q B}) / (measure Pentagon) = 17 / 128 := sorry

end probability_obtuse_angle_AQB_l478_478853


namespace janice_flights_of_stairs_l478_478806

theorem janice_flights_of_stairs (flights_up_per_trip flights_down_per_trip : ℕ) (trips_up trips_down : ℕ) :
    flights_up_per_trip = 3 →
    flights_down_per_trip = 3 →
    trips_up = 5 →
    trips_down = 3 →
    (flights_up_per_trip * trips_up) + (flights_down_per_trip * trips_down) = 24 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  norm_num

end janice_flights_of_stairs_l478_478806


namespace top_four_cards_probability_l478_478099

def num_cards : ℕ := 52

def num_hearts : ℕ := 13

def num_diamonds : ℕ := 13

def num_clubs : ℕ := 13

def prob_first_heart := (num_hearts : ℚ) / num_cards
def prob_second_heart := (num_hearts - 1 : ℚ) / (num_cards - 1)
def prob_third_diamond := (num_diamonds : ℚ) / (num_cards - 2)
def prob_fourth_club := (num_clubs : ℚ) / (num_cards - 3)

def combined_prob :=
  prob_first_heart * prob_second_heart * prob_third_diamond * prob_fourth_club

theorem top_four_cards_probability :
  combined_prob = 39 / 63875 := by
  sorry

end top_four_cards_probability_l478_478099


namespace prob_select_A_and_B_l478_478646

theorem prob_select_A_and_B (students : Finset ℕ) (A B : ℕ) (hA : A ∈ students) (hB : B ∈ students) (h_card : students.card = 5) :
  (students.choose 3).count {comb | A ∈ comb ∧ B ∈ comb} / (students.choose 3).card = 3 / 10 := sorry

end prob_select_A_and_B_l478_478646


namespace purely_imaginary_l478_478749

def z₁ : ℂ := 1 - complex.i
def z₂ (x : ℝ) : ℂ := -1 + x * complex.i

theorem purely_imaginary (x : ℝ) : (∃ (y : ℂ), z₁ * z₂ x = y * complex.i) ↔ x = 1 :=
by { simp [z₁, z₂], sorry }

end purely_imaginary_l478_478749


namespace integer_solutions_count_l478_478766

theorem integer_solutions_count : 
  {x : ℤ | (x^2 - x - 6)^(x + 3) = 1}.to_finset.card = 5 :=
sorry

end integer_solutions_count_l478_478766


namespace select_3_from_5_prob_l478_478425

theorem select_3_from_5_prob {students : Finset ℕ} 
  (h : students.card = 5) 
  {A B : ℕ} 
  (hA : A ∈ students) 
  (hB : B ∈ students) 
  (H : 3 ≤ students.card) : 
  (finset.card (students.filter (λ x, x ≠ A ∧ x ≠ B)) = 3) → 
  (finset.card ((students.filter (λ x, x ≠ A ∧ x ≠ B)).subtype (λ x, True)) = 3) → 
  ((finset.card (finset.filter (λ (students : Finset ℕ) A ∈ students ∧ B ∈ students ↔ 3 = students.card) 5)) = ↑10) → 
  ∃ students', (students'.card = 3 ∧ A ∈ students' ∧ B ∈ students') → 
  ∃ (p : ℚ), p = 3 / 10 ∧ students.filter (λ x, x = A ∨ x = B) = 2 :=
  sorry

end select_3_from_5_prob_l478_478425


namespace dispatch_plans_correct_l478_478969

inductive Teacher : Type
| A | B | C | D | E | F | G | H

open Teacher

def valid_selection (t : List Teacher) : Prop :=
  if (A ∈ t) then (C ∈ t ∧ B ∉ t) else (C ∉ t)

def number_of_selections : ℕ :=
  let selections := filter valid_selection (List.powersetOfLength 4 [A, B, C, D, E, F, G, H])
  selections.length

def valid_arrangements (t : List Teacher) : ℕ :=
  if valid_selection t then 4! else 0

def total_dispatch_plans : ℕ :=
  let selections := filter valid_selection (List.powersetOfLength 4 [A, B, C, D, E, F, G, H])
  selections.length * 24

theorem dispatch_plans_correct : total_dispatch_plans = 600 := by
  sorry

end dispatch_plans_correct_l478_478969


namespace smallest_positive_debt_l478_478923

noncomputable def pigs_value : ℤ := 300
noncomputable def goats_value : ℤ := 210

theorem smallest_positive_debt : ∃ D p g : ℤ, (D = pigs_value * p + goats_value * g) ∧ D > 0 ∧ ∀ D' p' g' : ℤ, (D' = pigs_value * p' + goats_value * g' ∧ D' > 0) → D ≤ D' :=
by
  sorry

end smallest_positive_debt_l478_478923


namespace sum_of_digits_of_greatest_prime_divisor_of_1023_l478_478054

theorem sum_of_digits_of_greatest_prime_divisor_of_1023 : 
  let d := 1023
  let greatest_prime_divisor := 31
  sum_of_digits := (3 + 1)
  in (∀ p : ℕ, prime p → p ∣ d → p ≤ greatest_prime_divisor) ∧
     (∀ s : ℕ, s < greatest_prime_divisor → ¬prime s) →
     sum_of_digits = 4 :=
by
  sorry

end sum_of_digits_of_greatest_prime_divisor_of_1023_l478_478054


namespace find_a_n_l478_478135

variable {a_n : ℕ → ℝ}
variable {S_n : ℕ → ℝ}

axiom cond1 : ∀ n, S_n + (1 + 2 / n) * a_n n = 4

noncomputable def required_formula (n : ℕ) : ℝ :=
  n / 2^(n-1)

theorem find_a_n : ∀ n, a_n n = required_formula n :=
by
  sorry

end find_a_n_l478_478135


namespace probability_A_and_B_selected_l478_478239

theorem probability_A_and_B_selected (students : Finset ℕ) (A B : ℕ) (h : A ∈ students ∧ B ∈ students ∧ students.card = 5) : 
  (students.choose 3).count (λ s, {A, B} ⊆ s) / (students.choose 3).card = 3 / 10 := by
  sorry

end probability_A_and_B_selected_l478_478239


namespace probability_of_selecting_A_and_B_l478_478276

-- Define the combination function
def combination (n r : ℕ) : ℕ := Nat.factorial n / (Nat.factorial r * Nat.factorial (n - r))

-- Define the conditions
def students : ℕ := 5
def to_select : ℕ := 3

-- Define the favorable outcomes where A and B are selected
def favorable_outcomes : ℕ := combination 3 1

-- Define the total outcomes of selecting 3 out of 5 students
def total_outcomes : ℕ := combination 5 3

-- Define the probability calculation
def probability_of_A_and_B : ℚ := favorable_outcomes / total_outcomes

-- Proposition to prove
theorem probability_of_selecting_A_and_B : probability_of_A_and_B = 3 / 10 :=
by
  sorry

end probability_of_selecting_A_and_B_l478_478276


namespace clock_mirror_image_correct_l478_478978

theorem clock_mirror_image_correct (t : ℕ) (H : t = 800) : 
    ∃ (D : string), D = "16:00" := 
sorry

end clock_mirror_image_correct_l478_478978


namespace probability_of_selecting_A_and_B_l478_478665

open_locale classical

noncomputable def select_probability (n k : ℕ) : ℚ :=
  (nat.choose n k)⁻¹

theorem probability_of_selecting_A_and_B :
  let total_students := 5,
      students_to_select := 3,
      remaining_students := 3 in
  select_probability total_students students_to_select * nat.choose remaining_students 1 = 3 / 10 :=
by
  sorry

end probability_of_selecting_A_and_B_l478_478665


namespace sqrt_500_simplified_l478_478869

theorem sqrt_500_simplified : sqrt 500 = 10 * sqrt 5 :=
by
sorry

end sqrt_500_simplified_l478_478869


namespace probability_both_A_B_selected_l478_478221

-- Define the problem with the necessary conditions.
def probability_AB_selected (total_students : ℕ) (select_students : ℕ) (A B : ℕ) : ℚ :=
  if A < total_students ∧ B < total_students ∧ total_students = 5 ∧ select_students = 3 then
    let total_ways := (Nat.factorial total_students) / ((Nat.factorial select_students) * (Nat.factorial (total_students - select_students)))
    let favorable_ways := (Nat.factorial (total_students - 2)) / ((Nat.factorial (select_students - 2)) * (Nat.factorial ((total_students - 2) - (select_students - 2))))
    favorable_ways / total_ways
  else 0

theorem probability_both_A_B_selected 
  (total_students : ℕ) (select_students : ℕ) (A B : ℕ) 
  (h1 : total_students = 5) (h2 : select_students = 3) 
  (h3 : A < total_students) (h4 : B < total_students) 
  : probability_AB_selected total_students select_students A B = 3 / 10 := 
by {
  -- Insert the delta and logic to prove the theorem here.
  sorry
}

end probability_both_A_B_selected_l478_478221


namespace select_3_from_5_prob_l478_478427

theorem select_3_from_5_prob {students : Finset ℕ} 
  (h : students.card = 5) 
  {A B : ℕ} 
  (hA : A ∈ students) 
  (hB : B ∈ students) 
  (H : 3 ≤ students.card) : 
  (finset.card (students.filter (λ x, x ≠ A ∧ x ≠ B)) = 3) → 
  (finset.card ((students.filter (λ x, x ≠ A ∧ x ≠ B)).subtype (λ x, True)) = 3) → 
  ((finset.card (finset.filter (λ (students : Finset ℕ) A ∈ students ∧ B ∈ students ↔ 3 = students.card) 5)) = ↑10) → 
  ∃ students', (students'.card = 3 ∧ A ∈ students' ∧ B ∈ students') → 
  ∃ (p : ℚ), p = 3 / 10 ∧ students.filter (λ x, x = A ∨ x = B) = 2 :=
  sorry

end select_3_from_5_prob_l478_478427


namespace probability_A_and_B_selected_l478_478317

open Finset

noncomputable def total_ways : ℕ := (choose 5 3)

noncomputable def favorable_ways : ℕ := (choose 3 1)

noncomputable def probability (total favorable : ℕ) : ℚ := favorable / total

theorem probability_A_and_B_selected :
  probability total_ways favorable_ways = 3 / 10 :=
by
  sorry

end probability_A_and_B_selected_l478_478317


namespace probability_A_and_B_selected_l478_478538

theorem probability_A_and_B_selected :
  let n := 5
  let r := 3
  let total_ways := Nat.choose n r
  let favorable_ways := Nat.choose 3 1
  let probability := favorable_ways / total_ways
  probability = (3 / 10) := 
by
  sorry

end probability_A_and_B_selected_l478_478538


namespace probability_A_and_B_selected_l478_478327

theorem probability_A_and_B_selected (A B : Type) (S : Finset (Type)) (h : S.card = 5) (hA : A ∈ S) (hB : B ∈ S) (T : Finset (Type)) (hT : T.card = 3) :
  (∃ T ⊆ S, A ∈ T ∧ B ∈ T ∧ T.card = 3) → 
  ∃ T ⊆ S, T.card = 3 ∧ (finset.card (T ∩ {A, B}) = 2) → 
  (finset.card (T ∩ {A, B}) = 2) = (3 / 10 : Real) := sorry

end probability_A_and_B_selected_l478_478327


namespace brendan_threw_back_l478_478122

-- Brendan's catches in the morning, throwing back x fish and catching more in the afternoon
def brendan_morning (x : ℕ) : ℕ := 8 - x
def brendan_afternoon : ℕ := 5

-- Brendan's and his dad's total catches
def brendan_total (x : ℕ) : ℕ := brendan_morning x + brendan_afternoon
def dad_total : ℕ := 13

-- Combined total fish caught by both
def total_fish (x : ℕ) : ℕ := brendan_total x + dad_total

-- The number of fish thrown back by Brendan
theorem brendan_threw_back : ∃ x : ℕ, total_fish x = 23 ∧ x = 3 :=
by
  sorry

end brendan_threw_back_l478_478122


namespace probability_both_A_B_selected_l478_478224

-- Define the problem with the necessary conditions.
def probability_AB_selected (total_students : ℕ) (select_students : ℕ) (A B : ℕ) : ℚ :=
  if A < total_students ∧ B < total_students ∧ total_students = 5 ∧ select_students = 3 then
    let total_ways := (Nat.factorial total_students) / ((Nat.factorial select_students) * (Nat.factorial (total_students - select_students)))
    let favorable_ways := (Nat.factorial (total_students - 2)) / ((Nat.factorial (select_students - 2)) * (Nat.factorial ((total_students - 2) - (select_students - 2))))
    favorable_ways / total_ways
  else 0

theorem probability_both_A_B_selected 
  (total_students : ℕ) (select_students : ℕ) (A B : ℕ) 
  (h1 : total_students = 5) (h2 : select_students = 3) 
  (h3 : A < total_students) (h4 : B < total_students) 
  : probability_AB_selected total_students select_students A B = 3 / 10 := 
by {
  -- Insert the delta and logic to prove the theorem here.
  sorry
}

end probability_both_A_B_selected_l478_478224


namespace probability_of_A_and_B_selected_l478_478492

variable {α : Type}

/-- From 5 students, we randomly select 3 students to participate in community service work. 
  Let A and B be two specific students among these five. The probability that 
  both A and B are selected is 3 / 10. -/
theorem probability_of_A_and_B_selected (A B : α) (s : Finset α) (h : s.card = 5) :
  (∃ t : Finset α, t ⊆ s ∧ A ∈ t ∧ B ∈ t ∧ (t.card = 3) ∧ (t.card = 3) →
   (@Finset.exists_subset_card_eq α t s 3) →
   (@Finset.exists_subset_card_eq α t (Finset.erase s A) 2) ∧
   (@Finset.exists_subset_card_eq α t (Finset.erase (Finset.erase s A) B) 1)) →
  (3 : ℚ) / 10 :=
by
  sorry -- Proof omitted as per instructions.

end probability_of_A_and_B_selected_l478_478492


namespace probability_of_selecting_A_and_B_l478_478295

open_locale classical

-- Define the number of students
def num_students := 5

-- Define the number of students to be selected
def students_to_select := 3

def combination (n k : ℕ) : ℕ := nat.choose n k

-- Define the correct answer
def correct_probability := 3 / 10

-- Define the probability calculation
def calc_probability := (combination 3 1) / (combination 5 3)

-- Statement of the theorem to be proved
theorem probability_of_selecting_A_and_B :
  calc_probability = correct_probability :=
by sorry

end probability_of_selecting_A_and_B_l478_478295


namespace probability_A_and_B_selected_l478_478316

open Finset

noncomputable def total_ways : ℕ := (choose 5 3)

noncomputable def favorable_ways : ℕ := (choose 3 1)

noncomputable def probability (total favorable : ℕ) : ℚ := favorable / total

theorem probability_A_and_B_selected :
  probability total_ways favorable_ways = 3 / 10 :=
by
  sorry

end probability_A_and_B_selected_l478_478316


namespace eccentricity_of_ellipse_l478_478733

theorem eccentricity_of_ellipse 
  (F1 F2 : ℝ)
  (focal_distance : dist F1 F2 = 4)
  (a perimeter_triangle_ABF2 : ℝ)
  (triangle_perimeter : 4 * a = 32)
  : e = c / a := 
begin
  have a_val : a = 8, by linarith,
  have c_val : c = 2, by linarith,
  have eccentricity : e = c / a, by field_simp [c_val, a_val],
  exact eccentricity,
end

end eccentricity_of_ellipse_l478_478733


namespace inscribed_cube_side_length_l478_478038

-- Given lengths a and b and the distance c
variables (a b c : ℝ)

-- Define the conditions
axiom edges_perpendicular : ∀ (a b c : ℝ), ∃ tetrahedron, 
  (tetrahedron.opposite_edges_perpendicular a b) ∧
  (tetrahedron.distance_between_opposite_edges a b = c)

-- Define the question
def side_length_of_inscribed_cube (a b c : ℝ) : ℝ :=
  a * b * c / (a * b + b * c + a * c) 

-- The theorem that needs to be proven
theorem inscribed_cube_side_length :
  ∀ (a b c : ℝ), 
  (∀ a b c, ∃ tetrahedron, (tetrahedron.opposite_edges_perpendicular a b) ∧
    (tetrahedron.distance_between_opposite_edges a b = c)) →
  side_length_of_inscribed_cube a b c = (a * b * c) / (a * b + b * c + a * c) :=
by
  -- Placeholder for the complete proof
  sorry

end inscribed_cube_side_length_l478_478038


namespace arlene_hike_distance_l478_478113

-- Define the conditions: Arlene's pace and the time she spent hiking
def arlene_pace : ℝ := 4 -- miles per hour
def arlene_time_hiking : ℝ := 6 -- hours

-- Define the problem statement and provide the mathematical proof
theorem arlene_hike_distance :
  arlene_pace * arlene_time_hiking = 24 :=
by
  -- This is where the proof would go
  sorry

end arlene_hike_distance_l478_478113


namespace probability_A_and_B_selected_l478_478528

theorem probability_A_and_B_selected :
  let n := 5
  let r := 3
  let total_ways := Nat.choose n r
  let favorable_ways := Nat.choose 3 1
  let probability := favorable_ways / total_ways
  probability = (3 / 10) := 
by
  sorry

end probability_A_and_B_selected_l478_478528


namespace fraction_flower_beds_l478_478966

theorem fraction_flower_beds (yard_length yard_width : ℕ) (side1 side2 flower_leg : ℕ) 
  (h1 : yard_length = 30) (h2 : yard_width = 6)
  (h3 : side1 = 20) (h4 : side2 = 30)
  (h5 : flower_leg = (side2 - side1) / 2):

  let yard_area := yard_length * yard_width in
  let flower_area := 2 * (1 / 2 : ℚ) * flower_leg * flower_leg in
  let fraction_occupied := flower_area / yard_area in
  fraction_occupied = (5 / 36 : ℚ) :=
by 
  sorry

end fraction_flower_beds_l478_478966


namespace select_3_from_5_prob_l478_478438

theorem select_3_from_5_prob {students : Finset ℕ} 
  (h : students.card = 5) 
  {A B : ℕ} 
  (hA : A ∈ students) 
  (hB : B ∈ students) 
  (H : 3 ≤ students.card) : 
  (finset.card (students.filter (λ x, x ≠ A ∧ x ≠ B)) = 3) → 
  (finset.card ((students.filter (λ x, x ≠ A ∧ x ≠ B)).subtype (λ x, True)) = 3) → 
  ((finset.card (finset.filter (λ (students : Finset ℕ) A ∈ students ∧ B ∈ students ↔ 3 = students.card) 5)) = ↑10) → 
  ∃ students', (students'.card = 3 ∧ A ∈ students' ∧ B ∈ students') → 
  ∃ (p : ℚ), p = 3 / 10 ∧ students.filter (λ x, x = A ∨ x = B) = 2 :=
  sorry

end select_3_from_5_prob_l478_478438


namespace probability_A_and_B_selected_l478_478587

/-- Let there be a group of 5 students including A and B. 
    If 3 students are selected randomly from these 5, the probability of selecting both A and B is 3/10. -/
theorem probability_A_and_B_selected : 
  let students := 5 in
  let selected := 3 in
  let combinations := students.choose selected in
  let favorable_combinations := (students - 2).choose (selected - 2) in
  let probability := (favorable_combinations : ℚ) / (combinations : ℚ) in
  probability = 3 / 10 := 
by sorry

end probability_A_and_B_selected_l478_478587


namespace probability_A_and_B_selected_l478_478537

theorem probability_A_and_B_selected :
  let n := 5
  let r := 3
  let total_ways := Nat.choose n r
  let favorable_ways := Nat.choose 3 1
  let probability := favorable_ways / total_ways
  probability = (3 / 10) := 
by
  sorry

end probability_A_and_B_selected_l478_478537


namespace probability_both_A_B_selected_l478_478220

-- Define the problem with the necessary conditions.
def probability_AB_selected (total_students : ℕ) (select_students : ℕ) (A B : ℕ) : ℚ :=
  if A < total_students ∧ B < total_students ∧ total_students = 5 ∧ select_students = 3 then
    let total_ways := (Nat.factorial total_students) / ((Nat.factorial select_students) * (Nat.factorial (total_students - select_students)))
    let favorable_ways := (Nat.factorial (total_students - 2)) / ((Nat.factorial (select_students - 2)) * (Nat.factorial ((total_students - 2) - (select_students - 2))))
    favorable_ways / total_ways
  else 0

theorem probability_both_A_B_selected 
  (total_students : ℕ) (select_students : ℕ) (A B : ℕ) 
  (h1 : total_students = 5) (h2 : select_students = 3) 
  (h3 : A < total_students) (h4 : B < total_students) 
  : probability_AB_selected total_students select_students A B = 3 / 10 := 
by {
  -- Insert the delta and logic to prove the theorem here.
  sorry
}

end probability_both_A_B_selected_l478_478220


namespace probability_A_and_B_selected_l478_478340

theorem probability_A_and_B_selected (A B : Type) (S : Finset (Type)) (h : S.card = 5) (hA : A ∈ S) (hB : B ∈ S) (T : Finset (Type)) (hT : T.card = 3) :
  (∃ T ⊆ S, A ∈ T ∧ B ∈ T ∧ T.card = 3) → 
  ∃ T ⊆ S, T.card = 3 ∧ (finset.card (T ∩ {A, B}) = 2) → 
  (finset.card (T ∩ {A, B}) = 2) = (3 / 10 : Real) := sorry

end probability_A_and_B_selected_l478_478340


namespace prob_select_A_and_B_l478_478653

theorem prob_select_A_and_B (students : Finset ℕ) (A B : ℕ) (hA : A ∈ students) (hB : B ∈ students) (h_card : students.card = 5) :
  (students.choose 3).count {comb | A ∈ comb ∧ B ∈ comb} / (students.choose 3).card = 3 / 10 := sorry

end prob_select_A_and_B_l478_478653


namespace factorization_of_polynomial_l478_478182

theorem factorization_of_polynomial (x : ℝ) : 16 * x ^ 2 - 40 * x + 25 = (4 * x - 5) ^ 2 := by 
  sorry

end factorization_of_polynomial_l478_478182


namespace factorize_quadratic_l478_478178

theorem factorize_quadratic (x : ℝ) : 
  16 * x^2 - 40 * x + 25 = (4 * x - 5)^2 :=
sorry

end factorize_quadratic_l478_478178


namespace probability_AB_selected_l478_478510

def factorial (n : Nat) : Nat := if n = 0 then 1 else n * factorial (n - 1)

def choose (n k : Nat) : Nat := factorial n / (factorial k * factorial (n - k))

theorem probability_AB_selected (h1 : 5 = 5) (h2 : 3 = 3) : (choose 3 1) / (choose 5 3) = 3 / 10 := by
  sorry

end probability_AB_selected_l478_478510


namespace probability_of_selecting_A_and_B_l478_478294

open_locale classical

-- Define the number of students
def num_students := 5

-- Define the number of students to be selected
def students_to_select := 3

def combination (n k : ℕ) : ℕ := nat.choose n k

-- Define the correct answer
def correct_probability := 3 / 10

-- Define the probability calculation
def calc_probability := (combination 3 1) / (combination 5 3)

-- Statement of the theorem to be proved
theorem probability_of_selecting_A_and_B :
  calc_probability = correct_probability :=
by sorry

end probability_of_selecting_A_and_B_l478_478294


namespace remainder_when_divided_by_five_l478_478148

theorem remainder_when_divided_by_five :
  let E := 1250 * 1625 * 1830 * 2075 + 245
  E % 5 = 0 := by
  sorry

end remainder_when_divided_by_five_l478_478148


namespace part1_f_two_x_is_1_level_distribution_part1_g_two_div_x_is_1_level_distribution_part2_varphi_is_2_level_distribution_part3_varphi_is_t_level_distribution_for_all_t_l478_478133

noncomputable def is_t_level_distribution (f : ℝ → ℝ) (x0 t : ℝ) : Prop :=
  f (x0 + t) = f x0 * f t

def f_x (x : ℝ) : ℝ := 2 * x

def g_x (x : ℝ) : ℝ := 2 / x

def varphi (a x : ℝ) : ℝ := Real.sqrt (a / (x^2 + 1))

theorem part1_f_two_x_is_1_level_distribution :
  ∃ x0, is_t_level_distribution f_x x0 1 :=
sorry

theorem part1_g_two_div_x_is_1_level_distribution :
  ∃ x0, is_t_level_distribution g_x x0 1 :=
sorry

theorem part2_varphi_is_2_level_distribution (a : ℝ) (h : 0 < a) :
  a ∈ set.Icc (15 - 10 * Real.sqrt 2) (15 + 10 * Real.sqrt 2) →
  ∃ x0, is_t_level_distribution (varphi a) x0 2 :=
sorry

theorem part3_varphi_is_t_level_distribution_for_all_t (a : ℝ) (h : 0 < a) :
  a = 1 → ∀ t, ∃ x0, is_t_level_distribution (varphi a) x0 t :=
sorry

end part1_f_two_x_is_1_level_distribution_part1_g_two_div_x_is_1_level_distribution_part2_varphi_is_2_level_distribution_part3_varphi_is_t_level_distribution_for_all_t_l478_478133


namespace probability_of_selecting_A_and_B_l478_478293

open_locale classical

-- Define the number of students
def num_students := 5

-- Define the number of students to be selected
def students_to_select := 3

def combination (n k : ℕ) : ℕ := nat.choose n k

-- Define the correct answer
def correct_probability := 3 / 10

-- Define the probability calculation
def calc_probability := (combination 3 1) / (combination 5 3)

-- Statement of the theorem to be proved
theorem probability_of_selecting_A_and_B :
  calc_probability = correct_probability :=
by sorry

end probability_of_selecting_A_and_B_l478_478293


namespace area_of_square_inscribed_in_right_triangle_l478_478874

theorem area_of_square_inscribed_in_right_triangle (AB CD : ℝ) (hAB : AB = 35) (hCD : CD = 65) : 
  ∃ x : ℝ, (Square x) ∧ (InscribedSquareInRightTriangle x AB CD) ∧ (x^2 = 2275) := 
by 
  have prop := (AB * CD) 
  have h := 35 * 65 
  exact ⟨2275, sorry⟩ 

end area_of_square_inscribed_in_right_triangle_l478_478874


namespace not_cube_of_sum_l478_478855

theorem not_cube_of_sum (a b : ℕ) : ¬ ∃ (k : ℤ), a^3 + b^3 + 4 = k^3 :=
by
  sorry

end not_cube_of_sum_l478_478855


namespace problem_statement_l478_478780

variable {ℝ : Type*}

def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f x = f (-x)

def is_monotonically_decreasing (f : ℝ → ℝ) (a b : ℝ): Prop :=
  ∀ x y : ℝ, a ≤ x ∧ x ≤ y ∧ y ≤ b → f x ≥ f y

theorem problem_statement {f : ℝ → ℝ}
  (h_even : is_even_function f)
  (h_decreasing : is_monotonically_decreasing f 0 3) :
  f (-1) > f 2 ∧ f 2 > f 3 :=
by
  sorry

end problem_statement_l478_478780


namespace min_students_with_same_score_l478_478033

theorem min_students_with_same_score
    (start_points : ℕ := 6)
    (correct_points : ℕ := 4)
    (wrong_points : ℕ := 1)
    (num_questions : ℕ := 6)
    (num_students : ℕ := 51) :
    ∃ n, n ≥ 3 ∧ ∀ scores : Finset ℕ, scores.card = 25 → (∃ score ∈ scores, score ≥ 3) := 
begin
    sorry
end

end min_students_with_same_score_l478_478033


namespace probability_A_B_selected_l478_478370

noncomputable def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

def combination (n r : ℕ) : ℕ :=
  factorial n / (factorial r * factorial (n - r))

theorem probability_A_B_selected :
  let total_ways := combination 5 3 in
  let ways_A_B := combination 3 1 in
  (ways_A_B : ℚ) / total_ways = 3 / 10 :=
by
  sorry

end probability_A_B_selected_l478_478370


namespace probability_of_selecting_A_and_B_is_three_tenths_l478_478589

-- Define the set of students
def students : Finset ℕ := {0, 1, 2, 3, 4}

-- Define subsets A and B
def student_A : ℕ := 0
def student_B : ℕ := 1

-- Total number of ways to select 3 students out of 5
def total_ways : ℕ := students.card.choose 3

-- Number of ways to select 3 students including A and B
def favorable_ways : ℕ := {2, 3, 4}.card.choose 1

-- The probability
def probability_of_selecting_A_and_B : ℚ := (favorable_ways : ℚ) / (total_ways : ℚ)

theorem probability_of_selecting_A_and_B_is_three_tenths :
  probability_of_selecting_A_and_B = 3 / 10 :=
by
  sorry

end probability_of_selecting_A_and_B_is_three_tenths_l478_478589


namespace probability_both_A_B_selected_l478_478207

-- Define the problem with the necessary conditions.
def probability_AB_selected (total_students : ℕ) (select_students : ℕ) (A B : ℕ) : ℚ :=
  if A < total_students ∧ B < total_students ∧ total_students = 5 ∧ select_students = 3 then
    let total_ways := (Nat.factorial total_students) / ((Nat.factorial select_students) * (Nat.factorial (total_students - select_students)))
    let favorable_ways := (Nat.factorial (total_students - 2)) / ((Nat.factorial (select_students - 2)) * (Nat.factorial ((total_students - 2) - (select_students - 2))))
    favorable_ways / total_ways
  else 0

theorem probability_both_A_B_selected 
  (total_students : ℕ) (select_students : ℕ) (A B : ℕ) 
  (h1 : total_students = 5) (h2 : select_students = 3) 
  (h3 : A < total_students) (h4 : B < total_students) 
  : probability_AB_selected total_students select_students A B = 3 / 10 := 
by {
  -- Insert the delta and logic to prove the theorem here.
  sorry
}

end probability_both_A_B_selected_l478_478207


namespace highest_exceeds_lowest_by_l478_478008

-- Definitions based on the problem's conditions
def avg_40_innings : ℕ := 50
def num_innings : ℕ := 40
def highest_score : ℕ := 174
def avg_drop : ℕ := 2
def num_excluded_innings : ℕ := 2

-- Derived Definitions
def total_runs_40 : ℕ := avg_40_innings * num_innings
def new_avg : ℕ := avg_40_innings - avg_drop
def num_remaining_innings : ℕ := num_innings - num_excluded_innings
def total_runs_remaining : ℕ := new_avg * num_remaining_innings

-- The formal statement to be proven in Lean
theorem highest_exceeds_lowest_by :
  let lowest_score := total_runs_40 - total_runs_remaining - highest_score in
  highest_score - lowest_score = 172 :=
by
  let lowest_score := total_runs_40 - total_runs_remaining - highest_score
  sorry

end highest_exceeds_lowest_by_l478_478008


namespace probability_A_and_B_selected_l478_478383

theorem probability_A_and_B_selected 
  (students : Finset ℕ) (A B : ℕ) 
  (h_students_size : students.card = 5) 
  (h_A_in_students : A ∈ students) 
  (h_B_in_students : B ∈ students)
  : (↑(Nat.choose 3) : ℚ) / (↑(Nat.choose 5) : ℚ) = 3 / 10 := 
by sorry

end probability_A_and_B_selected_l478_478383


namespace christine_more_money_l478_478991

theorem christine_more_money (total_money christine_money siri_money : ℝ) 
  (h1 : total_money = 21)
  (h2 : christine_money = 20.5)
  (h3 : siri_money = total_money - christine_money) :
  (christine_money - siri_money = 20) :=
by 
  have siri_money_eq : siri_money = 21 - 20.5, from congr_arg (λ x, 21 - x) h2
  rw siri_money_eq at h3
  simp at h3
  have christine_more : 20.5 - 0.5 = 20 := by norm_num
  exact christine_more

end christine_more_money_l478_478991


namespace probability_A_B_l478_478445

open Nat

-- Define a function for combination
def combination (n r : Nat) : Nat :=
  factorial n / (factorial r * factorial (n - r))

-- Define the set of students and the selection conditions
def num_students : Nat := 5
def num_selected : Nat := 3

-- Define A and B as selected students
def selected_A_B : Nat := combination 3 1

-- Define the probability calculation
def probability_A_B_selected : Rat :=
  (selected_A_B : Rat) / (combination num_students num_selected : Rat)

-- The statement of the problem
theorem probability_A_B : probability_A_B_selected = 3 / 10 := by
  sorry

end probability_A_B_l478_478445


namespace probability_of_selecting_A_and_B_l478_478673

open_locale classical

noncomputable def select_probability (n k : ℕ) : ℚ :=
  (nat.choose n k)⁻¹

theorem probability_of_selecting_A_and_B :
  let total_students := 5,
      students_to_select := 3,
      remaining_students := 3 in
  select_probability total_students students_to_select * nat.choose remaining_students 1 = 3 / 10 :=
by
  sorry

end probability_of_selecting_A_and_B_l478_478673


namespace harmonic_mean_closest_integer_l478_478893

theorem harmonic_mean_closest_integer (a b : ℝ) (ha : a = 1) (hb : b = 2016) :
  abs ((2 * a * b) / (a + b) - 2) < 1 :=
by
  sorry

end harmonic_mean_closest_integer_l478_478893


namespace rationalize_denominator_l478_478859

theorem rationalize_denominator :
  let A := 12
  let B := 5
  let C := -9
  let D := 7
  let E := 17
  4 * B + 3 * D ∧
  B < D ∧
  A + B + C + D + E = 32 :=
by
  sorry

end rationalize_denominator_l478_478859


namespace probability_of_selecting_A_and_B_l478_478672

open_locale classical

noncomputable def select_probability (n k : ℕ) : ℚ :=
  (nat.choose n k)⁻¹

theorem probability_of_selecting_A_and_B :
  let total_students := 5,
      students_to_select := 3,
      remaining_students := 3 in
  select_probability total_students students_to_select * nat.choose remaining_students 1 = 3 / 10 :=
by
  sorry

end probability_of_selecting_A_and_B_l478_478672


namespace fraction_upgraded_l478_478968

theorem fraction_upgraded :
  ∀ (N U : ℕ), 24 * N = 6 * U → (U : ℚ) / (24 * N + U) = 1 / 7 :=
by
  intros N U h_eq
  sorry

end fraction_upgraded_l478_478968


namespace probability_A_B_selected_l478_478367

noncomputable def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

def combination (n r : ℕ) : ℕ :=
  factorial n / (factorial r * factorial (n - r))

theorem probability_A_B_selected :
  let total_ways := combination 5 3 in
  let ways_A_B := combination 3 1 in
  (ways_A_B : ℚ) / total_ways = 3 / 10 :=
by
  sorry

end probability_A_B_selected_l478_478367


namespace probability_A_and_B_selected_l478_478685

/-- There are 5 students including A and B. The probability of selecting both A and B 
    when 3 students are selected randomly from 5 is 3/10. -/
theorem probability_A_and_B_selected :
  (∃ (students : List String) (h : students.length = 5),
    ∃ (selected : List String) (h' : selected.length = 3),
    "A" ∈ selected ∧ "B" ∈ selected) →
  ∃ (P : ℚ), P = 3 / 10 :=
by
  sorry

end probability_A_and_B_selected_l478_478685


namespace degree_of_poly_eq_3_l478_478012

-- Define the given polynomial
def poly := 2 * x ^ 2 - 4 * x ^ 3 + 1

-- State the theorem that the degree of the polynomial is 3
theorem degree_of_poly_eq_3 : degree poly = 3 := 
  sorry

end degree_of_poly_eq_3_l478_478012


namespace probability_A_and_B_selected_l478_478389

theorem probability_A_and_B_selected 
  (students : Finset ℕ) (A B : ℕ) 
  (h_students_size : students.card = 5) 
  (h_A_in_students : A ∈ students) 
  (h_B_in_students : B ∈ students)
  : (↑(Nat.choose 3) : ℚ) / (↑(Nat.choose 5) : ℚ) = 3 / 10 := 
by sorry

end probability_A_and_B_selected_l478_478389


namespace select_3_from_5_prob_l478_478421

theorem select_3_from_5_prob {students : Finset ℕ} 
  (h : students.card = 5) 
  {A B : ℕ} 
  (hA : A ∈ students) 
  (hB : B ∈ students) 
  (H : 3 ≤ students.card) : 
  (finset.card (students.filter (λ x, x ≠ A ∧ x ≠ B)) = 3) → 
  (finset.card ((students.filter (λ x, x ≠ A ∧ x ≠ B)).subtype (λ x, True)) = 3) → 
  ((finset.card (finset.filter (λ (students : Finset ℕ) A ∈ students ∧ B ∈ students ↔ 3 = students.card) 5)) = ↑10) → 
  ∃ students', (students'.card = 3 ∧ A ∈ students' ∧ B ∈ students') → 
  ∃ (p : ℚ), p = 3 / 10 ∧ students.filter (λ x, x = A ∨ x = B) = 2 :=
  sorry

end select_3_from_5_prob_l478_478421


namespace least_number_added_l478_478069

def lcm (a b : ℕ) : ℕ := Nat.lcm a b

theorem least_number_added (x : ℕ) :
  let n := 5432
  let lcm_5_6 := lcm 5 6
  let lcm_4_3 := lcm 4 3
  let lcm_all := lcm lcm_5_6 lcm_4_3
  x = lcm_all - (n % lcm_all) → x = 28 :=
by
  intros
  sorry

end least_number_added_l478_478069


namespace trigonometric_inequality_l478_478147

noncomputable def periodic_cos : ℝ → ℤ → ℝ := λ x k, real.cos (x - 2 * real.pi * k)
noncomputable def sine_identity : ℝ → ℝ := λ x, real.sin (real.pi - x)

theorem trigonometric_inequality :
  periodic_cos 8.5 1 < real.sin 3 ∧ real.sin 3 < real.sin 1.5 :=
by
  sorry

end trigonometric_inequality_l478_478147


namespace probability_of_selecting_A_and_B_l478_478683

open_locale classical

noncomputable def select_probability (n k : ℕ) : ℚ :=
  (nat.choose n k)⁻¹

theorem probability_of_selecting_A_and_B :
  let total_students := 5,
      students_to_select := 3,
      remaining_students := 3 in
  select_probability total_students students_to_select * nat.choose remaining_students 1 = 3 / 10 :=
by
  sorry

end probability_of_selecting_A_and_B_l478_478683


namespace fraction_greater_than_decimal_l478_478144

/-- 
  Prove that the fraction 1/3 is greater than the decimal 0.333 by the amount 1/(3 * 10^3)
-/
theorem fraction_greater_than_decimal :
  (1 / 3 : ℚ) = (333 / 1000 : ℚ) + (1 / (3 * 1000) : ℚ) :=
by
  sorry

end fraction_greater_than_decimal_l478_478144


namespace zebra_catches_tiger_in_6_hours_l478_478104

-- Definitions
def zebra_speed := 55 -- kmph
def tiger_speed := 30 -- kmph
def head_start_time := 5 -- hours

-- Distance covered by Tiger in the head-start time
def head_start_distance := tiger_speed * head_start_time

-- Relative speed of zebra over tiger
def relative_speed := zebra_speed - tiger_speed

-- Time for zebra to catch up to tiger
def catch_up_time : ℝ := head_start_distance / relative_speed

-- Prove that the catch-up time is 6 hours
theorem zebra_catches_tiger_in_6_hours : catch_up_time = 6 := by
  sorry

end zebra_catches_tiger_in_6_hours_l478_478104


namespace all_numbers_odd_l478_478002

-- Definitions for the problem
def unit_square (n: ℕ) := {(i, j): ℕ × ℕ | i < n ∧ j < n}
def colored_square (n: ℕ) := {(i, j): unit_square n | i < n ∧ j < n ∧ is_colored (i, j)}
def S (i j n : ℕ) := {(x, y) : colored_square n | x ≤ i ∧ y ≤ j}

theorem all_numbers_odd (n: ℕ) (is_colored : (ℕ × ℕ) → Prop) :
  ∀ N, ∃ k, ∀ (i j : ℕ), (i, j) ∈ colored_square n → 
    (∀ m < N, (number_in_square (T^m (S i j n))) (k i j) ≡ 1 [MOD 2]) := sorry

end all_numbers_odd_l478_478002


namespace negative_a_t_exists_l478_478811

open Classical
open BigOperators

noncomputable theory

theorem negative_a_t_exists 
(k : ℝ) 
(hk1 : 0 < k) 
(hk2 : k < 1) 
(a : ℕ → ℝ) 
(h_rec : ∀ n : ℕ, 1 ≤ n → a (n + 1) ≤ (1 + k / n) * a n - 1) : 
∃ t : ℕ, a t < 0 :=
by 
  sorry

end negative_a_t_exists_l478_478811


namespace probability_A_and_B_selected_l478_478348

theorem probability_A_and_B_selected (A B : Type) (S : Finset (Type)) (h : S.card = 5) (hA : A ∈ S) (hB : B ∈ S) (T : Finset (Type)) (hT : T.card = 3) :
  (∃ T ⊆ S, A ∈ T ∧ B ∈ T ∧ T.card = 3) → 
  ∃ T ⊆ S, T.card = 3 ∧ (finset.card (T ∩ {A, B}) = 2) → 
  (finset.card (T ∩ {A, B}) = 2) = (3 / 10 : Real) := sorry

end probability_A_and_B_selected_l478_478348


namespace fraction_of_mothers_with_full_time_jobs_l478_478793

theorem fraction_of_mothers_with_full_time_jobs :
  (0.4 : ℝ) * M = 0.3 →
  (9 / 10 : ℝ) * 0.6 = 0.54 →
  1 - 0.16 = 0.84 →
  0.84 - 0.54 = 0.3 →
  M = 3 / 4 :=
by
  intros h1 h2 h3 h4
  -- The proof steps would go here.
  sorry

end fraction_of_mothers_with_full_time_jobs_l478_478793


namespace probability_A_and_B_selected_l478_478579

/-- Let there be a group of 5 students including A and B. 
    If 3 students are selected randomly from these 5, the probability of selecting both A and B is 3/10. -/
theorem probability_A_and_B_selected : 
  let students := 5 in
  let selected := 3 in
  let combinations := students.choose selected in
  let favorable_combinations := (students - 2).choose (selected - 2) in
  let probability := (favorable_combinations : ℚ) / (combinations : ℚ) in
  probability = 3 / 10 := 
by sorry

end probability_A_and_B_selected_l478_478579


namespace Cindy_walking_speed_l478_478992

noncomputable def walking_speed (total_time : ℕ) (running_speed : ℕ) (running_distance : ℚ) (walking_distance : ℚ) : ℚ := 
  let time_to_run := running_distance / running_speed
  let walking_time := total_time - (time_to_run * 60)
  walking_distance / (walking_time / 60)

theorem Cindy_walking_speed : walking_speed 40 3 0.5 0.5 = 1 := 
  sorry

end Cindy_walking_speed_l478_478992


namespace find_numbers_l478_478013

variables {x y : ℤ}

theorem find_numbers (x y : ℤ) (h1 : x - y = 11) (h2 : x^2 + y^2 = 185) (h3 : (x - y)^2 = 121) :
  (x = 13 ∧ y = 2) ∨ (x = -5 ∧ y = -16) :=
sorry

end find_numbers_l478_478013


namespace factor_quadratic_l478_478166

theorem factor_quadratic : ∀ (x : ℝ), 16*x^2 - 40*x + 25 = (4*x - 5)^2 := by
  intro x
  sorry

end factor_quadratic_l478_478166


namespace triangle_properties_l478_478105

-- Define structures for our geometric entities.
structure Point (α : Type*) := 
(x : α) (y : α) (z : α)  -- Assuming 3D space

def midpoint {α : Type*} [Field α] (A B : Point α) : Point α :=
{ 
  x := (A.x + B.x) / 2,
  y := (A.y + B.y) / 2,
  z := (A.z + B.z) / 2
}

-- Define the given conditions as Lean predicates or definitions:
variables {α : Type*} [Field α]
variables (A B C N L M : Point α)
variables (AN AC LM AB CN AL : α)

-- Main theorem to prove:
theorem triangle_properties (h1 : N = midpoint A B)
  (h2 : AN = AC)
  (h3 : AC = LM)
  (h4 : AL = AB)
  (h5 : LM = AN)
  : AB * CN^2 = AL^3 := 
sorry -- Proof goes here

end triangle_properties_l478_478105


namespace probability_A_and_B_selected_l478_478306

open Finset

noncomputable def total_ways : ℕ := (choose 5 3)

noncomputable def favorable_ways : ℕ := (choose 3 1)

noncomputable def probability (total favorable : ℕ) : ℚ := favorable / total

theorem probability_A_and_B_selected :
  probability total_ways favorable_ways = 3 / 10 :=
by
  sorry

end probability_A_and_B_selected_l478_478306


namespace find_f_half_l478_478752

/-- Definition of the function f according to the problem statement -/
def f (x : ℝ) : ℝ := (1 - x^2) / x^2

/-- The theorem to prove f(1/2) = 15 given the problem conditions -/
theorem find_f_half : f (1 - 2 * (1 / 4)) = 15 := by
  let x := 1 / 4
  have hx : 1 - 2 * x = 1 / 2 := by
    calc
    1 - 2 * (1 / 4) = 1 - 1 / 2 : by rw [mul_div_cancel' (1 : ℝ) two_ne_zero']
                    ... = 1 / 2  : by norm_num
  calc
  f (1 - 2 * x) = (1 - (1/4)^2) / (1/4)^2 : by rw f ((1 - 2 * x) = 1 / 2); exact rfl
  ...          = (1 - 1/16) / (1/16)      : by norm_num
  ...          = (15/16) / (1/16)         : by norm_num
  ...          = 15                       : by field_simp

end find_f_half_l478_478752


namespace probability_AB_selected_l478_478513

def factorial (n : Nat) : Nat := if n = 0 then 1 else n * factorial (n - 1)

def choose (n k : Nat) : Nat := factorial n / (factorial k * factorial (n - k))

theorem probability_AB_selected (h1 : 5 = 5) (h2 : 3 = 3) : (choose 3 1) / (choose 5 3) = 3 / 10 := by
  sorry

end probability_AB_selected_l478_478513


namespace probability_of_selecting_A_and_B_l478_478287

open_locale classical

-- Define the number of students
def num_students := 5

-- Define the number of students to be selected
def students_to_select := 3

def combination (n k : ℕ) : ℕ := nat.choose n k

-- Define the correct answer
def correct_probability := 3 / 10

-- Define the probability calculation
def calc_probability := (combination 3 1) / (combination 5 3)

-- Statement of the theorem to be proved
theorem probability_of_selecting_A_and_B :
  calc_probability = correct_probability :=
by sorry

end probability_of_selecting_A_and_B_l478_478287


namespace probability_A_and_B_selected_l478_478694

/-- There are 5 students including A and B. The probability of selecting both A and B 
    when 3 students are selected randomly from 5 is 3/10. -/
theorem probability_A_and_B_selected :
  (∃ (students : List String) (h : students.length = 5),
    ∃ (selected : List String) (h' : selected.length = 3),
    "A" ∈ selected ∧ "B" ∈ selected) →
  ∃ (P : ℚ), P = 3 / 10 :=
by
  sorry

end probability_A_and_B_selected_l478_478694


namespace probability_both_A_and_B_selected_l478_478408

-- The universe of students we consider
inductive Student
| A : Student
| B : Student
| C : Student
| D : Student
| E : Student

open Student

-- Function to count combinations
def combinations (n k : ℕ) : ℕ := (Nat.factorial n) / ((Nat.factorial k) * (Nat.factorial (n - k)))

-- The set of all students
def students : List Student := [A, B, C, D, E]

-- The total number of ways to select 3 students from 5
def total_selections := combinations 5 3

-- The number of ways A and B are both selected by selecting 1 more from the remaining 3 students
def selections_AB := combinations 3 1

-- The probability that both A and B are selected
def probability_AB := (selections_AB : ℚ) / (total_selections : ℚ)

-- The proof statement
theorem probability_both_A_and_B_selected :
  probability_AB = 3 / 10 :=
sorry

end probability_both_A_and_B_selected_l478_478408


namespace balance_problem_l478_478849

variable {G B Y W : ℝ}

theorem balance_problem
  (h1 : 4 * G = 8 * B)
  (h2 : 3 * Y = 7.5 * B)
  (h3 : 5 * B = 3.5 * W) :
  5 * G + 4 * Y + 3 * W = (170 / 7) * B := by
  sorry

end balance_problem_l478_478849


namespace perfectCubesBetween200and1200_l478_478769

theorem perfectCubesBetween200and1200 : ∃ n m : ℕ, (n = 6) ∧ (m = 10) ∧ (m - n + 1 = 5) ∧ (n^3 ≥ 200) ∧ (m^3 ≤ 1200) := 
by
  have h1 : 6^3 ≥ 200 := by norm_num
  have h2 : 10^3 ≤ 1200 := by norm_num
  use [6, 10]
  constructor; {refl} -- n = 6
  constructor; {refl} -- m = 10
  constructor;
  { norm_num },
  constructor; 
  { exact h1 },
  { exact h2 }
  sorry

end perfectCubesBetween200and1200_l478_478769


namespace min_expression_value_l478_478823

theorem min_expression_value (a b c : ℝ) (ha : 1 ≤ a) (hbc : b ≥ a) (hcb : c ≥ b) (hc5 : c ≤ 5) :
  (a - 1)^2 + (b / a - 1)^2 + (c / b - 1)^2 + (5 / c - 1)^2 ≥ 2 * Real.sqrt 5 - 4 * Real.sqrt (5^(1/4)) + 4 :=
sorry

end min_expression_value_l478_478823


namespace rectangle_ratio_l478_478794

variable (AB BE BC : ℝ)
variable (AEFD_square : AB = BE + AE ∧ AE = BE)
variable (ratio_holds : AB / BE = BE / BC)

theorem rectangle_ratio (h : AEFD_square ∧ ratio_holds) : AB / BC = (3 + Real.sqrt 5) / 2 :=
by sorry

end rectangle_ratio_l478_478794


namespace determine_s_plus_u_l478_478136

theorem determine_s_plus_u (p r s u : ℂ) (q t : ℂ) (h₁ : q = 5)
    (h₂ : t = -p - r) (h₃ : p + q * I + r + s * I + t + u * I = 4 * I) : s + u = -1 :=
by
  sorry

end determine_s_plus_u_l478_478136


namespace probability_A_and_B_selected_l478_478392

theorem probability_A_and_B_selected 
  (students : Finset ℕ) (A B : ℕ) 
  (h_students_size : students.card = 5) 
  (h_A_in_students : A ∈ students) 
  (h_B_in_students : B ∈ students)
  : (↑(Nat.choose 3) : ℚ) / (↑(Nat.choose 5) : ℚ) = 3 / 10 := 
by sorry

end probability_A_and_B_selected_l478_478392


namespace select_3_from_5_prob_l478_478424

theorem select_3_from_5_prob {students : Finset ℕ} 
  (h : students.card = 5) 
  {A B : ℕ} 
  (hA : A ∈ students) 
  (hB : B ∈ students) 
  (H : 3 ≤ students.card) : 
  (finset.card (students.filter (λ x, x ≠ A ∧ x ≠ B)) = 3) → 
  (finset.card ((students.filter (λ x, x ≠ A ∧ x ≠ B)).subtype (λ x, True)) = 3) → 
  ((finset.card (finset.filter (λ (students : Finset ℕ) A ∈ students ∧ B ∈ students ↔ 3 = students.card) 5)) = ↑10) → 
  ∃ students', (students'.card = 3 ∧ A ∈ students' ∧ B ∈ students') → 
  ∃ (p : ℚ), p = 3 / 10 ∧ students.filter (λ x, x = A ∨ x = B) = 2 :=
  sorry

end select_3_from_5_prob_l478_478424


namespace probability_A_and_B_selected_l478_478384

theorem probability_A_and_B_selected 
  (students : Finset ℕ) (A B : ℕ) 
  (h_students_size : students.card = 5) 
  (h_A_in_students : A ∈ students) 
  (h_B_in_students : B ∈ students)
  : (↑(Nat.choose 3) : ℚ) / (↑(Nat.choose 5) : ℚ) = 3 / 10 := 
by sorry

end probability_A_and_B_selected_l478_478384


namespace area_of_L_shape_l478_478959

theorem area_of_L_shape :
  let large_rect_length := 10
  let large_rect_width := 7
  let small_rect_length := large_rect_length - 3
  let small_rect_width := large_rect_width - 3
  let large_rect_area := large_rect_length * large_rect_width
  let small_rect_area := small_rect_length * small_rect_width
  large_rect_area - small_rect_area = 42 :=
by
  let large_rect_length := 10
  let large_rect_width := 7
  let small_rect_length := large_rect_length - 3
  let small_rect_width := large_rect_width - 3
  let large_rect_area := large_rect_length * large_rect_width
  let small_rect_area := small_rect_length * small_rect_width
  have h_large_rect_area : large_rect_area = 70 := by rfl
  have h_small_rect_area : small_rect_area = 28 := by rfl
  have h_diff_area : large_rect_area - small_rect_area = 42 := by
    rw [h_large_rect_area, h_small_rect_area]
    norm_num
    rfl
  exact h_diff_area

end area_of_L_shape_l478_478959


namespace initial_distance_proof_l478_478041

noncomputable def initial_distance_between_stones (v0 H: ℝ) : ℝ :=
  let t := H / v0 in
  H * Real.sqrt 2

theorem initial_distance_proof :
  ∀ (v0 H: ℝ), H = 50 → initial_distance_between_stones v0 H = 50 * Real.sqrt 2 :=
by
  intros v0 H hH
  rw [initial_distance_between_stones]
  rw [hH]
  sorry

end initial_distance_proof_l478_478041


namespace probability_both_A_B_selected_l478_478210

-- Define the problem with the necessary conditions.
def probability_AB_selected (total_students : ℕ) (select_students : ℕ) (A B : ℕ) : ℚ :=
  if A < total_students ∧ B < total_students ∧ total_students = 5 ∧ select_students = 3 then
    let total_ways := (Nat.factorial total_students) / ((Nat.factorial select_students) * (Nat.factorial (total_students - select_students)))
    let favorable_ways := (Nat.factorial (total_students - 2)) / ((Nat.factorial (select_students - 2)) * (Nat.factorial ((total_students - 2) - (select_students - 2))))
    favorable_ways / total_ways
  else 0

theorem probability_both_A_B_selected 
  (total_students : ℕ) (select_students : ℕ) (A B : ℕ) 
  (h1 : total_students = 5) (h2 : select_students = 3) 
  (h3 : A < total_students) (h4 : B < total_students) 
  : probability_AB_selected total_students select_students A B = 3 / 10 := 
by {
  -- Insert the delta and logic to prove the theorem here.
  sorry
}

end probability_both_A_B_selected_l478_478210


namespace evaluate_powers_of_i_l478_478159

/-- Defines the imaginary unit i and its properties. -/
def i : ℂ := Complex.I

/-- i squared is -1. -/
lemma i_squared : i^2 = -1 :=
by simp [i, Complex.I_sq]

/-- i^4 is 1. -/
lemma i_fourth_power: i^4 = 1 :=
by simp [i, Complex.I_pow]

/-- The main theorem proving the given expression equals i. -/
theorem evaluate_powers_of_i :
  i^13 + i^18 + i^23 + i^28 + i^33 = i :=
by sorry

end evaluate_powers_of_i_l478_478159


namespace probability_both_A_and_B_selected_l478_478404

-- The universe of students we consider
inductive Student
| A : Student
| B : Student
| C : Student
| D : Student
| E : Student

open Student

-- Function to count combinations
def combinations (n k : ℕ) : ℕ := (Nat.factorial n) / ((Nat.factorial k) * (Nat.factorial (n - k)))

-- The set of all students
def students : List Student := [A, B, C, D, E]

-- The total number of ways to select 3 students from 5
def total_selections := combinations 5 3

-- The number of ways A and B are both selected by selecting 1 more from the remaining 3 students
def selections_AB := combinations 3 1

-- The probability that both A and B are selected
def probability_AB := (selections_AB : ℚ) / (total_selections : ℚ)

-- The proof statement
theorem probability_both_A_and_B_selected :
  probability_AB = 3 / 10 :=
sorry

end probability_both_A_and_B_selected_l478_478404


namespace probability_both_A_B_selected_l478_478225

-- Define the problem with the necessary conditions.
def probability_AB_selected (total_students : ℕ) (select_students : ℕ) (A B : ℕ) : ℚ :=
  if A < total_students ∧ B < total_students ∧ total_students = 5 ∧ select_students = 3 then
    let total_ways := (Nat.factorial total_students) / ((Nat.factorial select_students) * (Nat.factorial (total_students - select_students)))
    let favorable_ways := (Nat.factorial (total_students - 2)) / ((Nat.factorial (select_students - 2)) * (Nat.factorial ((total_students - 2) - (select_students - 2))))
    favorable_ways / total_ways
  else 0

theorem probability_both_A_B_selected 
  (total_students : ℕ) (select_students : ℕ) (A B : ℕ) 
  (h1 : total_students = 5) (h2 : select_students = 3) 
  (h3 : A < total_students) (h4 : B < total_students) 
  : probability_AB_selected total_students select_students A B = 3 / 10 := 
by {
  -- Insert the delta and logic to prove the theorem here.
  sorry
}

end probability_both_A_B_selected_l478_478225


namespace probability_both_A_B_selected_l478_478222

-- Define the problem with the necessary conditions.
def probability_AB_selected (total_students : ℕ) (select_students : ℕ) (A B : ℕ) : ℚ :=
  if A < total_students ∧ B < total_students ∧ total_students = 5 ∧ select_students = 3 then
    let total_ways := (Nat.factorial total_students) / ((Nat.factorial select_students) * (Nat.factorial (total_students - select_students)))
    let favorable_ways := (Nat.factorial (total_students - 2)) / ((Nat.factorial (select_students - 2)) * (Nat.factorial ((total_students - 2) - (select_students - 2))))
    favorable_ways / total_ways
  else 0

theorem probability_both_A_B_selected 
  (total_students : ℕ) (select_students : ℕ) (A B : ℕ) 
  (h1 : total_students = 5) (h2 : select_students = 3) 
  (h3 : A < total_students) (h4 : B < total_students) 
  : probability_AB_selected total_students select_students A B = 3 / 10 := 
by {
  -- Insert the delta and logic to prove the theorem here.
  sorry
}

end probability_both_A_B_selected_l478_478222


namespace sum_of_integers_from_neg20_to_10_l478_478125

-- Given conditions
def a : Int := -20
def l : Int := 10

-- Prove statement
theorem sum_of_integers_from_neg20_to_10 : 
  (∑ k in Finset.range (l - a + 1), a + k : Int) = -155 :=
by
  sorry

end sum_of_integers_from_neg20_to_10_l478_478125


namespace stability_comparison_probability_calculation_l478_478024

-- Define the scores for class A and B
def scoresA : List ℕ := [5, 8, 9, 9, 9]
def scoresB : List ℕ := [6, 7, 8, 9, 10]

-- Define helper functions to calculate mean and variance for a list
def mean (scores : List ℕ) : ℝ :=
  (scores.foldl (λ acc x => acc + x) 0 : ℝ) / scores.length

def variance (scores : List ℕ) : ℝ :=
  let μ := mean scores
  (scores.foldl (λ acc x => acc + (x : ℝ - μ) ^ 2) 0) / scores.length

def populationMeanB := mean scoresB
def populationVarianceA := variance scoresA
def populationVarianceB := variance scoresB

-- Define a function to calculate the absolute difference between the sample mean and population mean
def sampleMeanDifference (sample : List ℕ) : ℝ :=
  abs (mean sample - populationMeanB)

-- Define the possible samples of size 2 from scoresB
def samplesB : List (List ℕ) := (scoresB.combinations 2).map id

-- Calculate the probability that the absolute difference is not less than 1
def probabilityOfDifferenceNotLessThan1 : ℝ :=
  let satisfyingSamples := samplesB.filter (λ sample => sampleMeanDifference sample ≥ 1)
  satisfyingSamples.length / samplesB.length

theorem stability_comparison : populationVarianceB < populationVarianceA := by
  sorry

theorem probability_calculation : probabilityOfDifferenceNotLessThan1 = 2 / 5 := by
  sorry

end stability_comparison_probability_calculation_l478_478024


namespace probability_of_selecting_A_and_B_l478_478257

-- Define the combination function
def combination (n r : ℕ) : ℕ := Nat.factorial n / (Nat.factorial r * Nat.factorial (n - r))

-- Define the conditions
def students : ℕ := 5
def to_select : ℕ := 3

-- Define the favorable outcomes where A and B are selected
def favorable_outcomes : ℕ := combination 3 1

-- Define the total outcomes of selecting 3 out of 5 students
def total_outcomes : ℕ := combination 5 3

-- Define the probability calculation
def probability_of_A_and_B : ℚ := favorable_outcomes / total_outcomes

-- Proposition to prove
theorem probability_of_selecting_A_and_B : probability_of_A_and_B = 3 / 10 :=
by
  sorry

end probability_of_selecting_A_and_B_l478_478257


namespace find_DC_l478_478951

noncomputable def given_conditions (AB BC BK KC : ℝ) : Prop :=
  AB < BC ∧ BK < AB ∧ KC = (√7 - 1) ∧ (cos (angle_of_cos ((√7 + 1) / 4)).cos) = ((√7 + 1) / 4) ∧
  perimeter <| triangle B K C = 2 * (√7) + 4

theorem find_DC (AB BC BK DC KC : ℝ) : given_conditions AB BC BK KC → DC = BC :=
by
  intro h
  sorry

end find_DC_l478_478951


namespace probability_of_selecting_A_and_B_is_three_tenths_l478_478608

-- Define the set of students
def students : Finset ℕ := {0, 1, 2, 3, 4}

-- Define subsets A and B
def student_A : ℕ := 0
def student_B : ℕ := 1

-- Total number of ways to select 3 students out of 5
def total_ways : ℕ := students.card.choose 3

-- Number of ways to select 3 students including A and B
def favorable_ways : ℕ := {2, 3, 4}.card.choose 1

-- The probability
def probability_of_selecting_A_and_B : ℚ := (favorable_ways : ℚ) / (total_ways : ℚ)

theorem probability_of_selecting_A_and_B_is_three_tenths :
  probability_of_selecting_A_and_B = 3 / 10 :=
by
  sorry

end probability_of_selecting_A_and_B_is_three_tenths_l478_478608


namespace probability_A_and_B_selected_l478_478578

/-- Let there be a group of 5 students including A and B. 
    If 3 students are selected randomly from these 5, the probability of selecting both A and B is 3/10. -/
theorem probability_A_and_B_selected : 
  let students := 5 in
  let selected := 3 in
  let combinations := students.choose selected in
  let favorable_combinations := (students - 2).choose (selected - 2) in
  let probability := (favorable_combinations : ℚ) / (combinations : ℚ) in
  probability = 3 / 10 := 
by sorry

end probability_A_and_B_selected_l478_478578


namespace factorization_identity_l478_478173

theorem factorization_identity (x : ℝ) : 
  16 * x^2 - 40 * x + 25 = (4 * x - 5)^2 :=
  sorry

end factorization_identity_l478_478173


namespace probability_A_and_B_selected_l478_478242

theorem probability_A_and_B_selected (students : Finset ℕ) (A B : ℕ) (h : A ∈ students ∧ B ∈ students ∧ students.card = 5) : 
  (students.choose 3).count (λ s, {A, B} ⊆ s) / (students.choose 3).card = 3 / 10 := by
  sorry

end probability_A_and_B_selected_l478_478242


namespace probability_of_selecting_A_and_B_l478_478261

-- Define the combination function
def combination (n r : ℕ) : ℕ := Nat.factorial n / (Nat.factorial r * Nat.factorial (n - r))

-- Define the conditions
def students : ℕ := 5
def to_select : ℕ := 3

-- Define the favorable outcomes where A and B are selected
def favorable_outcomes : ℕ := combination 3 1

-- Define the total outcomes of selecting 3 out of 5 students
def total_outcomes : ℕ := combination 5 3

-- Define the probability calculation
def probability_of_A_and_B : ℚ := favorable_outcomes / total_outcomes

-- Proposition to prove
theorem probability_of_selecting_A_and_B : probability_of_A_and_B = 3 / 10 :=
by
  sorry

end probability_of_selecting_A_and_B_l478_478261


namespace probability_AB_selected_l478_478501

def factorial (n : Nat) : Nat := if n = 0 then 1 else n * factorial (n - 1)

def choose (n k : Nat) : Nat := factorial n / (factorial k * factorial (n - k))

theorem probability_AB_selected (h1 : 5 = 5) (h2 : 3 = 3) : (choose 3 1) / (choose 5 3) = 3 / 10 := by
  sorry

end probability_AB_selected_l478_478501


namespace probability_A_and_B_selected_l478_478520

theorem probability_A_and_B_selected :
  let n := 5
  let r := 3
  let total_ways := Nat.choose n r
  let favorable_ways := Nat.choose 3 1
  let probability := favorable_ways / total_ways
  probability = (3 / 10) := 
by
  sorry

end probability_A_and_B_selected_l478_478520


namespace probability_A_and_B_selected_l478_478336

theorem probability_A_and_B_selected (A B : Type) (S : Finset (Type)) (h : S.card = 5) (hA : A ∈ S) (hB : B ∈ S) (T : Finset (Type)) (hT : T.card = 3) :
  (∃ T ⊆ S, A ∈ T ∧ B ∈ T ∧ T.card = 3) → 
  ∃ T ⊆ S, T.card = 3 ∧ (finset.card (T ∩ {A, B}) = 2) → 
  (finset.card (T ∩ {A, B}) = 2) = (3 / 10 : Real) := sorry

end probability_A_and_B_selected_l478_478336


namespace area_of_triangle_ABC_is_414_67_l478_478788

def semi_perimeter (a b c : ℝ) : ℝ := (a + b + c) / 2

def heron_area (a b c : ℝ) : ℝ :=
  let s := semi_perimeter a b c
  real.sqrt (s * (s - a) * (s - b) * (s - c))

def triangle_area_ABC : ℝ :=
  heron_area 30 30 50

theorem area_of_triangle_ABC_is_414_67 : 
  abs (triangle_area_ABC - 414.67) < 1e-2 := 
sorry

end area_of_triangle_ABC_is_414_67_l478_478788


namespace babblian_word_count_l478_478851

theorem babblian_word_count (n : ℕ) (h1 : n = 6) : ∃ m, m = 258 := by
  sorry

end babblian_word_count_l478_478851


namespace right_triangle_leg_length_l478_478095

theorem right_triangle_leg_length (a b c : ℕ) (h₁ : a = 8) (h₂ : c = 17) (h₃ : a^2 + b^2 = c^2) : b = 15 := 
by
  sorry

end right_triangle_leg_length_l478_478095


namespace prob_select_A_and_B_l478_478613

theorem prob_select_A_and_B : 
  let total_students := 5
  let to_select := 3
  let favorable_outcomes := Nat.div_fact (Nat.factorial 3) (Nat.factorial 1)
  let total_outcomes := Nat.div_fact (Nat.factorial 5) (Nat.factorial 3 * Nat.factorial (5 - 3))
  (favorable_outcomes : ℚ) / total_outcomes = 3 / 10 :=
by
  sorry

end prob_select_A_and_B_l478_478613


namespace hair_color_distribution_l478_478791

noncomputable def kids_by_hair_color := (red: Nat, blonde: Nat, black: Nat, brown: Nat)

theorem hair_color_distribution : (let ratio := (3, 6, 7, 4)
                                   let kids_per_part := 3
                                   let red_kids := 9
                                   let blonde_kids := 18
                                   let black_kids := 9
                                   let brown_kids := 12
                                   let total_kids := 60

                                   ratio.1 = 3 ∧ 
                                   ratio.2 = 6 ∧ 
                                   ratio.3 = 7 ∧ 
                                   ratio.4 = 4 ∧ 
                                   kids_per_part = red_kids / ratio.1 ∧
                                   kids_per_part = blonde_kids / ratio.2 ∧
                                   kids_per_part = black_kids / ratio.3 ∧
                                   kids_per_part = brown_kids / ratio.4 ∧
                                   total_kids = ratio.1 * kids_per_part + 
                                                ratio.2 * kids_per_part + 
                                                ratio.3 * kids_per_part + 
                                                ratio.4 * kids_per_part
                                   ) := by 
  sorry

end hair_color_distribution_l478_478791


namespace disproved_option_a_disproved_option_b_disproved_option_c_proved_option_d_l478_478934

theorem disproved_option_a : ¬ ∀ m n : ℤ, abs m = abs n → m = n := 
by 
  intro h
  have h1 : abs (-3) = abs 3 := by norm_num
  have h2 : h (-3) 3 h1
  contradiction

theorem disproved_option_b : ¬ ∀ m n : ℤ, m > n → abs m > abs n := 
by 
  intro h
  have h1 : 1 > -3 := by norm_num
  have h2 : abs 1 = 1 := by norm_num
  have h3 : abs (-3) = 3 := by norm_num
  have h4 : h 1 (-3) h1
  contradiction

theorem disproved_option_c : ¬ ∀ m n : ℤ, abs m > abs n → m > n :=
by 
  intro h
  have h1 : abs (-3) > abs 1 := by norm_num
  have h2 : (-3) < 1 := by norm_num
  exact not_lt_of_gt h2 (h (-3) 1 h1)

theorem proved_option_d : ∀ m n : ℤ, m < n → n < 0 → abs m > abs n := 
by 
  intro m n hmn hn0
  simp only [abs_lt, int.lt_neg, int.abs]
  have h1 : -n < -m := by exact neg_lt_neg hmn
  exact_mod_cast h1

end disproved_option_a_disproved_option_b_disproved_option_c_proved_option_d_l478_478934


namespace Felix_distance_proof_l478_478884

def average_speed : ℕ := 66
def twice_speed : ℕ := 2 * average_speed
def driving_hours : ℕ := 4
def distance_covered : ℕ := twice_speed * driving_hours

theorem Felix_distance_proof : distance_covered = 528 := by
  sorry

end Felix_distance_proof_l478_478884


namespace exists_block_with_five_primes_l478_478909

theorem exists_block_with_five_primes :
  (∃ n : ℕ, ∀ i : ℕ, 2 ≤ i → i ≤ 1001 → ¬ prime (1001! + i)) →
  (∃ m : ℕ, ∃ k : ℕ, k ≤ 5 ∧ ∀ j : ℕ, j < 1000 → prime (m + j) ↔ j < 5) :=
by sorry

end exists_block_with_five_primes_l478_478909


namespace probability_both_A_and_B_selected_l478_478406

-- The universe of students we consider
inductive Student
| A : Student
| B : Student
| C : Student
| D : Student
| E : Student

open Student

-- Function to count combinations
def combinations (n k : ℕ) : ℕ := (Nat.factorial n) / ((Nat.factorial k) * (Nat.factorial (n - k)))

-- The set of all students
def students : List Student := [A, B, C, D, E]

-- The total number of ways to select 3 students from 5
def total_selections := combinations 5 3

-- The number of ways A and B are both selected by selecting 1 more from the remaining 3 students
def selections_AB := combinations 3 1

-- The probability that both A and B are selected
def probability_AB := (selections_AB : ℚ) / (total_selections : ℚ)

-- The proof statement
theorem probability_both_A_and_B_selected :
  probability_AB = 3 / 10 :=
sorry

end probability_both_A_and_B_selected_l478_478406


namespace sum_distances_iff_ellipse_l478_478132

-- Definitions derived from conditions
def sum_distances_constant (M F1 F2 : ℝ) := 
  ∃ k : ℝ, k > 0 ∧ (dist M F1 + dist M F2 = k)

def is_ellipse (M : ℝ) :=
  ∃ a b : ℝ, 0 < b ∧ (a > b) ∧ (dist M (0, b) + dist M (0, -b) = 2*a)

-- The theorem statement as per condition and question
theorem sum_distances_iff_ellipse 
  (M : ℝ) : 
  (∀ F1 F2 : ℝ, sum_distances_constant M F1 F2) ↔ (is_ellipse M) :=
begin
  sorry
end

end sum_distances_iff_ellipse_l478_478132


namespace total_value_of_goods_l478_478103

theorem total_value_of_goods (V : ℝ) (tax_paid : ℝ) (tax_exemption : ℝ) (tax_rate : ℝ) :
  tax_exemption = 600 → tax_rate = 0.11 → tax_paid = 123.2 → 0.11 * (V - 600) = tax_paid → V = 1720 :=
by
  sorry

end total_value_of_goods_l478_478103


namespace prob_select_A_and_B_l478_478641

theorem prob_select_A_and_B (students : Finset ℕ) (A B : ℕ) (hA : A ∈ students) (hB : B ∈ students) (h_card : students.card = 5) :
  (students.choose 3).count {comb | A ∈ comb ∧ B ∈ comb} / (students.choose 3).card = 3 / 10 := sorry

end prob_select_A_and_B_l478_478641


namespace percentage_girls_l478_478907

theorem percentage_girls (initial_boys : ℕ) (initial_girls : ℕ) (added_boys : ℕ) :
  initial_boys = 11 → initial_girls = 13 → added_boys = 1 → 
  100 * initial_girls / (initial_boys + added_boys + initial_girls) = 52 :=
by
  intros h_boys h_girls h_added
  sorry

end percentage_girls_l478_478907


namespace prob_select_A_and_B_l478_478625

theorem prob_select_A_and_B : 
  let total_students := 5
  let to_select := 3
  let favorable_outcomes := Nat.div_fact (Nat.factorial 3) (Nat.factorial 1)
  let total_outcomes := Nat.div_fact (Nat.factorial 5) (Nat.factorial 3 * Nat.factorial (5 - 3))
  (favorable_outcomes : ℚ) / total_outcomes = 3 / 10 :=
by
  sorry

end prob_select_A_and_B_l478_478625


namespace probability_both_A_B_selected_l478_478219

-- Define the problem with the necessary conditions.
def probability_AB_selected (total_students : ℕ) (select_students : ℕ) (A B : ℕ) : ℚ :=
  if A < total_students ∧ B < total_students ∧ total_students = 5 ∧ select_students = 3 then
    let total_ways := (Nat.factorial total_students) / ((Nat.factorial select_students) * (Nat.factorial (total_students - select_students)))
    let favorable_ways := (Nat.factorial (total_students - 2)) / ((Nat.factorial (select_students - 2)) * (Nat.factorial ((total_students - 2) - (select_students - 2))))
    favorable_ways / total_ways
  else 0

theorem probability_both_A_B_selected 
  (total_students : ℕ) (select_students : ℕ) (A B : ℕ) 
  (h1 : total_students = 5) (h2 : select_students = 3) 
  (h3 : A < total_students) (h4 : B < total_students) 
  : probability_AB_selected total_students select_students A B = 3 / 10 := 
by {
  -- Insert the delta and logic to prove the theorem here.
  sorry
}

end probability_both_A_B_selected_l478_478219


namespace find_n_l478_478926

theorem find_n : ∃ n : ℤ, 0 ≤ n ∧ n < 25 ∧ (-250 ≡ n [MOD 23]) ∧ n = 3 := by
  use 3
  -- Proof omitted
  sorry

end find_n_l478_478926


namespace travel_distance_l478_478072

theorem travel_distance (S : ℝ) (h1 : S > 0) (delay : ℝ) (extra_distance : ℝ) (h2 : delay = 2) (h3 : extra_distance = 50) :
  let D := 4 * S in
  let malfunction_time_new := (D - (S + extra_distance)) / (3/5 * S) + 1 + extra_distance / S in
  malfunction_time_new = 2 - (2/3) → D = 200 :=
sorry

end travel_distance_l478_478072


namespace prob_select_A_and_B_l478_478637

theorem prob_select_A_and_B (students : Finset ℕ) (A B : ℕ) (hA : A ∈ students) (hB : B ∈ students) (h_card : students.card = 5) :
  (students.choose 3).count {comb | A ∈ comb ∧ B ∈ comb} / (students.choose 3).card = 3 / 10 := sorry

end prob_select_A_and_B_l478_478637


namespace probability_of_selecting_A_and_B_l478_478684

open_locale classical

noncomputable def select_probability (n k : ℕ) : ℚ :=
  (nat.choose n k)⁻¹

theorem probability_of_selecting_A_and_B :
  let total_students := 5,
      students_to_select := 3,
      remaining_students := 3 in
  select_probability total_students students_to_select * nat.choose remaining_students 1 = 3 / 10 :=
by
  sorry

end probability_of_selecting_A_and_B_l478_478684


namespace solution_set_l478_478738

variable {ℝ : Type*}

noncomputable def f (x : ℝ) : ℝ := sorry  -- f is a function ℝ -> ℝ
noncomputable def f' (x : ℝ) : ℝ := sorry -- f' is the derivative of f

axiom deriv_f (x : ℝ) : f'(x) = (deriv f) x
axiom condition1 (x : ℝ) : f'(x) + 2 * f(x) > 0
axiom condition2 : f(-1) = 0

theorem solution_set (x : ℝ) : f x < 0 ↔ x < -1 := by
  sorry

end solution_set_l478_478738


namespace sqrt_nine_over_four_l478_478027

theorem sqrt_nine_over_four (x : ℝ) : x = 3 / 2 ∨ x = - (3 / 2) ↔ x * x = 9 / 4 :=
by {
  sorry
}

end sqrt_nine_over_four_l478_478027


namespace probability_A_and_B_selected_l478_478586

/-- Let there be a group of 5 students including A and B. 
    If 3 students are selected randomly from these 5, the probability of selecting both A and B is 3/10. -/
theorem probability_A_and_B_selected : 
  let students := 5 in
  let selected := 3 in
  let combinations := students.choose selected in
  let favorable_combinations := (students - 2).choose (selected - 2) in
  let probability := (favorable_combinations : ℚ) / (combinations : ℚ) in
  probability = 3 / 10 := 
by sorry

end probability_A_and_B_selected_l478_478586


namespace find_possible_A_values_l478_478045

def is_digit (n : ℕ) := n ≥ 0 ∧ n ≤ 9
def is_three_digit_number (A : ℕ) := A ≥ 100 ∧ A < 1000

def conditions_holds (A B: ℕ) : Prop :=
  -- A and B are both three-digit numbers.
  is_three_digit_number A ∧ is_three_digit_number B ∧
  -- Digits of a three-digit number: A = 100*a + 10*b + c
  -- No digit in B matches the digit in A at the same place.
  ∃ (a b c : ℕ), is_digit a ∧ is_digit b ∧ is_digit c ∧
  A = 100 * a + 10 * b + c ∧
  ((B = 100 * b + 10 * c + a ∨ B = 100 * c + 10 * a + b) ∧
  (100 * a + 10 * b + c - (100 * b + 10 * c + a) % 100 =  sq ∧
  ∃ sq: ℕ, sq^2 < 100)

theorem find_possible_A_values (A B : ℕ) :
  conditions_holds A B → 
  ∃ a b c : ℕ, is_digit a ∧ is_digit b ∧ is_digit c ∧ 
  A = 100 * a + 10 * b + c ∧
  ((A = 218 ∨ A = 329 ∨ A = 213 ∨ A = 324 ∨ A = 435 ∨ A = 546 ∨
  A = 657 ∨ A = 768 ∨ A = 879 ∨ A = 706 ∨ A = 817 ∨ A = 928 ∨
  A = 201 ∨ A = 312 ∨ A = 423 ∨ A = 534 ∨ A = 645 ∨ A = 756 ∨
  A = 867 ∨ A = 978) ∨ false) :=
sorry

end find_possible_A_values_l478_478045


namespace probability_A_and_B_selected_l478_478346

theorem probability_A_and_B_selected (A B : Type) (S : Finset (Type)) (h : S.card = 5) (hA : A ∈ S) (hB : B ∈ S) (T : Finset (Type)) (hT : T.card = 3) :
  (∃ T ⊆ S, A ∈ T ∧ B ∈ T ∧ T.card = 3) → 
  ∃ T ⊆ S, T.card = 3 ∧ (finset.card (T ∩ {A, B}) = 2) → 
  (finset.card (T ∩ {A, B}) = 2) = (3 / 10 : Real) := sorry

end probability_A_and_B_selected_l478_478346


namespace probability_AB_selected_l478_478504

def factorial (n : Nat) : Nat := if n = 0 then 1 else n * factorial (n - 1)

def choose (n k : Nat) : Nat := factorial n / (factorial k * factorial (n - k))

theorem probability_AB_selected (h1 : 5 = 5) (h2 : 3 = 3) : (choose 3 1) / (choose 5 3) = 3 / 10 := by
  sorry

end probability_AB_selected_l478_478504


namespace prob_select_A_and_B_l478_478632

theorem prob_select_A_and_B : 
  let total_students := 5
  let to_select := 3
  let favorable_outcomes := Nat.div_fact (Nat.factorial 3) (Nat.factorial 1)
  let total_outcomes := Nat.div_fact (Nat.factorial 5) (Nat.factorial 3 * Nat.factorial (5 - 3))
  (favorable_outcomes : ℚ) / total_outcomes = 3 / 10 :=
by
  sorry

end prob_select_A_and_B_l478_478632


namespace ratio_of_segments_similarity_of_triangles_l478_478789

variables {A B C Q M N P : Type*} [Euclidean_Space A B C]

-- Conditions
def IsoscelesTriangle (A B C : Type*) : Prop := AB = AC

def lies_on (Q : Type*) (BC : Line) : Prop := Q ∈ BC

def parallel_lines (Q CA AB : Type*) : Prop := ∃ M, parallel (line_through Q CA) (line_through M AB)

def parallel_lines (Q BA AC : Type*) : Prop := ∃ N, parallel (line_through Q BA) (line_through N AC)

def reflection (Q MN : Type*) : Type* := ∃ P, P = reflect Q MN

-- Questions to Prove
theorem ratio_of_segments (ABC_isosceles : IsoscelesTriangle A B C)
  (Q_on_BC : lies_on Q BC) (Q_parallel_CA : parallel_lines Q CA AB) 
  (Q_parallel_BA : parallel_lines Q BA AC) (P_reflection : reflection Q MN):
  (PB / PC = BQ / QC) :=
sorry

theorem similarity_of_triangles (ABC_isosceles : IsoscelesTriangle A B C)
  (Q_on_BC : lies_on Q BC) (Q_parallel_CA : parallel_lines Q CA AB) 
  (Q_parallel_BA : parallel_lines Q BA AC) (P_reflection : reflection Q MN):
  (triangle.similar PBC ANM) :=
sorry

end ratio_of_segments_similarity_of_triangles_l478_478789


namespace smallest_lambda_l478_478197

theorem smallest_lambda :
  ∃ λ : ℝ, (∀ (a1 a2 a3 b1 b2 b3 : ℝ),
    (a1 ∈ set.Icc 0 (1/2)) ∧ (a2 ∈ set.Icc 0 (1/2)) ∧ (a3 ∈ set.Icc 0 (1/2)) ∧
    (b1 > 0) ∧ (b2 > 0) ∧ (b3 > 0) ∧ (a1 + a2 + a3 = 1) ∧ (b1 + b2 + b3 = 1) →
    b1 * b2 * b3 ≤ λ * (a1 * b1 + a2 * b2 + a3 * b3)) ∧ (λ = 1 / 8) :=
begin
  existsi (1 / 8),
  split,
  { intros a1 a2 a3 b1 b2 b3 h_conditions,
    sorry },
  { refl }
end

end smallest_lambda_l478_478197


namespace probability_both_A_and_B_selected_l478_478417

-- The universe of students we consider
inductive Student
| A : Student
| B : Student
| C : Student
| D : Student
| E : Student

open Student

-- Function to count combinations
def combinations (n k : ℕ) : ℕ := (Nat.factorial n) / ((Nat.factorial k) * (Nat.factorial (n - k)))

-- The set of all students
def students : List Student := [A, B, C, D, E]

-- The total number of ways to select 3 students from 5
def total_selections := combinations 5 3

-- The number of ways A and B are both selected by selecting 1 more from the remaining 3 students
def selections_AB := combinations 3 1

-- The probability that both A and B are selected
def probability_AB := (selections_AB : ℚ) / (total_selections : ℚ)

-- The proof statement
theorem probability_both_A_and_B_selected :
  probability_AB = 3 / 10 :=
sorry

end probability_both_A_and_B_selected_l478_478417


namespace factorization_of_polynomial_l478_478183

theorem factorization_of_polynomial (x : ℝ) : 16 * x ^ 2 - 40 * x + 25 = (4 * x - 5) ^ 2 := by 
  sorry

end factorization_of_polynomial_l478_478183


namespace amusement_park_admission_l478_478006

theorem amusement_park_admission :
  let admission_fee_child := 1.5
  let admission_fee_adult := 4
  let total_fees_collected := 810
  let number_of_children := 180
  let total_fee_from_children := admission_fee_child * number_of_children
  let total_fee_from_adults := total_fees_collected - total_fee_from_children
  let number_of_adults := total_fee_from_adults / admission_fee_adult
  number_of_children + number_of_adults = 315 :=
by
  let admission_fee_child := 1.5
  let admission_fee_adult := 4
  let total_fees_collected := 810
  let number_of_children := 180
  let total_fee_from_children := admission_fee_child * number_of_children
  let total_fee_from_adults := total_fees_collected - total_fee_from_children
  let number_of_adults := total_fee_from_adults / admission_fee_adult
  have h1 : total_fee_from_children = 270 := by norm_num
  have h2 : total_fee_from_adults = 540 := by norm_num
  have h3 : number_of_adults = 135 := by norm_num
  have h4 : number_of_children + number_of_adults = 315 := by norm_num
  exact h4

end amusement_park_admission_l478_478006


namespace probability_of_selecting_A_and_B_l478_478267

-- Define the combination function
def combination (n r : ℕ) : ℕ := Nat.factorial n / (Nat.factorial r * Nat.factorial (n - r))

-- Define the conditions
def students : ℕ := 5
def to_select : ℕ := 3

-- Define the favorable outcomes where A and B are selected
def favorable_outcomes : ℕ := combination 3 1

-- Define the total outcomes of selecting 3 out of 5 students
def total_outcomes : ℕ := combination 5 3

-- Define the probability calculation
def probability_of_A_and_B : ℚ := favorable_outcomes / total_outcomes

-- Proposition to prove
theorem probability_of_selecting_A_and_B : probability_of_A_and_B = 3 / 10 :=
by
  sorry

end probability_of_selecting_A_and_B_l478_478267


namespace probability_A_and_B_selected_l478_478540

theorem probability_A_and_B_selected :
  let n := 5
  let r := 3
  let total_ways := Nat.choose n r
  let favorable_ways := Nat.choose 3 1
  let probability := favorable_ways / total_ways
  probability = (3 / 10) := 
by
  sorry

end probability_A_and_B_selected_l478_478540


namespace probability_A_and_B_selected_l478_478530

theorem probability_A_and_B_selected :
  let n := 5
  let r := 3
  let total_ways := Nat.choose n r
  let favorable_ways := Nat.choose 3 1
  let probability := favorable_ways / total_ways
  probability = (3 / 10) := 
by
  sorry

end probability_A_and_B_selected_l478_478530


namespace julie_monthly_salary_l478_478810

def hourly_rate : ℕ := 5
def hours_per_day : ℕ := 8
def days_per_week : ℕ := 6
def missed_days : ℕ := 1
def weeks_per_month : ℕ := 4

theorem julie_monthly_salary :
  (hourly_rate * hours_per_day * (days_per_week - missed_days) * weeks_per_month) = 920 :=
by
  sorry

end julie_monthly_salary_l478_478810


namespace probability_A_and_B_selected_l478_478303

open Finset

noncomputable def total_ways : ℕ := (choose 5 3)

noncomputable def favorable_ways : ℕ := (choose 3 1)

noncomputable def probability (total favorable : ℕ) : ℚ := favorable / total

theorem probability_A_and_B_selected :
  probability total_ways favorable_ways = 3 / 10 :=
by
  sorry

end probability_A_and_B_selected_l478_478303


namespace probability_A_B_selected_l478_478368

noncomputable def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

def combination (n r : ℕ) : ℕ :=
  factorial n / (factorial r * factorial (n - r))

theorem probability_A_B_selected :
  let total_ways := combination 5 3 in
  let ways_A_B := combination 3 1 in
  (ways_A_B : ℚ) / total_ways = 3 / 10 :=
by
  sorry

end probability_A_B_selected_l478_478368


namespace factorization_eq_l478_478188

theorem factorization_eq (x : ℝ) : 
  -3 * x^3 + 12 * x^2 - 12 * x = -3 * x * (x - 2)^2 :=
by
  sorry

end factorization_eq_l478_478188


namespace problem1_problem2_l478_478762

-- Vector definitions
def OA : ℝ × ℝ := (3, -4)
def OB : ℝ × ℝ := (6, -3)
def OC (m : ℝ) : ℝ × ℝ := (5 - m, -(3 + m))

-- Problem (1): Non-collinearity condition for m
theorem problem1 (m : ℝ) : ¬(m = 1 / 2) :=
sorry

-- Problem (2): Right-angled triangle condition at angle A
theorem problem2 (m : ℝ) (h : ∀ (a b c : ℝ × ℝ), a ≠ b → b ≠ c → c ≠ a → a ≠ (0,0) → b ≠ (0,0) → c ≠ (0,0)
  → let v1 := (b.1 - a.1, b.2 - a.2),
         v2 := (c.1 - a.1, c.2 - a.2)
    in v1.1 * v2.1 + v1.2 * v2.2 = 0) : m = 7 / 4 := 
sorry

end problem1_problem2_l478_478762


namespace right_prism_volume_l478_478010

variables (p α β : ℝ)

def is_isosceles_triangle (A B C : ℝ × ℝ) : Prop :=
  dist A B = dist A C

def triangle_perimeter (A B C : ℝ × ℝ) : ℝ :=
  dist A B + dist B C + dist C A

noncomputable def prism_volume (p α β : ℝ) : ℝ :=
  p^3 * (Real.tan ((π - α) / 4))^3 * Real.tan (α / 2) * Real.tan β

theorem right_prism_volume 
  (p α β : ℝ)
  (A B C A1 B1 C1 : ℝ × ℝ)
  (isosceles : is_isosceles_triangle A B C)
  (perimeter : triangle_perimeter A B C = 2 * p)
  (angle_plane : ∀ (plane : Set (ℝ × ℝ)), (BC ∈ plane ∧ A1 ∈ plane) → angle plane = β) 
  : volume = prism_volume p α β :=
sorry

end right_prism_volume_l478_478010


namespace composition_symmetries_parallel_lines_l478_478857

theorem composition_symmetries_parallel_lines (n : ℕ) (l : ℕ → ℝ → ℝ) 
  (parallel : ∀ i j, l i = λ x, l j x + c i j) :
  (even n → ∃ v: ℝ, ∀ x: ℝ, (composition of symmetries with respect to l₀, l₁, ..., lₙ) x = x + v) ∧
  (odd n → ∃ r: ℝ → ℝ, ∀ x: ℝ, (composition of symmetries with respect to l₀, l₁, ..., lₙ) x = r x) :=
by
  sorry

end composition_symmetries_parallel_lines_l478_478857


namespace probability_A_and_B_selected_l478_478328

theorem probability_A_and_B_selected (A B : Type) (S : Finset (Type)) (h : S.card = 5) (hA : A ∈ S) (hB : B ∈ S) (T : Finset (Type)) (hT : T.card = 3) :
  (∃ T ⊆ S, A ∈ T ∧ B ∈ T ∧ T.card = 3) → 
  ∃ T ⊆ S, T.card = 3 ∧ (finset.card (T ∩ {A, B}) = 2) → 
  (finset.card (T ∩ {A, B}) = 2) = (3 / 10 : Real) := sorry

end probability_A_and_B_selected_l478_478328


namespace probability_A_and_B_selected_l478_478568

/-- Let there be a group of 5 students including A and B. 
    If 3 students are selected randomly from these 5, the probability of selecting both A and B is 3/10. -/
theorem probability_A_and_B_selected : 
  let students := 5 in
  let selected := 3 in
  let combinations := students.choose selected in
  let favorable_combinations := (students - 2).choose (selected - 2) in
  let probability := (favorable_combinations : ℚ) / (combinations : ℚ) in
  probability = 3 / 10 := 
by sorry

end probability_A_and_B_selected_l478_478568


namespace parabola_vertex_coordinates_l478_478889

-- Define the parabola
def parabola (x : ℝ) : ℝ := -x^2 + 4*x - 5

-- Define the vertex coordinates
def vertex_x : ℝ := -4 / (2 * -1)
def vertex_y : ℝ := parabola 2

theorem parabola_vertex_coordinates :
  (vertex_x, vertex_y) = (2, -1) :=
by
  -- Proof is omitted
  sorry

end parabola_vertex_coordinates_l478_478889


namespace flower_problem_l478_478086

theorem flower_problem
  (O : ℕ) 
  (total : ℕ := 105)
  (pink_purple : ℕ := 30)
  (red := 2 * O)
  (yellow := 2 * O - 5)
  (pink := pink_purple / 2)
  (purple := pink)
  (H1 : pink + purple = pink_purple)
  (H2 : pink_purple = 30)
  (H3 : pink = purple)
  (H4 : O + red + yellow + pink + purple = total)
  (H5 : total = 105):
  O = 16 := 
by 
  sorry

end flower_problem_l478_478086


namespace probability_of_selecting_A_and_B_l478_478559

theorem probability_of_selecting_A_and_B (students : Finset ℕ) (A B : ℕ)
  (h_size : students.card = 5) (h_in : A ∈ students ∧ B ∈ students) : 
  (Finset.card {s : Finset ℕ // s ⊆ students ∧ s.card = 3 ∧ A ∈ s ∧ B ∈ s}.to_set) / 
  (Finset.card {s : Finset ℕ // s ⊆ students ∧ s.card = 3}.to_set) = 3 / 10 :=
sorry

end probability_of_selecting_A_and_B_l478_478559


namespace probability_A_and_B_selected_l478_478378

theorem probability_A_and_B_selected 
  (students : Finset ℕ) (A B : ℕ) 
  (h_students_size : students.card = 5) 
  (h_A_in_students : A ∈ students) 
  (h_B_in_students : B ∈ students)
  : (↑(Nat.choose 3) : ℚ) / (↑(Nat.choose 5) : ℚ) = 3 / 10 := 
by sorry

end probability_A_and_B_selected_l478_478378


namespace select_3_from_5_prob_l478_478429

theorem select_3_from_5_prob {students : Finset ℕ} 
  (h : students.card = 5) 
  {A B : ℕ} 
  (hA : A ∈ students) 
  (hB : B ∈ students) 
  (H : 3 ≤ students.card) : 
  (finset.card (students.filter (λ x, x ≠ A ∧ x ≠ B)) = 3) → 
  (finset.card ((students.filter (λ x, x ≠ A ∧ x ≠ B)).subtype (λ x, True)) = 3) → 
  ((finset.card (finset.filter (λ (students : Finset ℕ) A ∈ students ∧ B ∈ students ↔ 3 = students.card) 5)) = ↑10) → 
  ∃ students', (students'.card = 3 ∧ A ∈ students' ∧ B ∈ students') → 
  ∃ (p : ℚ), p = 3 / 10 ∧ students.filter (λ x, x = A ∨ x = B) = 2 :=
  sorry

end select_3_from_5_prob_l478_478429


namespace probability_AB_selected_l478_478514

def factorial (n : Nat) : Nat := if n = 0 then 1 else n * factorial (n - 1)

def choose (n k : Nat) : Nat := factorial n / (factorial k * factorial (n - k))

theorem probability_AB_selected (h1 : 5 = 5) (h2 : 3 = 3) : (choose 3 1) / (choose 5 3) = 3 / 10 := by
  sorry

end probability_AB_selected_l478_478514


namespace triangle_area_l478_478753

noncomputable def f (x : ℝ) : ℝ := 2 * sin x * cos x - 2 * sqrt 3 * (cos x)^2 + sqrt 3

def interval_of_increase (k : ℤ) : Set ℝ := Set.Icc (-π / 12 + k * π) (5 * π / 12 + k * π)

def range_of_f (x : ℝ) : ℝ := 
  if x ∈ Set.Icc (π / 3) (11 * π / 24) 
  then sqrt 3 
  else 2

def circumradius : ℝ := (3 / 4) * sqrt 2

theorem triangle_area :
  ∃ (b c : ℝ) (R : ℝ) (A : ℝ), 
  (b = 2) ∧ (c = sqrt 3) ∧ (R = circumradius) ∧ (A = (b * c * sqrt 6 / 3) / 2) → 
  (A = sqrt 2) :=
sorry

end triangle_area_l478_478753


namespace min_value_m_min_value_expression_m_max_value_x_max_value_expression_x_l478_478890

-- Part 1: Prove the minimum value of the algebraic expression m^2 - 6m + 10
theorem min_value_m (m : ℝ) : (m - 3)^2 + 1 ≥ 1 :=
by sorry

-- Part 1: Further deducing that min of m^2 - 6m + 10 is 1
theorem min_value_expression_m (m : ℝ) : m^2 - 6m + 10 ≥ 1 :=
by sorry

-- Part 2: Prove the maximum value of the algebraic expression -2x^2 - 4x + 3
theorem max_value_x (x : ℝ) : -2 * (x + 1)^2 + 5 ≤ 5 :=
by sorry

-- Part 2: Further deducing that max of -2x^2 - 4x + 3 is 5
theorem max_value_expression_x (x : ℝ) : -2 * x^2 - 4 * x + 3 ≤ 5 :=
by sorry

end min_value_m_min_value_expression_m_max_value_x_max_value_expression_x_l478_478890


namespace probability_of_selecting_A_and_B_is_three_tenths_l478_478605

-- Define the set of students
def students : Finset ℕ := {0, 1, 2, 3, 4}

-- Define subsets A and B
def student_A : ℕ := 0
def student_B : ℕ := 1

-- Total number of ways to select 3 students out of 5
def total_ways : ℕ := students.card.choose 3

-- Number of ways to select 3 students including A and B
def favorable_ways : ℕ := {2, 3, 4}.card.choose 1

-- The probability
def probability_of_selecting_A_and_B : ℚ := (favorable_ways : ℚ) / (total_ways : ℚ)

theorem probability_of_selecting_A_and_B_is_three_tenths :
  probability_of_selecting_A_and_B = 3 / 10 :=
by
  sorry

end probability_of_selecting_A_and_B_is_three_tenths_l478_478605


namespace probability_of_selecting_A_and_B_l478_478253

-- Define the combination function
def combination (n r : ℕ) : ℕ := Nat.factorial n / (Nat.factorial r * Nat.factorial (n - r))

-- Define the conditions
def students : ℕ := 5
def to_select : ℕ := 3

-- Define the favorable outcomes where A and B are selected
def favorable_outcomes : ℕ := combination 3 1

-- Define the total outcomes of selecting 3 out of 5 students
def total_outcomes : ℕ := combination 5 3

-- Define the probability calculation
def probability_of_A_and_B : ℚ := favorable_outcomes / total_outcomes

-- Proposition to prove
theorem probability_of_selecting_A_and_B : probability_of_A_and_B = 3 / 10 :=
by
  sorry

end probability_of_selecting_A_and_B_l478_478253


namespace probability_A_and_B_selected_l478_478307

open Finset

noncomputable def total_ways : ℕ := (choose 5 3)

noncomputable def favorable_ways : ℕ := (choose 3 1)

noncomputable def probability (total favorable : ℕ) : ℚ := favorable / total

theorem probability_A_and_B_selected :
  probability total_ways favorable_ways = 3 / 10 :=
by
  sorry

end probability_A_and_B_selected_l478_478307


namespace probability_both_A_and_B_selected_l478_478411

-- The universe of students we consider
inductive Student
| A : Student
| B : Student
| C : Student
| D : Student
| E : Student

open Student

-- Function to count combinations
def combinations (n k : ℕ) : ℕ := (Nat.factorial n) / ((Nat.factorial k) * (Nat.factorial (n - k)))

-- The set of all students
def students : List Student := [A, B, C, D, E]

-- The total number of ways to select 3 students from 5
def total_selections := combinations 5 3

-- The number of ways A and B are both selected by selecting 1 more from the remaining 3 students
def selections_AB := combinations 3 1

-- The probability that both A and B are selected
def probability_AB := (selections_AB : ℚ) / (total_selections : ℚ)

-- The proof statement
theorem probability_both_A_and_B_selected :
  probability_AB = 3 / 10 :=
sorry

end probability_both_A_and_B_selected_l478_478411


namespace select_3_from_5_prob_l478_478444

theorem select_3_from_5_prob {students : Finset ℕ} 
  (h : students.card = 5) 
  {A B : ℕ} 
  (hA : A ∈ students) 
  (hB : B ∈ students) 
  (H : 3 ≤ students.card) : 
  (finset.card (students.filter (λ x, x ≠ A ∧ x ≠ B)) = 3) → 
  (finset.card ((students.filter (λ x, x ≠ A ∧ x ≠ B)).subtype (λ x, True)) = 3) → 
  ((finset.card (finset.filter (λ (students : Finset ℕ) A ∈ students ∧ B ∈ students ↔ 3 = students.card) 5)) = ↑10) → 
  ∃ students', (students'.card = 3 ∧ A ∈ students' ∧ B ∈ students') → 
  ∃ (p : ℚ), p = 3 / 10 ∧ students.filter (λ x, x = A ∨ x = B) = 2 :=
  sorry

end select_3_from_5_prob_l478_478444


namespace allocation_schemes_for_5_teachers_to_3_buses_l478_478088

noncomputable def number_of_allocation_schemes (teachers : ℕ) (buses : ℕ) : ℕ :=
  if buses = 3 ∧ teachers = 5 then 150 else 0

theorem allocation_schemes_for_5_teachers_to_3_buses : 
  number_of_allocation_schemes 5 3 = 150 := 
by
  sorry

end allocation_schemes_for_5_teachers_to_3_buses_l478_478088


namespace probability_both_A_B_selected_l478_478209

-- Define the problem with the necessary conditions.
def probability_AB_selected (total_students : ℕ) (select_students : ℕ) (A B : ℕ) : ℚ :=
  if A < total_students ∧ B < total_students ∧ total_students = 5 ∧ select_students = 3 then
    let total_ways := (Nat.factorial total_students) / ((Nat.factorial select_students) * (Nat.factorial (total_students - select_students)))
    let favorable_ways := (Nat.factorial (total_students - 2)) / ((Nat.factorial (select_students - 2)) * (Nat.factorial ((total_students - 2) - (select_students - 2))))
    favorable_ways / total_ways
  else 0

theorem probability_both_A_B_selected 
  (total_students : ℕ) (select_students : ℕ) (A B : ℕ) 
  (h1 : total_students = 5) (h2 : select_students = 3) 
  (h3 : A < total_students) (h4 : B < total_students) 
  : probability_AB_selected total_students select_students A B = 3 / 10 := 
by {
  -- Insert the delta and logic to prove the theorem here.
  sorry
}

end probability_both_A_B_selected_l478_478209


namespace rectangle_area_l478_478019

theorem rectangle_area (b : ℕ) (l : ℕ) (h1 : l = 3 * b) (h2 : 2 * (l + b) = 48) : l * b = 108 := 
by
  sorry

end rectangle_area_l478_478019


namespace prob_select_A_and_B_l478_478643

theorem prob_select_A_and_B (students : Finset ℕ) (A B : ℕ) (hA : A ∈ students) (hB : B ∈ students) (h_card : students.card = 5) :
  (students.choose 3).count {comb | A ∈ comb ∧ B ∈ comb} / (students.choose 3).card = 3 / 10 := sorry

end prob_select_A_and_B_l478_478643


namespace max_squares_cut_diagonal_l478_478929

theorem max_squares_cut_diagonal (board : Matrix ℕ ℕ) (h_board : board.shape = (9, 9)) :
  ∃ max_squares, max_squares = 21 ∧
  (∀ cuts, cuts ≤ max_squares ∧ (board_does_not_fall_apart board cuts)) := 
sorry

end max_squares_cut_diagonal_l478_478929


namespace factorization_identity_l478_478172

theorem factorization_identity (x : ℝ) : 
  16 * x^2 - 40 * x + 25 = (4 * x - 5)^2 :=
  sorry

end factorization_identity_l478_478172


namespace area_comparison_of_convex_polygons_l478_478856

open EuclideanGeometry

-- Definitions based on given conditions
variables {n : ℕ} {A B : Fin n → Point}
variable [ConvexPolygon A]
variable [ConvexPolygon B]
variable (h_eq_sides : ∀ i, dist (A i) (A ((i + 1) % n)) = dist (B i) (B ((i + 1) % n)))
variable (h_circumscribed : ∃ (K : Circle), ∀ i, is_inscribed B K)

-- Problem Statement
theorem area_comparison_of_convex_polygons (h_eq_sides : ∀ i, dist (A i) (A ((i + 1) % n)) = dist (B i) (B ((i + 1) % n)))
    (h_circumscribed : ∃ (K : Circle), ∀ i, is_inscribed B K) :
    area B ≥ area A := 
    sorry

end area_comparison_of_convex_polygons_l478_478856


namespace percent_of_districts_with_fewer_than_50000_l478_478090

def districts_percentages (a b c : ℕ) : ℕ :=
  a + b

theorem percent_of_districts_with_fewer_than_50000 (a b c : ℕ) (ha : a = 35) (hb : b = 40) (hc : c = 25) :
  districts_percentages a b c = 75 :=
by
  rw [ha, hb]
  dsimp [districts_percentages]
  norm_num

end percent_of_districts_with_fewer_than_50000_l478_478090


namespace area_increase_l478_478092

-- Definitions based on conditions
def length : ℝ := 12
def width : ℝ := 8

-- Radii of the semicircles
def radius_large : ℝ := length / 2
def radius_small : ℝ := width / 2

-- Areas of the semicircles
def area_large_semicircles : ℝ := 2 * (Real.pi * radius_large ^ 2 / 2)
def area_small_semicircles : ℝ := 2 * (Real.pi * radius_small ^ 2 / 2)

-- Statement to prove
theorem area_increase :
  ((area_large_semicircles - area_small_semicircles) / area_small_semicircles * 100).to_nat = 125 :=
by
  sorry

end area_increase_l478_478092


namespace slip_4_5_in_cup_B_l478_478917

def slips := [1.5, 2, 2, 2.5, 2.5, 3, 3, 3.5, 3.5, 4, 4, 4.5, 5, 5.5, 6]
def cup_sum (cup: List ℝ) := cup.sum
def cup_A := [2]
def cup_C := [3.5]

theorem slip_4_5_in_cup_B (A B C D E : List ℝ)
    (h1 : (cup_A ++ A).sum = 8)
    (h2 : (B).sum = 9)
    (h3 : (cup_C ++ C).sum = 10)
    (h4 : (D).sum = 11)
    (h5 : (E).sum = 14)
    (h_slips : cup_A ++ A ++ B ++ cup_C ++ C ++ D ++ E = slips)
    : 4.5 ∈ B :=
by
  sorry

end slip_4_5_in_cup_B_l478_478917


namespace probability_of_selecting_A_and_B_l478_478545

theorem probability_of_selecting_A_and_B (students : Finset ℕ) (A B : ℕ)
  (h_size : students.card = 5) (h_in : A ∈ students ∧ B ∈ students) : 
  (Finset.card {s : Finset ℕ // s ⊆ students ∧ s.card = 3 ∧ A ∈ s ∧ B ∈ s}.to_set) / 
  (Finset.card {s : Finset ℕ // s ⊆ students ∧ s.card = 3}.to_set) = 3 / 10 :=
sorry

end probability_of_selecting_A_and_B_l478_478545


namespace probability_of_A_and_B_selected_l478_478481

variable {α : Type}

/-- From 5 students, we randomly select 3 students to participate in community service work. 
  Let A and B be two specific students among these five. The probability that 
  both A and B are selected is 3 / 10. -/
theorem probability_of_A_and_B_selected (A B : α) (s : Finset α) (h : s.card = 5) :
  (∃ t : Finset α, t ⊆ s ∧ A ∈ t ∧ B ∈ t ∧ (t.card = 3) ∧ (t.card = 3) →
   (@Finset.exists_subset_card_eq α t s 3) →
   (@Finset.exists_subset_card_eq α t (Finset.erase s A) 2) ∧
   (@Finset.exists_subset_card_eq α t (Finset.erase (Finset.erase s A) B) 1)) →
  (3 : ℚ) / 10 :=
by
  sorry -- Proof omitted as per instructions.

end probability_of_A_and_B_selected_l478_478481


namespace probability_A_and_B_selected_l478_478702

/-- There are 5 students including A and B. The probability of selecting both A and B 
    when 3 students are selected randomly from 5 is 3/10. -/
theorem probability_A_and_B_selected :
  (∃ (students : List String) (h : students.length = 5),
    ∃ (selected : List String) (h' : selected.length = 3),
    "A" ∈ selected ∧ "B" ∈ selected) →
  ∃ (P : ℚ), P = 3 / 10 :=
by
  sorry

end probability_A_and_B_selected_l478_478702


namespace distinct_even_numbers_l478_478020

theorem distinct_even_numbers:
  let digits := {0, 1, 2, 3} in
  let evens := {0, 2} in
  let valid_digit_placement := (d1 != 0 ∧ d2 ∈ digits ∧ d3 ∈ digits) in
  (∀ d1 d2 d3, d1 ∈ digits ∧ d2 ∈ digits ∧ d3 ∈ digits ∧ d1 ≠ 0 →
    ((d1 * 100 + d2 * 10 + d3) % 2 = 0)) → 
  ∑ (d1 ∈ evens ∧ d1 ≠ 0), 3 * 2 * 1 + ∑ (d2 ∈ evens ∧ d2 ≠ 0), 2 * 2 * 1 = 10 := 
sorry

end distinct_even_numbers_l478_478020


namespace part1_max_value_when_a_zero_part2_range_of_a_l478_478756

noncomputable def f (x a : ℝ) : ℝ := (2 * x) / (Real.exp x) + a * Real.log (x + 1)

theorem part1_max_value_when_a_zero :
  ∀ x : ℝ, f x 0 ≤ 2 / Real.exp 1 :=
sorry

theorem part2_range_of_a (a : ℝ) :
  (∀ x : ℝ, 0 ≤ x → f x a ≤ 0) → a ∈ set.Iic (-2) :=
sorry

end part1_max_value_when_a_zero_part2_range_of_a_l478_478756


namespace probability_of_selecting_A_and_B_l478_478262

-- Define the combination function
def combination (n r : ℕ) : ℕ := Nat.factorial n / (Nat.factorial r * Nat.factorial (n - r))

-- Define the conditions
def students : ℕ := 5
def to_select : ℕ := 3

-- Define the favorable outcomes where A and B are selected
def favorable_outcomes : ℕ := combination 3 1

-- Define the total outcomes of selecting 3 out of 5 students
def total_outcomes : ℕ := combination 5 3

-- Define the probability calculation
def probability_of_A_and_B : ℚ := favorable_outcomes / total_outcomes

-- Proposition to prove
theorem probability_of_selecting_A_and_B : probability_of_A_and_B = 3 / 10 :=
by
  sorry

end probability_of_selecting_A_and_B_l478_478262


namespace christopher_age_l478_478129

theorem christopher_age (G C : ℕ) (h1 : C = 2 * G) (h2 : C - 9 = 5 * (G - 9)) : C = 24 := 
by
  sorry

end christopher_age_l478_478129


namespace sum_of_roots_l478_478055

-- Define the given equation
def equation (x : ℝ) : ℝ := 3 * x^3 + 7 * x^2 - 9 * x

-- Prove the sum of the roots
theorem sum_of_roots (x1 x2 x3 : ℝ)
  (h1: equation x1 = 0)
  (h2: equation x2 = 0)
  (h3: equation x3 = 0)
  (h_distinct : x1 ≠ x2 ∧ x1 ≠ x3 ∧ x2 ≠ x3)
  (h_x1_zero : x1 = 0)
  (h_roots : ∀ x, equation x = 0 → x = x1 ∨ x = x2 ∨ x = x3) :
  x2 + x3 ≈ -2.33 :=
by sorry

end sum_of_roots_l478_478055


namespace probability_of_selecting_A_and_B_l478_478543

theorem probability_of_selecting_A_and_B (students : Finset ℕ) (A B : ℕ)
  (h_size : students.card = 5) (h_in : A ∈ students ∧ B ∈ students) : 
  (Finset.card {s : Finset ℕ // s ⊆ students ∧ s.card = 3 ∧ A ∈ s ∧ B ∈ s}.to_set) / 
  (Finset.card {s : Finset ℕ // s ⊆ students ∧ s.card = 3}.to_set) = 3 / 10 :=
sorry

end probability_of_selecting_A_and_B_l478_478543


namespace probability_A_and_B_selected_l478_478236

theorem probability_A_and_B_selected (students : Finset ℕ) (A B : ℕ) (h : A ∈ students ∧ B ∈ students ∧ students.card = 5) : 
  (students.choose 3).count (λ s, {A, B} ⊆ s) / (students.choose 3).card = 3 / 10 := by
  sorry

end probability_A_and_B_selected_l478_478236


namespace part1_part2_part3_l478_478725

variables (a S : ℕ → ℝ)
hypotheses
  (h_pos : ∀ n, a n > 0)
  (h_S : ∀ n, S n = (finset.range n).sum a)
  (h_a1 : a 1 = 1)
  (h_a_next : ∀ n : ℕ, n > 0 → a (n + 1) = 2 * real.sqrt (S n) + 1)

theorem part1 : a 2 = 3 :=
sorry

theorem part2 : ∀ n : ℕ, n > 0 → a n = 2 * n - 1 :=
sorry

theorem part3 : ¬ ∃ k : ℕ, k > 0 ∧ a k, S (2 * k - 1), and a (4 * k) form a geometric sequence :=
sorry

end part1_part2_part3_l478_478725


namespace probability_all_three_same_flips_l478_478918

theorem probability_all_three_same_flips :
  let p := (1 / 3) in
  let q := (2 / 3) in
  let individual_prob (n : ℕ) : ℝ := (q ^ (n - 1)) * p in
  let total_prob : ℝ := ∑' n, (individual_prob n)^3 in
  (total_prob = 1 / 19) :=
by
  let p := (1 / 3)
  let q := (2 / 3)
  let individual_prob := fun (n : ℕ) => (q ^ (n - 1)) * p
  let total_prob := tsum (fun (n : ℕ) => (individual_prob n)^3)
  sorry

end probability_all_three_same_flips_l478_478918


namespace evaluate_expression_l478_478160

theorem evaluate_expression : (9⁻¹ - 6⁻¹)⁻¹ = -18 := 
by
  -- Definition expressions extracted from conditions
  let a := 9⁻¹
  let b := 6⁻¹
  have frac_a : a = 1 / 9 := by sorry
  have frac_b : b = 1 / 6 := by sorry
  let expr := (a - b)⁻¹
  have common_denom : 1 / 9 - 1 / 6 = (2 - 3) / 18 := by sorry
  have expr_evaluation : expr = -18 := by sorry
  show expr = -18 from expr_evaluation

end evaluate_expression_l478_478160


namespace probability_A_and_B_selected_l478_478574

/-- Let there be a group of 5 students including A and B. 
    If 3 students are selected randomly from these 5, the probability of selecting both A and B is 3/10. -/
theorem probability_A_and_B_selected : 
  let students := 5 in
  let selected := 3 in
  let combinations := students.choose selected in
  let favorable_combinations := (students - 2).choose (selected - 2) in
  let probability := (favorable_combinations : ℚ) / (combinations : ℚ) in
  probability = 3 / 10 := 
by sorry

end probability_A_and_B_selected_l478_478574


namespace select_3_from_5_prob_l478_478440

theorem select_3_from_5_prob {students : Finset ℕ} 
  (h : students.card = 5) 
  {A B : ℕ} 
  (hA : A ∈ students) 
  (hB : B ∈ students) 
  (H : 3 ≤ students.card) : 
  (finset.card (students.filter (λ x, x ≠ A ∧ x ≠ B)) = 3) → 
  (finset.card ((students.filter (λ x, x ≠ A ∧ x ≠ B)).subtype (λ x, True)) = 3) → 
  ((finset.card (finset.filter (λ (students : Finset ℕ) A ∈ students ∧ B ∈ students ↔ 3 = students.card) 5)) = ↑10) → 
  ∃ students', (students'.card = 3 ∧ A ∈ students' ∧ B ∈ students') → 
  ∃ (p : ℚ), p = 3 / 10 ∧ students.filter (λ x, x = A ∨ x = B) = 2 :=
  sorry

end select_3_from_5_prob_l478_478440


namespace probability_A_B_l478_478461

open Nat

-- Define a function for combination
def combination (n r : Nat) : Nat :=
  factorial n / (factorial r * factorial (n - r))

-- Define the set of students and the selection conditions
def num_students : Nat := 5
def num_selected : Nat := 3

-- Define A and B as selected students
def selected_A_B : Nat := combination 3 1

-- Define the probability calculation
def probability_A_B_selected : Rat :=
  (selected_A_B : Rat) / (combination num_students num_selected : Rat)

-- The statement of the problem
theorem probability_A_B : probability_A_B_selected = 3 / 10 := by
  sorry

end probability_A_B_l478_478461


namespace christmas_bonus_remainder_l478_478844

theorem christmas_bonus_remainder (P : ℕ) (h : P % 5 = 2) : (3 * P) % 5 = 1 :=
by
  sorry

end christmas_bonus_remainder_l478_478844


namespace relationship_between_m_and_n_l478_478714

theorem relationship_between_m_and_n
  (a b : ℝ)
  (ha : a > 0)
  (hb : b > 0)
  (m : ℝ := Real.log10 ((Real.sqrt a + Real.sqrt b) / 2))
  (n : ℝ := Real.log10 ((Real.sqrt (a + b)) / 2)) :
  m > n :=
sorry

end relationship_between_m_and_n_l478_478714


namespace abs_five_minus_e_l478_478161

variable (e : ℝ)
#check Real

noncomputable def approx_e := 2.71828

theorem abs_five_minus_e : ∀ (e : ℝ), e = approx_e → |5 - e| = 2.28172 :=
by
  intros
  sorry

end abs_five_minus_e_l478_478161


namespace probability_of_selecting_A_and_B_l478_478541

theorem probability_of_selecting_A_and_B (students : Finset ℕ) (A B : ℕ)
  (h_size : students.card = 5) (h_in : A ∈ students ∧ B ∈ students) : 
  (Finset.card {s : Finset ℕ // s ⊆ students ∧ s.card = 3 ∧ A ∈ s ∧ B ∈ s}.to_set) / 
  (Finset.card {s : Finset ℕ // s ⊆ students ∧ s.card = 3}.to_set) = 3 / 10 :=
sorry

end probability_of_selecting_A_and_B_l478_478541


namespace probability_of_selecting_A_and_B_l478_478285

open_locale classical

-- Define the number of students
def num_students := 5

-- Define the number of students to be selected
def students_to_select := 3

def combination (n k : ℕ) : ℕ := nat.choose n k

-- Define the correct answer
def correct_probability := 3 / 10

-- Define the probability calculation
def calc_probability := (combination 3 1) / (combination 5 3)

-- Statement of the theorem to be proved
theorem probability_of_selecting_A_and_B :
  calc_probability = correct_probability :=
by sorry

end probability_of_selecting_A_and_B_l478_478285


namespace probability_A_B_selected_l478_478351

noncomputable def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

def combination (n r : ℕ) : ℕ :=
  factorial n / (factorial r * factorial (n - r))

theorem probability_A_B_selected :
  let total_ways := combination 5 3 in
  let ways_A_B := combination 3 1 in
  (ways_A_B : ℚ) / total_ways = 3 / 10 :=
by
  sorry

end probability_A_B_selected_l478_478351


namespace total_courses_l478_478839

-- Define the conditions as variables
def max_courses : Nat := 40
def sid_courses : Nat := 4 * max_courses

-- State the theorem we want to prove
theorem total_courses : max_courses + sid_courses = 200 := 
  by
    -- This is where the actual proof would go
    sorry

end total_courses_l478_478839


namespace probability_team_A_wins_first_game_l478_478004

theorem probability_team_A_wins_first_game :
  let games := ["A", "B"]
  let series := list.permutations games
  series.count (λ g, 
    let wins := g.count "A" = 3 ∧ g.count "B" = 2
    wins ∧ g.head = "A"
  ) / series.count (λ g, g.count "A" = 3 ∧ g.count "B" = 2) = 1 / 2 := 
sorry

end probability_team_A_wins_first_game_l478_478004


namespace probability_A_B_l478_478453

open Nat

-- Define a function for combination
def combination (n r : Nat) : Nat :=
  factorial n / (factorial r * factorial (n - r))

-- Define the set of students and the selection conditions
def num_students : Nat := 5
def num_selected : Nat := 3

-- Define A and B as selected students
def selected_A_B : Nat := combination 3 1

-- Define the probability calculation
def probability_A_B_selected : Rat :=
  (selected_A_B : Rat) / (combination num_students num_selected : Rat)

-- The statement of the problem
theorem probability_A_B : probability_A_B_selected = 3 / 10 := by
  sorry

end probability_A_B_l478_478453


namespace probability_of_selecting_A_and_B_is_three_tenths_l478_478609

-- Define the set of students
def students : Finset ℕ := {0, 1, 2, 3, 4}

-- Define subsets A and B
def student_A : ℕ := 0
def student_B : ℕ := 1

-- Total number of ways to select 3 students out of 5
def total_ways : ℕ := students.card.choose 3

-- Number of ways to select 3 students including A and B
def favorable_ways : ℕ := {2, 3, 4}.card.choose 1

-- The probability
def probability_of_selecting_A_and_B : ℚ := (favorable_ways : ℚ) / (total_ways : ℚ)

theorem probability_of_selecting_A_and_B_is_three_tenths :
  probability_of_selecting_A_and_B = 3 / 10 :=
by
  sorry

end probability_of_selecting_A_and_B_is_three_tenths_l478_478609


namespace probability_A_and_B_selected_l478_478703

/-- There are 5 students including A and B. The probability of selecting both A and B 
    when 3 students are selected randomly from 5 is 3/10. -/
theorem probability_A_and_B_selected :
  (∃ (students : List String) (h : students.length = 5),
    ∃ (selected : List String) (h' : selected.length = 3),
    "A" ∈ selected ∧ "B" ∈ selected) →
  ∃ (P : ℚ), P = 3 / 10 :=
by
  sorry

end probability_A_and_B_selected_l478_478703


namespace probability_A_and_B_selected_l478_478376

theorem probability_A_and_B_selected 
  (students : Finset ℕ) (A B : ℕ) 
  (h_students_size : students.card = 5) 
  (h_A_in_students : A ∈ students) 
  (h_B_in_students : B ∈ students)
  : (↑(Nat.choose 3) : ℚ) / (↑(Nat.choose 5) : ℚ) = 3 / 10 := 
by sorry

end probability_A_and_B_selected_l478_478376


namespace largest_apartment_size_l478_478112

theorem largest_apartment_size (rental_rate affordable_rent : ℝ)
  (h_rental_rate : rental_rate = 1.10)
  (h_affordable_rent : affordable_rent = 715) :
  affordable_rent / rental_rate = 650 :=
by
  simp [h_rental_rate, h_affordable_rent]
  norm_num
  sorry

end largest_apartment_size_l478_478112


namespace cone_to_sphere_ratio_l478_478967

noncomputable def sphere_volume (r : ℝ) : ℝ := (4/3) * π * r^3
noncomputable def cone_volume (r h : ℝ) : ℝ := (1/3) * π * (2*r)^2 * h

theorem cone_to_sphere_ratio (r h : ℝ) :
  cone_volume r h = (1/3) * sphere_volume r →
  h / (2 * r) = 1 / 6 :=
by
  sorry

end cone_to_sphere_ratio_l478_478967


namespace probability_A_and_B_selected_l478_478532

theorem probability_A_and_B_selected :
  let n := 5
  let r := 3
  let total_ways := Nat.choose n r
  let favorable_ways := Nat.choose 3 1
  let probability := favorable_ways / total_ways
  probability = (3 / 10) := 
by
  sorry

end probability_A_and_B_selected_l478_478532


namespace probability_of_selecting_A_and_B_l478_478264

-- Define the combination function
def combination (n r : ℕ) : ℕ := Nat.factorial n / (Nat.factorial r * Nat.factorial (n - r))

-- Define the conditions
def students : ℕ := 5
def to_select : ℕ := 3

-- Define the favorable outcomes where A and B are selected
def favorable_outcomes : ℕ := combination 3 1

-- Define the total outcomes of selecting 3 out of 5 students
def total_outcomes : ℕ := combination 5 3

-- Define the probability calculation
def probability_of_A_and_B : ℚ := favorable_outcomes / total_outcomes

-- Proposition to prove
theorem probability_of_selecting_A_and_B : probability_of_A_and_B = 3 / 10 :=
by
  sorry

end probability_of_selecting_A_and_B_l478_478264


namespace nickel_more_chocolates_than_robert_l478_478860

namespace Chocolates

def robert_chocolates : Nat := 7
def robert_cost_per_chocolate : Nat := 2
def nickel_chocolates : Nat := 5
def nickel_discount : Nat := 1.5
def robert_total_cost : Nat := robert_chocolates * robert_cost_per_chocolate
def nickel_cost_per_chocolate : Nat := robert_cost_per_chocolate - nickel_discount
def nickel_total_cost : Nat := nickel_chocolates * nickel_cost_per_chocolate

-- Prove Nickel could buy 21 more chocolates than Robert given the price difference
theorem nickel_more_chocolates_than_robert :
  robert_total_cost = 14 →
  ∃ (additional_chocolates : Nat), additional_chocolates = 21 :=
by
  sorry

end Chocolates

end nickel_more_chocolates_than_robert_l478_478860


namespace probability_A_B_selected_l478_478371

noncomputable def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

def combination (n r : ℕ) : ℕ :=
  factorial n / (factorial r * factorial (n - r))

theorem probability_A_B_selected :
  let total_ways := combination 5 3 in
  let ways_A_B := combination 3 1 in
  (ways_A_B : ℚ) / total_ways = 3 / 10 :=
by
  sorry

end probability_A_B_selected_l478_478371


namespace determine_b_l478_478003

theorem determine_b (b : ℝ) :
  (∀ x : ℝ, 2 * f_inv(x) + b * f_inv(x) = 1 -> f_inv(x) = (1 - 3 * x) / (3 * x)) → b = 3 := by
  sorry

end determine_b_l478_478003


namespace smallest_N_divisible_2_to_10_l478_478195

theorem smallest_N_divisible_2_to_10 :
  ∃ (N : ℕ), N = 2520 ∧ 
  (∀ k ∈ {2, 3, 4, 5, 6, 7, 8, 9, 10}, (N + k) % k = 0) := 
sorry

end smallest_N_divisible_2_to_10_l478_478195


namespace total_courses_attended_l478_478837

-- Define the number of courses attended by Max
def maxCourses : ℕ := 40

-- Define the number of courses attended by Sid (four times as many as Max)
def sidCourses : ℕ := 4 * maxCourses

-- Define the total number of courses attended by both Max and Sid
def totalCourses : ℕ := maxCourses + sidCourses

-- The proof statement
theorem total_courses_attended : totalCourses = 200 := by
  sorry

end total_courses_attended_l478_478837


namespace probability_of_selecting_A_and_B_is_three_tenths_l478_478597

-- Define the set of students
def students : Finset ℕ := {0, 1, 2, 3, 4}

-- Define subsets A and B
def student_A : ℕ := 0
def student_B : ℕ := 1

-- Total number of ways to select 3 students out of 5
def total_ways : ℕ := students.card.choose 3

-- Number of ways to select 3 students including A and B
def favorable_ways : ℕ := {2, 3, 4}.card.choose 1

-- The probability
def probability_of_selecting_A_and_B : ℚ := (favorable_ways : ℚ) / (total_ways : ℚ)

theorem probability_of_selecting_A_and_B_is_three_tenths :
  probability_of_selecting_A_and_B = 3 / 10 :=
by
  sorry

end probability_of_selecting_A_and_B_is_three_tenths_l478_478597


namespace range_of_a_inequality_holds_l478_478718

-- Define the function f(x)
def f (a x : ℝ) : ℝ := a * Real.log (x + 1) + 1 / (x + 1) + 3 * x - 1

-- Statement 1: Proving the range of a
theorem range_of_a (a : ℝ) : (∀ x ≥ 0, f a x ≥ 0) ↔ a ≥ -2 := 
sorry

-- Statement 2: Proving the inequality for all positive integers n
theorem inequality_holds (n : ℕ) (hn : 0 < n) : 
  (∑ k in Finset.range n, ↑(k + 1) / (4 * (↑k + 1)^2 - 1)) > 1 / 4 * Real.log (2 * ↑n + 1) := 
sorry

end range_of_a_inequality_holds_l478_478718


namespace probability_A_and_B_selected_l478_478338

theorem probability_A_and_B_selected (A B : Type) (S : Finset (Type)) (h : S.card = 5) (hA : A ∈ S) (hB : B ∈ S) (T : Finset (Type)) (hT : T.card = 3) :
  (∃ T ⊆ S, A ∈ T ∧ B ∈ T ∧ T.card = 3) → 
  ∃ T ⊆ S, T.card = 3 ∧ (finset.card (T ∩ {A, B}) = 2) → 
  (finset.card (T ∩ {A, B}) = 2) = (3 / 10 : Real) := sorry

end probability_A_and_B_selected_l478_478338


namespace prob_select_A_and_B_l478_478633

theorem prob_select_A_and_B : 
  let total_students := 5
  let to_select := 3
  let favorable_outcomes := Nat.div_fact (Nat.factorial 3) (Nat.factorial 1)
  let total_outcomes := Nat.div_fact (Nat.factorial 5) (Nat.factorial 3 * Nat.factorial (5 - 3))
  (favorable_outcomes : ℚ) / total_outcomes = 3 / 10 :=
by
  sorry

end prob_select_A_and_B_l478_478633


namespace brenda_peaches_left_brenda_peaches_left_correct_l478_478120

theorem brenda_peaches_left (total_peaches : ℕ) (fresh_pct : ℝ) (too_small_peaches : ℕ)
  (h1 : total_peaches = 250)
  (h2 : fresh_pct = 0.60)
  (h3 : too_small_peaches = 15) : ℕ :=
sorry

theorem brenda_peaches_left_correct : brenda_peaches_left 250 0.60 15 = 135 :=
by {
  rw brenda_peaches_left,
  exact sorry
}

end brenda_peaches_left_brenda_peaches_left_correct_l478_478120


namespace Kiley_ate_two_slices_l478_478157

-- Definitions of conditions
def calories_per_slice := 350
def total_calories := 2800
def percentage_eaten := 0.25

-- Statement of the problem
theorem Kiley_ate_two_slices (h1 : total_calories = 2800)
                            (h2 : calories_per_slice = 350)
                            (h3 : percentage_eaten = 0.25) :
  (total_calories / calories_per_slice * percentage_eaten) = 2 := 
sorry

end Kiley_ate_two_slices_l478_478157


namespace revenue_increase_is_sixty_percent_l478_478950

-- Define the initial conditions
def original_weight := 400 -- grams
def original_cost := 150 -- rubles
def new_weight := 300 -- grams
def new_cost := 180 -- rubles

-- Define the calculations for the percentage increase in revenue
def percentage_increase :=
  let original_cost_per_kg := (original_cost.to_rat / original_weight.to_rat) * 1200
  let new_cost_per_kg := (new_cost.to_rat / new_weight.to_rat) * 1200
  let increase := new_cost_per_kg - original_cost_per_kg
  (increase / original_cost_per_kg) * 100

-- The theorem statement representing the proof problem
theorem revenue_increase_is_sixty_percent : percentage_increase = 60 :=
by
  sorry

end revenue_increase_is_sixty_percent_l478_478950


namespace license_plate_combinations_l478_478982

theorem license_plate_combinations : 
  let letters := 26
  let non_repeated_letters := 25
  let positions_for_unique := Nat.choose 4 1
  let digits := 10 * 9 * 8
  (letters * non_repeated_letters * positions_for_unique * digits) = 187200 := by
  let letters := 26
  let non_repeated_letters := 25
  let positions_for_unique := Nat.choose 4 1
  let digits := 10 * 9 * 8
  have h : letters * non_repeated_letters * positions_for_unique * digits = 187200 := by sorry
  exact h


end license_plate_combinations_l478_478982


namespace probability_A_and_B_selected_l478_478326

theorem probability_A_and_B_selected (A B : Type) (S : Finset (Type)) (h : S.card = 5) (hA : A ∈ S) (hB : B ∈ S) (T : Finset (Type)) (hT : T.card = 3) :
  (∃ T ⊆ S, A ∈ T ∧ B ∈ T ∧ T.card = 3) → 
  ∃ T ⊆ S, T.card = 3 ∧ (finset.card (T ∩ {A, B}) = 2) → 
  (finset.card (T ∩ {A, B}) = 2) = (3 / 10 : Real) := sorry

end probability_A_and_B_selected_l478_478326


namespace prob_select_A_and_B_l478_478635

theorem prob_select_A_and_B : 
  let total_students := 5
  let to_select := 3
  let favorable_outcomes := Nat.div_fact (Nat.factorial 3) (Nat.factorial 1)
  let total_outcomes := Nat.div_fact (Nat.factorial 5) (Nat.factorial 3 * Nat.factorial (5 - 3))
  (favorable_outcomes : ℚ) / total_outcomes = 3 / 10 :=
by
  sorry

end prob_select_A_and_B_l478_478635


namespace number_of_allowed_pairs_l478_478773

theorem number_of_allowed_pairs (total_books : ℕ) (prohibited_books : ℕ) : ℕ :=
  let total_pairs := (total_books * (total_books - 1)) / 2
  let prohibited_pairs := (prohibited_books * (prohibited_books - 1)) / 2
  total_pairs - prohibited_pairs

example : number_of_allowed_pairs 15 3 = 102 :=
by
  sorry

end number_of_allowed_pairs_l478_478773


namespace probability_of_selecting_A_and_B_is_three_tenths_l478_478604

-- Define the set of students
def students : Finset ℕ := {0, 1, 2, 3, 4}

-- Define subsets A and B
def student_A : ℕ := 0
def student_B : ℕ := 1

-- Total number of ways to select 3 students out of 5
def total_ways : ℕ := students.card.choose 3

-- Number of ways to select 3 students including A and B
def favorable_ways : ℕ := {2, 3, 4}.card.choose 1

-- The probability
def probability_of_selecting_A_and_B : ℚ := (favorable_ways : ℚ) / (total_ways : ℚ)

theorem probability_of_selecting_A_and_B_is_three_tenths :
  probability_of_selecting_A_and_B = 3 / 10 :=
by
  sorry

end probability_of_selecting_A_and_B_is_three_tenths_l478_478604


namespace ganesh_average_speed_l478_478071

variable (D : ℝ) (hD : D > 0)

/-- Ganesh's average speed over the entire journey is 45 km/hr.
    Given:
    - Speed from X to Y is 60 km/hr
    - Speed from Y to X is 36 km/hr
--/
theorem ganesh_average_speed :
  let T1 := D / 60
  let T2 := D / 36
  let total_distance := 2 * D
  let total_time := T1 + T2
  (total_distance / total_time) = 45 :=
by
  sorry

end ganesh_average_speed_l478_478071


namespace select_3_from_5_prob_l478_478432

theorem select_3_from_5_prob {students : Finset ℕ} 
  (h : students.card = 5) 
  {A B : ℕ} 
  (hA : A ∈ students) 
  (hB : B ∈ students) 
  (H : 3 ≤ students.card) : 
  (finset.card (students.filter (λ x, x ≠ A ∧ x ≠ B)) = 3) → 
  (finset.card ((students.filter (λ x, x ≠ A ∧ x ≠ B)).subtype (λ x, True)) = 3) → 
  ((finset.card (finset.filter (λ (students : Finset ℕ) A ∈ students ∧ B ∈ students ↔ 3 = students.card) 5)) = ↑10) → 
  ∃ students', (students'.card = 3 ∧ A ∈ students' ∧ B ∈ students') → 
  ∃ (p : ℚ), p = 3 / 10 ∧ students.filter (λ x, x = A ∨ x = B) = 2 :=
  sorry

end select_3_from_5_prob_l478_478432


namespace tree_heights_l478_478031

theorem tree_heights :
  let Tree_A := 150
  let Tree_B := (2/3 : ℝ) * Tree_A
  let Tree_C := (1/2 : ℝ) * Tree_B
  let Tree_D := Tree_C + 25
  let Tree_E := 0.40 * Tree_A
  let Tree_F := (Tree_B + Tree_D) / 2
  let Tree_G := (3/8 : ℝ) * Tree_A
  let Tree_H := 1.25 * Tree_F
  let Tree_I := 0.60 * (Tree_E + Tree_G)
  let total_height := Tree_A + Tree_B + Tree_C + Tree_D + Tree_E + Tree_F + Tree_G + Tree_H + Tree_I
  Tree_A = 150 ∧
  Tree_B = 100 ∧
  Tree_C = 50 ∧
  Tree_D = 75 ∧
  Tree_E = 60 ∧
  Tree_F = 87.5 ∧
  Tree_G = 56.25 ∧
  Tree_H = 109.375 ∧
  Tree_I = 69.75 ∧
  total_height = 758.125 :=
by
  sorry

end tree_heights_l478_478031


namespace solution_set_inequality_l478_478191

def inequality (y : ℝ) : Prop := 1 / (y * (y + 2)) - 1 / ((y + 2) * (y + 4)) < 1 / 4

theorem solution_set_inequality :
  { y : ℝ | inequality y } = { y | y ∈ set.Union (λ x, [Iio (-4), Ioo (-2) 0, Ioi 1] x) } :=
by sorry

end solution_set_inequality_l478_478191


namespace probability_A_and_B_selected_l478_478243

theorem probability_A_and_B_selected (students : Finset ℕ) (A B : ℕ) (h : A ∈ students ∧ B ∈ students ∧ students.card = 5) : 
  (students.choose 3).count (λ s, {A, B} ⊆ s) / (students.choose 3).card = 3 / 10 := by
  sorry

end probability_A_and_B_selected_l478_478243


namespace probability_A_B_selected_l478_478356

noncomputable def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

def combination (n r : ℕ) : ℕ :=
  factorial n / (factorial r * factorial (n - r))

theorem probability_A_B_selected :
  let total_ways := combination 5 3 in
  let ways_A_B := combination 3 1 in
  (ways_A_B : ℚ) / total_ways = 3 / 10 :=
by
  sorry

end probability_A_B_selected_l478_478356


namespace select_3_from_5_prob_l478_478431

theorem select_3_from_5_prob {students : Finset ℕ} 
  (h : students.card = 5) 
  {A B : ℕ} 
  (hA : A ∈ students) 
  (hB : B ∈ students) 
  (H : 3 ≤ students.card) : 
  (finset.card (students.filter (λ x, x ≠ A ∧ x ≠ B)) = 3) → 
  (finset.card ((students.filter (λ x, x ≠ A ∧ x ≠ B)).subtype (λ x, True)) = 3) → 
  ((finset.card (finset.filter (λ (students : Finset ℕ) A ∈ students ∧ B ∈ students ↔ 3 = students.card) 5)) = ↑10) → 
  ∃ students', (students'.card = 3 ∧ A ∈ students' ∧ B ∈ students') → 
  ∃ (p : ℚ), p = 3 / 10 ∧ students.filter (λ x, x = A ∨ x = B) = 2 :=
  sorry

end select_3_from_5_prob_l478_478431


namespace exists_multiple_difference_seven_or_three_l478_478128

theorem exists_multiple_difference_seven_or_three (s : Finset ℤ) (h : s.card = 7) (h_bound : ∀ x ∈ s, 1 ≤ x ∧ x ≤ 2007) :
  ∃ (x y ∈ s), x ≠ y ∧ ((x - y) % 7 = 0 ∨ (x - y) % 3 = 0) :=
by sorry

end exists_multiple_difference_seven_or_three_l478_478128


namespace prob_select_A_and_B_l478_478642

theorem prob_select_A_and_B (students : Finset ℕ) (A B : ℕ) (hA : A ∈ students) (hB : B ∈ students) (h_card : students.card = 5) :
  (students.choose 3).count {comb | A ∈ comb ∧ B ∈ comb} / (students.choose 3).card = 3 / 10 := sorry

end prob_select_A_and_B_l478_478642


namespace probability_A_and_B_selected_l478_478312

open Finset

noncomputable def total_ways : ℕ := (choose 5 3)

noncomputable def favorable_ways : ℕ := (choose 3 1)

noncomputable def probability (total favorable : ℕ) : ℚ := favorable / total

theorem probability_A_and_B_selected :
  probability total_ways favorable_ways = 3 / 10 :=
by
  sorry

end probability_A_and_B_selected_l478_478312


namespace smallest_of_given_numbers_l478_478975

theorem smallest_of_given_numbers : 
  let x1 := 0
  let x2 := 1 / 2
  let x3 := -Real.pi
  let x4 := Real.sqrt 2
  x3 < x1 ∧ x3 < x2 ∧ x3 < x4 := 
by
  sorry

end smallest_of_given_numbers_l478_478975


namespace f_odd_f_increasing_l478_478824

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := (2^x - a) / (2^x + a)

-- Part 1: Proving that f(x) with a = 1 is an odd function
theorem f_odd : ∀ (x : ℝ), f 1 x = -f 1 (-x) :=
by sorry

-- Part 2: Proving that f(x) is strictly increasing when a > 0
theorem f_increasing (a : ℝ) (ha : a > 0) : StrictMono (f a) :=
by sorry

end f_odd_f_increasing_l478_478824


namespace brenda_peaches_remaining_l478_478118

theorem brenda_peaches_remaining (total_peaches : ℕ) (percent_fresh : ℚ) (thrown_away : ℕ) (fresh_peaches : ℕ) (remaining_peaches : ℕ) :
    total_peaches = 250 → 
    percent_fresh = 0.60 → 
    thrown_away = 15 → 
    fresh_peaches = total_peaches * percent_fresh → 
    remaining_peaches = fresh_peaches - thrown_away → 
    remaining_peaches = 135 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end brenda_peaches_remaining_l478_478118


namespace buckets_with_original_size_l478_478913

variable (C : ℝ) (N : ℝ)

theorem buckets_with_original_size :
  (C : ℝ) > 0 → (62.5 * (2 / 5) * C = N * C) → N = 25 :=
by
  intros hC hV
  have : C ≠ 0 := by linarith
  have h : (2 / 5) * 62.5 = 25 := by norm_num
  field_simp at hV
  rw h at hV
  assumption

end buckets_with_original_size_l478_478913


namespace probability_of_A_and_B_selected_l478_478491

variable {α : Type}

/-- From 5 students, we randomly select 3 students to participate in community service work. 
  Let A and B be two specific students among these five. The probability that 
  both A and B are selected is 3 / 10. -/
theorem probability_of_A_and_B_selected (A B : α) (s : Finset α) (h : s.card = 5) :
  (∃ t : Finset α, t ⊆ s ∧ A ∈ t ∧ B ∈ t ∧ (t.card = 3) ∧ (t.card = 3) →
   (@Finset.exists_subset_card_eq α t s 3) →
   (@Finset.exists_subset_card_eq α t (Finset.erase s A) 2) ∧
   (@Finset.exists_subset_card_eq α t (Finset.erase (Finset.erase s A) B) 1)) →
  (3 : ℚ) / 10 :=
by
  sorry -- Proof omitted as per instructions.

end probability_of_A_and_B_selected_l478_478491


namespace probability_A_and_B_selected_l478_478335

theorem probability_A_and_B_selected (A B : Type) (S : Finset (Type)) (h : S.card = 5) (hA : A ∈ S) (hB : B ∈ S) (T : Finset (Type)) (hT : T.card = 3) :
  (∃ T ⊆ S, A ∈ T ∧ B ∈ T ∧ T.card = 3) → 
  ∃ T ⊆ S, T.card = 3 ∧ (finset.card (T ∩ {A, B}) = 2) → 
  (finset.card (T ∩ {A, B}) = 2) = (3 / 10 : Real) := sorry

end probability_A_and_B_selected_l478_478335


namespace factorize_quadratic_l478_478181

theorem factorize_quadratic (x : ℝ) : 
  16 * x^2 - 40 * x + 25 = (4 * x - 5)^2 :=
sorry

end factorize_quadratic_l478_478181


namespace surface_area_div_by_pi_l478_478953

-- Define the given conditions about the circle and the sector
namespace ConeFromSector

variables (r : ℝ) (angle : ℝ)
def sector_arc_length (r : ℝ) (angle : ℝ) : ℝ := (angle / 360) * 2 * Real.pi * r

-- Define the radius as 15 obtained from the arc length calculation
noncomputable def cone_base_radius : ℝ := 15

-- Slant height (original circle radius) is given
noncomputable def cone_slant_height : ℝ := 20

-- Height of the cone obtained using Pythagorean theorem
noncomputable def cone_height (r : ℝ) (l : ℝ) : ℝ := Real.sqrt (l^2 - r^2)

-- Total surface area of the cone (base area + lateral surface area)
noncomputable def cone_total_surface_area (r : ℝ) (l : ℝ) : ℝ :=
  Real.pi * r * (r + l)

-- The proof problem, proving that the total surface area divided by π equals 525
theorem surface_area_div_by_pi
  (r : ℝ := cone_base_radius) 
  (l : ℝ := cone_slant_height) : 
  cone_total_surface_area r l / Real.pi = 525 := by
  sorry

end ConeFromSector

end surface_area_div_by_pi_l478_478953


namespace find_n_on_angle_bisector_l478_478744

theorem find_n_on_angle_bisector (M : ℝ × ℝ) (hM : M = (3 * n - 2, 2 * n + 7) ∧ M.1 + M.2 = 0) : 
    n = -1 :=
by
  sorry

end find_n_on_angle_bisector_l478_478744


namespace prob_select_A_and_B_l478_478621

theorem prob_select_A_and_B : 
  let total_students := 5
  let to_select := 3
  let favorable_outcomes := Nat.div_fact (Nat.factorial 3) (Nat.factorial 1)
  let total_outcomes := Nat.div_fact (Nat.factorial 5) (Nat.factorial 3 * Nat.factorial (5 - 3))
  (favorable_outcomes : ℚ) / total_outcomes = 3 / 10 :=
by
  sorry

end prob_select_A_and_B_l478_478621


namespace height_of_house_l478_478928

theorem height_of_house
(house_shadow tree_shadow tree_height : ℝ)
(ratio : house_shadow / tree_shadow = 9 / 4)
(house_shadow = 63) (tree_shadow = 28) (tree_height = 14) :
  let h := 14 * (9 / 4) in h = 32 :=
by
  sorry

end height_of_house_l478_478928


namespace probability_A_B_selected_l478_478354

noncomputable def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

def combination (n r : ℕ) : ℕ :=
  factorial n / (factorial r * factorial (n - r))

theorem probability_A_B_selected :
  let total_ways := combination 5 3 in
  let ways_A_B := combination 3 1 in
  (ways_A_B : ℚ) / total_ways = 3 / 10 :=
by
  sorry

end probability_A_B_selected_l478_478354


namespace probability_AB_selected_l478_478495

def factorial (n : Nat) : Nat := if n = 0 then 1 else n * factorial (n - 1)

def choose (n k : Nat) : Nat := factorial n / (factorial k * factorial (n - k))

theorem probability_AB_selected (h1 : 5 = 5) (h2 : 3 = 3) : (choose 3 1) / (choose 5 3) = 3 / 10 := by
  sorry

end probability_AB_selected_l478_478495


namespace probability_A_B_selected_l478_478358

noncomputable def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

def combination (n r : ℕ) : ℕ :=
  factorial n / (factorial r * factorial (n - r))

theorem probability_A_B_selected :
  let total_ways := combination 5 3 in
  let ways_A_B := combination 3 1 in
  (ways_A_B : ℚ) / total_ways = 3 / 10 :=
by
  sorry

end probability_A_B_selected_l478_478358


namespace max_edges_of_colored_graph_l478_478097

noncomputable def max_edges_colored_graph (G : SimpleGraph (Fin 2020)) (edges_colored : Fin 2020 → Fin 2020 → Prop) : ℕ :=
  sorry

theorem max_edges_of_colored_graph (G : SimpleGraph (Fin 2020)) (edges_colored : Fin 2020 → Fin 2020 → Prop) (h1 : ∀ u v, edges_colored u v = (edges_colored v u)) (h2 : ∀ cycle, is_monochromatic cycle -> even_length cycle) :
  max_edges_colored_graph G edges_colored = 1530150 :=
begin
  sorry,
end

end max_edges_of_colored_graph_l478_478097


namespace probability_of_selecting_A_and_B_l478_478547

theorem probability_of_selecting_A_and_B (students : Finset ℕ) (A B : ℕ)
  (h_size : students.card = 5) (h_in : A ∈ students ∧ B ∈ students) : 
  (Finset.card {s : Finset ℕ // s ⊆ students ∧ s.card = 3 ∧ A ∈ s ∧ B ∈ s}.to_set) / 
  (Finset.card {s : Finset ℕ // s ⊆ students ∧ s.card = 3}.to_set) = 3 / 10 :=
sorry

end probability_of_selecting_A_and_B_l478_478547


namespace sum_of_squares_is_289_l478_478901

theorem sum_of_squares_is_289 (x y : ℤ) (h1 : x * y = 120) (h2 : x + y = 23) : x^2 + y^2 = 289 :=
by
  sorry

end sum_of_squares_is_289_l478_478901


namespace unique_increasing_function_l478_478189

variable (f : ℝ → ℝ)
variables [Incr : StrictMono f]

-- Given Conditions
axiom h1 : f 1 = 1
axiom h2 : ∀ x y : ℝ, f (x + y) = f x + f y

theorem unique_increasing_function : ∀ x : ℝ, f x = x := 
by
  sorry

end unique_increasing_function_l478_478189


namespace probability_A_B_l478_478465

open Nat

-- Define a function for combination
def combination (n r : Nat) : Nat :=
  factorial n / (factorial r * factorial (n - r))

-- Define the set of students and the selection conditions
def num_students : Nat := 5
def num_selected : Nat := 3

-- Define A and B as selected students
def selected_A_B : Nat := combination 3 1

-- Define the probability calculation
def probability_A_B_selected : Rat :=
  (selected_A_B : Rat) / (combination num_students num_selected : Rat)

-- The statement of the problem
theorem probability_A_B : probability_A_B_selected = 3 / 10 := by
  sorry

end probability_A_B_l478_478465


namespace mod_exp_equivalence_l478_478131

theorem mod_exp_equivalence :
  (81^1814 - 25^1814) % 7 = 0 := by
  sorry

end mod_exp_equivalence_l478_478131


namespace jars_for_five_level_pyramid_max_levels_with_100_jars_l478_478763

theorem jars_for_five_level_pyramid : 
  ∀ k, sum (range 6) (λ k, k * (k + 1) / 2) = 35 := by sorry

theorem max_levels_with_100_jars :
  ∀ k, (sum (range (k+1)) (λ k, (k * (k + 1)) / 2)) ≤ 100 → k ≤ 7 := by sorry

end jars_for_five_level_pyramid_max_levels_with_100_jars_l478_478763


namespace shaded_area_fraction_is_one_eight_l478_478047

noncomputable def fraction_of_shaded_area :=
  let total_area := 15 * 20 in
  let half_rectangle_area := total_area / 2 in
  let shaded_area := (1 / 4) * half_rectangle_area in
  shaded_area / total_area

theorem shaded_area_fraction_is_one_eight :
  fraction_of_shaded_area = 1 / 8 :=
by
  sorry

end shaded_area_fraction_is_one_eight_l478_478047


namespace factorization_of_polynomial_l478_478185

theorem factorization_of_polynomial (x : ℝ) : 16 * x ^ 2 - 40 * x + 25 = (4 * x - 5) ^ 2 := by 
  sorry

end factorization_of_polynomial_l478_478185


namespace probability_both_A_and_B_selected_l478_478403

-- The universe of students we consider
inductive Student
| A : Student
| B : Student
| C : Student
| D : Student
| E : Student

open Student

-- Function to count combinations
def combinations (n k : ℕ) : ℕ := (Nat.factorial n) / ((Nat.factorial k) * (Nat.factorial (n - k)))

-- The set of all students
def students : List Student := [A, B, C, D, E]

-- The total number of ways to select 3 students from 5
def total_selections := combinations 5 3

-- The number of ways A and B are both selected by selecting 1 more from the remaining 3 students
def selections_AB := combinations 3 1

-- The probability that both A and B are selected
def probability_AB := (selections_AB : ℚ) / (total_selections : ℚ)

-- The proof statement
theorem probability_both_A_and_B_selected :
  probability_AB = 3 / 10 :=
sorry

end probability_both_A_and_B_selected_l478_478403


namespace probability_A_B_l478_478452

open Nat

-- Define a function for combination
def combination (n r : Nat) : Nat :=
  factorial n / (factorial r * factorial (n - r))

-- Define the set of students and the selection conditions
def num_students : Nat := 5
def num_selected : Nat := 3

-- Define A and B as selected students
def selected_A_B : Nat := combination 3 1

-- Define the probability calculation
def probability_A_B_selected : Rat :=
  (selected_A_B : Rat) / (combination num_students num_selected : Rat)

-- The statement of the problem
theorem probability_A_B : probability_A_B_selected = 3 / 10 := by
  sorry

end probability_A_B_l478_478452


namespace probability_of_selecting_A_and_B_l478_478278

open_locale classical

-- Define the number of students
def num_students := 5

-- Define the number of students to be selected
def students_to_select := 3

def combination (n k : ℕ) : ℕ := nat.choose n k

-- Define the correct answer
def correct_probability := 3 / 10

-- Define the probability calculation
def calc_probability := (combination 3 1) / (combination 5 3)

-- Statement of the theorem to be proved
theorem probability_of_selecting_A_and_B :
  calc_probability = correct_probability :=
by sorry

end probability_of_selecting_A_and_B_l478_478278


namespace probability_of_selecting_A_and_B_l478_478260

-- Define the combination function
def combination (n r : ℕ) : ℕ := Nat.factorial n / (Nat.factorial r * Nat.factorial (n - r))

-- Define the conditions
def students : ℕ := 5
def to_select : ℕ := 3

-- Define the favorable outcomes where A and B are selected
def favorable_outcomes : ℕ := combination 3 1

-- Define the total outcomes of selecting 3 out of 5 students
def total_outcomes : ℕ := combination 5 3

-- Define the probability calculation
def probability_of_A_and_B : ℚ := favorable_outcomes / total_outcomes

-- Proposition to prove
theorem probability_of_selecting_A_and_B : probability_of_A_and_B = 3 / 10 :=
by
  sorry

end probability_of_selecting_A_and_B_l478_478260


namespace percent_employed_females_l478_478802

theorem percent_employed_females (total_population employed_population employed_males : ℝ)
  (h1 : employed_population = 0.6 * total_population)
  (h2 : employed_males = 0.48 * total_population) :
  ((employed_population - employed_males) / employed_population) * 100 = 20 := 
by
  sorry

end percent_employed_females_l478_478802


namespace equation_of_parallel_line_passing_through_point_l478_478015

variable (x y : ℝ)

def is_point_on_line (x_val y_val : ℝ) (a b c : ℝ) : Prop := a * x_val + b * y_val + c = 0

def is_parallel (slope1 slope2 : ℝ) : Prop := slope1 = slope2

theorem equation_of_parallel_line_passing_through_point :
  (is_point_on_line (-1) 3 1 (-2) 7) ∧ (is_parallel (1 / 2) (1 / 2)) → (∀ x y, is_point_on_line x y 1 (-2) 7) :=
by
  sorry

end equation_of_parallel_line_passing_through_point_l478_478015


namespace probability_A_and_B_selected_l478_478696

/-- There are 5 students including A and B. The probability of selecting both A and B 
    when 3 students are selected randomly from 5 is 3/10. -/
theorem probability_A_and_B_selected :
  (∃ (students : List String) (h : students.length = 5),
    ∃ (selected : List String) (h' : selected.length = 3),
    "A" ∈ selected ∧ "B" ∈ selected) →
  ∃ (P : ℚ), P = 3 / 10 :=
by
  sorry

end probability_A_and_B_selected_l478_478696


namespace probability_A_and_B_selected_l478_478567

/-- Let there be a group of 5 students including A and B. 
    If 3 students are selected randomly from these 5, the probability of selecting both A and B is 3/10. -/
theorem probability_A_and_B_selected : 
  let students := 5 in
  let selected := 3 in
  let combinations := students.choose selected in
  let favorable_combinations := (students - 2).choose (selected - 2) in
  let probability := (favorable_combinations : ℚ) / (combinations : ℚ) in
  probability = 3 / 10 := 
by sorry

end probability_A_and_B_selected_l478_478567


namespace probability_of_selecting_A_and_B_l478_478674

open_locale classical

noncomputable def select_probability (n k : ℕ) : ℚ :=
  (nat.choose n k)⁻¹

theorem probability_of_selecting_A_and_B :
  let total_students := 5,
      students_to_select := 3,
      remaining_students := 3 in
  select_probability total_students students_to_select * nat.choose remaining_students 1 = 3 / 10 :=
by
  sorry

end probability_of_selecting_A_and_B_l478_478674


namespace probability_of_selecting_A_and_B_l478_478291

open_locale classical

-- Define the number of students
def num_students := 5

-- Define the number of students to be selected
def students_to_select := 3

def combination (n k : ℕ) : ℕ := nat.choose n k

-- Define the correct answer
def correct_probability := 3 / 10

-- Define the probability calculation
def calc_probability := (combination 3 1) / (combination 5 3)

-- Statement of the theorem to be proved
theorem probability_of_selecting_A_and_B :
  calc_probability = correct_probability :=
by sorry

end probability_of_selecting_A_and_B_l478_478291


namespace probability_of_selecting_A_and_B_is_three_tenths_l478_478611

-- Define the set of students
def students : Finset ℕ := {0, 1, 2, 3, 4}

-- Define subsets A and B
def student_A : ℕ := 0
def student_B : ℕ := 1

-- Total number of ways to select 3 students out of 5
def total_ways : ℕ := students.card.choose 3

-- Number of ways to select 3 students including A and B
def favorable_ways : ℕ := {2, 3, 4}.card.choose 1

-- The probability
def probability_of_selecting_A_and_B : ℚ := (favorable_ways : ℚ) / (total_ways : ℚ)

theorem probability_of_selecting_A_and_B_is_three_tenths :
  probability_of_selecting_A_and_B = 3 / 10 :=
by
  sorry

end probability_of_selecting_A_and_B_is_three_tenths_l478_478611


namespace rate_interest_l478_478940

variable (P A T : ℕ) (R : ℚ) 

theorem rate_interest : 
  P = 750 → A = 900 → T = 16 → 
  let SI := A - P in 
  R = (SI * 100) / (P * T) → 
  R = 1.25 :=
by
  intros hP hA hT hR
  sorry

end rate_interest_l478_478940


namespace probability_of_selecting_A_and_B_l478_478255

-- Define the combination function
def combination (n r : ℕ) : ℕ := Nat.factorial n / (Nat.factorial r * Nat.factorial (n - r))

-- Define the conditions
def students : ℕ := 5
def to_select : ℕ := 3

-- Define the favorable outcomes where A and B are selected
def favorable_outcomes : ℕ := combination 3 1

-- Define the total outcomes of selecting 3 out of 5 students
def total_outcomes : ℕ := combination 5 3

-- Define the probability calculation
def probability_of_A_and_B : ℚ := favorable_outcomes / total_outcomes

-- Proposition to prove
theorem probability_of_selecting_A_and_B : probability_of_A_and_B = 3 / 10 :=
by
  sorry

end probability_of_selecting_A_and_B_l478_478255


namespace probability_AB_selected_l478_478496

def factorial (n : Nat) : Nat := if n = 0 then 1 else n * factorial (n - 1)

def choose (n k : Nat) : Nat := factorial n / (factorial k * factorial (n - k))

theorem probability_AB_selected (h1 : 5 = 5) (h2 : 3 = 3) : (choose 3 1) / (choose 5 3) = 3 / 10 := by
  sorry

end probability_AB_selected_l478_478496


namespace sequence_length_l478_478767

theorem sequence_length (a d n : ℕ) (h1 : a = 3) (h2 : d = 5) (h3: 3 + (n-1) * d = 3008) : n = 602 := 
by
  sorry

end sequence_length_l478_478767


namespace rotationMappingTriangles_l478_478921

structure Point :=
  (x : ℝ)
  (y : ℝ)

structure Triangle :=
  (A : Point)
  (B : Point)
  (C : Point)

def rotate (θ : ℝ) (p q: Point) (t: Triangle) : Triangle :=
  sorry -- rotation function implementation is skipped

noncomputable def minimumRotationDegrees (t1 t2: Triangle) : ℝ :=
  sorry -- function to calculate minimum rotation degrees is skipped

def isRotationMapping (θ: ℝ) (p q: Point) (t1 t2: Triangle) : Prop :=
  rotate θ p q t1 = t2

def triangleXYZ : Triangle :=
{ A := ⟨0, 0⟩,
  B := ⟨0, 15⟩,
  C := ⟨20, 0⟩ }

def triangleX'Y'Z' : Triangle :=
{ A := ⟨30, 10⟩,
  B := ⟨40, 10⟩,
  C := ⟨30, 0⟩ }

def pxy : Point := ⟨-30, 5⟩ -- The point (p, q) of rotation

theorem rotationMappingTriangles :
  ∃ n : ℝ, isRotationMapping n pxy triangleXYZ triangleX'Y'Z' ∧ n + pxy.x + pxy.y = 65 :=
by
  sorry

end rotationMappingTriangles_l478_478921


namespace probability_of_selecting_A_and_B_is_three_tenths_l478_478610

-- Define the set of students
def students : Finset ℕ := {0, 1, 2, 3, 4}

-- Define subsets A and B
def student_A : ℕ := 0
def student_B : ℕ := 1

-- Total number of ways to select 3 students out of 5
def total_ways : ℕ := students.card.choose 3

-- Number of ways to select 3 students including A and B
def favorable_ways : ℕ := {2, 3, 4}.card.choose 1

-- The probability
def probability_of_selecting_A_and_B : ℚ := (favorable_ways : ℚ) / (total_ways : ℚ)

theorem probability_of_selecting_A_and_B_is_three_tenths :
  probability_of_selecting_A_and_B = 3 / 10 :=
by
  sorry

end probability_of_selecting_A_and_B_is_three_tenths_l478_478610


namespace probability_A_and_B_selected_l478_478342

theorem probability_A_and_B_selected (A B : Type) (S : Finset (Type)) (h : S.card = 5) (hA : A ∈ S) (hB : B ∈ S) (T : Finset (Type)) (hT : T.card = 3) :
  (∃ T ⊆ S, A ∈ T ∧ B ∈ T ∧ T.card = 3) → 
  ∃ T ⊆ S, T.card = 3 ∧ (finset.card (T ∩ {A, B}) = 2) → 
  (finset.card (T ∩ {A, B}) = 2) = (3 / 10 : Real) := sorry

end probability_A_and_B_selected_l478_478342


namespace probability_both_A_B_selected_l478_478228

-- Define the problem with the necessary conditions.
def probability_AB_selected (total_students : ℕ) (select_students : ℕ) (A B : ℕ) : ℚ :=
  if A < total_students ∧ B < total_students ∧ total_students = 5 ∧ select_students = 3 then
    let total_ways := (Nat.factorial total_students) / ((Nat.factorial select_students) * (Nat.factorial (total_students - select_students)))
    let favorable_ways := (Nat.factorial (total_students - 2)) / ((Nat.factorial (select_students - 2)) * (Nat.factorial ((total_students - 2) - (select_students - 2))))
    favorable_ways / total_ways
  else 0

theorem probability_both_A_B_selected 
  (total_students : ℕ) (select_students : ℕ) (A B : ℕ) 
  (h1 : total_students = 5) (h2 : select_students = 3) 
  (h3 : A < total_students) (h4 : B < total_students) 
  : probability_AB_selected total_students select_students A B = 3 / 10 := 
by {
  -- Insert the delta and logic to prove the theorem here.
  sorry
}

end probability_both_A_B_selected_l478_478228


namespace card_pair_probability_l478_478084
open_locale big_operators

theorem card_pair_probability (deck : finset ℕ) (num_removed : fin 2 → finset ℕ)
  (h_deck : deck.card = 50)
  (h_numbers : (∀ x ∈ deck, ∃ n : ℕ, x = n ∧ n ∈ (finset.range 10).map (nat.succ) ∧ 
                                      (deck.filter (λ y, y = n)).card = 5))
  (h_removed : (∀ i : fin 2, (num_removed i).card = 2 ∧ ∃ n : ℕ, 
                ∀ x ∈ num_removed i, x = n ∧ (∃ y ∈ deck, x = y ∧ 
                                                (deck.filter (λ z, z = y)).card = 2)))
  : let p := (8 * (4.choose 2) + 2 * (3.choose 2)) in
    p.gcd 1035 = 1 ∧ p + 1035 = 1121 := 
by
  let p := 86
  have h1 : p.gcd 1035 = 1 := by sorry
  have h2 : p + 1035 = 1121 := by sorry
  exact ⟨h1, h2⟩

end card_pair_probability_l478_478084


namespace arrangement_with_one_between_l478_478078

theorem arrangement_with_one_between :
  let n := 5
  let k := 1
  let total_ways := 36
  A B : ℕ in
  (∃ l : list ℕ, l.length = n ∧ (A ∈ l) ∧ (B ∈ l) ∧ (∀ i, l.nth i = some A → l.nth (i + k + 1) = some B) → 
   multiset.card (multiset.filter (λ l : list ℕ, l.length = n ∧ (A ∈ l) ∧ (B ∈ l) ∧ 
    (∀ i, l.nth i = some A → l.nth (i + k + 1) = some B)) {l | permutations l}.to_finset) = total_ways) :=
sorry

end arrangement_with_one_between_l478_478078


namespace exercise_l478_478025

noncomputable def a : ℕ → ℝ
| 0       := real.sqrt 2 / 2
| (n + 1) := real.sqrt 2 / 2 * real.sqrt (1 - real.sqrt (1 - (a n)^2))

noncomputable def b : ℕ → ℝ
| 0       := 1
| (n + 1) := (real.sqrt (1 + (b n)^2) - 1) / (b n)

theorem exercise (n : ℕ) : 2^(n + 2) * a n < real.pi ∧ real.pi < 2^(n + 2) * b n :=
sorry

end exercise_l478_478025


namespace probability_of_selecting_A_and_B_l478_478286

open_locale classical

-- Define the number of students
def num_students := 5

-- Define the number of students to be selected
def students_to_select := 3

def combination (n k : ℕ) : ℕ := nat.choose n k

-- Define the correct answer
def correct_probability := 3 / 10

-- Define the probability calculation
def calc_probability := (combination 3 1) / (combination 5 3)

-- Statement of the theorem to be proved
theorem probability_of_selecting_A_and_B :
  calc_probability = correct_probability :=
by sorry

end probability_of_selecting_A_and_B_l478_478286


namespace find_fx2_minus_1_l478_478825

theorem find_fx2_minus_1 (f : ℝ → ℝ) (h : ∀ x : ℝ, f(x^2 + 3) = x^4 + 6 * x^2 + 9) :
    f(x^2 - 1) = x^4 - 2 * x^2 - 8 := 
begin
  sorry
end

end find_fx2_minus_1_l478_478825


namespace prob_select_A_and_B_l478_478644

theorem prob_select_A_and_B (students : Finset ℕ) (A B : ℕ) (hA : A ∈ students) (hB : B ∈ students) (h_card : students.card = 5) :
  (students.choose 3).count {comb | A ∈ comb ∧ B ∈ comb} / (students.choose 3).card = 3 / 10 := sorry

end prob_select_A_and_B_l478_478644


namespace calculate_total_prime_dates_l478_478986

-- Define the prime months
def prime_months : List Nat := [2, 3, 5, 7, 11, 13]

-- Define the number of days in each month for a non-leap year
def days_in_month (month : Nat) : Nat :=
  if month = 2 then 28
  else if month = 3 then 31
  else if month = 5 then 31
  else if month = 7 then 31
  else if month = 11 then 30
  else if month = 13 then 31
  else 0

-- Define the prime days in a month
def prime_days : List Nat := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31]

-- Calculate the number of prime dates in a given month
def prime_dates_in_month (month : Nat) : Nat :=
  (prime_days.filter (λ d => d <= days_in_month month)).length

-- Calculate the total number of prime dates for the year
def total_prime_dates : Nat :=
  (prime_months.map prime_dates_in_month).sum

theorem calculate_total_prime_dates : total_prime_dates = 62 := by
  sorry

end calculate_total_prime_dates_l478_478986


namespace ratio_pentagram_to_circle_l478_478082

noncomputable def pentagram_area_ratio (r : ℝ) (A_circle : ℝ) (A_pentagon : ℝ) (ratio_pentagon_pentagram : ℝ) :=
  let A_pentagram := A_pentagon * ratio_pentagon_pentagram
  in (A_pentagram / A_circle)

theorem ratio_pentagram_to_circle {r A_circle A_pentagon ratio_pentagon_pentagram : ℝ}
  (h_r : r = 3)
  (h_A_circle : A_circle = π * r ^ 2)
  (h_A_pentagon : A_pentagon = (√(25 + 10 * √5) / 4) * r ^ 2)
  (h_ratio_pentagon_pentagram : ratio_pentagon_pentagram = 5 / 8) :
  pentagram_area_ratio r A_circle A_pentagon ratio_pentagon_pentagram = 5 * √(25 + 10 * √5) / (32 * π) :=
sorry

end ratio_pentagram_to_circle_l478_478082


namespace quadratic_equation_coefficients_l478_478058

theorem quadratic_equation_coefficients :
  ∀ (a b c : ℤ), (-a * x^2 + b * x = 1) → ((a = -1) ∧ (b = 3) ∧ (c = -1)) := 
by
  intros a b c h
  -- Transform the equation -x^2 + 3x - 1 = 0
  have eq := -a * x ^ 2 + b * x - c = 0,
  sorry

end quadratic_equation_coefficients_l478_478058


namespace prob_select_A_and_B_l478_478622

theorem prob_select_A_and_B : 
  let total_students := 5
  let to_select := 3
  let favorable_outcomes := Nat.div_fact (Nat.factorial 3) (Nat.factorial 1)
  let total_outcomes := Nat.div_fact (Nat.factorial 5) (Nat.factorial 3 * Nat.factorial (5 - 3))
  (favorable_outcomes : ℚ) / total_outcomes = 3 / 10 :=
by
  sorry

end prob_select_A_and_B_l478_478622


namespace slope_of_best_fitting_line_l478_478060

open Real

/-- The slope of the best fitting line for four points with linearly spaced x-values -/
theorem slope_of_best_fitting_line (x1 x2 x3 x4 y1 y2 y3 y4 d : ℝ)
  (h1 : x1 < x2) (h2 : x2 < x3) (h3 : x3 < x4)
  (h4 : x2 - x1 = d) (h5 : x3 - x2 = d) (h6 : x4 - x3 = d) :
  let m := (y4 - y1) / (x4 - x1)
  in ∃ m, m = (y4 - y1) / (x4 - x1) :=
  by            
  -- Variables y average, x average, and sum of products are established in proof steps
  sorry

end slope_of_best_fitting_line_l478_478060


namespace probability_A_and_B_selected_l478_478393

theorem probability_A_and_B_selected 
  (students : Finset ℕ) (A B : ℕ) 
  (h_students_size : students.card = 5) 
  (h_A_in_students : A ∈ students) 
  (h_B_in_students : B ∈ students)
  : (↑(Nat.choose 3) : ℚ) / (↑(Nat.choose 5) : ℚ) = 3 / 10 := 
by sorry

end probability_A_and_B_selected_l478_478393


namespace min_episodes_to_watch_l478_478005

theorem min_episodes_to_watch (T W H F Sa Su M trip_days total_episodes: ℕ)
  (hW: W = 1) (hTh: H = 1) (hF: F = 1) (hSa: Sa = 2) (hSu: Su = 2) (hMo: M = 0)
  (total_episodes_eq: total_episodes = 60)
  (trip_days_eq: trip_days = 17):
  total_episodes - ((4 * W + 2 * Sa + 1 * M) * (trip_days / 7) + (trip_days % 7) * (W + Sa + Su + Mo)) = 39 := 
by
  sorry

end min_episodes_to_watch_l478_478005


namespace probability_A_and_B_selected_l478_478517

theorem probability_A_and_B_selected :
  let n := 5
  let r := 3
  let total_ways := Nat.choose n r
  let favorable_ways := Nat.choose 3 1
  let probability := favorable_ways / total_ways
  probability = (3 / 10) := 
by
  sorry

end probability_A_and_B_selected_l478_478517


namespace probability_AB_selected_l478_478499

def factorial (n : Nat) : Nat := if n = 0 then 1 else n * factorial (n - 1)

def choose (n k : Nat) : Nat := factorial n / (factorial k * factorial (n - k))

theorem probability_AB_selected (h1 : 5 = 5) (h2 : 3 = 3) : (choose 3 1) / (choose 5 3) = 3 / 10 := by
  sorry

end probability_AB_selected_l478_478499


namespace number_times_frac_eq_cube_l478_478070

theorem number_times_frac_eq_cube (x : ℕ) : x * (1/6)^2 = 6^3 → x = 7776 :=
by
  intro h
  -- skipped proof
  sorry

end number_times_frac_eq_cube_l478_478070


namespace mean_of_xyz_l478_478007

theorem mean_of_xyz (x y z : ℝ) (seven_mean : ℝ)
  (h1 : seven_mean = 45)
  (h2 : (7 * seven_mean + x + y + z) / 10 = 58) :
  (x + y + z) / 3 = 265 / 3 :=
by
  sorry

end mean_of_xyz_l478_478007


namespace range_of_f_l478_478830

def f (x y : ℝ) : ℝ :=
  real.sqrt ((1 + x * y) / (1 + x^2)) + real.sqrt ((1 - x * y) / (1 + y^2))

theorem range_of_f : 
  ∀ (x y : ℝ), 
  0 ≤ x ∧ x ≤ 1 → 0 ≤ y ∧ y ≤ 1 → 
  1 ≤ f x y ∧ f x y ≤ 2 :=
by
  sorry

end range_of_f_l478_478830


namespace ellipse_properties_l478_478726

theorem ellipse_properties:
  ∀ (a c : ℝ) (F1 F2 : ℝ × ℝ),
      2 * a = 4 ∧ -- major axis length is 4
      F1 = (-1, 0) ∧ -- foci F1
      F2 = (1, 0) ∧  -- foci F2
      c = 1 → -- focal distance is 1
      -- conclusion for the standard equation and eccentricity
      (∃ b : ℝ, b^2 = a^2 - c^2 ∧
      (∀ (x y : ℝ), (x, y) ≠ (0, 0) → (x^2 / a^2 + y^2 / (a^2 - c^2) = 1) ) ∧
      (c / a = 1 / 2)) :=
by
  intros a c F1 F2 h,
  cases h with h_major_axis h1,
  cases h1 with h_foci1 h2,
  cases h2 with h_foci2 h3,
  cases h3 with h_c hc_val,
  -- Using h_major_axis, we have 2 * a = 4, so a = 2
  have ha : a = 2 := by {
    rw ← two_mul at h_major_axis,
    exact (mul_right_inj' two_ne_zero).mp h_major_axis 
  },
  -- Using h_foci1 and h_foci2, we have F1 = (-1, 0) and F2 = (1, 0)
  have hc : c = 1 := by exact hc_val,
  -- we can then find b² = a² - c² = 4-1=3
  have hb : b ∃ ℝ, b^2 = a^2 - c^2 := by {
    use sqrt (a^2 - c^2),
    have hsquared := a^2 - c^2,
    exact_mod_cast hsquared.div_eq_mul hc_val, ha,
    exact_mod_cast (ha.pow_two).sub (hc.pow_two)
  },
  use hb,
  have heccentricity := c / a = 1/2 := by {
    rw ha,
    exact one_div_two.add,
  }
  use heccentricity,
  sorry # skipping actual proof

end ellipse_properties_l478_478726


namespace probability_A_and_B_selected_l478_478707

/-- There are 5 students including A and B. The probability of selecting both A and B 
    when 3 students are selected randomly from 5 is 3/10. -/
theorem probability_A_and_B_selected :
  (∃ (students : List String) (h : students.length = 5),
    ∃ (selected : List String) (h' : selected.length = 3),
    "A" ∈ selected ∧ "B" ∈ selected) →
  ∃ (P : ℚ), P = 3 / 10 :=
by
  sorry

end probability_A_and_B_selected_l478_478707


namespace probability_of_A_and_B_selected_l478_478477

variable {α : Type}

/-- From 5 students, we randomly select 3 students to participate in community service work. 
  Let A and B be two specific students among these five. The probability that 
  both A and B are selected is 3 / 10. -/
theorem probability_of_A_and_B_selected (A B : α) (s : Finset α) (h : s.card = 5) :
  (∃ t : Finset α, t ⊆ s ∧ A ∈ t ∧ B ∈ t ∧ (t.card = 3) ∧ (t.card = 3) →
   (@Finset.exists_subset_card_eq α t s 3) →
   (@Finset.exists_subset_card_eq α t (Finset.erase s A) 2) ∧
   (@Finset.exists_subset_card_eq α t (Finset.erase (Finset.erase s A) B) 1)) →
  (3 : ℚ) / 10 :=
by
  sorry -- Proof omitted as per instructions.

end probability_of_A_and_B_selected_l478_478477


namespace borel_functions_l478_478858

open MeasureTheory

noncomputable def is_borel_function {α β : Type*} [MeasurableSpace α] [MeasurableSpace β] (f : α → β) : Prop :=
measurable f

theorem borel_functions :
  (∀ n : ℕ, is_borel_function (λ (x : ℝ), x ^ n)) ∧
  is_borel_function (λ (x : ℝ), max x 0) ∧
  is_borel_function (λ (x : ℝ), max (-x) 0) ∧
  is_borel_function (λ (x : ℝ), max x 0 + max (-x) 0) ∧
  (∀ {n : ℕ}, n ≥ 1 → ∀ f : ℝ^n → ℝ, continuous f → is_borel_function f) :=
by sorry

end borel_functions_l478_478858


namespace prob_select_A_and_B_l478_478648

theorem prob_select_A_and_B (students : Finset ℕ) (A B : ℕ) (hA : A ∈ students) (hB : B ∈ students) (h_card : students.card = 5) :
  (students.choose 3).count {comb | A ∈ comb ∧ B ∈ comb} / (students.choose 3).card = 3 / 10 := sorry

end prob_select_A_and_B_l478_478648


namespace gcd_polynomial_l478_478737

variable (a : ℕ)

def multiple_of_2345 (a : ℕ) := ∃ k : ℕ, a = 2345 * k

theorem gcd_polynomial (h : multiple_of_2345 a) : Nat.gcd (a^2 + 10*a + 25) (a + 5) = a + 5 := 
sorry

end gcd_polynomial_l478_478737


namespace prob_select_A_and_B_l478_478617

theorem prob_select_A_and_B : 
  let total_students := 5
  let to_select := 3
  let favorable_outcomes := Nat.div_fact (Nat.factorial 3) (Nat.factorial 1)
  let total_outcomes := Nat.div_fact (Nat.factorial 5) (Nat.factorial 3 * Nat.factorial (5 - 3))
  (favorable_outcomes : ℚ) / total_outcomes = 3 / 10 :=
by
  sorry

end prob_select_A_and_B_l478_478617


namespace find_y_value_l478_478993

theorem find_y_value :
  (let a1 := 1
   let r1 := (1 : ℝ) / 3
   let sum1 := a1 / (1 - r1)

   let a2 := 1
   let r2 := -(1 : ℝ) / 3
   let sum2 := a2 / (1 - r2)
   
   let product := sum1 * sum2
   let y := (9 : ℝ)
   
   product = (1 / (1 - 1 / y)))
  :=
  let sum1 : ℝ := 1 / (1 - 1 / 3)
  let sum2 : ℝ := 1 / (1 - (-1 / 3))
  let product : ℝ := sum1 * sum2
  product = (1 / (1 - 1 / 9)) :=
sorry

end find_y_value_l478_478993


namespace factorization_of_polynomial_l478_478184

theorem factorization_of_polynomial (x : ℝ) : 16 * x ^ 2 - 40 * x + 25 = (4 * x - 5) ^ 2 := by 
  sorry

end factorization_of_polynomial_l478_478184


namespace probability_A_and_B_selected_l478_478525

theorem probability_A_and_B_selected :
  let n := 5
  let r := 3
  let total_ways := Nat.choose n r
  let favorable_ways := Nat.choose 3 1
  let probability := favorable_ways / total_ways
  probability = (3 / 10) := 
by
  sorry

end probability_A_and_B_selected_l478_478525


namespace max_value_m_l478_478774

theorem max_value_m (m : ℝ) : 
  (∀ x : ℝ, (x^2 - 2 * x - 8 > 0) -> (x < m)) -> m = -2 :=
by
  sorry

end max_value_m_l478_478774


namespace pipe_filling_ratio_l478_478924

theorem pipe_filling_ratio (t_A t_B : ℝ) :
  t_B = 1 ∧ (1 / 4 * (1 / t_A) + 1 / 2) = 1 ∧ t_B = 1 →
  t_A / t_B = 1 / 2 :=
by
  intros h
  cases h with h1 h2
  cases h2 with h3 h4
  rw h1 at *
  rw h4 at *
  sorry

end pipe_filling_ratio_l478_478924


namespace alpha_parallel_beta_l478_478740

-- Defining lines and planes with their properties.
variables {Line Plane : Type} [Distinct Line] [Distinct Plane]
variables (l m : Line) (α β : Plane)

-- Defining predicates for parallel and perpendicular relationships.
def parallel (x y : Plane) : Prop := sorry
def perpendicular (x y : Plane) : Prop := sorry
def parallel_line_plane (l : Line) (α : Plane) : Prop := sorry
def perpendicular_line_plane (l : Line) (α : Plane) : Prop := sorry
def parallel_line_line (l m : Line) : Prop := sorry

-- The given conditions as hypotheses.
axiom h1 : parallel_line_line l m
axiom h2 : perpendicular_line_plane m β
axiom h3 : perpendicular_line_plane l α

-- The statement to prove.
theorem alpha_parallel_beta : parallel α β := sorry

end alpha_parallel_beta_l478_478740


namespace probability_under_20_l478_478065

theorem probability_under_20 (total_people : ℕ) (over_30 : ℕ) (under_20 : ℕ) 
  (h_total : total_people = 100) (h_over_30 : over_30 = 90) (h_under_20 : under_20 = total_people - over_30) : 
  under_20 / total_people = 0.1 := 
by
  sorry

end probability_under_20_l478_478065


namespace probability_A_and_B_selected_l478_478588

/-- Let there be a group of 5 students including A and B. 
    If 3 students are selected randomly from these 5, the probability of selecting both A and B is 3/10. -/
theorem probability_A_and_B_selected : 
  let students := 5 in
  let selected := 3 in
  let combinations := students.choose selected in
  let favorable_combinations := (students - 2).choose (selected - 2) in
  let probability := (favorable_combinations : ℚ) / (combinations : ℚ) in
  probability = 3 / 10 := 
by sorry

end probability_A_and_B_selected_l478_478588


namespace trajectory_eq_hyperbola_slope_range_fixed_point_exists_l478_478712

-- Define the foci points F1 and F2
def F1 := (-2, 0)
def F2 := (2, 0)

-- Define the trajectory condition
def P_condition (P : ℝ × ℝ) : Prop := 
  abs (dist P F1 - dist P F2) = 2

-- Problem 1: Show the equation of the trajectory Γ
theorem trajectory_eq_hyperbola : 
  ∀ P : ℝ × ℝ, P_condition P ↔ P ∈ { p : ℝ × ℝ | p.1^2 - p.2^2 / 3 = 1 ∧ p.1 ≥ 1 } :=
sorry

-- Problem 2: Determine the range of slope k
theorem slope_range (k : ℝ) : 
  k < -real.sqrt 3 ∨ k > real.sqrt 3 :=
sorry

-- Problem 3: Show there exists a fixed point M on the x-axis such that MA ⊥ MB 
theorem fixed_point_exists : 
  ∃ M : ℝ × ℝ, M = (-1, 0) ∧ ∀ (k : ℝ) (A B : ℝ × ℝ), 
  A ≠ B → line_through B F2 k → P_condition A → P_condition B → 
  inner_product (A.1 - M.1, A.2 - M.2) (B.1 - M.1, B.2 - M.2) = 0 :=
sorry

end trajectory_eq_hyperbola_slope_range_fixed_point_exists_l478_478712


namespace Debby_daily_bottles_is_six_l478_478140

def daily_bottles (total_bottles : ℕ) (total_days : ℕ) : ℕ :=
  total_bottles / total_days

theorem Debby_daily_bottles_is_six : daily_bottles 12 2 = 6 := by
  sorry

end Debby_daily_bottles_is_six_l478_478140


namespace probability_A_and_B_selected_l478_478322

open Finset

noncomputable def total_ways : ℕ := (choose 5 3)

noncomputable def favorable_ways : ℕ := (choose 3 1)

noncomputable def probability (total favorable : ℕ) : ℚ := favorable / total

theorem probability_A_and_B_selected :
  probability total_ways favorable_ways = 3 / 10 :=
by
  sorry

end probability_A_and_B_selected_l478_478322


namespace probability_both_A_and_B_selected_l478_478405

-- The universe of students we consider
inductive Student
| A : Student
| B : Student
| C : Student
| D : Student
| E : Student

open Student

-- Function to count combinations
def combinations (n k : ℕ) : ℕ := (Nat.factorial n) / ((Nat.factorial k) * (Nat.factorial (n - k)))

-- The set of all students
def students : List Student := [A, B, C, D, E]

-- The total number of ways to select 3 students from 5
def total_selections := combinations 5 3

-- The number of ways A and B are both selected by selecting 1 more from the remaining 3 students
def selections_AB := combinations 3 1

-- The probability that both A and B are selected
def probability_AB := (selections_AB : ℚ) / (total_selections : ℚ)

-- The proof statement
theorem probability_both_A_and_B_selected :
  probability_AB = 3 / 10 :=
sorry

end probability_both_A_and_B_selected_l478_478405


namespace probability_of_selecting_A_and_B_l478_478555

theorem probability_of_selecting_A_and_B (students : Finset ℕ) (A B : ℕ)
  (h_size : students.card = 5) (h_in : A ∈ students ∧ B ∈ students) : 
  (Finset.card {s : Finset ℕ // s ⊆ students ∧ s.card = 3 ∧ A ∈ s ∧ B ∈ s}.to_set) / 
  (Finset.card {s : Finset ℕ // s ⊆ students ∧ s.card = 3}.to_set) = 3 / 10 :=
sorry

end probability_of_selecting_A_and_B_l478_478555


namespace geom_sequence_common_ratio_l478_478792

variable {α : Type*} [LinearOrderedField α]

theorem geom_sequence_common_ratio (a1 q : α) (h : a1 > 0) (h_eq : a1 + a1 * q + a1 * q^2 + a1 * q = 9 * a1 * q^2) : q = 1 / 2 :=
by sorry

end geom_sequence_common_ratio_l478_478792


namespace factorize_quadratic_l478_478180

theorem factorize_quadratic (x : ℝ) : 
  16 * x^2 - 40 * x + 25 = (4 * x - 5)^2 :=
sorry

end factorize_quadratic_l478_478180


namespace defective_product_probabilities_l478_478980

-- Defining the probabilities of production by machines a, b, and c
def P_H1 := 0.20
def P_H2 := 0.35
def P_H3 := 0.45

-- Defining the defect rates for machines a, b, and c
def P_A_H1 := 0.03
def P_A_H2 := 0.02
def P_A_H3 := 0.04

-- Total Probability of defect
def P_A := (P_A_H1 * P_H1) + (P_A_H2 * P_H2) + (P_A_H3 * P_H3)

-- Conditional probabilities using Bayes' Theorem
def P_H1_A := (P_H1 * P_A_H1) / P_A
def P_H2_A := (P_H2 * P_A_H2) / P_A
def P_H3_A := (P_H3 * P_A_H3) / P_A

-- Main theorem
theorem defective_product_probabilities:
  P_H1_A ≈ 0.1936 ∧ P_H2_A ≈ 0.2258 ∧ P_H3_A ≈ 0.5806 :=
by
  sorry

end defective_product_probabilities_l478_478980


namespace factorization_identity_l478_478171

theorem factorization_identity (x : ℝ) : 
  16 * x^2 - 40 * x + 25 = (4 * x - 5)^2 :=
  sorry

end factorization_identity_l478_478171


namespace sphere_radius_eq_six_l478_478960

-- Conditions stated in the problem
def meter_stick_height : ℝ := 1.5
def meter_stick_shadow : ℝ := 3
def sphere_shadow : ℝ := 12

-- Tangent of the angle derived from the meter stick
def tan_theta : ℝ := meter_stick_height / meter_stick_shadow

-- The problem, restating the question as a proof of equality
theorem sphere_radius_eq_six (r : ℝ) 
  (h1 : tan_theta = meter_stick_height / meter_stick_shadow)
  (h2 : tan_theta = r / sphere_shadow) : 
  r = 6 :=
by 
  -- Placeholder for proof
  sorry

end sphere_radius_eq_six_l478_478960


namespace triangle_angle_sum_l478_478896

theorem triangle_angle_sum (a : ℝ) (x : ℝ) :
  0 < 2 * a + 20 ∧ 0 < 3 * a - 15 ∧ 0 < 175 - 5 * a ∧
  2 * a + 20 + 3 * a - 15 + x = 180 → 
  x = 175 - 5 * a ∧ max (2 * a + 20) (max (3 * a - 15) (175 - 5 * a)) = 88 :=
sorry

end triangle_angle_sum_l478_478896


namespace probability_A_and_B_selected_l478_478309

open Finset

noncomputable def total_ways : ℕ := (choose 5 3)

noncomputable def favorable_ways : ℕ := (choose 3 1)

noncomputable def probability (total favorable : ℕ) : ℚ := favorable / total

theorem probability_A_and_B_selected :
  probability total_ways favorable_ways = 3 / 10 :=
by
  sorry

end probability_A_and_B_selected_l478_478309


namespace f_2019_2020_l478_478739

def f (x : ℝ) : ℝ := if h : (0 < x ∧ x ≤ 1) then sin (real.pi * x / 2) - 1 else 0

@[odd_fun]
axiom odd_f : ∀ x : ℝ, f (-x) = -f x

@[periodic]
axiom periodic_f : ∀ x : ℝ, f (x + 3) = f x

theorem f_2019_2020 : f 2019 + f 2020 = 0 :=
  by sorry

end f_2019_2020_l478_478739


namespace probability_A_and_B_selected_l478_478305

open Finset

noncomputable def total_ways : ℕ := (choose 5 3)

noncomputable def favorable_ways : ℕ := (choose 3 1)

noncomputable def probability (total favorable : ℕ) : ℚ := favorable / total

theorem probability_A_and_B_selected :
  probability total_ways favorable_ways = 3 / 10 :=
by
  sorry

end probability_A_and_B_selected_l478_478305


namespace law_of_motion_l478_478965

variables (a t v_0 s_0 : Real)

-- Define velocity as a function of time given the initial conditions
def velocity (t : Real) := a * t + v_0

-- Define position as a function of time given the initial conditions
def position (t : Real) := (1 / 2) * a * t^2 + v_0 * t + s_0

-- Lean theorem stating that these definitions hold
theorem law_of_motion :
  ∀ t, velocity a t v_0 = a * t + v_0 ∧ position a t v_0 s_0 = (1/2) * a * t^2 + v_0 * t + s_0 := 
by
  intro t
  exact And.intro rfl rfl

end law_of_motion_l478_478965


namespace prob_select_A_and_B_l478_478656

theorem prob_select_A_and_B (students : Finset ℕ) (A B : ℕ) (hA : A ∈ students) (hB : B ∈ students) (h_card : students.card = 5) :
  (students.choose 3).count {comb | A ∈ comb ∧ B ∈ comb} / (students.choose 3).card = 3 / 10 := sorry

end prob_select_A_and_B_l478_478656


namespace probability_A_and_B_selected_l478_478329

theorem probability_A_and_B_selected (A B : Type) (S : Finset (Type)) (h : S.card = 5) (hA : A ∈ S) (hB : B ∈ S) (T : Finset (Type)) (hT : T.card = 3) :
  (∃ T ⊆ S, A ∈ T ∧ B ∈ T ∧ T.card = 3) → 
  ∃ T ⊆ S, T.card = 3 ∧ (finset.card (T ∩ {A, B}) = 2) → 
  (finset.card (T ∩ {A, B}) = 2) = (3 / 10 : Real) := sorry

end probability_A_and_B_selected_l478_478329


namespace angle_relationship_l478_478779

-- Conditions: The faces of two dihedral angles are perpendicular to each other
-- and their edges are parallel.
variables {α β : Type} [DihedralAngle α] [DihedralAngle β]

-- Assume faces are perpendicular
axiom faces_perpendicular : perpendicular (faces α) (faces β)

-- Assume edges are parallel
axiom edges_parallel : parallel (edges α) (edges β)

-- Theorem: The relationship between the angles is either equal or complementary.
theorem angle_relationship : equal (angles α) (angles β) ∨ complementary (angles α) (angles β) :=
sorry

end angle_relationship_l478_478779


namespace collinear_probability_l478_478861

def is_collinear (m n : ℕ) : Prop :=
  n = 2 * m

def valid_outcome (m n : ℕ) : Prop :=
  1 ≤ m ∧ m ≤ 6 ∧ 1 ≤ n ∧ n ≤ 6

theorem collinear_probability :
  (∑ m in finset.range 6, ∑ n in finset.range 6, if valid_outcome (m+1) (n+1) ∧ is_collinear (m+1) (n+1) then 1 else 0).to_real /
  (∑ m in finset.range 6, ∑ n in finset.range 6, if valid_outcome (m+1) (n+1) then 1 else 0).to_real = 1 / 12 :=
sorry

end collinear_probability_l478_478861


namespace probability_of_A_and_B_selected_l478_478490

variable {α : Type}

/-- From 5 students, we randomly select 3 students to participate in community service work. 
  Let A and B be two specific students among these five. The probability that 
  both A and B are selected is 3 / 10. -/
theorem probability_of_A_and_B_selected (A B : α) (s : Finset α) (h : s.card = 5) :
  (∃ t : Finset α, t ⊆ s ∧ A ∈ t ∧ B ∈ t ∧ (t.card = 3) ∧ (t.card = 3) →
   (@Finset.exists_subset_card_eq α t s 3) →
   (@Finset.exists_subset_card_eq α t (Finset.erase s A) 2) ∧
   (@Finset.exists_subset_card_eq α t (Finset.erase (Finset.erase s A) B) 1)) →
  (3 : ℚ) / 10 :=
by
  sorry -- Proof omitted as per instructions.

end probability_of_A_and_B_selected_l478_478490


namespace select_3_from_5_prob_l478_478428

theorem select_3_from_5_prob {students : Finset ℕ} 
  (h : students.card = 5) 
  {A B : ℕ} 
  (hA : A ∈ students) 
  (hB : B ∈ students) 
  (H : 3 ≤ students.card) : 
  (finset.card (students.filter (λ x, x ≠ A ∧ x ≠ B)) = 3) → 
  (finset.card ((students.filter (λ x, x ≠ A ∧ x ≠ B)).subtype (λ x, True)) = 3) → 
  ((finset.card (finset.filter (λ (students : Finset ℕ) A ∈ students ∧ B ∈ students ↔ 3 = students.card) 5)) = ↑10) → 
  ∃ students', (students'.card = 3 ∧ A ∈ students' ∧ B ∈ students') → 
  ∃ (p : ℚ), p = 3 / 10 ∧ students.filter (λ x, x = A ∨ x = B) = 2 :=
  sorry

end select_3_from_5_prob_l478_478428


namespace select_3_from_5_prob_l478_478426

theorem select_3_from_5_prob {students : Finset ℕ} 
  (h : students.card = 5) 
  {A B : ℕ} 
  (hA : A ∈ students) 
  (hB : B ∈ students) 
  (H : 3 ≤ students.card) : 
  (finset.card (students.filter (λ x, x ≠ A ∧ x ≠ B)) = 3) → 
  (finset.card ((students.filter (λ x, x ≠ A ∧ x ≠ B)).subtype (λ x, True)) = 3) → 
  ((finset.card (finset.filter (λ (students : Finset ℕ) A ∈ students ∧ B ∈ students ↔ 3 = students.card) 5)) = ↑10) → 
  ∃ students', (students'.card = 3 ∧ A ∈ students' ∧ B ∈ students') → 
  ∃ (p : ℚ), p = 3 / 10 ∧ students.filter (λ x, x = A ∨ x = B) = 2 :=
  sorry

end select_3_from_5_prob_l478_478426


namespace line_BP_intercepts_AQ_l478_478026

theorem line_BP_intercepts_AQ (A B C D P Q : Point) (n : ℕ) (h_parallelogram : Parallelogram A B C D)
  (h_division : divides AD n) (h_first_division_point : first_division_point P AD n)
  (h_connect_PB : connects P B) (h_diagonal_AC : is_diagonal AC A C) :
  intercepts_segment BP AC AQ → AQ = (1 / (n + 1)) * AC :=
begin
  sorry
end

end line_BP_intercepts_AQ_l478_478026


namespace probability_of_selecting_A_and_B_l478_478550

theorem probability_of_selecting_A_and_B (students : Finset ℕ) (A B : ℕ)
  (h_size : students.card = 5) (h_in : A ∈ students ∧ B ∈ students) : 
  (Finset.card {s : Finset ℕ // s ⊆ students ∧ s.card = 3 ∧ A ∈ s ∧ B ∈ s}.to_set) / 
  (Finset.card {s : Finset ℕ // s ⊆ students ∧ s.card = 3}.to_set) = 3 / 10 :=
sorry

end probability_of_selecting_A_and_B_l478_478550


namespace probability_A_and_B_selected_l478_478534

theorem probability_A_and_B_selected :
  let n := 5
  let r := 3
  let total_ways := Nat.choose n r
  let favorable_ways := Nat.choose 3 1
  let probability := favorable_ways / total_ways
  probability = (3 / 10) := 
by
  sorry

end probability_A_and_B_selected_l478_478534


namespace probability_A_and_B_selected_l478_478229

theorem probability_A_and_B_selected (students : Finset ℕ) (A B : ℕ) (h : A ∈ students ∧ B ∈ students ∧ students.card = 5) : 
  (students.choose 3).count (λ s, {A, B} ⊆ s) / (students.choose 3).card = 3 / 10 := by
  sorry

end probability_A_and_B_selected_l478_478229


namespace probability_A_and_B_selected_l478_478584

/-- Let there be a group of 5 students including A and B. 
    If 3 students are selected randomly from these 5, the probability of selecting both A and B is 3/10. -/
theorem probability_A_and_B_selected : 
  let students := 5 in
  let selected := 3 in
  let combinations := students.choose selected in
  let favorable_combinations := (students - 2).choose (selected - 2) in
  let probability := (favorable_combinations : ℚ) / (combinations : ℚ) in
  probability = 3 / 10 := 
by sorry

end probability_A_and_B_selected_l478_478584


namespace probability_of_selecting_A_and_B_l478_478560

theorem probability_of_selecting_A_and_B (students : Finset ℕ) (A B : ℕ)
  (h_size : students.card = 5) (h_in : A ∈ students ∧ B ∈ students) : 
  (Finset.card {s : Finset ℕ // s ⊆ students ∧ s.card = 3 ∧ A ∈ s ∧ B ∈ s}.to_set) / 
  (Finset.card {s : Finset ℕ // s ⊆ students ∧ s.card = 3}.to_set) = 3 / 10 :=
sorry

end probability_of_selecting_A_and_B_l478_478560


namespace probability_A_and_B_selected_l478_478248

theorem probability_A_and_B_selected (students : Finset ℕ) (A B : ℕ) (h : A ∈ students ∧ B ∈ students ∧ students.card = 5) : 
  (students.choose 3).count (λ s, {A, B} ⊆ s) / (students.choose 3).card = 3 / 10 := by
  sorry

end probability_A_and_B_selected_l478_478248


namespace prob_select_A_and_B_l478_478636

theorem prob_select_A_and_B : 
  let total_students := 5
  let to_select := 3
  let favorable_outcomes := Nat.div_fact (Nat.factorial 3) (Nat.factorial 1)
  let total_outcomes := Nat.div_fact (Nat.factorial 5) (Nat.factorial 3 * Nat.factorial (5 - 3))
  (favorable_outcomes : ℚ) / total_outcomes = 3 / 10 :=
by
  sorry

end prob_select_A_and_B_l478_478636


namespace probability_both_A_and_B_selected_l478_478407

-- The universe of students we consider
inductive Student
| A : Student
| B : Student
| C : Student
| D : Student
| E : Student

open Student

-- Function to count combinations
def combinations (n k : ℕ) : ℕ := (Nat.factorial n) / ((Nat.factorial k) * (Nat.factorial (n - k)))

-- The set of all students
def students : List Student := [A, B, C, D, E]

-- The total number of ways to select 3 students from 5
def total_selections := combinations 5 3

-- The number of ways A and B are both selected by selecting 1 more from the remaining 3 students
def selections_AB := combinations 3 1

-- The probability that both A and B are selected
def probability_AB := (selections_AB : ℚ) / (total_selections : ℚ)

-- The proof statement
theorem probability_both_A_and_B_selected :
  probability_AB = 3 / 10 :=
sorry

end probability_both_A_and_B_selected_l478_478407


namespace correct_operation_l478_478935

theorem correct_operation :
  (∀ a : ℝ, (a^5 * a^3 = a^15) = false) ∧
  (∀ a : ℝ, (a^5 - a^3 = a^2) = false) ∧
  (∀ a : ℝ, ((-a^5)^2 = a^10) = true) ∧
  (∀ a : ℝ, (a^6 / a^3 = a^2) = false) :=
by
  sorry

end correct_operation_l478_478935


namespace probability_of_selecting_A_and_B_l478_478299

open_locale classical

-- Define the number of students
def num_students := 5

-- Define the number of students to be selected
def students_to_select := 3

def combination (n k : ℕ) : ℕ := nat.choose n k

-- Define the correct answer
def correct_probability := 3 / 10

-- Define the probability calculation
def calc_probability := (combination 3 1) / (combination 5 3)

-- Statement of the theorem to be proved
theorem probability_of_selecting_A_and_B :
  calc_probability = correct_probability :=
by sorry

end probability_of_selecting_A_and_B_l478_478299


namespace probability_A_and_B_selected_l478_478343

theorem probability_A_and_B_selected (A B : Type) (S : Finset (Type)) (h : S.card = 5) (hA : A ∈ S) (hB : B ∈ S) (T : Finset (Type)) (hT : T.card = 3) :
  (∃ T ⊆ S, A ∈ T ∧ B ∈ T ∧ T.card = 3) → 
  ∃ T ⊆ S, T.card = 3 ∧ (finset.card (T ∩ {A, B}) = 2) → 
  (finset.card (T ∩ {A, B}) = 2) = (3 / 10 : Real) := sorry

end probability_A_and_B_selected_l478_478343


namespace quadrilateral_area_proof_l478_478115

-- Assume we have a rectangle with area 24 cm^2 and two triangles with total area 7.5 cm^2.
-- We want to prove the area of the quadrilateral ABCD is 16.5 cm^2 inside this rectangle.

def rectangle_area : ℝ := 24
def triangles_area : ℝ := 7.5
def quadrilateral_area : ℝ := rectangle_area - triangles_area

theorem quadrilateral_area_proof : quadrilateral_area = 16.5 := 
by
  exact sorry

end quadrilateral_area_proof_l478_478115


namespace probability_of_selecting_A_and_B_l478_478277

open_locale classical

-- Define the number of students
def num_students := 5

-- Define the number of students to be selected
def students_to_select := 3

def combination (n k : ℕ) : ℕ := nat.choose n k

-- Define the correct answer
def correct_probability := 3 / 10

-- Define the probability calculation
def calc_probability := (combination 3 1) / (combination 5 3)

-- Statement of the theorem to be proved
theorem probability_of_selecting_A_and_B :
  calc_probability = correct_probability :=
by sorry

end probability_of_selecting_A_and_B_l478_478277


namespace select_3_from_5_prob_l478_478442

theorem select_3_from_5_prob {students : Finset ℕ} 
  (h : students.card = 5) 
  {A B : ℕ} 
  (hA : A ∈ students) 
  (hB : B ∈ students) 
  (H : 3 ≤ students.card) : 
  (finset.card (students.filter (λ x, x ≠ A ∧ x ≠ B)) = 3) → 
  (finset.card ((students.filter (λ x, x ≠ A ∧ x ≠ B)).subtype (λ x, True)) = 3) → 
  ((finset.card (finset.filter (λ (students : Finset ℕ) A ∈ students ∧ B ∈ students ↔ 3 = students.card) 5)) = ↑10) → 
  ∃ students', (students'.card = 3 ∧ A ∈ students' ∧ B ∈ students') → 
  ∃ (p : ℚ), p = 3 / 10 ∧ students.filter (λ x, x = A ∨ x = B) = 2 :=
  sorry

end select_3_from_5_prob_l478_478442


namespace probability_A_and_B_selected_l478_478321

open Finset

noncomputable def total_ways : ℕ := (choose 5 3)

noncomputable def favorable_ways : ℕ := (choose 3 1)

noncomputable def probability (total favorable : ℕ) : ℚ := favorable / total

theorem probability_A_and_B_selected :
  probability total_ways favorable_ways = 3 / 10 :=
by
  sorry

end probability_A_and_B_selected_l478_478321


namespace probability_A_and_B_selected_l478_478234

theorem probability_A_and_B_selected (students : Finset ℕ) (A B : ℕ) (h : A ∈ students ∧ B ∈ students ∧ students.card = 5) : 
  (students.choose 3).count (λ s, {A, B} ⊆ s) / (students.choose 3).card = 3 / 10 := by
  sorry

end probability_A_and_B_selected_l478_478234


namespace probability_of_selecting_A_and_B_l478_478553

theorem probability_of_selecting_A_and_B (students : Finset ℕ) (A B : ℕ)
  (h_size : students.card = 5) (h_in : A ∈ students ∧ B ∈ students) : 
  (Finset.card {s : Finset ℕ // s ⊆ students ∧ s.card = 3 ∧ A ∈ s ∧ B ∈ s}.to_set) / 
  (Finset.card {s : Finset ℕ // s ⊆ students ∧ s.card = 3}.to_set) = 3 / 10 :=
sorry

end probability_of_selecting_A_and_B_l478_478553


namespace factorization_of_polynomial_l478_478187

theorem factorization_of_polynomial (x : ℝ) : 16 * x ^ 2 - 40 * x + 25 = (4 * x - 5) ^ 2 := by 
  sorry

end factorization_of_polynomial_l478_478187


namespace probability_of_selecting_A_and_B_l478_478556

theorem probability_of_selecting_A_and_B (students : Finset ℕ) (A B : ℕ)
  (h_size : students.card = 5) (h_in : A ∈ students ∧ B ∈ students) : 
  (Finset.card {s : Finset ℕ // s ⊆ students ∧ s.card = 3 ∧ A ∈ s ∧ B ∈ s}.to_set) / 
  (Finset.card {s : Finset ℕ // s ⊆ students ∧ s.card = 3}.to_set) = 3 / 10 :=
sorry

end probability_of_selecting_A_and_B_l478_478556


namespace probability_of_A_and_B_selected_l478_478487

variable {α : Type}

/-- From 5 students, we randomly select 3 students to participate in community service work. 
  Let A and B be two specific students among these five. The probability that 
  both A and B are selected is 3 / 10. -/
theorem probability_of_A_and_B_selected (A B : α) (s : Finset α) (h : s.card = 5) :
  (∃ t : Finset α, t ⊆ s ∧ A ∈ t ∧ B ∈ t ∧ (t.card = 3) ∧ (t.card = 3) →
   (@Finset.exists_subset_card_eq α t s 3) →
   (@Finset.exists_subset_card_eq α t (Finset.erase s A) 2) ∧
   (@Finset.exists_subset_card_eq α t (Finset.erase (Finset.erase s A) B) 1)) →
  (3 : ℚ) / 10 :=
by
  sorry -- Proof omitted as per instructions.

end probability_of_A_and_B_selected_l478_478487


namespace first_bell_weight_l478_478835

-- Given conditions from the problem
variable (x : ℕ) -- weight of the first bell in pounds
variable (total_weight : ℕ)

-- The condition as the sum of the weights
def bronze_weights (x total_weight : ℕ) : Prop :=
  x + 2 * x + 8 * 2 * x = total_weight

-- Prove that the weight of the first bell is 50 pounds given the total weight is 550 pounds
theorem first_bell_weight : bronze_weights x 550 → x = 50 := by
  intro h
  sorry

end first_bell_weight_l478_478835


namespace probability_both_A_B_selected_l478_478227

-- Define the problem with the necessary conditions.
def probability_AB_selected (total_students : ℕ) (select_students : ℕ) (A B : ℕ) : ℚ :=
  if A < total_students ∧ B < total_students ∧ total_students = 5 ∧ select_students = 3 then
    let total_ways := (Nat.factorial total_students) / ((Nat.factorial select_students) * (Nat.factorial (total_students - select_students)))
    let favorable_ways := (Nat.factorial (total_students - 2)) / ((Nat.factorial (select_students - 2)) * (Nat.factorial ((total_students - 2) - (select_students - 2))))
    favorable_ways / total_ways
  else 0

theorem probability_both_A_B_selected 
  (total_students : ℕ) (select_students : ℕ) (A B : ℕ) 
  (h1 : total_students = 5) (h2 : select_students = 3) 
  (h3 : A < total_students) (h4 : B < total_students) 
  : probability_AB_selected total_students select_students A B = 3 / 10 := 
by {
  -- Insert the delta and logic to prove the theorem here.
  sorry
}

end probability_both_A_B_selected_l478_478227


namespace probability_A_and_B_selected_l478_478324

open Finset

noncomputable def total_ways : ℕ := (choose 5 3)

noncomputable def favorable_ways : ℕ := (choose 3 1)

noncomputable def probability (total favorable : ℕ) : ℚ := favorable / total

theorem probability_A_and_B_selected :
  probability total_ways favorable_ways = 3 / 10 :=
by
  sorry

end probability_A_and_B_selected_l478_478324


namespace max_value_expression_l478_478820

open Complex

-- Conditions
variable (α β ω : ℂ)
variable (h1 : |β| = 1)
variable (h2 : β * conj α ≠ ω)
variable (h3 : |ω| < 1)

-- Statement
theorem max_value_expression :
  ∃ M, M = |(ω + 1) / 2| ∧ ∀ α β, |β| = 1 → β * conj α ≠ ω → |ω| < 1 → 
  ∀ z, z = (ω * β - α) / (1 - conj α * β) → |z| ≤ M :=
sorry

end max_value_expression_l478_478820


namespace profit_for_110_oranges_is_21_r_l478_478064

variable (r : ℝ)

-- Definition that represents the conditions
def cost_price (n : ℝ) := (10 * r) / 11 * n / 11
def sell_price (n : ℝ) := (11 * r) / 10 * n / 10
def profit (cost_price : ℝ) (sell_price : ℝ) := sell_price - cost_price

-- Proof statement
theorem profit_for_110_oranges_is_21_r : profit (cost_price 110) (sell_price 110) = 21 * r :=
by
  sorry

end profit_for_110_oranges_is_21_r_l478_478064


namespace positive_number_from_square_roots_l478_478785

theorem positive_number_from_square_roots (x n : ℕ) (h1 : (x + 1) = real.sqrt n) (h2 : (4 - 2 * x) = real.sqrt n) : n = 36 := by
    sorry

end positive_number_from_square_roots_l478_478785


namespace probability_A_and_B_selected_l478_478700

/-- There are 5 students including A and B. The probability of selecting both A and B 
    when 3 students are selected randomly from 5 is 3/10. -/
theorem probability_A_and_B_selected :
  (∃ (students : List String) (h : students.length = 5),
    ∃ (selected : List String) (h' : selected.length = 3),
    "A" ∈ selected ∧ "B" ∈ selected) →
  ∃ (P : ℚ), P = 3 / 10 :=
by
  sorry

end probability_A_and_B_selected_l478_478700


namespace probability_A_and_B_selected_l478_478318

open Finset

noncomputable def total_ways : ℕ := (choose 5 3)

noncomputable def favorable_ways : ℕ := (choose 3 1)

noncomputable def probability (total favorable : ℕ) : ℚ := favorable / total

theorem probability_A_and_B_selected :
  probability total_ways favorable_ways = 3 / 10 :=
by
  sorry

end probability_A_and_B_selected_l478_478318


namespace positive_number_is_36_l478_478782

theorem positive_number_is_36
  (x : ℝ) 
  (h : (x + 1) = (4 - 2x)) 
  (y : ℝ) 
  (hx1 : y = x + 1) 
  (hx2 : y = 4 - 2x) : 
  y^2 = 36 :=
by 
  -- Proof omitted
  sorry

end positive_number_is_36_l478_478782


namespace prod_three_consec_cubemultiple_of_504_l478_478911

theorem prod_three_consec_cubemultiple_of_504 (a : ℤ) : (a^3 - 1) * a^3 * (a^3 + 1) % 504 = 0 := by
  sorry

end prod_three_consec_cubemultiple_of_504_l478_478911


namespace probability_A_and_B_selected_l478_478527

theorem probability_A_and_B_selected :
  let n := 5
  let r := 3
  let total_ways := Nat.choose n r
  let favorable_ways := Nat.choose 3 1
  let probability := favorable_ways / total_ways
  probability = (3 / 10) := 
by
  sorry

end probability_A_and_B_selected_l478_478527


namespace probability_of_selecting_A_and_B_l478_478289

open_locale classical

-- Define the number of students
def num_students := 5

-- Define the number of students to be selected
def students_to_select := 3

def combination (n k : ℕ) : ℕ := nat.choose n k

-- Define the correct answer
def correct_probability := 3 / 10

-- Define the probability calculation
def calc_probability := (combination 3 1) / (combination 5 3)

-- Statement of the theorem to be proved
theorem probability_of_selecting_A_and_B :
  calc_probability = correct_probability :=
by sorry

end probability_of_selecting_A_and_B_l478_478289


namespace cube_edge_length_l478_478786

theorem cube_edge_length (a : ℝ) (h : 6 * a^2 = 24) : a = 2 :=
by sorry

end cube_edge_length_l478_478786


namespace find_Breadth1_l478_478949

-- Definitions
variables (Depth1 Length1 Breadth1 Depth2 Length2 Breadth2 : ℕ) (Time1 Time2 : ℕ)
-- Conditions
def condition1 : Depth1 = 100 := by rfl
def condition2 : Length1 = 25 := by rfl
def condition3 : Time1 = 12 := by rfl
def condition4 : Depth2 = 75 := by rfl
def condition5 : Length2 = 20 := by rfl
def condition6 : Breadth2 = 50 := by rfl
def condition7 : Time2 = 12 := by rfl

-- Problem Statement
theorem find_Breadth1 (Condition1 : Depth1 = 100) (Condition2 : Length1 = 25)
  (Condition3 : Time1 = 12) (Condition4 : Depth2 = 75) (Condition5 : Length2 = 20)
  (Condition6 : Breadth2 = 50) (Condition7 : Time2 = 12) : Breadth1 = 30 := 
sorry

end find_Breadth1_l478_478949


namespace probability_A_and_B_selected_l478_478570

/-- Let there be a group of 5 students including A and B. 
    If 3 students are selected randomly from these 5, the probability of selecting both A and B is 3/10. -/
theorem probability_A_and_B_selected : 
  let students := 5 in
  let selected := 3 in
  let combinations := students.choose selected in
  let favorable_combinations := (students - 2).choose (selected - 2) in
  let probability := (favorable_combinations : ℚ) / (combinations : ℚ) in
  probability = 3 / 10 := 
by sorry

end probability_A_and_B_selected_l478_478570


namespace max_fractions_with_integer_values_l478_478044

noncomputable def max_integer_valued_fractions (S : Set ℕ) : ℕ :=
  let fractions := { (a, b) | a ∈ S ∧ b ∈ S ∧ b ≠ 0 ∧ a % b = 0 }
  in fractions.to_finset.card

theorem max_fractions_with_integer_values (S : Set ℕ) (hS : S = {1, 2, ..., 22}) : max_integer_valued_fractions S = 10 :=
  sorry

end max_fractions_with_integer_values_l478_478044


namespace part_I_part_II_l478_478833
noncomputable section

open Real

/-- Given \( \cos B = \frac{3}{5} \), \( b = 2 \), and \( A = 30^{\circ} \),
    we want to show that \( a = \frac{5}{4} \) -/
theorem part_I (a b c : ℝ) (A B C : ℝ) (hB : cos B = 3/5) (hb : b = 2) (hA : A = π / 6) : 
  a = 5 / 4 :=
sorry

/-- Given \( \cos B = \frac{3}{5} \), \( b = 2 \), and the area of \( \triangle ABC \) is 3,
    we want to show that \( a + c = 2 \sqrt {7} \) -/
theorem part_II (a b c : ℝ) (A B C : ℝ) (hB : cos B = 3/5) (hb : b = 2) (hArea : 1 / 2 * a * c * sqrt(1 - (3/5)^2) = 3) : 
  a + c = 2 * sqrt 7 :=
sorry

end part_I_part_II_l478_478833


namespace composite_infinite_sequence_exists_l478_478110

-- Definitions as per conditions
def infinite_sequence (p : ℕ → ℕ) : Prop :=
  ∀ i, last_digit (p (i + 1)) ≠ 9 ∧ p i = omit_last_digit (p (i + 1))

-- Auxiliary functions for last digit and omitting last digit
def last_digit (n : ℕ) : ℕ := n % 10

def omit_last_digit (n : ℕ) : ℕ := n / 10

-- Main theorem that needs to be proven
theorem composite_infinite_sequence_exists (p : ℕ → ℕ) 
  (h : infinite_sequence p) : 
  ∃   infinitly_many_composite_numbers_in_sequence :=
sorry

end composite_infinite_sequence_exists_l478_478110


namespace probability_A_and_B_selected_l478_478315

open Finset

noncomputable def total_ways : ℕ := (choose 5 3)

noncomputable def favorable_ways : ℕ := (choose 3 1)

noncomputable def probability (total favorable : ℕ) : ℚ := favorable / total

theorem probability_A_and_B_selected :
  probability total_ways favorable_ways = 3 / 10 :=
by
  sorry

end probability_A_and_B_selected_l478_478315


namespace find_a_from_complex_condition_l478_478747

theorem find_a_from_complex_condition (a : ℝ) (x y : ℝ) 
  (h : x = -1 ∧ y = -2 * a)
  (h_line : x - y = 0) : a = 1 / 2 :=
by
  sorry

end find_a_from_complex_condition_l478_478747


namespace microwave_sales_l478_478999

theorem microwave_sales (p c : ℝ) (k : ℝ) (p' : ℝ) :
  (p * c = k) ∧ (p = 15) ∧ (c = 300) ∧ (k = 4500) ∧ (c' = 600 * 0.8) ∧ (c' = 480) 
  → round(p' * 480) = round(k) :=
by sorry

end microwave_sales_l478_478999


namespace felix_distance_l478_478886

theorem felix_distance : 
  ∀ (avg_speed: ℕ) (time: ℕ) (factor: ℕ), 
  avg_speed = 66 → factor = 2 → time = 4 → (factor * avg_speed * time = 528) := 
by
  intros avg_speed time factor h_avg_speed h_factor h_time
  rw [h_avg_speed, h_factor, h_time]
  norm_num
  sorry

end felix_distance_l478_478886


namespace probability_of_selecting_A_and_B_l478_478671

open_locale classical

noncomputable def select_probability (n k : ℕ) : ℚ :=
  (nat.choose n k)⁻¹

theorem probability_of_selecting_A_and_B :
  let total_students := 5,
      students_to_select := 3,
      remaining_students := 3 in
  select_probability total_students students_to_select * nat.choose remaining_students 1 = 3 / 10 :=
by
  sorry

end probability_of_selecting_A_and_B_l478_478671


namespace probability_of_selecting_A_and_B_is_three_tenths_l478_478601

-- Define the set of students
def students : Finset ℕ := {0, 1, 2, 3, 4}

-- Define subsets A and B
def student_A : ℕ := 0
def student_B : ℕ := 1

-- Total number of ways to select 3 students out of 5
def total_ways : ℕ := students.card.choose 3

-- Number of ways to select 3 students including A and B
def favorable_ways : ℕ := {2, 3, 4}.card.choose 1

-- The probability
def probability_of_selecting_A_and_B : ℚ := (favorable_ways : ℚ) / (total_ways : ℚ)

theorem probability_of_selecting_A_and_B_is_three_tenths :
  probability_of_selecting_A_and_B = 3 / 10 :=
by
  sorry

end probability_of_selecting_A_and_B_is_three_tenths_l478_478601


namespace find_locus_of_M_l478_478034

-- Define collinearity property for three points
def collinear (A B C : Point) : Prop :=
  ∃ l : Line, A ∈ l ∧ B ∈ l ∧ C ∈ l

-- Define the circle centered at point with radius
def circle (center : Point) (radius : ℝ) : Set Point :=
  { P | dist P center = radius }

-- The main statement
theorem find_locus_of_M (A B C M : Point) (r : ℝ) 
  (h_collinear : collinear A B C) (h_between : B between A and C)
  (h_circle_center : circle B r) 
  (h_tangents_intersect : ∃ θ : ℝ, is_tangent A M θ r ∧ is_tangent C M θ r) :
  ∃ K : Point, M ∈ arc_of_circle B K ∧ arc_constrained (line_perpendicular_through A (A - C)) (line_perpendicular_through C (A - C)) :=
sorry

end find_locus_of_M_l478_478034


namespace probability_of_A_and_B_selected_l478_478471

variable {α : Type}

/-- From 5 students, we randomly select 3 students to participate in community service work. 
  Let A and B be two specific students among these five. The probability that 
  both A and B are selected is 3 / 10. -/
theorem probability_of_A_and_B_selected (A B : α) (s : Finset α) (h : s.card = 5) :
  (∃ t : Finset α, t ⊆ s ∧ A ∈ t ∧ B ∈ t ∧ (t.card = 3) ∧ (t.card = 3) →
   (@Finset.exists_subset_card_eq α t s 3) →
   (@Finset.exists_subset_card_eq α t (Finset.erase s A) 2) ∧
   (@Finset.exists_subset_card_eq α t (Finset.erase (Finset.erase s A) B) 1)) →
  (3 : ℚ) / 10 :=
by
  sorry -- Proof omitted as per instructions.

end probability_of_A_and_B_selected_l478_478471


namespace value_of_f_at_neg_one_l478_478761

noncomputable def g (x : ℝ) : ℝ := 2 - 3 * x^2

noncomputable def f (x : ℝ) (h : x ≠ 0) : ℝ := (2 - 3 * x^2) / x^2

theorem value_of_f_at_neg_one : f (-1) (by norm_num) = -1 := 
sorry

end value_of_f_at_neg_one_l478_478761


namespace probability_both_A_B_selected_l478_478218

-- Define the problem with the necessary conditions.
def probability_AB_selected (total_students : ℕ) (select_students : ℕ) (A B : ℕ) : ℚ :=
  if A < total_students ∧ B < total_students ∧ total_students = 5 ∧ select_students = 3 then
    let total_ways := (Nat.factorial total_students) / ((Nat.factorial select_students) * (Nat.factorial (total_students - select_students)))
    let favorable_ways := (Nat.factorial (total_students - 2)) / ((Nat.factorial (select_students - 2)) * (Nat.factorial ((total_students - 2) - (select_students - 2))))
    favorable_ways / total_ways
  else 0

theorem probability_both_A_B_selected 
  (total_students : ℕ) (select_students : ℕ) (A B : ℕ) 
  (h1 : total_students = 5) (h2 : select_students = 3) 
  (h3 : A < total_students) (h4 : B < total_students) 
  : probability_AB_selected total_students select_students A B = 3 / 10 := 
by {
  -- Insert the delta and logic to prove the theorem here.
  sorry
}

end probability_both_A_B_selected_l478_478218


namespace probability_of_selecting_A_and_B_l478_478268

-- Define the combination function
def combination (n r : ℕ) : ℕ := Nat.factorial n / (Nat.factorial r * Nat.factorial (n - r))

-- Define the conditions
def students : ℕ := 5
def to_select : ℕ := 3

-- Define the favorable outcomes where A and B are selected
def favorable_outcomes : ℕ := combination 3 1

-- Define the total outcomes of selecting 3 out of 5 students
def total_outcomes : ℕ := combination 5 3

-- Define the probability calculation
def probability_of_A_and_B : ℚ := favorable_outcomes / total_outcomes

-- Proposition to prove
theorem probability_of_selecting_A_and_B : probability_of_A_and_B = 3 / 10 :=
by
  sorry

end probability_of_selecting_A_and_B_l478_478268


namespace solve_logarithmic_equation_evaluate_expression_l478_478945

/-- Prove the equivalence of logarithmic expressions given certain conditions -/
theorem solve_logarithmic_equation (a : ℝ) (h_pos : a > 0) (h_neq_one : a ≠ 1) :
    (∀ x : ℝ, 3 * x - 1 > 0 ∧ x - 1 > 0 ∧ 3 + x > 0 →
    log a (3 * x - 1) = log a (x - 1) + log a (3 + x)) ↔ (∀ x : ℝ, (3 * x - 1 = (x - 1) * (3 + x))) :=
sorry

/-- Evaluate the given expression involving logarithms and exponents -/
theorem evaluate_expression :
    lg 5 + lg 2 - (-1 / 3) ^ (-2 : ℝ) + (sqrt 2 - 1) ^ 0 + log 2 8 = 
    log 10 5 + log 10 2 - 9 + 1 + 3 :=
sorry

end solve_logarithmic_equation_evaluate_expression_l478_478945


namespace next_sales_amount_l478_478102

theorem next_sales_amount (profit1 : ℝ) (sales1 : ℝ) (profit2 : ℝ) (increase_percent : ℝ) :
  profit1 = 5 → sales1 = 15 → profit2 = 12 → increase_percent = 20.00000000000001 →
  let S := (profit2 * sales1) / (profit1 * (1 + increase_percent / 100)) in
  S = 30 :=
by
  intros h1 h2 h3 h4
  let S := (profit2 * sales1) / (profit1 * (1 + increase_percent / 100))
  sorry

end next_sales_amount_l478_478102


namespace basketball_free_throws_l478_478899

/-
Given the following conditions:
1. The players scored twice as many points with three-point shots as with two-point shots: \( 3b = 2a \).
2. The number of successful free throws was one more than the number of successful two-point shots: \( x = a + 1 \).
3. The team’s total score was 84 points: \( 2a + 3b + x = 84 \).

Prove that the number of free throws \( x \) equals 16.
-/
theorem basketball_free_throws (a b x : ℕ) 
  (h1 : 3 * b = 2 * a) 
  (h2 : x = a + 1) 
  (h3 : 2 * a + 3 * b + x = 84) : 
  x = 16 := 
  sorry

end basketball_free_throws_l478_478899


namespace probability_of_A_and_B_selected_l478_478482

variable {α : Type}

/-- From 5 students, we randomly select 3 students to participate in community service work. 
  Let A and B be two specific students among these five. The probability that 
  both A and B are selected is 3 / 10. -/
theorem probability_of_A_and_B_selected (A B : α) (s : Finset α) (h : s.card = 5) :
  (∃ t : Finset α, t ⊆ s ∧ A ∈ t ∧ B ∈ t ∧ (t.card = 3) ∧ (t.card = 3) →
   (@Finset.exists_subset_card_eq α t s 3) →
   (@Finset.exists_subset_card_eq α t (Finset.erase s A) 2) ∧
   (@Finset.exists_subset_card_eq α t (Finset.erase (Finset.erase s A) B) 1)) →
  (3 : ℚ) / 10 :=
by
  sorry -- Proof omitted as per instructions.

end probability_of_A_and_B_selected_l478_478482


namespace probability_A_and_B_selected_l478_478524

theorem probability_A_and_B_selected :
  let n := 5
  let r := 3
  let total_ways := Nat.choose n r
  let favorable_ways := Nat.choose 3 1
  let probability := favorable_ways / total_ways
  probability = (3 / 10) := 
by
  sorry

end probability_A_and_B_selected_l478_478524


namespace unique_positive_integer_divisibility_l478_478204

theorem unique_positive_integer_divisibility (n : ℕ) (h : n > 0) : 
  (5^(n-1) + 3^(n-1)) ∣ (5^n + 3^n) ↔ n = 1 :=
by
  sorry

end unique_positive_integer_divisibility_l478_478204


namespace probability_A_B_selected_l478_478353

noncomputable def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

def combination (n r : ℕ) : ℕ :=
  factorial n / (factorial r * factorial (n - r))

theorem probability_A_B_selected :
  let total_ways := combination 5 3 in
  let ways_A_B := combination 3 1 in
  (ways_A_B : ℚ) / total_ways = 3 / 10 :=
by
  sorry

end probability_A_B_selected_l478_478353


namespace probability_AB_selected_l478_478509

def factorial (n : Nat) : Nat := if n = 0 then 1 else n * factorial (n - 1)

def choose (n k : Nat) : Nat := factorial n / (factorial k * factorial (n - k))

theorem probability_AB_selected (h1 : 5 = 5) (h2 : 3 = 3) : (choose 3 1) / (choose 5 3) = 3 / 10 := by
  sorry

end probability_AB_selected_l478_478509


namespace prob_select_A_and_B_l478_478623

theorem prob_select_A_and_B : 
  let total_students := 5
  let to_select := 3
  let favorable_outcomes := Nat.div_fact (Nat.factorial 3) (Nat.factorial 1)
  let total_outcomes := Nat.div_fact (Nat.factorial 5) (Nat.factorial 3 * Nat.factorial (5 - 3))
  (favorable_outcomes : ℚ) / total_outcomes = 3 / 10 :=
by
  sorry

end prob_select_A_and_B_l478_478623


namespace solution_l478_478787

noncomputable def mathProblem : Prop :=
  let a := 13
  let b := 0
  let c := 2
  (sqrt 3 + 1 / sqrt 3 + sqrt 8 + 1 / sqrt 8) = (13 * sqrt 3 + 0 * sqrt 8) / 2

theorem solution : mathProblem → 15 := by
  intro h
  sorry

end solution_l478_478787


namespace probability_A_and_B_selected_l478_478569

/-- Let there be a group of 5 students including A and B. 
    If 3 students are selected randomly from these 5, the probability of selecting both A and B is 3/10. -/
theorem probability_A_and_B_selected : 
  let students := 5 in
  let selected := 3 in
  let combinations := students.choose selected in
  let favorable_combinations := (students - 2).choose (selected - 2) in
  let probability := (favorable_combinations : ℚ) / (combinations : ℚ) in
  probability = 3 / 10 := 
by sorry

end probability_A_and_B_selected_l478_478569


namespace probability_A_and_B_selected_l478_478575

/-- Let there be a group of 5 students including A and B. 
    If 3 students are selected randomly from these 5, the probability of selecting both A and B is 3/10. -/
theorem probability_A_and_B_selected : 
  let students := 5 in
  let selected := 3 in
  let combinations := students.choose selected in
  let favorable_combinations := (students - 2).choose (selected - 2) in
  let probability := (favorable_combinations : ℚ) / (combinations : ℚ) in
  probability = 3 / 10 := 
by sorry

end probability_A_and_B_selected_l478_478575


namespace probability_of_selecting_A_and_B_l478_478271

-- Define the combination function
def combination (n r : ℕ) : ℕ := Nat.factorial n / (Nat.factorial r * Nat.factorial (n - r))

-- Define the conditions
def students : ℕ := 5
def to_select : ℕ := 3

-- Define the favorable outcomes where A and B are selected
def favorable_outcomes : ℕ := combination 3 1

-- Define the total outcomes of selecting 3 out of 5 students
def total_outcomes : ℕ := combination 5 3

-- Define the probability calculation
def probability_of_A_and_B : ℚ := favorable_outcomes / total_outcomes

-- Proposition to prove
theorem probability_of_selecting_A_and_B : probability_of_A_and_B = 3 / 10 :=
by
  sorry

end probability_of_selecting_A_and_B_l478_478271


namespace class_percentage_of_girls_l478_478905

/-
Given:
- Initial number of boys in the class: 11
- Number of girls in the class: 13
- 1 boy is added to the class, resulting in the new total number of boys being 12

Prove:
- The percentage of the class that are girls is 52%.
-/
theorem class_percentage_of_girls (initial_boys : ℕ) (girls : ℕ) (added_boy : ℕ)
  (new_boy_total : ℕ) (total_students : ℕ) (percent_girls : ℕ) (h1 : initial_boys = 11) 
  (h2 : girls = 13) (h3 : added_boy = 1) (h4 : new_boy_total = initial_boys + added_boy) 
  (h5 : total_students = new_boy_total + girls) 
  (h6 : percent_girls = (girls * 100) / total_students) : percent_girls = 52 :=
sorry

end class_percentage_of_girls_l478_478905


namespace probability_of_A_and_B_selected_l478_478485

variable {α : Type}

/-- From 5 students, we randomly select 3 students to participate in community service work. 
  Let A and B be two specific students among these five. The probability that 
  both A and B are selected is 3 / 10. -/
theorem probability_of_A_and_B_selected (A B : α) (s : Finset α) (h : s.card = 5) :
  (∃ t : Finset α, t ⊆ s ∧ A ∈ t ∧ B ∈ t ∧ (t.card = 3) ∧ (t.card = 3) →
   (@Finset.exists_subset_card_eq α t s 3) →
   (@Finset.exists_subset_card_eq α t (Finset.erase s A) 2) ∧
   (@Finset.exists_subset_card_eq α t (Finset.erase (Finset.erase s A) B) 1)) →
  (3 : ℚ) / 10 :=
by
  sorry -- Proof omitted as per instructions.

end probability_of_A_and_B_selected_l478_478485


namespace probability_of_selecting_A_and_B_l478_478668

open_locale classical

noncomputable def select_probability (n k : ℕ) : ℚ :=
  (nat.choose n k)⁻¹

theorem probability_of_selecting_A_and_B :
  let total_students := 5,
      students_to_select := 3,
      remaining_students := 3 in
  select_probability total_students students_to_select * nat.choose remaining_students 1 = 3 / 10 :=
by
  sorry

end probability_of_selecting_A_and_B_l478_478668


namespace non_indian_percentage_l478_478068

theorem non_indian_percentage (total_men total_women total_children : ℕ)
  (perc_indian_men perc_indian_women perc_indian_children : ℝ) :
  total_men = 500 →
  total_women = 300 →
  total_children = 500 →
  perc_indian_men = 0.10 →
  perc_indian_women = 0.60 →
  perc_indian_children = 0.70 →
  ((total_men + total_women + total_children) - (perc_indian_men * total_men + perc_indian_women * total_women + perc_indian_children * total_children)) / (total_men + total_women + total_children) * 100 ≈ 55.38 :=
by
  sorry

end non_indian_percentage_l478_478068


namespace probability_of_selecting_A_and_B_l478_478557

theorem probability_of_selecting_A_and_B (students : Finset ℕ) (A B : ℕ)
  (h_size : students.card = 5) (h_in : A ∈ students ∧ B ∈ students) : 
  (Finset.card {s : Finset ℕ // s ⊆ students ∧ s.card = 3 ∧ A ∈ s ∧ B ∈ s}.to_set) / 
  (Finset.card {s : Finset ℕ // s ⊆ students ∧ s.card = 3}.to_set) = 3 / 10 :=
sorry

end probability_of_selecting_A_and_B_l478_478557


namespace two_points_P_for_given_area_max_area_incircle_abf1_l478_478751

-- Define the ellipse equation
def ellipse (x y : ℝ) := x^2 / 2 + y^2 = 1

-- Define the right focus F2
def F2 := (1 : ℝ, 0 : ℝ)

-- Define line l with a slope of 1 passing through F2
def line_l (x y : ℝ) := x - y - 1 = 0

-- Define intersection points A and B for the line and ellipse
def A := (0 : ℝ, -1 : ℝ)
def B := (4 / 3 : ℝ, 1 / 3 : ℝ)

-- Define the given area of triangle ABP
def given_area : ℝ := (2 * Real.sqrt 5 - 2) / 3

-- Define the maximum area of the incircle of triangle ABF1 and the line equation at this time
def max_incircle_area : ℝ := Real.pi / 8
def optimal_line_eq (x y : ℝ) := x = 1

-- Problem statements in Lean
theorem two_points_P_for_given_area : ∃ (P : ℝ × ℝ), ellipse P.1 P.2 ∧ (area_triangle A B P) = given_area :=
sorry

theorem max_area_incircle_abf1 : 
  ∃ (R : ℝ), (R = Real.sqrt 2 / 4) ∧ (max_incircle_area) ∧ (optimal_line_eq) :=
sorry

end two_points_P_for_given_area_max_area_incircle_abf1_l478_478751


namespace max_AB_CD_ratio_l478_478151

noncomputable def circle_radius : ℕ := 7

def on_circle (p : ℤ × ℤ) : Prop := p.1^2 + p.2^2 = circle_radius^2

def distance (p1 p2 : ℤ × ℤ) : ℝ :=
  real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem max_AB_CD_ratio :
    ∀ (A B C D : ℤ × ℤ),
    A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D ∧
    on_circle A ∧ on_circle B ∧ on_circle C ∧ on_circle D ∧
    ¬ integer (distance A B) ∧ ¬ integer (distance C D) →
    (real.to_nnreal (distance A B) / real.to_nnreal (distance C D) <= 1) :=
begin
  sorry
end

end max_AB_CD_ratio_l478_478151


namespace probability_A_and_B_selected_l478_478235

theorem probability_A_and_B_selected (students : Finset ℕ) (A B : ℕ) (h : A ∈ students ∧ B ∈ students ∧ students.card = 5) : 
  (students.choose 3).count (λ s, {A, B} ⊆ s) / (students.choose 3).card = 3 / 10 := by
  sorry

end probability_A_and_B_selected_l478_478235


namespace prob_select_A_and_B_l478_478645

theorem prob_select_A_and_B (students : Finset ℕ) (A B : ℕ) (hA : A ∈ students) (hB : B ∈ students) (h_card : students.card = 5) :
  (students.choose 3).count {comb | A ∈ comb ∧ B ∈ comb} / (students.choose 3).card = 3 / 10 := sorry

end prob_select_A_and_B_l478_478645


namespace probability_of_selecting_A_and_B_is_three_tenths_l478_478607

-- Define the set of students
def students : Finset ℕ := {0, 1, 2, 3, 4}

-- Define subsets A and B
def student_A : ℕ := 0
def student_B : ℕ := 1

-- Total number of ways to select 3 students out of 5
def total_ways : ℕ := students.card.choose 3

-- Number of ways to select 3 students including A and B
def favorable_ways : ℕ := {2, 3, 4}.card.choose 1

-- The probability
def probability_of_selecting_A_and_B : ℚ := (favorable_ways : ℚ) / (total_ways : ℚ)

theorem probability_of_selecting_A_and_B_is_three_tenths :
  probability_of_selecting_A_and_B = 3 / 10 :=
by
  sorry

end probability_of_selecting_A_and_B_is_three_tenths_l478_478607


namespace DE_angle_bisector_AD_l478_478709

noncomputable def point_on_circle (S : circle) (D : point) : Prop := sorry
noncomputable def perpendicular_to_diameter (AB : line) (D C : point) : Prop := sorry
noncomputable def circle_tangent_to_segments (S1 : circle) (CA CD : segment) (E D : point) : Prop := sorry
noncomputable def circle_tangent_to_circle (S S1 : circle) : Prop := sorry
noncomputable def angle_bisector (E D A C : point) : Prop := sorry

-- Defining the problem conditions
variables {S S1 : circle} {A B C D E : point} {AB CA CD : segment}

axiom D_on_circle_S : point_on_circle S D
axiom DC_perpendicular_AB : perpendicular_to_diameter AB D C
axiom S1_tangent_CA_CD_E : circle_tangent_to_segments S1 CA CD E D
axiom S1_tangent_to_S : circle_tangent_to_circle S S1

-- Stating the theorem
theorem DE_angle_bisector_AD : 
  point_on_circle S D → 
  perpendicular_to_diameter AB D C → 
  circle_tangent_to_segments S1 CA CD E D → 
  circle_tangent_to_circle S S1 → 
  angle_bisector D E A C :=
  begin
    sorry
  end

end DE_angle_bisector_AD_l478_478709


namespace probability_A_and_B_selected_l478_478529

theorem probability_A_and_B_selected :
  let n := 5
  let r := 3
  let total_ways := Nat.choose n r
  let favorable_ways := Nat.choose 3 1
  let probability := favorable_ways / total_ways
  probability = (3 / 10) := 
by
  sorry

end probability_A_and_B_selected_l478_478529


namespace probability_A_and_B_selected_l478_478334

theorem probability_A_and_B_selected (A B : Type) (S : Finset (Type)) (h : S.card = 5) (hA : A ∈ S) (hB : B ∈ S) (T : Finset (Type)) (hT : T.card = 3) :
  (∃ T ⊆ S, A ∈ T ∧ B ∈ T ∧ T.card = 3) → 
  ∃ T ⊆ S, T.card = 3 ∧ (finset.card (T ∩ {A, B}) = 2) → 
  (finset.card (T ∩ {A, B}) = 2) = (3 / 10 : Real) := sorry

end probability_A_and_B_selected_l478_478334


namespace greatest_possible_value_x_y_z_l478_478970

noncomputable def S := a + b + c + d + e

theorem greatest_possible_value_x_y_z
  (a b c d e : ℕ)
  (h_pairwise_sums : multiset.sum {210, 350, 305, 255, x, y, 400, 345, 380, z}) = 5 * S
  (H_known_pairwise_sums : multiset.sum {210, 350, 305, 255, 400, 345, 380}) = 1935 
  (h_max_sum : S = 1125) :
  x + y + z = 3690 :=
sorry

end greatest_possible_value_x_y_z_l478_478970


namespace black_pieces_count_l478_478848

theorem black_pieces_count :
  ∃ (grid : Fin 6 → Fin 6 → bool), -- grid configuration
  (∀ i : Fin 6, ∃! w_i, ∀ j : Fin 6, grid i j = (w_i > 0)) ∧ -- distinct number of white pieces per row
  (∃ c : Fin 6 → ℕ, ∀ j : Fin 6, ∀ i : Fin 6, grid i j = (c j > 0) ∧ (c 0 = c j)) → -- same number of white pieces per column
  ∑ i, ∑ j, if grid i j then 0 else 1 = 18 := -- total number of black pieces

begin
  sorry
end

end black_pieces_count_l478_478848


namespace digital_roots_squares_cubes_mod_9_l478_478046

theorem digital_roots_squares_cubes_mod_9 :
  (∀ n ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8}, ((n*n) % 9) ∈ {0, 1, 4, 7}) ∧ 
  (∀ n ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8}, ((n*n*n) % 9) ∈ {0, 1, 8}) :=
by {
  -- Proof would go here
  sorry
}

end digital_roots_squares_cubes_mod_9_l478_478046


namespace ribbon_problem_l478_478108

variable (Ribbon1 Ribbon2 : ℕ)
variable (L : ℕ)

theorem ribbon_problem
    (h1 : Ribbon1 = 8)
    (h2 : ∀ L, L > 0 → Ribbon1 % L = 0 → Ribbon2 % L = 0)
    (h3 : ∀ k, (k > 0 ∧ Ribbon1 % k = 0 ∧ Ribbon2 % k = 0) → k ≤ 8) :
    Ribbon2 = 8 := by
  sorry

end ribbon_problem_l478_478108


namespace probability_A_and_B_selected_l478_478580

/-- Let there be a group of 5 students including A and B. 
    If 3 students are selected randomly from these 5, the probability of selecting both A and B is 3/10. -/
theorem probability_A_and_B_selected : 
  let students := 5 in
  let selected := 3 in
  let combinations := students.choose selected in
  let favorable_combinations := (students - 2).choose (selected - 2) in
  let probability := (favorable_combinations : ℚ) / (combinations : ℚ) in
  probability = 3 / 10 := 
by sorry

end probability_A_and_B_selected_l478_478580


namespace probability_of_selecting_A_and_B_l478_478558

theorem probability_of_selecting_A_and_B (students : Finset ℕ) (A B : ℕ)
  (h_size : students.card = 5) (h_in : A ∈ students ∧ B ∈ students) : 
  (Finset.card {s : Finset ℕ // s ⊆ students ∧ s.card = 3 ∧ A ∈ s ∧ B ∈ s}.to_set) / 
  (Finset.card {s : Finset ℕ // s ⊆ students ∧ s.card = 3}.to_set) = 3 / 10 :=
sorry

end probability_of_selecting_A_and_B_l478_478558


namespace sequence_property_l478_478822

theorem sequence_property (a : ℕ → ℕ) (h1 : a 1 = 1)
  (h_rec : ∀ m n : ℕ, a (m + n) = a m + a n + m * n) :
  a 10 = 55 :=
sorry

end sequence_property_l478_478822


namespace probability_A_and_B_selected_l478_478331

theorem probability_A_and_B_selected (A B : Type) (S : Finset (Type)) (h : S.card = 5) (hA : A ∈ S) (hB : B ∈ S) (T : Finset (Type)) (hT : T.card = 3) :
  (∃ T ⊆ S, A ∈ T ∧ B ∈ T ∧ T.card = 3) → 
  ∃ T ⊆ S, T.card = 3 ∧ (finset.card (T ∩ {A, B}) = 2) → 
  (finset.card (T ∩ {A, B}) = 2) = (3 / 10 : Real) := sorry

end probability_A_and_B_selected_l478_478331


namespace total_courses_attended_l478_478836

-- Define the number of courses attended by Max
def maxCourses : ℕ := 40

-- Define the number of courses attended by Sid (four times as many as Max)
def sidCourses : ℕ := 4 * maxCourses

-- Define the total number of courses attended by both Max and Sid
def totalCourses : ℕ := maxCourses + sidCourses

-- The proof statement
theorem total_courses_attended : totalCourses = 200 := by
  sorry

end total_courses_attended_l478_478836


namespace center_incircle_l478_478925

noncomputable def semicircle_diameter (P Q : ℝ) : set ℝ := 
  {x | x >= P ∧ x <= Q}

noncomputable def inscribed_square (A B C D P Q : ℝ) : Prop :=
  A = P ∧ D = Q ∧ B ≠ D ∧ C ≠ D ∧
  exists r, 
    (B - A) ^ 2 + (r - D) ^ 2 = ((B - A) + (r - D))^2 ∧ 
    (C - A) ^ 2 + (r - D) ^ 2 = ((C - A) + (r - D))^2

noncomputable def right_triangle (P Q R : ℝ) : Prop :=
  (P, Q, R) is right triangle ∧ PQ = QR

noncomputable def triangle_area_equal_square (P Q R A B C D : ℝ) : Prop :=
  0.5 * (P * Q * R) = (A * B * C * D)

noncomputable def center_incircle_on_square_side (P Q R A B C D : ℝ) : Prop :=
  ∃ I, (∃ A* D*, inscribed_square A B C D P Q) ∧ right_triangle P Q R ∧ 
  triangle_area_equal_square P Q R A B C D →
  I lies on (A, D)

theorem center_incircle (P Q R A B C D : ℝ) : 
  semicircle_diameter P Q ∧ inscribed_square A B C D P Q ∧ 
  right_triangle P Q R ∧ triangle_area_equal_square P Q R A B C D
  → center_incircle_on_square_side P Q R A B C D := 
by 
  sorry

end center_incircle_l478_478925


namespace probability_AB_selected_l478_478507

def factorial (n : Nat) : Nat := if n = 0 then 1 else n * factorial (n - 1)

def choose (n k : Nat) : Nat := factorial n / (factorial k * factorial (n - k))

theorem probability_AB_selected (h1 : 5 = 5) (h2 : 3 = 3) : (choose 3 1) / (choose 5 3) = 3 / 10 := by
  sorry

end probability_AB_selected_l478_478507


namespace factorial_expression_value_l478_478931

theorem factorial_expression_value : (10.factorial * 7.factorial * 3.factorial) / (9.factorial * 8.factorial) = 15 / 7 := by
  sorry

end factorial_expression_value_l478_478931


namespace wyhoff_game_position_l478_478801

noncomputable def alpha : ℝ := (1 + Real.sqrt 5) / 2
noncomputable def beta : ℝ := (1 + Real.sqrt 5) / 2 + 1

def a (n : ℕ) : ℕ := ⌊alpha * n⌋₊
def b (n : ℕ) : ℕ := ⌊beta * n⌋₊

theorem wyhoff_game_position :
  b 100 - a 100 = 100 :=
sorry

end wyhoff_game_position_l478_478801


namespace prob_select_A_and_B_l478_478658

theorem prob_select_A_and_B (students : Finset ℕ) (A B : ℕ) (hA : A ∈ students) (hB : B ∈ students) (h_card : students.card = 5) :
  (students.choose 3).count {comb | A ∈ comb ∧ B ∈ comb} / (students.choose 3).card = 3 / 10 := sorry

end prob_select_A_and_B_l478_478658


namespace last_digit_to_appear_is_6_l478_478134

def modified_fib (n : ℕ) : ℕ :=
match n with
| 1 => 2
| 2 => 3
| n + 3 => modified_fib (n + 2) + modified_fib (n + 1)
| _ => 0 -- To silence the "missing cases" warning; won't be hit.

theorem last_digit_to_appear_is_6 :
  ∃ N : ℕ, ∀ n : ℕ, (n < N → ∃ d, d < 10 ∧ 
    (∀ m < n, (modified_fib m) % 10 ≠ d) ∧ d = 6) := sorry

end last_digit_to_appear_is_6_l478_478134


namespace probability_A_and_B_selected_l478_478245

theorem probability_A_and_B_selected (students : Finset ℕ) (A B : ℕ) (h : A ∈ students ∧ B ∈ students ∧ students.card = 5) : 
  (students.choose 3).count (λ s, {A, B} ⊆ s) / (students.choose 3).card = 3 / 10 := by
  sorry

end probability_A_and_B_selected_l478_478245


namespace probability_A_and_B_selected_l478_478688

/-- There are 5 students including A and B. The probability of selecting both A and B 
    when 3 students are selected randomly from 5 is 3/10. -/
theorem probability_A_and_B_selected :
  (∃ (students : List String) (h : students.length = 5),
    ∃ (selected : List String) (h' : selected.length = 3),
    "A" ∈ selected ∧ "B" ∈ selected) →
  ∃ (P : ℚ), P = 3 / 10 :=
by
  sorry

end probability_A_and_B_selected_l478_478688


namespace prob_select_A_and_B_l478_478628

theorem prob_select_A_and_B : 
  let total_students := 5
  let to_select := 3
  let favorable_outcomes := Nat.div_fact (Nat.factorial 3) (Nat.factorial 1)
  let total_outcomes := Nat.div_fact (Nat.factorial 5) (Nat.factorial 3 * Nat.factorial (5 - 3))
  (favorable_outcomes : ℚ) / total_outcomes = 3 / 10 :=
by
  sorry

end prob_select_A_and_B_l478_478628


namespace probability_A_B_l478_478455

open Nat

-- Define a function for combination
def combination (n r : Nat) : Nat :=
  factorial n / (factorial r * factorial (n - r))

-- Define the set of students and the selection conditions
def num_students : Nat := 5
def num_selected : Nat := 3

-- Define A and B as selected students
def selected_A_B : Nat := combination 3 1

-- Define the probability calculation
def probability_A_B_selected : Rat :=
  (selected_A_B : Rat) / (combination num_students num_selected : Rat)

-- The statement of the problem
theorem probability_A_B : probability_A_B_selected = 3 / 10 := by
  sorry

end probability_A_B_l478_478455


namespace h_of_2_eq_7_l478_478815

def f (x : ℝ) : ℝ := 2 * x + 5
def g (x : ℝ) : ℝ := Real.sqrt (f x) - 2
def h (x : ℝ) : ℝ := f (g x)

theorem h_of_2_eq_7 : h 2 = 7 := by
  sorry

end h_of_2_eq_7_l478_478815


namespace probability_AB_selected_l478_478505

def factorial (n : Nat) : Nat := if n = 0 then 1 else n * factorial (n - 1)

def choose (n k : Nat) : Nat := factorial n / (factorial k * factorial (n - k))

theorem probability_AB_selected (h1 : 5 = 5) (h2 : 3 = 3) : (choose 3 1) / (choose 5 3) = 3 / 10 := by
  sorry

end probability_AB_selected_l478_478505


namespace probability_both_A_and_B_selected_l478_478416

-- The universe of students we consider
inductive Student
| A : Student
| B : Student
| C : Student
| D : Student
| E : Student

open Student

-- Function to count combinations
def combinations (n k : ℕ) : ℕ := (Nat.factorial n) / ((Nat.factorial k) * (Nat.factorial (n - k)))

-- The set of all students
def students : List Student := [A, B, C, D, E]

-- The total number of ways to select 3 students from 5
def total_selections := combinations 5 3

-- The number of ways A and B are both selected by selecting 1 more from the remaining 3 students
def selections_AB := combinations 3 1

-- The probability that both A and B are selected
def probability_AB := (selections_AB : ℚ) / (total_selections : ℚ)

-- The proof statement
theorem probability_both_A_and_B_selected :
  probability_AB = 3 / 10 :=
sorry

end probability_both_A_and_B_selected_l478_478416


namespace dolphins_win_at_least_six_games_l478_478878

noncomputable def binomial_probability : ℕ → ℕ → ℝ → ℝ :=
  λ n k p, (Nat.choose n k) * (p ^ k) * ((1 - p) ^ (n - k))

noncomputable def dolphins_probability : ℝ :=
  let p := 0.5
  let n := 9
  (binomial_probability n 6 p) + (binomial_probability n 7 p) + 
  (binomial_probability n 8 p) + (binomial_probability n 9 p)

theorem dolphins_win_at_least_six_games : dolphins_probability = 65 / 256 :=
by
  sorry

end dolphins_win_at_least_six_games_l478_478878


namespace number_of_distinct_possible_values_c_l478_478829

noncomputable def count_complex_values (c : ℂ) (r s t : ℂ) : ℕ :=
if ∀ z : ℂ, (z - r) * (z - s) * (z - t) = (z - c * r) * (z - c * s) * (z - c * t) then 1 else 0

theorem number_of_distinct_possible_values_c :
  let r : ℂ := 1
      s : ℂ := complex.exp (2 * real.pi * complex.I / 3)
      t : ℂ := complex.exp (4 * real.pi * complex.I / 3) 
  in
  (r ≠ 0) ∧ (s ≠ 0) ∧ (t ≠ 0) ∧ (r ≠ s) ∧ (s ≠ t) ∧ (r ≠ t) →
  ∑ (c : ℂ) in {1, complex.exp (2 * real.pi * complex.I / 3), complex.exp (4 * real.pi * complex.I / 3)}, count_complex_values c r s t = 3 :=
by
  intros r s t h
  sorry

end number_of_distinct_possible_values_c_l478_478829


namespace probability_of_A_and_B_selected_l478_478469

variable {α : Type}

/-- From 5 students, we randomly select 3 students to participate in community service work. 
  Let A and B be two specific students among these five. The probability that 
  both A and B are selected is 3 / 10. -/
theorem probability_of_A_and_B_selected (A B : α) (s : Finset α) (h : s.card = 5) :
  (∃ t : Finset α, t ⊆ s ∧ A ∈ t ∧ B ∈ t ∧ (t.card = 3) ∧ (t.card = 3) →
   (@Finset.exists_subset_card_eq α t s 3) →
   (@Finset.exists_subset_card_eq α t (Finset.erase s A) 2) ∧
   (@Finset.exists_subset_card_eq α t (Finset.erase (Finset.erase s A) B) 1)) →
  (3 : ℚ) / 10 :=
by
  sorry -- Proof omitted as per instructions.

end probability_of_A_and_B_selected_l478_478469


namespace probability_of_selecting_A_and_B_is_three_tenths_l478_478591

-- Define the set of students
def students : Finset ℕ := {0, 1, 2, 3, 4}

-- Define subsets A and B
def student_A : ℕ := 0
def student_B : ℕ := 1

-- Total number of ways to select 3 students out of 5
def total_ways : ℕ := students.card.choose 3

-- Number of ways to select 3 students including A and B
def favorable_ways : ℕ := {2, 3, 4}.card.choose 1

-- The probability
def probability_of_selecting_A_and_B : ℚ := (favorable_ways : ℚ) / (total_ways : ℚ)

theorem probability_of_selecting_A_and_B_is_three_tenths :
  probability_of_selecting_A_and_B = 3 / 10 :=
by
  sorry

end probability_of_selecting_A_and_B_is_three_tenths_l478_478591


namespace more_radishes_correct_l478_478985

def total_radishes : ℕ := 88
def radishes_first_basket : ℕ := 37

def more_radishes_in_second_basket := total_radishes - radishes_first_basket - radishes_first_basket

theorem more_radishes_correct : more_radishes_in_second_basket = 14 :=
by
  sorry

end more_radishes_correct_l478_478985


namespace min_squares_to_cover_staircase_l478_478052

-- Definition of the staircase and the constraints
def is_staircase (n : ℕ) (s : ℕ → ℕ) : Prop :=
  ∀ i, i < n → s i = i + 1

-- The proof problem statement
theorem min_squares_to_cover_staircase : 
  ∀ n : ℕ, n = 15 →
  ∀ s : ℕ → ℕ, is_staircase n s →
  ∃ k : ℕ, k = 15 ∧ (∀ i, i < n → ∃ a b : ℕ, a ≤ i ∧ b ≤ s a ∧ ∃ (l : ℕ), l = 1) :=
by
  sorry

end min_squares_to_cover_staircase_l478_478052


namespace probability_A_and_B_selected_l478_478301

open Finset

noncomputable def total_ways : ℕ := (choose 5 3)

noncomputable def favorable_ways : ℕ := (choose 3 1)

noncomputable def probability (total favorable : ℕ) : ℚ := favorable / total

theorem probability_A_and_B_selected :
  probability total_ways favorable_ways = 3 / 10 :=
by
  sorry

end probability_A_and_B_selected_l478_478301


namespace probability_A_B_l478_478457

open Nat

-- Define a function for combination
def combination (n r : Nat) : Nat :=
  factorial n / (factorial r * factorial (n - r))

-- Define the set of students and the selection conditions
def num_students : Nat := 5
def num_selected : Nat := 3

-- Define A and B as selected students
def selected_A_B : Nat := combination 3 1

-- Define the probability calculation
def probability_A_B_selected : Rat :=
  (selected_A_B : Rat) / (combination num_students num_selected : Rat)

-- The statement of the problem
theorem probability_A_B : probability_A_B_selected = 3 / 10 := by
  sorry

end probability_A_B_l478_478457


namespace correct_statements_l478_478109

-- Definitions based on conditions
def synthetic_method_cause_to_effect : Prop := true
def analytic_method_indirect_proof : Prop := false
def analytic_method_effect_to_cause : Prop := true
def proof_by_contradiction_direct_proof : Prop := false

-- Theorem stating the correctness of statements 1 and 3 
theorem correct_statements : synthetic_method_cause_to_effect ∧ analytic_method_effect_to_cause :=
by {
  exact ⟨synthetic_method_cause_to_effect, analytic_method_effect_to_cause⟩,
  sorry
}

end correct_statements_l478_478109


namespace probability_A_and_B_selected_l478_478708

/-- There are 5 students including A and B. The probability of selecting both A and B 
    when 3 students are selected randomly from 5 is 3/10. -/
theorem probability_A_and_B_selected :
  (∃ (students : List String) (h : students.length = 5),
    ∃ (selected : List String) (h' : selected.length = 3),
    "A" ∈ selected ∧ "B" ∈ selected) →
  ∃ (P : ℚ), P = 3 / 10 :=
by
  sorry

end probability_A_and_B_selected_l478_478708


namespace prob_select_A_and_B_l478_478647

theorem prob_select_A_and_B (students : Finset ℕ) (A B : ℕ) (hA : A ∈ students) (hB : B ∈ students) (h_card : students.card = 5) :
  (students.choose 3).count {comb | A ∈ comb ∧ B ∈ comb} / (students.choose 3).card = 3 / 10 := sorry

end prob_select_A_and_B_l478_478647


namespace area_of_10th_square_l478_478153

noncomputable def area_of_square (n: ℕ) : ℚ :=
  if n = 1 then 4
  else 2 * (1 / 2)^(n - 1)

theorem area_of_10th_square : area_of_square 10 = 1 / 256 := 
  sorry

end area_of_10th_square_l478_478153


namespace good_pairs_total_l478_478998

def slope (a b : ℝ) : ℝ := a

def is_parallel (m1 m2 : ℝ) : Prop := m1 = m2

def is_perpendicular (m1 m2 : ℝ) : Prop := m1 * m2 = -1

def lines : List (ℝ × ℝ) := [(4, 1), (-5, -2), (4, 3), (1/5, 4), (-5, 1)]

def good_pairs_count : ℕ :=
  List.length (List.filter (λ (pair : (ℝ × ℝ)), 
    match pair with 
    | ((m1, _), (m2, _)) => is_parallel m1 m2 ∨ is_perpendicular m1 m2 
    end) 
    (List.zip lines lines))

theorem good_pairs_total : good_pairs_count = 4 := sorry

end good_pairs_total_l478_478998


namespace a_seq_formula_l478_478724

noncomputable def a_seq : ℕ → ℕ
| 1       := 3
| (n + 1) := 4 * a_seq n + 3

theorem a_seq_formula (n: ℕ) : a_seq n = 4^n - 1 := sorry

end a_seq_formula_l478_478724


namespace prob_select_A_and_B_l478_478620

theorem prob_select_A_and_B : 
  let total_students := 5
  let to_select := 3
  let favorable_outcomes := Nat.div_fact (Nat.factorial 3) (Nat.factorial 1)
  let total_outcomes := Nat.div_fact (Nat.factorial 5) (Nat.factorial 3 * Nat.factorial (5 - 3))
  (favorable_outcomes : ℚ) / total_outcomes = 3 / 10 :=
by
  sorry

end prob_select_A_and_B_l478_478620


namespace range_of_m_l478_478759

open Set

def A : Set ℝ := { x | -2 ≤ x ∧ x ≤ 7 }
def B (m : ℝ) : Set ℝ := { x | (m + 1) ≤ x ∧ x ≤ (2 * m - 1) }

theorem range_of_m (m : ℝ) : (A ∪ B m = A) → m ≤ 4 :=
by
  intro h
  sorry

end range_of_m_l478_478759


namespace bertha_daughters_and_granddaughters_have_no_daughters_l478_478984

/--
Bertha has 8 daughters and no sons.
Some of her daughters have 4 daughters each, and the rest have none.
Bertha has a total of 40 daughters and granddaughters.
Bertha has no great-granddaughters.
-/
theorem bertha_daughters_and_granddaughters_have_no_daughters :
  ∀ (daughters granddaughters daughters_with_children : ℕ),
  daughters = 8 ∧
  granddaughters = 40 - daughters ∧
  daughters_with_children * 4 = granddaughters ∧
  daughters_with_children = daughters →
  (daughters - daughters_with_children) + granddaughters = 32 :=
by
  intros daughters granddaughters daughters_with_children
  intro h
  -- assumptions from the conditions
  cases h with h1 h_rest
  cases h_rest with h2 h_rest
  cases h_rest with h3 h4
  -- the proof will use these assumptions to reach the final equality
  sorry

end bertha_daughters_and_granddaughters_have_no_daughters_l478_478984


namespace crates_of_oranges_l478_478032

theorem crates_of_oranges (C : ℕ) (h1 : ∀ crate, crate = 150) (h2 : ∀ box, box = 30) (num_boxes : ℕ) (total_fruits : ℕ) : 
  num_boxes = 16 → total_fruits = 2280 → 150 * C + 16 * 30 = 2280 → C = 12 :=
by
  intros num_boxes_eq total_fruits_eq fruit_eq
  sorry

end crates_of_oranges_l478_478032


namespace expected_value_median_l478_478912

noncomputable def median_of_three_dice_expected_value : ℚ :=
  7 / 2

theorem expected_value_median (a b : ℚ) (h₁ : a = 7) (h₂ : b = 2) :
  a + b = 9 :=
by
  rw [h₁, h₂]
  norm_num
  sorry

end expected_value_median_l478_478912


namespace probability_both_A_B_selected_l478_478208

-- Define the problem with the necessary conditions.
def probability_AB_selected (total_students : ℕ) (select_students : ℕ) (A B : ℕ) : ℚ :=
  if A < total_students ∧ B < total_students ∧ total_students = 5 ∧ select_students = 3 then
    let total_ways := (Nat.factorial total_students) / ((Nat.factorial select_students) * (Nat.factorial (total_students - select_students)))
    let favorable_ways := (Nat.factorial (total_students - 2)) / ((Nat.factorial (select_students - 2)) * (Nat.factorial ((total_students - 2) - (select_students - 2))))
    favorable_ways / total_ways
  else 0

theorem probability_both_A_B_selected 
  (total_students : ℕ) (select_students : ℕ) (A B : ℕ) 
  (h1 : total_students = 5) (h2 : select_students = 3) 
  (h3 : A < total_students) (h4 : B < total_students) 
  : probability_AB_selected total_students select_students A B = 3 / 10 := 
by {
  -- Insert the delta and logic to prove the theorem here.
  sorry
}

end probability_both_A_B_selected_l478_478208


namespace probability_A_and_B_selected_l478_478232

theorem probability_A_and_B_selected (students : Finset ℕ) (A B : ℕ) (h : A ∈ students ∧ B ∈ students ∧ students.card = 5) : 
  (students.choose 3).count (λ s, {A, B} ⊆ s) / (students.choose 3).card = 3 / 10 := by
  sorry

end probability_A_and_B_selected_l478_478232


namespace probability_of_selecting_A_and_B_l478_478272

-- Define the combination function
def combination (n r : ℕ) : ℕ := Nat.factorial n / (Nat.factorial r * Nat.factorial (n - r))

-- Define the conditions
def students : ℕ := 5
def to_select : ℕ := 3

-- Define the favorable outcomes where A and B are selected
def favorable_outcomes : ℕ := combination 3 1

-- Define the total outcomes of selecting 3 out of 5 students
def total_outcomes : ℕ := combination 5 3

-- Define the probability calculation
def probability_of_A_and_B : ℚ := favorable_outcomes / total_outcomes

-- Proposition to prove
theorem probability_of_selecting_A_and_B : probability_of_A_and_B = 3 / 10 :=
by
  sorry

end probability_of_selecting_A_and_B_l478_478272


namespace probability_of_selecting_A_and_B_l478_478266

-- Define the combination function
def combination (n r : ℕ) : ℕ := Nat.factorial n / (Nat.factorial r * Nat.factorial (n - r))

-- Define the conditions
def students : ℕ := 5
def to_select : ℕ := 3

-- Define the favorable outcomes where A and B are selected
def favorable_outcomes : ℕ := combination 3 1

-- Define the total outcomes of selecting 3 out of 5 students
def total_outcomes : ℕ := combination 5 3

-- Define the probability calculation
def probability_of_A_and_B : ℚ := favorable_outcomes / total_outcomes

-- Proposition to prove
theorem probability_of_selecting_A_and_B : probability_of_A_and_B = 3 / 10 :=
by
  sorry

end probability_of_selecting_A_and_B_l478_478266


namespace probability_A_B_selected_l478_478372

noncomputable def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

def combination (n r : ℕ) : ℕ :=
  factorial n / (factorial r * factorial (n - r))

theorem probability_A_B_selected :
  let total_ways := combination 5 3 in
  let ways_A_B := combination 3 1 in
  (ways_A_B : ℚ) / total_ways = 3 / 10 :=
by
  sorry

end probability_A_B_selected_l478_478372


namespace probability_A_B_selected_l478_478360

noncomputable def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

def combination (n r : ℕ) : ℕ :=
  factorial n / (factorial r * factorial (n - r))

theorem probability_A_B_selected :
  let total_ways := combination 5 3 in
  let ways_A_B := combination 3 1 in
  (ways_A_B : ℚ) / total_ways = 3 / 10 :=
by
  sorry

end probability_A_B_selected_l478_478360


namespace probability_of_A_and_B_selected_l478_478474

variable {α : Type}

/-- From 5 students, we randomly select 3 students to participate in community service work. 
  Let A and B be two specific students among these five. The probability that 
  both A and B are selected is 3 / 10. -/
theorem probability_of_A_and_B_selected (A B : α) (s : Finset α) (h : s.card = 5) :
  (∃ t : Finset α, t ⊆ s ∧ A ∈ t ∧ B ∈ t ∧ (t.card = 3) ∧ (t.card = 3) →
   (@Finset.exists_subset_card_eq α t s 3) →
   (@Finset.exists_subset_card_eq α t (Finset.erase s A) 2) ∧
   (@Finset.exists_subset_card_eq α t (Finset.erase (Finset.erase s A) B) 1)) →
  (3 : ℚ) / 10 :=
by
  sorry -- Proof omitted as per instructions.

end probability_of_A_and_B_selected_l478_478474


namespace sliding_segment_min_length_l478_478727

theorem sliding_segment_min_length (d : ℝ) : 
  (∀ x y z : ℝ, 0 ≤ x ∧ 0 ≤ y ∧ 0 ≤ z ∧ x + y + z = 1 → ∃ a b : ℝ, 0 ≤ a ∧ a ≤ 1 ∧ 0 ≤ b ∧ b ≤ 1 ∧ sqrt ((x - a)^2 + (y - b)^2) = d) ↔ d = 2/3 :=
sorry

end sliding_segment_min_length_l478_478727


namespace probability_of_selecting_A_and_B_l478_478549

theorem probability_of_selecting_A_and_B (students : Finset ℕ) (A B : ℕ)
  (h_size : students.card = 5) (h_in : A ∈ students ∧ B ∈ students) : 
  (Finset.card {s : Finset ℕ // s ⊆ students ∧ s.card = 3 ∧ A ∈ s ∧ B ∈ s}.to_set) / 
  (Finset.card {s : Finset ℕ // s ⊆ students ∧ s.card = 3}.to_set) = 3 / 10 :=
sorry

end probability_of_selecting_A_and_B_l478_478549


namespace buckets_required_l478_478915

theorem buckets_required (C : ℝ) (N : ℝ):
  (62.5 * (2 / 5) * C = N * C) → N = 25 :=
by
  sorry

end buckets_required_l478_478915


namespace boys_in_class_l478_478067

theorem boys_in_class (total_students : ℕ) (fraction_girls : ℝ) (fraction_girls_eq : fraction_girls = 1 / 4) (total_students_eq : total_students = 160) :
  (total_students - fraction_girls * total_students = 120) :=
by
  rw [fraction_girls_eq, total_students_eq]
  -- Here, additional lines proving the steps would follow, but we use sorry for completeness.
  sorry

end boys_in_class_l478_478067


namespace prob_first_class_individual_prob_at_least_one_first_class_part_l478_478834

open ProbabilityTheory

variable {Ω : Type*} {𝒜 : MeasurableSpace Ω} [ProbabilitySpace Ω]

-- Conditions as probabilities
variable (A B C : Set Ω)
variable (probAovrB : ℙ(A ∩ Bᶜ) = 1/4)
variable (probBovrC : ℙ(B ∩ Cᶜ) = 1/12)
variable (probAC : ℙ(A ∩ C) = 2/9)

-- Prove the individual probabilities
theorem prob_first_class_individual :
  ℙ(A) = 1/3 ∧
  ℙ(B) = 1/4 ∧
  ℙ(C) = 2/3 :=
sorry

-- Define event D for getting at least one first-class part
def D := (A ∪ B ∪ C)

-- Prove the probability of at least one first-class part
theorem prob_at_least_one_first_class_part :
  ℙ(D) = 5/6 :=
sorry

end prob_first_class_individual_prob_at_least_one_first_class_part_l478_478834


namespace probability_A_B_l478_478464

open Nat

-- Define a function for combination
def combination (n r : Nat) : Nat :=
  factorial n / (factorial r * factorial (n - r))

-- Define the set of students and the selection conditions
def num_students : Nat := 5
def num_selected : Nat := 3

-- Define A and B as selected students
def selected_A_B : Nat := combination 3 1

-- Define the probability calculation
def probability_A_B_selected : Rat :=
  (selected_A_B : Rat) / (combination num_students num_selected : Rat)

-- The statement of the problem
theorem probability_A_B : probability_A_B_selected = 3 / 10 := by
  sorry

end probability_A_B_l478_478464


namespace probability_AB_selected_l478_478498

def factorial (n : Nat) : Nat := if n = 0 then 1 else n * factorial (n - 1)

def choose (n k : Nat) : Nat := factorial n / (factorial k * factorial (n - k))

theorem probability_AB_selected (h1 : 5 = 5) (h2 : 3 = 3) : (choose 3 1) / (choose 5 3) = 3 / 10 := by
  sorry

end probability_AB_selected_l478_478498


namespace probability_of_A_and_B_selected_l478_478478

variable {α : Type}

/-- From 5 students, we randomly select 3 students to participate in community service work. 
  Let A and B be two specific students among these five. The probability that 
  both A and B are selected is 3 / 10. -/
theorem probability_of_A_and_B_selected (A B : α) (s : Finset α) (h : s.card = 5) :
  (∃ t : Finset α, t ⊆ s ∧ A ∈ t ∧ B ∈ t ∧ (t.card = 3) ∧ (t.card = 3) →
   (@Finset.exists_subset_card_eq α t s 3) →
   (@Finset.exists_subset_card_eq α t (Finset.erase s A) 2) ∧
   (@Finset.exists_subset_card_eq α t (Finset.erase (Finset.erase s A) B) 1)) →
  (3 : ℚ) / 10 :=
by
  sorry -- Proof omitted as per instructions.

end probability_of_A_and_B_selected_l478_478478


namespace carbon_copies_after_folding_l478_478939

-- Define the initial condition of sheets and carbon papers
def initial_sheets : ℕ := 3
def initial_carbons : ℕ := 2

-- Define the condition of folding the paper
def fold_paper (sheets carbons : ℕ) : ℕ := sheets * 2

-- Statement of the problem
theorem carbon_copies_after_folding : (fold_paper initial_sheets initial_carbons - initial_sheets + initial_carbons) = 4 :=
by
  sorry

end carbon_copies_after_folding_l478_478939


namespace probability_of_selecting_A_and_B_l478_478563

theorem probability_of_selecting_A_and_B (students : Finset ℕ) (A B : ℕ)
  (h_size : students.card = 5) (h_in : A ∈ students ∧ B ∈ students) : 
  (Finset.card {s : Finset ℕ // s ⊆ students ∧ s.card = 3 ∧ A ∈ s ∧ B ∈ s}.to_set) / 
  (Finset.card {s : Finset ℕ // s ⊆ students ∧ s.card = 3}.to_set) = 3 / 10 :=
sorry

end probability_of_selecting_A_and_B_l478_478563


namespace cars_difference_proof_l478_478847

theorem cars_difference_proof (U M : ℕ) :
  let initial_cars := 150
  let total_cars := 196
  let cars_from_uncle := U
  let cars_from_grandpa := 2 * U
  let cars_from_dad := 10
  let cars_from_auntie := U + 1
  let cars_from_mum := M
  let total_given_cars := cars_from_dad + cars_from_auntie + cars_from_uncle + cars_from_grandpa + cars_from_mum
  initial_cars + total_given_cars = total_cars ->
  (cars_from_mum - cars_from_dad = 5) := 
by
  sorry

end cars_difference_proof_l478_478847


namespace find_ordered_pair_l478_478894

theorem find_ordered_pair (s m : ℚ) :
  (∃ t : ℚ, (5 * s - 7 = 2) ∧ 
           ((∃ (t1 : ℚ), (x = s + 3 * t1) ∧  (y = 2 + m * t1)) 
           → (x = 24 / 5) → (y = 5))) →
  (s = 9 / 5 ∧ m = 3) :=
by
  sorry

end find_ordered_pair_l478_478894


namespace alcohol_percentage_correct_l478_478946

-- Define the initial conditions

def original_solution_volume : ℝ := 11
def added_water_volume : ℝ := 3
def alcohol_percentage_original : ℝ := 0.42

-- Calculate amount of alcohol in original solution
def alcohol_amount_original : ℝ := alcohol_percentage_original * original_solution_volume

-- Calculate total volume of new solution
def total_solution_volume : ℝ := original_solution_volume + added_water_volume

-- Definition: percentage of alcohol in new mixture
def alcohol_percentage_new : ℝ := (alcohol_amount_original / total_solution_volume) * 100

-- Proposition: the percentage of alcohol in new mixture is 33%
theorem alcohol_percentage_correct : alcohol_percentage_new = 33 := by
  -- Proof here
  sorry

#check alcohol_percentage_correct

end alcohol_percentage_correct_l478_478946


namespace probability_of_selecting_spring_or_dragon_boat_l478_478035

noncomputable def prob_at_least_one_festival_selected (total_festivals : ℕ) (selected_festivals : ℕ) : ℚ :=
  1 - (Nat.choose (total_festivals - selected_festivals + 1) selected_festivals / Nat.choose total_festivals selected_festivals)

theorem probability_of_selecting_spring_or_dragon_boat : 
  prob_at_least_one_festival_selected 5 2 = 0.7 := by
  sorry

end probability_of_selecting_spring_or_dragon_boat_l478_478035


namespace probability_of_A_and_B_selected_l478_478484

variable {α : Type}

/-- From 5 students, we randomly select 3 students to participate in community service work. 
  Let A and B be two specific students among these five. The probability that 
  both A and B are selected is 3 / 10. -/
theorem probability_of_A_and_B_selected (A B : α) (s : Finset α) (h : s.card = 5) :
  (∃ t : Finset α, t ⊆ s ∧ A ∈ t ∧ B ∈ t ∧ (t.card = 3) ∧ (t.card = 3) →
   (@Finset.exists_subset_card_eq α t s 3) →
   (@Finset.exists_subset_card_eq α t (Finset.erase s A) 2) ∧
   (@Finset.exists_subset_card_eq α t (Finset.erase (Finset.erase s A) B) 1)) →
  (3 : ℚ) / 10 :=
by
  sorry -- Proof omitted as per instructions.

end probability_of_A_and_B_selected_l478_478484


namespace vertical_asymptote_at_9_7_l478_478200

def has_vertical_asymptote_at (f : ℝ → ℝ) (x : ℝ) : Prop :=
  ∀ ε > 0, ∃ δ > 0, ∀ x', abs (x' - x) < δ → abs (f x') > ε

noncomputable def f : ℝ → ℝ := λ x, (2 * x + 3) / (7 * x - 9)

theorem vertical_asymptote_at_9_7 :
  has_vertical_asymptote_at f (9 / 7) :=
by
  sorry

end vertical_asymptote_at_9_7_l478_478200


namespace probability_of_selecting_A_and_B_l478_478283

open_locale classical

-- Define the number of students
def num_students := 5

-- Define the number of students to be selected
def students_to_select := 3

def combination (n k : ℕ) : ℕ := nat.choose n k

-- Define the correct answer
def correct_probability := 3 / 10

-- Define the probability calculation
def calc_probability := (combination 3 1) / (combination 5 3)

-- Statement of the theorem to be proved
theorem probability_of_selecting_A_and_B :
  calc_probability = correct_probability :=
by sorry

end probability_of_selecting_A_and_B_l478_478283


namespace part1_part2_l478_478073

theorem part1 (x : ℝ) : 3 + 2 * x > - x - 6 ↔ x > -3 := by
  sorry

theorem part2 (x : ℝ) : 2 * x + 1 ≤ x + 3 ∧ (2 * x + 1) / 3 > 1 ↔ 1 < x ∧ x ≤ 2 := by
  sorry

end part1_part2_l478_478073


namespace probability_A_and_B_selected_l478_478391

theorem probability_A_and_B_selected 
  (students : Finset ℕ) (A B : ℕ) 
  (h_students_size : students.card = 5) 
  (h_A_in_students : A ∈ students) 
  (h_B_in_students : B ∈ students)
  : (↑(Nat.choose 3) : ℚ) / (↑(Nat.choose 5) : ℚ) = 3 / 10 := 
by sorry

end probability_A_and_B_selected_l478_478391


namespace probability_of_selecting_A_and_B_l478_478279

open_locale classical

-- Define the number of students
def num_students := 5

-- Define the number of students to be selected
def students_to_select := 3

def combination (n k : ℕ) : ℕ := nat.choose n k

-- Define the correct answer
def correct_probability := 3 / 10

-- Define the probability calculation
def calc_probability := (combination 3 1) / (combination 5 3)

-- Statement of the theorem to be proved
theorem probability_of_selecting_A_and_B :
  calc_probability = correct_probability :=
by sorry

end probability_of_selecting_A_and_B_l478_478279


namespace problem_statement_l478_478781

variable {ℝ : Type*}

def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f x = f (-x)

def is_monotonically_decreasing (f : ℝ → ℝ) (a b : ℝ): Prop :=
  ∀ x y : ℝ, a ≤ x ∧ x ≤ y ∧ y ≤ b → f x ≥ f y

theorem problem_statement {f : ℝ → ℝ}
  (h_even : is_even_function f)
  (h_decreasing : is_monotonically_decreasing f 0 3) :
  f (-1) > f 2 ∧ f 2 > f 3 :=
by
  sorry

end problem_statement_l478_478781


namespace farmer_loss_representative_value_l478_478085

def check_within_loss_range (S L : ℝ) : Prop :=
  (S = 100000) → (20000 ≤ L ∧ L ≤ 25000)

theorem farmer_loss_representative_value : check_within_loss_range 100000 21987.53 :=
by
  intros hs
  sorry

end farmer_loss_representative_value_l478_478085


namespace alpha_more_economical_l478_478107

theorem alpha_more_economical (n : ℕ) : n ≥ 12 → 80 + 12 * n < 10 + 18 * n := 
by
  sorry

end alpha_more_economical_l478_478107


namespace probability_A_B_l478_478448

open Nat

-- Define a function for combination
def combination (n r : Nat) : Nat :=
  factorial n / (factorial r * factorial (n - r))

-- Define the set of students and the selection conditions
def num_students : Nat := 5
def num_selected : Nat := 3

-- Define A and B as selected students
def selected_A_B : Nat := combination 3 1

-- Define the probability calculation
def probability_A_B_selected : Rat :=
  (selected_A_B : Rat) / (combination num_students num_selected : Rat)

-- The statement of the problem
theorem probability_A_B : probability_A_B_selected = 3 / 10 := by
  sorry

end probability_A_B_l478_478448


namespace probability_AB_selected_l478_478502

def factorial (n : Nat) : Nat := if n = 0 then 1 else n * factorial (n - 1)

def choose (n k : Nat) : Nat := factorial n / (factorial k * factorial (n - k))

theorem probability_AB_selected (h1 : 5 = 5) (h2 : 3 = 3) : (choose 3 1) / (choose 5 3) = 3 / 10 := by
  sorry

end probability_AB_selected_l478_478502


namespace sequence_0_to_0_l478_478043

noncomputable def apply_operations : ℤ → List (ℤ → ℤ) → ℤ
| x, []       => x
| x, f :: fs  => apply_operations (f x) fs

theorem sequence_0_to_0 :
  apply_operations 0 [ (λ x => x + 1), (λ x => -x⁻¹), (λ x => x + 1) ] = 0 :=
sorry

end sequence_0_to_0_l478_478043


namespace probability_of_selecting_A_and_B_l478_478551

theorem probability_of_selecting_A_and_B (students : Finset ℕ) (A B : ℕ)
  (h_size : students.card = 5) (h_in : A ∈ students ∧ B ∈ students) : 
  (Finset.card {s : Finset ℕ // s ⊆ students ∧ s.card = 3 ∧ A ∈ s ∧ B ∈ s}.to_set) / 
  (Finset.card {s : Finset ℕ // s ⊆ students ∧ s.card = 3}.to_set) = 3 / 10 :=
sorry

end probability_of_selecting_A_and_B_l478_478551


namespace probability_AB_selected_l478_478500

def factorial (n : Nat) : Nat := if n = 0 then 1 else n * factorial (n - 1)

def choose (n k : Nat) : Nat := factorial n / (factorial k * factorial (n - k))

theorem probability_AB_selected (h1 : 5 = 5) (h2 : 3 = 3) : (choose 3 1) / (choose 5 3) = 3 / 10 := by
  sorry

end probability_AB_selected_l478_478500


namespace solve_system_of_equations_l478_478000

theorem solve_system_of_equations (x y : ℝ) (hx : x > 0) (hy : y > 0)
  (h1 : x^4 + y^4 - x^2 * y^2 = 13)
  (h2 : x^2 - y^2 + 2 * x * y = 1) :
  x = 1 ∧ y = 2 :=
sorry

end solve_system_of_equations_l478_478000


namespace sqrt_500_simplified_l478_478872

theorem sqrt_500_simplified : sqrt 500 = 10 * sqrt 5 :=
by
sorry

end sqrt_500_simplified_l478_478872


namespace steve_long_letter_writing_time_l478_478875

theorem steve_long_letter_writing_time :
  ∀ (days_in_month letters_every_n_days pages_per_month time_per_letter minutes_per_page double_time_factor : ℕ),
    (days_in_month = 30) →
    (letters_every_n_days = 3) →
    (pages_per_month = 24) →
    (time_per_letter = 20) →
    (minutes_per_page = 10) →
    (double_time_factor = 2) →
    let letters_per_month := days_in_month / letters_every_n_days in
    let pages_per_letter := time_per_letter / minutes_per_page in
    let regular_pages := letters_per_month * pages_per_letter in
    let long_letter_pages := pages_per_month - regular_pages in
    long_letter_pages * (minutes_per_page * double_time_factor) = 80 :=
by
  intros days_in_month letters_every_n_days pages_per_month time_per_letter minutes_per_page double_time_factor
  assume h1 h2 h3 h4 h5 h6
  let letters_per_month := days_in_month / letters_every_n_days
  let pages_per_letter := time_per_letter / minutes_per_page
  let regular_pages := letters_per_month * pages_per_letter
  let long_letter_pages := pages_per_month - regular_pages
  calc
    long_letter_pages * (minutes_per_page * double_time_factor) = (pages_per_month - regular_pages) * (minutes_per_page * double_time_factor) : by rfl
    ... = (24 - (30 / 3) * (20 / 10)) * (10 * 2) : by rw [h1, h2, h3, h4, h5, h6]
    ... = (24 - 10 * 2) * 20 : by norm_num
    ... = 4 * 20 : by norm_num
    ... = 80 : by norm_num
  sorry

end steve_long_letter_writing_time_l478_478875


namespace probability_A_B_selected_l478_478362

noncomputable def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

def combination (n r : ℕ) : ℕ :=
  factorial n / (factorial r * factorial (n - r))

theorem probability_A_B_selected :
  let total_ways := combination 5 3 in
  let ways_A_B := combination 3 1 in
  (ways_A_B : ℚ) / total_ways = 3 / 10 :=
by
  sorry

end probability_A_B_selected_l478_478362


namespace probability_A_and_B_selected_l478_478539

theorem probability_A_and_B_selected :
  let n := 5
  let r := 3
  let total_ways := Nat.choose n r
  let favorable_ways := Nat.choose 3 1
  let probability := favorable_ways / total_ways
  probability = (3 / 10) := 
by
  sorry

end probability_A_and_B_selected_l478_478539


namespace probability_A_B_l478_478450

open Nat

-- Define a function for combination
def combination (n r : Nat) : Nat :=
  factorial n / (factorial r * factorial (n - r))

-- Define the set of students and the selection conditions
def num_students : Nat := 5
def num_selected : Nat := 3

-- Define A and B as selected students
def selected_A_B : Nat := combination 3 1

-- Define the probability calculation
def probability_A_B_selected : Rat :=
  (selected_A_B : Rat) / (combination num_students num_selected : Rat)

-- The statement of the problem
theorem probability_A_B : probability_A_B_selected = 3 / 10 := by
  sorry

end probability_A_B_l478_478450


namespace probability_A_and_B_selected_l478_478565

/-- Let there be a group of 5 students including A and B. 
    If 3 students are selected randomly from these 5, the probability of selecting both A and B is 3/10. -/
theorem probability_A_and_B_selected : 
  let students := 5 in
  let selected := 3 in
  let combinations := students.choose selected in
  let favorable_combinations := (students - 2).choose (selected - 2) in
  let probability := (favorable_combinations : ℚ) / (combinations : ℚ) in
  probability = 3 / 10 := 
by sorry

end probability_A_and_B_selected_l478_478565


namespace ratio_approximation_l478_478039

noncomputable def ratio_condition (a b : ℝ) (h1 : a > b) (h2 : b > 0) (h3 : (a + b) / 2 = 3 * Real.sqrt (a * b)) : ℝ :=
  a / b

theorem ratio_approximation (a b : ℝ) (h1 : a > b) (h2 : b > 0) (h3 : (a + b) / 2 = 3 * Real.sqrt (a * b)) :
  (ratio_condition a b h1 h2 h3).to_nnreal ≈ 34 :=
by
  -- to fill in the proof 
  sorry

end ratio_approximation_l478_478039


namespace consecutive_ints_sum_l478_478900

theorem consecutive_ints_sum (n : ℕ) (h : n * (n + 1) = 2800) : n + (n + 1) = 105 :=
begin
  sorry
end

end consecutive_ints_sum_l478_478900


namespace new_mean_of_five_numbers_l478_478895

theorem new_mean_of_five_numbers (a b c d e : ℝ) 
  (h_mean : (a + b + c + d + e) / 5 = 25) :
  ((a + 5) + (b + 10) + (c + 15) + (d + 20) + (e + 25)) / 5 = 40 :=
by
  sorry

end new_mean_of_five_numbers_l478_478895


namespace mutual_fund_percent_increase_l478_478063

-- Lean statement of the problem
theorem mutual_fund_percent_increase 
  (P : ℝ)
  (h_first_quarter : 1.25 * P)
  (h_second_quarter : 1.55 * P) :
  ((1.55 - 1.25) / 1.25) * 100 = 24 := 
sorry

end mutual_fund_percent_increase_l478_478063


namespace boar_sausages_left_l478_478842

def boar_sausages_final_count(sausages_initial : ℕ) : ℕ :=
  let after_monday := sausages_initial - (2 / 5 * sausages_initial)
  let after_tuesday := after_monday - (1 / 2 * after_monday)
  let after_wednesday := after_tuesday - (1 / 4 * after_tuesday)
  let after_thursday := after_wednesday - (1 / 3 * after_wednesday)
  let after_sharing := after_thursday - (1 / 5 * after_thursday)
  let after_eating := after_sharing - (3 / 5 * after_sharing)
  after_eating

theorem boar_sausages_left : boar_sausages_final_count 1200 = 58 := 
  sorry

end boar_sausages_left_l478_478842


namespace probability_both_A_and_B_selected_l478_478413

-- The universe of students we consider
inductive Student
| A : Student
| B : Student
| C : Student
| D : Student
| E : Student

open Student

-- Function to count combinations
def combinations (n k : ℕ) : ℕ := (Nat.factorial n) / ((Nat.factorial k) * (Nat.factorial (n - k)))

-- The set of all students
def students : List Student := [A, B, C, D, E]

-- The total number of ways to select 3 students from 5
def total_selections := combinations 5 3

-- The number of ways A and B are both selected by selecting 1 more from the remaining 3 students
def selections_AB := combinations 3 1

-- The probability that both A and B are selected
def probability_AB := (selections_AB : ℚ) / (total_selections : ℚ)

-- The proof statement
theorem probability_both_A_and_B_selected :
  probability_AB = 3 / 10 :=
sorry

end probability_both_A_and_B_selected_l478_478413


namespace factorize_quadratic_l478_478179

theorem factorize_quadratic (x : ℝ) : 
  16 * x^2 - 40 * x + 25 = (4 * x - 5)^2 :=
sorry

end factorize_quadratic_l478_478179


namespace prob_select_A_and_B_l478_478651

theorem prob_select_A_and_B (students : Finset ℕ) (A B : ℕ) (hA : A ∈ students) (hB : B ∈ students) (h_card : students.card = 5) :
  (students.choose 3).count {comb | A ∈ comb ∧ B ∈ comb} / (students.choose 3).card = 3 / 10 := sorry

end prob_select_A_and_B_l478_478651


namespace probability_both_A_and_B_selected_l478_478415

-- The universe of students we consider
inductive Student
| A : Student
| B : Student
| C : Student
| D : Student
| E : Student

open Student

-- Function to count combinations
def combinations (n k : ℕ) : ℕ := (Nat.factorial n) / ((Nat.factorial k) * (Nat.factorial (n - k)))

-- The set of all students
def students : List Student := [A, B, C, D, E]

-- The total number of ways to select 3 students from 5
def total_selections := combinations 5 3

-- The number of ways A and B are both selected by selecting 1 more from the remaining 3 students
def selections_AB := combinations 3 1

-- The probability that both A and B are selected
def probability_AB := (selections_AB : ℚ) / (total_selections : ℚ)

-- The proof statement
theorem probability_both_A_and_B_selected :
  probability_AB = 3 / 10 :=
sorry

end probability_both_A_and_B_selected_l478_478415


namespace probability_A_and_B_selected_l478_478249

theorem probability_A_and_B_selected (students : Finset ℕ) (A B : ℕ) (h : A ∈ students ∧ B ∈ students ∧ students.card = 5) : 
  (students.choose 3).count (λ s, {A, B} ⊆ s) / (students.choose 3).card = 3 / 10 := by
  sorry

end probability_A_and_B_selected_l478_478249


namespace probability_A_and_B_selected_l478_478332

theorem probability_A_and_B_selected (A B : Type) (S : Finset (Type)) (h : S.card = 5) (hA : A ∈ S) (hB : B ∈ S) (T : Finset (Type)) (hT : T.card = 3) :
  (∃ T ⊆ S, A ∈ T ∧ B ∈ T ∧ T.card = 3) → 
  ∃ T ⊆ S, T.card = 3 ∧ (finset.card (T ∩ {A, B}) = 2) → 
  (finset.card (T ∩ {A, B}) = 2) = (3 / 10 : Real) := sorry

end probability_A_and_B_selected_l478_478332


namespace probability_both_A_and_B_selected_l478_478420

-- The universe of students we consider
inductive Student
| A : Student
| B : Student
| C : Student
| D : Student
| E : Student

open Student

-- Function to count combinations
def combinations (n k : ℕ) : ℕ := (Nat.factorial n) / ((Nat.factorial k) * (Nat.factorial (n - k)))

-- The set of all students
def students : List Student := [A, B, C, D, E]

-- The total number of ways to select 3 students from 5
def total_selections := combinations 5 3

-- The number of ways A and B are both selected by selecting 1 more from the remaining 3 students
def selections_AB := combinations 3 1

-- The probability that both A and B are selected
def probability_AB := (selections_AB : ℚ) / (total_selections : ℚ)

-- The proof statement
theorem probability_both_A_and_B_selected :
  probability_AB = 3 / 10 :=
sorry

end probability_both_A_and_B_selected_l478_478420


namespace probability_A_B_selected_l478_478363

noncomputable def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

def combination (n r : ℕ) : ℕ :=
  factorial n / (factorial r * factorial (n - r))

theorem probability_A_B_selected :
  let total_ways := combination 5 3 in
  let ways_A_B := combination 3 1 in
  (ways_A_B : ℚ) / total_ways = 3 / 10 :=
by
  sorry

end probability_A_B_selected_l478_478363


namespace solve_diff_eq_l478_478873

-- Define the differential equation and its components
def diff_eq (x : ℝ) (y y' y'' : ℝ→ℝ) : Prop := y'' x + 6 * y' x + 8 * y x = 4 * exp (-2 * x) / (2 + exp (2 * x))

-- Define the general solution of the homogeneous equation
def homogeneous_sol (x : ℝ) (C1 C2 : ℝ) : ℝ :=
  C1 * exp (-4 * x) + C2 * exp (-2 * x)

-- Define the particular solution components
def particular_C2 (x : ℝ) : ℝ := x - 0.5 * log (2 + exp (2 * x))
def particular_C1 (x : ℝ) : ℝ := - log (2 + exp (2 * x))

-- Define the overall particular solution
def particular_sol (x : ℝ) : ℝ :=
  particular_C1 x * exp (-4 * x) + particular_C2 x * exp (-2 * x)

-- Define the general solution
def general_solution (x C1 C2 : ℝ) : ℝ :=
  homogeneous_sol x C1 C2 - exp (-4 * x) * log (2 + exp (2 * x)) +
  (x - 0.5 * log (2 + exp (2 * x))) * exp (-2 * x)

-- The statement of the proof problem
theorem solve_diff_eq :
  ∃ (C1 C2 : ℝ), ∀ (y y' y'' : ℝ → ℝ),
    y = general_solution x C1 C2 →
    diff_eq x y y' y''
:= by
  sorry

end solve_diff_eq_l478_478873


namespace probability_red_odd_green_special_l478_478922

theorem probability_red_odd_green_special : 
  let red_die := {1, 2, 3, 4, 5, 6, 7, 8}
  let green_die := {1, 2, 3, 4, 5, 6, 7, 8}
  let is_odd (x : Nat) : Prop := x % 2 = 1
  let is_perfect_square (x : Nat) : Prop := x = 1 ∨ x = 4
  let is_prime (x : Nat) : Prop := x = 2 ∨ x = 3 ∨ x = 5 ∨ x = 7
  let is_special (x : Nat) : Prop := is_perfect_square x ∨ is_prime x
  let outcomes := red_die.Product green_die
  let successful_outcomes := {(r, g) | r ∈ red_die ∧ g ∈ green_die ∧ is_odd r ∧ is_special g}
  let total_outcomes := 8 * 8
  let successful_outcomes_count := 4 * 6
  let probability := successful_outcomes_count / total_outcomes
  in probability = 3 / 8 :=
by {
  sorry
}

end probability_red_odd_green_special_l478_478922


namespace exists_consecutive_set_divisor_lcm_l478_478075

theorem exists_consecutive_set_divisor_lcm (n : ℕ) (h : n ≥ 4) :
  ∃ (A : Finset ℕ), A.card = n ∧
  ∀ (m : ℕ), m ∈ A ∧ m = A.max' (Finset.card_pos.2 (Nat.pos_of_ne_zero (ne_of_gt h))) → 
  m ∣ ∏ x in A.erase m, x :=
by
  sorry

end exists_consecutive_set_divisor_lcm_l478_478075


namespace probability_of_selecting_A_and_B_is_three_tenths_l478_478593

-- Define the set of students
def students : Finset ℕ := {0, 1, 2, 3, 4}

-- Define subsets A and B
def student_A : ℕ := 0
def student_B : ℕ := 1

-- Total number of ways to select 3 students out of 5
def total_ways : ℕ := students.card.choose 3

-- Number of ways to select 3 students including A and B
def favorable_ways : ℕ := {2, 3, 4}.card.choose 1

-- The probability
def probability_of_selecting_A_and_B : ℚ := (favorable_ways : ℚ) / (total_ways : ℚ)

theorem probability_of_selecting_A_and_B_is_three_tenths :
  probability_of_selecting_A_and_B = 3 / 10 :=
by
  sorry

end probability_of_selecting_A_and_B_is_three_tenths_l478_478593


namespace probability_of_A_and_B_selected_l478_478483

variable {α : Type}

/-- From 5 students, we randomly select 3 students to participate in community service work. 
  Let A and B be two specific students among these five. The probability that 
  both A and B are selected is 3 / 10. -/
theorem probability_of_A_and_B_selected (A B : α) (s : Finset α) (h : s.card = 5) :
  (∃ t : Finset α, t ⊆ s ∧ A ∈ t ∧ B ∈ t ∧ (t.card = 3) ∧ (t.card = 3) →
   (@Finset.exists_subset_card_eq α t s 3) →
   (@Finset.exists_subset_card_eq α t (Finset.erase s A) 2) ∧
   (@Finset.exists_subset_card_eq α t (Finset.erase (Finset.erase s A) B) 1)) →
  (3 : ℚ) / 10 :=
by
  sorry -- Proof omitted as per instructions.

end probability_of_A_and_B_selected_l478_478483


namespace probability_both_A_and_B_selected_l478_478398

-- The universe of students we consider
inductive Student
| A : Student
| B : Student
| C : Student
| D : Student
| E : Student

open Student

-- Function to count combinations
def combinations (n k : ℕ) : ℕ := (Nat.factorial n) / ((Nat.factorial k) * (Nat.factorial (n - k)))

-- The set of all students
def students : List Student := [A, B, C, D, E]

-- The total number of ways to select 3 students from 5
def total_selections := combinations 5 3

-- The number of ways A and B are both selected by selecting 1 more from the remaining 3 students
def selections_AB := combinations 3 1

-- The probability that both A and B are selected
def probability_AB := (selections_AB : ℚ) / (total_selections : ℚ)

-- The proof statement
theorem probability_both_A_and_B_selected :
  probability_AB = 3 / 10 :=
sorry

end probability_both_A_and_B_selected_l478_478398


namespace probability_A_and_B_selected_l478_478699

/-- There are 5 students including A and B. The probability of selecting both A and B 
    when 3 students are selected randomly from 5 is 3/10. -/
theorem probability_A_and_B_selected :
  (∃ (students : List String) (h : students.length = 5),
    ∃ (selected : List String) (h' : selected.length = 3),
    "A" ∈ selected ∧ "B" ∈ selected) →
  ∃ (P : ℚ), P = 3 / 10 :=
by
  sorry

end probability_A_and_B_selected_l478_478699


namespace probability_of_selecting_A_and_B_l478_478275

-- Define the combination function
def combination (n r : ℕ) : ℕ := Nat.factorial n / (Nat.factorial r * Nat.factorial (n - r))

-- Define the conditions
def students : ℕ := 5
def to_select : ℕ := 3

-- Define the favorable outcomes where A and B are selected
def favorable_outcomes : ℕ := combination 3 1

-- Define the total outcomes of selecting 3 out of 5 students
def total_outcomes : ℕ := combination 5 3

-- Define the probability calculation
def probability_of_A_and_B : ℚ := favorable_outcomes / total_outcomes

-- Proposition to prove
theorem probability_of_selecting_A_and_B : probability_of_A_and_B = 3 / 10 :=
by
  sorry

end probability_of_selecting_A_and_B_l478_478275


namespace probability_A_and_B_selected_l478_478526

theorem probability_A_and_B_selected :
  let n := 5
  let r := 3
  let total_ways := Nat.choose n r
  let favorable_ways := Nat.choose 3 1
  let probability := favorable_ways / total_ways
  probability = (3 / 10) := 
by
  sorry

end probability_A_and_B_selected_l478_478526


namespace probability_of_selecting_A_and_B_is_three_tenths_l478_478596

-- Define the set of students
def students : Finset ℕ := {0, 1, 2, 3, 4}

-- Define subsets A and B
def student_A : ℕ := 0
def student_B : ℕ := 1

-- Total number of ways to select 3 students out of 5
def total_ways : ℕ := students.card.choose 3

-- Number of ways to select 3 students including A and B
def favorable_ways : ℕ := {2, 3, 4}.card.choose 1

-- The probability
def probability_of_selecting_A_and_B : ℚ := (favorable_ways : ℚ) / (total_ways : ℚ)

theorem probability_of_selecting_A_and_B_is_three_tenths :
  probability_of_selecting_A_and_B = 3 / 10 :=
by
  sorry

end probability_of_selecting_A_and_B_is_three_tenths_l478_478596


namespace projection_of_a_onto_b_l478_478730

def vector_a : Vector ℝ 3 := ![1, 0, -1]
def vector_b : Vector ℝ 3 := ![-1, 1, 0]

theorem projection_of_a_onto_b :
  (∥(vector_a ⬝ vector_b : ℝ) / ∥vector_b∥ ^ 2 • vector_b∥ = ![1 / 2, -1 / 2, 0]) :=
by sorry

end projection_of_a_onto_b_l478_478730


namespace select_3_from_5_prob_l478_478434

theorem select_3_from_5_prob {students : Finset ℕ} 
  (h : students.card = 5) 
  {A B : ℕ} 
  (hA : A ∈ students) 
  (hB : B ∈ students) 
  (H : 3 ≤ students.card) : 
  (finset.card (students.filter (λ x, x ≠ A ∧ x ≠ B)) = 3) → 
  (finset.card ((students.filter (λ x, x ≠ A ∧ x ≠ B)).subtype (λ x, True)) = 3) → 
  ((finset.card (finset.filter (λ (students : Finset ℕ) A ∈ students ∧ B ∈ students ↔ 3 = students.card) 5)) = ↑10) → 
  ∃ students', (students'.card = 3 ∧ A ∈ students' ∧ B ∈ students') → 
  ∃ (p : ℚ), p = 3 / 10 ∧ students.filter (λ x, x = A ∨ x = B) = 2 :=
  sorry

end select_3_from_5_prob_l478_478434


namespace probability_A_B_l478_478458

open Nat

-- Define a function for combination
def combination (n r : Nat) : Nat :=
  factorial n / (factorial r * factorial (n - r))

-- Define the set of students and the selection conditions
def num_students : Nat := 5
def num_selected : Nat := 3

-- Define A and B as selected students
def selected_A_B : Nat := combination 3 1

-- Define the probability calculation
def probability_A_B_selected : Rat :=
  (selected_A_B : Rat) / (combination num_students num_selected : Rat)

-- The statement of the problem
theorem probability_A_B : probability_A_B_selected = 3 / 10 := by
  sorry

end probability_A_B_l478_478458


namespace symmetry_g_l478_478754

noncomputable def φ : ℝ := π / 6

def f (x : ℝ) : ℝ := 2 * sin x * sin (x + 3 * φ)
def g (x : ℝ) : ℝ := cos (2 * x - φ)

theorem symmetry_g :
  (∀ x, f x = -f (-x)) → φ ∈ (0 : ℝ, π / 2) →
  ∃ k : ℤ, -5 * π / 12 = k * π / 2 + π / 12 :=
by
  sorry

end symmetry_g_l478_478754


namespace probability_both_A_B_selected_l478_478212

-- Define the problem with the necessary conditions.
def probability_AB_selected (total_students : ℕ) (select_students : ℕ) (A B : ℕ) : ℚ :=
  if A < total_students ∧ B < total_students ∧ total_students = 5 ∧ select_students = 3 then
    let total_ways := (Nat.factorial total_students) / ((Nat.factorial select_students) * (Nat.factorial (total_students - select_students)))
    let favorable_ways := (Nat.factorial (total_students - 2)) / ((Nat.factorial (select_students - 2)) * (Nat.factorial ((total_students - 2) - (select_students - 2))))
    favorable_ways / total_ways
  else 0

theorem probability_both_A_B_selected 
  (total_students : ℕ) (select_students : ℕ) (A B : ℕ) 
  (h1 : total_students = 5) (h2 : select_students = 3) 
  (h3 : A < total_students) (h4 : B < total_students) 
  : probability_AB_selected total_students select_students A B = 3 / 10 := 
by {
  -- Insert the delta and logic to prove the theorem here.
  sorry
}

end probability_both_A_B_selected_l478_478212


namespace probability_of_selecting_A_and_B_is_three_tenths_l478_478603

-- Define the set of students
def students : Finset ℕ := {0, 1, 2, 3, 4}

-- Define subsets A and B
def student_A : ℕ := 0
def student_B : ℕ := 1

-- Total number of ways to select 3 students out of 5
def total_ways : ℕ := students.card.choose 3

-- Number of ways to select 3 students including A and B
def favorable_ways : ℕ := {2, 3, 4}.card.choose 1

-- The probability
def probability_of_selecting_A_and_B : ℚ := (favorable_ways : ℚ) / (total_ways : ℚ)

theorem probability_of_selecting_A_and_B_is_three_tenths :
  probability_of_selecting_A_and_B = 3 / 10 :=
by
  sorry

end probability_of_selecting_A_and_B_is_three_tenths_l478_478603


namespace determine_x_l478_478150

theorem determine_x (x : ℝ) :
  (∀ y : ℝ, 10 * x * y - 15 * y + 3 * x - 9 / 2 = 0) → x = 3 / 2 :=
by
  intro h
  sorry

end determine_x_l478_478150


namespace probability_A_and_B_selected_l478_478706

/-- There are 5 students including A and B. The probability of selecting both A and B 
    when 3 students are selected randomly from 5 is 3/10. -/
theorem probability_A_and_B_selected :
  (∃ (students : List String) (h : students.length = 5),
    ∃ (selected : List String) (h' : selected.length = 3),
    "A" ∈ selected ∧ "B" ∈ selected) →
  ∃ (P : ℚ), P = 3 / 10 :=
by
  sorry

end probability_A_and_B_selected_l478_478706


namespace probability_of_selecting_A_and_B_l478_478296

open_locale classical

-- Define the number of students
def num_students := 5

-- Define the number of students to be selected
def students_to_select := 3

def combination (n k : ℕ) : ℕ := nat.choose n k

-- Define the correct answer
def correct_probability := 3 / 10

-- Define the probability calculation
def calc_probability := (combination 3 1) / (combination 5 3)

-- Statement of the theorem to be proved
theorem probability_of_selecting_A_and_B :
  calc_probability = correct_probability :=
by sorry

end probability_of_selecting_A_and_B_l478_478296


namespace probability_A_B_selected_l478_478361

noncomputable def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

def combination (n r : ℕ) : ℕ :=
  factorial n / (factorial r * factorial (n - r))

theorem probability_A_B_selected :
  let total_ways := combination 5 3 in
  let ways_A_B := combination 3 1 in
  (ways_A_B : ℚ) / total_ways = 3 / 10 :=
by
  sorry

end probability_A_B_selected_l478_478361


namespace probability_both_A_and_B_selected_l478_478401

-- The universe of students we consider
inductive Student
| A : Student
| B : Student
| C : Student
| D : Student
| E : Student

open Student

-- Function to count combinations
def combinations (n k : ℕ) : ℕ := (Nat.factorial n) / ((Nat.factorial k) * (Nat.factorial (n - k)))

-- The set of all students
def students : List Student := [A, B, C, D, E]

-- The total number of ways to select 3 students from 5
def total_selections := combinations 5 3

-- The number of ways A and B are both selected by selecting 1 more from the remaining 3 students
def selections_AB := combinations 3 1

-- The probability that both A and B are selected
def probability_AB := (selections_AB : ℚ) / (total_selections : ℚ)

-- The proof statement
theorem probability_both_A_and_B_selected :
  probability_AB = 3 / 10 :=
sorry

end probability_both_A_and_B_selected_l478_478401


namespace probability_A_and_B_selected_l478_478571

/-- Let there be a group of 5 students including A and B. 
    If 3 students are selected randomly from these 5, the probability of selecting both A and B is 3/10. -/
theorem probability_A_and_B_selected : 
  let students := 5 in
  let selected := 3 in
  let combinations := students.choose selected in
  let favorable_combinations := (students - 2).choose (selected - 2) in
  let probability := (favorable_combinations : ℚ) / (combinations : ℚ) in
  probability = 3 / 10 := 
by sorry

end probability_A_and_B_selected_l478_478571


namespace average_decrease_of_seventh_observation_l478_478882

theorem average_decrease_of_seventh_observation :
  (avg_decrease : ℝ) (obs_6_avg : ℝ) (obs_7_new : ℝ) (num_obs_6 : ℕ) (num_obs_7 : ℕ)
  (sum_6 : ℝ) (sum_7 : ℝ) (new_avg : ℝ) : 
  obs_6_avg = 12 → 
  obs_7_new = 5 → 
  num_obs_6 = 6 → 
  num_obs_7 = 7 →
  sum_6 = num_obs_6 * obs_6_avg →
  sum_7 = sum_6 + obs_7_new →
  new_avg = sum_7 / num_obs_7 →
  avg_decrease = obs_6_avg - new_avg →
  avg_decrease = 1 :=
by
  intros avg_decrease obs_6_avg obs_7_new num_obs_6 num_obs_7 sum_6 sum_7 new_avg 
  intro h1 h2 h3 h4 h5 h6 h7 h8
  sorry

end average_decrease_of_seventh_observation_l478_478882


namespace initially_calculated_average_weight_l478_478009

theorem initially_calculated_average_weight (n : ℕ) (misread_diff correct_avg_weight : ℝ)
  (hn : n = 20) (hmisread_diff : misread_diff = 10) (hcorrect_avg_weight : correct_avg_weight = 58.9) :
  ((correct_avg_weight * n - misread_diff) / n) = 58.4 :=
by
  rw [hn, hmisread_diff, hcorrect_avg_weight]
  sorry

end initially_calculated_average_weight_l478_478009


namespace probability_A_and_B_selected_l478_478396

theorem probability_A_and_B_selected 
  (students : Finset ℕ) (A B : ℕ) 
  (h_students_size : students.card = 5) 
  (h_A_in_students : A ∈ students) 
  (h_B_in_students : B ∈ students)
  : (↑(Nat.choose 3) : ℚ) / (↑(Nat.choose 5) : ℚ) = 3 / 10 := 
by sorry

end probability_A_and_B_selected_l478_478396


namespace probability_A_and_B_selected_l478_478252

theorem probability_A_and_B_selected (students : Finset ℕ) (A B : ℕ) (h : A ∈ students ∧ B ∈ students ∧ students.card = 5) : 
  (students.choose 3).count (λ s, {A, B} ⊆ s) / (students.choose 3).card = 3 / 10 := by
  sorry

end probability_A_and_B_selected_l478_478252


namespace probability_A_and_B_selected_l478_478395

theorem probability_A_and_B_selected 
  (students : Finset ℕ) (A B : ℕ) 
  (h_students_size : students.card = 5) 
  (h_A_in_students : A ∈ students) 
  (h_B_in_students : B ∈ students)
  : (↑(Nat.choose 3) : ℚ) / (↑(Nat.choose 5) : ℚ) = 3 / 10 := 
by sorry

end probability_A_and_B_selected_l478_478395


namespace probability_A_and_B_selected_l478_478323

open Finset

noncomputable def total_ways : ℕ := (choose 5 3)

noncomputable def favorable_ways : ℕ := (choose 3 1)

noncomputable def probability (total favorable : ℕ) : ℚ := favorable / total

theorem probability_A_and_B_selected :
  probability total_ways favorable_ways = 3 / 10 :=
by
  sorry

end probability_A_and_B_selected_l478_478323


namespace probability_A_and_B_selected_l478_478692

/-- There are 5 students including A and B. The probability of selecting both A and B 
    when 3 students are selected randomly from 5 is 3/10. -/
theorem probability_A_and_B_selected :
  (∃ (students : List String) (h : students.length = 5),
    ∃ (selected : List String) (h' : selected.length = 3),
    "A" ∈ selected ∧ "B" ∈ selected) →
  ∃ (P : ℚ), P = 3 / 10 :=
by
  sorry

end probability_A_and_B_selected_l478_478692


namespace probability_of_selecting_A_and_B_l478_478676

open_locale classical

noncomputable def select_probability (n k : ℕ) : ℚ :=
  (nat.choose n k)⁻¹

theorem probability_of_selecting_A_and_B :
  let total_students := 5,
      students_to_select := 3,
      remaining_students := 3 in
  select_probability total_students students_to_select * nat.choose remaining_students 1 = 3 / 10 :=
by
  sorry

end probability_of_selecting_A_and_B_l478_478676


namespace probability_of_selecting_A_and_B_l478_478682

open_locale classical

noncomputable def select_probability (n k : ℕ) : ℚ :=
  (nat.choose n k)⁻¹

theorem probability_of_selecting_A_and_B :
  let total_students := 5,
      students_to_select := 3,
      remaining_students := 3 in
  select_probability total_students students_to_select * nat.choose remaining_students 1 = 3 / 10 :=
by
  sorry

end probability_of_selecting_A_and_B_l478_478682


namespace select_3_from_5_prob_l478_478433

theorem select_3_from_5_prob {students : Finset ℕ} 
  (h : students.card = 5) 
  {A B : ℕ} 
  (hA : A ∈ students) 
  (hB : B ∈ students) 
  (H : 3 ≤ students.card) : 
  (finset.card (students.filter (λ x, x ≠ A ∧ x ≠ B)) = 3) → 
  (finset.card ((students.filter (λ x, x ≠ A ∧ x ≠ B)).subtype (λ x, True)) = 3) → 
  ((finset.card (finset.filter (λ (students : Finset ℕ) A ∈ students ∧ B ∈ students ↔ 3 = students.card) 5)) = ↑10) → 
  ∃ students', (students'.card = 3 ∧ A ∈ students' ∧ B ∈ students') → 
  ∃ (p : ℚ), p = 3 / 10 ∧ students.filter (λ x, x = A ∨ x = B) = 2 :=
  sorry

end select_3_from_5_prob_l478_478433


namespace probability_both_A_B_selected_l478_478214

-- Define the problem with the necessary conditions.
def probability_AB_selected (total_students : ℕ) (select_students : ℕ) (A B : ℕ) : ℚ :=
  if A < total_students ∧ B < total_students ∧ total_students = 5 ∧ select_students = 3 then
    let total_ways := (Nat.factorial total_students) / ((Nat.factorial select_students) * (Nat.factorial (total_students - select_students)))
    let favorable_ways := (Nat.factorial (total_students - 2)) / ((Nat.factorial (select_students - 2)) * (Nat.factorial ((total_students - 2) - (select_students - 2))))
    favorable_ways / total_ways
  else 0

theorem probability_both_A_B_selected 
  (total_students : ℕ) (select_students : ℕ) (A B : ℕ) 
  (h1 : total_students = 5) (h2 : select_students = 3) 
  (h3 : A < total_students) (h4 : B < total_students) 
  : probability_AB_selected total_students select_students A B = 3 / 10 := 
by {
  -- Insert the delta and logic to prove the theorem here.
  sorry
}

end probability_both_A_B_selected_l478_478214


namespace find_length_and_area_time_at_o_find_ab_equation_when_angle_is_90_l478_478746

-- Definitions from the conditions
def on_circle (A B : Point) : Prop := A ∈ circle(0, 2) ∧ B ∈ circle(0, 2)
def on_line (C : Point) : Prop := C ∈ line (1, 1, 2)
def ab_parallel_l (A B C : Point) : Prop := line_through(A, B) ∥ line (1, 1, 2)
def ab_through_origin (A B : Point) : Prop := origin ∈ line_through(A, B)

-- Proof goals equivalent to the original problem
theorem find_length_and_area_time_at_o (A B C : Point) 
  (hc : on_circle A B)
  (hl : on_line C)
  (hp : ab_parallel_l A B C)
  (ho : ab_through_origin A B) : 
  length (A, B) = 4 ∧ area (A, B, C) = 2 * √2 := 
sorry

theorem find_ab_equation_when_angle_is_90 (A B C : Point)
  (hc : on_circle A B)
  (hl : on_line C)
  (hp : ab_parallel_l A B C)
  (h_angle : angle_between_lines (line_through(B, C)) = π/2)
  (hmax : ∀D, length (A, D) ≤ length (A, C))
  : equation (line_through (A, B)) = "y=x-(2/3)" := 
sorry

end find_length_and_area_time_at_o_find_ab_equation_when_angle_is_90_l478_478746


namespace simplest_square_root_l478_478936

theorem simplest_square_root : 
  (∀ x, x ∈ {sqrt 2, sqrt 12, sqrt (1/5), 1 / sqrt 2} → (√2) ≤ x) :=
by 
  intros x hx
  cases hx with
  | inl h => rfl
  | inr h1 =>
    cases h1 with
    | inl h2 =>
      calc
        √ 2 ≤ 2 * √ 3 : by norm_num
    | inr h2 =>
      cases h2 with
      | inl h3 =>
        calc
          √ 2 ≤ √ 5 / 5 : by norm_num
        sorry -- detailed calculations would follow, skipping for brevity

end simplest_square_root_l478_478936


namespace a_must_be_negative_l478_478735

variable (a b c d e : ℝ)

theorem a_must_be_negative
  (h1 : a / b < -c / d)
  (hb : b > 0)
  (hd : d > 0)
  (he : e > 0)
  (h2 : a + e > 0) : a < 0 := by
  sorry

end a_must_be_negative_l478_478735


namespace probability_A_and_B_selected_l478_478690

/-- There are 5 students including A and B. The probability of selecting both A and B 
    when 3 students are selected randomly from 5 is 3/10. -/
theorem probability_A_and_B_selected :
  (∃ (students : List String) (h : students.length = 5),
    ∃ (selected : List String) (h' : selected.length = 3),
    "A" ∈ selected ∧ "B" ∈ selected) →
  ∃ (P : ℚ), P = 3 / 10 :=
by
  sorry

end probability_A_and_B_selected_l478_478690


namespace Julie_monthly_salary_l478_478808

theorem Julie_monthly_salary 
(hourly_rate : ℕ) (hours_per_day : ℕ) (days_per_week : ℕ) (weeks_per_month : ℕ) (missed_days : ℕ) 
(h1 : hourly_rate = 5) (h2 : hours_per_day = 8) 
(h3 : days_per_week = 6) (h4 : weeks_per_month = 4) 
(h5 : missed_days = 1) : 
hourly_rate * hours_per_day * days_per_week * weeks_per_month - hourly_rate * hours_per_day * missed_days = 920 :=
by sorry

end Julie_monthly_salary_l478_478808


namespace probability_A_and_B_selected_l478_478573

/-- Let there be a group of 5 students including A and B. 
    If 3 students are selected randomly from these 5, the probability of selecting both A and B is 3/10. -/
theorem probability_A_and_B_selected : 
  let students := 5 in
  let selected := 3 in
  let combinations := students.choose selected in
  let favorable_combinations := (students - 2).choose (selected - 2) in
  let probability := (favorable_combinations : ℚ) / (combinations : ℚ) in
  probability = 3 / 10 := 
by sorry

end probability_A_and_B_selected_l478_478573


namespace probability_of_selecting_A_and_B_is_three_tenths_l478_478612

-- Define the set of students
def students : Finset ℕ := {0, 1, 2, 3, 4}

-- Define subsets A and B
def student_A : ℕ := 0
def student_B : ℕ := 1

-- Total number of ways to select 3 students out of 5
def total_ways : ℕ := students.card.choose 3

-- Number of ways to select 3 students including A and B
def favorable_ways : ℕ := {2, 3, 4}.card.choose 1

-- The probability
def probability_of_selecting_A_and_B : ℚ := (favorable_ways : ℚ) / (total_ways : ℚ)

theorem probability_of_selecting_A_and_B_is_three_tenths :
  probability_of_selecting_A_and_B = 3 / 10 :=
by
  sorry

end probability_of_selecting_A_and_B_is_three_tenths_l478_478612


namespace tetrahedron_lateral_to_base_ratio_l478_478093

theorem tetrahedron_lateral_to_base_ratio (A B C D M N : Point)
  (h_tetrahedron : regular_tetrahedron A B C D)
  (h_midpoints : (2 : Point) → (BD = M ∧ CD = N))
  (h_plane : plane_intersection_through_midpoints A M N)
  : ratio_lateral_to_base A B C D = √6 :=
by
  sorry

end tetrahedron_lateral_to_base_ratio_l478_478093


namespace probability_AB_selected_l478_478503

def factorial (n : Nat) : Nat := if n = 0 then 1 else n * factorial (n - 1)

def choose (n k : Nat) : Nat := factorial n / (factorial k * factorial (n - k))

theorem probability_AB_selected (h1 : 5 = 5) (h2 : 3 = 3) : (choose 3 1) / (choose 5 3) = 3 / 10 := by
  sorry

end probability_AB_selected_l478_478503


namespace probability_of_selecting_A_and_B_l478_478664

open_locale classical

noncomputable def select_probability (n k : ℕ) : ℚ :=
  (nat.choose n k)⁻¹

theorem probability_of_selecting_A_and_B :
  let total_students := 5,
      students_to_select := 3,
      remaining_students := 3 in
  select_probability total_students students_to_select * nat.choose remaining_students 1 = 3 / 10 :=
by
  sorry

end probability_of_selecting_A_and_B_l478_478664


namespace probability_A_and_B_selected_l478_478379

theorem probability_A_and_B_selected 
  (students : Finset ℕ) (A B : ℕ) 
  (h_students_size : students.card = 5) 
  (h_A_in_students : A ∈ students) 
  (h_B_in_students : B ∈ students)
  : (↑(Nat.choose 3) : ℚ) / (↑(Nat.choose 5) : ℚ) = 3 / 10 := 
by sorry

end probability_A_and_B_selected_l478_478379


namespace probability_A_and_B_selected_l478_478531

theorem probability_A_and_B_selected :
  let n := 5
  let r := 3
  let total_ways := Nat.choose n r
  let favorable_ways := Nat.choose 3 1
  let probability := favorable_ways / total_ways
  probability = (3 / 10) := 
by
  sorry

end probability_A_and_B_selected_l478_478531


namespace factor_quadratic_l478_478165

theorem factor_quadratic : ∀ (x : ℝ), 16*x^2 - 40*x + 25 = (4*x - 5)^2 := by
  intro x
  sorry

end factor_quadratic_l478_478165


namespace probability_A_and_B_selected_l478_478583

/-- Let there be a group of 5 students including A and B. 
    If 3 students are selected randomly from these 5, the probability of selecting both A and B is 3/10. -/
theorem probability_A_and_B_selected : 
  let students := 5 in
  let selected := 3 in
  let combinations := students.choose selected in
  let favorable_combinations := (students - 2).choose (selected - 2) in
  let probability := (favorable_combinations : ℚ) / (combinations : ℚ) in
  probability = 3 / 10 := 
by sorry

end probability_A_and_B_selected_l478_478583


namespace probability_of_selecting_A_and_B_l478_478667

open_locale classical

noncomputable def select_probability (n k : ℕ) : ℚ :=
  (nat.choose n k)⁻¹

theorem probability_of_selecting_A_and_B :
  let total_students := 5,
      students_to_select := 3,
      remaining_students := 3 in
  select_probability total_students students_to_select * nat.choose remaining_students 1 = 3 / 10 :=
by
  sorry

end probability_of_selecting_A_and_B_l478_478667


namespace probability_A_and_B_selected_l478_478388

theorem probability_A_and_B_selected 
  (students : Finset ℕ) (A B : ℕ) 
  (h_students_size : students.card = 5) 
  (h_A_in_students : A ∈ students) 
  (h_B_in_students : B ∈ students)
  : (↑(Nat.choose 3) : ℚ) / (↑(Nat.choose 5) : ℚ) = 3 / 10 := 
by sorry

end probability_A_and_B_selected_l478_478388


namespace friend_driving_time_l478_478990

theorem friend_driving_time 
  (christina_speed : ℕ)
  (friend_speed : ℕ)
  (total_distance : ℕ)
  (christina_time_minutes : ℕ)
  (christina_distance : christina_speed * (christina_time_minutes / 60) = 30 * 3)
  (total_minus_christina_distance : total_distance - christina_distance = 210 - 90) :
  total_distance - christina_distance = friend_speed * (total_distance - christina_distance) / 40 ∧ friend_speed = 40 :=
by 
  sorry

end friend_driving_time_l478_478990


namespace triangle_CDM_area_l478_478920

theorem triangle_CDM_area :
  ∃ (m n p : ℕ), 
    (is_coprime m p) ∧ 
    (√n ∣ (m ∧ ¬ (∃ (k : ℕ), k^2 = n)) ) ∧ 
    { let AC := 8 : ℕ,
      let BC := 15 : ℕ,
      let AB := 17 : ℕ,
      let midAB := 17 / 2 : ℚ,
      let D := 17 : ℕ,
      let CN := 120 / 17 : ℚ,
      let AN := √(46 / 289) : ℚ,
      let MN := midAB - AN,
      let DM := √(867 / 4) : ℚ,
      let areaCDM := (1 / 2) * DM * MN,
      let form := 17 * √867 / 4,
      areaCDM = form
    } 
    ∧ m + n + p = 39963 :=
by sorry

end triangle_CDM_area_l478_478920


namespace probability_of_A_and_B_selected_l478_478486

variable {α : Type}

/-- From 5 students, we randomly select 3 students to participate in community service work. 
  Let A and B be two specific students among these five. The probability that 
  both A and B are selected is 3 / 10. -/
theorem probability_of_A_and_B_selected (A B : α) (s : Finset α) (h : s.card = 5) :
  (∃ t : Finset α, t ⊆ s ∧ A ∈ t ∧ B ∈ t ∧ (t.card = 3) ∧ (t.card = 3) →
   (@Finset.exists_subset_card_eq α t s 3) →
   (@Finset.exists_subset_card_eq α t (Finset.erase s A) 2) ∧
   (@Finset.exists_subset_card_eq α t (Finset.erase (Finset.erase s A) B) 1)) →
  (3 : ℚ) / 10 :=
by
  sorry -- Proof omitted as per instructions.

end probability_of_A_and_B_selected_l478_478486


namespace select_3_from_5_prob_l478_478423

theorem select_3_from_5_prob {students : Finset ℕ} 
  (h : students.card = 5) 
  {A B : ℕ} 
  (hA : A ∈ students) 
  (hB : B ∈ students) 
  (H : 3 ≤ students.card) : 
  (finset.card (students.filter (λ x, x ≠ A ∧ x ≠ B)) = 3) → 
  (finset.card ((students.filter (λ x, x ≠ A ∧ x ≠ B)).subtype (λ x, True)) = 3) → 
  ((finset.card (finset.filter (λ (students : Finset ℕ) A ∈ students ∧ B ∈ students ↔ 3 = students.card) 5)) = ↑10) → 
  ∃ students', (students'.card = 3 ∧ A ∈ students' ∧ B ∈ students') → 
  ∃ (p : ℚ), p = 3 / 10 ∧ students.filter (λ x, x = A ∨ x = B) = 2 :=
  sorry

end select_3_from_5_prob_l478_478423


namespace probability_A_and_B_selected_l478_478691

/-- There are 5 students including A and B. The probability of selecting both A and B 
    when 3 students are selected randomly from 5 is 3/10. -/
theorem probability_A_and_B_selected :
  (∃ (students : List String) (h : students.length = 5),
    ∃ (selected : List String) (h' : selected.length = 3),
    "A" ∈ selected ∧ "B" ∈ selected) →
  ∃ (P : ℚ), P = 3 / 10 :=
by
  sorry

end probability_A_and_B_selected_l478_478691


namespace probability_of_selecting_A_and_B_l478_478292

open_locale classical

-- Define the number of students
def num_students := 5

-- Define the number of students to be selected
def students_to_select := 3

def combination (n k : ℕ) : ℕ := nat.choose n k

-- Define the correct answer
def correct_probability := 3 / 10

-- Define the probability calculation
def calc_probability := (combination 3 1) / (combination 5 3)

-- Statement of the theorem to be proved
theorem probability_of_selecting_A_and_B :
  calc_probability = correct_probability :=
by sorry

end probability_of_selecting_A_and_B_l478_478292


namespace probability_AB_selected_l478_478516

def factorial (n : Nat) : Nat := if n = 0 then 1 else n * factorial (n - 1)

def choose (n k : Nat) : Nat := factorial n / (factorial k * factorial (n - k))

theorem probability_AB_selected (h1 : 5 = 5) (h2 : 3 = 3) : (choose 3 1) / (choose 5 3) = 3 / 10 := by
  sorry

end probability_AB_selected_l478_478516


namespace probability_both_A_B_selected_l478_478217

-- Define the problem with the necessary conditions.
def probability_AB_selected (total_students : ℕ) (select_students : ℕ) (A B : ℕ) : ℚ :=
  if A < total_students ∧ B < total_students ∧ total_students = 5 ∧ select_students = 3 then
    let total_ways := (Nat.factorial total_students) / ((Nat.factorial select_students) * (Nat.factorial (total_students - select_students)))
    let favorable_ways := (Nat.factorial (total_students - 2)) / ((Nat.factorial (select_students - 2)) * (Nat.factorial ((total_students - 2) - (select_students - 2))))
    favorable_ways / total_ways
  else 0

theorem probability_both_A_B_selected 
  (total_students : ℕ) (select_students : ℕ) (A B : ℕ) 
  (h1 : total_students = 5) (h2 : select_students = 3) 
  (h3 : A < total_students) (h4 : B < total_students) 
  : probability_AB_selected total_students select_students A B = 3 / 10 := 
by {
  -- Insert the delta and logic to prove the theorem here.
  sorry
}

end probability_both_A_B_selected_l478_478217


namespace probability_A_and_B_selected_l478_478244

theorem probability_A_and_B_selected (students : Finset ℕ) (A B : ℕ) (h : A ∈ students ∧ B ∈ students ∧ students.card = 5) : 
  (students.choose 3).count (λ s, {A, B} ⊆ s) / (students.choose 3).card = 3 / 10 := by
  sorry

end probability_A_and_B_selected_l478_478244


namespace david_average_marks_l478_478139

theorem david_average_marks :
  let English := 70
  let Mathematics := 63
  let Physics := 80
  let Chemistry := 63
  let Biology := 65
  (English + Mathematics + Physics + Chemistry + Biology) / 5 = 68.2 :=
by
  let English := 70
  let Mathematics := 63
  let Physics := 80
  let Chemistry := 63
  let Biology := 65
  have total_marks : ℕ := English + Mathematics + Physics + Chemistry + Biology
  have average_marks : ℝ := total_marks / 5
  show average_marks = 68.2 from sorry

end david_average_marks_l478_478139


namespace equation_solution_l478_478826

theorem equation_solution {n : ℕ} (hpos : 0 < n) 
  (hsum : ∑ i in finset.Ico 2 (nat.floor ((n - 1) / 3) + 1), i - 1 = 28) :
  n = 25 ∨ n = 28 :=
sorry

end equation_solution_l478_478826


namespace work_problem_l478_478941

theorem work_problem (x : ℕ) (h1 : A_rate = 3 * B_rate) (h2 : (1/B_rate + 1/A_rate) = 1/18) : A_days = 24 := by
  have hB : B_rate * 18 = x := 
  calc 
    ... : sorry

  have hA : A_rate * 18 = x / 3 := 
  calc 
    ... : sorry

  have combined_rate: 4 / x = 1 / 18 := by sorry

  have B_days: x = 72 := by 
  calc
    4 * 18 = x := 
    sorry

  have A_days: 24 = 72 / 3 := by sorry

  exact A_days

end work_problem_l478_478941


namespace candy_difference_l478_478976

theorem candy_difference :
  ∀ (given left : ℝ), given = 6.25 → left = 4.75 → (given - left) = 1.50 :=
by
  intros given left Hgiven Hleft
  rw [Hgiven, Hleft]
  norm_num
  sorry

end candy_difference_l478_478976


namespace probability_A_and_B_selected_l478_478330

theorem probability_A_and_B_selected (A B : Type) (S : Finset (Type)) (h : S.card = 5) (hA : A ∈ S) (hB : B ∈ S) (T : Finset (Type)) (hT : T.card = 3) :
  (∃ T ⊆ S, A ∈ T ∧ B ∈ T ∧ T.card = 3) → 
  ∃ T ⊆ S, T.card = 3 ∧ (finset.card (T ∩ {A, B}) = 2) → 
  (finset.card (T ∩ {A, B}) = 2) = (3 / 10 : Real) := sorry

end probability_A_and_B_selected_l478_478330


namespace Felix_distance_proof_l478_478883

def average_speed : ℕ := 66
def twice_speed : ℕ := 2 * average_speed
def driving_hours : ℕ := 4
def distance_covered : ℕ := twice_speed * driving_hours

theorem Felix_distance_proof : distance_covered = 528 := by
  sorry

end Felix_distance_proof_l478_478883


namespace height_at_time_two_l478_478111

-- Question conditions
def initial_velocity : ℝ := 50
def air_resistance_factor : ℝ := 0.05
def time : ℝ := 2

-- Given condition for height model
def height (t v0 k : ℝ) : ℝ := -16 * t^2 + v0 * t * (1 - k * t) + 140

-- Statement to prove the condition height at t=2 as 166 after considering air resistance
theorem height_at_time_two : height time initial_velocity air_resistance_factor = 166 := by
  -- Here we omit the proof steps, but we assert that the height is 166
  sorry

end height_at_time_two_l478_478111


namespace problem_l478_478141

-- Define the problem statement
theorem problem 
  (g : ℝ → ℝ := λ x, 3 * x + 2)
  (f : ℝ → ℝ := λ x, b * x + c)
  (h : ∀ x, g x = 2 * inverse f x + 4)
  (h_inv : ∀ x, inverse f (f x) = x)
  (b c : ℚ) :
  3 * b + 4 * c = 14 / 3 :=
sorry

end problem_l478_478141


namespace probability_A_and_B_selected_l478_478693

/-- There are 5 students including A and B. The probability of selecting both A and B 
    when 3 students are selected randomly from 5 is 3/10. -/
theorem probability_A_and_B_selected :
  (∃ (students : List String) (h : students.length = 5),
    ∃ (selected : List String) (h' : selected.length = 3),
    "A" ∈ selected ∧ "B" ∈ selected) →
  ∃ (P : ℚ), P = 3 / 10 :=
by
  sorry

end probability_A_and_B_selected_l478_478693


namespace probability_A_B_selected_l478_478357

noncomputable def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

def combination (n r : ℕ) : ℕ :=
  factorial n / (factorial r * factorial (n - r))

theorem probability_A_B_selected :
  let total_ways := combination 5 3 in
  let ways_A_B := combination 3 1 in
  (ways_A_B : ℚ) / total_ways = 3 / 10 :=
by
  sorry

end probability_A_B_selected_l478_478357


namespace probability_A_and_B_selected_l478_478325

theorem probability_A_and_B_selected (A B : Type) (S : Finset (Type)) (h : S.card = 5) (hA : A ∈ S) (hB : B ∈ S) (T : Finset (Type)) (hT : T.card = 3) :
  (∃ T ⊆ S, A ∈ T ∧ B ∈ T ∧ T.card = 3) → 
  ∃ T ⊆ S, T.card = 3 ∧ (finset.card (T ∩ {A, B}) = 2) → 
  (finset.card (T ∩ {A, B}) = 2) = (3 / 10 : Real) := sorry

end probability_A_and_B_selected_l478_478325


namespace range_g_l478_478930

noncomputable def g (x : ℝ) : ℝ := x / (x + 1)^2

theorem range_g : (Set.range (λ x : {x : ℝ // x ≠ -1}, g x) = Set.Iic (1/4)) :=
sorry

end range_g_l478_478930


namespace prob_select_A_and_B_l478_478629

theorem prob_select_A_and_B : 
  let total_students := 5
  let to_select := 3
  let favorable_outcomes := Nat.div_fact (Nat.factorial 3) (Nat.factorial 1)
  let total_outcomes := Nat.div_fact (Nat.factorial 5) (Nat.factorial 3 * Nat.factorial (5 - 3))
  (favorable_outcomes : ℚ) / total_outcomes = 3 / 10 :=
by
  sorry

end prob_select_A_and_B_l478_478629


namespace probability_A_and_B_selected_l478_478572

/-- Let there be a group of 5 students including A and B. 
    If 3 students are selected randomly from these 5, the probability of selecting both A and B is 3/10. -/
theorem probability_A_and_B_selected : 
  let students := 5 in
  let selected := 3 in
  let combinations := students.choose selected in
  let favorable_combinations := (students - 2).choose (selected - 2) in
  let probability := (favorable_combinations : ℚ) / (combinations : ℚ) in
  probability = 3 / 10 := 
by sorry

end probability_A_and_B_selected_l478_478572


namespace probability_of_A_and_B_selected_l478_478472

variable {α : Type}

/-- From 5 students, we randomly select 3 students to participate in community service work. 
  Let A and B be two specific students among these five. The probability that 
  both A and B are selected is 3 / 10. -/
theorem probability_of_A_and_B_selected (A B : α) (s : Finset α) (h : s.card = 5) :
  (∃ t : Finset α, t ⊆ s ∧ A ∈ t ∧ B ∈ t ∧ (t.card = 3) ∧ (t.card = 3) →
   (@Finset.exists_subset_card_eq α t s 3) →
   (@Finset.exists_subset_card_eq α t (Finset.erase s A) 2) ∧
   (@Finset.exists_subset_card_eq α t (Finset.erase (Finset.erase s A) B) 1)) →
  (3 : ℚ) / 10 :=
by
  sorry -- Proof omitted as per instructions.

end probability_of_A_and_B_selected_l478_478472


namespace max_yellow_apples_can_take_max_total_apples_can_take_l478_478029

structure Basket :=
  (total_apples : ℕ)
  (green_apples : ℕ)
  (yellow_apples : ℕ)
  (red_apples : ℕ)
  (green_lt_yellow : green_apples < yellow_apples)
  (yellow_lt_red : yellow_apples < red_apples)

def basket_conditions : Basket :=
  { total_apples := 44,
    green_apples := 11,
    yellow_apples := 14,
    red_apples := 19,
    green_lt_yellow := sorry,  -- 11 < 14
    yellow_lt_red := sorry }   -- 14 < 19

theorem max_yellow_apples_can_take : basket_conditions.yellow_apples = 14 := sorry

theorem max_total_apples_can_take : basket_conditions.green_apples 
                                     + basket_conditions.yellow_apples 
                                     + (basket_conditions.red_apples - 2) = 42 := sorry

end max_yellow_apples_can_take_max_total_apples_can_take_l478_478029


namespace factor_quadratic_l478_478169

theorem factor_quadratic : ∀ (x : ℝ), 16*x^2 - 40*x + 25 = (4*x - 5)^2 := by
  intro x
  sorry

end factor_quadratic_l478_478169


namespace average_germination_is_correct_l478_478202

structure Plot where
  seeds_planted: ℕ
  germination_rate: ℚ

def plot1 : Plot := { seeds_planted := 300, germination_rate := 0.25 }
def plot2 : Plot := { seeds_planted := 200, germination_rate := 0.40 }
def plot3 : Plot := { seeds_planted := 500, germination_rate := 0.30 }
def plot4 : Plot := { seeds_planted := 400, germination_rate := 0.35 }
def plot5 : Plot := { seeds_planted := 600, germination_rate := 0.20 }

def plots : List Plot := [plot1, plot2, plot3, plot4, plot5]

def total_germinated (plots: List Plot) : ℕ := 
  plots.foldl (λ acc plot => acc + (plot.seeds_planted * plot.germination_rate.num).nat_abs) 0

def total_planted (plots: List Plot) : ℕ := 
  plots.foldl (λ acc plot => acc + plot.seeds_planted) 0

def average_germination_percentage (plots: List Plot) : ℚ := 
  (total_germinated plots : ℚ) / (total_planted plots : ℚ) * 100

theorem average_germination_is_correct : average_germination_percentage plots = 28.25 := 
by
  -- Proof steps would go here
  sorry

end average_germination_is_correct_l478_478202


namespace prob_select_A_and_B_l478_478634

theorem prob_select_A_and_B : 
  let total_students := 5
  let to_select := 3
  let favorable_outcomes := Nat.div_fact (Nat.factorial 3) (Nat.factorial 1)
  let total_outcomes := Nat.div_fact (Nat.factorial 5) (Nat.factorial 3 * Nat.factorial (5 - 3))
  (favorable_outcomes : ℚ) / total_outcomes = 3 / 10 :=
by
  sorry

end prob_select_A_and_B_l478_478634


namespace base_number_is_4_l478_478056

theorem base_number_is_4 : 
  ∃ x : ℝ, (x^3 = 1024 * (1 / 4)^2) ∧ x = 4 :=
begin
  use 4,
  split,
  { 
    calc 4^3 = 64         : by norm_num
        ... = 1024 / 16   : by norm_num
        ... = 1024 * (1 / 4)^2 : by norm_num,
  },
  { 
    norm_num,
  }
end

end base_number_is_4_l478_478056


namespace visibility_time_correct_l478_478116

noncomputable def visibility_time (r : ℝ) (d : ℝ) (v_j : ℝ) (v_k : ℝ) : ℝ :=
  (d / (v_j + v_k)) * (r / (r * (v_j / v_k + 1)))

theorem visibility_time_correct :
  visibility_time 60 240 4 2 = 120 :=
by
  sorry

end visibility_time_correct_l478_478116


namespace sum_x_coordinates_of_solutions_l478_478198

noncomputable def x_solution_sum : ℝ :=
  let f := λ x : ℝ, abs (x^2 - 8 * x + 15)
  let g := λ x : ℝ, x + 2
  let solutions := { x : ℝ | f x = g x }
  set.toFinset solutions
  sorry -- We need to sum the x-coordinates of the solutions

theorem sum_x_coordinates_of_solutions :
  x_solution_sum = 9 :=
sorry

end sum_x_coordinates_of_solutions_l478_478198


namespace probability_of_A_and_B_selected_l478_478488

variable {α : Type}

/-- From 5 students, we randomly select 3 students to participate in community service work. 
  Let A and B be two specific students among these five. The probability that 
  both A and B are selected is 3 / 10. -/
theorem probability_of_A_and_B_selected (A B : α) (s : Finset α) (h : s.card = 5) :
  (∃ t : Finset α, t ⊆ s ∧ A ∈ t ∧ B ∈ t ∧ (t.card = 3) ∧ (t.card = 3) →
   (@Finset.exists_subset_card_eq α t s 3) →
   (@Finset.exists_subset_card_eq α t (Finset.erase s A) 2) ∧
   (@Finset.exists_subset_card_eq α t (Finset.erase (Finset.erase s A) B) 1)) →
  (3 : ℚ) / 10 :=
by
  sorry -- Proof omitted as per instructions.

end probability_of_A_and_B_selected_l478_478488


namespace probability_both_A_and_B_selected_l478_478402

-- The universe of students we consider
inductive Student
| A : Student
| B : Student
| C : Student
| D : Student
| E : Student

open Student

-- Function to count combinations
def combinations (n k : ℕ) : ℕ := (Nat.factorial n) / ((Nat.factorial k) * (Nat.factorial (n - k)))

-- The set of all students
def students : List Student := [A, B, C, D, E]

-- The total number of ways to select 3 students from 5
def total_selections := combinations 5 3

-- The number of ways A and B are both selected by selecting 1 more from the remaining 3 students
def selections_AB := combinations 3 1

-- The probability that both A and B are selected
def probability_AB := (selections_AB : ℚ) / (total_selections : ℚ)

-- The proof statement
theorem probability_both_A_and_B_selected :
  probability_AB = 3 / 10 :=
sorry

end probability_both_A_and_B_selected_l478_478402


namespace probability_A_and_B_selected_l478_478337

theorem probability_A_and_B_selected (A B : Type) (S : Finset (Type)) (h : S.card = 5) (hA : A ∈ S) (hB : B ∈ S) (T : Finset (Type)) (hT : T.card = 3) :
  (∃ T ⊆ S, A ∈ T ∧ B ∈ T ∧ T.card = 3) → 
  ∃ T ⊆ S, T.card = 3 ∧ (finset.card (T ∩ {A, B}) = 2) → 
  (finset.card (T ∩ {A, B}) = 2) = (3 / 10 : Real) := sorry

end probability_A_and_B_selected_l478_478337


namespace probability_A_and_B_selected_l478_478233

theorem probability_A_and_B_selected (students : Finset ℕ) (A B : ℕ) (h : A ∈ students ∧ B ∈ students ∧ students.card = 5) : 
  (students.choose 3).count (λ s, {A, B} ⊆ s) / (students.choose 3).card = 3 / 10 := by
  sorry

end probability_A_and_B_selected_l478_478233


namespace function_conditions_satisfied_l478_478776

noncomputable def function_satisfying_conditions : ℝ → ℝ := fun x => -2 * x^2 + 3 * x

theorem function_conditions_satisfied :
  (function_satisfying_conditions 1 = 1) ∧
  (∃ x y : ℝ, x > 0 ∧ y < 0 ∧ function_satisfying_conditions x = y) ∧
  (∀ x y : ℝ, x > 1 ∧ y = function_satisfying_conditions x → ∃ ε > 0, ∀ δ > 0, (x + δ > 1 → function_satisfying_conditions (x + δ) < y)) :=
by
  sorry

end function_conditions_satisfied_l478_478776


namespace smallest_positive_integer_l478_478053

theorem smallest_positive_integer (n : ℕ) :
  (∃ n : ℕ, n > 0 ∧ n % 30 = 0 ∧ n % 40 = 0 ∧ n % 16 ≠ 0 ∧ ∀ m : ℕ, (m > 0 ∧ m % 30 = 0 ∧ m % 40 = 0 ∧ m % 16 ≠ 0) → n ≤ m) ↔ n = 120 :=
by
  sorry

end smallest_positive_integer_l478_478053


namespace tank_capacity_l478_478101

theorem tank_capacity (C : ℝ) :
  (3/4 * C - 0.4 * (3/4 * C) + 0.3 * (3/4 * C - 0.4 * (3/4 * C))) = 4680 → C = 8000 :=
by
  sorry

end tank_capacity_l478_478101


namespace exists_eight_points_with_perpendicular_bisectors_l478_478805

/--
Given 8 different points on a plane, we need to prove that for any given pair of points,
there exist two other points such that they lie on the perpendicular bisector of 
the segment connecting the pair.
-/
theorem exists_eight_points_with_perpendicular_bisectors :
  ∃ (A B C D X Y Z U : Point), 
  (∀ (P Q : Point), P ≠ Q → {P, Q} ⊆ {A, B, C, D, X, Y, Z, U} →
  ∃ R S : Point, R ≠ S ∧ {R, S} ⊆ {A, B, C, D, X, Y, Z, U} ∧
   is_perpendicular_bisector (segment P Q) (line_through R S)) :=
sorry

end exists_eight_points_with_perpendicular_bisectors_l478_478805


namespace brenda_peaches_remaining_l478_478119

theorem brenda_peaches_remaining (total_peaches : ℕ) (percent_fresh : ℚ) (thrown_away : ℕ) (fresh_peaches : ℕ) (remaining_peaches : ℕ) :
    total_peaches = 250 → 
    percent_fresh = 0.60 → 
    thrown_away = 15 → 
    fresh_peaches = total_peaches * percent_fresh → 
    remaining_peaches = fresh_peaches - thrown_away → 
    remaining_peaches = 135 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end brenda_peaches_remaining_l478_478119


namespace probability_of_selecting_A_and_B_l478_478561

theorem probability_of_selecting_A_and_B (students : Finset ℕ) (A B : ℕ)
  (h_size : students.card = 5) (h_in : A ∈ students ∧ B ∈ students) : 
  (Finset.card {s : Finset ℕ // s ⊆ students ∧ s.card = 3 ∧ A ∈ s ∧ B ∈ s}.to_set) / 
  (Finset.card {s : Finset ℕ // s ⊆ students ∧ s.card = 3}.to_set) = 3 / 10 :=
sorry

end probability_of_selecting_A_and_B_l478_478561


namespace log_expression_defined_l478_478146

theorem log_expression_defined (x : ℝ) : x > 3 ↔ 
  (x^2 - x - 2 > 0 ∧ x^2 - x - 3 > 0) :=
by
  split
  · intro hx
    exact ⟨by linarith, by linarith⟩
  · rintro ⟨h1, h2⟩
    linarith

end log_expression_defined_l478_478146


namespace select_3_from_5_prob_l478_478436

theorem select_3_from_5_prob {students : Finset ℕ} 
  (h : students.card = 5) 
  {A B : ℕ} 
  (hA : A ∈ students) 
  (hB : B ∈ students) 
  (H : 3 ≤ students.card) : 
  (finset.card (students.filter (λ x, x ≠ A ∧ x ≠ B)) = 3) → 
  (finset.card ((students.filter (λ x, x ≠ A ∧ x ≠ B)).subtype (λ x, True)) = 3) → 
  ((finset.card (finset.filter (λ (students : Finset ℕ) A ∈ students ∧ B ∈ students ↔ 3 = students.card) 5)) = ↑10) → 
  ∃ students', (students'.card = 3 ∧ A ∈ students' ∧ B ∈ students') → 
  ∃ (p : ℚ), p = 3 / 10 ∧ students.filter (λ x, x = A ∨ x = B) = 2 :=
  sorry

end select_3_from_5_prob_l478_478436


namespace mod_equivalence_l478_478048

theorem mod_equivalence (n : ℤ) (hn₁ : 0 ≤ n) (hn₂ : n < 23) (hmod : -250 % 23 = n % 23) : n = 3 := by
  sorry

end mod_equivalence_l478_478048


namespace monkeys_minus_giraffes_eq_neg_106_l478_478126

noncomputable def number_of_zebras : ℕ := 12
noncomputable def number_of_camels : ℕ := number_of_zebras / 2
noncomputable def number_of_monkeys : ℕ := 4 * number_of_camels
noncomputable def number_of_parrots : ℕ := (2 * number_of_monkeys) - 5
noncomputable def number_of_giraffes : ℕ := (3 * number_of_parrots) + 1

theorem monkeys_minus_giraffes_eq_neg_106 :
  number_of_monkeys - number_of_giraffes = -106 :=
by
  sorry  -- Proof is not required

end monkeys_minus_giraffes_eq_neg_106_l478_478126


namespace line_bisects_segment_l478_478828

theorem line_bisects_segment
  (A B C H D K L M N : Point)
  (h_acute_triangle : is_acute_triangle A B C)
  (h_AB_lt_AC : dist A B < dist A C)
  (h_orthocenter : is_orthocenter H A B C)
  (h_circle_AD_center_A_radius_AC : circle_intersects A (dist A C) (circumcircle A B C) D)
  (h_circle_AK_center_A_radius_AB : circle_intersects_seg A (dist A B) (seg A D) K)
  (h_parallel_KL_CD : parallel (line K L) (line C D))
  (h_intersects_KL_BC : intersects_at (line K L) (line B C) L)
  (h_midpoint_M_BC : midpoint M B C)
  (h_perpendicular_HN_AL : perpendicular_to H N (line A L)) :
  bisects (line M N) (seg A H) :=
sorry

end line_bisects_segment_l478_478828


namespace probability_A_and_B_selected_l478_478374

theorem probability_A_and_B_selected 
  (students : Finset ℕ) (A B : ℕ) 
  (h_students_size : students.card = 5) 
  (h_A_in_students : A ∈ students) 
  (h_B_in_students : B ∈ students)
  : (↑(Nat.choose 3) : ℚ) / (↑(Nat.choose 5) : ℚ) = 3 / 10 := 
by sorry

end probability_A_and_B_selected_l478_478374


namespace probability_A_and_B_selected_l478_478373

theorem probability_A_and_B_selected 
  (students : Finset ℕ) (A B : ℕ) 
  (h_students_size : students.card = 5) 
  (h_A_in_students : A ∈ students) 
  (h_B_in_students : B ∈ students)
  : (↑(Nat.choose 3) : ℚ) / (↑(Nat.choose 5) : ℚ) = 3 / 10 := 
by sorry

end probability_A_and_B_selected_l478_478373


namespace probability_AB_selected_l478_478515

def factorial (n : Nat) : Nat := if n = 0 then 1 else n * factorial (n - 1)

def choose (n k : Nat) : Nat := factorial n / (factorial k * factorial (n - k))

theorem probability_AB_selected (h1 : 5 = 5) (h2 : 3 = 3) : (choose 3 1) / (choose 5 3) = 3 / 10 := by
  sorry

end probability_AB_selected_l478_478515


namespace exists_good_set_of_perimeter_l478_478734

noncomputable def good_set (Γ : Metric.Sphere (0 : ℝ) 1) (T : Set (Metric.Tri)) : Prop :=
  ∀ (t : Metric.Tri), t ∈ T → t.inscribed Γ ∧ ∀ t₁ t₂ ∈ T, t₁ ≠ t₂ → ¬ (t₁ ∩ t₂).internal_point

theorem exists_good_set_of_perimeter (t : ℝ) : t ∈ Ioo 0 4 → (∀ n : ℕ, ∃ T : Set (Metric.Tri), good_set (Metric.Sphere (0 : ℝ) 1) T ∧ T.card = n ∧ ∀ (τ : Metric.Tri), τ ∈ T → τ.perimeter > t) :=
by
  sorry

end exists_good_set_of_perimeter_l478_478734


namespace jordan_width_45_l478_478127

noncomputable def carolRectangleLength : ℕ := 15
noncomputable def carolRectangleWidth : ℕ := 24
noncomputable def jordanRectangleLength : ℕ := 8
noncomputable def carolRectangleArea : ℕ := carolRectangleLength * carolRectangleWidth
noncomputable def jordanRectangleWidth (area : ℕ) : ℕ := area / jordanRectangleLength

theorem jordan_width_45 : jordanRectangleWidth carolRectangleArea = 45 :=
by sorry

end jordan_width_45_l478_478127


namespace factorial_expression_value_l478_478932

theorem factorial_expression_value : (10.factorial * 7.factorial * 3.factorial) / (9.factorial * 8.factorial) = 15 / 7 := by
  sorry

end factorial_expression_value_l478_478932


namespace lcm_inequality_l478_478143

noncomputable def lcm (a b : ℕ) : ℕ := a * b / Int.gcd a b

theorem lcm_inequality (n : ℕ) (hpos : 0 < n) (h : ∀ k : ℕ, 1 ≤ k ∧ k ≤ 35 → lcm n (n + k) > lcm n (n + k + 1)) :
  lcm n (n + 35) > lcm n (n + 36) :=
sorry

end lcm_inequality_l478_478143


namespace sam_distance_l478_478863

theorem sam_distance :
  ∀ (speed time : ℕ), speed = 4 ∧ time = 2 → speed * time = 8 :=
by
  intros speed time h
  cases h with hs ht
  rw [hs, ht]
  norm_num

end sam_distance_l478_478863


namespace probability_A_and_B_selected_l478_478320

open Finset

noncomputable def total_ways : ℕ := (choose 5 3)

noncomputable def favorable_ways : ℕ := (choose 3 1)

noncomputable def probability (total favorable : ℕ) : ℚ := favorable / total

theorem probability_A_and_B_selected :
  probability total_ways favorable_ways = 3 / 10 :=
by
  sorry

end probability_A_and_B_selected_l478_478320


namespace probability_A_B_l478_478462

open Nat

-- Define a function for combination
def combination (n r : Nat) : Nat :=
  factorial n / (factorial r * factorial (n - r))

-- Define the set of students and the selection conditions
def num_students : Nat := 5
def num_selected : Nat := 3

-- Define A and B as selected students
def selected_A_B : Nat := combination 3 1

-- Define the probability calculation
def probability_A_B_selected : Rat :=
  (selected_A_B : Rat) / (combination num_students num_selected : Rat)

-- The statement of the problem
theorem probability_A_B : probability_A_B_selected = 3 / 10 := by
  sorry

end probability_A_B_l478_478462


namespace probability_A_B_l478_478468

open Nat

-- Define a function for combination
def combination (n r : Nat) : Nat :=
  factorial n / (factorial r * factorial (n - r))

-- Define the set of students and the selection conditions
def num_students : Nat := 5
def num_selected : Nat := 3

-- Define A and B as selected students
def selected_A_B : Nat := combination 3 1

-- Define the probability calculation
def probability_A_B_selected : Rat :=
  (selected_A_B : Rat) / (combination num_students num_selected : Rat)

-- The statement of the problem
theorem probability_A_B : probability_A_B_selected = 3 / 10 := by
  sorry

end probability_A_B_l478_478468


namespace even_function_and_monotonicity_l478_478142

noncomputable def f (x : ℝ) : ℝ := sorry

theorem even_function_and_monotonicity (f_symm : ∀ x : ℝ, f x = f (-x))
  (f_inc_neg : ∀ ⦃x1 x2 : ℝ⦄, x1 < x2 → x1 ≤ 0 → x2 ≤ 0 → f x1 < f x2)
  (n : ℕ) (hn : n > 0) :
  f (n + 1) < f (-n) ∧ f (-n) < f (n - 1) := 
sorry

end even_function_and_monotonicity_l478_478142


namespace prob_select_A_and_B_l478_478624

theorem prob_select_A_and_B : 
  let total_students := 5
  let to_select := 3
  let favorable_outcomes := Nat.div_fact (Nat.factorial 3) (Nat.factorial 1)
  let total_outcomes := Nat.div_fact (Nat.factorial 5) (Nat.factorial 3 * Nat.factorial (5 - 3))
  (favorable_outcomes : ℚ) / total_outcomes = 3 / 10 :=
by
  sorry

end prob_select_A_and_B_l478_478624


namespace hulk_jump_geometric_seq_l478_478880

theorem hulk_jump_geometric_seq :
  ∃ n : ℕ, (2 * 3^(n-1) > 2000) ∧ n = 8 :=
by
  sorry

end hulk_jump_geometric_seq_l478_478880


namespace power_division_l478_478049

theorem power_division : 2^16 / 16^2 = 256 := by
  have h1 : 16 = 2^4 := by norm_num
  have h2 : 16^2 = 2^8 := by rw [h1, pow_mul]; norm_num
  rw [h2, pow_sub (ne_of_gt (pow_pos (by norm_num) 16))]
  norm_num
  sorry

end power_division_l478_478049


namespace percent_students_in_range_l478_478957

theorem percent_students_in_range
    (n1 n2 n3 n4 n5 : ℕ)
    (h1 : n1 = 5)
    (h2 : n2 = 7)
    (h3 : n3 = 8)
    (h4 : n4 = 4)
    (h5 : n5 = 3) :
  ((n3 : ℝ) / (n1 + n2 + n3 + n4 + n5) * 100) = 29.63 :=
by
  sorry

end percent_students_in_range_l478_478957


namespace find_PC_l478_478919

-- Define points A, B, C and P.
variables (A B C P : Point)

-- Define distances PA, PB and the right angle at B in triangle ABC.
variables (PA PB PC : ℝ)
variable (right_angle_at_B : right_angle (∠ B))

-- Define the angles and their properties.
variables (angle_APB angle_BPC angle_CPA : ℝ)
variable (sum_of_angles : angle_APB + angle_BPC + angle_CPA = 360)
variables (equal_angles_120 : angle_APB = 120 ∧ angle_BPC = 120 ∧ angle_CPA = 120)

-- Define the lengths PA = 10 and PB = 6.
axiom PA_eq_10 : PA = 10
axiom PB_eq_6 : PB = 6

-- Prove that PC = 33.
theorem find_PC : PC = 33 :=
by {
  sorry
}

end find_PC_l478_478919


namespace total_courses_l478_478838

-- Define the conditions as variables
def max_courses : Nat := 40
def sid_courses : Nat := 4 * max_courses

-- State the theorem we want to prove
theorem total_courses : max_courses + sid_courses = 200 := 
  by
    -- This is where the actual proof would go
    sorry

end total_courses_l478_478838


namespace probability_A_B_l478_478460

open Nat

-- Define a function for combination
def combination (n r : Nat) : Nat :=
  factorial n / (factorial r * factorial (n - r))

-- Define the set of students and the selection conditions
def num_students : Nat := 5
def num_selected : Nat := 3

-- Define A and B as selected students
def selected_A_B : Nat := combination 3 1

-- Define the probability calculation
def probability_A_B_selected : Rat :=
  (selected_A_B : Rat) / (combination num_students num_selected : Rat)

-- The statement of the problem
theorem probability_A_B : probability_A_B_selected = 3 / 10 := by
  sorry

end probability_A_B_l478_478460


namespace delta_5_zero_for_all_n_l478_478201

def u (n : ℕ) : ℕ := n^4 + n^2

def Δ (k : ℕ) (u : ℕ → ℕ) : (ℕ → ℕ)
| 0       := u
| (k + 1) := λ n, Δ k u (n + 1) - Δ k u n

theorem delta_5_zero_for_all_n : 
  (Δ 5 (λ n, u n) = λ n, 0) :=
sorry

end delta_5_zero_for_all_n_l478_478201
