import Mathlib

namespace margaret_age_in_12_years_l806_80618

theorem margaret_age_in_12_years
  (brian_age : ℝ)
  (christian_age : ℝ)
  (margaret_age : ℝ)
  (h1 : christian_age = 3.5 * brian_age)
  (h2 : brian_age + 12 = 45)
  (h3 : margaret_age = christian_age - 10) :
  margaret_age + 12 = 117.5 :=
by
  sorry

end margaret_age_in_12_years_l806_80618


namespace parabola_and_length_ef_l806_80656

theorem parabola_and_length_ef :
  ∃ a b : ℝ, (∀ x : ℝ, (x + 1) * (x - 3) = 0 → a * x^2 + b * x + 3 = 0) ∧ 
            (∀ x : ℝ, -a * x^2 + b * x + 3 = 7 / 4 → 
              ∃ x1 x2 : ℝ, x1 = -1 / 2 ∧ x2 = 5 / 2 ∧ abs (x2 - x1) = 3) := 
sorry

end parabola_and_length_ef_l806_80656


namespace option_D_functions_same_l806_80631

theorem option_D_functions_same (x : ℝ) : (x^2) = (x^6)^(1/3) :=
by 
  sorry

end option_D_functions_same_l806_80631


namespace sum_of_x_coordinates_mod_20_l806_80638

theorem sum_of_x_coordinates_mod_20 (y x : ℤ) (h1 : y ≡ 7 * x + 3 [ZMOD 20]) (h2 : y ≡ 13 * x + 17 [ZMOD 20]) 
: ∃ (x1 x2 : ℤ), (0 ≤ x1 ∧ x1 < 20) ∧ (0 ≤ x2 ∧ x2 < 20) ∧ x1 ≡ 1 [ZMOD 10] ∧ x2 ≡ 11 [ZMOD 10] ∧ x1 + x2 = 12 := sorry

end sum_of_x_coordinates_mod_20_l806_80638


namespace notebook_and_pencil_cost_l806_80666

theorem notebook_and_pencil_cost :
  ∃ (x y : ℝ), 6 * x + 4 * y = 9.2 ∧ 3 * x + y = 3.8 ∧ x + y = 1.8 :=
by
  sorry

end notebook_and_pencil_cost_l806_80666


namespace greatest_int_with_gcd_of_24_eq_2_l806_80686

theorem greatest_int_with_gcd_of_24_eq_2 (n : ℕ) (h1 : n < 200) (h2 : Int.gcd n 24 = 2) : n = 194 := 
sorry

end greatest_int_with_gcd_of_24_eq_2_l806_80686


namespace solution_set_of_inequality_l806_80630

theorem solution_set_of_inequality :
  ∀ x : ℝ, (x-50)*(60-x) > 0 ↔ 50 < x ∧ x < 60 :=
by
  sorry

end solution_set_of_inequality_l806_80630


namespace find_water_in_sport_formulation_l806_80659

noncomputable def standard_formulation : ℚ × ℚ × ℚ := (1, 12, 30)
noncomputable def sport_flavoring_to_corn : ℚ := 3 * (1 / 12)
noncomputable def sport_flavoring_to_water : ℚ := (1 / 2) * (1 / 30)
noncomputable def sport_formulation (f : ℚ) (c : ℚ) (w : ℚ) : Prop :=
  f / c = sport_flavoring_to_corn ∧ f / w = sport_flavoring_to_water

noncomputable def given_corn_syrup : ℚ := 8

theorem find_water_in_sport_formulation :
  ∀ (f c w : ℚ), sport_formulation f c w → c = given_corn_syrup → w = 120 :=
by
  sorry

end find_water_in_sport_formulation_l806_80659


namespace maximize_operation_l806_80617

-- Definitions from the conditions
def is_three_digit_integer (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

-- The proof statement
theorem maximize_operation : ∃ n, is_three_digit_integer n ∧ (∀ m, is_three_digit_integer m → 3 * (300 - m) ≤ 600) :=
by {
  -- Placeholder for the actual proof
  sorry
}

end maximize_operation_l806_80617


namespace probability_two_win_one_lose_l806_80654

noncomputable def p_A : ℚ := 1 / 5
noncomputable def p_B : ℚ := 3 / 8
noncomputable def p_C : ℚ := 2 / 7

noncomputable def P_two_win_one_lose : ℚ :=
  p_A * p_B * (1 - p_C) +
  p_A * p_C * (1 - p_B) +
  p_B * p_C * (1 - p_A)

theorem probability_two_win_one_lose :
  P_two_win_one_lose = 49 / 280 :=
by
  sorry

end probability_two_win_one_lose_l806_80654


namespace triangle_perimeter_l806_80635

theorem triangle_perimeter (r : ℝ) (A B C P Q R S T : ℝ)
  (triangle_isosceles : A = C)
  (circle_tangent : P = A ∧ Q = B ∧ R = B ∧ S = C ∧ T = C)
  (center_dist : P + Q = 2 ∧ Q + R = 2 ∧ R + S = 2 ∧ S + T = 2) :
  2 * (A + B + C) = 6 := by
  sorry

end triangle_perimeter_l806_80635


namespace arithmetic_sequence_general_term_find_n_given_sum_l806_80651

theorem arithmetic_sequence_general_term 
  (a : ℕ → ℕ)
  (S : ℕ → ℕ)
  (h1 : a 10 = 30)
  (h2 : a 15 = 40)
  : ∃ a1 d, (∀ n, a n = a1 + (n - 1) * d) ∧ a 10 = 30 ∧ a 15 = 40 :=
by {
  sorry
}

theorem find_n_given_sum
  (a : ℕ → ℕ)
  (S : ℕ → ℕ)
  (a1 d : ℕ)
  (h_gen : ∀ n, a n = a1 + (n - 1) * d)
  (h_sum : ∀ n, S n = n * a1 + (n * (n - 1) * d) / 2)
  (h_a1 : a1 = 12)
  (h_d : d = 2)
  (h_Sn : S 14 = 210)
  : ∃ n, S n = 210 ∧ n = 14 :=
by {
  sorry
}

end arithmetic_sequence_general_term_find_n_given_sum_l806_80651


namespace arithmetic_sequence_sum_neq_l806_80612

theorem arithmetic_sequence_sum_neq (a : ℕ → ℤ) (d : ℤ) (S : ℕ → ℤ)
    (h_arith : ∀ n, a (n + 1) = a n + d)
    (h_sum : ∀ n, S n = (n * (a 1 + a n)) / 2)
    (h_abs_eq : abs (a 3) = abs (a 9))
    (h_d_neg : d < 0) : S 5 ≠ S 6 := by
  sorry

end arithmetic_sequence_sum_neq_l806_80612


namespace determine_weights_l806_80606

-- Definitions
variable {W : Type} [AddCommGroup W] [OrderedAddCommMonoid W]
variable (w : Fin 20 → W) -- List of weights for 20 people
variable (s : W) -- Total sum of weights
variable (lower upper : W) -- Lower and upper weight limits

-- Conditions
def weight_constraints : Prop :=
  (∀ i, lower ≤ w i ∧ w i ≤ upper) ∧ (Finset.univ.sum w = s)

-- Problem statement
theorem determine_weights (w : Fin 20 → ℝ) :
  weight_constraints w 60 90 3040 →
  ∃ w : Fin 20 → ℝ, weight_constraints w 60 90 3040 := by
  sorry

end determine_weights_l806_80606


namespace cats_to_dogs_l806_80657

theorem cats_to_dogs (c d : ℕ) (h1 : c = 24) (h2 : 4 * d = 5 * c) : d = 30 :=
by
  sorry

end cats_to_dogs_l806_80657


namespace area_enclosed_by_line_and_curve_l806_80623

theorem area_enclosed_by_line_and_curve :
  ∃ area, ∀ (x : ℝ), x^2 = 4 * (x - 4/2) → 
    area = ∫ (t : ℝ) in Set.Icc (-1 : ℝ) 2, (1/4 * t + 1/2 - 1/4 * t^2) :=
sorry

end area_enclosed_by_line_and_curve_l806_80623


namespace six_circles_distance_relation_l806_80613

/--
Prove that for any pair of non-touching circles (among six circles where each touches four of the remaining five),
their radii \( r_1 \) and \( r_2 \) and the distance \( d \) between their centers satisfy 

\[ d^{2}=r_{1}^{2}+r_{2}^{2} \pm 6r_{1}r_{2} \]

("plus" if the circles do not lie inside one another, "minus" otherwise).
-/
theorem six_circles_distance_relation 
  (r1 r2 d : ℝ) 
  (h : ∀ i : Fin 6, i < 6 → ∃ c : ℝ, (c = r1 ∨ c = r2) ∧ ∀ j : Fin 6, j ≠ i → abs (c - j) ≠ d ) :
  d^2 = r1^2 + r2^2 + 6 * r1 * r2 ∨ d^2 = r1^2 + r2^2 - 6 * r1 * r2 := 
  sorry

end six_circles_distance_relation_l806_80613


namespace water_ratio_horse_pig_l806_80626

-- Definitions based on conditions
def num_pigs : ℕ := 8
def water_per_pig : ℕ := 3
def num_horses : ℕ := 10
def water_for_chickens : ℕ := 30
def total_water : ℕ := 114

-- Statement of the problem
theorem water_ratio_horse_pig : 
  (total_water - (num_pigs * water_per_pig) - water_for_chickens) / num_horses / water_per_pig = 2 := 
by sorry

end water_ratio_horse_pig_l806_80626


namespace cos_A_is_one_l806_80698

-- Definitions as per Lean's requirement
variable (A B C D : Type) [Inhabited A] [Inhabited B] [Inhabited C] [Inhabited D]

-- Declaring the conditions are given
variables (α : ℝ) (cos_A : ℝ)
variables (AB CD AD BC : ℝ)
def is_convex_quadrilateral (A B C D : Type) : Prop := 
  sorry -- This would be a formal definition of convex quadrilateral

-- The conditions are specified in Lean terms
variables (h1 : is_convex_quadrilateral A B C D)
variables (h2 : α = 0) -- α = 0 implies cos(α) = 1
variables (h3 : AB = 240)
variables (h4 : CD = 240)
variables (h5 : AD ≠ BC)
variables (h6 : AB + CD + AD + BC = 960)

-- The proof statement to indicate that cos(α) = 1 under the given conditions
theorem cos_A_is_one : cos_A = 1 :=
by
  sorry -- Proof not included as per the instruction

end cos_A_is_one_l806_80698


namespace find_a7_l806_80676

-- Definitions for geometric progression and conditions
def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
∀ n, ∃ q : ℝ, a n = a 1 * q^(n-1)

def condition1 (a : ℕ → ℝ) : Prop :=
a 2 * a 4 * a 5 = a 3 * a 6

def condition2 (a : ℕ → ℝ) : Prop :=
a 9 * a 10 = -8

-- Theorem stating the equivalence given the conditions
theorem find_a7 (a : ℕ → ℝ) (h : is_geometric_sequence a) (h1 : condition1 a) (h2 : condition2 a) : 
    a 7 = -2 :=
sorry

end find_a7_l806_80676


namespace difference_of_squares_multiple_of_20_l806_80653

theorem difference_of_squares_multiple_of_20 (a b : ℕ) (h1 : a > b) (h2 : a + b = 10) (hb : b = 10 - a) : 
  ∃ k : ℕ, (9 * a + 10)^2 - (100 - 9 * a)^2 = 20 * k :=
by
  sorry

end difference_of_squares_multiple_of_20_l806_80653


namespace pick_two_black_cards_l806_80601

-- Definition: conditions
def total_cards : ℕ := 52
def cards_per_suit : ℕ := 13
def black_suits : ℕ := 2
def red_suits : ℕ := 2
def total_black_cards : ℕ := black_suits * cards_per_suit

-- Theorem: number of ways to pick two different black cards
theorem pick_two_black_cards :
  (total_black_cards * (total_black_cards - 1)) = 650 :=
by
  -- proof here
  sorry

end pick_two_black_cards_l806_80601


namespace eighth_grade_students_l806_80677

def avg_books (total_books : ℕ) (num_students : ℕ) : ℚ :=
  total_books / num_students

theorem eighth_grade_students (x : ℕ) (y : ℕ)
  (h1 : x + y = 1800)
  (h2 : y = x - 150)
  (h3 : avg_books x 1800 = 1.5 * avg_books (x - 150) 1800) :
  y = 450 :=
by {
  sorry
}

end eighth_grade_students_l806_80677


namespace temperature_difference_l806_80696

theorem temperature_difference (T_high T_low : ℝ) (h_high : T_high = 9) (h_low : T_low = -1) : 
  T_high - T_low = 10 :=
by
  rw [h_high, h_low]
  norm_num

end temperature_difference_l806_80696


namespace parabola_vertex_y_l806_80619

theorem parabola_vertex_y (x : ℝ) : (∃ (h k : ℝ), (4 * (x - h)^2 + k = 4 * x^2 + 16 * x + 11) ∧ k = -5) := 
  sorry

end parabola_vertex_y_l806_80619


namespace find_AB_l806_80639

theorem find_AB
  (r R : ℝ)
  (h : r < R) :
  ∃ AB : ℝ, AB = (4 * r * (Real.sqrt (R * r))) / (R + r) :=
by
  sorry

end find_AB_l806_80639


namespace fencing_required_l806_80648

theorem fencing_required (L W : ℝ) (hL : L = 20) (hA : 20 * W = 60) : 2 * W + L = 26 :=
by
  sorry

end fencing_required_l806_80648


namespace youngest_brother_age_difference_l806_80688

def Rick_age : ℕ := 15
def Oldest_brother_age : ℕ := 2 * Rick_age
def Middle_brother_age : ℕ := Oldest_brother_age / 3
def Smallest_brother_age : ℕ := Middle_brother_age / 2
def Youngest_brother_age : ℕ := 3

theorem youngest_brother_age_difference :
  Smallest_brother_age - Youngest_brother_age = 2 :=
by
  -- sorry to skip the proof
  sorry

end youngest_brother_age_difference_l806_80688


namespace largest_hole_leakage_rate_l806_80655

theorem largest_hole_leakage_rate (L : ℝ) (h1 : 600 = (L + L / 2 + L / 6) * 120) : 
  L = 3 :=
sorry

end largest_hole_leakage_rate_l806_80655


namespace students_without_favorite_subject_l806_80662

theorem students_without_favorite_subject
  (total_students : ℕ)
  (students_like_math : ℕ)
  (students_like_english : ℕ)
  (remaining_students : ℕ)
  (students_like_science : ℕ)
  (students_without_favorite : ℕ)
  (h1 : total_students = 30)
  (h2 : students_like_math = total_students * (1 / 5))
  (h3 : students_like_english = total_students * (1 / 3))
  (h4 : remaining_students = total_students - (students_like_math + students_like_english))
  (h5 : students_like_science = remaining_students * (1 / 7))
  (h6 : students_without_favorite = remaining_students - students_like_science) :
  students_without_favorite = 12 := by
  sorry

end students_without_favorite_subject_l806_80662


namespace day_crew_fraction_l806_80628

-- Definitions of number of boxes per worker for day crew, and workers for day crew
variables (D : ℕ) (W : ℕ)

-- Definitions of night crew loading rate and worker ratio based on given conditions
def night_boxes_per_worker := (3 / 4 : ℚ) * D
def night_workers := (2 / 3 : ℚ) * W

-- Definition of total boxes loaded by each crew
def day_crew_total := D * W
def night_crew_total := night_boxes_per_worker D * night_workers W

-- The proof problem shows fraction loaded by day crew equals 2/3
theorem day_crew_fraction : (day_crew_total D W) / (day_crew_total D W + night_crew_total D W) = (2 / 3 : ℚ) := by
  sorry

end day_crew_fraction_l806_80628


namespace trapezoid_area_l806_80694

theorem trapezoid_area (l : ℝ) (r : ℝ) (a b : ℝ) (h : ℝ) (A : ℝ) :
  l = 9 →
  r = 4 →
  a + b = l + l →
  h = 2 * r →
  (a + b) / 2 * h = A →
  A = 72 :=
by
  intro h1 h2 h3 h4 h5
  sorry

end trapezoid_area_l806_80694


namespace Ben_shirts_is_15_l806_80607

variable (Alex_shirts Joe_shirts Ben_shirts : Nat)

def Alex_has_4 : Alex_shirts = 4 := by sorry

def Joe_has_more_than_Alex : Joe_shirts = Alex_shirts + 3 := by sorry

def Ben_has_more_than_Joe : Ben_shirts = Joe_shirts + 8 := by sorry

theorem Ben_shirts_is_15 (h1 : Alex_shirts = 4) (h2 : Joe_shirts = Alex_shirts + 3) (h3 : Ben_shirts = Joe_shirts + 8) : Ben_shirts = 15 := by
  sorry

end Ben_shirts_is_15_l806_80607


namespace find_angle_l806_80691

theorem find_angle (x : ℝ) (h : 90 - x = 2 * x + 15) : x = 25 :=
by
  sorry

end find_angle_l806_80691


namespace derivative_at_zero_l806_80603

noncomputable def f (x : ℝ) : ℝ := x * (x - 1) * (x - 2) * (x - 3) * (x - 4) * (x - 5) * (x - 6)

theorem derivative_at_zero : deriv f 0 = 720 :=
by
  sorry

end derivative_at_zero_l806_80603


namespace max_intersections_l806_80681

/-- Given two different circles and three different straight lines, the maximum number of
points of intersection on a plane is 17. -/
theorem max_intersections (c1 c2 : Circle) (l1 l2 l3 : Line) (h_distinct_cir : c1 ≠ c2) (h_distinct_lines : ∀ (l1 l2 : Line), l1 ≠ l2) :
  ∃ (n : ℕ), n = 17 :=
by
  sorry

end max_intersections_l806_80681


namespace cats_left_l806_80689

theorem cats_left (siamese_cats : ℕ) (house_cats : ℕ) (cats_sold : ℕ) 
  (h1 : siamese_cats = 13) (h2 : house_cats = 5) (h3 : cats_sold = 10) : 
  siamese_cats + house_cats - cats_sold = 8 :=
by
  sorry

end cats_left_l806_80689


namespace polynomial_perfect_square_l806_80625

theorem polynomial_perfect_square (m : ℤ) : (∃ a : ℤ, a^2 = 25 ∧ x^2 + m*x + 25 = (x + a)^2) ↔ (m = 10 ∨ m = -10) :=
by sorry

end polynomial_perfect_square_l806_80625


namespace sam_sandwich_shop_cost_l806_80608

theorem sam_sandwich_shop_cost :
  let sandwich_cost := 4
  let soda_cost := 3
  let fries_cost := 2
  let num_sandwiches := 3
  let num_sodas := 7
  let num_fries := 5
  let total_cost := num_sandwiches * sandwich_cost + num_sodas * soda_cost + num_fries * fries_cost
  total_cost = 43 :=
by
  sorry

end sam_sandwich_shop_cost_l806_80608


namespace bruce_and_anne_clean_house_l806_80685

theorem bruce_and_anne_clean_house 
  (A : ℚ) (B : ℚ) (H1 : A = 1/12) 
  (H2 : 3 * (B + 2 * A) = 1) :
  1 / (B + A) = 4 := 
sorry

end bruce_and_anne_clean_house_l806_80685


namespace least_number_of_trees_l806_80675

theorem least_number_of_trees :
  ∃ n : ℕ, (n % 5 = 0) ∧ (n % 6 = 0) ∧ (n % 7 = 0) ∧ n = 210 :=
by
  sorry

end least_number_of_trees_l806_80675


namespace unique_integer_solution_l806_80678

-- Define the problem statement and the conditions: integers x, y such that x^4 - 2y^2 = 1
theorem unique_integer_solution (x y: ℤ) (h: x^4 - 2 * y^2 = 1) : (x = 1 ∧ y = 0) :=
sorry

end unique_integer_solution_l806_80678


namespace find_third_number_l806_80674

theorem find_third_number (x y : ℕ) (h1 : x = 3)
  (h2 : (x + 1) / (x + 5) = (x + 5) / (x + y)) : y = 13 :=
by
  sorry

end find_third_number_l806_80674


namespace larger_integer_l806_80687

theorem larger_integer (x y : ℕ) (h_diff : y - x = 8) (h_prod : x * y = 272) : y = 20 :=
by
  sorry

end larger_integer_l806_80687


namespace ratio_S7_S3_l806_80629

variable {a_n : ℕ → ℕ} -- Arithmetic sequence {a_n}
variable (S_n : ℕ → ℕ) -- Sum of the first n terms of the arithmetic sequence

-- Conditions
def ratio_a2_a4 (a_2 a_4 : ℕ) : Prop := a_2 = 7 * (a_4 / 6)
def sum_formula (n a_1 d : ℕ) : ℕ := n * (2 * a_1 + (n - 1) * d) / 2

-- Proof goal
theorem ratio_S7_S3 (a_1 d : ℕ) (h : ratio_a2_a4 (a_1 + d) (a_1 + 3 * d)): 
  (S_n 7 = sum_formula 7 a_1 d) ∧ (S_n 3 = sum_formula 3 a_1 d) →
  (S_n 7 / S_n 3 = 2) :=
by
  sorry

end ratio_S7_S3_l806_80629


namespace inverse_ratio_l806_80684

theorem inverse_ratio (a b c d : ℝ) :
  (∀ x, x ≠ -6 → (3 * x - 2) / (x + 6) = (a * x + b) / (c * x + d)) →
  a/c = -6 :=
by
  sorry

end inverse_ratio_l806_80684


namespace problem_l806_80661

noncomputable def p (k : ℝ) (x : ℝ) := k * (x - 5) * (x - 2)
noncomputable def q (x : ℝ) := (x - 5) * (x + 3)

theorem problem {p q : ℝ → ℝ} (k : ℝ) :
  (∀ x, q x = (x - 5) * (x + 3)) →
  (∀ x, p x = k * (x - 5) * (x - 2)) →
  (∀ x ≠ 5, (p x) / (q x) = (3 * (x - 2)) / (x + 3)) →
  p 3 / q 3 = 1 / 2 :=
by
  sorry

end problem_l806_80661


namespace circleII_area_l806_80645

noncomputable def area_of_circle (r : ℝ) : ℝ := Real.pi * r^2

theorem circleII_area (r₁ : ℝ) (h₁ : area_of_circle r₁ = 9) (h₂ : r₂ = 3 * 2 * r₁) : 
  area_of_circle r₂ = 324 :=
by
  sorry

end circleII_area_l806_80645


namespace time_to_fill_tank_with_two_pipes_simultaneously_l806_80615

def PipeA : ℝ := 30
def PipeB : ℝ := 45

theorem time_to_fill_tank_with_two_pipes_simultaneously :
  let A := 1 / PipeA
  let B := 1 / PipeB
  let combined_rate := A + B
  let time_to_fill_tank := 1 / combined_rate
  time_to_fill_tank = 18 := 
by
  sorry

end time_to_fill_tank_with_two_pipes_simultaneously_l806_80615


namespace intersection_in_fourth_quadrant_l806_80692

theorem intersection_in_fourth_quadrant (m : ℝ) :
  let x := (3 * m + 2) / 4
  let y := (-m - 2) / 8
  (x > 0) ∧ (y < 0) ↔ (m > -2 / 3) :=
by
  sorry

end intersection_in_fourth_quadrant_l806_80692


namespace loss_percentage_is_25_l806_80652

variables (C S : ℝ)
variables (h : 30 * C = 40 * S)

theorem loss_percentage_is_25 (h : 30 * C = 40 * S) : ((C - S) / C) * 100 = 25 :=
by
  -- proof skipped
  sorry

end loss_percentage_is_25_l806_80652


namespace negation_of_all_men_are_tall_l806_80695

variable {α : Type}
variable (man : α → Prop) (tall : α → Prop)

theorem negation_of_all_men_are_tall :
  (¬ ∀ x, man x → tall x) ↔ ∃ x, man x ∧ ¬ tall x :=
sorry

end negation_of_all_men_are_tall_l806_80695


namespace circle_equation_with_focus_center_and_tangent_directrix_l806_80621

theorem circle_equation_with_focus_center_and_tangent_directrix :
  ∃ (x y : ℝ), (∃ k : ℝ, y^2 = -8 * x ∧ k = 2 ∧ (x = -2 ∧ y = 0) ∧ (x + 2)^2 + y^2 = 16) :=
by
  sorry

end circle_equation_with_focus_center_and_tangent_directrix_l806_80621


namespace inheritance_amount_l806_80643

theorem inheritance_amount (x : ℝ) (total_taxes_paid : ℝ) (federal_tax_rate : ℝ) (state_tax_rate : ℝ) 
  (federal_tax_paid : ℝ) (state_tax_base : ℝ) (state_tax_paid : ℝ) 
  (federal_tax_eq : federal_tax_paid = federal_tax_rate * x)
  (state_tax_base_eq : state_tax_base = x - federal_tax_paid)
  (state_tax_eq : state_tax_paid = state_tax_rate * state_tax_base)
  (total_taxes_eq : total_taxes_paid = federal_tax_paid + state_tax_paid) 
  (total_taxes_val : total_taxes_paid = 18000)
  (federal_tax_rate_val : federal_tax_rate = 0.25)
  (state_tax_rate_val : state_tax_rate = 0.15)
  : x = 50000 :=
sorry

end inheritance_amount_l806_80643


namespace total_distance_covered_l806_80697

theorem total_distance_covered (h : ℝ) : (h > 0) → 
  ∑' n : ℕ, (h * (0.8 : ℝ) ^ n + h * (0.8 : ℝ) ^ (n + 1)) = 5 * h :=
  by
  sorry

end total_distance_covered_l806_80697


namespace combined_work_time_l806_80609

theorem combined_work_time (man_rate : ℚ := 1/5) (wife_rate : ℚ := 1/7) (son_rate : ℚ := 1/15) :
  (man_rate + wife_rate + son_rate)⁻¹ = 105 / 43 :=
by
  sorry

end combined_work_time_l806_80609


namespace solution_set_of_inequality_l806_80614

theorem solution_set_of_inequality :
  {x : ℝ | (x - 3) / x ≥ 0} = {x : ℝ | x < 0 ∨ x ≥ 3} :=
sorry

end solution_set_of_inequality_l806_80614


namespace number_of_teams_l806_80699

theorem number_of_teams (n : ℕ) (h : (n * (n - 1)) / 2 * 10 = 1050) : n = 15 :=
by 
  sorry

end number_of_teams_l806_80699


namespace exp_gt_one_iff_a_gt_one_l806_80671

theorem exp_gt_one_iff_a_gt_one (a : ℝ) : 
  (∀ x : ℝ, 0 < x → a^x > 1) ↔ a > 1 :=
by
  sorry

end exp_gt_one_iff_a_gt_one_l806_80671


namespace find_k_l806_80658

theorem find_k (k : ℝ) (h : (3:ℝ)^4 + k * (3:ℝ)^2 - 26 = 0) : k = -55 / 9 := 
by sorry

end find_k_l806_80658


namespace lucy_l806_80633

-- Define rounding function to nearest ten
def round_to_nearest_ten (x : Int) : Int :=
  if x % 10 < 5 then x - x % 10 else x + (10 - x % 10)

-- Define the problem with given conditions
def lucy_problem : Prop :=
  let sum := 68 + 57
  round_to_nearest_ten sum = 130

-- Statement of proof problem
theorem lucy's_correct_rounded_sum : lucy_problem := by
  sorry

end lucy_l806_80633


namespace enclosed_area_is_43pi_l806_80600

noncomputable def enclosed_area (x y : ℝ) : Prop :=
  (x^2 - 6*x + y^2 + 10*y = 9)

theorem enclosed_area_is_43pi :
  (∃ x y : ℝ, enclosed_area x y) → 
  ∃ A : ℝ, A = 43 * Real.pi :=
by
  sorry

end enclosed_area_is_43pi_l806_80600


namespace centers_of_parallelograms_l806_80680

def is_skew_lines (l1 l2 l3 l4 : Line) : Prop :=
  -- A function that checks if 4 lines are pairwise skew and no three of them are parallel to the same plane.
  sorry

def count_centers_of_parallelograms (l1 l2 l3 l4 : Line) : ℕ :=
  -- A function that counts the number of lines through which the centers of parallelograms formed by the intersections of the lines pass.
  sorry

theorem centers_of_parallelograms (l1 l2 l3 l4 : Line) (h_skew: is_skew_lines l1 l2 l3 l4) : count_centers_of_parallelograms l1 l2 l3 l4 = 3 :=
  sorry

end centers_of_parallelograms_l806_80680


namespace number_of_moles_of_HCl_l806_80672

-- Defining the chemical equation relationship
def reaction_relation (HCl NaHCO3 NaCl H2O CO2 : ℕ) : Prop :=
  H2O = HCl ∧ H2O = NaHCO3

-- Conditions
def conditions (HCl NaHCO3 H2O : ℕ) : Prop :=
  NaHCO3 = 3 ∧ H2O = 3

-- Theorem statement proving the number of moles of HCl given the conditions
theorem number_of_moles_of_HCl (HCl NaHCO3 NaCl H2O CO2 : ℕ) 
  (h1 : reaction_relation HCl NaHCO3 NaCl H2O CO2) 
  (h2 : conditions HCl NaHCO3 H2O) :
  HCl = 3 :=
sorry

end number_of_moles_of_HCl_l806_80672


namespace find_distance_CD_l806_80690

noncomputable def distance_CD : ℝ :=
  let C : ℝ × ℝ := (0, 0)
  let D : ℝ × ℝ := (3, 6)
  Real.sqrt ((D.1 - C.1)^2 + (D.2 - C.2)^2)

theorem find_distance_CD :
  ∀ (C D : ℝ × ℝ), 
  (C = (0, 0) ∧ D = (3, 6)) ∧ 
  (∃ x y : ℝ, (y^2 = 12 * x ∧ (x^2 + y^2 - 4 * x - 6 * y = 0))) → 
  distance_CD = 3 * Real.sqrt 5 :=
by
  sorry

end find_distance_CD_l806_80690


namespace gold_problem_proof_l806_80644

noncomputable def solve_gold_problem : Prop :=
  ∃ (a : ℕ → ℝ), 
  (a 1) + (a 2) + (a 3) = 4 ∧ 
  (a 8) + (a 9) + (a 10) = 3 ∧
  (a 5) + (a 6) = 7 / 3

theorem gold_problem_proof : solve_gold_problem := 
  sorry

end gold_problem_proof_l806_80644


namespace correct_removal_of_parentheses_C_incorrect_removal_of_parentheses_A_incorrect_removal_of_parentheses_B_incorrect_removal_of_parentheses_D_l806_80632

theorem correct_removal_of_parentheses_C (a : ℝ) :
    -(2 * a - 1) = -2 * a + 1 :=
by sorry

theorem incorrect_removal_of_parentheses_A (a : ℝ) :
    -(7 * a - 5) ≠ -7 * a - 5 :=
by sorry

theorem incorrect_removal_of_parentheses_B (a : ℝ) :
    -(-1 / 2 * a + 2) ≠ -1 / 2 * a - 2 :=
by sorry

theorem incorrect_removal_of_parentheses_D (a : ℝ) :
    -(-3 * a + 2) ≠ 3 * a + 2 :=
by sorry

end correct_removal_of_parentheses_C_incorrect_removal_of_parentheses_A_incorrect_removal_of_parentheses_B_incorrect_removal_of_parentheses_D_l806_80632


namespace symmetric_line_eq_l806_80670

theorem symmetric_line_eq (x y : ℝ) :
  (∀ x y, 2 * x - y + 1 = 0 → y = -x) → (∀ x y, x - 2 * y + 1 = 0) :=
by sorry

end symmetric_line_eq_l806_80670


namespace prime_dvd_square_l806_80605

theorem prime_dvd_square (p n : ℕ) (hp : Nat.Prime p) (h : p ∣ n^2) : p ∣ n :=
  sorry

end prime_dvd_square_l806_80605


namespace max_value_of_a_l806_80667
noncomputable def f (a x : ℝ) : ℝ :=
  if x < a then -a * x + 1 else (x - 2)^2

theorem max_value_of_a (a : ℝ) :
  (∃ (x : ℝ), ∀ (y : ℝ), f a x ≤ f a y) → a ≤ 1 := 
sorry

end max_value_of_a_l806_80667


namespace triangle_problems_l806_80650

open Real

variables {A B C a b c : ℝ}
variables {m n : ℝ × ℝ}

def triangle_sides_and_angles (a b c : ℝ) (A B C : ℝ) : Prop := a > 0 ∧ b > 0 ∧ c > 0 ∧ A + B + C = π

def perpendicular (m n : ℝ × ℝ) : Prop := m.1 * n.1 + m.2 * n.2 = 0

noncomputable def area_of_triangle (a b c : ℝ) (A : ℝ) : ℝ :=
  1 / 2 * b * c * sin A

theorem triangle_problems
  (h1 : triangle_sides_and_angles a b c A B C)
  (h2 : m = (1, 1))
  (h3 : n = (sqrt 3 / 2 - sin B * sin C, cos B * cos C))
  (h4 : perpendicular m n)
  (h5 : a = 1)
  (h6 : b = sqrt 3 * c) :
  A = π / 6 ∧ area_of_triangle a b c A = sqrt 3 / 4 :=
by
  sorry

end triangle_problems_l806_80650


namespace find_sum_of_perimeters_l806_80611

variables (x y : ℝ)
noncomputable def sum_of_perimeters := 4 * x + 4 * y

theorem find_sum_of_perimeters (h1 : x^2 + y^2 = 65) (h2 : x^2 - y^2 = 33) :
  sum_of_perimeters x y = 44 :=
sorry

end find_sum_of_perimeters_l806_80611


namespace find_reals_abc_d_l806_80679

theorem find_reals_abc_d (a b c d : ℝ)
  (h1 : a * b * c + a * b + b * c + c * a + a + b + c = 1)
  (h2 : b * c * d + b * c + c * d + d * b + b + c + d = 9)
  (h3 : c * d * a + c * d + d * a + a * c + c + d + a = 9)
  (h4 : d * a * b + d * a + a * b + b * d + d + a + b = 9) :
  a = b ∧ b = c ∧ c = (2 : ℝ)^(1/3) - 1 ∧ d = 5 * (2 : ℝ)^(1/3) - 1 :=
sorry

end find_reals_abc_d_l806_80679


namespace find_m_value_l806_80664

variable (m : ℝ)
noncomputable def a : ℝ × ℝ := (2 * Real.sqrt 2, 2)
noncomputable def b : ℝ × ℝ := (0, 2)
noncomputable def c (m : ℝ) : ℝ × ℝ := (m, Real.sqrt 2)

theorem find_m_value (h : (a.1 + 2 * b.1) * (m) + (a.2 + 2 * b.2) * (Real.sqrt 2) = 0) : m = -3 :=
by
  sorry

end find_m_value_l806_80664


namespace warehouse_capacity_l806_80642

theorem warehouse_capacity (total_bins num_20_ton_bins cap_20_ton_bin cap_15_ton_bin : Nat) 
  (h1 : total_bins = 30) 
  (h2 : num_20_ton_bins = 12) 
  (h3 : cap_20_ton_bin = 20) 
  (h4 : cap_15_ton_bin = 15) : 
  total_bins * cap_20_ton_bin + (total_bins - num_20_ton_bins) * cap_15_ton_bin = 510 := 
by
  sorry

end warehouse_capacity_l806_80642


namespace whale_sixth_hour_consumption_l806_80637

-- Definitions based on the given conditions
def consumption (x : ℕ) (hour : ℕ) : ℕ := x + 3 * (hour - 1)

def total_consumption (x : ℕ) : ℕ := 
  (consumption x 1) + (consumption x 2) + (consumption x 3) +
  (consumption x 4) + (consumption x 5) + (consumption x 6) + 
  (consumption x 7) + (consumption x 8) + (consumption x 9)

-- Given problem translated to Lean
theorem whale_sixth_hour_consumption (x : ℕ) (h1 : total_consumption x = 270) :
  consumption x 6 = 33 :=
sorry

end whale_sixth_hour_consumption_l806_80637


namespace problem_statement_l806_80622

theorem problem_statement (a b : ℕ) (h₁ : a ≠ 0) (h₂ : b ≠ 0) (h₃ : ∀ n : ℕ, n ≥ 1 → 2^n * b + 1 ∣ a^(2^n) - 1) : a = 1 := by
  sorry

end problem_statement_l806_80622


namespace geometric_series_seventh_term_l806_80616

theorem geometric_series_seventh_term (a₁ a₁₀ : ℝ) (n : ℝ) (r : ℝ) :
  a₁ = 4 →
  a₁₀ = 93312 →
  n = 10 →
  a₁₀ = a₁ * r^(n-1) →
  (∃ (r : ℝ), r = 6) →
  4 * 6^(7-1) = 186624 := by
  intros a1_eq a10_eq n_eq an_eq exists_r
  sorry

end geometric_series_seventh_term_l806_80616


namespace remaining_cookies_l806_80604

variable (total_initial_cookies : ℕ)
variable (cookies_taken_day1 : ℕ := 3)
variable (cookies_taken_day2 : ℕ := 3)
variable (cookies_eaten_day2 : ℕ := 1)
variable (cookies_put_back_day2 : ℕ := 2)
variable (cookies_taken_by_junior : ℕ := 7)

theorem remaining_cookies (total_initial_cookies cookies_taken_day1 cookies_taken_day2
                          cookies_eaten_day2 cookies_put_back_day2 cookies_taken_by_junior : ℕ) :
  (total_initial_cookies = 2 * (cookies_taken_day1 + cookies_eaten_day2 + cookies_taken_by_junior))
  → (total_initial_cookies - (cookies_taken_day1 + cookies_eaten_day2 + cookies_taken_by_junior) = 11) :=
by
  sorry

end remaining_cookies_l806_80604


namespace largest_spherical_ball_radius_in_torus_l806_80641

theorem largest_spherical_ball_radius_in_torus 
    (inner_radius outer_radius : ℝ) 
    (circle_center : ℝ × ℝ × ℝ) 
    (circle_radius : ℝ) 
    (r : ℝ)
    (h0 : inner_radius = 2)
    (h1 : outer_radius = 4)
    (h2 : circle_center = (3, 0, 1))
    (h3 : circle_radius = 1)
    (h4 : 3^2 + (r - 1)^2 = (r + 1)^2) :
    r = 9 / 4 :=
by
  sorry

end largest_spherical_ball_radius_in_torus_l806_80641


namespace negate_prop_l806_80668

theorem negate_prop (p : ∀ x : ℝ, 0 ≤ x ∧ x ≤ 2 * Real.pi → |Real.sin x| ≤ 1) :
  ¬ (∀ x : ℝ, 0 ≤ x ∧ x ≤ 2 * Real.pi → |Real.sin x| ≤ 1) ↔ ∃ x_0 : ℝ, 0 ≤ x_0 ∧ x_0 ≤ 2 * Real.pi ∧ |Real.sin x_0| > 1 :=
by sorry

end negate_prop_l806_80668


namespace expenditure_recording_l806_80646

theorem expenditure_recording (income expense : ℤ) (h1 : income = 100) (h2 : expense = -100)
  (h3 : income = -expense) : expense = -100 :=
by
  sorry

end expenditure_recording_l806_80646


namespace fraction_in_jug_x_after_pouring_water_l806_80620

-- Define capacities and initial fractions
def initial_fraction_x := 1 / 4
def initial_fraction_y := 2 / 3
def fill_needed_y := 1 - initial_fraction_y -- 1/3

-- Define capacity of original jugs
variable (C : ℚ) -- We can assume capacities are rational for simplicity

-- Define initial water amounts in jugs x and y
def initial_water_x := initial_fraction_x * C
def initial_water_y := initial_fraction_y * C

-- Define the water needed to fill jug y
def additional_water_needed_y := fill_needed_y * C

-- Define the final fraction of water in jug x
def final_fraction_x := initial_fraction_x / 2 -- since half of the initial water is poured out

theorem fraction_in_jug_x_after_pouring_water :
  final_fraction_x = 1 / 8 := by
  sorry

end fraction_in_jug_x_after_pouring_water_l806_80620


namespace number_of_games_l806_80660

-- Definitions based on the conditions
def initial_money : ℕ := 104
def cost_of_blades : ℕ := 41
def cost_per_game : ℕ := 9

-- Lean 4 statement asserting the number of games Will can buy is 7
theorem number_of_games : (initial_money - cost_of_blades) / cost_per_game = 7 := by
  sorry

end number_of_games_l806_80660


namespace highest_score_runs_l806_80624

theorem highest_score_runs 
  (avg : ℕ) (innings : ℕ) (total_runs : ℕ) (H L : ℕ)
  (diff_HL : ℕ) (excl_avg : ℕ) (excl_innings : ℕ) (excl_total_runs : ℕ) :
  avg = 60 → innings = 46 → total_runs = avg * innings →
  diff_HL = 180 → excl_avg = 58 → excl_innings = 44 → 
  excl_total_runs = excl_avg * excl_innings →
  H - L = diff_HL →
  total_runs = excl_total_runs + H + L →
  H = 194 :=
by
  intros h_avg h_innings h_total_runs h_diff_HL h_excl_avg h_excl_innings h_excl_total_runs h_H_minus_L h_total_eq
  sorry

end highest_score_runs_l806_80624


namespace find_number_l806_80610

theorem find_number 
  (x y n : ℝ)
  (h1 : n * x = 0.04 * y)
  (h2 : (y - x) / (y + x) = 0.948051948051948) :
  n = 37.5 :=
sorry  -- proof omitted

end find_number_l806_80610


namespace find_n_l806_80663

theorem find_n (n : ℕ) (h : 2^6 * 3^3 * n = Nat.factorial 10) : n = 2100 :=
by
sorry

end find_n_l806_80663


namespace decimal_2_09_is_209_percent_l806_80669

-- Definition of the conversion from decimal to percentage
def decimal_to_percentage (x : ℝ) := x * 100

-- Theorem statement
theorem decimal_2_09_is_209_percent : decimal_to_percentage 2.09 = 209 :=
by sorry

end decimal_2_09_is_209_percent_l806_80669


namespace find_triplets_of_real_numbers_l806_80649

theorem find_triplets_of_real_numbers (x y z : ℝ) :
  (x^2 + y^2 + 25 * z^2 = 6 * x * z + 8 * y * z) ∧ 
  (3 * x^2 + 2 * y^2 + z^2 = 240) → 
  (x = 6 ∧ y = 8 ∧ z = 2) ∨ (x = -6 ∧ y = -8 ∧ z = -2) := 
sorry

end find_triplets_of_real_numbers_l806_80649


namespace t_f_3_equals_sqrt_44_l806_80636

noncomputable def t (x : ℝ) : ℝ := Real.sqrt (4 * x + 4)
noncomputable def f (x : ℝ) : ℝ := 6 + t x

theorem t_f_3_equals_sqrt_44 : t (f 3) = Real.sqrt 44 := by
  sorry

end t_f_3_equals_sqrt_44_l806_80636


namespace problem_1_l806_80673

variable (x : ℝ) (a : ℝ)

theorem problem_1 (h1 : x - 1/x = 3) (h2 : a = x^2 + 1/x^2) : a = 11 := sorry

end problem_1_l806_80673


namespace find_divisor_l806_80647

theorem find_divisor (d : ℕ) : (55 / d) + 10 = 21 → d = 5 :=
by 
  sorry

end find_divisor_l806_80647


namespace gcd_xyz_square_of_diff_l806_80682

theorem gcd_xyz_square_of_diff {x y z : ℕ} 
    (h : (1 : ℚ) / x - (1 : ℚ) / y = (1 : ℚ) / z) : 
    ∃ n : ℕ, Nat.gcd x (Nat.gcd y z) * (y - x) = n ^ 2 :=
by
  sorry

end gcd_xyz_square_of_diff_l806_80682


namespace sum_geometric_series_l806_80602

theorem sum_geometric_series (x : ℂ) (h₀ : x ≠ 1) (h₁ : x^10 - 3*x + 2 = 0) : 
  x^9 + x^8 + x^7 + x^6 + x^5 + x^4 + x^3 + x^2 + x + 1 = 3 := 
by 
  sorry

end sum_geometric_series_l806_80602


namespace remaining_amount_needed_l806_80683

def goal := 150
def earnings_from_3_families := 3 * 10
def earnings_from_15_families := 15 * 5
def total_earnings := earnings_from_3_families + earnings_from_15_families
def remaining_amount := goal - total_earnings

theorem remaining_amount_needed : remaining_amount = 45 := by
  sorry

end remaining_amount_needed_l806_80683


namespace hamburgers_served_l806_80634

-- Definitions for the conditions
def hamburgers_made : ℕ := 9
def hamburgers_left_over : ℕ := 6

-- The main statement to prove
theorem hamburgers_served : hamburgers_made - hamburgers_left_over = 3 := by
  sorry

end hamburgers_served_l806_80634


namespace given_eqn_simplification_l806_80640

theorem given_eqn_simplification (x : ℝ) (h : 6 * x^2 - 4 * x - 3 = 0) : 
  (x - 1)^2 + x * (x + 2 / 3) = 2 :=
by
  sorry

end given_eqn_simplification_l806_80640


namespace fewest_coach_handshakes_l806_80627

theorem fewest_coach_handshakes (n m1 m2 : ℕ) 
  (handshakes_total : (n * (n - 1)) / 2 + m1 + m2 = 465) 
  (m1_m2_eq_n : m1 + m2 = n) : 
  n * (n - 1) / 2 = 465 → m1 + m2 = 0 :=
by 
  sorry

end fewest_coach_handshakes_l806_80627


namespace units_digit_sum_l806_80693

theorem units_digit_sum (n1 n2 : ℕ) (h1 : n1 % 10 = 1) (h2 : n2 % 10 = 3) : ((n1^3 + n2^3) % 10) = 8 := 
by
  sorry

end units_digit_sum_l806_80693


namespace nancy_total_spending_l806_80665

theorem nancy_total_spending :
  let crystal_bead_price := 9
  let metal_bead_price := 10
  let nancy_crystal_beads := 1
  let nancy_metal_beads := 2
  nancy_crystal_beads * crystal_bead_price + nancy_metal_beads * metal_bead_price = 29 := by
sorry

end nancy_total_spending_l806_80665
