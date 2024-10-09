import Mathlib

namespace smallest_lcm_not_multiple_of_25_l1169_116928

theorem smallest_lcm_not_multiple_of_25 (n : ℕ) (h1 : n % 36 = 0) (h2 : n % 45 = 0) (h3 : n % 25 ≠ 0) : n = 180 := 
by 
  sorry

end smallest_lcm_not_multiple_of_25_l1169_116928


namespace find_fraction_value_l1169_116997

theorem find_fraction_value (m n : ℝ) (h : 1/m - 1/n = 6) : (m * n) / (m - n) = -1/6 :=
sorry

end find_fraction_value_l1169_116997


namespace john_saves_1200_yearly_l1169_116967

noncomputable def former_rent_per_month (sq_ft_cost : ℝ) (sq_ft : ℝ) : ℝ :=
  sq_ft_cost * sq_ft

noncomputable def new_rent_per_month (total_cost : ℝ) (roommates : ℝ) : ℝ :=
  total_cost / roommates

noncomputable def monthly_savings (former_rent : ℝ) (new_rent : ℝ) : ℝ :=
  former_rent - new_rent

noncomputable def annual_savings (monthly_savings : ℝ) : ℝ :=
  monthly_savings * 12

theorem john_saves_1200_yearly :
  let former_rent := former_rent_per_month 2 750
  let new_rent := new_rent_per_month 2800 2
  let monthly_savings := monthly_savings former_rent new_rent
  annual_savings monthly_savings = 1200 := 
by 
  sorry

end john_saves_1200_yearly_l1169_116967


namespace increase_in_average_weight_l1169_116951

variable (A : ℝ)

theorem increase_in_average_weight (h1 : ∀ (A : ℝ), 4 * A - 65 + 71 = 4 * (A + 1.5)) :
  (71 - 65) / 4 = 1.5 :=
by
  sorry

end increase_in_average_weight_l1169_116951


namespace cody_marbles_l1169_116952

theorem cody_marbles (M : ℕ) (h1 : M / 3 + 5 + 7 = M) : M = 18 :=
by
  have h2 : 3 * M / 3 + 3 * 5 + 3 * 7 = 3 * M := by sorry
  have h3 : 3 * M / 3 = M := by sorry
  have h4 : 3 * 7 = 21 := by sorry
  have h5 : M + 15 + 21 = 3 * M := by sorry
  have h6 : M = 18 := by sorry
  exact h6

end cody_marbles_l1169_116952


namespace number_of_small_branches_l1169_116902

-- Define the number of small branches grown by each branch as a variable
variable (x : ℕ)

-- Define the total number of main stems, branches, and small branches
def total := 1 + x + x * x

theorem number_of_small_branches (h : total x = 91) : x = 9 :=
by
  -- Proof is not required as per instructions
  sorry

end number_of_small_branches_l1169_116902


namespace units_digit_of_516n_divisible_by_12_l1169_116958

theorem units_digit_of_516n_divisible_by_12 (n : ℕ) (h₀ : n ≤ 9) :
  (516 * 10 + n) % 12 = 0 ↔ n = 0 ∨ n = 4 :=
by 
  sorry

end units_digit_of_516n_divisible_by_12_l1169_116958


namespace intersection_count_l1169_116993

def M (x y : ℝ) : Prop := y^2 = x - 1
def N (x y m : ℝ) : Prop := y = 2 * x - 2 * m^2 + m - 2

theorem intersection_count (m x y : ℝ) :
  (M x y ∧ N x y m) → (∃ n : ℕ, n = 1 ∨ n = 2) :=
sorry

end intersection_count_l1169_116993


namespace calculate_expression_l1169_116973

theorem calculate_expression : (3.15 * 2.5) - 1.75 = 6.125 := 
by
  -- The proof is omitted, indicated by sorry
  sorry

end calculate_expression_l1169_116973


namespace sequence_contains_infinite_squares_l1169_116914

theorem sequence_contains_infinite_squares :
  ∃ f : ℕ → ℕ, ∀ m : ℕ, ∃ n : ℕ, f (n + m) * f (n + m) = 1 + 17 * (n + m) ^ 2 :=
sorry

end sequence_contains_infinite_squares_l1169_116914


namespace ladder_rungs_count_l1169_116980

theorem ladder_rungs_count :
  ∃ (n : ℕ), ∀ (start mid : ℕ),
    start = n / 2 →
    mid = ((start + 5 - 7) + 8 + 7) →
    mid = n →
    n = 27 :=
by
  sorry

end ladder_rungs_count_l1169_116980


namespace ratio_alan_to_ben_l1169_116947

theorem ratio_alan_to_ben (A B L : ℕ) (hA : A = 48) (hL : L = 36) (hB : B = L / 3) : A / B = 4 := by
  sorry

end ratio_alan_to_ben_l1169_116947


namespace recycling_drive_target_l1169_116969

-- Define the collection totals for each section
def section_collections_first_week : List ℝ := [260, 290, 250, 270, 300, 310, 280, 265]

-- Compute total collection for the first week
def total_first_week (collections: List ℝ) : ℝ := collections.sum

-- Compute collection for the second week with a 10% increase
def second_week_increase (collection: ℝ) : ℝ := collection * 1.10
def total_second_week (collections: List ℝ) : ℝ := (collections.map second_week_increase).sum

-- Compute collection for the third week with a 30% increase from the second week
def third_week_increase (collection: ℝ) : ℝ := collection * 1.30
def total_third_week (collections: List ℝ) : ℝ := (collections.map (second_week_increase)).sum * 1.30

-- Total target collection is the sum of collections for three weeks
def target (collections: List ℝ) : ℝ := total_first_week collections + total_second_week collections + total_third_week collections

-- Main theorem to prove
theorem recycling_drive_target : target section_collections_first_week = 7854.25 :=
by
  sorry -- skipping the proof

end recycling_drive_target_l1169_116969


namespace rahul_matches_played_l1169_116995

theorem rahul_matches_played
  (current_avg runs_today new_avg : ℕ)
  (h1 : current_avg = 51)
  (h2 : runs_today = 69)
  (h3 : new_avg = 54)
  : ∃ m : ℕ, ((51 * m + 69) / (m + 1) = 54) ∧ (m = 5) :=
by
  sorry

end rahul_matches_played_l1169_116995


namespace initial_money_l1169_116950

theorem initial_money (M : ℝ)
  (clothes : M * (1 / 3) = M - M * (2 / 3))
  (food : (M - M * (1 / 3)) * (1 / 5) = (M - M * (1 / 3)) - ((M - M * (1 / 3)) * (4 / 5)))
  (travel : ((M - M * (1 / 3)) - ((M - M * (1 / 3)) * (1 / 5))) * (1 / 4) = ((M - M * (1 / 3)) - ((M - M * (1 / 3)) * (1 / 5))) - (((M - M * (1 / 3)) - ((M - M * (1 / 3)) * (1 / 5)))) * (3 / 4))
  (left : ((M - M * (1 / 3)) - (((M - M * (1 / 3)) - ((M - M * (1 / 3)) * (1 / 5))) * (1 / 4))) = 400)
  : M = 1000 := 
sorry

end initial_money_l1169_116950


namespace two_equal_sum_partition_three_equal_sum_partition_l1169_116971

-- Definition 1: Sum of the set X_n
def sum_X_n (n : ℕ) : ℕ := n * (n + 1) / 2

-- Definition 2: Equivalences for partitioning X_n into two equal sum parts
def partition_two_equal_sum (n : ℕ) : Prop :=
  (n % 4 = 0 ∨ n % 4 = 3) ↔ ∃ (A B : Finset ℕ), A ∪ B = Finset.range n ∧ A ∩ B = ∅ ∧ A.sum id = B.sum id

-- Definition 3: Equivalences for partitioning X_n into three equal sum parts
def partition_three_equal_sum (n : ℕ) : Prop :=
  (n % 3 ≠ 1) ↔ ∃ (A B C : Finset ℕ), A ∪ B ∪ C = Finset.range n ∧ (A ∩ B = ∅ ∧ A ∩ C = ∅ ∧ B ∩ C = ∅) ∧ A.sum id = B.sum id ∧ B.sum id = C.sum id

-- Main theorem statements
theorem two_equal_sum_partition (n : ℕ) : partition_two_equal_sum n :=
  sorry

theorem three_equal_sum_partition (n : ℕ) : partition_three_equal_sum n :=
  sorry

end two_equal_sum_partition_three_equal_sum_partition_l1169_116971


namespace addition_comm_subtraction_compare_multiplication_comm_subtraction_greater_l1169_116933

theorem addition_comm (a b : ℕ) : a + b = b + a :=
by sorry

theorem subtraction_compare {a b c : ℕ} (h1 : a < b) (h2 : c = 28) : 56 - c < 65 - c :=
by sorry

theorem multiplication_comm (a b : ℕ) : a * b = b * a :=
by sorry

theorem subtraction_greater {a b c : ℕ} (h1 : a - b = 18) (h2 : a - c = 27) (h3 : 32 = b) (h4 : 23 = c) : a - b > a - c :=
by sorry

end addition_comm_subtraction_compare_multiplication_comm_subtraction_greater_l1169_116933


namespace circle_sector_radius_l1169_116900

theorem circle_sector_radius (r : ℝ) :
  (2 * r + (r * (Real.pi / 3)) = 144) → r = 432 / (6 + Real.pi) := by
  sorry

end circle_sector_radius_l1169_116900


namespace sum_of_b_is_negative_twelve_l1169_116916

-- Conditions: the quadratic equation and its property having exactly one solution
def quadratic_equation (b : ℝ) : Prop :=
  ∀ x : ℝ, 3 * x^2 + b * x + 6 * x + 10 = 0

-- Statement to prove: sum of the values of b is -12, 
-- given the condition that the equation has exactly one solution
theorem sum_of_b_is_negative_twelve :
  ∀ b1 b2 : ℝ, (quadratic_equation b1 ∧ quadratic_equation b2) ∧
  (∀ x : ℝ, 3 * x^2 + (b1 + 6) * x + 10 = 0 ∧ 3 * x^2 + (b2 + 6) * x + 10 = 0) ∧
  (∀ b : ℝ, b = b1 ∨ b = b2) →
  b1 + b2 = -12 :=
by
  sorry

end sum_of_b_is_negative_twelve_l1169_116916


namespace total_votes_l1169_116906

-- Define the conditions
variables (V : ℝ) (votes_second_candidate : ℝ) (percent_second_candidate : ℝ)
variables (h1 : votes_second_candidate = 240)
variables (h2 : percent_second_candidate = 0.30)

-- Statement: The total number of votes is 800 given the conditions.
theorem total_votes (h : percent_second_candidate * V = votes_second_candidate) : V = 800 :=
sorry

end total_votes_l1169_116906


namespace find_a6_geometric_sequence_l1169_116968

noncomputable def geom_seq (a : ℕ → ℝ) (q : ℝ) : Prop :=
∀ n, a (n + 1) = a n * q

theorem find_a6_geometric_sequence (a : ℕ → ℝ) (q : ℝ) (h1 : geom_seq a q) (h2 : a 4 = 7) (h3 : a 8 = 63) : 
  a 6 = 21 :=
sorry

end find_a6_geometric_sequence_l1169_116968


namespace determine_fake_coin_l1169_116919

theorem determine_fake_coin (N : ℕ) : 
  (∃ (n : ℕ), N = 2 * n + 2) ↔ (∃ (n : ℕ), N = 2 * n + 2) := by 
  sorry

end determine_fake_coin_l1169_116919


namespace find_brick_length_l1169_116940

-- Definitions of dimensions
def brick_width : ℝ := 11.25
def brick_height : ℝ := 6
def wall_length : ℝ := 750
def wall_height : ℝ := 600
def wall_thickness : ℝ := 22.5
def num_bricks : ℝ := 6000

-- Volume calculations
def volume_wall : ℝ := wall_length * wall_height * wall_thickness
def volume_brick (x : ℝ) : ℝ := x * brick_width * brick_height

-- Statement of the problem
theorem find_brick_length (length_of_brick : ℝ) :
  volume_wall = num_bricks * volume_brick length_of_brick → length_of_brick = 25 :=
by
  simp [volume_wall, volume_brick, num_bricks, brick_width, brick_height, wall_length, wall_height, wall_thickness]
  intro h 
  sorry

end find_brick_length_l1169_116940


namespace largest_gcd_l1169_116959

theorem largest_gcd (a b : ℕ) (h : a + b = 1008) : ∃ d, d = gcd a b ∧ (∀ d', d' = gcd a b → d' ≤ d) ∧ d = 504 :=
by
  sorry

end largest_gcd_l1169_116959


namespace girls_more_than_boys_by_155_l1169_116932

def number_of_girls : Real := 542.0
def number_of_boys : Real := 387.0
def difference : Real := number_of_girls - number_of_boys

theorem girls_more_than_boys_by_155 :
  difference = 155.0 := 
by
  sorry

end girls_more_than_boys_by_155_l1169_116932


namespace find_sum_on_si_l1169_116946

noncomputable def sum_invested_on_si (r1 r2 r3 : ℝ) (years_si: ℕ) (ci_rate: ℝ) (principal_ci: ℝ) (years_ci: ℕ) (times_compounded: ℕ) :=
  let ci_rate_period := ci_rate / times_compounded
  let amount_ci := principal_ci * (1 + ci_rate_period / 1)^(years_ci * times_compounded)
  let ci := amount_ci - principal_ci
  let si := ci / 2
  let total_si_rate := r1 / 100 + r2 / 100 + r3 / 100
  let principle_si := si / total_si_rate
  principle_si

theorem find_sum_on_si :
  sum_invested_on_si 0.05 0.06 0.07 3 0.10 4000 2 2 = 2394.51 :=
by
  sorry

end find_sum_on_si_l1169_116946


namespace solve_quadratic_eq_l1169_116961

theorem solve_quadratic_eq (x : ℝ) (h : x^2 + 2 * x - 15 = 0) : x = 3 ∨ x = -5 :=
by {
  sorry
}

end solve_quadratic_eq_l1169_116961


namespace minimum_value_of_quadratic_l1169_116954

theorem minimum_value_of_quadratic (x : ℝ) : ∃ (y : ℝ), (∀ x : ℝ, y ≤ x^2 + 2) ∧ (y = 2) :=
by
  sorry

end minimum_value_of_quadratic_l1169_116954


namespace triangle_sin_ratio_cos_side_l1169_116934

noncomputable section

variables (A B C a b c : ℝ)
variables (h1 : a + b + c = 5)
variables (h2 : Real.cos B = 1 / 4)
variables (h3 : Real.cos A - 2 * Real.cos C = (2 * c - a) / b * Real.cos B)

theorem triangle_sin_ratio_cos_side :
  (Real.sin C / Real.sin A = 2) ∧ (b = 2) :=
  sorry

end triangle_sin_ratio_cos_side_l1169_116934


namespace translated_parabola_eq_l1169_116966

-- Define the original parabola
def orig_parabola (x : ℝ) : ℝ := -2 * x^2

-- Define the translation function
def translate_upwards (f : ℝ → ℝ) (dy : ℝ) : (ℝ → ℝ) :=
  fun x => f x + dy

-- Define the translated parabola
def translated_parabola := translate_upwards orig_parabola 3

-- State the theorem
theorem translated_parabola_eq:
  translated_parabola = (fun x : ℝ => -2 * x^2 + 3) :=
by
  sorry

end translated_parabola_eq_l1169_116966


namespace infinite_seq_condition_l1169_116975

theorem infinite_seq_condition (x : ℕ → ℕ) (n m : ℕ) : 
  (∀ i, x i = 0 → x (i + m) = 1) → 
  (∀ i, x i = 1 → x (i + n) = 0) → 
  ∃ d p q : ℕ, n = 2^d * p ∧ m = 2^d * q ∧ p % 2 = 1 ∧ q % 2 = 1  :=
by 
  intros h1 h2 
  sorry

end infinite_seq_condition_l1169_116975


namespace sufficient_but_not_necessary_condition_l1169_116904

variable (a b : ℝ)

theorem sufficient_but_not_necessary_condition (h : |b| + a < 0) : b^2 < a^2 :=
  sorry

end sufficient_but_not_necessary_condition_l1169_116904


namespace distinct_real_roots_l1169_116942

noncomputable def g (x d : ℝ) : ℝ := x^2 + 4*x + d

theorem distinct_real_roots (d : ℝ) :
  (∃! x : ℝ, g (g x d) d = 0) ↔ d = 0 :=
sorry

end distinct_real_roots_l1169_116942


namespace who_finished_in_7th_place_l1169_116953

theorem who_finished_in_7th_place:
  ∀ (Alex Ben Charlie David Ethan : ℕ),
  (Ethan + 4 = Alex) →
  (David + 1 = Ben) →
  (Charlie = Ben + 3) →
  (Alex = Ben + 2) →
  (Ethan + 2 = David) →
  (Ben = 5) →
  Alex = 7 :=
by
  intros Alex Ben Charlie David Ethan h1 h2 h3 h4 h5 h6
  sorry

end who_finished_in_7th_place_l1169_116953


namespace problem_false_proposition_l1169_116912

def p : Prop := ∀ x : ℝ, |x| = x ↔ x > 0

def q : Prop := (¬ ∃ x : ℝ, x^2 - x > 0) ↔ ∀ x : ℝ, x^2 - x ≤ 0

theorem problem_false_proposition : ¬ (p ∧ q) :=
by
  sorry

end problem_false_proposition_l1169_116912


namespace tiffany_bags_difference_l1169_116908

theorem tiffany_bags_difference : 
  ∀ (monday_bags next_day_bags : ℕ), monday_bags = 7 → next_day_bags = 12 → next_day_bags - monday_bags = 5 := 
by
  intros monday_bags next_day_bags h1 h2
  sorry

end tiffany_bags_difference_l1169_116908


namespace deleted_files_l1169_116978

variable {initial_files : ℕ}
variable {files_per_folder : ℕ}
variable {folders : ℕ}

noncomputable def files_deleted (initial_files files_in_folders : ℕ) : ℕ :=
  initial_files - files_in_folders

theorem deleted_files (h1 : initial_files = 27) (h2 : files_per_folder = 6) (h3 : folders = 3) :
  files_deleted initial_files (files_per_folder * folders) = 9 :=
by
  sorry

end deleted_files_l1169_116978


namespace derivative_at_3_l1169_116910

def f (x : ℝ) : ℝ := x^2

theorem derivative_at_3 : deriv f 3 = 6 := by
  sorry

end derivative_at_3_l1169_116910


namespace no_integer_roots_l1169_116915

theorem no_integer_roots (a b : ℤ) : ¬∃ u : ℤ, u^2 + 3 * a * u + 3 * (2 - b^2) = 0 :=
by
  sorry

end no_integer_roots_l1169_116915


namespace math_problem_l1169_116972

variable {x y : ℝ}
variable (hx : x ≠ 0) (hy : y ≠ 0) (h := y^2 - 1 / x^2 ≠ 0) (h₁ := x^2 * y^2 ≠ 1)

theorem math_problem :
  (x^2 - 1 / y^2) / (y^2 - 1 / x^2) = x^2 / y^2 :=
sorry

end math_problem_l1169_116972


namespace bugs_meet_on_diagonal_l1169_116911

noncomputable def isosceles_trapezoid (A B C D : Type) : Prop :=
  ∃ (AB CD : ℝ), (AB > CD) ∧ (AB = AB) ∧ (CD = CD)

noncomputable def same_speeds (speed1 speed2 : ℝ) : Prop :=
  speed1 = speed2

noncomputable def opposite_directions (path1 path2 : ℝ → ℝ) (diagonal_length : ℝ) : Prop :=
  ∀ t, path1 t = diagonal_length - path2 t

noncomputable def bugs_meet (A B C D : Type) (path1 path2 : ℝ → ℝ) (T : ℝ) : Prop :=
  ∃ t ≤ T, path1 t = path2 t

theorem bugs_meet_on_diagonal :
  ∀ (A B C D : Type) (speed : ℝ) (path1 path2 : ℝ → ℝ) (diagonal_length cycle_period : ℝ),
  isosceles_trapezoid A B C D →
  same_speeds speed speed →
  (∀ t, 0 ≤ t → t ≤ cycle_period) →
  opposite_directions path1 path2 diagonal_length →
  bugs_meet A B C D path1 path2 cycle_period :=
by
  intros
  sorry

end bugs_meet_on_diagonal_l1169_116911


namespace original_rectangle_perimeter_difference_is_multiple_of_7_relationship_for_seamless_combination_l1169_116918

-- Condition declarations
variables (x y : ℕ)
variables (hx : x > 0) (hy : y > 0)
variables (h_sum : x + y = 25)
variables (h_diff : (x + 5) * (y + 5) - (x - 2) * (y - 2) = 196)

-- Lean 4 statements for the proof problem
theorem original_rectangle_perimeter (x y : ℕ) (hx : x > 0) (hy : y > 0) (h_sum : x + y = 25) : 2 * (x + y) = 50 := by
  sorry

theorem difference_is_multiple_of_7 (x y : ℕ) (hx : x > 0) (hy : y > 0) 
  (h_diff : (x + 5) * (y + 5) - (x - 2) * (y - 2) = 196) : 7 ∣ ((x + 5) * (y + 5) - (x - 2) * (y - 2)) := by
  sorry

theorem relationship_for_seamless_combination (x y : ℕ) (h_sum : x + y = 25) (h_seamless : (x+5)*(y+5) = (x*(y+5))) : x = y + 5 := by
  sorry

end original_rectangle_perimeter_difference_is_multiple_of_7_relationship_for_seamless_combination_l1169_116918


namespace find_area_MOI_l1169_116974

noncomputable def incenter_coords (a b c : ℝ) (A B C : (ℝ × ℝ)) : (ℝ × ℝ) :=
  ((a * A.1 + b * B.1 + c * C.1) / (a + b + c), (a * A.2 + b * B.2 + c * C.2) / (a + b + c))

noncomputable def shoelace_area (P Q R : (ℝ × ℝ)) : ℝ :=
  0.5 * abs (P.1 * (Q.2 - R.2) + Q.1 * (R.2 - P.2) + R.1 * (P.2 - Q.2))

theorem find_area_MOI :
  let A := (0, 0)
  let B := (8, 0)
  let C := (0, 17)
  let O := (4, 8.5)
  let I := incenter_coords 8 15 17 A B C
  let M := (6.25, 6.25)
  shoelace_area M O I = 25.78125 :=
by
  sorry

end find_area_MOI_l1169_116974


namespace primes_up_to_floor_implies_all_primes_l1169_116963

/-- Define the function f. -/
def f (x p : ℕ) : ℕ := x^2 + x + p

/-- Define the initial prime condition. -/
def primes_up_to_floor_sqrt_p_over_3 (p : ℕ) : Prop :=
  ∀ x, x ≤ Nat.floor (Nat.sqrt (p / 3)) → Nat.Prime (f x p)

/-- Define the property we want to prove. -/
def all_primes_up_to_p_minus_2 (p : ℕ) : Prop :=
  ∀ x, x ≤ p - 2 → Nat.Prime (f x p)

/-- The main theorem statement. -/
theorem primes_up_to_floor_implies_all_primes
  (p : ℕ) (h : primes_up_to_floor_sqrt_p_over_3 p) : all_primes_up_to_p_minus_2 p :=
sorry

end primes_up_to_floor_implies_all_primes_l1169_116963


namespace original_population_before_changes_l1169_116998

open Nat

def halved_population (p: ℕ) (years: ℕ) : ℕ := p / (2^years)

theorem original_population_before_changes (P_init P_final : ℕ)
    (new_people : ℕ) (people_moved_out : ℕ) :
    new_people = 100 →
    people_moved_out = 400 →
    ∀ years, (years = 4 → halved_population P_final years = 60) →
    ∃ P_before_change, P_before_change = 780 ∧
    P_init = P_before_change + new_people - people_moved_out ∧
    halved_population P_init years = P_final := 
by
  intros
  sorry

end original_population_before_changes_l1169_116998


namespace part1_part2_l1169_116937

noncomputable def f (x a : ℝ) : ℝ := |x - 1| + |2 * x + a|

theorem part1 (x : ℝ) : f x 1 + |x - 1| ≥ 3 :=
  sorry

theorem part2 (a : ℝ) (h : ∃ x : ℝ, f x a = 2) : a = 2 ∨ a = -6 :=
  sorry

end part1_part2_l1169_116937


namespace find_domain_l1169_116907

noncomputable def domain (x : ℝ) : Prop :=
  (2 * x + 1 ≥ 0) ∧ (3 - 4 * x ≥ 0)

theorem find_domain :
  {x : ℝ | domain x} = {x : ℝ | -1/2 ≤ x ∧ x ≤ 3/4} :=
by
  sorry

end find_domain_l1169_116907


namespace average_percent_score_is_77_l1169_116905

def numberOfStudents : ℕ := 100

def percentage_counts : List (ℕ × ℕ) :=
[(100, 7), (90, 18), (80, 35), (70, 25), (60, 10), (50, 3), (40, 2)]

noncomputable def average_score (counts : List (ℕ × ℕ)) : ℚ :=
  (counts.foldl (λ acc p => acc + (p.1 * p.2)) 0 : ℚ) / numberOfStudents

theorem average_percent_score_is_77 : average_score percentage_counts = 77 := by
  sorry

end average_percent_score_is_77_l1169_116905


namespace share_of_B_l1169_116965

noncomputable def problem_statement (A B C : ℝ) : Prop :=
  A + B + C = 595 ∧ A = (2/3) * B ∧ B = (1/4) * C

theorem share_of_B (A B C : ℝ) (h : problem_statement A B C) : B = 105 :=
by
  -- Proof omitted
  sorry

end share_of_B_l1169_116965


namespace find_fx_l1169_116926

theorem find_fx (f : ℝ → ℝ) 
  (h1 : ∀ x > 0, f (-x) = -(2 * x - 3)) 
  (h2 : ∀ x < 0, -f x = f (-x)) :
  ∀ x < 0, f x = 2 * x + 3 :=
by
  sorry

end find_fx_l1169_116926


namespace average_birth_rate_l1169_116903

theorem average_birth_rate (B : ℕ) (death_rate : ℕ) (net_increase : ℕ) (seconds_per_day : ℕ) 
  (two_sec_intervals : ℕ) (H1 : death_rate = 2) (H2 : net_increase = 86400) (H3 : seconds_per_day = 86400) 
  (H4 : two_sec_intervals = seconds_per_day / 2) 
  (H5 : net_increase = (B - death_rate) * two_sec_intervals) : B = 4 := 
by 
  sorry

end average_birth_rate_l1169_116903


namespace graph_of_direct_proportion_is_line_l1169_116962

-- Define the direct proportion function
def direct_proportion (k : ℝ) (x : ℝ) : ℝ :=
  k * x

-- State the theorem to prove the graph of this function is a straight line
theorem graph_of_direct_proportion_is_line (k : ℝ) :
  ∃ a b : ℝ, a ≠ 0 ∧ ∀ x : ℝ, direct_proportion k x = a * x + b ∧ b = 0 := 
by 
  sorry

end graph_of_direct_proportion_is_line_l1169_116962


namespace max_cookies_Andy_eats_l1169_116929

theorem max_cookies_Andy_eats (cookies_total : ℕ) (h_cookies_total : cookies_total = 30) 
  (exists_pos_a : ∃ a : ℕ, a > 0 ∧ 3 * a = 30 - a ∧ (∃ k : ℕ, 3 * a = k ∧ ∃ m : ℕ, a = m)) 
  : ∃ max_a : ℕ, max_a ≤ 7 ∧ 3 * max_a < cookies_total ∧ 3 * max_a ∣ cookies_total ∧ max_a = 6 :=
by
  sorry

end max_cookies_Andy_eats_l1169_116929


namespace sum_of_sides_is_seven_l1169_116964

def triangle_sides : ℕ := 3
def quadrilateral_sides : ℕ := 4
def sum_of_sides : ℕ := triangle_sides + quadrilateral_sides

theorem sum_of_sides_is_seven : sum_of_sides = 7 :=
by
  sorry

end sum_of_sides_is_seven_l1169_116964


namespace white_truck_chance_l1169_116986

-- Definitions from conditions
def trucks : ℕ := 50
def cars : ℕ := 40
def vans : ℕ := 30

def red_trucks : ℕ := 50 / 2
def black_trucks : ℕ := (20 * 50) / 100

-- The remaining percentage (30%) of trucks is assumed to be white.
def white_trucks : ℕ := (30 * 50) / 100

def total_vehicles : ℕ := trucks + cars + vans

-- Given
def percentage_white_truck : ℕ := (white_trucks * 100) / total_vehicles

-- Theorem that proves the problem statement
theorem white_truck_chance : percentage_white_truck = 13 := 
by
  -- Proof will be written here (currently stubbed)
  sorry

end white_truck_chance_l1169_116986


namespace carol_can_invite_friends_l1169_116922

-- Definitions based on the problem's conditions
def invitations_per_pack := 9
def packs_bought := 5

-- Required proof statement
theorem carol_can_invite_friends :
  invitations_per_pack * packs_bought = 45 :=
by
  sorry

end carol_can_invite_friends_l1169_116922


namespace find_b_plus_m_l1169_116988

def line1 (m : ℝ) (x : ℝ) : ℝ := m * x + 7
def line2 (b : ℝ) (x : ℝ) : ℝ := 4 * x + b

theorem find_b_plus_m :
  ∃ (m b : ℝ), line1 m 8 = 11 ∧ line2 b 8 = 11 ∧ b + m = -20.5 :=
sorry

end find_b_plus_m_l1169_116988


namespace initial_maple_trees_l1169_116960

theorem initial_maple_trees
  (initial_maple_trees : ℕ)
  (to_be_planted : ℕ)
  (final_maple_trees : ℕ)
  (h1 : to_be_planted = 9)
  (h2 : final_maple_trees = 11) :
  initial_maple_trees + to_be_planted = final_maple_trees → initial_maple_trees = 2 := 
by 
  sorry

end initial_maple_trees_l1169_116960


namespace cylindrical_coordinates_of_point_l1169_116927

noncomputable def cylindrical_coordinates (x y z : ℝ) : ℝ × ℝ × ℝ :=
  let r := Real.sqrt (x^2 + y^2)
  let θ := if x = -r then Real.pi else 0 -- From the step if cos θ = -1
  (r, θ, z)

theorem cylindrical_coordinates_of_point :
  cylindrical_coordinates (-5) 0 (-8) = (5, Real.pi, -8) :=
by
  -- placeholder for the actual proof
  sorry

end cylindrical_coordinates_of_point_l1169_116927


namespace three_hundred_thousand_times_three_hundred_thousand_minus_one_million_l1169_116983

theorem three_hundred_thousand_times_three_hundred_thousand_minus_one_million :
  (300000 * 300000) - 1000000 = 89990000000 := by
  sorry 

end three_hundred_thousand_times_three_hundred_thousand_minus_one_million_l1169_116983


namespace circle_chord_length_equal_l1169_116923

def equation_of_circle (D E F : ℝ) : Prop :=
  ∀ x y : ℝ, x^2 + y^2 + D * x + E * y + F = 0

def distances_equal (D E F : ℝ) : Prop :=
  (D^2 ≠ E^2 ∧ E^2 > 4 * F) → 
  (∀ x y : ℝ, (x^2 + y^2 + D * x + E * y + F = 0) → (x = -D/2) ∧ (y = -E/2) → (abs x = abs y))

theorem circle_chord_length_equal (D E F : ℝ) (h : D^2 ≠ E^2 ∧ E^2 > 4 * F) :
  distances_equal D E F :=
by
  sorry

end circle_chord_length_equal_l1169_116923


namespace even_iff_a_zero_max_value_f_l1169_116901

noncomputable def f (x a : ℝ) : ℝ := -x^2 + |x - a| + a + 1

theorem even_iff_a_zero (a : ℝ) : (∀ x, f x a = f (-x) a) ↔ a = 0 :=
by {
  -- Proof is omitted
  sorry
}

theorem max_value_f (a : ℝ) : 
  ∃ max_val : ℝ, 
    ( 
      (-1/2 < a ∧ a ≤ 0 ∧ max_val = 5/4) ∨ 
      (0 < a ∧ a < 1/2 ∧ max_val = 5/4 + 2*a) ∨ 
      ((a ≤ -1/2 ∨ a ≥ 1/2) ∧ max_val = -a^2 + a + 1)
    ) :=
by {
  -- Proof is omitted
  sorry
}

end even_iff_a_zero_max_value_f_l1169_116901


namespace trapezoid_diagonals_perpendicular_iff_geometric_mean_l1169_116990

structure Trapezoid :=
(a b c d e f : ℝ) -- lengths of sides a, b, c, d, and diagonals e, f.
(right_angle : d^2 = a^2 + c^2) -- Condition that makes it a right-angled trapezoid.

theorem trapezoid_diagonals_perpendicular_iff_geometric_mean (T : Trapezoid) :
  (T.e * T.e + T.f * T.f = T.a * T.a + T.b * T.b + T.c * T.c + T.d * T.d) ↔ 
  (T.d * T.d = T.a * T.c) := 
sorry

end trapezoid_diagonals_perpendicular_iff_geometric_mean_l1169_116990


namespace average_price_per_bottle_l1169_116949

/-
  Given:
  * Number of large bottles: 1300
  * Price per large bottle: 1.89
  * Number of small bottles: 750
  * Price per small bottle: 1.38
  
  Prove:
  The approximate average price per bottle is 1.70
-/
theorem average_price_per_bottle : 
  let num_large_bottles := 1300
  let price_per_large_bottle := 1.89
  let num_small_bottles := 750
  let price_per_small_bottle := 1.38
  let total_cost_large_bottles := num_large_bottles * price_per_large_bottle
  let total_cost_small_bottles := num_small_bottles * price_per_small_bottle
  let total_number_bottles := num_large_bottles + num_small_bottles
  let overall_total_cost := total_cost_large_bottles + total_cost_small_bottles
  let average_price := overall_total_cost / total_number_bottles
  average_price = 1.70 :=
by
  sorry

end average_price_per_bottle_l1169_116949


namespace gcd_lcm_sum_l1169_116920

-- Define the given numbers
def a1 := 54
def b1 := 24
def a2 := 48
def b2 := 18

-- Define the GCD and LCM functions in Lean
def gcd_ab := Nat.gcd a1 b1
def lcm_cd := Nat.lcm a2 b2

-- Define the final sum
def final_sum := gcd_ab + lcm_cd

-- State the equality that represents the problem
theorem gcd_lcm_sum : final_sum = 150 := by
  sorry

end gcd_lcm_sum_l1169_116920


namespace seven_divides_n_l1169_116957

theorem seven_divides_n (n : ℕ) (h1 : n ≥ 2) (h2 : n ∣ 3^n + 4^n) : 7 ∣ n :=
sorry

end seven_divides_n_l1169_116957


namespace proof_l1169_116944

open Set

variable (U M P : Set ℕ)

noncomputable def prob_statement : Prop :=
  let C_U (A : Set ℕ) : Set ℕ := {x ∈ U | x ∉ A}
  U = {1,2,3,4,5,6,7,8} ∧ M = {2,3,4} ∧ P = {1,3,6} ∧ C_U (M ∪ P) = {5,7,8}

theorem proof : prob_statement {1,2,3,4,5,6,7,8} {2,3,4} {1,3,6} :=
by
  sorry

end proof_l1169_116944


namespace marks_in_physics_l1169_116945

section
variables (P C M B CS : ℕ)

-- Given conditions
def condition_1 : Prop := P + C + M + B + CS = 375
def condition_2 : Prop := P + M + B = 255
def condition_3 : Prop := P + C + CS = 210

-- Prove that P = 90
theorem marks_in_physics : condition_1 P C M B CS → condition_2 P M B → condition_3 P C CS → P = 90 :=
by sorry
end

end marks_in_physics_l1169_116945


namespace students_chocolate_milk_l1169_116931

-- Definitions based on the problem conditions
def students_strawberry_milk : ℕ := 15
def students_regular_milk : ℕ := 3
def total_milks_taken : ℕ := 20

-- The proof goal
theorem students_chocolate_milk : total_milks_taken - (students_strawberry_milk + students_regular_milk) = 2 := by
  -- The proof steps will go here (not required as per instructions)
  sorry

end students_chocolate_milk_l1169_116931


namespace irreducible_positive_fraction_unique_l1169_116994

theorem irreducible_positive_fraction_unique
  (a b : ℕ) (h_pos : a > 0 ∧ b > 0) (h_gcd : Nat.gcd a b = 1)
  (h_eq : (a + 12) * b = 3 * a * (b + 12)) :
  a = 2 ∧ b = 9 :=
by
  sorry

end irreducible_positive_fraction_unique_l1169_116994


namespace first_lock_stall_time_eq_21_l1169_116976

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

end first_lock_stall_time_eq_21_l1169_116976


namespace value_of_y_l1169_116930

theorem value_of_y : (∃ y : ℝ, (1 / 3 - 1 / 4 = 4 / y) ∧ y = 48) :=
by
  sorry

end value_of_y_l1169_116930


namespace total_cookies_after_three_days_l1169_116936

-- Define the initial conditions
def cookies_baked_monday : ℕ := 32
def cookies_baked_tuesday : ℕ := cookies_baked_monday / 2
def cookies_baked_wednesday_before : ℕ := cookies_baked_tuesday * 3
def brother_ate : ℕ := 4

-- Define the total cookies before brother ate any
def total_cookies_before : ℕ := cookies_baked_monday + cookies_baked_tuesday + cookies_baked_wednesday_before

-- Define the total cookies after brother ate some
def total_cookies_after : ℕ := total_cookies_before - brother_ate

-- The proof statement
theorem total_cookies_after_three_days : total_cookies_after = 92 := by
  -- Here, we would provide the proof, but we add sorry for now to compile successfully.
  sorry

end total_cookies_after_three_days_l1169_116936


namespace stratified_sampling_workshops_l1169_116941

theorem stratified_sampling_workshops (units_A units_B units_C sample_B n : ℕ) 
(hA : units_A = 96) 
(hB : units_B = 84) 
(hC : units_C = 60) 
(hSample_B : sample_B = 7) 
(hn : (sample_B : ℚ) / n = (units_B : ℚ) / (units_A + units_B + units_C)) : 
  n = 70 :=
  by
  sorry

end stratified_sampling_workshops_l1169_116941


namespace number_of_cakes_sold_l1169_116987

namespace Bakery

variables (cakes pastries sold_cakes sold_pastries : ℕ)

-- Defining the conditions
def pastries_sold := 154
def more_pastries_than_cakes := 76

-- Defining the problem statement
theorem number_of_cakes_sold (h1 : sold_pastries = pastries_sold) 
                             (h2 : sold_pastries = sold_cakes + more_pastries_than_cakes) : 
                             sold_cakes = 78 :=
by {
  sorry
}

end Bakery

end number_of_cakes_sold_l1169_116987


namespace avg_growth_rate_equation_l1169_116977

/-- This theorem formalizes the problem of finding the equation for the average growth rate of working hours.
    Given that the average working hours in the first week are 40 hours and in the third week are 48.4 hours,
    we need to show that the equation for the growth rate \(x\) satisfies \( 40(1 + x)^2 = 48.4 \). -/
theorem avg_growth_rate_equation (x : ℝ) (first_week_hours third_week_hours : ℝ) 
  (h1: first_week_hours = 40) (h2: third_week_hours = 48.4) :
  40 * (1 + x) ^ 2 = 48.4 :=
sorry

end avg_growth_rate_equation_l1169_116977


namespace open_box_volume_l1169_116979

theorem open_box_volume (l w s : ℕ) (h1 : l = 50)
  (h2 : w = 36) (h3 : s = 8) : (l - 2 * s) * (w - 2 * s) * s = 5440 :=
by {
  sorry
}

end open_box_volume_l1169_116979


namespace total_wet_surface_area_l1169_116939

def cistern_length (L : ℝ) := L = 5
def cistern_width (W : ℝ) := W = 4
def water_depth (D : ℝ) := D = 1.25

theorem total_wet_surface_area (L W D A : ℝ) 
  (hL : cistern_length L) 
  (hW : cistern_width W) 
  (hD : water_depth D) :
  A = 42.5 :=
by
  subst hL
  subst hW
  subst hD
  sorry

end total_wet_surface_area_l1169_116939


namespace speed_of_man_rowing_upstream_l1169_116984

-- Define conditions
def V_m : ℝ := 20 -- speed of the man in still water (kmph)
def V_downstream : ℝ := 25 -- speed of the man rowing downstream (kmph)
def V_s : ℝ := V_downstream - V_m -- calculate the speed of the stream

-- Define the theorem to prove the speed of the man rowing upstream
theorem speed_of_man_rowing_upstream 
  (V_m : ℝ) (V_downstream : ℝ) (V_s : ℝ := V_downstream - V_m) : 
  V_upstream = V_m - V_s :=
by
  sorry

end speed_of_man_rowing_upstream_l1169_116984


namespace initial_birds_l1169_116909

-- Given conditions
def number_birds_initial (x : ℕ) : Prop :=
  ∃ (y : ℕ), y = 4 ∧ (x + y = 6)

-- Proof statement
theorem initial_birds : ∃ x : ℕ, number_birds_initial x ↔ x = 2 :=
by {
  sorry
}

end initial_birds_l1169_116909


namespace circus_tickets_l1169_116956

variable (L U : ℕ)

theorem circus_tickets (h1 : L + U = 80) (h2 : 30 * L + 20 * U = 2100) : L = 50 :=
by
  sorry

end circus_tickets_l1169_116956


namespace general_term_l1169_116924

noncomputable def F (n : ℕ) : ℝ :=
  1 / (Real.sqrt 5) * (((1 + Real.sqrt 5) / 2)^(n-2) - ((1 - Real.sqrt 5) / 2)^(n-2))

noncomputable def a : ℕ → ℝ
| 0 => 1
| 1 => 5
| n+2 => a (n+1) * a n / Real.sqrt ((a (n+1))^2 + (a n)^2 + 1)

theorem general_term (n : ℕ) :
  a n = (2^(F (n+2)) * 13^(F (n+1)) * 5^(-2 * F (n+1)) - 1)^(1/2) := sorry

end general_term_l1169_116924


namespace determine_k_l1169_116913

noncomputable def p (x y : ℝ) : ℝ := x^2 - y^2
noncomputable def q (x y : ℝ) : ℝ := Real.log (x - y)

def m (k : ℝ) : ℝ := 2 * k
def w (n : ℝ) : ℝ := n + 1

theorem determine_k (k : ℝ) (c : ℝ → ℝ → ℝ) (v : ℝ → ℝ → ℝ) (n : ℝ) :
  p 32 6 = k * c 32 6 ∧
  p 45 10 = m k * c 45 10 ∧
  q 15 5 = n * v 15 5 ∧
  q 28 7 = w n * v 28 7 →
  k = 1925 / 1976 :=
by
  sorry

end determine_k_l1169_116913


namespace fraction_simplification_l1169_116999

theorem fraction_simplification :
  (3 + 6 - 12 + 24 + 48 - 96) / (6 + 12 - 24 + 48 + 96 - 192) = 1 / 2 :=
by
  sorry

end fraction_simplification_l1169_116999


namespace gcd_1029_1437_5649_l1169_116981

theorem gcd_1029_1437_5649 : Nat.gcd (Nat.gcd 1029 1437) 5649 = 3 := by
  sorry

end gcd_1029_1437_5649_l1169_116981


namespace initial_green_hard_hats_l1169_116989

noncomputable def initial_pink_hard_hats : ℕ := 26
noncomputable def initial_yellow_hard_hats : ℕ := 24
noncomputable def carl_taken_pink_hard_hats : ℕ := 4
noncomputable def john_taken_pink_hard_hats : ℕ := 6
noncomputable def john_taken_green_hard_hats (G : ℕ) : ℕ := 2 * john_taken_pink_hard_hats
noncomputable def remaining_pink_hard_hats : ℕ := initial_pink_hard_hats - carl_taken_pink_hard_hats - john_taken_pink_hard_hats
noncomputable def total_remaining_hard_hats (G : ℕ) : ℕ := remaining_pink_hard_hats + (G - john_taken_green_hard_hats G) + initial_yellow_hard_hats

theorem initial_green_hard_hats (G : ℕ) :
  total_remaining_hard_hats G = 43 ↔ G = 15 := by
  sorry

end initial_green_hard_hats_l1169_116989


namespace lesser_number_is_32_l1169_116943

variable (x y : ℕ)

theorem lesser_number_is_32 (h1 : y = 2 * x) (h2 : x + y = 96) : x = 32 := 
sorry

end lesser_number_is_32_l1169_116943


namespace percent_fair_hair_l1169_116917

theorem percent_fair_hair (total_employees : ℕ) (total_women_fair_hair : ℕ)
  (percent_fair_haired_women : ℕ) (percent_women_fair_hair : ℕ)
  (h1 : total_women_fair_hair = (total_employees * percent_women_fair_hair) / 100)
  (h2 : percent_fair_haired_women * total_women_fair_hair = total_employees * 10) :
  (25 * total_employees = 100 * total_women_fair_hair) :=
by {
  sorry
}

end percent_fair_hair_l1169_116917


namespace solve_base_6_addition_l1169_116955

variables (X Y k : ℕ)

theorem solve_base_6_addition (h1 : Y + 3 = X) (h2 : ∃ k, X + 5 = 2 + 6 * k) : X + Y = 3 :=
sorry

end solve_base_6_addition_l1169_116955


namespace gcd_three_numbers_4557_1953_5115_l1169_116948

theorem gcd_three_numbers_4557_1953_5115 : Nat.gcd (Nat.gcd 4557 1953) 5115 = 93 := 
by 
  sorry

end gcd_three_numbers_4557_1953_5115_l1169_116948


namespace inverse_proportion_quadrants_l1169_116985

theorem inverse_proportion_quadrants (k : ℝ) :
  (∀ (x : ℝ), x ≠ 0 → ((x > 0 → (k - 3) / x > 0) ∧ (x < 0 → (k - 3) / x < 0))) → k > 3 :=
by
  intros h
  sorry

end inverse_proportion_quadrants_l1169_116985


namespace sum_of_excluded_numbers_l1169_116935

theorem sum_of_excluded_numbers (S : ℕ) (X : ℕ) (n m : ℕ) (averageN : ℕ) (averageM : ℕ)
  (h1 : S = 34 * 8) 
  (h2 : n = 8) 
  (h3 : m = 6) 
  (h4 : averageN = 34) 
  (h5 : averageM = 29) 
  (hS : S = n * averageN) 
  (hX : S - X = m * averageM) : 
  X = 98 := by
  sorry

end sum_of_excluded_numbers_l1169_116935


namespace maximize_profit_l1169_116996

noncomputable def profit (x : ℝ) : ℝ :=
  (x - 8) * (100 - 10 * (x - 10))

theorem maximize_profit :
  let max_price := 14
  let max_profit := 360
  (∀ x > 10, profit x ≤ profit max_price) ∧ profit max_price = max_profit :=
by
  let max_price := 14
  let max_profit := 360
  sorry

end maximize_profit_l1169_116996


namespace distance_between_chords_l1169_116991

theorem distance_between_chords (R AB CD : ℝ) (hR : R = 15) (hAB : AB = 18) (hCD : CD = 24) : 
  ∃ d : ℝ, d = 21 :=
by 
  sorry

end distance_between_chords_l1169_116991


namespace no_solution_when_k_eq_7_l1169_116992

theorem no_solution_when_k_eq_7 
  (x : ℝ) (h₁ : x ≠ 4) (h₂ : x ≠ 8) : 
  (∀ k : ℝ, (x - 3) / (x - 4) = (x - k) / (x - 8) → False) ↔ k = 7 :=
by
  sorry

end no_solution_when_k_eq_7_l1169_116992


namespace production_time_l1169_116938

variable (a m : ℝ) -- Define a and m as real numbers

-- State the problem as a theorem in Lean
theorem production_time : (a / m) * 200 = 200 * (a / m) := by
  sorry

end production_time_l1169_116938


namespace exactly_one_passes_l1169_116970

theorem exactly_one_passes (P_A P_B : ℚ) (hA : P_A = 3 / 5) (hB : P_B = 1 / 3) : 
  (1 - P_A) * P_B + P_A * (1 - P_B) = 8 / 15 :=
by
  -- skipping the proof as per requirement
  sorry

end exactly_one_passes_l1169_116970


namespace solution_l1169_116921

noncomputable def problem : Prop :=
  let num_apprentices := 200
  let num_junior := 20
  let num_intermediate := 60
  let num_senior := 60
  let num_technician := 40
  let num_senior_technician := 20
  let total_technician := num_technician + num_senior_technician
  let sampling_ratio := 10 / num_apprentices
  
  -- Number of technicians (including both technician and senior technicians) in the exchange group
  let num_technicians_selected := total_technician * sampling_ratio

  -- Probability Distribution of X
  let P_X_0 := 7 / 24
  let P_X_1 := 21 / 40
  let P_X_2 := 7 / 40
  let P_X_3 := 1 / 120

  -- Expected value of X
  let E_X := (0 * P_X_0) + (1 * P_X_1) + (2 * P_X_2) + (3 * P_X_3)
  E_X = 9 / 10

theorem solution : problem :=
  sorry

end solution_l1169_116921


namespace slices_per_sandwich_l1169_116925

theorem slices_per_sandwich (total_sandwiches : ℕ) (total_slices : ℕ) (h1 : total_sandwiches = 5) (h2 : total_slices = 15) :
  total_slices / total_sandwiches = 3 :=
by sorry

end slices_per_sandwich_l1169_116925


namespace smallest_product_not_factor_of_48_l1169_116982

theorem smallest_product_not_factor_of_48 (a b : ℕ) (h1 : a ≠ b) (h2 : a ∣ 48) (h3 : b ∣ 48) (h4 : ¬ (a * b ∣ 48)) : a * b = 32 :=
sorry

end smallest_product_not_factor_of_48_l1169_116982
