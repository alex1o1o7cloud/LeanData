import Mathlib

namespace NUMINAMATH_CALUDE_marias_first_stop_distance_l3416_341696

def total_distance : ℝ := 560

def distance_before_first_stop : ℝ → Prop := λ x =>
  let remaining_after_first := total_distance - x
  let second_stop_distance := (1/4) * remaining_after_first
  let final_leg := 210
  second_stop_distance + final_leg = remaining_after_first

theorem marias_first_stop_distance :
  ∃ x, distance_before_first_stop x ∧ x = 280 :=
sorry

end NUMINAMATH_CALUDE_marias_first_stop_distance_l3416_341696


namespace NUMINAMATH_CALUDE_inequality_holds_l3416_341602

theorem inequality_holds (a : ℝ) (h : a ≥ 7/2) : 
  ∀ θ : ℝ, 0 ≤ θ ∧ θ ≤ π/2 → 
    ∀ x : ℝ, (x + 3 + 2 * Real.sin θ * Real.cos θ)^2 + 
              (x + a * Real.sin θ + a * Real.cos θ)^2 ≥ 1/8 := by
  sorry

end NUMINAMATH_CALUDE_inequality_holds_l3416_341602


namespace NUMINAMATH_CALUDE_freshmen_sophomores_without_pets_l3416_341667

theorem freshmen_sophomores_without_pets (total_students : ℕ) 
  (freshmen_sophomore_ratio : ℚ) (pet_owner_ratio : ℚ) : ℕ :=
  by
  sorry

#check freshmen_sophomores_without_pets 400 (1/2) (1/5) = 160

end NUMINAMATH_CALUDE_freshmen_sophomores_without_pets_l3416_341667


namespace NUMINAMATH_CALUDE_larger_number_of_pair_l3416_341644

theorem larger_number_of_pair (x y : ℝ) (h1 : x + y = 29) (h2 : x - y = 5) : 
  max x y = 17 := by
sorry

end NUMINAMATH_CALUDE_larger_number_of_pair_l3416_341644


namespace NUMINAMATH_CALUDE_eighth_power_sum_exists_l3416_341687

theorem eighth_power_sum_exists (ζ₁ ζ₂ ζ₃ : ℂ) 
  (sum_condition : ζ₁ + ζ₂ + ζ₃ = 2)
  (square_sum_condition : ζ₁^2 + ζ₂^2 + ζ₃^2 = 6)
  (cube_sum_condition : ζ₁^3 + ζ₂^3 + ζ₃^3 = 18) :
  ∃ s₈ : ℂ, ζ₁^8 + ζ₂^8 + ζ₃^8 = s₈ := by
  sorry

end NUMINAMATH_CALUDE_eighth_power_sum_exists_l3416_341687


namespace NUMINAMATH_CALUDE_ice_cream_permutations_l3416_341628

theorem ice_cream_permutations :
  Finset.card (Finset.univ.image (fun σ : Equiv.Perm (Fin 5) => σ)) = 120 := by
  sorry

end NUMINAMATH_CALUDE_ice_cream_permutations_l3416_341628


namespace NUMINAMATH_CALUDE_quadratic_vertex_on_x_axis_l3416_341635

/-- The quadratic function -x^2 + 4x + t has its vertex on the x-axis if and only if t = -4 -/
theorem quadratic_vertex_on_x_axis (t : ℝ) : 
  (∃ x : ℝ, ∀ y : ℝ, y = -x^2 + 4*x + t → y = 0) ↔ t = -4 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_vertex_on_x_axis_l3416_341635


namespace NUMINAMATH_CALUDE_min_values_for_constrained_x_y_l3416_341685

theorem min_values_for_constrained_x_y :
  ∀ x y : ℝ, x > 0 → y > 0 → 2 * x + y = 1 →
  (∀ a b : ℝ, a > 0 → b > 0 → 2 * a + b = 1 → 2 / x + 1 / y ≤ 2 / a + 1 / b) ∧
  (∀ a b : ℝ, a > 0 → b > 0 → 2 * a + b = 1 → 4 * x^2 + y^2 ≤ 4 * a^2 + b^2) ∧
  (2 / x + 1 / y = 9) ∧
  (4 * x^2 + y^2 = 1/2) := by
sorry

end NUMINAMATH_CALUDE_min_values_for_constrained_x_y_l3416_341685


namespace NUMINAMATH_CALUDE_concert_group_discount_l3416_341661

theorem concert_group_discount (P : ℝ) (h : P > 0) :
  ∃ (x : ℕ), 3 * P = (3 + x) * (0.75 * P) ∧ 3 + x = 4 := by
  sorry

end NUMINAMATH_CALUDE_concert_group_discount_l3416_341661


namespace NUMINAMATH_CALUDE_average_string_length_l3416_341616

theorem average_string_length (s1 s2 s3 : ℝ) (h1 : s1 = 1) (h2 : s2 = 3) (h3 : s3 = 5) :
  (s1 + s2 + s3) / 3 = 3 := by
  sorry

end NUMINAMATH_CALUDE_average_string_length_l3416_341616


namespace NUMINAMATH_CALUDE_equation_solution_l3416_341699

theorem equation_solution : 
  ∀ x : ℝ, 
    (((x + 1)^2 + 1) / (x + 1) + ((x + 4)^2 + 4) / (x + 4) = 
     ((x + 2)^2 + 2) / (x + 2) + ((x + 3)^2 + 3) / (x + 3)) ↔ 
    (x = 0 ∨ x = -5/2) :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l3416_341699


namespace NUMINAMATH_CALUDE_fewer_girls_than_boys_l3416_341647

theorem fewer_girls_than_boys (total_students : ℕ) (girls_ratio boys_ratio : ℕ) : 
  total_students = 24 →
  girls_ratio = 3 →
  boys_ratio = 5 →
  total_students * girls_ratio / (girls_ratio + boys_ratio) = 9 ∧
  total_students * boys_ratio / (girls_ratio + boys_ratio) = 15 ∧
  15 - 9 = 6 :=
by sorry

end NUMINAMATH_CALUDE_fewer_girls_than_boys_l3416_341647


namespace NUMINAMATH_CALUDE_planes_parallel_if_perpendicular_to_parallel_lines_l3416_341686

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the parallel relation for lines
variable (parallel_lines : Line → Line → Prop)

-- Define the perpendicular relation between a line and a plane
variable (perpendicular_line_plane : Line → Plane → Prop)

-- Define the parallel relation for planes
variable (parallel_planes : Plane → Plane → Prop)

-- Theorem statement
theorem planes_parallel_if_perpendicular_to_parallel_lines
  (m n : Line) (α β : Plane) :
  parallel_lines m n →
  perpendicular_line_plane m α →
  perpendicular_line_plane n β →
  parallel_planes α β :=
sorry

end NUMINAMATH_CALUDE_planes_parallel_if_perpendicular_to_parallel_lines_l3416_341686


namespace NUMINAMATH_CALUDE_total_distance_walked_l3416_341605

/-- The total distance walked by two girls, given one walked twice as far as the other -/
theorem total_distance_walked (nadia_distance : ℝ) (h_nadia : nadia_distance = 18) 
  (h_twice : nadia_distance = 2 * (nadia_distance / 2)) : 
  nadia_distance + (nadia_distance / 2) = 27 := by
  sorry

#check total_distance_walked

end NUMINAMATH_CALUDE_total_distance_walked_l3416_341605


namespace NUMINAMATH_CALUDE_determinant_inequality_solution_l3416_341607

-- Define the determinant of a 2x2 matrix
def det (a b c d : ℝ) : ℝ := a * d - b * c

-- Define the solution set
def solution_set : Set ℝ := {x | -1 < x ∧ x < 3}

-- Theorem statement
theorem determinant_inequality_solution :
  {x : ℝ | det 1 2 x (x^2) < 3} = solution_set :=
sorry

end NUMINAMATH_CALUDE_determinant_inequality_solution_l3416_341607


namespace NUMINAMATH_CALUDE_local_max_implies_c_eq_six_l3416_341611

/-- Given a function f(x) = x(x-c)² where c is a constant, 
    if f has a local maximum at x = 2, then c = 6 -/
theorem local_max_implies_c_eq_six (c : ℝ) : 
  let f : ℝ → ℝ := λ x => x * (x - c)^2
  (∃ δ > 0, ∀ x ∈ Set.Ioo (2 - δ) (2 + δ), f x ≤ f 2) →
  c = 6 := by
  sorry

end NUMINAMATH_CALUDE_local_max_implies_c_eq_six_l3416_341611


namespace NUMINAMATH_CALUDE_g_forms_l3416_341620

-- Define the function f
def f (x : ℝ) : ℝ := x^2

-- Define the property for g
def g_property (g : ℝ → ℝ) : Prop :=
  ∀ x, f (g x) = 9*x^2 - 6*x + 1

-- Theorem statement
theorem g_forms {g : ℝ → ℝ} (h : g_property g) :
  (∀ x, g x = 3*x - 1) ∨ (∀ x, g x = -3*x + 1) := by
  sorry

end NUMINAMATH_CALUDE_g_forms_l3416_341620


namespace NUMINAMATH_CALUDE_book_arrangement_l3416_341625

theorem book_arrangement (n m : ℕ) (h : n + m = 8) :
  Nat.choose 8 n = 56 :=
sorry

end NUMINAMATH_CALUDE_book_arrangement_l3416_341625


namespace NUMINAMATH_CALUDE_product_remainder_mod_five_l3416_341658

theorem product_remainder_mod_five : ∃ k : ℕ, 114 * 232 * 454^2 * 678 = 5 * k + 4 := by
  sorry

end NUMINAMATH_CALUDE_product_remainder_mod_five_l3416_341658


namespace NUMINAMATH_CALUDE_mod_twelve_equiv_nine_l3416_341638

theorem mod_twelve_equiv_nine : 
  ∃! n : ℤ, 0 ≤ n ∧ n ≤ 11 ∧ n ≡ -2187 [ZMOD 12] ∧ n = 9 := by
  sorry

end NUMINAMATH_CALUDE_mod_twelve_equiv_nine_l3416_341638


namespace NUMINAMATH_CALUDE_equation_and_expression_proof_l3416_341609

theorem equation_and_expression_proof :
  (∃ x1 x2 : ℝ, x1 = 2 + Real.sqrt 7 ∧ x2 = 2 - Real.sqrt 7 ∧
    x1^2 - 4*x1 - 3 = 0 ∧ x2^2 - 4*x2 - 3 = 0) ∧
  ((-1)^2 + 2 * Real.sin (π/3) - Real.tan (π/4) = Real.sqrt 3) :=
by sorry

end NUMINAMATH_CALUDE_equation_and_expression_proof_l3416_341609


namespace NUMINAMATH_CALUDE_secret_organization_membership_l3416_341603

theorem secret_organization_membership (total_cents : ℕ) (max_members : ℕ) : 
  total_cents = 300737 ∧ max_members = 500 →
  ∃! (members : ℕ) (fee_cents : ℕ),
    members ≤ max_members ∧
    members * fee_cents = total_cents ∧
    members = 311 ∧
    fee_cents = 967 := by
  sorry

end NUMINAMATH_CALUDE_secret_organization_membership_l3416_341603


namespace NUMINAMATH_CALUDE_sally_eggs_l3416_341621

-- Define what a dozen is
def dozen : ℕ := 12

-- Define the number of dozens Sally bought
def dozens_bought : ℕ := 4

-- Theorem: Sally bought 48 eggs
theorem sally_eggs : dozens_bought * dozen = 48 := by
  sorry

end NUMINAMATH_CALUDE_sally_eggs_l3416_341621


namespace NUMINAMATH_CALUDE_number_exceeding_twelve_percent_l3416_341683

theorem number_exceeding_twelve_percent : ∃ x : ℝ, x = 0.12 * x + 52.8 ∧ x = 60 := by
  sorry

end NUMINAMATH_CALUDE_number_exceeding_twelve_percent_l3416_341683


namespace NUMINAMATH_CALUDE_sum_remainder_mod_17_l3416_341670

theorem sum_remainder_mod_17 : (82 + 83 + 84 + 85 + 86 + 87 + 88 + 89) % 17 = 11 := by
  sorry

end NUMINAMATH_CALUDE_sum_remainder_mod_17_l3416_341670


namespace NUMINAMATH_CALUDE_final_result_l3416_341660

/-- The number of different five-digit even numbers that can be formed using the digits 0, 1, 2, 3, and 4 -/
def even_numbers : ℕ := 60

/-- The number of different five-digit numbers that can be formed using the digits 1, 2, 3, 4, and 5 such that 2 and 3 are not adjacent -/
def non_adjacent_23 : ℕ := 72

/-- The number of different five-digit numbers that can be formed using the digits 1, 2, 3, 4, and 5 such that the digits 1, 2, and 3 must be arranged in descending order -/
def descending_123 : ℕ := 20

/-- The final result is the sum of the three subproblems -/
theorem final_result : even_numbers + non_adjacent_23 + descending_123 = 152 := by
  sorry

end NUMINAMATH_CALUDE_final_result_l3416_341660


namespace NUMINAMATH_CALUDE_inequality_preservation_l3416_341681

theorem inequality_preservation (a b c : ℝ) (h : a > b) : a + c^2 > b + c^2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_preservation_l3416_341681


namespace NUMINAMATH_CALUDE_equation_solutions_count_l3416_341675

theorem equation_solutions_count :
  ∃! (S : Set ℝ), (∀ x ∈ S, (x^2 - 7)^2 = 25) ∧ S.Finite ∧ S.ncard = 4 := by
  sorry

end NUMINAMATH_CALUDE_equation_solutions_count_l3416_341675


namespace NUMINAMATH_CALUDE_assignment_ways_theorem_l3416_341643

/-- The number of ways to assign 7 friends to 7 rooms with at most 3 friends per room -/
def assignment_ways : ℕ := 17640

/-- The number of rooms in the inn -/
def num_rooms : ℕ := 7

/-- The number of friends arriving -/
def num_friends : ℕ := 7

/-- The maximum number of friends allowed per room -/
def max_per_room : ℕ := 3

/-- Theorem stating that the number of ways to assign 7 friends to 7 rooms,
    with at most 3 friends per room, is equal to 17640 -/
theorem assignment_ways_theorem :
  ∃ (ways : ℕ → ℕ → ℕ → ℕ),
    ways num_rooms num_friends max_per_room = assignment_ways :=
by sorry

end NUMINAMATH_CALUDE_assignment_ways_theorem_l3416_341643


namespace NUMINAMATH_CALUDE_prime_diff_perfect_square_pairs_l3416_341637

theorem prime_diff_perfect_square_pairs (m n : ℕ+) (p : ℕ) :
  p.Prime →
  m - n = p →
  ∃ k : ℕ, m * n = k^2 →
  p % 2 = 1 ∧ m = ((p + 1)^2 / 4 : ℕ) ∧ n = ((p - 1)^2 / 4 : ℕ) := by
  sorry

end NUMINAMATH_CALUDE_prime_diff_perfect_square_pairs_l3416_341637


namespace NUMINAMATH_CALUDE_sum_of_fraction_parts_l3416_341673

def repeating_decimal : ℚ := 2.5252525

theorem sum_of_fraction_parts : ∃ (n d : ℕ), 
  repeating_decimal = n / d ∧ 
  Nat.gcd n d = 1 ∧ 
  n + d = 349 := by sorry

end NUMINAMATH_CALUDE_sum_of_fraction_parts_l3416_341673


namespace NUMINAMATH_CALUDE_correct_contribution_l3416_341697

/-- The cost of the project in billions of dollars -/
def project_cost : ℝ := 25

/-- The number of participants in millions -/
def num_participants : ℝ := 300

/-- The contribution required from each participant -/
def individual_contribution : ℝ := 83

/-- Theorem stating that the individual contribution is correct given the project cost and number of participants -/
theorem correct_contribution : 
  (project_cost * 1000) / num_participants = individual_contribution := by
  sorry

end NUMINAMATH_CALUDE_correct_contribution_l3416_341697


namespace NUMINAMATH_CALUDE_sum_two_condition_l3416_341684

theorem sum_two_condition (a b : ℝ) :
  (a = 1 ∧ b = 1 → a + b = 2) ∧
  (∃ a b : ℝ, a + b = 2 ∧ ¬(a = 1 ∧ b = 1)) :=
by sorry

end NUMINAMATH_CALUDE_sum_two_condition_l3416_341684


namespace NUMINAMATH_CALUDE_points_per_correct_answer_l3416_341632

theorem points_per_correct_answer 
  (total_questions : ℕ) 
  (correct_answers : ℕ) 
  (final_score : ℚ) 
  (incorrect_penalty : ℚ) 
  (h1 : total_questions = 120)
  (h2 : correct_answers = 104)
  (h3 : final_score = 100)
  (h4 : incorrect_penalty = -1/4)
  (h5 : correct_answers ≤ total_questions) :
  ∃ (points_per_correct : ℚ), 
    points_per_correct * correct_answers + 
    incorrect_penalty * (total_questions - correct_answers) = final_score ∧
    points_per_correct = 1 :=
by sorry

end NUMINAMATH_CALUDE_points_per_correct_answer_l3416_341632


namespace NUMINAMATH_CALUDE_series_sum_equals_three_fourths_l3416_341600

/-- The sum of the series 1/(n(n+2)) from n=1 to infinity equals 3/4 -/
theorem series_sum_equals_three_fourths :
  ∑' n, (1 : ℝ) / (n * (n + 2)) = 3/4 := by sorry

end NUMINAMATH_CALUDE_series_sum_equals_three_fourths_l3416_341600


namespace NUMINAMATH_CALUDE_prime_power_difference_l3416_341674

theorem prime_power_difference (n : ℕ) (p : ℕ) (k : ℕ) 
  (h1 : n > 0) 
  (h2 : Nat.Prime p) 
  (h3 : 3^n - 2^n = p^k) : 
  Nat.Prime n := by
sorry

end NUMINAMATH_CALUDE_prime_power_difference_l3416_341674


namespace NUMINAMATH_CALUDE_sum_of_solutions_l3416_341604

theorem sum_of_solutions (a₁ a₂ a₃ b₁ b₂ b₃ c₁ c₂ c₃ a b c : ℝ) 
  (eq1 : a₁ * (b₂ * c₃ - b₃ * c₂) - a₂ * (b₁ * c₃ - b₃ * c₁) + a₃ * (b₁ * c₂ - b₂ * c₁) = 9)
  (eq2 : a * (b₂ * c₃ - b₃ * c₂) - a₂ * (b * c₃ - b₃ * c) + a₃ * (b * c₂ - b₂ * c) = 17)
  (eq3 : a₁ * (b * c₃ - b₃ * c) - a * (b₁ * c₃ - b₃ * c₁) + a₃ * (b₁ * c - b * c₁) = -8)
  (eq4 : a₁ * (b₂ * c - b * c₂) - a₂ * (b₁ * c - b * c₁) + a * (b₁ * c₂ - b₂ * c₁) = 7)
  (sys1 : a₁ * x + a₂ * y + a₃ * z = a)
  (sys2 : b₁ * x + b₂ * y + b₃ * z = b)
  (sys3 : c₁ * x + c₂ * y + c₃ * z = c) :
  x + y + z = 16/9 := by
sorry

end NUMINAMATH_CALUDE_sum_of_solutions_l3416_341604


namespace NUMINAMATH_CALUDE_johns_speed_l3416_341651

/-- Prove that John's speed during his final push was 4.2 m/s given the race conditions --/
theorem johns_speed (initial_gap : ℝ) (steve_speed : ℝ) (final_gap : ℝ) (push_duration : ℝ) : 
  initial_gap = 14 →
  steve_speed = 3.7 →
  final_gap = 2 →
  push_duration = 32 →
  (initial_gap + final_gap) / push_duration + steve_speed = 4.2 := by
sorry

end NUMINAMATH_CALUDE_johns_speed_l3416_341651


namespace NUMINAMATH_CALUDE_max_large_chips_l3416_341619

theorem max_large_chips (total : ℕ) (small large : ℕ → ℕ) (composite : ℕ → ℕ) : 
  total = 72 →
  (∀ n, total = small n + large n) →
  (∀ n, small n = large n + composite n) →
  (∀ n, composite n ≥ 4) →
  (∃ max_large : ℕ, ∀ n, large n ≤ max_large ∧ (∃ m, large m = max_large)) →
  (∃ max_large : ℕ, max_large = 34 ∧ ∀ n, large n ≤ max_large) :=
by sorry

end NUMINAMATH_CALUDE_max_large_chips_l3416_341619


namespace NUMINAMATH_CALUDE_equilateral_triangle_l3416_341695

theorem equilateral_triangle (a b c : ℝ) 
  (h1 : a + b - c = 2) 
  (h2 : 2 * a * b - c^2 = 4) : 
  a = 2 ∧ b = 2 ∧ c = 2 := by
  sorry

end NUMINAMATH_CALUDE_equilateral_triangle_l3416_341695


namespace NUMINAMATH_CALUDE_unique_solution_is_four_l3416_341694

/-- Function that returns the product of digits of a positive integer -/
def digit_product (n : ℕ+) : ℕ :=
  sorry

/-- Theorem stating that 4 is the only positive integer solution to n^2 - 17n + 56 = a(n) -/
theorem unique_solution_is_four :
  ∃! (n : ℕ+), n^2 - 17*n + 56 = digit_product n :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_is_four_l3416_341694


namespace NUMINAMATH_CALUDE_quadratic_roots_integer_P_l3416_341641

theorem quadratic_roots_integer_P (P : ℤ) 
  (h1 : 5 < P) (h2 : P < 20) 
  (h3 : ∃ x y : ℤ, x^2 - 2*(2*P - 3)*x + 4*P^2 - 14*P + 8 = 0 ∧ 
                   y^2 - 2*(2*P - 3)*y + 4*P^2 - 14*P + 8 = 0 ∧ 
                   x ≠ y) : 
  P = 12 := by sorry

end NUMINAMATH_CALUDE_quadratic_roots_integer_P_l3416_341641


namespace NUMINAMATH_CALUDE_divisible_by_six_l3416_341652

theorem divisible_by_six (m : ℕ) : ∃ k : ℤ, (m : ℤ)^3 + 11 * m = 6 * k := by
  sorry

end NUMINAMATH_CALUDE_divisible_by_six_l3416_341652


namespace NUMINAMATH_CALUDE_smallest_valid_coloring_l3416_341672

/-- A coloring function that assigns a color (represented by a natural number) to each integer in the range [2, 31] -/
def Coloring := Fin 30 → Nat

/-- Predicate to check if a coloring is valid according to the given conditions -/
def IsValidColoring (c : Coloring) : Prop :=
  ∀ m n, 2 ≤ m ∧ m ≤ 31 ∧ 2 ≤ n ∧ n ≤ 31 →
    m ≠ n → m % n = 0 → c (m - 2) ≠ c (n - 2)

/-- The existence of a valid coloring using k colors -/
def ExistsValidColoring (k : Nat) : Prop :=
  ∃ c : Coloring, IsValidColoring c ∧ ∀ i, c i < k

/-- The main theorem: The smallest number of colors needed is 4 -/
theorem smallest_valid_coloring : (∃ k, ExistsValidColoring k ∧ ∀ j, j < k → ¬ExistsValidColoring j) ∧
                                   ExistsValidColoring 4 :=
sorry

end NUMINAMATH_CALUDE_smallest_valid_coloring_l3416_341672


namespace NUMINAMATH_CALUDE_membership_fee_increase_l3416_341626

/-- Proves that the yearly increase in membership fee is $10 given the initial and final fees -/
theorem membership_fee_increase
  (initial_fee : ℕ)
  (final_fee : ℕ)
  (initial_year : ℕ)
  (final_year : ℕ)
  (h1 : initial_fee = 80)
  (h2 : final_fee = 130)
  (h3 : initial_year = 1)
  (h4 : final_year = 6)
  (h5 : final_fee = initial_fee + (final_year - initial_year) * (yearly_increase : ℕ)) :
  yearly_increase = 10 := by
  sorry

end NUMINAMATH_CALUDE_membership_fee_increase_l3416_341626


namespace NUMINAMATH_CALUDE_sum_of_edge_lengths_specific_prism_l3416_341662

/-- Regular hexagonal prism with given base side length and height -/
structure RegularHexagonalPrism where
  base_side_length : ℝ
  height : ℝ

/-- Calculate the sum of the lengths of all edges of a regular hexagonal prism -/
def sum_of_edge_lengths (prism : RegularHexagonalPrism) : ℝ :=
  12 * prism.base_side_length + 6 * prism.height

/-- Theorem: The sum of edge lengths for a regular hexagonal prism with base side 6 cm and height 11 cm is 138 cm -/
theorem sum_of_edge_lengths_specific_prism :
  let prism : RegularHexagonalPrism := ⟨6, 11⟩
  sum_of_edge_lengths prism = 138 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_edge_lengths_specific_prism_l3416_341662


namespace NUMINAMATH_CALUDE_right_triangle_area_l3416_341668

/-- A right triangle ABC in the xy-plane with specific properties -/
structure RightTriangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  is_right_angle_at_C : (A.1 - C.1) * (B.1 - C.1) + (A.2 - C.2) * (B.2 - C.2) = 0
  hypotenuse_length : (A.1 - B.1)^2 + (A.2 - B.2)^2 = 900
  median_A_on_y_eq_x : A.1 = A.2
  median_B_on_y_eq_x_plus_1 : B.2 = B.1 + 1

/-- The area of the right triangle ABC is 448 -/
theorem right_triangle_area (t : RightTriangle) : 
  (1/2) * abs ((t.A.1 * (t.B.2 - t.C.2) + t.B.1 * (t.C.2 - t.A.2) + t.C.1 * (t.A.2 - t.B.2))) = 448 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_area_l3416_341668


namespace NUMINAMATH_CALUDE_trigonometric_system_solution_l3416_341646

theorem trigonometric_system_solution (x y : ℝ) :
  (Real.sin x * Real.sin y = 0.75) →
  (Real.tan x * Real.tan y = 3) →
  ∃ (k n : ℤ), 
    (x = π/3 + π*(k + n : ℝ) ∨ x = -π/3 + π*(k + n : ℝ)) ∧
    (y = π/3 + π*(n - k : ℝ) ∨ y = -π/3 + π*(n - k : ℝ)) := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_system_solution_l3416_341646


namespace NUMINAMATH_CALUDE_complex_modulus_problem_l3416_341614

theorem complex_modulus_problem (z : ℂ) (h : (1 - Complex.I) * z = 4 * Complex.I) : 
  Complex.abs z = 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_problem_l3416_341614


namespace NUMINAMATH_CALUDE_inequality_system_solution_l3416_341682

theorem inequality_system_solution :
  ∀ x : ℝ, (x + 2 > -1 ∧ x - 5 < 3 * (x - 1)) ↔ x > -1 :=
by sorry

end NUMINAMATH_CALUDE_inequality_system_solution_l3416_341682


namespace NUMINAMATH_CALUDE_average_of_other_results_l3416_341608

theorem average_of_other_results
  (n₁ : ℕ) (n₂ : ℕ) (avg₁ : ℚ) (avg_all : ℚ)
  (h₁ : n₁ = 60)
  (h₂ : n₂ = 40)
  (h₃ : avg₁ = 40)
  (h₄ : avg_all = 48)
  : (n₁ * avg₁ + n₂ * ((n₁ + n₂) * avg_all - n₁ * avg₁) / n₂) / (n₁ + n₂) = avg_all ∧
    ((n₁ + n₂) * avg_all - n₁ * avg₁) / n₂ = 60 :=
by sorry

end NUMINAMATH_CALUDE_average_of_other_results_l3416_341608


namespace NUMINAMATH_CALUDE_original_number_proof_l3416_341669

theorem original_number_proof (n : ℕ) (k : ℕ) : 
  (∃ m : ℕ, n + k = 5 * m) ∧ 
  (n + k = 2500) ∧ 
  (∀ j : ℕ, j < k → ¬∃ m : ℕ, n + j = 5 * m) →
  n = 2500 :=
by sorry

end NUMINAMATH_CALUDE_original_number_proof_l3416_341669


namespace NUMINAMATH_CALUDE_cos_beta_eq_four_fifths_l3416_341648

-- Define the quadrilateral EFGH
structure Quadrilateral :=
  (E F G H : ℝ × ℝ)

-- Define the properties of the quadrilateral
def is_convex (q : Quadrilateral) : Prop := sorry

def angle_E_eq_angle_G (q : Quadrilateral) (β : ℝ) : Prop := sorry

def side_EF_eq_side_GH (q : Quadrilateral) : Prop := sorry

def side_EH_ne_side_FG (q : Quadrilateral) : Prop := sorry

def perimeter (q : Quadrilateral) : ℝ := sorry

-- Main theorem
theorem cos_beta_eq_four_fifths (q : Quadrilateral) (β : ℝ) :
  is_convex q →
  angle_E_eq_angle_G q β →
  side_EF_eq_side_GH q →
  side_EH_ne_side_FG q →
  perimeter q = 720 →
  Real.cos β = 4/5 := by sorry

end NUMINAMATH_CALUDE_cos_beta_eq_four_fifths_l3416_341648


namespace NUMINAMATH_CALUDE_all_contradictions_valid_l3416_341629

/-- A type representing the different kinds of contradictions in a proof by contradiction -/
inductive ContradictionType
  | KnownFact
  | Assumption
  | DefinitionTheoremAxiomLaw
  | Fact

/-- Definition of a valid contradiction in a proof by contradiction -/
def is_valid_contradiction (c : ContradictionType) : Prop :=
  match c with
  | ContradictionType.KnownFact => True
  | ContradictionType.Assumption => True
  | ContradictionType.DefinitionTheoremAxiomLaw => True
  | ContradictionType.Fact => True

/-- Theorem stating that all types of contradictions are valid in a proof by contradiction -/
theorem all_contradictions_valid :
  ∀ (c : ContradictionType), is_valid_contradiction c :=
by sorry

end NUMINAMATH_CALUDE_all_contradictions_valid_l3416_341629


namespace NUMINAMATH_CALUDE_haji_mother_sales_l3416_341649

theorem haji_mother_sales (tough_week_sales : ℕ) (good_weeks : ℕ) (tough_weeks : ℕ)
  (h1 : tough_week_sales = 800)
  (h2 : tough_week_sales * 2 = tough_week_sales + tough_week_sales)
  (h3 : good_weeks = 5)
  (h4 : tough_weeks = 3) :
  tough_week_sales * tough_weeks + (tough_week_sales * 2) * good_weeks = 10400 := by
  sorry

end NUMINAMATH_CALUDE_haji_mother_sales_l3416_341649


namespace NUMINAMATH_CALUDE_arithmetic_mean_problem_l3416_341690

theorem arithmetic_mean_problem (x : ℚ) : 
  ((x + 10) + 20 + 3*x + 18 + (3*x + 6)) / 5 = 32 → x = 106 / 7 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_mean_problem_l3416_341690


namespace NUMINAMATH_CALUDE_valerie_light_bulb_purchase_l3416_341617

/-- Valerie's light bulb purchase problem -/
theorem valerie_light_bulb_purchase (small_bulb_cost large_bulb_cost small_bulb_count large_bulb_count leftover_money : ℕ) :
  small_bulb_cost = 8 →
  large_bulb_cost = 12 →
  small_bulb_count = 3 →
  large_bulb_count = 1 →
  leftover_money = 24 →
  small_bulb_cost * small_bulb_count + large_bulb_cost * large_bulb_count + leftover_money = 60 :=
by sorry

end NUMINAMATH_CALUDE_valerie_light_bulb_purchase_l3416_341617


namespace NUMINAMATH_CALUDE_arjun_has_largest_result_l3416_341633

def initial_number : ℕ := 15

def liam_result : ℕ := ((initial_number - 2) * 3) + 3

def maya_result : ℕ := ((initial_number * 3) - 4) + 5

def arjun_result : ℕ := ((initial_number - 3) + 4) * 3

theorem arjun_has_largest_result :
  arjun_result > liam_result ∧ arjun_result > maya_result :=
by sorry

end NUMINAMATH_CALUDE_arjun_has_largest_result_l3416_341633


namespace NUMINAMATH_CALUDE_magnitude_of_b_is_one_l3416_341627

/-- Given two vectors a and b in ℝ², prove that the magnitude of b is 1 -/
theorem magnitude_of_b_is_one (a b : ℝ × ℝ) : 
  (Real.cos (60 * π / 180) = a.fst * b.fst + a.snd * b.snd) →  -- angle between a and b is 60°
  (a.fst^2 + a.snd^2 = 1) →  -- |a| = 1
  ((2*a.fst - b.fst)^2 + (2*a.snd - b.snd)^2 = 3) →  -- |2a - b| = √3
  (b.fst^2 + b.snd^2 = 1) :=  -- |b| = 1
by sorry

end NUMINAMATH_CALUDE_magnitude_of_b_is_one_l3416_341627


namespace NUMINAMATH_CALUDE_log_equation_solution_l3416_341645

-- Define the logarithm function (base 10)
noncomputable def lg (x : ℝ) : ℝ := Real.log x / Real.log 10

-- State the theorem
theorem log_equation_solution (m n b : ℝ) (h : lg m = b - lg n) : m = 10^b / n := by
  sorry

end NUMINAMATH_CALUDE_log_equation_solution_l3416_341645


namespace NUMINAMATH_CALUDE_post_height_l3416_341613

/-- Calculates the height of a cylindrical post given the squirrel's travel conditions -/
theorem post_height (total_distance : ℝ) (post_circumference : ℝ) (height_per_circuit : ℝ) : 
  total_distance = 27 ∧ post_circumference = 3 ∧ height_per_circuit = 3 →
  (total_distance / post_circumference) * height_per_circuit = 27 := by
sorry

end NUMINAMATH_CALUDE_post_height_l3416_341613


namespace NUMINAMATH_CALUDE_mississippi_arrangements_l3416_341624

theorem mississippi_arrangements : 
  (11 : ℕ).factorial / ((4 : ℕ).factorial * (4 : ℕ).factorial * (2 : ℕ).factorial) = 34650 := by
  sorry

end NUMINAMATH_CALUDE_mississippi_arrangements_l3416_341624


namespace NUMINAMATH_CALUDE_campers_rowing_difference_l3416_341601

theorem campers_rowing_difference (morning_campers afternoon_campers evening_campers : ℕ) 
  (h1 : morning_campers = 44)
  (h2 : afternoon_campers = 39)
  (h3 : evening_campers = 31) :
  morning_campers - afternoon_campers = 5 := by
  sorry

end NUMINAMATH_CALUDE_campers_rowing_difference_l3416_341601


namespace NUMINAMATH_CALUDE_expression_simplification_and_evaluation_l3416_341615

theorem expression_simplification_and_evaluation :
  ∀ x : ℤ, -2 < x → x < 2 → x ≠ -1 → x ≠ 0 →
  (((x^2 - 1) / (x^2 + 2*x + 1)) / ((1 / (x + 1)) - 1) = (1 - x) / x) ∧
  (((x^2 - 1) / (x^2 + 2*x + 1)) / ((1 / (x + 1)) - 1) = 0) :=
by sorry

end NUMINAMATH_CALUDE_expression_simplification_and_evaluation_l3416_341615


namespace NUMINAMATH_CALUDE_smallest_perimeter_isosceles_triangle_l3416_341665

/-- Triangle with positive integer side lengths --/
structure IsoscelesTriangle where
  side : ℕ+
  base : ℕ+

/-- Point representing the intersection of angle bisectors --/
structure AngleBisectorIntersection where
  distance : ℕ+

/-- Theorem stating the smallest possible perimeter of the triangle --/
theorem smallest_perimeter_isosceles_triangle 
  (t : IsoscelesTriangle) 
  (j : AngleBisectorIntersection) 
  (h : j.distance = 10) : 
  2 * (t.side + t.base) ≥ 198 := by
  sorry

#check smallest_perimeter_isosceles_triangle

end NUMINAMATH_CALUDE_smallest_perimeter_isosceles_triangle_l3416_341665


namespace NUMINAMATH_CALUDE_line_not_in_first_quadrant_l3416_341630

/-- A line that does not pass through the first quadrant has a non-positive slope -/
def not_in_first_quadrant (t : ℝ) : Prop :=
  3 - 2 * t ≤ 0

/-- The range of t for which the line (2t-3)x + y + 6 = 0 does not pass through the first quadrant -/
def t_range : Set ℝ :=
  {t : ℝ | t ≥ 3/2}

theorem line_not_in_first_quadrant :
  ∀ t : ℝ, not_in_first_quadrant t ↔ t ∈ t_range :=
sorry

end NUMINAMATH_CALUDE_line_not_in_first_quadrant_l3416_341630


namespace NUMINAMATH_CALUDE_range_of_a_l3416_341656

def f (x a : ℝ) : ℝ := |x - 1| + |x - a|

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, f x a ≥ 2) → a ∈ Set.Ici 3 ∪ Set.Iic (-1) :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l3416_341656


namespace NUMINAMATH_CALUDE_consecutive_integers_square_difference_l3416_341610

theorem consecutive_integers_square_difference :
  ∃ n : ℕ, 
    (n > 0) ∧ 
    (n + (n + 1) + (n + 2) < 150) ∧ 
    ((n + 2)^2 - n^2 = 144) :=
by sorry

end NUMINAMATH_CALUDE_consecutive_integers_square_difference_l3416_341610


namespace NUMINAMATH_CALUDE_number_plus_sqrt_equals_24_l3416_341692

theorem number_plus_sqrt_equals_24 : ∃ x : ℝ, x + Real.sqrt (-4 + 6 * 4 * 3) = 24 := by
  sorry

end NUMINAMATH_CALUDE_number_plus_sqrt_equals_24_l3416_341692


namespace NUMINAMATH_CALUDE_max_area_prime_sides_l3416_341622

/-- A prime number is a natural number greater than 1 that has no positive divisors other than 1 and itself. -/
def isPrime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m > 0 → m < n → n % m ≠ 0

/-- The perimeter of the rectangle is 40 meters. -/
def perimeter : ℕ := 40

/-- The theorem stating that the maximum area of a rectangular enclosure with prime side lengths and a perimeter of 40 meters is 91 square meters. -/
theorem max_area_prime_sides : 
  ∀ l w : ℕ, 
    isPrime l → 
    isPrime w → 
    l + w = perimeter / 2 → 
    l * w ≤ 91 :=
sorry

end NUMINAMATH_CALUDE_max_area_prime_sides_l3416_341622


namespace NUMINAMATH_CALUDE_increasing_f_range_of_a_l3416_341623

-- Define the piecewise function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then (6 - a) * x - 4 * a
  else Real.log x / Real.log a

-- Theorem statement
theorem increasing_f_range_of_a :
  ∀ a : ℝ, (∀ x y : ℝ, x < y → f a x < f a y) →
  a ∈ Set.Icc (6/5) 6 ∧ a ≠ 6 :=
sorry

end NUMINAMATH_CALUDE_increasing_f_range_of_a_l3416_341623


namespace NUMINAMATH_CALUDE_initial_deposit_is_one_l3416_341679

def initial_amount : ℕ := 100
def weeks : ℕ := 52
def final_total : ℕ := 1478

def arithmetic_sum (a₁ : ℚ) (n : ℕ) : ℚ :=
  (n : ℚ) / 2 * (2 * a₁ + (n - 1 : ℚ))

theorem initial_deposit_is_one :
  ∃ (x : ℚ), 
    arithmetic_sum x weeks + initial_amount = final_total ∧ 
    x = 1 := by sorry

end NUMINAMATH_CALUDE_initial_deposit_is_one_l3416_341679


namespace NUMINAMATH_CALUDE_lake_circumference_diameter_ratio_l3416_341642

/-- For a circular lake with given diameter and circumference, 
    prove that the ratio of circumference to diameter is 3.14 -/
theorem lake_circumference_diameter_ratio :
  ∀ (diameter circumference : ℝ),
    diameter = 100 →
    circumference = 314 →
    circumference / diameter = 3.14 := by
  sorry

end NUMINAMATH_CALUDE_lake_circumference_diameter_ratio_l3416_341642


namespace NUMINAMATH_CALUDE_price_of_car_is_five_l3416_341606

/-- Calculates the price of one little car given the total earnings, cost of Legos, and number of cars sold. -/
def price_of_one_car (total_earnings : ℕ) (legos_cost : ℕ) (num_cars : ℕ) : ℚ :=
  (total_earnings - legos_cost : ℚ) / num_cars

/-- Theorem stating that the price of one little car is $5 given the problem conditions. -/
theorem price_of_car_is_five :
  price_of_one_car 45 30 3 = 5 := by
  sorry

#eval price_of_one_car 45 30 3

end NUMINAMATH_CALUDE_price_of_car_is_five_l3416_341606


namespace NUMINAMATH_CALUDE_probability_five_or_king_l3416_341678

/-- A standard deck of cards -/
structure Deck :=
  (cards : Nat)
  (ranks : Nat)
  (suits : Nat)
  (unique_combinations : Bool)

/-- The probability of drawing a specific set of cards from a deck -/
def probability (d : Deck) (favorable_outcomes : Nat) : ℚ :=
  favorable_outcomes / d.cards

/-- Theorem: The probability of drawing a 5 or King from a standard deck is 2/13 -/
theorem probability_five_or_king (d : Deck) 
  (h1 : d.cards = 52)
  (h2 : d.ranks = 13)
  (h3 : d.suits = 4)
  (h4 : d.unique_combinations = true) :
  probability d 8 = 2 / 13 := by
  sorry

end NUMINAMATH_CALUDE_probability_five_or_king_l3416_341678


namespace NUMINAMATH_CALUDE_maya_car_arrangement_l3416_341664

theorem maya_car_arrangement (current_cars : ℕ) (cars_per_row : ℕ) (additional_cars : ℕ) : 
  current_cars = 29 →
  cars_per_row = 7 →
  (current_cars + additional_cars) % cars_per_row = 0 →
  ∀ n : ℕ, n < additional_cars → (current_cars + n) % cars_per_row ≠ 0 →
  additional_cars = 6 := by
sorry

end NUMINAMATH_CALUDE_maya_car_arrangement_l3416_341664


namespace NUMINAMATH_CALUDE_shaded_to_unshaded_ratio_l3416_341663

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a square -/
structure Square where
  s : Point
  t : Point
  u : Point
  v : Point

/-- Represents a triangle -/
structure Triangle where
  a : Point
  b : Point
  c : Point

/-- Calculate the area of a triangle -/
def triangleArea (t : Triangle) : ℝ := sorry

/-- Calculate the area of a square -/
def squareArea (s : Square) : ℝ := sorry

/-- Check if a point is the midpoint of two other points -/
def isMidpoint (m : Point) (a : Point) (b : Point) : Prop := sorry

/-- Check if two line segments are parallel -/
def isParallel (a : Point) (b : Point) (c : Point) (d : Point) : Prop := sorry

/-- Check if two line segments are equal in length -/
def segmentEqual (a : Point) (b : Point) (c : Point) (d : Point) : Prop := sorry

theorem shaded_to_unshaded_ratio 
  (s : Square) 
  (q p r o : Point) 
  (t1 t2 t3 : Triangle) :
  isMidpoint q s.s s.t →
  isMidpoint p s.u s.v →
  segmentEqual p r q r →
  isParallel s.v q p r →
  t1 = Triangle.mk q o r →
  t2 = Triangle.mk p o r →
  t3 = Triangle.mk q p s.v →
  (triangleArea t1 + triangleArea t2 + triangleArea t3) / 
  (squareArea s - (triangleArea t1 + triangleArea t2 + triangleArea t3)) = 3 / 5 := by
  sorry

end NUMINAMATH_CALUDE_shaded_to_unshaded_ratio_l3416_341663


namespace NUMINAMATH_CALUDE_positive_sum_from_positive_difference_l3416_341659

theorem positive_sum_from_positive_difference (a b : ℝ) : a - |b| > 0 → b + a > 0 := by
  sorry

end NUMINAMATH_CALUDE_positive_sum_from_positive_difference_l3416_341659


namespace NUMINAMATH_CALUDE_collinear_points_sum_l3416_341618

/-- Three points in ℝ³ are collinear if they lie on the same line. -/
def collinear (A B C : ℝ × ℝ × ℝ) : Prop :=
  ∃ t : ℝ, C - A = t • (B - A)

/-- The theorem states that if A(1,3,-2), B(2,5,1), and C(p,7,q-2) are collinear in ℝ³, 
    then p+q = 9. -/
theorem collinear_points_sum (p q : ℝ) : 
  let A : ℝ × ℝ × ℝ := (1, 3, -2)
  let B : ℝ × ℝ × ℝ := (2, 5, 1)
  let C : ℝ × ℝ × ℝ := (p, 7, q-2)
  collinear A B C → p + q = 9 := by
  sorry

end NUMINAMATH_CALUDE_collinear_points_sum_l3416_341618


namespace NUMINAMATH_CALUDE_no_valid_ab_pairs_l3416_341612

theorem no_valid_ab_pairs : 
  ¬∃ (a b : ℝ), ∃ (x y : ℤ), 
    (3 * a * x + 7 * b * y = 3) ∧ 
    (x^2 + y^2 = 85) ∧ 
    (x % 5 = 0 ∨ y % 5 = 0) :=
sorry

end NUMINAMATH_CALUDE_no_valid_ab_pairs_l3416_341612


namespace NUMINAMATH_CALUDE_no_member_divisible_by_4_or_5_l3416_341640

def T : Set Int := {x | ∃ n : Int, x = (n-1)^2 + n^2 + (n+1)^2 + (n+2)^2}

theorem no_member_divisible_by_4_or_5 : ∀ x ∈ T, ¬(x % 4 = 0 ∨ x % 5 = 0) := by
  sorry

end NUMINAMATH_CALUDE_no_member_divisible_by_4_or_5_l3416_341640


namespace NUMINAMATH_CALUDE_beckys_necklaces_l3416_341631

theorem beckys_necklaces (initial_count : ℕ) (broken : ℕ) (new_purchases : ℕ) (final_count : ℕ)
  (h1 : initial_count = 50)
  (h2 : broken = 3)
  (h3 : new_purchases = 5)
  (h4 : final_count = 37) :
  initial_count - broken + new_purchases - final_count = 15 := by
  sorry

end NUMINAMATH_CALUDE_beckys_necklaces_l3416_341631


namespace NUMINAMATH_CALUDE_zachary_pushups_l3416_341653

theorem zachary_pushups (david_pushups : ℕ) (difference : ℕ) (zachary_pushups : ℕ) :
  david_pushups = 37 →
  david_pushups = zachary_pushups + difference →
  difference = 30 →
  zachary_pushups = 7 :=
by
  sorry

end NUMINAMATH_CALUDE_zachary_pushups_l3416_341653


namespace NUMINAMATH_CALUDE_slower_speed_percentage_l3416_341650

theorem slower_speed_percentage (usual_time slower_time : ℝ) 
  (h1 : usual_time = 8)
  (h2 : slower_time = usual_time + 24) :
  (usual_time / slower_time) * 100 = 25 := by
sorry

end NUMINAMATH_CALUDE_slower_speed_percentage_l3416_341650


namespace NUMINAMATH_CALUDE_quadratic_roots_complex_and_distinct_l3416_341689

-- Define the coefficients of the quadratic equation
def a : ℂ := 1
def b : ℂ := 2 + 2*Complex.I
def c : ℂ := 5

-- Define the discriminant
def discriminant : ℂ := b^2 - 4*a*c

-- Theorem statement
theorem quadratic_roots_complex_and_distinct :
  ¬(discriminant = 0) ∧ ¬(discriminant.im = 0) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_complex_and_distinct_l3416_341689


namespace NUMINAMATH_CALUDE_petya_can_prevent_natural_sum_l3416_341688

/-- Represents a player's turn in the game -/
structure Turn where
  player : Bool  -- true for Petya, false for Vasya
  fractions : List Nat  -- List of denominators of fractions written

/-- The state of the game board -/
structure GameState where
  turns : List Turn
  sum : Rat

/-- Vasya's strategy to choose fractions -/
def vasyaStrategy (state : GameState) : List Nat := sorry

/-- Petya's strategy to choose a fraction -/
def petyaStrategy (state : GameState) : Nat := sorry

/-- Checks if the sum of fractions is a natural number -/
def isNaturalSum (sum : Rat) : Bool := sorry

/-- Simulates the game for a given number of rounds -/
def playGame (rounds : Nat) : GameState := sorry

/-- Theorem stating that Petya can prevent Vasya from achieving a natural number sum -/
theorem petya_can_prevent_natural_sum :
  ∀ (rounds : Nat), ¬(isNaturalSum (playGame rounds).sum) := by sorry

end NUMINAMATH_CALUDE_petya_can_prevent_natural_sum_l3416_341688


namespace NUMINAMATH_CALUDE_S_finite_iff_power_of_two_l3416_341691

def S (k : ℕ) : Set (ℕ × ℕ × ℕ) :=
  {t | t.1 % 2 = 1 ∧ 
       Nat.gcd t.2.1 t.2.2 = 1 ∧ 
       t.2.1 + t.2.2 = k ∧ 
       t.1 ∣ (t.2.1 ^ t.1 + t.2.2 ^ t.1)}

theorem S_finite_iff_power_of_two (k : ℕ) (h : k > 1) :
  Set.Finite (S k) ↔ ∃ α : ℕ, k = 2^α ∧ α > 0 :=
sorry

end NUMINAMATH_CALUDE_S_finite_iff_power_of_two_l3416_341691


namespace NUMINAMATH_CALUDE_unique_solution_sin_equation_l3416_341676

theorem unique_solution_sin_equation :
  ∃! x : ℝ, x = Real.sin x + 1993 := by sorry

end NUMINAMATH_CALUDE_unique_solution_sin_equation_l3416_341676


namespace NUMINAMATH_CALUDE_circle_radius_l3416_341655

theorem circle_radius (A C : ℝ) (h : A / C = 15) : 
  ∃ r : ℝ, r > 0 ∧ A = π * r^2 ∧ C = 2 * π * r ∧ r = 30 := by
sorry

end NUMINAMATH_CALUDE_circle_radius_l3416_341655


namespace NUMINAMATH_CALUDE_marble_jar_count_l3416_341657

theorem marble_jar_count : ∃ (total : ℕ), 
  (total / 2 : ℕ) + (total / 4 : ℕ) + 27 + 14 = total ∧ total = 164 := by
  sorry

end NUMINAMATH_CALUDE_marble_jar_count_l3416_341657


namespace NUMINAMATH_CALUDE_jogger_train_distance_l3416_341654

/-- Calculates the distance a jogger is ahead of a train's engine given their speeds and the time it takes for the train to pass the jogger. -/
theorem jogger_train_distance
  (jogger_speed : ℝ)
  (train_speed : ℝ)
  (train_length : ℝ)
  (passing_time : ℝ)
  (h1 : jogger_speed = 9 / 3.6)  -- Convert 9 km/hr to m/s
  (h2 : train_speed = 45 / 3.6)  -- Convert 45 km/hr to m/s
  (h3 : train_length = 120)
  (h4 : passing_time = 32) :
  train_speed * passing_time - jogger_speed * passing_time - train_length = 200 :=
by sorry

end NUMINAMATH_CALUDE_jogger_train_distance_l3416_341654


namespace NUMINAMATH_CALUDE_complex_subtraction_example_l3416_341680

theorem complex_subtraction_example : (6 - 2*I) - (3*I + 1) = 5 - 5*I := by
  sorry

end NUMINAMATH_CALUDE_complex_subtraction_example_l3416_341680


namespace NUMINAMATH_CALUDE_root_difference_implies_k_l3416_341698

theorem root_difference_implies_k (k : ℝ) : 
  (∀ r s : ℝ, r^2 + k*r + 6 = 0 ∧ s^2 + k*s + 6 = 0 → 
    ∃ r' s' : ℝ, r'^2 - k*r' + 6 = 0 ∧ s'^2 - k*s' + 6 = 0 ∧ 
    r' = r + 5 ∧ s' = s + 5) → 
  k = 5 := by
sorry

end NUMINAMATH_CALUDE_root_difference_implies_k_l3416_341698


namespace NUMINAMATH_CALUDE_probability_all_white_balls_l3416_341639

def white_balls : ℕ := 7
def black_balls : ℕ := 7
def total_balls : ℕ := white_balls + black_balls
def num_draws : ℕ := 6

theorem probability_all_white_balls :
  (white_balls : ℚ) / total_balls ^ num_draws = 1 / 64 := by sorry

end NUMINAMATH_CALUDE_probability_all_white_balls_l3416_341639


namespace NUMINAMATH_CALUDE_point_in_second_quadrant_l3416_341693

theorem point_in_second_quadrant (m : ℝ) : 
  let p : ℝ × ℝ := (-1, m^2 + 1)
  p.1 < 0 ∧ p.2 > 0 :=
by sorry

end NUMINAMATH_CALUDE_point_in_second_quadrant_l3416_341693


namespace NUMINAMATH_CALUDE_max_x_minus_y_l3416_341677

theorem max_x_minus_y (x y : ℝ) (h : x^2 + y^2 - 4*x - 2*y - 4 = 0) :
  ∃ (z : ℝ), z = x - y ∧ z ≤ 1 + 3 * Real.sqrt 2 ∧
  ∀ (w : ℝ), (∃ (a b : ℝ), w = a - b ∧ a^2 + b^2 - 4*a - 2*b - 4 = 0) →
  w ≤ 1 + 3 * Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_max_x_minus_y_l3416_341677


namespace NUMINAMATH_CALUDE_sequence_formula_l3416_341636

theorem sequence_formula (a : ℕ+ → ℝ) (S : ℕ+ → ℝ) 
  (h : ∀ n : ℕ+, S n = (1/3) * (a n - 1)) :
  ∀ n : ℕ+, a n = n + 1 := by sorry

end NUMINAMATH_CALUDE_sequence_formula_l3416_341636


namespace NUMINAMATH_CALUDE_triangle_area_in_rectangle_config_l3416_341666

/-- The area of a triangle in a specific geometric configuration -/
theorem triangle_area_in_rectangle_config : 
  ∀ (base height : ℝ),
  base = 16 ∧ 
  height = 12 * (18 / 39) →
  (1 / 2 : ℝ) * base * height = 1536 / 13 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_in_rectangle_config_l3416_341666


namespace NUMINAMATH_CALUDE_hall_length_proof_l3416_341634

/-- Proves that a hall with given dimensions and mat cost has a specific length -/
theorem hall_length_proof (width height mat_cost_per_sqm total_cost : ℝ) 
  (h_width : width = 15)
  (h_height : height = 5)
  (h_mat_cost : mat_cost_per_sqm = 40)
  (h_total_cost : total_cost = 38000) :
  ∃ (length : ℝ), 
    length = 32 ∧ 
    total_cost = mat_cost_per_sqm * (length * width + 2 * length * height + 2 * width * height) :=
by sorry

end NUMINAMATH_CALUDE_hall_length_proof_l3416_341634


namespace NUMINAMATH_CALUDE_no_call_days_l3416_341671

theorem no_call_days (total_days : ℕ) (call_period1 call_period2 call_period3 : ℕ) : 
  total_days = 365 ∧ call_period1 = 2 ∧ call_period2 = 5 ∧ call_period3 = 7 →
  total_days - (
    (total_days / call_period1 + total_days / call_period2 + total_days / call_period3) -
    (total_days / (Nat.lcm call_period1 call_period2) + 
     total_days / (Nat.lcm call_period1 call_period3) + 
     total_days / (Nat.lcm call_period2 call_period3)) +
    total_days / (Nat.lcm call_period1 (Nat.lcm call_period2 call_period3))
  ) = 125 := by
  sorry

end NUMINAMATH_CALUDE_no_call_days_l3416_341671
