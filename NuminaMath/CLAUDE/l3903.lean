import Mathlib

namespace NUMINAMATH_CALUDE_expression_equals_500_l3903_390386

theorem expression_equals_500 : 88 * 4 + 37 * 4 = 500 := by
  sorry

end NUMINAMATH_CALUDE_expression_equals_500_l3903_390386


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l3903_390384

-- Define the quadratic function
def f (x : ℝ) := x^2 - 6*x + 8

-- Define the solution set
def solution_set : Set ℝ := {x | x < 2 ∨ x > 4}

-- Theorem statement
theorem quadratic_inequality_solution :
  {x : ℝ | f x > 0} = solution_set :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l3903_390384


namespace NUMINAMATH_CALUDE_f_min_at_4_l3903_390368

/-- The quadratic function f(x) = x^2 - 8x + 15 -/
def f (x : ℝ) : ℝ := x^2 - 8*x + 15

/-- The theorem stating that f(x) attains its minimum at x = 4 -/
theorem f_min_at_4 :
  ∀ x : ℝ, f x ≥ f 4 :=
sorry

end NUMINAMATH_CALUDE_f_min_at_4_l3903_390368


namespace NUMINAMATH_CALUDE_gymnastics_competition_participants_l3903_390382

/-- Represents the structure of a gymnastics competition layout --/
structure GymnasticsCompetition where
  rows : ℕ
  columns : ℕ
  front_position : ℕ
  back_position : ℕ
  left_position : ℕ
  right_position : ℕ

/-- Calculates the total number of participants in the gymnastics competition --/
def total_participants (gc : GymnasticsCompetition) : ℕ :=
  gc.rows * gc.columns

/-- Theorem stating that the total number of participants is 425 --/
theorem gymnastics_competition_participants :
  ∀ (gc : GymnasticsCompetition),
    gc.front_position = 6 →
    gc.back_position = 12 →
    gc.left_position = 15 →
    gc.right_position = 11 →
    gc.columns = gc.front_position + gc.back_position - 1 →
    gc.rows = gc.left_position + gc.right_position - 1 →
    total_participants gc = 425 := by
  sorry

#check gymnastics_competition_participants

end NUMINAMATH_CALUDE_gymnastics_competition_participants_l3903_390382


namespace NUMINAMATH_CALUDE_least_six_digit_congruent_to_3_mod_17_l3903_390302

theorem least_six_digit_congruent_to_3_mod_17 : ∃ (n : ℕ), 
  (n ≥ 100000 ∧ n < 1000000) ∧ 
  n % 17 = 3 ∧
  ∀ (m : ℕ), (m ≥ 100000 ∧ m < 1000000 ∧ m % 17 = 3) → n ≤ m :=
by sorry

end NUMINAMATH_CALUDE_least_six_digit_congruent_to_3_mod_17_l3903_390302


namespace NUMINAMATH_CALUDE_three_correct_letters_probability_l3903_390395

/-- The number of people and letters --/
def n : ℕ := 5

/-- The number of people who receive the correct letter --/
def k : ℕ := 3

/-- The probability of exactly k people receiving their correct letter when n letters are randomly distributed to n people --/
def prob_correct_letters (n k : ℕ) : ℚ :=
  (Nat.choose n k * Nat.factorial (n - k)) / Nat.factorial n

theorem three_correct_letters_probability :
  prob_correct_letters n k = 1 / 12 := by
  sorry

end NUMINAMATH_CALUDE_three_correct_letters_probability_l3903_390395


namespace NUMINAMATH_CALUDE_cylinder_height_comparison_l3903_390355

-- Define the cylinders
structure Cylinder where
  radius : ℝ
  height : ℝ

-- Define the theorem
theorem cylinder_height_comparison (c1 c2 : Cylinder) 
  (h_volume : π * c1.radius^2 * c1.height = π * c2.radius^2 * c2.height)
  (h_radius : c2.radius = 1.2 * c1.radius) :
  c1.height = 1.44 * c2.height := by
  sorry

end NUMINAMATH_CALUDE_cylinder_height_comparison_l3903_390355


namespace NUMINAMATH_CALUDE_function_strictly_increasing_iff_a_in_range_l3903_390394

/-- The function f(x) = (a-2)a^x is strictly increasing if and only if a is in the set (0,1) ∪ (2,+∞) -/
theorem function_strictly_increasing_iff_a_in_range (a : ℝ) (ha : a > 0 ∧ a ≠ 1) :
  (∀ (x₁ x₂ : ℝ), x₁ ≠ x₂ → (((a - 2) * a^x₁ - (a - 2) * a^x₂) / (x₁ - x₂)) > 0) ↔
  (a ∈ Set.Ioo 0 1 ∪ Set.Ioi 2) :=
sorry

end NUMINAMATH_CALUDE_function_strictly_increasing_iff_a_in_range_l3903_390394


namespace NUMINAMATH_CALUDE_female_student_count_l3903_390391

theorem female_student_count (total_students : ℕ) (selection_ways : ℕ) :
  total_students = 8 →
  selection_ways = 30 →
  (∃ (male_students : ℕ) (female_students : ℕ),
    male_students + female_students = total_students ∧
    (male_students.choose 2) * female_students = selection_ways ∧
    (female_students = 2 ∨ female_students = 3)) :=
by sorry

end NUMINAMATH_CALUDE_female_student_count_l3903_390391


namespace NUMINAMATH_CALUDE_unique_zamena_assignment_l3903_390312

def digit := Fin 5

structure Assignment where
  Z : digit
  A : digit
  M : digit
  E : digit
  N : digit
  H : digit

def satisfies_inequalities (a : Assignment) : Prop :=
  (3 > a.A.val + 1) ∧ 
  (a.A.val > a.M.val) ∧ 
  (a.M.val < a.E.val) ∧ 
  (a.E.val < a.H.val) ∧ 
  (a.H.val < a.A.val)

def all_different (a : Assignment) : Prop :=
  a.Z ≠ a.A ∧ a.Z ≠ a.M ∧ a.Z ≠ a.E ∧ a.Z ≠ a.N ∧ a.Z ≠ a.H ∧
  a.A ≠ a.M ∧ a.A ≠ a.E ∧ a.A ≠ a.N ∧ a.A ≠ a.H ∧
  a.M ≠ a.E ∧ a.M ≠ a.N ∧ a.M ≠ a.H ∧
  a.E ≠ a.N ∧ a.E ≠ a.H ∧
  a.N ≠ a.H

def zamena_value (a : Assignment) : ℕ :=
  100000 * (a.Z.val + 1) + 10000 * (a.A.val + 1) + 1000 * (a.M.val + 1) +
  100 * (a.E.val + 1) + 10 * (a.N.val + 1) + (a.A.val + 1)

theorem unique_zamena_assignment :
  ∀ a : Assignment, 
    satisfies_inequalities a → all_different a → 
    zamena_value a = 541234 :=
sorry

end NUMINAMATH_CALUDE_unique_zamena_assignment_l3903_390312


namespace NUMINAMATH_CALUDE_cube_edge_length_l3903_390379

theorem cube_edge_length (l w h : ℝ) (cube_edge : ℝ) : 
  l = 2 → w = 4 → h = 8 → l * w * h = cube_edge^3 → cube_edge = 4 := by
  sorry

end NUMINAMATH_CALUDE_cube_edge_length_l3903_390379


namespace NUMINAMATH_CALUDE_cost_difference_l3903_390374

def ice_cream_quantity : ℕ := 100
def yoghurt_quantity : ℕ := 35
def cheese_quantity : ℕ := 50
def milk_quantity : ℕ := 20

def ice_cream_price : ℚ := 12
def yoghurt_price : ℚ := 3
def cheese_price : ℚ := 8
def milk_price : ℚ := 4

def ice_cream_discount : ℚ := 0.05
def yoghurt_tax : ℚ := 0.08
def cheese_discount : ℚ := 0.10

def returned_ice_cream : ℕ := 10
def returned_cheese : ℕ := 5

def adjusted_ice_cream_cost : ℚ :=
  (ice_cream_quantity * ice_cream_price) * (1 - ice_cream_discount) -
  (returned_ice_cream * ice_cream_price)

def adjusted_yoghurt_cost : ℚ :=
  (yoghurt_quantity * yoghurt_price) * (1 + yoghurt_tax)

def adjusted_cheese_cost : ℚ :=
  (cheese_quantity * cheese_price) * (1 - cheese_discount) -
  (returned_cheese * cheese_price)

def adjusted_milk_cost : ℚ :=
  milk_quantity * milk_price

theorem cost_difference :
  adjusted_ice_cream_cost + adjusted_cheese_cost -
  (adjusted_yoghurt_cost + adjusted_milk_cost) = 1146.60 := by
  sorry

end NUMINAMATH_CALUDE_cost_difference_l3903_390374


namespace NUMINAMATH_CALUDE_v_1010_proof_l3903_390308

/-- Represents the last term of the nth group in the sequence -/
def f (n : ℕ) : ℕ := (5 * n^2 - 3 * n + 2) / 2

/-- Represents the total number of terms up to and including the nth group -/
def total_terms (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The 1010th term of the sequence -/
def v_1010 : ℕ := 4991

theorem v_1010_proof : 
  ∃ (group : ℕ), 
    total_terms group ≥ 1010 ∧ 
    total_terms (group - 1) < 1010 ∧
    v_1010 = f group - (total_terms group - 1010) :=
sorry

end NUMINAMATH_CALUDE_v_1010_proof_l3903_390308


namespace NUMINAMATH_CALUDE_sum_of_numbers_l3903_390372

/-- The set of digits used to create the three-digit numbers -/
def digits : Finset Nat := {2, 4, 6, 8}

/-- A function that generates all possible three-digit numbers using the given digits -/
def generateNumbers (digits : Finset Nat) : Finset Nat :=
  Finset.biUnion digits (λ h => 
    Finset.biUnion digits (λ t => 
      Finset.image (λ u => h * 100 + t * 10 + u) digits))

/-- The theorem stating that the sum of all possible three-digit numbers is 35,520 -/
theorem sum_of_numbers (digits : Finset Nat) : 
  (Finset.sum (generateNumbers digits) id) = 35520 :=
sorry

end NUMINAMATH_CALUDE_sum_of_numbers_l3903_390372


namespace NUMINAMATH_CALUDE_goldfish_equal_after_11_months_l3903_390371

/-- Number of months until Brent and Gretel have the same number of goldfish -/
def months_until_equal : ℕ := 11

/-- Brent's initial number of goldfish -/
def brent_initial : ℕ := 6

/-- Gretel's initial number of goldfish -/
def gretel_initial : ℕ := 150

/-- Brent's goldfish growth rate per month -/
def brent_growth_rate : ℝ := 2

/-- Gretel's goldfish growth rate per month -/
def gretel_growth_rate : ℝ := 1.5

/-- Brent's number of goldfish after n months -/
def brent_goldfish (n : ℕ) : ℝ := brent_initial * brent_growth_rate ^ n

/-- Gretel's number of goldfish after n months -/
def gretel_goldfish (n : ℕ) : ℝ := gretel_initial * gretel_growth_rate ^ n

theorem goldfish_equal_after_11_months :
  brent_goldfish months_until_equal = gretel_goldfish months_until_equal :=
sorry

end NUMINAMATH_CALUDE_goldfish_equal_after_11_months_l3903_390371


namespace NUMINAMATH_CALUDE_x_1997_value_l3903_390381

def x : ℕ → ℕ
  | 0 => 1
  | n + 1 => x n + (x n / (n + 1)) + 2

theorem x_1997_value : x 1996 = 23913 := by
  sorry

end NUMINAMATH_CALUDE_x_1997_value_l3903_390381


namespace NUMINAMATH_CALUDE_two_candles_burning_time_l3903_390363

/-- Proves that the time during which exactly two candles are burning simultaneously is 35 minutes -/
theorem two_candles_burning_time (t₁ t₂ t₃ : ℕ) 
  (h₁ : t₁ = 30) 
  (h₂ : t₂ = 40) 
  (h₃ : t₃ = 50) 
  (h_three : ℕ) 
  (h_three_eq : h_three = 10) 
  (h_one : ℕ) 
  (h_one_eq : h_one = 20) 
  (h_two : ℕ) 
  (h_total : h_one + 2 * h_two + 3 * h_three = t₁ + t₂ + t₃) : 
  h_two = 35 := by
  sorry

end NUMINAMATH_CALUDE_two_candles_burning_time_l3903_390363


namespace NUMINAMATH_CALUDE_smallest_unique_sum_l3903_390362

/-- 
Given two natural numbers a and b, if their sum c can be uniquely represented 
in the form A + B = AV (where A, B, and V are distinct letters representing 
distinct digits), then the smallest possible value of c is 10.
-/
theorem smallest_unique_sum (a b : ℕ) : 
  (∃! (A B V : ℕ), A < 10 ∧ B < 10 ∧ V < 10 ∧ A ≠ B ∧ A ≠ V ∧ B ≠ V ∧ 
    a + b = c ∧ 10 * A + V = c ∧ a = A ∧ b = B) → 
  (∀ c' : ℕ, c' < c → ¬∃! (A' B' V' : ℕ), A' < 10 ∧ B' < 10 ∧ V' < 10 ∧ 
    A' ≠ B' ∧ A' ≠ V' ∧ B' ≠ V' ∧ a + b = c' ∧ 10 * A' + V' = c' ∧ a = A' ∧ b = B') →
  c = 10 := by
sorry

end NUMINAMATH_CALUDE_smallest_unique_sum_l3903_390362


namespace NUMINAMATH_CALUDE_common_divisors_8400_7560_l3903_390344

theorem common_divisors_8400_7560 : Nat.card {d : ℕ | d ∣ 8400 ∧ d ∣ 7560} = 32 := by
  sorry

end NUMINAMATH_CALUDE_common_divisors_8400_7560_l3903_390344


namespace NUMINAMATH_CALUDE_preferred_groups_2000_l3903_390333

/-- The number of non-empty subsets of {1, 2, ..., n} whose sum is divisible by 5 -/
def preferred_groups (n : ℕ) : ℕ := sorry

/-- The formula for the number of preferred groups when n = 2000 -/
def preferred_groups_formula : ℕ := 
  2^400 * ((1 / 5) * (2^1600 - 1) + 1) - 1

theorem preferred_groups_2000 : 
  preferred_groups 2000 = preferred_groups_formula := by sorry

end NUMINAMATH_CALUDE_preferred_groups_2000_l3903_390333


namespace NUMINAMATH_CALUDE_amount_left_after_spending_l3903_390369

def mildred_spent : ℕ := 25
def candice_spent : ℕ := 35
def total_given : ℕ := 100

theorem amount_left_after_spending :
  total_given - (mildred_spent + candice_spent) = 40 :=
by sorry

end NUMINAMATH_CALUDE_amount_left_after_spending_l3903_390369


namespace NUMINAMATH_CALUDE_propositions_correctness_l3903_390354

-- Define the property of being even
def IsEven (n : ℤ) : Prop := ∃ k : ℤ, n = 2 * k

-- Define the property of being divisible by 2
def DivisibleBy2 (n : ℤ) : Prop := ∃ k : ℤ, n = 2 * k

-- Define the inequality from proposition ③
def Inequality (a x : ℝ) : Prop := (a^2 - 4) * x^2 + (a + 2) * x - 1 ≥ 0

theorem propositions_correctness :
  -- Proposition ②
  (¬ ∀ n : ℤ, DivisibleBy2 n → IsEven n) ↔ (∃ n : ℤ, DivisibleBy2 n ∧ ¬IsEven n)
  ∧
  -- Proposition ③
  ∃ a : ℝ, (¬ (|a| ≤ 1)) ∧ (∀ x : ℝ, ¬Inequality a x) :=
by sorry

end NUMINAMATH_CALUDE_propositions_correctness_l3903_390354


namespace NUMINAMATH_CALUDE_problem_1_problem_2_l3903_390393

-- Problem 1
theorem problem_1 : Real.sqrt 9 + |3 - Real.pi| - Real.sqrt ((-3)^2) = Real.pi - 3 := by
  sorry

-- Problem 2
theorem problem_2 : ∃ x : ℝ, 3 * (x - 1)^3 = 81 ∧ x = 4 := by
  sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_l3903_390393


namespace NUMINAMATH_CALUDE_M_intersect_N_equals_N_l3903_390329

-- Define the sets M and N
def M : Set ℝ := {x : ℝ | ∃ y : ℝ, y = x^2 - 2}
def N : Set ℝ := {y : ℝ | ∃ x : ℝ, y = x^2 - 2}

-- Statement to prove
theorem M_intersect_N_equals_N : M ∩ N = N := by sorry

end NUMINAMATH_CALUDE_M_intersect_N_equals_N_l3903_390329


namespace NUMINAMATH_CALUDE_statement_a_incorrect_statement_b_correct_statement_c_correct_statement_d_correct_l3903_390380

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the necessary operations and relations
variable (subset : Line → Plane → Prop)
variable (parallel : Line → Line → Prop)
variable (parallel_line_plane : Line → Plane → Prop)
variable (parallel_plane : Plane → Plane → Prop)
variable (intersect_line : Line → Line → Prop)
variable (intersect_plane : Plane → Plane → Line)

-- Statement A
theorem statement_a_incorrect 
  (a b : Line) (α : Plane) :
  ∃ a b α, subset b α ∧ parallel a b ∧ ¬(parallel_line_plane a α) := by sorry

-- Statement B
theorem statement_b_correct 
  (a b : Line) (α β : Plane) :
  parallel_line_plane a α → intersect_plane α β = b → subset a β → parallel a b := by sorry

-- Statement C
theorem statement_c_correct 
  (a b : Line) (α β : Plane) (p : Line) :
  subset a α → subset b α → intersect_line a b → 
  parallel_line_plane a β → parallel_line_plane b β → 
  parallel_plane α β := by sorry

-- Statement D
theorem statement_d_correct 
  (a b : Line) (α β γ : Plane) :
  parallel_plane α β → intersect_plane α γ = a → intersect_plane β γ = b → 
  parallel a b := by sorry

end NUMINAMATH_CALUDE_statement_a_incorrect_statement_b_correct_statement_c_correct_statement_d_correct_l3903_390380


namespace NUMINAMATH_CALUDE_inequality_abc_l3903_390346

theorem inequality_abc (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (ha : Real.sqrt a = x * (y - z)^2)
  (hb : Real.sqrt b = y * (z - x)^2)
  (hc : Real.sqrt c = z * (x - y)^2) :
  a^2 + b^2 + c^2 ≥ 2*(a*b + b*c + c*a) := by
sorry

end NUMINAMATH_CALUDE_inequality_abc_l3903_390346


namespace NUMINAMATH_CALUDE_carla_marbles_l3903_390342

/-- The number of marbles Carla bought -/
def marbles_bought : ℕ := 134

/-- The number of marbles Carla has now -/
def marbles_now : ℕ := 187

/-- The number of marbles Carla started with -/
def marbles_start : ℕ := marbles_now - marbles_bought

theorem carla_marbles : marbles_start = 53 := by
  sorry

end NUMINAMATH_CALUDE_carla_marbles_l3903_390342


namespace NUMINAMATH_CALUDE_average_xyz_l3903_390330

theorem average_xyz (x y z : ℝ) (h : (5 / 2) * (x + y + z) = 20) : 
  (x + y + z) / 3 = 8 / 3 := by
sorry

end NUMINAMATH_CALUDE_average_xyz_l3903_390330


namespace NUMINAMATH_CALUDE_stacy_height_l3903_390340

/-- Stacy's height problem -/
theorem stacy_height (stacy_last_year : ℕ) (brother_growth : ℕ) (stacy_extra_growth : ℕ) :
  stacy_last_year = 50 →
  brother_growth = 1 →
  stacy_extra_growth = 6 →
  stacy_last_year + brother_growth + stacy_extra_growth = 57 :=
by sorry

end NUMINAMATH_CALUDE_stacy_height_l3903_390340


namespace NUMINAMATH_CALUDE_arcade_spending_equals_allowance_l3903_390396

def dress_cost : ℕ := 80
def initial_savings : ℕ := 20
def weekly_allowance : ℕ := 30
def weeks_to_save : ℕ := 3

theorem arcade_spending_equals_allowance :
  ∃ (arcade_spending : ℕ),
    arcade_spending = weekly_allowance ∧
    initial_savings + weeks_to_save * weekly_allowance - weeks_to_save * arcade_spending = dress_cost :=
by sorry

end NUMINAMATH_CALUDE_arcade_spending_equals_allowance_l3903_390396


namespace NUMINAMATH_CALUDE_xy_value_l3903_390328

theorem xy_value (x y : ℝ) 
  (h1 : (4:ℝ)^x / (2:ℝ)^(x+y) = 16)
  (h2 : (9:ℝ)^(x+y) / (3:ℝ)^(5*y) = 81) : 
  x * y = 32 := by
  sorry

end NUMINAMATH_CALUDE_xy_value_l3903_390328


namespace NUMINAMATH_CALUDE_edward_picked_three_l3903_390343

/-- The number of pieces of paper Olivia picked up -/
def olivia_pieces : ℕ := 16

/-- The total number of pieces of paper picked up by Olivia and Edward -/
def total_pieces : ℕ := 19

/-- The number of pieces of paper Edward picked up -/
def edward_pieces : ℕ := total_pieces - olivia_pieces

theorem edward_picked_three : edward_pieces = 3 := by
  sorry

end NUMINAMATH_CALUDE_edward_picked_three_l3903_390343


namespace NUMINAMATH_CALUDE_range_of_a_l3903_390338

theorem range_of_a (a : ℝ) : 
  (∀ x ∈ (Set.Ioo 0 1), (x + Real.log a) / Real.exp x - a * Real.log x / x > 0) →
  a ∈ Set.Icc (Real.exp (-1)) 1 ∧ a ≠ 1 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l3903_390338


namespace NUMINAMATH_CALUDE_two_numbers_with_given_means_l3903_390399

theorem two_numbers_with_given_means (a b : ℝ) (ha : a > 0) (hb : b > 0) : 
  (Real.sqrt (a * b) = Real.sqrt 5) → 
  (2 / (1/a + 1/b) = 5/3) → 
  ((a = 5 ∧ b = 1) ∨ (a = 1 ∧ b = 5)) := by
sorry

end NUMINAMATH_CALUDE_two_numbers_with_given_means_l3903_390399


namespace NUMINAMATH_CALUDE_equation_represents_circle_l3903_390370

/-- The equation (x-3)^2 = -(3y+1)^2 + 45 represents a circle -/
theorem equation_represents_circle : ∃ (h k r : ℝ), ∀ (x y : ℝ),
  (x - 3)^2 = -(3*y + 1)^2 + 45 ↔ (x - h)^2 + (y - k)^2 = r^2 :=
sorry

end NUMINAMATH_CALUDE_equation_represents_circle_l3903_390370


namespace NUMINAMATH_CALUDE_new_barbell_cost_l3903_390365

/-- The cost of a new barbell that is 30% more expensive than an old barbell priced at $250 is $325. -/
theorem new_barbell_cost (old_price : ℝ) (new_price : ℝ) : 
  old_price = 250 →
  new_price = old_price * 1.3 →
  new_price = 325 := by
  sorry

end NUMINAMATH_CALUDE_new_barbell_cost_l3903_390365


namespace NUMINAMATH_CALUDE_inequality_system_solution_l3903_390356

theorem inequality_system_solution (x : ℝ) : 
  (1 - (2*x - 1)/2 > (3*x - 1)/4 ∧ 2 - 3*x ≤ 4 - x) ↔ -1 ≤ x ∧ x < 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_system_solution_l3903_390356


namespace NUMINAMATH_CALUDE_midpoint_square_sum_l3903_390351

def A : ℝ × ℝ := (2, 6)
def C : ℝ × ℝ := (4, 1)

theorem midpoint_square_sum (x y : ℝ) : 
  (∀ (p : ℝ × ℝ), p = ((A.1 + x) / 2, (A.2 + y) / 2) → p = C) →
  x^2 + y^2 = 52 := by
  sorry

end NUMINAMATH_CALUDE_midpoint_square_sum_l3903_390351


namespace NUMINAMATH_CALUDE_geometric_sequence_fifth_term_l3903_390385

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_fifth_term
  (a : ℕ → ℝ)
  (h_geom : geometric_sequence a)
  (h_roots : a 3 * a 7 = 3 ∧ a 3 + a 7 = 4) :
  a 5 = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_fifth_term_l3903_390385


namespace NUMINAMATH_CALUDE_linos_shells_l3903_390377

/-- The number of shells Lino picked up -/
def shells_picked_up : ℝ := 324.0

/-- The number of shells Lino put back -/
def shells_put_back : ℝ := 292.00

/-- The number of shells Lino has in all -/
def shells_remaining : ℝ := shells_picked_up - shells_put_back

/-- Theorem stating that the number of shells Lino has in all is 32.0 -/
theorem linos_shells : shells_remaining = 32.0 := by sorry

end NUMINAMATH_CALUDE_linos_shells_l3903_390377


namespace NUMINAMATH_CALUDE_brothers_identity_l3903_390387

-- Define the types for brothers and card suits
inductive Brother
| First
| Second

inductive Suit
| Black
| Red

-- Define the statements made by each brother
def firstBrotherStatement (secondBrotherName : String) (secondBrotherSuit : Suit) : Prop :=
  secondBrotherName = "Tweedledee" ∧ secondBrotherSuit = Suit.Black

def secondBrotherStatement (firstBrotherName : String) (firstBrotherSuit : Suit) : Prop :=
  firstBrotherName = "Tweedledum" ∧ firstBrotherSuit = Suit.Red

-- Define the theorem
theorem brothers_identity :
  ∃ (firstBrotherName secondBrotherName : String) 
    (firstBrotherSuit secondBrotherSuit : Suit),
    (firstBrotherName = "Tweedledee" ∧ secondBrotherName = "Tweedledum") ∧
    (firstBrotherSuit = Suit.Black ∧ secondBrotherSuit = Suit.Red) ∧
    (firstBrotherStatement secondBrotherName secondBrotherSuit ≠ 
     secondBrotherStatement firstBrotherName firstBrotherSuit) :=
by
  sorry

end NUMINAMATH_CALUDE_brothers_identity_l3903_390387


namespace NUMINAMATH_CALUDE_sin_210_degrees_l3903_390304

theorem sin_210_degrees : Real.sin (210 * π / 180) = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_sin_210_degrees_l3903_390304


namespace NUMINAMATH_CALUDE_moving_circle_center_path_l3903_390335

/-- A moving circle M with center (x, y) passes through (3, 2) and is tangent to y = 1 -/
def MovingCircle (x y : ℝ) : Prop :=
  (x - 3)^2 + (y - 2)^2 = (y - 1)^2

/-- The equation of the path traced by the center of the moving circle -/
def CenterPath (x y : ℝ) : Prop :=
  x^2 - 6*x + 2*y + 12 = 0

/-- Theorem: The equation of the path traced by the center of the moving circle
    is x^2 - 6x + 2y + 12 = 0 -/
theorem moving_circle_center_path :
  ∀ x y : ℝ, MovingCircle x y → CenterPath x y := by
  sorry

end NUMINAMATH_CALUDE_moving_circle_center_path_l3903_390335


namespace NUMINAMATH_CALUDE_max_area_equilateral_triangle_in_rectangle_l3903_390357

/-- The maximum area of an equilateral triangle inscribed in a 12x5 rectangle --/
theorem max_area_equilateral_triangle_in_rectangle :
  ∃ (A : ℝ),
    A = (25 : ℝ) * Real.sqrt 3 / 3 ∧
    ∀ (s : ℝ),
      s > 0 →
      s ≤ 12 →
      s * Real.sqrt 3 / 2 ≤ 5 →
      (Real.sqrt 3 / 4) * s^2 ≤ A :=
by sorry

end NUMINAMATH_CALUDE_max_area_equilateral_triangle_in_rectangle_l3903_390357


namespace NUMINAMATH_CALUDE_cube_volume_surface_area_l3903_390378

theorem cube_volume_surface_area (x : ℝ) : 
  (∃ s : ℝ, s > 0 ∧ s^3 = 8*x ∧ 6*s^2 = x) → x = 13824 := by
  sorry

end NUMINAMATH_CALUDE_cube_volume_surface_area_l3903_390378


namespace NUMINAMATH_CALUDE_prime_square_minus_one_div_24_l3903_390367

theorem prime_square_minus_one_div_24 (p : ℕ) (hp : Prime p) (hp_gt_3 : p > 3) : 
  24 ∣ (p^2 - 1) := by
sorry

end NUMINAMATH_CALUDE_prime_square_minus_one_div_24_l3903_390367


namespace NUMINAMATH_CALUDE_negation_of_existence_negation_of_quadratic_inequality_l3903_390303

theorem negation_of_existence (p : ℝ → Prop) :
  (¬ ∃ x, p x) ↔ (∀ x, ¬ p x) := by sorry

theorem negation_of_quadratic_inequality :
  (¬ ∃ x : ℝ, x^2 + x - 1 < 0) ↔ (∀ x : ℝ, x^2 + x - 1 ≥ 0) := by sorry

end NUMINAMATH_CALUDE_negation_of_existence_negation_of_quadratic_inequality_l3903_390303


namespace NUMINAMATH_CALUDE_unique_four_digit_number_with_reverse_property_l3903_390315

def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n < 10000

def reverse_digits (n : ℕ) : ℕ :=
  let d₁ := n / 1000
  let d₂ := (n / 100) % 10
  let d₃ := (n / 10) % 10
  let d₄ := n % 10
  d₄ * 1000 + d₃ * 100 + d₂ * 10 + d₁

theorem unique_four_digit_number_with_reverse_property :
  ∃! n : ℕ, is_four_digit n ∧ n + 7182 = reverse_digits n :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_unique_four_digit_number_with_reverse_property_l3903_390315


namespace NUMINAMATH_CALUDE_tan_alpha_3_implies_fraction_l3903_390321

theorem tan_alpha_3_implies_fraction (α : Real) (h : Real.tan α = 3) :
  (Real.sin α + 3 * Real.cos α) / (2 * Real.sin α + 5 * Real.cos α) = 6 / 11 := by
  sorry

end NUMINAMATH_CALUDE_tan_alpha_3_implies_fraction_l3903_390321


namespace NUMINAMATH_CALUDE_oak_willow_difference_l3903_390332

theorem oak_willow_difference (total_trees : ℕ) (willow_percent oak_percent : ℚ) : 
  total_trees = 712 →
  willow_percent = 34 / 100 →
  oak_percent = 45 / 100 →
  ⌊oak_percent * total_trees⌋ - ⌊willow_percent * total_trees⌋ = 78 := by
  sorry

end NUMINAMATH_CALUDE_oak_willow_difference_l3903_390332


namespace NUMINAMATH_CALUDE_logarithm_equality_l3903_390366

theorem logarithm_equality (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x^2 + 4*y^2 = 12*x*y) :
  Real.log (x + 2*y) / Real.log 10 - 2 * (Real.log 2 / Real.log 10) = 
  (1/2) * (Real.log x / Real.log 10 + Real.log y / Real.log 10) := by
  sorry

end NUMINAMATH_CALUDE_logarithm_equality_l3903_390366


namespace NUMINAMATH_CALUDE_diagonal_intersection_probability_l3903_390306

theorem diagonal_intersection_probability (n : ℕ) (h : n > 0) :
  let vertices := 2 * n + 1
  let total_diagonals := vertices * (vertices - 3) / 2
  let intersecting_diagonals := vertices.choose 4
  intersecting_diagonals / (total_diagonals.choose 2 : ℚ) = 
    n * (2 * n - 1) / (3 * (2 * n^2 - n - 2)) := by
  sorry

end NUMINAMATH_CALUDE_diagonal_intersection_probability_l3903_390306


namespace NUMINAMATH_CALUDE_max_value_trig_sum_l3903_390398

theorem max_value_trig_sum (x : ℝ) : 3 * Real.cos x + 4 * Real.sin x ≤ 5 := by
  sorry

end NUMINAMATH_CALUDE_max_value_trig_sum_l3903_390398


namespace NUMINAMATH_CALUDE_local_minimum_implies_a_eq_2_l3903_390313

/-- The function f(x) = x(x-a)² has a local minimum at x = 2 -/
def has_local_minimum (f : ℝ → ℝ) (a : ℝ) : Prop :=
  ∃ δ > 0, ∀ x, |x - 2| < δ → f x ≥ f 2

/-- The function f(x) = x(x-a)² -/
def f (a : ℝ) (x : ℝ) : ℝ := x * (x - a)^2

theorem local_minimum_implies_a_eq_2 :
  ∀ a : ℝ, has_local_minimum (f a) a → a = 2 := by
  sorry

end NUMINAMATH_CALUDE_local_minimum_implies_a_eq_2_l3903_390313


namespace NUMINAMATH_CALUDE_focus_of_parabola_is_correct_l3903_390389

/-- The focus of a parabola y^2 = 12x --/
def focus_of_parabola : ℝ × ℝ := (3, 0)

/-- The equation of the parabola --/
def parabola_equation (x y : ℝ) : Prop := y^2 = 12 * x

/-- Theorem: The focus of the parabola y^2 = 12x is at the point (3, 0) --/
theorem focus_of_parabola_is_correct :
  let (a, b) := focus_of_parabola
  ∀ x y : ℝ, parabola_equation x y → (x - a)^2 + y^2 = (x + a)^2 :=
sorry

end NUMINAMATH_CALUDE_focus_of_parabola_is_correct_l3903_390389


namespace NUMINAMATH_CALUDE_ezekiel_new_shoes_l3903_390341

/-- The number of pairs of shoes Ezekiel bought -/
def pairs_bought : ℕ := 3

/-- The number of shoes in each pair -/
def shoes_per_pair : ℕ := 2

/-- The total number of new shoes Ezekiel has now -/
def total_new_shoes : ℕ := pairs_bought * shoes_per_pair

theorem ezekiel_new_shoes : total_new_shoes = 6 := by
  sorry

end NUMINAMATH_CALUDE_ezekiel_new_shoes_l3903_390341


namespace NUMINAMATH_CALUDE_solve_problem_l3903_390360

def problem (x y : ℝ) : Prop :=
  x > 0 ∧ y > 0 ∧
  (x + y + 9 + 10 + 11) / 5 = 10 ∧
  ((x - 10)^2 + (y - 10)^2 + (9 - 10)^2 + (10 - 10)^2 + (11 - 10)^2) / 5 = 2

theorem solve_problem (x y : ℝ) (h : problem x y) : |x - y| = 4 :=
by sorry

end NUMINAMATH_CALUDE_solve_problem_l3903_390360


namespace NUMINAMATH_CALUDE_square_min_rotation_angle_l3903_390358

/-- The minimum rotation angle for a square to coincide with its original position -/
def min_rotation_angle_square : ℝ := 90

/-- A square has rotational symmetry of order 4 -/
def rotational_symmetry_order_square : ℕ := 4

theorem square_min_rotation_angle :
  min_rotation_angle_square = 360 / rotational_symmetry_order_square :=
by sorry

end NUMINAMATH_CALUDE_square_min_rotation_angle_l3903_390358


namespace NUMINAMATH_CALUDE_reduced_journey_time_l3903_390373

/-- Calculates the reduced time of a journey when speed is increased -/
theorem reduced_journey_time 
  (original_time : ℝ) 
  (original_speed : ℝ) 
  (new_speed : ℝ) 
  (h1 : original_time = 50) 
  (h2 : original_speed = 48) 
  (h3 : new_speed = 60) : 
  (original_time * original_speed) / new_speed = 40 := by
  sorry

end NUMINAMATH_CALUDE_reduced_journey_time_l3903_390373


namespace NUMINAMATH_CALUDE_ball_max_height_l3903_390322

/-- The height function of the ball -/
def f (t : ℝ) : ℝ := -16 * t^2 + 96 * t + 15

/-- Theorem stating that the maximum height of the ball is 159 feet -/
theorem ball_max_height :
  ∃ t_max : ℝ, ∀ t : ℝ, f t ≤ f t_max ∧ f t_max = 159 := by
  sorry

end NUMINAMATH_CALUDE_ball_max_height_l3903_390322


namespace NUMINAMATH_CALUDE_parabola_line_intersection_minimum_l3903_390334

/-- A parabola with equation y^2 = 16x --/
structure Parabola where
  equation : ℝ → ℝ → Prop
  focus : ℝ × ℝ

/-- A line passing through a point --/
structure Line where
  passingPoint : ℝ × ℝ

/-- Intersection points of a line and a parabola --/
structure Intersection where
  M : ℝ × ℝ
  N : ℝ × ℝ

/-- Distance between two points --/
def distance (p1 p2 : ℝ × ℝ) : ℝ := sorry

/-- The main theorem --/
theorem parabola_line_intersection_minimum (p : Parabola) (l : Line) (i : Intersection) :
  p.equation = (fun x y => y^2 = 16*x) →
  p.focus = (4, 0) →
  l.passingPoint = p.focus →
  (∃ (x y : ℝ), p.equation x y ∧ (x, y) = i.M) →
  (∃ (x y : ℝ), p.equation x y ∧ (x, y) = i.N) →
  (∀ NF MF : ℝ, NF = distance i.N p.focus → MF = distance i.M p.focus →
    NF / 9 - 4 / MF ≥ 1 / 3) ∧
  (∃ NF MF : ℝ, NF = distance i.N p.focus ∧ MF = distance i.M p.focus ∧
    NF / 9 - 4 / MF = 1 / 3) :=
sorry

end NUMINAMATH_CALUDE_parabola_line_intersection_minimum_l3903_390334


namespace NUMINAMATH_CALUDE_exists_function_satisfying_conditions_l3903_390376

-- Define the properties of the function f
def satisfies_conditions (f : ℝ → ℝ) : Prop :=
  (∀ x₁ x₂ : ℝ, x₁ + x₂ = 0 → f x₁ + f x₂ = 0) ∧ 
  (∀ x t : ℝ, t > 0 → f (x + t) > f x)

-- State the theorem
theorem exists_function_satisfying_conditions : 
  ∃ f : ℝ → ℝ, satisfies_conditions f ∧ f = fun x ↦ x^3 := by
  sorry

end NUMINAMATH_CALUDE_exists_function_satisfying_conditions_l3903_390376


namespace NUMINAMATH_CALUDE_basketball_highlight_film_avg_player_footage_l3903_390348

/-- Calculates the average player footage in minutes for a basketball highlight film --/
theorem basketball_highlight_film_avg_player_footage
  (point_guard_footage : ℕ)
  (shooting_guard_footage : ℕ)
  (small_forward_footage : ℕ)
  (power_forward_footage : ℕ)
  (center_footage : ℕ)
  (game_footage : ℕ)
  (interview_footage : ℕ)
  (opening_closing_footage : ℕ)
  (h1 : point_guard_footage = 130)
  (h2 : shooting_guard_footage = 145)
  (h3 : small_forward_footage = 85)
  (h4 : power_forward_footage = 60)
  (h5 : center_footage = 180)
  (h6 : game_footage = 120)
  (h7 : interview_footage = 90)
  (h8 : opening_closing_footage = 30) :
  (point_guard_footage + shooting_guard_footage + small_forward_footage + power_forward_footage + center_footage) / (5 * 60) = 2 :=
by sorry

end NUMINAMATH_CALUDE_basketball_highlight_film_avg_player_footage_l3903_390348


namespace NUMINAMATH_CALUDE_ivy_morning_cupcakes_l3903_390349

/-- The number of cupcakes Ivy baked in the morning -/
def morning_cupcakes : ℕ := sorry

/-- The number of cupcakes Ivy baked in the afternoon -/
def afternoon_cupcakes : ℕ := morning_cupcakes + 15

/-- The total number of cupcakes Ivy baked -/
def total_cupcakes : ℕ := 55

/-- Theorem stating that Ivy baked 20 cupcakes in the morning -/
theorem ivy_morning_cupcakes : 
  morning_cupcakes = 20 ∧ 
  afternoon_cupcakes = morning_cupcakes + 15 ∧ 
  total_cupcakes = morning_cupcakes + afternoon_cupcakes := by
  sorry

end NUMINAMATH_CALUDE_ivy_morning_cupcakes_l3903_390349


namespace NUMINAMATH_CALUDE_school_boys_count_l3903_390347

theorem school_boys_count :
  ∀ (boys girls : ℕ),
  (boys : ℚ) / girls = 5 / 13 →
  girls = boys + 80 →
  boys = 50 := by
sorry

end NUMINAMATH_CALUDE_school_boys_count_l3903_390347


namespace NUMINAMATH_CALUDE_line_equations_l3903_390352

/-- A line in 2D space represented by ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Check if a point (x, y) lies on a line -/
def Line.contains (l : Line) (x y : ℝ) : Prop :=
  l.a * x + l.b * y + l.c = 0

/-- Check if two lines are perpendicular -/
def Line.perpendicular (l₁ l₂ : Line) : Prop :=
  l₁.a * l₂.a + l₁.b * l₂.b = 0

/-- Check if a line has equal intercepts on both axes -/
def Line.equalIntercepts (l : Line) : Prop :=
  l.a = l.b ∧ l.a ≠ 0

theorem line_equations (l₁ : Line) :
  (l₁.contains 2 3) →
  (∃ l₂ : Line, l₂.a = 1 ∧ l₂.b = 2 ∧ l₂.c = 4 ∧ l₁.perpendicular l₂) →
  (l₁.a = 2 ∧ l₁.b = -1 ∧ l₁.c = -1) ∨
  (l₁.equalIntercepts → (l₁.a = 1 ∧ l₁.b = 1 ∧ l₁.c = -5) ∨ (l₁.a = 3 ∧ l₁.b = -2 ∧ l₁.c = 0)) :=
by sorry

end NUMINAMATH_CALUDE_line_equations_l3903_390352


namespace NUMINAMATH_CALUDE_parallel_vectors_x_value_l3903_390353

/-- Two vectors are parallel if their components are proportional -/
def are_parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 = a.2 * b.1

theorem parallel_vectors_x_value :
  ∀ x : ℝ, are_parallel (2, x) (1, 2) → x = 4 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_x_value_l3903_390353


namespace NUMINAMATH_CALUDE_max_red_socks_l3903_390375

theorem max_red_socks (x y : ℕ) : 
  x + y ≤ 2017 →
  (x * (x - 1) + y * (y - 1)) / ((x + y) * (x + y - 1)) = 1 / 2 →
  x ≤ 990 :=
by sorry

end NUMINAMATH_CALUDE_max_red_socks_l3903_390375


namespace NUMINAMATH_CALUDE_exponent_identities_l3903_390388

theorem exponent_identities (x a : ℝ) (h : a ≠ 0) : 
  (3 * x^2 * x^4 - (-x^3)^2 = 2 * x^6) ∧ 
  (a^3 * a + (-a^2)^3 / a^2 = 0) := by sorry

end NUMINAMATH_CALUDE_exponent_identities_l3903_390388


namespace NUMINAMATH_CALUDE_multiplication_by_hundred_l3903_390397

theorem multiplication_by_hundred : 38 * 100 = 3800 := by
  sorry

end NUMINAMATH_CALUDE_multiplication_by_hundred_l3903_390397


namespace NUMINAMATH_CALUDE_area_between_concentric_circles_l3903_390364

/-- The area between two concentric circles, where a chord of length 100 units
    is tangent to the smaller circle, is equal to 2500π square units. -/
theorem area_between_concentric_circles (R r : ℝ) : 
  R > r → r > 0 → R^2 - r^2 = 2500 → π * (R^2 - r^2) = 2500 * π := by
  sorry

#check area_between_concentric_circles

end NUMINAMATH_CALUDE_area_between_concentric_circles_l3903_390364


namespace NUMINAMATH_CALUDE_total_children_l3903_390307

theorem total_children (happy_children sad_children neutral_children boys girls happy_boys sad_girls neutral_boys : ℕ) : 
  happy_children = 30 →
  sad_children = 10 →
  neutral_children = 20 →
  boys = 22 →
  girls = 38 →
  happy_boys = 6 →
  sad_girls = 4 →
  neutral_boys = 10 →
  happy_children + sad_children + neutral_children = boys + girls :=
by
  sorry

end NUMINAMATH_CALUDE_total_children_l3903_390307


namespace NUMINAMATH_CALUDE_max_value_base_conversion_l3903_390359

theorem max_value_base_conversion (n A B C : ℕ) : 
  n > 0 →
  n = 64 * A + 8 * B + C →
  n = 81 * C + 9 * B + A →
  C % 2 = 0 →
  A ≤ 7 →
  B ≤ 7 →
  C ≤ 7 →
  n ≤ 64 :=
by sorry

end NUMINAMATH_CALUDE_max_value_base_conversion_l3903_390359


namespace NUMINAMATH_CALUDE_number_division_remainder_l3903_390337

theorem number_division_remainder (N : ℤ) (D : ℕ) 
  (h1 : N % 125 = 40) 
  (h2 : N % D = 11) : 
  D = 29 := by
sorry

end NUMINAMATH_CALUDE_number_division_remainder_l3903_390337


namespace NUMINAMATH_CALUDE_min_filtration_layers_l3903_390320

theorem min_filtration_layers (a : ℝ) (ha : a > 0) : 
  (∃ n : ℕ, n ≥ 5 ∧ a * (4/5)^n ≤ (1/3) * a ∧ ∀ m : ℕ, m < 5 → a * (4/5)^m > (1/3) * a) :=
sorry

end NUMINAMATH_CALUDE_min_filtration_layers_l3903_390320


namespace NUMINAMATH_CALUDE_complement_of_A_l3903_390310

def A : Set ℝ := {x : ℝ | x ≥ 1}

theorem complement_of_A : 
  (Set.univ : Set ℝ) \ A = {x : ℝ | x < 1} := by sorry

end NUMINAMATH_CALUDE_complement_of_A_l3903_390310


namespace NUMINAMATH_CALUDE_no_common_points_l3903_390324

theorem no_common_points : 
  ¬∃ (x y : ℝ), (x^2 + y^2 = 4) ∧ (x^2 + 2*y^2 = 2) := by
  sorry

end NUMINAMATH_CALUDE_no_common_points_l3903_390324


namespace NUMINAMATH_CALUDE_f_domain_is_open_interval_l3903_390318

/-- The domain of the function f(x) = ln((3 - x)(x + 1)) -/
def f_domain : Set ℝ :=
  {x : ℝ | (3 - x) * (x + 1) > 0}

/-- Theorem stating that the domain of f(x) = ln((3 - x)(x + 1)) is (-1, 3) -/
theorem f_domain_is_open_interval :
  f_domain = Set.Ioo (-1) 3 :=
by
  sorry

#check f_domain_is_open_interval

end NUMINAMATH_CALUDE_f_domain_is_open_interval_l3903_390318


namespace NUMINAMATH_CALUDE_hyperbola_properties_l3903_390319

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := y^2 - x^2 = 4

-- Define what it means for a focus to be on the y-axis
def focus_on_y_axis (h : (ℝ → ℝ → Prop)) : Prop :=
  ∃ c : ℝ, ∀ x y : ℝ, h x y → (x = 0 ∧ y = c) ∨ (x = 0 ∧ y = -c)

-- Define what it means for asymptotes to be perpendicular
def perpendicular_asymptotes (h : (ℝ → ℝ → Prop)) : Prop :=
  ∃ m : ℝ, ∀ x y : ℝ, h x y → (y = m*x ∨ y = -m*x) ∧ m * (-1/m) = -1

-- Theorem statement
theorem hyperbola_properties :
  focus_on_y_axis hyperbola ∧ perpendicular_asymptotes hyperbola :=
sorry

end NUMINAMATH_CALUDE_hyperbola_properties_l3903_390319


namespace NUMINAMATH_CALUDE_tara_savings_loss_l3903_390350

/-- The amount Tara had saved before losing all her savings -/
def amount_lost : ℕ := by sorry

theorem tara_savings_loss :
  let clarinet_cost : ℕ := 90
  let initial_savings : ℕ := 10
  let book_price : ℕ := 5
  let total_books_sold : ℕ := 25
  amount_lost = 45 := by sorry

end NUMINAMATH_CALUDE_tara_savings_loss_l3903_390350


namespace NUMINAMATH_CALUDE_right_to_left_eval_equals_56_over_9_l3903_390383

def right_to_left_eval : ℚ := by
  -- Define the operations
  let square (x : ℚ) := x * x
  let divide (x y : ℚ) := x / y
  let add (x y : ℚ) := x + y
  let multiply (x y : ℚ) := x * y

  -- Evaluate from right to left
  let step1 := square 6
  let step2 := divide 4 step1
  let step3 := add 3 step2
  let step4 := multiply 2 step3

  exact step4

-- Theorem statement
theorem right_to_left_eval_equals_56_over_9 : 
  right_to_left_eval = 56 / 9 := by
  sorry

end NUMINAMATH_CALUDE_right_to_left_eval_equals_56_over_9_l3903_390383


namespace NUMINAMATH_CALUDE_max_value_inequality_l3903_390317

theorem max_value_inequality (a b c : ℝ) (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : 0 ≤ c) 
  (h4 : a^2 + b^2 + c^2 = 1) : 2*a*b*Real.sqrt 2 + 2*b*c ≤ Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_max_value_inequality_l3903_390317


namespace NUMINAMATH_CALUDE_product_of_complex_magnitudes_l3903_390345

theorem product_of_complex_magnitudes : Complex.abs (4 - 3*I) * Complex.abs (4 + 3*I) = 25 := by
  sorry

end NUMINAMATH_CALUDE_product_of_complex_magnitudes_l3903_390345


namespace NUMINAMATH_CALUDE_freezer_ice_cubes_l3903_390392

/-- The minimum number of ice cubes in Jerry's freezer -/
def min_ice_cubes (num_cups : ℕ) (ice_per_cup : ℕ) : ℕ :=
  num_cups * ice_per_cup

/-- Theorem stating that the minimum number of ice cubes is the product of cups and ice per cup -/
theorem freezer_ice_cubes (num_cups : ℕ) (ice_per_cup : ℕ) :
  min_ice_cubes num_cups ice_per_cup = num_cups * ice_per_cup :=
by sorry

end NUMINAMATH_CALUDE_freezer_ice_cubes_l3903_390392


namespace NUMINAMATH_CALUDE_expected_ones_is_one_third_l3903_390323

/-- The probability of rolling a 1 on a standard die -/
def prob_one : ℚ := 1 / 6

/-- The probability of not rolling a 1 on a standard die -/
def prob_not_one : ℚ := 1 - prob_one

/-- The expected number of 1's when rolling two standard dice -/
def expected_ones : ℚ := 2 * (prob_one * prob_one) + 1 * (2 * prob_one * prob_not_one)

theorem expected_ones_is_one_third : expected_ones = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_expected_ones_is_one_third_l3903_390323


namespace NUMINAMATH_CALUDE_jeff_running_time_l3903_390325

/-- Represents Jeff's running schedule for a week --/
structure RunningSchedule where
  weekdayCommitment : ℕ  -- Minutes committed per weekday
  thursdayReduction : ℕ  -- Minutes reduced on Thursday
  fridayExtension : ℕ    -- Minutes extended on Friday

/-- Calculates the total running time for the week given a RunningSchedule --/
def totalRunningTime (schedule : RunningSchedule) : ℕ :=
  (3 * schedule.weekdayCommitment) +  -- Monday to Wednesday
  (schedule.weekdayCommitment - schedule.thursdayReduction) +  -- Thursday
  (schedule.weekdayCommitment + schedule.fridayExtension)  -- Friday

/-- Theorem stating that Jeff's total running time for the week is 290 minutes --/
theorem jeff_running_time (jeffSchedule : RunningSchedule)
    (h1 : jeffSchedule.weekdayCommitment = 60)
    (h2 : jeffSchedule.thursdayReduction = 20)
    (h3 : jeffSchedule.fridayExtension = 10) :
  totalRunningTime jeffSchedule = 290 := by
  sorry


end NUMINAMATH_CALUDE_jeff_running_time_l3903_390325


namespace NUMINAMATH_CALUDE_P_zero_equals_eleven_l3903_390339

variables (a b c : ℝ) (P : ℝ → ℝ)

/-- The roots of the cubic equation -/
axiom root_equation : a^3 + 3*a^2 + 5*a + 7 = 0 ∧ 
                      b^3 + 3*b^2 + 5*b + 7 = 0 ∧ 
                      c^3 + 3*c^2 + 5*c + 7 = 0

/-- Properties of polynomial P -/
axiom P_properties : P a = b + c ∧ 
                     P b = a + c ∧ 
                     P c = a + b ∧ 
                     P (a + b + c) = -16

/-- Theorem: P(0) equals 11 -/
theorem P_zero_equals_eleven : P 0 = 11 := by sorry

end NUMINAMATH_CALUDE_P_zero_equals_eleven_l3903_390339


namespace NUMINAMATH_CALUDE_project_time_difference_l3903_390390

/-- Represents the working times of three people on a project -/
structure ProjectTime where
  t1 : ℕ  -- Time of person 1
  t2 : ℕ  -- Time of person 2
  t3 : ℕ  -- Time of person 3

/-- The proposition that the working times are in the ratio 1:2:3 -/
def ratio_correct (pt : ProjectTime) : Prop :=
  2 * pt.t1 = pt.t2 ∧ 3 * pt.t1 = pt.t3

/-- The total project time is 120 hours -/
def total_time_correct (pt : ProjectTime) : Prop :=
  pt.t1 + pt.t2 + pt.t3 = 120

/-- The main theorem stating the difference between longest and shortest working times -/
theorem project_time_difference (pt : ProjectTime) 
  (h1 : ratio_correct pt) (h2 : total_time_correct pt) : 
  pt.t3 - pt.t1 = 40 := by
  sorry


end NUMINAMATH_CALUDE_project_time_difference_l3903_390390


namespace NUMINAMATH_CALUDE_irrational_approximation_l3903_390336

theorem irrational_approximation (x : ℝ) (h_irr : Irrational x) (h_pos : x > 0) :
  ∀ n : ℕ, ∃ p q : ℤ, q > n ∧ q > 0 ∧ |x - (p : ℝ) / q| < 1 / q^2 := by
  sorry

end NUMINAMATH_CALUDE_irrational_approximation_l3903_390336


namespace NUMINAMATH_CALUDE_total_blocks_traveled_l3903_390326

def blocks_to_garage : ℕ := 5
def blocks_to_post_office : ℕ := 20

theorem total_blocks_traveled : 
  (2 * (blocks_to_garage + blocks_to_post_office)) = 50 := by
  sorry

end NUMINAMATH_CALUDE_total_blocks_traveled_l3903_390326


namespace NUMINAMATH_CALUDE_specific_hyperbola_conjugate_axis_length_l3903_390314

/-- Represents a hyperbola with equation x^2 - y^2/m = 1 -/
structure Hyperbola where
  m : ℝ
  focus : ℝ × ℝ

/-- The length of the conjugate axis of a hyperbola -/
def conjugate_axis_length (h : Hyperbola) : ℝ := sorry

/-- Theorem stating the length of the conjugate axis for a specific hyperbola -/
theorem specific_hyperbola_conjugate_axis_length :
  ∀ (h : Hyperbola), 
  h.m > 0 ∧ h.focus = (-3, 0) → 
  conjugate_axis_length h = 4 * Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_specific_hyperbola_conjugate_axis_length_l3903_390314


namespace NUMINAMATH_CALUDE_max_shoe_pairs_l3903_390327

theorem max_shoe_pairs (initial_pairs : ℕ) (lost_shoes : ℕ) (max_remaining_pairs : ℕ) : 
  initial_pairs = 27 → lost_shoes = 9 → max_remaining_pairs = 18 →
  max_remaining_pairs = initial_pairs - lost_shoes / 2 := by
sorry

end NUMINAMATH_CALUDE_max_shoe_pairs_l3903_390327


namespace NUMINAMATH_CALUDE_simplify_expression_l3903_390300

variable (a b : ℝ)

theorem simplify_expression (hb : b ≠ 0) :
  6 * a^5 * b^2 / (3 * a^3 * b^2) + (2 * a * b^3)^2 / (-b^2)^3 = -2 * a^2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l3903_390300


namespace NUMINAMATH_CALUDE_f_properties_l3903_390361

noncomputable def f (x : ℝ) : ℝ := 1 + x - x^3

theorem f_properties :
  (∃! (a b : ℝ), a ≠ b ∧ (deriv f a = 0 ∧ deriv f b = 0) ∧
    ∀ x, deriv f x = 0 → (x = a ∨ x = b)) ∧
  (∃! (a b : ℝ), deriv f a = 0 ∧ deriv f b = 0 ∧ a + b = 0) ∧
  (∃! x, f x = 0) ∧
  (¬∃ x, f x = -x ∧ deriv f x = -1) :=
sorry

end NUMINAMATH_CALUDE_f_properties_l3903_390361


namespace NUMINAMATH_CALUDE_solution_to_equation_l3903_390309

theorem solution_to_equation : ∃ x : ℚ, (1/3 - 1/2) * x = 1 ∧ x = -6 := by sorry

end NUMINAMATH_CALUDE_solution_to_equation_l3903_390309


namespace NUMINAMATH_CALUDE_multiples_of_seven_ending_in_five_l3903_390305

/-- The count of positive multiples of 7 less than 2000 that end with the digit 5 -/
theorem multiples_of_seven_ending_in_five (n : ℕ) : 
  (∃ k : ℕ, n = 7 * k ∧ n < 2000 ∧ n % 10 = 5) ↔ n ∈ Finset.range 29 :=
sorry

end NUMINAMATH_CALUDE_multiples_of_seven_ending_in_five_l3903_390305


namespace NUMINAMATH_CALUDE_rhombicosidodecahedron_symmetries_l3903_390301

/-- Represents a rhombicosidodecahedron -/
structure Rhombicosidodecahedron where
  triangular_faces : ℕ
  square_faces : ℕ
  pentagonal_faces : ℕ
  is_archimedean : Prop
  is_convex : Prop
  is_isogonal : Prop
  is_nonprismatic : Prop

/-- The number of rotational symmetries of a rhombicosidodecahedron -/
def rotational_symmetries (r : Rhombicosidodecahedron) : ℕ := 60

/-- Theorem stating that a rhombicosidodecahedron has 60 rotational symmetries -/
theorem rhombicosidodecahedron_symmetries (r : Rhombicosidodecahedron) 
  (h1 : r.triangular_faces = 20)
  (h2 : r.square_faces = 30)
  (h3 : r.pentagonal_faces = 12)
  (h4 : r.is_archimedean)
  (h5 : r.is_convex)
  (h6 : r.is_isogonal)
  (h7 : r.is_nonprismatic) :
  rotational_symmetries r = 60 := by
  sorry

end NUMINAMATH_CALUDE_rhombicosidodecahedron_symmetries_l3903_390301


namespace NUMINAMATH_CALUDE_eq2_most_suitable_for_factorization_l3903_390316

/-- Represents a quadratic equation --/
inductive QuadraticEquation
  | Eq1 : QuadraticEquation  -- (x+1)(x-3)=2
  | Eq2 : QuadraticEquation  -- 2(x-2)^2=x^2-4
  | Eq3 : QuadraticEquation  -- x^2+3x-1=0
  | Eq4 : QuadraticEquation  -- 5(2-x)^2=3

/-- Predicate to determine if an equation is suitable for factorization --/
def isSuitableForFactorization : QuadraticEquation → Prop :=
  fun eq => match eq with
    | QuadraticEquation.Eq1 => False
    | QuadraticEquation.Eq2 => True
    | QuadraticEquation.Eq3 => False
    | QuadraticEquation.Eq4 => False

/-- Theorem stating that Eq2 is the most suitable for factorization --/
theorem eq2_most_suitable_for_factorization :
  ∀ eq : QuadraticEquation, 
    isSuitableForFactorization eq → eq = QuadraticEquation.Eq2 :=
by
  sorry

end NUMINAMATH_CALUDE_eq2_most_suitable_for_factorization_l3903_390316


namespace NUMINAMATH_CALUDE_solve_system_l3903_390311

theorem solve_system (c d : ℤ) 
  (eq1 : 5 + c = 6 - d) 
  (eq2 : 6 + d = 9 + c) : 
  5 - c = 6 := by
sorry

end NUMINAMATH_CALUDE_solve_system_l3903_390311


namespace NUMINAMATH_CALUDE_exclusive_or_implies_disjunction_l3903_390331

theorem exclusive_or_implies_disjunction (p q : Prop) : 
  ((p ∧ ¬q) ∨ (¬p ∧ q)) → (p ∨ q) :=
by
  sorry

end NUMINAMATH_CALUDE_exclusive_or_implies_disjunction_l3903_390331
