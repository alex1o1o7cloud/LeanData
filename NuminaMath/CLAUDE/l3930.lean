import Mathlib

namespace quadratic_roots_pure_imaginary_l3930_393041

theorem quadratic_roots_pure_imaginary (m : ℝ) (hm : m < 0) :
  ∀ (z : ℂ), 8 * z^2 + 4 * Complex.I * z - m = 0 →
  ∃ (y : ℝ), z = Complex.I * y :=
sorry

end quadratic_roots_pure_imaginary_l3930_393041


namespace solve_money_problem_l3930_393098

def money_problem (mildred_spent candice_spent amount_left : ℕ) : Prop :=
  let total_spent := mildred_spent + candice_spent
  let mom_gave := total_spent + amount_left
  mom_gave = mildred_spent + candice_spent + amount_left

theorem solve_money_problem :
  ∀ (mildred_spent candice_spent amount_left : ℕ),
  money_problem mildred_spent candice_spent amount_left :=
by
  sorry

end solve_money_problem_l3930_393098


namespace double_round_robin_max_teams_l3930_393047

/-- The maximum number of teams in a double round-robin tournament --/
def max_teams : ℕ := 6

/-- The number of weeks available for the tournament --/
def available_weeks : ℕ := 4

/-- The number of matches each team plays in a double round-robin tournament --/
def matches_per_team (n : ℕ) : ℕ := 2 * (n - 1)

/-- The total number of matches in a double round-robin tournament --/
def total_matches (n : ℕ) : ℕ := n * (n - 1)

/-- The maximum number of away matches a team can play in the available weeks --/
def max_away_matches : ℕ := available_weeks

theorem double_round_robin_max_teams :
  ∀ n : ℕ, n ≤ max_teams ∧ 
  matches_per_team n ≤ 2 * max_away_matches ∧
  (∀ m : ℕ, m > max_teams → matches_per_team m > 2 * max_away_matches) :=
by sorry

#check double_round_robin_max_teams

end double_round_robin_max_teams_l3930_393047


namespace intersection_M_N_l3930_393086

def M : Set ℝ := {x : ℝ | 0 ≤ x ∧ x < 2}
def N : Set ℝ := {x : ℝ | x^2 + 2*x - 3 < 0}

theorem intersection_M_N : M ∩ N = {x : ℝ | 0 ≤ x ∧ x < 1} := by sorry

end intersection_M_N_l3930_393086


namespace meaningful_expression_l3930_393091

theorem meaningful_expression (x : ℝ) : 
  (∃ y : ℝ, y = (Real.sqrt (x + 2)) / (x - 1)) ↔ x ≥ -2 ∧ x ≠ 1 := by sorry

end meaningful_expression_l3930_393091


namespace stratified_sample_size_l3930_393031

/-- Represents the total number of staff in each category -/
structure StaffCount where
  business : ℕ
  management : ℕ
  logistics : ℕ

/-- Calculates the total sample size for stratified sampling -/
def calculateSampleSize (staff : StaffCount) (managementSample : ℕ) : ℕ :=
  let totalStaff := staff.business + staff.management + staff.logistics
  let samplingFraction := managementSample / staff.management
  totalStaff * samplingFraction

/-- Theorem: Given the staff counts and management sample, the total sample size is 20 -/
theorem stratified_sample_size 
  (staff : StaffCount) 
  (h1 : staff.business = 120) 
  (h2 : staff.management = 24) 
  (h3 : staff.logistics = 16) 
  (h4 : calculateSampleSize staff 3 = 20) : 
  calculateSampleSize staff 3 = 20 := by
  sorry

#check stratified_sample_size

end stratified_sample_size_l3930_393031


namespace arc_length_30_degree_sector_l3930_393046

/-- The length of an arc in a circular sector with radius 1 cm and central angle 30° is π/6 cm. -/
theorem arc_length_30_degree_sector :
  let r : ℝ := 1  -- radius in cm
  let θ : ℝ := 30 * π / 180  -- central angle in radians
  let l : ℝ := r * θ  -- arc length formula
  l = π / 6 := by sorry

end arc_length_30_degree_sector_l3930_393046


namespace cubic_factorization_l3930_393051

theorem cubic_factorization (x y : ℝ) : x^3 - x*y^2 = x*(x-y)*(x+y) := by
  sorry

end cubic_factorization_l3930_393051


namespace max_same_count_2011_grid_max_same_count_2011_grid_achievable_l3930_393042

/-- Represents a configuration of napkins on a grid -/
structure NapkinConfiguration where
  grid_size : Nat
  napkin_size : Nat
  napkins : List (Nat × Nat)  -- List of (row, column) positions of napkin top-left corners

/-- Calculates the maximum number of cells with the same nonzero napkin count -/
def max_same_count (config : NapkinConfiguration) : Nat :=
  sorry

/-- The main theorem stating the maximum number of cells with the same nonzero napkin count -/
theorem max_same_count_2011_grid (config : NapkinConfiguration) 
  (h1 : config.grid_size = 2011)
  (h2 : config.napkin_size = 52) :
  max_same_count config ≤ 1994^2 + 37 * 17^2 :=
sorry

/-- The theorem stating that the upper bound is achievable -/
theorem max_same_count_2011_grid_achievable : 
  ∃ (config : NapkinConfiguration), 
    config.grid_size = 2011 ∧ 
    config.napkin_size = 52 ∧
    max_same_count config = 1994^2 + 37 * 17^2 :=
sorry

end max_same_count_2011_grid_max_same_count_2011_grid_achievable_l3930_393042


namespace x_range_theorem_l3930_393038

theorem x_range_theorem (a b : ℝ) (h1 : a + b = 1) (h2 : a > 0) (h3 : b > 0)
  (h4 : ∀ x : ℝ, (1/a) + (4/b) ≥ |2*x - 1| - |x + 1|) :
  ∀ x : ℝ, -7 ≤ x ∧ x ≤ 11 := by sorry

end x_range_theorem_l3930_393038


namespace area_of_triangle_KBC_l3930_393063

/-- Represents a point in 2D space -/
structure Point :=
  (x : ℝ) (y : ℝ)

/-- Represents a hexagon -/
structure Hexagon :=
  (A B C D E F : Point)

/-- Represents a square -/
structure Square :=
  (A B C D : Point)

/-- Represents a triangle -/
structure Triangle :=
  (A B C : Point)

/-- The area of a triangle -/
def triangleArea (t : Triangle) : ℝ := sorry

/-- The area of a square -/
def squareArea (s : Square) : ℝ := sorry

/-- Check if a hexagon is equiangular -/
def isEquiangular (h : Hexagon) : Prop := sorry

/-- Check if a triangle is isosceles -/
def isIsosceles (t : Triangle) : Prop := sorry

/-- The length of a line segment between two points -/
def distance (p1 p2 : Point) : ℝ := sorry

theorem area_of_triangle_KBC 
  (ABCDEF : Hexagon) 
  (ABJI FEHG : Square) 
  (JBK : Triangle) :
  isEquiangular ABCDEF →
  squareArea ABJI = 25 →
  squareArea FEHG = 49 →
  isIsosceles JBK →
  distance ABCDEF.F ABCDEF.E = distance ABCDEF.B ABCDEF.C →
  triangleArea ⟨JBK.B, ABCDEF.B, ABCDEF.C⟩ = 49 * Real.sqrt 3 / 4 := by
  sorry

end area_of_triangle_KBC_l3930_393063


namespace fractional_equation_solution_range_l3930_393025

theorem fractional_equation_solution_range (m : ℝ) : 
  (∃ x : ℝ, x ≥ 0 ∧ x ≠ 2 ∧ (m / (x - 2) + 1 = x / (2 - x))) ↔ 
  (m ≤ 2 ∧ m ≠ -2) :=
by sorry

end fractional_equation_solution_range_l3930_393025


namespace congruence_definition_l3930_393023

-- Define a type for geometric figures
def Figure : Type := sorry

-- Define a relation for figures that can completely overlap
def can_overlap (f1 f2 : Figure) : Prop := sorry

-- Define congruence for figures
def congruent (f1 f2 : Figure) : Prop := sorry

-- Theorem stating the definition of congruent figures
theorem congruence_definition :
  ∀ (f1 f2 : Figure), congruent f1 f2 ↔ can_overlap f1 f2 := by sorry

end congruence_definition_l3930_393023


namespace mollys_age_l3930_393002

theorem mollys_age (sandy_current : ℕ) (molly_current : ℕ) : 
  (sandy_current : ℚ) / molly_current = 4 / 3 →
  sandy_current + 6 = 38 →
  molly_current = 24 := by
sorry

end mollys_age_l3930_393002


namespace card_flipping_theorem_l3930_393082

/-- Represents the sum of visible numbers on cards after i flips -/
def sum_after_flips (n : ℕ) (initial_config : Fin n → Bool) (i : Fin (n + 1)) : ℕ :=
  sorry

/-- The statement to be proved -/
theorem card_flipping_theorem (n : ℕ) (h : 0 < n) :
  (∀ initial_config : Fin n → Bool,
    ∃ i j : Fin (n + 1), i ≠ j ∧ sum_after_flips n initial_config i ≠ sum_after_flips n initial_config j) ∧
  (∀ initial_config : Fin n → Bool,
    ∃ r : Fin (n + 1), sum_after_flips n initial_config r = n / 2 ∨ sum_after_flips n initial_config r = (n + 1) / 2) :=
by sorry

end card_flipping_theorem_l3930_393082


namespace x_eleven_percent_greater_than_70_l3930_393057

/-- If x is 11 percent greater than 70, then x = 77.7 -/
theorem x_eleven_percent_greater_than_70 (x : ℝ) : 
  x = 70 * (1 + 11 / 100) → x = 77.7 := by
sorry

end x_eleven_percent_greater_than_70_l3930_393057


namespace bottle_caps_eaten_l3930_393054

theorem bottle_caps_eaten (initial : ℕ) (final : ℕ) (eaten : ℕ) : 
  initial = 65 → final = 61 → initial - final = eaten → eaten = 4 := by
sorry

end bottle_caps_eaten_l3930_393054


namespace coefficient_value_l3930_393061

def P (c : ℝ) (x : ℝ) : ℝ := x^4 + 3*x^3 + 2*x^2 + c*x + 15

theorem coefficient_value (c : ℝ) :
  (∀ x, (x - 7 : ℝ) ∣ P c x) → c = -508 := by
  sorry

end coefficient_value_l3930_393061


namespace other_communities_count_l3930_393048

theorem other_communities_count (total : ℕ) (muslim_percent : ℚ) (hindu_percent : ℚ) (sikh_percent : ℚ) 
  (h_total : total = 400)
  (h_muslim : muslim_percent = 44/100)
  (h_hindu : hindu_percent = 28/100)
  (h_sikh : sikh_percent = 10/100) :
  (total : ℚ) * (1 - (muslim_percent + hindu_percent + sikh_percent)) = 72 := by
  sorry

end other_communities_count_l3930_393048


namespace cats_theorem_l3930_393036

def cats_problem (siamese house persian first_sale second_sale : ℕ) : Prop :=
  let initial_total : ℕ := siamese + house + persian
  let after_first_sale : ℕ := initial_total - first_sale
  let final_count : ℕ := after_first_sale - second_sale
  final_count = 17

theorem cats_theorem : cats_problem 23 17 29 40 12 := by
  sorry

end cats_theorem_l3930_393036


namespace choose_three_from_ten_l3930_393092

theorem choose_three_from_ten : Nat.choose 10 3 = 120 := by
  sorry

end choose_three_from_ten_l3930_393092


namespace contests_paths_l3930_393067

/-- Represents the number of choices for each step in the path, except the last --/
def choices : ℕ := 2

/-- Represents the number of starting points (number of "C"s at the base) --/
def starting_points : ℕ := 2

/-- Represents the number of steps in the path (length of "CONTESTS" - 1) --/
def path_length : ℕ := 7

theorem contests_paths :
  starting_points * (choices ^ path_length) = 256 := by
  sorry

end contests_paths_l3930_393067


namespace right_triangle_hypotenuse_l3930_393016

theorem right_triangle_hypotenuse : 
  ∀ (a b c : ℝ), 
  a = 10 ∧ b = 24 ∧ c^2 = a^2 + b^2 → c = 26 :=
by sorry

end right_triangle_hypotenuse_l3930_393016


namespace x_gets_thirty_paisa_l3930_393032

/-- Represents the share of each person in rupees -/
structure Share where
  w : ℝ
  x : ℝ
  y : ℝ

/-- The total amount distributed -/
def total_amount : ℝ := 15

/-- The share of w in rupees -/
def w_share : ℝ := 10

/-- The amount y gets for each rupee w gets, in rupees -/
def y_per_w : ℝ := 0.20

/-- Theorem stating that x gets 0.30 rupees for each rupee w gets -/
theorem x_gets_thirty_paisa (s : Share) 
  (h1 : s.w = w_share)
  (h2 : s.y = y_per_w * s.w)
  (h3 : s.w + s.x + s.y = total_amount) : 
  s.x / s.w = 0.30 := by sorry

end x_gets_thirty_paisa_l3930_393032


namespace max_colored_cells_l3930_393080

/-- Represents a cell in the 8x8 square --/
structure Cell :=
  (row : Fin 8)
  (col : Fin 8)

/-- Predicate to check if four cells form a rectangle with sides parallel to the edges --/
def formsRectangle (c1 c2 c3 c4 : Cell) : Prop :=
  (c1.row = c2.row ∧ c3.row = c4.row ∧ c1.col = c3.col ∧ c2.col = c4.col) ∨
  (c1.row = c3.row ∧ c2.row = c4.row ∧ c1.col = c2.col ∧ c3.col = c4.col)

/-- The main theorem --/
theorem max_colored_cells :
  ∃ (S : Finset Cell),
    S.card = 24 ∧
    (∀ (c1 c2 c3 c4 : Cell),
      c1 ∈ S → c2 ∈ S → c3 ∈ S → c4 ∈ S →
      c1 ≠ c2 → c1 ≠ c3 → c1 ≠ c4 → c2 ≠ c3 → c2 ≠ c4 → c3 ≠ c4 →
      ¬formsRectangle c1 c2 c3 c4) ∧
    (∀ (T : Finset Cell),
      T.card > 24 →
      ∃ (c1 c2 c3 c4 : Cell),
        c1 ∈ T ∧ c2 ∈ T ∧ c3 ∈ T ∧ c4 ∈ T ∧
        c1 ≠ c2 ∧ c1 ≠ c3 ∧ c1 ≠ c4 ∧ c2 ≠ c3 ∧ c2 ≠ c4 ∧ c3 ≠ c4 ∧
        formsRectangle c1 c2 c3 c4) :=
by sorry


end max_colored_cells_l3930_393080


namespace collinear_points_d_values_l3930_393053

/-- Four points in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Define the four points -/
def p1 (a : ℝ) : Point3D := ⟨2, 0, a⟩
def p2 (b : ℝ) : Point3D := ⟨b, 2, 0⟩
def p3 (c : ℝ) : Point3D := ⟨0, c, 2⟩
def p4 (d : ℝ) : Point3D := ⟨8*d, 8*d, -2*d⟩

/-- Define collinearity for four points -/
def collinear (p q r s : Point3D) : Prop :=
  ∃ (t₁ t₂ t₃ : ℝ), (q.x - p.x, q.y - p.y, q.z - p.z) = t₁ • (r.x - p.x, r.y - p.y, r.z - p.z)
                 ∧ (q.x - p.x, q.y - p.y, q.z - p.z) = t₂ • (s.x - p.x, s.y - p.y, s.z - p.z)
                 ∧ (r.x - p.x, r.y - p.y, r.z - p.z) = t₃ • (s.x - p.x, s.y - p.y, s.z - p.z)

/-- The main theorem -/
theorem collinear_points_d_values (a b c d : ℝ) :
  collinear (p1 a) (p2 b) (p3 c) (p4 d) → d = 1/8 ∨ d = -1/32 := by
  sorry

end collinear_points_d_values_l3930_393053


namespace box_height_l3930_393072

/-- Proves that a rectangular box with given dimensions has a height of 3 cm -/
theorem box_height (base_length base_width volume : ℝ) 
  (h1 : base_length = 2)
  (h2 : base_width = 5)
  (h3 : volume = 30) :
  volume / (base_length * base_width) = 3 := by
  sorry

end box_height_l3930_393072


namespace plumbing_job_washers_l3930_393005

/-- Calculates the number of remaining washers after a plumbing job --/
def remaining_washers (total_pipe_length : ℕ) (pipe_per_bolt : ℕ) (washers_per_bolt : ℕ) (total_washers : ℕ) : ℕ :=
  let bolts_needed := total_pipe_length / pipe_per_bolt
  let washers_used := bolts_needed * washers_per_bolt
  total_washers - washers_used

/-- Theorem stating that for the given plumbing job, 4 washers will remain --/
theorem plumbing_job_washers :
  remaining_washers 40 5 2 20 = 4 := by
  sorry

end plumbing_job_washers_l3930_393005


namespace root_product_expression_l3930_393070

theorem root_product_expression (p q : ℝ) (α β γ δ : ℂ) : 
  (α^2 - 2*p*α + 1 = 0) → 
  (β^2 - 2*p*β + 1 = 0) → 
  (γ^2 + q*γ + 2 = 0) → 
  (δ^2 + q*δ + 2 = 0) → 
  (α - γ) * (β - γ) * (α - δ) * (β - δ) = 2*(p - q)^2 :=
by sorry

end root_product_expression_l3930_393070


namespace number_categorization_l3930_393059

def numbers : List ℚ := [7, -3.14, -5, 1/8, 0, -7/4, -4/5]

def is_positive_rational (x : ℚ) : Prop := x > 0

def is_negative_fraction (x : ℚ) : Prop := x < 0 ∧ x ≠ ↑(⌊x⌋)

def is_integer (x : ℚ) : Prop := ∃ n : ℤ, x = ↑n

theorem number_categorization :
  (∀ x ∈ numbers, is_positive_rational x ↔ x ∈ [7, 1/8]) ∧
  (∀ x ∈ numbers, is_negative_fraction x ↔ x ∈ [-3.14, -7/4, -4/5]) ∧
  (∀ x ∈ numbers, is_integer x ↔ x ∈ [7, -5, 0]) :=
sorry

end number_categorization_l3930_393059


namespace kittens_problem_l3930_393060

/-- The number of kittens Tim gave to Jessica -/
def kittens_to_jessica (initial : ℕ) (to_sara : ℕ) (remaining : ℕ) : ℕ :=
  initial - to_sara - remaining

theorem kittens_problem (initial : ℕ) (to_sara : ℕ) (remaining : ℕ) 
  (h1 : initial = 18) 
  (h2 : to_sara = 6) 
  (h3 : remaining = 9) : 
  kittens_to_jessica initial to_sara remaining = 3 := by
  sorry

end kittens_problem_l3930_393060


namespace inverse_proportion_ratio_l3930_393030

theorem inverse_proportion_ratio (a₁ a₂ b₁ b₂ : ℝ) (h1 : a₁ ≠ 0) (h2 : a₂ ≠ 0) (h3 : b₁ ≠ 0) (h4 : b₂ ≠ 0) :
  (∃ k : ℝ, k ≠ 0 ∧ a₁ * b₁ = k ∧ a₂ * b₂ = k) →
  a₁ / a₂ = 3 / 4 →
  b₁ / b₂ = 4 / 3 := by
sorry

end inverse_proportion_ratio_l3930_393030


namespace birthday_cake_cost_l3930_393013

/-- Proves that the cost of the birthday cake is $25 given the conditions of Erika and Rick's gift-buying scenario. -/
theorem birthday_cake_cost (gift_cost : ℝ) (erika_savings : ℝ) (rick_savings : ℝ) (leftover : ℝ) :
  gift_cost = 250 →
  erika_savings = 155 →
  rick_savings = gift_cost / 2 →
  leftover = 5 →
  erika_savings + rick_savings - gift_cost - leftover = 25 := by
  sorry

#check birthday_cake_cost

end birthday_cake_cost_l3930_393013


namespace simplify_square_roots_l3930_393058

theorem simplify_square_roots : 16^(1/2) - 625^(1/2) = -21 := by
  sorry

end simplify_square_roots_l3930_393058


namespace calculate_expression_l3930_393090

theorem calculate_expression : 3000 * (3000 ^ 3000) + 3000 ^ 2 = 3000 ^ 3001 + 9000000 := by
  sorry

end calculate_expression_l3930_393090


namespace purely_imaginary_complex_number_l3930_393079

theorem purely_imaginary_complex_number (a : ℝ) : 
  let z : ℂ := Complex.mk (a^2 + 2*a - 3) (a + 3)
  (z.re = 0 ∧ z.im ≠ 0) → a = 1 := by
sorry

end purely_imaginary_complex_number_l3930_393079


namespace count_pairs_eq_28_l3930_393026

def count_pairs : ℕ :=
  (Finset.range 7).sum (λ m =>
    (Finset.range (8 - m)).card)

theorem count_pairs_eq_28 : count_pairs = 28 := by
  sorry

end count_pairs_eq_28_l3930_393026


namespace remainder_of_2614303940317_div_13_l3930_393037

theorem remainder_of_2614303940317_div_13 : 
  2614303940317 % 13 = 4 := by sorry

end remainder_of_2614303940317_div_13_l3930_393037


namespace inequality_proof_l3930_393012

theorem inequality_proof (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (Real.sqrt a + Real.sqrt b)^8 ≥ 64 * a * b * (a + b)^2 := by
  sorry

end inequality_proof_l3930_393012


namespace circle_f_value_l3930_393076

def Circle (d e f : ℝ) := {p : ℝ × ℝ | p.1^2 + p.2^2 + d * p.1 + e * p.2 + f = 0}

def isDiameter (c : ℝ × ℝ → Prop) (A B : ℝ × ℝ) : Prop :=
  let midpoint := ((A.1 + B.1) / 2, (A.2 + B.2) / 2)
  ∀ p : ℝ × ℝ, c p → 
    (p.1 - midpoint.1)^2 + (p.2 - midpoint.2)^2 ≤ ((A.1 - B.1)^2 + (A.2 - B.2)^2) / 4

theorem circle_f_value (d e f : ℝ) :
  isDiameter (Circle d e f) (20, 22) (10, 30) → f = 860 := by
  sorry

end circle_f_value_l3930_393076


namespace birthday_paradox_l3930_393089

theorem birthday_paradox (n : ℕ) (h : n = 367) :
  ∃ (f : Fin n → Fin 366), ¬Function.Injective f :=
sorry

end birthday_paradox_l3930_393089


namespace binary_of_25_l3930_393087

/-- Represents a binary number as a list of bits (least significant bit first) -/
def BinaryRepr := List Bool

/-- Converts a natural number to its binary representation -/
def toBinary (n : ℕ) : BinaryRepr :=
  if n = 0 then [] else (n % 2 = 1) :: toBinary (n / 2)

/-- Theorem: The binary representation of 25 is 11001 -/
theorem binary_of_25 :
  toBinary 25 = [true, false, false, true, true] := by
  sorry

end binary_of_25_l3930_393087


namespace difference_constant_sum_not_always_minimal_when_equal_l3930_393064

theorem difference_constant_sum_not_always_minimal_when_equal :
  ¬ (∀ (a b : ℝ) (d : ℝ), 
    a > 0 → b > 0 → a - b = d → 
    (∀ (x y : ℝ), x > 0 → y > 0 → x - y = d → a + b ≤ x + y)) :=
sorry

end difference_constant_sum_not_always_minimal_when_equal_l3930_393064


namespace function_characterization_l3930_393021

theorem function_characterization
  (f : ℕ → ℕ)
  (α : ℕ)
  (h1 : ∀ (m n : ℕ), f (m * n^2) = f (m * n) + α * f n)
  (h2 : ∀ (n : ℕ) (p : ℕ), Nat.Prime p → p ∣ n → f p ≠ 0 ∧ f p ∣ f n) :
  ∃ (c : ℕ), 
    (α = 1) ∧
    (∀ (n : ℕ), 
      f n = c * (Nat.factorization n).sum (fun _ e => e)) :=
sorry

end function_characterization_l3930_393021


namespace sector_area_l3930_393085

theorem sector_area (r : ℝ) (θ : ℝ) (h1 : r = 24) (h2 : θ = 110 * π / 180) :
  r^2 * θ / 2 = 176 * π := by
  sorry

end sector_area_l3930_393085


namespace f_properties_l3930_393078

-- Define the function f
noncomputable def f : ℝ → ℝ := λ x =>
  if x < 0 then (x - 1)^2
  else if x = 0 then 0
  else -(x + 1)^2

-- State the theorem
theorem f_properties :
  (∀ x, f (-x) = -f x) ∧  -- f is odd
  (∀ x > 0, f x = -(x + 1)^2) →  -- given condition
  (∀ x < 0, f x = (x - 1)^2) ∧  -- part of the analytic expression
  (f 0 = 0) ∧  -- part of the analytic expression
  (∀ m, f (m^2 + 2*m) + f m > 0 ↔ -3 < m ∧ m < 0) :=  -- range of m
by sorry

end f_properties_l3930_393078


namespace workers_in_first_group_l3930_393000

/-- The number of workers in the first group -/
def W : ℕ := 360

/-- The time taken by the first group to build the wall -/
def T1 : ℕ := 48

/-- The number of workers in the second group -/
def T2 : ℕ := 24

/-- The time taken by the second group to build the wall -/
def W2 : ℕ := 30

/-- Theorem stating that W is the correct number of workers in the first group -/
theorem workers_in_first_group :
  W * T1 = T2 * W2 := by sorry

end workers_in_first_group_l3930_393000


namespace peters_extra_pictures_l3930_393094

theorem peters_extra_pictures (randy_pictures : ℕ) (peter_pictures : ℕ) (quincy_pictures : ℕ) :
  randy_pictures = 5 →
  quincy_pictures = peter_pictures + 20 →
  randy_pictures + peter_pictures + quincy_pictures = 41 →
  peter_pictures - randy_pictures = 3 := by
  sorry

end peters_extra_pictures_l3930_393094


namespace factorization_sum_l3930_393008

theorem factorization_sum (x y : ℝ) : ∃ (a b c d e f g h j k : ℤ),
  (27 * x^9 - 512 * y^9 = (a * x + b * y) * (c * x^3 + d * x * y^2 + e * y^3) * 
                          (f * x + g * y) * (h * x^3 + j * x * y^2 + k * y^3)) ∧
  (a + b + c + d + e + f + g + h + j + k = 12) := by
sorry

end factorization_sum_l3930_393008


namespace rain_probability_l3930_393004

theorem rain_probability (p : ℚ) (n : ℕ) (hp : p = 3/5) (hn : n = 5) :
  1 - (1 - p)^n = 3093/3125 := by
  sorry

end rain_probability_l3930_393004


namespace fourth_power_sum_l3930_393009

theorem fourth_power_sum (x y z : ℝ) 
  (h1 : x + y + z = 2) 
  (h2 : x^2 + y^2 + z^2 = 6) 
  (h3 : x^3 + y^3 + z^3 = 8) : 
  x^4 + y^4 + z^4 = 26 := by
sorry

end fourth_power_sum_l3930_393009


namespace y_increase_proof_l3930_393035

/-- Represents a line in the Cartesian plane -/
structure Line where
  slope : ℝ

/-- Calculates the change in y given a change in x for a line -/
def Line.deltaY (l : Line) (deltaX : ℝ) : ℝ :=
  l.slope * deltaX

theorem y_increase_proof (l : Line) (h : l.deltaY 4 = 6) :
  l.deltaY 12 = 18 := by
  sorry

end y_increase_proof_l3930_393035


namespace taran_number_puzzle_l3930_393043

theorem taran_number_puzzle : ∃ x : ℕ, 
  ((x * 5 + 5 - 5 = 73) ∨ (x * 5 + 5 - 6 = 73) ∨ (x * 5 + 6 - 5 = 73) ∨ (x * 5 + 6 - 6 = 73) ∨
   (x * 6 + 5 - 5 = 73) ∨ (x * 6 + 5 - 6 = 73) ∨ (x * 6 + 6 - 5 = 73) ∨ (x * 6 + 6 - 6 = 73)) ∧
  x = 12 := by
  sorry

end taran_number_puzzle_l3930_393043


namespace second_train_speed_is_16_l3930_393022

/-- The speed of the second train given the conditions of the problem -/
def second_train_speed (first_train_speed : ℝ) (distance_between_stations : ℝ) (distance_difference : ℝ) : ℝ :=
  let v : ℝ := 16  -- Speed of the second train
  v

/-- Theorem stating that under the given conditions, the speed of the second train is 16 km/hr -/
theorem second_train_speed_is_16 :
  second_train_speed 20 495 55 = 16 := by
  sorry

#check second_train_speed_is_16

end second_train_speed_is_16_l3930_393022


namespace tan_alpha_3_implies_sin_2alpha_over_cos_alpha_squared_6_l3930_393083

theorem tan_alpha_3_implies_sin_2alpha_over_cos_alpha_squared_6 (α : Real) 
  (h : Real.tan α = 3) : 
  (Real.sin (2 * α)) / (Real.cos α)^2 = 6 := by
  sorry

end tan_alpha_3_implies_sin_2alpha_over_cos_alpha_squared_6_l3930_393083


namespace sqrt_x_equals_3_x_squared_equals_y_squared_l3930_393007

-- Define x and y as functions of a
def x (a : ℝ) : ℝ := 1 - 2*a
def y (a : ℝ) : ℝ := 3*a - 4

-- Theorem 1: When √x = 3, a = -4
theorem sqrt_x_equals_3 : ∃ a : ℝ, x a = 9 ∧ a = -4 := by sorry

-- Theorem 2: There exist values of a such that x² = y² = 1 or x² = y² = 25
theorem x_squared_equals_y_squared :
  (∃ a : ℝ, (x a)^2 = (y a)^2 ∧ (x a)^2 = 1) ∨
  (∃ a : ℝ, (x a)^2 = (y a)^2 ∧ (x a)^2 = 25) := by sorry

end sqrt_x_equals_3_x_squared_equals_y_squared_l3930_393007


namespace team_selection_combinations_l3930_393015

theorem team_selection_combinations (n m k : ℕ) (hn : n = 5) (hm : m = 5) (hk : k = 3) :
  (Nat.choose (n + m) k) - (Nat.choose n k) = 110 := by
  sorry

end team_selection_combinations_l3930_393015


namespace N_mod_100_l3930_393011

/-- The number of ways to select a group of singers satisfying given conditions -/
def N : ℕ := sorry

/-- The total number of tenors available -/
def num_tenors : ℕ := 8

/-- The total number of basses available -/
def num_basses : ℕ := 10

/-- The total number of singers to be selected -/
def total_singers : ℕ := 6

/-- Predicate to check if a group satisfies the conditions -/
def valid_group (tenors basses : ℕ) : Prop :=
  tenors + basses = total_singers ∧ 
  ∃ k : ℤ, tenors - basses = 4 * k

theorem N_mod_100 : N % 100 = 96 := by sorry

end N_mod_100_l3930_393011


namespace greatest_valid_n_l3930_393003

def is_valid (n : ℕ) : Prop :=
  n ≥ 100 ∧ n ≤ 999 ∧ ¬((Nat.factorial (n / 2)) % (n * (n + 1)) = 0)

theorem greatest_valid_n : 
  (∀ m : ℕ, m > 996 → m ≤ 999 → ¬(is_valid m)) ∧
  is_valid 996 := by sorry

end greatest_valid_n_l3930_393003


namespace exists_divisible_term_l3930_393093

/-- Sequence defined by a₀ = 5 and aₙ₊₁ = 2aₙ + 1 -/
def a : ℕ → ℕ
  | 0 => 5
  | n + 1 => 2 * a n + 1

/-- For every natural number n, there exists a different k such that a_n divides a_k -/
theorem exists_divisible_term (n : ℕ) : ∃ k : ℕ, k ≠ n ∧ a n ∣ a k := by
  sorry

end exists_divisible_term_l3930_393093


namespace function_range_contained_in_unit_interval_l3930_393097

/-- Given a function f: ℝ → ℝ satisfying (f x)^2 ≤ f y for all x > y,
    prove that the range of f is contained in [0, 1]. -/
theorem function_range_contained_in_unit_interval
  (f : ℝ → ℝ) (h : ∀ x y, x > y → (f x)^2 ≤ f y) :
  ∀ x, 0 ≤ f x ∧ f x ≤ 1 := by
  sorry

end function_range_contained_in_unit_interval_l3930_393097


namespace relationship_holds_l3930_393027

/-- A function representing the relationship between x and y -/
def f (x : ℕ) : ℕ := x^2 + 3*x + 1

/-- The set of x values given in the problem -/
def X : Finset ℕ := {1, 2, 3, 4, 5}

/-- The set of y values given in the problem -/
def Y : Finset ℕ := {5, 11, 19, 29, 41}

/-- Theorem stating that the function f correctly relates all given x and y values -/
theorem relationship_holds : ∀ x ∈ X, f x ∈ Y :=
  sorry

end relationship_holds_l3930_393027


namespace incoming_class_size_l3930_393040

theorem incoming_class_size : ∃! n : ℕ, 
  0 < n ∧ n < 1000 ∧ n % 25 = 18 ∧ n % 28 = 26 ∧ n = 418 := by
  sorry

end incoming_class_size_l3930_393040


namespace reciprocal_of_negative_half_l3930_393084

theorem reciprocal_of_negative_half : ((-1/2)⁻¹ : ℚ) = -2 := by sorry

end reciprocal_of_negative_half_l3930_393084


namespace perpendicular_lines_from_perpendicular_planes_l3930_393044

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular relation between lines and planes
variable (perp_line_plane : Line → Plane → Prop)

-- Define the perpendicular relation between planes
variable (perp_plane : Plane → Plane → Prop)

-- Define the perpendicular relation between lines
variable (perp_line : Line → Line → Prop)

-- Theorem statement
theorem perpendicular_lines_from_perpendicular_planes 
  (m n : Line) (α β : Plane) :
  perp_line_plane m α → 
  perp_line_plane n β → 
  perp_plane α β → 
  perp_line m n :=
sorry

end perpendicular_lines_from_perpendicular_planes_l3930_393044


namespace sum_of_a_and_d_l3930_393024

theorem sum_of_a_and_d (a b c d : ℤ) 
  (eq1 : a + b = 5)
  (eq2 : b + c = 6)
  (eq3 : c + d = 3) :
  a + d = 2 := by
sorry

end sum_of_a_and_d_l3930_393024


namespace unique_consecutive_sum_18_l3930_393039

/-- The sum of n consecutive integers starting from a -/
def consecutiveSum (a n : ℕ) : ℕ := n * (2 * a + n - 1) / 2

/-- A predicate that checks if a set of consecutive integers sums to 18 -/
def isValidSet (a n : ℕ) : Prop :=
  n ≥ 3 ∧ consecutiveSum a n = 18

theorem unique_consecutive_sum_18 :
  ∃! p : ℕ × ℕ, isValidSet p.1 p.2 :=
sorry

end unique_consecutive_sum_18_l3930_393039


namespace tangent_line_parallel_points_l3930_393045

def f (x : ℝ) : ℝ := x^3 - 2

theorem tangent_line_parallel_points :
  ∀ x y : ℝ, f x = y →
  (∃ k : ℝ, k * (x - 1) = y + 1 ∧ k = 3) ↔ 
  ((x = 1 ∧ y = -1) ∨ (x = -1 ∧ y = -3)) :=
by sorry

end tangent_line_parallel_points_l3930_393045


namespace solution_set_reciprocal_inequality_l3930_393019

theorem solution_set_reciprocal_inequality (x : ℝ) :
  (1 / x > 1) ↔ (0 < x ∧ x < 1) :=
by sorry

end solution_set_reciprocal_inequality_l3930_393019


namespace hash_sum_plus_six_l3930_393066

def hash (a b : ℕ) : ℕ := 4 * a^2 + 4 * b^2 + 8 * a * b

theorem hash_sum_plus_six (a b : ℕ) (h : hash a b = 100) : (a + b) + 6 = 11 := by
  sorry

end hash_sum_plus_six_l3930_393066


namespace reconstructed_text_is_correct_l3930_393069

-- Define the set of original characters
def OriginalChars : Set Char := {'А', 'Б', 'В', 'Г', 'Д', 'Е', 'Ё', 'Ж', 'З', 'И', 'Й', 'К', 'Л', 'М', 'Н', 'О', 'П', 'Р', 'С', 'Т', 'У', 'Ф', 'Х', 'Ц', 'Ч', 'Ш', 'Щ', 'Ъ', 'Ы', 'Ь', 'Э', 'Ю', 'Я'}

-- Define a mapping from distorted characters to original characters
def DistortedToOriginal : Char → Char := sorry

-- Define the reconstructed text
def ReconstructedText : String := "глобальное потепление"

-- Theorem stating that the reconstructed text is correct
theorem reconstructed_text_is_correct :
  ∀ c ∈ ReconstructedText.data, DistortedToOriginal c ∈ OriginalChars :=
sorry

#check reconstructed_text_is_correct

end reconstructed_text_is_correct_l3930_393069


namespace speed_ratio_l3930_393056

/-- Two cars traveling towards each other with constant speeds -/
structure TwoCars where
  v1 : ℝ  -- Speed of the first car
  v2 : ℝ  -- Speed of the second car
  d : ℝ   -- Distance between points A and B
  t : ℝ   -- Time until the cars meet

/-- The conditions of the problem -/
def MeetingConditions (cars : TwoCars) : Prop :=
  cars.v1 > 0 ∧ cars.v2 > 0 ∧ cars.d > 0 ∧ cars.t > 0 ∧
  cars.v1 * cars.t + cars.v2 * cars.t = cars.d ∧
  cars.d - cars.v1 * cars.t = cars.v1 ∧
  cars.d - cars.v2 * cars.t = 4 * cars.v2

/-- The theorem stating the ratio of speeds -/
theorem speed_ratio (cars : TwoCars) 
  (h : MeetingConditions cars) : cars.v1 / cars.v2 = 2 := by
  sorry

end speed_ratio_l3930_393056


namespace current_trees_count_l3930_393001

/-- The number of dogwood trees currently in the park -/
def current_trees : ℕ := sorry

/-- The number of trees to be planted today -/
def trees_today : ℕ := 3

/-- The number of trees to be planted tomorrow -/
def trees_tomorrow : ℕ := 2

/-- The total number of trees after planting -/
def total_trees : ℕ := 12

/-- Proof that the current number of trees is 7 -/
theorem current_trees_count : current_trees = 7 := by
  sorry

end current_trees_count_l3930_393001


namespace symmetry_y_axis_symmetry_x_axis_symmetry_origin_area_greater_than_pi_l3930_393034

-- Define the curve C
def C (x y : ℝ) : Prop := x^4 + y^2 = 1

-- Symmetry about y=0
theorem symmetry_y_axis (x y : ℝ) : C x y ↔ C x (-y) := by sorry

-- Symmetry about x=0
theorem symmetry_x_axis (x y : ℝ) : C x y ↔ C (-x) y := by sorry

-- Symmetry about (0,0)
theorem symmetry_origin (x y : ℝ) : C x y ↔ C (-x) (-y) := by sorry

-- Define the area of C
noncomputable def area_C : ℝ := sorry

-- Area of C is greater than π
theorem area_greater_than_pi : area_C > π := by sorry

end symmetry_y_axis_symmetry_x_axis_symmetry_origin_area_greater_than_pi_l3930_393034


namespace sandy_second_shop_amount_l3930_393018

/-- The amount Sandy paid for books from the second shop -/
def second_shop_amount (first_shop_books : ℕ) (second_shop_books : ℕ) 
  (first_shop_amount : ℚ) (average_price : ℚ) : ℚ :=
  (first_shop_books + second_shop_books) * average_price - first_shop_amount

/-- Proof that Sandy paid $900 for books from the second shop -/
theorem sandy_second_shop_amount :
  second_shop_amount 65 55 1380 19 = 900 := by
  sorry

end sandy_second_shop_amount_l3930_393018


namespace donation_difference_l3930_393096

def total_donation : ℕ := 1000
def treetown_forest_donation : ℕ := 570

theorem donation_difference : 
  treetown_forest_donation - (total_donation - treetown_forest_donation) = 140 := by
  sorry

end donation_difference_l3930_393096


namespace unique_albums_count_l3930_393006

/-- Represents the album collections of Andrew, John, and Bella -/
structure AlbumCollections where
  andrew_total : ℕ
  andrew_john_shared : ℕ
  john_unique : ℕ
  bella_andrew_overlap : ℕ

/-- Calculates the number of unique albums not shared among any two people -/
def unique_albums (collections : AlbumCollections) : ℕ :=
  (collections.andrew_total - collections.andrew_john_shared) + collections.john_unique

/-- Theorem stating that the number of unique albums is 18 given the problem conditions -/
theorem unique_albums_count (collections : AlbumCollections)
  (h1 : collections.andrew_total = 20)
  (h2 : collections.andrew_john_shared = 10)
  (h3 : collections.john_unique = 8)
  (h4 : collections.bella_andrew_overlap = 5)
  (h5 : collections.bella_andrew_overlap ≤ collections.andrew_total - collections.andrew_john_shared) :
  unique_albums collections = 18 := by
  sorry

#eval unique_albums { andrew_total := 20, andrew_john_shared := 10, john_unique := 8, bella_andrew_overlap := 5 }

end unique_albums_count_l3930_393006


namespace fifteen_factorial_base_eight_zeroes_l3930_393077

/-- The number of trailing zeroes in n! when written in base b -/
def trailingZeroes (n : ℕ) (b : ℕ) : ℕ :=
  sorry

/-- Theorem: 15! ends with 5 zeroes when written in base 8 -/
theorem fifteen_factorial_base_eight_zeroes :
  trailingZeroes 15 8 = 5 := by
  sorry

end fifteen_factorial_base_eight_zeroes_l3930_393077


namespace solution_set_a_3_range_of_a_non_negative_l3930_393029

-- Define the function f
def f (a x : ℝ) : ℝ := |x^2 - 2*x + a - 1| - a^2 - 2*a

-- Theorem 1: Solution set when a = 3
theorem solution_set_a_3 :
  {x : ℝ | f 3 x ≥ -10} = {x : ℝ | x ≥ 3 ∨ x ≤ -1} :=
sorry

-- Theorem 2: Range of a for f(x) ≥ 0 for all x
theorem range_of_a_non_negative :
  {a : ℝ | ∀ x, f a x ≥ 0} = {a : ℝ | -2 ≤ a ∧ a ≤ 0} :=
sorry

end solution_set_a_3_range_of_a_non_negative_l3930_393029


namespace draws_calculation_l3930_393014

def total_games : ℕ := 14
def wins : ℕ := 2
def losses : ℕ := 2

theorem draws_calculation : total_games - (wins + losses) = 10 := by
  sorry

end draws_calculation_l3930_393014


namespace robinson_family_has_six_children_l3930_393081

/-- Represents the Robinson family -/
structure RobinsonFamily where
  num_children : ℕ
  father_age : ℕ
  mother_age : ℕ
  children_ages : List ℕ

/-- The average age of a list of ages -/
def average_age (ages : List ℕ) : ℚ :=
  (ages.sum : ℚ) / ages.length

/-- The properties of the Robinson family -/
def is_robinson_family (family : RobinsonFamily) : Prop :=
  let total_ages := family.mother_age :: family.father_age :: family.children_ages
  average_age total_ages = 22 ∧
  family.father_age = 50 ∧
  average_age (family.mother_age :: family.children_ages) = 18

theorem robinson_family_has_six_children :
  ∀ family : RobinsonFamily, is_robinson_family family → family.num_children = 6 :=
by sorry

end robinson_family_has_six_children_l3930_393081


namespace binomial_expansion_problem_l3930_393088

theorem binomial_expansion_problem (m n : ℕ) (hm : m ≠ 0) (hn : n ≥ 2) :
  (∀ k, 0 ≤ k ∧ k ≤ n → (n.choose k) * m^k ≤ (n.choose 5) * m^5) ∧
  (n.choose 2) * m^2 = 9 * (n.choose 1) * m →
  m = 2 ∧ n = 10 ∧ (1 - 2 * 9)^10 % 6 = 1 :=
sorry

end binomial_expansion_problem_l3930_393088


namespace max_b_value_max_b_value_achieved_l3930_393049

theorem max_b_value (x b : ℤ) (h1 : x^2 + b*x = -21) (h2 : b > 0) : b ≤ 22 := by
  sorry

theorem max_b_value_achieved : ∃ x b : ℤ, x^2 + b*x = -21 ∧ b > 0 ∧ b = 22 := by
  sorry

end max_b_value_max_b_value_achieved_l3930_393049


namespace train_speed_l3930_393071

/-- The speed of a train given its length and time to pass a stationary point -/
theorem train_speed (length time : ℝ) (h1 : length = 160) (h2 : time = 8) :
  length / time = 20 := by
  sorry

end train_speed_l3930_393071


namespace min_p_plus_q_l3930_393075

theorem min_p_plus_q (p q : ℕ) : 
  p > 1 → q > 1 → 17 * (p + 1) = 21 * (q + 1) → 
  ∀ (p' q' : ℕ), p' > 1 → q' > 1 → 17 * (p' + 1) = 21 * (q' + 1) → 
  p + q ≤ p' + q' :=
by
  sorry

end min_p_plus_q_l3930_393075


namespace inequality_proof_l3930_393062

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  x^2 + y^4 + z^6 ≥ x*y^2 + y^2*z^3 + x*z^3 := by
  sorry

end inequality_proof_l3930_393062


namespace divisibility_by_five_l3930_393033

theorem divisibility_by_five (m n : ℕ) : 
  (∃ k : ℕ, m * n = 5 * k) → (∃ j : ℕ, m = 5 * j) ∨ (∃ l : ℕ, n = 5 * l) :=
by sorry

end divisibility_by_five_l3930_393033


namespace arithmetic_expression_equality_l3930_393095

theorem arithmetic_expression_equality : 60 + 5 * 12 / (180 / 3) = 61 := by
  sorry

end arithmetic_expression_equality_l3930_393095


namespace toys_per_day_l3930_393055

/-- Given a factory that produces toys, this theorem proves the number of toys produced each day. -/
theorem toys_per_day 
  (total_toys : ℕ)           -- Total number of toys produced per week
  (work_days : ℕ)            -- Number of work days per week
  (h1 : total_toys = 4560)   -- The factory produces 4560 toys per week
  (h2 : work_days = 4)       -- Workers work 4 days a week
  (h3 : total_toys % work_days = 0)  -- The number of toys produced is the same each day
  : total_toys / work_days = 1140 := by
  sorry

#check toys_per_day

end toys_per_day_l3930_393055


namespace gross_profit_calculation_l3930_393052

theorem gross_profit_calculation (sales_price : ℝ) (gross_profit_percentage : ℝ) :
  sales_price = 44 ∧ gross_profit_percentage = 1.2 →
  ∃ (cost : ℝ) (gross_profit : ℝ),
    sales_price = cost + gross_profit ∧
    gross_profit = gross_profit_percentage * cost ∧
    gross_profit = 24 := by
  sorry

end gross_profit_calculation_l3930_393052


namespace sets_problem_l3930_393065

-- Define the universe set U as the set of all real numbers
def U : Set ℝ := Set.univ

-- Define set A
def A : Set ℝ := {x | 0 ≤ x ∧ x ≤ 2}

-- Define set B (parameterized by a)
def B (a : ℝ) : Set ℝ := {x | a ≤ x ∧ x ≤ 3 - 2*a}

theorem sets_problem (a : ℝ) :
  (((Set.compl A) ∪ (B a)) = U ↔ a ≤ 0) ∧
  ((A ∩ (B a)) = (B a) ↔ a ≥ (1/2)) := by
  sorry


end sets_problem_l3930_393065


namespace peach_price_is_40_cents_l3930_393068

/-- Represents the store's discount policy -/
def discount_rate : ℚ := 2 / 10

/-- Represents the number of peaches bought -/
def num_peaches : ℕ := 400

/-- Represents the total amount paid after discount -/
def total_paid : ℚ := 128

/-- Calculates the price of each peach -/
def price_per_peach : ℚ :=
  let total_before_discount := total_paid / (1 - discount_rate)
  total_before_discount / num_peaches

/-- Proves that the price of each peach is $0.40 -/
theorem peach_price_is_40_cents : price_per_peach = 0.40 := by
  sorry

end peach_price_is_40_cents_l3930_393068


namespace weight_of_B_l3930_393017

theorem weight_of_B (A B C : ℝ) 
  (h1 : (A + B + C) / 3 = 45)
  (h2 : (A + B) / 2 = 40)
  (h3 : (B + C) / 2 = 44) :
  B = 33 := by
sorry

end weight_of_B_l3930_393017


namespace least_possible_bananas_l3930_393074

/-- Represents the distribution of bananas among three monkeys. -/
structure BananaDistribution where
  b₁ : ℕ  -- bananas taken by first monkey
  b₂ : ℕ  -- bananas taken by second monkey
  b₃ : ℕ  -- bananas taken by third monkey

/-- Checks if the given distribution satisfies all conditions of the problem. -/
def isValidDistribution (d : BananaDistribution) : Prop :=
  let m₁ := (2 * d.b₁) / 3 + d.b₂ / 3 + (7 * d.b₃) / 16
  let m₂ := d.b₁ / 6 + d.b₂ / 3 + (7 * d.b₃) / 16
  let m₃ := d.b₁ / 6 + d.b₂ / 3 + d.b₃ / 8
  (∀ n : ℕ, n ∈ [m₁, m₂, m₃] → n > 0) ∧  -- whole number condition
  5 * m₂ = 3 * m₁ ∧ 5 * m₃ = 2 * m₁       -- ratio condition

/-- The theorem stating the least possible total number of bananas. -/
theorem least_possible_bananas :
  ∃ (d : BananaDistribution),
    isValidDistribution d ∧
    d.b₁ + d.b₂ + d.b₃ = 336 ∧
    (∀ d' : BananaDistribution, isValidDistribution d' → d'.b₁ + d'.b₂ + d'.b₃ ≥ 336) :=
  sorry

end least_possible_bananas_l3930_393074


namespace cube_root_of_cube_l3930_393099

theorem cube_root_of_cube (x : ℝ) : x^(1/3)^3 = x := by
  sorry

end cube_root_of_cube_l3930_393099


namespace jermaine_earnings_difference_l3930_393028

def total_earnings : ℕ := 90
def terrence_earnings : ℕ := 30
def emilee_earnings : ℕ := 25

theorem jermaine_earnings_difference : 
  ∃ (jermaine_earnings : ℕ), 
    jermaine_earnings > terrence_earnings ∧
    jermaine_earnings + terrence_earnings + emilee_earnings = total_earnings ∧
    jermaine_earnings - terrence_earnings = 5 :=
by sorry

end jermaine_earnings_difference_l3930_393028


namespace hyperbola_foci_l3930_393010

/-- The hyperbola equation --/
def hyperbola_eq (x y : ℝ) : Prop := y^2 - x^2/3 = 1

/-- The focus coordinates --/
def focus_coords : Set (ℝ × ℝ) := {(0, 2), (0, -2)}

/-- Theorem: The given coordinates are the foci of the hyperbola --/
theorem hyperbola_foci : 
  ∀ (x y : ℝ), hyperbola_eq x y ↔ ∃ (f : ℝ × ℝ), f ∈ focus_coords ∧ 
    (x - f.1)^2 + (y - f.2)^2 = ((x + f.1)^2 + (y + f.2)^2) :=
sorry

end hyperbola_foci_l3930_393010


namespace quadratic_integer_values_l3930_393073

theorem quadratic_integer_values (a b c : ℝ) :
  (∀ x : ℤ, ∃ n : ℤ, a * x^2 + b * x + c = n) ↔
  (∃ m : ℤ, 2 * a = m) ∧ (∃ n : ℤ, a + b = n) ∧ (∃ p : ℤ, c = p) :=
by sorry

end quadratic_integer_values_l3930_393073


namespace pencil_distribution_l3930_393020

theorem pencil_distribution (num_students : ℕ) (num_pencils : ℕ) 
  (h1 : num_students = 2) 
  (h2 : num_pencils = 18) :
  num_pencils / num_students = 9 := by
  sorry

end pencil_distribution_l3930_393020


namespace tens_digit_of_7_power_2011_l3930_393050

theorem tens_digit_of_7_power_2011 : 7^2011 % 100 = 43 := by
  sorry

end tens_digit_of_7_power_2011_l3930_393050
