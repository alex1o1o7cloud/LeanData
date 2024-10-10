import Mathlib

namespace jan_cable_purchase_l31_3164

theorem jan_cable_purchase (section_length : ℕ) (sections_on_hand : ℕ) : 
  section_length = 25 →
  sections_on_hand = 15 →
  (4 : ℚ) * sections_on_hand = 3 * (2 * sections_on_hand) →
  (4 : ℚ) * sections_on_hand * section_length = 1000 := by
  sorry

end jan_cable_purchase_l31_3164


namespace jinas_mascots_l31_3155

/-- The number of mascots Jina has -/
def total_mascots (original_teddies bunny_to_teddy_ratio koala_bears additional_teddies_per_bunny : ℕ) : ℕ :=
  let bunnies := original_teddies * bunny_to_teddy_ratio
  let additional_teddies := bunnies * additional_teddies_per_bunny
  original_teddies + bunnies + koala_bears + additional_teddies

/-- Theorem stating the total number of mascots Jina has -/
theorem jinas_mascots : total_mascots 5 3 1 2 = 51 := by
  sorry

end jinas_mascots_l31_3155


namespace shooting_score_proof_l31_3107

theorem shooting_score_proof (total_shots : ℕ) (total_score : ℕ) (ten_point_shots : ℕ) (remaining_shots : ℕ) :
  total_shots = 10 →
  total_score = 90 →
  ten_point_shots = 4 →
  remaining_shots = total_shots - ten_point_shots →
  (∃ (seven_point_shots eight_point_shots nine_point_shots : ℕ),
    seven_point_shots + eight_point_shots + nine_point_shots = remaining_shots ∧
    7 * seven_point_shots + 8 * eight_point_shots + 9 * nine_point_shots = total_score - 10 * ten_point_shots) →
  (∃ (nine_point_shots : ℕ), nine_point_shots = 3) :=
by sorry

end shooting_score_proof_l31_3107


namespace insect_meeting_point_l31_3133

/-- Triangle PQR with given side lengths -/
structure Triangle (PQ QR PR : ℝ) where
  positive : 0 < PQ ∧ 0 < QR ∧ 0 < PR
  triangle_inequality : PQ + QR > PR ∧ QR + PR > PQ ∧ PR + PQ > QR

/-- Point S where insects meet -/
def MeetingPoint (t : Triangle PQ QR PR) := 
  {S : ℝ // 0 ≤ S ∧ S ≤ QR}

/-- Theorem stating that QS = 5 under given conditions -/
theorem insect_meeting_point 
  (t : Triangle 7 8 9) 
  (S : MeetingPoint t) : 
  S.val = 5 := by sorry

end insect_meeting_point_l31_3133


namespace ceiling_floor_product_l31_3114

theorem ceiling_floor_product (y : ℝ) : 
  y < 0 → ⌈y⌉ * ⌊y⌋ = 132 → -12 < y ∧ y < -11 := by
  sorry

end ceiling_floor_product_l31_3114


namespace interior_angles_increase_l31_3189

/-- The sum of interior angles of a convex polygon with n sides -/
def sum_interior_angles (n : ℕ) : ℝ := 180 * (n - 2)

theorem interior_angles_increase (n : ℕ) :
  sum_interior_angles n = 1620 → sum_interior_angles (n + 3) = 2160 := by
  sorry

end interior_angles_increase_l31_3189


namespace garden_perimeter_garden_perimeter_proof_l31_3144

/-- The perimeter of a rectangular garden with width 24 meters and the same area as a rectangular playground of length 16 meters and width 12 meters is equal to 64 meters. -/
theorem garden_perimeter : ℝ → ℝ → ℝ → ℝ → Prop :=
  fun (garden_width playground_length playground_width garden_perimeter : ℝ) =>
    garden_width = 24 ∧
    playground_length = 16 ∧
    playground_width = 12 ∧
    garden_width * (playground_length * playground_width / garden_width) = playground_length * playground_width →
    garden_perimeter = 2 * (garden_width + (playground_length * playground_width / garden_width)) →
    garden_perimeter = 64

/-- Proof of the garden_perimeter theorem -/
theorem garden_perimeter_proof : garden_perimeter 24 16 12 64 := by
  sorry

end garden_perimeter_garden_perimeter_proof_l31_3144


namespace greatest_7_power_divisor_l31_3197

/-- The number of divisors of a positive integer -/
def num_divisors (n : ℕ+) : ℕ := sorry

/-- n is divisible by 7^k -/
def divides_by_7_pow (n : ℕ+) (k : ℕ) : Prop := sorry

theorem greatest_7_power_divisor (n : ℕ+) (h1 : num_divisors n = 30) (h2 : num_divisors (7 * n) = 42) :
  ∃ k : ℕ, divides_by_7_pow n k ∧ k = 1 ∧ ∀ m : ℕ, divides_by_7_pow n m → m ≤ k :=
sorry

end greatest_7_power_divisor_l31_3197


namespace K2Cr2O7_molecular_weight_l31_3199

/-- Calculates the molecular weight of a compound given the number of atoms and their atomic weights -/
def molecular_weight (K_atoms Cr_atoms O_atoms : ℕ) (K_weight Cr_weight O_weight : ℝ) : ℝ :=
  K_atoms * K_weight + Cr_atoms * Cr_weight + O_atoms * O_weight

/-- The molecular weight of the compound K₂Cr₂O₇ is 294.192 g/mol -/
theorem K2Cr2O7_molecular_weight : 
  molecular_weight 2 2 7 39.10 51.996 16.00 = 294.192 := by
  sorry

#eval molecular_weight 2 2 7 39.10 51.996 16.00

end K2Cr2O7_molecular_weight_l31_3199


namespace negation_of_perpendicular_plane_l31_3162

-- Define the concept of a line
variable (Line : Type)

-- Define the concept of a plane
variable (Plane : Type)

-- Define what it means for a plane to be perpendicular to a line
variable (perpendicular : Plane → Line → Prop)

-- State the theorem
theorem negation_of_perpendicular_plane :
  (¬ ∀ l : Line, ∃ α : Plane, perpendicular α l) ↔ 
  (∃ l : Line, ∀ α : Plane, ¬ perpendicular α l) :=
by sorry

end negation_of_perpendicular_plane_l31_3162


namespace a_equals_seven_l31_3183

theorem a_equals_seven (A B : Set ℝ) (a : ℝ) : 
  A = {1, 2, a} → B = {1, 7} → B ⊆ A → a = 7 := by
  sorry

end a_equals_seven_l31_3183


namespace f_order_magnitude_l31_3188

-- Define f as a function from ℝ to ℝ
variable (f : ℝ → ℝ)

-- Define the properties of f
def is_even (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x
def is_increasing_on_nonneg (f : ℝ → ℝ) : Prop := ∀ x y, 0 ≤ x → x < y → f x < f y

-- State the theorem
theorem f_order_magnitude 
  (h1 : is_even f) 
  (h2 : is_increasing_on_nonneg f) : 
  f (-π) > f 3 ∧ f 3 > f (-2) :=
sorry

end f_order_magnitude_l31_3188


namespace positive_A_value_l31_3156

-- Define the # relation
def hash (A B : ℝ) : ℝ := A^2 + B^2

-- Theorem statement
theorem positive_A_value (A : ℝ) (h1 : hash A 7 = 130) (h2 : A > 0) : A = 9 := by
  sorry

end positive_A_value_l31_3156


namespace power_function_properties_l31_3172

noncomputable def f (x : ℝ) : ℝ := x ^ (1/2)

theorem power_function_properties :
  ∀ (x₁ x₂ : ℝ), 0 < x₁ ∧ x₁ < x₂ →
    (x₁ * f x₁ < x₂ * f x₂) ∧
    (f x₁ / x₁ > f x₂ / x₂) :=
by sorry

end power_function_properties_l31_3172


namespace angle_bisector_sum_l31_3148

-- Define the triangle vertices
def P : ℝ × ℝ := (-8, 2)
def Q : ℝ × ℝ := (-10, -10)
def R : ℝ × ℝ := (2, -4)

-- Define the angle bisector equation coefficients
def b : ℝ := sorry
def d : ℝ := sorry

-- State the theorem
theorem angle_bisector_sum (h : ∀ (x y : ℝ), b * x + 2 * y + d = 0 ↔ 
  (y - P.2) = (y - P.2) / (x - P.1) * (x - P.1)) : 
  abs (b + d + 64.226) < 0.001 := by sorry

end angle_bisector_sum_l31_3148


namespace complement_of_B_l31_3104

-- Define the set B
def B : Set ℝ := {x | x^2 - 3*x + 2 < 0}

-- Define the universal set U as the set of real numbers
def U : Set ℝ := Set.univ

-- State the theorem
theorem complement_of_B : 
  Set.compl B = {x : ℝ | x ≤ 1 ∨ x ≥ 2} := by sorry

end complement_of_B_l31_3104


namespace sixth_candy_to_pete_l31_3142

/-- Represents the recipients of candy wrappers -/
inductive Recipient : Type
  | Pete : Recipient
  | Vasey : Recipient

/-- Represents the sequence of candy wrapper distributions -/
def CandySequence : Fin 6 → Recipient
  | ⟨0, _⟩ => Recipient.Pete
  | ⟨1, _⟩ => Recipient.Pete
  | ⟨2, _⟩ => Recipient.Pete
  | ⟨3, _⟩ => Recipient.Vasey
  | ⟨4, _⟩ => Recipient.Vasey
  | ⟨5, _⟩ => Recipient.Pete

theorem sixth_candy_to_pete :
  CandySequence ⟨5, by norm_num⟩ = Recipient.Pete := by sorry

end sixth_candy_to_pete_l31_3142


namespace parallelogram_sides_sum_l31_3161

theorem parallelogram_sides_sum (x y : ℝ) : 
  (5 * x - 7 = 14) → 
  (3 * y + 4 = 8 * y - 3) → 
  x + y = 5.6 := by
sorry

end parallelogram_sides_sum_l31_3161


namespace function_satisfies_equation_l31_3135

noncomputable def f (x : ℝ) : ℝ := x + 1/x + 1/(x-1)

theorem function_satisfies_equation :
  ∀ x : ℝ, x ≠ 0 ∧ x ≠ 1 →
    f ((x - 1) / x) + f (1 / (1 - x)) = 2 - 2 * x :=
by
  sorry

end function_satisfies_equation_l31_3135


namespace cos_96_cos_24_minus_sin_96_cos_66_l31_3145

theorem cos_96_cos_24_minus_sin_96_cos_66 : 
  Real.cos (96 * π / 180) * Real.cos (24 * π / 180) - 
  Real.sin (96 * π / 180) * Real.cos (66 * π / 180) = -1/2 := by
  sorry

end cos_96_cos_24_minus_sin_96_cos_66_l31_3145


namespace three_zeros_implies_a_equals_four_l31_3117

-- Define the function f
def f (x a : ℝ) : ℝ := |x^2 - 4*x| - a

-- State the theorem
theorem three_zeros_implies_a_equals_four :
  (∃ (x₁ x₂ x₃ : ℝ), x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₂ ≠ x₃ ∧ 
    f x₁ a = 0 ∧ f x₂ a = 0 ∧ f x₃ a = 0 ∧
    (∀ x : ℝ, f x a = 0 → x = x₁ ∨ x = x₂ ∨ x = x₃)) →
  a = 4 :=
sorry

end three_zeros_implies_a_equals_four_l31_3117


namespace sequence_always_terminates_l31_3170

def last_digit (n : ℕ) : ℕ := n % 10

def remove_last_digit (n : ℕ) : ℕ := n / 10

def next_term (n : ℕ) : ℕ :=
  if n ≤ 5 then n
  else if last_digit n ≤ 5 then remove_last_digit n
  else 9 * n

def sequence_terminates (a₀ : ℕ) : Prop :=
  ∃ n : ℕ, (Nat.iterate next_term n a₀) ≤ 5

theorem sequence_always_terminates (a₀ : ℕ) : sequence_terminates a₀ := by
  sorry

#check sequence_always_terminates

end sequence_always_terminates_l31_3170


namespace sum_of_roots_quadratic_l31_3130

theorem sum_of_roots_quadratic : ∀ (a b c : ℝ), a ≠ 0 → 
  (∃ x y : ℝ, x ≠ y ∧ a * x^2 + b * x + c = 0 ∧ a * y^2 + b * y + c = 0) →
  (∃ s : ℝ, s = -(b / a) ∧ s = 7) := by
sorry

end sum_of_roots_quadratic_l31_3130


namespace cindy_calculation_l31_3134

theorem cindy_calculation (h : 50^2 = 2500) : 50^2 - 49^2 = 99 := by
  sorry

end cindy_calculation_l31_3134


namespace coconut_price_l31_3132

/-- The price of a coconut given the yield per tree, total money needed, and number of trees to harvest. -/
theorem coconut_price
  (yield_per_tree : ℕ)  -- Number of coconuts per tree
  (total_money : ℕ)     -- Total money needed in dollars
  (trees_to_harvest : ℕ) -- Number of trees to harvest
  (h1 : yield_per_tree = 5)
  (h2 : total_money = 90)
  (h3 : trees_to_harvest = 6) :
  total_money / (yield_per_tree * trees_to_harvest) = 3 :=
by sorry


end coconut_price_l31_3132


namespace correct_average_after_error_correction_l31_3175

theorem correct_average_after_error_correction 
  (n : Nat) 
  (initial_average : ℚ) 
  (wrong_number correct_number : ℚ) :
  n = 10 →
  initial_average = 5 →
  wrong_number = 26 →
  correct_number = 36 →
  (n : ℚ) * initial_average + (correct_number - wrong_number) = n * 6 :=
by sorry

end correct_average_after_error_correction_l31_3175


namespace tree_spacing_l31_3108

theorem tree_spacing (total_length : ℕ) (num_trees : ℕ) (tree_space : ℕ) 
  (h1 : total_length = 157)
  (h2 : num_trees = 13)
  (h3 : tree_space = 1) :
  (total_length - num_trees * tree_space) / (num_trees - 1) = 12 :=
sorry

end tree_spacing_l31_3108


namespace equation_solution_l31_3109

theorem equation_solution : ∃ x : ℝ, 
  (1 / (x + 9) + 1 / (x + 7) = 1 / (x + 10) + 1 / (x + 6)) ∧ 
  x = -8 := by sorry

end equation_solution_l31_3109


namespace squirrels_in_tree_l31_3176

theorem squirrels_in_tree (nuts : ℕ) (squirrels : ℕ) : 
  nuts = 2 → squirrels = nuts + 2 → squirrels = 4 := by
  sorry

end squirrels_in_tree_l31_3176


namespace line_not_in_third_quadrant_slope_l31_3195

/-- A line that does not pass through the third quadrant has a non-positive slope -/
theorem line_not_in_third_quadrant_slope (k : ℝ) :
  (∀ x y : ℝ, y = k * x + 3 → ¬(x < 0 ∧ y < 0)) →
  k ≤ 0 := by
  sorry

end line_not_in_third_quadrant_slope_l31_3195


namespace total_precious_stones_l31_3126

theorem total_precious_stones (agate olivine diamond : ℕ) : 
  olivine = agate + 5 →
  diamond = olivine + 11 →
  agate = 30 →
  agate + olivine + diamond = 111 := by
sorry

end total_precious_stones_l31_3126


namespace g_of_3_eq_15_l31_3174

/-- A function g satisfying the given conditions -/
def g (x : ℝ) : ℝ := sorry

/-- The theorem stating that g(3) = 15 -/
theorem g_of_3_eq_15 (h1 : g 1 = 7) (h2 : g 2 = 11) 
  (h3 : ∃ (c d : ℝ), ∀ x, g x = c * x + d * x + 3) : 
  g 3 = 15 := by sorry

end g_of_3_eq_15_l31_3174


namespace not_square_for_prime_l31_3124

theorem not_square_for_prime (p : ℕ) (h_prime : Nat.Prime p) : ¬∃ (a : ℤ), (7 * p + 3^p - 4 : ℤ) = a^2 := by
  sorry

end not_square_for_prime_l31_3124


namespace cuboid_volume_l31_3115

/-- Represents a cuboid with length, width, and height -/
structure Cuboid where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the volume of a cuboid -/
def Cuboid.volume (c : Cuboid) : ℝ :=
  c.length * c.width * c.height

/-- Calculates the surface area of a cuboid -/
def Cuboid.surfaceArea (c : Cuboid) : ℝ :=
  2 * (c.length * c.width + c.length * c.height + c.width * c.height)

/-- Theorem: The volume of the cuboid is 180 cm³ -/
theorem cuboid_volume (c : Cuboid) :
  (∀ (c' : Cuboid), c'.length = c.length ∧ c'.width = c.width ∧ c'.height = c.height + 1 →
    c'.length = c'.width ∧ c'.width = c'.height) →
  (∃ (c' : Cuboid), c'.length = c.length ∧ c'.width = c.width ∧ c'.height = c.height + 1 ∧
    c'.surfaceArea = c.surfaceArea + 24) →
  c.volume = 180 := by
  sorry

end cuboid_volume_l31_3115


namespace circle_condition_l31_3163

theorem circle_condition (a : ℝ) :
  (∃ (x y : ℝ), x^2 + y^2 + 2*x + a = 0) →
  (∃ (h k r : ℝ), ∀ (x y : ℝ), (x - h)^2 + (y - k)^2 = r^2) →
  a < 1 :=
by sorry

end circle_condition_l31_3163


namespace train_speed_l31_3151

/-- Proves that a train of length 480 meters crossing a telegraph post in 16 seconds has a speed of 108 km/h -/
theorem train_speed (train_length : Real) (crossing_time : Real) (speed_kmh : Real) : 
  train_length = 480 ∧ 
  crossing_time = 16 ∧ 
  speed_kmh = (train_length / crossing_time) * 3.6 → 
  speed_kmh = 108 := by
sorry

end train_speed_l31_3151


namespace beth_age_proof_l31_3157

/-- Beth's current age -/
def beth_age : ℕ := 18

/-- Beth's sister's current age -/
def sister_age : ℕ := 5

/-- Years into the future when Beth will be twice her sister's age -/
def future_years : ℕ := 8

theorem beth_age_proof :
  beth_age = 18 ∧
  sister_age = 5 ∧
  beth_age + future_years = 2 * (sister_age + future_years) :=
by sorry

end beth_age_proof_l31_3157


namespace root_of_f_equals_two_max_m_for_inequality_ab_equals_one_l31_3110

noncomputable section

-- Define the function f
def f (a b x : ℝ) : ℝ := a^x + b^x

-- Define the conditions on a and b
class PositiveNotOne (r : ℝ) : Prop where
  pos : r > 0
  not_one : r ≠ 1

-- Theorem 1a
theorem root_of_f_equals_two 
  (h₁ : PositiveNotOne 2) 
  (h₂ : PositiveNotOne (1/2)) :
  ∃ x : ℝ, f 2 (1/2) x = 2 ∧ x = 0 := by sorry

-- Theorem 1b
theorem max_m_for_inequality 
  (h₁ : PositiveNotOne 2) 
  (h₂ : PositiveNotOne (1/2)) :
  ∃ m : ℝ, (∀ x : ℝ, f 2 (1/2) (2*x) ≥ m * f 2 (1/2) x - 6) ∧ 
  (∀ m' : ℝ, (∀ x : ℝ, f 2 (1/2) (2*x) ≥ m' * f 2 (1/2) x - 6) → m' ≤ m) ∧
  m = 4 := by sorry

-- Define function g
def g (a b x : ℝ) : ℝ := f a b x - 2

-- Theorem 2
theorem ab_equals_one 
  (ha : 0 < a ∧ a < 1) 
  (hb : b > 1) 
  (h : PositiveNotOne a) 
  (h' : PositiveNotOne b) 
  (hg : ∃! x : ℝ, g a b x = 0) :
  a * b = 1 := by sorry

end root_of_f_equals_two_max_m_for_inequality_ab_equals_one_l31_3110


namespace w_squared_value_l31_3113

theorem w_squared_value (w : ℝ) (h : (w + 10)^2 = (4*w + 6)*(w + 5)) : w^2 = 70/3 := by
  sorry

end w_squared_value_l31_3113


namespace divisibility_problem_l31_3180

theorem divisibility_problem (a b c : ℤ) 
  (h1 : a ∣ b * c - 1) 
  (h2 : b ∣ c * a - 1) 
  (h3 : c ∣ a * b - 1) : 
  a * b * c ∣ a * b + b * c + c * a - 1 := by
  sorry

end divisibility_problem_l31_3180


namespace stating_magical_stack_size_magical_stack_n_l31_3111

/-- Represents a stack of cards with the described properties. -/
structure CardStack :=
  (n : ℕ)  -- Half the total number of cards
  (is_magical : Bool)  -- Whether the stack is magical after restacking

/-- 
  Theorem stating that a magical stack where card 101 retains its position
  must have 302 cards in total.
-/
theorem magical_stack_size 
  (stack : CardStack) 
  (h_magical : stack.is_magical = true) 
  (h_101_position : ∃ (pos : ℕ), pos ≤ 2 * stack.n ∧ pos = 101) :
  2 * stack.n = 302 := by
  sorry

/-- 
  Corollary: The value of n in a magical stack where card 101 
  retains its position is 151.
-/
theorem magical_stack_n 
  (stack : CardStack) 
  (h_magical : stack.is_magical = true) 
  (h_101_position : ∃ (pos : ℕ), pos ≤ 2 * stack.n ∧ pos = 101) :
  stack.n = 151 := by
  sorry

end stating_magical_stack_size_magical_stack_n_l31_3111


namespace range_of_m_l31_3131

-- Define the propositions p and q
def p (x : ℝ) : Prop := (x + 1) * (x - 3) < 0
def q (x m : ℝ) : Prop := 3 * x - 4 < m

-- Define the necessary but not sufficient condition
def necessary_but_not_sufficient (p q : ℝ → Prop) : Prop :=
  (∀ x, q x → p x) ∧ ∃ x, p x ∧ ¬q x

-- Theorem statement
theorem range_of_m (m : ℝ) :
  (∃ x, q x m) ∧ necessary_but_not_sufficient (p · ) (q · m) ↔ m ≥ 5 :=
sorry

end range_of_m_l31_3131


namespace tangent_line_range_l31_3100

/-- Given a circle and a line, if there exists a point on the line such that
    the tangents from this point to the circle form a 60° angle,
    then the parameter k in the line equation is between -2√2 and 2√2. -/
theorem tangent_line_range (k : ℝ) : 
  (∃ (x y : ℝ), x^2 + y^2 = 1 ∧ x + y + k = 0 ∧ 
   ∃ (p : ℝ × ℝ), p.1 + p.2 + k = 0 ∧ 
   ∃ (a b : ℝ × ℝ), a.1^2 + a.2^2 = 1 ∧ b.1^2 + b.2^2 = 1 ∧ 
   ((p.1 - a.1)*(b.1 - a.1) + (p.2 - a.2)*(b.2 - a.2))^2 = 
   ((p.1 - a.1)^2 + (p.2 - a.2)^2) * ((b.1 - a.1)^2 + (b.2 - a.2)^2) / 4) →
  -2 * Real.sqrt 2 ≤ k ∧ k ≤ 2 * Real.sqrt 2 :=
by sorry

end tangent_line_range_l31_3100


namespace largest_integer_satisfying_inequality_l31_3116

theorem largest_integer_satisfying_inequality :
  ∀ x : ℤ, x ≤ 4 ↔ (x : ℚ) / 4 + 3 / 5 < 7 / 4 :=
by sorry

end largest_integer_satisfying_inequality_l31_3116


namespace incorrect_expression_l31_3123

/-- Represents a repeating decimal with a 3-digit non-repeating part and a 2-digit repeating part -/
structure RepeatingDecimal where
  nonRepeating : ℕ  -- Represents P (3-digit non-repeating part)
  repeating : ℕ     -- Represents Q (2-digit repeating part)
  nonRepeating_three_digits : nonRepeating < 1000
  repeating_two_digits : repeating < 100

/-- Converts a RepeatingDecimal to its decimal representation -/
def toDecimal (d : RepeatingDecimal) : ℚ :=
  (d.nonRepeating : ℚ) / 1000 + (d.repeating : ℚ) / 99900

/-- The statement that the given expression is incorrect -/
theorem incorrect_expression (d : RepeatingDecimal) :
  ¬(10^3 * (10^2 - 1) * toDecimal d = (d.repeating : ℚ) * (100 * d.nonRepeating - 1)) := by
  sorry

end incorrect_expression_l31_3123


namespace squirrel_journey_time_l31_3171

/-- Proves that a squirrel traveling 0.5 miles at 6 mph and then 1.5 miles at 3 mph
    takes 35 minutes to complete a 2-mile journey. -/
theorem squirrel_journey_time :
  let total_distance : ℝ := 2
  let first_segment_distance : ℝ := 0.5
  let first_segment_speed : ℝ := 6
  let second_segment_distance : ℝ := 1.5
  let second_segment_speed : ℝ := 3
  let first_segment_time : ℝ := first_segment_distance / first_segment_speed
  let second_segment_time : ℝ := second_segment_distance / second_segment_speed
  let total_time_hours : ℝ := first_segment_time + second_segment_time
  let total_time_minutes : ℝ := total_time_hours * 60
  total_distance = first_segment_distance + second_segment_distance →
  total_time_minutes = 35 := by
  sorry

end squirrel_journey_time_l31_3171


namespace washing_time_calculation_l31_3160

def clothes_time : ℕ := 30

def towels_time (clothes_time : ℕ) : ℕ := 2 * clothes_time

def sheets_time (towels_time : ℕ) : ℕ := towels_time - 15

def total_washing_time (clothes_time towels_time sheets_time : ℕ) : ℕ :=
  clothes_time + towels_time + sheets_time

theorem washing_time_calculation :
  total_washing_time clothes_time (towels_time clothes_time) (sheets_time (towels_time clothes_time)) = 135 := by
  sorry

end washing_time_calculation_l31_3160


namespace joyce_bananas_l31_3198

/-- Given a number of boxes and bananas per box, calculates the total number of bananas -/
def total_bananas (num_boxes : ℕ) (bananas_per_box : ℕ) : ℕ :=
  num_boxes * bananas_per_box

/-- Proves that 10 boxes with 4 bananas each results in 40 bananas total -/
theorem joyce_bananas : total_bananas 10 4 = 40 := by
  sorry

end joyce_bananas_l31_3198


namespace exists_indivisible_treasure_l31_3196

/-- Represents a treasure of gold bars -/
structure Treasure where
  num_bars : ℕ
  total_value : ℕ
  bar_values : Fin num_bars → ℕ
  sum_constraint : (Finset.univ.sum bar_values) = total_value

/-- Represents an even division of a treasure among pirates -/
def EvenDivision (t : Treasure) (num_pirates : ℕ) : Prop :=
  ∃ (division : Fin t.num_bars → Fin num_pirates),
    ∀ p : Fin num_pirates,
      (Finset.univ.filter (λ i => division i = p)).sum t.bar_values =
        t.total_value / num_pirates

/-- The main theorem stating that there exists a treasure that cannot be evenly divided -/
theorem exists_indivisible_treasure :
  ∃ (t : Treasure),
    t.num_bars = 240 ∧
    t.total_value = 360 ∧
    (∀ i : Fin t.num_bars, t.bar_values i > 0) ∧
    ¬(EvenDivision t 3) := by
  sorry

end exists_indivisible_treasure_l31_3196


namespace ams_sequence_results_in_14_l31_3141

/-- Milly's operation: multiply by 3 -/
def milly (x : ℤ) : ℤ := 3 * x

/-- Abby's operation: add 2 -/
def abby (x : ℤ) : ℤ := x + 2

/-- Sam's operation: subtract 1 -/
def sam (x : ℤ) : ℤ := x - 1

/-- The theorem stating that applying Abby's, Milly's, and Sam's operations in order to 3 results in 14 -/
theorem ams_sequence_results_in_14 : sam (milly (abby 3)) = 14 := by
  sorry

end ams_sequence_results_in_14_l31_3141


namespace divisibility_implies_five_divisor_l31_3187

theorem divisibility_implies_five_divisor (n : ℕ) : 
  n > 1 → (6^n - 1) % n = 0 → n % 5 = 0 := by sorry

end divisibility_implies_five_divisor_l31_3187


namespace cosine_sine_graph_shift_l31_3127

theorem cosine_sine_graph_shift (x : ℝ) :
  3 * Real.cos (2 * x) = 3 * Real.sin (2 * (x + π / 6) + π / 6) := by
  sorry

end cosine_sine_graph_shift_l31_3127


namespace min_value_xyz_l31_3154

theorem min_value_xyz (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (h : x * y * z = 27) :
  x + 3 * y + 9 * z ≥ 27 ∧ ∃ (x₀ y₀ z₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ z₀ > 0 ∧ x₀ * y₀ * z₀ = 27 ∧ x₀ + 3 * y₀ + 9 * z₀ = 27 :=
by sorry

end min_value_xyz_l31_3154


namespace calculator_squared_key_l31_3173

theorem calculator_squared_key (n : ℕ) : (5 ^ (2 ^ n) > 10000) ↔ n ≥ 3 :=
  sorry

end calculator_squared_key_l31_3173


namespace regions_less_than_199_with_99_lines_l31_3105

/-- The number of regions created by dividing a plane with lines -/
def num_regions (num_lines : ℕ) (all_parallel : Bool) (all_concurrent : Bool) : ℕ :=
  if all_parallel then
    num_lines + 1
  else if all_concurrent then
    2 * num_lines - 1
  else
    1 + num_lines + (num_lines.choose 2)

/-- Theorem stating the possible number of regions less than 199 when 99 lines divide a plane -/
theorem regions_less_than_199_with_99_lines :
  let possible_regions := {n : ℕ | n < 199 ∧ ∃ (parallel concurrent : Bool), 
    num_regions 99 parallel concurrent = n}
  possible_regions = {100, 198} := by
  sorry

end regions_less_than_199_with_99_lines_l31_3105


namespace food_fraction_is_one_fifth_l31_3146

def salary : ℚ := 150000.00000000003
def house_rent_fraction : ℚ := 1/10
def clothes_fraction : ℚ := 3/5
def amount_left : ℚ := 15000

theorem food_fraction_is_one_fifth :
  let food_fraction := 1 - house_rent_fraction - clothes_fraction - amount_left / salary
  food_fraction = 1/5 := by sorry

end food_fraction_is_one_fifth_l31_3146


namespace parabola_point_k_value_l31_3190

/-- Given that the point (3,0) lies on the parabola y = 2x^2 + (k+2)x - k, prove that k = -12 -/
theorem parabola_point_k_value :
  ∀ k : ℝ, (2 * 3^2 + (k + 2) * 3 - k = 0) → k = -12 := by
  sorry

end parabola_point_k_value_l31_3190


namespace triangular_number_formula_l31_3182

/-- The triangular number sequence -/
def triangular_number : ℕ → ℕ
| 0 => 0
| (n + 1) => triangular_number n + n + 1

/-- Theorem: The nth triangular number is equal to n(n+1)/2 -/
theorem triangular_number_formula (n : ℕ) :
  triangular_number n = n * (n + 1) / 2 := by
  sorry

end triangular_number_formula_l31_3182


namespace paulas_walking_distance_l31_3139

/-- Represents a pedometer with a maximum step count before reset --/
structure Pedometer where
  max_steps : ℕ
  steps_per_km : ℕ

/-- Represents the yearly walking data --/
structure YearlyWalkingData where
  pedometer : Pedometer
  resets : ℕ
  final_reading : ℕ

def calculate_total_steps (data : YearlyWalkingData) : ℕ :=
  data.resets * (data.pedometer.max_steps + 1) + data.final_reading

def calculate_kilometers (data : YearlyWalkingData) : ℚ :=
  (calculate_total_steps data : ℚ) / data.pedometer.steps_per_km

theorem paulas_walking_distance (data : YearlyWalkingData) 
  (h1 : data.pedometer.max_steps = 49999)
  (h2 : data.pedometer.steps_per_km = 1200)
  (h3 : data.resets = 76)
  (h4 : data.final_reading = 25000) :
  ∃ (k : ℕ), k ≥ 3187 ∧ k ≤ 3200 ∧ calculate_kilometers data = k := by
  sorry

#eval calculate_kilometers {
  pedometer := { max_steps := 49999, steps_per_km := 1200 },
  resets := 76,
  final_reading := 25000
}

end paulas_walking_distance_l31_3139


namespace emily_new_salary_l31_3106

def emily_initial_salary : ℕ := 1000000
def employee_salaries : List ℕ := [30000, 30000, 25000, 35000, 20000]
def min_salary : ℕ := 35000
def tax_rate : ℚ := 15 / 100

def calculate_new_salary (initial_salary : ℕ) (employee_salaries : List ℕ) (min_salary : ℕ) (tax_rate : ℚ) : ℕ :=
  sorry

theorem emily_new_salary :
  calculate_new_salary emily_initial_salary employee_salaries min_salary tax_rate = 959750 :=
sorry

end emily_new_salary_l31_3106


namespace three_person_subcommittees_from_eight_l31_3119

theorem three_person_subcommittees_from_eight (n : ℕ) (k : ℕ) : n = 8 ∧ k = 3 → Nat.choose n k = 56 := by
  sorry

end three_person_subcommittees_from_eight_l31_3119


namespace function_properties_l31_3178

-- Define the function f(x) with parameter a
def f (a : ℝ) (x : ℝ) : ℝ := x^3 - a*x^2 - 9*x + 11

-- Define the derivative of f(x)
def f_derivative (a : ℝ) (x : ℝ) : ℝ := 3*x^2 - 2*a*x - 9

theorem function_properties :
  ∃ (a : ℝ),
    (f_derivative a 1 = -12) ∧
    (a = 3) ∧
    (∀ x, f a x ≤ 16) ∧
    (∃ x, f a x = 16) ∧
    (∀ x, f a x ≥ -16) ∧
    (∃ x, f a x = -16) :=
by sorry

end function_properties_l31_3178


namespace cyclic_wins_count_l31_3153

/-- Represents a round-robin tournament. -/
structure Tournament where
  /-- The number of teams in the tournament. -/
  num_teams : ℕ
  /-- The number of wins for each team. -/
  wins_per_team : ℕ
  /-- The number of losses for each team. -/
  losses_per_team : ℕ
  /-- No ties in the tournament. -/
  no_ties : wins_per_team + losses_per_team = num_teams - 1

/-- The number of sets of three teams {A, B, C} where A beat B, B beat C, and C beat A. -/
def cyclic_wins (t : Tournament) : ℕ := sorry

/-- The main theorem stating the number of cyclic win sets in the given tournament. -/
theorem cyclic_wins_count (t : Tournament) 
  (h1 : t.num_teams = 21)
  (h2 : t.wins_per_team = 10)
  (h3 : t.losses_per_team = 10) :
  cyclic_wins t = 385 := by sorry

end cyclic_wins_count_l31_3153


namespace shielas_neighbors_l31_3102

theorem shielas_neighbors (total_drawings : ℕ) (drawings_per_neighbor : ℕ) (h1 : total_drawings = 54) (h2 : drawings_per_neighbor = 9) (h3 : drawings_per_neighbor > 0) :
  total_drawings / drawings_per_neighbor = 6 := by
  sorry

end shielas_neighbors_l31_3102


namespace weight_loss_days_l31_3136

/-- Calculates the number of days required to lose a target weight given daily calorie intake, burn rate, and calories per pound of weight loss. -/
def daysToLoseWeight (caloriesEaten : ℕ) (caloriesBurned : ℕ) (caloriesPerPound : ℕ) (targetPounds : ℕ) : ℕ :=
  (caloriesPerPound * targetPounds) / (caloriesBurned - caloriesEaten)

theorem weight_loss_days :
  daysToLoseWeight 1800 2300 4000 10 = 80 := by
  sorry

end weight_loss_days_l31_3136


namespace tree_height_after_two_years_l31_3168

/-- The height of a tree after a given number of years, given that it triples its height each year -/
def tree_height (initial_height : ℝ) (years : ℕ) : ℝ :=
  initial_height * (3 ^ years)

/-- Theorem: A tree that triples its height every year and reaches 243 feet after 5 years has a height of 9 feet after 2 years -/
theorem tree_height_after_two_years :
  ∃ (initial_height : ℝ),
    tree_height initial_height 5 = 243 ∧
    tree_height initial_height 2 = 9 :=
by
  sorry

end tree_height_after_two_years_l31_3168


namespace largest_initial_number_prove_largest_initial_number_l31_3152

theorem largest_initial_number : ℕ → Prop :=
  fun n => n = 189 ∧
    ∃ (a b c d e : ℕ),
      n + a + b + c + d + e = 200 ∧
      a ≥ 2 ∧ b ≥ 2 ∧ c ≥ 2 ∧ d ≥ 2 ∧ e ≥ 2 ∧
      ¬(n % a = 0) ∧ ¬(n % b = 0) ∧ ¬(n % c = 0) ∧ ¬(n % d = 0) ∧ ¬(n % e = 0) ∧
      ∀ m : ℕ, m > n →
        ¬∃ (a' b' c' d' e' : ℕ),
          m + a' + b' + c' + d' + e' = 200 ∧
          a' ≥ 2 ∧ b' ≥ 2 ∧ c' ≥ 2 ∧ d' ≥ 2 ∧ e' ≥ 2 ∧
          ¬(m % a' = 0) ∧ ¬(m % b' = 0) ∧ ¬(m % c' = 0) ∧ ¬(m % d' = 0) ∧ ¬(m % e' = 0)

theorem prove_largest_initial_number : ∃ n : ℕ, largest_initial_number n := by
  sorry

end largest_initial_number_prove_largest_initial_number_l31_3152


namespace exists_k_for_A_l31_3137

theorem exists_k_for_A (n m : ℕ) (hn : n ≥ 2) (hm : m ≥ 2) :
  ∃ k : ℕ, ((n + Real.sqrt (n^2 - 4)) / 2)^m = (k + Real.sqrt (k^2 - 4)) / 2 := by
  sorry

end exists_k_for_A_l31_3137


namespace square_difference_equals_360_l31_3193

theorem square_difference_equals_360 :
  (15 + 12)^2 - (12^2 + 15^2) = 360 := by
  sorry

end square_difference_equals_360_l31_3193


namespace fraction_subtraction_l31_3167

theorem fraction_subtraction : (5/6 : ℚ) + (1/4 : ℚ) - (2/3 : ℚ) = (5/12 : ℚ) := by
  sorry

end fraction_subtraction_l31_3167


namespace max_value_inequality_l31_3103

theorem max_value_inequality (x : ℝ) : 
  (∀ y, y > x → (6 + 5*y + y^2) * Real.sqrt (2*y^2 - y^3 - y) > 0) → 
  ((6 + 5*x + x^2) * Real.sqrt (2*x^2 - x^3 - x) ≤ 0) → 
  x ≤ 1 := by
sorry

end max_value_inequality_l31_3103


namespace min_horizontal_distance_l31_3129

def f (x : ℝ) := x^3 - x^2 - x - 6

theorem min_horizontal_distance :
  ∃ (x1 x2 : ℝ),
    f x1 = 8 ∧
    f x2 = -8 ∧
    ∀ (y1 y2 : ℝ),
      f y1 = 8 → f y2 = -8 →
      |x1 - x2| ≤ |y1 - y2| ∧
      |x1 - x2| = 1 :=
sorry

end min_horizontal_distance_l31_3129


namespace smallest_digit_for_divisibility_by_9_l31_3138

theorem smallest_digit_for_divisibility_by_9 :
  ∃ (d : Nat), d < 10 ∧ (562000 + d * 100 + 48) % 9 = 0 ∧
  ∀ (k : Nat), k < d → k < 10 → (562000 + k * 100 + 48) % 9 ≠ 0 :=
by sorry

end smallest_digit_for_divisibility_by_9_l31_3138


namespace arithmetic_sequence_difference_l31_3192

theorem arithmetic_sequence_difference (a b c : ℚ) : 
  (∃ d : ℚ, d = (9 - 2) / 4 ∧ 
             a = 2 + d ∧ 
             b = 2 + 2*d ∧ 
             c = 2 + 3*d ∧ 
             9 = 2 + 4*d) → 
  c - a = 3.5 := by
sorry

end arithmetic_sequence_difference_l31_3192


namespace sin_cos_identity_l31_3121

theorem sin_cos_identity : 
  Real.sin (110 * π / 180) * Real.cos (40 * π / 180) - 
  Real.cos (70 * π / 180) * Real.sin (40 * π / 180) = 1/2 := by
sorry

end sin_cos_identity_l31_3121


namespace quadratic_one_solution_l31_3147

theorem quadratic_one_solution (k : ℚ) : 
  (∃! x, 3 * x^2 - 8 * x + k = 0) ↔ k = 16/3 := by
sorry

end quadratic_one_solution_l31_3147


namespace collinear_points_k_value_l31_3179

/-- Three points are collinear if they lie on the same straight line. -/
def collinear (p1 p2 p3 : ℝ × ℝ) : Prop :=
  (p2.2 - p1.2) * (p3.1 - p1.1) = (p3.2 - p1.2) * (p2.1 - p1.1)

/-- The theorem states that if the points (4,7), (0,k), and (-8,5) are collinear, then k = 19/3. -/
theorem collinear_points_k_value :
  collinear (4, 7) (0, k) (-8, 5) → k = 19/3 :=
by
  sorry

end collinear_points_k_value_l31_3179


namespace smallest_number_with_given_properties_l31_3177

theorem smallest_number_with_given_properties : ∃ n : ℕ, 
  (∀ m : ℕ, m < n → ¬(8 ∣ m ∧ m % 2 = 1 ∧ m % 3 = 1 ∧ m % 4 = 1 ∧ m % 5 = 1 ∧ m % 7 = 1)) ∧ 
  (8 ∣ n) ∧ 
  (n % 2 = 1) ∧ 
  (n % 3 = 1) ∧ 
  (n % 4 = 1) ∧ 
  (n % 5 = 1) ∧ 
  (n % 7 = 1) ∧ 
  n = 7141 :=
by
  sorry

end smallest_number_with_given_properties_l31_3177


namespace M_always_positive_l31_3169

theorem M_always_positive (x y : ℝ) : 3 * x^2 - 8 * x * y + 9 * y^2 - 4 * x + 6 * y + 13 > 0 := by
  sorry

end M_always_positive_l31_3169


namespace bill_bouquets_theorem_l31_3165

/-- Represents the number of roses in a bouquet Bill buys -/
def roses_per_bouquet_buy : ℕ := 7

/-- Represents the number of roses in a bouquet Bill sells -/
def roses_per_bouquet_sell : ℕ := 5

/-- Represents the price of a bouquet (both buying and selling) -/
def price_per_bouquet : ℕ := 20

/-- Represents the target profit -/
def target_profit : ℕ := 1000

/-- Calculates the number of bouquets Bill needs to buy to earn the target profit -/
def bouquets_to_buy : ℕ :=
  let bouquets_per_operation := roses_per_bouquet_sell
  let profit_per_operation := price_per_bouquet * bouquets_per_operation - price_per_bouquet * roses_per_bouquet_buy / roses_per_bouquet_sell
  let operations_needed := target_profit / profit_per_operation
  operations_needed * roses_per_bouquet_buy / roses_per_bouquet_sell

theorem bill_bouquets_theorem : bouquets_to_buy = 125 := by
  sorry

end bill_bouquets_theorem_l31_3165


namespace solution_set_for_a_equals_one_range_of_a_for_inequality_l31_3181

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |4*x - a| + a^2 - 4*a

-- Define the function g
def g (x : ℝ) : ℝ := |x - 1|

-- Theorem for part 1
theorem solution_set_for_a_equals_one :
  {x : ℝ | -2 ≤ f 1 x ∧ f 1 x ≤ 4} = 
  {x : ℝ | -3/2 ≤ x ∧ x ≤ 0} ∪ {x : ℝ | 1/2 ≤ x ∧ x ≤ 2} :=
by sorry

-- Theorem for part 2
theorem range_of_a_for_inequality :
  {a : ℝ | ∀ x : ℝ, f a x - 4 * g x ≤ 6} = 
  {a : ℝ | (5 - Real.sqrt 33) / 2 ≤ a ∧ a ≤ 5} :=
by sorry

end solution_set_for_a_equals_one_range_of_a_for_inequality_l31_3181


namespace parabola_tangent_intersection_l31_3150

noncomputable def parabola (x : ℝ) : ℝ := x^2

def point_A : ℝ × ℝ := (1, 1)

noncomputable def point_B (x2 : ℝ) : ℝ × ℝ := (x2, x2^2)

noncomputable def tangent_slope (x : ℝ) : ℝ := 2 * x

noncomputable def tangent_line_A (x : ℝ) : ℝ := 2 * (x - 1) + 1

noncomputable def tangent_line_B (x2 x : ℝ) : ℝ := 2 * x2 * (x - x2) + x2^2

noncomputable def intersection_point (x2 : ℝ) : ℝ × ℝ :=
  let x_c := (x2^2 - 1) / (2 - 2*x2)
  let y_c := 2 * x_c - 1
  (x_c, y_c)

noncomputable def vector_AC (x2 : ℝ) : ℝ × ℝ :=
  let C := intersection_point x2
  (C.1 - point_A.1, C.2 - point_A.2)

noncomputable def vector_BC (x2 : ℝ) : ℝ × ℝ :=
  let C := intersection_point x2
  let B := point_B x2
  (C.1 - B.1, C.2 - B.2)

noncomputable def dot_product (v1 v2 : ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

theorem parabola_tangent_intersection (x2 : ℝ) :
  dot_product (vector_AC x2) (vector_BC x2) = 0 → x2 = -1/4 :=
by sorry

end parabola_tangent_intersection_l31_3150


namespace parallel_vectors_k_value_l31_3128

/-- Two vectors in ℝ² are parallel if their components are proportional -/
def parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 = a.2 * b.1

theorem parallel_vectors_k_value (k : ℝ) :
  let a : ℝ × ℝ := (2, k)
  let b : ℝ × ℝ := (1, 2)
  parallel a b → k = 4 := by
  sorry

end parallel_vectors_k_value_l31_3128


namespace scientific_notation_of_0_00625_l31_3191

/-- Scientific notation representation of a real number -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  coeff_range : 1 ≤ coefficient ∧ coefficient < 10

/-- Convert a real number to scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

theorem scientific_notation_of_0_00625 :
  toScientificNotation 0.00625 = ScientificNotation.mk 6.25 (-3) sorry := by
  sorry

end scientific_notation_of_0_00625_l31_3191


namespace complex_number_additive_inverse_parts_l31_3166

theorem complex_number_additive_inverse_parts (b : ℝ) : 
  let z := (2 - b * Complex.I) / (1 + 2 * Complex.I)
  (z.re = -z.im) → b = -2 := by
sorry

end complex_number_additive_inverse_parts_l31_3166


namespace reflected_ray_equation_l31_3112

-- Define the points
def start_point : ℝ × ℝ := (-1, 3)
def end_point : ℝ × ℝ := (4, 6)

-- Define the reflection surface (x-axis)
def reflection_axis : Set (ℝ × ℝ) := {p | p.2 = 0}

-- Define the reflected ray
def reflected_ray : Set (ℝ × ℝ) := {p | ∃ t : ℝ, p = (1 - t) • start_point + t • end_point}

-- Theorem statement
theorem reflected_ray_equation :
  ∀ p ∈ reflected_ray, 9 * p.1 - 5 * p.2 - 6 = 0 :=
sorry

end reflected_ray_equation_l31_3112


namespace unique_prime_in_range_l31_3194

/-- The only prime number in the range (200, 220) is 211 -/
theorem unique_prime_in_range : ∃! (n : ℕ), 200 < n ∧ n < 220 ∧ Nat.Prime n :=
  sorry

end unique_prime_in_range_l31_3194


namespace intersection_when_m_is_3_range_of_m_when_union_equals_B_l31_3149

-- Define the sets A and B
def A : Set ℝ := {x | x < 0 ∨ x > 3}
def B (m : ℝ) : Set ℝ := {x | x < m - 1 ∨ x > 2 * m}

-- Part 1: Prove that when m = 3, A ∩ B = {x | x < 0 ∨ x > 6}
theorem intersection_when_m_is_3 : A ∩ B 3 = {x | x < 0 ∨ x > 6} := by sorry

-- Part 2: Prove that when B ∪ A = B, the range of m is [1, 3/2]
theorem range_of_m_when_union_equals_B :
  (∀ m : ℝ, B m ∪ A = B m) ↔ (∀ m : ℝ, 1 ≤ m ∧ m ≤ 3/2) := by sorry

end intersection_when_m_is_3_range_of_m_when_union_equals_B_l31_3149


namespace no_valid_n_exists_l31_3185

theorem no_valid_n_exists : ∀ n : ℕ, n ≥ 2 →
  ¬∃ p q r : ℕ, Prime p ∧ Prime q ∧ Prime r ∧ p ≠ q ∧ p ≠ r ∧ q ≠ r ∧
    ∀ (a : Fin n → ℕ), (∀ i j : Fin n, i.val < j.val → a i ≠ a j) →
      (∀ i j : Fin n, i.val ≤ j.val → (p ∣ a j - a i) ∨ (q ∣ a j - a i) ∨ (r ∣ a j - a i)) →
        ((∀ i j : Fin n, i.val < j.val → p ∣ a j - a i) ∨
         (∀ i j : Fin n, i.val < j.val → q ∣ a j - a i) ∨
         (∀ i j : Fin n, i.val < j.val → r ∣ a j - a i)) :=
by
  sorry

end no_valid_n_exists_l31_3185


namespace dot_product_range_l31_3158

theorem dot_product_range (M N : ℝ × ℝ) (a : ℝ × ℝ) :
  let (x₁, y₁) := M
  let (x₂, y₂) := N
  x₁ ≥ 0 ∧ y₁ ≥ 0 ∧ x₁ + 2 * y₁ ≤ 6 ∧ 3 * x₁ + y₁ ≤ 12 ∧
  x₂ ≥ 0 ∧ y₂ ≥ 0 ∧ x₂ + 2 * y₂ ≤ 6 ∧ 3 * x₂ + y₂ ≤ 12 ∧
  a = (1, -1) →
  -7 ≤ ((x₂ - x₁) * a.1 + (y₂ - y₁) * a.2) ∧ 
  ((x₂ - x₁) * a.1 + (y₂ - y₁) * a.2) ≤ 7 := by
sorry

end dot_product_range_l31_3158


namespace square_of_sqrt_17_l31_3101

theorem square_of_sqrt_17 : (Real.sqrt 17) ^ 2 = 17 := by
  sorry

end square_of_sqrt_17_l31_3101


namespace inverse_proposition_correct_l31_3118

-- Define the type for lines
def Line : Type := ℝ → ℝ → Prop

-- Define what it means for two lines to be parallel
def parallel (l₁ l₂ : Line) : Prop := sorry

-- Define what it means for angles to be supplementary
def supplementary_angles (θ₁ θ₂ : ℝ) : Prop := θ₁ + θ₂ = 180

-- Define the original proposition
def original_proposition (l₁ l₂ : Line) (θ₁ θ₂ : ℝ) : Prop :=
  parallel l₁ l₂ → supplementary_angles θ₁ θ₂

-- Define the inverse proposition
def inverse_proposition (l₁ l₂ : Line) (θ₁ θ₂ : ℝ) : Prop :=
  supplementary_angles θ₁ θ₂ → parallel l₁ l₂

-- Theorem stating that the inverse proposition is correct
theorem inverse_proposition_correct :
  ∀ (l₁ l₂ : Line) (θ₁ θ₂ : ℝ),
    inverse_proposition l₁ l₂ θ₁ θ₂ =
    (supplementary_angles θ₁ θ₂ → parallel l₁ l₂) :=
by
  sorry

end inverse_proposition_correct_l31_3118


namespace lcm_of_15_25_35_l31_3186

theorem lcm_of_15_25_35 : Nat.lcm (Nat.lcm 15 25) 35 = 525 := by
  sorry

end lcm_of_15_25_35_l31_3186


namespace raisin_mixture_problem_l31_3143

/-- The number of scoops of natural seedless raisins needed to create a mixture with
    a specific cost per scoop, given the costs and quantities of two types of raisins. -/
theorem raisin_mixture_problem (cost_natural : ℚ) (cost_golden : ℚ) (scoops_golden : ℕ) (cost_mixture : ℚ) :
  cost_natural = 345/100 →
  cost_golden = 255/100 →
  scoops_golden = 20 →
  cost_mixture = 3 →
  ∃ scoops_natural : ℕ,
    scoops_natural = 20 ∧
    (cost_natural * scoops_natural + cost_golden * scoops_golden) / (scoops_natural + scoops_golden) = cost_mixture :=
by sorry

end raisin_mixture_problem_l31_3143


namespace log_equation_sum_l31_3125

theorem log_equation_sum (A B C : ℕ+) 
  (h_coprime : Nat.gcd A.val (Nat.gcd B.val C.val) = 1)
  (h_eq : A * Real.log 5 / Real.log 100 + B * Real.log 2 / Real.log 100 = C) : 
  A + B + C = 5 := by
sorry

end log_equation_sum_l31_3125


namespace complement_intersection_theorem_l31_3159

open Set

def U : Set Nat := {1,2,3,4,5,6,7,8}
def A : Set Nat := {1,2,3}
def B : Set Nat := {2,3,4,5}

theorem complement_intersection_theorem : 
  (Aᶜ ∪ Bᶜ) ∩ U = {1,4,5,6,7,8} := by sorry

end complement_intersection_theorem_l31_3159


namespace polynomial_value_equals_one_l31_3120

theorem polynomial_value_equals_one (x₀ : ℂ) (h : x₀^2 + x₀ + 2 = 0) :
  x₀^4 + 2*x₀^3 + 3*x₀^2 + 2*x₀ + 1 = 1 := by
  sorry

end polynomial_value_equals_one_l31_3120


namespace shirts_not_all_on_sale_l31_3122

-- Define the universe of discourse
variable (Shirt : Type)
-- Define the property of being on sale
variable (on_sale : Shirt → Prop)
-- Define the property of being in the store
variable (in_store : Shirt → Prop)

-- Theorem statement
theorem shirts_not_all_on_sale 
  (h : ¬ (∀ s : Shirt, in_store s → on_sale s)) : 
  (∃ s : Shirt, in_store s ∧ ¬ on_sale s) ∧ 
  (¬ (∀ s : Shirt, in_store s → on_sale s)) := by
  sorry


end shirts_not_all_on_sale_l31_3122


namespace wendi_chicken_count_l31_3184

/-- The number of chickens Wendi has after various changes --/
def final_chicken_count (initial : ℕ) : ℕ :=
  let doubled := initial * 2
  let after_loss := doubled - 1
  let additional := 6
  after_loss + additional

/-- Theorem stating that starting with 4 chickens, Wendi ends up with 13 chickens --/
theorem wendi_chicken_count : final_chicken_count 4 = 13 := by
  sorry

end wendi_chicken_count_l31_3184


namespace figure_perimeter_l31_3140

/-- The figure in the coordinate plane defined by |x + y| + |x - y| = 8 -/
def Figure := {p : ℝ × ℝ | |p.1 + p.2| + |p.1 - p.2| = 8}

/-- The perimeter of a set in ℝ² -/
noncomputable def perimeter (S : Set (ℝ × ℝ)) : ℝ := sorry

theorem figure_perimeter : perimeter Figure = 16 * Real.sqrt 2 := by sorry

end figure_perimeter_l31_3140
