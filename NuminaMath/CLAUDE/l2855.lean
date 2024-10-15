import Mathlib

namespace NUMINAMATH_CALUDE_four_digit_number_proof_l2855_285537

theorem four_digit_number_proof :
  ∀ (a b : ℕ),
    (2^a * 9^b ≥ 1000) ∧ 
    (2^a * 9^b < 10000) ∧
    (2^a * 9^b = 2000 + 100*a + 90 + b) →
    a = 5 ∧ b = 2 := by
  sorry

end NUMINAMATH_CALUDE_four_digit_number_proof_l2855_285537


namespace NUMINAMATH_CALUDE_remaining_volume_cube_with_hole_l2855_285531

/-- The remaining volume of a cube after drilling a cylindrical hole -/
theorem remaining_volume_cube_with_hole (cube_side : Real) (hole_radius : Real) (hole_height : Real) :
  cube_side = 6 →
  hole_radius = 3 →
  hole_height = 4 →
  cube_side ^ 3 - π * hole_radius ^ 2 * hole_height = 216 - 36 * π := by
  sorry

#check remaining_volume_cube_with_hole

end NUMINAMATH_CALUDE_remaining_volume_cube_with_hole_l2855_285531


namespace NUMINAMATH_CALUDE_unique_records_count_l2855_285564

/-- The number of records in either Samantha's or Lily's collection, but not both -/
def unique_records (samantha_total : ℕ) (shared : ℕ) (lily_unique : ℕ) : ℕ :=
  (samantha_total - shared) + lily_unique

/-- Proof that the number of unique records is 18 -/
theorem unique_records_count :
  unique_records 24 15 9 = 18 := by
  sorry

end NUMINAMATH_CALUDE_unique_records_count_l2855_285564


namespace NUMINAMATH_CALUDE_alpha_value_at_negative_six_l2855_285551

/-- Given that α is inversely proportional to β², prove that α = 1/3 when β = -6,
    given the condition that α = 3 when β = 2. -/
theorem alpha_value_at_negative_six (α β : ℝ) (h : ∃ k, ∀ β ≠ 0, α = k / β^2) 
    (h_condition : α = 3 ∧ β = 2) : 
    (β = -6) → (α = 1/3) := by
  sorry

end NUMINAMATH_CALUDE_alpha_value_at_negative_six_l2855_285551


namespace NUMINAMATH_CALUDE_water_pouring_game_score_l2855_285505

/-- Represents the players in the game -/
inductive Player
| Xiaoming
| Xiaolin

/-- Defines the scoring rules for the water pouring game -/
def score (overflowPlayer : Option Player) : Nat :=
  match overflowPlayer with
  | some Player.Xiaoming => 10
  | some Player.Xiaolin => 9
  | none => 3

/-- Represents a round in the game -/
structure Round where
  xiaomingPour : Nat
  xiaolinPour : Nat
  overflowPlayer : Option Player

/-- The three rounds of the game -/
def round1 : Round := ⟨5, 5, some Player.Xiaolin⟩
def round2 : Round := ⟨2, 7, none⟩
def round3 : Round := ⟨13, 0, some Player.Xiaoming⟩

/-- Calculates the total score for the given rounds -/
def totalScore (rounds : List Round) : Nat :=
  rounds.foldl (fun acc r => acc + score r.overflowPlayer) 0

/-- The main theorem to prove -/
theorem water_pouring_game_score :
  totalScore [round1, round2, round3] = 22 := by
  sorry

end NUMINAMATH_CALUDE_water_pouring_game_score_l2855_285505


namespace NUMINAMATH_CALUDE_volume_is_one_sixth_l2855_285526

-- Define the region
def region (x y z : ℝ) : Prop :=
  abs x + abs y + abs z ≤ 1 ∧ abs x + abs y + abs (z - 1) ≤ 1

-- Define the volume of the region
noncomputable def volume_of_region : ℝ := sorry

-- Theorem statement
theorem volume_is_one_sixth : volume_of_region = 1/6 := by sorry

end NUMINAMATH_CALUDE_volume_is_one_sixth_l2855_285526


namespace NUMINAMATH_CALUDE_increasing_function_condition_l2855_285596

def f (a b x : ℝ) : ℝ := (a + 1) * x + b

theorem increasing_function_condition (a b : ℝ) :
  (∀ x y : ℝ, x < y → f a b x < f a b y) → a > -1 := by
  sorry

end NUMINAMATH_CALUDE_increasing_function_condition_l2855_285596


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l2855_285593

/-- The eccentricity of a hyperbola with specific properties -/
theorem hyperbola_eccentricity : ∀ (a b : ℝ), a > 0 → b > 0 →
  (∃ (x y : ℝ), x^2 / a^2 - y^2 / b^2 = 1 ∧ y = Real.sqrt x ∧
  (∃ (m : ℝ), m * (x + 1) = y ∧ m = 1 / (2 * Real.sqrt x))) →
  (∃ (c : ℝ), c^2 = a^2 + b^2 ∧ c = 1) →
  (a^2 + b^2) / a = (Real.sqrt 5 + 1) / 2 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l2855_285593


namespace NUMINAMATH_CALUDE_remainder_problem_l2855_285552

theorem remainder_problem (M : ℕ) (h1 : M % 24 = 13) (h2 : M = 3024) : M % 1821 = 1203 := by
  sorry

end NUMINAMATH_CALUDE_remainder_problem_l2855_285552


namespace NUMINAMATH_CALUDE_ellipse_slope_theorem_l2855_285580

/-- Given an ellipse with specific properties, prove the slope of a line passing through a point on the ellipse --/
theorem ellipse_slope_theorem (F₁ PF : ℝ) (k₂ : ℝ) :
  F₁ = (6/5) * Real.sqrt 5 →
  PF = (4/5) * Real.sqrt 5 →
  ∃ (k : ℝ), k = (3/2) * k₂ ∧ (k = (3 * Real.sqrt 5) / 10 ∨ k = -(3 * Real.sqrt 5) / 10) := by
  sorry

end NUMINAMATH_CALUDE_ellipse_slope_theorem_l2855_285580


namespace NUMINAMATH_CALUDE_tickets_to_be_sold_l2855_285571

theorem tickets_to_be_sold (total : ℕ) (jude andrea sandra : ℕ) : 
  total = 100 → 
  andrea = 2 * jude → 
  sandra = jude / 2 + 4 → 
  jude = 16 → 
  total - (jude + andrea + sandra) = 40 := by
sorry

end NUMINAMATH_CALUDE_tickets_to_be_sold_l2855_285571


namespace NUMINAMATH_CALUDE_constant_derivative_implies_linear_l2855_285589

/-- If a function's derivative is zero everywhere, then its graph is a straight line -/
theorem constant_derivative_implies_linear (f : ℝ → ℝ) :
  (∀ x : ℝ, deriv f x = 0) → ∃ a b : ℝ, ∀ x : ℝ, f x = a * x + b :=
sorry

end NUMINAMATH_CALUDE_constant_derivative_implies_linear_l2855_285589


namespace NUMINAMATH_CALUDE_series_sum_l2855_285582

theorem series_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a > b) :
  let series_term (n : ℕ) := 1 / (((2 * n - 3) * a - (n - 2) * b) * (2 * n * a - (2 * n - 1) * b))
  ∑' n, series_term n = 1 / ((a - b) * b) := by
  sorry

end NUMINAMATH_CALUDE_series_sum_l2855_285582


namespace NUMINAMATH_CALUDE_square_field_diagonal_l2855_285556

theorem square_field_diagonal (area : ℝ) (diagonal : ℝ) : 
  area = 128 → diagonal = 16 := by
  sorry

end NUMINAMATH_CALUDE_square_field_diagonal_l2855_285556


namespace NUMINAMATH_CALUDE_unknown_number_proof_l2855_285574

theorem unknown_number_proof (x : ℝ) : 3034 - (x / 200.4) = 3029 → x = 1002 := by
  sorry

end NUMINAMATH_CALUDE_unknown_number_proof_l2855_285574


namespace NUMINAMATH_CALUDE_union_of_sets_l2855_285515

theorem union_of_sets (a : ℝ) : 
  let A : Set ℝ := {a^2 + 1, 2*a}
  let B : Set ℝ := {a + 1, 0}
  (A ∩ B).Nonempty → A ∪ B = {0, 1} := by
  sorry

end NUMINAMATH_CALUDE_union_of_sets_l2855_285515


namespace NUMINAMATH_CALUDE_dot_product_equals_one_l2855_285575

def a : ℝ × ℝ := (1, -1)
def b : ℝ × ℝ := (-1, 2)

theorem dot_product_equals_one :
  (2 • a + b) • a = 1 := by sorry

end NUMINAMATH_CALUDE_dot_product_equals_one_l2855_285575


namespace NUMINAMATH_CALUDE_minimum_amount_spent_on_boxes_l2855_285511

/-- The minimum amount spent on boxes for packaging a collection --/
theorem minimum_amount_spent_on_boxes
  (box_length : ℝ) (box_width : ℝ) (box_height : ℝ)
  (cost_per_box : ℝ) (total_collection_volume : ℝ)
  (h1 : box_length = 20)
  (h2 : box_width = 20)
  (h3 : box_height = 12)
  (h4 : cost_per_box = 0.40)
  (h5 : total_collection_volume = 2160000) :
  ⌈total_collection_volume / (box_length * box_width * box_height)⌉ * cost_per_box = 180 := by
  sorry

#check minimum_amount_spent_on_boxes

end NUMINAMATH_CALUDE_minimum_amount_spent_on_boxes_l2855_285511


namespace NUMINAMATH_CALUDE_min_value_of_function_l2855_285563

theorem min_value_of_function (x : ℝ) (h1 : 0 < x) (h2 : x < 1/3) :
  (3 / (2 * x) + 2 / (1 - 3 * x)) ≥ 25/2 :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_function_l2855_285563


namespace NUMINAMATH_CALUDE_problem_solution_l2855_285545

theorem problem_solution (x y : ℝ) 
  (hx : x = 2 + Real.sqrt 3) 
  (hy : y = 2 - Real.sqrt 3) : 
  (x^2 + 2*x*y + y^2 = 16) ∧ (x^2 - y^2 = 8 * Real.sqrt 3) := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l2855_285545


namespace NUMINAMATH_CALUDE_union_of_M_and_N_l2855_285521

def M : Set ℝ := {x : ℝ | x^2 - x - 12 = 0}
def N : Set ℝ := {x : ℝ | x^2 + 3*x = 0}

theorem union_of_M_and_N : M ∪ N = {-3, 0, 4} := by
  sorry

end NUMINAMATH_CALUDE_union_of_M_and_N_l2855_285521


namespace NUMINAMATH_CALUDE_triangle_rectangle_perimeter_equality_l2855_285586

/-- The perimeter of an isosceles triangle with two sides of 12 cm and one side of 14 cm 
    is equal to the perimeter of a rectangle with width 8 cm and length x cm. -/
theorem triangle_rectangle_perimeter_equality (x : ℝ) : 
  (12 : ℝ) + 12 + 14 = 2 * (x + 8) → x = 11 := by
  sorry

end NUMINAMATH_CALUDE_triangle_rectangle_perimeter_equality_l2855_285586


namespace NUMINAMATH_CALUDE_scientific_notation_equality_l2855_285553

theorem scientific_notation_equality (n : ℝ) : n = 361000000 → n = 3.61 * 10^8 := by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_equality_l2855_285553


namespace NUMINAMATH_CALUDE_samples_are_stratified_l2855_285559

/-- Represents a sample of 10 student numbers -/
structure Sample :=
  (numbers : List Nat)
  (h_size : numbers.length = 10)
  (h_range : ∀ n ∈ numbers, 1 ≤ n ∧ n ≤ 270)

/-- Represents the distribution of students across grades -/
structure SchoolDistribution :=
  (total : Nat)
  (first_grade : Nat)
  (second_grade : Nat)
  (third_grade : Nat)
  (h_total : total = first_grade + second_grade + third_grade)

/-- Checks if a sample can represent stratified sampling for a given school distribution -/
def is_stratified_sampling (s : Sample) (sd : SchoolDistribution) : Prop :=
  ∃ (n1 n2 n3 : Nat),
    n1 + n2 + n3 = 10 ∧
    n1 ≤ sd.first_grade ∧
    n2 ≤ sd.second_grade ∧
    n3 ≤ sd.third_grade ∧
    (∀ n ∈ s.numbers, 
      (n ≤ sd.first_grade) ∨ 
      (sd.first_grade < n ∧ n ≤ sd.first_grade + sd.second_grade) ∨
      (sd.first_grade + sd.second_grade < n))

def sample1 : Sample := {
  numbers := [7, 34, 61, 88, 115, 142, 169, 196, 223, 250],
  h_size := by rfl,
  h_range := sorry
}

def sample3 : Sample := {
  numbers := [11, 38, 65, 92, 119, 146, 173, 200, 227, 254],
  h_size := by rfl,
  h_range := sorry
}

def school : SchoolDistribution := {
  total := 270,
  first_grade := 108,
  second_grade := 81,
  third_grade := 81,
  h_total := by rfl
}

theorem samples_are_stratified : 
  is_stratified_sampling sample1 school ∧ is_stratified_sampling sample3 school :=
sorry

end NUMINAMATH_CALUDE_samples_are_stratified_l2855_285559


namespace NUMINAMATH_CALUDE_other_communities_students_l2855_285535

theorem other_communities_students (total : ℕ) (muslim_percent hindu_percent sikh_percent christian_percent buddhist_percent : ℚ) :
  total = 1500 →
  muslim_percent = 38/100 →
  hindu_percent = 26/100 →
  sikh_percent = 12/100 →
  christian_percent = 6/100 →
  buddhist_percent = 4/100 →
  ↑(total * (1 - (muslim_percent + hindu_percent + sikh_percent + christian_percent + buddhist_percent))) = 210 :=
by
  sorry

end NUMINAMATH_CALUDE_other_communities_students_l2855_285535


namespace NUMINAMATH_CALUDE_model_y_completion_time_l2855_285539

/-- The time (in minutes) it takes for a Model Y computer to complete the task -/
def model_y_time : ℝ := 30

/-- The time (in minutes) it takes for a Model X computer to complete the task -/
def model_x_time : ℝ := 60

/-- The number of Model X computers used -/
def num_model_x : ℝ := 20

/-- The time (in minutes) it takes for both models working together to complete the task -/
def total_time : ℝ := 1

theorem model_y_completion_time :
  (num_model_x / model_x_time + num_model_x / model_y_time) * total_time = 1 :=
by sorry

end NUMINAMATH_CALUDE_model_y_completion_time_l2855_285539


namespace NUMINAMATH_CALUDE_number_problem_l2855_285579

theorem number_problem (N : ℝ) : 
  (1/4 : ℝ) * (1/3 : ℝ) * (2/5 : ℝ) * N = 17 → (40/100 : ℝ) * N = 204 := by
  sorry

end NUMINAMATH_CALUDE_number_problem_l2855_285579


namespace NUMINAMATH_CALUDE_alpha_range_l2855_285538

theorem alpha_range (α : Real) (h1 : 0 < α) (h2 : α < Real.pi / 2) 
  (h3 : Real.cos α - Real.sin α = Real.tan α) : 
  α ∈ Set.Ioo 0 (Real.pi / 6) := by
sorry

end NUMINAMATH_CALUDE_alpha_range_l2855_285538


namespace NUMINAMATH_CALUDE_polynomial_simplification_l2855_285567

theorem polynomial_simplification (x : ℝ) : 
  (7 * x^12 + 2 * x^10 + x^9) + (3 * x^11 + x^10 + 6 * x^9 + 5 * x^7) + (x^12 + 4 * x^10 + 2 * x^9 + x^3) = 
  8 * x^12 + 3 * x^11 + 7 * x^10 + 9 * x^9 + 5 * x^7 + x^3 :=
by sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l2855_285567


namespace NUMINAMATH_CALUDE_set_operations_l2855_285578

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | -1 < x ∧ x < 2}
def B : Set ℝ := {x : ℝ | x > 0}

-- Define the complement of B in ℝ
def C_R_B : Set ℝ := {x : ℝ | x ≤ 0}

-- State the theorem
theorem set_operations :
  (A ∩ B = {x : ℝ | 0 < x ∧ x < 2}) ∧
  (C_R_B ∪ A = {x : ℝ | x < 2}) := by
  sorry

end NUMINAMATH_CALUDE_set_operations_l2855_285578


namespace NUMINAMATH_CALUDE_min_knight_liar_pairs_l2855_285568

/-- Represents the type of people on the island -/
inductive Person
| Knight
| Liar

/-- Represents a friendship between two people -/
structure Friendship where
  person1 : Person
  person2 : Person

/-- The total number of people on the island -/
def total_people : Nat := 200

/-- The number of knights on the island -/
def num_knights : Nat := 100

/-- The number of liars on the island -/
def num_liars : Nat := 100

/-- The number of people who said "All my friends are knights" -/
def num_all_knight_claims : Nat := 100

/-- The number of people who said "All my friends are liars" -/
def num_all_liar_claims : Nat := 100

/-- Definition: Each person has at least one friend -/
axiom has_friend (p : Person) : ∃ (f : Friendship), f.person1 = p ∨ f.person2 = p

/-- Definition: Knights always tell the truth -/
axiom knight_truth (k : Person) (claim : Prop) : k = Person.Knight → (claim ↔ true)

/-- Definition: Liars always lie -/
axiom liar_lie (l : Person) (claim : Prop) : l = Person.Liar → (claim ↔ false)

/-- The main theorem to be proved -/
theorem min_knight_liar_pairs :
  ∃ (friendships : List Friendship),
    (∀ f ∈ friendships, (f.person1 = Person.Knight ∧ f.person2 = Person.Liar) ∨
                        (f.person1 = Person.Liar ∧ f.person2 = Person.Knight)) ∧
    friendships.length = 50 ∧
    (∀ friendships' : List Friendship,
      (∀ f' ∈ friendships', (f'.person1 = Person.Knight ∧ f'.person2 = Person.Liar) ∨
                            (f'.person1 = Person.Liar ∧ f'.person2 = Person.Knight)) →
      friendships'.length ≥ 50) := by
  sorry

end NUMINAMATH_CALUDE_min_knight_liar_pairs_l2855_285568


namespace NUMINAMATH_CALUDE_union_A_B_when_a_is_4_intersection_A_B_equals_A_l2855_285576

-- Define the sets A and B
def A (a : ℝ) : Set ℝ := {x | |x - a| ≤ 1}
def B : Set ℝ := {x | (4 - x) * (x - 1) ≤ 0}

-- Statement 1
theorem union_A_B_when_a_is_4 : 
  A 4 ∪ B = {x | x ≥ 3 ∨ x ≤ 1} := by sorry

-- Statement 2
theorem intersection_A_B_equals_A (a : ℝ) : 
  A a ∩ B = A a ↔ a ≥ 5 ∨ a ≤ 0 := by sorry

end NUMINAMATH_CALUDE_union_A_B_when_a_is_4_intersection_A_B_equals_A_l2855_285576


namespace NUMINAMATH_CALUDE_crypto_puzzle_l2855_285562

theorem crypto_puzzle (A B C D : Nat) : 
  A < 10 ∧ B < 10 ∧ C < 10 ∧ D < 10 ∧  -- Digits are 0-9
  A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D ∧  -- Unique digits
  A + B + C = D ∧
  B + C = 7 ∧
  A - B = 1 →
  D = 9 := by
sorry

end NUMINAMATH_CALUDE_crypto_puzzle_l2855_285562


namespace NUMINAMATH_CALUDE_complex_fraction_evaluation_l2855_285557

theorem complex_fraction_evaluation (a b : ℂ) (ha : a ≠ 0) (hb : b ≠ 0) 
  (h : a^4 + a^2 * b^2 + b^4 = 0) : 
  (a^15 + b^15) / (a + b)^15 = -2 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_evaluation_l2855_285557


namespace NUMINAMATH_CALUDE_problem_1_problem_2_l2855_285594

-- Define the logarithm function
noncomputable def lg (base : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log base

-- Theorem 1
theorem problem_1 : (lg 2 2)^2 + (lg 2 2) * (lg 2 5) + (lg 2 5) = 1 := by sorry

-- Theorem 2
theorem problem_2 : (2^(1/3) * 3^(1/2))^6 - 8 * (16/49)^(-1/2) - 2^(1/4) * 8^0.25 - (-2016)^0 = 91 := by sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_l2855_285594


namespace NUMINAMATH_CALUDE_suji_age_problem_l2855_285501

theorem suji_age_problem (abi_age suji_age : ℕ) : 
  (abi_age : ℚ) / suji_age = 5 / 4 →
  (abi_age + 3 : ℚ) / (suji_age + 3) = 11 / 9 →
  suji_age = 24 := by
sorry

end NUMINAMATH_CALUDE_suji_age_problem_l2855_285501


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l2855_285570

theorem quadratic_equation_roots (m : ℝ) : 
  (∃ x : ℝ, x^2 + m*x + 3 = 0 ∧ x = 1) → 
  (∃ y : ℝ, y^2 + m*y + 3 = 0 ∧ y = 3 ∧ m = -4) := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l2855_285570


namespace NUMINAMATH_CALUDE_subtract_negative_negative_two_minus_five_l2855_285500

theorem subtract_negative (a b : ℤ) : a - b = a + (-b) := by sorry

theorem negative_two_minus_five : (-2 : ℤ) - 5 = -7 := by sorry

end NUMINAMATH_CALUDE_subtract_negative_negative_two_minus_five_l2855_285500


namespace NUMINAMATH_CALUDE_circle_tangent_relation_l2855_285590

/-- Two circles with radii R₁ and R₂ are externally tangent. A line of length d is perpendicular to their common tangent. -/
structure CircleConfiguration where
  R₁ : ℝ
  R₂ : ℝ
  d : ℝ
  R₁_pos : 0 < R₁
  R₂_pos : 0 < R₂
  d_pos : 0 < d
  externally_tangent : R₁ + R₂ > 0

/-- The relation between the radii of two externally tangent circles and the length of a line perpendicular to their common tangent. -/
theorem circle_tangent_relation (c : CircleConfiguration) :
  1 / c.R₁ + 1 / c.R₂ = 2 / c.d := by
  sorry

end NUMINAMATH_CALUDE_circle_tangent_relation_l2855_285590


namespace NUMINAMATH_CALUDE_fence_length_for_specific_yard_l2855_285598

/-- A rectangular yard with given dimensions and area -/
structure RectangularYard where
  length : ℝ
  width : ℝ
  area : ℝ
  length_positive : 0 < length
  width_positive : 0 < width
  area_eq : area = length * width

/-- The fence length for a rectangular yard -/
def fence_length (yard : RectangularYard) : ℝ :=
  2 * yard.width + yard.length

/-- Theorem: For a rectangular yard with one side of 40 feet and an area of 240 square feet,
    the fence length (perimeter minus one side) is 52 feet -/
theorem fence_length_for_specific_yard :
  ∃ (yard : RectangularYard),
    yard.length = 40 ∧
    yard.area = 240 ∧
    fence_length yard = 52 := by
  sorry


end NUMINAMATH_CALUDE_fence_length_for_specific_yard_l2855_285598


namespace NUMINAMATH_CALUDE_y_value_proof_l2855_285508

theorem y_value_proof (y : ℝ) (h : (40 : ℝ) / 80 = Real.sqrt (y / 80)) : y = 20 := by
  sorry

end NUMINAMATH_CALUDE_y_value_proof_l2855_285508


namespace NUMINAMATH_CALUDE_joan_seashells_l2855_285595

/-- The number of seashells Joan has after giving some away -/
def remaining_seashells (initial : ℕ) (given_away : ℕ) : ℕ :=
  initial - given_away

/-- Theorem: Joan has 16 seashells after giving away 63 from her initial 79 -/
theorem joan_seashells : remaining_seashells 79 63 = 16 := by
  sorry

end NUMINAMATH_CALUDE_joan_seashells_l2855_285595


namespace NUMINAMATH_CALUDE_min_value_theorem_min_value_achievable_l2855_285517

theorem min_value_theorem (x : ℝ) : 
  (x^2 + 9) / Real.sqrt (x^2 + 3) ≥ 2 * Real.sqrt 6 := by
  sorry

theorem min_value_achievable : 
  ∃ x : ℝ, (x^2 + 9) / Real.sqrt (x^2 + 3) = 2 * Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_min_value_achievable_l2855_285517


namespace NUMINAMATH_CALUDE_tangent_circle_center_and_radius_l2855_285585

/-- A circle tangent to y=x, y=-x, and y=10 with center above (10,10) -/
structure TangentCircle where
  h : ℝ
  k : ℝ
  r : ℝ
  h_gt_ten : h > 10
  k_gt_ten : k > 10
  tangent_y_eq_x : r = |h - k| / Real.sqrt 2
  tangent_y_eq_neg_x : r = |h + k| / Real.sqrt 2
  tangent_y_eq_ten : r = k - 10

/-- The center and radius of a circle tangent to y=x, y=-x, and y=10 -/
theorem tangent_circle_center_and_radius (c : TangentCircle) :
  c.h = 10 + (1 + Real.sqrt 2) * c.r ∧ c.k = 10 + c.r :=
by sorry

end NUMINAMATH_CALUDE_tangent_circle_center_and_radius_l2855_285585


namespace NUMINAMATH_CALUDE_marbles_distribution_l2855_285581

theorem marbles_distribution (total_marbles : ℕ) (num_boys : ℕ) (marbles_per_boy : ℕ) : 
  total_marbles = 35 → num_boys = 5 → marbles_per_boy = total_marbles / num_boys → marbles_per_boy = 7 := by
  sorry

end NUMINAMATH_CALUDE_marbles_distribution_l2855_285581


namespace NUMINAMATH_CALUDE_triangle_equilateral_iff_equation_l2855_285527

/-- A triangle ABC with side lengths a, b, and c is equilateral if and only if
    a^4 + b^4 + c^4 - a^2b^2 - b^2c^2 - a^2c^2 = 0 -/
theorem triangle_equilateral_iff_equation (a b c : ℝ) (h : a > 0 ∧ b > 0 ∧ c > 0) :
  a^4 + b^4 + c^4 - a^2*b^2 - b^2*c^2 - a^2*c^2 = 0 ↔ a = b ∧ b = c := by
  sorry

end NUMINAMATH_CALUDE_triangle_equilateral_iff_equation_l2855_285527


namespace NUMINAMATH_CALUDE_lcm_of_8_9_5_10_l2855_285541

theorem lcm_of_8_9_5_10 : Nat.lcm 8 (Nat.lcm 9 (Nat.lcm 5 10)) = 360 := by
  sorry

end NUMINAMATH_CALUDE_lcm_of_8_9_5_10_l2855_285541


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l2855_285566

theorem complex_fraction_simplification :
  (7 + 18 * Complex.I) / (3 - 4 * Complex.I) = (-51 / 25 : ℝ) + (82 / 25 : ℝ) * Complex.I :=
by sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l2855_285566


namespace NUMINAMATH_CALUDE_book_length_is_4556_l2855_285565

/-- Represents the properties of a book and a reader's progress. -/
structure BookReading where
  total_hours : Nat
  pages_read : Nat
  speed_increase : Nat
  extra_pages : Nat

/-- Calculates the total number of pages in the book based on the given reading information. -/
def calculate_total_pages (reading : BookReading) : Nat :=
  reading.pages_read + (reading.pages_read - reading.extra_pages)

/-- Theorem stating that given the specific reading conditions, the total number of pages in the book is 4556. -/
theorem book_length_is_4556 (reading : BookReading)
  (h1 : reading.total_hours = 5)
  (h2 : reading.pages_read = 2323)
  (h3 : reading.speed_increase = 10)
  (h4 : reading.extra_pages = 90) :
  calculate_total_pages reading = 4556 := by
  sorry

#eval calculate_total_pages { total_hours := 5, pages_read := 2323, speed_increase := 10, extra_pages := 90 }

end NUMINAMATH_CALUDE_book_length_is_4556_l2855_285565


namespace NUMINAMATH_CALUDE_sequence_sum_l2855_285573

def is_geometric (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

def is_arithmetic (b : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, b (n + 1) = b n + d

theorem sequence_sum (a b : ℕ → ℝ) :
  is_geometric a →
  is_arithmetic b →
  a 3 * a 11 = 4 * a 7 →
  a 7 = b 7 →
  b 5 + b 9 = 8 := by
sorry

end NUMINAMATH_CALUDE_sequence_sum_l2855_285573


namespace NUMINAMATH_CALUDE_geometric_sequence_property_l2855_285584

theorem geometric_sequence_property (a b c q : ℝ) : 
  (∃ x : ℝ, x ≠ 0 ∧ 
    b + c - a = (a + b + c) * q ∧
    c + a - b = (a + b + c) * q^2 ∧
    a + b - c = (a + b + c) * q^3) →
  q^3 + q^2 + q = 1 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_property_l2855_285584


namespace NUMINAMATH_CALUDE_min_faces_for_conditions_l2855_285569

/-- Represents a pair of dice --/
structure DicePair :=
  (die1 : ℕ)
  (die2 : ℕ)

/-- Calculates the number of ways to roll a sum on a pair of dice --/
def waysToRollSum (d : DicePair) (sum : ℕ) : ℕ := sorry

/-- Checks if a pair of dice satisfies the given conditions --/
def satisfiesConditions (d : DicePair) : Prop :=
  d.die1 ≥ 6 ∧ d.die2 ≥ 6 ∧
  waysToRollSum d 8 * 12 = waysToRollSum d 11 * 5 ∧
  waysToRollSum d 14 * d.die1 * d.die2 = d.die1 * d.die2 / 15

/-- Theorem stating that the minimum number of faces on two dice satisfying the conditions is 27 --/
theorem min_faces_for_conditions :
  ∀ d : DicePair, satisfiesConditions d → d.die1 + d.die2 ≥ 27 :=
sorry

end NUMINAMATH_CALUDE_min_faces_for_conditions_l2855_285569


namespace NUMINAMATH_CALUDE_farmer_ploughing_problem_l2855_285555

/-- Farmer's field ploughing problem -/
theorem farmer_ploughing_problem 
  (initial_daily_area : ℝ) 
  (productivity_increase : ℝ) 
  (days_ahead : ℕ) 
  (total_field_area : ℝ) 
  (h1 : initial_daily_area = 120)
  (h2 : productivity_increase = 0.25)
  (h3 : days_ahead = 2)
  (h4 : total_field_area = 1440) :
  ∃ (planned_days : ℕ) (actual_days : ℕ),
    planned_days = 10 ∧ 
    actual_days = planned_days - days_ahead ∧
    actual_days * initial_daily_area + 
      (planned_days - actual_days) * (initial_daily_area * (1 + productivity_increase)) = 
    total_field_area :=
by sorry

end NUMINAMATH_CALUDE_farmer_ploughing_problem_l2855_285555


namespace NUMINAMATH_CALUDE_chords_from_eight_points_l2855_285518

/-- The number of chords that can be drawn from n points on a circle's circumference -/
def num_chords (n : ℕ) : ℕ := n.choose 2

/-- Theorem: The number of chords from 8 points on a circle is 28 -/
theorem chords_from_eight_points : num_chords 8 = 28 := by
  sorry

end NUMINAMATH_CALUDE_chords_from_eight_points_l2855_285518


namespace NUMINAMATH_CALUDE_exists_number_with_removable_digit_l2855_285591

-- Define a function to check if a number has a non-zero digit
def has_nonzero_digit (n : ℕ) : Prop :=
  ∃ (k : ℕ), (n / 10^k) % 10 ≠ 0

-- Define a function to check if a number can be obtained by removing a non-zero digit from another number
def can_remove_nonzero_digit (n n' : ℕ) : Prop :=
  ∃ (k : ℕ), 
    let d := (n / 10^k) % 10
    d ≠ 0 ∧ n' = (n / 10^(k+1)) * 10^k + n % 10^k

theorem exists_number_with_removable_digit (d : ℕ) (hd : d > 0) : 
  ∃ (n : ℕ), 
    n % d = 0 ∧ 
    has_nonzero_digit n ∧ 
    ∃ (n' : ℕ), can_remove_nonzero_digit n n' ∧ n' % d = 0 :=
sorry

end NUMINAMATH_CALUDE_exists_number_with_removable_digit_l2855_285591


namespace NUMINAMATH_CALUDE_cone_properties_l2855_285529

-- Define the cone
structure Cone where
  generatrix : ℝ
  base_diameter : ℝ

-- Define the theorem
theorem cone_properties (c : Cone) 
  (h1 : c.generatrix = 2 * Real.sqrt 5)
  (h2 : c.base_diameter = 4) :
  -- 1. Volume of the cone
  let volume := (1/3) * Real.pi * (c.base_diameter/2)^2 * Real.sqrt ((2*Real.sqrt 5)^2 - (c.base_diameter/2)^2)
  volume = (16/3) * Real.pi ∧
  -- 2. Minimum distance from any point on a parallel section to the vertex
  let min_distance := (4/5) * Real.sqrt 5
  (∀ r : ℝ, r > 0 → r < c.base_diameter/2 → 
    ∀ p : ℝ × ℝ, p.1^2 + p.2^2 = r^2 → 
      p.1^2 + (Real.sqrt ((2*Real.sqrt 5)^2 - (c.base_diameter/2)^2) - (c.base_diameter/2 - r))^2 ≥ min_distance^2) ∧
  -- 3. Area of the section when it's the center of the circumscribed sphere
  let section_radius := (3/5) * c.base_diameter/2
  Real.pi * section_radius^2 = (36/25) * Real.pi := by
sorry

end NUMINAMATH_CALUDE_cone_properties_l2855_285529


namespace NUMINAMATH_CALUDE_trigonometric_identity_l2855_285504

theorem trigonometric_identity (c : ℝ) (h : c = 2 * Real.pi / 9) :
  (Real.sin (2 * c) * Real.sin (5 * c) * Real.sin (8 * c) * Real.sin (11 * c) * Real.sin (14 * c)) /
  (Real.sin c * Real.sin (3 * c) * Real.sin (4 * c) * Real.sin (7 * c) * Real.sin (8 * c)) =
  Real.sin (80 * Real.pi / 180) :=
by sorry

end NUMINAMATH_CALUDE_trigonometric_identity_l2855_285504


namespace NUMINAMATH_CALUDE_at_least_one_outstanding_equiv_l2855_285588

/-- Represents whether a person is an outstanding student -/
def IsOutstandingStudent (person : Prop) : Prop := person

/-- The statement "At least one of person A and person B is an outstanding student" -/
def AtLeastOneOutstanding (A B : Prop) : Prop :=
  IsOutstandingStudent A ∨ IsOutstandingStudent B

theorem at_least_one_outstanding_equiv (A B : Prop) :
  AtLeastOneOutstanding A B ↔ (IsOutstandingStudent A ∨ IsOutstandingStudent B) :=
sorry

end NUMINAMATH_CALUDE_at_least_one_outstanding_equiv_l2855_285588


namespace NUMINAMATH_CALUDE_remainder_problem_l2855_285528

theorem remainder_problem (m : ℤ) (k : ℤ) (h : m = 100 * k - 2) : 
  (m^2 + 4*m + 6) % 100 = 2 := by
sorry

end NUMINAMATH_CALUDE_remainder_problem_l2855_285528


namespace NUMINAMATH_CALUDE_red_sector_overlap_l2855_285514

theorem red_sector_overlap (n : ℕ) (red_sectors : ℕ) (h1 : n = 1965) (h2 : red_sectors = 200) :
  ∃ (positions : Finset ℕ), 
    (Finset.card positions ≥ 60) ∧ 
    (∀ p ∈ positions, p < n) ∧
    (∀ p ∈ positions, (red_sectors * red_sectors - n * 20) / n ≤ red_sectors - 
      (red_sectors * red_sectors - (n - p) * red_sectors) / n) :=
sorry

end NUMINAMATH_CALUDE_red_sector_overlap_l2855_285514


namespace NUMINAMATH_CALUDE_no_integer_a_with_one_integer_solution_l2855_285587

theorem no_integer_a_with_one_integer_solution :
  ¬ ∃ (a : ℤ), ∃! (x : ℤ), x^3 - a*x^2 - 6*a*x + a^2 - 3 = 0 :=
by sorry

end NUMINAMATH_CALUDE_no_integer_a_with_one_integer_solution_l2855_285587


namespace NUMINAMATH_CALUDE_quadratic_equations_solutions_l2855_285536

theorem quadratic_equations_solutions :
  (∃ x1 x2 : ℝ, (x1^2 - 4*x1 = 5 ∧ x2^2 - 4*x2 = 5) ∧ (x1 = 5 ∧ x2 = -1)) ∧
  (∃ y1 y2 : ℝ, (y1^2 + 7*y1 - 18 = 0 ∧ y2^2 + 7*y2 - 18 = 0) ∧ (y1 = -9 ∧ y2 = 2)) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equations_solutions_l2855_285536


namespace NUMINAMATH_CALUDE_pie_eating_contest_l2855_285502

theorem pie_eating_contest (student1 student2 student3 : ℚ) 
  (h1 : student1 = 5/6)
  (h2 : student2 = 7/8)
  (h3 : student3 = 1/2) :
  student1 + student2 - student3 = 29/24 := by
  sorry

end NUMINAMATH_CALUDE_pie_eating_contest_l2855_285502


namespace NUMINAMATH_CALUDE_infinite_binomial_congruence_pairs_l2855_285533

theorem infinite_binomial_congruence_pairs :
  ∀ p : ℕ, Prime p → p ≠ 2 →
  ∃ a b : ℕ,
    a > b ∧
    a + b = 2 * p ∧
    (Nat.choose (2 * p) a) % (2 * p) = (Nat.choose (2 * p) b) % (2 * p) ∧
    (Nat.choose (2 * p) a) % (2 * p) ≠ 0 :=
by sorry

end NUMINAMATH_CALUDE_infinite_binomial_congruence_pairs_l2855_285533


namespace NUMINAMATH_CALUDE_complex_sum_problem_l2855_285548

theorem complex_sum_problem (p q r s t u : ℝ) : 
  q = 4 → 
  t = -p - r → 
  (p + q * I) + (r + s * I) + (t + u * I) = 3 * I → 
  s + u = -1 := by
sorry

end NUMINAMATH_CALUDE_complex_sum_problem_l2855_285548


namespace NUMINAMATH_CALUDE_value_of_y_l2855_285506

theorem value_of_y (x y : ℝ) (h1 : x - y = 16) (h2 : x + y = 4) : y = -6 := by
  sorry

end NUMINAMATH_CALUDE_value_of_y_l2855_285506


namespace NUMINAMATH_CALUDE_derivative_f_at_zero_l2855_285597

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≠ 0 then
    Real.sqrt (1 + Real.log (1 + 3 * x^2 * Real.cos (2 / x))) - 1
  else
    0

-- State the theorem
theorem derivative_f_at_zero :
  deriv f 0 = 0 := by
  sorry

end NUMINAMATH_CALUDE_derivative_f_at_zero_l2855_285597


namespace NUMINAMATH_CALUDE_romeo_chocolate_profit_l2855_285503

/-- Calculates the profit for Romeo's chocolate business -/
theorem romeo_chocolate_profit :
  let total_revenue : ℕ := 340
  let chocolate_cost : ℕ := 175
  let packaging_cost : ℕ := 60
  let advertising_cost : ℕ := 20
  let total_cost : ℕ := chocolate_cost + packaging_cost + advertising_cost
  let profit : ℕ := total_revenue - total_cost
  profit = 85 := by sorry

end NUMINAMATH_CALUDE_romeo_chocolate_profit_l2855_285503


namespace NUMINAMATH_CALUDE_equation_solution_l2855_285542

theorem equation_solution : ∃! x : ℝ, (3 / (x - 3) = 1 / (x - 1)) ∧ x = 0 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2855_285542


namespace NUMINAMATH_CALUDE_jar_flipping_problem_l2855_285534

theorem jar_flipping_problem (total_jars : Nat) (max_flip_per_move : Nat) (n_upper_bound : Nat) : 
  total_jars = 343 →
  max_flip_per_move = 27 →
  n_upper_bound = 2021 →
  (∃ (n : Nat), n ≥ (total_jars + max_flip_per_move - 1) / max_flip_per_move ∧ 
                n ≤ n_upper_bound ∧
                n % 2 = 1) →
  (Finset.filter (fun x => x % 2 = 1) (Finset.range (n_upper_bound + 1))).card = 1005 := by
sorry

end NUMINAMATH_CALUDE_jar_flipping_problem_l2855_285534


namespace NUMINAMATH_CALUDE_store_dvds_count_l2855_285549

def total_dvds : ℕ := 10
def online_dvds : ℕ := 2

theorem store_dvds_count : total_dvds - online_dvds = 8 := by
  sorry

end NUMINAMATH_CALUDE_store_dvds_count_l2855_285549


namespace NUMINAMATH_CALUDE_initial_population_initial_population_approx_l2855_285532

/-- Calculates the initial population of a village given the population changes over 5 years and the final population. -/
theorem initial_population (final_population : ℝ) : ℝ :=
  let year1_change := 1.05
  let year2_change := 0.93
  let year3_change := 1.03
  let year4_change := 1.10
  let year5_change := 0.95
  final_population / (year1_change * year2_change * year3_change * year4_change * year5_change)

/-- The initial population of the village is approximately 10,457. -/
theorem initial_population_approx : 
  ∃ (ε : ℝ), ε > 0 ∧ ε < 1 ∧ |initial_population 10450 - 10457| < ε :=
sorry

end NUMINAMATH_CALUDE_initial_population_initial_population_approx_l2855_285532


namespace NUMINAMATH_CALUDE_total_yellow_balls_is_30_l2855_285540

/-- The number of boxes containing balls -/
def num_boxes : ℕ := 6

/-- The number of yellow balls in each box -/
def yellow_balls_per_box : ℕ := 5

/-- The total number of yellow balls across all boxes -/
def total_yellow_balls : ℕ := num_boxes * yellow_balls_per_box

theorem total_yellow_balls_is_30 : total_yellow_balls = 30 := by
  sorry

end NUMINAMATH_CALUDE_total_yellow_balls_is_30_l2855_285540


namespace NUMINAMATH_CALUDE_quadratic_root_in_interval_l2855_285524

theorem quadratic_root_in_interval
  (a b c : ℝ)
  (h_two_roots : ∃ x y : ℝ, x ≠ y ∧ a * x^2 + b * x + c = 0 ∧ a * y^2 + b * y + c = 0)
  (h_inequality : |a * (b - c)| > |b^2 - a * c| + |c^2 - a * b|) :
  ∃ α : ℝ, 0 < α ∧ α < 2 ∧ a * α^2 + b * α + c = 0 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_root_in_interval_l2855_285524


namespace NUMINAMATH_CALUDE_pens_problem_l2855_285507

theorem pens_problem (initial_pens : ℕ) (final_pens : ℕ) (sharon_pens : ℕ) 
  (h1 : initial_pens = 20)
  (h2 : final_pens = 65)
  (h3 : sharon_pens = 19) :
  ∃ (mike_pens : ℕ), 2 * (initial_pens + mike_pens) - sharon_pens = final_pens ∧ mike_pens = 22 := by
  sorry

end NUMINAMATH_CALUDE_pens_problem_l2855_285507


namespace NUMINAMATH_CALUDE_sphere_volume_from_inscribed_box_l2855_285550

/-- The volume of a sphere given an inscribed rectangular box --/
theorem sphere_volume_from_inscribed_box (AB BC AA₁ : ℝ) (h1 : AB = 2) (h2 : BC = 2) (h3 : AA₁ = 2 * Real.sqrt 2) :
  let box_diagonal := Real.sqrt (AB^2 + BC^2 + AA₁^2)
  let sphere_radius := box_diagonal / 2
  let sphere_volume := (4 / 3) * Real.pi * sphere_radius^3
  sphere_volume = (32 * Real.pi) / 3 := by
  sorry

end NUMINAMATH_CALUDE_sphere_volume_from_inscribed_box_l2855_285550


namespace NUMINAMATH_CALUDE_correct_result_l2855_285560

def add_subtract_round (a b c : ℕ) : ℕ :=
  let result := a + b - c
  let remainder := result % 5
  if remainder < 3 then result - remainder else result + (5 - remainder)

theorem correct_result : add_subtract_round 82 56 15 = 125 := by
  sorry

end NUMINAMATH_CALUDE_correct_result_l2855_285560


namespace NUMINAMATH_CALUDE_arithmetic_progression_1980_l2855_285509

/-- An arithmetic progression of natural numbers. -/
structure ArithProgression where
  first : ℕ
  diff : ℕ

/-- Check if a natural number belongs to an arithmetic progression. -/
def belongsTo (n : ℕ) (ap : ArithProgression) : Prop :=
  ∃ k : ℕ, n = ap.first + k * ap.diff

/-- The main theorem statement. -/
theorem arithmetic_progression_1980 (P₁ P₂ P₃ : ArithProgression) :
  (∀ n : ℕ, n ≤ 8 → belongsTo n P₁ ∨ belongsTo n P₂ ∨ belongsTo n P₃) →
  belongsTo 1980 P₁ ∨ belongsTo 1980 P₂ ∨ belongsTo 1980 P₃ := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_progression_1980_l2855_285509


namespace NUMINAMATH_CALUDE_cubic_geometric_roots_property_l2855_285544

/-- A cubic equation with coefficients a, b, c has three nonzero real roots in geometric progression -/
structure CubicWithGeometricRoots (a b c : ℝ) : Prop where
  roots_exist : ∃ (d q : ℝ), d ≠ 0 ∧ q ≠ 0 ∧ q ≠ 1
  root_equation : ∀ (d q : ℝ), d ≠ 0 → q ≠ 0 → q ≠ 1 →
    d^3 + a*d^2 + b*d + c = 0 ∧
    (d*q)^3 + a*(d*q)^2 + b*(d*q) + c = 0 ∧
    (d*q^2)^3 + a*(d*q^2)^2 + b*(d*q^2) + c = 0

/-- The main theorem -/
theorem cubic_geometric_roots_property {a b c : ℝ} (h : CubicWithGeometricRoots a b c) :
  a^3 * c - b^3 = 0 := by
  sorry

end NUMINAMATH_CALUDE_cubic_geometric_roots_property_l2855_285544


namespace NUMINAMATH_CALUDE_equal_expressions_l2855_285543

theorem equal_expressions : 
  (2^3 ≠ 3^2) ∧ 
  (-3^3 = (-3)^3) ∧ 
  (-2^2 ≠ (-2)^2) ∧ 
  (-|-2| ≠ -(-2)) := by
  sorry

end NUMINAMATH_CALUDE_equal_expressions_l2855_285543


namespace NUMINAMATH_CALUDE_rectangular_map_area_l2855_285583

/-- The area of a rectangular map with given length and width. -/
def map_area (length width : ℝ) : ℝ := length * width

/-- Theorem: The area of a rectangular map with length 5 meters and width 2 meters is 10 square meters. -/
theorem rectangular_map_area :
  map_area 5 2 = 10 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_map_area_l2855_285583


namespace NUMINAMATH_CALUDE_root_equation_q_value_l2855_285530

theorem root_equation_q_value (a b m p q : ℝ) : 
  (a^2 - m*a + (3/2) = 0) →
  (b^2 - m*b + (3/2) = 0) →
  ((a + 1/b)^2 - p*(a + 1/b) + q = 0) →
  ((b + 1/a)^2 - p*(b + 1/a) + q = 0) →
  q = 19/6 := by
sorry

end NUMINAMATH_CALUDE_root_equation_q_value_l2855_285530


namespace NUMINAMATH_CALUDE_unique_intersection_point_l2855_285546

def f (x : ℝ) : ℝ := x^3 + 3*x^2 + 9*x + 15

theorem unique_intersection_point :
  ∃! a : ℝ, f a = a ∧ a = -1 := by sorry

end NUMINAMATH_CALUDE_unique_intersection_point_l2855_285546


namespace NUMINAMATH_CALUDE_sphere_surface_area_of_circumscribed_cube_l2855_285516

theorem sphere_surface_area_of_circumscribed_cube (R : ℝ) :
  let cube_edge_1 : ℝ := 2
  let cube_edge_2 : ℝ := 3
  let cube_edge_3 : ℝ := 1
  let cube_diagonal : ℝ := (cube_edge_1^2 + cube_edge_2^2 + cube_edge_3^2).sqrt
  R = cube_diagonal / 2 →
  4 * Real.pi * R^2 = 14 * Real.pi :=
by sorry

end NUMINAMATH_CALUDE_sphere_surface_area_of_circumscribed_cube_l2855_285516


namespace NUMINAMATH_CALUDE_solution_set_part1_solution_set_part2_l2855_285554

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x - a|

-- Part 1
theorem solution_set_part1 :
  {x : ℝ | |x - 2| ≥ 4 - |x - 4|} = {x : ℝ | x ≤ 1 ∨ x ≥ 5} :=
by sorry

-- Part 2
theorem solution_set_part2 (a : ℝ) (h : a > 1) :
  ({x : ℝ | |f a (2*x + a) - 2*f a x| ≤ 2} = {x : ℝ | 1 ≤ x ∧ x ≤ 2}) →
  a = 3 :=
by sorry

end NUMINAMATH_CALUDE_solution_set_part1_solution_set_part2_l2855_285554


namespace NUMINAMATH_CALUDE_sqrt_sum_inequality_l2855_285512

theorem sqrt_sum_inequality : Real.sqrt 2 + Real.sqrt 3 ≠ Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_inequality_l2855_285512


namespace NUMINAMATH_CALUDE_peanut_eating_interval_l2855_285510

/-- Proves that given a flight duration of 2 hours and 4 bags of peanuts with 30 peanuts each,
    if all peanuts are consumed at equally spaced intervals during the flight,
    the time between eating each peanut is 1 minute. -/
theorem peanut_eating_interval (flight_duration : ℕ) (bags : ℕ) (peanuts_per_bag : ℕ) :
  flight_duration = 2 →
  bags = 4 →
  peanuts_per_bag = 30 →
  (flight_duration * 60) / (bags * peanuts_per_bag) = 1 := by
  sorry

#check peanut_eating_interval

end NUMINAMATH_CALUDE_peanut_eating_interval_l2855_285510


namespace NUMINAMATH_CALUDE_outfit_combinations_l2855_285577

/-- Represents the number of items of each type (shirts, pants, hats) -/
def num_items : ℕ := 7

/-- Represents the number of colors available for each item type -/
def num_colors : ℕ := 7

/-- Calculates the number of valid outfit combinations where no two items are the same color -/
def valid_outfits : ℕ := num_colors * (num_colors - 1) * (num_colors - 2)

/-- Proves that the number of valid outfit combinations is 210 -/
theorem outfit_combinations :
  valid_outfits = 210 :=
by sorry

end NUMINAMATH_CALUDE_outfit_combinations_l2855_285577


namespace NUMINAMATH_CALUDE_f_composition_value_l2855_285522

noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 0 then Real.sin (5 * Real.pi * x / 2)
  else 1/6 - Real.log x / Real.log 3

theorem f_composition_value : f (f (3 * Real.sqrt 3)) = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_f_composition_value_l2855_285522


namespace NUMINAMATH_CALUDE_shopping_tax_calculation_l2855_285547

theorem shopping_tax_calculation (total_amount : ℝ) (h_positive : total_amount > 0) :
  let clothing_percent : ℝ := 0.5
  let food_percent : ℝ := 0.25
  let other_percent : ℝ := 0.25
  let clothing_tax_rate : ℝ := 0.1
  let food_tax_rate : ℝ := 0
  let other_tax_rate : ℝ := 0.2
  
  let clothing_amount := clothing_percent * total_amount
  let food_amount := food_percent * total_amount
  let other_amount := other_percent * total_amount
  
  let clothing_tax := clothing_amount * clothing_tax_rate
  let food_tax := food_amount * food_tax_rate
  let other_tax := other_amount * other_tax_rate
  
  let total_tax := clothing_tax + food_tax + other_tax
  
  (total_tax / total_amount) = 0.1 := by sorry

end NUMINAMATH_CALUDE_shopping_tax_calculation_l2855_285547


namespace NUMINAMATH_CALUDE_deduce_day_from_statements_l2855_285513

structure Animal where
  name : String
  lying_days : Finset Nat

def day_of_week (d : Nat) : Nat :=
  d % 7

theorem deduce_day_from_statements
  (lion unicorn : Animal)
  (today yesterday : Nat)
  (h_lion_statement : day_of_week yesterday ∈ lion.lying_days)
  (h_unicorn_statement : day_of_week yesterday ∈ unicorn.lying_days)
  (h_common_lying_day : ∃! d, d ∈ lion.lying_days ∧ d ∈ unicorn.lying_days)
  (h_today_yesterday : day_of_week today = (day_of_week yesterday + 1) % 7) :
  ∃ (common_day : Nat),
    day_of_week yesterday = common_day ∧
    common_day ∈ lion.lying_days ∧
    common_day ∈ unicorn.lying_days ∧
    day_of_week today = (common_day + 1) % 7 :=
by sorry

end NUMINAMATH_CALUDE_deduce_day_from_statements_l2855_285513


namespace NUMINAMATH_CALUDE_intersection_characterization_l2855_285572

def M : Set ℝ := {x | x > 1}
def N : Set ℝ := {x | x^2 - 2*x < 0}

theorem intersection_characterization :
  ∀ x : ℝ, x ∈ (M ∩ N) ↔ (1 < x ∧ x < 2) :=
sorry

end NUMINAMATH_CALUDE_intersection_characterization_l2855_285572


namespace NUMINAMATH_CALUDE_inequality_proof_l2855_285525

theorem inequality_proof (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (h : a > b) :
  1 / (a * b^2) > 1 / (a^2 * b) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2855_285525


namespace NUMINAMATH_CALUDE_intersection_properties_l2855_285592

/-- Two lines intersecting at point P -/
def line1 (m : ℝ) (x y : ℝ) : Prop := m * x - y - 3 * m + 1 = 0
def line2 (m : ℝ) (x y : ℝ) : Prop := x + m * y - 3 * m - 1 = 0

/-- Circle C -/
def circle_C (x y : ℝ) : Prop := (x + 2)^2 + (y + 1)^2 = 4

/-- Point P satisfies both lines -/
def point_P (m : ℝ) (x y : ℝ) : Prop := line1 m x y ∧ line2 m x y

/-- AB is a chord of circle C with length 2√3 -/
def chord_AB (xa ya xb yb : ℝ) : Prop :=
  circle_C xa ya ∧ circle_C xb yb ∧ (xa - xb)^2 + (ya - yb)^2 = 12

/-- Q is the midpoint of AB -/
def midpoint_Q (xa ya xb yb xq yq : ℝ) : Prop :=
  xq = (xa + xb) / 2 ∧ yq = (ya + yb) / 2

theorem intersection_properties (m : ℝ) :
  ∃ x y xa ya xb yb xq yq,
    point_P m x y ∧
    chord_AB xa ya xb yb ∧
    midpoint_Q xa ya xb yb xq yq →
    (¬ circle_C x y) ∧  -- P lies outside circle C
    (∃ pq_max, pq_max = 6 + Real.sqrt 2 ∧
      ∀ x' y', point_P m x' y' →
        ∀ xa' ya' xb' yb' xq' yq',
          chord_AB xa' ya' xb' yb' ∧
          midpoint_Q xa' ya' xb' yb' xq' yq' →
            ((x' - xq')^2 + (y' - yq')^2)^(1/2) ≤ pq_max) ∧  -- Max length of PQ
    (∃ pa_pb_min, pa_pb_min = 15 - 8 * Real.sqrt 2 ∧
      ∀ x' y', point_P m x' y' →
        ∀ xa' ya' xb' yb',
          chord_AB xa' ya' xb' yb' →
            (x' - xa') * (x' - xb') + (y' - ya') * (y' - yb') ≥ pa_pb_min)  -- Min value of PA · PB
  := by sorry

end NUMINAMATH_CALUDE_intersection_properties_l2855_285592


namespace NUMINAMATH_CALUDE_ratio_of_numbers_l2855_285523

theorem ratio_of_numbers (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a > b) (h4 : a + b = 7 * (a - b)) :
  a / b = 2 := by
sorry

end NUMINAMATH_CALUDE_ratio_of_numbers_l2855_285523


namespace NUMINAMATH_CALUDE_orange_theorem_l2855_285561

def orange_problem (betty_oranges bill_oranges frank_multiplier seeds_per_orange oranges_per_tree : ℕ) : ℕ :=
  let total_betty_bill := betty_oranges + bill_oranges
  let frank_oranges := frank_multiplier * total_betty_bill
  let total_seeds := frank_oranges * seeds_per_orange
  let total_oranges := total_seeds * oranges_per_tree
  total_oranges

theorem orange_theorem : 
  orange_problem 15 12 3 2 5 = 810 := by
  sorry

end NUMINAMATH_CALUDE_orange_theorem_l2855_285561


namespace NUMINAMATH_CALUDE_tip_difference_calculation_l2855_285558

/-- Calculates the difference in euro cents between a good tip and a bad tip -/
def tip_difference (initial_bill : ℝ) (bad_tip_percent : ℝ) (good_tip_percent : ℝ) 
  (discount_percent : ℝ) (tax_percent : ℝ) (usd_to_eur : ℝ) : ℝ :=
  let discounted_bill := initial_bill * (1 - discount_percent)
  let final_bill := discounted_bill * (1 + tax_percent)
  let bad_tip := final_bill * bad_tip_percent
  let good_tip := final_bill * good_tip_percent
  let difference_usd := good_tip - bad_tip
  let difference_eur := difference_usd * usd_to_eur
  difference_eur * 100  -- Convert to cents

theorem tip_difference_calculation :
  tip_difference 26 0.05 0.20 0.08 0.07 0.85 = 326.33 := by
  sorry

end NUMINAMATH_CALUDE_tip_difference_calculation_l2855_285558


namespace NUMINAMATH_CALUDE_carton_height_proof_l2855_285599

/-- Given a carton and soap boxes with specific dimensions, prove the height of the carton. -/
theorem carton_height_proof (carton_length carton_width carton_height : ℝ)
  (box_length box_width box_height : ℝ) (max_boxes : ℕ) :
  carton_length = 25 ∧ 
  carton_width = 48 ∧
  box_length = 8 ∧
  box_width = 6 ∧
  box_height = 5 ∧
  max_boxes = 300 ∧
  (carton_length * carton_width * carton_height) = 
    (↑max_boxes * box_length * box_width * box_height) →
  carton_height = 60 := by
sorry

end NUMINAMATH_CALUDE_carton_height_proof_l2855_285599


namespace NUMINAMATH_CALUDE_ellipse_focus_d_l2855_285519

/-- An ellipse in the first quadrant tangent to both x-axis and y-axis with foci at (4,8) and (d,8) -/
structure Ellipse where
  d : ℝ
  tangent_to_axes : Bool
  in_first_quadrant : Bool
  focus1 : ℝ × ℝ := (4, 8)
  focus2 : ℝ × ℝ := (d, 8)

/-- The value of d for the given ellipse is 15 -/
theorem ellipse_focus_d (e : Ellipse) (h1 : e.tangent_to_axes) (h2 : e.in_first_quadrant) :
  e.d = 15 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_focus_d_l2855_285519


namespace NUMINAMATH_CALUDE_friday_lunch_customers_l2855_285520

theorem friday_lunch_customers (breakfast : ℕ) (dinner : ℕ) (saturday_prediction : ℕ) :
  breakfast = 73 →
  dinner = 87 →
  saturday_prediction = 574 →
  ∃ (lunch : ℕ), lunch = saturday_prediction / 2 - breakfast - dinner ∧ lunch = 127 :=
by sorry

end NUMINAMATH_CALUDE_friday_lunch_customers_l2855_285520
