import Mathlib

namespace NUMINAMATH_CALUDE_sum_extrema_l1317_131762

theorem sum_extrema (x y z w : ℝ) (h_nonneg : x ≥ 0 ∧ y ≥ 0 ∧ z ≥ 0 ∧ w ≥ 0) 
  (h_eq : x^2 + y^2 + z^2 + w^2 + x + 2*y + 3*z + 4*w = 17/2) : 
  (∀ a b c d : ℝ, a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧ d ≥ 0 → 
    a^2 + b^2 + c^2 + d^2 + a + 2*b + 3*c + 4*d = 17/2 → 
    a + b + c + d ≤ 3) ∧ 
  (x + y + z + w ≥ -2 + 5/2 * Real.sqrt 2) ∧
  (∃ a b c d : ℝ, a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧ d ≥ 0 ∧ 
    a^2 + b^2 + c^2 + d^2 + a + 2*b + 3*c + 4*d = 17/2 ∧ 
    a + b + c + d = 3) ∧
  (∃ a b c d : ℝ, a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧ d ≥ 0 ∧ 
    a^2 + b^2 + c^2 + d^2 + a + 2*b + 3*c + 4*d = 17/2 ∧ 
    a + b + c + d = -2 + 5/2 * Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_sum_extrema_l1317_131762


namespace NUMINAMATH_CALUDE_ellipse_eccentricity_l1317_131777

/-- The eccentricity of an ellipse with equation x²/a² + y² = 1, where a > 1 and the major axis length is 4 -/
theorem ellipse_eccentricity (a : ℝ) (h1 : a > 1) (h2 : 2 * a = 4) :
  let c := Real.sqrt (a^2 - 1)
  (c / a) = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_eccentricity_l1317_131777


namespace NUMINAMATH_CALUDE_thief_reasoning_flaw_l1317_131769

/-- Represents the components of the thief's argument --/
inductive ArgumentComponent
  | MajorPremise
  | MinorPremise
  | Conclusion

/-- Represents the thief's ability to open a video recorder --/
def can_open (x : Prop) : Prop := x

/-- Represents the ownership of the video recorder --/
def is_mine (x : Prop) : Prop := x

/-- The thief's argument structure --/
def thief_argument (recorder : Prop) : Prop :=
  (is_mine recorder → can_open recorder) ∧
  (can_open recorder) ∧
  (is_mine recorder)

/-- The flaw in the thief's reasoning --/
def flaw_in_reasoning (component : ArgumentComponent) : Prop :=
  component = ArgumentComponent.MajorPremise

/-- Theorem stating that the flaw in the thief's reasoning is in the major premise --/
theorem thief_reasoning_flaw (recorder : Prop) :
  thief_argument recorder → flaw_in_reasoning ArgumentComponent.MajorPremise :=
by sorry

end NUMINAMATH_CALUDE_thief_reasoning_flaw_l1317_131769


namespace NUMINAMATH_CALUDE_count_squares_below_line_l1317_131711

/-- The number of 1x1 squares in the first quadrant entirely below the line 6x + 143y = 858 -/
def squares_below_line : ℕ :=
  355

/-- The equation of the line -/
def line_equation (x y : ℚ) : Prop :=
  6 * x + 143 * y = 858

theorem count_squares_below_line :
  squares_below_line = 355 :=
sorry

end NUMINAMATH_CALUDE_count_squares_below_line_l1317_131711


namespace NUMINAMATH_CALUDE_ghee_composition_l1317_131738

theorem ghee_composition (original_quantity : ℝ) (vanaspati_percentage : ℝ) 
  (added_pure_ghee : ℝ) (new_vanaspati_percentage : ℝ) :
  original_quantity = 10 →
  vanaspati_percentage = 40 →
  added_pure_ghee = 10 →
  new_vanaspati_percentage = 20 →
  (vanaspati_percentage / 100) * original_quantity = 
    (new_vanaspati_percentage / 100) * (original_quantity + added_pure_ghee) →
  (100 - vanaspati_percentage) = 60 := by
sorry

end NUMINAMATH_CALUDE_ghee_composition_l1317_131738


namespace NUMINAMATH_CALUDE_age_problem_l1317_131725

theorem age_problem (a b c : ℕ) : 
  a = b + 2 →
  b = 2 * c →
  a + b + c = 17 →
  b = 6 :=
by
  sorry

end NUMINAMATH_CALUDE_age_problem_l1317_131725


namespace NUMINAMATH_CALUDE_sprite_to_coke_ratio_l1317_131764

/-- Represents a drink mixture with three components -/
structure Drink where
  total : ℝ
  coke : ℝ
  sprite : ℝ
  mountainDew : ℝ
  cokeParts : ℝ
  mountainDewParts : ℝ

/-- Theorem stating the ratio of Sprite to Coke in the drink -/
theorem sprite_to_coke_ratio (d : Drink) 
  (h1 : d.total = 18)
  (h2 : d.coke = 6)
  (h3 : d.cokeParts = 2)
  (h4 : d.mountainDewParts = 3)
  (h5 : d.total = d.coke + d.sprite + d.mountainDew)
  (h6 : d.coke / d.cokeParts = d.mountainDew / d.mountainDewParts) : 
  d.sprite / d.coke = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sprite_to_coke_ratio_l1317_131764


namespace NUMINAMATH_CALUDE_equal_selection_probability_l1317_131746

/-- Represents the selection process for voluntary labor --/
structure SelectionProcess where
  total_students : ℕ
  excluded : ℕ
  selected : ℕ
  h_total : total_students = 1008
  h_excluded : excluded = 8
  h_selected : selected = 20
  h_remaining : total_students - excluded = 1000

/-- The probability of being selected for an individual student --/
def selection_probability (process : SelectionProcess) : ℚ :=
  process.selected / process.total_students

/-- States that the selection probability is equal for all students --/
theorem equal_selection_probability (process : SelectionProcess) :
  ∀ (student1 student2 : Fin process.total_students),
    selection_probability process = selection_probability process :=
by sorry

end NUMINAMATH_CALUDE_equal_selection_probability_l1317_131746


namespace NUMINAMATH_CALUDE_units_digit_of_7_pow_5_l1317_131744

-- Define a function to get the units digit of a natural number
def unitsDigit (n : ℕ) : ℕ := n % 10

-- State the theorem
theorem units_digit_of_7_pow_5 : unitsDigit (7^5) = 7 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_7_pow_5_l1317_131744


namespace NUMINAMATH_CALUDE_min_cuboids_for_cube_l1317_131755

def cuboid_length : ℕ := 3
def cuboid_width : ℕ := 4
def cuboid_height : ℕ := 5

theorem min_cuboids_for_cube : 
  let cube_side := Nat.lcm (Nat.lcm cuboid_length cuboid_width) cuboid_height
  let cube_volume := cube_side ^ 3
  let cuboid_volume := cuboid_length * cuboid_width * cuboid_height
  cube_volume / cuboid_volume = 3600 := by
  sorry

end NUMINAMATH_CALUDE_min_cuboids_for_cube_l1317_131755


namespace NUMINAMATH_CALUDE_total_license_plates_l1317_131748

/-- The number of vowels in the license plate system -/
def num_vowels : ℕ := 8

/-- The number of consonants in the license plate system -/
def num_consonants : ℕ := 26 - num_vowels

/-- The number of even digits (0, 2, 4, 6, 8) -/
def num_even_digits : ℕ := 5

/-- The structure of a license plate: consonant, vowel, consonant, even digit, even digit -/
def license_plate_structure := 
  num_consonants * num_vowels * num_consonants * num_even_digits * num_even_digits

/-- The total number of possible license plates -/
theorem total_license_plates : license_plate_structure = 25920 := by
  sorry

end NUMINAMATH_CALUDE_total_license_plates_l1317_131748


namespace NUMINAMATH_CALUDE_problem_solution_l1317_131713

theorem problem_solution (x y : ℝ) (hx : x = 2 + Real.sqrt 3) (hy : y = 2 - Real.sqrt 3) :
  (x^2 + y^2 = 14) ∧ (x / y - y / x = 8 * Real.sqrt 3) := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1317_131713


namespace NUMINAMATH_CALUDE_no_solution_exponential_equation_l1317_131798

theorem no_solution_exponential_equation :
  ¬ ∃ z : ℝ, (16 : ℝ) ^ (3 * z) = (64 : ℝ) ^ (2 * z + 5) := by
  sorry

end NUMINAMATH_CALUDE_no_solution_exponential_equation_l1317_131798


namespace NUMINAMATH_CALUDE_arctan_equation_solution_l1317_131732

theorem arctan_equation_solution (x : ℝ) :
  Real.arctan (1 / x) + Real.arctan (1 / x^3) = π / 4 → x = (1 + Real.sqrt 5) / 2 := by
sorry

end NUMINAMATH_CALUDE_arctan_equation_solution_l1317_131732


namespace NUMINAMATH_CALUDE_minimum_age_vasily_l1317_131783

theorem minimum_age_vasily (n : ℕ) (h_n : n = 64) :
  ∃ (V F : ℕ),
    V = F + 2 ∧
    F ≥ 5 ∧
    (∀ k : ℕ, k ≥ F → Nat.choose n k > Nat.choose n (k + 2)) ∧
    (∀ V' F' : ℕ, V' = F' + 2 → F' ≥ 5 → 
      (∀ k : ℕ, k ≥ F' → Nat.choose n k > Nat.choose n (k + 2)) → V' ≥ V) ∧
    V = 34 := by
  sorry

end NUMINAMATH_CALUDE_minimum_age_vasily_l1317_131783


namespace NUMINAMATH_CALUDE_equal_cost_at_four_students_agency_a_cheaper_for_ten_students_l1317_131706

/-- Represents a travel agency with a pricing strategy -/
structure Agency where
  teacherDiscount : ℝ  -- Discount for the teacher (0 means full price)
  studentDiscount : ℝ  -- Discount for students

/-- Calculates the total cost for a given number of students -/
def totalCost (a : Agency) (numStudents : ℕ) (fullPrice : ℝ) : ℝ :=
  fullPrice * (1 - a.teacherDiscount) + numStudents * fullPrice * (1 - a.studentDiscount)

/-- The full price of a ticket -/
def fullPrice : ℝ := 240

/-- Agency A's pricing strategy -/
def agencyA : Agency := ⟨0, 0.5⟩

/-- Agency B's pricing strategy -/
def agencyB : Agency := ⟨0.4, 0.4⟩

theorem equal_cost_at_four_students :
  ∃ n : ℕ, n = 4 ∧ totalCost agencyA n fullPrice = totalCost agencyB n fullPrice :=
sorry

theorem agency_a_cheaper_for_ten_students :
  totalCost agencyA 10 fullPrice < totalCost agencyB 10 fullPrice :=
sorry

end NUMINAMATH_CALUDE_equal_cost_at_four_students_agency_a_cheaper_for_ten_students_l1317_131706


namespace NUMINAMATH_CALUDE_equation_solution_l1317_131727

theorem equation_solution : ∃ x : ℝ, 3 * x + 2 * (x - 2) = 6 ∧ x = 2 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1317_131727


namespace NUMINAMATH_CALUDE_largest_share_in_startup_l1317_131733

def profit_split (total_profit : ℚ) (ratios : List ℚ) : List ℚ :=
  let sum_ratios := ratios.sum
  ratios.map (λ r => (r / sum_ratios) * total_profit)

theorem largest_share_in_startup (total_profit : ℚ) :
  let ratios : List ℚ := [3, 4, 4, 6, 7]
  let shares := profit_split total_profit ratios
  total_profit = 48000 →
  shares.maximum = some 14000 := by
sorry

end NUMINAMATH_CALUDE_largest_share_in_startup_l1317_131733


namespace NUMINAMATH_CALUDE_expansion_coefficient_l1317_131710

/-- The coefficient of a^2 * b^3 * c^3 in the expansion of (a + b + c)^8 -/
def coefficient_a2b3c3 : ℕ :=
  Nat.choose 8 5 * Nat.choose 5 3

theorem expansion_coefficient :
  coefficient_a2b3c3 = 560 := by sorry

end NUMINAMATH_CALUDE_expansion_coefficient_l1317_131710


namespace NUMINAMATH_CALUDE_binary_1010101_to_decimal_l1317_131780

/-- Converts a list of binary digits to its decimal equivalent -/
def binary_to_decimal (binary : List Nat) : Nat :=
  binary.enum.foldl (fun acc (i, b) => acc + b * 2^i) 0

/-- The binary representation of the number 1010101₂ -/
def binary_1010101 : List Nat := [1, 0, 1, 0, 1, 0, 1]

/-- Theorem: The decimal equivalent of 1010101₂ is 85 -/
theorem binary_1010101_to_decimal :
  binary_to_decimal binary_1010101.reverse = 85 := by
  sorry

end NUMINAMATH_CALUDE_binary_1010101_to_decimal_l1317_131780


namespace NUMINAMATH_CALUDE_stepped_design_reduces_blind_spots_l1317_131729

/-- Represents a hall design --/
structure HallDesign where
  shape : String
  is_stepped : Bool

/-- Represents the visibility in a hall --/
structure Visibility where
  blind_spots : ℕ

/-- A function that calculates visibility based on hall design --/
def calculate_visibility (design : HallDesign) : Visibility :=
  sorry

/-- The theorem stating that a stepped design reduces blind spots --/
theorem stepped_design_reduces_blind_spots 
  (flat_design stepped_design : HallDesign)
  (h1 : flat_design.shape = "flat")
  (h2 : flat_design.is_stepped = false)
  (h3 : stepped_design.shape = "stepped")
  (h4 : stepped_design.is_stepped = true) :
  (calculate_visibility stepped_design).blind_spots < (calculate_visibility flat_design).blind_spots :=
sorry

end NUMINAMATH_CALUDE_stepped_design_reduces_blind_spots_l1317_131729


namespace NUMINAMATH_CALUDE_triangle_conjugates_l1317_131784

/-- Triangle ABC with side lengths a, b, and c -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  ha : a > 0
  hb : b > 0
  hc : c > 0

/-- Barycentric coordinates -/
structure BarycentricCoord where
  α : ℝ
  β : ℝ
  γ : ℝ
  h_pos : α > 0 ∧ β > 0 ∧ γ > 0

/-- Isotomically conjugate points -/
def isIsotomicallyConjugate (tri : Triangle) (p q : BarycentricCoord) : Prop :=
  p.α * q.α = p.β * q.β ∧ p.β * q.β = p.γ * q.γ ∧ p.γ * q.γ = p.α * q.α

/-- Isogonally conjugate points -/
def isIsogonallyConjugate (tri : Triangle) (p q : BarycentricCoord) : Prop :=
  p.α * q.α = tri.a^2 ∧ p.β * q.β = tri.b^2 ∧ p.γ * q.γ = tri.c^2

/-- Main theorem -/
theorem triangle_conjugates (tri : Triangle) (p : BarycentricCoord) :
  let q₁ : BarycentricCoord := ⟨p.α⁻¹, p.β⁻¹, p.γ⁻¹, sorry⟩
  let q₂ : BarycentricCoord := ⟨tri.a^2 / p.α, tri.b^2 / p.β, tri.c^2 / p.γ, sorry⟩
  isIsotomicallyConjugate tri p q₁ ∧ isIsogonallyConjugate tri p q₂ := by
  sorry

end NUMINAMATH_CALUDE_triangle_conjugates_l1317_131784


namespace NUMINAMATH_CALUDE_total_pencils_l1317_131770

/-- Given 11 children, with each child having 2 pencils, the total number of pencils is 22. -/
theorem total_pencils (num_children : Nat) (pencils_per_child : Nat) (total_pencils : Nat) : 
  num_children = 11 → pencils_per_child = 2 → total_pencils = num_children * pencils_per_child →
  total_pencils = 22 := by
  sorry

end NUMINAMATH_CALUDE_total_pencils_l1317_131770


namespace NUMINAMATH_CALUDE_integer_root_b_values_l1317_131703

def has_integer_root (b : ℤ) : Prop :=
  ∃ x : ℤ, x^3 + 4*x^2 + b*x + 12 = 0

def valid_b_values : Set ℤ :=
  {-193, -97, -62, -35, -25, -18, -17, -14, -3, -1, 2, 9}

theorem integer_root_b_values :
  ∀ b : ℤ, has_integer_root b ↔ b ∈ valid_b_values :=
sorry

end NUMINAMATH_CALUDE_integer_root_b_values_l1317_131703


namespace NUMINAMATH_CALUDE_train_speed_l1317_131789

/-- Calculates the speed of a train passing a bridge -/
theorem train_speed (train_length bridge_length : ℝ) (time : ℝ) :
  train_length = 360 →
  bridge_length = 140 →
  time = 50 →
  (train_length + bridge_length) / time * 3.6 = 36 := by
  sorry

end NUMINAMATH_CALUDE_train_speed_l1317_131789


namespace NUMINAMATH_CALUDE_mean_height_is_60_l1317_131712

/-- Represents the stem and leaf plot of player heights --/
def stemAndLeaf : List (Nat × List Nat) := [
  (4, [9]),
  (5, [2, 3, 5, 8, 8, 9]),
  (6, [0, 1, 1, 2, 6, 8, 9, 9])
]

/-- Calculates the total sum of heights from the stem and leaf plot --/
def sumHeights (plot : List (Nat × List Nat)) : Nat :=
  plot.foldl (fun acc (stem, leaves) => 
    acc + stem * 10 * leaves.length + leaves.sum
  ) 0

/-- Calculates the number of players from the stem and leaf plot --/
def countPlayers (plot : List (Nat × List Nat)) : Nat :=
  plot.foldl (fun acc (_, leaves) => acc + leaves.length) 0

/-- The mean height of the players --/
def meanHeight : ℚ := (sumHeights stemAndLeaf : ℚ) / (countPlayers stemAndLeaf : ℚ)

theorem mean_height_is_60 : meanHeight = 60 := by
  sorry

end NUMINAMATH_CALUDE_mean_height_is_60_l1317_131712


namespace NUMINAMATH_CALUDE_tetrahedron_subdivision_existence_l1317_131786

theorem tetrahedron_subdivision_existence : ∃ k : ℕ, (1 / 2 : ℝ) ^ k < (1 / 100 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_tetrahedron_subdivision_existence_l1317_131786


namespace NUMINAMATH_CALUDE_local_minimum_condition_l1317_131775

/-- The function f(x) defined as x^3 - 3bx + b -/
def f (b : ℝ) (x : ℝ) : ℝ := x^3 - 3*b*x + b

/-- Theorem stating the condition for f(x) to have a local minimum in (0,1) -/
theorem local_minimum_condition (b : ℝ) :
  (∃ x, x ∈ Set.Ioo 0 1 ∧ IsLocalMin (f b) x) ↔ b ∈ Set.Ioo 0 1 := by
  sorry

#check local_minimum_condition

end NUMINAMATH_CALUDE_local_minimum_condition_l1317_131775


namespace NUMINAMATH_CALUDE_square_configuration_l1317_131782

theorem square_configuration (a b : ℕ) (ha : a > 0) (hb : b > 0) :
  let x : ℝ := (a - Real.sqrt 2) / b
  2 * Real.sqrt 2 * x + x = 1 →
  a + b = 11 := by
sorry

end NUMINAMATH_CALUDE_square_configuration_l1317_131782


namespace NUMINAMATH_CALUDE_isosceles_right_triangle_exists_l1317_131759

/-- A coloring of an infinite grid using three colors -/
def GridColoring := ℤ × ℤ → Fin 3

/-- An isosceles right triangle on the grid -/
structure IsoscelesRightTriangle where
  a : ℤ × ℤ
  b : ℤ × ℤ
  c : ℤ × ℤ
  is_right : (b.1 - a.1) * (c.1 - a.1) + (b.2 - a.2) * (c.2 - a.2) = 0
  is_isosceles : (b.1 - a.1)^2 + (b.2 - a.2)^2 = (c.1 - a.1)^2 + (c.2 - a.2)^2

/-- The main theorem: In any three-coloring of an infinite grid, 
    there exists an isosceles right triangle with vertices of the same color -/
theorem isosceles_right_triangle_exists (coloring : GridColoring) : 
  ∃ t : IsoscelesRightTriangle, 
    coloring t.a = coloring t.b ∧ coloring t.b = coloring t.c :=
sorry

end NUMINAMATH_CALUDE_isosceles_right_triangle_exists_l1317_131759


namespace NUMINAMATH_CALUDE_inverse_sum_equals_negative_two_l1317_131715

def g (x : ℝ) : ℝ := x^3

theorem inverse_sum_equals_negative_two :
  (Function.invFun g) 8 + (Function.invFun g) (-64) = -2 := by
  sorry

end NUMINAMATH_CALUDE_inverse_sum_equals_negative_two_l1317_131715


namespace NUMINAMATH_CALUDE_work_completion_time_l1317_131741

/-- The number of days A takes to complete the work -/
def a_days : ℝ := 12

/-- B's efficiency compared to A -/
def b_efficiency : ℝ := 1.2

/-- The number of days B takes to complete the work -/
def b_days : ℝ := 10

theorem work_completion_time :
  a_days * b_efficiency = b_days := by sorry

end NUMINAMATH_CALUDE_work_completion_time_l1317_131741


namespace NUMINAMATH_CALUDE_sum_remainder_zero_l1317_131765

theorem sum_remainder_zero (m : ℤ) : (11 - m + (m + 5)) % 8 = 0 := by
  sorry

end NUMINAMATH_CALUDE_sum_remainder_zero_l1317_131765


namespace NUMINAMATH_CALUDE_class_average_problem_l1317_131740

theorem class_average_problem (total_students : ℕ) (excluded_students : ℕ) 
  (excluded_avg : ℝ) (remaining_avg : ℝ) :
  total_students = 20 →
  excluded_students = 2 →
  excluded_avg = 45 →
  remaining_avg = 95 →
  (total_students * (total_students - excluded_students) * remaining_avg + 
   excluded_students * excluded_avg) / total_students = 90 := by
  sorry

end NUMINAMATH_CALUDE_class_average_problem_l1317_131740


namespace NUMINAMATH_CALUDE_scout_weights_l1317_131737

/-- The weight measurement error of the scale -/
def error : ℝ := 2

/-- Míša's measured weight -/
def misa_measured : ℝ := 30

/-- Emil's measured weight -/
def emil_measured : ℝ := 28

/-- Combined measured weight of Míša and Emil -/
def combined_measured : ℝ := 56

/-- Míša's actual weight -/
def misa_actual : ℝ := misa_measured - error

/-- Emil's actual weight -/
def emil_actual : ℝ := emil_measured - error

theorem scout_weights :
  misa_actual = 28 ∧ emil_actual = 26 ∧
  misa_actual + emil_actual = combined_measured - error := by
  sorry

end NUMINAMATH_CALUDE_scout_weights_l1317_131737


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1317_131749

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a b : ℕ → ℝ) :
  is_arithmetic_sequence a →
  (∀ n : ℕ+, a n * b n = 2 * n^2 - n) →
  5 * a 4 = 7 * a 3 →
  a 1 + b 1 = 2 →
  a 9 + b 10 = 27 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1317_131749


namespace NUMINAMATH_CALUDE_function_inequality_l1317_131714

open Real

theorem function_inequality (f g : ℝ → ℝ) (hf : Differentiable ℝ f) (hg : Differentiable ℝ g)
  (hpos_f : ∀ x, f x > 0) (hpos_g : ∀ x, g x > 0)
  (h_inequality : ∀ x, (deriv^[2] f) x * g x - f x * (deriv^[2] g) x < 0)
  (a b x : ℝ) (hx : b < x ∧ x < a) :
  f x * g a > f a * g x :=
by sorry

end NUMINAMATH_CALUDE_function_inequality_l1317_131714


namespace NUMINAMATH_CALUDE_algebraic_expression_value_l1317_131700

theorem algebraic_expression_value (m n : ℝ) (h : m^2 + 3*n - 1 = 2) :
  2*m^2 + 6*n + 1 = 7 := by
  sorry

end NUMINAMATH_CALUDE_algebraic_expression_value_l1317_131700


namespace NUMINAMATH_CALUDE_faye_bought_30_songs_l1317_131720

/-- The number of songs Faye bought -/
def total_songs (country_albums pop_albums songs_per_album : ℕ) : ℕ :=
  (country_albums + pop_albums) * songs_per_album

/-- Proof that Faye bought 30 songs -/
theorem faye_bought_30_songs :
  total_songs 2 3 6 = 30 := by
  sorry

end NUMINAMATH_CALUDE_faye_bought_30_songs_l1317_131720


namespace NUMINAMATH_CALUDE_dropped_students_score_l1317_131735

theorem dropped_students_score (initial_students : ℕ) (remaining_students : ℕ) 
  (initial_average : ℚ) (remaining_average : ℚ) 
  (h1 : initial_students = 30) 
  (h2 : remaining_students = 26) 
  (h3 : initial_average = 60.25) 
  (h4 : remaining_average = 63.75) :
  (initial_students : ℚ) * initial_average - 
  (remaining_students : ℚ) * remaining_average = 150 := by
  sorry


end NUMINAMATH_CALUDE_dropped_students_score_l1317_131735


namespace NUMINAMATH_CALUDE_permutations_of_red_l1317_131752

-- Define the number of letters in 'red'
def n : ℕ := 3

-- Theorem to prove
theorem permutations_of_red : Nat.factorial n = 6 := by
  sorry

end NUMINAMATH_CALUDE_permutations_of_red_l1317_131752


namespace NUMINAMATH_CALUDE_computer_price_increase_l1317_131758

theorem computer_price_increase (y : ℝ) (h1 : 1.30 * y = 351) (h2 : 2 * y = 540) :
  2 * y = 540 := by
  sorry

end NUMINAMATH_CALUDE_computer_price_increase_l1317_131758


namespace NUMINAMATH_CALUDE_smallest_integer_in_set_l1317_131705

theorem smallest_integer_in_set (n : ℤ) : 
  (n + 6 < 3 * ((7 * n + 21) / 7)) → 
  (∀ k : ℤ, k < n → k + 6 < 3 * ((7 * k + 21) / 7)) →
  n = -1 :=
by sorry

end NUMINAMATH_CALUDE_smallest_integer_in_set_l1317_131705


namespace NUMINAMATH_CALUDE_cosine_sine_sum_l1317_131757

theorem cosine_sine_sum (α : ℝ) : 
  (Real.cos (2 * α)) / (Real.sin (α - π/4)) = -Real.sqrt 2 / 2 → 
  Real.cos α + Real.sin α = 1/2 := by
sorry

end NUMINAMATH_CALUDE_cosine_sine_sum_l1317_131757


namespace NUMINAMATH_CALUDE_comic_book_frames_l1317_131716

/-- The number of frames per page in Julian's comic book -/
def frames_per_page : ℝ := 143.0

/-- The number of pages in Julian's comic book -/
def pages : ℝ := 11.0

/-- The total number of frames in Julian's comic book -/
def total_frames : ℝ := frames_per_page * pages

theorem comic_book_frames :
  total_frames = 1573.0 := by sorry

end NUMINAMATH_CALUDE_comic_book_frames_l1317_131716


namespace NUMINAMATH_CALUDE_min_value_problem_equality_condition_l1317_131745

theorem min_value_problem (x : ℝ) (h : x > 0) : 3 * x + 4 / x ≥ 4 * Real.sqrt 3 := by
  sorry

theorem equality_condition : ∃ x : ℝ, x > 0 ∧ 3 * x + 4 / x = 4 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_min_value_problem_equality_condition_l1317_131745


namespace NUMINAMATH_CALUDE_limit_of_f_at_one_l1317_131721

noncomputable def f (x : ℝ) : ℝ := (3 * x^2 - x - 2) / (4 * x^2 - 5 * x + 1)

theorem limit_of_f_at_one :
  ∃ (L : ℝ), L = 5/3 ∧ ∀ ε > 0, ∃ δ > 0, ∀ x : ℝ, 0 < |x - 1| ∧ |x - 1| < δ → |f x - L| < ε :=
sorry

end NUMINAMATH_CALUDE_limit_of_f_at_one_l1317_131721


namespace NUMINAMATH_CALUDE_parabola_circle_intersection_l1317_131747

/-- Parabola C₁: y² = 2px where p > 0 -/
structure Parabola where
  p : ℝ
  hp : p > 0

/-- Circle C₂: (x-1)² + y² = 1 -/
def Circle (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 1

/-- Point on the parabola -/
structure PointOnParabola (C : Parabola) where
  x : ℝ
  y : ℝ
  hy : y^2 = 2 * C.p * x

/-- Only the vertex of C₁ is on C₂, all other points are outside -/
axiom vertex_on_circle_others_outside (C : Parabola) :
  Circle 0 0 ∧ ∀ (P : PointOnParabola C), P.x ≠ 0 → ¬Circle P.x P.y

/-- Fixed point M on C₁ with y₀ > 0 -/
structure FixedPoint (C : Parabola) extends PointOnParabola C where
  hy_pos : y > 0

/-- Two points A and B on C₁ -/
structure IntersectionPoints (C : Parabola) where
  A : PointOnParabola C
  B : PointOnParabola C

/-- Slopes of MA and MB exist and their angles are complementary -/
axiom complementary_slopes (C : Parabola) (M : FixedPoint C) (I : IntersectionPoints C) :
  ∃ (k : ℝ), k ≠ 0 ∧
    (I.A.y - M.y) / (I.A.x - M.x) = k ∧
    (I.B.y - M.y) / (I.B.x - M.x) = -k

/-- Main theorem -/
theorem parabola_circle_intersection (C : Parabola) (M : FixedPoint C) (I : IntersectionPoints C) :
  C.p ≥ 1 ∧
  ∃ (slope : ℝ), slope = -C.p / M.y ∧ slope ≠ 0 ∧
    (I.B.y - I.A.y) / (I.B.x - I.A.x) = slope := by sorry

end NUMINAMATH_CALUDE_parabola_circle_intersection_l1317_131747


namespace NUMINAMATH_CALUDE_smallest_union_size_l1317_131781

theorem smallest_union_size (A B : Finset ℕ) : 
  Finset.card A = 30 → 
  Finset.card B = 20 → 
  Finset.card (A ∩ B) ≥ 10 → 
  Finset.card (A ∪ B) ≥ 40 ∧ 
  ∃ (C D : Finset ℕ), Finset.card C = 30 ∧ 
                      Finset.card D = 20 ∧ 
                      Finset.card (C ∩ D) ≥ 10 ∧ 
                      Finset.card (C ∪ D) = 40 :=
by sorry

end NUMINAMATH_CALUDE_smallest_union_size_l1317_131781


namespace NUMINAMATH_CALUDE_min_value_complex_expression_l1317_131760

theorem min_value_complex_expression (w : ℂ) (h : Complex.abs (w - (3 - I)) = 3) :
  Complex.abs (w + (1 - I))^2 + Complex.abs (w - (7 - 2*I))^2 = 38 := by
  sorry

end NUMINAMATH_CALUDE_min_value_complex_expression_l1317_131760


namespace NUMINAMATH_CALUDE_expression_simplification_l1317_131790

/-- Proves that the given expression simplifies to the expected result. -/
theorem expression_simplification (x y : ℝ) :
  3 * x + 4 * x^2 + 2 - (5 - 3 * x - 5 * x^2 + 2 * y) = 9 * x^2 + 6 * x - 2 * y - 3 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l1317_131790


namespace NUMINAMATH_CALUDE_rectangular_solid_depth_l1317_131730

theorem rectangular_solid_depth
  (length width surface_area : ℝ)
  (h_length : length = 9)
  (h_width : width = 8)
  (h_surface_area : surface_area = 314)
  (h_formula : surface_area = 2 * length * width + 2 * length * depth + 2 * width * depth) :
  depth = 5 :=
by
  sorry

end NUMINAMATH_CALUDE_rectangular_solid_depth_l1317_131730


namespace NUMINAMATH_CALUDE_equal_volumes_of_modified_cylinders_l1317_131739

/-- Theorem: Equal volumes of modified cylinders -/
theorem equal_volumes_of_modified_cylinders (r h : ℝ) (y : ℝ) 
  (hr : r = 5) (hh : h = 4) (hy : y ≠ 0) :
  π * (r + 2)^2 * h = π * r^2 * (h + y) → y = 96 / 25 := by
  sorry

end NUMINAMATH_CALUDE_equal_volumes_of_modified_cylinders_l1317_131739


namespace NUMINAMATH_CALUDE_division_mistake_remainder_l1317_131795

theorem division_mistake_remainder (d q r : ℕ) (h1 : d > 0) (h2 : 472 = d * q + r) (h3 : 427 = d * (q - 5) + r) : r = 4 := by
  sorry

end NUMINAMATH_CALUDE_division_mistake_remainder_l1317_131795


namespace NUMINAMATH_CALUDE_sin_shift_l1317_131771

theorem sin_shift (x : ℝ) : 
  Real.sin (4 * x - π / 3) = Real.sin (4 * (x - π / 12)) := by
  sorry

end NUMINAMATH_CALUDE_sin_shift_l1317_131771


namespace NUMINAMATH_CALUDE_quadratic_roots_sum_l1317_131785

theorem quadratic_roots_sum (a b : ℝ) : 
  (∃ x : ℝ, x^2 + x - 2023 = 0) → 
  (a^2 + a - 2023 = 0) → 
  (b^2 + b - 2023 = 0) → 
  (a ≠ b) →
  a^2 + 2*a + b = 2022 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_sum_l1317_131785


namespace NUMINAMATH_CALUDE_no_quadratic_function_exists_l1317_131788

theorem no_quadratic_function_exists : 
  ¬ ∃ (b c : ℝ), 
    ((-4)^2 + b*(-4) + c = 1) ∧ 
    (∀ x : ℝ, 6*x ≤ 3*x^2 + 3 ∧ 3*x^2 + 3 ≤ x^2 + b*x + c) :=
by sorry

end NUMINAMATH_CALUDE_no_quadratic_function_exists_l1317_131788


namespace NUMINAMATH_CALUDE_line_points_k_value_l1317_131717

/-- A line contains the points (3, 10), (1, k), and (-7, 2). Prove that k = 8.4. -/
theorem line_points_k_value (k : ℝ) : 
  (∃ (m b : ℝ), 
    (3 * m + b = 10) ∧ 
    (1 * m + b = k) ∧ 
    (-7 * m + b = 2)) → 
  k = 8.4 := by
sorry

end NUMINAMATH_CALUDE_line_points_k_value_l1317_131717


namespace NUMINAMATH_CALUDE_remainder_113_pow_113_plus_113_mod_137_l1317_131718

theorem remainder_113_pow_113_plus_113_mod_137 
  (h1 : Prime 113) 
  (h2 : Prime 137) 
  (h3 : 113 < 137) : 
  (113^113 + 113) % 137 = 89 := by
sorry

end NUMINAMATH_CALUDE_remainder_113_pow_113_plus_113_mod_137_l1317_131718


namespace NUMINAMATH_CALUDE_smallest_divisor_k_l1317_131723

def f (z : ℂ) : ℂ := z^10 + z^9 + z^8 + z^6 + z^5 + z^4 + z + 1

theorem smallest_divisor_k : 
  (∀ z : ℂ, f z = 0 → z^84 = 1) ∧ 
  (∀ k : ℕ, k < 84 → ∃ z : ℂ, f z = 0 ∧ z^k ≠ 1) :=
sorry

end NUMINAMATH_CALUDE_smallest_divisor_k_l1317_131723


namespace NUMINAMATH_CALUDE_distance_after_two_hours_l1317_131797

/-- Alice's walking speed in miles per minute -/
def alice_speed : ℚ := 1 / 20

/-- Bob's jogging speed in miles per minute -/
def bob_speed : ℚ := 3 / 40

/-- Duration of travel in minutes -/
def duration : ℕ := 120

/-- The distance between Alice and Bob after the given duration -/
def distance_between (alice_speed bob_speed : ℚ) (duration : ℕ) : ℚ :=
  (alice_speed * duration) + (bob_speed * duration)

theorem distance_after_two_hours :
  distance_between alice_speed bob_speed duration = 15 := by
  sorry

end NUMINAMATH_CALUDE_distance_after_two_hours_l1317_131797


namespace NUMINAMATH_CALUDE_negation_of_universal_statement_l1317_131767

theorem negation_of_universal_statement :
  (¬ ∀ x : ℝ, x^3 - 3*x > 0) ↔ (∃ x : ℝ, x^3 - 3*x ≤ 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_universal_statement_l1317_131767


namespace NUMINAMATH_CALUDE_least_number_to_add_l1317_131728

theorem least_number_to_add (n : ℕ) : 
  (∀ m : ℕ, m < 7 → ¬((1789 + m) % 6 = 0 ∧ (1789 + m) % 4 = 0 ∧ (1789 + m) % 3 = 0)) ∧
  ((1789 + 7) % 6 = 0 ∧ (1789 + 7) % 4 = 0 ∧ (1789 + 7) % 3 = 0) :=
by sorry

end NUMINAMATH_CALUDE_least_number_to_add_l1317_131728


namespace NUMINAMATH_CALUDE_concentric_circles_ratio_l1317_131707

theorem concentric_circles_ratio (a b : ℝ) (h : a > 0) (k : b > 0) :
  (π * b^2 - π * a^2 = 4 * π * a^2) → (a / b = 1 / Real.sqrt 5) := by
  sorry

end NUMINAMATH_CALUDE_concentric_circles_ratio_l1317_131707


namespace NUMINAMATH_CALUDE_probability_reach_bottom_is_one_fifth_l1317_131763

/-- Represents a dodecahedron -/
structure Dodecahedron where
  top_vertex : Vertex
  bottom_vertex : Vertex
  middle_vertices : Finset Vertex
  adjacent : Vertex → Finset Vertex

/-- The probability of an ant reaching the bottom vertex in two steps -/
def probability_reach_bottom (d : Dodecahedron) : ℚ :=
  1 / 5

/-- Theorem stating the probability of reaching the bottom vertex in two steps -/
theorem probability_reach_bottom_is_one_fifth (d : Dodecahedron) :
  probability_reach_bottom d = 1 / 5 :=
by sorry

end NUMINAMATH_CALUDE_probability_reach_bottom_is_one_fifth_l1317_131763


namespace NUMINAMATH_CALUDE_shortest_time_5x6_checkerboard_l1317_131724

/-- Represents a checkerboard with alternating black and white squares. -/
structure Checkerboard where
  rows : Nat
  cols : Nat
  squareSize : ℝ
  normalSpeed : ℝ
  slowSpeed : ℝ

/-- Calculates the shortest time to travel from bottom-left to top-right corner of the checkerboard. -/
def shortestTravelTime (board : Checkerboard) : ℝ :=
  sorry

/-- The theorem stating the shortest travel time for the specific checkerboard. -/
theorem shortest_time_5x6_checkerboard :
  let board : Checkerboard := {
    rows := 5
    cols := 6
    squareSize := 1
    normalSpeed := 2
    slowSpeed := 1
  }
  shortestTravelTime board = (1 + 5 * Real.sqrt 2) / 2 := by
  sorry

end NUMINAMATH_CALUDE_shortest_time_5x6_checkerboard_l1317_131724


namespace NUMINAMATH_CALUDE_sin_sixty_degrees_l1317_131722

theorem sin_sixty_degrees : Real.sin (π / 3) = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_sixty_degrees_l1317_131722


namespace NUMINAMATH_CALUDE_impossible_to_turn_all_off_l1317_131736

/-- Represents the state of a lightning bug (on or off) -/
inductive BugState
| On
| Off

/-- Represents a 6x6 grid of lightning bugs -/
def Grid := Fin 6 → Fin 6 → BugState

/-- Represents a move on the grid -/
inductive Move
| Horizontal (row : Fin 6) (start_col : Fin 6)
| Vertical (col : Fin 6) (start_row : Fin 6)

/-- Applies a move to a grid -/
def applyMove (grid : Grid) (move : Move) : Grid :=
  sorry

/-- Checks if all bugs in the grid are off -/
def allOff (grid : Grid) : Prop :=
  ∀ (row col : Fin 6), grid row col = BugState.Off

/-- Initial grid configuration with one bug on -/
def initialGrid : Grid :=
  sorry

/-- Theorem stating the impossibility of turning all bugs off -/
theorem impossible_to_turn_all_off :
  ¬∃ (moves : List Move), allOff (moves.foldl applyMove initialGrid) :=
  sorry

end NUMINAMATH_CALUDE_impossible_to_turn_all_off_l1317_131736


namespace NUMINAMATH_CALUDE_tadd_500th_number_l1317_131791

def tadd_sequence (n : ℕ) : ℕ := (3 * n - 2) ^ 2

theorem tadd_500th_number : tadd_sequence 500 = 2244004 := by
  sorry

end NUMINAMATH_CALUDE_tadd_500th_number_l1317_131791


namespace NUMINAMATH_CALUDE_prob_of_specific_sums_is_five_eighteenths_l1317_131743

/-- Represents the faces of a die -/
def Die := List Nat

/-- The first die with faces 1, 3, 3, 5, 5, 7 -/
def die1 : Die := [1, 3, 3, 5, 5, 7]

/-- The second die with faces 2, 4, 4, 6, 6, 8 -/
def die2 : Die := [2, 4, 4, 6, 6, 8]

/-- Calculates the probability of a specific sum occurring when rolling two dice -/
def probOfSum (d1 d2 : Die) (sum : Nat) : Rat :=
  sorry

/-- Calculates the probability of the sum being 8, 10, or 12 when rolling the two specified dice -/
def probOfSpecificSums (d1 d2 : Die) : Rat :=
  (probOfSum d1 d2 8) + (probOfSum d1 d2 10) + (probOfSum d1 d2 12)

/-- Theorem stating that the probability of getting a sum of 8, 10, or 12 with the given dice is 5/18 -/
theorem prob_of_specific_sums_is_five_eighteenths :
  probOfSpecificSums die1 die2 = 5 / 18 := by
  sorry

end NUMINAMATH_CALUDE_prob_of_specific_sums_is_five_eighteenths_l1317_131743


namespace NUMINAMATH_CALUDE_percent_decrease_proof_l1317_131751

theorem percent_decrease_proof (original_price sale_price : ℝ) 
  (h1 : original_price = 100)
  (h2 : sale_price = 70) :
  (original_price - sale_price) / original_price * 100 = 30 := by
  sorry

end NUMINAMATH_CALUDE_percent_decrease_proof_l1317_131751


namespace NUMINAMATH_CALUDE_john_booking_l1317_131792

/-- Calculates the number of nights booked given the nightly rate, discount, and total paid -/
def nights_booked (nightly_rate : ℕ) (discount : ℕ) (total_paid : ℕ) : ℕ :=
  (total_paid + discount) / nightly_rate

theorem john_booking :
  nights_booked 250 100 650 = 3 := by
  sorry

end NUMINAMATH_CALUDE_john_booking_l1317_131792


namespace NUMINAMATH_CALUDE_parabola_focus_directrix_distance_l1317_131794

/-- Given a parabola defined by y = 4x^2, the distance from its focus to its directrix is 1/8. -/
theorem parabola_focus_directrix_distance (x y : ℝ) : 
  y = 4 * x^2 → (distance_focus_to_directrix : ℝ) = 1/8 :=
by sorry

end NUMINAMATH_CALUDE_parabola_focus_directrix_distance_l1317_131794


namespace NUMINAMATH_CALUDE_triangle_similarity_l1317_131793

-- Define the points as complex numbers
variable (z₁ z₂ z₃ t₁ t₂ t₃ z₁' z₂' z₃' : ℂ)

-- Define the similarity relation
def similar (a b c d e f : ℂ) : Prop :=
  (e - d) / (f - d) = (b - a) / (c - a)

-- State the theorem
theorem triangle_similarity :
  similar z₁ z₂ z₃ t₁ t₂ t₃ →  -- DBC similar to ABC
  similar z₂ z₃ z₁ t₂ t₃ t₁ →  -- ECA similar to ABC
  similar z₃ z₁ z₂ t₃ t₁ t₂ →  -- FAB similar to ABC
  similar t₂ t₃ t₁ z₁' t₃ t₂ →  -- A'FE similar to DBC
  similar t₃ t₁ t₂ z₂' t₁ t₃ →  -- B'DF similar to ECA
  similar t₁ t₂ t₃ z₃' t₂ t₁ →  -- C'ED similar to FAB
  similar z₁ z₂ z₃ z₁' z₂' z₃'  -- A'B'C' similar to ABC
:= by sorry

end NUMINAMATH_CALUDE_triangle_similarity_l1317_131793


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l1317_131742

-- Define set A
def A : Set ℝ := {x | x^2 - 2*x - 3 < 0}

-- Define set B
def B : Set ℝ := {x | ∃ y, y = Real.log (2 - x)}

-- Theorem statement
theorem intersection_of_A_and_B : A ∩ B = {x | -1 < x ∧ x < 2} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l1317_131742


namespace NUMINAMATH_CALUDE_smallest_root_of_equation_l1317_131776

theorem smallest_root_of_equation : 
  let eq := fun x : ℝ => 2 * (x - 3 * Real.sqrt 5) * (x - 5 * Real.sqrt 3)
  ∃ (r : ℝ), eq r = 0 ∧ r = 3 * Real.sqrt 5 ∧ ∀ (s : ℝ), eq s = 0 → r ≤ s :=
by sorry

end NUMINAMATH_CALUDE_smallest_root_of_equation_l1317_131776


namespace NUMINAMATH_CALUDE_inequality_proof_l1317_131796

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a + 1/b - c) * (b + 1/c - a) + (b + 1/c - a) * (c + 1/a - b) + (c + 1/a - b) * (a + 1/b - c) ≥ 3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1317_131796


namespace NUMINAMATH_CALUDE_dr_jones_remaining_money_l1317_131719

theorem dr_jones_remaining_money :
  let monthly_earnings : ℕ := 6000
  let house_rental : ℕ := 640
  let food_expense : ℕ := 380
  let electric_water_bill : ℕ := monthly_earnings / 4
  let insurance_cost : ℕ := monthly_earnings / 5
  let total_expenses : ℕ := house_rental + food_expense + electric_water_bill + insurance_cost
  let remaining_money : ℕ := monthly_earnings - total_expenses
  remaining_money = 2280 := by
  sorry

end NUMINAMATH_CALUDE_dr_jones_remaining_money_l1317_131719


namespace NUMINAMATH_CALUDE_exists_same_color_four_directions_l1317_131709

/-- A color in the grid -/
inductive Color
| Red
| Yellow
| Green
| Blue

/-- A position in the grid -/
structure Position where
  x : Fin 50
  y : Fin 50

/-- A coloring of the grid -/
def Coloring := Position → Color

/-- A position has a same-colored square above it -/
def has_same_color_above (c : Coloring) (p : Position) : Prop :=
  ∃ q : Position, q.x = p.x ∧ q.y > p.y ∧ c q = c p

/-- A position has a same-colored square below it -/
def has_same_color_below (c : Coloring) (p : Position) : Prop :=
  ∃ q : Position, q.x = p.x ∧ q.y < p.y ∧ c q = c p

/-- A position has a same-colored square to its left -/
def has_same_color_left (c : Coloring) (p : Position) : Prop :=
  ∃ q : Position, q.y = p.y ∧ q.x < p.x ∧ c q = c p

/-- A position has a same-colored square to its right -/
def has_same_color_right (c : Coloring) (p : Position) : Prop :=
  ∃ q : Position, q.y = p.y ∧ q.x > p.x ∧ c q = c p

/-- Main theorem: There exists a position with same-colored squares in all four directions -/
theorem exists_same_color_four_directions (c : Coloring) : 
  ∃ p : Position, 
    has_same_color_above c p ∧ 
    has_same_color_below c p ∧ 
    has_same_color_left c p ∧ 
    has_same_color_right c p := by
  sorry

end NUMINAMATH_CALUDE_exists_same_color_four_directions_l1317_131709


namespace NUMINAMATH_CALUDE_orchestra_members_count_l1317_131773

theorem orchestra_members_count : ∃! n : ℕ, 
  150 < n ∧ n < 300 ∧ 
  n % 6 = 2 ∧ 
  n % 8 = 4 ∧ 
  n % 9 = 6 ∧
  n = 212 := by sorry

end NUMINAMATH_CALUDE_orchestra_members_count_l1317_131773


namespace NUMINAMATH_CALUDE_right_triangle_inradius_l1317_131702

/-- The inradius of a right triangle with sides 9, 40, and 41 is 4 -/
theorem right_triangle_inradius : ∀ (a b c r : ℝ),
  a = 9 ∧ b = 40 ∧ c = 41 →
  a^2 + b^2 = c^2 →
  r = (a * b) / (2 * (a + b + c)) →
  r = 4 := by sorry

end NUMINAMATH_CALUDE_right_triangle_inradius_l1317_131702


namespace NUMINAMATH_CALUDE_line_slope_slope_value_l1317_131778

theorem line_slope (x y : ℝ) :
  x + Real.sqrt 3 * y + 1 = 0 → (y = -(Real.sqrt 3 / 3) * x - (1 / Real.sqrt 3)) := by
  sorry

theorem slope_value :
  let m := -(Real.sqrt 3 / 3)
  ∀ x y : ℝ, x + Real.sqrt 3 * y + 1 = 0 → y = m * x - (1 / Real.sqrt 3) := by
  sorry

end NUMINAMATH_CALUDE_line_slope_slope_value_l1317_131778


namespace NUMINAMATH_CALUDE_function_value_problem_l1317_131779

theorem function_value_problem (f : ℝ → ℝ) (a : ℝ) :
  (∀ x, f (2 * x - 1) = 3 * x + a) →
  f 3 = 2 →
  a = -4 :=
by sorry

end NUMINAMATH_CALUDE_function_value_problem_l1317_131779


namespace NUMINAMATH_CALUDE_fruit_box_arrangement_l1317_131756

-- Define the fruits
inductive Fruit
  | Apple
  | Pear
  | Orange
  | Banana

-- Define a type for box numbers
inductive BoxNumber
  | One
  | Two
  | Three
  | Four

-- Define a function type for box labels
def BoxLabel := BoxNumber → Fruit

-- Define a function type for the actual content of boxes
def BoxContent := BoxNumber → Fruit

-- Define the property that all labels are incorrect
def AllLabelsIncorrect (label : BoxLabel) (content : BoxContent) : Prop :=
  ∀ b : BoxNumber, label b ≠ content b

-- Define the specific labels for each box
def SpecificLabels (label : BoxLabel) : Prop :=
  label BoxNumber.One = Fruit.Orange ∧
  label BoxNumber.Two = Fruit.Pear ∧
  (label BoxNumber.Three = Fruit.Apple ∨ label BoxNumber.Three = Fruit.Pear) ∧
  label BoxNumber.Four = Fruit.Apple

-- Define the conditional statement for Box 3
def Box3Condition (content : BoxContent) : Prop :=
  content BoxNumber.One = Fruit.Banana →
  (content BoxNumber.Three = Fruit.Apple ∨ content BoxNumber.Three = Fruit.Pear)

-- The main theorem
theorem fruit_box_arrangement :
  ∀ (label : BoxLabel) (content : BoxContent),
    AllLabelsIncorrect label content →
    SpecificLabels label →
    ¬Box3Condition content →
    content BoxNumber.One = Fruit.Banana ∧
    content BoxNumber.Two = Fruit.Apple ∧
    content BoxNumber.Three = Fruit.Orange ∧
    content BoxNumber.Four = Fruit.Pear :=
by
  sorry

end NUMINAMATH_CALUDE_fruit_box_arrangement_l1317_131756


namespace NUMINAMATH_CALUDE_min_value_x_l1317_131701

theorem min_value_x (a b x : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : x > 0)
  (h4 : ∀ a b, a > 0 → b > 0 → (1 / a^2 + 16 / b^2 ≥ 1 + x / 2 - x^2))
  (h5 : ∀ a b, a > 0 → b > 0 → (4*a + b*(1 - a) = 0)) :
  x ≥ 1 :=
sorry

end NUMINAMATH_CALUDE_min_value_x_l1317_131701


namespace NUMINAMATH_CALUDE_sqrt_12_times_sqrt_75_l1317_131753

theorem sqrt_12_times_sqrt_75 : Real.sqrt 12 * Real.sqrt 75 = 30 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_12_times_sqrt_75_l1317_131753


namespace NUMINAMATH_CALUDE_acme_profit_l1317_131772

/-- Calculates the profit for a horseshoe manufacturing company. -/
def calculate_profit (initial_outlay : ℝ) (cost_per_set : ℝ) (selling_price : ℝ) (num_sets : ℕ) : ℝ :=
  let revenue := selling_price * num_sets
  let total_cost := initial_outlay + cost_per_set * num_sets
  revenue - total_cost

/-- Proves that the profit for Acme's horseshoe manufacturing is $15,337.50 -/
theorem acme_profit :
  calculate_profit 12450 20.75 50 950 = 15337.50 := by
  sorry

end NUMINAMATH_CALUDE_acme_profit_l1317_131772


namespace NUMINAMATH_CALUDE_modular_inverse_of_two_mod_127_l1317_131708

theorem modular_inverse_of_two_mod_127 : ∃ x : ℕ, x < 127 ∧ (2 * x) % 127 = 1 :=
  by
    use 64
    sorry

end NUMINAMATH_CALUDE_modular_inverse_of_two_mod_127_l1317_131708


namespace NUMINAMATH_CALUDE_new_numbers_mean_l1317_131766

/-- Given 7 numbers with mean 36 and 3 new numbers making a total of 10 with mean 48,
    prove that the mean of the 3 new numbers is 76. -/
theorem new_numbers_mean (original_count : Nat) (new_count : Nat) 
  (original_mean : ℝ) (new_mean : ℝ) : 
  original_count = 7 →
  new_count = 3 →
  original_mean = 36 →
  new_mean = 48 →
  (original_count * original_mean + new_count * 
    ((original_count + new_count) * new_mean - original_count * original_mean) / new_count) / 
    new_count = 76 := by
  sorry

end NUMINAMATH_CALUDE_new_numbers_mean_l1317_131766


namespace NUMINAMATH_CALUDE_ball_distribution_l1317_131774

/-- The number of ways to distribute n indistinguishable balls into k boxes -/
def distribute (n k : ℕ) : ℕ := Nat.choose (n + k - 1) n

/-- The number of ways to distribute balls into three boxes with minimum requirements -/
def distributeWithMinimum (total : ℕ) (min1 min2 min3 : ℕ) : ℕ :=
  distribute (total - min1 - min2 - min3) 3

theorem ball_distribution :
  distributeWithMinimum 20 1 2 3 = 120 := by sorry

end NUMINAMATH_CALUDE_ball_distribution_l1317_131774


namespace NUMINAMATH_CALUDE_two_std_dev_below_mean_l1317_131731

/-- For a normal distribution with mean μ and standard deviation σ,
    the value 2σ below the mean is μ - 2σ. -/
theorem two_std_dev_below_mean (μ σ : ℝ) (hμ : μ = 15.5) (hσ : σ = 1.5) :
  μ - 2 * σ = 12.5 := by
  sorry

#check two_std_dev_below_mean

end NUMINAMATH_CALUDE_two_std_dev_below_mean_l1317_131731


namespace NUMINAMATH_CALUDE_remainder_problem_l1317_131761

theorem remainder_problem (f y : ℤ) : 
  y % 5 = 4 → (f + y) % 5 = 2 → f % 5 = 3 :=
by sorry

end NUMINAMATH_CALUDE_remainder_problem_l1317_131761


namespace NUMINAMATH_CALUDE_fencing_cost_l1317_131799

/-- The total cost of fencing a rectangular field with a square pond -/
theorem fencing_cost (field_area : ℝ) (outer_fence_cost : ℝ) (pond_fence_cost : ℝ) : 
  field_area = 10800 ∧ 
  outer_fence_cost = 1.5 ∧ 
  pond_fence_cost = 1 → 
  ∃ (short_side long_side pond_side : ℝ),
    short_side * long_side = field_area ∧
    long_side = (4/3) * short_side ∧
    pond_side = (1/6) * short_side ∧
    2 * (short_side + long_side) * outer_fence_cost + 4 * pond_side * pond_fence_cost = 690 :=
by sorry

end NUMINAMATH_CALUDE_fencing_cost_l1317_131799


namespace NUMINAMATH_CALUDE_radio_sale_profit_percentage_l1317_131726

/-- Represents the problem of calculating profit percentage for a radio sale --/
theorem radio_sale_profit_percentage 
  (original_cost_usd : ℝ) 
  (exchange_rate : ℝ) 
  (discount_rate : ℝ) 
  (tax_rate : ℝ) 
  (final_price : ℝ) 
  (h1 : original_cost_usd = 110)
  (h2 : exchange_rate = 30)
  (h3 : discount_rate = 0.15)
  (h4 : tax_rate = 0.12)
  (h5 : final_price = 4830) :
  let original_cost_inr : ℝ := original_cost_usd * exchange_rate
  let selling_price_before_tax : ℝ := final_price / (1 + tax_rate)
  let profit : ℝ := selling_price_before_tax - original_cost_inr
  let profit_percentage : ℝ := (profit / original_cost_inr) * 100
  ∃ (ε : ℝ), abs (profit_percentage - 30.68) < ε ∧ ε > 0 := by
  sorry


end NUMINAMATH_CALUDE_radio_sale_profit_percentage_l1317_131726


namespace NUMINAMATH_CALUDE_triangle_side_values_l1317_131734

theorem triangle_side_values (a b c : ℝ) (A B C : ℝ) :
  -- Define triangle ABC
  (0 < a ∧ 0 < b ∧ 0 < c) →
  (0 < A ∧ 0 < B ∧ 0 < C) →
  (A + B + C = π) →
  -- Area condition
  (1/2 * b * c * Real.sin A = Real.sqrt 3 / 2) →
  -- Given conditions
  (c = 2) →
  (A = π/3) →
  -- Conclusion
  (a = Real.sqrt 3 ∧ b = 1) := by
sorry

end NUMINAMATH_CALUDE_triangle_side_values_l1317_131734


namespace NUMINAMATH_CALUDE_conference_handshakes_l1317_131704

/-- The number of handshakes in a conference with multiple companies --/
def number_of_handshakes (num_companies : ℕ) (reps_per_company : ℕ) : ℕ :=
  let total_participants := num_companies * reps_per_company
  let handshakes_per_person := total_participants - reps_per_company
  (total_participants * handshakes_per_person) / 2

/-- Theorem: In a conference with 3 companies, each having 5 representatives,
    where every person shakes hands once with every person except those from their own company,
    the total number of handshakes is 75. --/
theorem conference_handshakes :
  number_of_handshakes 3 5 = 75 := by
  sorry

end NUMINAMATH_CALUDE_conference_handshakes_l1317_131704


namespace NUMINAMATH_CALUDE_box_volume_increase_l1317_131754

theorem box_volume_increase (l w h : ℝ) 
  (volume : l * w * h = 5000)
  (surface_area : 2 * (l * w + l * h + w * h) = 1950)
  (edge_sum : 4 * (l + w + h) = 240) :
  (l + 2) * (w + 2) * (h + 2) = 7198 := by
  sorry

end NUMINAMATH_CALUDE_box_volume_increase_l1317_131754


namespace NUMINAMATH_CALUDE_loan_future_value_l1317_131750

/-- Represents the relationship between principal and future value for a loan -/
theorem loan_future_value 
  (P A : ℝ) -- Principal and future value
  (r : ℝ) -- Annual interest rate
  (n : ℕ) -- Number of times interest is compounded per year
  (t : ℕ) -- Number of years
  (h1 : r = 0.12) -- Interest rate is 12%
  (h2 : n = 2) -- Compounded half-yearly
  (h3 : t = 20) -- Loan period is 20 years
  : A = P * (1 + r/n)^(n*t) :=
by sorry

end NUMINAMATH_CALUDE_loan_future_value_l1317_131750


namespace NUMINAMATH_CALUDE_tangent_slopes_reciprocal_implies_a_between_one_and_two_l1317_131768

open Real

theorem tangent_slopes_reciprocal_implies_a_between_one_and_two 
  (f : ℝ → ℝ) (a : ℝ) (l₁ l₂ : ℝ → ℝ) :
  a ≠ 0 →
  (∀ x, f x = log x - a * (x - 1)) →
  (∃ x₁ y₁, l₁ 0 = 0 ∧ l₁ x₁ = y₁ ∧ y₁ = f x₁) →
  (∃ x₂ y₂, l₂ 0 = 0 ∧ l₂ x₂ = y₂ ∧ y₂ = exp x₂) →
  (∃ k₁ k₂, (∀ x, l₁ x = k₁ * x) ∧ (∀ x, l₂ x = k₂ * x) ∧ k₁ * k₂ = 1) →
  1 < a ∧ a < 2 := by
sorry

end NUMINAMATH_CALUDE_tangent_slopes_reciprocal_implies_a_between_one_and_two_l1317_131768


namespace NUMINAMATH_CALUDE_complex_equation_solution_l1317_131787

-- Define the complex number i
def i : ℂ := Complex.I

-- State the theorem
theorem complex_equation_solution :
  ∃ x : ℂ, (3 - 2 * i * x = 6 + i * x) ∧ (x = i) := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l1317_131787
