import Mathlib

namespace NUMINAMATH_CALUDE_rand_code_is_1236_l1537_153713

/-- Represents a coding system for words -/
structure CodeSystem where
  range_code : Nat
  random_code : Nat

/-- Extracts the code for a given letter based on its position in a word -/
def extract_code (n : Nat) (code : Nat) : Nat :=
  (code / (10 ^ (5 - n))) % 10

/-- Determines the code for "rand" based on the given coding system -/
def rand_code (cs : CodeSystem) : Nat :=
  let r := extract_code 1 cs.range_code
  let a := extract_code 2 cs.range_code
  let n := extract_code 3 cs.range_code
  let d := extract_code 4 cs.random_code
  r * 1000 + a * 100 + n * 10 + d

/-- Theorem stating that the code for "rand" is 1236 given the specified coding system -/
theorem rand_code_is_1236 (cs : CodeSystem) 
    (h1 : cs.range_code = 12345) 
    (h2 : cs.random_code = 123678) : 
  rand_code cs = 1236 := by
  sorry

end NUMINAMATH_CALUDE_rand_code_is_1236_l1537_153713


namespace NUMINAMATH_CALUDE_total_lobster_amount_l1537_153702

/-- The amount of lobster in pounds for each harbor -/
structure HarborLobster where
  hooperBay : ℝ
  harborA : ℝ
  harborB : ℝ
  harborC : ℝ
  harborD : ℝ

/-- The conditions for the lobster distribution problem -/
def LobsterDistribution (h : HarborLobster) : Prop :=
  h.harborA = 50 ∧
  h.harborB = 70.5 ∧
  h.harborC = (2/3) * h.harborB ∧
  h.harborD = h.harborA - 0.15 * h.harborA ∧
  h.hooperBay = 3 * (h.harborA + h.harborB + h.harborC + h.harborD)

/-- The theorem stating that the total amount of lobster is 840 pounds -/
theorem total_lobster_amount (h : HarborLobster) 
  (hDist : LobsterDistribution h) : 
  h.hooperBay + h.harborA + h.harborB + h.harborC + h.harborD = 840 := by
  sorry

end NUMINAMATH_CALUDE_total_lobster_amount_l1537_153702


namespace NUMINAMATH_CALUDE_equation_solution_l1537_153730

theorem equation_solution (x y : ℝ) :
  x * y^3 - y^2 = y * x^3 - x^2 → y = -x ∨ y = x ∨ y = 1 / x :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l1537_153730


namespace NUMINAMATH_CALUDE_point_p_position_l1537_153727

/-- Given seven points O, A, B, C, D, E, F on a line, with specified distances from O,
    and a point P between D and E satisfying a ratio condition,
    prove that OP has a specific value. -/
theorem point_p_position
  (a b c d e f : ℝ)  -- Real parameters for distances
  (O A B C D E F P : ℝ)  -- Points on the real line
  (h1 : O = 0)  -- O is the origin
  (h2 : A = 2*a)
  (h3 : B = 5*b)
  (h4 : C = 9*c)
  (h5 : D = 12*d)
  (h6 : E = 15*e)
  (h7 : F = 20*f)
  (h8 : D ≤ P ∧ P ≤ E)  -- P is between D and E
  (h9 : (P - A) / (F - P) = (P - D) / (E - P))  -- Ratio condition
  : P = (300*a*e - 240*d*f) / (2*a - 15*e + 20*f) :=
sorry

end NUMINAMATH_CALUDE_point_p_position_l1537_153727


namespace NUMINAMATH_CALUDE_period_start_time_l1537_153755

def period_end : Nat := 17  -- 5 pm in 24-hour format
def rain_duration : Nat := 2
def no_rain_duration : Nat := 6

theorem period_start_time : 
  period_end - (rain_duration + no_rain_duration) = 9 := by
  sorry

end NUMINAMATH_CALUDE_period_start_time_l1537_153755


namespace NUMINAMATH_CALUDE_pots_needed_for_path_l1537_153781

/-- Calculate the number of pots needed for a path with given specifications. -/
def calculate_pots (path_length : ℕ) (pot_distance : ℕ) : ℕ :=
  let pots_per_side := path_length / pot_distance + 1
  2 * pots_per_side

/-- Theorem stating that 152 pots are needed for the given path specifications. -/
theorem pots_needed_for_path : calculate_pots 150 2 = 152 := by
  sorry

#eval calculate_pots 150 2

end NUMINAMATH_CALUDE_pots_needed_for_path_l1537_153781


namespace NUMINAMATH_CALUDE_cube_sum_of_roots_l1537_153799

theorem cube_sum_of_roots (p q r : ℂ) : 
  (p^3 - 2*p^2 + 3*p - 4 = 0) →
  (q^3 - 2*q^2 + 3*q - 4 = 0) →
  (r^3 - 2*r^2 + 3*r - 4 = 0) →
  p^3 + q^3 + r^3 = 2 := by
sorry

end NUMINAMATH_CALUDE_cube_sum_of_roots_l1537_153799


namespace NUMINAMATH_CALUDE_sum_of_three_consecutive_even_numbers_l1537_153774

theorem sum_of_three_consecutive_even_numbers (n : ℕ) (h : n = 52) :
  n + (n + 2) + (n + 4) = 162 := by
sorry

end NUMINAMATH_CALUDE_sum_of_three_consecutive_even_numbers_l1537_153774


namespace NUMINAMATH_CALUDE_students_not_eating_lunch_l1537_153751

theorem students_not_eating_lunch (total : ℕ) (cafeteria : ℕ) (bring_lunch_multiplier : ℕ) :
  total = 90 →
  bring_lunch_multiplier = 4 →
  cafeteria = 12 →
  total - (cafeteria + bring_lunch_multiplier * cafeteria) = 30 :=
by sorry

end NUMINAMATH_CALUDE_students_not_eating_lunch_l1537_153751


namespace NUMINAMATH_CALUDE_sector_arc_length_l1537_153716

theorem sector_arc_length (s r p : ℝ) : 
  s = 4 → r = 2 → s = (1/2) * r * p → p = 4 := by sorry

end NUMINAMATH_CALUDE_sector_arc_length_l1537_153716


namespace NUMINAMATH_CALUDE_complex_cube_root_l1537_153743

theorem complex_cube_root : ∃ (z : ℂ), z^2 + 2 = 0 → z^3 = 2 * Real.sqrt 2 * I ∨ z^3 = -2 * Real.sqrt 2 * I := by
  sorry

end NUMINAMATH_CALUDE_complex_cube_root_l1537_153743


namespace NUMINAMATH_CALUDE_plane_determining_pairs_count_l1537_153762

/-- A regular tetrahedron -/
structure RegularTetrahedron where
  /-- The number of edges in a regular tetrahedron -/
  num_edges : ℕ
  /-- The number of edges that intersect with each edge -/
  intersecting_edges : ℕ
  /-- Property that the number of edges is 6 -/
  edge_count : num_edges = 6
  /-- Property that each edge intersects with 2 other edges -/
  intersect_count : intersecting_edges = 2
  /-- Property that there are no skew edges -/
  no_skew_edges : True

/-- The number of unordered pairs of edges that determine a plane in a regular tetrahedron -/
def plane_determining_pairs (t : RegularTetrahedron) : ℕ :=
  t.num_edges * t.intersecting_edges / 2

/-- Theorem stating that the number of unordered pairs of edges that determine a plane in a regular tetrahedron is 6 -/
theorem plane_determining_pairs_count (t : RegularTetrahedron) :
  plane_determining_pairs t = 6 := by
  sorry

end NUMINAMATH_CALUDE_plane_determining_pairs_count_l1537_153762


namespace NUMINAMATH_CALUDE_largest_valid_partition_l1537_153750

/-- Represents a partition of the set {1, 2, ..., m} into n subsets -/
def Partition (m : ℕ) (n : ℕ) := Fin n → Finset (Fin m)

/-- Checks if a partition satisfies the condition that the product of two different
    elements in the same subset is never a perfect square -/
def ValidPartition (p : Partition m n) : Prop :=
  ∀ i : Fin n, ∀ x y : Fin m, x ∈ p i → y ∈ p i → x ≠ y →
    ¬ ∃ z : ℕ, (x.val + 1) * (y.val + 1) = z * z

/-- The main theorem stating that n^2 + 2n is the largest m for which
    a valid partition exists -/
theorem largest_valid_partition (n : ℕ) (h : 0 < n) :
  (∃ p : Partition (n^2 + 2*n) n, ValidPartition p) ∧
  (∀ m : ℕ, m > n^2 + 2*n → ¬ ∃ p : Partition m n, ValidPartition p) :=
sorry

end NUMINAMATH_CALUDE_largest_valid_partition_l1537_153750


namespace NUMINAMATH_CALUDE_calculation_proof_l1537_153794

theorem calculation_proof :
  (1 / 6 + 2 / 3) * (-24) = -20 ∧
  (-3)^2 * (2 - (-6)) + 30 / (-5) = 66 := by
sorry

end NUMINAMATH_CALUDE_calculation_proof_l1537_153794


namespace NUMINAMATH_CALUDE_prob_sum_less_than_7_is_5_12_l1537_153784

/-- The number of possible outcomes when throwing two dice -/
def total_outcomes : ℕ := 36

/-- The number of favorable outcomes (sum less than 7) -/
def favorable_outcomes : ℕ := 15

/-- The probability of getting a sum less than 7 when throwing two dice -/
def prob_sum_less_than_7 : ℚ := favorable_outcomes / total_outcomes

theorem prob_sum_less_than_7_is_5_12 : prob_sum_less_than_7 = 5 / 12 := by
  sorry

end NUMINAMATH_CALUDE_prob_sum_less_than_7_is_5_12_l1537_153784


namespace NUMINAMATH_CALUDE_sum_of_coordinates_D_l1537_153711

/-- Given that M(5,3) is the midpoint of segment CD and C(2,6), prove that the sum of D's coordinates is 8 -/
theorem sum_of_coordinates_D (C D M : ℝ × ℝ) : 
  C = (2, 6) →
  M = (5, 3) →
  M.1 = (C.1 + D.1) / 2 →
  M.2 = (C.2 + D.2) / 2 →
  D.1 + D.2 = 8 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_coordinates_D_l1537_153711


namespace NUMINAMATH_CALUDE_f_2018_equals_1_l1537_153714

def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

theorem f_2018_equals_1
  (f : ℝ → ℝ)
  (h_even : is_even_function f)
  (h_f0 : f 0 = -1)
  (h_fx : ∀ x, f x = -f (2 - x)) :
  f 2018 = 1 := by
  sorry

end NUMINAMATH_CALUDE_f_2018_equals_1_l1537_153714


namespace NUMINAMATH_CALUDE_intersection_volume_is_half_l1537_153763

/-- A regular tetrahedron -/
structure RegularTetrahedron where
  volume : ℝ

/-- The intersection of two regular tetrahedra -/
def tetrahedra_intersection (t1 t2 : RegularTetrahedron) : ℝ := sorry

/-- Reflection of a regular tetrahedron through its center -/
def reflect_tetrahedron (t : RegularTetrahedron) : RegularTetrahedron := sorry

theorem intersection_volume_is_half (t : RegularTetrahedron) 
  (h : t.volume = 1) : 
  tetrahedra_intersection t (reflect_tetrahedron t) = 1/2 := by sorry

end NUMINAMATH_CALUDE_intersection_volume_is_half_l1537_153763


namespace NUMINAMATH_CALUDE_fixed_charge_is_six_l1537_153722

/-- Represents Elvin's telephone bill components and totals -/
structure PhoneBill where
  fixed_charge : ℝ  -- Fixed monthly charge for internet service
  january_call_charge : ℝ  -- Charge for calls made in January
  january_total : ℝ  -- Total bill for January
  february_total : ℝ  -- Total bill for February

/-- Theorem stating that given the conditions, the fixed monthly charge is $6 -/
theorem fixed_charge_is_six (bill : PhoneBill) 
  (h1 : bill.fixed_charge + bill.january_call_charge = bill.january_total)
  (h2 : bill.fixed_charge + 2 * bill.january_call_charge = bill.february_total)
  (h3 : bill.january_total = 48)
  (h4 : bill.february_total = 90) :
  bill.fixed_charge = 6 := by
  sorry

end NUMINAMATH_CALUDE_fixed_charge_is_six_l1537_153722


namespace NUMINAMATH_CALUDE_tan_x0_equals_3_l1537_153759

/-- Given a function f(x) = sin x - cos x, prove that if f''(x₀) = 2f(x₀), then tan x₀ = 3 -/
theorem tan_x0_equals_3 (x₀ : ℝ) : 
  let f : ℝ → ℝ := λ x => Real.sin x - Real.cos x
  (deriv (deriv f)) x₀ = 2 * f x₀ → Real.tan x₀ = 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_x0_equals_3_l1537_153759


namespace NUMINAMATH_CALUDE_tangent_line_point_on_circle_l1537_153745

/-- Given a line ax + by - 1 = 0 tangent to the circle x² + y² = 1,
    prove that the point P(a, b) lies on the circle. -/
theorem tangent_line_point_on_circle (a b : ℝ) :
  (∀ x y, x^2 + y^2 = 1 → (a*x + b*y = 1)) →  -- Line is tangent to circle
  a^2 + b^2 = 1  -- Point P(a, b) is on the circle
  := by sorry

end NUMINAMATH_CALUDE_tangent_line_point_on_circle_l1537_153745


namespace NUMINAMATH_CALUDE_compound_interest_rate_l1537_153736

/-- Compound interest calculation --/
theorem compound_interest_rate (P : ℝ) (t : ℝ) (n : ℝ) (CI : ℝ) (r : ℝ) : 
  P = 20000 →
  t = 2 →
  n = 2 →
  CI = 1648.64 →
  (P + CI) = P * (1 + r / n) ^ (n * t) →
  r = 0.04 := by
sorry

end NUMINAMATH_CALUDE_compound_interest_rate_l1537_153736


namespace NUMINAMATH_CALUDE_tax_calculation_correct_l1537_153797

/-- Represents the tax rate for a given income range -/
structure TaxBracket where
  lower : ℝ
  upper : Option ℝ
  rate : ℝ

/-- Calculates the tax for a given taxable income -/
def calculateTax (brackets : List TaxBracket) (taxableIncome : ℝ) : ℝ :=
  sorry

/-- Represents the tax system with its parameters -/
structure TaxSystem where
  threshold : ℝ
  brackets : List TaxBracket
  elderlyDeduction : ℝ

/-- Calculates the after-tax income given a pre-tax income and tax system -/
def afterTaxIncome (preTaxIncome : ℝ) (system : TaxSystem) : ℝ :=
  sorry

theorem tax_calculation_correct (preTaxIncome : ℝ) (system : TaxSystem) :
  let taxPaid := 180
  let afterTax := 9720
  system.threshold = 5000 ∧
  system.elderlyDeduction = 1000 ∧
  system.brackets = [
    ⟨0, some 3000, 0.03⟩,
    ⟨3000, some 12000, 0.10⟩,
    ⟨12000, some 25000, 0.20⟩,
    ⟨25000, none, 0.25⟩
  ] →
  calculateTax system.brackets (preTaxIncome - system.threshold - system.elderlyDeduction) = taxPaid ∧
  afterTaxIncome preTaxIncome system = afterTax :=
by sorry

end NUMINAMATH_CALUDE_tax_calculation_correct_l1537_153797


namespace NUMINAMATH_CALUDE_arcsin_neg_sqrt2_over_2_l1537_153766

theorem arcsin_neg_sqrt2_over_2 : Real.arcsin (-Real.sqrt 2 / 2) = -π / 4 := by sorry

end NUMINAMATH_CALUDE_arcsin_neg_sqrt2_over_2_l1537_153766


namespace NUMINAMATH_CALUDE_isotopes_same_count_atom_molecule_same_count_molecules_same_count_cations_same_count_anions_same_count_different_elements_different_count_atom_ion_different_count_molecule_ion_different_count_anion_cation_different_count_l1537_153718

-- Define the basic types
inductive ParticleType
| Atom
| Molecule
| Cation
| Anion

-- Define a particle
structure Particle where
  type : ParticleType
  protons : ℕ
  electrons : ℕ

-- Define the property of having the same number of protons and electrons
def sameProtonElectronCount (p1 p2 : Particle) : Prop :=
  p1.protons = p2.protons ∧ p1.electrons = p2.electrons

-- Theorem: Two different atoms (isotopes) can have the same number of protons and electrons
theorem isotopes_same_count :
  ∃ (p1 p2 : Particle), p1.type = ParticleType.Atom ∧ p2.type = ParticleType.Atom ∧
  p1 ≠ p2 ∧ sameProtonElectronCount p1 p2 :=
sorry

-- Theorem: An atom and a molecule can have the same number of protons and electrons
theorem atom_molecule_same_count :
  ∃ (p1 p2 : Particle), p1.type = ParticleType.Atom ∧ p2.type = ParticleType.Molecule ∧
  sameProtonElectronCount p1 p2 :=
sorry

-- Theorem: Two different molecules can have the same number of protons and electrons
theorem molecules_same_count :
  ∃ (p1 p2 : Particle), p1.type = ParticleType.Molecule ∧ p2.type = ParticleType.Molecule ∧
  p1 ≠ p2 ∧ sameProtonElectronCount p1 p2 :=
sorry

-- Theorem: Two different cations can have the same number of protons and electrons
theorem cations_same_count :
  ∃ (p1 p2 : Particle), p1.type = ParticleType.Cation ∧ p2.type = ParticleType.Cation ∧
  p1 ≠ p2 ∧ sameProtonElectronCount p1 p2 :=
sorry

-- Theorem: Two different anions can have the same number of protons and electrons
theorem anions_same_count :
  ∃ (p1 p2 : Particle), p1.type = ParticleType.Anion ∧ p2.type = ParticleType.Anion ∧
  p1 ≠ p2 ∧ sameProtonElectronCount p1 p2 :=
sorry

-- Theorem: Atoms of two different elements cannot have the same number of protons and electrons
theorem different_elements_different_count :
  ∀ (p1 p2 : Particle), p1.type = ParticleType.Atom ∧ p2.type = ParticleType.Atom ∧
  p1.protons ≠ p2.protons → ¬(sameProtonElectronCount p1 p2) :=
sorry

-- Theorem: An atom and an ion cannot have the same number of protons and electrons
theorem atom_ion_different_count :
  ∀ (p1 p2 : Particle), p1.type = ParticleType.Atom ∧ (p2.type = ParticleType.Cation ∨ p2.type = ParticleType.Anion) →
  ¬(sameProtonElectronCount p1 p2) :=
sorry

-- Theorem: A molecule and an ion cannot have the same number of protons and electrons
theorem molecule_ion_different_count :
  ∀ (p1 p2 : Particle), p1.type = ParticleType.Molecule ∧ (p2.type = ParticleType.Cation ∨ p2.type = ParticleType.Anion) →
  ¬(sameProtonElectronCount p1 p2) :=
sorry

-- Theorem: An anion and a cation cannot have the same number of protons and electrons
theorem anion_cation_different_count :
  ∀ (p1 p2 : Particle), p1.type = ParticleType.Anion ∧ p2.type = ParticleType.Cation →
  ¬(sameProtonElectronCount p1 p2) :=
sorry

end NUMINAMATH_CALUDE_isotopes_same_count_atom_molecule_same_count_molecules_same_count_cations_same_count_anions_same_count_different_elements_different_count_atom_ion_different_count_molecule_ion_different_count_anion_cation_different_count_l1537_153718


namespace NUMINAMATH_CALUDE_sum_of_digits_up_to_2023_l1537_153728

/-- Sum of digits of a natural number -/
def sumOfDigits (n : ℕ) : ℕ := sorry

/-- Sum of digits of all numbers from 1 to n -/
def sumOfDigitsUpTo (n : ℕ) : ℕ := 
  (List.range n).map (λ i => sumOfDigits (i + 1)) |>.sum

/-- The sum of digits of all numbers from 1 to 2023 is 27314 -/
theorem sum_of_digits_up_to_2023 : sumOfDigitsUpTo 2023 = 27314 := by sorry

end NUMINAMATH_CALUDE_sum_of_digits_up_to_2023_l1537_153728


namespace NUMINAMATH_CALUDE_linear_function_expression_y_value_at_negative_four_l1537_153733

/-- A linear function passing through two given points -/
structure LinearFunction where
  k : ℝ
  b : ℝ
  point1 : k * 1 + b = 5
  point2 : k * (-1) + b = 1

/-- The unique linear function passing through (1, 5) and (-1, 1) -/
def uniqueLinearFunction : LinearFunction where
  k := 2
  b := 3
  point1 := by sorry
  point2 := by sorry

theorem linear_function_expression (f : LinearFunction) :
  f.k = 2 ∧ f.b = 3 := by sorry

theorem y_value_at_negative_four (f : LinearFunction) :
  f.k * (-4) + f.b = -5 := by sorry

end NUMINAMATH_CALUDE_linear_function_expression_y_value_at_negative_four_l1537_153733


namespace NUMINAMATH_CALUDE_vector_addition_l1537_153749

/-- Given two vectors a and b in ℝ², prove that 2b + 3a equals (6,1) -/
theorem vector_addition (a b : ℝ × ℝ) (h1 : a = (2, 1)) (h2 : b = (0, -1)) :
  2 • b + 3 • a = (6, 1) := by
  sorry

end NUMINAMATH_CALUDE_vector_addition_l1537_153749


namespace NUMINAMATH_CALUDE_expand_product_l1537_153731

theorem expand_product (x : ℝ) : (x + 4) * (x - 9) = x^2 - 5*x - 36 := by
  sorry

end NUMINAMATH_CALUDE_expand_product_l1537_153731


namespace NUMINAMATH_CALUDE_average_score_is_94_l1537_153756

def june_score : ℝ := 97
def patty_score : ℝ := 85
def josh_score : ℝ := 100
def henry_score : ℝ := 94

def num_children : ℕ := 4

theorem average_score_is_94 :
  (june_score + patty_score + josh_score + henry_score) / num_children = 94 := by
  sorry

end NUMINAMATH_CALUDE_average_score_is_94_l1537_153756


namespace NUMINAMATH_CALUDE_composite_product_division_l1537_153773

def first_five_composites : List Nat := [12, 14, 15, 16, 18]
def next_five_composites : List Nat := [21, 22, 24, 25, 26]

def product_of_list (l : List Nat) : Nat :=
  l.foldl (· * ·) 1

theorem composite_product_division :
  (product_of_list first_five_composites) / (product_of_list next_five_composites) = 72 / 715 := by
  sorry

end NUMINAMATH_CALUDE_composite_product_division_l1537_153773


namespace NUMINAMATH_CALUDE_chord_ratio_is_sqrt6_to_2_l1537_153758

-- Define the points and circles
structure PointOnLine where
  position : ℝ

structure Circle where
  center : ℝ
  radius : ℝ

-- Define the problem setup
def setup (A B C D : PointOnLine) (circle_AB circle_BC circle_CD : Circle) :=
  -- Points are on a line and equally spaced
  B.position - A.position = C.position - B.position ∧
  C.position - B.position = D.position - C.position ∧
  -- Circles have diameters AB, BC, and CD
  circle_AB.radius = (B.position - A.position) / 2 ∧
  circle_BC.radius = (C.position - B.position) / 2 ∧
  circle_CD.radius = (D.position - C.position) / 2 ∧
  circle_AB.center = (A.position + B.position) / 2 ∧
  circle_BC.center = (B.position + C.position) / 2 ∧
  circle_CD.center = (C.position + D.position) / 2

-- Define the tangent line and chords
def tangent_and_chords (A : PointOnLine) (circle_CD : Circle) (chord_AB chord_BC : ℝ) :=
  ∃ (l : ℝ → ℝ), 
    -- l is tangent to circle_CD at point A
    (l A.position - circle_CD.center)^2 = circle_CD.radius^2 ∧
    -- chord_AB and chord_BC are the lengths of the chords cut by l on circles with diameters AB and BC
    chord_AB > 0 ∧ chord_BC > 0

-- The main theorem
theorem chord_ratio_is_sqrt6_to_2 
  (A B C D : PointOnLine) 
  (circle_AB circle_BC circle_CD : Circle) 
  (chord_AB chord_BC : ℝ) :
  setup A B C D circle_AB circle_BC circle_CD →
  tangent_and_chords A circle_CD chord_AB chord_BC →
  chord_AB / chord_BC = Real.sqrt 6 / 2 :=
sorry

end NUMINAMATH_CALUDE_chord_ratio_is_sqrt6_to_2_l1537_153758


namespace NUMINAMATH_CALUDE_exists_winning_strategy_l1537_153707

/-- Represents the state of the switch -/
inductive SwitchState
| Left
| Right

/-- Represents a trainee's action in the room -/
inductive TraineeAction
| FlipSwitch
| DoNothing
| Declare

/-- Represents the result of the challenge -/
inductive ChallengeResult
| Success
| Failure

/-- The strategy function type for a trainee -/
def TraineeStrategy := Nat → SwitchState → TraineeAction

/-- The type representing the challenge setup -/
structure Challenge where
  numTrainees : Nat
  initialState : SwitchState

/-- The function to simulate the challenge -/
noncomputable def simulateChallenge (c : Challenge) (strategies : List TraineeStrategy) : ChallengeResult :=
  sorry

/-- The main theorem to prove -/
theorem exists_winning_strategy :
  ∃ (strategies : List TraineeStrategy),
    strategies.length = 42 ∧
    ∀ (c : Challenge),
      c.numTrainees = 42 →
      simulateChallenge c strategies = ChallengeResult.Success :=
sorry

end NUMINAMATH_CALUDE_exists_winning_strategy_l1537_153707


namespace NUMINAMATH_CALUDE_abs_sum_complex_roots_l1537_153760

/-- Given complex numbers a, b, and c satisfying certain conditions,
    prove that |a + b + c| is either 0 or 1. -/
theorem abs_sum_complex_roots (a b c : ℂ) 
    (h1 : Complex.abs a = 1)
    (h2 : Complex.abs b = 1)
    (h3 : Complex.abs c = 1)
    (h4 : a^2 * b + b^2 * c + c^2 * a = 0) :
    Complex.abs (a + b + c) = 0 ∨ Complex.abs (a + b + c) = 1 := by
  sorry

end NUMINAMATH_CALUDE_abs_sum_complex_roots_l1537_153760


namespace NUMINAMATH_CALUDE_circle_area_ratio_after_radius_increase_l1537_153739

theorem circle_area_ratio_after_radius_increase (r : ℝ) (h : r > 0) : 
  (π * r^2) / (π * (1.5 * r)^2) = 4/9 := by sorry

end NUMINAMATH_CALUDE_circle_area_ratio_after_radius_increase_l1537_153739


namespace NUMINAMATH_CALUDE_gcd_2134_1455_ternary_l1537_153769

theorem gcd_2134_1455_ternary : 
  ∃ m : ℕ, 
    Nat.gcd 2134 1455 = m ∧ 
    (Nat.digits 3 m).reverse = [1, 0, 1, 2, 1] :=
by sorry

end NUMINAMATH_CALUDE_gcd_2134_1455_ternary_l1537_153769


namespace NUMINAMATH_CALUDE_initial_children_meals_l1537_153720

/-- Calculates the number of meals initially available for children given the total adult meals and remaining meals after some adults eat. -/
def children_meals (total_adult_meals : ℕ) (adults_eaten : ℕ) (remaining_child_meals : ℕ) : ℕ :=
  (total_adult_meals * remaining_child_meals) / (total_adult_meals - adults_eaten)

/-- Proves that the number of meals initially available for children is 90. -/
theorem initial_children_meals :
  children_meals 70 14 72 = 90 := by
  sorry

end NUMINAMATH_CALUDE_initial_children_meals_l1537_153720


namespace NUMINAMATH_CALUDE_only_one_statement_correct_l1537_153703

-- Define the concept of opposite numbers
def are_opposites (a b : ℝ) : Prop := a + b = 0

-- Define the four statements
def statement1 : Prop := ∀ a b : ℝ, (a > 0 ∧ b < 0) ∨ (a < 0 ∧ b > 0) → are_opposites a b
def statement2 : Prop := ∀ a : ℝ, ∃ b : ℝ, are_opposites a b ∧ b < 0
def statement3 : Prop := ∀ a b : ℝ, are_opposites a b → a + b = 0
def statement4 : Prop := ∀ a b : ℝ, are_opposites a b → (a > 0 ∧ b < 0) ∨ (a < 0 ∧ b > 0)

-- Theorem stating that only one of the statements is correct
theorem only_one_statement_correct :
  (¬statement1 ∧ ¬statement2 ∧ statement3 ∧ ¬statement4) ∧
  ¬(statement1 ∨ statement2 ∨ statement4) :=
sorry

end NUMINAMATH_CALUDE_only_one_statement_correct_l1537_153703


namespace NUMINAMATH_CALUDE_collinear_vectors_x_value_l1537_153735

/-- Two vectors are collinear if one is a scalar multiple of the other -/
def collinear (a b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, a.1 * b.2 = k * a.2 * b.1

theorem collinear_vectors_x_value :
  let a : ℝ × ℝ := (2, -3)
  let b : ℝ × ℝ := (x, 6)
  collinear a b → x = -4 :=
by
  sorry

end NUMINAMATH_CALUDE_collinear_vectors_x_value_l1537_153735


namespace NUMINAMATH_CALUDE_solution_set_equivalence_l1537_153778

theorem solution_set_equivalence (x : ℝ) :
  (x - 2 > 1 ∧ x < 4) ↔ (3 < x ∧ x < 4) := by sorry

end NUMINAMATH_CALUDE_solution_set_equivalence_l1537_153778


namespace NUMINAMATH_CALUDE_positive_integer_sum_greater_than_product_l1537_153785

theorem positive_integer_sum_greater_than_product (a b : ℕ+) :
  a + b > a * b ↔ a = 1 ∨ b = 1 := by sorry

end NUMINAMATH_CALUDE_positive_integer_sum_greater_than_product_l1537_153785


namespace NUMINAMATH_CALUDE_propositions_truth_l1537_153775

theorem propositions_truth :
  (¬ ∀ x : ℝ, x^4 > x^2) ∧
  (∃ α : ℝ, Real.sin (3 * α) = 3 * Real.sin α) ∧
  (¬ ∃ a : ℝ, ∀ x : ℝ, x^2 + 2*x + a < 0) :=
by sorry

end NUMINAMATH_CALUDE_propositions_truth_l1537_153775


namespace NUMINAMATH_CALUDE_equal_numbers_product_l1537_153790

theorem equal_numbers_product (a b c d : ℝ) : 
  (a + b + c + d) / 4 = 20 ∧ 
  a = 12 ∧ 
  b = 22 ∧ 
  c = d → 
  c * d = 529 := by
sorry

end NUMINAMATH_CALUDE_equal_numbers_product_l1537_153790


namespace NUMINAMATH_CALUDE_complex_power_sum_l1537_153786

theorem complex_power_sum (z : ℂ) (h : z + 1/z = 2 * Real.cos (5 * π / 180)) :
  z^1800 + 1/(z^1800) = -2 := by
  sorry

end NUMINAMATH_CALUDE_complex_power_sum_l1537_153786


namespace NUMINAMATH_CALUDE_key_chain_manufacturing_cost_l1537_153772

theorem key_chain_manufacturing_cost 
  (selling_price : ℝ)
  (old_profit_percentage : ℝ)
  (new_profit_percentage : ℝ)
  (new_manufacturing_cost : ℝ)
  (h1 : old_profit_percentage = 0.3)
  (h2 : new_profit_percentage = 0.5)
  (h3 : new_manufacturing_cost = 50)
  (h4 : selling_price = new_manufacturing_cost / (1 - new_profit_percentage)) :
  selling_price * (1 - old_profit_percentage) = 70 := by
sorry

end NUMINAMATH_CALUDE_key_chain_manufacturing_cost_l1537_153772


namespace NUMINAMATH_CALUDE_derivative_f_at_negative_two_l1537_153704

-- Define the function f
def f (x : ℝ) : ℝ := x^3

-- State the theorem
theorem derivative_f_at_negative_two :
  deriv f (-2) = 0 := by sorry

end NUMINAMATH_CALUDE_derivative_f_at_negative_two_l1537_153704


namespace NUMINAMATH_CALUDE_sixth_degree_polynomial_identity_l1537_153719

theorem sixth_degree_polynomial_identity (b₁ b₂ b₃ c₁ c₂ c₃ : ℝ) 
  (h : ∀ x : ℝ, x^6 + x^5 + x^4 + x^3 + x^2 + x + 1 = 
     (x^2 + b₁*x + c₁) * (x^2 + b₂*x + c₂) * (x^2 + b₃*x + c₃)) : 
  b₁^2 + b₂^2 + b₃^2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_sixth_degree_polynomial_identity_l1537_153719


namespace NUMINAMATH_CALUDE_division_value_problem_l1537_153783

theorem division_value_problem (x : ℝ) : 
  ((7.5 / x) * 12 = 15) → x = 6 := by
  sorry

end NUMINAMATH_CALUDE_division_value_problem_l1537_153783


namespace NUMINAMATH_CALUDE_cats_remaining_after_sale_l1537_153798

theorem cats_remaining_after_sale (siamese : ℕ) (house : ℕ) (sold : ℕ) : 
  siamese = 13 → house = 5 → sold = 10 → siamese + house - sold = 8 := by
sorry

end NUMINAMATH_CALUDE_cats_remaining_after_sale_l1537_153798


namespace NUMINAMATH_CALUDE_factorial_500_trailing_zeroes_l1537_153734

/-- The number of trailing zeroes in n! -/
def trailingZeroes (n : ℕ) : ℕ :=
  (n / 5) + (n / 25) + (n / 125)

/-- Theorem: The number of trailing zeroes in 500! is 124 -/
theorem factorial_500_trailing_zeroes :
  trailingZeroes 500 = 124 := by
  sorry

end NUMINAMATH_CALUDE_factorial_500_trailing_zeroes_l1537_153734


namespace NUMINAMATH_CALUDE_solution_values_l1537_153795

def has_55_solutions (n : ℕ+) : Prop :=
  (Finset.filter (fun (x, y, z) => 3 * x + 3 * y + z = n ∧ x > 0 ∧ y > 0 ∧ z > 0)
    (Finset.product (Finset.range n) (Finset.product (Finset.range n) (Finset.range n)))).card = 55

theorem solution_values (n : ℕ+) (h : has_55_solutions n) : n = 34 ∨ n = 37 := by
  sorry

end NUMINAMATH_CALUDE_solution_values_l1537_153795


namespace NUMINAMATH_CALUDE_trash_outside_classrooms_l1537_153747

theorem trash_outside_classrooms 
  (total_trash : ℕ) 
  (classroom_trash : ℕ) 
  (h1 : total_trash = 1576) 
  (h2 : classroom_trash = 344) : 
  total_trash - classroom_trash = 1232 := by
sorry

end NUMINAMATH_CALUDE_trash_outside_classrooms_l1537_153747


namespace NUMINAMATH_CALUDE_second_year_sampled_is_thirteen_l1537_153764

/-- Calculates the number of second-year students sampled in a stratified survey. -/
def second_year_sampled (total_population : ℕ) (second_year_population : ℕ) (total_sampled : ℕ) : ℕ :=
  (second_year_population * total_sampled) / total_population

/-- Proves that the number of second-year students sampled is 13 given the problem conditions. -/
theorem second_year_sampled_is_thirteen :
  second_year_sampled 2100 780 35 = 13 := by
  sorry

end NUMINAMATH_CALUDE_second_year_sampled_is_thirteen_l1537_153764


namespace NUMINAMATH_CALUDE_courtyard_width_l1537_153779

theorem courtyard_width (length : ℝ) (width : ℝ) (stone_length : ℝ) (stone_width : ℝ) 
  (total_stones : ℕ) (h1 : length = 30) (h2 : stone_length = 2) (h3 : stone_width = 1) 
  (h4 : total_stones = 240) (h5 : length * width = stone_length * stone_width * total_stones) : 
  width = 16 := by
  sorry

end NUMINAMATH_CALUDE_courtyard_width_l1537_153779


namespace NUMINAMATH_CALUDE_fifth_term_equals_fourth_l1537_153742

/-- A geometric sequence of positive integers -/
structure GeometricSequence where
  a : ℕ+  -- first term
  r : ℕ+  -- common ratio

/-- The nth term of a geometric sequence -/
def nthTerm (seq : GeometricSequence) (n : ℕ) : ℕ+ :=
  seq.a * (seq.r ^ (n - 1))

theorem fifth_term_equals_fourth (seq : GeometricSequence) 
  (h1 : seq.a = 4)
  (h2 : nthTerm seq 4 = 324) :
  nthTerm seq 5 = 324 := by
  sorry

end NUMINAMATH_CALUDE_fifth_term_equals_fourth_l1537_153742


namespace NUMINAMATH_CALUDE_pseudo_symmetry_point_l1537_153724

noncomputable def f (x : ℝ) : ℝ := x^2 - 6*x + 4*Real.log x

noncomputable def g (x₀ x : ℝ) : ℝ := 
  (2*x₀ + 4/x₀ - 6)*(x - x₀) + x₀^2 - 6*x₀ + 4*Real.log x₀

theorem pseudo_symmetry_point :
  ∃! x₀ : ℝ, x₀ > 0 ∧ 
  ∀ x, x > 0 → x ≠ x₀ → (f x - g x₀ x) / (x - x₀) > 0 :=
sorry

end NUMINAMATH_CALUDE_pseudo_symmetry_point_l1537_153724


namespace NUMINAMATH_CALUDE_subtraction_of_negatives_l1537_153701

theorem subtraction_of_negatives : (-2) - (-4) = 2 := by
  sorry

end NUMINAMATH_CALUDE_subtraction_of_negatives_l1537_153701


namespace NUMINAMATH_CALUDE_circle_B_radius_l1537_153771

-- Define the circles
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the problem setup
def problem_setup (A B C D : Circle) : Prop :=
  -- Circles A, B, and C are externally tangent to each other
  (A.radius + B.radius = dist A.center B.center) ∧
  (A.radius + C.radius = dist A.center C.center) ∧
  (B.radius + C.radius = dist B.center C.center) ∧
  -- Circles A, B, and C are internally tangent to circle D
  (D.radius - A.radius = dist D.center A.center) ∧
  (D.radius - B.radius = dist D.center B.center) ∧
  (D.radius - C.radius = dist D.center C.center) ∧
  -- Circles B and C are congruent
  (B.radius = C.radius) ∧
  -- Circle A has radius 1
  (A.radius = 1) ∧
  -- Circle A passes through the center of D
  (dist A.center D.center = A.radius + D.radius)

-- Theorem statement
theorem circle_B_radius (A B C D : Circle) :
  problem_setup A B C D → B.radius = 8/9 := by
  sorry

end NUMINAMATH_CALUDE_circle_B_radius_l1537_153771


namespace NUMINAMATH_CALUDE_triple_nested_log_sum_l1537_153768

-- Define the logarithm function
noncomputable def log (b : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log b

-- Define the theorem
theorem triple_nested_log_sum (x y z : ℝ) :
  log 3 (log 4 (log 5 x)) = 0 ∧
  log 4 (log 5 (log 3 y)) = 0 ∧
  log 5 (log 3 (log 4 z)) = 0 →
  x + y + z = 932 := by
  sorry

end NUMINAMATH_CALUDE_triple_nested_log_sum_l1537_153768


namespace NUMINAMATH_CALUDE_chopped_cube_height_l1537_153761

/-- The height of a cube with a chopped corner -/
theorem chopped_cube_height (s : ℝ) (h_s : s = 2) : 
  let diagonal := s * Real.sqrt 3
  let triangle_side := Real.sqrt (2 * s^2)
  let triangle_area := (Real.sqrt 3 / 4) * triangle_side^2
  let pyramid_volume := (1 / 6) * s^3
  let pyramid_height := 3 * pyramid_volume / triangle_area
  s - pyramid_height = (2 * Real.sqrt 3 - 1) / Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_chopped_cube_height_l1537_153761


namespace NUMINAMATH_CALUDE_school_greening_area_equation_l1537_153732

/-- Represents the growth of a greening area over time -/
def greeningAreaGrowth (initialArea finalArea : ℝ) (years : ℕ) (growthRate : ℝ) : Prop :=
  initialArea * (1 + growthRate) ^ years = finalArea

/-- The equation for the school's greening area growth -/
theorem school_greening_area_equation :
  greeningAreaGrowth 1000 1440 2 x ↔ 1000 * (1 + x)^2 = 1440 := by
  sorry

end NUMINAMATH_CALUDE_school_greening_area_equation_l1537_153732


namespace NUMINAMATH_CALUDE_unique_positive_solution_l1537_153770

theorem unique_positive_solution :
  ∃! x : ℝ, x > 0 ∧ x ≤ 1 ∧ Real.cos (Real.arctan (Real.sin (Real.arccos x))) = x :=
by
  sorry

end NUMINAMATH_CALUDE_unique_positive_solution_l1537_153770


namespace NUMINAMATH_CALUDE_franks_trivia_score_l1537_153700

/-- Frank's trivia game score calculation -/
theorem franks_trivia_score :
  ∀ (first_half second_half points_per_question : ℕ),
    first_half = 3 →
    second_half = 2 →
    points_per_question = 3 →
    (first_half + second_half) * points_per_question = 15 :=
by
  sorry

end NUMINAMATH_CALUDE_franks_trivia_score_l1537_153700


namespace NUMINAMATH_CALUDE_danny_bottle_caps_l1537_153757

/-- Represents the number of bottle caps Danny found at the park -/
def new_bottle_caps : ℕ := 50

/-- Represents the number of old bottle caps Danny threw away -/
def thrown_away_caps : ℕ := 6

/-- Represents the current number of bottle caps in Danny's collection -/
def current_collection : ℕ := 60

/-- Represents the difference between found and thrown away caps -/
def difference_found_thrown : ℕ := 44

theorem danny_bottle_caps :
  new_bottle_caps = thrown_away_caps + difference_found_thrown ∧
  current_collection = (new_bottle_caps + thrown_away_caps) - thrown_away_caps :=
by sorry

end NUMINAMATH_CALUDE_danny_bottle_caps_l1537_153757


namespace NUMINAMATH_CALUDE_max_value_of_f_l1537_153715

-- Define the function f
def f (x : ℝ) : ℝ := x^3 - 3*x^2 - 9*x + 2

-- Define the interval
def interval : Set ℝ := Set.Icc (-2) 2

-- Statement to prove
theorem max_value_of_f :
  ∃ (c : ℝ), c ∈ interval ∧ ∀ (x : ℝ), x ∈ interval → f x ≤ f c ∧ f c = 7 :=
sorry

end NUMINAMATH_CALUDE_max_value_of_f_l1537_153715


namespace NUMINAMATH_CALUDE_intersection_complement_A_and_B_range_of_a_for_C_subset_A_l1537_153789

-- Define the sets A and B
def A : Set ℝ := {x | x^2 + 2*x < 0}
def B : Set ℝ := {x | ∃ y, y = Real.sqrt (x + 1)}

-- Define the set C
def C (a : ℝ) : Set ℝ := {x | a < x ∧ x < 2*a + 1}

-- Statement 1
theorem intersection_complement_A_and_B :
  (Set.compl A ∩ B) = {x : ℝ | x ≥ 0} := by sorry

-- Statement 2
theorem range_of_a_for_C_subset_A :
  {a : ℝ | C a ⊆ A} = {a : ℝ | a ≤ -1/2} := by sorry

end NUMINAMATH_CALUDE_intersection_complement_A_and_B_range_of_a_for_C_subset_A_l1537_153789


namespace NUMINAMATH_CALUDE_swim_club_additional_capacity_l1537_153746

/-- Represents the swimming club's transportation setup -/
structure SwimClubTransport where
  num_cars : ℕ
  num_vans : ℕ
  people_per_car : ℕ
  people_per_van : ℕ
  max_car_capacity : ℕ
  max_van_capacity : ℕ

/-- Calculates the additional capacity of the swim club's transportation -/
def additional_capacity (t : SwimClubTransport) : ℕ :=
  (t.num_cars * t.max_car_capacity + t.num_vans * t.max_van_capacity) -
  (t.num_cars * t.people_per_car + t.num_vans * t.people_per_van)

/-- Theorem stating that the additional capacity for the given scenario is 17 -/
theorem swim_club_additional_capacity :
  let t : SwimClubTransport :=
    { num_cars := 2
    , num_vans := 3
    , people_per_car := 5
    , people_per_van := 3
    , max_car_capacity := 6
    , max_van_capacity := 8
    }
  additional_capacity t = 17 := by
  sorry


end NUMINAMATH_CALUDE_swim_club_additional_capacity_l1537_153746


namespace NUMINAMATH_CALUDE_total_eyes_count_l1537_153725

theorem total_eyes_count (num_boys : ℕ) (eyes_per_boy : ℕ) (h1 : num_boys = 23) (h2 : eyes_per_boy = 2) :
  num_boys * eyes_per_boy = 46 := by
  sorry

end NUMINAMATH_CALUDE_total_eyes_count_l1537_153725


namespace NUMINAMATH_CALUDE_six_digit_square_number_puzzle_l1537_153741

theorem six_digit_square_number_puzzle :
  ∃ (n x y : ℕ), 
    100000 ≤ n^2 ∧ n^2 < 1000000 ∧
    10 ≤ x ∧ x ≤ 99 ∧
    0 ≤ y ∧ y ≤ 9 ∧
    n^2 = 10101 * x + y^2 ∧
    (n^2 = 232324 ∨ n^2 = 595984 ∨ n^2 = 929296) :=
by sorry

end NUMINAMATH_CALUDE_six_digit_square_number_puzzle_l1537_153741


namespace NUMINAMATH_CALUDE_race_problem_l1537_153729

/-- Race problem statement -/
theorem race_problem (race_length : ℕ) (distance_between : ℕ) (jack_distance : ℕ) :
  race_length = 1000 →
  distance_between = 848 →
  jack_distance = race_length - distance_between →
  jack_distance = 152 :=
by sorry

end NUMINAMATH_CALUDE_race_problem_l1537_153729


namespace NUMINAMATH_CALUDE_gcd_lcm_product_180_l1537_153765

theorem gcd_lcm_product_180 (a b : ℕ+) :
  (Nat.gcd a b) * (Nat.lcm a b) = 180 →
  (∃ s : Finset ℕ+, s.card = 9 ∧ ∀ x, x ∈ s ↔ ∃ c d : ℕ+, (Nat.gcd c d) * (Nat.lcm c d) = 180 ∧ Nat.gcd c d = x) :=
by sorry

end NUMINAMATH_CALUDE_gcd_lcm_product_180_l1537_153765


namespace NUMINAMATH_CALUDE_maxim_birth_probability_maxim_birth_probability_proof_l1537_153776

/-- The year Maxim starts first grade -/
def start_year : ℕ := 2014

/-- The month Maxim starts first grade (September = 9) -/
def start_month : ℕ := 9

/-- The day Maxim starts first grade -/
def start_day : ℕ := 1

/-- Maxim's age when he starts first grade -/
def start_age : ℕ := 6

/-- The year we're interested in for Maxim's birth -/
def birth_year_of_interest : ℕ := 2008

/-- Function to determine if a year is a leap year -/
def is_leap_year (year : ℕ) : Bool :=
  (year % 4 == 0 && year % 100 ≠ 0) || (year % 400 == 0)

/-- Function to get the number of days in a month -/
def days_in_month (year : ℕ) (month : ℕ) : ℕ :=
  if month == 2 then
    if is_leap_year year then 29 else 28
  else if month ∈ [4, 6, 9, 11] then 30
  else 31

/-- The probability that Maxim was born in 2008 -/
theorem maxim_birth_probability : ℚ :=
  244 / 365

/-- Proof of the probability calculation -/
theorem maxim_birth_probability_proof :
  maxim_birth_probability = 244 / 365 := by
  sorry

end NUMINAMATH_CALUDE_maxim_birth_probability_maxim_birth_probability_proof_l1537_153776


namespace NUMINAMATH_CALUDE_set_equals_naturals_l1537_153721

def is_closed_under_multiplication_by_four (X : Set ℕ) : Prop :=
  ∀ x ∈ X, (4 * x) ∈ X

def is_closed_under_floor_sqrt (X : Set ℕ) : Prop :=
  ∀ x ∈ X, Nat.sqrt x ∈ X

theorem set_equals_naturals (X : Set ℕ) 
  (h_nonempty : X.Nonempty)
  (h_mul_four : is_closed_under_multiplication_by_four X)
  (h_floor_sqrt : is_closed_under_floor_sqrt X) : 
  X = Set.univ :=
sorry

end NUMINAMATH_CALUDE_set_equals_naturals_l1537_153721


namespace NUMINAMATH_CALUDE_integral_reciprocal_plus_x_l1537_153788

theorem integral_reciprocal_plus_x : ∫ x in (2 : ℝ)..4, (1 / x + x) = Real.log 2 + 6 := by
  sorry

end NUMINAMATH_CALUDE_integral_reciprocal_plus_x_l1537_153788


namespace NUMINAMATH_CALUDE_remainder_of_m_l1537_153777

theorem remainder_of_m (m : ℕ) (h1 : m^2 % 7 = 1) (h2 : m^3 % 7 = 6) : m % 7 = 6 := by
  sorry

end NUMINAMATH_CALUDE_remainder_of_m_l1537_153777


namespace NUMINAMATH_CALUDE_complement_P_inter_Q_l1537_153796

open Set

def P : Set ℝ := { x | x - 1 ≤ 0 }
def Q : Set ℝ := { x | 0 < x ∧ x ≤ 2 }

theorem complement_P_inter_Q : (P.compl ∩ Q) = Ioo 1 2 := by sorry

end NUMINAMATH_CALUDE_complement_P_inter_Q_l1537_153796


namespace NUMINAMATH_CALUDE_absolute_value_equation_unique_solution_l1537_153787

theorem absolute_value_equation_unique_solution :
  ∃! x : ℝ, |x - 3| = |x + 5| := by
sorry

end NUMINAMATH_CALUDE_absolute_value_equation_unique_solution_l1537_153787


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1537_153726

/-- Given an arithmetic sequence {a_n} with S_n as the sum of its first n terms,
    if -a_{2015} < a_1 < -a_{2016}, then S_{2015} > 0 and S_{2016} < 0. -/
theorem arithmetic_sequence_sum (a : ℕ → ℝ) (S : ℕ → ℝ)
  (h_arithmetic : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1))
  (h_sum : ∀ n, S n = n * (a 1 + a n) / 2)
  (h_inequality : -a 2015 < a 1 ∧ a 1 < -a 2016) :
  S 2015 > 0 ∧ S 2016 < 0 := by
  sorry


end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1537_153726


namespace NUMINAMATH_CALUDE_sum_of_reciprocal_squares_l1537_153791

-- Define the cubic equation
def cubic_equation (x : ℝ) : Prop := x^3 - 6*x^2 + 11*x - 6 = 0

-- Define the roots of the equation
def roots (a b c : ℝ) : Prop := cubic_equation a ∧ cubic_equation b ∧ cubic_equation c

-- Theorem statement
theorem sum_of_reciprocal_squares (a b c : ℝ) :
  roots a b c → 1/a^2 + 1/b^2 + 1/c^2 = 49/36 := by sorry

end NUMINAMATH_CALUDE_sum_of_reciprocal_squares_l1537_153791


namespace NUMINAMATH_CALUDE_garden_breadth_is_100_l1537_153754

/-- Represents a rectangular garden -/
structure RectangularGarden where
  length : ℝ
  breadth : ℝ

/-- Calculates the perimeter of a rectangular garden -/
def perimeter (garden : RectangularGarden) : ℝ :=
  2 * (garden.length + garden.breadth)

theorem garden_breadth_is_100 :
  ∃ (garden : RectangularGarden),
    garden.length = 250 ∧
    perimeter garden = 700 ∧
    garden.breadth = 100 := by
  sorry

end NUMINAMATH_CALUDE_garden_breadth_is_100_l1537_153754


namespace NUMINAMATH_CALUDE_first_player_winning_strategy_l1537_153708

/-- Represents the number of points on the circle -/
def n : Nat := 98

/-- Represents the number of moves to connect n-2 points -/
def N (n : Nat) : Nat := (n - 3) * (n - 4) / 2

/-- Represents whether a number is odd -/
def isOdd (m : Nat) : Prop := ∃ k, m = 2 * k + 1

/-- Represents the winning condition for the first player -/
def firstPlayerWins (n : Nat) : Prop := isOdd (N n)

/-- Theorem stating that the first player has a winning strategy -/
theorem first_player_winning_strategy : firstPlayerWins n := by
  sorry

end NUMINAMATH_CALUDE_first_player_winning_strategy_l1537_153708


namespace NUMINAMATH_CALUDE_price_adjustment_l1537_153748

theorem price_adjustment (a : ℝ) : 
  let price_after_reductions := a * (1 - 0.1) * (1 - 0.1)
  let final_price := price_after_reductions * (1 + 0.2)
  final_price = 0.972 * a :=
by sorry

end NUMINAMATH_CALUDE_price_adjustment_l1537_153748


namespace NUMINAMATH_CALUDE_system_solution_l1537_153738

theorem system_solution :
  ∃ (x y : ℚ), 
    (7 * x = -9 - 3 * y) ∧ 
    (4 * x = 5 * y - 34) ∧ 
    (x = -413 / 235) ∧ 
    (y = -202 / 47) := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l1537_153738


namespace NUMINAMATH_CALUDE_journey_distance_l1537_153712

theorem journey_distance (total_journey : ℕ) (remaining : ℕ) (driven : ℕ) : 
  total_journey = 1200 → remaining = 277 → driven = total_journey - remaining → driven = 923 := by
sorry

end NUMINAMATH_CALUDE_journey_distance_l1537_153712


namespace NUMINAMATH_CALUDE_federal_guideline_requirement_l1537_153793

/-- The daily minimum requirement of vegetables in cups according to federal guidelines. -/
def daily_requirement : ℕ := 3

/-- The number of days Sarah has been eating vegetables. -/
def days_counted : ℕ := 5

/-- The total amount of vegetables Sarah has eaten in cups. -/
def vegetables_eaten : ℕ := 8

/-- Sarah's daily consumption needed to meet the minimum requirement. -/
def sarah_daily_need : ℕ := 3

theorem federal_guideline_requirement :
  daily_requirement = sarah_daily_need :=
by sorry

end NUMINAMATH_CALUDE_federal_guideline_requirement_l1537_153793


namespace NUMINAMATH_CALUDE_line_segment_parameterization_sum_of_squares_l1537_153706

/-- Given a line segment connecting (-4, 10) and (2, -3), parameterized by x = at + b and y = ct + d
    where -1 ≤ t ≤ 1 and t = -1 corresponds to (-4, 10), prove that a^2 + b^2 + c^2 + d^2 = 321 -/
theorem line_segment_parameterization_sum_of_squares :
  ∀ (a b c d : ℝ),
  (∀ t : ℝ, -1 ≤ t → t ≤ 1 → ∃ x y : ℝ, x = a * t + b ∧ y = c * t + d) →
  (a * (-1) + b = -4 ∧ c * (-1) + d = 10) →
  (a * 1 + b = 2 ∧ c * 1 + d = -3) →
  a^2 + b^2 + c^2 + d^2 = 321 :=
by sorry

end NUMINAMATH_CALUDE_line_segment_parameterization_sum_of_squares_l1537_153706


namespace NUMINAMATH_CALUDE_sum_of_page_numbers_constant_l1537_153723

/-- Represents a magazine with nested double sheets. -/
structure Magazine where
  num_double_sheets : ℕ
  pages_per_double_sheet : ℕ

/-- Calculates the sum of page numbers on a double sheet. -/
def sum_of_page_numbers (m : Magazine) (sheet_number : ℕ) : ℕ :=
  sorry

/-- Theorem: The sum of page numbers on each double sheet is always 130. -/
theorem sum_of_page_numbers_constant (m : Magazine) (sheet_number : ℕ) :
  m.num_double_sheets = 16 →
  m.pages_per_double_sheet = 4 →
  sheet_number ≤ m.num_double_sheets →
  sum_of_page_numbers m sheet_number = 130 :=
sorry

end NUMINAMATH_CALUDE_sum_of_page_numbers_constant_l1537_153723


namespace NUMINAMATH_CALUDE_inclination_angle_range_l1537_153744

/-- The range of inclination angles for a line with equation x*cos(θ) + √3*y - 1 = 0 -/
theorem inclination_angle_range (θ : ℝ) :
  let l : Set (ℝ × ℝ) := {(x, y) | x * Real.cos θ + Real.sqrt 3 * y - 1 = 0}
  let α := Real.arctan (-Real.sqrt 3 / 3 * Real.cos θ)
  α ∈ Set.union (Set.Icc 0 (Real.pi / 6)) (Set.Icc (5 * Real.pi / 6) Real.pi) :=
by sorry


end NUMINAMATH_CALUDE_inclination_angle_range_l1537_153744


namespace NUMINAMATH_CALUDE_two_digit_puzzle_solution_l1537_153753

theorem two_digit_puzzle_solution :
  ∃ (A B : ℕ), 
    A ≠ B ∧ 
    A ≠ 0 ∧ 
    A < 10 ∧ 
    B < 10 ∧ 
    A * B + A + B = 10 * A + B :=
by
  sorry

end NUMINAMATH_CALUDE_two_digit_puzzle_solution_l1537_153753


namespace NUMINAMATH_CALUDE_repeating_decimal_proof_l1537_153709

/-- The repeating decimal 0.817817817... as a real number -/
def F : ℚ := 817 / 999

/-- The difference between the denominator and numerator of F when expressed as a fraction in lowest terms -/
def denominator_numerator_difference : ℕ := 999 - 817

theorem repeating_decimal_proof :
  F = 817 / 999 ∧ denominator_numerator_difference = 182 :=
sorry

end NUMINAMATH_CALUDE_repeating_decimal_proof_l1537_153709


namespace NUMINAMATH_CALUDE_problem_statement_l1537_153710

theorem problem_statement : 
  (∀ x : ℝ, x^2 + x + 1 > 0) ∧ (∃ x : ℝ, x^3 = 1 - x^2) := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l1537_153710


namespace NUMINAMATH_CALUDE_dog_age_is_12_l1537_153717

def cat_age : ℕ := 8

def rabbit_age (cat_age : ℕ) : ℕ := cat_age / 2

def dog_age (rabbit_age : ℕ) : ℕ := 3 * rabbit_age

theorem dog_age_is_12 : dog_age (rabbit_age cat_age) = 12 := by
  sorry

end NUMINAMATH_CALUDE_dog_age_is_12_l1537_153717


namespace NUMINAMATH_CALUDE_converse_statement_l1537_153705

/-- Given that m is a real number, prove that the converse of the statement 
    "If m > 0, then the equation x^2 + x - m = 0 has real roots" 
    is "If the equation x^2 + x - m = 0 has real roots, then m > 0" -/
theorem converse_statement (m : ℝ) : 
  (∃ x : ℝ, x^2 + x - m = 0) → m > 0 :=
sorry

end NUMINAMATH_CALUDE_converse_statement_l1537_153705


namespace NUMINAMATH_CALUDE_max_area_is_eight_l1537_153737

/-- A line in the form kx - y + 2 = 0 -/
structure Line where
  k : ℝ

/-- A circle in the form x^2 + y^2 - 4x - 12 = 0 -/
def Circle : Type := Unit

/-- Points of intersection between the line and the circle -/
structure Intersection where
  Q : ℝ × ℝ
  R : ℝ × ℝ

/-- The maximum area of triangle QRC given a line and a circle -/
def max_area (l : Line) (C : Circle) (i : Intersection) : ℝ := 8

/-- Theorem stating that the maximum area of triangle QRC is 8 -/
theorem max_area_is_eight (l : Line) (C : Circle) (i : Intersection) :
  max_area l C i = 8 := by sorry

end NUMINAMATH_CALUDE_max_area_is_eight_l1537_153737


namespace NUMINAMATH_CALUDE_swimmers_pass_178_times_l1537_153752

/-- Represents a swimmer in the pool --/
structure Swimmer where
  speed : ℝ
  startPosition : ℝ

/-- Represents the swimming pool scenario --/
structure PoolScenario where
  poolLength : ℝ
  swimmer1 : Swimmer
  swimmer2 : Swimmer
  totalTime : ℝ

/-- Calculates the number of times swimmers pass each other --/
def calculatePassings (scenario : PoolScenario) : ℕ :=
  sorry

/-- The specific pool scenario from the problem --/
def problemScenario : PoolScenario :=
  { poolLength := 100
    swimmer1 := { speed := 4, startPosition := 0 }
    swimmer2 := { speed := 3, startPosition := 100 }
    totalTime := 20 * 60 }  -- 20 minutes in seconds

theorem swimmers_pass_178_times :
  calculatePassings problemScenario = 178 :=
sorry

end NUMINAMATH_CALUDE_swimmers_pass_178_times_l1537_153752


namespace NUMINAMATH_CALUDE_inequality_condition_max_area_ellipse_l1537_153740

-- Define the line l: y = k(x+1)
def line_l (k : ℝ) (x : ℝ) : ℝ := k * (x + 1)

-- Define the ellipse: x^2 + 4y^2 = a^2
def ellipse (a : ℝ) (x y : ℝ) : Prop := x^2 + 4*y^2 = a^2

-- Define the intersection points A and B
def intersection_points (k a : ℝ) : Prop :=
  ∃ x1 y1 x2 y2 : ℝ,
    ellipse a x1 y1 ∧ y1 = line_l k x1 ∧
    ellipse a x2 y2 ∧ y2 = line_l k x2 ∧
    (x1 ≠ x2 ∨ y1 ≠ y2)

-- Define point C as the intersection of line l with x-axis
def point_c (k : ℝ) : ℝ := -1

-- Define the condition AC = 2CB
def ac_twice_cb (k a : ℝ) : Prop :=
  ∃ x1 y1 x2 y2 : ℝ,
    ellipse a x1 y1 ∧ y1 = line_l k x1 ∧
    ellipse a x2 y2 ∧ y2 = line_l k x2 ∧
    (x1 - point_c k) = 2 * (point_c k - x2)

-- Theorem 1: a^2 > 4k^2 / (1+k^2)
theorem inequality_condition (k a : ℝ) (h1 : a > 0) (h2 : intersection_points k a) :
  a^2 > 4*k^2 / (1 + k^2) := by sorry

-- Theorem 2: When the area of triangle OAB is maximized, the equation of the ellipse is x^2 + 4y^2 = 5
theorem max_area_ellipse (k a : ℝ) (h1 : a > 0) (h2 : intersection_points k a) (h3 : ac_twice_cb k a) :
  (∀ x y : ℝ, ellipse a x y ↔ x^2 + 4*y^2 = 5) := by sorry

end NUMINAMATH_CALUDE_inequality_condition_max_area_ellipse_l1537_153740


namespace NUMINAMATH_CALUDE_cone_cylinder_volume_ratio_l1537_153780

theorem cone_cylinder_volume_ratio (r h : ℝ) (hr : r > 0) (hh : h > 0) : 
  (1 / 3 * π * r^2 * (h / 3)) / (π * r^2 * h) = 1 / 9 := by
  sorry

end NUMINAMATH_CALUDE_cone_cylinder_volume_ratio_l1537_153780


namespace NUMINAMATH_CALUDE_some_students_not_club_members_l1537_153782

-- Define the universe
variable (U : Type)

-- Define predicates
variable (Student : U → Prop)
variable (StudyScience : U → Prop)
variable (ClubMember : U → Prop)
variable (Honest : U → Prop)

-- Define the conditions
variable (h1 : ∃ x, Student x ∧ ¬StudyScience x)
variable (h2 : ∀ x, ClubMember x → (StudyScience x ∧ Honest x))

-- State the theorem
theorem some_students_not_club_members :
  ∃ x, Student x ∧ ¬ClubMember x :=
sorry

end NUMINAMATH_CALUDE_some_students_not_club_members_l1537_153782


namespace NUMINAMATH_CALUDE_fourth_tea_price_theorem_l1537_153767

/-- Calculates the price of the fourth tea variety given the prices of three varieties,
    their mixing ratios, and the final mixture price. -/
def fourth_tea_price (p1 p2 p3 mix_price : ℚ) : ℚ :=
  let r1 : ℚ := 2
  let r2 : ℚ := 3
  let r3 : ℚ := 4
  let r4 : ℚ := 5
  let total_ratio : ℚ := r1 + r2 + r3 + r4
  (mix_price * total_ratio - (p1 * r1 + p2 * r2 + p3 * r3)) / r4

/-- Theorem stating that given the prices of three tea varieties, their mixing ratios,
    and the final mixture price, the price of the fourth variety is 205.8. -/
theorem fourth_tea_price_theorem (p1 p2 p3 mix_price : ℚ) 
  (h1 : p1 = 126) (h2 : p2 = 135) (h3 : p3 = 156) (h4 : mix_price = 165) :
  fourth_tea_price p1 p2 p3 mix_price = 205.8 := by
  sorry

#eval fourth_tea_price 126 135 156 165

end NUMINAMATH_CALUDE_fourth_tea_price_theorem_l1537_153767


namespace NUMINAMATH_CALUDE_range_of_a_l1537_153792

theorem range_of_a (a : ℝ) : 
  (∃ x₀ : ℝ, x₀ ∈ [0, 1] ∧ 
    x₀ + (Real.exp 2 - 1) * Real.log a ≥ (2 * a / Real.exp x₀) + Real.exp 2 * x₀ - 2) →
  a ∈ Set.Icc 1 (Real.exp 3) :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l1537_153792
