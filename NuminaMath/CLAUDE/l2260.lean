import Mathlib

namespace jack_needs_more_money_l2260_226017

def sock_price : ℝ := 9.50
def shoe_price : ℝ := 92
def jack_money : ℝ := 40
def num_socks : ℕ := 2

theorem jack_needs_more_money :
  let total_cost := num_socks * sock_price + shoe_price
  total_cost - jack_money = 71 := by sorry

end jack_needs_more_money_l2260_226017


namespace sum_of_smallest_multiples_l2260_226050

def smallest_two_digit_multiple_of_3 : ℕ → Prop :=
  λ n => n ≥ 10 ∧ n < 100 ∧ 3 ∣ n ∧ ∀ m, m ≥ 10 ∧ m < 100 ∧ 3 ∣ m → n ≤ m

def smallest_three_digit_multiple_of_4 : ℕ → Prop :=
  λ n => n ≥ 100 ∧ n < 1000 ∧ 4 ∣ n ∧ ∀ m, m ≥ 100 ∧ m < 1000 ∧ 4 ∣ m → n ≤ m

theorem sum_of_smallest_multiples : 
  ∀ a b : ℕ, smallest_two_digit_multiple_of_3 a → smallest_three_digit_multiple_of_4 b → 
  a + b = 112 := by
sorry

end sum_of_smallest_multiples_l2260_226050


namespace locus_of_P_l2260_226021

-- Define the lines l₁ and l₂
def l₁ (x y : ℝ) : Prop := x - 2*y - 1 = 0
def l₂ (x y : ℝ) : Prop := 2*x - y - 2 = 0

-- Define point Q
def Q : ℝ × ℝ := (2, -1)

-- Define the condition for a point P to be on the locus
def on_locus (P : ℝ × ℝ) : Prop :=
  P.1 - P.2 - 1 = 0 ∧ P ≠ (3, 4) ∧ P ≠ (-2, -3) ∧ P ≠ (1, 0)

-- State the theorem
theorem locus_of_P (P A B : ℝ × ℝ) :
  l₁ A.1 A.2 →
  l₂ B.1 B.2 →
  (∃ (t : ℝ), P = (1 - t) • A + t • B) →
  P ≠ Q →
  (P.1 - A.1) / (B.1 - P.1) = (Q.1 - A.1) / (B.1 - Q.1) →
  (P.2 - A.2) / (B.2 - P.2) = (Q.2 - A.2) / (B.2 - Q.2) →
  on_locus P :=
sorry

end locus_of_P_l2260_226021


namespace equation_solutions_l2260_226030

theorem equation_solutions :
  (∃ x₁ x₂ : ℝ, (x₁ = 1 ∧ x₂ = 4) ∧ 
    (∀ x : ℝ, (x - 1)^2 = 3*(x - 1) ↔ x = x₁ ∨ x = x₂)) ∧
  (∃ y₁ y₂ : ℝ, (y₁ = 2 + Real.sqrt 3 ∧ y₂ = 2 - Real.sqrt 3) ∧ 
    (∀ x : ℝ, x^2 - 4*x + 1 = 0 ↔ x = y₁ ∨ x = y₂)) :=
by sorry

end equation_solutions_l2260_226030


namespace even_mono_increasing_inequality_l2260_226053

/-- A function is even if f(-x) = f(x) for all x -/
def IsEven (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

/-- A function is monotonically increasing on [0, +∞) if f(x) ≤ f(y) for 0 ≤ x ≤ y -/
def MonoIncreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, 0 ≤ x → x ≤ y → f x ≤ f y

theorem even_mono_increasing_inequality (f : ℝ → ℝ) 
    (h_even : IsEven f) (h_mono : MonoIncreasing f) : 
    f (-2) > f 1 ∧ f 1 > f 0 := by
  sorry

end even_mono_increasing_inequality_l2260_226053


namespace hyperbola_eccentricity_l2260_226038

/-- The eccentricity of a hyperbola with the given conditions -/
theorem hyperbola_eccentricity (k a b : ℝ) (h1 : a > 0) (h2 : b > 0) : 
  (∃ (c : ℝ), c > 0 ∧ 
    (∀ (x y : ℝ), k * x + y - Real.sqrt 2 * k = 0 → 
      x^2 / a^2 - y^2 / b^2 = 1 → 
      (∃ (m : ℝ), y = m * x ∧ abs (Real.sqrt 2 * k / Real.sqrt (1 + k^2)) = 4/3))) → 
  Real.sqrt (1 + b^2 / a^2) = 3 :=
by sorry

end hyperbola_eccentricity_l2260_226038


namespace boxes_sold_theorem_l2260_226059

/-- Represents the number of boxes sold on each day --/
structure BoxesSold where
  friday : ℕ
  saturday : ℕ
  sunday : ℕ

/-- Calculates the total number of boxes sold over three days --/
def totalBoxesSold (boxes : BoxesSold) : ℕ :=
  boxes.friday + boxes.saturday + boxes.sunday

/-- Theorem stating the total number of boxes sold over three days --/
theorem boxes_sold_theorem (boxes : BoxesSold) 
  (h1 : boxes.friday = 40)
  (h2 : boxes.saturday = 2 * boxes.friday - 10)
  (h3 : boxes.sunday = boxes.saturday / 2) :
  totalBoxesSold boxes = 145 := by
  sorry

#check boxes_sold_theorem

end boxes_sold_theorem_l2260_226059


namespace find_number_l2260_226011

theorem find_number (x : ℤ) : 
  (∃ q r : ℤ, 5 * (x + 3) = 8 * q + r ∧ q = 156 ∧ r = 2) → x = 247 := by
sorry

end find_number_l2260_226011


namespace tetrahedron_passage_l2260_226025

/-- The minimal radius through which a regular tetrahedron with edge length 1 can pass -/
def min_radius : ℝ := 0.4478

/-- A regular tetrahedron with edge length 1 -/
structure RegularTetrahedron where
  edge_length : ℝ
  is_regular : edge_length = 1

/-- A circular hole -/
structure CircularHole where
  radius : ℝ

/-- Predicate for whether a tetrahedron can pass through a hole -/
def can_pass_through (t : RegularTetrahedron) (h : CircularHole) : Prop :=
  h.radius ≥ min_radius

/-- Theorem stating the condition for a regular tetrahedron to pass through a circular hole -/
theorem tetrahedron_passage (t : RegularTetrahedron) (h : CircularHole) :
  can_pass_through t h ↔ h.radius ≥ min_radius :=
sorry

end tetrahedron_passage_l2260_226025


namespace roots_reciprocal_sum_squared_l2260_226034

theorem roots_reciprocal_sum_squared (a b c r s : ℝ) (ha : a ≠ 0) (hc : c ≠ 0)
  (hr : a * r^2 + b * r + c = 0) (hs : a * s^2 + b * s + c = 0) :
  1 / r^2 + 1 / s^2 = (b^2 - 2*a*c) / c^2 := by
  sorry

end roots_reciprocal_sum_squared_l2260_226034


namespace double_iced_cubes_count_l2260_226033

/-- Represents a cube cake with icing -/
structure IcedCake where
  size : Nat
  has_top_icing : Bool
  has_side_icing : Bool
  middle_icing_height : Rat

/-- Counts cubes with exactly two iced sides in an iced cake -/
def count_double_iced_cubes (cake : IcedCake) : Nat :=
  sorry

/-- The main theorem to be proved -/
theorem double_iced_cubes_count (cake : IcedCake) : 
  cake.size = 5 ∧ 
  cake.has_top_icing = true ∧ 
  cake.has_side_icing = true ∧ 
  cake.middle_icing_height = 5/2 →
  count_double_iced_cubes cake = 72 :=
by sorry

end double_iced_cubes_count_l2260_226033


namespace opposite_of_negative_five_l2260_226068

theorem opposite_of_negative_five :
  ∀ x : ℤ, ((-5 : ℤ) + x = 0) → x = 5 := by
sorry

end opposite_of_negative_five_l2260_226068


namespace cube_root_squared_times_fifth_root_l2260_226035

theorem cube_root_squared_times_fifth_root (x : ℝ) (h : x > 0) :
  (x^(1/3))^2 * x^(1/5) = x^(13/15) := by
  sorry

end cube_root_squared_times_fifth_root_l2260_226035


namespace pentagon_area_l2260_226060

theorem pentagon_area (a b c d e : ℝ) (h₁ : a = 18) (h₂ : b = 25) (h₃ : c = 30) (h₄ : d = 28) (h₅ : e = 25)
  (h₆ : a * b / 2 + (b + c) * d / 2 = 995) : 
  ∃ (pentagon_area : ℝ), pentagon_area = 995 := by
  sorry

end pentagon_area_l2260_226060


namespace basketball_team_combinations_l2260_226090

theorem basketball_team_combinations :
  let total_players : ℕ := 15
  let team_size : ℕ := 6
  let must_include : ℕ := 2
  let remaining_slots : ℕ := team_size - must_include
  let remaining_players : ℕ := total_players - must_include
  Nat.choose remaining_players remaining_slots = 715 :=
by sorry

end basketball_team_combinations_l2260_226090


namespace max_value_complex_expression_l2260_226004

theorem max_value_complex_expression (z : ℂ) (h : Complex.abs z = 1) :
  Complex.abs (z^3 - 3*z + 2) ≤ 3 * Real.sqrt 3 :=
by sorry

end max_value_complex_expression_l2260_226004


namespace correct_committee_count_l2260_226095

/-- Represents a department in the division of mathematical sciences --/
inductive Department
| Mathematics
| Statistics
| ComputerScience

/-- Represents the gender of a professor --/
inductive Gender
| Male
| Female

/-- Represents the composition of professors in a department --/
structure DepartmentComposition where
  department : Department
  maleCount : Nat
  femaleCount : Nat

/-- Represents the requirements for forming a committee --/
structure CommitteeRequirements where
  totalSize : Nat
  femaleCount : Nat
  maleCount : Nat
  mathDepartmentCount : Nat
  minDepartmentsRepresented : Nat

def divisionComposition : List DepartmentComposition := [
  ⟨Department.Mathematics, 3, 3⟩,
  ⟨Department.Statistics, 2, 3⟩,
  ⟨Department.ComputerScience, 2, 3⟩
]

def committeeReqs : CommitteeRequirements := {
  totalSize := 7,
  femaleCount := 4,
  maleCount := 3,
  mathDepartmentCount := 2,
  minDepartmentsRepresented := 3
}

/-- Calculates the number of possible committees given the division composition and requirements --/
def countPossibleCommittees (composition : List DepartmentComposition) (reqs : CommitteeRequirements) : Nat :=
  sorry

theorem correct_committee_count :
  countPossibleCommittees divisionComposition committeeReqs = 1050 :=
sorry

end correct_committee_count_l2260_226095


namespace intersection_limit_l2260_226073

noncomputable def L (m : ℝ) : ℝ := -Real.sqrt (m + 8)

theorem intersection_limit :
  ∀ ε > 0, ∃ δ > 0, ∀ m : ℝ, 
    0 < |m| ∧ |m| < δ ∧ -8 < m ∧ m < 8 → 
    |((L (-m) - L m) / m) - 1 / (2 * Real.sqrt 2)| < ε := by
  sorry

end intersection_limit_l2260_226073


namespace smallest_unique_sum_l2260_226083

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

end smallest_unique_sum_l2260_226083


namespace count_valid_words_l2260_226041

/-- The number of letters in each word -/
def word_length : ℕ := 4

/-- The number of available letters -/
def alphabet_size : ℕ := 5

/-- The number of letters that must be included -/
def required_letters : ℕ := 2

/-- The number of 4-letter words that can be formed using the letters A, B, C, D, and E, 
    with repetition allowed, and including both A and E at least once -/
def valid_words : ℕ := alphabet_size^word_length - 2*(alphabet_size-1)^word_length + (alphabet_size-2)^word_length

theorem count_valid_words : valid_words = 194 := by sorry

end count_valid_words_l2260_226041


namespace molecular_weight_calculation_molecular_weight_proof_l2260_226076

/-- Given the molecular weight of 10 moles of a substance, 
    calculate the molecular weight of x moles of the same substance. -/
theorem molecular_weight_calculation 
  (mw_10_moles : ℝ)  -- molecular weight of 10 moles
  (x : ℝ)            -- number of moles we want to calculate
  (h : mw_10_moles > 0) -- assumption that molecular weight is positive
  : ℝ :=
  (mw_10_moles / 10) * x

/-- Prove that the molecular weight calculation is correct -/
theorem molecular_weight_proof 
  (mw_10_moles : ℝ)  -- molecular weight of 10 moles
  (x : ℝ)            -- number of moles we want to calculate
  (h : mw_10_moles > 0) -- assumption that molecular weight is positive
  : molecular_weight_calculation mw_10_moles x h = (mw_10_moles / 10) * x :=
by sorry

end molecular_weight_calculation_molecular_weight_proof_l2260_226076


namespace benny_seashells_l2260_226015

/-- Represents the number of seashells Benny has after giving some to Jason -/
def seashells_remaining (initial : ℕ) (given_away : ℕ) : ℕ :=
  initial - given_away

/-- Proves that Benny has 14 seashells remaining -/
theorem benny_seashells : seashells_remaining 66 52 = 14 := by
  sorry

end benny_seashells_l2260_226015


namespace circle_op_inequality_l2260_226045

def circle_op (x y : ℝ) : ℝ := x * (1 - y)

theorem circle_op_inequality (a : ℝ) : 
  (∀ x : ℝ, circle_op (x - a) (x + a) < 1) → -1/2 < a ∧ a < 3/2 := by
  sorry

end circle_op_inequality_l2260_226045


namespace marked_price_calculation_l2260_226063

/-- Given a pair of articles bought for $50 with a 30% discount, 
    prove that the marked price of each article is 50 / 1.4 -/
theorem marked_price_calculation (total_price : ℝ) (discount_percent : ℝ) 
    (h1 : total_price = 50)
    (h2 : discount_percent = 30) : 
  (total_price / (2 * (1 - discount_percent / 100))) = 50 / 1.4 := by
  sorry

#eval (50 : Float) / 1.4

end marked_price_calculation_l2260_226063


namespace smallest_x_for_cube_l2260_226027

def is_perfect_cube (n : ℕ) : Prop := ∃ m : ℕ, n = m^3

theorem smallest_x_for_cube : 
  (∀ y : ℕ, y > 0 ∧ y < 7350 → ¬ is_perfect_cube (1260 * y)) ∧ 
  is_perfect_cube (1260 * 7350) := by
  sorry

end smallest_x_for_cube_l2260_226027


namespace vector_sum_and_scalar_mult_l2260_226043

/-- Prove that the sum of the vector (3, -2, 5) and 2 times the vector (-1, 4, -3) is equal to the vector (1, 6, -1). -/
theorem vector_sum_and_scalar_mult :
  let v₁ : Fin 3 → ℝ := ![3, -2, 5]
  let v₂ : Fin 3 → ℝ := ![-1, 4, -3]
  v₁ + 2 • v₂ = ![1, 6, -1] := by
  sorry

end vector_sum_and_scalar_mult_l2260_226043


namespace outfit_combinations_l2260_226066

def number_of_shirts : ℕ := 5
def number_of_pants : ℕ := 4
def number_of_hats : ℕ := 2

theorem outfit_combinations : 
  number_of_shirts * number_of_pants * number_of_hats = 40 := by
  sorry

end outfit_combinations_l2260_226066


namespace cube_volume_surface_area_l2260_226080

theorem cube_volume_surface_area (x : ℝ) : 
  (∃ s : ℝ, s > 0 ∧ s^3 = 8*x ∧ 6*s^2 = x) → x = 13824 := by
  sorry

end cube_volume_surface_area_l2260_226080


namespace democrat_ratio_is_one_third_l2260_226098

/-- Represents the number of participants in each category -/
structure Participants where
  total : ℕ
  female : ℕ
  male : ℕ
  femaleDemocrats : ℕ
  maleDemocrats : ℕ

/-- The ratio of democrats to total participants -/
def democratRatio (p : Participants) : ℚ :=
  (p.femaleDemocrats + p.maleDemocrats : ℚ) / p.total

theorem democrat_ratio_is_one_third (p : Participants) 
  (h1 : p.total = 660)
  (h2 : p.female + p.male = p.total)
  (h3 : p.femaleDemocrats = p.female / 2)
  (h4 : p.maleDemocrats = p.male / 4)
  (h5 : p.femaleDemocrats = 110) :
  democratRatio p = 1 / 3 := by
  sorry

end democrat_ratio_is_one_third_l2260_226098


namespace sin_2alpha_value_l2260_226049

theorem sin_2alpha_value (α : Real) 
  (h1 : π / 2 < α ∧ α < π) 
  (h2 : 3 * Real.cos (2 * α) = Real.cos (π / 4 + α)) : 
  Real.sin (2 * α) = -17 / 18 := by
  sorry

end sin_2alpha_value_l2260_226049


namespace f_properties_l2260_226082

noncomputable def f (x : ℝ) : ℝ := 1 + x - x^3

theorem f_properties :
  (∃! (a b : ℝ), a ≠ b ∧ (deriv f a = 0 ∧ deriv f b = 0) ∧
    ∀ x, deriv f x = 0 → (x = a ∨ x = b)) ∧
  (∃! (a b : ℝ), deriv f a = 0 ∧ deriv f b = 0 ∧ a + b = 0) ∧
  (∃! x, f x = 0) ∧
  (¬∃ x, f x = -x ∧ deriv f x = -1) :=
sorry

end f_properties_l2260_226082


namespace locker_count_proof_l2260_226000

/-- The cost of each digit in cents -/
def digit_cost : ℚ := 3

/-- The total cost of labeling all lockers in dollars -/
def total_cost : ℚ := 771.90

/-- The number of lockers -/
def num_lockers : ℕ := 6369

/-- The cost of labeling lockers from 1 to n -/
def labeling_cost (n : ℕ) : ℚ :=
  let one_digit := (min n 9 : ℚ) * digit_cost / 100
  let two_digit := (min n 99 - min n 9 : ℚ) * 2 * digit_cost / 100
  let three_digit := (min n 999 - min n 99 : ℚ) * 3 * digit_cost / 100
  let four_digit := (min n 9999 - min n 999 : ℚ) * 4 * digit_cost / 100
  let five_digit := (n - min n 9999 : ℚ) * 5 * digit_cost / 100
  one_digit + two_digit + three_digit + four_digit + five_digit

theorem locker_count_proof :
  labeling_cost num_lockers = total_cost := by
  sorry

end locker_count_proof_l2260_226000


namespace tailor_time_ratio_l2260_226018

theorem tailor_time_ratio (num_shirts : ℕ) (num_pants : ℕ) (shirt_time : ℚ) 
  (hourly_rate : ℚ) (total_cost : ℚ) :
  num_shirts = 10 →
  num_pants = 12 →
  shirt_time = 3/2 →
  hourly_rate = 30 →
  total_cost = 1530 →
  ∃ (pants_time : ℚ), 
    pants_time / shirt_time = 2 ∧
    total_cost = hourly_rate * (num_shirts * shirt_time + num_pants * pants_time) :=
by sorry

end tailor_time_ratio_l2260_226018


namespace square_field_side_length_l2260_226077

theorem square_field_side_length (area : ℝ) (side_length : ℝ) :
  area = 100 ∧ area = side_length ^ 2 → side_length = 10 := by
  sorry

end square_field_side_length_l2260_226077


namespace range_of_a_l2260_226001

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, 2 * x^2 + (a - 1) * x + 1/2 > 0) → 
  -1 < a ∧ a < 3 := by
sorry

end range_of_a_l2260_226001


namespace midpoint_sum_coordinates_l2260_226081

/-- The sum of the coordinates of the midpoint of a line segment with endpoints (2, 3) and (8, 15) is 14. -/
theorem midpoint_sum_coordinates : 
  let x₁ : ℝ := 2
  let y₁ : ℝ := 3
  let x₂ : ℝ := 8
  let y₂ : ℝ := 15
  let midpoint_x : ℝ := (x₁ + x₂) / 2
  let midpoint_y : ℝ := (y₁ + y₂) / 2
  midpoint_x + midpoint_y = 14 := by
  sorry


end midpoint_sum_coordinates_l2260_226081


namespace pole_area_after_cuts_l2260_226075

/-- The area of a rectangular pole after two cuts -/
theorem pole_area_after_cuts (original_length original_width : ℝ)
  (length_cut_percentage width_cut_percentage : ℝ) :
  original_length = 20 →
  original_width = 2 →
  length_cut_percentage = 0.3 →
  width_cut_percentage = 0.25 →
  let new_length := original_length * (1 - length_cut_percentage)
  let new_width := original_width * (1 - width_cut_percentage)
  new_length * new_width = 21 := by
  sorry

end pole_area_after_cuts_l2260_226075


namespace range_of_a_l2260_226044

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, |x - 1| + |x + 1| ≥ 3 * a) →
  (∀ x y : ℝ, x < y → (2 * a - 1) ^ x > (2 * a - 1) ^ y) →
  1/2 < a ∧ a ≤ 2/3 := by
  sorry

end range_of_a_l2260_226044


namespace x_range_for_quartic_equation_l2260_226003

theorem x_range_for_quartic_equation (k x : ℝ) :
  x^4 - 2*k*x^2 + k^2 + 2*k - 3 = 0 → -Real.sqrt 2 ≤ x ∧ x ≤ Real.sqrt 2 := by
  sorry

end x_range_for_quartic_equation_l2260_226003


namespace sin_cos_difference_equals_sqrt_three_over_two_l2260_226072

theorem sin_cos_difference_equals_sqrt_three_over_two :
  Real.sin (5 * π / 180) * Real.cos (55 * π / 180) -
  Real.cos (175 * π / 180) * Real.sin (55 * π / 180) =
  Real.sqrt 3 / 2 := by
  sorry

end sin_cos_difference_equals_sqrt_three_over_two_l2260_226072


namespace mass_percentage_Cl_is_66_04_l2260_226096

/-- The mass percentage of Cl in a certain compound -/
def mass_percentage_Cl : ℝ := 66.04

/-- Theorem stating that the mass percentage of Cl is 66.04% -/
theorem mass_percentage_Cl_is_66_04 : mass_percentage_Cl = 66.04 := by
  sorry

end mass_percentage_Cl_is_66_04_l2260_226096


namespace rectangle_to_square_possible_l2260_226016

/-- Represents a rectangle with integer side lengths -/
structure Rectangle where
  length : ℕ
  width : ℕ

/-- Represents a square with integer side length -/
structure Square where
  side : ℕ

/-- Represents a rectangular piece cut from the original rectangle -/
structure Piece where
  length : ℕ
  width : ℕ

def Rectangle.area (r : Rectangle) : ℕ := r.length * r.width

def Square.area (s : Square) : ℕ := s.side * s.side

def can_form_square (r : Rectangle) (s : Square) (pieces : List Piece) : Prop :=
  r.area = s.area ∧
  (pieces.foldl (fun acc p => acc + p.length * p.width) 0 = r.area) ∧
  (∀ p ∈ pieces, p.length ≤ r.length ∧ p.width ≤ r.width)

theorem rectangle_to_square_possible (r : Rectangle) (h1 : r.length = 16) (h2 : r.width = 9) :
  ∃ (s : Square) (pieces : List Piece), can_form_square r s pieces ∧ pieces.length ≤ 2 := by
  sorry

end rectangle_to_square_possible_l2260_226016


namespace coin_toss_probability_l2260_226065

/-- The probability of a coin with diameter 1/2 not touching any lattice lines when tossed onto a 1x1 square -/
def coin_probability : ℚ := 1 / 4

/-- The diameter of the coin -/
def coin_diameter : ℚ := 1 / 2

/-- The side length of the square -/
def square_side : ℚ := 1

theorem coin_toss_probability :
  coin_probability = (square_side - coin_diameter)^2 / square_side^2 :=
by sorry

end coin_toss_probability_l2260_226065


namespace special_function_max_l2260_226042

open Real

/-- A continuous function satisfying the given properties -/
def SpecialFunction (f : ℝ → ℝ) : Prop :=
  Continuous f ∧
  (∀ x y, f (x + y) * f (x - y) = f x ^ 2 - f y ^ 2) ∧
  (∀ x, f (x + 2 * π) = f x) ∧
  (∀ a, 0 < a → a < 2 * π → ∃ x, f (x + a) ≠ f x)

/-- The main theorem to be proved -/
theorem special_function_max (f : ℝ → ℝ) (hf : SpecialFunction f) :
  ∀ x, |f (π / 2)| ≥ f x :=
sorry

end special_function_max_l2260_226042


namespace two_ice_cream_cones_cost_l2260_226069

/-- The cost of an ice cream cone in cents -/
def ice_cream_cost : ℕ := 99

/-- The number of ice cream cones -/
def num_cones : ℕ := 2

/-- Theorem: The cost of 2 ice cream cones is 198 cents -/
theorem two_ice_cream_cones_cost : 
  ice_cream_cost * num_cones = 198 := by
  sorry

end two_ice_cream_cones_cost_l2260_226069


namespace cards_per_layer_in_house_of_cards_l2260_226078

/-- Proves that given 16 decks of 52 cards each, and a house of cards with 32 layers
    where each layer has the same number of cards, the number of cards per layer is 26. -/
theorem cards_per_layer_in_house_of_cards 
  (num_decks : ℕ) 
  (cards_per_deck : ℕ) 
  (num_layers : ℕ) 
  (h1 : num_decks = 16) 
  (h2 : cards_per_deck = 52) 
  (h3 : num_layers = 32) : 
  (num_decks * cards_per_deck) / num_layers = 26 := by
  sorry

#eval (16 * 52) / 32  -- Expected output: 26

end cards_per_layer_in_house_of_cards_l2260_226078


namespace number_of_recitation_orders_l2260_226058

/-- The number of high school seniors --/
def total_students : ℕ := 7

/-- The number of students to be selected --/
def selected_students : ℕ := 4

/-- The number of special students (A, B, C) --/
def special_students : ℕ := 3

/-- Function to calculate the number of recitation orders --/
def recitation_orders : ℕ := sorry

/-- Theorem stating the number of recitation orders --/
theorem number_of_recitation_orders :
  recitation_orders = 768 := by sorry

end number_of_recitation_orders_l2260_226058


namespace min_sum_squares_l2260_226029

def S : Finset Int := {-8, -6, -4, -1, 1, 3, 5, 14}

theorem min_sum_squares (p q r s t u v w : Int) 
  (h_distinct : p ≠ q ∧ p ≠ r ∧ p ≠ s ∧ p ≠ t ∧ p ≠ u ∧ p ≠ v ∧ p ≠ w ∧
                q ≠ r ∧ q ≠ s ∧ q ≠ t ∧ q ≠ u ∧ q ≠ v ∧ q ≠ w ∧
                r ≠ s ∧ r ≠ t ∧ r ≠ u ∧ r ≠ v ∧ r ≠ w ∧
                s ≠ t ∧ s ≠ u ∧ s ≠ v ∧ s ≠ w ∧
                t ≠ u ∧ t ≠ v ∧ t ≠ w ∧
                u ≠ v ∧ u ≠ w ∧
                v ≠ w)
  (h_in_S : p ∈ S ∧ q ∈ S ∧ r ∈ S ∧ s ∈ S ∧ t ∈ S ∧ u ∈ S ∧ v ∈ S ∧ w ∈ S) :
  (p + q + r + s)^2 + (t + u + v + w)^2 ≥ 8 :=
sorry

end min_sum_squares_l2260_226029


namespace smaller_number_proof_l2260_226009

theorem smaller_number_proof (x y : ℝ) 
  (h1 : y = 71.99999999999999)
  (h2 : y - x = (1/3) * y) : 
  x = 48 := by
  sorry

end smaller_number_proof_l2260_226009


namespace num_divisors_360_eq_24_l2260_226012

/-- The number of positive divisors of 360 -/
def num_divisors_360 : ℕ := sorry

/-- Theorem stating that the number of positive divisors of 360 is 24 -/
theorem num_divisors_360_eq_24 : num_divisors_360 = 24 := by sorry

end num_divisors_360_eq_24_l2260_226012


namespace right_to_left_eval_equals_56_over_9_l2260_226094

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

end right_to_left_eval_equals_56_over_9_l2260_226094


namespace fraction_equality_l2260_226064

theorem fraction_equality (w z : ℝ) (h : (1/w + 1/z)/(1/w - 1/z) = 2023) :
  (w + z)/(w - z) = -1012 := by
sorry

end fraction_equality_l2260_226064


namespace attic_boxes_count_l2260_226089

/-- Represents the problem of arranging teacups in an attic --/
def TeacupArrangement (B : ℕ) : Prop :=
  let boxes_without_pans := B - 6
  let boxes_with_teacups := boxes_without_pans / 2
  let cups_per_box := 5 * 4
  let broken_cups := 2 * boxes_with_teacups
  let original_cups := cups_per_box * boxes_with_teacups
  original_cups = 180 + broken_cups

/-- Theorem stating that there are 26 boxes in the attic --/
theorem attic_boxes_count : ∃ B : ℕ, TeacupArrangement B ∧ B = 26 := by
  sorry

end attic_boxes_count_l2260_226089


namespace sum_of_repeating_decimals_l2260_226013

def repeating_decimal_1 : ℚ := 1 / 9
def repeating_decimal_2 : ℚ := 2 / 99
def repeating_decimal_3 : ℚ := 3 / 999

theorem sum_of_repeating_decimals :
  repeating_decimal_1 + repeating_decimal_2 + repeating_decimal_3 = 134 / 999 := by
  sorry

end sum_of_repeating_decimals_l2260_226013


namespace linos_shells_l2260_226079

/-- The number of shells Lino picked up -/
def shells_picked_up : ℝ := 324.0

/-- The number of shells Lino put back -/
def shells_put_back : ℝ := 292.00

/-- The number of shells Lino has in all -/
def shells_remaining : ℝ := shells_picked_up - shells_put_back

/-- Theorem stating that the number of shells Lino has in all is 32.0 -/
theorem linos_shells : shells_remaining = 32.0 := by sorry

end linos_shells_l2260_226079


namespace range_of_a_l2260_226061

theorem range_of_a (a : ℝ) : 
  (∀ x ∈ Set.Icc 1 2, x^2 - a ≥ 0) ∧ 
  (∃ x₀ : ℝ, x₀^2 + 2*a*x₀ + 2 - a = 0) → 
  a = 1 ∨ a ≤ -2 :=
by sorry

end range_of_a_l2260_226061


namespace bank_deposit_calculation_l2260_226028

theorem bank_deposit_calculation (initial_amount : ℝ) : 
  (0.20 * 0.25 * 0.30 * initial_amount = 750) → initial_amount = 50000 := by
  sorry

end bank_deposit_calculation_l2260_226028


namespace johns_age_l2260_226023

theorem johns_age (j d m : ℕ) 
  (h1 : j = d - 20)
  (h2 : j = m - 15)
  (h3 : j + d = 80)
  (h4 : m = d + 5) :
  j = 30 := by
  sorry

end johns_age_l2260_226023


namespace amp_composition_l2260_226085

-- Define the & operation
def amp (x : ℝ) : ℝ := 9 - x

-- Define the & operation
def amp_rev (x : ℝ) : ℝ := x - 9

-- Theorem statement
theorem amp_composition : amp_rev (amp 15) = -15 := by sorry

end amp_composition_l2260_226085


namespace mary_berry_spending_l2260_226091

theorem mary_berry_spending (total apples peaches : ℝ) (h1 : total = 34.72) (h2 : apples = 14.33) (h3 : peaches = 9.31) :
  total - (apples + peaches) = 11.08 := by
  sorry

end mary_berry_spending_l2260_226091


namespace equal_boy_girl_division_theorem_l2260_226008

/-- Represents a student arrangement as a list of integers, where 1 represents a boy and -1 represents a girl -/
def StudentArrangement := List Int

/-- Checks if a given arrangement can be divided into two parts with equal number of boys and girls -/
def canBeDivided (arrangement : StudentArrangement) : Bool :=
  sorry

/-- Counts the number of arrangements where division is impossible -/
def countImpossibleDivisions (n : Nat) : Nat :=
  sorry

/-- Counts the number of arrangements where exactly one division is possible -/
def countSingleDivisions (n : Nat) : Nat :=
  sorry

theorem equal_boy_girl_division_theorem (n : Nat) (h : n ≥ 2) :
  countSingleDivisions (2 * n) = 2 * countImpossibleDivisions (2 * n) :=
by
  sorry

end equal_boy_girl_division_theorem_l2260_226008


namespace equation_solution_l2260_226070

theorem equation_solution :
  ∃ x : ℚ, (0.05 * x + 0.12 * (30 + x) = 18) ∧ (x = 144 / 17) := by
  sorry

end equation_solution_l2260_226070


namespace valid_numbers_l2260_226071

def is_valid (n : ℕ+) : Prop :=
  ∀ a : ℕ+, (a ≤ 1 + Real.sqrt n.val) → (Nat.gcd a.val n.val = 1) →
    ∃ x : ℤ, (a.val : ℤ) ≡ x^2 [ZMOD n.val]

theorem valid_numbers : {n : ℕ+ | is_valid n} = {1, 2, 12} := by sorry

end valid_numbers_l2260_226071


namespace isosceles_triangle_perimeter_l2260_226039

-- Define the isosceles triangle
structure IsoscelesTriangle where
  base : ℝ
  leg : ℝ

-- Define the conditions
def triangle_conditions (t : IsoscelesTriangle) : Prop :=
  t.base = 4 ∧ t.leg^2 - 5*t.leg + 6 = 0

-- Define the perimeter
def perimeter (t : IsoscelesTriangle) : ℝ :=
  t.base + 2*t.leg

-- Theorem statement
theorem isosceles_triangle_perimeter :
  ∀ t : IsoscelesTriangle, triangle_conditions t → perimeter t = 10 := by
  sorry

end isosceles_triangle_perimeter_l2260_226039


namespace pencil_count_problem_l2260_226051

/-- The number of pencils in a drawer after a series of additions and removals. -/
def final_pencil_count (initial : ℕ) (sara_adds : ℕ) (john_adds : ℕ) (ben_removes : ℕ) : ℕ :=
  initial + sara_adds + john_adds - ben_removes

/-- Theorem stating that given the initial number of pencils and the changes made by Sara, John, and Ben, the final number of pencils is 245. -/
theorem pencil_count_problem : final_pencil_count 115 100 75 45 = 245 := by
  sorry

end pencil_count_problem_l2260_226051


namespace polynomial_simplification_l2260_226088

/-- The sum of two polynomials is equal to the simplified polynomial. -/
theorem polynomial_simplification (x : ℝ) :
  (12 * x^10 + 6 * x^9 + 3 * x^8) + (2 * x^11 + x^10 + 4 * x^9 + x^7 + 4 * x^4 + 7 * x + 9) =
  2 * x^11 + 13 * x^10 + 10 * x^9 + 3 * x^8 + x^7 + 4 * x^4 + 7 * x + 9 := by
  sorry

end polynomial_simplification_l2260_226088


namespace james_has_43_oreos_l2260_226040

/-- The number of Oreos James has -/
def james_oreos (jordan_oreos : ℕ) : ℕ := 4 * jordan_oreos + 7

/-- The total number of Oreos -/
def total_oreos : ℕ := 52

theorem james_has_43_oreos :
  ∃ (jordan_oreos : ℕ), 
    james_oreos jordan_oreos + jordan_oreos = total_oreos ∧
    james_oreos jordan_oreos = 43 := by
  sorry

end james_has_43_oreos_l2260_226040


namespace laura_friends_count_l2260_226024

def total_blocks : ℕ := 28
def blocks_per_friend : ℕ := 7

theorem laura_friends_count : total_blocks / blocks_per_friend = 4 := by
  sorry

end laura_friends_count_l2260_226024


namespace jean_burglary_charges_l2260_226055

/-- Represents the charges and sentences for Jean's case -/
structure CriminalCase where
  arson_counts : ℕ
  burglary_charges : ℕ
  petty_larceny_charges : ℕ
  arson_sentence : ℕ
  burglary_sentence : ℕ
  petty_larceny_sentence : ℕ
  total_sentence : ℕ

/-- Calculates the total sentence for a given criminal case -/
def total_sentence (case : CriminalCase) : ℕ :=
  case.arson_counts * case.arson_sentence +
  case.burglary_charges * case.burglary_sentence +
  case.petty_larceny_charges * case.petty_larceny_sentence

/-- Theorem stating that Jean's case has 2 burglary charges -/
theorem jean_burglary_charges :
  ∃ (case : CriminalCase),
    case.arson_counts = 3 ∧
    case.petty_larceny_charges = 6 * case.burglary_charges ∧
    case.arson_sentence = 36 ∧
    case.burglary_sentence = 18 ∧
    case.petty_larceny_sentence = case.burglary_sentence / 3 ∧
    total_sentence case = 216 ∧
    case.burglary_charges = 2 :=
sorry

end jean_burglary_charges_l2260_226055


namespace focus_coordinates_for_specific_ellipse_l2260_226031

/-- Represents an ellipse with its center and axis endpoints -/
structure Ellipse where
  center : ℝ × ℝ
  major_axis_endpoints : (ℝ × ℝ) × (ℝ × ℝ)
  minor_axis_endpoints : (ℝ × ℝ) × (ℝ × ℝ)

/-- Calculates the coordinates of the focus with greater x-coordinate for a given ellipse -/
def focus_with_greater_x (e : Ellipse) : ℝ × ℝ :=
  sorry

/-- Theorem stating that for the given ellipse, the focus with greater x-coordinate
    has coordinates (3.5 + √6/2, 0) -/
theorem focus_coordinates_for_specific_ellipse :
  let e : Ellipse := {
    center := (3.5, 0),
    major_axis_endpoints := ((0, 0), (7, 0)),
    minor_axis_endpoints := ((3.5, 2.5), (3.5, -2.5))
  }
  focus_with_greater_x e = (3.5 + Real.sqrt 6 / 2, 0) := by
  sorry

end focus_coordinates_for_specific_ellipse_l2260_226031


namespace circle_and_ngons_inequalities_l2260_226062

/-- Given a circle and two regular n-gons (one inscribed, one circumscribed),
    prove the relationships between their areas and perimeters. -/
theorem circle_and_ngons_inequalities 
  (n : ℕ) 
  (S : ℝ) 
  (L : ℝ) 
  (S₁ : ℝ) 
  (S₂ : ℝ) 
  (P₁ : ℝ) 
  (P₂ : ℝ) 
  (h_n : n ≥ 3) 
  (h_S : S > 0) 
  (h_L : L > 0) 
  (h_S₁ : S₁ > 0) 
  (h_S₂ : S₂ > 0) 
  (h_P₁ : P₁ > 0) 
  (h_P₂ : P₂ > 0) 
  (h_inscribed : S₁ < S) 
  (h_circumscribed : S₂ > S) : 
  (S^2 > S₁ * S₂) ∧ (L^2 < P₁ * P₂) := by
  sorry


end circle_and_ngons_inequalities_l2260_226062


namespace factor_expression_l2260_226052

theorem factor_expression (x : ℝ) : x * (x + 3) + 2 * (x + 3) = (x + 2) * (x + 3) := by
  sorry

end factor_expression_l2260_226052


namespace infinite_geometric_series_first_term_l2260_226048

theorem infinite_geometric_series_first_term 
  (r : ℚ) (S : ℚ) (h1 : r = -1/3) (h2 : S = 10) :
  let a := S * (1 - r)
  a = 40/3 := by sorry

end infinite_geometric_series_first_term_l2260_226048


namespace mango_buying_rate_l2260_226037

/-- Represents the rate at which mangoes are bought and sold -/
structure MangoRate where
  buy : ℚ  -- Buying rate (rupees per x mangoes)
  sell : ℚ  -- Selling rate (mangoes per rupee)

/-- Calculates the profit percentage given buying and selling rates -/
def profit_percentage (rate : MangoRate) : ℚ :=
  (rate.sell⁻¹ / rate.buy - 1) * 100

/-- Proves that the buying rate is 2 rupees for x mangoes given the conditions -/
theorem mango_buying_rate (rate : MangoRate) 
  (h_sell : rate.sell = 3)
  (h_profit : profit_percentage rate = 50) :
  rate.buy = 2 := by
  sorry

end mango_buying_rate_l2260_226037


namespace smallest_denominator_fraction_l2260_226010

-- Define the fraction type
structure Fraction where
  numerator : ℕ
  denominator : ℕ
  denom_pos : denominator > 0

-- Define the property of being in the open interval
def inOpenInterval (f : Fraction) : Prop :=
  47 / 245 < f.numerator / f.denominator ∧ f.numerator / f.denominator < 34 / 177

-- Define the property of having the smallest denominator
def hasSmallestDenominator (f : Fraction) : Prop :=
  ∀ g : Fraction, inOpenInterval g → f.denominator ≤ g.denominator

-- The main theorem
theorem smallest_denominator_fraction :
  ∃ f : Fraction, f.numerator = 19 ∧ f.denominator = 99 ∧
  inOpenInterval f ∧ hasSmallestDenominator f :=
sorry

end smallest_denominator_fraction_l2260_226010


namespace max_students_on_field_trip_l2260_226099

def budget : ℕ := 350
def bus_rental : ℕ := 100
def admission_cost : ℕ := 10

theorem max_students_on_field_trip : 
  (budget - bus_rental) / admission_cost = 25 := by
  sorry

end max_students_on_field_trip_l2260_226099


namespace helen_raisin_cookie_difference_l2260_226007

/-- The number of raisin cookies Helen baked yesterday -/
def raisin_cookies_yesterday : ℕ := 300

/-- The number of raisin cookies Helen baked today -/
def raisin_cookies_today : ℕ := 280

/-- The difference in raisin cookies baked between yesterday and today -/
def raisin_cookie_difference : ℕ := raisin_cookies_yesterday - raisin_cookies_today

theorem helen_raisin_cookie_difference : raisin_cookie_difference = 20 := by
  sorry

end helen_raisin_cookie_difference_l2260_226007


namespace exchange_rate_problem_l2260_226067

theorem exchange_rate_problem (x : ℕ) : 
  (8 * x / 5 : ℚ) - 80 = x →
  (x / 100 + (x % 100) / 10 + x % 10 : ℕ) = 7 := by
  sorry

end exchange_rate_problem_l2260_226067


namespace aquarium_visitors_l2260_226092

theorem aquarium_visitors (total : ℕ) (ill_percent : ℚ) (not_ill : ℕ) 
  (h1 : ill_percent = 40 / 100)
  (h2 : not_ill = 300)
  (h3 : (1 - ill_percent) * total = not_ill) : 
  total = 500 := by
  sorry

end aquarium_visitors_l2260_226092


namespace two_digit_numbers_with_special_property_l2260_226056

def is_valid_number (n : ℕ) : Prop :=
  10 ≤ n ∧ n ≤ 99 ∧
  n % 7 = 1 ∧
  (10 * (n % 10) + n / 10) % 7 = 1

theorem two_digit_numbers_with_special_property :
  {n : ℕ | is_valid_number n} = {22, 29, 92, 99} :=
by sorry

end two_digit_numbers_with_special_property_l2260_226056


namespace fill_pipe_fraction_l2260_226005

/-- Represents the fraction of a cistern that can be filled in a given time -/
def FractionFilled (time : ℝ) : ℝ := sorry

theorem fill_pipe_fraction :
  let fill_time : ℝ := 30
  let fraction := FractionFilled fill_time
  (∃ (f : ℝ), FractionFilled fill_time = f ∧ f * fill_time = fill_time) →
  fraction = 1 := by sorry

end fill_pipe_fraction_l2260_226005


namespace reader_collection_pages_l2260_226087

def book1_chapters : List Nat := [24, 32, 40, 20]
def book2_chapters : List Nat := [48, 52, 36]
def book3_chapters : List Nat := [16, 28, 44, 22, 34]

def total_pages (chapters : List Nat) : Nat :=
  chapters.sum

theorem reader_collection_pages :
  total_pages book1_chapters +
  total_pages book2_chapters +
  total_pages book3_chapters = 396 := by
  sorry

end reader_collection_pages_l2260_226087


namespace no_rational_roots_for_three_digit_prime_quadratic_l2260_226093

def is_three_digit_prime (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000 ∧ Nat.Prime n

def digits_of_three_digit_number (n : ℕ) : ℕ × ℕ × ℕ :=
  let a := n / 100
  let b := (n / 10) % 10
  let c := n % 10
  (a, b, c)

theorem no_rational_roots_for_three_digit_prime_quadratic :
  ∀ A : ℕ, is_three_digit_prime A →
    let (a, b, c) := digits_of_three_digit_number A
    ∀ x : ℚ, a * x^2 + b * x + c ≠ 0 :=
by sorry

end no_rational_roots_for_three_digit_prime_quadratic_l2260_226093


namespace teachers_on_field_trip_l2260_226002

theorem teachers_on_field_trip 
  (num_students : ℕ) 
  (student_ticket_cost : ℕ) 
  (teacher_ticket_cost : ℕ) 
  (total_cost : ℕ) :
  num_students = 12 →
  student_ticket_cost = 1 →
  teacher_ticket_cost = 3 →
  total_cost = 24 →
  ∃ (num_teachers : ℕ), 
    num_students * student_ticket_cost + num_teachers * teacher_ticket_cost = total_cost ∧
    num_teachers = 4 := by
sorry

end teachers_on_field_trip_l2260_226002


namespace arithmetic_sequence_unique_n_l2260_226019

/-- An arithmetic sequence with n terms, where a₁ is the first term and d is the common difference. -/
structure ArithmeticSequence where
  n : ℕ
  a₁ : ℚ
  d : ℚ

/-- The sum of the first k terms of an arithmetic sequence. -/
def sum_first_k (seq : ArithmeticSequence) (k : ℕ) : ℚ :=
  k * (2 * seq.a₁ + (k - 1) * seq.d) / 2

/-- The sum of the last k terms of an arithmetic sequence. -/
def sum_last_k (seq : ArithmeticSequence) (k : ℕ) : ℚ :=
  k * (2 * (seq.a₁ + (seq.n - k) * seq.d) + (k - 1) * seq.d) / 2

/-- The sum of all terms in an arithmetic sequence. -/
def sum_all (seq : ArithmeticSequence) : ℚ :=
  seq.n * (2 * seq.a₁ + (seq.n - 1) * seq.d) / 2

theorem arithmetic_sequence_unique_n (seq : ArithmeticSequence) :
  sum_first_k seq 4 = 40 →
  sum_last_k seq 4 = 80 →
  sum_all seq = 210 →
  seq.n = 14 := by
  sorry

end arithmetic_sequence_unique_n_l2260_226019


namespace product_72_difference_equals_sum_l2260_226032

theorem product_72_difference_equals_sum (P Q R S : ℕ+) : 
  P ≠ Q ∧ P ≠ R ∧ P ≠ S ∧ Q ≠ R ∧ Q ≠ S ∧ R ≠ S →
  P * Q = 72 →
  R * S = 72 →
  P - Q = R + S →
  P = 12 := by
sorry

end product_72_difference_equals_sum_l2260_226032


namespace curve_symmetry_condition_l2260_226014

/-- Given a curve y = (px + q) / (rx - s) with nonzero p, q, r, s,
    if y = -x is an axis of symmetry, then r + s = 0 -/
theorem curve_symmetry_condition (p q r s : ℝ) (hp : p ≠ 0) (hq : q ≠ 0) (hr : r ≠ 0) (hs : s ≠ 0) :
  (∀ x y : ℝ, y = (p * x + q) / (r * x - s) ↔ (-x) = (p * (-y) + q) / (r * (-y) - s)) →
  r + s = 0 :=
by sorry

end curve_symmetry_condition_l2260_226014


namespace triangle_median_altitude_equations_l2260_226006

/-- Triangle ABC with vertices A(-5, 0), B(4, -4), and C(0, 2) -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- The equation of a line in the form ax + by + c = 0 -/
structure LineEquation where
  a : ℝ
  b : ℝ
  c : ℝ

def triangle_ABC : Triangle :=
  { A := (-5, 0)
    B := (4, -4)
    C := (0, 2) }

/-- The equation of the line on which the median to side BC lies -/
def median_equation (t : Triangle) : LineEquation :=
  { a := 1
    b := 7
    c := 5 }

/-- The equation of the line on which the altitude from A to side BC lies -/
def altitude_equation (t : Triangle) : LineEquation :=
  { a := 2
    b := -3
    c := 10 }

theorem triangle_median_altitude_equations :
  (median_equation triangle_ABC).a = 1 ∧
  (median_equation triangle_ABC).b = 7 ∧
  (median_equation triangle_ABC).c = 5 ∧
  (altitude_equation triangle_ABC).a = 2 ∧
  (altitude_equation triangle_ABC).b = -3 ∧
  (altitude_equation triangle_ABC).c = 10 := by
  sorry

end triangle_median_altitude_equations_l2260_226006


namespace gianna_savings_l2260_226022

def total_savings : ℕ := 14235
def days_in_year : ℕ := 365
def daily_savings : ℚ := total_savings / days_in_year

theorem gianna_savings : daily_savings = 39 := by
  sorry

end gianna_savings_l2260_226022


namespace squares_end_same_digit_l2260_226086

theorem squares_end_same_digit (a b : ℤ) : 
  (a + b) % 10 = 0 → a^2 % 10 = b^2 % 10 := by
  sorry

end squares_end_same_digit_l2260_226086


namespace balloon_difference_is_two_l2260_226097

/-- The number of balloons Allan brought to the park -/
def allan_balloons : ℕ := 5

/-- The number of balloons Jake brought to the park -/
def jake_balloons : ℕ := 3

/-- The difference in the number of balloons between Allan and Jake -/
def balloon_difference : ℕ := allan_balloons - jake_balloons

theorem balloon_difference_is_two : balloon_difference = 2 := by sorry

end balloon_difference_is_two_l2260_226097


namespace two_candles_burning_time_l2260_226084

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

end two_candles_burning_time_l2260_226084


namespace kite_area_theorem_l2260_226036

/-- A symmetrical quadrilateral kite -/
structure Kite where
  base : ℝ
  height : ℝ

/-- The area of a kite -/
def kite_area (k : Kite) : ℝ := k.base * k.height

/-- Theorem: The area of a kite with base 35 and height 15 is 525 -/
theorem kite_area_theorem (k : Kite) (h1 : k.base = 35) (h2 : k.height = 15) :
  kite_area k = 525 := by
  sorry

#check kite_area_theorem

end kite_area_theorem_l2260_226036


namespace light_travel_distance_l2260_226074

/-- The distance light travels in one year, in miles -/
def light_year_distance : ℕ := 5870000000000

/-- The number of years we want to calculate the light travel distance for -/
def years : ℕ := 200

/-- Theorem stating that light travels 1174 × 10^12 miles in 200 years -/
theorem light_travel_distance : 
  (light_year_distance * years : ℚ) = 1174 * (10^12 : ℚ) := by
  sorry

end light_travel_distance_l2260_226074


namespace complex_number_quadrant_l2260_226020

theorem complex_number_quadrant : ∃ (z : ℂ), z = 2 / (1 - Complex.I) ∧ z.re > 0 ∧ z.im > 0 := by
  sorry

end complex_number_quadrant_l2260_226020


namespace part_one_part_two_l2260_226047

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x - a| + |x - 3|

-- Part 1
theorem part_one (a : ℝ) (h1 : a < 3) 
  (h2 : ∀ x, f a x ≥ 4 ↔ x ≤ 1/2 ∨ x ≥ 9/2) : 
  a = 2 := by sorry

-- Part 2
theorem part_two (a : ℝ) (h1 : a < 3) 
  (h2 : ∀ x, f a x + |x - 3| ≥ 1) : 
  a ≤ 2 := by sorry

end part_one_part_two_l2260_226047


namespace smallest_integer_with_gcd_lcm_constraint_l2260_226026

theorem smallest_integer_with_gcd_lcm_constraint (x : ℕ) (m n : ℕ) 
  (h1 : x > 0)
  (h2 : m = 30)
  (h3 : Nat.gcd m n = x + 3)
  (h4 : Nat.lcm m n = x * (x + 3)) :
  n ≥ 162 ∧ ∃ (x : ℕ), x > 0 ∧ 
    Nat.gcd 30 162 = x + 3 ∧ 
    Nat.lcm 30 162 = x * (x + 3) :=
by sorry

end smallest_integer_with_gcd_lcm_constraint_l2260_226026


namespace min_sum_absolute_values_l2260_226054

theorem min_sum_absolute_values (x : ℝ) : 
  |x + 3| + |x + 6| + |x + 7| + 2 ≥ 8 ∧ 
  ∃ y : ℝ, |y + 3| + |y + 6| + |y + 7| + 2 = 8 :=
sorry

end min_sum_absolute_values_l2260_226054


namespace trailing_zeros_30_factorial_l2260_226057

/-- The number of trailing zeros in n! -/
def trailingZeros (n : ℕ) : ℕ :=
  sorry

/-- Theorem: The number of trailing zeros in 30! is 7 -/
theorem trailing_zeros_30_factorial : trailingZeros 30 = 7 := by
  sorry

end trailing_zeros_30_factorial_l2260_226057


namespace smallest_lcm_with_gcd_five_l2260_226046

theorem smallest_lcm_with_gcd_five (k l : ℕ) : 
  1000 ≤ k ∧ k < 10000 ∧ 
  1000 ≤ l ∧ l < 10000 ∧ 
  Nat.gcd k l = 5 →
  201000 ≤ Nat.lcm k l ∧ 
  ∃ (k' l' : ℕ), 1000 ≤ k' ∧ k' < 10000 ∧ 
                 1000 ≤ l' ∧ l' < 10000 ∧ 
                 Nat.gcd k' l' = 5 ∧ 
                 Nat.lcm k' l' = 201000 :=
by sorry

end smallest_lcm_with_gcd_five_l2260_226046
