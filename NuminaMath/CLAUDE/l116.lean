import Mathlib

namespace NUMINAMATH_CALUDE_five_pointed_star_angle_sum_l116_11633

/-- An irregular five-pointed star -/
structure FivePointedStar where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ
  E : ℝ × ℝ

/-- The angles at the vertices of a five-pointed star -/
structure StarAngles where
  α : ℝ
  β : ℝ
  γ : ℝ
  δ : ℝ
  ε : ℝ

/-- The sum of angles in a five-pointed star is 180 degrees -/
theorem five_pointed_star_angle_sum (star : FivePointedStar) (angles : StarAngles) :
  angles.α + angles.β + angles.γ + angles.δ + angles.ε = 180 := by
  sorry

#check five_pointed_star_angle_sum

end NUMINAMATH_CALUDE_five_pointed_star_angle_sum_l116_11633


namespace NUMINAMATH_CALUDE_total_chips_bags_l116_11635

theorem total_chips_bags (total_bags : ℕ) (doritos_bags : ℕ) : 
  (4 * doritos_bags = total_bags) →  -- One quarter of the bags are Doritos
  (4 * 5 = doritos_bags) →           -- Doritos bags can be split into 4 equal piles with 5 bags in each
  total_bags = 80 :=                 -- Prove that the total number of bags is 80
by
  sorry

end NUMINAMATH_CALUDE_total_chips_bags_l116_11635


namespace NUMINAMATH_CALUDE_intersection_complement_equality_l116_11639

def U : Set ℕ := {1,2,3,4,5,6,7}
def A : Set ℕ := {2,4,6}
def B : Set ℕ := {1,3,5,7}

theorem intersection_complement_equality : A ∩ (U \ B) = {2,4,6} := by
  sorry

end NUMINAMATH_CALUDE_intersection_complement_equality_l116_11639


namespace NUMINAMATH_CALUDE_false_propositions_count_l116_11669

-- Define the original proposition
def original_prop (m n : ℝ) : Prop := m > -n → m^2 > n^2

-- Define the contrapositive
def contrapositive (m n : ℝ) : Prop := ¬(m^2 > n^2) → ¬(m > -n)

-- Define the inverse
def inverse (m n : ℝ) : Prop := m^2 > n^2 → m > -n

-- Define the negation
def negation (m n : ℝ) : Prop := ¬(m > -n → m^2 > n^2)

-- Theorem statement
theorem false_propositions_count :
  ∃ (m n : ℝ), (¬(original_prop m n) ∧ ¬(contrapositive m n) ∧ ¬(inverse m n) ∧ ¬(negation m n)) :=
by sorry

end NUMINAMATH_CALUDE_false_propositions_count_l116_11669


namespace NUMINAMATH_CALUDE_solution_set_x_abs_x_leq_one_l116_11655

theorem solution_set_x_abs_x_leq_one (x : ℝ) : x * |x| ≤ 1 ↔ x ≤ 1 := by sorry

end NUMINAMATH_CALUDE_solution_set_x_abs_x_leq_one_l116_11655


namespace NUMINAMATH_CALUDE_product_digit_sum_base7_l116_11651

/-- Converts a base-7 number to base-10 --/
def toBase10 (n : ℕ) : ℕ := sorry

/-- Converts a base-10 number to base-7 --/
def toBase7 (n : ℕ) : ℕ := sorry

/-- Calculates the sum of digits of a number in base-7 --/
def sumOfDigitsBase7 (n : ℕ) : ℕ := sorry

/-- The main theorem --/
theorem product_digit_sum_base7 :
  sumOfDigitsBase7 (toBase7 (toBase10 35 * toBase10 13)) = 11 := by sorry

end NUMINAMATH_CALUDE_product_digit_sum_base7_l116_11651


namespace NUMINAMATH_CALUDE_tax_discount_commute_petes_equals_pollys_l116_11600

/-- Proves that the order of applying tax and discount doesn't affect the final price -/
theorem tax_discount_commute (price : ℝ) (tax_rate discount_rate : ℝ) 
  (h1 : 0 ≤ tax_rate) (h2 : 0 ≤ discount_rate) (h3 : discount_rate ≤ 1) :
  price * (1 + tax_rate) * (1 - discount_rate) = price * (1 - discount_rate) * (1 + tax_rate) :=
by sorry

/-- Calculates Pete's method: tax then discount -/
def petes_method (price : ℝ) (tax_rate discount_rate : ℝ) : ℝ :=
  price * (1 + tax_rate) * (1 - discount_rate)

/-- Calculates Polly's method: discount then tax -/
def pollys_method (price : ℝ) (tax_rate discount_rate : ℝ) : ℝ :=
  price * (1 - discount_rate) * (1 + tax_rate)

/-- Proves that Pete's and Polly's methods yield the same result -/
theorem petes_equals_pollys (price : ℝ) (tax_rate discount_rate : ℝ) 
  (h1 : 0 ≤ tax_rate) (h2 : 0 ≤ discount_rate) (h3 : discount_rate ≤ 1) :
  petes_method price tax_rate discount_rate = pollys_method price tax_rate discount_rate :=
by sorry

end NUMINAMATH_CALUDE_tax_discount_commute_petes_equals_pollys_l116_11600


namespace NUMINAMATH_CALUDE_min_value_of_f_l116_11603

-- Define the expression as a function of x
def f (x : ℝ) : ℝ := (15 - x) * (8 - x) * (15 + x) * (8 + x) + 200

-- State the theorem
theorem min_value_of_f :
  ∀ x : ℝ, f x ≥ -6290.25 ∧ ∃ y : ℝ, f y = -6290.25 := by sorry

end NUMINAMATH_CALUDE_min_value_of_f_l116_11603


namespace NUMINAMATH_CALUDE_largest_n_satisfying_property_l116_11649

/-- A function that checks if a number is prime -/
def isPrime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → n % m ≠ 0

/-- A function that checks if a number is an odd prime -/
def isOddPrime (p : ℕ) : Prop := isPrime p ∧ p % 2 ≠ 0

/-- The property that n satisfies: for any odd prime p < n, n - p is prime -/
def satisfiesProperty (n : ℕ) : Prop :=
  ∀ p : ℕ, p < n → isOddPrime p → isPrime (n - p)

theorem largest_n_satisfying_property :
  (satisfiesProperty 10) ∧ 
  (∀ m : ℕ, m > 10 → ¬(satisfiesProperty m)) :=
sorry

end NUMINAMATH_CALUDE_largest_n_satisfying_property_l116_11649


namespace NUMINAMATH_CALUDE_mortgage_duration_l116_11614

theorem mortgage_duration (house_price deposit monthly_payment : ℕ) :
  house_price = 280000 →
  deposit = 40000 →
  monthly_payment = 2000 →
  (house_price - deposit) / monthly_payment / 12 = 10 := by
  sorry

end NUMINAMATH_CALUDE_mortgage_duration_l116_11614


namespace NUMINAMATH_CALUDE_smallest_solution_congruence_l116_11661

theorem smallest_solution_congruence :
  ∃ (x : ℕ), x > 0 ∧ (6 * x) % 35 = 17 % 35 ∧
  ∀ (y : ℕ), y > 0 ∧ (6 * y) % 35 = 17 % 35 → x ≤ y :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_smallest_solution_congruence_l116_11661


namespace NUMINAMATH_CALUDE_work_completion_time_l116_11630

theorem work_completion_time (renu_rate suma_rate : ℚ) 
  (h1 : renu_rate = 1 / 8)
  (h2 : suma_rate = 1 / (24 / 5))
  : (1 / (renu_rate + suma_rate) : ℚ) = 3 := by
  sorry

end NUMINAMATH_CALUDE_work_completion_time_l116_11630


namespace NUMINAMATH_CALUDE_remaining_oil_after_350km_distance_when_8_liters_left_l116_11666

-- Define the initial conditions
def initial_oil : ℝ := 56
def oil_consumption_rate : ℝ := 0.08

-- Define the relationship between remaining oil and distance traveled
def remaining_oil (x : ℝ) : ℝ := initial_oil - oil_consumption_rate * x

-- Theorem to prove the remaining oil after 350 km
theorem remaining_oil_after_350km :
  remaining_oil 350 = 28 := by sorry

-- Theorem to prove the distance traveled when 8 liters are left
theorem distance_when_8_liters_left :
  ∃ x : ℝ, remaining_oil x = 8 ∧ x = 600 := by sorry

end NUMINAMATH_CALUDE_remaining_oil_after_350km_distance_when_8_liters_left_l116_11666


namespace NUMINAMATH_CALUDE_number_of_factors_of_30_l116_11670

theorem number_of_factors_of_30 : Finset.card (Nat.divisors 30) = 8 := by
  sorry

end NUMINAMATH_CALUDE_number_of_factors_of_30_l116_11670


namespace NUMINAMATH_CALUDE_quadratic_roots_l116_11675

/-- A quadratic function passing through specific points -/
def f (a b c : ℝ) : ℝ → ℝ := fun x ↦ a * x^2 + b * x + c

/-- The theorem stating that the quadratic equation has specific roots -/
theorem quadratic_roots (a b c : ℝ) : 
  f a b c (-2) = 21 ∧ 
  f a b c (-1) = 12 ∧ 
  f a b c 1 = 0 ∧ 
  f a b c 2 = -3 ∧ 
  f a b c 4 = -3 → 
  (∀ x, f a b c x = 0 ↔ x = 1 ∨ x = 5) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_l116_11675


namespace NUMINAMATH_CALUDE_unique_solution_l116_11652

-- Define the type for digits from 2 to 9
def Digit := Fin 8

-- Define a function to map letters to digits
def LetterToDigit := Char → Digit

-- Define the letters used in the problem
def Letters : List Char := ['N', 'I', 'E', 'O', 'T', 'W', 'S', 'X']

-- Define the function to convert a word to a number
def wordToNumber (f : LetterToDigit) (word : List Char) : Nat :=
  word.foldl (fun acc c => 10 * acc + (f c).val.succ.succ) 0

-- State the theorem
theorem unique_solution :
  ∃! f : LetterToDigit,
    (∀ c₁ c₂ : Char, c₁ ∈ Letters → c₂ ∈ Letters → c₁ ≠ c₂ → f c₁ ≠ f c₂) ∧
    (wordToNumber f ['O', 'N', 'E']) + 
    (wordToNumber f ['T', 'W', 'O']) + 
    (wordToNumber f ['S', 'I', 'X']) = 
    (wordToNumber f ['N', 'I', 'N', 'E']) ∧
    (wordToNumber f ['N', 'I', 'N', 'E']) = 2526 :=
  sorry

end NUMINAMATH_CALUDE_unique_solution_l116_11652


namespace NUMINAMATH_CALUDE_optimal_sampling_methods_l116_11631

/-- Represents different sampling methods -/
inductive SamplingMethod
  | SimpleRandom
  | Systematic
  | Stratified

/-- Represents income levels -/
inductive IncomeLevel
  | High
  | Middle
  | Low

/-- Community structure -/
structure Community where
  totalFamilies : Nat
  highIncomeFamilies : Nat
  middleIncomeFamilies : Nat
  lowIncomeFamilies : Nat

/-- Sample size for family survey -/
def familySampleSize : Nat := 100

/-- Student selection parameters -/
structure StudentSelection where
  totalStudents : Nat
  studentsToSelect : Nat

/-- Function to determine the optimal sampling method for family survey -/
def optimalFamilySamplingMethod (c : Community) : SamplingMethod := sorry

/-- Function to determine the optimal sampling method for student selection -/
def optimalStudentSamplingMethod (s : StudentSelection) : SamplingMethod := sorry

/-- Theorem stating the optimal sampling methods for the given scenario -/
theorem optimal_sampling_methods 
  (community : Community)
  (studentSelection : StudentSelection)
  (h1 : community.totalFamilies = 800)
  (h2 : community.highIncomeFamilies = 200)
  (h3 : community.middleIncomeFamilies = 480)
  (h4 : community.lowIncomeFamilies = 120)
  (h5 : studentSelection.totalStudents = 10)
  (h6 : studentSelection.studentsToSelect = 3) :
  optimalFamilySamplingMethod community = SamplingMethod.Stratified ∧
  optimalStudentSamplingMethod studentSelection = SamplingMethod.SimpleRandom := by
  sorry


end NUMINAMATH_CALUDE_optimal_sampling_methods_l116_11631


namespace NUMINAMATH_CALUDE_set_union_equality_implies_m_range_l116_11677

theorem set_union_equality_implies_m_range (m : ℝ) : 
  let A : Set ℝ := {x | x^2 - 3*x - 10 ≤ 0}
  let B : Set ℝ := {x | m + 1 ≤ x ∧ x ≤ 2*m - 1}
  A ∪ B = A → m ∈ Set.Icc (-3) 3 := by
  sorry

end NUMINAMATH_CALUDE_set_union_equality_implies_m_range_l116_11677


namespace NUMINAMATH_CALUDE_plane_perpendicularity_l116_11616

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (parallel : Line → Line → Prop)
variable (perpendicular : Line → Plane → Prop)
variable (contained_in : Line → Plane → Prop)
variable (plane_perpendicular : Plane → Plane → Prop)

-- State the theorem
theorem plane_perpendicularity 
  (m n : Line) (α β : Plane) :
  parallel m n → 
  contained_in m α → 
  perpendicular n β → 
  plane_perpendicular α β := by
sorry

end NUMINAMATH_CALUDE_plane_perpendicularity_l116_11616


namespace NUMINAMATH_CALUDE_sum_of_reciprocals_of_roots_l116_11610

theorem sum_of_reciprocals_of_roots (r₁ r₂ : ℝ) : 
  r₁^2 - 26*r₁ + 12 = 0 → 
  r₂^2 - 26*r₂ + 12 = 0 → 
  r₁ ≠ r₂ →
  (1/r₁ + 1/r₂) = 13/6 := by
sorry

end NUMINAMATH_CALUDE_sum_of_reciprocals_of_roots_l116_11610


namespace NUMINAMATH_CALUDE_car_trip_distance_l116_11625

theorem car_trip_distance (D : ℝ) :
  let remaining_after_first_stop := D / 2
  let remaining_after_second_stop := remaining_after_first_stop * 2 / 3
  let remaining_after_third_stop := remaining_after_second_stop * 3 / 5
  remaining_after_third_stop = 180
  → D = 900 := by
  sorry

end NUMINAMATH_CALUDE_car_trip_distance_l116_11625


namespace NUMINAMATH_CALUDE_mutual_fund_share_price_increase_l116_11672

theorem mutual_fund_share_price_increase (initial_price : ℝ) : 
  let first_quarter_price := initial_price * 1.25
  let second_quarter_price := initial_price * 1.55
  (second_quarter_price - first_quarter_price) / first_quarter_price * 100 = 24 := by
  sorry

end NUMINAMATH_CALUDE_mutual_fund_share_price_increase_l116_11672


namespace NUMINAMATH_CALUDE_andrew_grape_purchase_l116_11650

/-- The amount of grapes Andrew purchased in kg -/
def G : ℝ := by sorry

/-- The price of grapes per kg -/
def grape_price : ℝ := 70

/-- The amount of mangoes Andrew purchased in kg -/
def mango_amount : ℝ := 9

/-- The price of mangoes per kg -/
def mango_price : ℝ := 55

/-- The total amount Andrew paid -/
def total_paid : ℝ := 1055

theorem andrew_grape_purchase :
  G * grape_price + mango_amount * mango_price = total_paid ∧ G = 8 := by sorry

end NUMINAMATH_CALUDE_andrew_grape_purchase_l116_11650


namespace NUMINAMATH_CALUDE_cube_sum_equals_407_l116_11686

theorem cube_sum_equals_407 (x y : ℝ) (h1 : x + y = 11) (h2 : x^2 * y = 36) :
  x^3 + y^3 = 407 := by
sorry

end NUMINAMATH_CALUDE_cube_sum_equals_407_l116_11686


namespace NUMINAMATH_CALUDE_min_sum_of_two_digits_is_one_l116_11683

/-- A digit is a natural number from 0 to 9 -/
def Digit := { n : ℕ // n ≤ 9 }

/-- The theorem states that the minimum sum of two digits P and Q is 1,
    given that P, Q, R, and S are four different digits,
    and (P+Q)/(R+S) is an integer and as small as possible. -/
theorem min_sum_of_two_digits_is_one
  (P Q R S : Digit)
  (h_distinct : P ≠ Q ∧ P ≠ R ∧ P ≠ S ∧ Q ≠ R ∧ Q ≠ S ∧ R ≠ S)
  (h_integer : ∃ k : ℕ, (P.val + Q.val : ℚ) / (R.val + S.val) = k)
  (h_min : ∀ (P' Q' R' S' : Digit),
           P' ≠ Q' ∧ P' ≠ R' ∧ P' ≠ S' ∧ Q' ≠ R' ∧ Q' ≠ S' ∧ R' ≠ S' →
           (∃ k : ℕ, (P'.val + Q'.val : ℚ) / (R'.val + S'.val) = k) →
           (P.val + Q.val : ℚ) / (R.val + S.val) ≤ (P'.val + Q'.val : ℚ) / (R'.val + S'.val)) :
  P.val + Q.val = 1 :=
sorry

end NUMINAMATH_CALUDE_min_sum_of_two_digits_is_one_l116_11683


namespace NUMINAMATH_CALUDE_capri_sun_pouches_per_box_l116_11692

theorem capri_sun_pouches_per_box 
  (total_boxes : ℕ) 
  (total_paid : ℚ) 
  (cost_per_pouch : ℚ) 
  (h1 : total_boxes = 10) 
  (h2 : total_paid = 12) 
  (h3 : cost_per_pouch = 1/5) : 
  (total_paid / cost_per_pouch) / total_boxes = 6 := by
sorry

end NUMINAMATH_CALUDE_capri_sun_pouches_per_box_l116_11692


namespace NUMINAMATH_CALUDE_parallel_lines_m_value_l116_11641

/-- Two lines are parallel if and only if they have the same slope -/
axiom parallel_lines_same_slope {m₁ m₂ b₁ b₂ : ℝ} :
  (∀ x y : ℝ, y = m₁ * x + b₁ ↔ y = m₂ * x + b₂) ↔ m₁ = m₂

/-- The equation of line l₁ -/
def line_l₁ (m : ℝ) (x y : ℝ) : Prop := m * x + y - 2 = 0

/-- The equation of line l₂ -/
def line_l₂ (x y : ℝ) : Prop := y = 2 * x - 1

theorem parallel_lines_m_value :
  (∀ x y : ℝ, line_l₁ m x y ↔ line_l₂ x y) → m = -2 :=
by sorry

end NUMINAMATH_CALUDE_parallel_lines_m_value_l116_11641


namespace NUMINAMATH_CALUDE_converse_and_inverse_false_l116_11660

-- Define the properties of triangles
def Equilateral (t : Type) : Prop := sorry
def Isosceles (t : Type) : Prop := sorry

-- State the given true statement
axiom equilateral_implies_isosceles : ∀ t : Type, Equilateral t → Isosceles t

-- Define the converse and inverse
def converse : Prop := ∀ t : Type, Isosceles t → Equilateral t
def inverse : Prop := ∀ t : Type, ¬(Equilateral t) → ¬(Isosceles t)

-- Theorem to prove
theorem converse_and_inverse_false : ¬converse ∧ ¬inverse := by
  sorry

end NUMINAMATH_CALUDE_converse_and_inverse_false_l116_11660


namespace NUMINAMATH_CALUDE_parabola_rotation_l116_11685

/-- A parabola in the xy-plane -/
structure Parabola where
  a : ℝ  -- coefficient of (x-h)^2
  h : ℝ  -- x-coordinate of vertex
  k : ℝ  -- y-coordinate of vertex

/-- Rotate a parabola by 180 degrees around its vertex -/
def rotate180 (p : Parabola) : Parabola :=
  { a := -p.a, h := p.h, k := p.k }

theorem parabola_rotation (p : Parabola) (hp : p = ⟨2, 3, -2⟩) :
  rotate180 p = ⟨-2, 3, -2⟩ := by
  sorry

#check parabola_rotation

end NUMINAMATH_CALUDE_parabola_rotation_l116_11685


namespace NUMINAMATH_CALUDE_inner_hexagon_area_lower_bound_l116_11647

/-- A regular hexagon -/
structure RegularHexagon where
  vertices : Fin 6 → ℝ × ℝ
  is_regular : sorry

/-- A point inside a regular hexagon -/
structure PointInHexagon (h : RegularHexagon) where
  point : ℝ × ℝ
  is_inside : sorry

/-- The hexagon formed by connecting a point to the vertices of a regular hexagon -/
def inner_hexagon (h : RegularHexagon) (p : PointInHexagon h) : Set (ℝ × ℝ) :=
  sorry

/-- The area of a set in ℝ² -/
noncomputable def area (s : Set (ℝ × ℝ)) : ℝ :=
  sorry

/-- The theorem stating that the area of the inner hexagon is at least 2/3 of the original hexagon -/
theorem inner_hexagon_area_lower_bound (h : RegularHexagon) (p : PointInHexagon h) :
  area (inner_hexagon h p) ≥ (2/3) * area (Set.range h.vertices) :=
sorry

end NUMINAMATH_CALUDE_inner_hexagon_area_lower_bound_l116_11647


namespace NUMINAMATH_CALUDE_mike_working_time_l116_11688

/-- Calculates the total working time in hours for Mike's car service tasks. -/
def calculate_working_time (wash_time min_per_car : ℕ) (oil_change_time min_per_car : ℕ)
  (tire_change_time min_per_set : ℕ) (cars_washed : ℕ) (oil_changes : ℕ)
  (tire_sets_changed : ℕ) : ℚ :=
  let total_minutes := wash_time * cars_washed +
                       oil_change_time * oil_changes +
                       tire_change_time * tire_sets_changed
  total_minutes / 60

theorem mike_working_time :
  calculate_working_time 10 15 30 9 6 2 = 4 := by
  sorry

end NUMINAMATH_CALUDE_mike_working_time_l116_11688


namespace NUMINAMATH_CALUDE_max_value_a_l116_11608

theorem max_value_a (a b c d : ℕ+) (h1 : a < 3 * b) (h2 : b < 4 * c) (h3 : c < 5 * d) 
  (h4 : Even c) (h5 : d < 150) : a ≤ 8924 := by
  sorry

end NUMINAMATH_CALUDE_max_value_a_l116_11608


namespace NUMINAMATH_CALUDE_matrix_inverse_proof_l116_11601

def A : Matrix (Fin 2) (Fin 2) ℚ := !![5, 4; -2, 8]

def A_inv : Matrix (Fin 2) (Fin 2) ℚ := !![1/6, -1/12; 1/24, 5/48]

theorem matrix_inverse_proof :
  A * A_inv = 1 ∧ A_inv * A = 1 :=
by sorry

end NUMINAMATH_CALUDE_matrix_inverse_proof_l116_11601


namespace NUMINAMATH_CALUDE_first_term_is_two_l116_11695

/-- An arithmetic sequence with the given properties -/
structure ArithmeticSequence where
  a : ℕ → ℝ
  monotone_increasing : ∀ n, a n < a (n + 1)
  is_arithmetic : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)
  sum_first_three : a 1 + a 2 + a 3 = 12
  product_first_three : a 1 * a 2 * a 3 = 48

/-- The first term of the arithmetic sequence is 2 -/
theorem first_term_is_two (seq : ArithmeticSequence) : seq.a 1 = 2 := by
  sorry

end NUMINAMATH_CALUDE_first_term_is_two_l116_11695


namespace NUMINAMATH_CALUDE_cone_prism_volume_ratio_l116_11671

/-- The ratio of the volume of a right circular cone to the volume of its circumscribing right rectangular prism -/
theorem cone_prism_volume_ratio :
  ∀ (r h : ℝ), r > 0 → h > 0 →
  (1 / 3 * π * r^2 * h) / (9 * r^2 * h) = π / 27 := by
sorry

end NUMINAMATH_CALUDE_cone_prism_volume_ratio_l116_11671


namespace NUMINAMATH_CALUDE_accessory_percentage_l116_11621

def computer_cost : ℝ := 3000
def initial_money : ℝ := 3 * computer_cost
def money_left : ℝ := 2700

theorem accessory_percentage :
  let total_spent := initial_money - money_left
  let accessory_cost := total_spent - computer_cost
  (accessory_cost / computer_cost) * 100 = 110 := by sorry

end NUMINAMATH_CALUDE_accessory_percentage_l116_11621


namespace NUMINAMATH_CALUDE_sequence_formula_l116_11615

theorem sequence_formula (a : ℕ → ℚ) :
  a 1 = 1 ∧
  (∀ n : ℕ, n ≥ 1 → 3 * a (n + 1) + 2 * a (n + 1) * a n - a n = 0) →
  ∀ n : ℕ, n ≥ 1 → a n = 1 / (2 * 3^(n - 1) - 1) :=
by sorry

end NUMINAMATH_CALUDE_sequence_formula_l116_11615


namespace NUMINAMATH_CALUDE_exactly_one_from_each_class_passing_at_least_one_student_passing_l116_11612

-- Define the probability of a student passing
def p_pass : ℝ := 0.6

-- Define the number of students from each class
def n_students : ℕ := 2

-- Define the probability of exactly one student from a class passing
def p_one_pass : ℝ := n_students * p_pass * (1 - p_pass)

-- Theorem for the first question
theorem exactly_one_from_each_class_passing : 
  p_one_pass * p_one_pass = 0.2304 := by sorry

-- Theorem for the second question
theorem at_least_one_student_passing : 
  1 - (1 - p_pass)^(2 * n_students) = 0.9744 := by sorry

end NUMINAMATH_CALUDE_exactly_one_from_each_class_passing_at_least_one_student_passing_l116_11612


namespace NUMINAMATH_CALUDE_jamesFinalNumber_l116_11636

-- Define a function to calculate the sum of digits
def sumOfDigits (n : ℕ) : ℕ := sorry

-- Define Kyle's result
def kylesResult : ℕ := sumOfDigits (2014^2014)

-- Define Shannon's result
def shannonsResult : ℕ := sumOfDigits kylesResult

-- Theorem to prove
theorem jamesFinalNumber : sumOfDigits shannonsResult = 7 := by sorry

end NUMINAMATH_CALUDE_jamesFinalNumber_l116_11636


namespace NUMINAMATH_CALUDE_no_integer_solution_l116_11619

theorem no_integer_solution : ¬ ∃ (a b c d : ℤ), 
  (a * 19^3 + b * 19^2 + c * 19 + d = 1) ∧ 
  (a * 62^3 + b * 62^2 + c * 62 + d = 2) := by
sorry

end NUMINAMATH_CALUDE_no_integer_solution_l116_11619


namespace NUMINAMATH_CALUDE_average_problem_l116_11624

theorem average_problem (numbers : List ℕ) (x : ℕ) : 
  numbers = [201, 202, 204, 205, 206, 209, 209, 210, 212] →
  (numbers.sum + x) / 10 = 207 →
  x = 212 := by
sorry

end NUMINAMATH_CALUDE_average_problem_l116_11624


namespace NUMINAMATH_CALUDE_range_of_a_minus_b_l116_11673

theorem range_of_a_minus_b (a b : ℝ) (h1 : -1 < a) (h2 : a < b) (h3 : b < 2) :
  -3 < a - b ∧ a - b < 0 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_minus_b_l116_11673


namespace NUMINAMATH_CALUDE_x_values_l116_11697

theorem x_values (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0)
  (h1 : x + 1 / y = 7) (h2 : y + 1 / x = 1 / 3) :
  x = 3 ∨ x = 4 := by
  sorry

end NUMINAMATH_CALUDE_x_values_l116_11697


namespace NUMINAMATH_CALUDE_unique_four_digit_number_l116_11623

theorem unique_four_digit_number : ∃! n : ℕ,
  (1000 ≤ n ∧ n < 10000) ∧ 
  (∃ a : ℕ, n = a^2) ∧
  (∃ b : ℕ, n % 1000 = b^3) ∧
  (∃ c : ℕ, n % 100 = c^4) ∧
  n = 9216 :=
by sorry

end NUMINAMATH_CALUDE_unique_four_digit_number_l116_11623


namespace NUMINAMATH_CALUDE_circle_passes_through_points_l116_11622

/-- The equation of a circle passing through points A(2, 0), B(4, 0), and C(1, 2) -/
def circle_equation (x y : ℝ) : Prop :=
  x^2 + y^2 - 6*x - (7/2)*y + 8 = 0

/-- Point A -/
def point_A : ℝ × ℝ := (2, 0)

/-- Point B -/
def point_B : ℝ × ℝ := (4, 0)

/-- Point C -/
def point_C : ℝ × ℝ := (1, 2)

/-- Theorem stating that the circle equation passes through points A, B, and C -/
theorem circle_passes_through_points :
  circle_equation point_A.1 point_A.2 ∧
  circle_equation point_B.1 point_B.2 ∧
  circle_equation point_C.1 point_C.2 :=
by sorry

end NUMINAMATH_CALUDE_circle_passes_through_points_l116_11622


namespace NUMINAMATH_CALUDE_min_value_expression_l116_11607

theorem min_value_expression :
  ∃ (s₀ t₀ : ℝ), ∀ (s t : ℝ), (s + 5 - 3 * |Real.cos t|)^2 + (s - 2 * |Real.sin t|)^2 ≥ 2 ∧
  (s₀ + 5 - 3 * |Real.cos t₀|)^2 + (s₀ - 2 * |Real.sin t₀|)^2 = 2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_expression_l116_11607


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l116_11699

theorem complex_fraction_simplification (x y : ℚ) (hx : x = 3) (hy : y = 4) :
  (x / (y + 1)) / (y / (x + 2)) = 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l116_11699


namespace NUMINAMATH_CALUDE_central_angle_regular_hexagon_l116_11648

/-- The central angle of a regular hexagon is 60 degrees. -/
theorem central_angle_regular_hexagon :
  ∀ (full_circle_degrees : ℝ) (num_sides : ℕ),
    full_circle_degrees = 360 →
    num_sides = 6 →
    full_circle_degrees / num_sides = 60 := by
  sorry

end NUMINAMATH_CALUDE_central_angle_regular_hexagon_l116_11648


namespace NUMINAMATH_CALUDE_tan_sum_simplification_l116_11698

theorem tan_sum_simplification :
  (∀ x y, Real.tan (x + y) = (Real.tan x + Real.tan y) / (1 - Real.tan x * Real.tan y)) →
  Real.tan (45 * π / 180) = 1 →
  (1 + Real.tan (10 * π / 180)) * (1 + Real.tan (35 * π / 180)) = 2 := by
  sorry

end NUMINAMATH_CALUDE_tan_sum_simplification_l116_11698


namespace NUMINAMATH_CALUDE_carriage_hourly_rate_l116_11687

/-- Calculates the hourly rate for a carriage given the journey details and costs. -/
theorem carriage_hourly_rate
  (distance : ℝ)
  (speed : ℝ)
  (flat_fee : ℝ)
  (total_cost : ℝ)
  (h1 : distance = 20)
  (h2 : speed = 10)
  (h3 : flat_fee = 20)
  (h4 : total_cost = 80) :
  (total_cost - flat_fee) / (distance / speed) = 30 := by
  sorry

#check carriage_hourly_rate

end NUMINAMATH_CALUDE_carriage_hourly_rate_l116_11687


namespace NUMINAMATH_CALUDE_max_value_of_expression_l116_11602

theorem max_value_of_expression (x : ℝ) :
  (x^6) / (x^10 + 3*x^8 - 6*x^6 + 12*x^4 + 32) ≤ 1/18 ∧
  (2^6) / (2^10 + 3*2^8 - 6*2^6 + 12*2^4 + 32) = 1/18 := by
  sorry

end NUMINAMATH_CALUDE_max_value_of_expression_l116_11602


namespace NUMINAMATH_CALUDE_A_intersect_B_l116_11657

def A : Set ℕ := {2, 3, 4}
def B : Set ℕ := {0, 1, 2}

theorem A_intersect_B : A ∩ B = {2} := by sorry

end NUMINAMATH_CALUDE_A_intersect_B_l116_11657


namespace NUMINAMATH_CALUDE_triangle_area_l116_11684

theorem triangle_area (a b c : ℝ) (h_right_angle : a^2 + b^2 = c^2) 
  (h_angle : Real.cos (30 * π / 180) = c / (2 * a)) (h_side : b = 8) : 
  (1/2) * a * b = 32 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_l116_11684


namespace NUMINAMATH_CALUDE_plums_picked_total_l116_11678

/-- The number of plums Alyssa picked -/
def alyssas_plums : ℕ := 17

/-- The number of plums Jason picked -/
def jasons_plums : ℕ := 10

/-- The total number of plums picked -/
def total_plums : ℕ := alyssas_plums + jasons_plums

theorem plums_picked_total :
  total_plums = 27 := by sorry

end NUMINAMATH_CALUDE_plums_picked_total_l116_11678


namespace NUMINAMATH_CALUDE_complex_root_quadratic_l116_11618

theorem complex_root_quadratic (a : ℝ) : 
  (∃ x : ℂ, x^2 - 2*a*x + a^2 - 4*a + 6 = 0) ∧ 
  (Complex.I^2 = -1) ∧
  ((1 : ℂ) + Complex.I * Real.sqrt 2)^2 - 2*a*((1 : ℂ) + Complex.I * Real.sqrt 2) + a^2 - 4*a + 6 = 0
  → a = 1 := by
sorry

end NUMINAMATH_CALUDE_complex_root_quadratic_l116_11618


namespace NUMINAMATH_CALUDE_final_candy_count_l116_11645

-- Define the variables
def initial_candy : ℕ := 47
def eaten_candy : ℕ := 25
def received_candy : ℕ := 40

-- State the theorem
theorem final_candy_count :
  initial_candy - eaten_candy + received_candy = 62 := by
  sorry

end NUMINAMATH_CALUDE_final_candy_count_l116_11645


namespace NUMINAMATH_CALUDE_freds_shopping_cost_l116_11681

/-- Calculates the total cost of Fred's shopping trip --/
def calculate_shopping_cost (orange_price : ℚ) (orange_quantity : ℕ)
                            (cereal_price : ℚ) (cereal_quantity : ℕ)
                            (bread_price : ℚ) (bread_quantity : ℕ)
                            (cookie_price : ℚ) (cookie_quantity : ℕ)
                            (bakery_discount_threshold : ℚ)
                            (bakery_discount_rate : ℚ)
                            (coupon_threshold : ℚ)
                            (coupon_value : ℚ) : ℚ :=
  let orange_total := orange_price * orange_quantity
  let cereal_total := cereal_price * cereal_quantity
  let bread_total := bread_price * bread_quantity
  let cookie_total := cookie_price * cookie_quantity
  let bakery_total := bread_total + cookie_total
  let total_before_discounts := orange_total + cereal_total + bakery_total
  let bakery_discount := if bakery_total > bakery_discount_threshold then bakery_total * bakery_discount_rate else 0
  let total_after_bakery_discount := total_before_discounts - bakery_discount
  let final_total := if total_before_discounts ≥ coupon_threshold then total_after_bakery_discount - coupon_value else total_after_bakery_discount
  final_total

/-- Theorem stating that Fred's shopping cost is $30.5 --/
theorem freds_shopping_cost :
  calculate_shopping_cost 2 4 4 3 3 3 6 1 15 (1/10) 30 3 = 30.5 := by
  sorry

end NUMINAMATH_CALUDE_freds_shopping_cost_l116_11681


namespace NUMINAMATH_CALUDE_set_d_forms_triangle_l116_11674

/-- Triangle Inequality Theorem: The sum of the lengths of any two sides of a triangle
    must be greater than the length of the remaining side. --/
def satisfies_triangle_inequality (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

/-- Check if three lengths can form a triangle --/
def can_form_triangle (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ satisfies_triangle_inequality a b c

theorem set_d_forms_triangle :
  can_form_triangle 6 6 6 := by
  sorry

end NUMINAMATH_CALUDE_set_d_forms_triangle_l116_11674


namespace NUMINAMATH_CALUDE_system_solution_l116_11653

theorem system_solution :
  ∃! (x y z : ℚ),
    2 * x - 3 * y + z = 8 ∧
    4 * x - 6 * y + 2 * z = 16 ∧
    x + y - z = 1 ∧
    x = 11 / 3 ∧
    y = 1 ∧
    z = 11 / 3 := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l116_11653


namespace NUMINAMATH_CALUDE_f_derivative_at_2_l116_11679

def f (x : ℝ) : ℝ := x^3 + 4*x - 5

theorem f_derivative_at_2 : (deriv f) 2 = 16 := by sorry

end NUMINAMATH_CALUDE_f_derivative_at_2_l116_11679


namespace NUMINAMATH_CALUDE_line_inclination_angle_l116_11691

def line_equation (x y : ℝ) : Prop := Real.sqrt 3 * x + 3 * y + 2 = 0

def inclination_angle (eq : (ℝ → ℝ → Prop)) : ℝ := sorry

theorem line_inclination_angle :
  inclination_angle line_equation = 150 * Real.pi / 180 :=
sorry

end NUMINAMATH_CALUDE_line_inclination_angle_l116_11691


namespace NUMINAMATH_CALUDE_rice_sales_problem_l116_11689

/-- Represents the daily rice sales function -/
structure RiceSales where
  k : ℝ
  b : ℝ
  y : ℝ → ℝ
  h1 : ∀ x, y x = k * x + b
  h2 : y 5 = 950
  h3 : y 6 = 900

/-- Calculates the profit for a given price -/
def profit (price : ℝ) (sales : ℝ) : ℝ := (price - 4) * sales

theorem rice_sales_problem (rs : RiceSales) :
  (rs.y = λ x => -50 * x + 1200) ∧
  (∃ x ∈ Set.Icc 4 7, profit x (rs.y x) = 1800 ∧ x = 6) ∧
  (∀ x ∈ Set.Icc 4 7, profit x (rs.y x) ≤ 2550) ∧
  (profit 7 (rs.y 7) = 2550) := by
  sorry

end NUMINAMATH_CALUDE_rice_sales_problem_l116_11689


namespace NUMINAMATH_CALUDE_stripe_area_on_cylindrical_tank_l116_11646

/-- The area of a stripe painted on a cylindrical tank -/
theorem stripe_area_on_cylindrical_tank 
  (diameter : ℝ) 
  (stripe_width : ℝ) 
  (revolutions : ℕ) 
  (h1 : diameter = 40)
  (h2 : stripe_width = 4)
  (h3 : revolutions = 3) : 
  stripe_width * (Real.pi * diameter * revolutions) = 480 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_stripe_area_on_cylindrical_tank_l116_11646


namespace NUMINAMATH_CALUDE_f_of_2_equals_3_l116_11606

/-- Given a function f(x) = x^2 - 2x + 3, prove that f(2) = 3 -/
theorem f_of_2_equals_3 : let f : ℝ → ℝ := fun x ↦ x^2 - 2*x + 3
  f 2 = 3 := by
  sorry

end NUMINAMATH_CALUDE_f_of_2_equals_3_l116_11606


namespace NUMINAMATH_CALUDE_greatest_negative_value_x_minus_y_l116_11644

theorem greatest_negative_value_x_minus_y :
  ∃ (x y : ℝ), 
    (Real.sin x + Real.sin y) * (Real.cos x - Real.cos y) = 1/2 + Real.sin (x - y) * Real.cos (x + y) ∧
    x - y = -π/6 ∧
    ∀ (a b : ℝ), 
      (Real.sin a + Real.sin b) * (Real.cos a - Real.cos b) = 1/2 + Real.sin (a - b) * Real.cos (a + b) →
      a - b < 0 →
      a - b ≤ -π/6 :=
by sorry

end NUMINAMATH_CALUDE_greatest_negative_value_x_minus_y_l116_11644


namespace NUMINAMATH_CALUDE_find_y_l116_11642

theorem find_y (x y : ℤ) (h1 : x + y = 300) (h2 : x - y = 200) : y = 50 := by
  sorry

end NUMINAMATH_CALUDE_find_y_l116_11642


namespace NUMINAMATH_CALUDE_multiples_of_five_most_representative_l116_11696

/-- Represents a sampling method for the math test --/
inductive SamplingMethod
  | TopStudents
  | BottomStudents
  | FemaleStudents
  | MultiplesOfFive

/-- Represents a student in the seventh grade --/
structure Student where
  id : Nat
  gender : Bool  -- True for female, False for male
  score : Nat

/-- The population of students who took the test --/
def population : Finset Student := sorry

/-- The total number of students in the population --/
axiom total_students : Finset.card population = 400

/-- Defines what makes a sampling method representative --/
def is_representative (method : SamplingMethod) : Prop := sorry

/-- Theorem stating that selecting students with numbers that are multiples of 5 
    is the most representative sampling method --/
theorem multiples_of_five_most_representative : 
  is_representative SamplingMethod.MultiplesOfFive ∧ 
  ∀ m : SamplingMethod, m ≠ SamplingMethod.MultiplesOfFive → 
    ¬(is_representative m) :=
sorry

end NUMINAMATH_CALUDE_multiples_of_five_most_representative_l116_11696


namespace NUMINAMATH_CALUDE_total_cars_is_32_l116_11690

/-- The number of cars owned by each person -/
structure CarOwnership where
  cathy : ℕ
  lindsey : ℕ
  carol : ℕ
  susan : ℕ

/-- The conditions of car ownership -/
def car_ownership_conditions (c : CarOwnership) : Prop :=
  c.lindsey = c.cathy + 4 ∧
  c.susan = c.carol - 2 ∧
  c.carol = 2 * c.cathy ∧
  c.cathy = 5

/-- The total number of cars owned by all four people -/
def total_cars (c : CarOwnership) : ℕ :=
  c.cathy + c.lindsey + c.carol + c.susan

/-- Theorem stating that the total number of cars is 32 -/
theorem total_cars_is_32 (c : CarOwnership) (h : car_ownership_conditions c) :
  total_cars c = 32 := by
  sorry

end NUMINAMATH_CALUDE_total_cars_is_32_l116_11690


namespace NUMINAMATH_CALUDE_chess_players_lost_to_ai_castor_island_ai_losses_l116_11605

/-- The number of chess players who have lost to a computer at least once on Castor island -/
theorem chess_players_lost_to_ai (total_players : ℝ) (never_lost_fraction : ℚ) : ℝ :=
  let never_lost := total_players * (never_lost_fraction : ℝ)
  let lost_to_ai := total_players - never_lost
  ⌊lost_to_ai + 0.5⌋

/-- Given the conditions on Castor island, prove that approximately 48 players have lost to a computer -/
theorem castor_island_ai_losses : 
  ⌊chess_players_lost_to_ai 157.83 (37/53) + 0.5⌋ = 48 := by
sorry

end NUMINAMATH_CALUDE_chess_players_lost_to_ai_castor_island_ai_losses_l116_11605


namespace NUMINAMATH_CALUDE_eight_power_fifteen_div_sixtyfour_power_six_l116_11629

theorem eight_power_fifteen_div_sixtyfour_power_six : 8^15 / 64^6 = 512 := by
  sorry

end NUMINAMATH_CALUDE_eight_power_fifteen_div_sixtyfour_power_six_l116_11629


namespace NUMINAMATH_CALUDE_pipe_fill_time_l116_11620

theorem pipe_fill_time (T : ℝ) (h1 : T > 0) (h2 : 1/T - 1/4.5 = 1/9) : T = 3 := by
  sorry

end NUMINAMATH_CALUDE_pipe_fill_time_l116_11620


namespace NUMINAMATH_CALUDE_additive_function_value_l116_11654

/-- A function satisfying f(x+y) = f(x) + f(y) for all real x and y -/
def AdditiveFunctionR (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x + y) = f x + f y

/-- Theorem: If f is an additive function on ℝ and f(2) = 4, then f(1) = 2 -/
theorem additive_function_value (f : ℝ → ℝ) (h1 : AdditiveFunctionR f) (h2 : f 2 = 4) : f 1 = 2 := by
  sorry

end NUMINAMATH_CALUDE_additive_function_value_l116_11654


namespace NUMINAMATH_CALUDE_wood_measurement_problem_l116_11658

theorem wood_measurement_problem (x y : ℝ) :
  (x + 4.5 = y ∧ x + 1 = (1/2) * y) ↔
  (∃ (wood_length rope_length : ℝ),
    wood_length = x ∧
    rope_length = y ∧
    wood_length + 4.5 = rope_length ∧
    wood_length + 1 = (1/2) * rope_length) :=
by sorry

end NUMINAMATH_CALUDE_wood_measurement_problem_l116_11658


namespace NUMINAMATH_CALUDE_transform_invariant_l116_11663

def initial_point : ℝ × ℝ × ℝ := (1, 1, 1)

def rotate_z_90 (p : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  let (x, y, z) := p
  (-y, x, z)

def reflect_yz (p : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  let (x, y, z) := p
  (-x, y, z)

def reflect_xz (p : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  let (x, y, z) := p
  (x, -y, z)

def reflect_xy (p : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  let (x, y, z) := p
  (x, y, -z)

def transform (p : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  p
  |> rotate_z_90
  |> reflect_yz
  |> reflect_xz
  |> rotate_z_90
  |> reflect_xy
  |> reflect_xz

theorem transform_invariant : transform initial_point = initial_point := by
  sorry

end NUMINAMATH_CALUDE_transform_invariant_l116_11663


namespace NUMINAMATH_CALUDE_tissue_magnification_l116_11667

/-- Given a circular piece of tissue magnified by an electron microscope, 
    this theorem proves the relationship between the magnified image diameter 
    and the actual tissue diameter. -/
theorem tissue_magnification (magnification : ℝ) (magnified_diameter : ℝ) 
  (h1 : magnification = 1000) 
  (h2 : magnified_diameter = 2) :
  magnified_diameter / magnification = 0.002 := by
  sorry

end NUMINAMATH_CALUDE_tissue_magnification_l116_11667


namespace NUMINAMATH_CALUDE_trig_inequality_l116_11664

theorem trig_inequality : Real.tan (55 * π / 180) > Real.cos (55 * π / 180) ∧ Real.cos (55 * π / 180) > Real.sin (33 * π / 180) := by
  sorry

end NUMINAMATH_CALUDE_trig_inequality_l116_11664


namespace NUMINAMATH_CALUDE_multiplication_properties_l116_11609

theorem multiplication_properties :
  (∃ (p q : Nat), Prime p ∧ Prime q ∧ ¬(Prime (p * q))) ∧
  (∀ (a b : Int), ∃ (c : Int), (a^2) * (b^2) = c^2) ∧
  (∀ (m n : Int), Odd m → Odd n → Odd (m * n)) ∧
  (∀ (x y : Int), Even x → Even y → Even (x * y)) :=
by sorry

#check multiplication_properties

end NUMINAMATH_CALUDE_multiplication_properties_l116_11609


namespace NUMINAMATH_CALUDE_drive_time_calculation_l116_11668

/-- Given a person drives 120 miles in 3 hours, prove that driving 200 miles
    at the same speed will take 5 hours. -/
theorem drive_time_calculation (distance1 : ℝ) (time1 : ℝ) (distance2 : ℝ)
    (h1 : distance1 = 120)
    (h2 : time1 = 3)
    (h3 : distance2 = 200) :
  let speed := distance1 / time1
  distance2 / speed = 5 := by
  sorry

end NUMINAMATH_CALUDE_drive_time_calculation_l116_11668


namespace NUMINAMATH_CALUDE_complement_of_union_equals_four_l116_11638

universe u

def U : Set ℕ := {1, 2, 3, 4}
def M : Set ℕ := {1, 2}
def N : Set ℕ := {2, 3}

theorem complement_of_union_equals_four :
  (M ∪ N)ᶜ = {4} := by sorry

end NUMINAMATH_CALUDE_complement_of_union_equals_four_l116_11638


namespace NUMINAMATH_CALUDE_reduced_rate_fraction_l116_11611

/-- Represents the fraction of a day with reduced rates -/
def weekdayReducedRateFraction : ℚ := 12 / 24

/-- Represents the fraction of a day with reduced rates on weekends -/
def weekendReducedRateFraction : ℚ := 1

/-- Represents the number of weekdays in a week -/
def weekdaysPerWeek : ℕ := 5

/-- Represents the number of weekend days in a week -/
def weekendDaysPerWeek : ℕ := 2

/-- Represents the total number of days in a week -/
def daysPerWeek : ℕ := 7

/-- Theorem stating that the fraction of a week with reduced rates is 9/14 -/
theorem reduced_rate_fraction :
  (weekdayReducedRateFraction * weekdaysPerWeek + weekendReducedRateFraction * weekendDaysPerWeek) / daysPerWeek = 9 / 14 := by
  sorry


end NUMINAMATH_CALUDE_reduced_rate_fraction_l116_11611


namespace NUMINAMATH_CALUDE_binomial_coefficient_28_5_l116_11643

theorem binomial_coefficient_28_5 (h1 : Nat.choose 26 3 = 2600)
                                  (h2 : Nat.choose 26 4 = 14950)
                                  (h3 : Nat.choose 26 5 = 65780) :
  Nat.choose 28 5 = 98280 := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_28_5_l116_11643


namespace NUMINAMATH_CALUDE_no_rain_probability_l116_11682

theorem no_rain_probability (p : ℚ) (h : p = 2/3) :
  (1 - p)^5 = 1/243 := by
  sorry

end NUMINAMATH_CALUDE_no_rain_probability_l116_11682


namespace NUMINAMATH_CALUDE_triangle_base_length_l116_11628

/-- Given a triangle with area 615 and height 10, prove its base is 123 -/
theorem triangle_base_length (area : ℝ) (height : ℝ) (base : ℝ) 
  (h_area : area = 615) 
  (h_height : height = 10) 
  (h_triangle_area : area = (base * height) / 2) : base = 123 := by
  sorry

end NUMINAMATH_CALUDE_triangle_base_length_l116_11628


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l116_11662

/-- An arithmetic sequence is a sequence where the difference between each consecutive term is constant. -/
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- Given an arithmetic sequence a with a₁ = 3 and a₃ = 5, prove that a₅ = 7 -/
theorem arithmetic_sequence_problem (a : ℕ → ℝ) 
  (h_arith : is_arithmetic_sequence a) 
  (h_a1 : a 1 = 3) 
  (h_a3 : a 3 = 5) : 
  a 5 = 7 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l116_11662


namespace NUMINAMATH_CALUDE_tan_product_pi_ninths_l116_11637

theorem tan_product_pi_ninths : 
  Real.tan (π / 9) * Real.tan (2 * π / 9) * Real.tan (4 * π / 9) = 3 := by sorry

end NUMINAMATH_CALUDE_tan_product_pi_ninths_l116_11637


namespace NUMINAMATH_CALUDE_ellipse_line_intersection_range_l116_11604

-- Define the ellipse equation
def ellipse (x y m : ℝ) : Prop := x^2/3 + y^2/m = 1

-- Define the line equation
def line (x y : ℝ) : Prop := x + 2*y - 2 = 0

-- Define the intersection condition
def intersect_at_two_points (m : ℝ) : Prop :=
  ∃ x₁ y₁ x₂ y₂, x₁ ≠ x₂ ∧ ellipse x₁ y₁ m ∧ ellipse x₂ y₂ m ∧ 
                  line x₁ y₁ ∧ line x₂ y₂

-- Theorem statement
theorem ellipse_line_intersection_range :
  ∀ m : ℝ, intersect_at_two_points m ↔ (1/4 < m ∧ m < 3) ∨ m > 3 :=
sorry

end NUMINAMATH_CALUDE_ellipse_line_intersection_range_l116_11604


namespace NUMINAMATH_CALUDE_odd_function_property_l116_11617

/-- A function f is odd if f(-x) = -f(x) for all x -/
def IsOdd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

theorem odd_function_property (f : ℝ → ℝ) (h1 : IsOdd f) (h2 : f (-3) = -2) :
  f 3 + f 0 = 2 := by
  sorry

end NUMINAMATH_CALUDE_odd_function_property_l116_11617


namespace NUMINAMATH_CALUDE_odd_function_properties_l116_11676

-- Define the function f
def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^3 + b * x^2 + c * x

-- State the theorem
theorem odd_function_properties :
  ∀ (a b c : ℝ),
  (∀ x, f a b c (-x) = -(f a b c x)) →  -- f is odd
  f a b c (-Real.sqrt 2) = Real.sqrt 2 →  -- f(-√2) = √2
  f a b c (2 * Real.sqrt 2) = 10 * Real.sqrt 2 →  -- f(2√2) = 10√2
  (∃ (f' : ℝ → ℝ),
    (∀ x, f a b c x = x^3 - 3*x) ∧  -- f(x) = x³ - 3x
    (∀ x, x < -1 → f' x > 0) ∧  -- f is increasing on (-∞, -1)
    (∀ x, -1 < x ∧ x < 1 → f' x < 0) ∧  -- f is decreasing on (-1, 1)
    (∀ x, 1 < x → f' x > 0) ∧  -- f is increasing on (1, +∞)
    (∀ m, (∃ x y z, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧
           f a b c x + m = 0 ∧ f a b c y + m = 0 ∧ f a b c z + m = 0) ↔
          -2 < m ∧ m < 2)) -- f(x) + m = 0 has three distinct roots iff m ∈ (-2, 2)
  := by sorry

end NUMINAMATH_CALUDE_odd_function_properties_l116_11676


namespace NUMINAMATH_CALUDE_toys_per_day_l116_11693

-- Define the weekly toy production
def weekly_production : ℕ := 4340

-- Define the number of working days per week
def working_days : ℕ := 2

-- Define the daily toy production
def daily_production : ℕ := weekly_production / working_days

-- Theorem to prove
theorem toys_per_day : daily_production = 2170 := by
  sorry

end NUMINAMATH_CALUDE_toys_per_day_l116_11693


namespace NUMINAMATH_CALUDE_candy_store_revenue_l116_11634

/-- Represents the revenue calculation for a candy store sale --/
theorem candy_store_revenue : 
  let fudge_pounds : ℝ := 37
  let fudge_price : ℝ := 2.5
  let truffle_count : ℝ := 82
  let truffle_price : ℝ := 1.5
  let pretzel_count : ℝ := 48
  let pretzel_price : ℝ := 2
  let fudge_discount : ℝ := 0.1
  let sales_tax : ℝ := 0.05

  let fudge_revenue := fudge_pounds * fudge_price
  let truffle_revenue := truffle_count * truffle_price
  let pretzel_revenue := pretzel_count * pretzel_price

  let total_before_discount := fudge_revenue + truffle_revenue + pretzel_revenue
  let fudge_discount_amount := fudge_revenue * fudge_discount
  let total_after_discount := total_before_discount - fudge_discount_amount
  let tax_amount := total_after_discount * sales_tax
  let final_revenue := total_after_discount + tax_amount

  final_revenue = 317.36
  := by sorry


end NUMINAMATH_CALUDE_candy_store_revenue_l116_11634


namespace NUMINAMATH_CALUDE_cyclic_sum_inequality_l116_11665

theorem cyclic_sum_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  ((2*a + b + c)^2) / (2*a^2 + (b + c)^2) +
  ((2*b + c + a)^2) / (2*b^2 + (c + a)^2) +
  ((2*c + a + b)^2) / (2*c^2 + (a + b)^2) ≤ 8 := by
  sorry

end NUMINAMATH_CALUDE_cyclic_sum_inequality_l116_11665


namespace NUMINAMATH_CALUDE_consecutive_sum_18_l116_11627

def consecutive_sum (start : ℕ) (length : ℕ) : ℕ :=
  (length * (2 * start + length - 1)) / 2

theorem consecutive_sum_18 :
  ∃! (start length : ℕ), 2 ≤ length ∧ consecutive_sum start length = 18 :=
sorry

end NUMINAMATH_CALUDE_consecutive_sum_18_l116_11627


namespace NUMINAMATH_CALUDE_adult_ticket_cost_l116_11656

theorem adult_ticket_cost (child_ticket_cost : ℝ) (total_tickets : ℕ) (total_revenue : ℝ) (child_tickets : ℕ) : ℝ :=
  let adult_tickets := total_tickets - child_tickets
  let child_revenue := child_ticket_cost * child_tickets
  let adult_revenue := total_revenue - child_revenue
  let adult_ticket_cost := adult_revenue / adult_tickets
  by
    have h1 : child_ticket_cost = 4.5 := by sorry
    have h2 : total_tickets = 400 := by sorry
    have h3 : total_revenue = 2100 := by sorry
    have h4 : child_tickets = 200 := by sorry
    sorry

end NUMINAMATH_CALUDE_adult_ticket_cost_l116_11656


namespace NUMINAMATH_CALUDE_greatest_common_multiple_15_20_less_than_150_l116_11680

def is_common_multiple (m n k : ℕ) : Prop := k % m = 0 ∧ k % n = 0

theorem greatest_common_multiple_15_20_less_than_150 : 
  ∃ (k : ℕ), k = 120 ∧ 
  is_common_multiple 15 20 k ∧ 
  k < 150 ∧ 
  ∀ (m : ℕ), is_common_multiple 15 20 m → m < 150 → m ≤ k :=
sorry

end NUMINAMATH_CALUDE_greatest_common_multiple_15_20_less_than_150_l116_11680


namespace NUMINAMATH_CALUDE_gala_tree_count_l116_11632

/-- Represents an apple orchard with Fuji and Gala trees -/
structure AppleOrchard where
  totalTrees : ℕ
  pureFuji : ℕ
  pureGala : ℕ
  crossPollinated : ℕ

/-- The conditions of the orchard as described in the problem -/
def orchardConditions (o : AppleOrchard) : Prop :=
  o.crossPollinated = o.totalTrees / 10 ∧
  o.pureFuji = (o.totalTrees * 3) / 4 ∧
  o.pureFuji + o.crossPollinated = 170 ∧
  o.totalTrees = o.pureFuji + o.pureGala + o.crossPollinated

/-- The theorem stating that under the given conditions, there are 50 pure Gala trees -/
theorem gala_tree_count (o : AppleOrchard) 
  (h : orchardConditions o) : o.pureGala = 50 := by
  sorry

#check gala_tree_count

end NUMINAMATH_CALUDE_gala_tree_count_l116_11632


namespace NUMINAMATH_CALUDE_basketball_game_theorem_l116_11659

/-- Represents the score of a team for each quarter -/
structure GameScore :=
  (q1 q2 q3 q4 : ℚ)

/-- Checks if a GameScore forms a geometric sequence with ratio 3 -/
def isGeometricSequence (score : GameScore) : Prop :=
  score.q2 = 3 * score.q1 ∧ score.q3 = 3 * score.q2 ∧ score.q4 = 3 * score.q3

/-- Checks if a GameScore forms an arithmetic sequence with difference 12 -/
def isArithmeticSequence (score : GameScore) : Prop :=
  score.q2 = score.q1 + 12 ∧ score.q3 = score.q2 + 12 ∧ score.q4 = score.q3 + 12

/-- Calculates the total score for a GameScore -/
def totalScore (score : GameScore) : ℚ :=
  score.q1 + score.q2 + score.q3 + score.q4

/-- Calculates the first half score for a GameScore -/
def firstHalfScore (score : GameScore) : ℚ :=
  score.q1 + score.q2

theorem basketball_game_theorem (eagles lions : GameScore) : 
  eagles.q1 = lions.q1 →  -- Tied at the end of first quarter
  isGeometricSequence eagles →
  isArithmeticSequence lions →
  totalScore eagles = totalScore lions + 3 →  -- Eagles won by 3 points
  totalScore eagles ≤ 120 →
  totalScore lions ≤ 120 →
  firstHalfScore eagles + firstHalfScore lions = 15 :=
by sorry

end NUMINAMATH_CALUDE_basketball_game_theorem_l116_11659


namespace NUMINAMATH_CALUDE_margo_age_in_three_years_l116_11640

/-- Margo's age in three years given Benjie's current age and their age difference -/
def margos_future_age (benjies_age : ℕ) (age_difference : ℕ) : ℕ :=
  (benjies_age - age_difference) + 3

theorem margo_age_in_three_years :
  margos_future_age 6 5 = 4 := by
  sorry

end NUMINAMATH_CALUDE_margo_age_in_three_years_l116_11640


namespace NUMINAMATH_CALUDE_sequence_proof_l116_11626

theorem sequence_proof (a : Fin 8 → ℕ) 
  (h1 : a 0 = 11)
  (h2 : a 7 = 12)
  (h3 : ∀ i : Fin 6, a i + a (i + 1) + a (i + 2) = 50) :
  a = ![11, 12, 27, 11, 12, 27, 11, 12] := by
sorry

end NUMINAMATH_CALUDE_sequence_proof_l116_11626


namespace NUMINAMATH_CALUDE_square_plus_self_even_l116_11694

theorem square_plus_self_even (n : ℤ) : Even (n^2 + n) := by sorry

end NUMINAMATH_CALUDE_square_plus_self_even_l116_11694


namespace NUMINAMATH_CALUDE_sum_of_coefficients_abs_l116_11613

theorem sum_of_coefficients_abs (a₀ a₁ a₂ a₃ a₄ a₅ a₆ : ℤ) :
  (∀ x, (2 - x)^6 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5 + a₆*x^6) →
  |a₁| + |a₂| + |a₃| + |a₄| + |a₅| + |a₆| = 665 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_abs_l116_11613
