import Mathlib

namespace NUMINAMATH_CALUDE_cameron_house_paintable_area_l473_47393

/-- Calculates the total paintable area of walls in multiple bedrooms -/
def total_paintable_area (num_bedrooms : ℕ) (length width height : ℝ) (unpaintable_area : ℝ) : ℝ :=
  let wall_area := 2 * (length * height + width * height)
  let paintable_area := wall_area - unpaintable_area
  num_bedrooms * paintable_area

/-- Theorem stating that the total paintable area of walls in Cameron's house is 1840 square feet -/
theorem cameron_house_paintable_area :
  total_paintable_area 4 15 12 10 80 = 1840 := by
  sorry

#eval total_paintable_area 4 15 12 10 80

end NUMINAMATH_CALUDE_cameron_house_paintable_area_l473_47393


namespace NUMINAMATH_CALUDE_brads_running_speed_l473_47383

/-- Prove that Brad's running speed is 6 km/h given the conditions of the problem -/
theorem brads_running_speed (maxwell_speed : ℝ) (total_distance : ℝ) (brad_delay : ℝ) (total_time : ℝ)
  (h1 : maxwell_speed = 4)
  (h2 : total_distance = 74)
  (h3 : brad_delay = 1)
  (h4 : total_time = 8) :
  (total_distance - maxwell_speed * total_time) / (total_time - brad_delay) = 6 := by
  sorry

end NUMINAMATH_CALUDE_brads_running_speed_l473_47383


namespace NUMINAMATH_CALUDE_sled_distance_l473_47349

/-- Represents the distance traveled by a sled in a given second -/
def distance_in_second (n : ℕ) : ℕ := 6 + (n - 1) * 8

/-- Calculates the total distance traveled by the sled over a given number of seconds -/
def total_distance (seconds : ℕ) : ℕ :=
  (seconds * (distance_in_second 1 + distance_in_second seconds)) / 2

/-- Theorem stating that a sled sliding for 20 seconds travels 1640 inches -/
theorem sled_distance : total_distance 20 = 1640 := by
  sorry

end NUMINAMATH_CALUDE_sled_distance_l473_47349


namespace NUMINAMATH_CALUDE_min_colours_for_cube_l473_47308

/-- Represents a colouring of a cube's faces -/
def CubeColouring := Fin 6 → ℕ

/-- Checks if two face indices are adjacent on a cube -/
def are_adjacent (i j : Fin 6) : Prop :=
  (i.val + j.val) % 2 = 1 ∧ i ≠ j

/-- A valid colouring has different colours for adjacent faces -/
def is_valid_colouring (c : CubeColouring) : Prop :=
  ∀ i j : Fin 6, are_adjacent i j → c i ≠ c j

/-- The number of colours used in a colouring -/
def num_colours (c : CubeColouring) : ℕ :=
  Finset.card (Finset.image c Finset.univ)

/-- There exists a valid 3-colouring of a cube -/
axiom exists_valid_3_colouring : ∃ c : CubeColouring, is_valid_colouring c ∧ num_colours c = 3

/-- Any valid colouring of a cube uses at least 3 colours -/
axiom valid_colouring_needs_at_least_3 : ∀ c : CubeColouring, is_valid_colouring c → num_colours c ≥ 3

theorem min_colours_for_cube : ∃ n : ℕ, n = 3 ∧
  (∃ c : CubeColouring, is_valid_colouring c ∧ num_colours c = n) ∧
  (∀ c : CubeColouring, is_valid_colouring c → num_colours c ≥ n) :=
sorry

end NUMINAMATH_CALUDE_min_colours_for_cube_l473_47308


namespace NUMINAMATH_CALUDE_cooking_time_for_remaining_potatoes_l473_47309

/-- Given a chef cooking potatoes with the following conditions:
  - The total number of potatoes to cook is 16
  - The number of potatoes already cooked is 7
  - Each potato takes 5 minutes to cook
  Prove that the time required to cook the remaining potatoes is 45 minutes. -/
theorem cooking_time_for_remaining_potatoes :
  ∀ (total_potatoes cooked_potatoes cooking_time_per_potato : ℕ),
    total_potatoes = 16 →
    cooked_potatoes = 7 →
    cooking_time_per_potato = 5 →
    (total_potatoes - cooked_potatoes) * cooking_time_per_potato = 45 :=
by sorry

end NUMINAMATH_CALUDE_cooking_time_for_remaining_potatoes_l473_47309


namespace NUMINAMATH_CALUDE_fixed_point_implies_stable_point_exists_stable_point_not_fixed_point_l473_47342

/-- A monotonically decreasing function -/
def MonoDecreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f y < f x

/-- Definition of a fixed point -/
def IsFixedPoint (f : ℝ → ℝ) (x : ℝ) : Prop :=
  f x = x

/-- Definition of a stable point -/
def IsStablePoint (f : ℝ → ℝ) (x : ℝ) : Prop :=
  f x = Function.invFun f x

theorem fixed_point_implies_stable_point
    (f : ℝ → ℝ) (hf : MonoDecreasing f) (x : ℝ) :
    IsFixedPoint f x → IsStablePoint f x :=
  sorry

theorem exists_stable_point_not_fixed_point
    (f : ℝ → ℝ) (hf : MonoDecreasing f) :
    ∃ x, IsStablePoint f x ∧ ¬IsFixedPoint f x :=
  sorry

end NUMINAMATH_CALUDE_fixed_point_implies_stable_point_exists_stable_point_not_fixed_point_l473_47342


namespace NUMINAMATH_CALUDE_letters_identity_l473_47329

-- Define the Letter type
inductive Letter
| A
| B

-- Define a function to represent whether a letter tells the truth
def tellsTruth (l : Letter) : Bool :=
  match l with
  | Letter.A => true
  | Letter.B => false

-- Define the statements made by each letter
def statement1 (l1 l2 l3 : Letter) : Prop :=
  (l1 = l2 ∧ l1 ≠ l3) ∨ (l1 = l3 ∧ l1 ≠ l2)

def statement2 (l1 l2 l3 : Letter) : Prop :=
  (l1 = Letter.A ∧ l2 = Letter.A ∧ l3 = Letter.B) ∨
  (l1 = Letter.A ∧ l2 = Letter.B ∧ l3 = Letter.B) ∨
  (l1 = Letter.B ∧ l2 = Letter.A ∧ l3 = Letter.B) ∨
  (l1 = Letter.B ∧ l2 = Letter.B ∧ l3 = Letter.B)

def statement3 (l1 l2 l3 : Letter) : Prop :=
  (l1 = Letter.B ∧ l2 ≠ Letter.B ∧ l3 ≠ Letter.B) ∨
  (l1 ≠ Letter.B ∧ l2 = Letter.B ∧ l3 ≠ Letter.B) ∨
  (l1 ≠ Letter.B ∧ l2 ≠ Letter.B ∧ l3 = Letter.B)

-- Define the main theorem
theorem letters_identity :
  ∃! (l1 l2 l3 : Letter),
    (tellsTruth l1 = statement1 l1 l2 l3) ∧
    (tellsTruth l2 = statement2 l1 l2 l3) ∧
    (tellsTruth l3 = statement3 l1 l2 l3) ∧
    l1 = Letter.B ∧ l2 = Letter.A ∧ l3 = Letter.A :=
by sorry

end NUMINAMATH_CALUDE_letters_identity_l473_47329


namespace NUMINAMATH_CALUDE_volume_for_56_ounces_l473_47331

/-- A substance with volume directly proportional to weight -/
structure Substance where
  /-- Constant of proportionality between volume and weight -/
  k : ℚ
  /-- Assumption that k is positive -/
  k_pos : k > 0

/-- The volume of the substance given its weight -/
def volume (s : Substance) (weight : ℚ) : ℚ :=
  s.k * weight

theorem volume_for_56_ounces (s : Substance) 
  (h : volume s 112 = 48) : volume s 56 = 24 := by
  sorry

#check volume_for_56_ounces

end NUMINAMATH_CALUDE_volume_for_56_ounces_l473_47331


namespace NUMINAMATH_CALUDE_square_and_cube_sum_l473_47359

theorem square_and_cube_sum (p q : ℝ) (h1 : p * q = 8) (h2 : p + q = 7) :
  p^2 + q^2 = 33 ∧ p^3 + q^3 = 175 := by
  sorry

end NUMINAMATH_CALUDE_square_and_cube_sum_l473_47359


namespace NUMINAMATH_CALUDE_quadratic_inequality_l473_47378

theorem quadratic_inequality (x : ℝ) : x^2 - 5*x + 6 < 0 ↔ 2 < x ∧ x < 3 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_l473_47378


namespace NUMINAMATH_CALUDE_cube_with_tunnel_surface_area_l473_47319

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a cube with a tunnel drilled through it -/
structure CubeWithTunnel where
  /-- Length of the cube's edge -/
  edgeLength : ℝ
  /-- Point P, a corner of the cube -/
  p : Point3D
  /-- Point L on PQ -/
  l : Point3D
  /-- Point M on PR -/
  m : Point3D
  /-- Point N on PC -/
  n : Point3D

/-- The surface area of the cube with tunnel can be expressed as x + y√z -/
def surfaceAreaExpression (c : CubeWithTunnel) : ℕ × ℕ × ℕ :=
  sorry

theorem cube_with_tunnel_surface_area 
  (c : CubeWithTunnel)
  (h1 : c.edgeLength = 10)
  (h2 : c.p.x = 10 ∧ c.p.y = 10 ∧ c.p.z = 10)
  (h3 : c.l.x = 7.5 ∧ c.l.y = 10 ∧ c.l.z = 10)
  (h4 : c.m.x = 10 ∧ c.m.y = 7.5 ∧ c.m.z = 10)
  (h5 : c.n.x = 10 ∧ c.n.y = 10 ∧ c.n.z = 7.5) :
  let (x, y, z) := surfaceAreaExpression c
  x + y + z = 639 ∧ 
  (∀ p : ℕ, Prime p → ¬(p^2 ∣ z)) :=
sorry

end NUMINAMATH_CALUDE_cube_with_tunnel_surface_area_l473_47319


namespace NUMINAMATH_CALUDE_milk_problem_l473_47396

theorem milk_problem (M : ℝ) : 
  M > 0 → 
  (1 - 2/3) * (1 - 2/5) * (1 - 1/6) * M = 120 → 
  M = 720 :=
by
  sorry

end NUMINAMATH_CALUDE_milk_problem_l473_47396


namespace NUMINAMATH_CALUDE_problem_solution_l473_47379

theorem problem_solution (m n a : ℝ) : 
  (m^2 - 2*m - 1 = 0) →
  (n^2 - 2*n - 1 = 0) →
  (7*m^2 - 14*m + a)*(3*n^2 - 6*n - 7) = 8 →
  a = -9 := by sorry

end NUMINAMATH_CALUDE_problem_solution_l473_47379


namespace NUMINAMATH_CALUDE_sum_of_fourth_powers_less_than_150_l473_47316

theorem sum_of_fourth_powers_less_than_150 : 
  (Finset.filter (fun n : ℕ => n^4 < 150) (Finset.range 150)).sum id = 98 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_fourth_powers_less_than_150_l473_47316


namespace NUMINAMATH_CALUDE_unique_valid_number_l473_47350

def is_valid_number (abc : ℕ) : Prop :=
  let a := abc / 100
  let b := (abc / 10) % 10
  let c := abc % 10
  let sum := a + b + c
  abc % sum = 1 ∧
  (c * 100 + b * 10 + a) % sum = 1 ∧
  a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
  a > c

theorem unique_valid_number : ∃! abc : ℕ, 100 ≤ abc ∧ abc < 1000 ∧ is_valid_number abc :=
  sorry

end NUMINAMATH_CALUDE_unique_valid_number_l473_47350


namespace NUMINAMATH_CALUDE_sony_johnny_fish_ratio_l473_47324

def total_fishes : ℕ := 40
def johnny_fishes : ℕ := 8

theorem sony_johnny_fish_ratio :
  (total_fishes - johnny_fishes) / johnny_fishes = 4 := by
  sorry

end NUMINAMATH_CALUDE_sony_johnny_fish_ratio_l473_47324


namespace NUMINAMATH_CALUDE_baker_cookies_total_l473_47304

theorem baker_cookies_total (chocolate_chip_batches oatmeal_batches : ℕ)
  (chocolate_chip_per_batch oatmeal_per_batch : ℕ)
  (sugar_cookies double_chocolate_cookies : ℕ) :
  chocolate_chip_batches = 5 →
  oatmeal_batches = 3 →
  chocolate_chip_per_batch = 8 →
  oatmeal_per_batch = 7 →
  sugar_cookies = 10 →
  double_chocolate_cookies = 6 →
  chocolate_chip_batches * chocolate_chip_per_batch +
  oatmeal_batches * oatmeal_per_batch +
  sugar_cookies + double_chocolate_cookies = 77 :=
by sorry

end NUMINAMATH_CALUDE_baker_cookies_total_l473_47304


namespace NUMINAMATH_CALUDE_base_7_321_equals_162_l473_47376

def base_7_to_10 (a b c : ℕ) : ℕ := a * 7^2 + b * 7 + c

theorem base_7_321_equals_162 : base_7_to_10 3 2 1 = 162 := by
  sorry

end NUMINAMATH_CALUDE_base_7_321_equals_162_l473_47376


namespace NUMINAMATH_CALUDE_complex_multiplication_l473_47362

theorem complex_multiplication (i : ℂ) :
  i * i = -1 →
  (-1 + i) * (2 - i) = -1 + 3 * i :=
by sorry

end NUMINAMATH_CALUDE_complex_multiplication_l473_47362


namespace NUMINAMATH_CALUDE_school_population_theorem_l473_47314

theorem school_population_theorem :
  ∀ (boys girls : ℕ),
  boys + girls = 150 →
  girls = (boys * 100) / 150 →
  boys = 90 := by
sorry

end NUMINAMATH_CALUDE_school_population_theorem_l473_47314


namespace NUMINAMATH_CALUDE_complex_equation_solution_l473_47374

theorem complex_equation_solution (z : ℂ) 
  (h1 : Complex.abs (1 - z) + z = 10 - 3*I) :
  ∃ (m n : ℝ), 
    z = 5 - 3*I ∧ 
    z^2 + m*z + n = 1 - 3*I ∧ 
    m = -9 ∧ 
    n = 30 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l473_47374


namespace NUMINAMATH_CALUDE_power_equality_l473_47336

theorem power_equality (k : ℕ) : 9^4 = 3^k → k = 8 := by
  sorry

end NUMINAMATH_CALUDE_power_equality_l473_47336


namespace NUMINAMATH_CALUDE_count_nonzero_monomials_l473_47368

/-- The number of monomials with non-zero coefficients in the expansion of (x+y+z)^2028 + (x-y-z)^2028 -/
def num_nonzero_monomials : ℕ := 1030225

/-- The exponent in the given expression -/
def exponent : ℕ := 2028

theorem count_nonzero_monomials :
  num_nonzero_monomials = (exponent / 2 + 1)^2 := by sorry

end NUMINAMATH_CALUDE_count_nonzero_monomials_l473_47368


namespace NUMINAMATH_CALUDE_probability_at_least_one_male_doctor_l473_47358

/-- The probability of selecting at least one male doctor when choosing 3 doctors from 4 female and 3 male doctors. -/
theorem probability_at_least_one_male_doctor : 
  let total_doctors : ℕ := 7
  let female_doctors : ℕ := 4
  let male_doctors : ℕ := 3
  let doctors_to_select : ℕ := 3
  let total_combinations := Nat.choose total_doctors doctors_to_select
  let favorable_outcomes := 
    Nat.choose male_doctors 1 * Nat.choose female_doctors 2 +
    Nat.choose male_doctors 2 * Nat.choose female_doctors 1 +
    Nat.choose male_doctors 3
  (favorable_outcomes : ℚ) / total_combinations = 31 / 35 :=
by sorry

end NUMINAMATH_CALUDE_probability_at_least_one_male_doctor_l473_47358


namespace NUMINAMATH_CALUDE_quadratic_touch_existence_l473_47351

theorem quadratic_touch_existence (p q r : ℤ) : 
  (∃ x : ℝ, p * x^2 + q * x + r = 0 ∧ ∀ y : ℝ, p * y^2 + q * y + r ≥ 0) →
  ∃ a b : ℤ, 
    (∃ x : ℝ, p * x^2 + q * x + r = (b : ℝ)) ∧
    (∃ x : ℝ, x^2 + (a : ℝ) * x + (b : ℝ) = 0 ∧ ∀ y : ℝ, y^2 + (a : ℝ) * y + (b : ℝ) ≥ 0) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_touch_existence_l473_47351


namespace NUMINAMATH_CALUDE_simplify_expression_l473_47341

theorem simplify_expression : 
  5 * Real.sqrt 3 + (Real.sqrt 4 + 2 * Real.sqrt 3) = 7 * Real.sqrt 3 + 2 :=
by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l473_47341


namespace NUMINAMATH_CALUDE_equation_one_solutions_equation_two_solutions_l473_47327

-- Equation 1
theorem equation_one_solutions (x : ℝ) : 
  x * (x - 2) = x - 2 ↔ x = 1 ∨ x = 2 := by sorry

-- Equation 2
theorem equation_two_solutions (x : ℝ) : 
  x^2 - 6*x + 5 = 0 ↔ x = 1 ∨ x = 5 := by sorry

end NUMINAMATH_CALUDE_equation_one_solutions_equation_two_solutions_l473_47327


namespace NUMINAMATH_CALUDE_perfect_square_equation_l473_47373

theorem perfect_square_equation (a b c : ℤ) 
  (h : (a - 5)^2 + (b - 12)^2 - (c - 13)^2 = a^2 + b^2 - c^2) : 
  ∃ k : ℤ, a^2 + b^2 - c^2 = k^2 := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_equation_l473_47373


namespace NUMINAMATH_CALUDE_number_thought_of_l473_47335

theorem number_thought_of : ∃ x : ℝ, (x / 6) + 5 = 17 := by
  sorry

end NUMINAMATH_CALUDE_number_thought_of_l473_47335


namespace NUMINAMATH_CALUDE_garden_ratio_l473_47363

/-- Represents a rectangular garden -/
structure RectangularGarden where
  width : ℝ
  length : ℝ
  area : ℝ

/-- Theorem: For a rectangular garden with area 675 sq meters and width 15 meters, 
    the ratio of length to width is 3:1 -/
theorem garden_ratio (g : RectangularGarden) 
    (h1 : g.area = 675)
    (h2 : g.width = 15)
    (h3 : g.area = g.length * g.width) :
  g.length / g.width = 3 := by
  sorry

#check garden_ratio

end NUMINAMATH_CALUDE_garden_ratio_l473_47363


namespace NUMINAMATH_CALUDE_gcd_459_357_l473_47377

def euclidean_gcd (a b : ℕ) : ℕ := sorry

def successive_subtraction_gcd (a b : ℕ) : ℕ := sorry

theorem gcd_459_357 : 
  euclidean_gcd 459 357 = 51 ∧ 
  successive_subtraction_gcd 459 357 = 51 := by sorry

end NUMINAMATH_CALUDE_gcd_459_357_l473_47377


namespace NUMINAMATH_CALUDE_lisas_marbles_problem_l473_47338

/-- The minimum number of additional marbles needed -/
def min_additional_marbles (num_friends : ℕ) (initial_marbles : ℕ) : ℕ :=
  (num_friends * (num_friends + 1)) / 2 - initial_marbles

/-- Theorem stating the minimum number of additional marbles needed for Lisa's problem -/
theorem lisas_marbles_problem :
  min_additional_marbles 12 40 = 38 := by
  sorry

#eval min_additional_marbles 12 40

end NUMINAMATH_CALUDE_lisas_marbles_problem_l473_47338


namespace NUMINAMATH_CALUDE_intersection_area_l473_47305

theorem intersection_area (f g : ℝ → ℝ) (P Q : ℝ × ℝ) (B A : ℝ × ℝ) :
  (f = λ x => 2 * Real.cos (3 * x) + 1) →
  (g = λ x => - Real.cos (2 * x)) →
  (∃ x₁ x₂, 17 * π / 4 < x₁ ∧ x₁ < 21 * π / 4 ∧
            17 * π / 4 < x₂ ∧ x₂ < 21 * π / 4 ∧
            P = (x₁, f x₁) ∧ Q = (x₂, f x₂) ∧
            f x₁ = g x₁ ∧ f x₂ = g x₂) →
  (∃ m b, ∀ x, P.2 + m * (x - P.1) = b * x) →
  B.2 = 0 →
  A.1 = 0 →
  (area_triangle : ℝ × ℝ → ℝ × ℝ → ℝ × ℝ → ℝ) →
  area_triangle (0, 0) B A = 361 * π / 8 := by
sorry

end NUMINAMATH_CALUDE_intersection_area_l473_47305


namespace NUMINAMATH_CALUDE_farm_fencing_cost_l473_47355

/-- Calculates the cost of fencing a rectangular farm -/
theorem farm_fencing_cost 
  (area : ℝ) 
  (short_side : ℝ) 
  (cost_per_meter : ℝ) 
  (h_area : area = 1200) 
  (h_short : short_side = 30) 
  (h_cost : cost_per_meter = 14) : 
  let long_side := area / short_side
  let diagonal := Real.sqrt (long_side^2 + short_side^2)
  let total_length := long_side + short_side + diagonal
  cost_per_meter * total_length = 1680 :=
by sorry

end NUMINAMATH_CALUDE_farm_fencing_cost_l473_47355


namespace NUMINAMATH_CALUDE_product_of_solutions_with_positive_real_part_l473_47354

theorem product_of_solutions_with_positive_real_part (x : ℂ) : 
  (x^8 = -256) → 
  (∃ (S : Finset ℂ), 
    (∀ z ∈ S, z^8 = -256 ∧ z.re > 0) ∧ 
    (∀ z, z^8 = -256 ∧ z.re > 0 → z ∈ S) ∧ 
    (S.prod id = 8)) :=
by sorry

end NUMINAMATH_CALUDE_product_of_solutions_with_positive_real_part_l473_47354


namespace NUMINAMATH_CALUDE_system_solution_l473_47344

theorem system_solution (x y z : ℝ) 
  (eq1 : x + 3*y = 20)
  (eq2 : x + y + z = 25)
  (eq3 : x - z = 5) :
  x = 14 ∧ y = 2 ∧ z = 9 := by
sorry

end NUMINAMATH_CALUDE_system_solution_l473_47344


namespace NUMINAMATH_CALUDE_marks_lost_is_one_l473_47380

/-- Represents an examination with given parameters -/
structure Examination where
  total_questions : ℕ
  marks_per_correct : ℕ
  total_score : ℕ
  correct_answers : ℕ

/-- Calculates the marks lost per wrong answer -/
def marks_lost_per_wrong (exam : Examination) : ℚ :=
  (exam.marks_per_correct * exam.correct_answers - exam.total_score) / (exam.total_questions - exam.correct_answers)

/-- Theorem stating that for the given examination parameters, 
    the marks lost per wrong answer is 1 -/
theorem marks_lost_is_one (exam : Examination) 
  (h1 : exam.total_questions = 60)
  (h2 : exam.marks_per_correct = 4)
  (h3 : exam.total_score = 140)
  (h4 : exam.correct_answers = 40) : 
  marks_lost_per_wrong exam = 1 := by
  sorry

#eval marks_lost_per_wrong ⟨60, 4, 140, 40⟩

end NUMINAMATH_CALUDE_marks_lost_is_one_l473_47380


namespace NUMINAMATH_CALUDE_triangle_area_with_arithmetic_sides_l473_47315

/-- Given a triangle ABC with one angle of 120° and sides in arithmetic progression with common difference 2, its area is 15√3/4 -/
theorem triangle_area_with_arithmetic_sides : ∀ (a b c : ℝ),
  0 < a ∧ 0 < b ∧ 0 < c →
  ∃ (θ : ℝ), θ = 2 * π / 3 →
  ∃ (d : ℝ), d = 2 →
  b = a + d ∧ c = b + d →
  c^2 = a^2 + b^2 - 2*a*b*Real.cos θ →
  (1/2) * a * b * Real.sin θ = 15 * Real.sqrt 3 / 4 := by
  sorry


end NUMINAMATH_CALUDE_triangle_area_with_arithmetic_sides_l473_47315


namespace NUMINAMATH_CALUDE_range_of_a_l473_47390

open Set
open Real

theorem range_of_a (a : ℝ) : 
  (¬ ∃ x₀ : ℝ, x₀^2 - a*x₀ + a = 0) →
  (∀ x : ℝ, x > 1 → x + 1/(x-1) ≥ a) →
  a ∈ Ioo 0 3 := by
sorry

end NUMINAMATH_CALUDE_range_of_a_l473_47390


namespace NUMINAMATH_CALUDE_max_a_fourth_quadrant_l473_47307

theorem max_a_fourth_quadrant (a : ℤ) : 
  let z : ℂ := (2 + a * Complex.I) / (1 + 2 * Complex.I)
  (0 < z.re ∧ z.im < 0) → a ≤ 3 ∧ ∃ (b : ℤ), b ≤ 3 ∧ 
    let w : ℂ := (2 + b * Complex.I) / (1 + 2 * Complex.I)
    0 < w.re ∧ w.im < 0 := by
  sorry

end NUMINAMATH_CALUDE_max_a_fourth_quadrant_l473_47307


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_relation_l473_47312

def geometric_sequence (n : ℕ) : ℝ := 2^(n-1)

def sum_geometric_sequence (n : ℕ) : ℝ := 2^n - 1

theorem geometric_sequence_sum_relation (n : ℕ) :
  sum_geometric_sequence n = 2 * geometric_sequence n - 1 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_relation_l473_47312


namespace NUMINAMATH_CALUDE_price_reduction_equation_l473_47303

/-- Theorem: For an item with an original price of 289 yuan and a final price of 256 yuan
    after two consecutive price reductions, where x represents the average percentage
    reduction each time, the equation 289(1-x)^2 = 256 holds true. -/
theorem price_reduction_equation (x : ℝ) 
  (h1 : 0 ≤ x) (h2 : x < 1) : 289 * (1 - x)^2 = 256 := by
  sorry

#check price_reduction_equation

end NUMINAMATH_CALUDE_price_reduction_equation_l473_47303


namespace NUMINAMATH_CALUDE_complex_division_equality_l473_47388

theorem complex_division_equality : ∀ (i : ℂ), i^2 = -1 →
  (3 - 2*i) / (2 + i) = 4/5 - 7/5*i :=
by sorry

end NUMINAMATH_CALUDE_complex_division_equality_l473_47388


namespace NUMINAMATH_CALUDE_parallel_line_difference_l473_47367

/-- Given two points (-1, q) and (-3, r) on a line parallel to y = (3/2)x + 1, 
    prove that r - q = -3 -/
theorem parallel_line_difference (q r : ℝ) : 
  (∃ (m b : ℝ), m = 3/2 ∧ 
    (∀ (x y : ℝ), y = m * x + b ↔ (x = -1 ∧ y = q) ∨ (x = -3 ∧ y = r))) →
  r - q = -3 :=
by sorry

end NUMINAMATH_CALUDE_parallel_line_difference_l473_47367


namespace NUMINAMATH_CALUDE_smallest_a_for_sqrt_50a_l473_47397

theorem smallest_a_for_sqrt_50a (a : ℕ) : (∃ k : ℕ, k^2 = 50 * a) → a ≥ 2 :=
sorry

end NUMINAMATH_CALUDE_smallest_a_for_sqrt_50a_l473_47397


namespace NUMINAMATH_CALUDE_rectangle_width_equal_square_side_l473_47364

theorem rectangle_width_equal_square_side 
  (square_side : ℝ) 
  (rect_length : ℝ) 
  (h1 : square_side = 3)
  (h2 : rect_length = 3)
  (h3 : square_side * square_side = rect_length * (square_side)) :
  square_side = rect_length :=
by sorry

end NUMINAMATH_CALUDE_rectangle_width_equal_square_side_l473_47364


namespace NUMINAMATH_CALUDE_min_value_arithmetic_seq_l473_47369

/-- An arithmetic sequence with positive terms and a_4 = 5 -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  (∀ n, a n > 0) ∧
  (∃ d, ∀ n, a (n + 1) = a n + d) ∧
  a 4 = 5

/-- The minimum value of 1/a_2 + 16/a_6 for the given arithmetic sequence -/
theorem min_value_arithmetic_seq (a : ℕ → ℝ) (h : ArithmeticSequence a) :
    (∀ n, a n > 0) → (1 / a 2 + 16 / a 6) ≥ 5/2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_arithmetic_seq_l473_47369


namespace NUMINAMATH_CALUDE_andrews_age_l473_47348

theorem andrews_age :
  ∀ (a g : ℚ),
  g = 15 * a →
  g - a = 55 →
  a = 55 / 14 := by
sorry

end NUMINAMATH_CALUDE_andrews_age_l473_47348


namespace NUMINAMATH_CALUDE_rectangle_longer_side_l473_47375

/-- Given a rectangular plot with one side of 10 meters, where fence poles are placed 5 meters apart
    and 24 poles are needed in total, the length of the longer side is 40 meters. -/
theorem rectangle_longer_side (width : ℝ) (length : ℝ) (poles : ℕ) :
  width = 10 →
  poles = 24 →
  (2 * width + 2 * length) / 5 = poles →
  length = 40 :=
by sorry

end NUMINAMATH_CALUDE_rectangle_longer_side_l473_47375


namespace NUMINAMATH_CALUDE_men_entered_room_l473_47384

theorem men_entered_room (initial_men : ℕ) (initial_women : ℕ) (men_entered : ℕ) : 
  (initial_men : ℚ) / initial_women = 4 / 5 →
  2 * (initial_women - 3) = 24 →
  initial_men + men_entered = 14 →
  men_entered = 2 := by
  sorry

end NUMINAMATH_CALUDE_men_entered_room_l473_47384


namespace NUMINAMATH_CALUDE_lens_discount_percentage_l473_47352

theorem lens_discount_percentage (original_price : ℝ) (discounted_price : ℝ) (saving : ℝ) :
  original_price = 300 ∧ 
  discounted_price = 220 ∧ 
  saving = 20 →
  (original_price - (discounted_price + saving)) / original_price * 100 = 20 := by
  sorry

end NUMINAMATH_CALUDE_lens_discount_percentage_l473_47352


namespace NUMINAMATH_CALUDE_reservoir_fullness_after_storm_l473_47372

theorem reservoir_fullness_after_storm 
  (original_content : ℝ) 
  (original_percentage : ℝ) 
  (storm_deposit : ℝ) 
  (h1 : original_content = 245)
  (h2 : original_percentage = 54.44444444444444)
  (h3 : storm_deposit = 115) :
  let total_capacity := original_content / (original_percentage / 100)
  let new_content := original_content + storm_deposit
  (new_content / total_capacity) * 100 = 80 := by
sorry

end NUMINAMATH_CALUDE_reservoir_fullness_after_storm_l473_47372


namespace NUMINAMATH_CALUDE_kevin_kangaroo_four_hops_l473_47323

def hop_distance (remaining : ℚ) : ℚ := (1 / 4) * remaining

def total_distance (n : ℕ) : ℚ :=
  let goal := 2
  let rec distance_after_hops (k : ℕ) (remaining : ℚ) (acc : ℚ) : ℚ :=
    if k = 0 then acc
    else
      let hop := hop_distance remaining
      distance_after_hops (k - 1) (remaining - hop) (acc + hop)
  distance_after_hops n goal 0

theorem kevin_kangaroo_four_hops :
  total_distance 4 = 175 / 128 := by sorry

end NUMINAMATH_CALUDE_kevin_kangaroo_four_hops_l473_47323


namespace NUMINAMATH_CALUDE_silas_payment_ratio_l473_47333

theorem silas_payment_ratio (total_bill : ℚ) (friend_payment : ℚ) 
  (h1 : total_bill = 150)
  (h2 : friend_payment = 18)
  (h3 : (5 : ℚ) * friend_payment + (total_bill / 10) = total_bill + (total_bill / 10) - (total_bill / 2)) :
  (total_bill / 2) / total_bill = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_silas_payment_ratio_l473_47333


namespace NUMINAMATH_CALUDE_tangent_problem_l473_47328

theorem tangent_problem (α β : Real) 
  (h1 : Real.tan (α + β) = 2/5)
  (h2 : Real.tan (β + Real.pi/4) = 1/4) :
  Real.tan (α - Real.pi/4) = 3/22 := by
  sorry

end NUMINAMATH_CALUDE_tangent_problem_l473_47328


namespace NUMINAMATH_CALUDE_quadratic_intersection_points_specific_quadratic_roots_l473_47387

theorem quadratic_intersection_points (a b c : ℝ) (h : a ≠ 0) :
  (∃ x y : ℝ, x ≠ y ∧ a * x^2 + b * x + c = 0 ∧ a * y^2 + b * y + c = 0) ↔
  b^2 - 4*a*c > 0 :=
by sorry

theorem specific_quadratic_roots :
  ∃ x y : ℝ, x ≠ y ∧ 2 * x^2 + 3 * x - 2 = 0 ∧ 2 * y^2 + 3 * y - 2 = 0 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_intersection_points_specific_quadratic_roots_l473_47387


namespace NUMINAMATH_CALUDE_infinite_triples_with_gcd_one_l473_47382

theorem infinite_triples_with_gcd_one (m n : ℕ+) :
  ∃ (a b c : ℕ+),
    a = m^2 + m * n + n^2 ∧
    b = m^2 - m * n ∧
    c = n^2 - m * n ∧
    Nat.gcd a (Nat.gcd b c) = 1 ∧
    a^2 = b^2 + c^2 + b * c :=
by sorry

end NUMINAMATH_CALUDE_infinite_triples_with_gcd_one_l473_47382


namespace NUMINAMATH_CALUDE_rachel_reading_homework_l473_47337

/-- The number of pages of reading homework Rachel had to complete -/
def reading_pages : ℕ := by sorry

/-- The number of pages of math homework Rachel had to complete -/
def math_pages : ℕ := 4

/-- The relationship between math and reading homework pages -/
axiom math_reading_relation : math_pages = reading_pages + 2

theorem rachel_reading_homework : reading_pages = 2 := by sorry

end NUMINAMATH_CALUDE_rachel_reading_homework_l473_47337


namespace NUMINAMATH_CALUDE_fraction_simplification_l473_47392

theorem fraction_simplification (x y : ℝ) (hx : x = 3) (hy : y = 2) :
  (x^8 + 2*x^4*y^2 + y^4) / (x^4 + y^2) = 85 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l473_47392


namespace NUMINAMATH_CALUDE_class_group_size_l473_47394

theorem class_group_size (boys girls groups : ℕ) 
  (h_boys : boys = 9) 
  (h_girls : girls = 12) 
  (h_groups : groups = 7) : 
  (boys + girls) / groups = 3 := by
sorry

end NUMINAMATH_CALUDE_class_group_size_l473_47394


namespace NUMINAMATH_CALUDE_factor_t_squared_minus_64_l473_47360

theorem factor_t_squared_minus_64 (t : ℝ) : t^2 - 64 = (t - 8) * (t + 8) := by
  sorry

end NUMINAMATH_CALUDE_factor_t_squared_minus_64_l473_47360


namespace NUMINAMATH_CALUDE_connie_grandmother_brother_birth_year_l473_47301

/-- The year Connie's grandmother's older brother was born -/
def older_brother_birth_year : ℕ := sorry

/-- The year Connie's grandmother's older sister was born -/
def older_sister_birth_year : ℕ := 1936

/-- The year Connie's grandmother was born -/
def grandmother_birth_year : ℕ := 1944

theorem connie_grandmother_brother_birth_year :
  (grandmother_birth_year - older_sister_birth_year = 2 * (older_sister_birth_year - older_brother_birth_year)) →
  older_brother_birth_year = 1932 := by
  sorry

end NUMINAMATH_CALUDE_connie_grandmother_brother_birth_year_l473_47301


namespace NUMINAMATH_CALUDE_circle_equation_l473_47395

/-- The standard equation of a circle with center (-3, 4) and radius 2 is (x+3)^2 + (y-4)^2 = 4 -/
theorem circle_equation (x y : ℝ) : 
  let center : ℝ × ℝ := (-3, 4)
  let radius : ℝ := 2
  (x + 3)^2 + (y - 4)^2 = 4 ↔ 
    ((x - center.1)^2 + (y - center.2)^2 = radius^2) :=
by sorry

end NUMINAMATH_CALUDE_circle_equation_l473_47395


namespace NUMINAMATH_CALUDE_pyramid_edges_cannot_form_closed_polygon_l473_47389

/-- Represents a line segment in 3D space -/
structure Segment3D where
  parallel_to_plane : Bool

/-- Represents a collection of line segments in 3D space -/
structure SegmentCollection where
  segments : List Segment3D
  parallel_count : Nat
  non_parallel_count : Nat

/-- Checks if a collection of segments can form a closed polygon -/
def can_form_closed_polygon (collection : SegmentCollection) : Prop :=
  collection.parallel_count = collection.non_parallel_count ∧
  collection.parallel_count + collection.non_parallel_count = collection.segments.length

theorem pyramid_edges_cannot_form_closed_polygon :
  ¬ ∃ (collection : SegmentCollection),
    collection.parallel_count = 171 ∧
    collection.non_parallel_count = 171 ∧
    can_form_closed_polygon collection :=
by sorry

end NUMINAMATH_CALUDE_pyramid_edges_cannot_form_closed_polygon_l473_47389


namespace NUMINAMATH_CALUDE_distance_center_to_point_l473_47332

-- Define the circle equation
def circle_equation (x y : ℝ) : Prop :=
  x^2 + y^2 = 6*x + 10*y + 9

-- Define the center of the circle
def circle_center : ℝ × ℝ :=
  let a := 3
  let b := 5
  (a, b)

-- Define the given point
def given_point : ℝ × ℝ := (-4, -2)

-- Theorem statement
theorem distance_center_to_point :
  let (cx, cy) := circle_center
  let (px, py) := given_point
  Real.sqrt ((cx - px)^2 + (cy - py)^2) = 7 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_distance_center_to_point_l473_47332


namespace NUMINAMATH_CALUDE_sum_of_roots_eq_six_l473_47311

theorem sum_of_roots_eq_six : 
  let f : ℝ → ℝ := λ x => (x - 3)^2 - 16
  ∃ x₁ x₂ : ℝ, (f x₁ = 0 ∧ f x₂ = 0 ∧ x₁ ≠ x₂) ∧ x₁ + x₂ = 6 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_roots_eq_six_l473_47311


namespace NUMINAMATH_CALUDE_equation_solution_l473_47317

theorem equation_solution :
  ∃ x : ℝ, (7 - 2*x = -3) ∧ (x = 5) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l473_47317


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l473_47343

theorem geometric_sequence_sum (a : ℚ) (r : ℚ) (n : ℕ) (h1 : a = 1/4) (h2 : r = 1/4) (h3 : n = 6) :
  a * (1 - r^n) / (1 - r) = 1365/4096 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l473_47343


namespace NUMINAMATH_CALUDE_halls_per_floor_wing2_is_9_l473_47306

/-- A hotel with two wings -/
structure Hotel where
  total_rooms : ℕ
  wing1_floors : ℕ
  wing1_halls_per_floor : ℕ
  wing1_rooms_per_hall : ℕ
  wing2_floors : ℕ
  wing2_rooms_per_hall : ℕ

/-- The number of halls on each floor of the second wing -/
def halls_per_floor_wing2 (h : Hotel) : ℕ :=
  (h.total_rooms - h.wing1_floors * h.wing1_halls_per_floor * h.wing1_rooms_per_hall) /
  (h.wing2_floors * h.wing2_rooms_per_hall)

/-- Theorem stating that the number of halls on each floor of the second wing is 9 -/
theorem halls_per_floor_wing2_is_9 (h : Hotel)
  (h_total : h.total_rooms = 4248)
  (h_wing1_floors : h.wing1_floors = 9)
  (h_wing1_halls : h.wing1_halls_per_floor = 6)
  (h_wing1_rooms : h.wing1_rooms_per_hall = 32)
  (h_wing2_floors : h.wing2_floors = 7)
  (h_wing2_rooms : h.wing2_rooms_per_hall = 40) :
  halls_per_floor_wing2 h = 9 := by
  sorry

#eval halls_per_floor_wing2 {
  total_rooms := 4248,
  wing1_floors := 9,
  wing1_halls_per_floor := 6,
  wing1_rooms_per_hall := 32,
  wing2_floors := 7,
  wing2_rooms_per_hall := 40
}

end NUMINAMATH_CALUDE_halls_per_floor_wing2_is_9_l473_47306


namespace NUMINAMATH_CALUDE_initial_books_l473_47330

theorem initial_books (initial : ℕ) (sold : ℕ) (bought : ℕ) (final : ℕ) : 
  sold = 11 → bought = 23 → final = 45 → initial - sold + bought = final → initial = 33 := by
  sorry

end NUMINAMATH_CALUDE_initial_books_l473_47330


namespace NUMINAMATH_CALUDE_original_square_area_l473_47356

theorem original_square_area : ∃ s : ℝ, s > 0 ∧ s^2 = 400 ∧ (s + 5)^2 = s^2 + 225 := by
  sorry

end NUMINAMATH_CALUDE_original_square_area_l473_47356


namespace NUMINAMATH_CALUDE_perfect_square_factors_count_l473_47345

/-- The number of positive factors of 450 that are perfect squares -/
def perfect_square_factors_of_450 : ℕ :=
  let prime_factorization : ℕ × ℕ × ℕ := (1, 2, 2)  -- Exponents of 2, 3, and 5 in 450's factorization
  2 * 2 * 2  -- Number of ways to choose even exponents for each prime factor

theorem perfect_square_factors_count :
  perfect_square_factors_of_450 = 8 :=
by sorry

end NUMINAMATH_CALUDE_perfect_square_factors_count_l473_47345


namespace NUMINAMATH_CALUDE_arc_length_ln_sin_l473_47353

open Real MeasureTheory

/-- The arc length of the curve y = ln(sin x) from x = π/3 to x = π/2 is (1/2) ln 3 -/
theorem arc_length_ln_sin (f : ℝ → ℝ) (h : ∀ x, f x = Real.log (Real.sin x)) :
  ∫ x in Set.Icc (π/3) (π/2), sqrt (1 + (deriv f x)^2) = (1/2) * Real.log 3 := by
  sorry

end NUMINAMATH_CALUDE_arc_length_ln_sin_l473_47353


namespace NUMINAMATH_CALUDE_product_of_repeating_decimal_and_eleven_l473_47361

/-- The repeating decimal 0.4567 as a rational number -/
def repeating_decimal : ℚ := 4567 / 9999

theorem product_of_repeating_decimal_and_eleven :
  11 * repeating_decimal = 50237 / 9999 := by sorry

end NUMINAMATH_CALUDE_product_of_repeating_decimal_and_eleven_l473_47361


namespace NUMINAMATH_CALUDE_olivia_dvd_count_l473_47334

theorem olivia_dvd_count (dvds_per_season : ℕ) (seasons_bought : ℕ) : 
  dvds_per_season = 8 → seasons_bought = 5 → dvds_per_season * seasons_bought = 40 :=
by sorry

end NUMINAMATH_CALUDE_olivia_dvd_count_l473_47334


namespace NUMINAMATH_CALUDE_tangent_slope_at_two_l473_47371

/-- The function representing the curve y = x^2 + 3x -/
def f (x : ℝ) : ℝ := x^2 + 3*x

/-- The derivative of the function f -/
def f' (x : ℝ) : ℝ := 2*x + 3

theorem tangent_slope_at_two :
  f' 2 = 7 := by sorry

end NUMINAMATH_CALUDE_tangent_slope_at_two_l473_47371


namespace NUMINAMATH_CALUDE_extreme_values_and_inequality_l473_47320

def f (a b c x : ℝ) : ℝ := x^3 - a*x^2 + b*x + c

theorem extreme_values_and_inequality 
  (a b c : ℝ) 
  (h1 : ∃ y, (deriv (f a b c)) (-1) = y ∧ y = 0)
  (h2 : ∃ y, (deriv (f a b c)) 3 = y ∧ y = 0)
  (h3 : ∀ x ∈ Set.Icc (-2) 6, f a b c x < c^2 + 4*c) :
  a = 3 ∧ b = -9 ∧ (c > 6 ∨ c < -9) := by sorry

end NUMINAMATH_CALUDE_extreme_values_and_inequality_l473_47320


namespace NUMINAMATH_CALUDE_train_speed_calculation_l473_47385

/-- Proves that given a train journey with an original time of 50 minutes and a reduced time of 40 minutes
    at a speed of 60 km/h, the original average speed of the train is 48 km/h. -/
theorem train_speed_calculation (distance : ℝ) (original_time : ℝ) (reduced_time : ℝ) (new_speed : ℝ) :
  original_time = 50 / 60 →
  reduced_time = 40 / 60 →
  new_speed = 60 →
  distance = new_speed * reduced_time →
  distance / original_time = 48 :=
by sorry

end NUMINAMATH_CALUDE_train_speed_calculation_l473_47385


namespace NUMINAMATH_CALUDE_arrangement_speeches_not_adjacent_l473_47399

theorem arrangement_speeches_not_adjacent (n : ℕ) (m : ℕ) :
  n = 5 ∧ m = 3 →
  (n.factorial * (n + 1).factorial / ((n + 1 - m).factorial)) = 14400 :=
sorry

end NUMINAMATH_CALUDE_arrangement_speeches_not_adjacent_l473_47399


namespace NUMINAMATH_CALUDE_subset_sum_indivisibility_implies_equality_l473_47325

theorem subset_sum_indivisibility_implies_equality (m : ℕ) (a : Fin m → ℕ) :
  (∀ i, a i ∈ Finset.range m) →
  (∀ s : Finset (Fin m), (s.sum a) % (m + 1) ≠ 0) →
  ∀ i j, a i = a j :=
sorry

end NUMINAMATH_CALUDE_subset_sum_indivisibility_implies_equality_l473_47325


namespace NUMINAMATH_CALUDE_smallest_consecutive_sum_l473_47321

theorem smallest_consecutive_sum (n : ℕ) (a : ℕ) (h1 : n > 1) 
  (h2 : n * a + n * (n - 1) / 2 = 2016) : 
  ∃ (m : ℕ), m ≥ 1 ∧ m * a + m * (m - 1) / 2 = 2016 ∧ m > 1 → a ≥ 1 :=
by sorry

end NUMINAMATH_CALUDE_smallest_consecutive_sum_l473_47321


namespace NUMINAMATH_CALUDE_parabola_equation_proof_l473_47310

/-- A parabola is defined by three points: A(4,0), C(0,-4), and B(-1,0) -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The equation of the parabola is y = ax^2 + bx + c -/
def parabola_equation (p : Parabola) (x : ℝ) : ℝ :=
  p.a * x^2 + p.b * x + p.c

/-- The parabola passes through point A(4,0) -/
def passes_through_A (p : Parabola) : Prop :=
  parabola_equation p 4 = 0

/-- The parabola passes through point C(0,-4) -/
def passes_through_C (p : Parabola) : Prop :=
  parabola_equation p 0 = -4

/-- The parabola passes through point B(-1,0) -/
def passes_through_B (p : Parabola) : Prop :=
  parabola_equation p (-1) = 0

/-- The theorem states that the parabola passing through A, C, and B
    has the equation y = x^2 - 3x - 4 -/
theorem parabola_equation_proof :
  ∃ p : Parabola,
    passes_through_A p ∧
    passes_through_C p ∧
    passes_through_B p ∧
    p.a = 1 ∧ p.b = -3 ∧ p.c = -4 :=
  sorry

end NUMINAMATH_CALUDE_parabola_equation_proof_l473_47310


namespace NUMINAMATH_CALUDE_even_periodic_function_range_l473_47326

/-- A function is even if f(-x) = f(x) for all x -/
def IsEven (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

/-- A function has period p if f(x + p) = f(x) for all x -/
def HasPeriod (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x, f (x + p) = f x

theorem even_periodic_function_range (f : ℝ → ℝ) (a : ℝ) :
  IsEven f →
  HasPeriod f 3 →
  f 1 < 1 →
  f 5 = (2*a - 3) / (a + 1) →
  -1 < a ∧ a < 4 := by
  sorry

end NUMINAMATH_CALUDE_even_periodic_function_range_l473_47326


namespace NUMINAMATH_CALUDE_complement_intersection_equality_l473_47322

def I : Set ℕ := {1, 2, 3, 4, 5}
def M : Set ℕ := {1, 2, 3}
def N : Set ℕ := {3, 4, 5}

theorem complement_intersection_equality :
  (I \ N) ∩ M = {1, 2} := by sorry

end NUMINAMATH_CALUDE_complement_intersection_equality_l473_47322


namespace NUMINAMATH_CALUDE_inequality_proof_l473_47339

theorem inequality_proof (a b : ℝ) (h : a + b > 0) :
  a / b^2 + b / a^2 ≥ 1 / a + 1 / b :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l473_47339


namespace NUMINAMATH_CALUDE_largest_band_members_l473_47302

theorem largest_band_members :
  ∀ (m r x : ℕ),
    m < 100 →
    m = r * x + 4 →
    m = (r - 3) * (x + 2) →
    (∀ m' r' x' : ℕ,
      m' < 100 →
      m' = r' * x' + 4 →
      m' = (r' - 3) * (x' + 2) →
      m' ≤ m) →
    m = 88 :=
by sorry

end NUMINAMATH_CALUDE_largest_band_members_l473_47302


namespace NUMINAMATH_CALUDE_cube_cut_surface_area_l473_47370

/-- Represents a piece of the cube -/
structure Piece where
  height : ℝ

/-- Represents the solid formed by rearranging the cube pieces -/
structure Solid where
  pieces : List Piece

/-- Calculates the surface area of the solid -/
def surfaceArea (s : Solid) : ℝ :=
  sorry

theorem cube_cut_surface_area :
  let cube_volume : ℝ := 1
  let cut1 : ℝ := 1/2
  let cut2 : ℝ := 1/3
  let cut3 : ℝ := 1/17
  let piece_A : Piece := ⟨cut1⟩
  let piece_B : Piece := ⟨cut2⟩
  let piece_C : Piece := ⟨cut3⟩
  let piece_D : Piece := ⟨1 - (cut1 + cut2 + cut3)⟩
  let solid : Solid := ⟨[piece_A, piece_B, piece_C, piece_D]⟩
  surfaceArea solid = 11 :=
sorry

end NUMINAMATH_CALUDE_cube_cut_surface_area_l473_47370


namespace NUMINAMATH_CALUDE_asterisk_replacement_l473_47340

/-- The expression after substituting 2x for * and expanding -/
def expanded_expression (x : ℝ) : ℝ := x^6 + x^4 + 4*x^2 + 4

/-- The number of terms in the expanded expression -/
def num_terms (x : ℝ) : ℕ := 4

theorem asterisk_replacement :
  ∀ x : ℝ, (x^3 - 2)^2 + (x^2 + 2*x)^2 = expanded_expression x ∧
           num_terms x = 4 :=
by sorry

end NUMINAMATH_CALUDE_asterisk_replacement_l473_47340


namespace NUMINAMATH_CALUDE_factorization_equality_l473_47365

theorem factorization_equality (x : ℝ) : (x^2 - 1)^2 - 6*(x^2 - 1) + 9 = (x - 2)^2 * (x + 2)^2 := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l473_47365


namespace NUMINAMATH_CALUDE_sum_of_fractions_l473_47300

theorem sum_of_fractions (p q r : ℝ) 
  (h1 : p + q + r = 5) 
  (h2 : 1 / (p + q) + 1 / (q + r) + 1 / (p + r) = 9) : 
  r / (p + q) + p / (q + r) + q / (p + r) = 42 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_fractions_l473_47300


namespace NUMINAMATH_CALUDE_carpet_cost_l473_47357

/-- The cost of a carpet with increased dimensions -/
theorem carpet_cost (b₁ : ℝ) (l₁_factor : ℝ) (l₂_increase : ℝ) (b₂_increase : ℝ) (rate : ℝ) :
  b₁ = 6 →
  l₁_factor = 1.44 →
  l₂_increase = 0.4 →
  b₂_increase = 0.25 →
  rate = 45 →
  (b₁ * (1 + b₂_increase)) * (b₁ * l₁_factor * (1 + l₂_increase)) * rate = 4082.4 := by
  sorry

end NUMINAMATH_CALUDE_carpet_cost_l473_47357


namespace NUMINAMATH_CALUDE_unique_prime_with_remainder_l473_47318

theorem unique_prime_with_remainder : ∃! p : ℕ, 
  Prime p ∧ 
  20 < p ∧ p < 35 ∧ 
  p % 11 = 7 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_unique_prime_with_remainder_l473_47318


namespace NUMINAMATH_CALUDE_min_sum_squares_over_a_squared_l473_47391

theorem min_sum_squares_over_a_squared (a b c : ℝ) 
  (h1 : a > b) (h2 : b > c) (h3 : a ≠ 0) : 
  ((a + b)^2 + (b + c)^2 + (c + a)^2) / a^2 ≥ 4 := by
  sorry

end NUMINAMATH_CALUDE_min_sum_squares_over_a_squared_l473_47391


namespace NUMINAMATH_CALUDE_total_travel_ways_l473_47398

-- Define the number of services for each mode of transportation
def bus_services : ℕ := 8
def train_services : ℕ := 3
def ferry_services : ℕ := 2

-- Theorem statement
theorem total_travel_ways : bus_services + train_services + ferry_services = 13 := by
  sorry

end NUMINAMATH_CALUDE_total_travel_ways_l473_47398


namespace NUMINAMATH_CALUDE_quadratic_factorization_l473_47347

theorem quadratic_factorization (x : ℝ) :
  16 * x^2 + 8 * x - 24 = 8 * (2 * x + 3) * (x - 1) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_factorization_l473_47347


namespace NUMINAMATH_CALUDE_molecular_weight_proof_l473_47366

-- Define atomic weights
def atomic_weight_Al : ℝ := 26.98
def atomic_weight_O : ℝ := 16.00
def atomic_weight_Fe : ℝ := 55.85
def atomic_weight_H : ℝ := 1.01

-- Define the number of atoms for each element
def num_Al : ℕ := 2
def num_O : ℕ := 3
def num_Fe : ℕ := 2
def num_H : ℕ := 4

-- Define the molecular weight calculation function
def molecular_weight : ℝ :=
  (num_Al : ℝ) * atomic_weight_Al +
  (num_O : ℝ) * atomic_weight_O +
  (num_Fe : ℝ) * atomic_weight_Fe +
  (num_H : ℝ) * atomic_weight_H

-- Theorem statement
theorem molecular_weight_proof :
  molecular_weight = 217.70 := by sorry

end NUMINAMATH_CALUDE_molecular_weight_proof_l473_47366


namespace NUMINAMATH_CALUDE_ali_wallet_final_amount_l473_47381

def initial_wallet_value : ℕ := 7 * 5 + 1 * 10 + 3 * 20 + 1 * 50 + 8 * 1

def grocery_spending : ℕ := 65

def change_received : ℕ := 1 * 5 + 5 * 1

def friend_payment : ℕ := 2 * 20 + 2 * 1

theorem ali_wallet_final_amount :
  initial_wallet_value - grocery_spending + change_received + friend_payment = 150 := by
  sorry

end NUMINAMATH_CALUDE_ali_wallet_final_amount_l473_47381


namespace NUMINAMATH_CALUDE_hyperbola_asymptotes_l473_47313

/-- The focus of a parabola y² = 12x -/
def parabola_focus : ℝ × ℝ := (3, 0)

/-- The equation of a hyperbola -/
def is_hyperbola (a : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 - y^2 = 1

/-- The equation of asymptotes of a hyperbola -/
def is_asymptote (k : ℝ) (x y : ℝ) : Prop :=
  y = k * x ∨ y = -k * x

/-- Main theorem -/
theorem hyperbola_asymptotes :
  ∃ (a : ℝ), (is_hyperbola a (parabola_focus.1) (parabola_focus.2)) →
  (∀ (x y : ℝ), is_asymptote (1/3) x y ↔ is_hyperbola a x y) :=
sorry

end NUMINAMATH_CALUDE_hyperbola_asymptotes_l473_47313


namespace NUMINAMATH_CALUDE_triangle_ABC_properties_l473_47386

open Real

theorem triangle_ABC_properties (A B C a b c : Real) : 
  0 < A ∧ 0 < B ∧ 0 < C ∧ 
  A + B + C = π ∧
  2 * sin A * sin C * (1 / (tan A * tan C) - 1) = -1 ∧
  a + c = 3 * sqrt 3 / 2 ∧
  b = sqrt 3 →
  B = π / 3 ∧ 
  (1 / 2) * a * c * sin B = 5 * sqrt 3 / 16 :=
by sorry

end NUMINAMATH_CALUDE_triangle_ABC_properties_l473_47386


namespace NUMINAMATH_CALUDE_petya_ice_cream_l473_47346

theorem petya_ice_cream (ice_cream_cost : ℕ) (petya_money : ℕ) : 
  ice_cream_cost = 2000 →
  petya_money = 400^5 - 399^2 * (400^3 + 2 * 400^2 + 3 * 400 + 4) →
  petya_money < ice_cream_cost :=
by
  sorry

end NUMINAMATH_CALUDE_petya_ice_cream_l473_47346
