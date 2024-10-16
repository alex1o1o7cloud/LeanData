import Mathlib

namespace NUMINAMATH_CALUDE_inequality_theorem_l3405_340550

theorem inequality_theorem (a b : ℝ) : 
  (a * b > 0 → b / a + a / b ≥ 2) ∧ 
  (a + 2 * b = 1 → 3^a + 9^b ≥ 2 * Real.sqrt 3) := by
  sorry

end NUMINAMATH_CALUDE_inequality_theorem_l3405_340550


namespace NUMINAMATH_CALUDE_folded_paper_distance_l3405_340538

/-- Given a square sheet of paper with area 12 cm², prove that when folded so
    that a corner point B rests on the diagonal and the visible red area equals
    the visible blue area, the distance from B to its original position is 4 cm. -/
theorem folded_paper_distance (sheet_area : ℝ) (fold_length : ℝ) :
  sheet_area = 12 →
  fold_length^2 / 2 = sheet_area - fold_length^2 →
  Real.sqrt (2 * fold_length^2) = 4 :=
by sorry

end NUMINAMATH_CALUDE_folded_paper_distance_l3405_340538


namespace NUMINAMATH_CALUDE_factorization_proof_l3405_340508

theorem factorization_proof : 989 * 1001 * 1007 + 320 = 991 * 997 * 1009 := by
  sorry

end NUMINAMATH_CALUDE_factorization_proof_l3405_340508


namespace NUMINAMATH_CALUDE_inequality_proof_l3405_340574

theorem inequality_proof (x y : ℝ) (hx : x ≥ 0) (hy : y ≥ 0) (hxy : x + y ≤ 1) :
  12 * x * y ≤ 4 * x * (1 - y) + 9 * y * (1 - x) := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l3405_340574


namespace NUMINAMATH_CALUDE_percentage_to_pass_l3405_340530

/-- Given a test with maximum marks and a student's performance, 
    calculate the percentage needed to pass the test. -/
theorem percentage_to_pass (max_marks : ℕ) (student_marks : ℕ) (fail_margin : ℕ) :
  max_marks = 300 →
  student_marks = 80 →
  fail_margin = 10 →
  (((student_marks + fail_margin : ℝ) / max_marks) * 100 : ℝ) = 30 := by
  sorry

end NUMINAMATH_CALUDE_percentage_to_pass_l3405_340530


namespace NUMINAMATH_CALUDE_lucy_popsicles_l3405_340558

/-- The maximum number of popsicles Lucy can buy given her funds and the pricing structure -/
def max_popsicles (total_funds : ℚ) (first_tier_price : ℚ) (second_tier_price : ℚ) (first_tier_limit : ℕ) : ℕ :=
  let first_tier_cost := first_tier_limit * first_tier_price
  let remaining_funds := total_funds - first_tier_cost
  let additional_popsicles := (remaining_funds / second_tier_price).floor
  first_tier_limit + additional_popsicles.toNat

/-- Theorem stating that Lucy can buy 15 popsicles -/
theorem lucy_popsicles :
  max_popsicles 25.5 1.75 1.5 8 = 15 := by
  sorry

#eval max_popsicles 25.5 1.75 1.5 8

end NUMINAMATH_CALUDE_lucy_popsicles_l3405_340558


namespace NUMINAMATH_CALUDE_root_implies_q_value_l3405_340571

-- Define the complex number i
def i : ℂ := Complex.I

-- Define the equation
def is_root (x p q : ℂ) : Prop := x^2 + p*x + q = 0

-- State the theorem
theorem root_implies_q_value (p q : ℝ) :
  is_root (2 + 3*i) p q → q = 13 := by
  sorry

end NUMINAMATH_CALUDE_root_implies_q_value_l3405_340571


namespace NUMINAMATH_CALUDE_probability_three_integer_points_l3405_340585

/-- Square with diagonal endpoints (1/4, 3/4) and (-1/4, -3/4) -/
def S : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | ∃ (t : ℝ), 0 ≤ t ∧ t ≤ 1 ∧
    p.1 = t/4 - (1-t)/4 ∧ p.2 = 3*t/4 - 3*(1-t)/4}

/-- Random point v = (x, y) where 0 ≤ x ≤ 100 and 0 ≤ y ≤ 100 -/
def V : Set (ℝ × ℝ) :=
  {v : ℝ × ℝ | 0 ≤ v.1 ∧ v.1 ≤ 100 ∧ 0 ≤ v.2 ∧ v.2 ≤ 100}

/-- Translated copy of S centered at v -/
def T (v : ℝ × ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | ∃ (q : ℝ × ℝ), q ∈ S ∧ p.1 = q.1 + v.1 ∧ p.2 = q.2 + v.2}

/-- Set of integer points -/
def IntegerPoints : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | ∃ (m n : ℤ), p.1 = m ∧ p.2 = n}

/-- Probability measure on V -/
noncomputable def P : (Set (ℝ × ℝ)) → ℝ := sorry

theorem probability_three_integer_points :
  P {v ∈ V | (T v ∩ IntegerPoints).ncard = 3} = 3/100 := sorry

end NUMINAMATH_CALUDE_probability_three_integer_points_l3405_340585


namespace NUMINAMATH_CALUDE_problem_solving_probability_l3405_340531

theorem problem_solving_probability (p1 p2 p3 : ℝ) 
  (h1 : p1 = 1/5) (h2 : p2 = 1/3) (h3 : p3 = 1/4) :
  1 - (1 - p1) * (1 - p2) * (1 - p3) = 3/5 := by
  sorry

end NUMINAMATH_CALUDE_problem_solving_probability_l3405_340531


namespace NUMINAMATH_CALUDE_unattainable_y_value_l3405_340586

theorem unattainable_y_value (x : ℝ) (h : x ≠ -4/3) :
  ¬∃ y : ℝ, y = -1/3 ∧ y = (2 - x) / (3*x + 4) := by
  sorry

end NUMINAMATH_CALUDE_unattainable_y_value_l3405_340586


namespace NUMINAMATH_CALUDE_debate_club_election_l3405_340524

def election_ways (n m k : ℕ) : ℕ :=
  (n - k).factorial / ((n - k - m).factorial * m.factorial) +
  k.factorial * (n - k).choose (m - k)

theorem debate_club_election :
  election_ways 30 5 4 = 6378720 :=
sorry

end NUMINAMATH_CALUDE_debate_club_election_l3405_340524


namespace NUMINAMATH_CALUDE_urn_probability_l3405_340566

/-- Represents the contents of the urn -/
structure UrnContents :=
  (red : ℕ)
  (blue : ℕ)

/-- The operation of drawing a ball and adding another of the same color -/
def draw_and_add (contents : UrnContents) : UrnContents → ℕ → ℝ
  | contents, n => sorry

/-- The probability of having a specific urn content after n operations -/
def prob_after_operations (initial : UrnContents) (final : UrnContents) (n : ℕ) : ℝ :=
  sorry

/-- The probability of removing a specific color ball given the urn contents -/
def prob_remove_color (contents : UrnContents) (remove_red : Bool) : ℝ :=
  sorry

/-- The main theorem to prove -/
theorem urn_probability :
  let initial := UrnContents.mk 2 1
  let final := UrnContents.mk 4 4
  let operations := 6
  (prob_after_operations initial (UrnContents.mk 5 4) operations *
   prob_remove_color (UrnContents.mk 5 4) true) = 5/63 :=
by sorry

end NUMINAMATH_CALUDE_urn_probability_l3405_340566


namespace NUMINAMATH_CALUDE_solve_equation_l3405_340547

theorem solve_equation (x : ℤ) (h : 9773 + x = 13200) : x = 3427 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l3405_340547


namespace NUMINAMATH_CALUDE_purely_imaginary_complex_l3405_340554

theorem purely_imaginary_complex (a : ℝ) :
  let z : ℂ := Complex.mk (a + 1) (1 + a^2)
  (z.re = 0 ∧ z.im ≠ 0) → a = -1 := by
  sorry

end NUMINAMATH_CALUDE_purely_imaginary_complex_l3405_340554


namespace NUMINAMATH_CALUDE_gmat_question_percentage_l3405_340570

theorem gmat_question_percentage
  (second_correct : Real)
  (neither_correct : Real)
  (both_correct : Real)
  (h1 : second_correct = 0.8)
  (h2 : neither_correct = 0.05)
  (h3 : both_correct = 0.7) :
  ∃ (first_correct : Real),
    first_correct = 0.85 ∧
    first_correct + second_correct - both_correct = 1 - neither_correct :=
by
  sorry

end NUMINAMATH_CALUDE_gmat_question_percentage_l3405_340570


namespace NUMINAMATH_CALUDE_divisibility_property_l3405_340581

theorem divisibility_property (n : ℕ) : 
  ∃ (a b : ℕ), (a * n + 1)^6 + b ≡ 0 [MOD (n^2 + n + 1)] :=
by
  use 2, 27
  sorry

end NUMINAMATH_CALUDE_divisibility_property_l3405_340581


namespace NUMINAMATH_CALUDE_treasury_problem_l3405_340599

theorem treasury_problem (T : ℚ) : 
  (T - T / 13 - (T - T / 13) / 17 = 150) → 
  T = 172 + 21 / 32 :=
by sorry

end NUMINAMATH_CALUDE_treasury_problem_l3405_340599


namespace NUMINAMATH_CALUDE_flag_paint_cost_l3405_340513

/-- Calculates the cost of paint for a flag given its dimensions and paint properties -/
theorem flag_paint_cost (width height : ℝ) (paint_cost_per_quart : ℝ) (coverage_per_quart : ℝ) : 
  width = 5 → height = 4 → paint_cost_per_quart = 2 → coverage_per_quart = 4 → 
  (2 * width * height / coverage_per_quart) * paint_cost_per_quart = 20 := by
sorry


end NUMINAMATH_CALUDE_flag_paint_cost_l3405_340513


namespace NUMINAMATH_CALUDE_intersection_of_P_and_Q_l3405_340582

-- Define the sets P and Q
def P : Set ℝ := {x | ∃ a : ℝ, x = a^2 + 1}
def Q : Set ℝ := {x | ∃ y : ℝ, y = Real.log (2 - x)}

-- State the theorem
theorem intersection_of_P_and_Q : P ∩ Q = Set.Icc 1 2 := by sorry

end NUMINAMATH_CALUDE_intersection_of_P_and_Q_l3405_340582


namespace NUMINAMATH_CALUDE_scientific_notation_18_million_l3405_340569

theorem scientific_notation_18_million :
  (18000000 : ℝ) = 1.8 * (10 : ℝ) ^ 7 :=
sorry

end NUMINAMATH_CALUDE_scientific_notation_18_million_l3405_340569


namespace NUMINAMATH_CALUDE_expression_evaluation_l3405_340534

theorem expression_evaluation :
  (2^(2+1) - 2*(2-1)^(2+1))^2 = 36 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3405_340534


namespace NUMINAMATH_CALUDE_tangent_circles_theorem_l3405_340514

/-- Two concentric circles with radii 1 and 3 -/
def inner_radius : ℝ := 1
def outer_radius : ℝ := 3

/-- Radius of circles tangent to both concentric circles -/
def tangent_circle_radius : ℝ := 1

/-- Maximum number of non-overlapping tangent circles -/
def max_tangent_circles : ℕ := 6

/-- Theorem stating the radius of tangent circles and the maximum number of such circles -/
theorem tangent_circles_theorem :
  (tangent_circle_radius = 1) ∧
  (max_tangent_circles = 6) := by
  sorry

#check tangent_circles_theorem

end NUMINAMATH_CALUDE_tangent_circles_theorem_l3405_340514


namespace NUMINAMATH_CALUDE_inequality_proof_l3405_340546

theorem inequality_proof (x : ℝ) (hx : x > 0) : (x + 1) * Real.sqrt (x + 1) ≥ Real.sqrt 2 * (x + Real.sqrt x) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3405_340546


namespace NUMINAMATH_CALUDE_triangle_area_rational_l3405_340590

/-- A point on the unit circle with rational coordinates -/
structure RationalUnitCirclePoint where
  x : ℚ
  y : ℚ
  on_circle : x^2 + y^2 = 1

/-- The area of a triangle with vertices on the unit circle is rational -/
theorem triangle_area_rational (p₁ p₂ p₃ : RationalUnitCirclePoint) :
  ∃ a : ℚ, a = (1/2) * |p₁.x * (p₂.y - p₃.y) + p₂.x * (p₃.y - p₁.y) + p₃.x * (p₁.y - p₂.y)| :=
sorry

end NUMINAMATH_CALUDE_triangle_area_rational_l3405_340590


namespace NUMINAMATH_CALUDE_probability_at_least_two_red_is_half_l3405_340539

def total_balls : ℕ := 6
def red_balls : ℕ := 3
def white_balls : ℕ := 2
def black_balls : ℕ := 1
def drawn_balls : ℕ := 3

def probability_at_least_two_red : ℚ :=
  (Nat.choose red_balls drawn_balls + 
   Nat.choose red_balls (drawn_balls - 1) * Nat.choose (white_balls + black_balls) 1) / 
  Nat.choose total_balls drawn_balls

theorem probability_at_least_two_red_is_half :
  probability_at_least_two_red = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_probability_at_least_two_red_is_half_l3405_340539


namespace NUMINAMATH_CALUDE_incorrect_representation_of_roots_l3405_340527

theorem incorrect_representation_of_roots : ∃ x : ℝ, x^2 - 3*x = 0 ∧ ¬(x = x ∧ x = 2*x) :=
by sorry

end NUMINAMATH_CALUDE_incorrect_representation_of_roots_l3405_340527


namespace NUMINAMATH_CALUDE_quadratic_equation_from_means_l3405_340567

theorem quadratic_equation_from_means (α β : ℝ) : 
  (α + β) / 2 = 8 → α * β = 144 → 
  ∀ x, x^2 - 16*x + 144 = 0 ↔ (x = α ∨ x = β) := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_from_means_l3405_340567


namespace NUMINAMATH_CALUDE_descent_time_l3405_340505

/-- Represents the scenario of Clea walking down an escalator -/
structure EscalatorScenario where
  /-- Clea's speed when walking down a stationary escalator (distance/time) -/
  clea_speed : ℝ
  /-- Length of the escalator (distance) -/
  escalator_length : ℝ
  /-- Speed of the moving escalator (distance/time) -/
  escalator_speed : ℝ
  /-- Time taken to walk down when escalator is stationary (seconds) -/
  stationary_time : ℝ
  /-- Time taken to walk down when escalator is moving (seconds) -/
  moving_time : ℝ
  /-- Duration of escalator stoppage during descent (seconds) -/
  stoppage_time : ℝ

/-- Theorem stating the total time for Clea's descent -/
theorem descent_time (scenario : EscalatorScenario) 
  (h1 : scenario.stationary_time = 80)
  (h2 : scenario.moving_time = 40)
  (h3 : scenario.stoppage_time = 20)
  (h4 : scenario.escalator_length = scenario.clea_speed * scenario.stationary_time)
  (h5 : scenario.escalator_length = (scenario.clea_speed + scenario.escalator_speed) * scenario.moving_time) :
  scenario.stoppage_time + (scenario.escalator_length - scenario.clea_speed * scenario.stoppage_time) / (scenario.clea_speed + scenario.escalator_speed) = 50 := by
  sorry


end NUMINAMATH_CALUDE_descent_time_l3405_340505


namespace NUMINAMATH_CALUDE_a_power_six_bounds_l3405_340560

theorem a_power_six_bounds (a : ℝ) (h : a^5 - a^3 + a = 2) : 3 < a^6 ∧ a^6 < 4 := by
  sorry

end NUMINAMATH_CALUDE_a_power_six_bounds_l3405_340560


namespace NUMINAMATH_CALUDE_clown_count_l3405_340591

/-- The number of clown mobiles -/
def num_mobiles : ℕ := 5

/-- The number of clowns in each mobile -/
def clowns_per_mobile : ℕ := 28

/-- The total number of clowns in all mobiles -/
def total_clowns : ℕ := num_mobiles * clowns_per_mobile

theorem clown_count : total_clowns = 140 := by
  sorry

end NUMINAMATH_CALUDE_clown_count_l3405_340591


namespace NUMINAMATH_CALUDE_fabric_price_system_l3405_340588

/-- Represents the price per foot of damask fabric in cents -/
def damask_price : ℝ := sorry

/-- Represents the price per foot of gauze fabric in cents -/
def gauze_price : ℝ := sorry

/-- The length of the damask fabric in feet -/
def damask_length : ℝ := 7

/-- The length of the gauze fabric in feet -/
def gauze_length : ℝ := 9

/-- The price difference per foot between damask and gauze fabrics in cents -/
def price_difference : ℝ := 36

theorem fabric_price_system :
  (damask_length * damask_price = gauze_length * gauze_price) ∧
  (damask_price - gauze_price = price_difference) := by sorry

end NUMINAMATH_CALUDE_fabric_price_system_l3405_340588


namespace NUMINAMATH_CALUDE_product_of_primes_l3405_340545

theorem product_of_primes (p q : ℕ) (hp : Prime p) (hq : Prime q)
  (h_range : 15 < p * q ∧ p * q < 36)
  (hp_range : 2 < p ∧ p < 6)
  (hq_range : 8 < q ∧ q < 24) :
  p * q = 33 := by
sorry

end NUMINAMATH_CALUDE_product_of_primes_l3405_340545


namespace NUMINAMATH_CALUDE_power_of_two_in_product_l3405_340562

theorem power_of_two_in_product (w : ℕ+) : 
  (3^3 ∣ (1452 * w)) ∧ 
  (13^3 ∣ (1452 * w)) ∧ 
  (∀ x : ℕ+, (3^3 ∣ (1452 * x)) ∧ (13^3 ∣ (1452 * x)) → w ≤ x) ∧
  w = 468 →
  ∃ (n : ℕ), 2^2 * n = 1452 * w ∧ ¬(2 ∣ n) := by
sorry

end NUMINAMATH_CALUDE_power_of_two_in_product_l3405_340562


namespace NUMINAMATH_CALUDE_star_equality_implies_power_equality_l3405_340596

/-- The k-th smallest positive integer not in X -/
def f (X : Finset Nat) (k : Nat) : Nat :=
  (Finset.range k.succ \ X).min' sorry

/-- The * operation for finite sets of positive integers -/
def star (X Y : Finset Nat) : Finset Nat :=
  X ∪ (Y.image (f X))

/-- Repeated application of star operation -/
def repeat_star (X : Finset Nat) : Nat → Finset Nat
  | 0 => X
  | n + 1 => star X (repeat_star X n)

theorem star_equality_implies_power_equality
  (A B : Finset Nat) (a b : Nat) (ha : a > 0) (hb : b > 0) :
  star A B = star B A →
  repeat_star A b = repeat_star B a :=
sorry

end NUMINAMATH_CALUDE_star_equality_implies_power_equality_l3405_340596


namespace NUMINAMATH_CALUDE_power_function_satisfies_no_equation_l3405_340595

theorem power_function_satisfies_no_equation (a : ℝ) :
  ¬(∀ x y : ℝ, (x*y)^a = x^a + y^a) ∧
  ¬(∀ x y : ℝ, (x+y)^a = x^a * y^a) ∧
  ¬(∀ x y : ℝ, (x+y)^a = x^a + y^a) :=
by sorry

end NUMINAMATH_CALUDE_power_function_satisfies_no_equation_l3405_340595


namespace NUMINAMATH_CALUDE_spatial_relationships_l3405_340502

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (perpendicular : Line → Plane → Prop)
variable (parallel_lines : Line → Line → Prop)
variable (parallel_planes : Plane → Plane → Prop)
variable (subset : Line → Plane → Prop)

-- State the theorem
theorem spatial_relationships 
  (m n : Line) (α β : Plane) : 
  (∀ (m n : Line) (β : Plane), 
    perpendicular m β → perpendicular n β → parallel_lines m n) ∧
  (∀ (m : Line) (α β : Plane), 
    perpendicular m α → perpendicular m β → parallel_planes α β) :=
by sorry

end NUMINAMATH_CALUDE_spatial_relationships_l3405_340502


namespace NUMINAMATH_CALUDE_sqrt_eight_and_nine_sixteenths_l3405_340536

theorem sqrt_eight_and_nine_sixteenths (x : ℝ) : 
  x = Real.sqrt (8 + 9 / 16) → x = Real.sqrt 137 / 4 :=
by sorry

end NUMINAMATH_CALUDE_sqrt_eight_and_nine_sixteenths_l3405_340536


namespace NUMINAMATH_CALUDE_carly_job_applications_l3405_340583

theorem carly_job_applications : ∃ (x : ℕ), x + 2*x = 600 ∧ x = 200 := by sorry

end NUMINAMATH_CALUDE_carly_job_applications_l3405_340583


namespace NUMINAMATH_CALUDE_point_B_coordinates_l3405_340540

-- Define the coordinates of points A and B
def A (a : ℝ) : ℝ × ℝ := (a - 1, a + 1)
def B (a : ℝ) : ℝ × ℝ := (a + 3, a - 5)

-- Theorem statement
theorem point_B_coordinates :
  ∀ a : ℝ, (A a).1 = 0 → B a = (4, -4) := by
  sorry

end NUMINAMATH_CALUDE_point_B_coordinates_l3405_340540


namespace NUMINAMATH_CALUDE_encryption_proof_l3405_340587

def encrypt (x : ℕ) : ℕ :=
  if x % 2 = 1 ∧ 1 ≤ x ∧ x ≤ 26 then
    (x + 1) / 2
  else if x % 2 = 0 ∧ 1 ≤ x ∧ x ≤ 26 then
    x / 2 + 13
  else
    0

def letter_to_num (c : Char) : ℕ :=
  match c with
  | 'a' => 1 | 'b' => 2 | 'c' => 3 | 'd' => 4 | 'e' => 5
  | 'f' => 6 | 'g' => 7 | 'h' => 8 | 'i' => 9 | 'j' => 10
  | 'k' => 11 | 'l' => 12 | 'm' => 13 | 'n' => 14 | 'o' => 15
  | 'p' => 16 | 'q' => 17 | 'r' => 18 | 's' => 19 | 't' => 20
  | 'u' => 21 | 'v' => 22 | 'w' => 23 | 'x' => 24 | 'y' => 25
  | 'z' => 26
  | _ => 0

def num_to_letter (n : ℕ) : Char :=
  match n with
  | 1 => 'a' | 2 => 'b' | 3 => 'c' | 4 => 'd' | 5 => 'e'
  | 6 => 'f' | 7 => 'g' | 8 => 'h' | 9 => 'i' | 10 => 'j'
  | 11 => 'k' | 12 => 'l' | 13 => 'm' | 14 => 'n' | 15 => 'o'
  | 16 => 'p' | 17 => 'q' | 18 => 'r' | 19 => 's' | 20 => 't'
  | 21 => 'u' | 22 => 'v' | 23 => 'w' | 24 => 'x' | 25 => 'y'
  | 26 => 'z'
  | _ => ' '

theorem encryption_proof :
  (encrypt (letter_to_num 'l'), 
   encrypt (letter_to_num 'o'), 
   encrypt (letter_to_num 'v'), 
   encrypt (letter_to_num 'e')) = 
  (letter_to_num 's', 
   letter_to_num 'h', 
   letter_to_num 'x', 
   letter_to_num 'c') := by
  sorry

end NUMINAMATH_CALUDE_encryption_proof_l3405_340587


namespace NUMINAMATH_CALUDE_octagon_side_length_l3405_340519

/-- Given an octagon-shaped box with a perimeter of 72 cm, prove that each side length is 9 cm. -/
theorem octagon_side_length (perimeter : ℝ) (num_sides : ℕ) : 
  perimeter = 72 ∧ num_sides = 8 → perimeter / num_sides = 9 := by
  sorry

end NUMINAMATH_CALUDE_octagon_side_length_l3405_340519


namespace NUMINAMATH_CALUDE_area_enclosed_by_four_circles_l3405_340553

/-- The area of the figure enclosed by four identical circles inscribed in a larger circle -/
theorem area_enclosed_by_four_circles (R : ℝ) : 
  ∃ (area : ℝ), 
    area = R^2 * (4 - π) * (3 - 2 * Real.sqrt 2) ∧
    (∀ (r : ℝ), 
      r = R * (Real.sqrt 2 - 1) →
      area = 4 * r^2 - π * r^2 ∧
      (∃ (O₁ O₂ O₃ O₄ : ℝ × ℝ),
        -- Four circles with centers O₁, O₂, O₃, O₄ and radius r
        -- Each touching two others and the larger circle with radius R
        True)) :=
by sorry

end NUMINAMATH_CALUDE_area_enclosed_by_four_circles_l3405_340553


namespace NUMINAMATH_CALUDE_fence_cost_for_square_plot_l3405_340563

theorem fence_cost_for_square_plot (area : Real) (price_per_foot : Real) : 
  area = 289 → price_per_foot = 55 → 4 * Real.sqrt area * price_per_foot = 3740 := by
  sorry

end NUMINAMATH_CALUDE_fence_cost_for_square_plot_l3405_340563


namespace NUMINAMATH_CALUDE_right_triangle_perimeter_l3405_340559

theorem right_triangle_perimeter (a b c : ℝ) : 
  a > 0 ∧ b > 0 ∧ c > 0 →  -- Positive side lengths
  a^2 + b^2 = c^2 →        -- Pythagorean theorem
  (1/2) * a * b = (1/2) * c →  -- Area condition
  a + b + c = 2 * (Real.sqrt 2 + 1) :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_perimeter_l3405_340559


namespace NUMINAMATH_CALUDE_parabola_y_intercepts_l3405_340551

/-- The number of y-intercepts of the parabola x = 3y^2 - 5y - 2 -/
def num_y_intercepts : ℕ := 2

/-- The equation of the parabola -/
def parabola_equation (y : ℝ) : ℝ := 3 * y^2 - 5 * y - 2

theorem parabola_y_intercepts :
  (∃ (s : Finset ℝ), s.card = num_y_intercepts ∧
    ∀ y ∈ s, parabola_equation y = 0) :=
by sorry

end NUMINAMATH_CALUDE_parabola_y_intercepts_l3405_340551


namespace NUMINAMATH_CALUDE_sum_of_roots_l3405_340572

theorem sum_of_roots (k d : ℝ) (x₁ x₂ : ℝ) : 
  x₁ ≠ x₂ → 
  (4 * x₁^2 - k * x₁ = d) → 
  (4 * x₂^2 - k * x₂ = d) → 
  x₁ + x₂ = k / 4 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_roots_l3405_340572


namespace NUMINAMATH_CALUDE_lcm_48_180_l3405_340589

theorem lcm_48_180 : Nat.lcm 48 180 = 720 := by
  sorry

end NUMINAMATH_CALUDE_lcm_48_180_l3405_340589


namespace NUMINAMATH_CALUDE_paper_area_difference_paper_area_difference_proof_l3405_340555

/-- The difference in combined area (front and back) between a square sheet of paper
    with side length 11 inches and a rectangular sheet of paper measuring 5.5 inches
    by 11 inches is 121 square inches. -/
theorem paper_area_difference : ℝ → Prop :=
  λ (inch : ℝ) =>
    let square_sheet_side := 11 * inch
    let rect_sheet_length := 5.5 * inch
    let rect_sheet_width := 11 * inch
    let square_sheet_area := 2 * (square_sheet_side * square_sheet_side)
    let rect_sheet_area := 2 * (rect_sheet_length * rect_sheet_width)
    square_sheet_area - rect_sheet_area = 121 * inch * inch

/-- Proof of the paper_area_difference theorem. -/
theorem paper_area_difference_proof : paper_area_difference 1 := by
  sorry

end NUMINAMATH_CALUDE_paper_area_difference_paper_area_difference_proof_l3405_340555


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l3405_340541

/-- Definition of an arithmetic sequence -/
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The problem statement -/
theorem arithmetic_sequence_problem (a : ℕ → ℝ) 
  (h_arith : is_arithmetic_sequence a)
  (h_3 : a 3 = 9)
  (h_9 : a 9 = 3) :
  (∀ n : ℕ, a n = 12 - n) ∧ 
  (∀ n : ℕ, n ≥ 13 → a n < 0) ∧
  (∀ n : ℕ, n < 13 → a n ≥ 0) :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l3405_340541


namespace NUMINAMATH_CALUDE_no_line_bisected_by_P_intersects_hyperbola_l3405_340521

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a line in 2D space -/
structure Line where
  m : ℝ  -- slope
  b : ℝ  -- y-intercept

/-- The hyperbola equation -/
def isOnHyperbola (p : Point) : Prop :=
  p.x^2 / 9 - p.y^2 / 4 = 1

/-- Check if a point is on a line -/
def isOnLine (p : Point) (l : Line) : Prop :=
  p.y = l.m * p.x + l.b

/-- Check if a point bisects a line segment -/
def isMidpoint (p : Point) (p1 : Point) (p2 : Point) : Prop :=
  p.x = (p1.x + p2.x) / 2 ∧ p.y = (p1.y + p2.y) / 2

/-- The main theorem -/
theorem no_line_bisected_by_P_intersects_hyperbola :
  ¬ ∃ (l : Line) (p1 p2 : Point),
    p1 ≠ p2 ∧
    isOnHyperbola p1 ∧
    isOnHyperbola p2 ∧
    isOnLine p1 l ∧
    isOnLine p2 l ∧
    isMidpoint ⟨2, 1⟩ p1 p2 :=
  sorry

end NUMINAMATH_CALUDE_no_line_bisected_by_P_intersects_hyperbola_l3405_340521


namespace NUMINAMATH_CALUDE_angle_measure_in_triangle_l3405_340597

theorem angle_measure_in_triangle (P Q R : ℝ) 
  (h1 : P = 75)
  (h2 : Q = 2 * R - 15)
  (h3 : P + Q + R = 180) :
  R = 40 := by
  sorry

end NUMINAMATH_CALUDE_angle_measure_in_triangle_l3405_340597


namespace NUMINAMATH_CALUDE_goldfish_count_l3405_340509

/-- The number of goldfish in Catriona's aquarium -/
def num_goldfish : ℕ := 8

/-- The number of angelfish in Catriona's aquarium -/
def num_angelfish : ℕ := num_goldfish + 4

/-- The number of guppies in Catriona's aquarium -/
def num_guppies : ℕ := 2 * num_angelfish

/-- The total number of fish in Catriona's aquarium -/
def total_fish : ℕ := 44

/-- Theorem stating that the number of goldfish is 8 -/
theorem goldfish_count : num_goldfish = 8 ∧ 
  num_angelfish = num_goldfish + 4 ∧ 
  num_guppies = 2 * num_angelfish ∧ 
  total_fish = num_goldfish + num_angelfish + num_guppies :=
by sorry

end NUMINAMATH_CALUDE_goldfish_count_l3405_340509


namespace NUMINAMATH_CALUDE_clarence_oranges_left_l3405_340564

def initial_oranges : ℕ := 5
def received_oranges : ℕ := 3
def total_oranges : ℕ := initial_oranges + received_oranges

theorem clarence_oranges_left : (total_oranges / 2 : ℕ) = 4 := by
  sorry

end NUMINAMATH_CALUDE_clarence_oranges_left_l3405_340564


namespace NUMINAMATH_CALUDE_notebook_profit_l3405_340511

/-- Calculates the profit from selling notebooks -/
def calculate_profit (
  num_notebooks : ℕ
  ) (purchase_price : ℚ)
    (sell_price : ℚ) : ℚ :=
  num_notebooks * sell_price - num_notebooks * purchase_price

/-- Proves that the profit from selling 1200 notebooks, 
    purchased at 4 for $5 and sold at 5 for $8, is $420 -/
theorem notebook_profit : 
  calculate_profit 1200 (5/4) (8/5) = 420 := by
  sorry

end NUMINAMATH_CALUDE_notebook_profit_l3405_340511


namespace NUMINAMATH_CALUDE_situps_theorem_l3405_340573

/-- The number of situps Diana did per minute -/
def diana_rate : ℕ := 4

/-- The total number of situps Diana did -/
def diana_total : ℕ := 40

/-- The difference in situps per minute between Hani and Diana -/
def rate_difference : ℕ := 3

/-- The number of minutes Diana took to do her situps -/
def duration : ℕ := diana_total / diana_rate

/-- The number of situps Hani did per minute -/
def hani_rate : ℕ := diana_rate + rate_difference

/-- The total number of situps Hani did -/
def hani_total : ℕ := hani_rate * duration

/-- The total number of situps Hani and Diana did together -/
def total_situps : ℕ := diana_total + hani_total

theorem situps_theorem : total_situps = 110 := by
  sorry

end NUMINAMATH_CALUDE_situps_theorem_l3405_340573


namespace NUMINAMATH_CALUDE_working_light_bulbs_l3405_340532

theorem working_light_bulbs (total_lamps : ℕ) (bulbs_per_lamp : ℕ) 
  (lamps_with_two_burnt : ℕ) (lamps_with_one_burnt : ℕ) (lamps_with_three_burnt : ℕ) :
  total_lamps = 60 →
  bulbs_per_lamp = 7 →
  lamps_with_two_burnt = total_lamps / 3 →
  lamps_with_one_burnt = total_lamps / 4 →
  lamps_with_three_burnt = total_lamps / 5 →
  (total_lamps - (lamps_with_two_burnt + lamps_with_one_burnt + lamps_with_three_burnt)) * bulbs_per_lamp +
  lamps_with_two_burnt * (bulbs_per_lamp - 2) +
  lamps_with_one_burnt * (bulbs_per_lamp - 1) +
  lamps_with_three_burnt * (bulbs_per_lamp - 3) = 329 :=
by
  sorry


end NUMINAMATH_CALUDE_working_light_bulbs_l3405_340532


namespace NUMINAMATH_CALUDE_sine_function_period_l3405_340525

/-- Given a sinusoidal function y = 2sin(ωx + φ) with ω > 0,
    if the maximum value 2 occurs at x = π/6 and
    the minimum value -2 occurs at x = 2π/3,
    then ω = 2. -/
theorem sine_function_period (ω φ : ℝ) (h_ω_pos : ω > 0) :
  (∀ x : ℝ, 2 * Real.sin (ω * x + φ) ≤ 2) ∧
  (2 * Real.sin (ω * (π / 6) + φ) = 2) ∧
  (2 * Real.sin (ω * (2 * π / 3) + φ) = -2) →
  ω = 2 := by
  sorry

end NUMINAMATH_CALUDE_sine_function_period_l3405_340525


namespace NUMINAMATH_CALUDE_triangle_inequality_l3405_340520

theorem triangle_inequality (a b c : ℝ) (h : 0 < a ∧ 0 < b ∧ 0 < c ∧ a + b > c ∧ b + c > a ∧ c + a > b) :
  (a^2 / (b + c - a)) + (b^2 / (c + a - b)) + (c^2 / (a + b - c)) ≥ a + b + c := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_l3405_340520


namespace NUMINAMATH_CALUDE_percentage_of_female_employees_l3405_340503

theorem percentage_of_female_employees 
  (total_employees : ℕ) 
  (male_computer_literate_ratio : ℚ) 
  (total_computer_literate_ratio : ℚ) 
  (female_computer_literate : ℕ) 
  (h1 : total_employees = 1400)
  (h2 : male_computer_literate_ratio = 1/2)
  (h3 : total_computer_literate_ratio = 62/100)
  (h4 : female_computer_literate = 588) :
  (total_employees - (total_computer_literate_ratio * total_employees - female_computer_literate) / male_computer_literate_ratio) / total_employees = 3/5 := by
sorry


end NUMINAMATH_CALUDE_percentage_of_female_employees_l3405_340503


namespace NUMINAMATH_CALUDE_fruit_display_problem_l3405_340544

theorem fruit_display_problem (bananas oranges apples : ℕ) 
  (h1 : apples = 2 * oranges)
  (h2 : oranges = 2 * bananas)
  (h3 : bananas + oranges + apples = 35) :
  bananas = 5 := by
  sorry

end NUMINAMATH_CALUDE_fruit_display_problem_l3405_340544


namespace NUMINAMATH_CALUDE_factorization_equality_l3405_340518

theorem factorization_equality (a b : ℝ) : 9 * a * b - a^3 * b = a * b * (3 + a) * (3 - a) := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l3405_340518


namespace NUMINAMATH_CALUDE_hamburger_cost_is_correct_l3405_340500

/-- The cost of a hamburger given the conditions of Robert and Teddy's snack purchase --/
def hamburger_cost : ℚ :=
  let pizza_box_cost : ℚ := 10
  let soft_drink_cost : ℚ := 2
  let robert_pizza_boxes : ℕ := 5
  let robert_soft_drinks : ℕ := 10
  let teddy_hamburgers : ℕ := 6
  let teddy_soft_drinks : ℕ := 10
  let total_spent : ℚ := 106

  let robert_spent : ℚ := pizza_box_cost * robert_pizza_boxes + soft_drink_cost * robert_soft_drinks
  let teddy_spent : ℚ := total_spent - robert_spent
  let teddy_hamburgers_cost : ℚ := teddy_spent - soft_drink_cost * teddy_soft_drinks

  teddy_hamburgers_cost / teddy_hamburgers

theorem hamburger_cost_is_correct :
  hamburger_cost = 267/100 := by
  sorry

end NUMINAMATH_CALUDE_hamburger_cost_is_correct_l3405_340500


namespace NUMINAMATH_CALUDE_min_purchase_price_l3405_340535

/-- Represents the coin denominations available on the Moon -/
def moon_coins : List Nat := [1, 15, 50]

/-- Theorem stating the minimum possible price of a purchase on the Moon -/
theorem min_purchase_price :
  ∀ (payment : List Nat) (change : List Nat),
    (∀ c ∈ payment, c ∈ moon_coins) →
    (∀ c ∈ change, c ∈ moon_coins) →
    (change.length = payment.length + 1) →
    (payment.sum - change.sum ≥ 6) →
    ∃ (p : List Nat) (c : List Nat),
      (∀ x ∈ p, x ∈ moon_coins) ∧
      (∀ x ∈ c, x ∈ moon_coins) ∧
      (c.length = p.length + 1) ∧
      (p.sum - c.sum = 6) :=
by sorry

end NUMINAMATH_CALUDE_min_purchase_price_l3405_340535


namespace NUMINAMATH_CALUDE_handshake_theorem_l3405_340526

theorem handshake_theorem (n : ℕ) (h : n = 8) :
  let total_people := n
  let num_teams := n / 2
  let handshakes_per_person := total_people - 2
  (total_people * handshakes_per_person) / 2 = 24 :=
by sorry

end NUMINAMATH_CALUDE_handshake_theorem_l3405_340526


namespace NUMINAMATH_CALUDE_train_passing_time_l3405_340579

/-- The time it takes for a train to pass a man running in the opposite direction -/
theorem train_passing_time (train_length : ℝ) (train_speed : ℝ) (man_speed : ℝ) : 
  train_length = 110 →
  train_speed = 90 * (1000 / 3600) →
  man_speed = 9 * (1000 / 3600) →
  (train_length / (train_speed + man_speed)) = 4 := by
  sorry

#check train_passing_time

end NUMINAMATH_CALUDE_train_passing_time_l3405_340579


namespace NUMINAMATH_CALUDE_sqrt_minus_three_minus_m_real_l3405_340542

theorem sqrt_minus_three_minus_m_real (m : ℝ) :
  (∃ (x : ℝ), x ^ 2 = -3 - m) ↔ m ≤ -3 :=
by sorry

end NUMINAMATH_CALUDE_sqrt_minus_three_minus_m_real_l3405_340542


namespace NUMINAMATH_CALUDE_specific_jump_record_l3405_340594

/-- The standard distance for the long jump competition -/
def standard_distance : ℝ := 4.00

/-- Calculate the recorded result for a given jump distance -/
def record_jump (jump_distance : ℝ) : ℝ :=
  jump_distance - standard_distance

/-- The specific jump distance we want to prove about -/
def specific_jump : ℝ := 3.85

/-- Theorem stating that the record for the specific jump should be -0.15 -/
theorem specific_jump_record :
  record_jump specific_jump = -0.15 := by sorry

end NUMINAMATH_CALUDE_specific_jump_record_l3405_340594


namespace NUMINAMATH_CALUDE_greatest_common_divisor_with_digit_sum_l3405_340598

def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + sum_of_digits (n / 10)

def has_same_remainder (a b n : ℕ) : Prop :=
  a % n = b % n

theorem greatest_common_divisor_with_digit_sum (a b : ℕ) :
  ∃ (n : ℕ), n > 0 ∧
  n ∣ (a - b) ∧
  has_same_remainder a b n ∧
  sum_of_digits n = 4 ∧
  (∀ m : ℕ, m > n → m ∣ (a - b) → sum_of_digits m ≠ 4) →
  1120 ∣ n ∧ ∀ k : ℕ, k < 1120 → ¬(n ∣ k) :=
sorry

end NUMINAMATH_CALUDE_greatest_common_divisor_with_digit_sum_l3405_340598


namespace NUMINAMATH_CALUDE_abs_p_minus_q_equals_five_l3405_340549

theorem abs_p_minus_q_equals_five (p q : ℝ) (h1 : p * q = 6) (h2 : p + q = 7) : |p - q| = 5 := by
  sorry

end NUMINAMATH_CALUDE_abs_p_minus_q_equals_five_l3405_340549


namespace NUMINAMATH_CALUDE_arithmetic_sequence_terms_l3405_340507

theorem arithmetic_sequence_terms (a₁ l d : ℤ) (h₁ : a₁ = 165) (h₂ : l = 30) (h₃ : d = -5) :
  ∃ n : ℕ, n = 28 ∧ l = a₁ + (n - 1) * d :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_terms_l3405_340507


namespace NUMINAMATH_CALUDE_softball_team_composition_l3405_340504

theorem softball_team_composition (total : ℕ) (ratio : ℚ) : 
  total = 16 ∧ ratio = 5/11 → ∃ (men women : ℕ), 
    men + women = total ∧ 
    (men : ℚ) / (women : ℚ) = ratio ∧ 
    women - men = 6 := by
  sorry

end NUMINAMATH_CALUDE_softball_team_composition_l3405_340504


namespace NUMINAMATH_CALUDE_greatest_common_divisor_780_180_240_l3405_340501

theorem greatest_common_divisor_780_180_240 :
  (∃ (d : ℕ), d ∣ 780 ∧ d ∣ 180 ∧ d ∣ 240 ∧ d < 100 ∧
    ∀ (x : ℕ), x ∣ 780 ∧ x ∣ 180 ∧ x ∣ 240 ∧ x < 100 → x ≤ d) ∧
  (60 ∣ 780 ∧ 60 ∣ 180 ∧ 60 ∣ 240 ∧ 60 < 100) :=
by sorry

end NUMINAMATH_CALUDE_greatest_common_divisor_780_180_240_l3405_340501


namespace NUMINAMATH_CALUDE_photo_album_and_film_prices_l3405_340578

theorem photo_album_and_film_prices :
  ∀ (x y : ℚ),
    5 * x + 4 * y = 139 →
    4 * x + 5 * y = 140 →
    x = 15 ∧ y = 16 := by
  sorry

end NUMINAMATH_CALUDE_photo_album_and_film_prices_l3405_340578


namespace NUMINAMATH_CALUDE_star_four_three_l3405_340529

-- Define the new operation
def star (a b : ℤ) : ℤ := a^2 - a*b + b^2

-- State the theorem
theorem star_four_three : star 4 3 = 13 := by
  sorry

end NUMINAMATH_CALUDE_star_four_three_l3405_340529


namespace NUMINAMATH_CALUDE_algebraic_expression_value_l3405_340575

theorem algebraic_expression_value (x : ℝ) : 
  2 * x^2 + 3 * x + 7 = 8 → 4 * x^2 + 6 * x - 9 = -7 := by
  sorry

end NUMINAMATH_CALUDE_algebraic_expression_value_l3405_340575


namespace NUMINAMATH_CALUDE_discount_calculation_l3405_340556

theorem discount_calculation (price_per_person : ℕ) (num_people : ℕ) (total_cost_with_discount : ℕ) 
  (h1 : price_per_person = 147)
  (h2 : num_people = 2)
  (h3 : total_cost_with_discount = 266) :
  (price_per_person * num_people - total_cost_with_discount) / num_people = 14 := by
  sorry

end NUMINAMATH_CALUDE_discount_calculation_l3405_340556


namespace NUMINAMATH_CALUDE_unique_solution_for_x_l3405_340552

theorem unique_solution_for_x : ∃! x : ℝ, 
  (x ≠ 2) ∧ 
  ((x^3 - 8) / (x - 2) = 3 * x^2) ∧ 
  (x = -1) := by
sorry

end NUMINAMATH_CALUDE_unique_solution_for_x_l3405_340552


namespace NUMINAMATH_CALUDE_soccer_team_average_age_l3405_340548

def ages : List ℕ := [13, 14, 15, 16, 17, 18]
def players : List ℕ := [2, 6, 8, 3, 2, 1]

theorem soccer_team_average_age :
  (List.sum (List.zipWith (· * ·) ages players)) / (List.sum players) = 15 := by
  sorry

end NUMINAMATH_CALUDE_soccer_team_average_age_l3405_340548


namespace NUMINAMATH_CALUDE_disjoint_subsets_equal_sum_l3405_340577

theorem disjoint_subsets_equal_sum (n : ℕ) (A : Finset ℕ) : 
  A.card = n → 
  (∀ a ∈ A, a > 0) → 
  A.sum id < 2^n - 1 → 
  ∃ (B C : Finset ℕ), B ⊆ A ∧ C ⊆ A ∧ B ≠ ∅ ∧ C ≠ ∅ ∧ B ∩ C = ∅ ∧ B.sum id = C.sum id :=
sorry

end NUMINAMATH_CALUDE_disjoint_subsets_equal_sum_l3405_340577


namespace NUMINAMATH_CALUDE_perfect_square_with_conditions_l3405_340580

theorem perfect_square_with_conditions : ∃ (N : ℕ), ∃ (K : ℕ), ∃ (X : ℕ), 
  N = K^2 ∧ 
  K % 20 = 5 ∧ 
  K % 21 = 3 ∧ 
  1000 ≤ X ∧ X < 10000 ∧
  N = X - (X / 1000 + (X / 100 % 10) + (X / 10 % 10) + (X % 10)) ∧
  N = 2025 := by
sorry

end NUMINAMATH_CALUDE_perfect_square_with_conditions_l3405_340580


namespace NUMINAMATH_CALUDE_complex_inverse_calculation_l3405_340593

theorem complex_inverse_calculation (i : ℂ) (h : i^2 = -1) : 
  (2*i - 3*i⁻¹)⁻¹ = -i/5 := by sorry

end NUMINAMATH_CALUDE_complex_inverse_calculation_l3405_340593


namespace NUMINAMATH_CALUDE_geometric_series_first_term_l3405_340533

theorem geometric_series_first_term
  (a r : ℝ)
  (h1 : a / (1 - r) = 20)
  (h2 : a^2 / (1 - r^2) = 80)
  : a = 20 / 3 := by
  sorry

end NUMINAMATH_CALUDE_geometric_series_first_term_l3405_340533


namespace NUMINAMATH_CALUDE_quadratic_solution_l3405_340557

theorem quadratic_solution : ∃ x : ℝ, x^2 - 2*x + 1 = 0 ∧ x = 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_solution_l3405_340557


namespace NUMINAMATH_CALUDE_lines_perpendicular_iff_m_eq_one_l3405_340576

/-- Two lines are perpendicular if and only if the product of their slopes is -1 -/
def perpendicular (m1 m2 : ℝ) : Prop := m1 * m2 = -1

/-- The slope of the line x - y = 0 -/
def slope1 : ℝ := 1

/-- The slope of the line x + my = 0 -/
def slope2 (m : ℝ) : ℝ := -m

/-- Theorem: The lines x - y = 0 and x + my = 0 are perpendicular if and only if m = 1 -/
theorem lines_perpendicular_iff_m_eq_one (m : ℝ) :
  perpendicular slope1 (slope2 m) ↔ m = 1 := by
  sorry

end NUMINAMATH_CALUDE_lines_perpendicular_iff_m_eq_one_l3405_340576


namespace NUMINAMATH_CALUDE_tank_fill_time_l3405_340515

def fill_time_A : ℝ := 30
def fill_rate_B_multiplier : ℝ := 5

theorem tank_fill_time :
  let fill_time_B := fill_time_A / fill_rate_B_multiplier
  let fill_rate_A := 1 / fill_time_A
  let fill_rate_B := 1 / fill_time_B
  let combined_fill_rate := fill_rate_A + fill_rate_B
  1 / combined_fill_rate = 5 := by sorry

end NUMINAMATH_CALUDE_tank_fill_time_l3405_340515


namespace NUMINAMATH_CALUDE_square_sum_reciprocal_l3405_340584

theorem square_sum_reciprocal (x : ℝ) (h : x + (1 / x) = 5) : x^2 + (1 / x)^2 = 23 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_reciprocal_l3405_340584


namespace NUMINAMATH_CALUDE_a_profit_calculation_l3405_340592

def business_profit (a_investment b_investment total_profit : ℕ) : ℕ :=
  let total_investment := a_investment + b_investment
  let management_fee := total_profit / 10
  let remaining_profit := total_profit - management_fee
  let a_share := remaining_profit * a_investment / total_investment
  management_fee + a_share

theorem a_profit_calculation (a_investment b_investment total_profit : ℕ) :
  a_investment = 15000 →
  b_investment = 25000 →
  total_profit = 9600 →
  business_profit a_investment b_investment total_profit = 4200 := by
sorry

#eval business_profit 15000 25000 9600

end NUMINAMATH_CALUDE_a_profit_calculation_l3405_340592


namespace NUMINAMATH_CALUDE_jay_painting_time_l3405_340506

theorem jay_painting_time (bong_time : ℝ) (combined_time : ℝ) (jay_time : ℝ) : 
  bong_time = 3 → 
  combined_time = 1.2 → 
  (1 / jay_time) + (1 / bong_time) = (1 / combined_time) → 
  jay_time = 2 := by
sorry

end NUMINAMATH_CALUDE_jay_painting_time_l3405_340506


namespace NUMINAMATH_CALUDE_trig_identity_l3405_340528

theorem trig_identity (θ : Real) (h : Real.tan θ = Real.sqrt 3) :
  Real.sin (2 * θ) / (1 + Real.cos (2 * θ)) = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_l3405_340528


namespace NUMINAMATH_CALUDE_m_range_l3405_340543

-- Define the function
def f (x : ℝ) : ℝ := x^2 - 2*x + 3

-- State the theorem
theorem m_range (m : ℝ) :
  (∀ x ∈ Set.Icc 0 m, f x ≤ 3) ∧
  (∃ x ∈ Set.Icc 0 m, f x = 3) ∧
  (∀ x ∈ Set.Icc 0 m, f x ≥ 2) ∧
  (∃ x ∈ Set.Icc 0 m, f x = 2) →
  m ∈ Set.Icc 1 2 :=
by sorry

end NUMINAMATH_CALUDE_m_range_l3405_340543


namespace NUMINAMATH_CALUDE_sequence_inequality_l3405_340517

/-- A sequence satisfying the given conditions -/
def SequenceSatisfyingConditions (a : ℕ → ℝ) : Prop :=
  (∀ n, a n ≥ 0) ∧ 
  (∀ m n : ℕ, a (m + n) ≤ a m + a n)

/-- The main theorem to be proved -/
theorem sequence_inequality (a : ℕ → ℝ) (h : SequenceSatisfyingConditions a) :
    ∀ m n : ℕ, n ≥ m → a n ≤ m * a 1 + (n / m - 1) * a m := by
  sorry


end NUMINAMATH_CALUDE_sequence_inequality_l3405_340517


namespace NUMINAMATH_CALUDE_probability_of_blue_is_four_thirteenths_l3405_340568

/-- Represents the number of jelly beans of each color in the bag -/
structure JellyBeanBag where
  red : ℕ
  green : ℕ
  yellow : ℕ
  blue : ℕ

/-- Calculates the total number of jelly beans in the bag -/
def totalJellyBeans (bag : JellyBeanBag) : ℕ :=
  bag.red + bag.green + bag.yellow + bag.blue

/-- Calculates the probability of selecting a blue jelly bean -/
def probabilityOfBlue (bag : JellyBeanBag) : ℚ :=
  bag.blue / (totalJellyBeans bag)

/-- Theorem: The probability of selecting a blue jelly bean from the given bag is 4/13 -/
theorem probability_of_blue_is_four_thirteenths :
  let bag : JellyBeanBag := { red := 5, green := 6, yellow := 7, blue := 8 }
  probabilityOfBlue bag = 4 / 13 := by
  sorry

end NUMINAMATH_CALUDE_probability_of_blue_is_four_thirteenths_l3405_340568


namespace NUMINAMATH_CALUDE_right_triangle_third_side_square_l3405_340561

theorem right_triangle_third_side_square (a b c : ℝ) : 
  a = 4 → b = 5 → c > 0 → 
  (a^2 + b^2 = c^2 ∨ a^2 + c^2 = b^2 ∨ b^2 + c^2 = a^2) → 
  c^2 = 9 ∨ c^2 = 41 := by
sorry

end NUMINAMATH_CALUDE_right_triangle_third_side_square_l3405_340561


namespace NUMINAMATH_CALUDE_sliced_meat_cost_l3405_340510

/-- Given a 4-pack of sliced meat costing $40.00 with a 30% rush delivery fee,
    the cost per type of sliced meat is $13.00. -/
theorem sliced_meat_cost (pack_cost : ℝ) (num_types : ℕ) (rush_fee_percent : ℝ) :
  pack_cost = 40 →
  num_types = 4 →
  rush_fee_percent = 0.3 →
  (pack_cost + pack_cost * rush_fee_percent) / num_types = 13 := by
  sorry

end NUMINAMATH_CALUDE_sliced_meat_cost_l3405_340510


namespace NUMINAMATH_CALUDE_mayo_bottles_count_l3405_340537

/-- Given a ratio of ketchup : mustard : mayo bottles as 3 : 3 : 2, 
    and 6 ketchup bottles, prove that there are 4 mayo bottles. -/
theorem mayo_bottles_count 
  (ratio_ketchup : ℕ) 
  (ratio_mustard : ℕ) 
  (ratio_mayo : ℕ) 
  (ketchup_bottles : ℕ) 
  (h_ratio : ratio_ketchup = 3 ∧ ratio_mustard = 3 ∧ ratio_mayo = 2)
  (h_ketchup : ketchup_bottles = 6) : 
  ketchup_bottles * ratio_mayo / ratio_ketchup = 4 := by
sorry


end NUMINAMATH_CALUDE_mayo_bottles_count_l3405_340537


namespace NUMINAMATH_CALUDE_average_score_calculation_l3405_340523

theorem average_score_calculation (T : ℝ) (h : T > 0) :
  let male_ratio : ℝ := 0.4
  let female_ratio : ℝ := 1 - male_ratio
  let male_avg : ℝ := 75
  let female_avg : ℝ := 80
  let total_score : ℝ := male_ratio * T * male_avg + female_ratio * T * female_avg
  total_score / T = 78 := by
  sorry

end NUMINAMATH_CALUDE_average_score_calculation_l3405_340523


namespace NUMINAMATH_CALUDE_odd_painted_faces_count_l3405_340512

/-- Represents a cube with its number of painted faces -/
structure Cube where
  painted_faces : Nat

/-- Represents the block of cubes -/
def Block := List Cube

/-- Creates a 6x6x1 block of painted cubes -/
def create_block : Block :=
  sorry

/-- Counts the number of cubes with an odd number of painted faces -/
def count_odd_painted (block : Block) : Nat :=
  sorry

/-- Theorem stating that the number of cubes with an odd number of painted faces is 16 -/
theorem odd_painted_faces_count (block : Block) : 
  block = create_block → count_odd_painted block = 16 := by
  sorry

end NUMINAMATH_CALUDE_odd_painted_faces_count_l3405_340512


namespace NUMINAMATH_CALUDE_smallest_four_digit_multiple_of_18_l3405_340565

theorem smallest_four_digit_multiple_of_18 :
  ∃ n : ℕ, n = 1008 ∧ 
  (∀ m : ℕ, m ≥ 1000 ∧ m < 10000 ∧ 18 ∣ m → n ≤ m) ∧
  n ≥ 1000 ∧ n < 10000 ∧ 18 ∣ n :=
by sorry

end NUMINAMATH_CALUDE_smallest_four_digit_multiple_of_18_l3405_340565


namespace NUMINAMATH_CALUDE_ellipse_and_circle_properties_l3405_340516

/-- An ellipse with the given conditions -/
structure Ellipse where
  a : ℝ
  b : ℝ
  c : ℝ
  h_ab : a > b
  h_b_pos : b > 0
  h_chord : 2 * b^2 / a = 3
  h_foci : 2 * c = a
  h_arithmetic : ∀ (x y : ℝ), x^2/a^2 + y^2/b^2 = 1 →
    ∃ (k : ℝ), (x + c)^2 + y^2 = k^2 ∧
               4 * c^2 = (k + 1)^2 ∧
               (x - c)^2 + y^2 = (k + 2)^2

/-- The theorem to be proved -/
theorem ellipse_and_circle_properties (E : Ellipse) :
  (E.a = 2 ∧ E.b = Real.sqrt 3) ∧
  ∃ (r : ℝ), r^2 = 12/7 ∧
    ∀ (k m : ℝ),
      (∀ (x y : ℝ), y = k*x + m → x^2 + y^2 = r^2) →
      ∃ (x₁ y₁ x₂ y₂ : ℝ),
        x₁^2/E.a^2 + y₁^2/E.b^2 = 1 ∧
        x₂^2/E.a^2 + y₂^2/E.b^2 = 1 ∧
        y₁ = k*x₁ + m ∧
        y₂ = k*x₂ + m ∧
        x₁*x₂ + y₁*y₂ = 0 :=
by sorry

end NUMINAMATH_CALUDE_ellipse_and_circle_properties_l3405_340516


namespace NUMINAMATH_CALUDE_max_value_of_expression_l3405_340522

theorem max_value_of_expression (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (h : a * b * c + a + c = b) :
  2 / (a^2 + 1) - 2 / (b^2 + 1) + 3 / (c^2 + 1) ≤ 26 / 5 := by
  sorry

end NUMINAMATH_CALUDE_max_value_of_expression_l3405_340522
