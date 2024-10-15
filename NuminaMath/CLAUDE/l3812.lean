import Mathlib

namespace NUMINAMATH_CALUDE_product_inequality_l3812_381253

theorem product_inequality (a b c : ℝ) (ha : 0 ≤ a) (hb : 0 ≤ b) (hc : 0 ≤ c) (habc : a * b * c = 1) :
  (a - 1 + 1 / b) * (b - 1 + 1 / c) * (c - 1 + 1 / a) ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_product_inequality_l3812_381253


namespace NUMINAMATH_CALUDE_union_condition_intersection_condition_l3812_381271

-- Define the sets A and B
def A (a : ℝ) : Set ℝ := {x | a - 1 < x ∧ x < 2 * a + 1}
def B : Set ℝ := {x | 0 < x ∧ x < 1}

-- Theorem 1
theorem union_condition (a : ℝ) : A a ∪ B = A a ↔ 0 ≤ a ∧ a ≤ 1 := by sorry

-- Theorem 2
theorem intersection_condition (a : ℝ) : A a ∩ B = ∅ ↔ a ≤ -1/2 ∨ a ≥ 2 := by sorry

end NUMINAMATH_CALUDE_union_condition_intersection_condition_l3812_381271


namespace NUMINAMATH_CALUDE_store_profit_loss_l3812_381252

theorem store_profit_loss (price : ℝ) (profit_margin loss_margin : ℝ) : 
  price = 168 ∧ profit_margin = 0.2 ∧ loss_margin = 0.2 →
  (price - price / (1 + profit_margin)) + (price - price / (1 - loss_margin)) = -14 := by
  sorry

end NUMINAMATH_CALUDE_store_profit_loss_l3812_381252


namespace NUMINAMATH_CALUDE_compressor_stations_configuration_l3812_381225

/-- Represents a triangle with side lengths x, y, and z. -/
structure Triangle where
  x : ℝ
  y : ℝ
  z : ℝ
  triangle_inequality : x < y + z ∧ y < x + z ∧ z < x + y

/-- Theorem about the specific triangle configuration described in the problem. -/
theorem compressor_stations_configuration (a : ℝ) :
  ∃ (t : Triangle),
    t.x + t.y = 3 * t.z ∧
    t.z + t.y = t.x + a ∧
    t.x + t.z = 60 →
    0 < a ∧ a < 60 ∧
    (a = 30 → t.x = 35 ∧ t.y = 40 ∧ t.z = 25) :=
by sorry

#check compressor_stations_configuration

end NUMINAMATH_CALUDE_compressor_stations_configuration_l3812_381225


namespace NUMINAMATH_CALUDE_area_of_shaded_region_l3812_381215

-- Define the side lengths of the squares
def side1 : ℝ := 6
def side2 : ℝ := 8

-- Define π as 3.14
def π : ℝ := 3.14

-- Define the area of the shaded region
def shaded_area : ℝ := 50.24

-- Theorem statement
theorem area_of_shaded_region :
  ∃ (f : ℝ → ℝ → ℝ → ℝ), f side1 side2 π = shaded_area :=
sorry

end NUMINAMATH_CALUDE_area_of_shaded_region_l3812_381215


namespace NUMINAMATH_CALUDE_accepted_to_rejected_ratio_egg_processing_change_l3812_381203

/-- Represents the daily egg processing at a plant -/
structure EggProcessing where
  total : ℕ
  accepted : ℕ
  rejected : ℕ
  h_total : total = accepted + rejected

/-- The original egg processing scenario -/
def original : EggProcessing :=
  { total := 400,
    accepted := 384,
    rejected := 16,
    h_total := rfl }

/-- The modified egg processing scenario -/
def modified : EggProcessing :=
  { total := 400,
    accepted := 396,
    rejected := 4,
    h_total := rfl }

/-- Theorem stating the ratio of accepted to rejected eggs in the modified scenario -/
theorem accepted_to_rejected_ratio :
  modified.accepted / modified.rejected = 99 := by
  sorry

/-- Proof that the ratio of accepted to rejected eggs changes as described -/
theorem egg_processing_change (orig : EggProcessing) (mod : EggProcessing)
  (h_orig : orig = original)
  (h_mod : mod = modified)
  (h_total_unchanged : orig.total = mod.total)
  (h_accepted_increase : mod.accepted = orig.accepted + 12) :
  mod.accepted / mod.rejected = 99 := by
  sorry

end NUMINAMATH_CALUDE_accepted_to_rejected_ratio_egg_processing_change_l3812_381203


namespace NUMINAMATH_CALUDE_arithmetic_sequence_75th_term_l3812_381274

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℚ) : Prop :=
  ∃ d : ℚ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_75th_term
  (a : ℕ → ℚ)
  (h_arith : arithmetic_sequence a)
  (h_15 : a 15 = 8)
  (h_60 : a 60 = 20) :
  a 75 = 24 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_75th_term_l3812_381274


namespace NUMINAMATH_CALUDE_puzzle_pieces_sum_l3812_381240

/-- The number of pieces in the first puzzle -/
def first_puzzle_pieces : ℕ := 1000

/-- The number of pieces in the second and third puzzles -/
def other_puzzle_pieces : ℕ := first_puzzle_pieces + first_puzzle_pieces / 2

/-- The total number of pieces in all puzzles -/
def total_pieces : ℕ := first_puzzle_pieces + 2 * other_puzzle_pieces

theorem puzzle_pieces_sum :
  total_pieces = 4000 :=
by sorry

end NUMINAMATH_CALUDE_puzzle_pieces_sum_l3812_381240


namespace NUMINAMATH_CALUDE_quadratic_integer_roots_l3812_381231

-- Define the quadratic equation
def quadratic_equation (a : ℝ) (x : ℝ) : ℝ := x^2 + a*x + 12*a

-- Define a function to check if a number is an integer
def is_integer (x : ℝ) : Prop := ∃ n : ℤ, x = n

-- Define a function to count the number of real a values that satisfy the condition
def count_a_values : ℕ := sorry

-- Theorem statement
theorem quadratic_integer_roots :
  count_a_values = 15 := by sorry

end NUMINAMATH_CALUDE_quadratic_integer_roots_l3812_381231


namespace NUMINAMATH_CALUDE_imaginary_part_of_product_l3812_381250

theorem imaginary_part_of_product : Complex.im ((3 - 4*Complex.I) * (1 + 2*Complex.I)) = 2 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_product_l3812_381250


namespace NUMINAMATH_CALUDE_arithmetic_sequence_count_100_l3812_381290

/-- The number of ways to select 3 different numbers from 1 to 100 
    that form an arithmetic sequence in their original order -/
def arithmeticSequenceCount : ℕ := 2450

/-- A function that counts the number of arithmetic sequences of length 3
    that can be formed from numbers 1 to n -/
def countArithmeticSequences (n : ℕ) : ℕ :=
  sorry

theorem arithmetic_sequence_count_100 : 
  countArithmeticSequences 100 = arithmeticSequenceCount := by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_count_100_l3812_381290


namespace NUMINAMATH_CALUDE_league_members_l3812_381236

/-- The cost of a pair of socks in dollars -/
def sock_cost : ℕ := 6

/-- The cost of a T-shirt in dollars -/
def tshirt_cost : ℕ := sock_cost + 7

/-- The cost of shorts in dollars -/
def shorts_cost : ℕ := tshirt_cost

/-- The total cost for one member's equipment in dollars -/
def member_cost : ℕ := 2 * (sock_cost + tshirt_cost + shorts_cost)

/-- The total cost for the league's equipment in dollars -/
def total_cost : ℕ := 4719

/-- The number of members in the league -/
def num_members : ℕ := 74

theorem league_members : 
  sock_cost = 6 ∧ 
  tshirt_cost = sock_cost + 7 ∧ 
  shorts_cost = tshirt_cost ∧
  member_cost = 2 * (sock_cost + tshirt_cost + shorts_cost) ∧
  total_cost = 4719 → 
  num_members * member_cost = total_cost :=
by sorry

end NUMINAMATH_CALUDE_league_members_l3812_381236


namespace NUMINAMATH_CALUDE_stationery_store_problem_l3812_381294

/-- Represents the cost and quantity of pencils in a packet -/
structure Packet where
  cost : ℝ
  quantity : ℝ

/-- The stationery store problem -/
theorem stationery_store_problem (a : ℝ) (h_pos : a > 0) :
  let s : Packet := ⟨a, 1⟩
  let m : Packet := ⟨1.2 * a, 1.5⟩
  let l : Packet := ⟨1.6 * a, 1.875⟩
  (m.cost / m.quantity < l.cost / l.quantity) ∧
  (l.cost / l.quantity < s.cost / s.quantity) := by
  sorry

#check stationery_store_problem

end NUMINAMATH_CALUDE_stationery_store_problem_l3812_381294


namespace NUMINAMATH_CALUDE_equilateral_triangle_third_vertex_y_coord_l3812_381247

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- An equilateral triangle with two vertices given -/
structure EquilateralTriangle where
  v1 : Point
  v2 : Point
  third_in_first_quadrant : Bool

/-- The y-coordinate of the third vertex of an equilateral triangle -/
def third_vertex_y_coord (t : EquilateralTriangle) : ℝ :=
  sorry

theorem equilateral_triangle_third_vertex_y_coord 
  (t : EquilateralTriangle) 
  (h1 : t.v1 = ⟨1, 3⟩) 
  (h2 : t.v2 = ⟨9, 3⟩) 
  (h3 : t.third_in_first_quadrant = true) : 
  third_vertex_y_coord t = 3 + 4 * Real.sqrt 3 :=
sorry

end NUMINAMATH_CALUDE_equilateral_triangle_third_vertex_y_coord_l3812_381247


namespace NUMINAMATH_CALUDE_inequality_system_solution_l3812_381230

theorem inequality_system_solution (p : ℝ) :
  (19 * p < 10 ∧ p > 1/2) ↔ (1/2 < p ∧ p < 10/19) := by
  sorry

end NUMINAMATH_CALUDE_inequality_system_solution_l3812_381230


namespace NUMINAMATH_CALUDE_intersection_A_B_l3812_381293

def A : Set ℝ := {x : ℝ | ∃ y : ℝ, y = Real.sqrt (x - 1)}
def B : Set ℝ := {x : ℝ | |x| ≤ 1}

theorem intersection_A_B : A ∩ B = {1} := by sorry

end NUMINAMATH_CALUDE_intersection_A_B_l3812_381293


namespace NUMINAMATH_CALUDE_luncheon_cost_is_105_l3812_381248

/-- The cost of a luncheon consisting of one sandwich, one cup of coffee, and one piece of pie -/
def luncheon_cost (s c p : ℚ) : ℚ := s + c + p

/-- The cost of the first luncheon combination -/
def first_combination (s c p : ℚ) : ℚ := 3 * s + 7 * c + p

/-- The cost of the second luncheon combination -/
def second_combination (s c p : ℚ) : ℚ := 4 * s + 10 * c + p

theorem luncheon_cost_is_105 
  (s c p : ℚ) 
  (h1 : first_combination s c p = 315/100) 
  (h2 : second_combination s c p = 420/100) : 
  luncheon_cost s c p = 105/100 := by
  sorry

end NUMINAMATH_CALUDE_luncheon_cost_is_105_l3812_381248


namespace NUMINAMATH_CALUDE_remaining_money_l3812_381273

def initial_amount : ℕ := 43
def pencil_cost : ℕ := 20
def candy_cost : ℕ := 5

theorem remaining_money :
  initial_amount - (pencil_cost + candy_cost) = 18 := by sorry

end NUMINAMATH_CALUDE_remaining_money_l3812_381273


namespace NUMINAMATH_CALUDE_square_sum_ge_double_product_l3812_381258

theorem square_sum_ge_double_product (a b : ℝ) : (a^2 + b^2 > 2*a*b) ∨ (a^2 + b^2 = 2*a*b) := by
  sorry

end NUMINAMATH_CALUDE_square_sum_ge_double_product_l3812_381258


namespace NUMINAMATH_CALUDE_equal_hire_probability_l3812_381283

/-- Represents the hiring process for a factory with n job openings and n applicants. -/
structure HiringProcess (n : ℕ) where
  (n_ge_3 : n ≥ 3)
  (job_openings : Fin n)
  (applicants : Fin n)
  (qualified : Fin n → Fin n → Prop)
  (qualified_condition : ∀ i j : Fin n, qualified i j ↔ i.val ≥ j.val)
  (arrival_order : Fin n → Fin n)
  (is_hired : Fin n → Prop)

/-- The probability of an applicant being hired. -/
def hire_probability (hp : HiringProcess n) (applicant : Fin n) : ℝ :=
  sorry

/-- Theorem stating that applicants n and n-1 have the same probability of being hired. -/
theorem equal_hire_probability (hp : HiringProcess n) :
  hire_probability hp ⟨n - 1, sorry⟩ = hire_probability hp ⟨n - 2, sorry⟩ :=
sorry

end NUMINAMATH_CALUDE_equal_hire_probability_l3812_381283


namespace NUMINAMATH_CALUDE_total_age_proof_l3812_381255

/-- Given three people a, b, and c, where:
  - a is two years older than b
  - b is twice as old as c
  - b is 8 years old
  Prove that the total of their ages is 22 years. -/
theorem total_age_proof (a b c : ℕ) : 
  b = 8 → a = b + 2 → b = 2 * c → a + b + c = 22 := by
  sorry

end NUMINAMATH_CALUDE_total_age_proof_l3812_381255


namespace NUMINAMATH_CALUDE_sum_divisibility_l3812_381257

theorem sum_divisibility (y : ℕ) : 
  y = 36 + 48 + 72 + 144 + 216 + 432 + 1296 →
  3 ∣ y ∧ 4 ∣ y ∧ 6 ∣ y ∧ 12 ∣ y :=
by sorry

end NUMINAMATH_CALUDE_sum_divisibility_l3812_381257


namespace NUMINAMATH_CALUDE_z_value_theorem_l3812_381298

theorem z_value_theorem (x y z : ℝ) (h : x ≠ 0 ∧ y ≠ 0 ∧ y ≠ x) 
  (eq : 1 / x - 1 / y = 1 / z) : z = (x * y) / (y - x) := by
  sorry

end NUMINAMATH_CALUDE_z_value_theorem_l3812_381298


namespace NUMINAMATH_CALUDE_circle_passes_through_points_circle_equation_equivalence_l3812_381209

-- Define the circle passing through points O(0,0), A(1,1), and B(4,2)
def circle_through_points (x y : ℝ) : Prop :=
  x^2 + y^2 - 8*x + 6*y = 0

-- Define the standard form of the circle
def circle_standard_form (x y : ℝ) : Prop :=
  (x - 4)^2 + (y + 3)^2 = 25

-- Theorem stating that the circle passes through the given points
theorem circle_passes_through_points :
  circle_through_points 0 0 ∧
  circle_through_points 1 1 ∧
  circle_through_points 4 2 := by sorry

-- Theorem stating the equivalence of the general and standard forms
theorem circle_equation_equivalence :
  ∀ x y : ℝ, circle_through_points x y ↔ circle_standard_form x y := by sorry

end NUMINAMATH_CALUDE_circle_passes_through_points_circle_equation_equivalence_l3812_381209


namespace NUMINAMATH_CALUDE_subcommittee_count_l3812_381235

/-- The number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := sorry

/-- The number of valid subcommittees -/
def validSubcommittees (totalMembers teachers subcommitteeSize : ℕ) : ℕ :=
  choose totalMembers subcommitteeSize - choose (totalMembers - teachers) subcommitteeSize

theorem subcommittee_count :
  validSubcommittees 12 5 5 = 771 := by sorry

end NUMINAMATH_CALUDE_subcommittee_count_l3812_381235


namespace NUMINAMATH_CALUDE_symmetric_line_equation_l3812_381263

-- Define the points P and Q
def P : ℝ × ℝ := (3, 2)
def Q : ℝ × ℝ := (1, 4)

-- Define the line l as a function ax + by + c = 0
def line_l (a b c : ℝ) (x y : ℝ) : Prop := a * x + b * y + c = 0

-- Define symmetry with respect to a line
def symmetric_wrt_line (P Q : ℝ × ℝ) (a b c : ℝ) : Prop :=
  let midpoint := ((P.1 + Q.1) / 2, (P.2 + Q.2) / 2)
  line_l a b c midpoint.1 midpoint.2 ∧
  a * (Q.2 - P.2) = b * (P.1 - Q.1)

-- Theorem statement
theorem symmetric_line_equation :
  ∃ (a b c : ℝ), a ≠ 0 ∧ b ≠ 0 ∧ symmetric_wrt_line P Q a b c ∧ line_l a b c = line_l 1 (-1) 1 :=
by sorry

end NUMINAMATH_CALUDE_symmetric_line_equation_l3812_381263


namespace NUMINAMATH_CALUDE_congruentAngles_equivalence_sameRemainder_equivalence_l3812_381217

-- Define a type for angles
structure Angle where
  measure : ℝ

-- Define congruence relation for angles
def congruentAngles (a b : Angle) : Prop := a.measure = b.measure

-- Define a type for integers with a specific modulus
structure ModInt (m : ℕ) where
  value : ℤ

-- Define same remainder relation for ModInt
def sameRemainder {m : ℕ} (a b : ModInt m) : Prop := a.value % m = b.value % m

-- Theorem: Congruence of angles is an equivalence relation
theorem congruentAngles_equivalence : Equivalence congruentAngles := by sorry

-- Theorem: Same remainder when divided by a certain number is an equivalence relation
theorem sameRemainder_equivalence (m : ℕ) : Equivalence (@sameRemainder m) := by sorry

end NUMINAMATH_CALUDE_congruentAngles_equivalence_sameRemainder_equivalence_l3812_381217


namespace NUMINAMATH_CALUDE_max_consecutive_integers_sum_l3812_381222

theorem max_consecutive_integers_sum (n : ℕ) : n = 31 ↔ 
  (n ≥ 3 ∧ 
   (∀ k : ℕ, k ≥ 3 → k ≤ n → (k * (k + 1)) / 2 - 3 ≤ 500) ∧
   (∀ m : ℕ, m > n → (m * (m + 1)) / 2 - 3 > 500)) :=
by sorry

end NUMINAMATH_CALUDE_max_consecutive_integers_sum_l3812_381222


namespace NUMINAMATH_CALUDE_max_sock_pairs_l3812_381206

theorem max_sock_pairs (initial_pairs : ℕ) (lost_socks : ℕ) (max_pairs : ℕ) : 
  initial_pairs = 10 →
  lost_socks = 5 →
  max_pairs = 5 →
  max_pairs = initial_pairs - (lost_socks / 2 + lost_socks % 2) :=
by sorry

end NUMINAMATH_CALUDE_max_sock_pairs_l3812_381206


namespace NUMINAMATH_CALUDE_smallest_value_complex_sum_l3812_381238

theorem smallest_value_complex_sum (p q r : ℤ) (ω : ℂ) : 
  p ≠ q → q ≠ r → r ≠ p → 
  (p = 0 ∨ q = 0 ∨ r = 0) →
  ω^3 = 1 →
  ω ≠ 1 →
  ∃ (min : ℝ), min = Real.sqrt 3 ∧ 
    (∀ (p' q' r' : ℤ), p' ≠ q' → q' ≠ r' → r' ≠ p' → 
      (p' = 0 ∨ q' = 0 ∨ r' = 0) → 
      Complex.abs (↑p' + ↑q' * ω^2 + ↑r' * ω) ≥ min) ∧
    (Complex.abs (↑p + ↑q * ω^2 + ↑r * ω) = min ∨
     Complex.abs (↑q + ↑r * ω^2 + ↑p * ω) = min ∨
     Complex.abs (↑r + ↑p * ω^2 + ↑q * ω) = min) :=
by sorry

end NUMINAMATH_CALUDE_smallest_value_complex_sum_l3812_381238


namespace NUMINAMATH_CALUDE_specific_tetrahedron_volume_l3812_381275

/-- Represents a tetrahedron PQRS with given edge lengths -/
structure Tetrahedron where
  PQ : ℝ
  PR : ℝ
  PS : ℝ
  QR : ℝ
  QS : ℝ
  RS : ℝ

/-- Calculates the volume of a tetrahedron -/
def tetrahedronVolume (t : Tetrahedron) : ℝ := sorry

/-- Theorem stating that the volume of the specific tetrahedron is 10 -/
theorem specific_tetrahedron_volume :
  let t : Tetrahedron := {
    PQ := 3,
    PR := 4,
    PS := 5,
    QR := 5,
    QS := Real.sqrt 34,
    RS := Real.sqrt 41
  }
  tetrahedronVolume t = 10 := by sorry

end NUMINAMATH_CALUDE_specific_tetrahedron_volume_l3812_381275


namespace NUMINAMATH_CALUDE_odd_function_implies_a_zero_l3812_381267

/-- A function f is odd if f(-x) = -f(x) for all x in its domain -/
def IsOdd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

/-- The function f(x) = (x^2+1)(x+a) -/
def f (a : ℝ) : ℝ → ℝ := fun x ↦ (x^2 + 1) * (x + a)

theorem odd_function_implies_a_zero (a : ℝ) : IsOdd (f a) → a = 0 := by
  sorry

end NUMINAMATH_CALUDE_odd_function_implies_a_zero_l3812_381267


namespace NUMINAMATH_CALUDE_caleb_dandelion_puffs_l3812_381218

/-- Represents the problem of Caleb's dandelion puffs distribution --/
def dandelion_puffs_problem (total : ℕ) (sister grandmother dog : ℕ) (friends num_per_friend : ℕ) : Prop :=
  ∃ (mom : ℕ),
    total = mom + sister + grandmother + dog + (friends * num_per_friend) ∧
    total = 40 ∧
    sister = 3 ∧
    grandmother = 5 ∧
    dog = 2 ∧
    friends = 3 ∧
    num_per_friend = 9

/-- The solution to Caleb's dandelion puffs problem --/
theorem caleb_dandelion_puffs :
  dandelion_puffs_problem 40 3 5 2 3 9 → ∃ (mom : ℕ), mom = 3 := by
  sorry

end NUMINAMATH_CALUDE_caleb_dandelion_puffs_l3812_381218


namespace NUMINAMATH_CALUDE_range_of_x_l3812_381268

theorem range_of_x (x : Real) 
  (h1 : 0 ≤ x ∧ x ≤ 2 * Real.pi)
  (h2 : Real.sqrt (1 - Real.sin (2 * x)) = Real.sin x - Real.cos x) :
  π / 4 ≤ x ∧ x ≤ 5 * π / 4 := by
  sorry

end NUMINAMATH_CALUDE_range_of_x_l3812_381268


namespace NUMINAMATH_CALUDE_min_students_with_both_devices_l3812_381208

theorem min_students_with_both_devices (n : ℕ) (laptop_users tablet_users : ℕ) : 
  laptop_users = (3 * n) / 7 →
  tablet_users = (5 * n) / 6 →
  ∃ (both : ℕ), both ≥ 11 ∧ n ≥ laptop_users + tablet_users - both :=
sorry

end NUMINAMATH_CALUDE_min_students_with_both_devices_l3812_381208


namespace NUMINAMATH_CALUDE_parabola_p_value_l3812_381277

/-- A parabola with equation y^2 = 2px and directrix x = -2 has p = 4 -/
theorem parabola_p_value (y x p : ℝ) : 
  (∀ y x, y^2 = 2*p*x) →  -- Condition 1: Parabola equation
  (x = -2)               -- Condition 2: Directrix equation
  → p = 4 :=             -- Conclusion: p = 4
by sorry

end NUMINAMATH_CALUDE_parabola_p_value_l3812_381277


namespace NUMINAMATH_CALUDE_fourth_episode_duration_l3812_381211

theorem fourth_episode_duration (episode1 episode2 episode3 : ℕ) 
  (total_duration : ℕ) (h1 : episode1 = 58) (h2 : episode2 = 62) 
  (h3 : episode3 = 65) (h4 : total_duration = 4 * 60) : 
  total_duration - (episode1 + episode2 + episode3) = 55 := by
  sorry

end NUMINAMATH_CALUDE_fourth_episode_duration_l3812_381211


namespace NUMINAMATH_CALUDE_sqrt_eight_minus_sqrt_two_l3812_381242

theorem sqrt_eight_minus_sqrt_two : Real.sqrt 8 - Real.sqrt 2 = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_eight_minus_sqrt_two_l3812_381242


namespace NUMINAMATH_CALUDE_remainder_problem_l3812_381229

theorem remainder_problem (k : ℕ) (h1 : k > 0) (h2 : k < 84) 
  (h3 : k % 5 = 2) (h4 : k % 6 = 5) (h5 : k % 8 = 7) : k % 9 = 2 := by
  sorry

end NUMINAMATH_CALUDE_remainder_problem_l3812_381229


namespace NUMINAMATH_CALUDE_painted_cubes_4x4x4_l3812_381299

/-- The number of unit cubes with at least one face painted in a 4x4x4 cube -/
def painted_cubes (n : Nat) : Nat :=
  n^3 - (n - 2)^3

/-- The proposition that the number of painted cubes in a 4x4x4 cube is 41 -/
theorem painted_cubes_4x4x4 :
  painted_cubes 4 = 41 := by
  sorry

end NUMINAMATH_CALUDE_painted_cubes_4x4x4_l3812_381299


namespace NUMINAMATH_CALUDE_inequality_proof_l3812_381221

theorem inequality_proof (x y z : ℝ) 
  (h_pos_x : x > 0) (h_pos_y : y > 0) (h_pos_z : z > 0)
  (h_sum : x + y + z = 1) : 
  (x^4)/(y*(1-y^2)) + (y^4)/(z*(1-z^2)) + (z^4)/(x*(1-x^2)) ≥ 1/8 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3812_381221


namespace NUMINAMATH_CALUDE_zoo_feeding_days_l3812_381297

def num_lions : ℕ := 3
def num_tigers : ℕ := 2
def num_leopards : ℕ := 5
def num_hyenas : ℕ := 4

def lion_consumption : ℕ := 25
def tiger_consumption : ℕ := 20
def leopard_consumption : ℕ := 15
def hyena_consumption : ℕ := 10

def total_meat : ℕ := 1200

def daily_consumption : ℕ :=
  num_lions * lion_consumption +
  num_tigers * tiger_consumption +
  num_leopards * leopard_consumption +
  num_hyenas * hyena_consumption

theorem zoo_feeding_days :
  (total_meat / daily_consumption : ℕ) = 5 := by sorry

end NUMINAMATH_CALUDE_zoo_feeding_days_l3812_381297


namespace NUMINAMATH_CALUDE_house_price_ratio_l3812_381254

def total_price : ℕ := 600000
def first_house_price : ℕ := 200000

theorem house_price_ratio :
  (total_price - first_house_price) / first_house_price = 2 :=
by sorry

end NUMINAMATH_CALUDE_house_price_ratio_l3812_381254


namespace NUMINAMATH_CALUDE_min_tangent_length_l3812_381220

/-- The minimum length of a tangent from a point on y = x + 2 to (x-3)² + (y+1)² = 2 is 4 -/
theorem min_tangent_length : 
  let line := {p : ℝ × ℝ | p.2 = p.1 + 2}
  let circle := {p : ℝ × ℝ | (p.1 - 3)^2 + (p.2 + 1)^2 = 2}
  ∃ (min_length : ℝ), 
    (∀ (p : ℝ × ℝ) (q : ℝ × ℝ), p ∈ line → q ∈ circle → 
      ‖p - q‖ ≥ min_length) ∧
    (∃ (p : ℝ × ℝ) (q : ℝ × ℝ), p ∈ line ∧ q ∈ circle ∧ ‖p - q‖ = min_length) ∧
    min_length = 4 :=
by
  sorry

end NUMINAMATH_CALUDE_min_tangent_length_l3812_381220


namespace NUMINAMATH_CALUDE_classmate_height_most_suitable_for_census_l3812_381205

/-- Represents a survey option -/
inductive SurveyOption
  | LightBulbLifespan
  | ClassmateHeight
  | NationwideStudentViewing
  | MissileAccuracy

/-- Characteristics of a survey -/
structure SurveyCharacteristics where
  population_size : ℕ
  is_destructive : Bool
  data_collection_difficulty : ℕ

/-- Defines the characteristics of each survey option -/
def survey_characteristics : SurveyOption → SurveyCharacteristics
  | SurveyOption.LightBulbLifespan => ⟨100, true, 5⟩
  | SurveyOption.ClassmateHeight => ⟨30, false, 1⟩
  | SurveyOption.NationwideStudentViewing => ⟨1000000, false, 8⟩
  | SurveyOption.MissileAccuracy => ⟨50, true, 7⟩

/-- Determines if a survey is suitable for a census based on its characteristics -/
def is_census_suitable (s : SurveyCharacteristics) : Bool :=
  s.population_size ≤ 100 ∧ ¬s.is_destructive ∧ s.data_collection_difficulty ≤ 3

/-- Theorem stating that classmate height survey is most suitable for census -/
theorem classmate_height_most_suitable_for_census :
  ∀ (option : SurveyOption),
    option ≠ SurveyOption.ClassmateHeight →
    is_census_suitable (survey_characteristics SurveyOption.ClassmateHeight) ∧
    ¬is_census_suitable (survey_characteristics option) :=
  sorry


end NUMINAMATH_CALUDE_classmate_height_most_suitable_for_census_l3812_381205


namespace NUMINAMATH_CALUDE_expression_simplification_l3812_381284

theorem expression_simplification (m n x : ℚ) :
  (5 * m + 3 * n - 7 * m - n = -2 * m + 2 * n) ∧
  (2 * x^2 - (3 * x - 2 * (x^2 - x + 3) + 2 * x^2) = 2 * x^2 - 5 * x + 6) :=
by sorry

end NUMINAMATH_CALUDE_expression_simplification_l3812_381284


namespace NUMINAMATH_CALUDE_blood_concentration_reaches_target_target_time_is_correct_l3812_381295

/-- Represents the blood drug concentration at a given time -/
def blood_concentration (peak_concentration : ℝ) (time : ℕ) : ℝ :=
  if time ≤ 3 then peak_concentration
  else peak_concentration * (0.4 ^ ((time - 3) / 2))

/-- Theorem stating that the blood concentration reaches 1.024% of peak after 13 hours -/
theorem blood_concentration_reaches_target (peak_concentration : ℝ) :
  blood_concentration peak_concentration 13 = 0.01024 * peak_concentration :=
by
  sorry

/-- Time when blood concentration reaches 1.024% of peak -/
def target_time : ℕ := 13

/-- Theorem proving that target_time is correct -/
theorem target_time_is_correct (peak_concentration : ℝ) :
  blood_concentration peak_concentration target_time = 0.01024 * peak_concentration :=
by
  sorry

end NUMINAMATH_CALUDE_blood_concentration_reaches_target_target_time_is_correct_l3812_381295


namespace NUMINAMATH_CALUDE_f_of_a_plus_one_l3812_381207

/-- The function f(x) = x^2 + 1 -/
def f (x : ℝ) : ℝ := x^2 + 1

/-- Theorem: For the function f(x) = x^2 + 1, f(a+1) = a^2 + 2a + 2 for any real number a -/
theorem f_of_a_plus_one (a : ℝ) : f (a + 1) = a^2 + 2*a + 2 := by
  sorry

end NUMINAMATH_CALUDE_f_of_a_plus_one_l3812_381207


namespace NUMINAMATH_CALUDE_equality_of_positive_integers_l3812_381270

theorem equality_of_positive_integers (a b : ℕ+) (p : ℕ) (h_prime : Nat.Prime p) 
  (h_p_eq : p = a + b + 1) (h_divides : p ∣ 4 * a * b - 1) : a = b := by
  sorry

end NUMINAMATH_CALUDE_equality_of_positive_integers_l3812_381270


namespace NUMINAMATH_CALUDE_hyperbola_equation_l3812_381237

theorem hyperbola_equation (a b : ℝ) (h1 : a > 0) (h2 : b > 0) : 
  (∃ x y : ℝ, x^2/a^2 - y^2/b^2 = 1) →
  ((a - 4)^2 + b^2 = 16) →
  (a^2 + b^2 = 16) →
  (x^2/4 - y^2/12 = 1) := by
sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l3812_381237


namespace NUMINAMATH_CALUDE_least_digit_sum_multiple_2003_l3812_381292

/-- Sum of decimal digits of a natural number -/
def S (n : ℕ) : ℕ := sorry

/-- The least value of S(m) where m is a multiple of 2003 -/
theorem least_digit_sum_multiple_2003 : 
  (∃ m : ℕ, m % 2003 = 0 ∧ S m = 3) ∧ 
  (∀ m : ℕ, m % 2003 = 0 → S m ≥ 3) := by sorry

end NUMINAMATH_CALUDE_least_digit_sum_multiple_2003_l3812_381292


namespace NUMINAMATH_CALUDE_boat_stream_speed_l3812_381241

/-- Proves that the speed of a stream is 5 km/hr given the conditions of the boat problem -/
theorem boat_stream_speed 
  (boat_speed : ℝ) 
  (distance : ℝ) 
  (time : ℝ) 
  (h1 : boat_speed = 22) 
  (h2 : distance = 81) 
  (h3 : time = 3) : 
  ∃ stream_speed : ℝ, 
    stream_speed = 5 ∧ 
    (boat_speed + stream_speed) * time = distance := by
  sorry

end NUMINAMATH_CALUDE_boat_stream_speed_l3812_381241


namespace NUMINAMATH_CALUDE_ngo_employees_l3812_381200

/-- The number of illiterate employees -/
def num_illiterate : ℕ := 20

/-- The decrease in daily average wages of illiterate employees in Rs -/
def wage_decrease_illiterate : ℕ := 15

/-- The decrease in average salary of all employees in Rs per day -/
def avg_salary_decrease : ℕ := 10

/-- The number of literate employees -/
def num_literate : ℕ := 10

theorem ngo_employees :
  num_literate = 10 :=
by sorry

end NUMINAMATH_CALUDE_ngo_employees_l3812_381200


namespace NUMINAMATH_CALUDE_min_value_sum_reciprocals_l3812_381249

theorem min_value_sum_reciprocals (p q r s t u : ℝ) 
  (hp : p > 0) (hq : q > 0) (hr : r > 0) (hs : s > 0) (ht : t > 0) (hu : u > 0)
  (sum_eq_8 : p + q + r + s + t + u = 8) : 
  (1/p + 4/q + 9/r + 16/s + 25/t + 49/u) ≥ 60.5 := by
  sorry

end NUMINAMATH_CALUDE_min_value_sum_reciprocals_l3812_381249


namespace NUMINAMATH_CALUDE_line_parallel_value_l3812_381232

-- Define the lines l₁ and l₂
def l₁ (a : ℝ) (x y : ℝ) : Prop := a * x + 2 * y + 6 = 0
def l₂ (a : ℝ) (x y : ℝ) : Prop := x + (a - 1) * y + (a^2 - 1) = 0

-- Define the parallel condition for two lines
def parallel (A₁ B₁ C₁ A₂ B₂ C₂ : ℝ) : Prop :=
  (A₁ * B₂ - A₂ * B₁ = 0) ∧ ((A₁ * C₂ - A₂ * C₁ ≠ 0) ∨ (B₁ * C₂ - B₂ * C₁ ≠ 0))

-- Define the coincident condition for two lines
def coincident (A₁ B₁ C₁ A₂ B₂ C₂ : ℝ) : Prop :=
  (A₁ * B₂ - A₂ * B₁ = 0) ∧ (A₁ * C₂ - A₂ * C₁ = 0) ∧ (B₁ * C₂ - B₂ * C₁ = 0)

-- Theorem statement
theorem line_parallel_value (a : ℝ) : 
  (parallel a 2 6 1 (a-1) (a^2-1)) ∧ 
  ¬(coincident a 2 6 1 (a-1) (a^2-1)) → 
  a = -1 := by sorry

end NUMINAMATH_CALUDE_line_parallel_value_l3812_381232


namespace NUMINAMATH_CALUDE_matthews_crackers_l3812_381280

/-- The number of friends Matthew has -/
def num_friends : ℕ := 4

/-- The number of cakes Matthew had initially -/
def initial_cakes : ℕ := 8

/-- The number of cakes each person ate -/
def cakes_eaten_per_person : ℕ := 2

/-- The number of crackers Matthew had initially -/
def initial_crackers : ℕ := 8

theorem matthews_crackers :
  initial_crackers = 8 :=
by
  sorry

end NUMINAMATH_CALUDE_matthews_crackers_l3812_381280


namespace NUMINAMATH_CALUDE_shaded_area_is_110_l3812_381214

/-- Represents a triangle inscribed in a hexagon --/
inductive InscribedTriangle
  | Small
  | Medium
  | Large

/-- The area of an inscribed triangle in terms of the number of unit triangles it contains --/
def triangle_area (t : InscribedTriangle) : ℕ :=
  match t with
  | InscribedTriangle.Small => 1
  | InscribedTriangle.Medium => 3
  | InscribedTriangle.Large => 7

/-- The area of a unit equilateral triangle in the hexagon --/
def unit_triangle_area : ℕ := 10

/-- The total area of the shaded part --/
def shaded_area : ℕ :=
  (triangle_area InscribedTriangle.Small +
   triangle_area InscribedTriangle.Medium +
   triangle_area InscribedTriangle.Large) * unit_triangle_area

theorem shaded_area_is_110 : shaded_area = 110 := by
  sorry


end NUMINAMATH_CALUDE_shaded_area_is_110_l3812_381214


namespace NUMINAMATH_CALUDE_gcd_12m_18n_lower_bound_l3812_381228

theorem gcd_12m_18n_lower_bound (m n : ℕ+) (h : Nat.gcd m n = 18) :
  Nat.gcd (12 * m) (18 * n) ≥ 108 := by
  sorry

end NUMINAMATH_CALUDE_gcd_12m_18n_lower_bound_l3812_381228


namespace NUMINAMATH_CALUDE_probability_different_colors_7_5_l3812_381272

/-- The probability of drawing two chips of different colors from a bag -/
def probability_different_colors (blue_chips yellow_chips : ℕ) : ℚ :=
  let total_chips := blue_chips + yellow_chips
  let prob_blue_then_yellow := (blue_chips : ℚ) / total_chips * yellow_chips / (total_chips - 1)
  let prob_yellow_then_blue := (yellow_chips : ℚ) / total_chips * blue_chips / (total_chips - 1)
  prob_blue_then_yellow + prob_yellow_then_blue

/-- Theorem stating the probability of drawing two chips of different colors -/
theorem probability_different_colors_7_5 :
  probability_different_colors 7 5 = 35 / 66 := by
  sorry

end NUMINAMATH_CALUDE_probability_different_colors_7_5_l3812_381272


namespace NUMINAMATH_CALUDE_sons_age_l3812_381246

/-- Proves that given the conditions, the son's age is 35 years. -/
theorem sons_age (son_age father_age : ℕ) : 
  father_age = son_age + 37 →
  father_age + 2 = 2 * (son_age + 2) →
  son_age = 35 := by
sorry

end NUMINAMATH_CALUDE_sons_age_l3812_381246


namespace NUMINAMATH_CALUDE_impossibility_of_tiling_l3812_381251

/-- Represents a checkerboard -/
structure Checkerboard :=
  (rows : ℕ)
  (cols : ℕ)
  (missing_corner : Bool)

/-- Represents a trimino -/
structure Trimino :=
  (length : ℕ)
  (width : ℕ)

/-- Determines if a checkerboard can be tiled with triminos -/
def can_tile (board : Checkerboard) (tile : Trimino) : Prop :=
  ∃ (tiling : ℕ), 
    (board.rows * board.cols - if board.missing_corner then 1 else 0) = 
    tiling * (tile.length * tile.width)

theorem impossibility_of_tiling (board : Checkerboard) (tile : Trimino) : 
  (board.rows = 8 ∧ board.cols = 8 ∧ tile.length = 3 ∧ tile.width = 1) →
  (¬ can_tile board tile) ∧ 
  (¬ can_tile {rows := board.rows, cols := board.cols, missing_corner := true} tile) :=
sorry

end NUMINAMATH_CALUDE_impossibility_of_tiling_l3812_381251


namespace NUMINAMATH_CALUDE_student_arrangement_l3812_381201

theorem student_arrangement (n : ℕ) (k : ℕ) (m : ℕ) : 
  n = 7 → k = 6 → m = 3 → 
  (n.choose k) * (k.choose m) * ((k - m).choose m) = 140 :=
sorry

end NUMINAMATH_CALUDE_student_arrangement_l3812_381201


namespace NUMINAMATH_CALUDE_milk_can_problem_l3812_381278

theorem milk_can_problem :
  ∃! (x y : ℕ), 10 * x + 17 * y = 206 :=
by sorry

end NUMINAMATH_CALUDE_milk_can_problem_l3812_381278


namespace NUMINAMATH_CALUDE_rectangle_cut_perimeter_l3812_381261

/-- Given a rectangle with perimeter 10, prove that when cut twice parallel to its
    length and width to form 9 smaller rectangles, the total perimeter of these
    9 rectangles is 30. -/
theorem rectangle_cut_perimeter (a b : ℝ) : 
  (2 * (a + b) = 10) →  -- Perimeter of original rectangle
  (∃ x y z w : ℝ, 
    x > 0 ∧ y > 0 ∧ z > 0 ∧ w > 0 ∧  -- Cuts are positive
    x + y + z = a ∧ w + y + z = b) →  -- Cuts divide length and width
  (2 * (a + b) + 4 * (a + b) = 30) :=  -- Total perimeter after cuts
by sorry

end NUMINAMATH_CALUDE_rectangle_cut_perimeter_l3812_381261


namespace NUMINAMATH_CALUDE_area_of_twelve_sided_figure_l3812_381224

/-- A vertex is represented by its x and y coordinates -/
structure Vertex :=
  (x : ℝ)
  (y : ℝ)

/-- A polygon is represented by a list of vertices -/
def Polygon := List Vertex

/-- The vertices of our 12-sided figure -/
def twelveSidedFigure : Polygon := [
  ⟨1, 3⟩, ⟨2, 4⟩, ⟨2, 5⟩, ⟨3, 6⟩, ⟨4, 6⟩, ⟨5, 5⟩,
  ⟨6, 4⟩, ⟨6, 3⟩, ⟨5, 2⟩, ⟨4, 1⟩, ⟨3, 1⟩, ⟨2, 2⟩
]

/-- Function to calculate the area of a polygon -/
def areaOfPolygon (p : Polygon) : ℝ := sorry

/-- Theorem stating that the area of our 12-sided figure is 16 cm² -/
theorem area_of_twelve_sided_figure :
  areaOfPolygon twelveSidedFigure = 16 := by sorry

end NUMINAMATH_CALUDE_area_of_twelve_sided_figure_l3812_381224


namespace NUMINAMATH_CALUDE_cookie_sales_difference_l3812_381266

/-- The number of cookie boxes sold by Kim -/
def kim_boxes : ℕ := 54

/-- The number of cookie boxes sold by Jennifer -/
def jennifer_boxes : ℕ := 71

/-- Theorem stating the difference in cookie sales between Jennifer and Kim -/
theorem cookie_sales_difference :
  jennifer_boxes > kim_boxes ∧
  jennifer_boxes - kim_boxes = 17 :=
sorry

end NUMINAMATH_CALUDE_cookie_sales_difference_l3812_381266


namespace NUMINAMATH_CALUDE_distinguishable_triangles_count_l3812_381243

/-- Represents the number of available colors for the triangles -/
def num_colors : ℕ := 8

/-- Represents the number of smaller triangles used to construct a large triangle -/
def triangles_per_large : ℕ := 4

/-- Calculates the number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ :=
  if k > n then 0
  else Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

/-- Calculates the number of distinguishable large triangles -/
def num_distinguishable_triangles : ℕ :=
  let corner_same := num_colors -- All corners same color
  let corner_two_same := num_colors * (num_colors - 1) -- Two corners same, one different
  let corner_all_diff := choose num_colors 3 -- All corners different
  let total_corners := corner_same + corner_two_same + corner_all_diff
  total_corners * num_colors -- Multiply by choices for center triangle

theorem distinguishable_triangles_count :
  num_distinguishable_triangles = 960 :=
sorry

end NUMINAMATH_CALUDE_distinguishable_triangles_count_l3812_381243


namespace NUMINAMATH_CALUDE_angle_properties_l3812_381281

theorem angle_properties (a θ : ℝ) (h : a > 0) 
  (h_point : ∃ (x y : ℝ), x = 3 * a ∧ y = 4 * a ∧ (Real.cos θ = x / Real.sqrt (x^2 + y^2)) ∧ (Real.sin θ = y / Real.sqrt (x^2 + y^2))) :
  (Real.sin θ = 4/5) ∧ 
  (Real.sin (3 * Real.pi / 2 - θ) + Real.cos (θ - Real.pi) = -6/5) := by
  sorry


end NUMINAMATH_CALUDE_angle_properties_l3812_381281


namespace NUMINAMATH_CALUDE_smallest_common_flock_size_l3812_381260

theorem smallest_common_flock_size : ∃ n : ℕ, 
  n > 0 ∧ 
  n % 13 = 0 ∧ 
  n % 14 = 0 ∧ 
  (∀ m : ℕ, m > 0 → m % 13 = 0 → m % 14 = 0 → m ≥ n) ∧
  n = 182 := by
sorry

end NUMINAMATH_CALUDE_smallest_common_flock_size_l3812_381260


namespace NUMINAMATH_CALUDE_binomial_13_11_l3812_381286

theorem binomial_13_11 : Nat.choose 13 11 = 78 := by
  sorry

end NUMINAMATH_CALUDE_binomial_13_11_l3812_381286


namespace NUMINAMATH_CALUDE_correct_calculation_l3812_381256

theorem correct_calculation (a : ℝ) : 8 * a^2 - 5 * a^2 = 3 * a^2 := by
  sorry

end NUMINAMATH_CALUDE_correct_calculation_l3812_381256


namespace NUMINAMATH_CALUDE_arithmetic_equation_l3812_381296

theorem arithmetic_equation : 
  (5 / 6 : ℚ) - (-2 : ℚ) + (1 + 1 / 6 : ℚ) = 4 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_equation_l3812_381296


namespace NUMINAMATH_CALUDE_cost_price_calculation_l3812_381202

def selling_price : ℝ := 24000

def discount_rate : ℝ := 0.1

def profit_rate : ℝ := 0.08

theorem cost_price_calculation (cp : ℝ) : 
  cp = 20000 ↔ 
  (selling_price * (1 - discount_rate) = cp * (1 + profit_rate)) ∧
  (selling_price > 0) ∧ 
  (cp > 0) :=
sorry

end NUMINAMATH_CALUDE_cost_price_calculation_l3812_381202


namespace NUMINAMATH_CALUDE_triangle_angle_measure_l3812_381216

theorem triangle_angle_measure (A B C : Real) (a b c : Real) :
  -- Given conditions
  (4 * (Real.cos A)^2 + 4 * Real.cos B * Real.cos C + 1 = 4 * Real.sin B * Real.sin C) →
  (A < B) →
  (a = 2 * Real.sqrt 3) →
  (a / Real.sin A = 4) →  -- Circumradius condition
  -- Conclusion
  A = π / 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_measure_l3812_381216


namespace NUMINAMATH_CALUDE_smallest_addition_for_divisibility_l3812_381212

theorem smallest_addition_for_divisibility : ∃! x : ℕ, 
  (x ≤ y → ∀ y : ℕ, (758492136547 + y) % 17 = 0 ∧ (758492136547 + y) % 3 = 0) ∧
  (758492136547 + x) % 17 = 0 ∧ 
  (758492136547 + x) % 3 = 0 := by
  sorry

end NUMINAMATH_CALUDE_smallest_addition_for_divisibility_l3812_381212


namespace NUMINAMATH_CALUDE_sqrt_five_lt_sqrt_two_plus_one_l3812_381288

theorem sqrt_five_lt_sqrt_two_plus_one : Real.sqrt 5 < Real.sqrt 2 + 1 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_five_lt_sqrt_two_plus_one_l3812_381288


namespace NUMINAMATH_CALUDE_semicircle_area_with_inscribed_rectangle_l3812_381204

theorem semicircle_area_with_inscribed_rectangle (π : Real) :
  let rectangle_width : Real := 1
  let rectangle_length : Real := 3
  let diameter : Real := (rectangle_width ^ 2 + rectangle_length ^ 2).sqrt
  let radius : Real := diameter / 2
  let semicircle_area : Real := π * radius ^ 2 / 2
  semicircle_area = 13 * π / 8 := by
  sorry

end NUMINAMATH_CALUDE_semicircle_area_with_inscribed_rectangle_l3812_381204


namespace NUMINAMATH_CALUDE_unique_integer_solution_l3812_381244

theorem unique_integer_solution : ∃! (x : ℤ), (45 + x / 89) * 89 = 4028 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_unique_integer_solution_l3812_381244


namespace NUMINAMATH_CALUDE_student_average_age_l3812_381282

theorem student_average_age 
  (num_students : ℕ) 
  (teacher_age : ℕ) 
  (total_average : ℕ) 
  (h1 : num_students = 30)
  (h2 : teacher_age = 46)
  (h3 : total_average = 16)
  : (((num_students + 1) * total_average - teacher_age) / num_students : ℚ) = 15 := by
  sorry

end NUMINAMATH_CALUDE_student_average_age_l3812_381282


namespace NUMINAMATH_CALUDE_min_value_of_f_l3812_381245

/-- The function to be minimized -/
def f (x y : ℝ) : ℝ := 2 * x^2 + 3 * x * y + 4 * y^2 - 8 * x - 6 * y

/-- The theorem stating that the minimum value of f is -14 -/
theorem min_value_of_f :
  ∃ (min : ℝ), min = -14 ∧ ∀ (x y : ℝ), f x y ≥ min :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_f_l3812_381245


namespace NUMINAMATH_CALUDE_prob_tails_heads_heads_l3812_381233

-- Define a coin flip as a type with two possible outcomes
inductive CoinFlip : Type
| Heads : CoinFlip
| Tails : CoinFlip

-- Define a sequence of three coin flips
def ThreeFlips := (CoinFlip × CoinFlip × CoinFlip)

-- Define the probability of getting tails on a single flip
def prob_tails : ℚ := 1 / 2

-- Define the desired outcome: Tails, Heads, Heads
def desired_outcome : ThreeFlips := (CoinFlip.Tails, CoinFlip.Heads, CoinFlip.Heads)

-- Theorem: The probability of getting the desired outcome is 1/8
theorem prob_tails_heads_heads : 
  (prob_tails * (1 - prob_tails) * (1 - prob_tails) : ℚ) = 1 / 8 := by
  sorry

end NUMINAMATH_CALUDE_prob_tails_heads_heads_l3812_381233


namespace NUMINAMATH_CALUDE_factorial_equation_solutions_l3812_381223

theorem factorial_equation_solutions :
  ∀ x y z : ℕ+,
    (2 ^ x.val + 3 ^ y.val - 7 = Nat.factorial z.val) ↔ 
    ((x, y, z) = (2, 2, 3) ∨ (x, y, z) = (2, 3, 4)) := by
  sorry

end NUMINAMATH_CALUDE_factorial_equation_solutions_l3812_381223


namespace NUMINAMATH_CALUDE_cellphone_selection_theorem_l3812_381226

/-- The number of service providers -/
def total_providers : ℕ := 25

/-- The number of siblings (including Laura) -/
def num_siblings : ℕ := 4

/-- The number of ways to select providers for all siblings -/
def ways_to_select_providers : ℕ := 
  (total_providers - 1) * (total_providers - 2) * (total_providers - 3)

theorem cellphone_selection_theorem :
  ways_to_select_providers = 12144 := by
  sorry

end NUMINAMATH_CALUDE_cellphone_selection_theorem_l3812_381226


namespace NUMINAMATH_CALUDE_jessie_muffins_theorem_l3812_381279

/-- The number of muffins made when Jessie and her friends each receive an equal amount -/
def total_muffins (num_friends : ℕ) (muffins_per_person : ℕ) : ℕ :=
  (num_friends + 1) * muffins_per_person

/-- Theorem stating that when Jessie has 4 friends and each person gets 4 muffins, the total is 20 -/
theorem jessie_muffins_theorem :
  total_muffins 4 4 = 20 := by
  sorry

end NUMINAMATH_CALUDE_jessie_muffins_theorem_l3812_381279


namespace NUMINAMATH_CALUDE_fire_in_city_a_l3812_381291

-- Define the cities
inductive City
| A
| B
| C

-- Define the possible statements
inductive Statement
| Fire
| LocationC

-- Define the behavior of residents in each city
def always_truth (c : City) : Prop :=
  c = City.A

def always_lie (c : City) : Prop :=
  c = City.B

def alternate (c : City) : Prop :=
  c = City.C

-- Define the caller's statements
def caller_statements : List Statement :=
  [Statement.Fire, Statement.LocationC]

-- Define the property of the actual fire location
def is_actual_fire_location (c : City) : Prop :=
  ∀ (s : Statement), s ∈ caller_statements → 
    (always_truth c → s = Statement.Fire) ∧
    (always_lie c → s ≠ Statement.LocationC) ∧
    (alternate c → (s = Statement.Fire ↔ s ≠ Statement.LocationC))

-- Theorem: The actual fire location is City A
theorem fire_in_city_a :
  is_actual_fire_location City.A :=
sorry

end NUMINAMATH_CALUDE_fire_in_city_a_l3812_381291


namespace NUMINAMATH_CALUDE_quadratic_expression_value_l3812_381265

theorem quadratic_expression_value (x y : ℝ) 
  (h1 : 3 * x + 2 * y = 7) 
  (h2 : 2 * x + 3 * y = 8) : 
  13 * x^2 + 24 * x * y + 13 * y^2 = 113 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_expression_value_l3812_381265


namespace NUMINAMATH_CALUDE_equal_revenue_both_options_l3812_381259

/-- Represents the fishing company's financial model -/
structure FishingCompany where
  initial_cost : ℕ
  first_year_expenses : ℕ
  annual_expense_increase : ℕ
  annual_revenue : ℕ

/-- Calculates the net profit for a given number of years -/
def net_profit (company : FishingCompany) (years : ℕ) : ℤ :=
  (company.annual_revenue * years : ℤ) -
  ((company.first_year_expenses + (years - 1) * company.annual_expense_increase / 2) * years : ℤ) -
  company.initial_cost

/-- Calculates the total revenue when selling at maximum average annual profit -/
def revenue_max_avg_profit (company : FishingCompany) (sell_price : ℕ) : ℤ :=
  net_profit company 7 + sell_price

/-- Calculates the total revenue when selling at maximum total net profit -/
def revenue_max_total_profit (company : FishingCompany) (sell_price : ℕ) : ℤ :=
  net_profit company 10 + sell_price

/-- Theorem stating that both selling options result in the same total revenue -/
theorem equal_revenue_both_options (company : FishingCompany) :
  revenue_max_avg_profit company 2600000 = revenue_max_total_profit company 800000 :=
by sorry

end NUMINAMATH_CALUDE_equal_revenue_both_options_l3812_381259


namespace NUMINAMATH_CALUDE_daily_harvest_l3812_381239

/-- The number of sacks harvested from each section daily -/
def sacks_per_section : ℕ := 65

/-- The number of sections in the orchard -/
def number_of_sections : ℕ := 12

/-- The total number of sacks harvested daily -/
def total_sacks : ℕ := sacks_per_section * number_of_sections

theorem daily_harvest : total_sacks = 780 := by
  sorry

end NUMINAMATH_CALUDE_daily_harvest_l3812_381239


namespace NUMINAMATH_CALUDE_bake_sale_group_composition_l3812_381213

theorem bake_sale_group_composition (total : ℕ) (girls : ℕ) : 
  girls = (60 : ℕ) * total / 100 →
  (girls - 3 : ℕ) * 2 = total →
  girls = 18 :=
by
  sorry

end NUMINAMATH_CALUDE_bake_sale_group_composition_l3812_381213


namespace NUMINAMATH_CALUDE_proposition_p_equivalence_l3812_381234

theorem proposition_p_equivalence :
  (∃ x, x < 1 ∧ x^2 < 1) ↔ ¬(∀ x, x < 1 → x^2 ≥ 1) :=
by sorry

end NUMINAMATH_CALUDE_proposition_p_equivalence_l3812_381234


namespace NUMINAMATH_CALUDE_auston_taller_than_emma_l3812_381276

def inch_to_cm (inches : ℝ) : ℝ := inches * 2.54

def height_difference_cm (auston_height_inch : ℝ) (emma_height_inch : ℝ) : ℝ :=
  inch_to_cm auston_height_inch - inch_to_cm emma_height_inch

theorem auston_taller_than_emma : 
  height_difference_cm 60 54 = 15.24 := by sorry

end NUMINAMATH_CALUDE_auston_taller_than_emma_l3812_381276


namespace NUMINAMATH_CALUDE_plush_bear_distribution_l3812_381264

theorem plush_bear_distribution (total_bears : ℕ) (kindergarten_bears : ℕ) (num_classes : ℕ) :
  total_bears = 48 →
  kindergarten_bears = 15 →
  num_classes = 3 →
  (total_bears - kindergarten_bears) / num_classes = 11 :=
by sorry

end NUMINAMATH_CALUDE_plush_bear_distribution_l3812_381264


namespace NUMINAMATH_CALUDE_square_count_figure_100_l3812_381210

/-- Represents the number of squares in the nth figure -/
def f (n : ℕ) : ℕ := 3 * n^2 + 3 * n + 1

theorem square_count_figure_100 :
  f 0 = 1 ∧ f 1 = 7 ∧ f 2 = 19 ∧ f 3 = 37 → f 100 = 30301 := by
  sorry

end NUMINAMATH_CALUDE_square_count_figure_100_l3812_381210


namespace NUMINAMATH_CALUDE_division_of_terms_l3812_381262

theorem division_of_terms (a b : ℝ) (h : b ≠ 0) : 3 * a^2 * b / b = 3 * a^2 := by
  sorry

end NUMINAMATH_CALUDE_division_of_terms_l3812_381262


namespace NUMINAMATH_CALUDE_greatest_integer_for_fraction_twenty_nine_satisfies_twenty_nine_is_greatest_l3812_381289

def is_integer (x : ℝ) : Prop := ∃ n : ℤ, x = n

theorem greatest_integer_for_fraction : 
  ∀ x : ℤ, (is_integer ((x^2 + 3*x + 8) / (x - 3))) → x ≤ 29 :=
by sorry

theorem twenty_nine_satisfies :
  is_integer ((29^2 + 3*29 + 8) / (29 - 3)) :=
by sorry

theorem twenty_nine_is_greatest :
  ∀ x : ℤ, x > 29 → ¬(is_integer ((x^2 + 3*x + 8) / (x - 3))) :=
by sorry

end NUMINAMATH_CALUDE_greatest_integer_for_fraction_twenty_nine_satisfies_twenty_nine_is_greatest_l3812_381289


namespace NUMINAMATH_CALUDE_equidistant_point_x_coordinate_l3812_381285

theorem equidistant_point_x_coordinate :
  ∃ (x y : ℝ),
    (abs x = abs y) ∧                             -- Equally distant from x-axis and y-axis
    (abs x = abs (x + y - 3) / Real.sqrt 2) ∧     -- Equally distant from the line x + y = 3
    (x = 3/2) := by
  sorry

end NUMINAMATH_CALUDE_equidistant_point_x_coordinate_l3812_381285


namespace NUMINAMATH_CALUDE_juniors_average_score_l3812_381219

theorem juniors_average_score 
  (total_students : ℝ) 
  (junior_ratio : ℝ) 
  (senior_ratio : ℝ) 
  (class_average : ℝ) 
  (senior_average : ℝ) 
  (h1 : junior_ratio = 0.2)
  (h2 : senior_ratio = 0.8)
  (h3 : junior_ratio + senior_ratio = 1)
  (h4 : class_average = 86)
  (h5 : senior_average = 85) :
  (class_average * total_students - senior_average * (senior_ratio * total_students)) / (junior_ratio * total_students) = 90 :=
by sorry

end NUMINAMATH_CALUDE_juniors_average_score_l3812_381219


namespace NUMINAMATH_CALUDE_ship_length_proof_l3812_381269

/-- The length of the ship in terms of Emily's normal steps -/
def ship_length : ℕ := 120

/-- The number of steps Emily takes with wind behind her -/
def steps_with_wind : ℕ := 300

/-- The number of steps Emily takes against the wind -/
def steps_against_wind : ℕ := 75

/-- The number of extra steps the wind allows Emily to take in the direction it blows -/
def wind_effect : ℕ := 20

theorem ship_length_proof :
  ∀ (E S : ℝ),
  E > 0 ∧ S > 0 →
  (steps_with_wind + wind_effect : ℝ) * E = ship_length + (steps_with_wind + wind_effect) * S →
  (steps_against_wind - wind_effect : ℝ) * E = ship_length - (steps_against_wind - wind_effect) * S →
  ship_length = 120 := by
  sorry

end NUMINAMATH_CALUDE_ship_length_proof_l3812_381269


namespace NUMINAMATH_CALUDE_water_ratio_is_two_to_one_l3812_381287

/-- Represents the water usage scenario of a water tower and four neighborhoods --/
structure WaterUsage where
  total : ℕ
  first : ℕ
  fourth : ℕ
  third_excess : ℕ

/-- Calculates the ratio of water used by the second neighborhood to the first neighborhood --/
def water_ratio (w : WaterUsage) : ℚ :=
  let second := (w.total - w.first - w.fourth - (w.total - w.first - w.fourth - w.third_excess)) / 2
  second / w.first

/-- Theorem stating that given the specific conditions, the water ratio is 2:1 --/
theorem water_ratio_is_two_to_one (w : WaterUsage) 
  (h1 : w.total = 1200)
  (h2 : w.first = 150)
  (h3 : w.fourth = 350)
  (h4 : w.third_excess = 100) :
  water_ratio w = 2 := by
  sorry

#eval water_ratio { total := 1200, first := 150, fourth := 350, third_excess := 100 }

end NUMINAMATH_CALUDE_water_ratio_is_two_to_one_l3812_381287


namespace NUMINAMATH_CALUDE_cut_cube_edge_count_l3812_381227

/-- Represents a cube with smaller cubes removed from its corners -/
structure CutCube where
  side_length : ℝ
  cut_length : ℝ

/-- Calculates the number of edges in a CutCube -/
def edge_count (c : CutCube) : ℕ :=
  12 + 8 * 9 / 3

/-- Theorem stating that a cube of side length 4 with corners of side length 1.5 removed has 36 edges -/
theorem cut_cube_edge_count :
  let c : CutCube := { side_length := 4, cut_length := 1.5 }
  edge_count c = 36 := by
  sorry

end NUMINAMATH_CALUDE_cut_cube_edge_count_l3812_381227
