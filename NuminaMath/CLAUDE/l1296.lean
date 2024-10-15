import Mathlib

namespace NUMINAMATH_CALUDE_line_contains_point_l1296_129650

/-- The value of k for which the line 2 - 2kx = -4y contains the point (3, -2) -/
theorem line_contains_point (k : ℝ) : 
  (2 - 2 * k * 3 = -4 * (-2)) ↔ k = -1 := by sorry

end NUMINAMATH_CALUDE_line_contains_point_l1296_129650


namespace NUMINAMATH_CALUDE_student_selection_theorem_l1296_129629

def number_of_boys : ℕ := 4
def number_of_girls : ℕ := 3
def total_to_select : ℕ := 3

theorem student_selection_theorem :
  (Nat.choose number_of_boys 2 * Nat.choose number_of_girls 1) +
  (Nat.choose number_of_boys 1 * Nat.choose number_of_girls 2) = 30 := by
  sorry

end NUMINAMATH_CALUDE_student_selection_theorem_l1296_129629


namespace NUMINAMATH_CALUDE_total_spent_proof_l1296_129675

-- Define the original prices and discount rates
def tshirt_price : ℚ := 20
def tshirt_discount : ℚ := 0.4
def hat_price : ℚ := 15
def hat_discount : ℚ := 0.6
def accessory_price : ℚ := 10
def bracelet_discount : ℚ := 0.3
def belt_discount : ℚ := 0.5
def sales_tax : ℚ := 0.05

-- Define the number of friends and their purchases
def total_friends : ℕ := 4
def bracelet_buyers : ℕ := 1
def belt_buyers : ℕ := 3

-- Define the function to calculate discounted price
def discounted_price (original_price : ℚ) (discount : ℚ) : ℚ :=
  original_price * (1 - discount)

-- Define the theorem
theorem total_spent_proof :
  let tshirt_discounted := discounted_price tshirt_price tshirt_discount
  let hat_discounted := discounted_price hat_price hat_discount
  let bracelet_discounted := discounted_price accessory_price bracelet_discount
  let belt_discounted := discounted_price accessory_price belt_discount
  let bracelet_total := tshirt_discounted + hat_discounted + bracelet_discounted
  let belt_total := tshirt_discounted + hat_discounted + belt_discounted
  let subtotal := bracelet_total * bracelet_buyers + belt_total * belt_buyers
  let total := subtotal * (1 + sales_tax)
  total = 98.7 := by
    sorry

end NUMINAMATH_CALUDE_total_spent_proof_l1296_129675


namespace NUMINAMATH_CALUDE_wire_length_for_square_field_l1296_129638

-- Define the area of the square field
def field_area : ℝ := 24336

-- Define the number of times the wire goes around the field
def num_rounds : ℕ := 13

-- Theorem statement
theorem wire_length_for_square_field :
  let side_length := Real.sqrt field_area
  let perimeter := 4 * side_length
  let wire_length := num_rounds * perimeter
  wire_length = 8112 := by sorry

end NUMINAMATH_CALUDE_wire_length_for_square_field_l1296_129638


namespace NUMINAMATH_CALUDE_usual_time_calculation_l1296_129652

/-- Proves that given a constant distance and the fact that at 60% of usual speed 
    it takes 35 minutes more, the usual time to cover the distance is 52.5 minutes. -/
theorem usual_time_calculation (distance : ℝ) (usual_speed : ℝ) (usual_time : ℝ) 
    (h1 : usual_speed > 0) 
    (h2 : usual_time > 0)
    (h3 : distance = usual_speed * usual_time)
    (h4 : distance = (0.6 * usual_speed) * (usual_time + 35/60)) :
  usual_time = 52.5 / 60 := by
sorry

end NUMINAMATH_CALUDE_usual_time_calculation_l1296_129652


namespace NUMINAMATH_CALUDE_complement_A_intersect_B_l1296_129697

-- Define set A
def A : Set ℝ := {x | x^2 - 2*x > 0}

-- Define set B
def B : Set ℝ := {y | ∃ x, y = Real.sin x}

-- State the theorem
theorem complement_A_intersect_B :
  (Set.univ \ A) ∩ B = Set.Icc 0 1 := by sorry

end NUMINAMATH_CALUDE_complement_A_intersect_B_l1296_129697


namespace NUMINAMATH_CALUDE_least_prime_factor_of_11_5_minus_11_4_l1296_129695

theorem least_prime_factor_of_11_5_minus_11_4 :
  Nat.minFac (11^5 - 11^4) = 2 := by sorry

end NUMINAMATH_CALUDE_least_prime_factor_of_11_5_minus_11_4_l1296_129695


namespace NUMINAMATH_CALUDE_candy_left_l1296_129642

theorem candy_left (houses : ℕ) (candies_per_house : ℕ) (people : ℕ) (candies_eaten_per_person : ℕ) : 
  houses = 15 → 
  candies_per_house = 8 → 
  people = 3 → 
  candies_eaten_per_person = 6 → 
  houses * candies_per_house - people * candies_eaten_per_person = 102 := by
sorry

end NUMINAMATH_CALUDE_candy_left_l1296_129642


namespace NUMINAMATH_CALUDE_product_profit_l1296_129619

theorem product_profit (original_price : ℝ) (cost_price : ℝ) : 
  cost_price > 0 →
  original_price > 0 →
  (0.8 * original_price = 1.2 * cost_price) →
  (original_price - cost_price) / cost_price = 0.5 := by
sorry

end NUMINAMATH_CALUDE_product_profit_l1296_129619


namespace NUMINAMATH_CALUDE_system_solution_l1296_129683

theorem system_solution (a b c : ℝ) :
  ∃ x y z : ℝ,
  (a * x^3 + b * y = c * z^5 ∧
   a * z^3 + b * x = c * y^5 ∧
   a * y^3 + b * z = c * x^5) →
  ((x = 0 ∧ y = 0 ∧ z = 0) ∨
   ∃ s t : ℝ, s^2 = (a + t * Real.sqrt (a^2 + 4*b*c)) / (2*c) ∧
             (x = s ∧ y = s ∧ z = s) ∧
             (t = 1 ∨ t = -1)) :=
by sorry

end NUMINAMATH_CALUDE_system_solution_l1296_129683


namespace NUMINAMATH_CALUDE_min_sum_squares_l1296_129660

def S : Finset Int := {-8, -6, -4, -1, 1, 3, 7, 12}

theorem min_sum_squares (p q r s t u v w : Int) 
  (h_distinct : p ≠ q ∧ p ≠ r ∧ p ≠ s ∧ p ≠ t ∧ p ≠ u ∧ p ≠ v ∧ p ≠ w ∧
                q ≠ r ∧ q ≠ s ∧ q ≠ t ∧ q ≠ u ∧ q ≠ v ∧ q ≠ w ∧
                r ≠ s ∧ r ≠ t ∧ r ≠ u ∧ r ≠ v ∧ r ≠ w ∧
                s ≠ t ∧ s ≠ u ∧ s ≠ v ∧ s ≠ w ∧
                t ≠ u ∧ t ≠ v ∧ t ≠ w ∧
                u ≠ v ∧ u ≠ w ∧
                v ≠ w)
  (h_in_S : p ∈ S ∧ q ∈ S ∧ r ∈ S ∧ s ∈ S ∧ t ∈ S ∧ u ∈ S ∧ v ∈ S ∧ w ∈ S) :
  (∀ a b c d e f g h : Int, 
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧ a ≠ g ∧ a ≠ h ∧
    b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧ b ≠ g ∧ b ≠ h ∧
    c ≠ d ∧ c ≠ e ∧ c ≠ f ∧ c ≠ g ∧ c ≠ h ∧
    d ≠ e ∧ d ≠ f ∧ d ≠ g ∧ d ≠ h ∧
    e ≠ f ∧ e ≠ g ∧ e ≠ h ∧
    f ≠ g ∧ f ≠ h ∧
    g ≠ h →
    a ∈ S ∧ b ∈ S ∧ c ∈ S ∧ d ∈ S ∧ e ∈ S ∧ f ∈ S ∧ g ∈ S ∧ h ∈ S →
    (a + b + c + d)^2 + (e + f + g + h)^2 ≥ 10) ∧
  (p + q + r + s)^2 + (t + u + v + w)^2 = 10 :=
sorry

end NUMINAMATH_CALUDE_min_sum_squares_l1296_129660


namespace NUMINAMATH_CALUDE_initial_number_of_persons_l1296_129655

theorem initial_number_of_persons (n : ℕ) 
  (h1 : (3 : ℝ) * n = 24) : n = 8 := by
  sorry

#check initial_number_of_persons

end NUMINAMATH_CALUDE_initial_number_of_persons_l1296_129655


namespace NUMINAMATH_CALUDE_instantaneous_velocity_at_3_seconds_l1296_129685

-- Define the motion equation
def s (t : ℝ) : ℝ := 1 - t + t^2

-- Define the instantaneous velocity function
def v (t : ℝ) : ℝ := -1 + 2*t

-- Theorem statement
theorem instantaneous_velocity_at_3_seconds :
  v 3 = 5 := by
  sorry

end NUMINAMATH_CALUDE_instantaneous_velocity_at_3_seconds_l1296_129685


namespace NUMINAMATH_CALUDE_marks_future_age_l1296_129658

def amy_age : ℕ := 15
def age_difference : ℕ := 7
def years_in_future : ℕ := 5

theorem marks_future_age :
  amy_age + age_difference + years_in_future = 27 := by
  sorry

end NUMINAMATH_CALUDE_marks_future_age_l1296_129658


namespace NUMINAMATH_CALUDE_f_min_max_l1296_129631

-- Define the function
def f (x y : ℝ) : ℝ := 3 * x^2 * y - 2 * x * y^2

-- State the theorem
theorem f_min_max :
  (∀ x y : ℝ, 0 ≤ x ∧ x ≤ 1 ∧ 0 ≤ y ∧ y ≤ 1 → f x y ≥ -1/3) ∧
  (∀ x y : ℝ, 0 ≤ x ∧ x ≤ 1 ∧ 0 ≤ y ∧ y ≤ 1 → f x y ≤ 9/8) ∧
  (∃ x y : ℝ, 0 ≤ x ∧ x ≤ 1 ∧ 0 ≤ y ∧ y ≤ 1 ∧ f x y = -1/3) ∧
  (∃ x y : ℝ, 0 ≤ x ∧ x ≤ 1 ∧ 0 ≤ y ∧ y ≤ 1 ∧ f x y = 9/8) :=
by sorry

end NUMINAMATH_CALUDE_f_min_max_l1296_129631


namespace NUMINAMATH_CALUDE_ferry_route_ratio_l1296_129648

-- Define the parameters
def ferry_p_speed : ℝ := 6
def ferry_p_time : ℝ := 3
def ferry_q_speed_difference : ℝ := 3
def ferry_q_time_difference : ℝ := 3

-- Define the theorem
theorem ferry_route_ratio :
  let ferry_p_distance := ferry_p_speed * ferry_p_time
  let ferry_q_speed := ferry_p_speed + ferry_q_speed_difference
  let ferry_q_time := ferry_p_time + ferry_q_time_difference
  let ferry_q_distance := ferry_q_speed * ferry_q_time
  ferry_q_distance / ferry_p_distance = 3 := by
  sorry


end NUMINAMATH_CALUDE_ferry_route_ratio_l1296_129648


namespace NUMINAMATH_CALUDE_alicia_score_l1296_129635

theorem alicia_score (total_score : ℕ) (other_players : ℕ) (avg_score : ℕ) (alicia_score : ℕ) : 
  total_score = 75 →
  other_players = 8 →
  avg_score = 6 →
  total_score = other_players * avg_score + alicia_score →
  alicia_score = 27 := by
sorry

end NUMINAMATH_CALUDE_alicia_score_l1296_129635


namespace NUMINAMATH_CALUDE_field_length_width_ratio_l1296_129693

/-- Proves that the ratio of a rectangular field's length to its width is 2:1,
    given specific conditions about the field and a pond within it. -/
theorem field_length_width_ratio :
  ∀ (field_length field_width pond_side : ℝ),
  field_length = 48 →
  pond_side = 8 →
  pond_side * pond_side = (field_length * field_width) / 18 →
  field_length / field_width = 2 := by
  sorry

end NUMINAMATH_CALUDE_field_length_width_ratio_l1296_129693


namespace NUMINAMATH_CALUDE_eight_queens_exists_l1296_129625

/-- Represents a position on the chessboard -/
structure Position :=
  (row : Fin 8)
  (col : Fin 8)

/-- Checks if two positions are on the same diagonal -/
def sameDiagonal (p1 p2 : Position) : Prop :=
  (p1.row.val : Int) - (p1.col.val : Int) = (p2.row.val : Int) - (p2.col.val : Int) ∨
  (p1.row.val : Int) + (p1.col.val : Int) = (p2.row.val : Int) + (p2.col.val : Int)

/-- Checks if two queens threaten each other -/
def threaten (p1 p2 : Position) : Prop :=
  p1.row = p2.row ∨ p1.col = p2.col ∨ sameDiagonal p1 p2

/-- Represents an arrangement of eight queens on the chessboard -/
def QueenArrangement := Fin 8 → Position

/-- Checks if a queen arrangement is valid (no queens threaten each other) -/
def validArrangement (arrangement : QueenArrangement) : Prop :=
  ∀ i j : Fin 8, i ≠ j → ¬threaten (arrangement i) (arrangement j)

/-- Theorem: There exists a valid arrangement of eight queens on an 8x8 chessboard -/
theorem eight_queens_exists : ∃ arrangement : QueenArrangement, validArrangement arrangement :=
sorry

end NUMINAMATH_CALUDE_eight_queens_exists_l1296_129625


namespace NUMINAMATH_CALUDE_parallel_lines_angle_problem_l1296_129611

-- Define the angles as real numbers
variable (AXE CYX BXY : ℝ)

-- State the theorem
theorem parallel_lines_angle_problem 
  (h1 : AXE = 4 * CYX - 120) -- Given condition
  (h2 : AXE = CYX) -- From parallel lines property
  : BXY = 40 := by
  sorry

end NUMINAMATH_CALUDE_parallel_lines_angle_problem_l1296_129611


namespace NUMINAMATH_CALUDE_max_ages_for_given_params_l1296_129672

/-- Calculates the maximum number of different integer ages within one standard deviation of the average age. -/
def max_different_ages (average_age : ℤ) (std_dev : ℤ) : ℕ :=
  let lower_bound := average_age - std_dev
  let upper_bound := average_age + std_dev
  (upper_bound - lower_bound + 1).toNat

/-- Theorem stating that for an average age of 10 and standard deviation of 8,
    the maximum number of different integer ages within one standard deviation is 17. -/
theorem max_ages_for_given_params :
  max_different_ages 10 8 = 17 := by
  sorry

#eval max_different_ages 10 8

end NUMINAMATH_CALUDE_max_ages_for_given_params_l1296_129672


namespace NUMINAMATH_CALUDE_max_min_values_l1296_129682

-- Define the constraint function
def constraint (x y : ℝ) : Prop :=
  |5 * x + y| + |5 * x - y| = 20

-- Define the expression to be maximized/minimized
def expr (x y : ℝ) : ℝ :=
  x^2 - x*y + y^2

-- Statement of the theorem
theorem max_min_values :
  (∃ x y : ℝ, constraint x y ∧ expr x y = 124) ∧
  (∃ x y : ℝ, constraint x y ∧ expr x y = 3) ∧
  (∀ x y : ℝ, constraint x y → 3 ≤ expr x y ∧ expr x y ≤ 124) :=
sorry

end NUMINAMATH_CALUDE_max_min_values_l1296_129682


namespace NUMINAMATH_CALUDE_atomic_number_relation_l1296_129649

-- Define the compound Y₂X₃
structure Compound where
  X : ℕ  -- Atomic number of X
  Y : ℕ  -- Atomic number of Y

-- Define the property of X being a short-period non-metal element
def isShortPeriodNonMetal (x : ℕ) : Prop :=
  x ≤ 18  -- Assuming short-period elements have atomic numbers up to 18

-- Define the compound formation rule
def formsCompound (c : Compound) : Prop :=
  isShortPeriodNonMetal c.X ∧ c.Y > 0

-- Theorem statement
theorem atomic_number_relation (n : ℕ) :
  ∀ c : Compound, formsCompound c → c.X = n → c.Y ≠ n + 2 := by
  sorry

end NUMINAMATH_CALUDE_atomic_number_relation_l1296_129649


namespace NUMINAMATH_CALUDE_plant_structure_l1296_129612

/-- Represents the structure of a plant with branches and small branches. -/
structure Plant where
  branches : ℕ
  smallBranchesPerBranch : ℕ

/-- The total count of parts in the plant (main stem + branches + small branches). -/
def Plant.totalCount (p : Plant) : ℕ :=
  1 + p.branches + p.branches * p.smallBranchesPerBranch

/-- The plant satisfies the given conditions. -/
def validPlant (p : Plant) : Prop :=
  p.branches = p.smallBranchesPerBranch ∧ p.totalCount = 43

theorem plant_structure : ∃ (p : Plant), validPlant p ∧ p.smallBranchesPerBranch = 6 := by
  sorry

end NUMINAMATH_CALUDE_plant_structure_l1296_129612


namespace NUMINAMATH_CALUDE_flag_stripes_l1296_129679

theorem flag_stripes :
  ∀ (S : ℕ), 
    S > 0 →
    (10 * (1 + (S - 1) / 2 : ℚ) = 70) →
    S = 13 := by
  sorry

end NUMINAMATH_CALUDE_flag_stripes_l1296_129679


namespace NUMINAMATH_CALUDE_garden_area_is_2400_l1296_129622

/-- Represents a rectangular garden with given properties -/
structure Garden where
  length : ℝ
  width : ℝ
  length_walk : ℕ
  perimeter_walk : ℕ
  total_distance : ℝ
  len_condition : length * length_walk = total_distance
  peri_condition : (2 * length + 2 * width) * perimeter_walk = total_distance

/-- Calculates the area of a rectangular garden -/
def garden_area (g : Garden) : ℝ :=
  g.length * g.width

/-- Theorem stating that a garden with the given properties has an area of 2400 square meters -/
theorem garden_area_is_2400 (g : Garden) 
  (h1 : g.length_walk = 50)
  (h2 : g.perimeter_walk = 15)
  (h3 : g.total_distance = 3000) : 
  garden_area g = 2400 := by
  sorry

end NUMINAMATH_CALUDE_garden_area_is_2400_l1296_129622


namespace NUMINAMATH_CALUDE_no_non_zero_solution_l1296_129666

theorem no_non_zero_solution (a b c n : ℤ) :
  6 * (6 * a^2 + 3 * b^2 + c^2) = 5 * n^2 → a = 0 ∧ b = 0 ∧ c = 0 ∧ n = 0 := by
  sorry

end NUMINAMATH_CALUDE_no_non_zero_solution_l1296_129666


namespace NUMINAMATH_CALUDE_inscribed_rectangle_area_l1296_129609

/-- A right triangle with an inscribed rectangle -/
structure InscribedRectangle where
  /-- Length of AB in the right triangle AGD -/
  ab : ℝ
  /-- Length of CD in the right triangle AGD -/
  cd : ℝ
  /-- Length of BC in the inscribed rectangle BCFE -/
  bc : ℝ
  /-- Length of FE in the inscribed rectangle BCFE -/
  fe : ℝ
  /-- BC is parallel to AD -/
  bc_parallel_ad : True
  /-- FE is parallel to AD -/
  fe_parallel_ad : True
  /-- Length of BC is one-third of FE -/
  bc_one_third_fe : bc = fe / 3
  /-- AB = 40 units -/
  ab_eq_40 : ab = 40
  /-- CD = 70 units -/
  cd_eq_70 : cd = 70

/-- The area of the inscribed rectangle BCFE is 2800 square units -/
theorem inscribed_rectangle_area (rect : InscribedRectangle) : 
  rect.bc * rect.fe = 2800 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_rectangle_area_l1296_129609


namespace NUMINAMATH_CALUDE_cubic_equation_root_problem_l1296_129600

theorem cubic_equation_root_problem (c d : ℚ) : 
  (∃ x : ℝ, x^3 + c*x^2 + d*x + 15 = 0 ∧ x = 3 + Real.sqrt 5) → d = -37/2 :=
by sorry

end NUMINAMATH_CALUDE_cubic_equation_root_problem_l1296_129600


namespace NUMINAMATH_CALUDE_ruby_candy_sharing_l1296_129601

theorem ruby_candy_sharing (total_candies : ℕ) (candies_per_friend : ℕ) 
  (h1 : total_candies = 36)
  (h2 : candies_per_friend = 4) :
  total_candies / candies_per_friend = 9 := by
  sorry

end NUMINAMATH_CALUDE_ruby_candy_sharing_l1296_129601


namespace NUMINAMATH_CALUDE_smallest_n_divisible_by_two_primes_l1296_129662

/-- A function that returns true if a number is divisible by at least two different primes -/
def divisible_by_two_primes (x : ℕ) : Prop :=
  ∃ p q : ℕ, Nat.Prime p ∧ Nat.Prime q ∧ p ≠ q ∧ p ∣ x ∧ q ∣ x

/-- The theorem stating that 5 is the smallest positive integer n ≥ 5 such that n^2 - n + 6 is divisible by at least two different primes -/
theorem smallest_n_divisible_by_two_primes :
  ∀ n : ℕ, n ≥ 5 → (divisible_by_two_primes (n^2 - n + 6) → n ≥ 5) ∧
  (n = 5 → divisible_by_two_primes (5^2 - 5 + 6)) :=
by sorry

#check smallest_n_divisible_by_two_primes

end NUMINAMATH_CALUDE_smallest_n_divisible_by_two_primes_l1296_129662


namespace NUMINAMATH_CALUDE_system_solution_sum_l1296_129627

theorem system_solution_sum (a b : ℝ) : 
  (1 : ℝ) * a + 2 = -1 ∧ 2 * (1 : ℝ) - b * 2 = 0 → a + b = -2 := by
  sorry

end NUMINAMATH_CALUDE_system_solution_sum_l1296_129627


namespace NUMINAMATH_CALUDE_f_inequality_l1296_129687

-- Define the function f
variable (f : ℝ → ℝ)

-- State the theorem
theorem f_inequality (h1 : f 1 = 1) (h2 : ∀ x, deriv f x < 2) :
  ∀ x, f x < 2 * x - 1 ↔ x > 1 := by
  sorry

end NUMINAMATH_CALUDE_f_inequality_l1296_129687


namespace NUMINAMATH_CALUDE_inequality_proof_l1296_129610

theorem inequality_proof (x y z : ℝ) : 
  x^2 / (x^2 + 2*y*z) + y^2 / (y^2 + 2*z*x) + z^2 / (z^2 + 2*x*y) ≥ 1 := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l1296_129610


namespace NUMINAMATH_CALUDE_symmetric_function_k_range_l1296_129626

/-- A function f is symmetric if it's monotonic on its domain D and there exists an interval [a,b] ⊆ D such that the range of f on [a,b] is [-b,-a] -/
def IsSymmetric (f : ℝ → ℝ) (D : Set ℝ) : Prop :=
  Monotone f ∧ ∃ a b, a < b ∧ Set.Icc a b ⊆ D ∧ Set.image f (Set.Icc a b) = Set.Icc (-b) (-a)

/-- The main theorem stating that if f(x) = √(2 - x) - k is symmetric on (-∞, 2], then k ∈ [2, 9/4) -/
theorem symmetric_function_k_range :
  ∀ k : ℝ, IsSymmetric (fun x ↦ Real.sqrt (2 - x) - k) (Set.Iic 2) →
  k ∈ Set.Icc 2 (9/4) := by sorry

end NUMINAMATH_CALUDE_symmetric_function_k_range_l1296_129626


namespace NUMINAMATH_CALUDE_pure_imaginary_complex_number_l1296_129691

theorem pure_imaginary_complex_number (x : ℝ) :
  let z : ℂ := Complex.mk (x^2 - 3*x + 2) (x - 1)
  (z.re = 0 ∧ z.im ≠ 0) → x = 2 := by
  sorry

end NUMINAMATH_CALUDE_pure_imaginary_complex_number_l1296_129691


namespace NUMINAMATH_CALUDE_min_k_value_l1296_129645

theorem min_k_value (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∀ k : ℝ, 1 / a + 1 / b + k / (a + b) ≥ 0) → 
  ∀ k : ℝ, k ≥ -4 ∧ ∃ k₀ : ℝ, k₀ = -4 ∧ 1 / a + 1 / b + k₀ / (a + b) ≥ 0 :=
by sorry

end NUMINAMATH_CALUDE_min_k_value_l1296_129645


namespace NUMINAMATH_CALUDE_remainder_theorem_l1296_129647

theorem remainder_theorem (n : ℕ) (h : n % 7 = 5) : (3 * n + 2)^2 % 11 = 3 := by
  sorry

end NUMINAMATH_CALUDE_remainder_theorem_l1296_129647


namespace NUMINAMATH_CALUDE_sandy_change_proof_l1296_129632

/-- Calculates the change received from a purchase given the payment amount and the costs of individual items. -/
def calculate_change (payment : ℚ) (item1_cost : ℚ) (item2_cost : ℚ) : ℚ :=
  payment - (item1_cost + item2_cost)

/-- Proves that given a $20 bill payment and purchases of $9.24 and $8.25, the change received is $2.51. -/
theorem sandy_change_proof :
  calculate_change 20 9.24 8.25 = 2.51 := by
  sorry

end NUMINAMATH_CALUDE_sandy_change_proof_l1296_129632


namespace NUMINAMATH_CALUDE_log_half_increasing_interval_l1296_129617

noncomputable def y (x a : ℝ) : ℝ := Real.log (x^2 - 2*a*x + 3) / Real.log (1/2)

def is_increasing (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a < x ∧ x < y ∧ y < b → f x < f y

theorem log_half_increasing_interval (a : ℝ) (h : a > 0) :
  (∃ m, m > 0 ∧ is_increasing (y · a) 0 m) ↔
  ((0 < a ∧ a ≤ Real.sqrt 3 ∧ ∃ m, 0 < m ∧ m ≤ a) ∨
   (a > Real.sqrt 3 ∧ ∃ m, 0 < m ∧ m ≤ a - Real.sqrt (a^2 - 3))) :=
sorry

end NUMINAMATH_CALUDE_log_half_increasing_interval_l1296_129617


namespace NUMINAMATH_CALUDE_matrices_are_inverses_l1296_129681

theorem matrices_are_inverses : 
  let A : Matrix (Fin 2) (Fin 2) ℝ := !![4, -7; -5, 9]
  let B : Matrix (Fin 2) (Fin 2) ℝ := !![9, 7; 5, 4]
  A * B = 1 ∧ B * A = 1 := by
  sorry

end NUMINAMATH_CALUDE_matrices_are_inverses_l1296_129681


namespace NUMINAMATH_CALUDE_modified_cube_surface_area_l1296_129654

/-- Represents a cube with its dimensions -/
structure Cube where
  size : Nat

/-- Represents the large cube and its properties -/
structure LargeCube where
  size : Nat
  smallCubeSize : Nat
  totalSmallCubes : Nat

/-- Calculates the surface area of the modified structure -/
def calculateSurfaceArea (lc : LargeCube) : Nat :=
  sorry

/-- Theorem stating the surface area of the modified structure -/
theorem modified_cube_surface_area 
  (lc : LargeCube) 
  (h1 : lc.size = 12) 
  (h2 : lc.smallCubeSize = 3) 
  (h3 : lc.totalSmallCubes = 64) : 
  calculateSurfaceArea lc = 2454 := by
  sorry

end NUMINAMATH_CALUDE_modified_cube_surface_area_l1296_129654


namespace NUMINAMATH_CALUDE_root_sum_sixth_power_l1296_129618

theorem root_sum_sixth_power (r s : ℝ) : 
  r^2 - 2*r*Real.sqrt 3 + 1 = 0 →
  s^2 - 2*s*Real.sqrt 3 + 1 = 0 →
  r ≠ s →
  r^6 + s^6 = 970 := by
sorry

end NUMINAMATH_CALUDE_root_sum_sixth_power_l1296_129618


namespace NUMINAMATH_CALUDE_smallest_possible_c_l1296_129616

theorem smallest_possible_c (a b c : ℝ) : 
  1 < a → a < b → b < c →
  1 + a ≤ b →
  1 / a + 1 / b ≤ 1 / c →
  c ≥ (3 + Real.sqrt 5) / 2 :=
sorry

end NUMINAMATH_CALUDE_smallest_possible_c_l1296_129616


namespace NUMINAMATH_CALUDE_directed_segment_length_equal_l1296_129684

-- Define a vector space
variable {V : Type*} [NormedAddCommGroup V]

-- Define two points in the vector space
variable (M N : V)

-- Define the directed line segment from M to N
def directed_segment (M N : V) : V := N - M

-- Theorem statement
theorem directed_segment_length_equal :
  ‖directed_segment M N‖ = ‖directed_segment N M‖ := by sorry

end NUMINAMATH_CALUDE_directed_segment_length_equal_l1296_129684


namespace NUMINAMATH_CALUDE_smallest_four_digit_negative_congruent_to_one_mod_37_l1296_129646

theorem smallest_four_digit_negative_congruent_to_one_mod_37 :
  ∀ x : ℤ, x < 0 ∧ x ≥ -9999 ∧ x ≡ 1 [ZMOD 37] → x ≥ -1034 :=
by sorry

end NUMINAMATH_CALUDE_smallest_four_digit_negative_congruent_to_one_mod_37_l1296_129646


namespace NUMINAMATH_CALUDE_answer_key_combinations_l1296_129668

/-- The number of ways to answer a single true-false question -/
def true_false_options : ℕ := 2

/-- The number of true-false questions in the quiz -/
def num_true_false : ℕ := 4

/-- The number of ways to answer a single multiple-choice question -/
def multiple_choice_options : ℕ := 4

/-- The number of multiple-choice questions in the quiz -/
def num_multiple_choice : ℕ := 2

/-- The total number of possible answer combinations for true-false questions -/
def total_true_false_combinations : ℕ := true_false_options ^ num_true_false

/-- The number of invalid true-false combinations (all true or all false) -/
def invalid_true_false_combinations : ℕ := 2

/-- The number of valid true-false combinations -/
def valid_true_false_combinations : ℕ := total_true_false_combinations - invalid_true_false_combinations

/-- The number of ways to answer all multiple-choice questions -/
def multiple_choice_combinations : ℕ := multiple_choice_options ^ num_multiple_choice

/-- The total number of ways to create an answer key for the quiz -/
def total_answer_key_combinations : ℕ := valid_true_false_combinations * multiple_choice_combinations

theorem answer_key_combinations : total_answer_key_combinations = 224 := by
  sorry

end NUMINAMATH_CALUDE_answer_key_combinations_l1296_129668


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_l1296_129639

theorem simplify_and_evaluate (m : ℝ) (h : m = Real.sqrt 16 + Real.tan (45 * π / 180)) :
  (m + 2 + 5 / (2 - m)) * ((2 * m - 4) / (3 - m)) = -16 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_l1296_129639


namespace NUMINAMATH_CALUDE_pencil_distribution_l1296_129656

/-- The number of ways to distribute n identical objects among k people, 
    where each person gets at least one object. -/
def distribute (n k : ℕ) : ℕ := sorry

/-- The number of ways to distribute 8 identical pencils among 4 friends, 
    where each friend has at least one pencil. -/
theorem pencil_distribution : distribute 8 4 = 35 := by sorry

end NUMINAMATH_CALUDE_pencil_distribution_l1296_129656


namespace NUMINAMATH_CALUDE_system_has_three_solutions_l1296_129614

/-- The floor function -/
noncomputable def floor (x : ℝ) : ℤ :=
  Int.floor x

/-- The system of equations -/
def system (x : ℝ) : Prop :=
  3 * x^2 - 45 * (floor x) + 60 = 0 ∧ 2 * x - 3 * (floor x) + 1 = 0

/-- The theorem stating that the system has exactly 3 real solutions -/
theorem system_has_three_solutions :
  ∃ (s : Finset ℝ), s.card = 3 ∧ ∀ x, x ∈ s ↔ system x :=
sorry

end NUMINAMATH_CALUDE_system_has_three_solutions_l1296_129614


namespace NUMINAMATH_CALUDE_cash_realized_specific_case_l1296_129603

/-- Given a total amount including brokerage and a brokerage rate,
    calculates the cash realized without brokerage -/
def cash_realized (total : ℚ) (brokerage_rate : ℚ) : ℚ :=
  total / (1 + brokerage_rate)

/-- Theorem stating that given the specific conditions of the problem,
    the cash realized is equal to 43200/401 -/
theorem cash_realized_specific_case :
  cash_realized 108 (1/400) = 43200/401 := by
  sorry

end NUMINAMATH_CALUDE_cash_realized_specific_case_l1296_129603


namespace NUMINAMATH_CALUDE_function_value_at_3000_l1296_129661

/-- Given a function f: ℕ → ℕ satisfying the following properties:
  1) f(0) = 1
  2) For all x, f(x + 3) = f(x) + 2x + 3
  Prove that f(3000) = 3000001 -/
theorem function_value_at_3000 (f : ℕ → ℕ) 
  (h1 : f 0 = 1) 
  (h2 : ∀ x, f (x + 3) = f x + 2 * x + 3) : 
  f 3000 = 3000001 := by
  sorry

end NUMINAMATH_CALUDE_function_value_at_3000_l1296_129661


namespace NUMINAMATH_CALUDE_collinear_points_theorem_l1296_129606

/-- Three points are collinear if the slope between any two pairs of points is the same. -/
def collinear (x₁ y₁ x₂ y₂ x₃ y₃ : ℝ) : Prop :=
  (y₂ - y₁) * (x₃ - x₂) = (y₃ - y₂) * (x₂ - x₁)

/-- The theorem states that if points A(a, 1), B(9, 0), and C(-3, 4) are collinear, then a = 6. -/
theorem collinear_points_theorem (a : ℝ) :
  collinear a 1 9 0 (-3) 4 → a = 6 := by
  sorry

#check collinear_points_theorem

end NUMINAMATH_CALUDE_collinear_points_theorem_l1296_129606


namespace NUMINAMATH_CALUDE_train_crossing_time_l1296_129694

/-- Proves that a train with given length and speed takes the calculated time to cross an electric pole -/
theorem train_crossing_time (train_length : Real) (train_speed_kmh : Real) (crossing_time : Real) :
  train_length = 2500 ∧ 
  train_speed_kmh = 90 →
  crossing_time = 100 := by
  sorry

#check train_crossing_time

end NUMINAMATH_CALUDE_train_crossing_time_l1296_129694


namespace NUMINAMATH_CALUDE_rogers_reading_rate_l1296_129651

/-- Roger's book reading problem -/
theorem rogers_reading_rate (total_books : ℕ) (weeks : ℕ) (books_per_week : ℕ) 
  (h1 : total_books = 30)
  (h2 : weeks = 5)
  (h3 : books_per_week * weeks = total_books) :
  books_per_week = 6 := by
sorry

end NUMINAMATH_CALUDE_rogers_reading_rate_l1296_129651


namespace NUMINAMATH_CALUDE_cube_root_equivalence_l1296_129615

theorem cube_root_equivalence (x : ℝ) (hx : x > 0) : 
  (x^2 * x^(1/4))^(1/3) = x^(3/4) := by sorry

end NUMINAMATH_CALUDE_cube_root_equivalence_l1296_129615


namespace NUMINAMATH_CALUDE_largest_circular_pool_diameter_l1296_129667

/-- Given a rectangular garden with area 180 square meters and length three times its width,
    the diameter of the largest circular pool that can be outlined by the garden's perimeter
    is 16√15/π meters. -/
theorem largest_circular_pool_diameter (width : ℝ) (length : ℝ) : 
  width > 0 →
  length = 3 * width →
  width * length = 180 →
  (2 * (width + length)) / π = 16 * Real.sqrt 15 / π :=
by sorry

end NUMINAMATH_CALUDE_largest_circular_pool_diameter_l1296_129667


namespace NUMINAMATH_CALUDE_coefficient_of_linear_term_l1296_129690

theorem coefficient_of_linear_term (a b c : ℝ) : 
  (fun x : ℝ => a * x^2 + b * x + c) = (fun x : ℝ => x^2 - 2*x + 3) → 
  b = -2 := by
sorry

end NUMINAMATH_CALUDE_coefficient_of_linear_term_l1296_129690


namespace NUMINAMATH_CALUDE_tan_pi_minus_alpha_l1296_129696

theorem tan_pi_minus_alpha (α : Real) (h : 3 * Real.sin (α - Real.pi) = Real.cos α) :
  Real.tan (Real.pi - α) = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_tan_pi_minus_alpha_l1296_129696


namespace NUMINAMATH_CALUDE_vector_sum_parallel_l1296_129605

theorem vector_sum_parallel (y : ℝ) : 
  let a : Fin 2 → ℝ := ![1, 2]
  let b : Fin 2 → ℝ := ![2, y]
  (∃ (k : ℝ), a = k • b) →
  (a + 2 • b) = ![5, 10] := by
sorry

end NUMINAMATH_CALUDE_vector_sum_parallel_l1296_129605


namespace NUMINAMATH_CALUDE_floor_difference_equals_ten_l1296_129607

theorem floor_difference_equals_ten : 
  ⌊(2010^4 / (2008 * 2009 : ℝ)) - (2008^4 / (2009 * 2010 : ℝ))⌋ = 10 := by sorry

end NUMINAMATH_CALUDE_floor_difference_equals_ten_l1296_129607


namespace NUMINAMATH_CALUDE_car_part_payment_l1296_129678

theorem car_part_payment (remaining_payment : ℝ) (part_payment_percentage : ℝ) 
  (h1 : remaining_payment = 5700)
  (h2 : part_payment_percentage = 0.05) : 
  (remaining_payment / (1 - part_payment_percentage)) * part_payment_percentage = 300 := by
  sorry

end NUMINAMATH_CALUDE_car_part_payment_l1296_129678


namespace NUMINAMATH_CALUDE_bag_cost_theorem_l1296_129665

def total_money : ℕ := 50
def tshirt_cost : ℕ := 8
def keychain_cost : ℚ := 2 / 3
def tshirts_bought : ℕ := 2
def bags_bought : ℕ := 2
def keychains_bought : ℕ := 21

theorem bag_cost_theorem :
  ∃ (bag_cost : ℚ),
    bag_cost * bags_bought = 
      total_money - 
      (tshirt_cost * tshirts_bought) - 
      (keychain_cost * keychains_bought) ∧
    bag_cost = 10 := by
  sorry

end NUMINAMATH_CALUDE_bag_cost_theorem_l1296_129665


namespace NUMINAMATH_CALUDE_symmetric_points_sum_l1296_129624

/-- Given two points A and B that are symmetric with respect to the x-axis,
    prove that the sum of their y-coordinate and x-coordinate respectively is 5. -/
theorem symmetric_points_sum (a b : ℝ) : 
  (2 : ℝ) = b ∧ a = 3 → a + b = 5 := by sorry

end NUMINAMATH_CALUDE_symmetric_points_sum_l1296_129624


namespace NUMINAMATH_CALUDE_train_true_speed_l1296_129628

/-- The true speed of a train given its length, crossing time, and opposing wind speed -/
theorem train_true_speed (train_length : ℝ) (crossing_time : ℝ) (wind_speed : ℝ) :
  train_length = 200 →
  crossing_time = 20 →
  wind_speed = 5 →
  (train_length / crossing_time) + wind_speed = 15 := by
  sorry


end NUMINAMATH_CALUDE_train_true_speed_l1296_129628


namespace NUMINAMATH_CALUDE_pink_crayons_count_l1296_129636

/-- Represents the number of crayons of each color in a crayon box. -/
structure CrayonBox where
  total : ℕ
  red : ℕ
  blue : ℕ
  green : ℕ
  pink : ℕ

/-- Theorem stating the number of pink crayons in the given crayon box. -/
theorem pink_crayons_count (box : CrayonBox) : box.pink = 6 :=
  by
  have h1 : box.total = 24 := by sorry
  have h2 : box.red = 8 := by sorry
  have h3 : box.blue = 6 := by sorry
  have h4 : box.green = 4 := by sorry
  have h5 : box.green = (2 * box.blue) / 3 := by sorry
  have h6 : box.total = box.red + box.blue + box.green + box.pink := by sorry
  sorry


end NUMINAMATH_CALUDE_pink_crayons_count_l1296_129636


namespace NUMINAMATH_CALUDE_triangle_properties_l1296_129633

noncomputable section

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- Vector in 2D space -/
structure Vector2D where
  x : ℝ
  y : ℝ

/-- Two vectors are parallel -/
def parallel (v w : Vector2D) : Prop :=
  ∃ (k : ℝ), v.x * w.y = k * v.y * w.x

variable (ABC : Triangle)
variable (m n : Vector2D)

/-- The given conditions -/
axiom cond1 : m = ⟨2 * Real.sin ABC.B, -Real.sqrt 3⟩
axiom cond2 : n = ⟨Real.cos (2 * ABC.B), 2 * (Real.cos ABC.B)^2 - 1⟩
axiom cond3 : parallel m n
axiom cond4 : ABC.b = 2

/-- The theorem to be proved -/
theorem triangle_properties :
  ABC.B = Real.pi / 3 ∧
  (∀ (S : ℝ), S = 1/2 * ABC.a * ABC.c * Real.sin ABC.B → S ≤ Real.sqrt 3) :=
sorry

end NUMINAMATH_CALUDE_triangle_properties_l1296_129633


namespace NUMINAMATH_CALUDE_greatest_common_factor_372_72_under_50_l1296_129676

def is_greatest_common_factor (n : ℕ) : Prop :=
  n ∣ 372 ∧ n < 50 ∧ n ∣ 72 ∧
  ∀ m : ℕ, m ∣ 372 → m < 50 → m ∣ 72 → m ≤ n

theorem greatest_common_factor_372_72_under_50 :
  is_greatest_common_factor 12 := by
sorry

end NUMINAMATH_CALUDE_greatest_common_factor_372_72_under_50_l1296_129676


namespace NUMINAMATH_CALUDE_inscribed_sphere_theorem_l1296_129659

/-- A right cone with a sphere inscribed in it -/
structure InscribedSphere where
  cone_base_radius : ℝ
  cone_height : ℝ
  sphere_radius : ℝ

/-- The condition that the sphere is inscribed in the cone -/
def is_inscribed (s : InscribedSphere) : Prop :=
  s.sphere_radius * (s.cone_base_radius^2 + s.cone_height^2).sqrt =
    s.cone_base_radius * (s.cone_height - s.sphere_radius)

/-- The theorem to be proved -/
theorem inscribed_sphere_theorem (b d : ℝ) :
  let s := InscribedSphere.mk 15 20 (b * d.sqrt - b)
  is_inscribed s → b + d = 12 := by
  sorry


end NUMINAMATH_CALUDE_inscribed_sphere_theorem_l1296_129659


namespace NUMINAMATH_CALUDE_nancy_football_games_l1296_129692

theorem nancy_football_games (games_this_month games_last_month games_next_month total_games : ℕ) :
  games_this_month = 9 →
  games_last_month = 8 →
  games_next_month = 7 →
  total_games = 24 →
  games_this_month + games_last_month + games_next_month = total_games :=
by sorry

end NUMINAMATH_CALUDE_nancy_football_games_l1296_129692


namespace NUMINAMATH_CALUDE_units_digit_of_27_times_36_l1296_129657

-- Define a function to get the units digit of a number
def unitsDigit (n : ℕ) : ℕ := n % 10

-- Theorem statement
theorem units_digit_of_27_times_36 : unitsDigit (27 * 36) = 2 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_27_times_36_l1296_129657


namespace NUMINAMATH_CALUDE_integral_polynomial_l1296_129699

theorem integral_polynomial (x : ℝ) :
  deriv (fun x => x^3 - x^2 + 5*x) x = 3*x^2 - 2*x + 5 := by
  sorry

end NUMINAMATH_CALUDE_integral_polynomial_l1296_129699


namespace NUMINAMATH_CALUDE_zero_neither_positive_nor_negative_l1296_129604

theorem zero_neither_positive_nor_negative :
  ¬(0 > 0) ∧ ¬(0 < 0) :=
by
  sorry

#check zero_neither_positive_nor_negative

end NUMINAMATH_CALUDE_zero_neither_positive_nor_negative_l1296_129604


namespace NUMINAMATH_CALUDE_part1_part2_l1296_129608

-- Define complex numbers
def z1 (m : ℝ) : ℂ := m - 2*Complex.I
def z2 (n : ℝ) : ℂ := 1 - n*Complex.I

-- Part 1
theorem part1 : Complex.abs (z1 1 + z2 (-1)) = Real.sqrt 5 := by sorry

-- Part 2
theorem part2 : z1 0 = (z2 1)^2 := by sorry

end NUMINAMATH_CALUDE_part1_part2_l1296_129608


namespace NUMINAMATH_CALUDE_camp_average_age_of_adults_l1296_129621

theorem camp_average_age_of_adults 
  (total_members : ℕ) 
  (overall_average : ℝ) 
  (num_girls num_boys num_adults : ℕ) 
  (avg_age_girls avg_age_boys : ℝ) 
  (h1 : total_members = 40)
  (h2 : overall_average = 17)
  (h3 : num_girls = 20)
  (h4 : num_boys = 15)
  (h5 : num_adults = 5)
  (h6 : avg_age_girls = 15)
  (h7 : avg_age_boys = 16)
  (h8 : total_members = num_girls + num_boys + num_adults) :
  (total_members : ℝ) * overall_average - 
  (num_girls : ℝ) * avg_age_girls - 
  (num_boys : ℝ) * avg_age_boys = 
  (num_adults : ℝ) * 28 :=
by sorry

end NUMINAMATH_CALUDE_camp_average_age_of_adults_l1296_129621


namespace NUMINAMATH_CALUDE_pizza_division_l1296_129653

theorem pizza_division (total_pizza : ℚ) (num_employees : ℕ) :
  total_pizza = 5 / 8 ∧ num_employees = 4 →
  total_pizza / num_employees = 5 / 32 := by
  sorry

end NUMINAMATH_CALUDE_pizza_division_l1296_129653


namespace NUMINAMATH_CALUDE_equation_solutions_l1296_129620

theorem equation_solutions (x : ℝ) : 
  (1 / x^2 + 2 / x = 5/4) ↔ (x = 2 ∨ x = -2/5) :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l1296_129620


namespace NUMINAMATH_CALUDE_smallest_determinant_and_minimal_pair_l1296_129674

def determinant (a b : ℤ) : ℤ := 36 * b - 81 * a

theorem smallest_determinant_and_minimal_pair :
  (∃ c : ℕ+, ∀ a b : ℤ, determinant a b ≠ 0 → c ≤ |determinant a b|) ∧
  (∃ a b : ℕ, determinant a b = 9 ∧
    ∀ a' b' : ℕ, determinant a' b' = 9 → a + b ≤ a' + b') :=
by sorry

end NUMINAMATH_CALUDE_smallest_determinant_and_minimal_pair_l1296_129674


namespace NUMINAMATH_CALUDE_women_count_is_twenty_l1296_129641

/-- Represents a social event with dancing participants -/
structure DancingEvent where
  num_men : ℕ
  num_women : ℕ
  dances_per_man : ℕ
  dances_per_woman : ℕ

/-- The number of women at the event given the conditions -/
def women_count (event : DancingEvent) : ℕ :=
  (event.num_men * event.dances_per_man) / event.dances_per_woman

/-- Theorem stating that the number of women at the event is 20 -/
theorem women_count_is_twenty (event : DancingEvent) 
  (h1 : event.num_men = 15)
  (h2 : event.dances_per_man = 4)
  (h3 : event.dances_per_woman = 3) :
  women_count event = 20 := by
  sorry

end NUMINAMATH_CALUDE_women_count_is_twenty_l1296_129641


namespace NUMINAMATH_CALUDE_triangle_side_lengths_l1296_129689

theorem triangle_side_lengths (a b c : ℝ) (angleC : ℝ) (area : ℝ) :
  a = 3 →
  angleC = 2 * Real.pi / 3 →
  area = 3 * Real.sqrt 3 / 4 →
  1/2 * a * b * Real.sin angleC = area →
  Real.cos angleC = (a^2 + b^2 - c^2) / (2 * a * b) →
  b = 1 ∧ c = Real.sqrt 13 := by
  sorry


end NUMINAMATH_CALUDE_triangle_side_lengths_l1296_129689


namespace NUMINAMATH_CALUDE_obtain_100_with_fewer_sevens_l1296_129637

theorem obtain_100_with_fewer_sevens : ∃ (expr : ℕ), 
  (expr = 100) ∧ 
  (∃ (a b c d e f g h i : ℕ), 
    (a + b + c + d + e + f + g + h + i < 10) ∧
    (expr = (777 / 7 - 77 / 7) ∨ 
     expr = (7 * 7 + 7 * 7 + 7 / 7 + 7 / 7))) :=
by sorry

end NUMINAMATH_CALUDE_obtain_100_with_fewer_sevens_l1296_129637


namespace NUMINAMATH_CALUDE_some_number_value_l1296_129664

theorem some_number_value (x : ℝ) : 65 + 5 * x / (180 / 3) = 66 → x = 12 := by
  sorry

end NUMINAMATH_CALUDE_some_number_value_l1296_129664


namespace NUMINAMATH_CALUDE_coat_cost_l1296_129602

def weekly_savings : ℕ := 25
def weeks_of_saving : ℕ := 6
def bill_fraction : ℚ := 1 / 3
def dad_contribution : ℕ := 70

theorem coat_cost : 
  weekly_savings * weeks_of_saving - 
  (weekly_savings * weeks_of_saving : ℚ) * bill_fraction +
  dad_contribution = 170 := by sorry

end NUMINAMATH_CALUDE_coat_cost_l1296_129602


namespace NUMINAMATH_CALUDE_sum_of_differences_l1296_129630

def S : Finset ℕ := Finset.range 9

def diff_sum (s : Finset ℕ) : ℕ :=
  s.sum (fun i => s.sum (fun j => if i > j then 2^i - 2^j else 0))

theorem sum_of_differences : diff_sum S = 3096 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_differences_l1296_129630


namespace NUMINAMATH_CALUDE_square_area_from_diagonal_l1296_129663

/-- The area of a square with diagonal length 8√2 is 64 -/
theorem square_area_from_diagonal : 
  ∀ (s : ℝ), s > 0 → s * s * 2 = (8 * Real.sqrt 2) ^ 2 → s * s = 64 := by
sorry

end NUMINAMATH_CALUDE_square_area_from_diagonal_l1296_129663


namespace NUMINAMATH_CALUDE_bob_guaranteed_victory_l1296_129669

/-- Represents a grid in the game -/
def Grid := Matrix (Fin 2011) (Fin 2011) ℕ

/-- The size of the grid -/
def gridSize : ℕ := 2011

/-- The total number of grids Alice has -/
def aliceGridCount : ℕ := 2010

/-- Checks if a grid is valid (strictly increasing across rows and down columns) -/
def isValidGrid (g : Grid) : Prop :=
  ∀ i j k, i < j → g i k < g j k ∧ g k i < g k j

/-- Checks if two grids are different -/
def areDifferentGrids (g1 g2 : Grid) : Prop :=
  ∃ i j, g1 i j ≠ g2 i j

/-- Checks if Bob wins against a given grid -/
def bobWins (bobGrid aliceGrid : Grid) : Prop :=
  ∃ i j k, aliceGrid i j = bobGrid k i ∧ aliceGrid i k = bobGrid k j

/-- Theorem: Bob can guarantee victory with at most 1 swap -/
theorem bob_guaranteed_victory :
  ∃ (initialBobGrid : Grid) (swappedBobGrid : Grid),
    isValidGrid initialBobGrid ∧
    isValidGrid swappedBobGrid ∧
    (∀ (aliceGrids : Fin aliceGridCount → Grid),
      (∀ i, isValidGrid (aliceGrids i)) →
      (∀ i j, i ≠ j → areDifferentGrids (aliceGrids i) (aliceGrids j)) →
      (bobWins initialBobGrid (aliceGrids i) ∨
       bobWins swappedBobGrid (aliceGrids i))) :=
sorry

end NUMINAMATH_CALUDE_bob_guaranteed_victory_l1296_129669


namespace NUMINAMATH_CALUDE_hydrogen_chloride_production_l1296_129673

/-- Represents the balanced chemical equation for the reaction between methane and chlorine -/
structure BalancedEquation where
  methane : ℕ
  chlorine : ℕ
  tetrachloromethane : ℕ
  hydrogen_chloride : ℕ
  balanced : methane = 1 ∧ chlorine = 4 ∧ tetrachloromethane = 1 ∧ hydrogen_chloride = 4

/-- Represents the given reaction conditions -/
structure ReactionConditions where
  methane : ℕ
  chlorine : ℕ
  tetrachloromethane : ℕ
  methane_eq : methane = 3
  chlorine_eq : chlorine = 12
  tetrachloromethane_eq : tetrachloromethane = 3

/-- Theorem stating that given the reaction conditions, 12 moles of hydrogen chloride are produced -/
theorem hydrogen_chloride_production 
  (balanced : BalancedEquation) 
  (conditions : ReactionConditions) : 
  conditions.methane * balanced.hydrogen_chloride = 12 := by
  sorry

end NUMINAMATH_CALUDE_hydrogen_chloride_production_l1296_129673


namespace NUMINAMATH_CALUDE_function_property_l1296_129686

-- Define the functions
def f1 (x : ℝ) := |x|
def f2 (x : ℝ) := x - |x|
def f3 (x : ℝ) := x + 1
def f4 (x : ℝ) := -x

-- Define the property we're checking
def satisfiesProperty (f : ℝ → ℝ) : Prop :=
  ∀ x, f (2 * x) = 2 * f x

-- Theorem statement
theorem function_property :
  satisfiesProperty f1 ∧
  satisfiesProperty f2 ∧
  ¬satisfiesProperty f3 ∧
  satisfiesProperty f4 :=
sorry

end NUMINAMATH_CALUDE_function_property_l1296_129686


namespace NUMINAMATH_CALUDE_average_length_is_10_over_3_l1296_129613

-- Define the lengths of the strings
def string1_length : ℚ := 2
def string2_length : ℚ := 5
def string3_length : ℚ := 3

-- Define the number of strings
def num_strings : ℕ := 3

-- Define the average length calculation
def average_length : ℚ := (string1_length + string2_length + string3_length) / num_strings

-- Theorem statement
theorem average_length_is_10_over_3 : average_length = 10 / 3 := by
  sorry

end NUMINAMATH_CALUDE_average_length_is_10_over_3_l1296_129613


namespace NUMINAMATH_CALUDE_average_weight_of_students_l1296_129670

theorem average_weight_of_students (girls_count boys_count : ℕ) 
  (girls_avg_weight boys_avg_weight : ℝ) :
  girls_count = 5 →
  boys_count = 5 →
  girls_avg_weight = 45 →
  boys_avg_weight = 55 →
  let total_count := girls_count + boys_count
  let total_weight := girls_count * girls_avg_weight + boys_count * boys_avg_weight
  (total_weight / total_count : ℝ) = 50 := by
sorry

end NUMINAMATH_CALUDE_average_weight_of_students_l1296_129670


namespace NUMINAMATH_CALUDE_y_value_when_x_is_one_l1296_129643

-- Define the inverse square relationship between x and y
def inverse_square_relation (k : ℝ) (x y : ℝ) : Prop :=
  x = k / (y ^ 2)

-- Theorem statement
theorem y_value_when_x_is_one 
  (k : ℝ) 
  (h1 : inverse_square_relation k 0.1111111111111111 6) 
  (h2 : k > 0) :
  ∃ y : ℝ, inverse_square_relation k 1 y ∧ y = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_y_value_when_x_is_one_l1296_129643


namespace NUMINAMATH_CALUDE_double_division_remainder_l1296_129688

def p (x : ℝ) : ℝ := x^10

def q1 (x : ℝ) : ℝ := 
  x^9 + 2*x^8 + 4*x^7 + 8*x^6 + 16*x^5 + 32*x^4 + 64*x^3 + 128*x^2 + 256*x + 512

theorem double_division_remainder (x : ℝ) : 
  ∃ (q2 : ℝ → ℝ) (r2 : ℝ), p x = (x - 2) * ((x - 2) * q2 x + q1 2) + r2 ∧ r2 = 5120 := by
  sorry

end NUMINAMATH_CALUDE_double_division_remainder_l1296_129688


namespace NUMINAMATH_CALUDE_greatest_n_condition_l1296_129698

def sum_of_squares (n : ℕ) : ℕ := n * (n + 1) * (2 * n + 1) / 6

def is_perfect_square (m : ℕ) : Prop := ∃ k : ℕ, m = k * k

def condition (n : ℕ) : Prop :=
  is_perfect_square (sum_of_squares n * (sum_of_squares (2 * n) - sum_of_squares n))

theorem greatest_n_condition :
  (1921 ≤ 2023) ∧ 
  condition 1921 ∧
  ∀ m : ℕ, (m > 1921 ∧ m ≤ 2023) → ¬(condition m) :=
sorry

end NUMINAMATH_CALUDE_greatest_n_condition_l1296_129698


namespace NUMINAMATH_CALUDE_min_fixed_amount_l1296_129671

def fixed_amount (F : ℝ) : Prop :=
  ∀ (S : ℝ), S ≥ 7750 → F + 0.04 * S ≥ 500

theorem min_fixed_amount :
  ∃ (F : ℝ), F ≥ 190 ∧ fixed_amount F :=
sorry

end NUMINAMATH_CALUDE_min_fixed_amount_l1296_129671


namespace NUMINAMATH_CALUDE_black_blue_difference_l1296_129677

/-- Represents Sam's pen collection -/
structure PenCollection where
  black : ℕ
  blue : ℕ
  red : ℕ
  pencils : ℕ

/-- Conditions for Sam's pen collection -/
def validCollection (c : PenCollection) : Prop :=
  c.black > c.blue ∧
  c.blue = 2 * c.pencils ∧
  c.pencils = 8 ∧
  c.red = c.pencils - 2 ∧
  c.black + c.blue + c.red = 48

/-- Theorem stating the difference between black and blue pens -/
theorem black_blue_difference (c : PenCollection) 
  (h : validCollection c) : c.black - c.blue = 10 := by
  sorry


end NUMINAMATH_CALUDE_black_blue_difference_l1296_129677


namespace NUMINAMATH_CALUDE_d_values_l1296_129680

def a (n : ℕ) : ℕ := 20 + n^2

def d (n : ℕ) : ℕ := Nat.gcd (a n) (a (n + 1))

theorem d_values : {n : ℕ | n > 0} → {d n | n : ℕ} = {1, 3, 9, 27, 81} := by sorry

end NUMINAMATH_CALUDE_d_values_l1296_129680


namespace NUMINAMATH_CALUDE_hyperbola_triangle_perimeter_l1296_129644

/-- The hyperbola equation -/
def hyperbola_equation (x y : ℝ) : Prop := x^2 / 9 - y^2 / 16 = 1

/-- The distance between foci -/
def focal_distance : ℝ := 10

/-- Point P is on the hyperbola -/
def point_on_hyperbola (P : ℝ × ℝ) : Prop :=
  hyperbola_equation P.1 P.2

/-- The distance from P to the right focus F₂ -/
def distance_PF₂ : ℝ := 7

/-- The perimeter of triangle F₁PF₂ -/
def triangle_perimeter (d_PF₁ : ℝ) : ℝ :=
  d_PF₁ + distance_PF₂ + focal_distance

theorem hyperbola_triangle_perimeter :
  ∀ P : ℝ × ℝ, point_on_hyperbola P →
  ∃ d_PF₁ : ℝ, triangle_perimeter d_PF₁ = 30 :=
sorry

end NUMINAMATH_CALUDE_hyperbola_triangle_perimeter_l1296_129644


namespace NUMINAMATH_CALUDE_interest_problem_l1296_129640

/-- Given compound and simple interest conditions, prove the principal amount -/
theorem interest_problem (P R : ℝ) : 
  P * ((1 + R / 100) ^ 2 - 1) = 11730 →
  (P * R * 2) / 100 = 10200 →
  P = 34000 := by
  sorry

end NUMINAMATH_CALUDE_interest_problem_l1296_129640


namespace NUMINAMATH_CALUDE_rectangle_area_and_ratio_l1296_129634

/-- Given a rectangle with original length a and width b -/
structure Rectangle where
  length : ℝ
  width : ℝ

/-- The new rectangle after increasing dimensions -/
def new_rectangle (r : Rectangle) : Rectangle :=
  { length := 1.12 * r.length,
    width := 1.15 * r.width }

/-- The area of a rectangle -/
def area (r : Rectangle) : ℝ := r.length * r.width

/-- The perimeter of a rectangle -/
def perimeter (r : Rectangle) : ℝ := 2 * (r.length + r.width)

/-- Theorem: The area increase and length-to-width ratio of the rectangle -/
theorem rectangle_area_and_ratio (r : Rectangle) :
  (area (new_rectangle r) = 1.288 * area r) ∧
  (perimeter (new_rectangle r) = 1.13 * perimeter r → r.length = 2 * r.width) := by
  sorry


end NUMINAMATH_CALUDE_rectangle_area_and_ratio_l1296_129634


namespace NUMINAMATH_CALUDE_completing_square_equivalence_l1296_129623

theorem completing_square_equivalence (x : ℝ) :
  (x^2 - 6*x + 4 = 0) ↔ ((x - 3)^2 = 5) := by
  sorry

end NUMINAMATH_CALUDE_completing_square_equivalence_l1296_129623
