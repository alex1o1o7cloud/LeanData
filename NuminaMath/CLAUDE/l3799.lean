import Mathlib

namespace n3_equals_9_l3799_379997

def is_multiple_of_9 (n : ℕ) : Prop := ∃ k, n = 9 * k

def sum_of_digits (n : ℕ) : ℕ := sorry

theorem n3_equals_9 
  (N : ℕ) 
  (h1 : 10^1989 ≤ 16*N ∧ 16*N < 10^1990) 
  (h2 : is_multiple_of_9 (16*N)) 
  (N1 : ℕ) (h3 : N1 = sum_of_digits N)
  (N2 : ℕ) (h4 : N2 = sum_of_digits N1)
  (N3 : ℕ) (h5 : N3 = sum_of_digits N2) :
  N3 = 9 :=
sorry

end n3_equals_9_l3799_379997


namespace boat_speed_theorem_l3799_379932

/-- Represents the speed of a boat in a stream -/
structure BoatInStream where
  boatSpeed : ℝ  -- Speed of the boat in still water
  streamSpeed : ℝ  -- Speed of the stream

/-- Calculates the effective speed of the boat -/
def BoatInStream.effectiveSpeed (b : BoatInStream) (upstream : Bool) : ℝ :=
  if upstream then b.boatSpeed - b.streamSpeed else b.boatSpeed + b.streamSpeed

/-- Theorem: If the time taken to row upstream is twice the time taken to row downstream
    for the same distance, and the stream speed is 24, then the boat speed in still water is 72 -/
theorem boat_speed_theorem (b : BoatInStream) (distance : ℝ) 
    (h1 : b.streamSpeed = 24)
    (h2 : distance / b.effectiveSpeed true = 2 * (distance / b.effectiveSpeed false)) :
    b.boatSpeed = 72 := by
  sorry

#check boat_speed_theorem

end boat_speed_theorem_l3799_379932


namespace tom_barbados_trip_cost_l3799_379926

/-- The total cost for Tom's trip to Barbados -/
def total_cost (num_vaccines : ℕ) (vaccine_cost : ℚ) (doctor_visit_cost : ℚ) 
  (insurance_coverage : ℚ) (trip_cost : ℚ) : ℚ :=
  let medical_cost := num_vaccines * vaccine_cost + doctor_visit_cost
  let out_of_pocket_medical := medical_cost * (1 - insurance_coverage)
  out_of_pocket_medical + trip_cost

/-- Theorem stating the total cost for Tom's trip to Barbados -/
theorem tom_barbados_trip_cost :
  total_cost 10 45 250 0.8 1200 = 1340 := by
  sorry

end tom_barbados_trip_cost_l3799_379926


namespace sin_sum_special_angles_l3799_379962

theorem sin_sum_special_angles : 
  Real.sin (Real.arcsin (4/5) + Real.arctan (Real.sqrt 3)) = (2 + 3 * Real.sqrt 3) / 10 := by
  sorry

end sin_sum_special_angles_l3799_379962


namespace arithmetic_sequence_and_sum_l3799_379911

-- Define the arithmetic sequence a_n
def a (n : ℕ) : ℚ := sorry

-- Define the conditions
axiom a7_eq_4 : a 7 = 4
axiom a19_eq_2a9 : a 19 = 2 * a 9

-- Define b_n
def b (n : ℕ) : ℚ := 1 / (2 * n * a n)

-- Define the sum of the first n terms of b_n
def S (n : ℕ) : ℚ := sorry

theorem arithmetic_sequence_and_sum :
  (∀ n : ℕ, a n = (n + 1) / 2) ∧
  (∀ n : ℕ, S n = n / (n + 1)) := by sorry

end arithmetic_sequence_and_sum_l3799_379911


namespace total_paid_is_117_l3799_379940

/-- Calculates the total amount paid after applying a senior citizen discount on Tuesday --/
def total_paid_after_discount (jimmy_shorts : ℕ) (jimmy_short_price : ℚ) 
                               (irene_shirts : ℕ) (irene_shirt_price : ℚ) 
                               (discount_rate : ℚ) : ℚ :=
  let total_before_discount := jimmy_shorts * jimmy_short_price + irene_shirts * irene_shirt_price
  let discount_amount := total_before_discount * discount_rate
  total_before_discount - discount_amount

/-- Proves that the total amount paid after the senior citizen discount is $117 --/
theorem total_paid_is_117 : 
  total_paid_after_discount 3 15 5 17 (1/10) = 117 := by
  sorry

end total_paid_is_117_l3799_379940


namespace canoe_production_sum_l3799_379928

theorem canoe_production_sum : 
  let a : ℕ := 5  -- first term
  let r : ℕ := 3  -- common ratio
  let n : ℕ := 8  -- number of terms
  a * (r^n - 1) / (r - 1) = 16400 :=
by sorry

end canoe_production_sum_l3799_379928


namespace mother_daughter_age_relation_l3799_379996

theorem mother_daughter_age_relation : 
  ∀ (mother_current_age daughter_future_age : ℕ),
    mother_current_age = 41 →
    daughter_future_age = 26 →
    ∃ (years_ago : ℕ),
      years_ago = 5 ∧
      mother_current_age - years_ago = 2 * (daughter_future_age - 3 - years_ago) :=
by
  sorry

end mother_daughter_age_relation_l3799_379996


namespace x_plus_2y_equals_100_l3799_379910

theorem x_plus_2y_equals_100 (x y : ℝ) (h1 : y = 25) (h2 : x = 50) : x + 2*y = 100 := by
  sorry

end x_plus_2y_equals_100_l3799_379910


namespace function_equality_implies_m_value_l3799_379927

theorem function_equality_implies_m_value :
  ∀ (m : ℚ),
  let f : ℚ → ℚ := λ x => x^2 - 3*x + m
  let g : ℚ → ℚ := λ x => x^2 - 3*x + 5*m
  3 * f 5 = 2 * g 5 →
  m = 10/7 :=
by
  sorry

end function_equality_implies_m_value_l3799_379927


namespace complement_intersection_theorem_l3799_379965

def U : Set Int := {-2, -1, 0, 1, 2, 3}
def A : Set Int := {-1, 0, 1}
def B : Set Int := {1, 2, 3}

theorem complement_intersection_theorem :
  (U \ (A ∩ B)) = {-2, -1, 0, 2, 3} := by sorry

end complement_intersection_theorem_l3799_379965


namespace smallest_ten_digit_max_sum_l3799_379930

def sum_of_digits (n : Nat) : Nat :=
  if n < 10 then n else (n % 10) + sum_of_digits (n / 10)

def is_ten_digit (n : Nat) : Prop :=
  1000000000 ≤ n ∧ n < 10000000000

theorem smallest_ten_digit_max_sum : 
  ∀ n : Nat, is_ten_digit n → n < 1999999999 → sum_of_digits n < sum_of_digits 1999999999 :=
sorry

#eval sum_of_digits 1999999999

end smallest_ten_digit_max_sum_l3799_379930


namespace rectangle_tiling_l3799_379952

/-- A rectangle can be tiled with 4x4 squares -/
def is_tileable (m n : ℕ) : Prop :=
  ∃ (a b : ℕ), m = 4 * a ∧ n = 4 * b

/-- If a rectangle with dimensions m × n can be tiled with 4 × 4 squares, 
    then m and n are divisible by 4 -/
theorem rectangle_tiling (m n : ℕ) :
  is_tileable m n → (4 ∣ m) ∧ (4 ∣ n) := by
  sorry

end rectangle_tiling_l3799_379952


namespace expression_simplification_l3799_379922

theorem expression_simplification (x : ℝ) (h : x = 2) :
  (1 / (x - 3)) / (1 / (x^2 - 9)) - (x / (x + 1)) * ((x^2 + x) / x^2) = 4 := by
  sorry

end expression_simplification_l3799_379922


namespace four_prime_pairs_sum_50_l3799_379950

/-- A function that returns true if a natural number is prime, false otherwise -/
def isPrime (n : ℕ) : Prop := sorry

/-- A function that returns the number of unordered pairs of prime numbers that sum to a given number -/
def countPrimePairs (sum : ℕ) : ℕ := sorry

/-- Theorem stating that there are exactly 4 unordered pairs of prime numbers that sum to 50 -/
theorem four_prime_pairs_sum_50 : countPrimePairs 50 = 4 := by sorry

end four_prime_pairs_sum_50_l3799_379950


namespace man_business_ownership_l3799_379915

/-- 
Given a business valued at 10000 rs, if a man sells 3/5 of his shares for 2000 rs,
then he originally owned 1/3 of the business.
-/
theorem man_business_ownership (man_share : ℚ) : 
  (3 / 5 : ℚ) * man_share * 10000 = 2000 → man_share = 1 / 3 := by
  sorry

end man_business_ownership_l3799_379915


namespace isosceles_right_triangle_example_l3799_379977

/-- A triangle with sides a, b, and c is an isosceles right triangle -/
def is_isosceles_right_triangle (a b c : ℝ) : Prop :=
  (a = b ∨ b = c ∨ a = c) ∧  -- Two sides are equal
  (a^2 + b^2 = c^2 ∨ b^2 + c^2 = a^2 ∨ a^2 + c^2 = b^2)  -- Pythagorean theorem holds

/-- The set {5, 5, 5√2} represents the sides of an isosceles right triangle -/
theorem isosceles_right_triangle_example : is_isosceles_right_triangle 5 5 (5 * Real.sqrt 2) := by
  sorry

end isosceles_right_triangle_example_l3799_379977


namespace circle_radius_l3799_379971

theorem circle_radius (x y : ℝ) (h : x + y = 100 * Real.pi) : 
  (∃ r : ℝ, x = Real.pi * r^2 ∧ y = 2 * Real.pi * r) → 
  (∃ r : ℝ, x = Real.pi * r^2 ∧ y = 2 * Real.pi * r ∧ r = 10) :=
by sorry

end circle_radius_l3799_379971


namespace factorization_problems_l3799_379970

theorem factorization_problems :
  (∀ x : ℝ, x^3 - 9*x = x*(x+3)*(x-3)) ∧
  (∀ a b : ℝ, a^3*b - 2*a^2*b + a*b = a*b*(a-1)^2) := by
  sorry

end factorization_problems_l3799_379970


namespace gcd_119_153_l3799_379923

theorem gcd_119_153 : Nat.gcd 119 153 = 17 := by
  -- The proof would go here
  sorry

end gcd_119_153_l3799_379923


namespace cube_edge_length_from_paint_cost_l3799_379956

/-- Proves that a cube with a specific edge length costs $1.60 to paint given certain paint properties -/
theorem cube_edge_length_from_paint_cost 
  (paint_cost_per_quart : ℝ) 
  (paint_coverage_per_quart : ℝ) 
  (total_paint_cost : ℝ) : 
  paint_cost_per_quart = 3.20 →
  paint_coverage_per_quart = 1200 →
  total_paint_cost = 1.60 →
  ∃ (edge_length : ℝ), 
    edge_length = 10 ∧ 
    total_paint_cost = (6 * edge_length^2) / paint_coverage_per_quart * paint_cost_per_quart :=
by
  sorry


end cube_edge_length_from_paint_cost_l3799_379956


namespace percentage_error_division_vs_multiplication_l3799_379916

theorem percentage_error_division_vs_multiplication (x : ℝ) : 
  let correct_result := 10 * x
  let incorrect_result := x / 10
  let error := correct_result - incorrect_result
  let percentage_error := (error / correct_result) * 100
  percentage_error = 99 := by
sorry

end percentage_error_division_vs_multiplication_l3799_379916


namespace sum_of_roots_l3799_379969

-- Define the function f
def f (x : ℝ) : ℝ := x^3 + 3*x^2 + 6*x + 14

-- State the theorem
theorem sum_of_roots (a b : ℝ) (ha : f a = 1) (hb : f b = 19) : a + b = -2 := by
  sorry

end sum_of_roots_l3799_379969


namespace equation_solution_l3799_379944

theorem equation_solution (x : ℝ) : 
  x ≠ -3 → (-x^2 = (3*x + 1) / (x + 3) ↔ x = -1) :=
by
  sorry

end equation_solution_l3799_379944


namespace constant_in_toll_formula_l3799_379941

/-- The toll formula for a truck crossing a bridge -/
def toll (x : ℕ) (constant : ℝ) : ℝ :=
  1.50 + 0.50 * (x - constant)

/-- The number of axles on an 18-wheel truck with 2 wheels on its front axle and 2 wheels on each of its other axles -/
def axles_18_wheel_truck : ℕ := 9

theorem constant_in_toll_formula :
  ∃ (constant : ℝ), 
    toll axles_18_wheel_truck constant = 5 ∧ 
    constant = 2 := by sorry

end constant_in_toll_formula_l3799_379941


namespace total_wrapping_cost_l3799_379925

/-- Represents a wrapping paper design with its cost and wrapping capacities -/
structure WrappingPaper where
  cost : ℝ
  shirtBoxCapacity : ℕ
  xlBoxCapacity : ℕ
  xxlBoxCapacity : ℕ

/-- Calculates the number of rolls needed for a given number of boxes -/
def rollsNeeded (boxes : ℕ) (capacity : ℕ) : ℕ :=
  (boxes + capacity - 1) / capacity

/-- Calculates the cost for wrapping a specific type of box -/
def costForBoxType (paper : WrappingPaper) (boxes : ℕ) (capacity : ℕ) : ℝ :=
  paper.cost * (rollsNeeded boxes capacity : ℝ)

/-- Theorem stating the total cost of wrapping all boxes -/
theorem total_wrapping_cost (design1 design2 design3 : WrappingPaper)
    (shirtBoxes xlBoxes xxlBoxes : ℕ) :
    design1.cost = 4 →
    design1.shirtBoxCapacity = 5 →
    design2.cost = 8 →
    design2.xlBoxCapacity = 4 →
    design3.cost = 12 →
    design3.xxlBoxCapacity = 4 →
    shirtBoxes = 20 →
    xlBoxes = 12 →
    xxlBoxes = 6 →
    costForBoxType design1 shirtBoxes design1.shirtBoxCapacity +
    costForBoxType design2 xlBoxes design2.xlBoxCapacity +
    costForBoxType design3 xxlBoxes design3.xxlBoxCapacity = 76 := by
  sorry

end total_wrapping_cost_l3799_379925


namespace crabapple_theorem_l3799_379979

/-- The number of possible sequences of crabapple recipients in a week -/
def crabapple_sequences (num_students : ℕ) (classes_per_week : ℕ) : ℕ :=
  num_students ^ classes_per_week

/-- Theorem stating the number of possible sequences for Mrs. Crabapple's class -/
theorem crabapple_theorem :
  crabapple_sequences 15 5 = 759375 := by
  sorry

end crabapple_theorem_l3799_379979


namespace expression_value_l3799_379955

theorem expression_value (a b c d x : ℝ) : 
  (c / 3 = -(-2 * d)) →
  (2 * a = 1 / (-b)) →
  (|x| = 9) →
  (2 * a * b - 6 * d + c - x / 3 = -4 ∨ 2 * a * b - 6 * d + c - x / 3 = 2) :=
by sorry

end expression_value_l3799_379955


namespace thirty_three_not_enrolled_l3799_379993

/-- Calculates the number of students not enrolled in either French or German --/
def students_not_enrolled (total : ℕ) (french : ℕ) (german : ℕ) (both : ℕ) : ℕ :=
  total - (french + german - both)

/-- Theorem stating that 33 students are not enrolled in either French or German --/
theorem thirty_three_not_enrolled : 
  students_not_enrolled 87 41 22 9 = 33 := by
  sorry

end thirty_three_not_enrolled_l3799_379993


namespace sum_of_integers_l3799_379959

theorem sum_of_integers (x y : ℕ+) 
  (h1 : x^2 + y^2 = 250)
  (h2 : x * y = 120)
  (h3 : x^2 - y^2 = 130) :
  x + y = 10 * Real.sqrt 4.9 := by
  sorry

end sum_of_integers_l3799_379959


namespace fraction_sum_equality_l3799_379998

theorem fraction_sum_equality : (3 : ℚ) / 5 - 2 / 15 + 1 / 3 = 4 / 5 := by
  sorry

end fraction_sum_equality_l3799_379998


namespace pascal_triangle_probability_l3799_379912

/-- Represents Pascal's Triangle up to a given number of rows -/
def PascalTriangle (n : ℕ) : List (List ℕ) :=
  sorry

/-- Counts the number of elements equal to a given value in Pascal's Triangle -/
def countElementsEqual (triangle : List (List ℕ)) (value : ℕ) : ℕ :=
  sorry

/-- Calculates the total number of elements in Pascal's Triangle -/
def totalElements (triangle : List (List ℕ)) : ℕ :=
  sorry

theorem pascal_triangle_probability (n : ℕ) :
  n = 20 →
  let triangle := PascalTriangle n
  let ones := countElementsEqual triangle 1
  let twos := countElementsEqual triangle 2
  let total := totalElements triangle
  (ones + twos : ℚ) / total = 57 / 210 := by
  sorry

end pascal_triangle_probability_l3799_379912


namespace geometric_sequence_sum_l3799_379901

/-- A geometric sequence with positive terms -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, q > 0 ∧ ∀ n : ℕ, a (n + 1) = a n * q

theorem geometric_sequence_sum (a : ℕ → ℝ) :
  geometric_sequence a →
  a 1 = 3 →
  a 1 + a 2 + a 3 = 21 →
  a 4 + a 5 + a 6 = 168 := by
sorry

end geometric_sequence_sum_l3799_379901


namespace harry_apples_l3799_379917

/-- The number of apples Harry ends up with after buying more -/
def final_apples (initial : ℕ) (bought : ℕ) : ℕ := initial + bought

/-- Theorem: Harry ends up with 84 apples -/
theorem harry_apples : final_apples 79 5 = 84 := by
  sorry

end harry_apples_l3799_379917


namespace maurice_cookout_packages_l3799_379975

/-- The number of packages of ground beef Maurice needs to purchase for his cookout --/
def packages_needed (guests : ℕ) (burger_weight : ℕ) (package_weight : ℕ) : ℕ :=
  let total_people := guests + 1  -- Adding Maurice himself
  let total_weight := total_people * burger_weight
  (total_weight + package_weight - 1) / package_weight  -- Ceiling division

/-- Theorem stating that Maurice needs to purchase 4 packages of ground beef --/
theorem maurice_cookout_packages : packages_needed 9 2 5 = 4 := by
  sorry

#eval packages_needed 9 2 5

end maurice_cookout_packages_l3799_379975


namespace correct_sums_l3799_379948

theorem correct_sums (total : ℕ) (wrong_ratio : ℕ) (h1 : total = 54) (h2 : wrong_ratio = 2) :
  ∃ (correct : ℕ), correct * (1 + wrong_ratio) = total ∧ correct = 18 :=
by sorry

end correct_sums_l3799_379948


namespace smallest_sum_reciprocals_l3799_379988

theorem smallest_sum_reciprocals (x y : ℕ+) (h1 : x ≠ y) (h2 : (1 : ℚ) / x + (1 : ℚ) / y = (1 : ℚ) / 24) :
  ∃ (a b : ℕ+), a ≠ b ∧ (1 : ℚ) / a + (1 : ℚ) / b = (1 : ℚ) / 24 ∧ (a : ℕ) + b = 98 ∧
  ∀ (c d : ℕ+), c ≠ d → (1 : ℚ) / c + (1 : ℚ) / d = (1 : ℚ) / 24 → (c : ℕ) + d ≥ 98 :=
sorry

end smallest_sum_reciprocals_l3799_379988


namespace derivative_exp_sin_l3799_379946

theorem derivative_exp_sin (x : ℝ) : 
  deriv (fun x => Real.exp x * Real.sin x) x = Real.exp x * (Real.sin x + Real.cos x) := by
sorry

end derivative_exp_sin_l3799_379946


namespace square_root_of_nine_l3799_379958

-- Define the square root function
def square_root (x : ℝ) : Set ℝ :=
  {y : ℝ | y * y = x}

-- Theorem statement
theorem square_root_of_nine :
  square_root 9 = {3, -3} := by
  sorry

end square_root_of_nine_l3799_379958


namespace complex_fourth_quadrant_m_range_l3799_379964

theorem complex_fourth_quadrant_m_range (m : ℝ) : 
  let z : ℂ := Complex.mk (m + 3) (m - 1)
  (0 < z.re ∧ z.im < 0) → -3 < m ∧ m < 1 := by
  sorry

end complex_fourth_quadrant_m_range_l3799_379964


namespace linda_tees_sold_l3799_379942

/-- Calculates the number of tees sold given the prices, number of jeans sold, and total money -/
def tees_sold (jeans_price tee_price : ℕ) (jeans_sold : ℕ) (total_money : ℕ) : ℕ :=
  (total_money - jeans_price * jeans_sold) / tee_price

theorem linda_tees_sold :
  tees_sold 11 8 4 100 = 7 := by
  sorry

end linda_tees_sold_l3799_379942


namespace yoongis_pets_l3799_379961

theorem yoongis_pets (dogs : ℕ) (cats : ℕ) : dogs = 5 → cats = 2 → dogs + cats = 7 := by
  sorry

end yoongis_pets_l3799_379961


namespace maximize_x2y5_l3799_379924

theorem maximize_x2y5 (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 50) :
  x^2 * y^5 ≤ (100/7)^2 * (250/7)^5 ∧ 
  (x^2 * y^5 = (100/7)^2 * (250/7)^5 ↔ x = 100/7 ∧ y = 250/7) := by
sorry

end maximize_x2y5_l3799_379924


namespace inequality_solution_set_l3799_379903

theorem inequality_solution_set (a b : ℝ) (h : |a - b| > 2) : ∀ x : ℝ, |x - a| + |x - b| > 2 := by
  sorry

end inequality_solution_set_l3799_379903


namespace least_beads_beads_solution_l3799_379986

theorem least_beads (b : ℕ) : 
  (b % 6 = 5) ∧ (b % 8 = 3) ∧ (b % 9 = 7) → b ≥ 179 :=
by
  sorry

theorem beads_solution : 
  ∃ (b : ℕ), (b % 6 = 5) ∧ (b % 8 = 3) ∧ (b % 9 = 7) ∧ b = 179 :=
by
  sorry

end least_beads_beads_solution_l3799_379986


namespace class_size_is_24_l3799_379947

-- Define the number of candidates
def num_candidates : Nat := 4

-- Define the number of absent students
def absent_students : Nat := 5

-- Define the function to calculate votes needed to win
def votes_to_win (x : Nat) : Nat :=
  if x % 2 = 0 then x / 2 + 1 else (x + 1) / 2

-- Define the function to calculate votes received by each candidate
def votes_received (x : Nat) (missed_by : Nat) : Nat :=
  votes_to_win x - missed_by

-- Define the theorem
theorem class_size_is_24 :
  ∃ (x : Nat),
    -- x is the number of students who voted
    x + absent_students = 24 ∧
    -- Sum of votes received by all candidates equals x
    votes_received x 3 + votes_received x 9 + votes_received x 5 + votes_received x 4 = x :=
by sorry

end class_size_is_24_l3799_379947


namespace greatest_integer_third_side_l3799_379976

theorem greatest_integer_third_side (a b c : ℕ) : 
  a = 7 ∧ b = 10 ∧ 
  (a + b > c) ∧ (a + c > b) ∧ (b + c > a) ∧  -- Triangle inequality
  c ≤ a + b - 1 →                           -- Strict inequality
  (∀ d : ℕ, d > c → ¬(a + b > d ∧ a + d > b ∧ b + d > a)) →
  c = 16 := by
sorry

end greatest_integer_third_side_l3799_379976


namespace reciprocal_of_neg_sqrt_two_l3799_379933

theorem reciprocal_of_neg_sqrt_two :
  (1 : ℝ) / (-Real.sqrt 2) = -(Real.sqrt 2) / 2 := by
  sorry

end reciprocal_of_neg_sqrt_two_l3799_379933


namespace geometric_sequence_increasing_iff_135_l3799_379974

/-- A geometric sequence -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q

/-- The sequence is positive -/
def positive_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a n > 0

/-- The sequence is increasing -/
def increasing_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) > a n

/-- The condition a_1 < a_3 < a_5 -/
def condition_135 (a : ℕ → ℝ) : Prop :=
  a 1 < a 3 ∧ a 3 < a 5

theorem geometric_sequence_increasing_iff_135 
  (a : ℕ → ℝ) 
  (h_geom : geometric_sequence a) 
  (h_pos : positive_sequence a) :
  increasing_sequence a ↔ condition_135 a :=
sorry

end geometric_sequence_increasing_iff_135_l3799_379974


namespace sum_congruence_mod_seven_l3799_379906

theorem sum_congruence_mod_seven :
  (2 + 33 + 444 + 5555 + 66666 + 777777 + 8888888 + 99999999) % 7 = 5 := by
  sorry

end sum_congruence_mod_seven_l3799_379906


namespace max_angle_APB_l3799_379990

-- Define the circles C and M
def circle_C (x y : ℝ) : Prop := (x - 3)^2 + y^2 = 1
def circle_M (x y θ : ℝ) : Prop := (x - 3 - 3 * Real.cos θ)^2 + (y - 3 * Real.sin θ)^2 = 1

-- Define a point P on circle M
def point_on_M (P : ℝ × ℝ) (θ : ℝ) : Prop := circle_M P.1 P.2 θ

-- Define points A and B on circle C
def points_on_C (A B : ℝ × ℝ) : Prop := circle_C A.1 A.2 ∧ circle_C B.1 B.2

-- Define the line PAB touching circle C
def line_touches_C (P A B : ℝ × ℝ) : Prop := 
  ∃ θ : ℝ, point_on_M P θ ∧ points_on_C A B

-- Theorem stating the maximum value of angle APB
theorem max_angle_APB : 
  ∀ P A B : ℝ × ℝ, line_touches_C P A B → 
  ∃ angle : ℝ, angle ≤ π / 3 ∧ 
  (∀ P' A' B' : ℝ × ℝ, line_touches_C P' A' B' → 
   ∃ angle' : ℝ, angle' ≤ angle) :=
sorry

end max_angle_APB_l3799_379990


namespace count_marquis_duels_l3799_379968

theorem count_marquis_duels (counts dukes marquises : ℕ) 
  (h1 : counts > 0) (h2 : dukes > 0) (h3 : marquises > 0)
  (h4 : 3 * counts = 2 * dukes)
  (h5 : 6 * dukes = 3 * marquises)
  (h6 : 2 * marquises = 2 * counts * k)
  (h7 : k > 0) :
  k = 6 := by
  sorry

end count_marquis_duels_l3799_379968


namespace implicit_function_derivatives_l3799_379963

/-- Given an implicit function defined by x^y - y^x = 0, this theorem proves
    the expressions for its first and second derivatives. -/
theorem implicit_function_derivatives
  (x y : ℝ) (h : x^y = y^x) (hx : x > 0) (hy : y > 0) :
  let y' := (y^2 * (Real.log x - 1)) / (x^2 * (Real.log y - 1))
  let y'' := (x * (3 - 2 * Real.log x) * (Real.log y - 1)^2 +
              (Real.log x - 1)^2 * (2 * Real.log y - 3) * y) *
             y^2 / (x^4 * (Real.log y - 1)^3)
  ∃ f : ℝ → ℝ, (∀ t, t^(f t) = (f t)^t) ∧
               (deriv f x = y') ∧
               (deriv (deriv f) x = y'') := by
  sorry

end implicit_function_derivatives_l3799_379963


namespace function_determination_l3799_379931

/-- Given a function f(x) = x³ - ax + b where x ∈ ℝ, 
    and the tangent line to f(x) at (1, f(1)) is 2x - y + 3 = 0,
    prove that f(x) = x³ - x + 5 -/
theorem function_determination (a b : ℝ) :
  (∀ x : ℝ, ∃ f : ℝ → ℝ, f x = x^3 - a*x + b) →
  (∃ f : ℝ → ℝ, ∀ x : ℝ, f x = x^3 - a*x + b ∧ 
    (2 * 1 - f 1 + 3 = 0) ∧
    (∀ x : ℝ, (2 * x - f x + 3 = 0) → x = 1)) →
  (∀ x : ℝ, ∃ f : ℝ → ℝ, f x = x^3 - x + 5) :=
by sorry

end function_determination_l3799_379931


namespace student_count_l3799_379935

/-- The number of students in the group -/
def num_students : ℕ := 6

/-- The weight decrease when replacing the heavier student with the lighter one -/
def weight_difference : ℕ := 80 - 62

/-- The average weight decrease per student -/
def avg_weight_decrease : ℕ := 3

theorem student_count :
  num_students * avg_weight_decrease = weight_difference :=
sorry

end student_count_l3799_379935


namespace comparison_theorem_l3799_379983

theorem comparison_theorem :
  (∀ m n : ℝ, m > n → -2*m + 1 < -2*n + 1) ∧
  (∀ m n a : ℝ, 
    (m < n ∧ a = 0 → m*a = n*a) ∧
    (m < n ∧ a > 0 → m*a < n*a) ∧
    (m < n ∧ a < 0 → m*a > n*a)) := by
  sorry

end comparison_theorem_l3799_379983


namespace unique_mapping_l3799_379987

-- Define the property for the mapping
def SatisfiesProperty (f : ℕ → ℕ) : Prop :=
  ∀ n : ℕ, f (f n) ≤ (n + f n) / 2

-- Define the identity function on ℕ
def IdentityFunc : ℕ → ℕ := λ n => n

-- Theorem statement
theorem unique_mapping :
  ∀ f : ℕ → ℕ, Function.Injective f → SatisfiesProperty f → f = IdentityFunc :=
sorry

end unique_mapping_l3799_379987


namespace mom_tshirt_count_l3799_379938

/-- The number of t-shirts in each package -/
def shirts_per_package : ℕ := 6

/-- The number of packages Mom buys -/
def packages_bought : ℕ := 71

/-- The total number of t-shirts Mom will have -/
def total_shirts : ℕ := shirts_per_package * packages_bought

theorem mom_tshirt_count : total_shirts = 426 := by
  sorry

end mom_tshirt_count_l3799_379938


namespace exercise_distance_l3799_379919

/-- 
Proves that given a person who walks x miles at 3 miles per hour, 
runs 10 miles at 5 miles per hour, repeats this exercise 7 times a week, 
and spends a total of 21 hours exercising per week, 
the value of x must be 3 miles.
-/
theorem exercise_distance (x : ℝ) 
  (h_walk_speed : ℝ := 3)
  (h_run_speed : ℝ := 5)
  (h_run_distance : ℝ := 10)
  (h_days_per_week : ℕ := 7)
  (h_total_hours : ℝ := 21)
  (h_exercise_time : ℝ := h_total_hours / h_days_per_week)
  (h_time_equation : x / h_walk_speed + h_run_distance / h_run_speed = h_exercise_time) :
  x = 3 := by
sorry

end exercise_distance_l3799_379919


namespace digit_sum_subtraction_l3799_379981

theorem digit_sum_subtraction (M N P Q : ℕ) : 
  (M ≤ 9 ∧ N ≤ 9 ∧ P ≤ 9 ∧ Q ≤ 9) →
  (10 * M + N) + (10 * P + M) = 10 * Q + N →
  (10 * M + N) - (10 * P + M) = N →
  Q = 0 := by
  sorry

end digit_sum_subtraction_l3799_379981


namespace subtract_problem_l3799_379913

theorem subtract_problem (x : ℤ) (h : x - 46 = 15) : x - 29 = 32 := by
  sorry

end subtract_problem_l3799_379913


namespace untouchedShapesAfterGame_l3799_379989

/-- Represents a shape made of matches -/
inductive Shape
| Triangle
| Square
| Pentagon

/-- Represents the game state -/
structure GameState where
  triangles : Nat
  squares : Nat
  pentagons : Nat
  untouchedShapes : Nat
  currentPlayer : Bool  -- true for Petya, false for Vasya

/-- Represents a player's move -/
structure Move where
  shapeType : Shape
  isNewShape : Bool

/-- Optimal strategy for a player -/
def optimalMove (state : GameState) : Move :=
  sorry

/-- Apply a move to the game state -/
def applyMove (state : GameState) (move : Move) : GameState :=
  sorry

/-- Play the game for a given number of turns -/
def playGame (initialState : GameState) (turns : Nat) : GameState :=
  sorry

/-- The main theorem to prove -/
theorem untouchedShapesAfterGame :
  let initialState : GameState := {
    triangles := 3,
    squares := 4,
    pentagons := 5,
    untouchedShapes := 12,
    currentPlayer := true
  }
  let finalState := playGame initialState 10
  finalState.untouchedShapes = 6 := by
  sorry

end untouchedShapesAfterGame_l3799_379989


namespace triangle_ABC_properties_l3799_379908

open Real

theorem triangle_ABC_properties (a b c : ℝ) (A B C : ℝ) :
  c = Real.sqrt 3 →
  C = π / 3 →
  2 * sin (2 * A) + sin (A - B) = sin C →
  (A = π / 2 ∨ A = π / 6) ∧
  2 * Real.sqrt 3 ≤ a + b + c ∧ a + b + c ≤ 3 * Real.sqrt 3 :=
by sorry

end triangle_ABC_properties_l3799_379908


namespace test_scores_mode_l3799_379936

/-- Represents a stem-and-leaf plot entry --/
structure StemLeafEntry where
  stem : ℕ
  leaves : List ℕ

/-- Finds the mode of a list of numbers --/
def mode (l : List ℕ) : ℕ := sorry

/-- Converts a stem-and-leaf plot to a list of numbers --/
def stemLeafToList (plot : List StemLeafEntry) : List ℕ := sorry

theorem test_scores_mode (plot : List StemLeafEntry) 
  (h1 : plot = [
    ⟨5, [1, 1]⟩,
    ⟨6, [5]⟩,
    ⟨7, [2, 4]⟩,
    ⟨8, [0, 3, 6, 6]⟩,
    ⟨9, [1, 5, 5, 5, 8, 8, 8]⟩,
    ⟨10, [2, 2, 2, 2, 4]⟩,
    ⟨11, [0, 0, 0]⟩
  ]) : 
  mode (stemLeafToList plot) = 102 := by sorry

end test_scores_mode_l3799_379936


namespace total_vegetarian_eaters_l3799_379945

/-- Represents the dietary preferences in a family -/
structure DietaryPreferences where
  vegetarian : ℕ
  nonVegetarian : ℕ
  bothVegNonVeg : ℕ
  vegan : ℕ
  veganAndVegetarian : ℕ
  pescatarian : ℕ
  pescatarianAndBoth : ℕ

/-- Theorem stating the total number of people eating vegetarian meals -/
theorem total_vegetarian_eaters (prefs : DietaryPreferences)
  (h1 : prefs.vegetarian = 13)
  (h2 : prefs.nonVegetarian = 7)
  (h3 : prefs.bothVegNonVeg = 8)
  (h4 : prefs.vegan = 5)
  (h5 : prefs.veganAndVegetarian = 3)
  (h6 : prefs.pescatarian = 4)
  (h7 : prefs.pescatarianAndBoth = 2) :
  prefs.vegetarian + prefs.bothVegNonVeg + (prefs.vegan - prefs.veganAndVegetarian) = 23 := by
  sorry

end total_vegetarian_eaters_l3799_379945


namespace dress_price_difference_l3799_379921

theorem dress_price_difference (original_price : ℝ) : 
  (original_price * 0.85 = 71.4) →
  (original_price - (71.4 * 1.25)) = 5.25 := by
sorry

end dress_price_difference_l3799_379921


namespace smallest_d_for_inverse_l3799_379980

/-- The function g(x) = (x - 3)^2 - 7 -/
def g (x : ℝ) : ℝ := (x - 3)^2 - 7

/-- The property of being strictly increasing on an interval -/
def StrictlyIncreasing (f : ℝ → ℝ) (a : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x < y → f x < f y

/-- The smallest value of d for which g has an inverse function on [d, ∞) -/
theorem smallest_d_for_inverse : 
  (∃ d : ℝ, StrictlyIncreasing g d ∧ 
    (∀ c : ℝ, c < d → ¬StrictlyIncreasing g c)) ∧ 
  (∀ d : ℝ, StrictlyIncreasing g d → d ≥ 3) ∧
  StrictlyIncreasing g 3 :=
sorry

end smallest_d_for_inverse_l3799_379980


namespace reflection_of_point_l3799_379943

/-- Reflects a point across the x-axis in a 2D Cartesian coordinate system -/
def reflect_x (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.1, -p.2)

/-- Theorem: The reflection of the point (5,2) across the x-axis is (5,-2) -/
theorem reflection_of_point : reflect_x (5, 2) = (5, -2) := by
  sorry

end reflection_of_point_l3799_379943


namespace trig_identity_l3799_379909

theorem trig_identity (α : Real) (h : Real.sin (α - π / 12) = 1 / 3) :
  Real.cos (α + 17 * π / 12) = 1 / 3 := by
  sorry

end trig_identity_l3799_379909


namespace davids_average_marks_l3799_379929

def english_marks : ℝ := 90
def mathematics_marks : ℝ := 92
def physics_marks : ℝ := 85
def chemistry_marks : ℝ := 87
def biology_marks : ℝ := 85

def total_marks : ℝ := english_marks + mathematics_marks + physics_marks + chemistry_marks + biology_marks
def number_of_subjects : ℝ := 5

theorem davids_average_marks :
  total_marks / number_of_subjects = 87.8 := by
  sorry

end davids_average_marks_l3799_379929


namespace solution_set_when_a_eq_3_range_of_a_l3799_379973

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := |x - a| - |x + 1|

-- Part 1
theorem solution_set_when_a_eq_3 :
  {x : ℝ | f 3 x ≥ 2*x + 3} = {x : ℝ | x ≤ -1/4} := by sorry

-- Part 2
theorem range_of_a :
  ∀ a : ℝ, (∃ x ∈ Set.Icc 1 2, f a x ≤ |x - 5|) → a ∈ Set.Icc (-4) 7 := by sorry

end solution_set_when_a_eq_3_range_of_a_l3799_379973


namespace min_handshakes_in_gathering_l3799_379949

/-- Represents a gathering of people and their handshakes -/
structure Gathering where
  people : Nat
  min_handshakes_per_person : Nat
  total_handshakes : Nat

/-- The minimum number of handshakes in a gathering of 30 people 
    where each person shakes hands with at least 3 others -/
theorem min_handshakes_in_gathering (g : Gathering) 
  (h1 : g.people = 30)
  (h2 : g.min_handshakes_per_person ≥ 3) :
  g.total_handshakes ≥ 45 ∧ 
  ∃ (arrangement : Gathering), 
    arrangement.people = 30 ∧ 
    arrangement.min_handshakes_per_person = 3 ∧ 
    arrangement.total_handshakes = 45 := by
  sorry

end min_handshakes_in_gathering_l3799_379949


namespace max_pieces_is_100_l3799_379994

/-- The size of the large cake in inches -/
def large_cake_size : ℕ := 20

/-- The size of the small cake pieces in inches -/
def small_piece_size : ℕ := 2

/-- The area of the large cake in square inches -/
def large_cake_area : ℕ := large_cake_size * large_cake_size

/-- The area of a small cake piece in square inches -/
def small_piece_area : ℕ := small_piece_size * small_piece_size

/-- The maximum number of small pieces that can be cut from the large cake -/
def max_pieces : ℕ := large_cake_area / small_piece_area

theorem max_pieces_is_100 : max_pieces = 100 := by
  sorry

end max_pieces_is_100_l3799_379994


namespace bart_earnings_l3799_379992

/-- The amount of money Bart receives for each question he answers in a survey. -/
def amount_per_question : ℚ := 1/5

/-- The number of questions in each survey. -/
def questions_per_survey : ℕ := 10

/-- The number of surveys Bart completed on Monday. -/
def monday_surveys : ℕ := 3

/-- The number of surveys Bart completed on Tuesday. -/
def tuesday_surveys : ℕ := 4

/-- The total amount of money Bart earned for the surveys completed on Monday and Tuesday. -/
def total_earnings : ℚ := 14

theorem bart_earnings : 
  amount_per_question * (questions_per_survey * (monday_surveys + tuesday_surveys)) = total_earnings := by
  sorry

end bart_earnings_l3799_379992


namespace intersection_property_l3799_379957

/-- Given a function f(x) = |sin x| and a line y = kx (k > 0) that intersect at exactly three points,
    with the maximum x-coordinate of the intersections being α, prove that:
    cos α / (sin α + sin 3α) = (1 + α²) / (4α) -/
theorem intersection_property (k α : ℝ) (hk : k > 0) 
    (h_intersections : ∃ (x₁ x₂ x₃ : ℝ), x₁ < x₂ ∧ x₂ < x₃ ∧ x₃ ≤ α ∧
      (∀ x, k * x = |Real.sin x| ↔ x = x₁ ∨ x = x₂ ∨ x = x₃))
    (h_max : ∀ x, k * x = |Real.sin x| → x ≤ α) :
  Real.cos α / (Real.sin α + Real.sin (3 * α)) = (1 + α^2) / (4 * α) := by
  sorry

end intersection_property_l3799_379957


namespace min_value_sqrt_plus_reciprocal_l3799_379967

theorem min_value_sqrt_plus_reciprocal (x : ℝ) (hx : x > 0) : 
  2 * Real.sqrt x + 1 / x ≥ 3 ∧ 
  (2 * Real.sqrt x + 1 / x = 3 ↔ x = 1) := by
sorry

end min_value_sqrt_plus_reciprocal_l3799_379967


namespace reciprocal_sum_pairs_l3799_379972

theorem reciprocal_sum_pairs : 
  ∃! (s : Finset (ℕ × ℕ)), 
    (∀ (p : ℕ × ℕ), p ∈ s ↔ 
      p.1 > 0 ∧ p.2 > 0 ∧ (1 : ℚ) / p.1 + (1 : ℚ) / p.2 = (1 : ℚ) / 4) ∧
    s.card = 5 :=
sorry

end reciprocal_sum_pairs_l3799_379972


namespace expected_jumps_is_eight_l3799_379978

/-- Represents the behavior of a trainer --/
structure Trainer where
  jumps : ℕ
  gives_treat : Bool

/-- The expected number of jumps before getting a treat --/
def expected_jumps (trainers : List Trainer) : ℝ :=
  sorry

/-- The list of trainers with their behaviors --/
def dog_trainers : List Trainer :=
  [{ jumps := 0, gives_treat := true },
   { jumps := 5, gives_treat := true },
   { jumps := 3, gives_treat := false }]

/-- The main theorem stating the expected number of jumps --/
theorem expected_jumps_is_eight :
  expected_jumps dog_trainers = 8 := by
  sorry

end expected_jumps_is_eight_l3799_379978


namespace equation_2x_squared_eq_1_is_quadratic_l3799_379953

/-- Definition of a quadratic equation -/
def is_quadratic_equation (f : ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

/-- The function representing 2x^2 = 1 -/
def f (x : ℝ) : ℝ := 2 * x^2 - 1

/-- Theorem: The equation 2x^2 = 1 is a quadratic equation -/
theorem equation_2x_squared_eq_1_is_quadratic : is_quadratic_equation f := by
  sorry

end equation_2x_squared_eq_1_is_quadratic_l3799_379953


namespace monomial_sum_l3799_379939

/-- If the sum of two monomials is still a monomial, then it equals -5xy^2 --/
theorem monomial_sum (a b : ℕ) : 
  (∃ (c : ℚ) (d e : ℕ), (-4 * 10^a * X^a * Y^2 + 35 * X * Y^(b-2) = c * X^d * Y^e)) →
  (-4 * 10^a * X^a * Y^2 + 35 * X * Y^(b-2) = -5 * X * Y^2) :=
by sorry

end monomial_sum_l3799_379939


namespace square_root_of_negative_two_fourth_power_l3799_379999

theorem square_root_of_negative_two_fourth_power :
  Real.sqrt ((-2)^4) = 4 ∨ Real.sqrt ((-2)^4) = -4 := by
  sorry

end square_root_of_negative_two_fourth_power_l3799_379999


namespace probability_no_three_consecutive_as_l3799_379900

/-- A string of length 6 using symbols A, B, and C -/
def String6ABC := Fin 6 → Fin 3

/-- Check if a string contains three consecutive A's -/
def hasThreeConsecutiveAs (s : String6ABC) : Prop :=
  ∃ i : Fin 4, s i = 0 ∧ s (i + 1) = 0 ∧ s (i + 2) = 0

/-- The total number of possible strings -/
def totalStrings : ℕ := 3^6

/-- The number of strings without three consecutive A's -/
def stringsWithoutThreeAs : ℕ := 680

/-- The probability of a random string not having three consecutive A's -/
def probabilityNoThreeAs : ℚ := stringsWithoutThreeAs / totalStrings

theorem probability_no_three_consecutive_as :
  probabilityNoThreeAs = 680 / 729 :=
sorry

end probability_no_three_consecutive_as_l3799_379900


namespace unique_solution_exponential_equation_l3799_379905

theorem unique_solution_exponential_equation :
  ∃! x : ℝ, (2003 : ℝ) ^ x + (2004 : ℝ) ^ x = (2005 : ℝ) ^ x := by
  sorry

end unique_solution_exponential_equation_l3799_379905


namespace movie_group_composition_l3799_379904

-- Define the ticket prices and group information
def adult_price : ℚ := 9.5
def child_price : ℚ := 6.5
def total_people : ℕ := 7
def total_paid : ℚ := 54.5

-- Define the theorem
theorem movie_group_composition :
  ∃ (adults : ℕ) (children : ℕ),
    adults + children = total_people ∧
    (adult_price * adults + child_price * children : ℚ) = total_paid ∧
    adults = 3 := by
  sorry

end movie_group_composition_l3799_379904


namespace ratio_solution_l3799_379937

theorem ratio_solution (a b : ℝ) (h1 : a ≠ b) 
  (h2 : a / b + (3 * a + 4 * b) / (b + 12 * a) = 2) : 
  a / b = (5 - Real.sqrt 19) / 6 ∨ a / b = (5 + Real.sqrt 19) / 6 := by
  sorry

end ratio_solution_l3799_379937


namespace quadratic_inequality_solution_set_l3799_379954

theorem quadratic_inequality_solution_set :
  {x : ℝ | -x^2 + 4*x - 3 > 0} = Set.Ioo 1 3 :=
by sorry

end quadratic_inequality_solution_set_l3799_379954


namespace max_log_product_l3799_379914

theorem max_log_product (a b : ℝ) (ha : a > 1) (hb : b > 1) (hab : a * b = 100) :
  Real.log a * Real.log b ≤ 1 ∧ 
  ∃ (a₀ b₀ : ℝ), a₀ > 1 ∧ b₀ > 1 ∧ a₀ * b₀ = 100 ∧ Real.log a₀ * Real.log b₀ = 1 :=
by sorry

end max_log_product_l3799_379914


namespace wire_cutting_problem_l3799_379902

theorem wire_cutting_problem (total_length : ℝ) (ratio : ℝ) :
  total_length = 35 →
  ratio = 2 / 5 →
  ∃ shorter_length longer_length : ℝ,
    shorter_length + longer_length = total_length ∧
    shorter_length = ratio * longer_length ∧
    shorter_length = 10 := by
  sorry

end wire_cutting_problem_l3799_379902


namespace quadratic_one_solution_l3799_379920

theorem quadratic_one_solution (q : ℝ) :
  (∃! x : ℝ, q * x^2 - 10 * x + 2 = 0) ↔ q = 12.5 := by
sorry

end quadratic_one_solution_l3799_379920


namespace coat_price_l3799_379984

/-- The final price of a coat after discounts and tax -/
def finalPrice (originalPrice discountOne discountTwo coupon salesTax : ℚ) : ℚ :=
  ((originalPrice * (1 - discountOne) * (1 - discountTwo) - coupon) * (1 + salesTax))

/-- Theorem stating the final price of the coat -/
theorem coat_price : 
  finalPrice 150 0.3 0.1 10 0.05 = 88.725 := by sorry

end coat_price_l3799_379984


namespace benny_apples_l3799_379934

theorem benny_apples (total : ℕ) (dan_apples : ℕ) (benny_apples : ℕ) :
  total = 11 → dan_apples = 9 → total = dan_apples + benny_apples → benny_apples = 2 := by
  sorry

end benny_apples_l3799_379934


namespace sum_of_four_rationals_l3799_379982

theorem sum_of_four_rationals (a₁ a₂ a₃ a₄ : ℚ) : 
  ({a₁ * a₂, a₁ * a₃, a₁ * a₄, a₂ * a₃, a₂ * a₄, a₃ * a₄} : Set ℚ) = 
    {-24, -2, -3/2, -1/8, 1, 3} → 
  a₁ + a₂ + a₃ + a₄ = 9/4 ∨ a₁ + a₂ + a₃ + a₄ = -9/4 := by
  sorry

end sum_of_four_rationals_l3799_379982


namespace board_cut_ratio_l3799_379960

/-- Proves that the ratio of the shorter piece to the longer piece is 1:1 for a 20-foot board cut into two pieces -/
theorem board_cut_ratio (total_length : ℝ) (shorter_length : ℝ) (longer_length : ℝ) :
  total_length = 20 →
  shorter_length = 8 →
  shorter_length = longer_length + 4 →
  shorter_length / longer_length = 1 := by
  sorry

end board_cut_ratio_l3799_379960


namespace megan_earnings_l3799_379951

/-- The amount of money Megan earned from selling necklaces -/
def money_earned (bead_necklaces gem_necklaces cost_per_necklace : ℕ) : ℕ :=
  (bead_necklaces + gem_necklaces) * cost_per_necklace

/-- Theorem stating that Megan earned 90 dollars from selling necklaces -/
theorem megan_earnings : money_earned 7 3 9 = 90 := by
  sorry

end megan_earnings_l3799_379951


namespace cos_330_degrees_l3799_379995

theorem cos_330_degrees : Real.cos (330 * π / 180) = Real.sqrt 3 / 2 := by
  sorry

end cos_330_degrees_l3799_379995


namespace sum_of_magnitudes_of_roots_l3799_379918

theorem sum_of_magnitudes_of_roots (z₁ z₂ z₃ z₄ : ℂ) : 
  (z₁^4 + 3*z₁^3 + 3*z₁^2 + 3*z₁ + 1 = 0) →
  (z₂^4 + 3*z₂^3 + 3*z₂^2 + 3*z₂ + 1 = 0) →
  (z₃^4 + 3*z₃^3 + 3*z₃^2 + 3*z₃ + 1 = 0) →
  (z₄^4 + 3*z₄^3 + 3*z₄^2 + 3*z₄ + 1 = 0) →
  Complex.abs z₁ + Complex.abs z₂ + Complex.abs z₃ + Complex.abs z₄ = (7 + Real.sqrt 5) / 2 := by
  sorry

end sum_of_magnitudes_of_roots_l3799_379918


namespace point_on_xOz_plane_l3799_379907

/-- A point in 3D Cartesian space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- The xOz plane in 3D Cartesian space -/
def xOzPlane : Set Point3D :=
  {p : Point3D | p.y = 0}

/-- The given point (1, 0, 4) -/
def givenPoint : Point3D :=
  ⟨1, 0, 4⟩

/-- Theorem: The given point (1, 0, 4) lies on the xOz plane -/
theorem point_on_xOz_plane : givenPoint ∈ xOzPlane := by
  sorry

end point_on_xOz_plane_l3799_379907


namespace fraction_zero_implies_x_equals_three_l3799_379985

theorem fraction_zero_implies_x_equals_three (x : ℝ) :
  (x^2 - 9) / (x + 3) = 0 ∧ x + 3 ≠ 0 → x = 3 := by
  sorry

end fraction_zero_implies_x_equals_three_l3799_379985


namespace total_legs_puppies_and_chicks_l3799_379966

/-- The number of legs for puppies and chicks -/
def total_legs (num_puppies num_chicks : ℕ) (puppy_legs chick_legs : ℕ) : ℕ :=
  num_puppies * puppy_legs + num_chicks * chick_legs

/-- Theorem: Given 3 puppies and 7 chicks, where puppies have 4 legs each and chicks have 2 legs each, the total number of legs is 26. -/
theorem total_legs_puppies_and_chicks :
  total_legs 3 7 4 2 = 26 := by
  sorry

end total_legs_puppies_and_chicks_l3799_379966


namespace fraction_equality_l3799_379991

theorem fraction_equality (a b : ℝ) (h : a ≠ -b) : (-a + b) / (-a - b) = (a - b) / (a + b) := by
  sorry

end fraction_equality_l3799_379991
