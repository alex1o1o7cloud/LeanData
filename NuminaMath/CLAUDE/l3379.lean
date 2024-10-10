import Mathlib

namespace min_max_values_min_value_three_variables_l3379_337955

-- Problem 1
theorem min_max_values (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) :
  (a + 1 / a^2)^2 + (b + 1 / b^2)^2 ≥ 25/4 ∧ (a + 1/a) * (b + 1/b) ≤ 25/4 := by
  sorry

-- Problem 2
theorem min_value_three_variables (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (habc : a + b + c = 1) :
  (a + 1/a)^3 + (b + 1/b)^3 + (c + 1/c)^3 ≥ 1000/9 := by
  sorry

end min_max_values_min_value_three_variables_l3379_337955


namespace consecutive_integers_product_sum_l3379_337904

theorem consecutive_integers_product_sum (n : ℤ) : 
  (n - 1) * n * (n + 1) = 336 → (n - 1) + n + (n + 1) = 21 := by
sorry

end consecutive_integers_product_sum_l3379_337904


namespace pecan_weight_in_mixture_l3379_337924

/-- A mixture of pecans and cashews -/
structure NutMixture where
  pecan_price : ℝ
  cashew_price : ℝ
  cashew_weight : ℝ
  total_weight : ℝ

/-- The amount of pecans in the mixture -/
def pecan_weight (m : NutMixture) : ℝ :=
  m.total_weight - m.cashew_weight

/-- Theorem stating the amount of pecans in the specific mixture -/
theorem pecan_weight_in_mixture (m : NutMixture) 
  (h1 : m.pecan_price = 5.60)
  (h2 : m.cashew_price = 3.50)
  (h3 : m.cashew_weight = 2)
  (h4 : m.total_weight = 3.33333333333) :
  pecan_weight m = 1.33333333333 := by
  sorry

#check pecan_weight_in_mixture

end pecan_weight_in_mixture_l3379_337924


namespace price_reduction_5_price_reduction_20_no_2200_profit_l3379_337960

/-- Represents the supermarket's sales model -/
structure SupermarketSales where
  initial_profit : ℕ
  initial_sales : ℕ
  price_reduction : ℕ
  sales_increase_rate : ℕ

/-- Calculates the new sales volume after a price reduction -/
def new_sales_volume (s : SupermarketSales) : ℕ :=
  s.initial_sales + s.price_reduction * s.sales_increase_rate

/-- Calculates the new profit per item after a price reduction -/
def new_profit_per_item (s : SupermarketSales) : ℤ :=
  s.initial_profit - s.price_reduction

/-- Calculates the total daily profit after a price reduction -/
def total_daily_profit (s : SupermarketSales) : ℤ :=
  (new_profit_per_item s) * (new_sales_volume s)

/-- Theorem: A price reduction of $5 results in 40 items sold and $1800 daily profit -/
theorem price_reduction_5 (s : SupermarketSales) 
  (h1 : s.initial_profit = 50)
  (h2 : s.initial_sales = 30)
  (h3 : s.sales_increase_rate = 2)
  (h4 : s.price_reduction = 5) :
  new_sales_volume s = 40 ∧ total_daily_profit s = 1800 := by sorry

/-- Theorem: A price reduction of $20 results in $2100 daily profit -/
theorem price_reduction_20 (s : SupermarketSales)
  (h1 : s.initial_profit = 50)
  (h2 : s.initial_sales = 30)
  (h3 : s.sales_increase_rate = 2)
  (h4 : s.price_reduction = 20) :
  total_daily_profit s = 2100 := by sorry

/-- Theorem: There is no price reduction that results in $2200 daily profit -/
theorem no_2200_profit (s : SupermarketSales)
  (h1 : s.initial_profit = 50)
  (h2 : s.initial_sales = 30)
  (h3 : s.sales_increase_rate = 2) :
  ∀ (x : ℕ), total_daily_profit { s with price_reduction := x } ≠ 2200 := by sorry

end price_reduction_5_price_reduction_20_no_2200_profit_l3379_337960


namespace cookie_sales_proof_l3379_337936

theorem cookie_sales_proof (total_value : ℝ) (choc_price plain_price : ℝ) (plain_boxes : ℝ) :
  total_value = 1586.25 →
  choc_price = 1.25 →
  plain_price = 0.75 →
  plain_boxes = 793.125 →
  ∃ (choc_boxes : ℝ), 
    choc_price * choc_boxes + plain_price * plain_boxes = total_value ∧
    choc_boxes + plain_boxes = 1586.25 :=
by sorry

end cookie_sales_proof_l3379_337936


namespace correct_propositions_l3379_337973

-- Define the propositions
def proposition1 : Prop := sorry
def proposition2 : Prop := sorry
def proposition3 : Prop := sorry
def proposition4 : Prop := sorry

-- Define a function to check if a proposition is correct
def is_correct (p : Prop) : Prop := sorry

-- Theorem statement
theorem correct_propositions :
  is_correct proposition2 ∧ 
  is_correct proposition3 ∧ 
  ¬is_correct proposition1 ∧ 
  ¬is_correct proposition4 :=
sorry

end correct_propositions_l3379_337973


namespace ellipse_and_line_theorem_l3379_337980

-- Define the ellipse C
def ellipse_C (x y : ℝ) : Prop := x^2 / 4 + y^2 = 1

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4 * x

-- Define the line l
def line_l (x y : ℝ) (m : ℝ) : Prop := x = m * y + 1

-- Define perpendicularity of vectors
def perpendicular (x1 y1 x2 y2 : ℝ) : Prop := x1 * x2 + y1 * y2 = 0

theorem ellipse_and_line_theorem :
  -- Conditions
  (ellipse_C (-2) 0) ∧
  (ellipse_C (Real.sqrt 2) ((Real.sqrt 2) / 2)) ∧
  -- Existence of intersection points M and N
  ∃ (x1 y1 x2 y2 : ℝ),
    (ellipse_C x1 y1) ∧
    (ellipse_C x2 y2) ∧
    (∃ (m : ℝ), line_l x1 y1 m ∧ line_l x2 y2 m) ∧
    -- M and N are distinct
    (x1 ≠ x2 ∨ y1 ≠ y2) ∧
    -- OM ⊥ ON
    (perpendicular x1 y1 x2 y2) →
  -- Conclusion
  ∃ (m : ℝ), (m = 1/2 ∨ m = -1/2) ∧ line_l 1 0 m := by
  sorry

end ellipse_and_line_theorem_l3379_337980


namespace burger_cost_proof_l3379_337962

/-- The cost of Uri's purchase in cents -/
def uri_cost : ℕ := 450

/-- The cost of Gen's purchase in cents -/
def gen_cost : ℕ := 480

/-- The number of burgers Uri bought -/
def uri_burgers : ℕ := 3

/-- The number of sodas Uri bought -/
def uri_sodas : ℕ := 2

/-- The number of burgers Gen bought -/
def gen_burgers : ℕ := 2

/-- The number of sodas Gen bought -/
def gen_sodas : ℕ := 3

/-- The cost of a burger in cents -/
def burger_cost : ℕ := 78

theorem burger_cost_proof :
  ∃ (soda_cost : ℕ),
    uri_burgers * burger_cost + uri_sodas * soda_cost = uri_cost ∧
    gen_burgers * burger_cost + gen_sodas * soda_cost = gen_cost :=
by sorry

end burger_cost_proof_l3379_337962


namespace arithmetic_sequence_property_l3379_337969

/-- An arithmetic sequence is a sequence where the difference between
    any two consecutive terms is constant. -/
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- Given an arithmetic sequence a_n where a_1 + 3a_8 + a_15 = 60,
    prove that 2a_9 - a_10 = 12 -/
theorem arithmetic_sequence_property
  (a : ℕ → ℝ)
  (h_arith : is_arithmetic_sequence a)
  (h_sum : a 1 + 3 * a 8 + a 15 = 60) :
  2 * a 9 - a 10 = 12 :=
sorry

end arithmetic_sequence_property_l3379_337969


namespace electric_bicycle_sales_l3379_337986

theorem electric_bicycle_sales (model_a_first_quarter : Real) 
  (model_bc_first_quarter : Real) (a : Real) :
  model_a_first_quarter = 0.56 ∧ 
  model_bc_first_quarter = 1 - model_a_first_quarter ∧
  0.56 * (1 + 0.23) + (1 - 0.56) * (1 - a / 100) = 1 + 0.12 →
  a = 2 := by
sorry

end electric_bicycle_sales_l3379_337986


namespace parabola_h_values_l3379_337915

/-- Represents a parabola of the form y = -(x - h)² -/
def Parabola (h : ℝ) : ℝ → ℝ := fun x ↦ -((x - h)^2)

/-- The domain of x values -/
def Domain : Set ℝ := {x | 3 ≤ x ∧ x ≤ 7}

theorem parabola_h_values (h : ℝ) :
  (∀ x ∈ Domain, Parabola h x ≤ -1) ∧
  (∃ x ∈ Domain, Parabola h x = -1) →
  h = 2 ∨ h = 8 := by sorry

end parabola_h_values_l3379_337915


namespace roots_difference_implies_k_value_l3379_337975

theorem roots_difference_implies_k_value (k : ℝ) : 
  (∃ r s : ℝ, 
    (r^2 + k*r + 10 = 0 ∧ s^2 + k*s + 10 = 0) ∧ 
    ((r+4)^2 - k*(r+4) + 10 = 0 ∧ (s+4)^2 - k*(s+4) + 10 = 0)) → 
  k = 4 := by
sorry

end roots_difference_implies_k_value_l3379_337975


namespace tan_alpha_value_l3379_337947

theorem tan_alpha_value (α : Real) (h : Real.tan (α / 2) = 2) :
  Real.tan α = -4/3 := by
  sorry

end tan_alpha_value_l3379_337947


namespace kelly_cheese_packages_l3379_337984

-- Define the problem parameters
def days_per_week : ℕ := 5
def oldest_child_cheese_per_day : ℕ := 2
def youngest_child_cheese_per_day : ℕ := 1
def cheese_per_package : ℕ := 30
def weeks : ℕ := 4

-- Define the theorem
theorem kelly_cheese_packages :
  (days_per_week * (oldest_child_cheese_per_day + youngest_child_cheese_per_day) * weeks + cheese_per_package - 1) / cheese_per_package = 2 :=
by sorry

end kelly_cheese_packages_l3379_337984


namespace quadratic_satisfies_conditions_l3379_337989

-- Define the quadratic function
def f (x : ℝ) : ℝ := -2 * x^2 + 12 * x - 10

-- State the theorem
theorem quadratic_satisfies_conditions : 
  f 1 = 0 ∧ f 5 = 0 ∧ f 3 = 8 := by sorry

end quadratic_satisfies_conditions_l3379_337989


namespace smallest_number_of_cars_l3379_337913

theorem smallest_number_of_cars (N : ℕ) : 
  N > 2 ∧ 
  N % 5 = 2 ∧ 
  N % 6 = 2 ∧ 
  N % 7 = 2 → 
  (∀ m : ℕ, m > 2 ∧ m % 5 = 2 ∧ m % 6 = 2 ∧ m % 7 = 2 → m ≥ N) →
  N = 212 :=
by sorry

end smallest_number_of_cars_l3379_337913


namespace distinct_reals_integer_combination_l3379_337971

theorem distinct_reals_integer_combination (x y : ℝ) (h : x ≠ y) :
  ∃ (m n : ℤ), m * x + n * y > 0 ∧ n * x + m * y < 0 := by
  sorry

end distinct_reals_integer_combination_l3379_337971


namespace problem_statement_l3379_337948

theorem problem_statement (r p q : ℝ) 
  (hr : r > 0) 
  (hpq : p * q ≠ 0) 
  (hineq : p^2 * r > q^2 * r) : 
  p^2 > q^2 := by
  sorry

end problem_statement_l3379_337948


namespace xy_greater_than_xz_l3379_337968

theorem xy_greater_than_xz (x y z : ℝ) 
  (h1 : x > y) (h2 : y > z) (h3 : x + y + z = 1) : x * y > x * z := by
  sorry

end xy_greater_than_xz_l3379_337968


namespace percentage_difference_l3379_337972

theorem percentage_difference (C : ℝ) (A B : ℝ) 
  (hA : A = 0.7 * C) 
  (hB : B = 0.63 * C) : 
  (A - B) / A = 0.1 := by
sorry

end percentage_difference_l3379_337972


namespace black_area_after_three_changes_l3379_337909

/-- Represents the fraction of black area remaining after a single change -/
def blackAreaAfterOneChange (initialBlackArea : ℚ) : ℚ :=
  initialBlackArea * (5/6) * (9/10)

/-- Calculates the fraction of black area remaining after n changes -/
def blackAreaAfterNChanges (n : ℕ) : ℚ :=
  match n with
  | 0 => 1
  | n+1 => blackAreaAfterOneChange (blackAreaAfterNChanges n)

/-- The main theorem stating that after 3 changes, 27/64 of the original area remains black -/
theorem black_area_after_three_changes :
  blackAreaAfterNChanges 3 = 27/64 := by
  sorry

end black_area_after_three_changes_l3379_337909


namespace line_through_point_parallel_to_original_l3379_337923

-- Define a line in 2D space
structure Line2D where
  f : ℝ → ℝ → ℝ
  is_line : ∀ x y, f x y = 0 ↔ (∃ a b c : ℝ, a * x + b * y + c = 0 ∧ (a ≠ 0 ∨ b ≠ 0))

-- Define a point in 2D space
structure Point2D where
  x : ℝ
  y : ℝ

-- Define what it means for a point to be on a line
def on_line (p : Point2D) (l : Line2D) : Prop :=
  l.f p.x p.y = 0

-- Define what it means for a point to not be on a line
def not_on_line (p : Point2D) (l : Line2D) : Prop :=
  l.f p.x p.y ≠ 0

-- Define parallel lines
def parallel (l1 l2 : Line2D) : Prop :=
  ∃ k : ℝ, k ≠ 0 ∧ ∀ x y, l1.f x y = k * l2.f x y

-- Theorem statement
theorem line_through_point_parallel_to_original 
  (l : Line2D) (p1 p2 : Point2D) 
  (h1 : on_line p1 l) 
  (h2 : not_on_line p2 l) : 
  ∃ l2 : Line2D, 
    (∀ x y, l2.f x y = l.f x y - l.f p1.x p1.y - l.f p2.x p2.y) ∧ 
    on_line p2 l2 ∧ 
    parallel l l2 := by
  sorry

end line_through_point_parallel_to_original_l3379_337923


namespace largest_two_digit_number_with_one_l3379_337928

def digits : List Nat := [1, 5, 6, 9]

def is_valid_number (n : Nat) : Prop :=
  n ≥ 10 ∧ n < 100 ∧
  n % 10 = 1 ∧
  (n / 10) ∈ digits ∧
  1 ∈ digits

theorem largest_two_digit_number_with_one :
  ∀ n : Nat, is_valid_number n → n ≤ 91 :=
by sorry

end largest_two_digit_number_with_one_l3379_337928


namespace range_of_a_l3379_337908

theorem range_of_a (a : ℝ) : 
  (¬∃ x : ℝ, |x - a| + |x + 1| ≤ 2) ↔ (a < -3 ∨ a > 1) := by
  sorry

end range_of_a_l3379_337908


namespace button_numbers_l3379_337903

theorem button_numbers (x y : ℕ) 
  (h1 : y - x = 480) 
  (h2 : y = 4 * x + 30) : 
  y = 630 := by
  sorry

end button_numbers_l3379_337903


namespace integer_product_condition_l3379_337991

theorem integer_product_condition (a : ℝ) : 
  (∀ n : ℕ, ∃ m : ℤ, a * n * (n + 2) * (n + 4) = m) ↔ 
  (∃ k : ℤ, a = k / 3) := by
sorry

end integer_product_condition_l3379_337991


namespace train_speed_calculation_l3379_337966

/-- Calculates the speed of a train crossing a bridge -/
theorem train_speed_calculation (train_length bridge_length : ℝ) (crossing_time : ℝ) 
  (h1 : train_length = 110)
  (h2 : bridge_length = 290)
  (h3 : crossing_time = 23.998080153587715) :
  (((train_length + bridge_length) / crossing_time) * 3.6) = 60 := by
  sorry

end train_speed_calculation_l3379_337966


namespace prove_total_workers_l3379_337963

def total_workers : ℕ := 9
def other_workers : ℕ := 7
def chosen_workers : ℕ := 2

theorem prove_total_workers :
  (total_workers = other_workers + 2) →
  (Nat.choose total_workers chosen_workers = 36) →
  (1 / (Nat.choose total_workers chosen_workers : ℚ) = 1 / 36) →
  total_workers = 9 := by
sorry

end prove_total_workers_l3379_337963


namespace subcommittee_count_l3379_337944

theorem subcommittee_count (total_members : ℕ) (subcommittees_per_member : ℕ) (members_per_subcommittee : ℕ) :
  total_members = 360 →
  subcommittees_per_member = 3 →
  members_per_subcommittee = 6 →
  (total_members * subcommittees_per_member) / members_per_subcommittee = 180 :=
by sorry

end subcommittee_count_l3379_337944


namespace proportional_function_k_l3379_337998

theorem proportional_function_k (k : ℝ) (h1 : k ≠ 0) (h2 : -5 = k * 3) : k = -5/3 := by
  sorry

end proportional_function_k_l3379_337998


namespace complex_equation_solution_l3379_337982

theorem complex_equation_solution (z : ℂ) : (3 + Complex.I) * z = 4 - 2 * Complex.I → z = 1 - Complex.I := by
  sorry

end complex_equation_solution_l3379_337982


namespace equation_solutions_l3379_337925

theorem equation_solutions : 
  (∃ x : ℝ, x^2 - 2*x - 7 = 0 ↔ x = 1 + 2*Real.sqrt 2 ∨ x = 1 - 2*Real.sqrt 2) ∧
  (∃ x : ℝ, 3*(x-2)^2 = x*(x-2) ↔ x = 2 ∨ x = 3) := by
  sorry

end equation_solutions_l3379_337925


namespace angle_A_is_pi_third_max_perimeter_is_3_sqrt_3_l3379_337935

/-- Triangle ABC with angles A, B, C and opposite sides a, b, c -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- Vectors m and n are orthogonal -/
def vectors_orthogonal (t : Triangle) : Prop :=
  t.a * (Real.cos t.C + Real.sqrt 3 * Real.sin t.C) + (t.b + t.c) * (-1) = 0

/-- Theorem: If vectors are orthogonal, then angle A is π/3 -/
theorem angle_A_is_pi_third (t : Triangle) (h : vectors_orthogonal t) : t.A = π / 3 := by
  sorry

/-- Maximum perimeter when a = √3 -/
def max_perimeter (t : Triangle) : ℝ :=
  t.a + t.b + t.c

/-- Theorem: When a = √3, the maximum perimeter is 3√3 -/
theorem max_perimeter_is_3_sqrt_3 (t : Triangle) (h : t.a = Real.sqrt 3) :
  max_perimeter t ≤ 3 * Real.sqrt 3 := by
  sorry

end angle_A_is_pi_third_max_perimeter_is_3_sqrt_3_l3379_337935


namespace stream_speed_ratio_l3379_337932

/-- Given a boat and a stream where:
  1. It takes twice as long to row against the stream as with it for the same distance.
  2. The speed of the boat in still water is three times the speed of the stream.
  This theorem proves that the speed of the stream is one-third of the speed of the boat in still water. -/
theorem stream_speed_ratio (B S : ℝ) (h1 : B = 3 * S) 
  (h2 : (1 : ℝ) / (B - S) = 2 * (1 / (B + S))) : S / B = 1 / 3 := by
  sorry

end stream_speed_ratio_l3379_337932


namespace p_and_q_properties_l3379_337953

-- Define proposition p
def p : Prop := ∃ x : ℝ, Real.sqrt x = Real.sqrt (2 * x + 1)

-- Define proposition q
def q : Prop := ∀ x : ℝ, x > 0 → x^2 < x^3

-- Theorem stating the properties of p and q
theorem p_and_q_properties :
  (∃ x : ℝ, Real.sqrt x = Real.sqrt (2 * x + 1)) ∧  -- p is existential
  ¬p ∧                                             -- p is false
  (∀ x : ℝ, x > 0 → x^2 < x^3) ∧                   -- q is universal
  ¬q                                               -- q is false
  := by sorry

end p_and_q_properties_l3379_337953


namespace prob_at_least_two_successes_l3379_337997

/-- The probability of success in a single trial -/
def p : ℝ := 0.6

/-- The number of trials -/
def n : ℕ := 3

/-- The probability of exactly k successes in n trials -/
def binomialProb (k : ℕ) : ℝ :=
  (n.choose k) * p^k * (1 - p)^(n - k)

/-- The probability of at least 2 successes in 3 trials -/
theorem prob_at_least_two_successes : 
  binomialProb 2 + binomialProb 3 = 81/125 := by sorry

end prob_at_least_two_successes_l3379_337997


namespace plane_equation_correct_l3379_337929

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents the coefficients of a plane equation Ax + By + Cz + D = 0 -/
structure PlaneCoefficients where
  A : ℤ
  B : ℤ
  C : ℤ
  D : ℤ

/-- The foot of the perpendicular from the origin to the plane -/
def footPoint : Point3D := ⟨10, -5, 2⟩

/-- The coefficients of the plane equation -/
def planeCoeffs : PlaneCoefficients := ⟨10, -5, 2, -129⟩

/-- Checks if a point satisfies the plane equation -/
def satisfiesPlaneEquation (p : Point3D) (c : PlaneCoefficients) : Prop :=
  c.A * p.x + c.B * p.y + c.C * p.z + c.D = 0

/-- Checks if a vector is perpendicular to another vector -/
def isPerpendicular (v1 v2 : Point3D) : Prop :=
  v1.x * v2.x + v1.y * v2.y + v1.z * v2.z = 0

/-- Theorem stating that the given plane equation is correct -/
theorem plane_equation_correct :
  satisfiesPlaneEquation footPoint planeCoeffs ∧
  isPerpendicular footPoint ⟨planeCoeffs.A, planeCoeffs.B, planeCoeffs.C⟩ ∧
  planeCoeffs.A > 0 ∧
  Nat.gcd (Nat.gcd (Int.natAbs planeCoeffs.A) (Int.natAbs planeCoeffs.B))
          (Nat.gcd (Int.natAbs planeCoeffs.C) (Int.natAbs planeCoeffs.D)) = 1 :=
by sorry

end plane_equation_correct_l3379_337929


namespace cos_three_pi_four_plus_two_alpha_l3379_337994

theorem cos_three_pi_four_plus_two_alpha (α : ℝ) 
  (h : Real.cos (π / 8 - α) = 1 / 6) : 
  Real.cos (3 * π / 4 + 2 * α) = 17 / 18 := by
  sorry

end cos_three_pi_four_plus_two_alpha_l3379_337994


namespace license_plate_palindrome_probability_l3379_337958

/-- The number of possible digits (0-9) -/
def num_digits : ℕ := 10

/-- The number of possible letters (A-Z) -/
def num_letters : ℕ := 26

/-- The probability of a four-digit palindrome -/
def prob_digit_palindrome : ℚ := 1 / 100

/-- The probability of a four-letter palindrome with at least one 'X' -/
def prob_letter_palindrome : ℚ := 1 / 8784

/-- The probability of both four-digit and four-letter palindromes occurring -/
def prob_both_palindromes : ℚ := prob_digit_palindrome * prob_letter_palindrome

/-- The probability of at least one palindrome in a license plate -/
def prob_at_least_one_palindrome : ℚ := 
  prob_digit_palindrome + prob_letter_palindrome - prob_both_palindromes

theorem license_plate_palindrome_probability :
  prob_at_least_one_palindrome = 8883 / 878400 := by
  sorry

end license_plate_palindrome_probability_l3379_337958


namespace corrected_mean_calculation_l3379_337919

theorem corrected_mean_calculation (n : ℕ) (original_mean : ℚ) (incorrect_value correct_value : ℚ) :
  n = 50 →
  original_mean = 36 →
  incorrect_value = 23 →
  correct_value = 48 →
  let original_sum := n * original_mean
  let difference := correct_value - incorrect_value
  let corrected_sum := original_sum + difference
  corrected_sum / n = 36.5 := by
sorry

end corrected_mean_calculation_l3379_337919


namespace parallel_vectors_k_l3379_337938

/-- Two 2D vectors are parallel if their cross product is zero -/
def parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 = a.2 * b.1

theorem parallel_vectors_k (k : ℝ) :
  let a : ℝ × ℝ := (2*k + 2, 4)
  let b : ℝ × ℝ := (k + 1, 8)
  parallel a b → k = -1 := by
sorry

end parallel_vectors_k_l3379_337938


namespace function_equality_l3379_337993

noncomputable def f (x : ℝ) : ℝ := 
  if x ≤ 1 then -x^2 + 1 else x - 1

theorem function_equality (a : ℝ) : f (a + 1) = f a ↔ a = -1/2 ∨ a = (-1 + Real.sqrt 5) / 2 := by
  sorry

end function_equality_l3379_337993


namespace remainder_of_geometric_sum_l3379_337900

def geometric_sum (r : ℕ) (n : ℕ) : ℕ :=
  (r^(n + 1) - 1) / (r - 1)

theorem remainder_of_geometric_sum :
  (geometric_sum 7 2004) % 1000 = 801 := by
  sorry

end remainder_of_geometric_sum_l3379_337900


namespace inverse_variation_problem_l3379_337945

/-- Given that x and y are positive real numbers, 3x² and y vary inversely,
    y = 18 when x = 3, and y = 2400, prove that x = 9√6 / 85. -/
theorem inverse_variation_problem (x y : ℝ) (hx : x > 0) (hy : y > 0)
    (h_inverse : ∃ k, k > 0 ∧ ∀ x' y', x' > 0 → y' > 0 → 3 * x'^2 * y' = k)
    (h_initial : 3 * 3^2 * 18 = 3 * x^2 * 2400) :
    x = 9 * Real.sqrt 6 / 85 := by
  sorry

end inverse_variation_problem_l3379_337945


namespace largest_number_l3379_337921

theorem largest_number (a b c d : ℝ) (ha : a = 1) (hb : b = -2) (hc : c = 0) (hd : d = Real.sqrt 3) :
  d = max a (max b (max c d)) :=
by sorry

end largest_number_l3379_337921


namespace volleyball_team_lineups_l3379_337950

def team_size : ℕ := 16
def quadruplet_size : ℕ := 4
def starter_size : ℕ := 6

def valid_lineups : ℕ := Nat.choose team_size starter_size - Nat.choose (team_size - quadruplet_size) (starter_size - quadruplet_size)

theorem volleyball_team_lineups : valid_lineups = 7942 := by
  sorry

end volleyball_team_lineups_l3379_337950


namespace intersection_of_A_and_B_l3379_337933

def A : Set ℝ := {x | x < 2}
def B : Set ℝ := {-1, 0, 1, 2}

theorem intersection_of_A_and_B : A ∩ B = {-1, 0, 1} := by
  sorry

end intersection_of_A_and_B_l3379_337933


namespace ants_in_park_l3379_337917

-- Define the dimensions of the park in meters
def park_width : ℝ := 100
def park_length : ℝ := 130

-- Define the ant density per square centimeter
def ants_per_sq_cm : ℝ := 1.2

-- Define the conversion factor from meters to centimeters
def cm_per_meter : ℝ := 100

-- Theorem statement
theorem ants_in_park :
  let park_area_sq_cm := park_width * park_length * cm_per_meter^2
  let total_ants := park_area_sq_cm * ants_per_sq_cm
  total_ants = 156000000 := by
  sorry

end ants_in_park_l3379_337917


namespace cube_volume_from_face_diagonal_l3379_337970

theorem cube_volume_from_face_diagonal (d : ℝ) (h : d = 6 * Real.sqrt 2) :
  let s := d / Real.sqrt 2
  s ^ 3 = 216 := by
  sorry

end cube_volume_from_face_diagonal_l3379_337970


namespace n_is_composite_l3379_337914

/-- The number of zeros in the given number -/
def num_zeros : ℕ := 2^1974 + 2^1000 - 1

/-- The number to be proven composite -/
def n : ℕ := 10^(num_zeros + 1) + 1

/-- Theorem stating that n is composite -/
theorem n_is_composite : ∃ (a b : ℕ), a > 1 ∧ b > 1 ∧ n = a * b :=
sorry

end n_is_composite_l3379_337914


namespace triangle_side_length_l3379_337934

/-- Given a triangle ABC with side lengths and altitude, prove BC = 4 -/
theorem triangle_side_length (A B C : ℝ × ℝ) (h : ℝ) : 
  let d := (fun (P Q : ℝ × ℝ) => Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2))
  (d A B = 2 * Real.sqrt 3) →
  (d A C = 2) →
  (h = Real.sqrt 3) →
  (h * d B C = 2 * Real.sqrt 3 * 2) →
  d B C = 4 := by
sorry

end triangle_side_length_l3379_337934


namespace tournament_points_l3379_337918

-- Define the type for teams
inductive Team : Type
  | A | B | C | D | E

-- Define the function for points
def points : Team → ℕ
  | Team.A => 7
  | Team.B => 6
  | Team.C => 4
  | Team.D => 5
  | Team.E => 2

-- Define the properties of the tournament
axiom different_points : ∀ t1 t2 : Team, t1 ≠ t2 → points t1 ≠ points t2
axiom a_most_points : ∀ t : Team, t ≠ Team.A → points Team.A > points t
axiom b_beat_a : points Team.B > points Team.A
axiom b_no_loss : ∀ t : Team, t ≠ Team.B → points Team.B ≥ points t
axiom c_no_loss : ∀ t : Team, t ≠ Team.C → points Team.C ≥ points t
axiom d_more_than_c : points Team.D > points Team.C

-- Theorem to prove
theorem tournament_points : 
  (points Team.A = 7 ∧ 
   points Team.B = 6 ∧ 
   points Team.C = 4 ∧ 
   points Team.D = 5 ∧ 
   points Team.E = 2) := by
  sorry

end tournament_points_l3379_337918


namespace green_chips_count_l3379_337979

theorem green_chips_count (total : ℕ) (blue : ℕ) (white : ℕ) (green : ℕ) : 
  blue = 3 →
  blue = (10 * total) / 100 →
  white = (50 * total) / 100 →
  green = total - blue - white →
  green = 12 := by
sorry

end green_chips_count_l3379_337979


namespace negative_product_sum_l3379_337951

theorem negative_product_sum (a b : ℚ) (h1 : a * b > 0) (h2 : a + b < 0) : a < 0 ∧ b < 0 := by
  sorry

end negative_product_sum_l3379_337951


namespace student_team_signup_l3379_337911

/-- The number of students --/
def num_students : ℕ := 4

/-- The number of sports teams --/
def num_teams : ℕ := 3

/-- The function that calculates the number of ways students can sign up for teams --/
def ways_to_sign_up (students : ℕ) (teams : ℕ) : ℕ := teams ^ students

/-- Theorem stating that there are 81 ways for 4 students to sign up for 3 teams --/
theorem student_team_signup :
  ways_to_sign_up num_students num_teams = 81 := by
  sorry

end student_team_signup_l3379_337911


namespace savings_ratio_l3379_337978

/-- Proves that the ratio of Nora's savings to Lulu's savings is 5:1 given the problem conditions -/
theorem savings_ratio (debt : ℝ) (lulu_savings : ℝ) (remaining_per_person : ℝ)
  (h1 : debt = 40)
  (h2 : lulu_savings = 6)
  (h3 : remaining_per_person = 2)
  (h4 : ∃ (x : ℝ), x > 0 ∧ ∃ (tamara_savings : ℝ),
    x * lulu_savings = 3 * tamara_savings ∧
    debt + 3 * remaining_per_person = lulu_savings + x * lulu_savings + tamara_savings) :
  ∃ (nora_savings : ℝ), nora_savings / lulu_savings = 5 := by
  sorry

end savings_ratio_l3379_337978


namespace badminton_purchase_costs_l3379_337981

/-- Represents the cost calculation for badminton equipment purchases --/
structure BadmintonPurchase where
  num_rackets : ℕ
  num_shuttlecocks : ℕ
  racket_price : ℕ
  shuttlecock_price : ℕ
  store_a_promotion : Bool
  store_b_discount : ℚ

/-- Calculates the cost at Store A --/
def cost_store_a (p : BadmintonPurchase) : ℕ :=
  p.num_rackets * p.racket_price + (p.num_shuttlecocks - p.num_rackets) * p.shuttlecock_price

/-- Calculates the cost at Store B --/
def cost_store_b (p : BadmintonPurchase) : ℚ :=
  ((p.num_rackets * p.racket_price + p.num_shuttlecocks * p.shuttlecock_price : ℚ) * (1 - p.store_b_discount))

/-- The main theorem to prove --/
theorem badminton_purchase_costs 
  (x : ℕ) 
  (h : x > 16) :
  let p : BadmintonPurchase := {
    num_rackets := 16,
    num_shuttlecocks := x,
    racket_price := 150,
    shuttlecock_price := 40,
    store_a_promotion := true,
    store_b_discount := 1/5
  }
  cost_store_a p = 1760 + 40 * x ∧ 
  cost_store_b p = 1920 + 32 * x := by
  sorry

#check badminton_purchase_costs

end badminton_purchase_costs_l3379_337981


namespace other_communities_count_l3379_337943

theorem other_communities_count (total_boys : ℕ) (muslim_percent : ℚ) (hindu_percent : ℚ) (sikh_percent : ℚ) :
  total_boys = 850 →
  muslim_percent = 34/100 →
  hindu_percent = 28/100 →
  sikh_percent = 10/100 →
  (total_boys : ℚ) * (1 - (muslim_percent + hindu_percent + sikh_percent)) = 238 := by
  sorry

end other_communities_count_l3379_337943


namespace triangle_abc_properties_l3379_337941

theorem triangle_abc_properties (A : Real) (h : Real.sin A + Real.cos A = 1/5) :
  (Real.sin A * Real.cos A = -12/25) ∧
  (π/2 < A ∧ A < π) ∧
  (Real.tan A = -4/3) := by
  sorry

end triangle_abc_properties_l3379_337941


namespace no_solution_l3379_337956

def connection (a b : ℕ+) : ℚ :=
  (Nat.lcm a.val b.val : ℚ) / (a.val * b.val)

theorem no_solution : ¬ ∃ y : ℕ+, y.val < 50 ∧ connection y 13 = 3/5 := by
  sorry

end no_solution_l3379_337956


namespace original_triangle_area_l3379_337930

-- Define the properties of the triangle
def is_oblique_projection (original : Real → Real → Real → Real) 
  (projected : Real → Real → Real → Real) : Prop := sorry

def is_equilateral (triangle : Real → Real → Real → Real) : Prop := sorry

def side_length (triangle : Real → Real → Real → Real) : Real := sorry

def area (triangle : Real → Real → Real → Real) : Real := sorry

-- Theorem statement
theorem original_triangle_area 
  (original projected : Real → Real → Real → Real) :
  is_oblique_projection original projected →
  is_equilateral projected →
  side_length projected = 1 →
  (area projected) / (area original) = Real.sqrt 2 / 4 →
  area original = Real.sqrt 6 / 2 := by sorry

end original_triangle_area_l3379_337930


namespace logical_equivalence_l3379_337987

theorem logical_equivalence (P Q R : Prop) :
  (¬P ∧ ¬Q → R) ↔ (P ∨ Q ∨ R) := by sorry

end logical_equivalence_l3379_337987


namespace relative_error_comparison_l3379_337965

theorem relative_error_comparison :
  let line1_length : ℝ := 25
  let line1_error : ℝ := 0.05
  let line2_length : ℝ := 125
  let line2_error : ℝ := 0.25
  let relative_error1 : ℝ := line1_error / line1_length
  let relative_error2 : ℝ := line2_error / line2_length
  relative_error1 = relative_error2 :=
by sorry

end relative_error_comparison_l3379_337965


namespace cube_volume_from_space_diagonal_l3379_337916

/-- The volume of a cube given its space diagonal -/
theorem cube_volume_from_space_diagonal (d : ℝ) (h : d = 6 * Real.sqrt 3) :
  (d / Real.sqrt 3) ^ 3 = 216 := by
  sorry

end cube_volume_from_space_diagonal_l3379_337916


namespace union_of_A_and_B_l3379_337920

open Set

def A : Set ℝ := {x : ℝ | -1 < x ∧ x < 3}
def B : Set ℝ := {x : ℝ | 2 ≤ x ∧ x ≤ 4}

theorem union_of_A_and_B : A ∪ B = {x : ℝ | -1 < x ∧ x ≤ 4} := by sorry

end union_of_A_and_B_l3379_337920


namespace sequence_problem_l3379_337940

theorem sequence_problem (a b : ℕ → ℝ) : 
  (∀ n, a (n + 1) - a n = a 2 - a 1) →  -- arithmetic sequence condition
  (∀ n, b (n + 1) / b n = b 2 / b 1) →  -- geometric sequence condition
  a 1 + a 2 = 10 →
  a 4 - a 3 = 2 →
  b 2 = a 3 →
  b 3 = a 7 →
  b 5 = 64 := by
sorry

end sequence_problem_l3379_337940


namespace sum_y_z_is_twice_x_l3379_337901

theorem sum_y_z_is_twice_x (x y z : ℝ) 
  (h1 : 0.6 * (x - y) = 0.3 * (x + y)) 
  (h2 : 0.4 * (x + z) = 0.2 * (y + z)) : 
  (y + z) / x = 2 := by
sorry

end sum_y_z_is_twice_x_l3379_337901


namespace total_players_in_ground_l3379_337931

theorem total_players_in_ground (cricket_players hockey_players football_players softball_players : ℕ) 
  (h1 : cricket_players = 22)
  (h2 : hockey_players = 15)
  (h3 : football_players = 21)
  (h4 : softball_players = 19) :
  cricket_players + hockey_players + football_players + softball_players = 77 := by
  sorry

end total_players_in_ground_l3379_337931


namespace round_trip_time_l3379_337959

/-- Calculates the time for a round trip given boat speed, current speed, and distance -/
theorem round_trip_time (boat_speed : ℝ) (current_speed : ℝ) (distance : ℝ) :
  boat_speed = 18 ∧ current_speed = 4 ∧ distance = 85.56 →
  (distance / (boat_speed - current_speed) + distance / (boat_speed + current_speed)) = 10 := by
  sorry

end round_trip_time_l3379_337959


namespace tan_45_degrees_equals_one_l3379_337937

theorem tan_45_degrees_equals_one : Real.tan (π / 4) = 1 := by sorry

end tan_45_degrees_equals_one_l3379_337937


namespace quadratic_inequality_solution_l3379_337927

theorem quadratic_inequality_solution (x : ℝ) :
  (2 * x^2 - 5 * x + 2 > 0) ↔ (x < (1 : ℝ) / 2 ∨ x > 2) :=
by sorry

end quadratic_inequality_solution_l3379_337927


namespace mersenne_prime_condition_l3379_337996

theorem mersenne_prime_condition (a n : ℕ) : 
  a > 1 → n > 1 → Nat.Prime (a^n - 1) → a = 2 ∧ Nat.Prime n :=
sorry

end mersenne_prime_condition_l3379_337996


namespace quadratic_interval_l3379_337922

theorem quadratic_interval (x : ℝ) : 
  (6 ≤ x^2 + 5*x + 6 ∧ x^2 + 5*x + 6 ≤ 12) ↔ ((-6 ≤ x ∧ x ≤ -5) ∨ (0 ≤ x ∧ x ≤ 1)) :=
by sorry

end quadratic_interval_l3379_337922


namespace product_equals_one_l3379_337988

theorem product_equals_one :
  (∀ a b c : ℝ, a * b * c = Real.sqrt ((a + 2) * (b + 3)) / (c + 1)) →
  6 * 15 * 11 = 1 := by
sorry

end product_equals_one_l3379_337988


namespace toy_store_revenue_l3379_337985

theorem toy_store_revenue (D : ℚ) (D_pos : D > 0) : 
  let nov := (2 / 5 : ℚ) * D
  let jan := (1 / 5 : ℚ) * nov
  let avg := (nov + jan) / 2
  D / avg = 25 / 6 := by
sorry

end toy_store_revenue_l3379_337985


namespace transformed_circle_center_l3379_337907

def reflect_x (p : ℝ × ℝ) : ℝ × ℝ := (p.1, -p.2)

def translate (p : ℝ × ℝ) (dx dy : ℝ) : ℝ × ℝ := (p.1 + dx, p.2 + dy)

def circle_center : ℝ × ℝ := (4, -3)

theorem transformed_circle_center :
  let reflected := reflect_x circle_center
  let translated_right := translate reflected 5 0
  let final_position := translate translated_right 0 3
  final_position = (9, 6) := by sorry

end transformed_circle_center_l3379_337907


namespace rectangle_y_value_l3379_337974

theorem rectangle_y_value (y : ℝ) (h1 : y > 0) : 
  let vertices := [(1, y), (-5, y), (1, -2), (-5, -2)]
  let length := 1 - (-5)
  let height := y - (-2)
  let area := length * height
  area = 56 → y = 22/3 := by
  sorry

end rectangle_y_value_l3379_337974


namespace sign_white_area_l3379_337949

/-- Represents the dimensions and areas of the letters in the sign --/
structure LetterAreas where
  m_area : ℝ
  a_area : ℝ
  t_area : ℝ
  h_area : ℝ

/-- Calculates the white area of the sign after drawing the letters "MATH" --/
def white_area (sign_width sign_height : ℝ) (letters : LetterAreas) : ℝ :=
  sign_width * sign_height - (letters.m_area + letters.a_area + letters.t_area + letters.h_area)

/-- Theorem stating that the white area of the sign is 42.5 square units --/
theorem sign_white_area :
  let sign_width := 20
  let sign_height := 4
  let letters := LetterAreas.mk 12 7.5 7 11
  white_area sign_width sign_height letters = 42.5 := by
  sorry

end sign_white_area_l3379_337949


namespace factorization_72_P_72_l3379_337964

/-- P(n) represents the number of ways to write a positive integer n 
    as a product of integers greater than 1, where order matters. -/
def P (n : ℕ+) : ℕ := sorry

/-- The prime factorization of 72 is 2^3 * 3^2 -/
theorem factorization_72 : 72 = 2^3 * 3^2 := sorry

/-- The main theorem: P(72) = 17 -/
theorem P_72 : P 72 = 17 := sorry

end factorization_72_P_72_l3379_337964


namespace simplify_expression_1_l3379_337961

theorem simplify_expression_1 : 
  Real.sqrt 8 + Real.sqrt (1/3) - 2 * Real.sqrt 2 = Real.sqrt 3 / 3 := by sorry

end simplify_expression_1_l3379_337961


namespace girl_pairs_in_circular_arrangement_l3379_337967

/-- 
Given a circular arrangement of boys and girls:
- n_boys: number of boys
- n_girls: number of girls
- boy_pairs: number of pairs of boys sitting next to each other
- girl_pairs: number of pairs of girls sitting next to each other
-/
def circular_arrangement (n_boys n_girls boy_pairs girl_pairs : ℕ) : Prop :=
  n_boys + n_girls > 0 ∧ boy_pairs ≤ n_boys ∧ girl_pairs ≤ n_girls

theorem girl_pairs_in_circular_arrangement 
  (n_boys n_girls boy_pairs girl_pairs : ℕ) 
  (h_arrangement : circular_arrangement n_boys n_girls boy_pairs girl_pairs)
  (h_boys : n_boys = 10)
  (h_girls : n_girls = 15)
  (h_boy_pairs : boy_pairs = 5) :
  girl_pairs = 10 := by
  sorry

end girl_pairs_in_circular_arrangement_l3379_337967


namespace unique_function_identity_l3379_337906

theorem unique_function_identity (f : ℝ → ℝ) 
  (h1 : ∀ x ≠ 0, f x = x^2 * f (1/x))
  (h2 : ∀ x y, f (x + y) = f x + f y)
  (h3 : f 1 = 1) :
  ∀ x, f x = x :=
sorry

end unique_function_identity_l3379_337906


namespace race_distance_p_l3379_337977

/-- The distance P runs in a race where:
  1. P's speed is 20% faster than Q's speed
  2. Q starts 300 meters ahead of P
  3. P and Q finish the race at the same time
-/
theorem race_distance_p (vq : ℝ) : ∃ dp : ℝ,
  let vp := 1.2 * vq
  let dq := dp - 300
  dp / vp = dq / vq ∧ dp = 1800 := by
  sorry

end race_distance_p_l3379_337977


namespace g_composed_four_times_is_even_l3379_337905

def is_even_function (g : ℝ → ℝ) : Prop :=
  ∀ x, g (-x) = g x

theorem g_composed_four_times_is_even 
  (g : ℝ → ℝ) 
  (h : is_even_function g) : 
  is_even_function (fun x ↦ g (g (g (g x)))) :=
by
  sorry

end g_composed_four_times_is_even_l3379_337905


namespace intersection_M_N_l3379_337946

def M : Set ℝ := {x | x^2 + x - 2 < 0}
def N : Set ℝ := {x | Real.log x / Real.log (1/2) > -1}

theorem intersection_M_N : M ∩ N = {x : ℝ | 0 < x ∧ x < 1} := by sorry

end intersection_M_N_l3379_337946


namespace quadratic_root_ratio_l3379_337983

theorem quadratic_root_ratio (a b c : ℝ) (h : a ≠ 0) :
  (∃ x y : ℝ, x = 4*y ∧ a*x^2 + b*x + c = 0 ∧ a*y^2 + b*y + c = 0) →
  16 * b^2 / (a * c) = 100 := by
sorry

end quadratic_root_ratio_l3379_337983


namespace disjunction_false_l3379_337990

theorem disjunction_false :
  ¬(
    (∃ x : ℝ, x^2 + 1 < 2*x) ∨
    (∀ m : ℝ, (∀ x : ℝ, m*x^2 - m*x + 1 > 0) → (0 < m ∧ m < 4))
  ) := by sorry

end disjunction_false_l3379_337990


namespace max_value_polynomial_l3379_337957

theorem max_value_polynomial (x y : ℝ) (h : x + y = 5) :
  x^4*y + x^3*y + x^2*y + x*y + x*y^2 + x*y^3 + x*y^4 ≤ 6084/17 ∧
  ∃ (x₀ y₀ : ℝ), x₀ + y₀ = 5 ∧ 
    x₀^4*y₀ + x₀^3*y₀ + x₀^2*y₀ + x₀*y₀ + x₀*y₀^2 + x₀*y₀^3 + x₀*y₀^4 = 6084/17 :=
by sorry

end max_value_polynomial_l3379_337957


namespace vector_perpendicular_problem_l3379_337902

theorem vector_perpendicular_problem (a b c : ℝ × ℝ) (k : ℝ) :
  a = (1, 2) →
  b = (1, 1) →
  c = (a.1 + k * b.1, a.2 + k * b.2) →
  b.1 * c.1 + b.2 * c.2 = 0 →
  k = -3/2 := by
  sorry

end vector_perpendicular_problem_l3379_337902


namespace composition_equality_l3379_337910

theorem composition_equality (f g : ℝ → ℝ) (a : ℝ) :
  (∀ x, f x = x / 3 + 2) →
  (∀ x, g x = 5 - 2 * x) →
  f (g a) = 4 →
  a = -1/2 := by
sorry

end composition_equality_l3379_337910


namespace polynomial_division_remainder_l3379_337954

theorem polynomial_division_remainder : ∃ q : Polynomial ℝ, 
  (X : Polynomial ℝ)^5 + 1 = (X^2 - 4*X + 5) * q + 76 := by
  sorry

end polynomial_division_remainder_l3379_337954


namespace sufficient_but_not_necessary_parallel_l3379_337999

/-- Two lines are parallel if their slopes are equal and they are not identical -/
def are_parallel (a b c d e f : ℝ) : Prop :=
  (a * f = b * d) ∧ (a * e ≠ b * c)

theorem sufficient_but_not_necessary_parallel :
  (are_parallel 3 2 1 3 2 (-2)) ∧
  (∃ a : ℝ, a ≠ 3 ∧ are_parallel a 2 1 3 (a - 1) (-2)) :=
by sorry

end sufficient_but_not_necessary_parallel_l3379_337999


namespace bisecting_line_sum_l3379_337912

/-- Triangle PQR with vertices P(0, 10), Q(3, 0), and R(9, 0) -/
structure Triangle where
  P : ℝ × ℝ
  Q : ℝ × ℝ
  R : ℝ × ℝ

/-- The triangle PQR with given coordinates -/
def trianglePQR : Triangle :=
  { P := (0, 10)
    Q := (3, 0)
    R := (9, 0) }

/-- A line represented by its slope and y-intercept -/
structure Line where
  slope : ℝ
  yIntercept : ℝ

/-- The line that bisects the area of triangle PQR and passes through Q -/
def bisectingLine (t : Triangle) : Line :=
  sorry

/-- Theorem: The sum of the slope and y-intercept of the bisecting line is -20/3 -/
theorem bisecting_line_sum (t : Triangle) (h : t = trianglePQR) :
    (bisectingLine t).slope + (bisectingLine t).yIntercept = -20/3 := by
  sorry

end bisecting_line_sum_l3379_337912


namespace holistic_substitution_l3379_337926

theorem holistic_substitution (a : ℝ) (x : ℝ) :
  (a^2 + 3*a - 2 = 0) →
  (5*a^3 + 15*a^2 - 10*a + 2020 = 2020) ∧
  ((x^2 + 2*x - 3 = 0) → 
   (x = 1 ∨ x = -3) →
   ((2*x + 3)^2 + 2*(2*x + 3) - 3 = 0) →
   (x = -1 ∨ x = -3)) :=
by sorry

end holistic_substitution_l3379_337926


namespace inequality_solution_l3379_337952

theorem inequality_solution (x : ℝ) : 
  (2 * Real.sqrt ((4 * x - 9)^2) + Real.sqrt (3 * Real.sqrt x - 5 + 2 * |x - 2|) ≤ 18 - 8 * x) ↔ 
  (x = 0) :=
by sorry

end inequality_solution_l3379_337952


namespace differential_pricing_profitability_l3379_337992

theorem differential_pricing_profitability 
  (n : ℝ) (t : ℝ) (h_n_pos : n > 0) (h_t_pos : t > 0) : 
  let shorts_ratio : ℝ := 0.75
  let suits_ratio : ℝ := 0.25
  let businessmen_ratio : ℝ := 0.8
  let tourists_ratio : ℝ := 0.2
  let uniform_revenue := n * t
  let differential_revenue (X : ℝ) := 
    (shorts_ratio * n * t) + 
    (suits_ratio * businessmen_ratio * n * (t + X))
  ∃ X : ℝ, X ≥ 0 ∧ 
    ∀ Y : ℝ, Y ≥ 0 → differential_revenue Y ≥ uniform_revenue → Y ≥ X ∧
    differential_revenue X = uniform_revenue ∧
    X = t / 4 :=
by
  sorry

end differential_pricing_profitability_l3379_337992


namespace boiling_temperature_calculation_boiling_temperature_proof_l3379_337942

theorem boiling_temperature_calculation (initial_temp : ℝ) (temp_increase : ℝ) 
  (pasta_time : ℝ) (total_time : ℝ) : ℝ :=
  let mixing_time := pasta_time / 3
  let cooking_and_mixing_time := pasta_time + mixing_time
  let time_to_boil := total_time - cooking_and_mixing_time
  let temp_increase_total := time_to_boil * temp_increase
  initial_temp + temp_increase_total

theorem boiling_temperature_proof :
  boiling_temperature_calculation 41 3 12 73 = 212 := by
  sorry

end boiling_temperature_calculation_boiling_temperature_proof_l3379_337942


namespace coin_flip_probability_l3379_337976

theorem coin_flip_probability :
  let n : ℕ := 6  -- total number of coins
  let k : ℕ := 3  -- number of specific coins we're interested in
  let total_outcomes : ℕ := 2^n
  let favorable_outcomes : ℕ := 2^(n - k)
  (favorable_outcomes : ℚ) / total_outcomes = 1 / 8 :=
by sorry

end coin_flip_probability_l3379_337976


namespace sqrt_sum_inequality_l3379_337939

theorem sqrt_sum_inequality (a : ℝ) (ha : a > 0) :
  Real.sqrt a + Real.sqrt (a + 5) < Real.sqrt (a + 2) + Real.sqrt (a + 3) := by
  sorry

end sqrt_sum_inequality_l3379_337939


namespace first_box_not_empty_count_l3379_337995

/-- The number of ways to distribute three distinct balls into four boxes. -/
def total_distributions : ℕ := 4^3

/-- The number of ways to distribute three distinct balls into four boxes
    such that the first box is empty. -/
def distributions_with_empty_first_box : ℕ := 3^3

theorem first_box_not_empty_count :
  total_distributions - distributions_with_empty_first_box = 37 := by
  sorry

end first_box_not_empty_count_l3379_337995
