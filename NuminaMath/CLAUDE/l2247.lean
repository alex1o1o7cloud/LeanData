import Mathlib

namespace NUMINAMATH_CALUDE_power_of_729_two_thirds_l2247_224770

theorem power_of_729_two_thirds : (729 : ℝ) ^ (2/3) = 81 := by
  sorry

end NUMINAMATH_CALUDE_power_of_729_two_thirds_l2247_224770


namespace NUMINAMATH_CALUDE_two_distinct_zeros_implies_m_3_or_4_l2247_224785

-- Define the function f(x)
def f (m : ℝ) (x : ℝ) : ℝ := m * x^2 + (m - 1) * x - 1

-- Define the theorem
theorem two_distinct_zeros_implies_m_3_or_4 :
  ∀ m : ℝ,
  (∃ x y : ℝ, x ≠ y ∧ x ∈ Set.Icc (-1) 2 ∧ y ∈ Set.Icc (-1) 2 ∧ f m x = 0 ∧ f m y = 0) →
  (m = 3 ∨ m = 4) :=
by sorry

end NUMINAMATH_CALUDE_two_distinct_zeros_implies_m_3_or_4_l2247_224785


namespace NUMINAMATH_CALUDE_age_problem_solution_l2247_224719

/-- Represents the ages of James and Joe -/
structure Ages where
  james : ℕ
  joe : ℕ

/-- The conditions of the problem -/
def satisfiesConditions (ages : Ages) : Prop :=
  ages.joe = ages.james + 10 ∧
  2 * (ages.joe + 8) = 3 * (ages.james + 8)

/-- The theorem to prove -/
theorem age_problem_solution :
  ∃ (ages : Ages), satisfiesConditions ages ∧ ages.james = 12 ∧ ages.joe = 22 := by
  sorry


end NUMINAMATH_CALUDE_age_problem_solution_l2247_224719


namespace NUMINAMATH_CALUDE_ramu_car_profit_percent_l2247_224761

/-- Represents the profit percentage calculation for Ramu's car transaction --/
theorem ramu_car_profit_percent :
  let initial_cost_rs : ℚ := 45000
  let engine_repair_rs : ℚ := 17000
  let bodywork_repair_rs : ℚ := 25000
  let selling_price_rs : ℚ := 80000
  let total_cost_rs : ℚ := initial_cost_rs + engine_repair_rs + bodywork_repair_rs
  let profit_rs : ℚ := selling_price_rs - total_cost_rs
  let profit_percent : ℚ := profit_rs / total_cost_rs * 100
  profit_percent = -806 / 100 := by sorry

end NUMINAMATH_CALUDE_ramu_car_profit_percent_l2247_224761


namespace NUMINAMATH_CALUDE_solve_equation_l2247_224735

theorem solve_equation (x : ℝ) : (x / 5) + 3 = 4 → x = 5 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l2247_224735


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sine_problem_l2247_224715

theorem arithmetic_sequence_sine_problem (a : ℕ → ℝ) :
  (∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)) →  -- arithmetic sequence condition
  a 5 + a 6 = 10 * Real.pi / 3 →                    -- given condition
  Real.sin (a 4 + a 7) = -Real.sqrt 3 / 2 :=        -- conclusion to prove
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sine_problem_l2247_224715


namespace NUMINAMATH_CALUDE_max_height_particle_from_wheel_l2247_224703

/-- The maximum height reached by a particle thrown off a rolling wheel -/
theorem max_height_particle_from_wheel
  (r : ℝ) -- radius of the wheel
  (ω : ℝ) -- angular velocity of the wheel
  (g : ℝ) -- acceleration due to gravity
  (h_pos : r > 0) -- radius is positive
  (ω_pos : ω > 0) -- angular velocity is positive
  (g_pos : g > 0) -- gravity is positive
  (h_ω : ω > Real.sqrt (g / r)) -- condition on angular velocity
  : ∃ (h : ℝ), h = (r * ω + g / ω)^2 / (2 * g) ∧
    ∀ (h' : ℝ), h' ≤ h :=
by sorry

end NUMINAMATH_CALUDE_max_height_particle_from_wheel_l2247_224703


namespace NUMINAMATH_CALUDE_charlie_share_l2247_224773

def distribute_money (total : ℕ) (ratio1 ratio2 ratio3 : ℕ) (deduct1 deduct2 deduct3 : ℕ) : ℕ × ℕ × ℕ :=
  sorry

theorem charlie_share :
  let (alice, bond, charlie) := distribute_money 1105 11 18 24 10 20 15
  charlie = 495 := by sorry

end NUMINAMATH_CALUDE_charlie_share_l2247_224773


namespace NUMINAMATH_CALUDE_page_number_added_twice_l2247_224706

theorem page_number_added_twice (n : ℕ) : 
  (n * (n + 1) / 2 ≤ 1986) ∧ 
  ((n + 1) * (n + 2) / 2 > 1986) →
  1986 - (n * (n + 1) / 2) = 33 := by
sorry

end NUMINAMATH_CALUDE_page_number_added_twice_l2247_224706


namespace NUMINAMATH_CALUDE_unique_intersection_l2247_224794

/-- The parabola equation -/
def parabola (x : ℝ) : ℝ := x^2 - x + 1

/-- The line equation -/
def line (k : ℝ) (x : ℝ) : ℝ := 4*x + k

/-- Theorem stating the condition for exactly one intersection point -/
theorem unique_intersection (k : ℝ) : 
  (∃! x, parabola x = line k x) ↔ k = -21/4 := by
  sorry

end NUMINAMATH_CALUDE_unique_intersection_l2247_224794


namespace NUMINAMATH_CALUDE_inequality_proof_l2247_224776

theorem inequality_proof (n : ℕ) (a b : ℝ) 
  (h1 : n ≠ 1) (h2 : a > b) (h3 : b > 0) : 
  ((a + b) / 2) ^ n < (a ^ n + b ^ n) / 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2247_224776


namespace NUMINAMATH_CALUDE_triangle_area_change_l2247_224777

theorem triangle_area_change (base height : ℝ) (base_new height_new area area_new : ℝ) 
  (h1 : base_new = 1.10 * base) 
  (h2 : height_new = 0.95 * height) 
  (h3 : area = (base * height) / 2) 
  (h4 : area_new = (base_new * height_new) / 2) :
  area_new = 1.045 * area := by
  sorry

#check triangle_area_change

end NUMINAMATH_CALUDE_triangle_area_change_l2247_224777


namespace NUMINAMATH_CALUDE_function_properties_l2247_224730

def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

def is_increasing_on (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x → x < y → y ≤ b → f x ≤ f y

theorem function_properties
  (f : ℝ → ℝ)
  (h_even : is_even f)
  (h_neg : ∀ x, f (x + 1) = -f x)
  (h_incr : is_increasing_on f (-1) 0) :
  (∀ x, f (x + 2) = f x) ∧
  (∀ x, f (2 - x) = f x) ∧
  f 2 = f 0 :=
sorry

end NUMINAMATH_CALUDE_function_properties_l2247_224730


namespace NUMINAMATH_CALUDE_pascal_triangle_fifth_number_l2247_224724

theorem pascal_triangle_fifth_number : 
  let row := List.cons 1 (List.cons 15 (List.replicate 3 0))  -- represents the start of the row
  let fifth_number := Nat.choose 15 4  -- represents ₁₅C₄
  fifth_number = 1365 := by
  sorry

end NUMINAMATH_CALUDE_pascal_triangle_fifth_number_l2247_224724


namespace NUMINAMATH_CALUDE_angle_between_vectors_l2247_224766

/-- The angle between two planar vectors satisfying given conditions -/
theorem angle_between_vectors (a b : ℝ × ℝ) 
  (h1 : (a.1 + b.1) * b.1 + (a.2 + b.2) * b.2 = 7)
  (h2 : Real.sqrt (a.1^2 + a.2^2) = Real.sqrt 3)
  (h3 : Real.sqrt (b.1^2 + b.2^2) = 2) :
  Real.arccos ((a.1 * b.1 + a.2 * b.2) / (Real.sqrt (a.1^2 + a.2^2) * Real.sqrt (b.1^2 + b.2^2))) = π / 6 := by
  sorry

end NUMINAMATH_CALUDE_angle_between_vectors_l2247_224766


namespace NUMINAMATH_CALUDE_jones_elementary_population_l2247_224713

theorem jones_elementary_population :
  ∀ (total_students : ℕ) (boys_percentage : ℚ) (sample_size : ℕ),
    boys_percentage = 60 / 100 →
    sample_size = 90 →
    (sample_size : ℚ) / boys_percentage = total_students →
    total_students = 150 := by
  sorry

end NUMINAMATH_CALUDE_jones_elementary_population_l2247_224713


namespace NUMINAMATH_CALUDE_playground_girls_l2247_224759

theorem playground_girls (total_children : ℕ) (boys : ℕ) 
  (h1 : total_children = 62) 
  (h2 : boys = 27) : 
  total_children - boys = 35 := by
  sorry

end NUMINAMATH_CALUDE_playground_girls_l2247_224759


namespace NUMINAMATH_CALUDE_trapezoid_pq_length_l2247_224718

/-- Represents a trapezoid ABCD with a parallel line PQ intersecting diagonals -/
structure Trapezoid :=
  (a : ℝ) -- Length of base BC
  (b : ℝ) -- Length of base AD
  (pl : ℝ) -- Length of PL
  (lr : ℝ) -- Length of LR

/-- The main theorem about the length of PQ in a trapezoid -/
theorem trapezoid_pq_length (t : Trapezoid) (h : t.pl = t.lr) :
  ∃ (pq : ℝ), pq = (3 * t.a * t.b) / (2 * t.a + t.b) ∨ pq = (3 * t.a * t.b) / (t.a + 2 * t.b) :=
sorry

end NUMINAMATH_CALUDE_trapezoid_pq_length_l2247_224718


namespace NUMINAMATH_CALUDE_total_power_cost_l2247_224762

def refrigerator_cost (water_heater_cost : ℝ) : ℝ := 3 * water_heater_cost

def electric_oven_cost : ℝ := 500

theorem total_power_cost (water_heater_cost : ℝ) 
  (h1 : electric_oven_cost = 2 * water_heater_cost) :
  water_heater_cost + refrigerator_cost water_heater_cost + electric_oven_cost = 1500 := by
  sorry

end NUMINAMATH_CALUDE_total_power_cost_l2247_224762


namespace NUMINAMATH_CALUDE_largest_possible_number_david_l2247_224740

/-- Represents a decimal number with up to two digits before and after the decimal point -/
structure DecimalNumber :=
  (beforeDecimal : Fin 100)
  (afterDecimal : Fin 100)

/-- Checks if a DecimalNumber has mutually different digits -/
def hasMutuallyDifferentDigits (n : DecimalNumber) : Prop :=
  sorry

/-- Checks if a DecimalNumber has exactly two identical digits -/
def hasExactlyTwoIdenticalDigits (n : DecimalNumber) : Prop :=
  sorry

/-- Converts a DecimalNumber to a rational number -/
def toRational (n : DecimalNumber) : ℚ :=
  sorry

/-- The sum of two DecimalNumbers -/
def sum (a b : DecimalNumber) : ℚ :=
  toRational a + toRational b

theorem largest_possible_number_david
  (jana david : DecimalNumber)
  (h_sum : sum jana david = 11.11)
  (h_david_digits : hasMutuallyDifferentDigits david)
  (h_jana_digits : hasExactlyTwoIdenticalDigits jana) :
  toRational david ≤ 0.9 :=
sorry

end NUMINAMATH_CALUDE_largest_possible_number_david_l2247_224740


namespace NUMINAMATH_CALUDE_s_1000_eq_720_l2247_224793

def s : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | n + 2 => 
    if n % 2 = 0 then s (n / 2)
    else if (n - 1) % 4 = 0 then s ((n - 1) / 2 + 1)
    else s ((n + 1) / 2 - 1) + (s ((n + 1) / 2 - 1))^2 / s ((n + 1) / 4 - 1)

theorem s_1000_eq_720 : s 1000 = 720 := by
  sorry

end NUMINAMATH_CALUDE_s_1000_eq_720_l2247_224793


namespace NUMINAMATH_CALUDE_sams_tuna_discount_l2247_224716

/-- Calculates the discount per coupon for a tuna purchase. -/
def discount_per_coupon (num_cans : ℕ) (num_coupons : ℕ) (paid : ℕ) (change : ℕ) (cost_per_can : ℕ) : ℕ :=
  let total_paid := paid - change
  let total_cost := num_cans * cost_per_can
  let total_discount := total_cost - total_paid
  total_discount / num_coupons

/-- Proves that the discount per coupon is 25 cents for Sam's tuna purchase. -/
theorem sams_tuna_discount :
  discount_per_coupon 9 5 2000 550 175 = 25 := by
  sorry

end NUMINAMATH_CALUDE_sams_tuna_discount_l2247_224716


namespace NUMINAMATH_CALUDE_average_of_combined_results_l2247_224758

theorem average_of_combined_results (n₁ : ℕ) (n₂ : ℕ) (avg₁ : ℝ) (avg₂ : ℝ) :
  n₁ = 60 →
  n₂ = 40 →
  avg₁ = 40 →
  avg₂ = 60 →
  (n₁ * avg₁ + n₂ * avg₂) / (n₁ + n₂) = 48 := by
  sorry

end NUMINAMATH_CALUDE_average_of_combined_results_l2247_224758


namespace NUMINAMATH_CALUDE_bug_return_probability_l2247_224780

/-- Probability of the bug being at the starting vertex after n moves -/
def Q : ℕ → ℚ
  | 0 => 1
  | n + 1 => 1/2 * (1 - Q n)

/-- The probability of returning to the starting vertex on the eighth move -/
theorem bug_return_probability : Q 8 = 43/128 := by
  sorry

end NUMINAMATH_CALUDE_bug_return_probability_l2247_224780


namespace NUMINAMATH_CALUDE_union_equality_iff_m_range_l2247_224722

-- Define sets A and B
def A : Set ℝ := {x | -1 ≤ x ∧ x ≤ 2}
def B (m : ℝ) : Set ℝ := {x | x^2 - (2*m + 1)*x + 2*m < 0}

-- State the theorem
theorem union_equality_iff_m_range :
  ∀ m : ℝ, (A ∪ B m = A) ↔ (-1/2 ≤ m ∧ m ≤ 1) := by
  sorry

end NUMINAMATH_CALUDE_union_equality_iff_m_range_l2247_224722


namespace NUMINAMATH_CALUDE_min_participants_is_61_l2247_224752

/-- Represents the number of participants in the race. -/
def n : ℕ := 61

/-- Represents the number of people who finished before Andrei. -/
def x : ℕ := 20

/-- Represents the number of people who finished before Dima. -/
def y : ℕ := 15

/-- Represents the number of people who finished before Lenya. -/
def z : ℕ := 12

/-- Theorem stating that 61 is the minimum number of participants satisfying the given conditions. -/
theorem min_participants_is_61 :
  (x + 1 + 2 * x = n) ∧
  (y + 1 + 3 * y = n) ∧
  (z + 1 + 4 * z = n) ∧
  (∀ m : ℕ, m < n → ¬((m - 1) % 3 = 0 ∧ (m - 1) % 4 = 0 ∧ (m - 1) % 5 = 0)) :=
by sorry

#check min_participants_is_61

end NUMINAMATH_CALUDE_min_participants_is_61_l2247_224752


namespace NUMINAMATH_CALUDE_binary_decimal_octal_conversion_l2247_224709

/-- Converts a binary number represented as a list of bits to a decimal number -/
def binary_to_decimal (bits : List Bool) : ℕ :=
  bits.enum.foldl (fun acc (i, b) => acc + if b then 2^i else 0) 0

/-- Converts a decimal number to an octal number represented as a list of digits -/
def decimal_to_octal (n : ℕ) : List ℕ :=
  if n = 0 then [0] else
    let rec go (m : ℕ) (acc : List ℕ) :=
      if m = 0 then acc
      else go (m / 8) ((m % 8) :: acc)
    go n []

/-- The binary representation of 11011100₂ -/
def binary_num : List Bool := [false, false, true, true, true, false, true, true]

theorem binary_decimal_octal_conversion :
  (binary_to_decimal binary_num = 110) ∧
  (decimal_to_octal 110 = [1, 5, 6]) := by
  sorry


end NUMINAMATH_CALUDE_binary_decimal_octal_conversion_l2247_224709


namespace NUMINAMATH_CALUDE_polynomial_equation_sum_l2247_224702

theorem polynomial_equation_sum (a b : ℤ) : 
  (∀ x : ℝ, 2 * x^3 - a * x^2 - 5 * x + 5 = (2 * x^2 + a * x - 1) * (x - b) + 3) → 
  a + b = 4 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_equation_sum_l2247_224702


namespace NUMINAMATH_CALUDE_yummy_kibble_percentage_proof_l2247_224796

/-- The number of vets in the state -/
def total_vets : ℕ := 1000

/-- The percentage of vets recommending Puppy Kibble -/
def puppy_kibble_percentage : ℚ := 20 / 100

/-- The number of additional vets recommending Yummy Dog Kibble compared to Puppy Kibble -/
def additional_yummy_kibble_vets : ℕ := 100

/-- The percentage of vets recommending Yummy Dog Kibble -/
def yummy_kibble_percentage : ℚ := 30 / 100

theorem yummy_kibble_percentage_proof :
  (puppy_kibble_percentage * total_vets + additional_yummy_kibble_vets : ℚ) / total_vets = yummy_kibble_percentage := by
  sorry

end NUMINAMATH_CALUDE_yummy_kibble_percentage_proof_l2247_224796


namespace NUMINAMATH_CALUDE_contestant_selection_probabilities_l2247_224721

/-- Represents the probability of selecting two females from a group of contestants. -/
def prob_two_females (total : ℕ) (females : ℕ) : ℚ :=
  (females.choose 2 : ℚ) / (total.choose 2 : ℚ)

/-- Represents the probability of selecting at least one male from a group of contestants. -/
def prob_at_least_one_male (total : ℕ) (females : ℕ) : ℚ :=
  1 - prob_two_females total females

/-- Theorem stating the probabilities for selecting contestants from a group of 8 with 5 females and 3 males. -/
theorem contestant_selection_probabilities :
  let total := 8
  let females := 5
  prob_two_females total females = 5 / 14 ∧
  prob_at_least_one_male total females = 9 / 14 := by
  sorry

end NUMINAMATH_CALUDE_contestant_selection_probabilities_l2247_224721


namespace NUMINAMATH_CALUDE_function_passes_through_point_l2247_224728

theorem function_passes_through_point (a : ℝ) (ha : a > 0) (ha1 : a ≠ 1) :
  let f : ℝ → ℝ := λ x ↦ a^(x + 4) + 3
  f (-4) = 4 := by sorry

end NUMINAMATH_CALUDE_function_passes_through_point_l2247_224728


namespace NUMINAMATH_CALUDE_custom_op_result_l2247_224747

/-- Define the custom operation * -/
def custom_op (a b : ℝ) (x y : ℝ) : ℝ := a * x + b * y + 1

/-- Theorem stating the result of the custom operation given the conditions -/
theorem custom_op_result (a b : ℝ) :
  (custom_op a b 3 5 = 15) →
  (custom_op a b 4 7 = 28) →
  (custom_op a b 1 1 = -11) := by
  sorry

end NUMINAMATH_CALUDE_custom_op_result_l2247_224747


namespace NUMINAMATH_CALUDE_max_value_of_f_on_interval_l2247_224781

def f (x : ℝ) : ℝ := -4 * x^3 + 3 * x + 2

theorem max_value_of_f_on_interval :
  ∃ (x : ℝ), x ∈ Set.Icc 0 1 ∧
  (∀ (y : ℝ), y ∈ Set.Icc 0 1 → f y ≤ f x) ∧
  f x = 3 :=
sorry

end NUMINAMATH_CALUDE_max_value_of_f_on_interval_l2247_224781


namespace NUMINAMATH_CALUDE_adult_ticket_cost_l2247_224727

/-- Proves that the cost of an adult ticket is $12 given the conditions of the problem -/
theorem adult_ticket_cost (total_tickets : ℕ) (total_receipts : ℕ) (adult_tickets : ℕ) (child_ticket_cost : ℕ) :
  total_tickets = 130 →
  total_receipts = 840 →
  adult_tickets = 40 →
  child_ticket_cost = 4 →
  ∃ (adult_ticket_cost : ℕ),
    adult_ticket_cost * adult_tickets + child_ticket_cost * (total_tickets - adult_tickets) = total_receipts ∧
    adult_ticket_cost = 12 := by
  sorry

end NUMINAMATH_CALUDE_adult_ticket_cost_l2247_224727


namespace NUMINAMATH_CALUDE_largest_negative_integer_congruence_l2247_224726

theorem largest_negative_integer_congruence :
  ∃ (x : ℤ), x = -6 ∧
  (34 * x + 6) % 20 = 2 % 20 ∧
  ∀ (y : ℤ), y < 0 → (34 * y + 6) % 20 = 2 % 20 → y ≤ x :=
by sorry

end NUMINAMATH_CALUDE_largest_negative_integer_congruence_l2247_224726


namespace NUMINAMATH_CALUDE_evaluate_expression_l2247_224783

theorem evaluate_expression : 
  |7 - (8^2) * (3 - 12)| - |(5^3) - (Real.sqrt 11)^4| = 579 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l2247_224783


namespace NUMINAMATH_CALUDE_triangle_similarity_problem_l2247_224756

-- Define the triangles and their properties
structure Triangle :=
  (side1 : ℝ)
  (side2 : ℝ)
  (height : ℝ)

-- Define the similarity relation
def similar (t1 t2 : Triangle) : Prop := sorry

-- State the theorem
theorem triangle_similarity_problem 
  (FGH IJH : Triangle)
  (h_similar : similar FGH IJH)
  (h_GH : FGH.side1 = 25)
  (h_JH : IJH.side1 = 15)
  (h_height : FGH.height = 15) :
  IJH.side2 = 9 := by sorry

end NUMINAMATH_CALUDE_triangle_similarity_problem_l2247_224756


namespace NUMINAMATH_CALUDE_large_triangle_toothpicks_l2247_224774

/-- The number of small triangles in the base row of the large equilateral triangle -/
def base_triangles : ℕ := 100

/-- The total number of small triangles in the large equilateral triangle -/
def total_triangles : ℕ := (base_triangles * (base_triangles + 1)) / 2

/-- The number of toothpicks required to assemble the large equilateral triangle -/
def toothpicks_required : ℕ := ((3 * total_triangles) / 2) + (3 * base_triangles)

theorem large_triangle_toothpicks :
  toothpicks_required = 7875 := by sorry

end NUMINAMATH_CALUDE_large_triangle_toothpicks_l2247_224774


namespace NUMINAMATH_CALUDE_passes_through_origin_symmetric_about_y_axis_symmetric_expression_inequality_condition_l2247_224799

-- Define the quadratic function
def f (m : ℝ) (x : ℝ) : ℝ := x^2 - 2*(m-1)*x - 2*m + m^2

-- Theorem 1: The graph passes through the origin when m = 0 or m = 2
theorem passes_through_origin (m : ℝ) : 
  f m 0 = 0 ↔ m = 0 ∨ m = 2 := by sorry

-- Theorem 2: The graph is symmetric about the y-axis when m = 1
theorem symmetric_about_y_axis (m : ℝ) :
  (∀ x, f m x = f m (-x)) ↔ m = 1 := by sorry

-- Theorem 3: Expression when symmetric about y-axis
theorem symmetric_expression (x : ℝ) :
  f 1 x = x^2 - 1 := by sorry

-- Theorem 4: Condition for f(x) ≥ 3 in the interval [1, 3]
theorem inequality_condition (m : ℝ) :
  (∀ x ∈ Set.Icc 1 3, f m x ≥ 3) ↔ m ≤ 0 ∨ m ≥ 6 := by sorry

end NUMINAMATH_CALUDE_passes_through_origin_symmetric_about_y_axis_symmetric_expression_inequality_condition_l2247_224799


namespace NUMINAMATH_CALUDE_apple_basket_problem_l2247_224755

def arithmetic_sum (a₁ : ℕ) (d : ℕ) (n : ℕ) : ℕ :=
  n * (2 * a₁ + (n - 1) * d) / 2

theorem apple_basket_problem (total_apples : ℕ) (first_basket : ℕ) (increment : ℕ) :
  total_apples = 495 →
  first_basket = 25 →
  increment = 2 →
  ∃ x : ℕ, x = 13 ∧ arithmetic_sum first_basket increment x = total_apples :=
by sorry

end NUMINAMATH_CALUDE_apple_basket_problem_l2247_224755


namespace NUMINAMATH_CALUDE_h_zero_at_seven_fifths_l2247_224797

-- Define the function h
def h (x : ℝ) : ℝ := 5 * x - 7

-- Theorem statement
theorem h_zero_at_seven_fifths : h (7 / 5) = 0 := by
  sorry

end NUMINAMATH_CALUDE_h_zero_at_seven_fifths_l2247_224797


namespace NUMINAMATH_CALUDE_smallest_b_value_l2247_224790

theorem smallest_b_value (a b : ℤ) (h1 : 9 < a ∧ a < 21) (h2 : b < 31) (h3 : (a : ℚ) / b = 2/3) :
  ∃ (n : ℤ), n = 14 ∧ n < b ∧ ∀ m, m < b → m ≤ n :=
sorry

end NUMINAMATH_CALUDE_smallest_b_value_l2247_224790


namespace NUMINAMATH_CALUDE_vase_transport_problem_l2247_224775

theorem vase_transport_problem (x : ℕ) : 
  (∃ C : ℝ, 
    (10 * (x - 50) - C = -300) ∧ 
    (12 * (x - 50) - C = 800)) → 
  x = 600 := by
  sorry

end NUMINAMATH_CALUDE_vase_transport_problem_l2247_224775


namespace NUMINAMATH_CALUDE_solution_set_quadratic_inequality_l2247_224748

theorem solution_set_quadratic_inequality :
  {x : ℝ | x^2 + x - 6 < 0} = {x : ℝ | -3 < x ∧ x < 2} := by sorry

end NUMINAMATH_CALUDE_solution_set_quadratic_inequality_l2247_224748


namespace NUMINAMATH_CALUDE_exactly_one_double_root_l2247_224751

/-- The function f(x) representing the left side of the equation -/
def f (a : ℝ) (x : ℝ) : ℝ := (x + 2)^2 * (x + 7)^2 + a

/-- The theorem stating the condition for exactly one double-root -/
theorem exactly_one_double_root (a : ℝ) : 
  (∃! x : ℝ, f a x = 0 ∧ (∀ y : ℝ, y ≠ x → f a y > 0)) ↔ a = -39.0625 := by sorry

end NUMINAMATH_CALUDE_exactly_one_double_root_l2247_224751


namespace NUMINAMATH_CALUDE_supplementary_angle_measures_l2247_224712

theorem supplementary_angle_measures :
  ∃ (possible_measures : Finset ℕ),
    (∀ A ∈ possible_measures,
      ∃ (B : ℕ) (k : ℕ),
        A > 0 ∧ B > 0 ∧ k > 0 ∧
        A + B = 180 ∧
        A = k * B) ∧
    (∀ A : ℕ,
      (∃ (B : ℕ) (k : ℕ),
        A > 0 ∧ B > 0 ∧ k > 0 ∧
        A + B = 180 ∧
        A = k * B) →
      A ∈ possible_measures) ∧
    Finset.card possible_measures = 17 :=
by sorry

end NUMINAMATH_CALUDE_supplementary_angle_measures_l2247_224712


namespace NUMINAMATH_CALUDE_equilateral_triangle_area_perimeter_ratio_l2247_224760

theorem equilateral_triangle_area_perimeter_ratio :
  let side_length : ℝ := 6
  let area : ℝ := (side_length^2 * Real.sqrt 3) / 4
  let perimeter : ℝ := 3 * side_length
  area / perimeter = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_equilateral_triangle_area_perimeter_ratio_l2247_224760


namespace NUMINAMATH_CALUDE_wood_measurement_l2247_224745

theorem wood_measurement (x : ℝ) : 
  (∃ rope : ℝ, rope = x + 4.5 ∧ rope / 2 = x + 1) → 
  (1/2 : ℝ) * (x + 4.5) = x - 1 :=
by sorry

end NUMINAMATH_CALUDE_wood_measurement_l2247_224745


namespace NUMINAMATH_CALUDE_compound_carbon_atoms_l2247_224779

/-- Represents the number of Carbon atoms in a compound -/
def carbonAtoms (molecularWeight : ℕ) (hydrogenAtoms : ℕ) : ℕ :=
  (molecularWeight - hydrogenAtoms) / 12

/-- Proves that a compound with 6 Hydrogen atoms and a molecular weight of 78 amu contains 6 Carbon atoms -/
theorem compound_carbon_atoms :
  carbonAtoms 78 6 = 6 :=
by
  sorry

#eval carbonAtoms 78 6

end NUMINAMATH_CALUDE_compound_carbon_atoms_l2247_224779


namespace NUMINAMATH_CALUDE_runs_by_running_percentage_l2247_224701

def total_runs : ℕ := 120
def num_boundaries : ℕ := 3
def num_sixes : ℕ := 8
def runs_per_boundary : ℕ := 4
def runs_per_six : ℕ := 6

theorem runs_by_running_percentage : 
  (total_runs - (num_boundaries * runs_per_boundary + num_sixes * runs_per_six)) / total_runs * 100 = 50 := by
  sorry

end NUMINAMATH_CALUDE_runs_by_running_percentage_l2247_224701


namespace NUMINAMATH_CALUDE_distinct_primes_in_product_l2247_224749

theorem distinct_primes_in_product : ∃ (s : Finset Nat), 
  (∀ p ∈ s, Nat.Prime p) ∧ 
  (∀ p : Nat, Nat.Prime p → (85 * 87 * 88 * 90) % p = 0 → p ∈ s) ∧ 
  Finset.card s = 6 := by
  sorry

end NUMINAMATH_CALUDE_distinct_primes_in_product_l2247_224749


namespace NUMINAMATH_CALUDE_committee_arrangement_l2247_224711

theorem committee_arrangement (n m : ℕ) (hn : n = 6) (hm : m = 4) : 
  Nat.choose (n + m) m = 210 := by
  sorry

end NUMINAMATH_CALUDE_committee_arrangement_l2247_224711


namespace NUMINAMATH_CALUDE_semicircle_perimeter_approx_l2247_224743

/-- The perimeter of a semicircle with radius 12 units is approximately 61.7 units. -/
theorem semicircle_perimeter_approx : ∃ (ε : ℝ), ε > 0 ∧ ε < 0.1 ∧ 
  |π * 12 + 24 - 61.7| < ε := by
  sorry

end NUMINAMATH_CALUDE_semicircle_perimeter_approx_l2247_224743


namespace NUMINAMATH_CALUDE_largest_product_l2247_224750

def S : Finset Int := {-4, -3, -1, 5, 6, 7}

def isConsecutive (a b : Int) : Prop := b = a + 1 ∨ a = b + 1

def fourDistinctElements (a b c d : Int) : Prop :=
  a ∈ S ∧ b ∈ S ∧ c ∈ S ∧ d ∈ S ∧
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d

def twoConsecutive (a b c d : Int) : Prop :=
  isConsecutive a b ∨ isConsecutive a c ∨ isConsecutive a d ∨
  isConsecutive b c ∨ isConsecutive b d ∨ isConsecutive c d

theorem largest_product :
  ∀ a b c d : Int,
    fourDistinctElements a b c d →
    twoConsecutive a b c d →
    a * b * c * d ≤ -210 :=
by sorry

end NUMINAMATH_CALUDE_largest_product_l2247_224750


namespace NUMINAMATH_CALUDE_monotone_decreasing_range_l2247_224757

/-- A function f is monotonically decreasing on ℝ if for all x, y ∈ ℝ, x < y implies f(x) > f(y) -/
def MonotonicallyDecreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, x < y → f x > f y

/-- The cubic function f(x) = -x³ + ax² - x - 1 -/
def f (a : ℝ) (x : ℝ) : ℝ := -x^3 + a*x^2 - x - 1

theorem monotone_decreasing_range (a : ℝ) :
  MonotonicallyDecreasing (f a) → a ∈ Set.Icc (-Real.sqrt 3) (Real.sqrt 3) :=
by sorry

end NUMINAMATH_CALUDE_monotone_decreasing_range_l2247_224757


namespace NUMINAMATH_CALUDE_water_level_rise_rate_l2247_224765

/-- The rate at which the water level rises in a cylinder when water drains from a cube -/
theorem water_level_rise_rate (cube_side : ℝ) (cylinder_radius : ℝ) (cube_fall_rate : ℝ) :
  cube_side = 100 →
  cylinder_radius = 100 →
  cube_fall_rate = 1 →
  (cylinder_radius ^ 2 * π) * (cube_side ^ 2 * cube_fall_rate) / (cylinder_radius ^ 2 * π) ^ 2 = 1 / π := by
  sorry

end NUMINAMATH_CALUDE_water_level_rise_rate_l2247_224765


namespace NUMINAMATH_CALUDE_four_square_games_l2247_224705

/-- The number of players in the four-square league -/
def total_players : ℕ := 12

/-- The number of players in each game -/
def players_per_game : ℕ := 6

/-- The number of times two specific players play together -/
def games_together : ℕ := 210

/-- The total number of possible game combinations -/
def total_combinations : ℕ := Nat.choose total_players players_per_game

theorem four_square_games (player1 player2 : Fin total_players) 
  (h_distinct : player1 ≠ player2) :
  (Nat.choose (total_players - 2) (players_per_game - 2) = games_together) ∧
  (total_combinations = Nat.choose total_players players_per_game) ∧
  (2 * games_together = players_per_game * (total_combinations / total_players)) :=
sorry

end NUMINAMATH_CALUDE_four_square_games_l2247_224705


namespace NUMINAMATH_CALUDE_polynomial_derivative_sum_l2247_224784

theorem polynomial_derivative_sum (a₀ a₁ a₂ a₃ a₄ a₅ a₆ : ℝ) :
  (∀ x, (2*x - 1)^6 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5 + a₆*x^6) →
  a₁ + 2*a₂ + 3*a₃ + 4*a₄ + 5*a₅ + 6*a₆ = 12 := by
sorry

end NUMINAMATH_CALUDE_polynomial_derivative_sum_l2247_224784


namespace NUMINAMATH_CALUDE_expression_value_approx_l2247_224720

def x : Real := 2.2
def a : Real := 3.6
def b : Real := 0.48
def c : Real := 2.50
def d : Real := 0.12
def e : Real := 0.09
def f : Real := 0.5

theorem expression_value_approx :
  ∃ (ε : Real), ε > 0 ∧ ε < 0.01 ∧ 
  |3 * x * ((a^2 * b * Real.log c) / (Real.sqrt d * Real.sin e * Real.log f)) + 720.72| < ε :=
by sorry

end NUMINAMATH_CALUDE_expression_value_approx_l2247_224720


namespace NUMINAMATH_CALUDE_min_sum_product_72_l2247_224704

theorem min_sum_product_72 (a b : ℤ) (h : a * b = 72) :
  ∀ x y : ℤ, x * y = 72 → a + b ≤ x + y ∧ ∃ a₀ b₀ : ℤ, a₀ * b₀ = 72 ∧ a₀ + b₀ = -17 :=
by sorry

end NUMINAMATH_CALUDE_min_sum_product_72_l2247_224704


namespace NUMINAMATH_CALUDE_initial_visual_range_proof_l2247_224725

/-- The initial visual range without the telescope -/
def initial_range : ℝ := 50

/-- The visual range with the telescope -/
def telescope_range : ℝ := 150

/-- The percentage increase in visual range -/
def percentage_increase : ℝ := 200

theorem initial_visual_range_proof :
  initial_range = telescope_range / (1 + percentage_increase / 100) :=
by sorry

end NUMINAMATH_CALUDE_initial_visual_range_proof_l2247_224725


namespace NUMINAMATH_CALUDE_rachel_homework_difference_l2247_224739

theorem rachel_homework_difference :
  let math_pages : ℕ := 2
  let reading_pages : ℕ := 3
  let total_pages : ℕ := 15
  let biology_pages : ℕ := total_pages - (math_pages + reading_pages)
  biology_pages - reading_pages = 7 :=
by sorry

end NUMINAMATH_CALUDE_rachel_homework_difference_l2247_224739


namespace NUMINAMATH_CALUDE_diagonals_in_150_degree_polygon_l2247_224763

/-- A polygon where all interior angles are 150 degrees -/
structure RegularPolygon where
  interior_angle : ℝ
  interior_angle_eq : interior_angle = 150

/-- The number of diagonals from one vertex in a RegularPolygon -/
def diagonals_from_vertex (p : RegularPolygon) : ℕ :=
  9

/-- Theorem: In a polygon where all interior angles are 150°, 
    the number of diagonals that can be drawn from one vertex is 9 -/
theorem diagonals_in_150_degree_polygon (p : RegularPolygon) :
  diagonals_from_vertex p = 9 := by
  sorry

end NUMINAMATH_CALUDE_diagonals_in_150_degree_polygon_l2247_224763


namespace NUMINAMATH_CALUDE_part_one_part_two_l2247_224742

-- Define the propositions p and q
def p (a x : ℝ) : Prop := a < x ∧ x < 3 * a

def q (x : ℝ) : Prop := 2 < x ∧ x < 3

-- Part 1
theorem part_one (a x : ℝ) (h1 : a > 0) (h2 : a = 1) (h3 : p a x ∧ q x) :
  2 < x ∧ x < 3 := by sorry

-- Part 2
theorem part_two (a : ℝ) (h1 : a > 0) 
  (h2 : ∀ x, q x → p a x) 
  (h3 : ∃ x, p a x ∧ ¬q x) :
  1 ≤ a ∧ a ≤ 2 := by sorry

end NUMINAMATH_CALUDE_part_one_part_two_l2247_224742


namespace NUMINAMATH_CALUDE_same_color_marble_probability_l2247_224771

theorem same_color_marble_probability : 
  let total_marbles : ℕ := 3 + 6 + 8
  let red_marbles : ℕ := 3
  let white_marbles : ℕ := 6
  let blue_marbles : ℕ := 8
  let drawn_marbles : ℕ := 4
  
  (Nat.choose white_marbles drawn_marbles + Nat.choose blue_marbles drawn_marbles : ℚ) /
  (Nat.choose total_marbles drawn_marbles : ℚ) = 17 / 476 :=
by sorry

end NUMINAMATH_CALUDE_same_color_marble_probability_l2247_224771


namespace NUMINAMATH_CALUDE_vector_operations_l2247_224789

def vector_a : ℝ × ℝ := (2, 0)
def vector_b : ℝ × ℝ := (-1, 3)

theorem vector_operations :
  (vector_a.1 + vector_b.1, vector_a.2 + vector_b.2) = (1, 3) ∧
  (vector_a.1 - vector_b.1, vector_a.2 - vector_b.2) = (3, -3) := by
  sorry

end NUMINAMATH_CALUDE_vector_operations_l2247_224789


namespace NUMINAMATH_CALUDE_combined_mpg_l2247_224791

/-- Combined miles per gallon calculation -/
theorem combined_mpg (alice_mpg bob_mpg alice_miles bob_miles : ℚ) 
  (h1 : alice_mpg = 30)
  (h2 : bob_mpg = 20)
  (h3 : alice_miles = 120)
  (h4 : bob_miles = 180) :
  (alice_miles + bob_miles) / (alice_miles / alice_mpg + bob_miles / bob_mpg) = 300 / 13 := by
  sorry

#eval (120 + 180) / (120 / 30 + 180 / 20) -- For verification

end NUMINAMATH_CALUDE_combined_mpg_l2247_224791


namespace NUMINAMATH_CALUDE_tommy_balloons_l2247_224795

/-- Prove that Tommy started with 26 balloons given the conditions of the problem -/
theorem tommy_balloons (initial : ℕ) (from_mom : ℕ) (after_mom : ℕ) (total : ℕ) : 
  after_mom = 26 → total = 60 → initial + from_mom = total → initial = 26 := by
  sorry

end NUMINAMATH_CALUDE_tommy_balloons_l2247_224795


namespace NUMINAMATH_CALUDE_difference_of_squares_special_case_l2247_224772

theorem difference_of_squares_special_case : (532 * 532) - (531 * 533) = 1 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_special_case_l2247_224772


namespace NUMINAMATH_CALUDE_optimal_seedlings_optimal_seedlings_count_l2247_224769

/-- Represents the profit per pot as a function of the number of seedlings -/
def profit_per_pot (n : ℕ) : ℝ :=
  n * (5 - 0.5 * (n - 4 : ℝ))

/-- The target profit per pot -/
def target_profit : ℝ := 24

/-- Theorem stating that 6 seedlings per pot achieves the target profit while minimizing costs -/
theorem optimal_seedlings :
  (profit_per_pot 6 = target_profit) ∧
  (∀ m : ℕ, m < 6 → profit_per_pot m < target_profit) ∧
  (∀ m : ℕ, m > 6 → profit_per_pot m ≤ target_profit) :=
sorry

/-- Corollary: 6 is the optimal number of seedlings per pot -/
theorem optimal_seedlings_count : ℕ :=
6

end NUMINAMATH_CALUDE_optimal_seedlings_optimal_seedlings_count_l2247_224769


namespace NUMINAMATH_CALUDE_triangle_angle_A_l2247_224708

theorem triangle_angle_A (A B C : ℝ) (a b : ℝ) (angleB : ℝ) : 
  a = Real.sqrt 3 →
  b = Real.sqrt 2 →
  angleB = π / 4 →
  (∃ (angleA : ℝ), (angleA = π / 3 ∨ angleA = 2 * π / 3) ∧ 
    a / Real.sin angleA = b / Real.sin angleB) :=
by sorry

end NUMINAMATH_CALUDE_triangle_angle_A_l2247_224708


namespace NUMINAMATH_CALUDE_quadratic_problem_l2247_224729

-- Define the quadratic function
def f (a : ℝ) (x : ℝ) := a * x^2 + 5 * x - 2

-- Define the solution set condition
def solution_set (a : ℝ) : Set ℝ := {x | 1/2 < x ∧ x < 2}

-- Define the second quadratic function
def g (a : ℝ) (x : ℝ) := a * x^2 - 5 * x + a^2 - 1

-- Theorem statement
theorem quadratic_problem (a : ℝ) :
  (∀ x, f a x > 0 ↔ x ∈ solution_set a) →
  (a = -2 ∧ 
   ∀ x, g a x > 0 ↔ -3 < x ∧ x < 1/2) := by
sorry

end NUMINAMATH_CALUDE_quadratic_problem_l2247_224729


namespace NUMINAMATH_CALUDE_geometric_progression_and_sum_l2247_224767

theorem geometric_progression_and_sum : ∃ x : ℝ,
  let a₁ := 10 + x
  let a₂ := 30 + x
  let a₃ := 60 + x
  (a₂ / a₁ = a₃ / a₂) ∧ (a₁ + a₂ + a₃ = 190) ∧ x = 30 := by
  sorry

end NUMINAMATH_CALUDE_geometric_progression_and_sum_l2247_224767


namespace NUMINAMATH_CALUDE_last_boat_passengers_l2247_224710

/-- The number of people on a boat trip -/
def boat_trip (m : ℕ) : Prop :=
  ∃ (total : ℕ),
    -- Condition 1: m boats with 10 seats each leaves 8 people without seats
    total = 10 * m + 8 ∧
    -- Condition 2 & 3: Using boats with 16 seats each, 1 fewer boat is rented, and last boat is not full
    ∃ (last_boat : ℕ), last_boat > 0 ∧ last_boat < 16 ∧
      total = 16 * (m - 1) + last_boat

/-- The number of people on the last boat with 16 seats -/
theorem last_boat_passengers (m : ℕ) (h : boat_trip m) :
  ∃ (last_boat : ℕ), last_boat = 40 - 6 * m :=
by sorry

end NUMINAMATH_CALUDE_last_boat_passengers_l2247_224710


namespace NUMINAMATH_CALUDE_sequence_nonpositive_l2247_224778

theorem sequence_nonpositive (N : ℕ) (a : ℕ → ℝ)
  (h0 : a 0 = 0)
  (hN : a N = 0)
  (h_rec : ∀ i ∈ Finset.range (N - 1), a (i + 2) - 2 * a (i + 1) + a i = (a (i + 1))^2) :
  ∀ i ∈ Finset.range (N - 1), a (i + 1) ≤ 0 := by
sorry

end NUMINAMATH_CALUDE_sequence_nonpositive_l2247_224778


namespace NUMINAMATH_CALUDE_evaluate_expression_l2247_224792

theorem evaluate_expression : 12.543 - 3.219 + 1.002 = 10.326 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l2247_224792


namespace NUMINAMATH_CALUDE_stratified_sample_male_athletes_l2247_224737

/-- Represents the number of male athletes drawn in a stratified sample -/
def male_athletes_drawn (total_athletes : ℕ) (male_athletes : ℕ) (sample_size : ℕ) : ℕ :=
  (sample_size * male_athletes) / total_athletes

/-- Theorem stating that in a stratified sample of 21 athletes from a population of 84 athletes 
    (48 male and 36 female), the number of male athletes drawn is 12 -/
theorem stratified_sample_male_athletes :
  male_athletes_drawn 84 48 21 = 12 := by
  sorry

#eval male_athletes_drawn 84 48 21

end NUMINAMATH_CALUDE_stratified_sample_male_athletes_l2247_224737


namespace NUMINAMATH_CALUDE_min_value_of_expression_l2247_224753

theorem min_value_of_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 1/a + 1/b = 1) :
  a/(a-1) + 4*b/(b-1) ≥ 9 ∧ ∃ (a₀ b₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ 1/a₀ + 1/b₀ = 1 ∧ a₀/(a₀-1) + 4*b₀/(b₀-1) = 9 :=
sorry

end NUMINAMATH_CALUDE_min_value_of_expression_l2247_224753


namespace NUMINAMATH_CALUDE_ratio_of_repeating_decimals_l2247_224714

/-- Represents a repeating decimal with a two-digit repetend -/
def RepeatingDecimal (a b : ℕ) : ℚ := (10 * a + b : ℚ) / 99

/-- The main theorem stating that the ratio of 0.overline{63} to 0.overline{21} is 3 -/
theorem ratio_of_repeating_decimals : 
  (RepeatingDecimal 6 3) / (RepeatingDecimal 2 1) = 3 := by sorry

end NUMINAMATH_CALUDE_ratio_of_repeating_decimals_l2247_224714


namespace NUMINAMATH_CALUDE_cost_type_B_calculation_l2247_224744

/-- The cost of purchasing type B books given the total number of books and the number of type A books purchased. -/
def cost_type_B (total_books : ℕ) (price_A : ℕ) (price_B : ℕ) (x : ℕ) : ℕ :=
  price_B * (total_books - x)

/-- Theorem stating that the cost of purchasing type B books is 8(100-x) yuan -/
theorem cost_type_B_calculation (x : ℕ) (h : x ≤ 100) :
  cost_type_B 100 10 8 x = 8 * (100 - x) := by
  sorry

end NUMINAMATH_CALUDE_cost_type_B_calculation_l2247_224744


namespace NUMINAMATH_CALUDE_inscribed_circle_radius_rhombus_l2247_224734

/-- The radius of a circle inscribed in a rhombus with given diagonals -/
theorem inscribed_circle_radius_rhombus (d1 d2 : ℝ) (h1 : d1 = 12) (h2 : d2 = 16) :
  let a := Real.sqrt ((d1/2)^2 + (d2/2)^2)
  let r := (d1 * d2) / (4 * a)
  r = 24 / 5 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_circle_radius_rhombus_l2247_224734


namespace NUMINAMATH_CALUDE_fourth_root_unity_sum_l2247_224768

/-- Given a nonreal complex number ω that is a fourth root of unity,
    prove that (1 - ω + ω^3)^4 + (1 + ω - ω^3)^4 = -14 -/
theorem fourth_root_unity_sum (ω : ℂ) 
  (h1 : ω^4 = 1) 
  (h2 : ω ≠ 1 ∧ ω ≠ -1 ∧ ω ≠ Complex.I ∧ ω ≠ -Complex.I) : 
  (1 - ω + ω^3)^4 + (1 + ω - ω^3)^4 = -14 := by
  sorry

end NUMINAMATH_CALUDE_fourth_root_unity_sum_l2247_224768


namespace NUMINAMATH_CALUDE_regression_lines_intersection_l2247_224700

/-- A linear regression line -/
structure RegressionLine where
  slope : ℝ
  intercept : ℝ

/-- The point where a regression line passes through -/
def passes_through (l : RegressionLine) (x y : ℝ) : Prop :=
  y = l.slope * x + l.intercept

theorem regression_lines_intersection
  (l₁ l₂ : RegressionLine)
  (s t : ℝ)
  (h₁ : passes_through l₁ s t)
  (h₂ : passes_through l₂ s t) :
  ∃ (x y : ℝ), passes_through l₁ x y ∧ passes_through l₂ x y ∧ x = s ∧ y = t :=
sorry

end NUMINAMATH_CALUDE_regression_lines_intersection_l2247_224700


namespace NUMINAMATH_CALUDE_partner_A_share_is_8160_l2247_224746

/-- Calculates the share of profit for partner A in a business partnership --/
def partner_A_share (total_profit : ℚ) (A_investment : ℚ) (B_investment : ℚ) (management_fee_percent : ℚ) : ℚ :=
  let management_fee := total_profit * management_fee_percent / 100
  let remaining_profit := total_profit - management_fee
  let total_investment := A_investment + B_investment
  let A_proportion := A_investment / total_investment
  management_fee + (remaining_profit * A_proportion)

/-- Theorem stating that partner A's share is 8160 Rs under given conditions --/
theorem partner_A_share_is_8160 :
  partner_A_share 9600 5000 1000 10 = 8160 := by
  sorry

end NUMINAMATH_CALUDE_partner_A_share_is_8160_l2247_224746


namespace NUMINAMATH_CALUDE_sum_interior_angles_regular_polygon_l2247_224732

/-- For a regular polygon with exterior angles of 40 degrees, 
    the sum of interior angles is 1260 degrees. -/
theorem sum_interior_angles_regular_polygon : 
  ∀ n : ℕ, 
  n > 2 → 
  360 / n = 40 → 
  (n - 2) * 180 = 1260 :=
by
  sorry

end NUMINAMATH_CALUDE_sum_interior_angles_regular_polygon_l2247_224732


namespace NUMINAMATH_CALUDE_cauchy_equation_on_X_l2247_224738

-- Define the set X
def X : Set ℝ := {x : ℝ | ∃ (a b : ℤ), x = a + b * Real.sqrt 2}

-- Define the Cauchy equation property
def is_cauchy (f : X → ℝ) : Prop :=
  ∀ (x y : X), f (⟨x + y, sorry⟩) = f x + f y

-- State the theorem
theorem cauchy_equation_on_X (f : X → ℝ) (hf : is_cauchy f) :
  ∀ (a b : ℤ), f ⟨a + b * Real.sqrt 2, sorry⟩ = a * f ⟨1, sorry⟩ + b * f ⟨Real.sqrt 2, sorry⟩ :=
sorry

end NUMINAMATH_CALUDE_cauchy_equation_on_X_l2247_224738


namespace NUMINAMATH_CALUDE_not_closed_sequence_3_pow_arithmetic_closed_sequence_iff_l2247_224782

/-- Definition of a closed sequence -/
def is_closed_sequence (a : ℕ → ℝ) : Prop :=
  ∀ m n : ℕ, ∃ k : ℕ, a m + a n = a k

/-- The sequence a_n = 3^n is not a closed sequence -/
theorem not_closed_sequence_3_pow : ¬ is_closed_sequence (λ n => 3^n) := by sorry

/-- Definition of an arithmetic sequence -/
def is_arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + d

/-- Necessary and sufficient condition for an arithmetic sequence to be a closed sequence -/
theorem arithmetic_closed_sequence_iff (a : ℕ → ℝ) (d : ℝ) :
  is_arithmetic_sequence a d →
  (is_closed_sequence a ↔ ∃ m : ℤ, m ≥ -1 ∧ a 1 = m * d) := by sorry

end NUMINAMATH_CALUDE_not_closed_sequence_3_pow_arithmetic_closed_sequence_iff_l2247_224782


namespace NUMINAMATH_CALUDE_angle_A_is_pi_over_three_max_area_when_a_is_four_l2247_224707

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the given condition
def condition (t : Triangle) : Prop :=
  (t.a + t.b) * (Real.sin t.A - Real.sin t.B) = t.c * (Real.sin t.C - Real.sin t.B)

-- Theorem for part 1
theorem angle_A_is_pi_over_three (t : Triangle) (h : condition t) : t.A = π / 3 := by
  sorry

-- Theorem for part 2
theorem max_area_when_a_is_four (t : Triangle) (h1 : condition t) (h2 : t.a = 4) :
  ∃ (S : ℝ), S = 4 * Real.sqrt 3 ∧ ∀ (S' : ℝ), S' = 1/2 * t.b * t.c * Real.sin t.A → S' ≤ S := by
  sorry

end NUMINAMATH_CALUDE_angle_A_is_pi_over_three_max_area_when_a_is_four_l2247_224707


namespace NUMINAMATH_CALUDE_inequality_bound_l2247_224741

theorem inequality_bound (x y z w : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (hw : w > 0) :
  Real.sqrt (x / (y + z + w)) + Real.sqrt (y / (x + z + w)) + 
  Real.sqrt (z / (x + y + w)) + Real.sqrt (w / (x + y + z)) < 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_bound_l2247_224741


namespace NUMINAMATH_CALUDE_second_ruler_alignment_l2247_224717

/-- Represents a small ruler in relation to the large ruler -/
structure SmallRuler where
  large_units : ℚ  -- Number of units on the large ruler
  small_units : ℚ  -- Number of units on the small ruler

/-- Represents the set square system with two small rulers and a large ruler -/
structure SetSquare where
  first_ruler : SmallRuler
  second_ruler : SmallRuler
  point_b : ℚ  -- Position of point B on the large ruler

/-- Main theorem statement -/
theorem second_ruler_alignment (s : SetSquare) : 
  s.first_ruler = SmallRuler.mk 11 10 →   -- First ruler divides 11 units into 10
  s.second_ruler = SmallRuler.mk 9 10 →   -- Second ruler divides 9 units into 10
  18 < s.point_b ∧ s.point_b < 19 →       -- Point B is between 18 and 19
  (s.point_b + 3 * s.first_ruler.large_units / s.first_ruler.small_units).floor = 
    (s.point_b + 3 * s.first_ruler.large_units / s.first_ruler.small_units) →  
    -- 3rd unit of first ruler coincides with an integer
  ∃ k : ℕ, (s.point_b + 7 * s.second_ruler.large_units / s.second_ruler.small_units) = ↑k :=
by sorry

end NUMINAMATH_CALUDE_second_ruler_alignment_l2247_224717


namespace NUMINAMATH_CALUDE_pencils_in_drawer_proof_l2247_224786

/-- The number of pencils initially in the drawer -/
def initial_drawer_pencils : ℕ := 43

/-- The number of pencils initially on the desk -/
def initial_desk_pencils : ℕ := 19

/-- The number of pencils added to the desk -/
def added_desk_pencils : ℕ := 16

/-- The total number of pencils -/
def total_pencils : ℕ := 78

/-- Theorem stating that the initial number of pencils in the drawer is correct -/
theorem pencils_in_drawer_proof :
  initial_drawer_pencils = total_pencils - (initial_desk_pencils + added_desk_pencils) :=
by sorry

end NUMINAMATH_CALUDE_pencils_in_drawer_proof_l2247_224786


namespace NUMINAMATH_CALUDE_max_sum_after_erasing_l2247_224787

-- Define the initial set of numbers
def initial_numbers : List ℕ := List.range 13 |>.map (· + 4)

-- Define a function to check if a list can be divided into groups with equal sums
def can_be_divided_equally (numbers : List ℕ) : Prop :=
  ∃ (k : ℕ) (groups : List (List ℕ)),
    k > 1 ∧
    groups.length = k ∧
    groups.all (λ group ↦ group.sum = (numbers.sum / k)) ∧
    groups.join.toFinset = numbers.toFinset

-- Define the theorem
theorem max_sum_after_erasing (numbers : List ℕ) :
  numbers.sum = 121 →
  numbers ⊆ initial_numbers →
  ¬ can_be_divided_equally numbers →
  ∀ (other_numbers : List ℕ),
    other_numbers ⊆ initial_numbers →
    other_numbers.sum > 121 →
    can_be_divided_equally other_numbers :=
sorry

end NUMINAMATH_CALUDE_max_sum_after_erasing_l2247_224787


namespace NUMINAMATH_CALUDE_part1_part2_l2247_224764

-- Definition of "shifted equation"
def is_shifted_equation (f g : ℝ → ℝ) : Prop :=
  ∃ x y : ℝ, f x = 0 ∧ g y = 0 ∧ x = y + 1

-- Part 1
theorem part1 : is_shifted_equation (λ x => 2*x + 1) (λ x => 2*x + 3) := by sorry

-- Part 2
theorem part2 : ∃ m : ℝ, 
  is_shifted_equation 
    (λ x => 3*(x-1) - m - (m+3)/2) 
    (λ x => 2*(x-3) - 1 - (3-(x+1))) ∧ 
  m = 5 := by sorry

end NUMINAMATH_CALUDE_part1_part2_l2247_224764


namespace NUMINAMATH_CALUDE_min_value_of_f_min_value_of_sum_squares_l2247_224736

-- Define the function f
def f (x : ℝ) : ℝ := 2 * abs (x - 1) + abs (2 * x + 1)

-- Theorem 1: The minimum value of f(x) is 3
theorem min_value_of_f : ∃ k : ℝ, k = 3 ∧ ∀ x : ℝ, f x ≥ k :=
sorry

-- Theorem 2: Minimum value of a² + b² + c² given the conditions
theorem min_value_of_sum_squares :
  ∀ a b c : ℝ,
  a > 0 → b > 0 → c > 0 →
  3 * a + 2 * b + c = 3 →
  a^2 + b^2 + c^2 ≥ 9/14 :=
sorry

end NUMINAMATH_CALUDE_min_value_of_f_min_value_of_sum_squares_l2247_224736


namespace NUMINAMATH_CALUDE_election_winner_percentage_l2247_224731

theorem election_winner_percentage (total_votes winner_votes margin : ℕ) : 
  winner_votes = 992 →
  margin = 384 →
  total_votes = winner_votes + (winner_votes - margin) →
  (winner_votes : ℚ) / (total_votes : ℚ) = 62 / 100 :=
by
  sorry

end NUMINAMATH_CALUDE_election_winner_percentage_l2247_224731


namespace NUMINAMATH_CALUDE_power_addition_l2247_224754

theorem power_addition (x y : ℝ) (a b : ℝ) 
  (h1 : (4 : ℝ) ^ x = a) 
  (h2 : (4 : ℝ) ^ y = b) : 
  (4 : ℝ) ^ (x + y) = a * b := by
  sorry

end NUMINAMATH_CALUDE_power_addition_l2247_224754


namespace NUMINAMATH_CALUDE_monica_money_exchange_l2247_224723

def exchange_rate : ℚ := 8 / 5

theorem monica_money_exchange (x : ℚ) : 
  (exchange_rate * x - 40 = x) → x = 200 := by
  sorry

end NUMINAMATH_CALUDE_monica_money_exchange_l2247_224723


namespace NUMINAMATH_CALUDE_dividend_proof_l2247_224733

theorem dividend_proof (dividend quotient remainder : ℕ) : 
  dividend / 9 = quotient → 
  dividend % 9 = remainder →
  quotient = 9 →
  remainder = 2 →
  dividend = 83 := by
sorry

end NUMINAMATH_CALUDE_dividend_proof_l2247_224733


namespace NUMINAMATH_CALUDE_unit_digit_of_3_to_58_l2247_224788

theorem unit_digit_of_3_to_58 : 3^58 % 10 = 9 := by
  sorry

end NUMINAMATH_CALUDE_unit_digit_of_3_to_58_l2247_224788


namespace NUMINAMATH_CALUDE_buffet_dishes_l2247_224798

theorem buffet_dishes (mango_salsa_dishes : ℕ) (mango_jelly_dishes : ℕ) (oliver_edible_dishes : ℕ) 
  (fresh_mango_ratio : ℚ) (oliver_pick_out_dishes : ℕ) :
  mango_salsa_dishes = 3 →
  mango_jelly_dishes = 1 →
  fresh_mango_ratio = 1 / 6 →
  oliver_pick_out_dishes = 2 →
  oliver_edible_dishes = 28 →
  ∃ (total_dishes : ℕ), 
    total_dishes = 36 ∧ 
    (fresh_mango_ratio * total_dishes : ℚ).num = oliver_pick_out_dishes + 
      (total_dishes - oliver_edible_dishes - mango_salsa_dishes - mango_jelly_dishes) :=
by sorry

end NUMINAMATH_CALUDE_buffet_dishes_l2247_224798
