import Mathlib

namespace no_nonzero_solutions_l3116_311619

theorem no_nonzero_solutions (a b : ℝ) (ha : a ≥ 0) (hb : b ≥ 0) :
  (Real.sqrt (a^2 + b^2) = 0 ↔ a = 0 ∧ b = 0) ∧
  (Real.sqrt (a^2 + b^2) = (a + b) / 2 ↔ a = 0 ∧ b = 0) ∧
  (Real.sqrt (a^2 + b^2) = Real.sqrt a + Real.sqrt b ↔ a = 0 ∧ b = 0) ∧
  (Real.sqrt (a^2 + b^2) = a + b - 1 ↔ a = 0 ∧ b = 0) :=
by sorry

end no_nonzero_solutions_l3116_311619


namespace all_boys_are_brothers_l3116_311671

/-- A type representing the group of boys -/
def Boys := Fin 7

/-- A relation indicating whether two boys are brothers -/
def is_brother (a b : Boys) : Prop := sorry

/-- Axiom: Each boy has at least 3 brothers among the others -/
axiom at_least_three_brothers (b : Boys) : 
  ∃ (s : Finset Boys), s.card ≥ 3 ∧ ∀ x ∈ s, x ≠ b ∧ is_brother x b

/-- Theorem: All seven boys are brothers -/
theorem all_boys_are_brothers : ∀ (a b : Boys), is_brother a b :=
sorry

end all_boys_are_brothers_l3116_311671


namespace trigonometric_identity_l3116_311605

theorem trigonometric_identity (c d : ℝ) (θ : ℝ) 
  (h : (Real.sin θ)^2 / c + (Real.cos θ)^2 / d = 1 / (c + d)) 
  (h1 : c ≠ 0) (h2 : d ≠ 0) (h3 : d ≠ 1) :
  (Real.sin θ)^4 / c^2 + (Real.cos θ)^4 / d^2 = 2 * (c - d)^2 / (c^2 * d^2 * (d - 1)^2) := by
  sorry

end trigonometric_identity_l3116_311605


namespace rotation_90_degrees_l3116_311680

def rotate90(z : ℂ) : ℂ := z * Complex.I

theorem rotation_90_degrees :
  rotate90 (-4 - 2 * Complex.I) = 2 - 4 * Complex.I :=
by sorry

end rotation_90_degrees_l3116_311680


namespace perpendicular_tangents_imply_a_value_l3116_311673

/-- Given two curves C₁ and C₂, where C₁ is y = ax³ - x² + 2x and C₂ is y = e^x,
    if their tangent lines are perpendicular at x = 1, then a = -1/(3e) -/
theorem perpendicular_tangents_imply_a_value (a : ℝ) :
  let C₁ : ℝ → ℝ := λ x ↦ a * x^3 - x^2 + 2*x
  let C₂ : ℝ → ℝ := λ x ↦ Real.exp x
  let tangent_C₁ : ℝ := 3*a - 2 + 2  -- Derivative of C₁ at x = 1
  let tangent_C₂ : ℝ := Real.exp 1   -- Derivative of C₂ at x = 1
  (tangent_C₁ * tangent_C₂ = -1) →   -- Condition for perpendicular tangents
  a = -1 / (3 * Real.exp 1) :=
by sorry

end perpendicular_tangents_imply_a_value_l3116_311673


namespace ones_digit_of_largest_power_of_two_dividing_32_factorial_l3116_311620

def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

def largest_power_of_two_dividing (n : ℕ) : ℕ :=
  -- This function is not implemented, but represents the concept
  sorry

def ones_digit (n : ℕ) : ℕ :=
  n % 10

theorem ones_digit_of_largest_power_of_two_dividing_32_factorial :
  ones_digit (largest_power_of_two_dividing (factorial 32)) = 8 := by
  sorry

end ones_digit_of_largest_power_of_two_dividing_32_factorial_l3116_311620


namespace small_cuboid_width_is_four_l3116_311667

/-- Represents the dimensions of a cuboid -/
structure CuboidDimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the volume of a cuboid given its dimensions -/
def cuboidVolume (d : CuboidDimensions) : ℝ :=
  d.length * d.width * d.height

theorem small_cuboid_width_is_four
  (large : CuboidDimensions)
  (small_length : ℝ)
  (small_height : ℝ)
  (num_small_cuboids : ℕ)
  (h1 : large.length = 16)
  (h2 : large.width = 10)
  (h3 : large.height = 12)
  (h4 : small_length = 5)
  (h5 : small_height = 3)
  (h6 : num_small_cuboids = 32)
  (h7 : ∃ (small_width : ℝ),
    cuboidVolume large = num_small_cuboids * cuboidVolume
      { length := small_length
        width := small_width
        height := small_height }) :
  ∃ (small_width : ℝ), small_width = 4 := by
  sorry

end small_cuboid_width_is_four_l3116_311667


namespace triangle_angle_calculation_l3116_311683

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    if a = 1, b = √2, and A = 30°, then B = 45° or B = 135°. -/
theorem triangle_angle_calculation (a b c A B C : Real) : 
  a = 1 → b = Real.sqrt 2 → A = π / 6 → 
  (B = π / 4 ∨ B = 3 * π / 4) := by
  sorry


end triangle_angle_calculation_l3116_311683


namespace solution_to_equation_l3116_311629

theorem solution_to_equation (x y z : ℝ) 
  (hx : x > 1) (hy : y > 1) (hz : z > 1)
  (h_eq : x + y + z + 3 / (x - 1) + 3 / (y - 1) + 3 / (z - 1) = 
          2 * (Real.sqrt (x + 2) + Real.sqrt (y + 2) + Real.sqrt (z + 2))) :
  x = (3 + Real.sqrt 13) / 2 ∧ 
  y = (3 + Real.sqrt 13) / 2 ∧ 
  z = (3 + Real.sqrt 13) / 2 := by
sorry

end solution_to_equation_l3116_311629


namespace new_person_weight_l3116_311675

/-- Given a group of 8 people where one person weighing 55 kg is replaced,
    if the average weight increases by 2.5 kg, then the new person weighs 75 kg. -/
theorem new_person_weight (initial_total : ℝ) (new_weight : ℝ) : 
  (initial_total - 55 + new_weight) / 8 = initial_total / 8 + 2.5 →
  new_weight = 75 := by
sorry

end new_person_weight_l3116_311675


namespace magical_stack_with_79_fixed_l3116_311631

/-- A stack of cards is magical if it satisfies certain conditions -/
def magical_stack (n : ℕ) : Prop :=
  ∃ (card_position : ℕ → ℕ),
    (∀ i, i ≤ 2*n → card_position i ≤ 2*n) ∧
    (∃ i ≤ n, card_position i = i) ∧
    (∃ i > n, i ≤ 2*n ∧ card_position i = i) ∧
    (∀ i ≤ 2*n, i % 2 = 1 → card_position i ≤ n) ∧
    (∀ i ≤ 2*n, i % 2 = 0 → card_position i > n)

theorem magical_stack_with_79_fixed (n : ℕ) :
  magical_stack n ∧ n ≥ 79 ∧ (∃ card_position : ℕ → ℕ, card_position 79 = 79) →
  2 * n = 236 :=
sorry

end magical_stack_with_79_fixed_l3116_311631


namespace min_trucks_required_l3116_311679

/-- Represents the weight capacity of a truck in tons -/
def truck_capacity : ℝ := 3

/-- Represents the total weight of all boxes in tons -/
def total_weight : ℝ := 10

/-- Represents the maximum weight of a single box in tons -/
def max_box_weight : ℝ := 1

/-- The minimum number of trucks required -/
def min_trucks : ℕ := 5

/-- Theorem stating that the minimum number of trucks required is 5 -/
theorem min_trucks_required :
  ∀ (box_weights : List ℝ),
    (box_weights.sum = total_weight) →
    (∀ w ∈ box_weights, w ≤ max_box_weight) →
    (∀ n : ℕ, n < min_trucks → n * truck_capacity < total_weight) →
    (min_trucks * truck_capacity ≥ total_weight) :=
by sorry

end min_trucks_required_l3116_311679


namespace right_triangle_geometric_sequence_ratio_l3116_311676

theorem right_triangle_geometric_sequence_ratio :
  ∀ (a b c : ℝ),
    a > 0 →
    b > 0 →
    c > 0 →
    a < b →
    b < c →
    a^2 + b^2 = c^2 →
    (∃ r : ℝ, r > 1 ∧ b = a * r ∧ c = a * r^2) →
    c / a = (1 + Real.sqrt 5) / 2 :=
by sorry

end right_triangle_geometric_sequence_ratio_l3116_311676


namespace tan_theta_in_terms_of_x_l3116_311626

theorem tan_theta_in_terms_of_x (θ : Real) (x : Real) 
  (h_acute : 0 < θ ∧ θ < Real.pi / 2)
  (h_sin : Real.sin (θ / 2) = Real.sqrt ((x - 1) / (2 * x)))
  (h_x_pos : x > 0) : 
  Real.tan θ = Real.sqrt (x^2 - 1) := by
  sorry

end tan_theta_in_terms_of_x_l3116_311626


namespace valid_sets_l3116_311600

theorem valid_sets (A : Set ℕ) : 
  (∀ m n : ℕ, m + n ∈ A → m * n ∈ A) ↔ 
  A = ∅ ∨ A = {0} ∨ A = {0, 1} ∨ A = {0, 1, 2} ∨ 
  A = {0, 1, 2, 3} ∨ A = {0, 1, 2, 3, 4} ∨ A = Set.univ :=
sorry

end valid_sets_l3116_311600


namespace sequence_term_number_l3116_311630

theorem sequence_term_number : 
  let a : ℕ → ℝ := fun n => Real.sqrt (2 * n - 1)
  ∃ n : ℕ, n = 23 ∧ a n = 3 * Real.sqrt 5 := by
  sorry

end sequence_term_number_l3116_311630


namespace parallel_iff_plane_intersects_parallel_transitive_l3116_311614

-- Define the concept of a line in 3D space
def Line : Type := ℝ × ℝ × ℝ → Prop

-- Define the concept of a plane in 3D space
def Plane : Type := ℝ × ℝ × ℝ → Prop

-- Define parallelism for lines
def parallel (a b : Line) : Prop := sorry

-- Define intersection between a plane and a line
def intersects (p : Plane) (l : Line) : Prop := sorry

-- Define unique intersection
def uniqueIntersection (p : Plane) (l : Line) : Prop := sorry

theorem parallel_iff_plane_intersects (a b : Line) : 
  parallel a b ↔ ∀ (p : Plane), intersects p a → uniqueIntersection p b := by sorry

theorem parallel_transitive (a b c : Line) :
  parallel a b → parallel b c → parallel a c := by sorry

end parallel_iff_plane_intersects_parallel_transitive_l3116_311614


namespace class_test_probability_l3116_311653

theorem class_test_probability (p_first p_second p_neither : ℝ) 
  (h1 : p_first = 0.63)
  (h2 : p_second = 0.49)
  (h3 : p_neither = 0.20) :
  p_first + p_second - (1 - p_neither) = 0.32 := by
    sorry

end class_test_probability_l3116_311653


namespace kendy_bank_transactions_l3116_311660

theorem kendy_bank_transactions (X : ℝ) : 
  X - 60 - 30 - 0.25 * (X - 60 - 30) - 10 = 100 → X = 236.67 :=
by sorry

end kendy_bank_transactions_l3116_311660


namespace shifted_line_equation_l3116_311618

/-- Represents a linear function in the form y = mx + b -/
structure LinearFunction where
  slope : ℝ
  yIntercept : ℝ

/-- Shifts a linear function horizontally and vertically -/
def shiftLinearFunction (f : LinearFunction) (horizontalShift : ℝ) (verticalShift : ℝ) : LinearFunction :=
  { slope := f.slope
    yIntercept := f.slope * (-horizontalShift) + f.yIntercept + verticalShift }

theorem shifted_line_equation (f : LinearFunction) :
  let f' := shiftLinearFunction f 2 3
  f.slope = 2 ∧ f.yIntercept = -3 → f'.slope = 2 ∧ f'.yIntercept = 4 := by
  sorry

#check shifted_line_equation

end shifted_line_equation_l3116_311618


namespace min_cost_is_58984_l3116_311639

/-- Represents a travel agency with its pricing structure -/
structure TravelAgency where
  name : String
  young_age_limit : Nat
  young_price : Nat
  adult_price : Nat
  discount_or_commission : Float
  is_discount : Bool

/-- Represents a family member -/
structure FamilyMember where
  age : Nat

/-- Calculates the total cost for a family's vacation with a given travel agency -/
def calculate_total_cost (agency : TravelAgency) (family : List FamilyMember) : Float :=
  sorry

/-- The Dorokhov family -/
def dorokhov_family : List FamilyMember :=
  [⟨35⟩, ⟨35⟩, ⟨5⟩]  -- Assuming parents are 35 years old

/-- Globus travel agency -/
def globus : TravelAgency :=
  ⟨"Globus", 5, 11200, 25400, 0.02, true⟩

/-- Around the World travel agency -/
def around_the_world : TravelAgency :=
  ⟨"Around the World", 6, 11400, 23500, 0.01, false⟩

/-- Theorem: The minimum cost for the Dorokhov family's vacation is 58984 rubles -/
theorem min_cost_is_58984 :
  min (calculate_total_cost globus dorokhov_family)
      (calculate_total_cost around_the_world dorokhov_family) = 58984 :=
  sorry

end min_cost_is_58984_l3116_311639


namespace division_by_fraction_problem_solution_l3116_311648

theorem division_by_fraction (a b c : ℚ) (hb : b ≠ 0) (hc : c ≠ 0) :
  a / (b / c) = (a * c) / b :=
by sorry

theorem problem_solution : (5 : ℚ) / ((7 : ℚ) / 13) = 65 / 7 :=
by sorry

end division_by_fraction_problem_solution_l3116_311648


namespace x_cubed_plus_x_cubed_l3116_311604

theorem x_cubed_plus_x_cubed (x : ℝ) (h : x > 0) : 
  (x^3 + x^3 = 2*x^3) ∧ 
  (x^3 + x^3 ≠ x^6) ∧ 
  (x^3 + x^3 ≠ (3*x)^3) ∧ 
  (x^3 + x^3 ≠ (x^3)^2) :=
by sorry

end x_cubed_plus_x_cubed_l3116_311604


namespace divisibility_condition_l3116_311698

theorem divisibility_condition (a b : ℕ+) : 
  (∃ k : ℕ, (a^2 * b + a + b : ℕ) = k * (a * b^2 + b + 7)) ↔ 
  ((a = 11 ∧ b = 1) ∨ (a = 49 ∧ b = 1) ∨ (∃ k : ℕ+, a = 7 * k^2 ∧ b = 7 * k)) :=
by sorry

end divisibility_condition_l3116_311698


namespace arithmetic_expression_equality_l3116_311634

theorem arithmetic_expression_equality : 9 - 8 + 7 * 6 + 5 - 4 * 3 + 2 - 1 = 37 := by
  sorry

end arithmetic_expression_equality_l3116_311634


namespace batch_size_proof_l3116_311615

/-- The number of parts in the batch -/
def total_parts : ℕ := 1150

/-- The fraction of work A completes when cooperating with the master -/
def a_work_fraction : ℚ := 1/5

/-- The fraction of work B completes when cooperating with the master -/
def b_work_fraction : ℚ := 2/5

/-- The number of fewer parts B processes when A joins -/
def b_fewer_parts : ℕ := 60

theorem batch_size_proof :
  (b_work_fraction * total_parts : ℚ) - 
  ((1 - a_work_fraction - b_work_fraction) / 
   (1 + (1 - a_work_fraction - b_work_fraction) / a_work_fraction) * total_parts : ℚ) = 
  b_fewer_parts := by sorry

end batch_size_proof_l3116_311615


namespace transformed_point_difference_l3116_311689

def rotate90CounterClockwise (x y xc yc : ℝ) : ℝ × ℝ :=
  (xc - (y - yc), yc + (x - xc))

def reflectAboutYEqualsX (x y : ℝ) : ℝ × ℝ :=
  (y, x)

theorem transformed_point_difference (a b : ℝ) :
  let (x1, y1) := rotate90CounterClockwise a b 2 3
  let (x2, y2) := reflectAboutYEqualsX x1 y1
  (x2 = 4 ∧ y2 = 1) → b - a = 1 := by
  sorry

end transformed_point_difference_l3116_311689


namespace min_value_theorem_l3116_311691

theorem min_value_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : 1/a + 1/b = 1) :
  ∃ (min : ℝ), min = 2 * Real.sqrt 6 ∧ ∀ (x : ℝ), x = 3/(a-1) + 2/(b-1) → x ≥ min :=
sorry

end min_value_theorem_l3116_311691


namespace reservoir_ratio_l3116_311655

theorem reservoir_ratio : 
  ∀ (total_capacity normal_level end_month_amount : ℝ),
  end_month_amount = 6 →
  end_month_amount = 0.6 * total_capacity →
  normal_level = total_capacity - 5 →
  end_month_amount / normal_level = 1.2 := by
sorry

end reservoir_ratio_l3116_311655


namespace max_students_distribution_l3116_311610

theorem max_students_distribution (pens pencils : ℕ) (h1 : pens = 1340) (h2 : pencils = 1280) :
  (∃ (students : ℕ), students > 0 ∧ pens % students = 0 ∧ pencils % students = 0 ∧
    ∀ (n : ℕ), n > students → (pens % n ≠ 0 ∨ pencils % n ≠ 0)) ↔
  (∃ (max_students : ℕ), max_students = Nat.gcd pens pencils) :=
sorry

end max_students_distribution_l3116_311610


namespace nested_function_evaluation_l3116_311694

def a (k : ℕ) : ℕ := (k + 1)^2

theorem nested_function_evaluation :
  let k : ℕ := 1
  a (a (a (a k))) = 458329 := by
  sorry

end nested_function_evaluation_l3116_311694


namespace stratified_sampling_group_b_l3116_311668

theorem stratified_sampling_group_b (total_cities : ℕ) (group_b_cities : ℕ) (sample_size : ℕ) :
  total_cities = 24 →
  group_b_cities = 12 →
  sample_size = 6 →
  (group_b_cities * sample_size) / total_cities = 3 :=
by sorry

end stratified_sampling_group_b_l3116_311668


namespace largest_number_with_digit_sum_14_l3116_311658

def is_valid_number (n : ℕ) : Prop :=
  ∀ d : ℕ, d ∈ n.digits 10 → d = 2 ∨ d = 3

def digit_sum (n : ℕ) : ℕ :=
  (n.digits 10).sum

theorem largest_number_with_digit_sum_14 :
  ∀ n : ℕ, 
    is_valid_number n → 
    digit_sum n = 14 → 
    n ≤ 3222233 :=
by
  sorry

end largest_number_with_digit_sum_14_l3116_311658


namespace skipping_competition_probability_l3116_311693

theorem skipping_competition_probability :
  let total_boys : ℕ := 4
  let total_girls : ℕ := 6
  let selected_boys : ℕ := 2
  let selected_girls : ℕ := 2
  let total_selections : ℕ := (Nat.choose total_boys selected_boys) * (Nat.choose total_girls selected_girls)
  let selections_without_A_and_B : ℕ := (Nat.choose (total_boys - 1) selected_boys) * (Nat.choose (total_girls - 1) selected_girls)
  (total_selections - selections_without_A_and_B) / total_selections = 2 / 3 :=
by sorry

end skipping_competition_probability_l3116_311693


namespace min_value_of_f_l3116_311627

-- Define the quadratic function
def f (x : ℝ) : ℝ := x^2 + 6*x + 9

-- Theorem stating that the minimum value of f is 0
theorem min_value_of_f :
  ∃ (x₀ : ℝ), ∀ (x : ℝ), f x₀ ≤ f x ∧ f x₀ = 0 :=
sorry

end min_value_of_f_l3116_311627


namespace frank_cans_total_l3116_311638

/-- The number of cans Frank picked up on Saturday and Sunday combined -/
def total_cans (saturday_bags : ℕ) (sunday_bags : ℕ) (cans_per_bag : ℕ) : ℕ :=
  (saturday_bags + sunday_bags) * cans_per_bag

/-- Theorem stating that Frank picked up 40 cans in total -/
theorem frank_cans_total : total_cans 5 3 5 = 40 := by
  sorry

end frank_cans_total_l3116_311638


namespace tea_price_calculation_l3116_311625

/-- The price of the first variety of tea in rupees per kg -/
def first_tea_price : ℝ := 126

/-- The price of the second variety of tea in rupees per kg -/
def second_tea_price : ℝ := 135

/-- The price of the third variety of tea in rupees per kg -/
def third_tea_price : ℝ := 175.5

/-- The price of the mixture in rupees per kg -/
def mixture_price : ℝ := 153

/-- The ratio of the first variety in the mixture -/
def first_ratio : ℝ := 1

/-- The ratio of the second variety in the mixture -/
def second_ratio : ℝ := 1

/-- The ratio of the third variety in the mixture -/
def third_ratio : ℝ := 2

theorem tea_price_calculation :
  first_tea_price * first_ratio + 
  second_tea_price * second_ratio + 
  third_tea_price * third_ratio = 
  mixture_price * (first_ratio + second_ratio + third_ratio) := by
  sorry

#check tea_price_calculation

end tea_price_calculation_l3116_311625


namespace second_class_average_l3116_311645

/-- Given two classes of students, this theorem proves that the average mark of the second class
    is 80, based on the given conditions. -/
theorem second_class_average (n₁ n₂ : ℕ) (avg₁ avg_total : ℚ) : 
  n₁ = 30 →
  n₂ = 50 →
  avg₁ = 40 →
  avg_total = 65 →
  let total_students : ℕ := n₁ + n₂
  let total_marks : ℚ := avg_total * total_students
  let first_class_marks : ℚ := avg₁ * n₁
  let second_class_marks : ℚ := total_marks - first_class_marks
  let avg₂ : ℚ := second_class_marks / n₂
  avg₂ = 80 := by sorry

end second_class_average_l3116_311645


namespace max_areas_circular_disk_l3116_311688

/-- 
Given a circular disk divided by 2n equally spaced radii and two secant lines 
that do not intersect at the same point on the circumference, the maximum number 
of non-overlapping areas into which the disk can be divided is 4n + 4.
-/
theorem max_areas_circular_disk (n : ℕ) : ℕ := by
  sorry

#check max_areas_circular_disk

end max_areas_circular_disk_l3116_311688


namespace hyperbola_right_directrix_l3116_311651

/-- Given a parabola and a hyperbola with a shared focus, this theorem proves 
    the equation of the right directrix of the hyperbola. -/
theorem hyperbola_right_directrix 
  (a : ℝ) 
  (h_a_pos : a > 0) 
  (h_focus : ∀ x y : ℝ, y^2 = 8*x → x^2/a^2 - y^2/3 = 1 → x = 2 ∧ y = 0) :
  ∃ x : ℝ, x = 1/2 ∧ 
    ∀ y : ℝ, (∃ t : ℝ, t^2/a^2 - y^2/3 = 1 ∧ t > x) → 
      x = a^2 / (2 * (a^2 + 3)^(1/2)) := by
  sorry

end hyperbola_right_directrix_l3116_311651


namespace sin_2theta_value_l3116_311656

theorem sin_2theta_value (θ : Real) (h : Real.cos θ + Real.sin θ = 3/2) : 
  Real.sin (2 * θ) = 5/4 := by
  sorry

end sin_2theta_value_l3116_311656


namespace sum_of_four_triangles_l3116_311686

/-- The value of a square -/
def square_value : ℝ := sorry

/-- The value of a triangle -/
def triangle_value : ℝ := sorry

/-- All squares have the same value -/
axiom square_constant : ∀ s : ℝ, s = square_value

/-- All triangles have the same value -/
axiom triangle_constant : ∀ t : ℝ, t = triangle_value

/-- First equation: square + triangle + square + triangle + square = 27 -/
axiom equation_1 : 3 * square_value + 2 * triangle_value = 27

/-- Second equation: triangle + square + triangle + square + triangle = 23 -/
axiom equation_2 : 2 * square_value + 3 * triangle_value = 23

/-- The sum of four triangles equals 12 -/
theorem sum_of_four_triangles : 4 * triangle_value = 12 := by sorry

end sum_of_four_triangles_l3116_311686


namespace farm_tax_percentage_l3116_311641

theorem farm_tax_percentage (total_tax collection_tax : ℝ) : 
  total_tax > 0 → 
  collection_tax > 0 → 
  collection_tax ≤ total_tax → 
  (collection_tax / total_tax) * 100 = 12.5 → 
  total_tax = 3840 ∧ collection_tax = 480 :=
by
  sorry

end farm_tax_percentage_l3116_311641


namespace range_of_a_for_monotonic_f_l3116_311672

/-- Given a > 0 and f(x) = x^3 - ax is monotonically increasing on [1, +∞),
    prove that the range of values for a is (0, 3]. -/
theorem range_of_a_for_monotonic_f (a : ℝ) (f : ℝ → ℝ) :
  a > 0 →
  (∀ x, f x = x^3 - a*x) →
  (∀ x y, 1 ≤ x → x < y → f x < f y) →
  ∃ S, S = Set.Ioo 0 3 ∧ a ∈ S :=
sorry

end range_of_a_for_monotonic_f_l3116_311672


namespace intersection_A_B_l3116_311674

def A : Set ℤ := {-1, 0, 1, 2}
def B : Set ℤ := {x | x^2 ≥ 1}

theorem intersection_A_B : A ∩ B = {-1, 1} := by sorry

end intersection_A_B_l3116_311674


namespace equation1_solutions_equation2_solution_l3116_311662

-- Define the equations
def equation1 (x : ℝ) : Prop := 2 * x^2 - 32 = 0
def equation2 (x : ℝ) : Prop := (x + 4)^3 + 64 = 0

-- Theorem for the first equation
theorem equation1_solutions :
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ equation1 x₁ ∧ equation1 x₂ ∧ x₁ = 4 ∧ x₂ = -4 :=
sorry

-- Theorem for the second equation
theorem equation2_solution :
  ∃ x : ℝ, equation2 x ∧ x = -8 :=
sorry

end equation1_solutions_equation2_solution_l3116_311662


namespace three_friends_came_later_l3116_311602

/-- The number of friends who came over later -/
def friends_came_later (initial_friends final_total : ℕ) : ℕ :=
  final_total - initial_friends

/-- Theorem: Given 4 initial friends and a final total of 7 people,
    prove that 3 friends came over later -/
theorem three_friends_came_later :
  friends_came_later 4 7 = 3 := by
  sorry

end three_friends_came_later_l3116_311602


namespace percentage_equivalence_l3116_311616

theorem percentage_equivalence (x : ℝ) (h : (30/100) * ((15/100) * x) = 27) :
  (15/100) * ((30/100) * x) = 27 := by
  sorry

end percentage_equivalence_l3116_311616


namespace f_neg_two_eq_neg_two_l3116_311687

/-- A polynomial function of degree 5 -/
def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^5 + b * x^3 + 4 * x + c

/-- Theorem stating that f(-2) = -2 given the conditions -/
theorem f_neg_two_eq_neg_two (a b c : ℝ) :
  (f a b c 5 + f a b c (-5) = 6) →
  (f a b c 2 = 8) →
  f a b c (-2) = -2 := by
  sorry

end f_neg_two_eq_neg_two_l3116_311687


namespace magician_tricks_conversion_l3116_311644

/-- Converts a base-9 number represented as a list of digits to its base-10 equivalent -/
def base9ToBase10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (9 ^ i)) 0

/-- The given number of tricks in base 9 -/
def tricksBase9 : List Nat := [2, 3, 4, 5]

theorem magician_tricks_conversion :
  base9ToBase10 tricksBase9 = 3998 := by
  sorry

end magician_tricks_conversion_l3116_311644


namespace division_problem_l3116_311681

theorem division_problem (L S q : ℕ) : 
  L - S = 1325 → 
  L = 1650 → 
  L = S * q + 5 → 
  q = 5 := by
sorry

end division_problem_l3116_311681


namespace events_mutually_exclusive_not_complementary_l3116_311633

-- Define the set of people
inductive Person : Type
| A : Person
| B : Person
| C : Person
| D : Person

-- Define the set of cards
inductive Card : Type
| Red : Card
| Black : Card
| White : Card
| Blue : Card

-- Define a distribution as a function from Person to Card
def Distribution := Person → Card

-- Define the event "Person A gets the red card"
def event_A_red (d : Distribution) : Prop := d Person.A = Card.Red

-- Define the event "Person B gets the red card"
def event_B_red (d : Distribution) : Prop := d Person.B = Card.Red

-- Define mutual exclusivity
def mutually_exclusive (e1 e2 : Distribution → Prop) : Prop :=
  ∀ d : Distribution, ¬(e1 d ∧ e2 d)

-- Define complementary events
def complementary (e1 e2 : Distribution → Prop) : Prop :=
  ∀ d : Distribution, e1 d ↔ ¬(e2 d)

-- Theorem statement
theorem events_mutually_exclusive_not_complementary :
  mutually_exclusive event_A_red event_B_red ∧
  ¬(complementary event_A_red event_B_red) :=
sorry

end events_mutually_exclusive_not_complementary_l3116_311633


namespace preimage_of_one_is_zero_one_neg_one_l3116_311607

-- Define the sets A and B as subsets of ℝ
variable (A B : Set ℝ)

-- Define the function f: A → B
def f (x : ℝ) : ℝ := x^3 - x + 1

-- Define the set of elements in A that map to 1 under f
def preimage_of_one (A : Set ℝ) : Set ℝ := {x ∈ A | f x = 1}

-- Theorem statement
theorem preimage_of_one_is_zero_one_neg_one (A B : Set ℝ) :
  preimage_of_one A = {0, 1, -1} := by sorry

end preimage_of_one_is_zero_one_neg_one_l3116_311607


namespace regular_price_correct_l3116_311670

/-- The regular price of one tire -/
def regular_price : ℝ := 108

/-- The sale price for three tires -/
def sale_price : ℝ := 270

/-- The theorem stating that the regular price of one tire is correct given the sale conditions -/
theorem regular_price_correct : 
  2 * regular_price + regular_price / 2 = sale_price :=
by sorry

end regular_price_correct_l3116_311670


namespace root_implies_a_range_l3116_311663

def f (a : ℝ) (x : ℝ) : ℝ := 2 * a * x + 4

theorem root_implies_a_range :
  ∀ a : ℝ, (∃ x : ℝ, x ∈ Set.Icc (-2) 1 ∧ f a x = 0) → a ∈ Set.Icc (-2) 1 := by
sorry

end root_implies_a_range_l3116_311663


namespace seeds_per_small_garden_l3116_311640

/-- Proves that given the initial number of seeds, seeds planted in the big garden,
    and the number of small gardens, the number of seeds in each small garden is correct. -/
theorem seeds_per_small_garden 
  (total_seeds : ℕ) 
  (big_garden_seeds : ℕ) 
  (small_gardens : ℕ) 
  (h1 : total_seeds = 56)
  (h2 : big_garden_seeds = 35)
  (h3 : small_gardens = 7)
  : (total_seeds - big_garden_seeds) / small_gardens = 3 := by
  sorry

end seeds_per_small_garden_l3116_311640


namespace cubic_equation_sum_of_cubes_l3116_311643

theorem cubic_equation_sum_of_cubes :
  ∃ (u v w : ℝ),
    (u - Real.rpow 7 (1/3 : ℝ)) * (u - Real.rpow 29 (1/3 : ℝ)) * (u - Real.rpow 61 (1/3 : ℝ)) = 1/5 ∧
    (v - Real.rpow 7 (1/3 : ℝ)) * (v - Real.rpow 29 (1/3 : ℝ)) * (v - Real.rpow 61 (1/3 : ℝ)) = 1/5 ∧
    (w - Real.rpow 7 (1/3 : ℝ)) * (w - Real.rpow 29 (1/3 : ℝ)) * (w - Real.rpow 61 (1/3 : ℝ)) = 1/5 ∧
    u ≠ v ∧ v ≠ w ∧ u ≠ w ∧
    u^3 + v^3 + w^3 = 97.6 :=
by sorry

end cubic_equation_sum_of_cubes_l3116_311643


namespace min_sum_squares_l3116_311697

theorem min_sum_squares (x y z : ℝ) (h : 2 * x + 3 * y + 3 * z = 1) :
  (∀ a b c : ℝ, 2 * a + 3 * b + 3 * c = 1 → x^2 + y^2 + z^2 ≤ a^2 + b^2 + c^2) →
  x^2 + y^2 + z^2 = 1 / 22 :=
by sorry

end min_sum_squares_l3116_311697


namespace invertible_elements_mod_8_l3116_311677

theorem invertible_elements_mod_8 :
  ∀ a : ℤ, a ∈ ({1, 3, 5, 7} : Set ℤ) ↔
    (∃ b : ℤ, (a * b) % 8 = 1 ∧ (a * a) % 8 = 1) :=
by sorry

end invertible_elements_mod_8_l3116_311677


namespace b_value_l3116_311654

def consecutive_odd_numbers (a b c d e : ℤ) : Prop :=
  b = a + 2 ∧ c = b + 2 ∧ d = c + 2 ∧ e = d + 2

theorem b_value (a b c d e : ℤ) 
  (h1 : consecutive_odd_numbers a b c d e)
  (h2 : a + c = 146)
  (h3 : e = 79) : 
  b = 73 := by
  sorry

end b_value_l3116_311654


namespace root_sum_property_l3116_311611

theorem root_sum_property (x₁ x₂ : ℝ) (n : ℕ) (hn : n ≥ 1) :
  (x₁^2 - 6*x₁ + 1 = 0) → (x₂^2 - 6*x₂ + 1 = 0) →
  (∃ (m : ℤ), x₁^n + x₂^n = m) ∧ ¬(∃ (k : ℤ), x₁^n + x₂^n = 5*k) := by
  sorry

end root_sum_property_l3116_311611


namespace det_of_specific_matrix_l3116_311621

theorem det_of_specific_matrix :
  let A : Matrix (Fin 2) (Fin 2) ℤ := !![5, -4; 2, 3]
  Matrix.det A = 23 := by
sorry

end det_of_specific_matrix_l3116_311621


namespace triangle_midpoint_sum_l3116_311632

theorem triangle_midpoint_sum (a b c : ℝ) : 
  a + b + c = 15 → 
  (a + b) / 2 + (a + c) / 2 + (b + c) / 2 = 15 := by
sorry

end triangle_midpoint_sum_l3116_311632


namespace equilateral_triangle_side_length_squared_l3116_311612

/-- An ellipse with equation 9x^2 + 25y^2 = 225 -/
def Ellipse : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | 9 * p.1^2 + 25 * p.2^2 = 225}

/-- A point on the ellipse -/
def PointOnEllipse (p : ℝ × ℝ) : Prop :=
  p ∈ Ellipse

/-- An equilateral triangle -/
structure EquilateralTriangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  equilateral : (A.1 - B.1)^2 + (A.2 - B.2)^2 = (B.1 - C.1)^2 + (B.2 - C.2)^2 ∧
                (B.1 - C.1)^2 + (B.2 - C.2)^2 = (C.1 - A.1)^2 + (C.2 - A.2)^2

/-- The triangle is inscribed in the ellipse -/
def TriangleInscribed (t : EquilateralTriangle) : Prop :=
  PointOnEllipse t.A ∧ PointOnEllipse t.B ∧ PointOnEllipse t.C

/-- One vertex is at (5/3, 0) -/
def VertexAtGivenPoint (t : EquilateralTriangle) : Prop :=
  t.A = (5/3, 0) ∨ t.B = (5/3, 0) ∨ t.C = (5/3, 0)

/-- One altitude is contained in the x-axis -/
def AltitudeOnXAxis (t : EquilateralTriangle) : Prop :=
  (t.A.2 = 0 ∧ t.B.2 = -t.C.2) ∨ (t.B.2 = 0 ∧ t.A.2 = -t.C.2) ∨ (t.C.2 = 0 ∧ t.A.2 = -t.B.2)

/-- The main theorem -/
theorem equilateral_triangle_side_length_squared 
  (t : EquilateralTriangle) 
  (h1 : TriangleInscribed t) 
  (h2 : VertexAtGivenPoint t) 
  (h3 : AltitudeOnXAxis t) : 
  (t.A.1 - t.B.1)^2 + (t.A.2 - t.B.2)^2 = 1475/196 :=
sorry

end equilateral_triangle_side_length_squared_l3116_311612


namespace nicole_bike_time_l3116_311659

/-- Given Nicole's biking information, calculate the time to ride 5 miles -/
theorem nicole_bike_time (distance_to_nathan : ℝ) (time_to_nathan : ℝ) (distance_to_patrick : ℝ)
  (h1 : distance_to_nathan = 2)
  (h2 : time_to_nathan = 8)
  (h3 : distance_to_patrick = 5) :
  distance_to_patrick / (distance_to_nathan / time_to_nathan) = 20 := by
  sorry

#check nicole_bike_time

end nicole_bike_time_l3116_311659


namespace percentage_without_full_time_jobs_survey_result_l3116_311669

theorem percentage_without_full_time_jobs 
  (mother_ratio : Real) 
  (father_ratio : Real) 
  (women_ratio : Real) : Real :=
  let total_parents := 100
  let women_count := women_ratio * total_parents
  let men_count := total_parents - women_count
  let employed_women := mother_ratio * women_count
  let employed_men := father_ratio * men_count
  let total_employed := employed_women + employed_men
  let unemployed := total_parents - total_employed
  unemployed / total_parents * 100

theorem survey_result : 
  percentage_without_full_time_jobs (5/6) (3/4) 0.6 = 20 := by
  sorry

end percentage_without_full_time_jobs_survey_result_l3116_311669


namespace modular_difference_in_range_l3116_311664

theorem modular_difference_in_range (a b : ℤ) : 
  a ≡ 25 [ZMOD 60] →
  b ≡ 84 [ZMOD 60] →
  ∃! n : ℤ, 150 ≤ n ∧ n ≤ 200 ∧ a - b ≡ n [ZMOD 60] ∧ n = 181 :=
by sorry

end modular_difference_in_range_l3116_311664


namespace three_W_five_l3116_311649

-- Define the operation W
def W (a b : ℤ) : ℤ := b + 7 * a - a ^ 2

-- Theorem to prove
theorem three_W_five : W 3 5 = 17 := by
  sorry

end three_W_five_l3116_311649


namespace min_dot_product_l3116_311635

/-- Circle C with center (t, t-2) and radius 1 -/
def Circle (t : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - t)^2 + (p.2 - (t-2))^2 = 1}

/-- Point P -/
def P : ℝ × ℝ := (-1, 1)

/-- Tangent points A and B (existence assumed) -/
def TangentPoints (t : ℝ) : (ℝ × ℝ) × (ℝ × ℝ) := sorry

/-- Dot product of vectors PA and PB -/
def DotProduct (t : ℝ) : ℝ :=
  let (A, B) := TangentPoints t
  ((A.1 - P.1) * (B.1 - P.1) + (A.2 - P.2) * (B.2 - P.2))

theorem min_dot_product :
  ∃ (m : ℝ), (∀ t, DotProduct t ≥ m) ∧ (∃ t₀, DotProduct t₀ = m) ∧ m = 21/4 := by
  sorry

end min_dot_product_l3116_311635


namespace square_root_range_l3116_311601

theorem square_root_range (x : ℝ) : 
  (∃ y : ℝ, y^2 = x - 4) → x ≥ 4 := by sorry

end square_root_range_l3116_311601


namespace practice_coincidence_l3116_311666

def trumpet_interval : ℕ := 11
def flute_interval : ℕ := 3

theorem practice_coincidence : Nat.lcm trumpet_interval flute_interval = 33 := by
  sorry

end practice_coincidence_l3116_311666


namespace complex_product_quadrant_l3116_311617

theorem complex_product_quadrant : 
  let z₁ : ℂ := 1 - 2*I
  let z₂ : ℂ := 2 + I
  let product := z₁ * z₂
  (product.re > 0 ∧ product.im < 0) := by sorry

end complex_product_quadrant_l3116_311617


namespace eleven_by_seven_max_squares_l3116_311622

/-- Represents the dimensions of a rectangle --/
structure Rectangle where
  length : ℕ
  width : ℕ

/-- Calculates the maximum number of squares that can be cut from a rectangle --/
def maxSquares (rect : Rectangle) : ℕ :=
  sorry

/-- The theorem stating the maximum number of squares for an 11x7 rectangle --/
theorem eleven_by_seven_max_squares :
  maxSquares ⟨11, 7⟩ = 6 := by
  sorry

end eleven_by_seven_max_squares_l3116_311622


namespace complex_magnitude_l3116_311646

theorem complex_magnitude (w : ℂ) (h : w^2 + 2*w = 11 - 16*I) : 
  Complex.abs w = 17 ∨ Complex.abs w = Real.sqrt 89 := by
  sorry

end complex_magnitude_l3116_311646


namespace oranges_taken_away_l3116_311684

/-- Represents the number of fruits in Tina's bag -/
structure FruitBag where
  apples : Nat
  oranges : Nat
  tangerines : Nat

/-- Represents the number of fruits Tina took away -/
structure FruitsTakenAway where
  oranges : Nat
  tangerines : Nat

def initial_bag : FruitBag := { apples := 9, oranges := 5, tangerines := 17 }

def fruits_taken : FruitsTakenAway := { oranges := 2, tangerines := 10 }

theorem oranges_taken_away (bag : FruitBag) (taken : FruitsTakenAway) : 
  taken.oranges = 2 ↔ 
    (bag.tangerines - taken.tangerines = (bag.oranges - taken.oranges) + 4) ∧
    (taken.tangerines = 10) ∧
    (bag = initial_bag) := by
  sorry

end oranges_taken_away_l3116_311684


namespace initial_files_correct_l3116_311609

/-- The number of files Nancy had initially -/
def initial_files : ℕ := 80

/-- The number of files Nancy deleted -/
def deleted_files : ℕ := 31

/-- The number of folders Nancy ended up with -/
def num_folders : ℕ := 7

/-- The number of files in each folder -/
def files_per_folder : ℕ := 7

/-- Theorem stating that the initial number of files is correct -/
theorem initial_files_correct : 
  initial_files = deleted_files + num_folders * files_per_folder := by
  sorry

end initial_files_correct_l3116_311609


namespace drawer_probability_verify_drawer_probability_l3116_311606

/-- The probability of selecting one shirt, one pair of shorts, and one pair of socks
    from a drawer with 6 shirts, 7 pairs of shorts, and 8 pairs of socks
    when randomly removing three articles of clothing. -/
theorem drawer_probability : ℕ → ℕ → ℕ → ℚ
  | 6, 7, 8 => 168/665
  | _, _, _ => 0

/-- Verifies that the probability is correct for the given problem. -/
theorem verify_drawer_probability :
  drawer_probability 6 7 8 = 168/665 := by sorry

end drawer_probability_verify_drawer_probability_l3116_311606


namespace inequality_implication_l3116_311642

theorem inequality_implication (a b : ℝ) (h : a < b) : -3 * a > -3 * b := by
  sorry

end inequality_implication_l3116_311642


namespace second_divisor_problem_l3116_311608

theorem second_divisor_problem (n : Nat) (h1 : n > 13) (h2 : n ∣ 192) : 
  (197 % n = 5 ∧ ∀ m : Nat, m > 13 → m < n → m ∣ 192 → 197 % m ≠ 5) → n = 16 := by
  sorry

end second_divisor_problem_l3116_311608


namespace fraction_equals_negative_one_l3116_311682

theorem fraction_equals_negative_one (a b : ℝ) (h : a + b ≠ 0) :
  (-a - b) / (a + b) = -1 := by
  sorry

end fraction_equals_negative_one_l3116_311682


namespace topsoil_cost_l3116_311690

/-- The cost of topsoil in dollars per cubic foot -/
def topsoil_cost_per_cubic_foot : ℝ := 6

/-- The number of cubic feet in one cubic yard -/
def cubic_feet_per_cubic_yard : ℝ := 27

/-- The number of cubic yards of topsoil -/
def cubic_yards_of_topsoil : ℝ := 5

/-- The theorem stating the cost of the given amount of topsoil -/
theorem topsoil_cost : 
  cubic_yards_of_topsoil * cubic_feet_per_cubic_yard * topsoil_cost_per_cubic_foot = 810 := by
  sorry

end topsoil_cost_l3116_311690


namespace hunting_ratio_l3116_311699

theorem hunting_ratio : 
  ∀ (sam rob mark peter total : ℕ) (mark_fraction : ℚ),
    sam = 6 →
    rob = sam / 2 →
    mark = mark_fraction * (sam + rob) →
    peter = 3 * mark →
    sam + rob + mark + peter = 21 →
    mark_fraction = 1 / 3 :=
by
  sorry

end hunting_ratio_l3116_311699


namespace absolute_value_plus_pi_minus_two_to_zero_l3116_311623

theorem absolute_value_plus_pi_minus_two_to_zero :
  |(-3 : ℝ)| + (π - 2)^(0 : ℝ) = 4 := by sorry

end absolute_value_plus_pi_minus_two_to_zero_l3116_311623


namespace first_apartment_utility_cost_l3116_311696

/-- Represents the monthly cost structure for an apartment --/
structure ApartmentCost where
  rent : ℝ
  utilities : ℝ
  drivingDistance : ℝ

/-- Calculates the total monthly cost for an apartment --/
def totalMonthlyCost (apt : ApartmentCost) (drivingCostPerMile : ℝ) (workingDays : ℝ) : ℝ :=
  apt.rent + apt.utilities + (apt.drivingDistance * drivingCostPerMile * workingDays)

/-- Theorem stating the utility cost of the first apartment --/
theorem first_apartment_utility_cost :
  let firstApt : ApartmentCost := { rent := 800, utilities := U, drivingDistance := 31 }
  let secondApt : ApartmentCost := { rent := 900, utilities := 200, drivingDistance := 21 }
  let drivingCostPerMile : ℝ := 0.58
  let workingDays : ℝ := 20
  totalMonthlyCost firstApt drivingCostPerMile workingDays - 
    totalMonthlyCost secondApt drivingCostPerMile workingDays = 76 →
  U = 259.60 := by
  sorry


end first_apartment_utility_cost_l3116_311696


namespace product_of_numbers_l3116_311613

theorem product_of_numbers (x y : ℝ) (h1 : x + y = 60) (h2 : x - y = 10) : x * y = 875 := by
  sorry

end product_of_numbers_l3116_311613


namespace certain_number_proof_l3116_311652

theorem certain_number_proof (x : ℚ) (n : ℚ) : 
  x = 6 → 9 - (4/x) = n + (8/x) → n = 7 := by
  sorry

end certain_number_proof_l3116_311652


namespace point_on_unit_circle_l3116_311665

/-- The coordinates of a point on the unit circle after moving counterclockwise from (1,0) by an arc length of 4π/3 -/
theorem point_on_unit_circle (Q : ℝ × ℝ) : 
  (∃ θ : ℝ, θ = 4 * Real.pi / 3 ∧ 
   Q.1 = Real.cos θ ∧ 
   Q.2 = Real.sin θ) →
  Q = (-1/2, -Real.sqrt 3 / 2) := by
sorry


end point_on_unit_circle_l3116_311665


namespace circle_translation_l3116_311657

/-- Given a circle equation, prove its center, radius, and translated form -/
theorem circle_translation (x y : ℝ) :
  let original_eq := x^2 + y^2 - 4*x + 6*y - 68 = 0
  let center := (2, -3)
  let radius := 9
  let X := x - 2
  let Y := y + 3
  let translated_eq := X^2 + Y^2 = 81
  original_eq → (
    (x - center.1)^2 + (y - center.2)^2 = radius^2 ∧
    translated_eq
  ) := by sorry

end circle_translation_l3116_311657


namespace gcf_lcm_sum_9_18_36_l3116_311650

theorem gcf_lcm_sum_9_18_36 : 
  let A := Nat.gcd 9 (Nat.gcd 18 36)
  let B := Nat.lcm 9 (Nat.lcm 18 36)
  A + B = 45 := by sorry

end gcf_lcm_sum_9_18_36_l3116_311650


namespace quadratic_root_implies_m_value_l3116_311628

theorem quadratic_root_implies_m_value (m n : ℝ) :
  (Complex.I : ℂ).re = 0 ∧ (Complex.I : ℂ).im = 1 →
  ((-3 : ℂ) + 2 * Complex.I) ^ 2 + m * ((-3 : ℂ) + 2 * Complex.I) + n = 0 →
  m = 6 := by sorry

end quadratic_root_implies_m_value_l3116_311628


namespace no_prime_base_n_representation_l3116_311695

def base_n_representation (n : ℕ) : ℕ := n^4 + n^2 + 1

theorem no_prime_base_n_representation :
  ∀ n : ℕ, n ≥ 2 → ¬(Nat.Prime (base_n_representation n)) :=
by sorry

end no_prime_base_n_representation_l3116_311695


namespace fibonacci_eight_sum_not_equal_single_l3116_311636

def fibonacci : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | (n + 2) => fibonacci (n + 1) + fibonacci n

theorem fibonacci_eight_sum_not_equal_single (k : ℕ) : 
  ¬∃ m : ℕ, 
    fibonacci k + fibonacci (k + 1) + fibonacci (k + 2) + fibonacci (k + 3) + 
    fibonacci (k + 4) + fibonacci (k + 5) + fibonacci (k + 6) + fibonacci (k + 7) = 
    fibonacci m := by
  sorry

end fibonacci_eight_sum_not_equal_single_l3116_311636


namespace cake_slices_l3116_311678

/-- The cost of ingredients and number of slices eaten by Laura's mother and the dog --/
structure CakeData where
  flour_cost : ℝ
  sugar_cost : ℝ
  butter_cost : ℝ
  eggs_cost : ℝ
  mother_slices : ℕ
  dog_cost : ℝ

/-- The total number of slices Laura cut the cake into --/
def total_slices (data : CakeData) : ℕ :=
  sorry

/-- Theorem stating that the total number of slices is 6 --/
theorem cake_slices (data : CakeData) 
  (h1 : data.flour_cost = 4)
  (h2 : data.sugar_cost = 2)
  (h3 : data.butter_cost = 2.5)
  (h4 : data.eggs_cost = 0.5)
  (h5 : data.mother_slices = 2)
  (h6 : data.dog_cost = 6) :
  total_slices data = 6 :=
sorry

end cake_slices_l3116_311678


namespace sufficient_not_necessary_condition_l3116_311637

theorem sufficient_not_necessary_condition :
  (∀ a b : ℝ, a > b ∧ b > 0 → a^2 > b^2) ∧
  (∃ a b : ℝ, a^2 > b^2 ∧ ¬(a > b ∧ b > 0)) := by
  sorry

end sufficient_not_necessary_condition_l3116_311637


namespace cara_seating_arrangement_l3116_311685

def choose (n k : ℕ) : ℕ := (Nat.factorial n) / ((Nat.factorial k) * (Nat.factorial (n - k)))

theorem cara_seating_arrangement :
  choose 5 2 = 10 := by
  sorry

end cara_seating_arrangement_l3116_311685


namespace yellow_tiled_area_l3116_311603

theorem yellow_tiled_area (length : ℝ) (width : ℝ) (yellow_ratio : ℝ) : 
  length = 3.6 → 
  width = 2.5 * length → 
  yellow_ratio = 1 / 2 → 
  yellow_ratio * (length * width) = 16.2 := by
sorry

end yellow_tiled_area_l3116_311603


namespace equation_solutions_inequality_solutions_l3116_311624

/-- Part a: Solutions for 1/x + 1/y + 1/z = 1 where x, y, z are natural numbers -/
def solutions_a : Set (ℕ × ℕ × ℕ) :=
  {(3, 3, 3), (6, 3, 2), (4, 4, 2)}

/-- Part b: Solutions for 1/x + 1/y + 1/z > 1 where x, y, z are natural numbers greater than 1 -/
def solutions_b : Set (ℕ × ℕ × ℕ) :=
  {(x, 2, 2) | x > 1} ∪ {(3, 3, 2), (4, 3, 2), (5, 3, 2)}

theorem equation_solutions (x y z : ℕ) :
  (1 / x + 1 / y + 1 / z = 1) ↔ (x, y, z) ∈ solutions_a := by
  sorry

theorem inequality_solutions (x y z : ℕ) :
  (x > 1 ∧ y > 1 ∧ z > 1 ∧ 1 / x + 1 / y + 1 / z > 1) ↔ (x, y, z) ∈ solutions_b := by
  sorry

end equation_solutions_inequality_solutions_l3116_311624


namespace jordan_seven_miles_time_l3116_311692

/-- Jordan's running time for a given distance -/
def jordanTime (distance : ℝ) : ℝ := sorry

/-- Steve's running time for a given distance -/
def steveTime (distance : ℝ) : ℝ := sorry

/-- Theorem stating Jordan's time for 7 miles given the conditions -/
theorem jordan_seven_miles_time :
  (jordanTime 3 = 2 / 3 * steveTime 5) →
  (steveTime 5 = 40) →
  (∀ d₁ d₂ : ℝ, jordanTime d₁ / d₁ = jordanTime d₂ / d₂) →
  jordanTime 7 = 185 / 3 := by
  sorry

end jordan_seven_miles_time_l3116_311692


namespace square_difference_identity_l3116_311661

theorem square_difference_identity : (15 + 5)^2 - (15^2 + 5^2) = 150 := by
  sorry

end square_difference_identity_l3116_311661


namespace germs_per_dish_l3116_311647

theorem germs_per_dish :
  let total_germs : ℝ := 5.4 * 10^6
  let total_dishes : ℝ := 10800
  let germs_per_dish : ℝ := total_germs / total_dishes
  germs_per_dish = 500 :=
by sorry

end germs_per_dish_l3116_311647
