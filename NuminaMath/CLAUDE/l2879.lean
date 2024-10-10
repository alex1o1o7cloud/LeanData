import Mathlib

namespace problem_statement_l2879_287903

-- Define proposition p
def p : Prop := ∃ x₀ : ℝ, x₀^2 - x₀ + 1 < 0

-- Define proposition q
def q : Prop := ∀ a b : ℝ, a^2 < b^2 → a < b

-- Theorem to prove
theorem problem_statement : ¬p ∧ ¬q := by sorry

end problem_statement_l2879_287903


namespace correct_marked_price_l2879_287911

/-- Represents the pricing structure of a book -/
structure BookPricing where
  cost_price : ℝ
  marked_price : ℝ
  first_discount_rate : ℝ
  additional_discount_rate : ℝ
  profit_rate : ℝ
  commission_rate : ℝ

/-- Calculates the final selling price after all discounts and commissions -/
def final_selling_price (b : BookPricing) : ℝ :=
  let price_after_first_discount := b.marked_price * (1 - b.first_discount_rate)
  let price_after_additional_discount := price_after_first_discount * (1 - b.additional_discount_rate)
  let commission := price_after_first_discount * b.commission_rate
  price_after_additional_discount + commission

/-- Theorem stating the correct marked price for the given conditions -/
theorem correct_marked_price :
  ∃ (b : BookPricing),
    b.cost_price = 75 ∧
    b.first_discount_rate = 0.12 ∧
    b.additional_discount_rate = 0.05 ∧
    b.profit_rate = 0.3 ∧
    b.commission_rate = 0.1 ∧
    b.marked_price = 99.35 ∧
    final_selling_price b = b.cost_price * (1 + b.profit_rate) :=
by
  sorry


end correct_marked_price_l2879_287911


namespace scaled_equation_l2879_287991

theorem scaled_equation (h : 2994 * 14.5 = 175) : 29.94 * 1.45 = 1.75 := by
  sorry

end scaled_equation_l2879_287991


namespace rals_age_is_26_l2879_287967

/-- Ral's current age -/
def rals_age : ℕ := 26

/-- Suri's current age -/
def suris_age : ℕ := 13

/-- Ral is twice as old as Suri -/
axiom ral_twice_suri : rals_age = 2 * suris_age

/-- In 3 years, Suri's current age will be 16 -/
axiom suri_age_in_3_years : suris_age + 3 = 16

/-- Theorem: Ral's current age is 26 years old -/
theorem rals_age_is_26 : rals_age = 26 := by
  sorry

end rals_age_is_26_l2879_287967


namespace negative_three_less_than_negative_two_l2879_287908

theorem negative_three_less_than_negative_two : -3 < -2 := by
  sorry

end negative_three_less_than_negative_two_l2879_287908


namespace isabel_homework_problem_l2879_287953

/-- The total number of homework problems Isabel had -/
def total_problems (finished : ℕ) (pages_left : ℕ) (problems_per_page : ℕ) : ℕ :=
  finished + pages_left * problems_per_page

/-- Theorem stating that Isabel had 72 homework problems in total -/
theorem isabel_homework_problem :
  total_problems 32 5 8 = 72 := by
  sorry

end isabel_homework_problem_l2879_287953


namespace son_age_proof_l2879_287996

theorem son_age_proof (father_age son_age : ℝ) : 
  father_age = son_age + 35 →
  father_age + 5 = 3 * (son_age + 5) →
  son_age = 12.5 := by
sorry

end son_age_proof_l2879_287996


namespace translated_point_coordinates_l2879_287971

-- Define the points in the 2D plane
def A : ℝ × ℝ := (-2, 0)
def B : ℝ × ℝ := (0, 3)
def A' : ℝ × ℝ := (2, 1)

-- Define the translation vector
def translation_vector : ℝ × ℝ := (A'.1 - A.1, A'.2 - A.2)

-- Define the translated point B'
def B' : ℝ × ℝ := (B.1 + translation_vector.1, B.2 + translation_vector.2)

-- Theorem statement
theorem translated_point_coordinates :
  B' = (4, 4) := by sorry

end translated_point_coordinates_l2879_287971


namespace rectangle_z_value_l2879_287932

/-- A rectangle with given vertices and area -/
structure Rectangle where
  z : ℝ
  area : ℝ
  h_vertices : z > 5
  h_area : area = 64

/-- The value of z for the given rectangle is 13 -/
theorem rectangle_z_value (rect : Rectangle) : rect.z = 13 := by
  sorry

end rectangle_z_value_l2879_287932


namespace paint_cost_most_cost_effective_l2879_287977

/-- Represents the payment options for the house painting job -/
inductive PaymentOption
  | WorkerDay
  | PaintCost
  | PaintedArea
  | HourlyRate

/-- Calculates the cost of a payment option given the job parameters -/
def calculate_cost (option : PaymentOption) (workers : ℕ) (hours_per_day : ℕ) (days : ℕ) 
  (paint_cost : ℕ) (painted_area : ℕ) : ℕ :=
  match option with
  | PaymentOption.WorkerDay => workers * days * 30
  | PaymentOption.PaintCost => (paint_cost * 30) / 100
  | PaymentOption.PaintedArea => painted_area * 12
  | PaymentOption.HourlyRate => workers * hours_per_day * days * 4

/-- Theorem stating that the PaintCost option is the most cost-effective -/
theorem paint_cost_most_cost_effective (workers : ℕ) (hours_per_day : ℕ) (days : ℕ) 
  (paint_cost : ℕ) (painted_area : ℕ) 
  (h1 : workers = 5)
  (h2 : hours_per_day = 8)
  (h3 : days = 10)
  (h4 : paint_cost = 4800)
  (h5 : painted_area = 150) :
  ∀ option, option ≠ PaymentOption.PaintCost → 
    calculate_cost PaymentOption.PaintCost workers hours_per_day days paint_cost painted_area ≤ 
    calculate_cost option workers hours_per_day days paint_cost painted_area :=
by sorry

end paint_cost_most_cost_effective_l2879_287977


namespace ring_area_l2879_287949

theorem ring_area (r : ℝ) (h : r > 0) :
  let outer_radius : ℝ := 3 * r
  let inner_radius : ℝ := r
  let width : ℝ := 3
  outer_radius - inner_radius = width →
  (π * outer_radius^2 - π * inner_radius^2) = 72 * π :=
by
  sorry

end ring_area_l2879_287949


namespace new_average_is_75_l2879_287959

/-- Calculates the new average daily production after adding a new day's production. -/
def new_average_production (past_days : ℕ) (past_average : ℚ) (today_production : ℕ) : ℚ :=
  (past_average * past_days + today_production) / (past_days + 1)

/-- Theorem stating that given the conditions, the new average daily production is 75 units. -/
theorem new_average_is_75 :
  let past_days : ℕ := 3
  let past_average : ℚ := 70
  let today_production : ℕ := 90
  new_average_production past_days past_average today_production = 75 := by
sorry

end new_average_is_75_l2879_287959


namespace quadratic_function_k_value_l2879_287948

/-- A quadratic function with integer coefficients -/
def QuadraticFunction (a b c : ℤ) : ℝ → ℝ := fun x ↦ (a * x^2 : ℝ) + (b * x : ℝ) + (c : ℝ)

theorem quadratic_function_k_value
  (a b c k : ℤ)
  (h1 : QuadraticFunction a b c 1 = 0)
  (h2 : 60 < QuadraticFunction a b c 7 ∧ QuadraticFunction a b c 7 < 70)
  (h3 : 80 < QuadraticFunction a b c 8 ∧ QuadraticFunction a b c 8 < 90)
  (h4 : (2000 : ℝ) * (k : ℝ) < QuadraticFunction a b c 50 ∧
        QuadraticFunction a b c 50 < (2000 : ℝ) * ((k + 1) : ℝ)) :
  k = 1 := by
  sorry

end quadratic_function_k_value_l2879_287948


namespace set_c_forms_triangle_l2879_287973

/-- Triangle inequality theorem for a set of three line segments -/
def satisfies_triangle_inequality (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

/-- Theorem: The set of line segments (4, 5, 6) can form a triangle -/
theorem set_c_forms_triangle : satisfies_triangle_inequality 4 5 6 := by
  sorry

end set_c_forms_triangle_l2879_287973


namespace triangle_angle_sum_equivalent_to_parallel_postulate_l2879_287928

-- Define Euclidean geometry
axiom EuclideanGeometry : Type

-- Define the parallel postulate
axiom parallel_postulate : EuclideanGeometry → Prop

-- Define the triangle angle sum theorem
axiom triangle_angle_sum : EuclideanGeometry → Prop

-- Theorem statement
theorem triangle_angle_sum_equivalent_to_parallel_postulate :
  ∀ (E : EuclideanGeometry), triangle_angle_sum E ↔ parallel_postulate E :=
sorry

end triangle_angle_sum_equivalent_to_parallel_postulate_l2879_287928


namespace complement_of_A_l2879_287925

def A : Set ℝ := {x : ℝ | -1 ≤ x ∧ x < 2}

theorem complement_of_A : 
  (Set.univ : Set ℝ) \ A = {x : ℝ | x < -1 ∨ x ≥ 2} := by sorry

end complement_of_A_l2879_287925


namespace triangle_special_angle_l2879_287990

/-- Given a triangle with side lengths a, b, and c satisfying the equation
    (c^2)/(a+b) + (a^2)/(b+c) = b, the angle opposite side b is 60°. -/
theorem triangle_special_angle (a b c : ℝ) (h : 0 < a ∧ 0 < b ∧ 0 < c) 
    (eq : c^2/(a+b) + a^2/(b+c) = b) : 
  let B := Real.arccos ((a^2 + c^2 - b^2) / (2*a*c))
  B = π/3 := by
sorry

end triangle_special_angle_l2879_287990


namespace expression_simplification_l2879_287916

theorem expression_simplification (x y : ℝ) (h : x * y ≠ 0) :
  ((x^3 + 1) / x * (y^2 + 1) / y) - ((x^2 - 1) / y * (y^3 - 1) / x) =
  (x^3 * y^2 - x^2 * y^3 + x^3 + x^2 + y^2 + y^3) / (x * y) := by
sorry

end expression_simplification_l2879_287916


namespace square_inequality_condition_l2879_287938

theorem square_inequality_condition (a b : ℝ) :
  (∀ a b : ℝ, a > b ∧ b > 0 → a^2 > b^2) ∧
  (∃ a b : ℝ, a^2 > b^2 ∧ ¬(a > b ∧ b > 0)) :=
by sorry

end square_inequality_condition_l2879_287938


namespace assignment_ways_20_3_l2879_287904

/-- The number of ways to assign 3 distinct items from a set of 20 items -/
def assignmentWays (n : ℕ) (k : ℕ) : ℕ :=
  n * (n - 1) * (n - 2)

/-- Theorem: The number of ways to assign 3 distinct items from a set of 20 items is 6840 -/
theorem assignment_ways_20_3 :
  assignmentWays 20 3 = 6840 := by
  sorry

#eval assignmentWays 20 3

end assignment_ways_20_3_l2879_287904


namespace interest_difference_theorem_l2879_287921

theorem interest_difference_theorem (P : ℝ) : 
  let r : ℝ := 0.04  -- 4% annual interest rate
  let t : ℕ := 2     -- 2 years time period
  let compound_interest := P * (1 + r)^t - P
  let simple_interest := P * r * t
  compound_interest - simple_interest = 1 → P = 625 := by
sorry

end interest_difference_theorem_l2879_287921


namespace f_and_g_are_even_and_increasing_l2879_287994

-- Define the functions
def f (x : ℝ) : ℝ := |2 * x|
def g (x : ℝ) : ℝ := 2 * x^2 + 3

-- Define evenness
def is_even (h : ℝ → ℝ) : Prop := ∀ x, h x = h (-x)

-- Define monotonically increasing on an interval
def is_monotone_increasing_on (h : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a < x → x < y → y < b → h x ≤ h y

-- Theorem statement
theorem f_and_g_are_even_and_increasing :
  (is_even f ∧ is_monotone_increasing_on f 0 1) ∧
  (is_even g ∧ is_monotone_increasing_on g 0 1) :=
sorry

end f_and_g_are_even_and_increasing_l2879_287994


namespace max_value_abcd_l2879_287910

theorem max_value_abcd (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) :
  (a * b * c * d * (a + b + c + d)) / ((a + b)^2 * (c + d)^2) ≤ (1 : ℝ) / 4 :=
by sorry

end max_value_abcd_l2879_287910


namespace triangle_to_decagon_area_ratio_l2879_287920

/-- A regular decagon -/
structure RegularDecagon where
  -- Add necessary fields here
  area : ℝ

/-- A triangle within a regular decagon formed by connecting three non-adjacent vertices -/
structure TriangleInDecagon (d : RegularDecagon) where
  -- Add necessary fields here
  area : ℝ

/-- The ratio of the area of a triangle to the area of the regular decagon it's inscribed in is 1/5 -/
theorem triangle_to_decagon_area_ratio 
  (d : RegularDecagon) 
  (t : TriangleInDecagon d) : 
  t.area / d.area = 1 / 5 := by sorry

end triangle_to_decagon_area_ratio_l2879_287920


namespace x_plus_y_equals_four_l2879_287995

theorem x_plus_y_equals_four (x y : ℝ) 
  (h1 : |x| + x + y = 12) 
  (h2 : x + |y| - y = 16) : 
  x + y = 4 := by
sorry

end x_plus_y_equals_four_l2879_287995


namespace right_triangle_angles_l2879_287968

/-- Represents a right triangle with external angles on the hypotenuse in the ratio 9:11 -/
structure RightTriangle where
  -- First acute angle in degrees
  α : ℝ
  -- Second acute angle in degrees
  β : ℝ
  -- The triangle is right-angled
  right_angle : α + β = 90
  -- The external angles on the hypotenuse are in the ratio 9:11
  external_angle_ratio : (180 - α) / (90 + α) = 9 / 11

/-- Theorem stating the acute angles of the specified right triangle -/
theorem right_triangle_angles (t : RightTriangle) : t.α = 58.5 ∧ t.β = 31.5 := by
  sorry

end right_triangle_angles_l2879_287968


namespace train_travel_time_l2879_287930

/-- Represents the problem of calculating the travel time of two trains --/
theorem train_travel_time 
  (cattle_speed : ℝ) 
  (speed_difference : ℝ) 
  (head_start : ℝ) 
  (total_distance : ℝ) 
  (h1 : cattle_speed = 56) 
  (h2 : speed_difference = 33) 
  (h3 : head_start = 6) 
  (h4 : total_distance = 1284) :
  ∃ t : ℝ, 
    t > 0 ∧ 
    cattle_speed * (t + head_start) + (cattle_speed - speed_difference) * t = total_distance ∧ 
    t = 12 := by
  sorry


end train_travel_time_l2879_287930


namespace max_distance_MN_l2879_287907

noncomputable def f (x : ℝ) : ℝ := Real.sin x
noncomputable def g (x : ℝ) : ℝ := 2 * (Real.cos x)^2 - 1

theorem max_distance_MN :
  ∃ (max_dist : ℝ),
    (∀ (a : ℝ),
      let M := (a, f a)
      let N := (a, g a)
      let dist_MN := |f a - g a|
      dist_MN ≤ max_dist) ∧
    (∃ (a : ℝ),
      let M := (a, f a)
      let N := (a, g a)
      let dist_MN := |f a - g a|
      dist_MN = max_dist) ∧
    max_dist = 2 := by sorry

end max_distance_MN_l2879_287907


namespace practice_time_ratio_l2879_287918

/-- Represents the practice time in minutes for each day of the week -/
structure PracticeTime where
  monday : ℕ
  tuesday : ℕ
  wednesday : ℕ
  thursday : ℕ
  friday : ℕ

/-- Theorem stating the ratio of practice time on Monday to Tuesday is 2:1 -/
theorem practice_time_ratio (p : PracticeTime) : 
  p.thursday = 50 ∧ 
  p.wednesday = p.thursday + 5 ∧ 
  p.tuesday = p.wednesday - 10 ∧ 
  p.friday = 60 ∧ 
  p.monday + p.tuesday + p.wednesday + p.thursday + p.friday = 300 →
  p.monday = 2 * p.tuesday :=
by sorry

end practice_time_ratio_l2879_287918


namespace power_of_two_with_ones_and_twos_l2879_287954

theorem power_of_two_with_ones_and_twos (N : ℕ) : 
  ∃ k : ℕ, ∃ m : ℕ, 2^k ≡ m [ZMOD 10^N] ∧ 
  ∀ d : ℕ, d < N → (m / 10^d % 10 = 1 ∨ m / 10^d % 10 = 2) :=
sorry

end power_of_two_with_ones_and_twos_l2879_287954


namespace sufficient_not_necessary_l2879_287975

theorem sufficient_not_necessary (a b : ℝ) :
  (∀ a b, a = 0 → a * b = 0) ∧
  (∃ a b, a * b = 0 ∧ a ≠ 0) := by
  sorry

end sufficient_not_necessary_l2879_287975


namespace james_living_room_cost_l2879_287937

def couch_price : ℝ := 2500
def sectional_price : ℝ := 3500
def entertainment_center_price : ℝ := 1500
def rug_price : ℝ := 800
def coffee_table_price : ℝ := 700
def accessories_price : ℝ := 500

def couch_discount : ℝ := 0.10
def sectional_discount : ℝ := 0.10
def entertainment_center_discount : ℝ := 0.05
def rug_discount : ℝ := 0.05
def coffee_table_discount : ℝ := 0.12
def accessories_discount : ℝ := 0.15

def sales_tax_rate : ℝ := 0.0825
def service_fee : ℝ := 250

def total_cost : ℝ := 9587.65

theorem james_living_room_cost : 
  (couch_price * (1 - couch_discount) + 
   sectional_price * (1 - sectional_discount) + 
   entertainment_center_price * (1 - entertainment_center_discount) + 
   rug_price * (1 - rug_discount) + 
   coffee_table_price * (1 - coffee_table_discount) + 
   accessories_price * (1 - accessories_discount)) * 
  (1 + sales_tax_rate) + service_fee = total_cost := by
  sorry

end james_living_room_cost_l2879_287937


namespace ellipse_focal_distances_l2879_287946

theorem ellipse_focal_distances (x y : ℝ) (P : ℝ × ℝ) (F1 F2 : ℝ × ℝ) :
  x^2 / 25 + y^2 = 1 →  -- P is on the ellipse
  P = (x, y) →  -- P's coordinates
  (∃ d : ℝ, d = 2 ∧ (Real.sqrt ((P.1 - F1.1)^2 + (P.2 - F1.2)^2) = d ∨
                     Real.sqrt ((P.1 - F2.1)^2 + (P.2 - F2.2)^2) = d)) →
  Real.sqrt ((P.1 - F1.1)^2 + (P.2 - F1.2)^2) +
  Real.sqrt ((P.1 - F2.1)^2 + (P.2 - F2.2)^2) = 10 →
  (∃ d : ℝ, d = 8 ∧ (Real.sqrt ((P.1 - F1.1)^2 + (P.2 - F1.2)^2) = d ∨
                     Real.sqrt ((P.1 - F2.1)^2 + (P.2 - F2.2)^2) = d)) :=
by sorry

end ellipse_focal_distances_l2879_287946


namespace quadratic_sum_l2879_287970

/-- Given a quadratic expression 4x^2 - 8x + 1, when expressed in the form a(x-h)^2 + k,
    the sum of a, h, and k equals 2. -/
theorem quadratic_sum (a h k : ℝ) : 
  (∀ x, 4 * x^2 - 8 * x + 1 = a * (x - h)^2 + k) → a + h + k = 2 := by
  sorry

end quadratic_sum_l2879_287970


namespace builder_cost_l2879_287983

/-- The cost of hiring builders to construct houses -/
theorem builder_cost (builders_per_floor : ℕ) (days_per_floor : ℕ) (daily_wage : ℕ)
  (num_builders : ℕ) (num_houses : ℕ) (floors_per_house : ℕ) :
  builders_per_floor = 3 →
  days_per_floor = 30 →
  daily_wage = 100 →
  num_builders = 6 →
  num_houses = 5 →
  floors_per_house = 6 →
  (num_houses * floors_per_house * days_per_floor * daily_wage * num_builders) / builders_per_floor = 270000 :=
by sorry

end builder_cost_l2879_287983


namespace sum_first_150_remainder_l2879_287923

theorem sum_first_150_remainder (n : Nat) (h : n = 150) : 
  (List.range n).sum % 5600 = 125 := by
  sorry

end sum_first_150_remainder_l2879_287923


namespace arithmetic_sequence_property_l2879_287987

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- Proof that for an arithmetic sequence and distinct positive integers m, n, and p,
    the equation m(a_p - a_n) + n(a_m - a_p) + p(a_n - a_m) = 0 holds -/
theorem arithmetic_sequence_property
  (a : ℕ → ℝ) (m n p : ℕ) (h_arith : ArithmeticSequence a) (h_distinct : m ≠ n ∧ n ≠ p ∧ m ≠ p) :
  m * (a p - a n) + n * (a m - a p) + p * (a n - a m) = 0 :=
by sorry

end arithmetic_sequence_property_l2879_287987


namespace goldfish_death_rate_l2879_287914

/-- The number of goldfish that die each week -/
def goldfish_deaths_per_week : ℕ := 5

/-- The initial number of goldfish -/
def initial_goldfish : ℕ := 18

/-- The number of goldfish purchased each week -/
def goldfish_purchased_per_week : ℕ := 3

/-- The number of weeks -/
def weeks : ℕ := 7

/-- The final number of goldfish -/
def final_goldfish : ℕ := 4

theorem goldfish_death_rate : 
  initial_goldfish + (goldfish_purchased_per_week * weeks) - (goldfish_deaths_per_week * weeks) = final_goldfish :=
by sorry

end goldfish_death_rate_l2879_287914


namespace systematic_sampling_sum_l2879_287957

/-- Systematic sampling function -/
def systematicSample (n : ℕ) (sampleSize : ℕ) (start : ℕ) : List ℕ :=
  List.range sampleSize |>.map (fun i => (start + i * (n / sampleSize)) % n + 1)

theorem systematic_sampling_sum (n : ℕ) (sampleSize : ℕ) (start : ℕ) :
  n = 50 →
  sampleSize = 5 →
  start ≤ n →
  systematicSample n sampleSize start = [4, a, 24, b, 44] →
  a + b = 48 :=
by
  sorry

end systematic_sampling_sum_l2879_287957


namespace function_inequality_implies_m_value_l2879_287981

theorem function_inequality_implies_m_value (m : ℝ) : 
  (∀ x : ℝ, (x^2 - 3*x + m ≥ 2*x^2 - 4*x) ↔ (-1 ≤ x ∧ x ≤ 2)) → m = 2 := by
  sorry

end function_inequality_implies_m_value_l2879_287981


namespace melody_reading_pages_l2879_287905

def english_pages : ℕ := 20
def science_pages : ℕ := 16
def civics_pages : ℕ := 8
def chinese_pages : ℕ := 12

def pages_to_read (total_pages : ℕ) : ℕ := total_pages / 4

theorem melody_reading_pages : 
  pages_to_read english_pages + 
  pages_to_read science_pages + 
  pages_to_read civics_pages + 
  pages_to_read chinese_pages = 14 := by
  sorry

end melody_reading_pages_l2879_287905


namespace investment_difference_l2879_287940

def compound_interest (principal : ℝ) (rate : ℝ) : ℝ :=
  principal * (1 + rate)

def emma_investment (initial : ℝ) (rate1 rate2 rate3 : ℝ) : ℝ :=
  compound_interest (compound_interest (compound_interest initial rate1) rate2) rate3

def briana_investment (initial : ℝ) (rate1 rate2 rate3 : ℝ) : ℝ :=
  compound_interest (compound_interest (compound_interest initial rate1) rate2) rate3

theorem investment_difference :
  let emma_initial := 300
  let briana_initial := 500
  let emma_rate1 := 0.15
  let emma_rate2 := 0.12
  let emma_rate3 := 0.18
  let briana_rate1 := 0.10
  let briana_rate2 := 0.08
  let briana_rate3 := 0.14
  briana_investment briana_initial briana_rate1 briana_rate2 briana_rate3 -
  emma_investment emma_initial emma_rate1 emma_rate2 emma_rate3 = 220.808 := by
  sorry

end investment_difference_l2879_287940


namespace nickel_count_is_three_l2879_287901

/-- Represents the number of coins of each type -/
structure CoinCount where
  pennies : ℕ
  nickels : ℕ
  dimes : ℕ

/-- The total value of coins in cents -/
def totalValue (c : CoinCount) : ℕ :=
  c.pennies * 1 + c.nickels * 5 + c.dimes * 10

/-- The total number of coins -/
def totalCoins (c : CoinCount) : ℕ :=
  c.pennies + c.nickels + c.dimes

theorem nickel_count_is_three :
  ∃ (c : CoinCount),
    totalCoins c = 8 ∧
    totalValue c = 47 ∧
    c.pennies ≥ 1 ∧
    c.nickels ≥ 1 ∧
    c.dimes ≥ 1 ∧
    (∀ (c' : CoinCount),
      totalCoins c' = 8 →
      totalValue c' = 47 →
      c'.pennies ≥ 1 →
      c'.nickels ≥ 1 →
      c'.dimes ≥ 1 →
      c'.nickels = 3) :=
by
  sorry

end nickel_count_is_three_l2879_287901


namespace books_given_difference_l2879_287979

theorem books_given_difference (mike_books_tuesday : ℕ) (mike_gave : ℕ) (lily_total : ℕ)
  (h1 : mike_books_tuesday = 45)
  (h2 : mike_gave = 10)
  (h3 : lily_total = 35) :
  lily_total - mike_gave - (mike_gave) = 15 := by
  sorry

end books_given_difference_l2879_287979


namespace product_unit_digit_l2879_287924

-- Define a function to get the unit digit of a number
def unitDigit (n : ℕ) : ℕ := n % 10

-- State the theorem
theorem product_unit_digit : 
  unitDigit (624 * 708 * 913 * 463) = 8 := by
  sorry

end product_unit_digit_l2879_287924


namespace bill_initial_money_l2879_287955

theorem bill_initial_money (ann_initial bill_initial : ℕ) (transfer : ℕ) : 
  ann_initial = 777 →
  transfer = 167 →
  ann_initial + transfer = bill_initial - transfer →
  bill_initial = 1111 := by
sorry

end bill_initial_money_l2879_287955


namespace other_factor_power_of_two_l2879_287958

def w : ℕ := 144

theorem other_factor_power_of_two :
  (∃ (k : ℕ), 936 * w = k * (3^3) * (12^2)) →
  (∀ (m : ℕ), m < w → ¬(∃ (l : ℕ), 936 * m = l * (3^3) * (12^2))) →
  (∃ (x : ℕ), 2^x ∣ (936 * w) ∧ x = 4) :=
by sorry

end other_factor_power_of_two_l2879_287958


namespace transformed_area_theorem_l2879_287913

-- Define the matrix
def A : Matrix (Fin 2) (Fin 2) ℝ := !![2, 1; 7, -3]

-- Define the original region's area
def S_area : ℝ := 10

-- Theorem statement
theorem transformed_area_theorem :
  let det := Matrix.det A
  let scale_factor := |det|
  scale_factor * S_area = 130 := by sorry

end transformed_area_theorem_l2879_287913


namespace lee_earnings_theorem_l2879_287944

/-- Represents the lawn care services and their charges -/
structure LawnCareService where
  mowing : Nat
  trimming : Nat
  weedRemoval : Nat
  leafBlowing : Nat
  fertilizing : Nat

/-- Represents the number of services provided -/
structure ServicesProvided where
  mowing : Nat
  trimming : Nat
  weedRemoval : Nat
  leafBlowing : Nat
  fertilizing : Nat

/-- Represents the tips received for each service -/
structure TipsReceived where
  mowing : List Nat
  trimming : List Nat
  weedRemoval : List Nat
  leafBlowing : List Nat

/-- Calculates the total earnings from services and tips -/
def calculateTotalEarnings (charges : LawnCareService) (services : ServicesProvided) (tips : TipsReceived) : Nat :=
  let serviceEarnings := 
    charges.mowing * services.mowing +
    charges.trimming * services.trimming +
    charges.weedRemoval * services.weedRemoval +
    charges.leafBlowing * services.leafBlowing +
    charges.fertilizing * services.fertilizing
  let tipEarnings :=
    tips.mowing.sum + tips.trimming.sum + tips.weedRemoval.sum + tips.leafBlowing.sum
  serviceEarnings + tipEarnings

/-- Theorem stating that Lee's total earnings are $923 -/
theorem lee_earnings_theorem (charges : LawnCareService) (services : ServicesProvided) (tips : TipsReceived)
    (h1 : charges = { mowing := 33, trimming := 15, weedRemoval := 10, leafBlowing := 20, fertilizing := 25 })
    (h2 : services = { mowing := 16, trimming := 8, weedRemoval := 5, leafBlowing := 4, fertilizing := 3 })
    (h3 : tips = { mowing := [10, 10, 12, 15], trimming := [5, 7], weedRemoval := [5], leafBlowing := [6] }) :
    calculateTotalEarnings charges services tips = 923 := by
  sorry


end lee_earnings_theorem_l2879_287944


namespace mans_age_twice_sons_l2879_287998

theorem mans_age_twice_sons (son_age : ℕ) (age_difference : ℕ) : son_age = 26 → age_difference = 28 → 
  ∃ y : ℕ, (son_age + y + age_difference) = 2 * (son_age + y) ∧ y = 2 := by
  sorry

end mans_age_twice_sons_l2879_287998


namespace derivative_sin_minus_exp_two_l2879_287900

theorem derivative_sin_minus_exp_two (x : ℝ) :
  deriv (fun x => Real.sin x - (2 : ℝ)^x) x = Real.cos x - (2 : ℝ)^x * Real.log 2 := by
  sorry

end derivative_sin_minus_exp_two_l2879_287900


namespace product_expansion_l2879_287986

theorem product_expansion (x : ℝ) : 
  (x^2 - 3*x + 3) * (x^2 + x + 1) = x^4 - 2*x^3 + x^2 + 3 := by
  sorry

end product_expansion_l2879_287986


namespace total_weight_is_5040_l2879_287917

/-- The weight of all settings for a catering event. -/
def total_weight_of_settings : ℕ :=
  let silverware_weight_per_piece : ℕ := 4
  let silverware_pieces_per_setting : ℕ := 3
  let plate_weight : ℕ := 12
  let plates_per_setting : ℕ := 2
  let tables : ℕ := 15
  let settings_per_table : ℕ := 8
  let backup_settings : ℕ := 20
  
  let total_settings : ℕ := tables * settings_per_table + backup_settings
  let weight_per_setting : ℕ := silverware_weight_per_piece * silverware_pieces_per_setting + 
                                 plate_weight * plates_per_setting
  
  total_settings * weight_per_setting

theorem total_weight_is_5040 : total_weight_of_settings = 5040 := by
  sorry

end total_weight_is_5040_l2879_287917


namespace rectangle_area_l2879_287943

/-- The length of the shorter side of the smaller rectangles -/
def short_side : ℝ := 7

/-- The length of the longer side of the smaller rectangles -/
def long_side : ℝ := 3 * short_side

/-- The width of the larger rectangle EFGH -/
def width : ℝ := long_side

/-- The length of the larger rectangle EFGH -/
def length : ℝ := long_side + short_side

/-- The area of the larger rectangle EFGH -/
def area : ℝ := length * width

theorem rectangle_area : area = 588 := by
  sorry

end rectangle_area_l2879_287943


namespace girls_count_l2879_287976

theorem girls_count (total : ℕ) (difference : ℕ) (girls : ℕ) : 
  total = 600 → 
  difference = 30 → 
  girls + (girls - difference) = total → 
  girls = 315 := by
sorry

end girls_count_l2879_287976


namespace min_value_of_f_l2879_287988

-- Define the function f(x)
def f (x : ℝ) : ℝ := 3 * x^2 - 18 * x + 2000

-- Theorem statement
theorem min_value_of_f :
  ∃ (x_min : ℝ), ∀ (x : ℝ), f x ≥ f x_min ∧ f x_min = 1973 :=
by sorry

end min_value_of_f_l2879_287988


namespace green_balls_count_l2879_287909

/-- The number of green balls in a bag with specific conditions -/
def num_green_balls (total : ℕ) (white : ℕ) (yellow : ℕ) (red : ℕ) (purple : ℕ) 
    (prob_not_red_purple : ℚ) : ℕ :=
  total - (white + yellow + red + purple)

theorem green_balls_count :
  let total := 60
  let white := 22
  let yellow := 2
  let red := 15
  let purple := 3
  let prob_not_red_purple := 7/10
  num_green_balls total white yellow red purple prob_not_red_purple = 18 := by
  sorry

end green_balls_count_l2879_287909


namespace problem_solution_l2879_287902

def A : Set ℝ := {x | (x - 1/2) * (x - 3) = 0}

def B (a : ℝ) : Set ℝ := {x | Real.log (x^2 + a*x + a + 9/4) = 0}

theorem problem_solution :
  (∀ a : ℝ, (∃! x : ℝ, x ∈ B a) → (a = 5 ∨ a = -1)) ∧
  (∀ a : ℝ, B a ⊂ A → a ∈ Set.Icc (-1) 5) :=
sorry

end problem_solution_l2879_287902


namespace frog_jump_probability_l2879_287935

-- Define the frog's jump
structure Jump where
  length : ℝ
  direction : ℝ × ℝ

-- Define the frog's position
def Position := ℝ × ℝ

-- Define a function to calculate the final position after n jumps
def finalPosition (jumps : List Jump) : Position :=
  sorry

-- Define a function to calculate the distance between two positions
def distance (p1 p2 : Position) : ℝ :=
  sorry

-- Define the probability function
def probability (n : ℕ) (jumpLength : ℝ) (maxDistance : ℝ) : ℝ :=
  sorry

-- Theorem statement
theorem frog_jump_probability :
  probability 5 1 1.5 = 1/8 :=
sorry

end frog_jump_probability_l2879_287935


namespace coloring_arrangements_l2879_287952

/-- The number of ways to arrange n distinct objects into n distinct positions -/
def permutations (n : ℕ) : ℕ := Nat.factorial n

/-- The number of parts to be colored -/
def num_parts : ℕ := 4

/-- The number of colors available -/
def num_colors : ℕ := 4

/-- Theorem: The number of ways to color 4 distinct parts with 4 distinct colors, 
    where each part must have a different color, is equal to 24 -/
theorem coloring_arrangements : permutations num_parts = 24 := by
  sorry

end coloring_arrangements_l2879_287952


namespace largest_n_sin_cos_inequality_l2879_287956

theorem largest_n_sin_cos_inequality : 
  ∃ (n : ℕ), n > 0 ∧ 
  (∀ (x : ℝ), (Real.sin x)^n + (Real.cos x)^n ≥ 2/n) ∧
  (∀ (m : ℕ), m > n → ∃ (y : ℝ), (Real.sin y)^m + (Real.cos y)^m < 2/m) :=
by sorry

end largest_n_sin_cos_inequality_l2879_287956


namespace smallest_coefficient_value_l2879_287992

-- Define the ratio condition
def ratio_condition (n : ℕ) : Prop :=
  (6^n) / (2^n) = 729

-- Define the function to get the coefficient of the term with the smallest coefficient
def smallest_coefficient (n : ℕ) : ℤ :=
  (-1)^(n - 3) * (Nat.choose n 3)

-- Theorem statement
theorem smallest_coefficient_value :
  ∃ n : ℕ, ratio_condition n ∧ smallest_coefficient n = -20 := by
  sorry

end smallest_coefficient_value_l2879_287992


namespace gcd_of_three_numbers_l2879_287941

theorem gcd_of_three_numbers : Nat.gcd 9240 (Nat.gcd 12240 33720) = 240 := by
  sorry

end gcd_of_three_numbers_l2879_287941


namespace largest_whole_number_inequality_l2879_287961

theorem largest_whole_number_inequality (x : ℕ) : x ≤ 3 ↔ (1 / 4 : ℚ) + (x : ℚ) / 5 < 1 := by
  sorry

end largest_whole_number_inequality_l2879_287961


namespace percentage_of_C_grades_l2879_287945

/-- Represents a grade with its lower and upper bounds -/
structure Grade where
  letter : String
  lower : Nat
  upper : Nat

/-- Checks if a score falls within a grade range -/
def isInGradeRange (score : Nat) (grade : Grade) : Bool :=
  score >= grade.lower ∧ score <= grade.upper

/-- The grading scale -/
def gradingScale : List Grade := [
  ⟨"A", 95, 100⟩,
  ⟨"A-", 90, 94⟩,
  ⟨"B+", 85, 89⟩,
  ⟨"B", 80, 84⟩,
  ⟨"C+", 77, 79⟩,
  ⟨"C", 73, 76⟩,
  ⟨"D", 70, 72⟩,
  ⟨"F", 0, 69⟩
]

/-- The list of student scores -/
def scores : List Nat := [98, 75, 86, 77, 60, 94, 72, 79, 69, 82, 70, 93, 74, 87, 78, 84, 95, 73]

/-- Theorem stating that the percentage of students who received a grade of C is 16.67% -/
theorem percentage_of_C_grades (ε : Real) (h : ε > 0) : 
  ∃ (p : Real), abs (p - 16.67) < ε ∧ 
  p = (100 : Real) * (scores.filter (fun score => 
    ∃ (g : Grade), g ∈ gradingScale ∧ g.letter = "C" ∧ isInGradeRange score g
  )).length / scores.length :=
sorry

end percentage_of_C_grades_l2879_287945


namespace gcd_84_126_l2879_287927

theorem gcd_84_126 : Nat.gcd 84 126 = 42 := by
  sorry

end gcd_84_126_l2879_287927


namespace george_hourly_rate_l2879_287934

/-- Calculates the hourly rate given total income and hours worked -/
def hourly_rate (total_income : ℚ) (total_hours : ℚ) : ℚ :=
  total_income / total_hours

theorem george_hourly_rate :
  let monday_hours : ℚ := 7
  let tuesday_hours : ℚ := 2
  let total_hours : ℚ := monday_hours + tuesday_hours
  let total_income : ℚ := 45
  hourly_rate total_income total_hours = 5 := by
  sorry

end george_hourly_rate_l2879_287934


namespace custom_operation_theorem_l2879_287922

theorem custom_operation_theorem (a b : ℚ) : 
  a ≠ 0 → b ≠ 0 → a - b = 9 → a / b = 20 → 1 / a + 1 / b = 19 / 60 := by
  sorry

end custom_operation_theorem_l2879_287922


namespace total_borrowed_by_lunchtime_l2879_287974

/-- Represents the number of books on a shelf at different times of the day -/
structure ShelfState where
  initial : ℕ
  added : ℕ
  borrowed_morning : ℕ
  borrowed_afternoon : ℕ
  remaining : ℕ

/-- Calculates the number of books borrowed by lunchtime for a given shelf -/
def borrowed_by_lunchtime (shelf : ShelfState) : ℕ :=
  shelf.initial + shelf.added - (shelf.remaining + shelf.borrowed_afternoon)

/-- The state of shelf A -/
def shelf_a : ShelfState := {
  initial := 100,
  added := 40,
  borrowed_morning := 0,  -- Unknown, to be calculated
  borrowed_afternoon := 30,
  remaining := 60
}

/-- The state of shelf B -/
def shelf_b : ShelfState := {
  initial := 150,
  added := 20,
  borrowed_morning := 50,
  borrowed_afternoon := 0,  -- Not needed for the calculation
  remaining := 80
}

/-- The state of shelf C -/
def shelf_c : ShelfState := {
  initial := 200,
  added := 10,
  borrowed_morning := 0,  -- Unknown, to be calculated
  borrowed_afternoon := 45,
  remaining := 200 + 10 - 130  -- 130 is total borrowed throughout the day
}

/-- Theorem stating that the total number of books borrowed by lunchtime across all shelves is 165 -/
theorem total_borrowed_by_lunchtime :
  borrowed_by_lunchtime shelf_a + borrowed_by_lunchtime shelf_b + borrowed_by_lunchtime shelf_c = 165 := by
  sorry

end total_borrowed_by_lunchtime_l2879_287974


namespace apple_percentage_is_fifty_percent_l2879_287912

-- Define the initial number of apples and oranges
def initial_apples : ℕ := 10
def initial_oranges : ℕ := 5

-- Define the number of oranges added
def added_oranges : ℕ := 5

-- Define the total number of fruits after adding oranges
def total_fruits : ℕ := initial_apples + initial_oranges + added_oranges

-- Define the percentage of apples
def apple_percentage : ℚ := (initial_apples : ℚ) / (total_fruits : ℚ) * 100

-- Theorem statement
theorem apple_percentage_is_fifty_percent :
  apple_percentage = 50 := by sorry

end apple_percentage_is_fifty_percent_l2879_287912


namespace binomial_expansion_problem_l2879_287999

theorem binomial_expansion_problem (n : ℕ) (a : ℕ → ℝ) : 
  (∀ k, 0 ≤ k ∧ k ≤ n → a k = (-1)^k * (n.choose k)) →
  (2 * (n.choose 2) - a (n - 5) = 0) →
  n = 8 := by sorry

end binomial_expansion_problem_l2879_287999


namespace smallest_games_for_score_l2879_287929

theorem smallest_games_for_score (win_points loss_points final_score : ℤ)
  (win_points_pos : win_points > 0)
  (loss_points_pos : loss_points > 0)
  (final_score_pos : final_score > 0)
  (h : win_points = 25 ∧ loss_points = 13 ∧ final_score = 2007) :
  ∃ (wins losses : ℕ),
    wins * win_points - losses * loss_points = final_score ∧
    wins + losses = 87 ∧
    ∀ (w l : ℕ), w * win_points - l * loss_points = final_score →
      w + l ≥ 87 := by
sorry

end smallest_games_for_score_l2879_287929


namespace replaced_person_weight_l2879_287978

/-- The weight of the replaced person given the conditions of the problem -/
def weight_of_replaced_person (initial_count : ℕ) (average_increase : ℚ) (new_person_weight : ℚ) : ℚ :=
  new_person_weight - (initial_count : ℚ) * average_increase

/-- Theorem stating that the weight of the replaced person is 67 kg -/
theorem replaced_person_weight :
  weight_of_replaced_person 8 (5/2) 87 = 67 := by
  sorry

end replaced_person_weight_l2879_287978


namespace range_of_k_l2879_287926

-- Define the propositions p and q
def p (k : ℝ) : Prop := ∃ (x y : ℝ), x^2/k + y^2/(4-k) = 1 ∧ k > 0 ∧ 4 - k > 0

def q (k : ℝ) : Prop := ∃ (x y : ℝ), x^2/(k-1) + y^2/(k-3) = 1 ∧ (k-1)*(k-3) < 0

-- State the theorem
theorem range_of_k (k : ℝ) : (p k ∨ q k) → 1 < k ∧ k < 4 := by
  sorry

end range_of_k_l2879_287926


namespace impossible_average_l2879_287931

theorem impossible_average (test1 test2 test3 test4 test5 test6 : ℕ) 
  (h1 : test1 = 85)
  (h2 : test2 = 79)
  (h3 : test3 = 92)
  (h4 : test4 = 84)
  (h5 : test5 = 88)
  (h6 : test6 = 7)
  : ¬ ∃ (test7 test8 : ℕ), (test1 + test2 + test3 + test4 + test5 + test6 + test7 + test8) / 8 = 87 :=
sorry

end impossible_average_l2879_287931


namespace plane_at_distance_from_point_and_through_axis_l2879_287919

-- Define the necessary types and structures
structure Point where
  x : ℝ
  y : ℝ
  z : ℝ

structure Plane where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

structure Sphere where
  center : Point
  radius : ℝ

def ProjectionAxis : Set Point := sorry

-- Define the distance between a point and a plane
def distancePointPlane (p : Point) (plane : Plane) : ℝ := sorry

-- Define a predicate for a plane passing through the projection axis
def passesThroughProjectionAxis (plane : Plane) : Prop := sorry

-- Define a predicate for a plane being tangent to a sphere
def isTangentTo (plane : Plane) (sphere : Sphere) : Prop := sorry

-- The main theorem
theorem plane_at_distance_from_point_and_through_axis
  (A : Point) (d : ℝ) (P : Plane) :
  (distancePointPlane A P = d ∧ passesThroughProjectionAxis P) ↔
  (isTangentTo P (Sphere.mk A d) ∧ passesThroughProjectionAxis P) := by
  sorry

end plane_at_distance_from_point_and_through_axis_l2879_287919


namespace cricket_bat_profit_percentage_cricket_bat_profit_is_twenty_percent_l2879_287972

/-- Calculates the profit percentage for seller A given the conditions of the cricket bat sale --/
theorem cricket_bat_profit_percentage 
  (cost_price_A : ℝ) 
  (profit_percentage_B : ℝ) 
  (selling_price_C : ℝ) : ℝ :=
  let selling_price_B := selling_price_C / (1 + profit_percentage_B)
  let profit_A := selling_price_B - cost_price_A
  let profit_percentage_A := (profit_A / cost_price_A) * 100
  
  profit_percentage_A

/-- The profit percentage for A when selling the cricket bat to B is 20% --/
theorem cricket_bat_profit_is_twenty_percent : 
  cricket_bat_profit_percentage 152 0.25 228 = 20 := by
  sorry

end cricket_bat_profit_percentage_cricket_bat_profit_is_twenty_percent_l2879_287972


namespace heptagon_coloring_l2879_287984

-- Define the color type
inductive Color
| Red
| Blue
| Yellow
| Green

-- Define the heptagon type
def Heptagon := Fin 7 → Color

-- Define the coloring conditions
def validColoring (h : Heptagon) : Prop :=
  ∀ i : Fin 7,
    (h i = Color.Red ∨ h i = Color.Blue →
      h ((i + 1) % 7) ≠ Color.Blue ∧ h ((i + 1) % 7) ≠ Color.Green ∧
      h ((i + 4) % 7) ≠ Color.Blue ∧ h ((i + 4) % 7) ≠ Color.Green) ∧
    (h i = Color.Yellow ∨ h i = Color.Green →
      h ((i + 1) % 7) ≠ Color.Red ∧ h ((i + 1) % 7) ≠ Color.Yellow ∧
      h ((i + 4) % 7) ≠ Color.Red ∧ h ((i + 4) % 7) ≠ Color.Yellow)

-- Theorem statement
theorem heptagon_coloring (h : Heptagon) (hvalid : validColoring h) :
  ∃ c : Color, ∀ i : Fin 7, h i = c :=
sorry

end heptagon_coloring_l2879_287984


namespace festival_attendance_l2879_287947

/-- Proves the attendance on the second day of a three-day festival -/
theorem festival_attendance (total : ℕ) (day1 day2 day3 : ℕ) : 
  total = 2700 →
  day2 = day1 / 2 →
  day3 = 3 * day1 →
  total = day1 + day2 + day3 →
  day2 = 300 := by
  sorry

end festival_attendance_l2879_287947


namespace correct_placement_l2879_287936

/-- Represents the participants in the competition -/
inductive Participant
| Olya
| Oleg
| Polya
| Pasha

/-- Represents the possible placements in the competition -/
inductive Place
| First
| Second
| Third
| Fourth

/-- Represents whether a participant is a boy or a girl -/
inductive Gender
| Boy
| Girl

/-- Defines the gender of each participant -/
def participantGender (p : Participant) : Gender :=
  match p with
  | Participant.Olya => Gender.Girl
  | Participant.Oleg => Gender.Boy
  | Participant.Polya => Gender.Girl
  | Participant.Pasha => Gender.Boy

/-- Defines whether a participant's name starts with 'O' -/
def nameStartsWithO (p : Participant) : Prop :=
  match p with
  | Participant.Olya => true
  | Participant.Oleg => true
  | Participant.Polya => false
  | Participant.Pasha => false

/-- Defines whether a place is odd-numbered -/
def isOddPlace (p : Place) : Prop :=
  match p with
  | Place.First => true
  | Place.Second => false
  | Place.Third => true
  | Place.Fourth => false

/-- Defines whether two places are consecutive -/
def areConsecutivePlaces (p1 p2 : Place) : Prop :=
  (p1 = Place.First ∧ p2 = Place.Second) ∨
  (p1 = Place.Second ∧ p2 = Place.Third) ∨
  (p1 = Place.Third ∧ p2 = Place.Fourth) ∨
  (p2 = Place.First ∧ p1 = Place.Second) ∨
  (p2 = Place.Second ∧ p1 = Place.Third) ∨
  (p2 = Place.Third ∧ p1 = Place.Fourth)

/-- Represents the final placement of participants -/
def Placement := Participant → Place

/-- Theorem stating the correct placement given the conditions -/
theorem correct_placement (placement : Placement) : 
  (∃! p : Participant, placement p = Place.First) ∧
  (∃! p : Participant, placement p = Place.Second) ∧
  (∃! p : Participant, placement p = Place.Third) ∧
  (∃! p : Participant, placement p = Place.Fourth) ∧
  (∃! p : Participant, (placement p = Place.First → 
    (∀ p' : Place, isOddPlace p' → ∃ p'' : Participant, placement p'' = p' ∧ participantGender p'' = Gender.Boy) ∧
    (areConsecutivePlaces (placement Participant.Oleg) (placement Participant.Olya)) ∧
    (∀ p' : Place, isOddPlace p' → ∃ p'' : Participant, placement p'' = p' ∧ nameStartsWithO p''))) →
  placement Participant.Oleg = Place.First ∧
  placement Participant.Olya = Place.Second ∧
  placement Participant.Polya = Place.Third ∧
  placement Participant.Pasha = Place.Fourth :=
by sorry

end correct_placement_l2879_287936


namespace fathers_sons_age_product_l2879_287997

theorem fathers_sons_age_product (father_age son_age : ℕ) : 
  father_age > 0 ∧ son_age > 0 ∧
  father_age = 7 * (son_age / 3) ∧
  (father_age + 6) = 2 * (son_age + 6) →
  father_age * son_age = 756 := by
sorry

end fathers_sons_age_product_l2879_287997


namespace not_right_triangle_1_5_2_3_l2879_287965

/-- A function that checks if three numbers can form the sides of a right triangle -/
def is_right_triangle (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ (a^2 + b^2 = c^2 ∨ a^2 + c^2 = b^2 ∨ b^2 + c^2 = a^2)

/-- Theorem stating that 1.5, 2, and 3 cannot form the sides of a right triangle -/
theorem not_right_triangle_1_5_2_3 : ¬ is_right_triangle 1.5 2 3 := by
  sorry

end not_right_triangle_1_5_2_3_l2879_287965


namespace inequality_proof_l2879_287963

theorem inequality_proof (a b : ℝ) (h : a < b ∧ b < 0) : a^2 > a*b ∧ a*b > b^2 := by
  sorry

end inequality_proof_l2879_287963


namespace wendy_running_distance_l2879_287960

/-- The distance Wendy walked in miles -/
def distance_walked : ℝ := 9.17

/-- The additional distance Wendy ran compared to what she walked in miles -/
def additional_distance_ran : ℝ := 10.67

/-- The total distance Wendy ran in miles -/
def distance_ran : ℝ := distance_walked + additional_distance_ran

theorem wendy_running_distance : distance_ran = 19.84 := by
  sorry

end wendy_running_distance_l2879_287960


namespace double_average_marks_l2879_287915

theorem double_average_marks (n : ℕ) (initial_avg : ℝ) (h1 : n = 30) (h2 : initial_avg = 45) :
  let total_marks := n * initial_avg
  let new_total_marks := 2 * total_marks
  let new_avg := new_total_marks / n
  new_avg = 90 := by sorry

end double_average_marks_l2879_287915


namespace intercept_sum_l2879_287962

/-- A line is described by the equation y + 3 = -3(x + 2) -/
def line_equation (x y : ℝ) : Prop := y + 3 = -3 * (x + 2)

/-- The x-intercept of the line -/
def x_intercept : ℝ := -3

/-- The y-intercept of the line -/
def y_intercept : ℝ := -9

theorem intercept_sum :
  line_equation x_intercept 0 ∧ line_equation 0 y_intercept ∧ x_intercept + y_intercept = -12 := by
  sorry

end intercept_sum_l2879_287962


namespace trapezoid_area_theorem_l2879_287993

/-- Represents a trapezoid with given diagonals and height -/
structure Trapezoid where
  diagonal1 : ℝ
  diagonal2 : ℝ
  height : ℝ

/-- Calculates the possible areas of a trapezoid given its diagonals and height -/
def trapezoid_areas (t : Trapezoid) : Set ℝ :=
  {900, 780}

/-- Theorem stating that a trapezoid with diagonals 17 and 113, and height 15 has an area of either 900 or 780 -/
theorem trapezoid_area_theorem (t : Trapezoid) 
    (h1 : t.diagonal1 = 17) 
    (h2 : t.diagonal2 = 113) 
    (h3 : t.height = 15) : 
  ∃ (area : ℝ), area ∈ trapezoid_areas t ∧ (area = 900 ∨ area = 780) := by
  sorry


end trapezoid_area_theorem_l2879_287993


namespace village_population_l2879_287980

theorem village_population (P : ℝ) : 0.85 * (0.95 * P) = 3294 → P = 4080 := by
  sorry

end village_population_l2879_287980


namespace system_solution_l2879_287933

theorem system_solution (x y : ℝ) (h1 : 3 * x + y = 21) (h2 : x + 3 * y = 1) : 2 * x + 2 * y = 11 := by
  sorry

end system_solution_l2879_287933


namespace black_beads_fraction_l2879_287966

/-- Proves that the fraction of black beads pulled out is 1/6 given the initial conditions -/
theorem black_beads_fraction (total_white : ℕ) (total_black : ℕ) (total_pulled : ℕ) :
  total_white = 51 →
  total_black = 90 →
  total_pulled = 32 →
  (total_pulled - (total_white / 3)) / total_black = 1 / 6 := by
  sorry

end black_beads_fraction_l2879_287966


namespace matrix_pattern_l2879_287989

/-- Given a 2x2 matrix [[a, 2], [5, 6]] where a is unknown, 
    if (5 * 6) = (a * 2) * 3, then a = 5 -/
theorem matrix_pattern (a : ℝ) : (5 * 6 : ℝ) = (a * 2) * 3 → a = 5 := by
  sorry

end matrix_pattern_l2879_287989


namespace base12_remainder_theorem_l2879_287942

/-- Converts a base-12 integer to decimal --/
def base12ToDecimal (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (12 ^ (digits.length - 1 - i))) 0

/-- The base-12 representation of 2543₁₂ --/
def base12Number : List Nat := [2, 5, 4, 3]

/-- The theorem stating that the remainder of 2543₁₂ divided by 9 is 8 --/
theorem base12_remainder_theorem :
  (base12ToDecimal base12Number) % 9 = 8 := by
  sorry

end base12_remainder_theorem_l2879_287942


namespace go_stones_problem_l2879_287951

theorem go_stones_problem (total : ℕ) (difference_result : ℕ) 
  (h_total : total = 6000)
  (h_difference : difference_result = 4800) :
  ∃ (white black : ℕ), 
    white + black = total ∧ 
    white > black ∧ 
    total - (white - black) = difference_result ∧
    white = 3600 := by
  sorry

end go_stones_problem_l2879_287951


namespace digit_sum_power_property_l2879_287906

/-- Sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- The property that the fifth power of the sum of digits equals the square of the number -/
def has_property (n : ℕ) : Prop := (sum_of_digits n)^5 = n^2

/-- Theorem stating that only 1 and 243 satisfy the property -/
theorem digit_sum_power_property :
  ∀ n : ℕ, has_property n ↔ n = 1 ∨ n = 243 := by sorry

end digit_sum_power_property_l2879_287906


namespace f_max_min_on_interval_l2879_287950

-- Define the function f(x) = x^3 - 3x + 1
def f (x : ℝ) : ℝ := x^3 - 3*x + 1

-- Define the closed interval [-3, 0]
def I : Set ℝ := Set.Icc (-3) 0

-- Statement of the theorem
theorem f_max_min_on_interval :
  (∃ x ∈ I, ∀ y ∈ I, f y ≤ f x) ∧
  (∃ x ∈ I, ∀ y ∈ I, f x ≤ f y) ∧
  (∃ x ∈ I, f x = 3) ∧
  (∃ x ∈ I, f x = -17) :=
sorry

end f_max_min_on_interval_l2879_287950


namespace bobby_candy_consumption_l2879_287964

theorem bobby_candy_consumption (initial : ℕ) (additional : ℕ) : 
  initial = 26 → additional = 17 → initial + additional = 43 := by
  sorry

end bobby_candy_consumption_l2879_287964


namespace fourth_degree_equation_roots_l2879_287982

theorem fourth_degree_equation_roots :
  ∃ (r₁ r₂ r₃ r₄ : ℂ),
    (∀ x : ℂ, 3 * x^4 + 2 * x^3 - 7 * x^2 + 2 * x + 3 = 0 ↔ x = r₁ ∨ x = r₂ ∨ x = r₃ ∨ x = r₄) :=
by sorry

end fourth_degree_equation_roots_l2879_287982


namespace range_of_a_l2879_287969

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, a * x^2 + a * x + 1 > 0) ∧ 
  (¬ ∃ x : ℝ, x^2 - x + a = 0) ∧
  ((∀ x : ℝ, a * x^2 + a * x + 1 > 0) ∨ (∃ x : ℝ, x^2 - x + a = 0)) ∧
  ¬((∀ x : ℝ, a * x^2 + a * x + 1 > 0) ∧ (∃ x : ℝ, x^2 - x + a = 0)) →
  a > 1/4 ∧ a < 4 :=
by sorry

end range_of_a_l2879_287969


namespace line_plane_relationships_l2879_287985

-- Define the types for lines and planes
def Line : Type := sorry
def Plane : Type := sorry

-- Define the relationships between lines and planes
def parallel_line_plane (l : Line) (p : Plane) : Prop := sorry
def perpendicular_line_plane (l : Line) (p : Plane) : Prop := sorry
def parallel_planes (p1 p2 : Plane) : Prop := sorry
def perpendicular_planes (p1 p2 : Plane) : Prop := sorry
def perpendicular_lines (l1 l2 : Line) : Prop := sorry

-- State the theorem
theorem line_plane_relationships 
  (m n : Line) (α β : Plane) 
  (h_diff_lines : m ≠ n) 
  (h_diff_planes : α ≠ β) :
  ((perpendicular_line_plane m α ∧ 
    parallel_line_plane n β ∧ 
    parallel_planes α β) → 
   perpendicular_lines m n) ∧
  ((perpendicular_line_plane m α ∧ 
    perpendicular_line_plane n β ∧ 
    perpendicular_planes α β) → 
   perpendicular_lines m n) :=
by sorry

end line_plane_relationships_l2879_287985


namespace fundraiser_result_l2879_287939

def fundraiser (num_students : ℕ) (initial_needed : ℕ) (additional_needed : ℕ) 
               (day1 : ℕ) (day2 : ℕ) (day3 : ℕ) (num_half_days : ℕ) : ℕ :=
  let total_per_student := initial_needed + additional_needed
  let total_needed := num_students * total_per_student
  let first_three_days := day1 + day2 + day3
  let half_day_amount := first_three_days / 2
  let total_raised := first_three_days + num_half_days * half_day_amount
  total_raised - total_needed

theorem fundraiser_result : 
  fundraiser 6 450 475 600 900 400 4 = 150 := by
  sorry

end fundraiser_result_l2879_287939
