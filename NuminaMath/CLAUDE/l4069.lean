import Mathlib

namespace NUMINAMATH_CALUDE_power_value_from_equation_l4069_406911

theorem power_value_from_equation (x y : ℝ) 
  (h : |x - 2| + Real.sqrt (y + 3) = 0) : 
  y ^ x = 9 := by sorry

end NUMINAMATH_CALUDE_power_value_from_equation_l4069_406911


namespace NUMINAMATH_CALUDE_crescent_moon_division_l4069_406980

/-- The maximum number of parts a crescent moon can be divided into with n straight cuts -/
def max_parts (n : ℕ) : ℕ := (n^2 + 3*n) / 2 + 1

/-- The number of straight cuts used -/
def num_cuts : ℕ := 5

theorem crescent_moon_division :
  max_parts num_cuts = 21 :=
sorry

end NUMINAMATH_CALUDE_crescent_moon_division_l4069_406980


namespace NUMINAMATH_CALUDE_probability_at_least_one_from_subset_l4069_406995

def total_elements : ℕ := 4
def selected_elements : ℕ := 2
def subset_size : ℕ := 2

theorem probability_at_least_one_from_subset :
  (1 : ℚ) - (Nat.choose (total_elements - subset_size) selected_elements : ℚ) / 
  (Nat.choose total_elements selected_elements : ℚ) = 5/6 := by sorry

end NUMINAMATH_CALUDE_probability_at_least_one_from_subset_l4069_406995


namespace NUMINAMATH_CALUDE_smallest_k_with_remainders_l4069_406902

theorem smallest_k_with_remainders : ∃! k : ℕ,
  k > 1 ∧
  k % 19 = 1 ∧
  k % 7 = 1 ∧
  k % 3 = 1 ∧
  ∀ m : ℕ, m > 1 → m % 19 = 1 → m % 7 = 1 → m % 3 = 1 → k ≤ m :=
by
  use 400
  sorry

end NUMINAMATH_CALUDE_smallest_k_with_remainders_l4069_406902


namespace NUMINAMATH_CALUDE_gregs_shopping_expenditure_l4069_406994

/-- Greg's shopping expenditure theorem -/
theorem gregs_shopping_expenditure (shirt_cost shoes_cost : ℕ) : 
  shirt_cost + shoes_cost = 300 →
  shoes_cost = 2 * shirt_cost + 9 →
  shirt_cost = 97 := by
  sorry

end NUMINAMATH_CALUDE_gregs_shopping_expenditure_l4069_406994


namespace NUMINAMATH_CALUDE_jeremy_songs_l4069_406950

def songs_problem (songs_yesterday : ℕ) (difference : ℕ) : Prop :=
  let songs_today : ℕ := songs_yesterday + difference
  let total_songs : ℕ := songs_yesterday + songs_today
  total_songs = 23

theorem jeremy_songs :
  songs_problem 9 5 := by
  sorry

end NUMINAMATH_CALUDE_jeremy_songs_l4069_406950


namespace NUMINAMATH_CALUDE_circle_area_from_polar_equation_l4069_406960

/-- The area of the circle described by the polar equation r = 3 cos θ - 4 sin θ is 25π/4 -/
theorem circle_area_from_polar_equation :
  let r : ℝ → ℝ := λ θ => 3 * Real.cos θ - 4 * Real.sin θ
  ∃ (center : ℝ × ℝ) (radius : ℝ),
    (∀ θ, (r θ * Real.cos θ - center.1)^2 + (r θ * Real.sin θ - center.2)^2 = radius^2) ∧
    π * radius^2 = 25 * π / 4 := by
  sorry

end NUMINAMATH_CALUDE_circle_area_from_polar_equation_l4069_406960


namespace NUMINAMATH_CALUDE_trapezoid_area_trapezoid_area_proof_l4069_406932

/-- The area of a trapezoid bounded by y = x + 1, y = 12, y = 7, and the y-axis -/
theorem trapezoid_area : ℝ :=
  let line1 : ℝ → ℝ := λ x ↦ x + 1
  let line2 : ℝ → ℝ := λ _ ↦ 12
  let line3 : ℝ → ℝ := λ _ ↦ 7
  let y_axis : ℝ → ℝ := λ _ ↦ 0
  42.5
  
#check trapezoid_area

/-- Proof that the area of the trapezoid is 42.5 square units -/
theorem trapezoid_area_proof : trapezoid_area = 42.5 := by
  sorry

end NUMINAMATH_CALUDE_trapezoid_area_trapezoid_area_proof_l4069_406932


namespace NUMINAMATH_CALUDE_max_towns_is_four_l4069_406938

/-- Represents the type of link between two towns -/
inductive LinkType
| Air
| Bus
| Train

/-- Represents a town -/
structure Town where
  id : Nat

/-- Represents a link between two towns -/
structure Link where
  town1 : Town
  town2 : Town
  linkType : LinkType

/-- A network of towns and their connections -/
structure TownNetwork where
  towns : List Town
  links : List Link

/-- Checks if a given network satisfies all the required conditions -/
def isValidNetwork (network : TownNetwork) : Prop :=
  -- Each pair of towns is linked by exactly one type of link
  (∀ t1 t2 : Town, t1 ∈ network.towns → t2 ∈ network.towns → t1 ≠ t2 →
    ∃! link : Link, link ∈ network.links ∧ 
    ((link.town1 = t1 ∧ link.town2 = t2) ∨ (link.town1 = t2 ∧ link.town2 = t1))) ∧
  -- At least one pair is linked by each type
  (∃ link : Link, link ∈ network.links ∧ link.linkType = LinkType.Air) ∧
  (∃ link : Link, link ∈ network.links ∧ link.linkType = LinkType.Bus) ∧
  (∃ link : Link, link ∈ network.links ∧ link.linkType = LinkType.Train) ∧
  -- No town has all three types of links
  (∀ t : Town, t ∈ network.towns →
    ¬(∃ l1 l2 l3 : Link, l1 ∈ network.links ∧ l2 ∈ network.links ∧ l3 ∈ network.links ∧
      (l1.town1 = t ∨ l1.town2 = t) ∧ (l2.town1 = t ∨ l2.town2 = t) ∧ (l3.town1 = t ∨ l3.town2 = t) ∧
      l1.linkType = LinkType.Air ∧ l2.linkType = LinkType.Bus ∧ l3.linkType = LinkType.Train)) ∧
  -- No three towns form a triangle with all sides of the same type
  (∀ t1 t2 t3 : Town, t1 ∈ network.towns → t2 ∈ network.towns → t3 ∈ network.towns →
    t1 ≠ t2 ∧ t2 ≠ t3 ∧ t1 ≠ t3 →
    ¬(∃ l1 l2 l3 : Link, l1 ∈ network.links ∧ l2 ∈ network.links ∧ l3 ∈ network.links ∧
      ((l1.town1 = t1 ∧ l1.town2 = t2) ∨ (l1.town1 = t2 ∧ l1.town2 = t1)) ∧
      ((l2.town1 = t2 ∧ l2.town2 = t3) ∨ (l2.town1 = t3 ∧ l2.town2 = t2)) ∧
      ((l3.town1 = t3 ∧ l3.town2 = t1) ∨ (l3.town1 = t1 ∧ l3.town2 = t3)) ∧
      l1.linkType = l2.linkType ∧ l2.linkType = l3.linkType))

/-- The theorem stating that the maximum number of towns in a valid network is 4 -/
theorem max_towns_is_four :
  (∃ network : TownNetwork, isValidNetwork network ∧ network.towns.length = 4) ∧
  (∀ network : TownNetwork, isValidNetwork network → network.towns.length ≤ 4) :=
sorry

end NUMINAMATH_CALUDE_max_towns_is_four_l4069_406938


namespace NUMINAMATH_CALUDE_total_flowers_l4069_406972

theorem total_flowers (num_pots : ℕ) (flowers_per_pot : ℕ) 
  (h1 : num_pots = 544) (h2 : flowers_per_pot = 32) : 
  num_pots * flowers_per_pot = 17408 := by
  sorry

end NUMINAMATH_CALUDE_total_flowers_l4069_406972


namespace NUMINAMATH_CALUDE_power_zero_is_one_l4069_406954

theorem power_zero_is_one (x : ℝ) (h : x ≠ 0) : x^0 = 1 := by
  sorry

end NUMINAMATH_CALUDE_power_zero_is_one_l4069_406954


namespace NUMINAMATH_CALUDE_parallel_lines_circle_intersection_l4069_406924

theorem parallel_lines_circle_intersection (r : ℝ) : 
  ∀ d : ℝ, 
    (17682 + (21/4) * d^2 = 42 * r^2) ∧ 
    (4394 + (117/4) * d^2 = 26 * r^2) → 
    d = Real.sqrt 127 := by
  sorry

end NUMINAMATH_CALUDE_parallel_lines_circle_intersection_l4069_406924


namespace NUMINAMATH_CALUDE_last_term_is_one_l4069_406979

/-- A sequence is k-th order repeatable if there exist two sets of consecutive k terms that match in order. -/
def kth_order_repeatable (a : ℕ → Fin 2) (m k : ℕ) : Prop :=
  ∃ i j, i ≠ j ∧ i + k ≤ m ∧ j + k ≤ m ∧ ∀ t, t < k → a (i + t) = a (j + t)

theorem last_term_is_one
  (a : ℕ → Fin 2)
  (m : ℕ)
  (h_m : m ≥ 3)
  (h_not_5th : ¬ kth_order_repeatable a m 5)
  (h_5th_after : ∀ b : Fin 2, kth_order_repeatable (Function.update a m b) (m + 1) 5)
  (h_a4 : a 4 ≠ 1) :
  a m = 1 :=
sorry

end NUMINAMATH_CALUDE_last_term_is_one_l4069_406979


namespace NUMINAMATH_CALUDE_line_segments_in_proportion_l4069_406969

theorem line_segments_in_proportion : 
  let a : ℝ := 2
  let b : ℝ := Real.sqrt 5
  let c : ℝ := 2 * Real.sqrt 3
  let d : ℝ := Real.sqrt 15
  a * d = b * c := by sorry

end NUMINAMATH_CALUDE_line_segments_in_proportion_l4069_406969


namespace NUMINAMATH_CALUDE_six_grade_assignments_l4069_406901

/-- Number of ways to assign n grades, where each grade is 2, 3, or 4, and no two consecutive grades can both be 2 -/
def gradeAssignments : ℕ → ℕ
  | 0 => 1
  | 1 => 3
  | n + 2 => 2 * gradeAssignments (n + 1) + 2 * gradeAssignments n

/-- The number of ways to assign 6 grades under the given conditions is 448 -/
theorem six_grade_assignments : gradeAssignments 6 = 448 := by
  sorry

end NUMINAMATH_CALUDE_six_grade_assignments_l4069_406901


namespace NUMINAMATH_CALUDE_yellow_straight_probability_l4069_406905

structure Garden where
  roses : ℝ
  daffodils : ℝ
  tulips : ℝ
  green_prob : ℝ
  straight_prob : ℝ
  rose_straight_prob : ℝ
  daffodil_curved_prob : ℝ
  tulip_straight_prob : ℝ

def is_valid_garden (g : Garden) : Prop :=
  g.roses + g.daffodils + g.tulips = 1 ∧
  g.roses = 1/4 ∧
  g.daffodils = 1/2 ∧
  g.tulips = 1/4 ∧
  g.green_prob = 2/3 ∧
  g.straight_prob = 1/2 ∧
  g.rose_straight_prob = 1/6 ∧
  g.daffodil_curved_prob = 1/3 ∧
  g.tulip_straight_prob = 1/8

theorem yellow_straight_probability (g : Garden) 
  (h : is_valid_garden g) : 
  (1 - g.green_prob) * g.straight_prob = 1/6 := by
  sorry

end NUMINAMATH_CALUDE_yellow_straight_probability_l4069_406905


namespace NUMINAMATH_CALUDE_eds_remaining_money_l4069_406992

/-- Calculates the remaining money after a hotel stay -/
def remaining_money (initial_amount : ℝ) (night_rate : ℝ) (morning_rate : ℝ) 
  (night_hours : ℝ) (morning_hours : ℝ) : ℝ :=
  initial_amount - (night_rate * night_hours + morning_rate * morning_hours)

/-- Theorem: Ed's remaining money after his hotel stay -/
theorem eds_remaining_money :
  remaining_money 80 1.5 2 6 4 = 63 := by
  sorry

end NUMINAMATH_CALUDE_eds_remaining_money_l4069_406992


namespace NUMINAMATH_CALUDE_turner_rides_l4069_406971

theorem turner_rides (rollercoaster_rides : ℕ) (ferris_wheel_rides : ℕ) 
  (rollercoaster_cost : ℕ) (catapult_cost : ℕ) (ferris_wheel_cost : ℕ) 
  (total_tickets : ℕ) :
  rollercoaster_rides = 3 →
  ferris_wheel_rides = 1 →
  rollercoaster_cost = 4 →
  catapult_cost = 4 →
  ferris_wheel_cost = 1 →
  total_tickets = 21 →
  ∃ catapult_rides : ℕ, 
    catapult_rides * catapult_cost + 
    rollercoaster_rides * rollercoaster_cost + 
    ferris_wheel_rides * ferris_wheel_cost = total_tickets ∧
    catapult_rides = 2 :=
by sorry

end NUMINAMATH_CALUDE_turner_rides_l4069_406971


namespace NUMINAMATH_CALUDE_binary_110011_equals_51_l4069_406966

def binary_to_decimal (b : List Bool) : ℕ :=
  b.enum.foldl (fun acc (i, bit) => acc + if bit then 2^i else 0) 0

theorem binary_110011_equals_51 :
  binary_to_decimal [true, true, false, false, true, true] = 51 := by sorry

end NUMINAMATH_CALUDE_binary_110011_equals_51_l4069_406966


namespace NUMINAMATH_CALUDE_circle_intersection_and_tangent_line_l4069_406933

theorem circle_intersection_and_tangent_line :
  ∃ (A B C : ℝ),
    (∀ x y : ℝ, A * x^2 + A * y^2 + B * x + C = 0) ∧
    (∀ x y : ℝ, (A * x^2 + A * y^2 + B * x + C = 0) →
      ((x^2 + y^2 - 1 = 0) ∧ (x^2 - 4*x + y^2 = 0)) ∨
      ((x^2 + y^2 - 1 ≠ 0) ∧ (x^2 - 4*x + y^2 ≠ 0))) ∧
    (∃ x₀ y₀ : ℝ,
      A * x₀^2 + A * y₀^2 + B * x₀ + C = 0 ∧
      x₀ - Real.sqrt 3 * y₀ - 6 = 0 ∧
      ∀ x y : ℝ, A * x^2 + A * y^2 + B * x + C = 0 →
        (x - Real.sqrt 3 * y - 6)^2 ≥ (x₀ - Real.sqrt 3 * y₀ - 6)^2) :=
by sorry

end NUMINAMATH_CALUDE_circle_intersection_and_tangent_line_l4069_406933


namespace NUMINAMATH_CALUDE_abs_negative_2022_l4069_406951

theorem abs_negative_2022 : |(-2022 : ℤ)| = 2022 := by
  sorry

end NUMINAMATH_CALUDE_abs_negative_2022_l4069_406951


namespace NUMINAMATH_CALUDE_polynomial_evaluation_l4069_406958

theorem polynomial_evaluation (x : ℝ) (h : x = 2) : 3 * x^2 + 5 * x - 2 = 20 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_evaluation_l4069_406958


namespace NUMINAMATH_CALUDE_matrix_power_1000_l4069_406944

def A : Matrix (Fin 2) (Fin 2) ℝ := !![1, 0; 2, 1]

theorem matrix_power_1000 :
  A ^ 1000 = !![1, 0; 2000, 1] := by sorry

end NUMINAMATH_CALUDE_matrix_power_1000_l4069_406944


namespace NUMINAMATH_CALUDE_painted_cube_theorem_l4069_406915

/-- Represents a cube composed of unit cubes -/
structure PaintedCube where
  size : ℕ
  totalCubes : ℕ
  surfacePainted : Bool

/-- Counts the number of unit cubes with a specific number of faces painted -/
def countPaintedFaces (cube : PaintedCube) (numFaces : ℕ) : ℕ :=
  match numFaces with
  | 3 => 8
  | 2 => 12 * (cube.size - 2)
  | 1 => 6 * (cube.size - 2)^2
  | 0 => (cube.size - 2)^3
  | _ => 0

theorem painted_cube_theorem (cube : PaintedCube) 
  (h1 : cube.size = 10) 
  (h2 : cube.totalCubes = 1000) 
  (h3 : cube.surfacePainted = true) :
  (countPaintedFaces cube 3 = 8) ∧
  (countPaintedFaces cube 2 = 96) ∧
  (countPaintedFaces cube 1 = 384) ∧
  (countPaintedFaces cube 0 = 512) := by
  sorry

end NUMINAMATH_CALUDE_painted_cube_theorem_l4069_406915


namespace NUMINAMATH_CALUDE_cube_root_function_l4069_406936

theorem cube_root_function (k : ℝ) :
  (∃ y : ℝ, y = k * (27 : ℝ)^(1/3) ∧ y = 3 * Real.sqrt 3) →
  k * (8 : ℝ)^(1/3) = 2 * Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_cube_root_function_l4069_406936


namespace NUMINAMATH_CALUDE_divisibility_of_n_l4069_406999

theorem divisibility_of_n : ∀ (n : ℕ),
  n = (2^4 - 1) * (3^6 - 1) * (5^10 - 1) * (7^12 - 1) →
  5 ∣ n ∧ 7 ∣ n ∧ 11 ∣ n ∧ 13 ∣ n :=
by
  sorry

end NUMINAMATH_CALUDE_divisibility_of_n_l4069_406999


namespace NUMINAMATH_CALUDE_candy_bar_sales_earnings_candy_bar_sales_proof_l4069_406964

/-- Calculates the total amount earned from candy bar sales given the specified conditions --/
theorem candy_bar_sales_earnings (num_members : ℕ) (type_a_price type_b_price : ℚ) 
  (avg_total_bars avg_type_a avg_type_b : ℕ) : ℚ :=
  let total_bars := num_members * avg_total_bars
  let total_type_a := num_members * avg_type_a
  let total_type_b := num_members * avg_type_b
  let earnings_type_a := total_type_a * type_a_price
  let earnings_type_b := total_type_b * type_b_price
  earnings_type_a + earnings_type_b

/-- Proves that the group earned $95 from their candy bar sales --/
theorem candy_bar_sales_proof :
  candy_bar_sales_earnings 20 (1/2) (3/4) 8 5 3 = 95 := by
  sorry

end NUMINAMATH_CALUDE_candy_bar_sales_earnings_candy_bar_sales_proof_l4069_406964


namespace NUMINAMATH_CALUDE_find_d_value_l4069_406977

theorem find_d_value (x y d : ℝ) : 
  7^(3*x - 1) * 3^(4*y - 3) = 49^x * d^y ∧ x + y = 4 → d = 27 := by
  sorry

end NUMINAMATH_CALUDE_find_d_value_l4069_406977


namespace NUMINAMATH_CALUDE_equation_solution_inequality_solution_l4069_406917

-- Part 1: Equation solution
theorem equation_solution :
  ∀ x : ℚ, (1 / (x + 1) - 1 / (x + 2) = 1 / (x + 3) - 1 / (x + 4)) ↔ (x = -5/2) :=
sorry

-- Part 2: Inequality solution
theorem inequality_solution (a : ℚ) (x : ℚ) :
  x^2 - (a + 1) * x + a ≤ 0 ↔
    (a = 1 ∧ x = 1) ∨
    (a < 1 ∧ a ≤ x ∧ x ≤ 1) ∨
    (a > 1 ∧ 1 ≤ x ∧ x ≤ a) :=
sorry

end NUMINAMATH_CALUDE_equation_solution_inequality_solution_l4069_406917


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l4069_406926

-- Define the sets A and B
def A : Set ℝ := {x | |x - 1| < 2}
def B : Set ℝ := {x | x^2 + x - 2 > 0}

-- State the theorem
theorem intersection_of_A_and_B :
  A ∩ B = {x : ℝ | 1 < x ∧ x < 3} :=
sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l4069_406926


namespace NUMINAMATH_CALUDE_min_ab_value_l4069_406941

theorem min_ab_value (a b : ℕ+) (h : (a : ℚ)⁻¹ + (3 * b : ℚ)⁻¹ = (9 : ℚ)⁻¹) :
  (a * b : ℕ) ≥ 60 ∧ ∃ (a₀ b₀ : ℕ+), (a₀ : ℚ)⁻¹ + (3 * b₀ : ℚ)⁻¹ = (9 : ℚ)⁻¹ ∧ (a₀ * b₀ : ℕ) = 60 :=
by sorry

end NUMINAMATH_CALUDE_min_ab_value_l4069_406941


namespace NUMINAMATH_CALUDE_fort_soldiers_count_l4069_406973

/-- The initial number of soldiers in the fort -/
def initial_soldiers : ℕ := 480

/-- The number of additional soldiers joining the fort -/
def additional_soldiers : ℕ := 528

/-- The number of days provisions last with initial soldiers -/
def initial_days : ℕ := 30

/-- The number of days provisions last with additional soldiers -/
def new_days : ℕ := 25

/-- The daily consumption per soldier initially (in kg) -/
def initial_consumption : ℚ := 3

/-- The daily consumption per soldier after additional soldiers join (in kg) -/
def new_consumption : ℚ := 5/2

theorem fort_soldiers_count :
  initial_soldiers * initial_consumption * initial_days =
  (initial_soldiers + additional_soldiers) * new_consumption * new_days :=
sorry

end NUMINAMATH_CALUDE_fort_soldiers_count_l4069_406973


namespace NUMINAMATH_CALUDE_sqrt_32_div_sqrt_8_eq_2_l4069_406906

theorem sqrt_32_div_sqrt_8_eq_2 : Real.sqrt 32 / Real.sqrt 8 = 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_32_div_sqrt_8_eq_2_l4069_406906


namespace NUMINAMATH_CALUDE_tablet_down_payment_is_100_l4069_406947

/-- The down payment for a tablet purchase with given conditions. -/
def tablet_down_payment (cash_price installment_total first_4_months next_4_months last_4_months cash_savings : ℕ) : ℕ :=
  installment_total - (4 * first_4_months + 4 * next_4_months + 4 * last_4_months)

/-- Theorem stating that the down payment for the tablet is $100 under given conditions. -/
theorem tablet_down_payment_is_100 :
  tablet_down_payment 450 520 40 35 30 70 = 100 := by
  sorry

end NUMINAMATH_CALUDE_tablet_down_payment_is_100_l4069_406947


namespace NUMINAMATH_CALUDE_quadratic_roots_distinct_l4069_406953

theorem quadratic_roots_distinct (a b c : ℝ) (h : a ≠ 0) :
  let discriminant := b^2 - 4*a*c
  a*x^2 + b*x + c = 0 ∧ discriminant > 0 → ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ a*x₁^2 + b*x₁ + c = 0 ∧ a*x₂^2 + b*x₂ + c = 0 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_distinct_l4069_406953


namespace NUMINAMATH_CALUDE_test_scores_l4069_406908

theorem test_scores (joao_score claudia_score : ℕ) : 
  (10 ≤ joao_score ∧ joao_score < 100) →  -- João's score is a two-digit number
  (10 ≤ claudia_score ∧ claudia_score < 100) →  -- Cláudia's score is a two-digit number
  claudia_score = joao_score + 13 →  -- Cláudia scored 13 points more than João
  joao_score + claudia_score = 149 →  -- Their combined score is 149
  joao_score = 68 ∧ claudia_score = 81 :=
by sorry

end NUMINAMATH_CALUDE_test_scores_l4069_406908


namespace NUMINAMATH_CALUDE_smallest_number_of_weights_l4069_406963

/-- A function that determines if a given number of weights can measure all masses -/
def can_measure_all (n : ℕ) : Prop :=
  ∃ (weights : Fin n → ℝ), 
    (∀ i, weights i ≥ 0.01) ∧ 
    (∀ m : ℝ, 0 ≤ m ∧ m ≤ 20.2 → 
      ∃ (subset : Fin n → Bool), 
        abs (m - (Finset.sum (Finset.filter (λ i => subset i = true) Finset.univ) weights)) ≤ 0.01)

theorem smallest_number_of_weights : 
  (∀ k < 2020, ¬ can_measure_all k) ∧ can_measure_all 2020 :=
sorry

end NUMINAMATH_CALUDE_smallest_number_of_weights_l4069_406963


namespace NUMINAMATH_CALUDE_initial_balance_was_200_l4069_406914

/-- Represents the balance of Yasmin's bank account throughout the week --/
structure BankAccount where
  initial : ℝ
  monday : ℝ
  tuesday : ℝ
  wednesday : ℝ
  thursday : ℝ

/-- Calculates the final balance of Yasmin's account after all transactions --/
def finalBalance (account : BankAccount) : ℝ :=
  account.thursday

/-- Theorem stating that the initial balance was $200 --/
theorem initial_balance_was_200 (account : BankAccount) :
  account.initial = 200 ∧
  account.monday = account.initial / 2 ∧
  account.tuesday = account.monday + 30 ∧
  account.wednesday = 200 ∧
  account.thursday = account.wednesday - 20 ∧
  finalBalance account = 160 :=
by sorry

end NUMINAMATH_CALUDE_initial_balance_was_200_l4069_406914


namespace NUMINAMATH_CALUDE_range_of_negative_two_a_plus_three_l4069_406985

theorem range_of_negative_two_a_plus_three (a : ℝ) : 
  a < 1 → -2*a + 3 > 1 := by
sorry

end NUMINAMATH_CALUDE_range_of_negative_two_a_plus_three_l4069_406985


namespace NUMINAMATH_CALUDE_prob_at_most_one_value_l4069_406934

/-- The probability that A hits the target -/
def prob_A : ℝ := 0.6

/-- The probability that B hits the target -/
def prob_B : ℝ := 0.7

/-- The probability that at most one of A and B hits the target -/
def prob_at_most_one : ℝ := 1 - prob_A * prob_B

theorem prob_at_most_one_value : prob_at_most_one = 0.58 := by
  sorry

end NUMINAMATH_CALUDE_prob_at_most_one_value_l4069_406934


namespace NUMINAMATH_CALUDE_art_museum_survey_l4069_406935

theorem art_museum_survey (V : ℕ) (E U : ℕ) : 
  E = U →                                     -- Number who enjoyed equals number who understood
  (3 : ℚ) / 4 * V = E →                       -- 3/4 of visitors both enjoyed and understood
  V = 520 →                                   -- Total number of visitors
  V - E = 130                                 -- Number who didn't enjoy and didn't understand
  := by sorry

end NUMINAMATH_CALUDE_art_museum_survey_l4069_406935


namespace NUMINAMATH_CALUDE_valid_committee_count_l4069_406981

/-- Represents the number of male professors in each department -/
def male_profs : Fin 3 → Nat
  | 0 => 3  -- Physics
  | 1 => 2  -- Chemistry
  | 2 => 2  -- Biology

/-- Represents the number of female professors in each department -/
def female_profs : Fin 3 → Nat
  | 0 => 3  -- Physics
  | 1 => 2  -- Chemistry
  | 2 => 3  -- Biology

/-- The total number of departments -/
def num_departments : Nat := 3

/-- The required committee size -/
def committee_size : Nat := 6

/-- The required number of male professors in the committee -/
def required_males : Nat := 3

/-- The required number of female professors in the committee -/
def required_females : Nat := 3

/-- Calculates the number of valid committee formations -/
def count_valid_committees : Nat := sorry

theorem valid_committee_count :
  count_valid_committees = 864 := by sorry

end NUMINAMATH_CALUDE_valid_committee_count_l4069_406981


namespace NUMINAMATH_CALUDE_congruent_triangles_corresponding_angles_l4069_406918

-- Define a triangle
def Triangle := ℝ × ℝ × ℝ

-- Define congruence for triangles
def congruent (t1 t2 : Triangle) : Prop := sorry

-- Define the property of corresponding angles being congruent
def corresponding_angles_congruent (t1 t2 : Triangle) : Prop := sorry

-- Theorem statement
theorem congruent_triangles_corresponding_angles 
  (t1 t2 : Triangle) : congruent t1 t2 → corresponding_angles_congruent t1 t2 := by
  sorry

end NUMINAMATH_CALUDE_congruent_triangles_corresponding_angles_l4069_406918


namespace NUMINAMATH_CALUDE_race_permutations_eq_24_l4069_406925

/-- The number of different possible orders for a race with 4 distinct participants and no ties -/
def race_permutations : ℕ := 24

/-- The number of participants in the race -/
def num_participants : ℕ := 4

/-- Theorem: The number of different possible orders for a race with 4 distinct participants and no ties is 24 -/
theorem race_permutations_eq_24 : race_permutations = Nat.factorial num_participants := by
  sorry

end NUMINAMATH_CALUDE_race_permutations_eq_24_l4069_406925


namespace NUMINAMATH_CALUDE_fraction_multiplication_result_l4069_406975

theorem fraction_multiplication_result : (3 / 4 : ℚ) * (1 / 2 : ℚ) * (2 / 5 : ℚ) * 5000 = 750 := by
  sorry

end NUMINAMATH_CALUDE_fraction_multiplication_result_l4069_406975


namespace NUMINAMATH_CALUDE_triangle_perimeter_l4069_406937

/-- Given a triangle with inradius 2.5 cm and area 25 cm², prove its perimeter is 20 cm -/
theorem triangle_perimeter (r : ℝ) (A : ℝ) (p : ℝ) 
  (h1 : r = 2.5)
  (h2 : A = 25)
  (h3 : A = r * (p / 2)) :
  p = 20 := by
  sorry

end NUMINAMATH_CALUDE_triangle_perimeter_l4069_406937


namespace NUMINAMATH_CALUDE_dining_bill_share_l4069_406907

/-- Given a total bill, number of people, and tip percentage, calculates each person's share --/
def calculate_share (total_bill : ℚ) (num_people : ℕ) (tip_percentage : ℚ) : ℚ :=
  (total_bill * (1 + tip_percentage)) / num_people

/-- Proves that the calculated share for the given conditions is approximately $48.53 --/
theorem dining_bill_share :
  let total_bill : ℚ := 211
  let num_people : ℕ := 5
  let tip_percentage : ℚ := 15 / 100
  abs (calculate_share total_bill num_people tip_percentage - 48.53) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_dining_bill_share_l4069_406907


namespace NUMINAMATH_CALUDE_computation_proof_l4069_406993

theorem computation_proof : (143 + 29) * 2 + 25 + 13 = 382 := by
  sorry

end NUMINAMATH_CALUDE_computation_proof_l4069_406993


namespace NUMINAMATH_CALUDE_estimate_sqrt_expression_l4069_406961

theorem estimate_sqrt_expression :
  7 < Real.sqrt 36 * Real.sqrt (1/2) + Real.sqrt 8 ∧
  Real.sqrt 36 * Real.sqrt (1/2) + Real.sqrt 8 < 8 :=
by sorry

end NUMINAMATH_CALUDE_estimate_sqrt_expression_l4069_406961


namespace NUMINAMATH_CALUDE_remainder_of_x_50_divided_by_x2_minus_4x_plus_3_l4069_406986

theorem remainder_of_x_50_divided_by_x2_minus_4x_plus_3 :
  ∀ (x : ℝ), ∃ (Q : ℝ → ℝ) (R : ℝ → ℝ),
    x^50 = (x^2 - 4*x + 3) * Q x + R x ∧
    (∀ (y : ℝ), R y = (3^50 - 1)/2 * y + (3 - 3^50)/2) ∧
    (∀ (y : ℝ), ∃ (a b : ℝ), R y = a * y + b) :=
by sorry

end NUMINAMATH_CALUDE_remainder_of_x_50_divided_by_x2_minus_4x_plus_3_l4069_406986


namespace NUMINAMATH_CALUDE_train_length_problem_l4069_406931

/-- Given a platform length, time to pass, and train speed, calculates the length of the train -/
def train_length (platform_length time_to_pass train_speed : ℝ) : ℝ :=
  train_speed * time_to_pass - platform_length

/-- Theorem stating that under the given conditions, the train length is 50 meters -/
theorem train_length_problem :
  let platform_length : ℝ := 100
  let time_to_pass : ℝ := 10
  let train_speed : ℝ := 15
  train_length platform_length time_to_pass train_speed = 50 := by
sorry

#eval train_length 100 10 15

end NUMINAMATH_CALUDE_train_length_problem_l4069_406931


namespace NUMINAMATH_CALUDE_triangle_properties_l4069_406940

theorem triangle_properties (a b c : ℝ) (A B C : ℝ) :
  a > 0 ∧ b > 0 ∧ c > 0 →
  A > 0 ∧ B > 0 ∧ C > 0 →
  A + B + C = π →
  a * Real.sin (2 * B) = Real.sqrt 3 * b * Real.sin A →
  Real.cos A = 1 / 3 →
  B = π / 6 ∧ Real.sin C = (2 * Real.sqrt 6 + 1) / 6 := by
sorry

end NUMINAMATH_CALUDE_triangle_properties_l4069_406940


namespace NUMINAMATH_CALUDE_nancy_savings_l4069_406942

/-- The value of a quarter in dollars -/
def quarter_value : ℚ := 0.25

/-- The value of a dime in dollars -/
def dime_value : ℚ := 0.10

/-- The value of a nickel in dollars -/
def nickel_value : ℚ := 0.05

/-- The number of coins in a dozen -/
def dozen : ℕ := 12

/-- The number of quarters Nancy has -/
def nancy_quarters : ℕ := 3 * dozen

/-- The number of dimes Nancy has -/
def nancy_dimes : ℕ := 2 * dozen

/-- The number of nickels Nancy has -/
def nancy_nickels : ℕ := 5 * dozen

/-- The total monetary value of Nancy's coins -/
def nancy_total : ℚ := 
  (nancy_quarters : ℚ) * quarter_value + 
  (nancy_dimes : ℚ) * dime_value + 
  (nancy_nickels : ℚ) * nickel_value

theorem nancy_savings : nancy_total = 14.40 := by
  sorry

end NUMINAMATH_CALUDE_nancy_savings_l4069_406942


namespace NUMINAMATH_CALUDE_complement_of_P_in_U_l4069_406939

def U : Set Int := {-1, 0, 1, 2}

def P : Set Int := {x : Int | x^2 < 2}

theorem complement_of_P_in_U : {2} = U \ P := by sorry

end NUMINAMATH_CALUDE_complement_of_P_in_U_l4069_406939


namespace NUMINAMATH_CALUDE_positive_function_condition_l4069_406991

theorem positive_function_condition (a : ℝ) : 
  (∀ x : ℝ, x ∈ Set.Icc 0 1 → (2 - a^2) * x + a > 0) ↔ (0 < a ∧ a < 2) := by
  sorry

end NUMINAMATH_CALUDE_positive_function_condition_l4069_406991


namespace NUMINAMATH_CALUDE_cot_thirteen_pi_fourths_l4069_406976

theorem cot_thirteen_pi_fourths : Real.cos (13 * π / 4) / Real.sin (13 * π / 4) = -1 := by
  sorry

end NUMINAMATH_CALUDE_cot_thirteen_pi_fourths_l4069_406976


namespace NUMINAMATH_CALUDE_sum_of_square_roots_inequality_l4069_406913

theorem sum_of_square_roots_inequality (a b c d : ℝ) 
  (non_neg_a : 0 ≤ a) (non_neg_b : 0 ≤ b) (non_neg_c : 0 ≤ c) (non_neg_d : 0 ≤ d)
  (sum_eq_four : a + b + c + d = 4) : 
  Real.sqrt (a + b + c) + Real.sqrt (b + c + d) + Real.sqrt (c + d + a) + Real.sqrt (d + a + b) ≥ 6 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_square_roots_inequality_l4069_406913


namespace NUMINAMATH_CALUDE_arithmetic_progression_rth_term_l4069_406927

/-- The sum of the first n terms of an arithmetic progression -/
def S (n : ℕ) : ℕ := 5 * n + 4 * n^2

/-- The r-th term of the arithmetic progression -/
def a (r : ℕ) : ℕ := S r - S (r - 1)

theorem arithmetic_progression_rth_term (r : ℕ) (h : r > 0) : a r = 8 * r + 1 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_progression_rth_term_l4069_406927


namespace NUMINAMATH_CALUDE_dave_apps_problem_l4069_406945

theorem dave_apps_problem (initial_apps final_apps : ℕ) 
  (h1 : initial_apps = 15)
  (h2 : final_apps = 14)
  (h3 : ∃ (added deleted : ℕ), initial_apps + added - deleted = final_apps ∧ deleted = added + 1) :
  ∃ (added : ℕ), added = 0 ∧ initial_apps + added - (added + 1) = final_apps :=
by sorry

end NUMINAMATH_CALUDE_dave_apps_problem_l4069_406945


namespace NUMINAMATH_CALUDE_integer_solution_system_l4069_406903

theorem integer_solution_system (m n : ℤ) : 
  m * (m + n) = n * 12 ∧ n * (m + n) = m * 3 → m = 4 ∧ n = 2 := by
  sorry

end NUMINAMATH_CALUDE_integer_solution_system_l4069_406903


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l4069_406949

-- Define an arithmetic sequence
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

-- State the theorem
theorem arithmetic_sequence_problem (a : ℕ → ℝ) 
  (h1 : is_arithmetic_sequence a) 
  (h2 : a 4 + a 7 + a 10 = 30) : 
  a 3 - 2 * a 5 = -10 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l4069_406949


namespace NUMINAMATH_CALUDE_pepsi_amount_l4069_406930

/-- Represents the drink inventory and packing constraints -/
structure DrinkInventory where
  maaza : ℕ
  sprite : ℕ
  total_cans : ℕ
  pepsi : ℕ

/-- Calculates the greatest common divisor of two natural numbers -/
def gcd (a b : ℕ) : ℕ := sorry

/-- Theorem: Given the inventory and constraints, the amount of Pepsi is 144 liters -/
theorem pepsi_amount (inventory : DrinkInventory) 
  (h1 : inventory.maaza = 80)
  (h2 : inventory.sprite = 368)
  (h3 : inventory.total_cans = 37)
  (h4 : ∃ (can_size : ℕ), can_size > 0 ∧ 
        inventory.maaza % can_size = 0 ∧ 
        inventory.sprite % can_size = 0 ∧
        inventory.pepsi % can_size = 0 ∧
        inventory.total_cans = inventory.maaza / can_size + inventory.sprite / can_size + inventory.pepsi / can_size)
  : inventory.pepsi = 144 := by
  sorry

end NUMINAMATH_CALUDE_pepsi_amount_l4069_406930


namespace NUMINAMATH_CALUDE_first_term_of_geometric_series_first_term_of_geometric_series_l4069_406916

/-- Given an infinite geometric series with sum 18 and sum of squares 72, 
    prove that the first term of the series is 72/11 -/
theorem first_term_of_geometric_series 
  (a : ℝ) -- First term of the series
  (r : ℝ) -- Common ratio of the series
  (h1 : a / (1 - r) = 18) -- Sum of the series is 18
  (h2 : a^2 / (1 - r^2) = 72) -- Sum of squares is 72
  : a = 72 / 11 := by
sorry

/-- Alternative formulation using a function for the series -/
theorem first_term_of_geometric_series' 
  (S : ℕ → ℝ) -- Geometric series as a function
  (h1 : ∃ r : ℝ, ∀ n : ℕ, S (n + 1) = r * S n) -- S is a geometric series
  (h2 : ∑' n, S n = 18) -- Sum of the series is 18
  (h3 : ∑' n, (S n)^2 = 72) -- Sum of squares is 72
  : S 0 = 72 / 11 := by
sorry

end NUMINAMATH_CALUDE_first_term_of_geometric_series_first_term_of_geometric_series_l4069_406916


namespace NUMINAMATH_CALUDE_correct_systematic_sample_l4069_406948

/-- Represents a systematic sampling scheme. -/
structure SystematicSampling where
  totalItems : ℕ
  sampleSize : ℕ
  startingPoint : ℕ
  samplingInterval : ℕ

/-- Generates the sample numbers for a given systematic sampling scheme. -/
def generateSampleNumbers (s : SystematicSampling) : List ℕ :=
  List.range s.sampleSize |>.map (fun i => s.startingPoint + i * s.samplingInterval)

/-- Theorem: The correct sample numbers for systematic sampling of 5 items from 50 products
    are 9, 19, 29, 39, 49. -/
theorem correct_systematic_sample :
  let s : SystematicSampling := {
    totalItems := 50,
    sampleSize := 5,
    startingPoint := 9,
    samplingInterval := 10
  }
  generateSampleNumbers s = [9, 19, 29, 39, 49] := by
  sorry


end NUMINAMATH_CALUDE_correct_systematic_sample_l4069_406948


namespace NUMINAMATH_CALUDE_line_passes_through_point_with_slope_l4069_406962

/-- The slope of the line -/
def m : ℝ := 2

/-- The x-coordinate of the point P -/
def x₀ : ℝ := 3

/-- The y-coordinate of the point P -/
def y₀ : ℝ := 4

/-- The equation of the line passing through (x₀, y₀) with slope m -/
def line_equation (x y : ℝ) : Prop := 2 * x - y - 2 = 0

theorem line_passes_through_point_with_slope :
  line_equation x₀ y₀ ∧ 
  ∀ x y : ℝ, line_equation x y → (y - y₀) = m * (x - x₀) :=
sorry

end NUMINAMATH_CALUDE_line_passes_through_point_with_slope_l4069_406962


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l4069_406956

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- Theorem: In an arithmetic sequence {a_n}, if a_1 + 3a_8 + a_15 = 120, then a_8 = 24 -/
theorem arithmetic_sequence_problem (a : ℕ → ℝ) 
  (h_arith : arithmetic_sequence a) 
  (h_sum : a 1 + 3 * a 8 + a 15 = 120) : 
  a 8 = 24 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l4069_406956


namespace NUMINAMATH_CALUDE_dale_had_two_eggs_l4069_406923

/-- The cost of breakfast for Dale and Andrew -/
def breakfast_cost (dale_eggs : ℕ) : ℝ :=
  (2 * 1 + dale_eggs * 3) + (1 * 1 + 2 * 3)

/-- Theorem: Dale had 2 eggs -/
theorem dale_had_two_eggs : 
  ∃ (dale_eggs : ℕ), breakfast_cost dale_eggs = 15 ∧ dale_eggs = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_dale_had_two_eggs_l4069_406923


namespace NUMINAMATH_CALUDE_dog_food_cans_per_package_adam_dog_food_cans_l4069_406968

theorem dog_food_cans_per_package (cat_packages : Nat) (dog_packages : Nat) 
  (cat_cans_per_package : Nat) (extra_cat_cans : Nat) : Nat :=
  let total_cat_cans := cat_packages * cat_cans_per_package
  let dog_cans_per_package := (total_cat_cans - extra_cat_cans) / dog_packages
  dog_cans_per_package

/-- The number of cans in each package of dog food is 5. -/
theorem adam_dog_food_cans : dog_food_cans_per_package 9 7 10 55 = 5 := by
  sorry

end NUMINAMATH_CALUDE_dog_food_cans_per_package_adam_dog_food_cans_l4069_406968


namespace NUMINAMATH_CALUDE_blind_box_equations_l4069_406910

/-- Represents the blind box production scenario -/
structure BlindBoxProduction where
  total_fabric : ℝ
  fabric_for_a : ℝ
  fabric_for_b : ℝ

/-- Conditions for the blind box production -/
def valid_production (p : BlindBoxProduction) : Prop :=
  p.total_fabric = 135 ∧
  p.fabric_for_a + p.fabric_for_b = p.total_fabric ∧
  2 * p.fabric_for_a = 3 * p.fabric_for_b

/-- Theorem stating the correct system of equations for the blind box production -/
theorem blind_box_equations (p : BlindBoxProduction) :
  valid_production p →
  p.fabric_for_a + p.fabric_for_b = 135 ∧ 2 * p.fabric_for_a = 3 * p.fabric_for_b := by
  sorry

end NUMINAMATH_CALUDE_blind_box_equations_l4069_406910


namespace NUMINAMATH_CALUDE_fencing_cost_per_foot_l4069_406920

/-- The cost of fencing per foot -/
def cost_per_foot (side_length back_length total_cost : ℚ) : ℚ :=
  let total_length := 2 * side_length + back_length
  let neighbor_back_contribution := back_length / 2
  let neighbor_left_contribution := side_length / 3
  let cole_length := total_length - neighbor_back_contribution - neighbor_left_contribution
  total_cost / cole_length

/-- Theorem stating that the cost per foot of fencing is $3 -/
theorem fencing_cost_per_foot :
  cost_per_foot 9 18 72 = 3 :=
sorry

end NUMINAMATH_CALUDE_fencing_cost_per_foot_l4069_406920


namespace NUMINAMATH_CALUDE_square_difference_equals_1380_l4069_406922

theorem square_difference_equals_1380 : (23 + 15)^2 - (23 - 15)^2 = 1380 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_equals_1380_l4069_406922


namespace NUMINAMATH_CALUDE_equation_relationship_l4069_406943

/-- Represents a relationship between x and y --/
inductive Relationship
  | Direct
  | Inverse
  | Neither

/-- Determines the relationship between x and y in the equation 2x + 3y = 15 --/
def relationshipInEquation : Relationship := sorry

/-- Theorem stating that the relationship in the equation 2x + 3y = 15 is neither direct nor inverse proportionality --/
theorem equation_relationship :
  relationshipInEquation = Relationship.Neither := by sorry

end NUMINAMATH_CALUDE_equation_relationship_l4069_406943


namespace NUMINAMATH_CALUDE_workshop_production_theorem_l4069_406900

/-- Represents the factory workshop setup and production requirements -/
structure Workshop where
  total_workers : ℕ
  type_a_production : ℕ
  type_b_production : ℕ
  type_a_required : ℕ
  type_b_required : ℕ
  type_a_cost : ℕ
  type_b_cost : ℕ

/-- Calculates the number of workers assigned to type A parts -/
def workers_for_type_a (w : Workshop) : ℕ :=
  sorry

/-- Calculates the total processing cost for all workers in one day -/
def total_processing_cost (w : Workshop) : ℕ :=
  sorry

/-- The main theorem stating the correct number of workers for type A and total cost -/
theorem workshop_production_theorem (w : Workshop) 
  (h1 : w.total_workers = 50)
  (h2 : w.type_a_production = 30)
  (h3 : w.type_b_production = 20)
  (h4 : w.type_a_required = 7)
  (h5 : w.type_b_required = 2)
  (h6 : w.type_a_cost = 10)
  (h7 : w.type_b_cost = 12) :
  workers_for_type_a w = 35 ∧ total_processing_cost w = 14100 :=
by sorry

end NUMINAMATH_CALUDE_workshop_production_theorem_l4069_406900


namespace NUMINAMATH_CALUDE_joe_cars_l4069_406957

theorem joe_cars (initial_cars additional_cars : ℕ) :
  initial_cars = 50 → additional_cars = 12 → initial_cars + additional_cars = 62 := by
  sorry

end NUMINAMATH_CALUDE_joe_cars_l4069_406957


namespace NUMINAMATH_CALUDE_sqrt_product_equality_l4069_406989

theorem sqrt_product_equality (x : ℝ) (h1 : x > 0) 
  (h2 : Real.sqrt (12 * x) * Real.sqrt (20 * x) * Real.sqrt (5 * x) * Real.sqrt (30 * x) = 30) :
  x = 1 / Real.sqrt 20 := by
sorry

end NUMINAMATH_CALUDE_sqrt_product_equality_l4069_406989


namespace NUMINAMATH_CALUDE_cheryl_skittles_l4069_406952

/-- 
Given that Cheryl starts with a certain number of Skittles and receives additional Skittles,
this theorem proves the total number of Skittles Cheryl ends up with.
-/
theorem cheryl_skittles (initial : ℕ) (additional : ℕ) :
  initial = 8 → additional = 89 → initial + additional = 97 := by
  sorry

end NUMINAMATH_CALUDE_cheryl_skittles_l4069_406952


namespace NUMINAMATH_CALUDE_max_fraction_value_l4069_406928

theorem max_fraction_value (a b : ℝ) 
  (ha : 100 ≤ a ∧ a ≤ 500) (hb : 500 ≤ b ∧ b ≤ 1500) : 
  (b - 100) / (a + 50) ≤ 28/3 := by
  sorry

end NUMINAMATH_CALUDE_max_fraction_value_l4069_406928


namespace NUMINAMATH_CALUDE_diagonal_path_crosses_12_tiles_l4069_406946

/-- Represents a rectangular floor tiled with 1x2 foot tiles -/
structure TiledFloor where
  width : ℕ
  length : ℕ

/-- Calculates the number of tiles crossed by a diagonal path on a tiled floor -/
def tilesCrossed (floor : TiledFloor) : ℕ :=
  floor.width / Nat.gcd floor.width floor.length +
  floor.length / Nat.gcd floor.width floor.length - 1

/-- Theorem stating that a diagonal path on an 8x18 foot floor crosses 12 tiles -/
theorem diagonal_path_crosses_12_tiles :
  let floor : TiledFloor := { width := 8, length := 18 }
  tilesCrossed floor = 12 := by sorry

end NUMINAMATH_CALUDE_diagonal_path_crosses_12_tiles_l4069_406946


namespace NUMINAMATH_CALUDE_chess_team_girls_l4069_406919

theorem chess_team_girls (total : ℕ) (attended : ℕ) (boys : ℕ) (girls : ℕ) : 
  total = 26 →
  attended = 16 →
  boys + girls = total →
  boys + (girls / 2) = attended →
  girls = 20 := by
sorry

end NUMINAMATH_CALUDE_chess_team_girls_l4069_406919


namespace NUMINAMATH_CALUDE_system_solution_ratio_l4069_406984

theorem system_solution_ratio (x y c d : ℝ) 
  (eq1 : 8 * x - 6 * y = c)
  (eq2 : 9 * y - 12 * x = d)
  (x_nonzero : x ≠ 0)
  (y_nonzero : y ≠ 0)
  (d_nonzero : d ≠ 0) :
  c / d = -2 / 3 := by
sorry

end NUMINAMATH_CALUDE_system_solution_ratio_l4069_406984


namespace NUMINAMATH_CALUDE_count_four_digit_divisible_by_13_l4069_406959

theorem count_four_digit_divisible_by_13 : 
  (Finset.filter (fun n => n % 13 = 0) (Finset.range 9000)).card = 689 := by
  sorry

end NUMINAMATH_CALUDE_count_four_digit_divisible_by_13_l4069_406959


namespace NUMINAMATH_CALUDE_basketball_height_data_field_survey_l4069_406967

def HeightData := List Nat

def isFieldSurveyMethod (data : HeightData) : Prop :=
  data.all (λ h => h ≥ 150 ∧ h ≤ 200) ∧ 
  data.length > 0 ∧
  data.length ≤ 20

def basketballTeamHeights : HeightData :=
  [167, 168, 167, 164, 168, 168, 163, 168, 167, 160]

theorem basketball_height_data_field_survey :
  isFieldSurveyMethod basketballTeamHeights := by
  sorry

end NUMINAMATH_CALUDE_basketball_height_data_field_survey_l4069_406967


namespace NUMINAMATH_CALUDE_pigs_joined_l4069_406990

def initial_pigs : ℕ := 64
def final_pigs : ℕ := 86

theorem pigs_joined (initial : ℕ) (final : ℕ) (h1 : initial = initial_pigs) (h2 : final = final_pigs) :
  final - initial = 22 :=
by sorry

end NUMINAMATH_CALUDE_pigs_joined_l4069_406990


namespace NUMINAMATH_CALUDE_final_sum_theorem_l4069_406978

/-- The number of participants in the game --/
def participants : ℕ := 53

/-- The initial value of the first calculator --/
def calc1_initial : ℤ := 2

/-- The initial value of the second calculator --/
def calc2_initial : ℤ := -2

/-- The initial value of the third calculator --/
def calc3_initial : ℕ := 5

/-- The operation applied to the first calculator --/
def op1 (n : ℤ) : ℤ := n ^ 2

/-- The operation applied to the second calculator --/
def op2 (n : ℤ) : ℤ := n ^ 3

/-- The operation applied to the third calculator --/
def op3 (n : ℕ) : ℕ := n + 2

/-- The final value of the first calculator after all participants --/
def calc1_final : ℤ := calc1_initial ^ (2 ^ participants)

/-- The final value of the second calculator after all participants --/
def calc2_final : ℤ := calc2_initial ^ (3 ^ participants)

/-- The final value of the third calculator after all participants --/
def calc3_final : ℕ := calc3_initial + 2 * participants

/-- The theorem stating the final sum of all calculators --/
theorem final_sum_theorem : 
  calc1_final + calc2_final + calc3_final = 
  calc1_initial ^ (2 ^ participants) + calc2_initial ^ (3 ^ participants) + (calc3_initial + 2 * participants) := by
  sorry

end NUMINAMATH_CALUDE_final_sum_theorem_l4069_406978


namespace NUMINAMATH_CALUDE_luke_trivia_game_l4069_406983

/-- Given a trivia game where a player gains a constant number of points per round
    and achieves a total score, calculate the number of rounds played. -/
def rounds_played (points_per_round : ℕ) (total_points : ℕ) : ℕ :=
  total_points / points_per_round

/-- Luke's trivia game scenario -/
theorem luke_trivia_game : rounds_played 3 78 = 26 := by
  sorry

end NUMINAMATH_CALUDE_luke_trivia_game_l4069_406983


namespace NUMINAMATH_CALUDE_binomial_distribution_p_value_l4069_406912

/-- A binomial distribution with parameters n and p -/
structure BinomialDistribution where
  n : ℕ
  p : ℝ
  h1 : 0 ≤ p
  h2 : p ≤ 1

/-- The expected value of a binomial distribution -/
def expectedValue (X : BinomialDistribution) : ℝ := X.n * X.p

/-- The variance of a binomial distribution -/
def variance (X : BinomialDistribution) : ℝ := X.n * X.p * (1 - X.p)

theorem binomial_distribution_p_value 
  (X : BinomialDistribution) 
  (h_exp : expectedValue X = 300)
  (h_var : variance X = 200) :
  X.p = 1/3 := by
sorry

end NUMINAMATH_CALUDE_binomial_distribution_p_value_l4069_406912


namespace NUMINAMATH_CALUDE_rod_cutting_l4069_406909

theorem rod_cutting (total_length : Real) (num_pieces : Nat) :
  total_length = 42.5 → num_pieces = 50 →
  (total_length / num_pieces) * 100 = 85 := by
  sorry

end NUMINAMATH_CALUDE_rod_cutting_l4069_406909


namespace NUMINAMATH_CALUDE_positive_product_of_positive_factors_l4069_406921

theorem positive_product_of_positive_factors (a b : ℝ) : a > 0 → b > 0 → a * b > 0 := by
  sorry

end NUMINAMATH_CALUDE_positive_product_of_positive_factors_l4069_406921


namespace NUMINAMATH_CALUDE_quadratic_solution_sum_l4069_406965

theorem quadratic_solution_sum (x : ℝ) (m n p : ℕ) : 
  x * (4 * x - 9) = -4 ∧ 
  (∃ (r : ℝ), r * r = n ∧ 
    (x = (m + r) / p ∨ x = (m - r) / p)) ∧
  Nat.gcd m (Nat.gcd n p) = 1 →
  m + n + p = 34 := by
sorry

end NUMINAMATH_CALUDE_quadratic_solution_sum_l4069_406965


namespace NUMINAMATH_CALUDE_rectangular_prism_sum_l4069_406904

/-- A rectangular prism is a three-dimensional shape with 6 faces, 12 edges, and 8 vertices. -/
structure RectangularPrism where
  faces : Nat
  edges : Nat
  vertices : Nat
  faces_eq : faces = 6
  edges_eq : edges = 12
  vertices_eq : vertices = 8

/-- The sum of faces, edges, and vertices of a rectangular prism is 26. -/
theorem rectangular_prism_sum (rp : RectangularPrism) : 
  rp.faces + rp.edges + rp.vertices = 26 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_prism_sum_l4069_406904


namespace NUMINAMATH_CALUDE_garden_occupation_fraction_l4069_406970

theorem garden_occupation_fraction :
  ∀ (garden_length garden_width : ℝ)
    (trapezoid_short_side trapezoid_long_side : ℝ)
    (sandbox_side : ℝ),
  garden_length = 40 →
  garden_width = 8 →
  trapezoid_long_side - trapezoid_short_side = 10 →
  trapezoid_short_side + trapezoid_long_side = garden_length →
  sandbox_side = 5 →
  let triangle_leg := (trapezoid_long_side - trapezoid_short_side) / 2
  let triangle_area := triangle_leg ^ 2 / 2
  let total_triangles_area := 2 * triangle_area
  let sandbox_area := sandbox_side ^ 2
  let occupied_area := total_triangles_area + sandbox_area
  let garden_area := garden_length * garden_width
  occupied_area / garden_area = 5 / 32 :=
by sorry

end NUMINAMATH_CALUDE_garden_occupation_fraction_l4069_406970


namespace NUMINAMATH_CALUDE_equation_equals_24_l4069_406982

theorem equation_equals_24 : (2 + 2 / 11) * 11 = 24 := by
  sorry

#check equation_equals_24

end NUMINAMATH_CALUDE_equation_equals_24_l4069_406982


namespace NUMINAMATH_CALUDE_ellipse_equation_l4069_406998

/-- An ellipse with a line passing through its vertex and focus -/
structure EllipseWithLine where
  /-- The semi-major axis length of the ellipse -/
  a : ℝ
  /-- The semi-minor axis length of the ellipse -/
  b : ℝ
  /-- Condition that a > b > 0 -/
  h1 : a > b ∧ b > 0
  /-- The line equation x - 2y + 4 = 0 -/
  line_eq : ℝ → ℝ → Prop := fun x y => x - 2*y + 4 = 0
  /-- The line passes through a vertex and focus of the ellipse -/
  line_through_vertex_focus : ∃ (x₁ y₁ x₂ y₂ : ℝ),
    line_eq x₁ y₁ ∧ line_eq x₂ y₂ ∧
    ((x₁^2 / a^2 + y₁^2 / b^2 = 1 ∧ (x₁ = a ∨ x₁ = -a ∨ y₁ = b ∨ y₁ = -b)) ∨
     (x₂^2 / a^2 + y₂^2 / b^2 > 1 ∧ x₂^2 - y₂^2 = a^2 - b^2))

/-- The theorem stating the standard equation of the ellipse -/
theorem ellipse_equation (e : EllipseWithLine) :
  ∀ (x y : ℝ), x^2/20 + y^2/4 = 1 ↔ x^2/e.a^2 + y^2/e.b^2 = 1 :=
sorry

end NUMINAMATH_CALUDE_ellipse_equation_l4069_406998


namespace NUMINAMATH_CALUDE_total_tickets_is_150_l4069_406996

/-- The number of tickets Alan handed out -/
def alan_tickets : ℕ := 26

/-- The number of tickets Marcy handed out -/
def marcy_tickets : ℕ := 5 * alan_tickets - 6

/-- The total number of tickets handed out by Alan and Marcy -/
def total_tickets : ℕ := alan_tickets + marcy_tickets

/-- Theorem stating that the total number of tickets handed out is 150 -/
theorem total_tickets_is_150 : total_tickets = 150 := by
  sorry

end NUMINAMATH_CALUDE_total_tickets_is_150_l4069_406996


namespace NUMINAMATH_CALUDE_hyperbola_dimensions_l4069_406929

/-- Given a hyperbola with equation x²/a² - y²/b² = 1 where a > 0 and b > 0,
    if the length of the real axis is 2 units greater than the length of the imaginary axis
    and the focal length is 10, then a = 4 and b = 3. -/
theorem hyperbola_dimensions (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  2*a - 2*b = 2 → a^2 + b^2 = 25 → a = 4 ∧ b = 3 := by sorry

end NUMINAMATH_CALUDE_hyperbola_dimensions_l4069_406929


namespace NUMINAMATH_CALUDE_max_dogs_and_fish_l4069_406955

/-- Represents the count of each animal type in the pet shop -/
structure PetShop where
  dogs : ℕ
  cats : ℕ
  bunnies : ℕ
  parrots : ℕ
  fish : ℕ

/-- Checks if the given pet shop counts satisfy the ratio constraint -/
def satisfiesRatio (shop : PetShop) : Prop :=
  7 * shop.cats = 7 * shop.dogs ∧
  8 * shop.cats = 7 * shop.bunnies ∧
  3 * shop.cats = 7 * shop.parrots ∧
  5 * shop.cats = 7 * shop.fish

/-- Checks if the total number of dogs and bunnies is 330 -/
def totalDogsAndBunnies330 (shop : PetShop) : Prop :=
  shop.dogs + shop.bunnies = 330

/-- Checks if there are at least twice as many fish as cats -/
def twiceAsManyFishAsCats (shop : PetShop) : Prop :=
  shop.fish ≥ 2 * shop.cats

/-- Theorem stating the maximum number of dogs and corresponding number of fish -/
theorem max_dogs_and_fish (shop : PetShop) 
  (h1 : satisfiesRatio shop) 
  (h2 : totalDogsAndBunnies330 shop) 
  (h3 : twiceAsManyFishAsCats shop) :
  shop.dogs ≤ 154 ∧ (shop.dogs = 154 → shop.fish = 308) :=
sorry

end NUMINAMATH_CALUDE_max_dogs_and_fish_l4069_406955


namespace NUMINAMATH_CALUDE_intersection_of_lines_l4069_406987

/-- Parametric equation of a line in 2D space -/
structure ParametricLine2D where
  point : ℝ × ℝ
  direction : ℝ × ℝ

/-- Check if a point lies on a parametric line -/
def pointOnLine (p : ℝ × ℝ) (l : ParametricLine2D) : Prop :=
  ∃ t : ℝ, p = (l.point.1 + t * l.direction.1, l.point.2 + t * l.direction.2)

theorem intersection_of_lines (line1 line2 : ParametricLine2D)
    (h1 : line1 = ParametricLine2D.mk (5, 1) (3, -2))
    (h2 : line2 = ParametricLine2D.mk (2, 8) (5, -3)) :
    ∃! p : ℝ × ℝ, pointOnLine p line1 ∧ pointOnLine p line2 ∧ p = (-73, 53) := by
  sorry

#check intersection_of_lines

end NUMINAMATH_CALUDE_intersection_of_lines_l4069_406987


namespace NUMINAMATH_CALUDE_anns_age_l4069_406988

theorem anns_age (A B t T : ℕ) : 
  A + B = 44 →
  B = A - t →
  B - t = A - T →
  B - T = A / 2 →
  A = 24 :=
by sorry

end NUMINAMATH_CALUDE_anns_age_l4069_406988


namespace NUMINAMATH_CALUDE_exists_diagonal_le_two_l4069_406997

-- Define a convex hexagon
structure ConvexHexagon where
  vertices : Fin 6 → ℝ × ℝ
  is_convex : sorry -- Add convexity condition

-- Define the property that all sides have length ≤ 1
def all_sides_le_one (h : ConvexHexagon) : Prop :=
  ∀ i : Fin 6, dist (h.vertices i) (h.vertices ((i + 1) % 6)) ≤ 1

-- Define a diagonal of the hexagon
def diagonal (h : ConvexHexagon) (i j : Fin 6) : ℝ :=
  dist (h.vertices i) (h.vertices j)

-- Theorem statement
theorem exists_diagonal_le_two (h : ConvexHexagon) (h_sides : all_sides_le_one h) :
  ∃ (i j : Fin 6), i ≠ j ∧ diagonal h i j ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_exists_diagonal_le_two_l4069_406997


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_l4069_406974

theorem simplify_and_evaluate (m n : ℝ) :
  (m + n)^2 - 2*m*(m + n) = n^2 - m^2 ∧
  (let m := 2; let n := -3; (m + n)^2 - 2*m*(m + n) = 5) :=
by sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_l4069_406974
