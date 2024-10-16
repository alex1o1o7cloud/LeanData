import Mathlib

namespace NUMINAMATH_CALUDE_unique_root_condition_l2329_232907

/-- The system of equations has only one root if and only if m is 0 or 2 -/
theorem unique_root_condition (m : ℝ) : 
  (∃! p : ℝ × ℝ, p.1^2 = 2*|p.1| ∧ |p.1| - p.2 - m = 1 - p.2^2) ↔ (m = 0 ∨ m = 2) :=
sorry

end NUMINAMATH_CALUDE_unique_root_condition_l2329_232907


namespace NUMINAMATH_CALUDE_sqrt_sum_eq_sum_iff_two_zero_l2329_232937

theorem sqrt_sum_eq_sum_iff_two_zero (a b c : ℝ) (ha : a ≥ 0) (hb : b ≥ 0) (hc : c ≥ 0) :
  Real.sqrt (a^2 + b^2 + c^2) = a + b + c ↔ (a = 0 ∧ b = 0) ∨ (a = 0 ∧ c = 0) ∨ (b = 0 ∧ c = 0) :=
by sorry

end NUMINAMATH_CALUDE_sqrt_sum_eq_sum_iff_two_zero_l2329_232937


namespace NUMINAMATH_CALUDE_range_of_m_l2329_232905

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | -3 ≤ x ∧ x ≤ 4}
def B (m : ℝ) : Set ℝ := {x : ℝ | 2*m - 1 < x ∧ x < m + 1}

-- State the theorem
theorem range_of_m (m : ℝ) : (B m ⊆ A) → m ≥ -1 := by
  sorry

end NUMINAMATH_CALUDE_range_of_m_l2329_232905


namespace NUMINAMATH_CALUDE_matinee_customers_count_l2329_232948

/-- Represents the revenue calculation for a movie theater. -/
def theater_revenue (matinee_customers : ℕ) : ℕ :=
  let matinee_price : ℕ := 5
  let evening_price : ℕ := 7
  let opening_night_price : ℕ := 10
  let popcorn_price : ℕ := 10
  let evening_customers : ℕ := 40
  let opening_night_customers : ℕ := 58
  let total_customers : ℕ := matinee_customers + evening_customers + opening_night_customers
  let popcorn_customers : ℕ := total_customers / 2

  matinee_price * matinee_customers +
  evening_price * evening_customers +
  opening_night_price * opening_night_customers +
  popcorn_price * popcorn_customers

/-- Theorem stating that the number of matinee customers is 32. -/
theorem matinee_customers_count : ∃ (n : ℕ), theater_revenue n = 1670 ∧ n = 32 :=
sorry

end NUMINAMATH_CALUDE_matinee_customers_count_l2329_232948


namespace NUMINAMATH_CALUDE_tomato_price_problem_l2329_232909

/-- Represents the problem of determining the price per kilogram of tomatoes --/
theorem tomato_price_problem (crate_capacity : ℕ) (num_crates : ℕ) (purchase_cost : ℕ) 
  (rotten_weight : ℕ) (profit : ℕ) : 
  crate_capacity = 20 →
  num_crates = 3 →
  purchase_cost = 330 →
  rotten_weight = 3 →
  profit = 12 →
  (purchase_cost + profit) / (crate_capacity * num_crates - rotten_weight) = 6 := by
  sorry

#check tomato_price_problem

end NUMINAMATH_CALUDE_tomato_price_problem_l2329_232909


namespace NUMINAMATH_CALUDE_ten_times_a_l2329_232939

theorem ten_times_a (a : ℝ) (h : a = 6) : 10 * a = 60 := by
  sorry

end NUMINAMATH_CALUDE_ten_times_a_l2329_232939


namespace NUMINAMATH_CALUDE_final_sum_is_212_l2329_232900

/-- Represents a person in the debt settlement problem -/
inductive Person
| Earl
| Fred
| Greg
| Hannah

/-- Represents the initial amount of money each person has -/
def initial_amount (p : Person) : Int :=
  match p with
  | Person.Earl => 90
  | Person.Fred => 48
  | Person.Greg => 36
  | Person.Hannah => 72

/-- Represents the amount one person owes to another -/
def debt (debtor receiver : Person) : Int :=
  match debtor, receiver with
  | Person.Earl, Person.Fred => 28
  | Person.Earl, Person.Hannah => 30
  | Person.Fred, Person.Greg => 32
  | Person.Fred, Person.Hannah => 10
  | Person.Greg, Person.Earl => 40
  | Person.Greg, Person.Hannah => 20
  | Person.Hannah, Person.Greg => 15
  | Person.Hannah, Person.Earl => 25
  | _, _ => 0

/-- Calculates the final amount a person has after settling all debts -/
def final_amount (p : Person) : Int :=
  initial_amount p
  + (debt Person.Earl p + debt Person.Fred p + debt Person.Greg p + debt Person.Hannah p)
  - (debt p Person.Earl + debt p Person.Fred + debt p Person.Greg + debt p Person.Hannah)

/-- Theorem stating that the sum of Greg's, Earl's, and Hannah's money after settling debts is $212 -/
theorem final_sum_is_212 :
  final_amount Person.Greg + final_amount Person.Earl + final_amount Person.Hannah = 212 :=
by sorry

end NUMINAMATH_CALUDE_final_sum_is_212_l2329_232900


namespace NUMINAMATH_CALUDE_counterexample_exists_l2329_232918

-- Define the types for points, lines, and planes
variable (Point Line Plane : Type)

-- Define the relation for a point being on a line or in a plane
variable (on_line : Point → Line → Prop)
variable (in_plane : Point → Plane → Prop)

-- Define the relation for a line being a subset of a plane
variable (line_subset_plane : Line → Plane → Prop)

-- Theorem statement
theorem counterexample_exists (l : Line) (α : Plane) (A : Point) 
  (h1 : ¬ line_subset_plane l α) 
  (h2 : on_line A l) :
  ¬ (∀ A, on_line A l → ¬ in_plane A α) :=
by sorry

end NUMINAMATH_CALUDE_counterexample_exists_l2329_232918


namespace NUMINAMATH_CALUDE_xyz_range_l2329_232997

theorem xyz_range (x y z : ℝ) 
  (sum_condition : x + y + z = 1) 
  (square_sum_condition : x^2 + y^2 + z^2 = 3) : 
  -1 ≤ x * y * z ∧ x * y * z ≤ 5/27 := by
  sorry

end NUMINAMATH_CALUDE_xyz_range_l2329_232997


namespace NUMINAMATH_CALUDE_double_acute_angle_less_than_180_degrees_l2329_232904

theorem double_acute_angle_less_than_180_degrees (α : Real) :
  (0 < α ∧ α < Real.pi / 2) → 2 * α < Real.pi := by
  sorry

end NUMINAMATH_CALUDE_double_acute_angle_less_than_180_degrees_l2329_232904


namespace NUMINAMATH_CALUDE_stratified_sampling_theorem_l2329_232979

/-- Represents the sample size for each category of students -/
structure SampleSizes where
  junior : ℕ
  undergraduate : ℕ
  graduate : ℕ

/-- Calculates the stratified sample sizes given the total population, category populations, and total sample size -/
def calculateSampleSizes (totalPopulation : ℕ) (juniorPopulation : ℕ) (undergradPopulation : ℕ) (sampleSize : ℕ) : SampleSizes :=
  let juniorSample := (juniorPopulation * sampleSize) / totalPopulation
  let undergradSample := (undergradPopulation * sampleSize) / totalPopulation
  let gradSample := sampleSize - juniorSample - undergradSample
  { junior := juniorSample,
    undergraduate := undergradSample,
    graduate := gradSample }

theorem stratified_sampling_theorem (totalPopulation : ℕ) (juniorPopulation : ℕ) (undergradPopulation : ℕ) (sampleSize : ℕ)
    (h1 : totalPopulation = 5600)
    (h2 : juniorPopulation = 1300)
    (h3 : undergradPopulation = 3000)
    (h4 : sampleSize = 280) :
    calculateSampleSizes totalPopulation juniorPopulation undergradPopulation sampleSize =
    { junior := 65, undergraduate := 150, graduate := 65 } := by
  sorry

#check stratified_sampling_theorem

end NUMINAMATH_CALUDE_stratified_sampling_theorem_l2329_232979


namespace NUMINAMATH_CALUDE_unique_solution_sum_in_base7_l2329_232952

/-- Represents a digit in base 7 --/
def Digit7 := Fin 7

/-- Addition in base 7 --/
def add7 (a b : Digit7) : Digit7 × Bool :=
  let sum := a.val + b.val
  (⟨sum % 7, by sorry⟩, sum ≥ 7)

/-- Represents the equation in base 7 --/
def equation (A B C : Digit7) : Prop :=
  ∃ (carry1 carry2 : Bool),
    let (units, carry1) := add7 B C
    let (tens, carry2) := add7 A B
    units = A ∧
    (if carry1 then add7 (⟨1, by sorry⟩) tens else (tens, false)).1 = C ∧
    (if carry2 then add7 (⟨1, by sorry⟩) A else (A, false)).1 = A

theorem unique_solution :
  ∃! (A B C : Digit7),
    A.val ≠ 0 ∧ B.val ≠ 0 ∧ C.val ≠ 0 ∧
    A.val ≠ B.val ∧ A.val ≠ C.val ∧ B.val ≠ C.val ∧
    equation A B C ∧
    A.val = 6 ∧ B.val = 3 ∧ C.val = 5 :=
  sorry

theorem sum_in_base7 (A B C : Digit7) 
  (h : A.val = 6 ∧ B.val = 3 ∧ C.val = 5) :
  (A.val + B.val + C.val : ℕ) % 49 = 20 :=
  sorry

end NUMINAMATH_CALUDE_unique_solution_sum_in_base7_l2329_232952


namespace NUMINAMATH_CALUDE_b_current_age_l2329_232969

/-- Given two people A and B, prove B's current age is 38 years old. -/
theorem b_current_age (a b : ℕ) : 
  (a + 10 = 2 * (b - 10)) → -- A's age in 10 years = 2 * (B's age 10 years ago)
  (a = b + 8) →             -- A is currently 8 years older than B
  b = 38 :=                 -- B's current age is 38
by sorry

end NUMINAMATH_CALUDE_b_current_age_l2329_232969


namespace NUMINAMATH_CALUDE_total_cost_calculation_l2329_232944

def muffin_cost : ℚ := 0.75
def juice_cost : ℚ := 1.45
def muffin_count : ℕ := 3

theorem total_cost_calculation : 
  (muffin_count : ℚ) * muffin_cost + juice_cost = 3.70 := by
  sorry

end NUMINAMATH_CALUDE_total_cost_calculation_l2329_232944


namespace NUMINAMATH_CALUDE_complex_multiplication_subtraction_l2329_232999

theorem complex_multiplication_subtraction : ∃ (i : ℂ), i^2 = -1 ∧ (4 - 3*i) * (2 + 5*i) - (6 - 2*i) = 17 + 16*i := by
  sorry

end NUMINAMATH_CALUDE_complex_multiplication_subtraction_l2329_232999


namespace NUMINAMATH_CALUDE_investment_result_l2329_232912

/-- Calculates the final amount after compound interest --/
def compound_interest (principal : ℝ) (rate : ℝ) (time : ℕ) : ℝ :=
  principal * (1 + rate) ^ time

/-- Proves that the given investment scenario results in approximately $3045.28 --/
theorem investment_result :
  let principal : ℝ := 1500
  let rate : ℝ := 0.04
  let time : ℕ := 21
  let result := compound_interest principal rate time
  ∃ ε > 0, |result - 3045.28| < ε :=
sorry

end NUMINAMATH_CALUDE_investment_result_l2329_232912


namespace NUMINAMATH_CALUDE_brenda_spay_cats_l2329_232931

/-- Represents the number of cats Brenda needs to spay -/
def num_cats : ℕ := sorry

/-- Represents the number of dogs Brenda needs to spay -/
def num_dogs : ℕ := sorry

/-- The total number of animals Brenda needs to spay -/
def total_animals : ℕ := 21

theorem brenda_spay_cats :
  (num_cats + num_dogs = total_animals) →
  (num_dogs = 2 * num_cats) →
  num_cats = 7 := by
  sorry

end NUMINAMATH_CALUDE_brenda_spay_cats_l2329_232931


namespace NUMINAMATH_CALUDE_tricia_age_l2329_232924

theorem tricia_age (tricia amilia yorick eugene khloe rupert vincent : ℕ) : 
  tricia = amilia / 3 →
  amilia = yorick / 4 →
  ∃ k : ℕ, yorick = k * eugene →
  khloe = eugene / 3 →
  rupert = khloe + 10 →
  rupert = vincent - 2 →
  vincent = 22 →
  tricia = 5 →
  tricia = 5 := by sorry

end NUMINAMATH_CALUDE_tricia_age_l2329_232924


namespace NUMINAMATH_CALUDE_farmer_max_animals_l2329_232901

/-- Represents the farmer's animal purchasing problem --/
def FarmerProblem (budget goatCost sheepCost : ℕ) : Prop :=
  ∃ (goats sheep : ℕ),
    goats > 0 ∧
    sheep > 0 ∧
    goats = 2 * sheep ∧
    goatCost * goats + sheepCost * sheep ≤ budget ∧
    ∀ (g s : ℕ),
      g > 0 →
      s > 0 →
      g = 2 * s →
      goatCost * g + sheepCost * s ≤ budget →
      g + s ≤ goats + sheep

theorem farmer_max_animals :
  FarmerProblem 2000 35 40 →
  ∃ (goats sheep : ℕ),
    goats = 36 ∧
    sheep = 18 ∧
    goats + sheep = 54 ∧
    FarmerProblem 2000 35 40 :=
by sorry

end NUMINAMATH_CALUDE_farmer_max_animals_l2329_232901


namespace NUMINAMATH_CALUDE_cherry_pies_count_l2329_232954

theorem cherry_pies_count (total_pies : ℕ) (apple_ratio blueberry_ratio cherry_ratio : ℕ) 
  (h_total : total_pies = 36)
  (h_ratio : apple_ratio = 2 ∧ blueberry_ratio = 5 ∧ cherry_ratio = 4) :
  (cherry_ratio * total_pies) / (apple_ratio + blueberry_ratio + cherry_ratio) = 13 := by
  sorry

end NUMINAMATH_CALUDE_cherry_pies_count_l2329_232954


namespace NUMINAMATH_CALUDE_blue_face_prob_half_l2329_232970

/-- A rectangular prism with colored faces -/
structure ColoredPrism where
  green_faces : ℕ
  yellow_faces : ℕ
  blue_faces : ℕ

/-- The probability of rolling a blue face on a colored prism -/
def blue_face_probability (prism : ColoredPrism) : ℚ :=
  prism.blue_faces / (prism.green_faces + prism.yellow_faces + prism.blue_faces)

/-- Theorem: The probability of rolling a blue face on the given prism is 1/2 -/
theorem blue_face_prob_half :
  let prism : ColoredPrism := ⟨4, 2, 6⟩
  blue_face_probability prism = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_blue_face_prob_half_l2329_232970


namespace NUMINAMATH_CALUDE_difference_ones_zeros_157_l2329_232903

def binary_representation (n : ℕ) : List ℕ :=
  sorry

theorem difference_ones_zeros_157 :
  let binary := binary_representation 157
  let x := (binary.filter (· = 0)).length
  let y := (binary.filter (· = 1)).length
  y - x = 2 := by sorry

end NUMINAMATH_CALUDE_difference_ones_zeros_157_l2329_232903


namespace NUMINAMATH_CALUDE_nina_widget_purchase_l2329_232986

/-- The number of widgets Nina can purchase at the original price -/
def widgets_purchased (total_money : ℕ) (original_price : ℕ) : ℕ :=
  total_money / original_price

/-- The condition that if the price is reduced by 1, Nina can buy exactly 8 widgets -/
def price_reduction_condition (original_price : ℕ) (total_money : ℕ) : Prop :=
  8 * (original_price - 1) = total_money

theorem nina_widget_purchase :
  ∀ (original_price : ℕ),
    original_price > 0 →
    price_reduction_condition original_price 24 →
    widgets_purchased 24 original_price = 6 := by
  sorry

end NUMINAMATH_CALUDE_nina_widget_purchase_l2329_232986


namespace NUMINAMATH_CALUDE_cake_is_circle_with_radius_three_l2329_232996

/-- The equation of the cake's boundary -/
def cake_boundary (x y : ℝ) : Prop := x^2 + y^2 + 1 = 2*x + 6*y

/-- The cake is circular with radius 3 -/
theorem cake_is_circle_with_radius_three :
  ∃ (h k : ℝ), ∀ (x y : ℝ), cake_boundary x y ↔ (x - h)^2 + (y - k)^2 = 3^2 :=
sorry

end NUMINAMATH_CALUDE_cake_is_circle_with_radius_three_l2329_232996


namespace NUMINAMATH_CALUDE_time_after_duration_l2329_232955

/-- Represents time in a 12-hour format -/
structure Time12 where
  hour : Nat
  minute : Nat
  second : Nat
  isPM : Bool

/-- Adds a duration to a given time -/
def addDuration (t : Time12) (hours minutes seconds : Nat) : Time12 :=
  sorry

/-- Converts the hour component to 12-hour format -/
def to12Hour (h : Nat) : Nat :=
  sorry

theorem time_after_duration (initial : Time12) (final : Time12) :
  initial = Time12.mk 3 15 15 true →
  final = addDuration initial 196 58 16 →
  final.hour = 8 ∧ 
  final.minute = 13 ∧ 
  final.second = 31 ∧ 
  final.isPM = true ∧
  final.hour + final.minute + final.second = 52 :=
sorry

end NUMINAMATH_CALUDE_time_after_duration_l2329_232955


namespace NUMINAMATH_CALUDE_range_of_f_l2329_232968

def f (x : ℤ) : ℤ := x^2 - 1

def domain : Set ℤ := {-1, 0, 1}

theorem range_of_f : 
  {y | ∃ x ∈ domain, f x = y} = {-1, 0} := by sorry

end NUMINAMATH_CALUDE_range_of_f_l2329_232968


namespace NUMINAMATH_CALUDE_complex_equation_solution_l2329_232994

theorem complex_equation_solution (z : ℂ) : (Complex.I * z = 4 + 3 * Complex.I) → z = 3 - 4 * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l2329_232994


namespace NUMINAMATH_CALUDE_even_multiples_of_45_l2329_232978

theorem even_multiples_of_45 :
  let lower_bound := 449
  let upper_bound := 990
  let count_even_multiples := (upper_bound - lower_bound) / (45 * 2)
  count_even_multiples = 6.022222222222222 := by
  sorry

end NUMINAMATH_CALUDE_even_multiples_of_45_l2329_232978


namespace NUMINAMATH_CALUDE_expression_simplification_l2329_232919

theorem expression_simplification (x : ℝ) : 
  3 * x - 7 * x^2 + 5 - (6 - 5 * x + 7 * x^2) = -14 * x^2 + 8 * x - 1 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l2329_232919


namespace NUMINAMATH_CALUDE_tim_younger_than_jenny_l2329_232988

-- Define the ages
def tim_age : ℕ := 5
def rommel_age : ℕ := 3 * tim_age
def jenny_age : ℕ := rommel_age + 2

-- Theorem statement
theorem tim_younger_than_jenny : jenny_age - tim_age = 12 := by
  sorry

end NUMINAMATH_CALUDE_tim_younger_than_jenny_l2329_232988


namespace NUMINAMATH_CALUDE_xiaodong_election_l2329_232963

theorem xiaodong_election (V : ℝ) (h : V > 0) : 
  let votes_needed := (3/4 : ℝ) * V
  let votes_calculated := (2/3 : ℝ) * V
  let votes_obtained := (5/6 : ℝ) * votes_calculated
  let votes_remaining := V - votes_calculated
  let additional_votes_needed := votes_needed - votes_obtained
  (additional_votes_needed / votes_remaining) = (7/12 : ℝ) := by
sorry

end NUMINAMATH_CALUDE_xiaodong_election_l2329_232963


namespace NUMINAMATH_CALUDE_chocolate_ratio_l2329_232941

/-- Proves that the ratio of chocolates with nuts to chocolates without nuts is 1:1 given the problem conditions. -/
theorem chocolate_ratio (total : ℕ) (eaten_with_nuts : ℚ) (eaten_without_nuts : ℚ) (left : ℕ)
  (h_total : total = 80)
  (h_eaten_with_nuts : eaten_with_nuts = 4/5)
  (h_eaten_without_nuts : eaten_without_nuts = 1/2)
  (h_left : left = 28) :
  ∃ (with_nuts without_nuts : ℕ),
    with_nuts + without_nuts = total ∧
    (1 - eaten_with_nuts) * with_nuts + (1 - eaten_without_nuts) * without_nuts = left ∧
    with_nuts = without_nuts := by
  sorry

#check chocolate_ratio

end NUMINAMATH_CALUDE_chocolate_ratio_l2329_232941


namespace NUMINAMATH_CALUDE_parabola_line_intersection_midpoint_line_equation_min_distance_product_parabola_line_intersection_properties_l2329_232992

-- Define the parabola
def parabola (x y : ℝ) : Prop := x^2 = 4*y

-- Define the line passing through P(-2, 2)
def line (k : ℝ) (x y : ℝ) : Prop := y = k*(x + 2) + 2

-- Define the focus of the parabola
def focus : ℝ × ℝ := (0, 1)

-- Define the distance from a point to the focus
def distToFocus (x y : ℝ) : ℝ := y + 1

theorem parabola_line_intersection (k : ℝ) :
  ∃ (x₁ y₁ x₂ y₂ : ℝ),
    parabola x₁ y₁ ∧ parabola x₂ y₂ ∧
    line k x₁ y₁ ∧ line k x₂ y₂ ∧
    x₁ ≠ x₂ := by sorry

theorem midpoint_line_equation :
  ∃ (x₁ y₁ x₂ y₂ : ℝ),
    parabola x₁ y₁ ∧ parabola x₂ y₂ ∧
    line (-1) x₁ y₁ ∧ line (-1) x₂ y₂ ∧
    x₁ + x₂ = -4 ∧ y₁ + y₂ = 4 := by sorry

theorem min_distance_product :
  ∃ (k : ℝ),
    ∀ (x₁ y₁ x₂ y₂ : ℝ),
      parabola x₁ y₁ → parabola x₂ y₂ →
      line k x₁ y₁ → line k x₂ y₂ →
      distToFocus x₁ y₁ * distToFocus x₂ y₂ ≥ 9/2 := by sorry

-- Main theorems to prove
theorem parabola_line_intersection_properties :
  -- 1) When P(-2, 2) is the midpoint of AB, the equation of line AB is x + y = 0
  (∀ (x y : ℝ), line (-1) x y ↔ x + y = 0) ∧
  -- 2) The minimum value of |AF|•|BF| is 9/2
  (∃ (k : ℝ),
    ∀ (x₁ y₁ x₂ y₂ : ℝ),
      parabola x₁ y₁ → parabola x₂ y₂ →
      line k x₁ y₁ → line k x₂ y₂ →
      distToFocus x₁ y₁ * distToFocus x₂ y₂ = 9/2) := by sorry

end NUMINAMATH_CALUDE_parabola_line_intersection_midpoint_line_equation_min_distance_product_parabola_line_intersection_properties_l2329_232992


namespace NUMINAMATH_CALUDE_tetrahedron_volume_is_4_sqrt_6_div_3_l2329_232977

/-- Tetrahedron with specific face angles and areas -/
structure Tetrahedron where
  /-- Face angle APB -/
  angle_APB : ℝ
  /-- Face angle BPC -/
  angle_BPC : ℝ
  /-- Face angle CPA -/
  angle_CPA : ℝ
  /-- Area of face PAB -/
  area_PAB : ℝ
  /-- Area of face PBC -/
  area_PBC : ℝ
  /-- Area of face PCA -/
  area_PCA : ℝ
  /-- All face angles are 60° -/
  angle_constraint : angle_APB = 60 ∧ angle_BPC = 60 ∧ angle_CPA = 60
  /-- Areas of faces are √3/2, 2, and 1 -/
  area_constraint : area_PAB = Real.sqrt 3 / 2 ∧ area_PBC = 2 ∧ area_PCA = 1

/-- Volume of a tetrahedron -/
noncomputable def tetrahedronVolume (t : Tetrahedron) : ℝ := sorry

/-- Theorem: The volume of the specified tetrahedron is 4√6/3 -/
theorem tetrahedron_volume_is_4_sqrt_6_div_3 (t : Tetrahedron) :
  tetrahedronVolume t = 4 * Real.sqrt 6 / 3 := by
  sorry


end NUMINAMATH_CALUDE_tetrahedron_volume_is_4_sqrt_6_div_3_l2329_232977


namespace NUMINAMATH_CALUDE_total_boys_l2329_232976

theorem total_boys (total_children happy_children sad_children neutral_children : ℕ)
  (girls happy_boys sad_girls neutral_boys : ℕ)
  (h1 : total_children = 60)
  (h2 : happy_children = 30)
  (h3 : sad_children = 10)
  (h4 : neutral_children = 20)
  (h5 : girls = 41)
  (h6 : happy_boys = 6)
  (h7 : sad_girls = 4)
  (h8 : neutral_boys = 7)
  (h9 : total_children = happy_children + sad_children + neutral_children)
  (h10 : total_children = girls + (happy_boys + (sad_children - sad_girls) + neutral_boys)) :
  happy_boys + (sad_children - sad_girls) + neutral_boys = 19 := by
  sorry

#check total_boys

end NUMINAMATH_CALUDE_total_boys_l2329_232976


namespace NUMINAMATH_CALUDE_system_inequality_equivalence_l2329_232984

theorem system_inequality_equivalence (x y m : ℝ) :
  (x - 2*y = 1 ∧ 2*x + y = 4*m) → (x + 3*y < 6 ↔ m < 7/4) := by
  sorry

end NUMINAMATH_CALUDE_system_inequality_equivalence_l2329_232984


namespace NUMINAMATH_CALUDE_urn_problem_l2329_232971

theorem urn_problem (N : ℝ) : 
  (5 / 10 * 20 / (20 + N) + 5 / 10 * N / (20 + N) = 0.6) → N = 20 := by
  sorry

end NUMINAMATH_CALUDE_urn_problem_l2329_232971


namespace NUMINAMATH_CALUDE_problem_statement_l2329_232928

def p (a : ℝ) : Prop := ∀ x ∈ Set.Icc 1 2, x^2 - a ≥ 0

def q (a : ℝ) : Prop := ∃ x : ℝ, x^2 + 2*a*x + 2 - a = 0

theorem problem_statement (a : ℝ) :
  (p a ↔ a ≤ 1) ∧
  ((p a ∨ q a) ∧ ¬(p a ∧ q a) ↔ a > 1 ∨ (-2 < a ∧ a < 1)) :=
sorry

end NUMINAMATH_CALUDE_problem_statement_l2329_232928


namespace NUMINAMATH_CALUDE_smallest_integer_l2329_232935

theorem smallest_integer (a b : ℕ) (ha : a = 60) (h_lcm_gcd : Nat.lcm a b / Nat.gcd a b = 45) :
  b ≥ 1080 ∧ ∀ c : ℕ, c < 1080 → Nat.lcm a c / Nat.gcd a c ≠ 45 := by
  sorry

end NUMINAMATH_CALUDE_smallest_integer_l2329_232935


namespace NUMINAMATH_CALUDE_parabola_focal_line_properties_l2329_232958

/-- Parabola structure -/
structure Parabola where
  p : ℝ
  pos_p : p > 0

/-- Point on a parabola -/
structure ParabolaPoint (para : Parabola) where
  x : ℝ
  y : ℝ
  on_parabola : y^2 = 2 * para.p * x

/-- Line through focus intersecting parabola -/
structure FocalLine (para : Parabola) where
  A : ParabolaPoint para
  B : ParabolaPoint para

/-- Theorem statement -/
theorem parabola_focal_line_properties (para : Parabola) (l : FocalLine para) :
  ∃ (N : ℝ × ℝ) (P : ℝ × ℝ),
    -- 1. FN = 1/2 * AB
    (N.1 - para.p/2)^2 + N.2^2 = (1/2)^2 * ((l.A.x - l.B.x)^2 + (l.A.y - l.B.y)^2) ∧
    -- 2. Trajectory of P
    P.1 + para.p/2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_parabola_focal_line_properties_l2329_232958


namespace NUMINAMATH_CALUDE_city_mpg_is_24_l2329_232967

/-- Represents the fuel efficiency of a car in different driving conditions. -/
structure CarFuelEfficiency where
  highway_miles_per_tankful : ℝ
  city_miles_per_tankful : ℝ
  highway_city_mpg_difference : ℝ

/-- Calculates the city miles per gallon given the car's fuel efficiency data. -/
def city_mpg (car : CarFuelEfficiency) : ℝ :=
  -- The actual calculation is not provided here
  sorry

/-- Theorem stating that for the given car fuel efficiency data, 
    the city miles per gallon is 24. -/
theorem city_mpg_is_24 (car : CarFuelEfficiency) 
  (h1 : car.highway_miles_per_tankful = 462)
  (h2 : car.city_miles_per_tankful = 336)
  (h3 : car.highway_city_mpg_difference = 9) :
  city_mpg car = 24 := by
  sorry

end NUMINAMATH_CALUDE_city_mpg_is_24_l2329_232967


namespace NUMINAMATH_CALUDE_identify_burned_bulb_l2329_232922

/-- Represents the time in seconds for screwing or unscrewing a bulb -/
def operation_time : ℕ := 10

/-- Represents the number of bulbs in the series -/
def num_bulbs : ℕ := 4

/-- Represents the minimum time to identify the burned-out bulb -/
def min_identification_time : ℕ := 60

/-- Theorem stating that the minimum time to identify the burned-out bulb is 60 seconds -/
theorem identify_burned_bulb :
  ∀ (burned_bulb_position : Fin num_bulbs),
  min_identification_time = operation_time * (2 * (num_bulbs - 1)) :=
by sorry

end NUMINAMATH_CALUDE_identify_burned_bulb_l2329_232922


namespace NUMINAMATH_CALUDE_sum_and_count_theorem_l2329_232946

def sumIntegers (a b : ℕ) : ℕ := (b - a + 1) * (a + b) / 2

def countEvenIntegers (a b : ℕ) : ℕ := (b - a) / 2 + 1

theorem sum_and_count_theorem :
  let x := sumIntegers 30 40
  let y := countEvenIntegers 30 40
  x + y = 391 := by sorry

end NUMINAMATH_CALUDE_sum_and_count_theorem_l2329_232946


namespace NUMINAMATH_CALUDE_bus_equations_l2329_232942

/-- Given m buses and n people, if 40 people per bus leaves 10 people without a seat
    and 43 people per bus leaves 1 person without a seat, then two equations hold. -/
theorem bus_equations (m n : ℕ) 
    (h1 : 40 * m + 10 = n) 
    (h2 : 43 * m + 1 = n) : 
    (40 * m + 10 = 43 * m + 1) ∧ ((n - 10) / 40 = (n - 1) / 43) := by
  sorry

end NUMINAMATH_CALUDE_bus_equations_l2329_232942


namespace NUMINAMATH_CALUDE_solutions_to_equation_all_solutions_l2329_232953

def solutions : Set ℂ := {1 + Complex.I * (2 : ℂ)^(1/3 : ℂ), 
                          1 - Complex.I * (2 : ℂ)^(1/3 : ℂ), 
                          -1 + Complex.I * (2 : ℂ)^(1/3 : ℂ), 
                          -1 - Complex.I * (2 : ℂ)^(1/3 : ℂ), 
                          Complex.I * (2 : ℂ)^(1/3 : ℂ), 
                          -Complex.I * (2 : ℂ)^(1/3 : ℂ)}

theorem solutions_to_equation : ∀ z ∈ solutions, z^6 = -8 :=
by sorry

theorem all_solutions : ∀ z : ℂ, z^6 = -8 → z ∈ solutions :=
by sorry

end NUMINAMATH_CALUDE_solutions_to_equation_all_solutions_l2329_232953


namespace NUMINAMATH_CALUDE_negative_sum_l2329_232915

theorem negative_sum : (-2) + (-5) = -7 := by
  sorry

end NUMINAMATH_CALUDE_negative_sum_l2329_232915


namespace NUMINAMATH_CALUDE_problem_solution_l2329_232991

theorem problem_solution : Real.sqrt 9 + 2⁻¹ + (-1)^2023 = 5/2 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l2329_232991


namespace NUMINAMATH_CALUDE_round_trip_average_speed_l2329_232980

/-- Calculates the average speed of a round trip given the following conditions:
  * The total distance of the round trip is 4 miles
  * The outbound journey of 2 miles takes 1 hour
  * The return journey of 2 miles is completed at a speed of 6.000000000000002 miles/hour
-/
theorem round_trip_average_speed : 
  let total_distance : ℝ := 4
  let outbound_distance : ℝ := 2
  let outbound_time : ℝ := 1
  let return_speed : ℝ := 6.000000000000002
  let return_time : ℝ := outbound_distance / return_speed
  let total_time : ℝ := outbound_time + return_time
  total_distance / total_time = 3 := by
sorry

end NUMINAMATH_CALUDE_round_trip_average_speed_l2329_232980


namespace NUMINAMATH_CALUDE_binary_38_correct_l2329_232990

/-- Converts a binary number represented as a list of bits to its decimal equivalent -/
def binary_to_decimal (bits : List Bool) : Nat :=
  bits.enum.foldl (fun acc (i, b) => acc + if b then 2^i else 0) 0

/-- The binary representation of 38 -/
def binary_38 : List Bool := [false, true, true, false, false, true]

/-- Theorem stating that the binary representation of 38 is correct -/
theorem binary_38_correct : binary_to_decimal binary_38 = 38 := by
  sorry

#eval binary_to_decimal binary_38

end NUMINAMATH_CALUDE_binary_38_correct_l2329_232990


namespace NUMINAMATH_CALUDE_initial_shells_amount_l2329_232964

/-- The amount of shells initially in Jovana's bucket -/
def initial_shells : ℕ := sorry

/-- The amount of shells added to fill the bucket -/
def added_shells : ℕ := 12

/-- The total amount of shells after filling the bucket -/
def total_shells : ℕ := 17

/-- Theorem stating that the initial amount of shells is 5 pounds -/
theorem initial_shells_amount : initial_shells = 5 := by
  sorry

end NUMINAMATH_CALUDE_initial_shells_amount_l2329_232964


namespace NUMINAMATH_CALUDE_solve_for_y_l2329_232930

theorem solve_for_y (x y : ℚ) (h1 : x = 202) (h2 : x^3 * y - 4 * x^2 * y + 2 * x * y = 808080) : y = 1/10 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_y_l2329_232930


namespace NUMINAMATH_CALUDE_penalty_kicks_count_l2329_232932

theorem penalty_kicks_count (total_players : ℕ) (goalies : ℕ) : 
  total_players = 20 ∧ goalies = 3 → 
  (total_players - goalies) * goalies + goalies * (goalies - 1) = 57 := by
  sorry

end NUMINAMATH_CALUDE_penalty_kicks_count_l2329_232932


namespace NUMINAMATH_CALUDE_cuts_equality_l2329_232925

/-- Represents a bagel -/
structure Bagel :=
  (intact : Bool)

/-- Represents the result of cutting a bagel -/
inductive CutResult
  | Log
  | TwoSectors

/-- Function to cut a bagel -/
def cut_bagel (b : Bagel) (result : CutResult) : Nat :=
  match result with
  | CutResult.Log => 1
  | CutResult.TwoSectors => 1

/-- Theorem stating that the number of cuts is the same for both operations -/
theorem cuts_equality (b : Bagel) :
  cut_bagel b CutResult.Log = cut_bagel b CutResult.TwoSectors :=
by
  sorry

end NUMINAMATH_CALUDE_cuts_equality_l2329_232925


namespace NUMINAMATH_CALUDE_number_of_students_l2329_232959

/-- Given an initial average of 100, and a correction of one student's mark from 60 to 10
    resulting in a new average of 98, prove that the number of students in the class is 25. -/
theorem number_of_students (initial_average : ℝ) (wrong_mark : ℝ) (correct_mark : ℝ) (new_average : ℝ)
  (h1 : initial_average = 100)
  (h2 : wrong_mark = 60)
  (h3 : correct_mark = 10)
  (h4 : new_average = 98) :
  ∃ n : ℕ, n * new_average = n * initial_average - (wrong_mark - correct_mark) ∧ n = 25 := by
  sorry

end NUMINAMATH_CALUDE_number_of_students_l2329_232959


namespace NUMINAMATH_CALUDE_complex_equation_solution_l2329_232929

/-- Given a, b ∈ ℝ and a - bi = (1 + i)i³, prove that a = 1 and b = -1 -/
theorem complex_equation_solution (a b : ℝ) : 
  (Complex.mk a (-b) = Complex.I * Complex.I * Complex.I * (1 + Complex.I)) → 
  (a = 1 ∧ b = -1) :=
by sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l2329_232929


namespace NUMINAMATH_CALUDE_pedestrian_cyclist_speeds_l2329_232995

/-- Proves that given the conditions of the problem, the pedestrian's speed is 5 km/h and the cyclist's speed is 11 km/h -/
theorem pedestrian_cyclist_speeds :
  ∀ (v₁ v₂ : ℝ),
    (27 : ℝ) > 0 →  -- Distance from A to B is 27 km
    (12 / 5 * v₁ - v₂ = 1) →  -- After 1 hour of cyclist's travel, they were 1 km behind the pedestrian
    (27 - 17 / 5 * v₁ = 2 * (27 - 2 * v₂)) →  -- After 2 hours of cyclist's travel, the cyclist had half the distance to B remaining compared to the pedestrian
    v₁ = 5 ∧ v₂ = 11 := by
  sorry

#check pedestrian_cyclist_speeds

end NUMINAMATH_CALUDE_pedestrian_cyclist_speeds_l2329_232995


namespace NUMINAMATH_CALUDE_max_chords_for_ten_points_l2329_232906

/-- Given n points on a circle, max_chords_no_triangle calculates the maximum number of chords
    that can be drawn between these points without forming any triangles. -/
def max_chords_no_triangle (n : ℕ) : ℕ :=
  (n^2) / 4

/-- Theorem stating that for 10 points on a circle, the maximum number of chords
    that can be drawn without forming triangles is 25. -/
theorem max_chords_for_ten_points :
  max_chords_no_triangle 10 = 25 :=
by sorry

end NUMINAMATH_CALUDE_max_chords_for_ten_points_l2329_232906


namespace NUMINAMATH_CALUDE_square_sum_minus_triple_product_l2329_232920

theorem square_sum_minus_triple_product (x y : ℝ) 
  (h1 : x * y = 3) 
  (h2 : x + y = 4) : 
  x^2 + y^2 - 3*x*y = 1 := by
sorry

end NUMINAMATH_CALUDE_square_sum_minus_triple_product_l2329_232920


namespace NUMINAMATH_CALUDE_trig_expression_equals_one_l2329_232951

theorem trig_expression_equals_one : 
  (2 * Real.sin (46 * π / 180) - Real.sqrt 3 * Real.cos (74 * π / 180)) / Real.cos (16 * π / 180) = 1 := by
  sorry

end NUMINAMATH_CALUDE_trig_expression_equals_one_l2329_232951


namespace NUMINAMATH_CALUDE_sum_becomes_27_l2329_232956

def numbers : List ℝ := [1.05, 1.15, 1.25, 1.4, 1.5, 1.6, 1.75, 1.85, 1.95]

def sum_with_error (nums : List ℝ) (error_index : Nat) : ℝ :=
  let error_value := nums[error_index]! * 10
  (nums.sum - nums[error_index]!) + error_value

theorem sum_becomes_27 :
  ∃ (i : Nat), i < numbers.length ∧ sum_with_error numbers i = 27 := by
  sorry

end NUMINAMATH_CALUDE_sum_becomes_27_l2329_232956


namespace NUMINAMATH_CALUDE_inequality_proof_l2329_232934

theorem inequality_proof (a b c d : ℝ) 
  (h_pos : a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0) 
  (h_sum : a + b + c + d = 3) : 
  1/a^3 + 1/b^3 + 1/c^3 + 1/d^3 ≤ 1/(a^3 * b^3 * c^3 * d^3) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2329_232934


namespace NUMINAMATH_CALUDE_triangle_sum_theorem_l2329_232936

noncomputable def triangle_sum (AB AC BC CX₁ : ℝ) : ℝ :=
  let M := BC / 2
  let NC := (5 / 13) * CX₁
  let X₁C := Real.sqrt (CX₁^2 - NC^2)
  let BN := BC - NC
  let X₁B := Real.sqrt (BN^2 + X₁C^2)
  let X₂X₁ := X₁B * (16 / 63)
  let ratio := 1 - (X₁B * (65 / 63) / AB)
  (X₁B + X₂X₁) / (1 - ratio)

theorem triangle_sum_theorem (AB AC BC CX₁ : ℝ) 
  (h1 : AB = 182) (h2 : AC = 182) (h3 : BC = 140) (h4 : CX₁ = 130) :
  triangle_sum AB AC BC CX₁ = 1106 / 5 :=
by sorry

end NUMINAMATH_CALUDE_triangle_sum_theorem_l2329_232936


namespace NUMINAMATH_CALUDE_complex_squared_norm_l2329_232945

theorem complex_squared_norm (w : ℂ) (h : w^2 + Complex.abs w^2 = 7 + 2*I) : 
  Complex.abs w^2 = 53/14 := by
  sorry

end NUMINAMATH_CALUDE_complex_squared_norm_l2329_232945


namespace NUMINAMATH_CALUDE_inequalities_comparison_l2329_232908

theorem inequalities_comparison (a b : ℝ) (h1 : 0 < a) (h2 : a < b) (h3 : b < 1) :
  ((1/2 : ℝ)^a > (1/2 : ℝ)^b) ∧
  (1/a > 1/b) ∧
  (b^2 > a^2) ∧
  (¬(Real.log a > Real.log b)) := by
  sorry

end NUMINAMATH_CALUDE_inequalities_comparison_l2329_232908


namespace NUMINAMATH_CALUDE_distance_between_circle_centers_l2329_232910

/-- The distance between the centers of two circles with polar equations ρ = 2cos(θ) and ρ = 4sin(θ) is √5. -/
theorem distance_between_circle_centers :
  let circle1 : ℝ → ℝ := fun θ ↦ 2 * Real.cos θ
  let circle2 : ℝ → ℝ := fun θ ↦ 4 * Real.sin θ
  let center1 : ℝ × ℝ := (1, 0)
  let center2 : ℝ × ℝ := (0, 2)
  (center1.1 - center2.1)^2 + (center1.2 - center2.2)^2 = 5 := by
  sorry

#check distance_between_circle_centers

end NUMINAMATH_CALUDE_distance_between_circle_centers_l2329_232910


namespace NUMINAMATH_CALUDE_floor_ceiling_sum_l2329_232923

theorem floor_ceiling_sum : ⌊(-3.87 : ℝ)⌋ + ⌈(30.75 : ℝ)⌉ = 27 := by
  sorry

end NUMINAMATH_CALUDE_floor_ceiling_sum_l2329_232923


namespace NUMINAMATH_CALUDE_range_of_a_l2329_232917

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x + a * x

theorem range_of_a (m n : ℝ) (h1 : 0 < m) (h2 : m < n) (h3 : 0 ≤ a) 
  (h4 : Set.Icc m n ⊆ Set.range (f a))
  (h5 : Set.Icc m n ⊆ Set.range (f a ∘ f a)) :
  1 - Real.exp (-1) ≤ a ∧ a < 1 := by sorry

end NUMINAMATH_CALUDE_range_of_a_l2329_232917


namespace NUMINAMATH_CALUDE_plywood_cut_perimeter_difference_l2329_232957

/-- Represents the dimensions of a rectangle --/
structure Rectangle where
  length : ℝ
  width : ℝ

/-- Calculates the perimeter of a rectangle --/
def perimeter (r : Rectangle) : ℝ := 2 * (r.length + r.width)

/-- Represents the plywood and its cutting --/
structure Plywood where
  original : Rectangle
  pieces : Fin 8 → Rectangle

/-- The original plywood dimensions --/
def original_plywood : Rectangle := { length := 16, width := 4 }

theorem plywood_cut_perimeter_difference :
  ∃ (max_cut min_cut : Plywood),
    (∀ i : Fin 8, perimeter (max_cut.pieces i) ≥ perimeter (min_cut.pieces i)) ∧
    (∀ cut : Plywood, ∀ i : Fin 8, 
      perimeter (cut.pieces i) ≤ perimeter (max_cut.pieces i) ∧
      perimeter (cut.pieces i) ≥ perimeter (min_cut.pieces i)) ∧
    (∀ i j : Fin 8, max_cut.pieces i = max_cut.pieces j) ∧
    (∀ i j : Fin 8, min_cut.pieces i = min_cut.pieces j) ∧
    (max_cut.original = original_plywood) ∧
    (min_cut.original = original_plywood) ∧
    (perimeter (max_cut.pieces 0) - perimeter (min_cut.pieces 0) = 21) :=
by sorry

end NUMINAMATH_CALUDE_plywood_cut_perimeter_difference_l2329_232957


namespace NUMINAMATH_CALUDE_cube_sum_reciprocal_l2329_232974

theorem cube_sum_reciprocal (x : ℝ) (h : x^3 + 1/x^3 = 110) : x + 1/x = 5 := by
  sorry

end NUMINAMATH_CALUDE_cube_sum_reciprocal_l2329_232974


namespace NUMINAMATH_CALUDE_largest_good_and_smallest_bad_l2329_232985

def is_good_number (M : ℕ) : Prop :=
  ∃ a b c d : ℕ, M ≤ a ∧ a < b ∧ b ≤ c ∧ c < d ∧ d ≤ M + 49 ∧ a * d = b * c

theorem largest_good_and_smallest_bad :
  (is_good_number 576) ∧
  (∀ M : ℕ, M ≥ 577 → ¬(is_good_number M)) ∧
  (¬(is_good_number 443)) ∧
  (∀ M : ℕ, 288 < M ∧ M ≤ 442 → is_good_number M) :=
by sorry

end NUMINAMATH_CALUDE_largest_good_and_smallest_bad_l2329_232985


namespace NUMINAMATH_CALUDE_eds_initial_money_l2329_232993

def night_rate : ℚ := 1.5
def morning_rate : ℚ := 2
def night_hours : ℕ := 6
def morning_hours : ℕ := 4
def remaining_money : ℚ := 63

theorem eds_initial_money :
  night_rate * night_hours + morning_rate * morning_hours + remaining_money = 80 := by
  sorry

end NUMINAMATH_CALUDE_eds_initial_money_l2329_232993


namespace NUMINAMATH_CALUDE_isosceles_triangles_in_right_triangle_l2329_232921

theorem isosceles_triangles_in_right_triangle :
  ∀ (a b c : ℝ) (S₁ S₂ S₃ : ℝ) (x : ℝ),
    a = 1 →
    b = Real.sqrt 3 →
    c^2 = a^2 + b^2 →
    S₁ + S₂ + S₃ = (1/2) * a * b →
    S₁ = (1/2) * (a/3) * x →
    S₂ = (1/2) * (b/3) * x →
    S₃ = (1/2) * (c/3) * x →
    x = Real.sqrt 109 / 6 :=
by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangles_in_right_triangle_l2329_232921


namespace NUMINAMATH_CALUDE_increasing_function_inequality_l2329_232913

theorem increasing_function_inequality (f : ℝ → ℝ) (a : ℝ) :
  (∀ x y, x < y → x ∈ Set.Icc (-1) 2 → y ∈ Set.Icc (-1) 2 → f x < f y) →
  (∀ x, x ∈ Set.Iio 2 → f x ≠ 0) →
  f (a - 1) > f (1 - 3 * a) →
  (1/2 : ℝ) < a ∧ a ≤ (2/3 : ℝ) := by
  sorry


end NUMINAMATH_CALUDE_increasing_function_inequality_l2329_232913


namespace NUMINAMATH_CALUDE_complex_number_in_fourth_quadrant_l2329_232989

theorem complex_number_in_fourth_quadrant :
  let z : ℂ := (2 - I) / (3 + 4*I)
  (z.re > 0 ∧ z.im < 0) := by sorry

end NUMINAMATH_CALUDE_complex_number_in_fourth_quadrant_l2329_232989


namespace NUMINAMATH_CALUDE_monkey_peaches_l2329_232927

theorem monkey_peaches : ∃ (n : ℕ) (m : ℕ), 
  n > 0 ∧ 
  n % 3 = 0 ∧ 
  m % n = 27 ∧ 
  (m - 27) / n = 5 ∧ 
  ∃ (x : ℕ), 0 < x ∧ x < 7 ∧ m = 7 * n - x ∧
  m = 102 := by
  sorry

end NUMINAMATH_CALUDE_monkey_peaches_l2329_232927


namespace NUMINAMATH_CALUDE_problem_l2329_232998

def is_divisor (d n : ℕ) : Prop := n % d = 0

theorem problem (n : ℕ) (d : ℕ → ℕ) :
  n > 0 ∧
  (∀ i j, 1 ≤ i ∧ i < j ∧ j ≤ 15 → d i < d j) ∧
  (∀ i, 1 ≤ i ∧ i ≤ 15 → is_divisor (d i) n) ∧
  d 1 = 1 ∧
  n = d 13 + d 14 + d 15 ∧
  (d 5 + 1)^3 = d 15 + 1 →
  n = 1998 := by
sorry

end NUMINAMATH_CALUDE_problem_l2329_232998


namespace NUMINAMATH_CALUDE_smallest_m_for_integral_solutions_l2329_232943

theorem smallest_m_for_integral_solutions : ∃ (m : ℕ), 
  (∀ (k : ℕ), k < m → ¬∃ (x y : ℤ), 15 * x^2 - k * x + 315 = 0 ∧ 15 * y^2 - k * y + 315 = 0 ∧ x ≠ y) ∧
  (∃ (x y : ℤ), 15 * x^2 - m * x + 315 = 0 ∧ 15 * y^2 - m * y + 315 = 0 ∧ x ≠ y) ∧
  m = 150 := by
sorry

end NUMINAMATH_CALUDE_smallest_m_for_integral_solutions_l2329_232943


namespace NUMINAMATH_CALUDE_sum_of_fourth_powers_inequality_l2329_232981

theorem sum_of_fourth_powers_inequality (x y z : ℝ) 
  (h : x^2 + y^2 + z^2 + 9 = 4*(x + y + z)) :
  x^4 + y^4 + z^4 + 16*(x^2 + y^2 + z^2) ≥ 8*(x^3 + y^3 + z^3) + 27 ∧
  (x^4 + y^4 + z^4 + 16*(x^2 + y^2 + z^2) = 8*(x^3 + y^3 + z^3) + 27 ↔ 
   (x = 1 ∨ x = 3) ∧ (y = 1 ∨ y = 3) ∧ (z = 1 ∨ z = 3)) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_fourth_powers_inequality_l2329_232981


namespace NUMINAMATH_CALUDE_registration_methods_l2329_232933

/-- The number of students signing up for interest groups -/
def num_students : ℕ := 4

/-- The number of interest groups available -/
def num_groups : ℕ := 3

/-- Theorem stating the total number of registration methods -/
theorem registration_methods :
  (num_groups ^ num_students : ℕ) = 81 := by
  sorry

end NUMINAMATH_CALUDE_registration_methods_l2329_232933


namespace NUMINAMATH_CALUDE_sally_mcqueen_cost_l2329_232961

/-- The cost of Sally McQueen given the costs of Lightning McQueen and Mater -/
theorem sally_mcqueen_cost 
  (lightning_cost : ℝ) 
  (mater_cost_percentage : ℝ) 
  (sally_cost_multiplier : ℝ) 
  (h1 : lightning_cost = 140000)
  (h2 : mater_cost_percentage = 0.1)
  (h3 : sally_cost_multiplier = 3) : 
  sally_cost_multiplier * (mater_cost_percentage * lightning_cost) = 42000 := by
  sorry

end NUMINAMATH_CALUDE_sally_mcqueen_cost_l2329_232961


namespace NUMINAMATH_CALUDE_levi_basketball_score_l2329_232987

theorem levi_basketball_score (levi_initial : ℕ) (brother_initial : ℕ) (brother_additional : ℕ) (goal_difference : ℕ) :
  levi_initial = 8 →
  brother_initial = 12 →
  brother_additional = 3 →
  goal_difference = 5 →
  (brother_initial + brother_additional + goal_difference) - levi_initial = 12 :=
by sorry

end NUMINAMATH_CALUDE_levi_basketball_score_l2329_232987


namespace NUMINAMATH_CALUDE_tree_height_differences_l2329_232965

def pine_height : ℚ := 15 + 1/4
def birch_height : ℚ := 20 + 1/2
def maple_height : ℚ := 18 + 3/4

theorem tree_height_differences :
  (birch_height - pine_height = 5 + 1/4) ∧
  (birch_height - maple_height = 1 + 3/4) := by
  sorry

end NUMINAMATH_CALUDE_tree_height_differences_l2329_232965


namespace NUMINAMATH_CALUDE_females_in_town_l2329_232940

/-- Represents the population of a town with a given male to female ratio. -/
structure TownPopulation where
  total : ℕ
  maleRatio : ℕ
  femaleRatio : ℕ

/-- Calculates the number of females in the town. -/
def femaleCount (town : TownPopulation) : ℕ :=
  (town.total * town.femaleRatio) / (town.maleRatio + town.femaleRatio)

/-- Theorem stating that in a town of 500 people with a 3:2 male to female ratio, there are 200 females. -/
theorem females_in_town (town : TownPopulation) 
    (h1 : town.total = 500) 
    (h2 : town.maleRatio = 3) 
    (h3 : town.femaleRatio = 2) : 
  femaleCount town = 200 := by
  sorry

end NUMINAMATH_CALUDE_females_in_town_l2329_232940


namespace NUMINAMATH_CALUDE_four_common_divisors_l2329_232916

/-- The number of positive integer divisors that simultaneously divide 60, 84, and 126 -/
def common_divisors : Nat :=
  (Nat.divisors 60 ∩ Nat.divisors 84 ∩ Nat.divisors 126).card

/-- Theorem stating that there are exactly 4 positive integers that simultaneously divide 60, 84, and 126 -/
theorem four_common_divisors : common_divisors = 4 := by
  sorry

end NUMINAMATH_CALUDE_four_common_divisors_l2329_232916


namespace NUMINAMATH_CALUDE_necessary_but_not_sufficient_l2329_232938

theorem necessary_but_not_sufficient : 
  (∀ x : ℝ, x * (x - 3) < 0 → |x - 1| < 2) ∧ 
  (∃ x : ℝ, |x - 1| < 2 ∧ x * (x - 3) ≥ 0) := by
  sorry

end NUMINAMATH_CALUDE_necessary_but_not_sufficient_l2329_232938


namespace NUMINAMATH_CALUDE_trigonometric_identity_l2329_232914

theorem trigonometric_identity : 
  (Real.sin (10 * π / 180) * Real.sin (80 * π / 180)) / 
  (Real.cos (35 * π / 180) ^ 2 - Real.sin (35 * π / 180) ^ 2) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identity_l2329_232914


namespace NUMINAMATH_CALUDE_grace_garden_seeds_l2329_232983

/-- Represents the number of large beds in Grace's garden -/
def num_large_beds : Nat := 2

/-- Represents the number of medium beds in Grace's garden -/
def num_medium_beds : Nat := 2

/-- Represents the number of rows in a large bed -/
def rows_large_bed : Nat := 4

/-- Represents the number of rows in a medium bed -/
def rows_medium_bed : Nat := 3

/-- Represents the number of seeds per row in a large bed -/
def seeds_per_row_large : Nat := 25

/-- Represents the number of seeds per row in a medium bed -/
def seeds_per_row_medium : Nat := 20

/-- Calculates the total number of seeds Grace can plant in her raised bed garden -/
def total_seeds : Nat :=
  num_large_beds * rows_large_bed * seeds_per_row_large +
  num_medium_beds * rows_medium_bed * seeds_per_row_medium

/-- Proves that the total number of seeds Grace can plant is 320 -/
theorem grace_garden_seeds : total_seeds = 320 := by
  sorry

end NUMINAMATH_CALUDE_grace_garden_seeds_l2329_232983


namespace NUMINAMATH_CALUDE_willy_finishes_series_in_30_days_l2329_232966

/-- Calculates the number of days needed to finish a TV series -/
def days_to_finish_series (seasons : ℕ) (episodes_per_season : ℕ) (episodes_per_day : ℕ) : ℕ :=
  (seasons * episodes_per_season) / episodes_per_day

/-- Proves that it takes 30 days to finish the given TV series -/
theorem willy_finishes_series_in_30_days :
  days_to_finish_series 3 20 2 = 30 := by
  sorry

end NUMINAMATH_CALUDE_willy_finishes_series_in_30_days_l2329_232966


namespace NUMINAMATH_CALUDE_third_degree_polynomial_property_l2329_232950

/-- A third-degree polynomial with real coefficients -/
def ThirdDegreePolynomial := ℝ → ℝ

/-- The property that |f(x)| = 15 for x ∈ {2, 4, 5, 6, 8, 9} -/
def HasAbsoluteValue15 (f : ThirdDegreePolynomial) : Prop :=
  ∀ x ∈ ({2, 4, 5, 6, 8, 9} : Set ℝ), |f x| = 15

theorem third_degree_polynomial_property (f : ThirdDegreePolynomial) 
  (h : HasAbsoluteValue15 f) : |f 0| = 135 := by
  sorry

end NUMINAMATH_CALUDE_third_degree_polynomial_property_l2329_232950


namespace NUMINAMATH_CALUDE_quadratic_radical_equivalence_l2329_232975

theorem quadratic_radical_equivalence (x : ℝ) :
  (∃ (y : ℝ), y > 0 ∧ y * y = x - 1 ∧ (∀ (z : ℝ), z > 0 → z * z = 8 → ∃ (k : ℚ), y = k * z)) →
  x = 9 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_radical_equivalence_l2329_232975


namespace NUMINAMATH_CALUDE_theater_ticket_pricing_l2329_232902

theorem theater_ticket_pricing (total_tickets : ℕ) (total_revenue : ℕ) 
  (balcony_price : ℕ) (balcony_orchestra_diff : ℕ) :
  total_tickets = 370 →
  total_revenue = 3320 →
  balcony_price = 8 →
  balcony_orchestra_diff = 190 →
  ∃ (orchestra_price : ℕ),
    orchestra_price = 12 ∧
    (total_tickets - balcony_orchestra_diff) / 2 * orchestra_price + 
    (total_tickets + balcony_orchestra_diff) / 2 * balcony_price = total_revenue :=
by sorry

end NUMINAMATH_CALUDE_theater_ticket_pricing_l2329_232902


namespace NUMINAMATH_CALUDE_goat_cost_is_400_l2329_232962

/-- The cost of a single goat in dollars -/
def goat_cost : ℝ := sorry

/-- The number of goats purchased -/
def num_goats : ℕ := 3

/-- The number of llamas purchased -/
def num_llamas : ℕ := 6

/-- The cost of a single llama in terms of goat cost -/
def llama_cost : ℝ := 1.5 * goat_cost

/-- The total amount spent on all animals -/
def total_spent : ℝ := 4800

theorem goat_cost_is_400 : goat_cost = 400 :=
  sorry

end NUMINAMATH_CALUDE_goat_cost_is_400_l2329_232962


namespace NUMINAMATH_CALUDE_gcd_cube_plus_square_and_linear_l2329_232973

theorem gcd_cube_plus_square_and_linear (n m : ℤ) (hn : n > 2^3) : 
  Int.gcd (n^3 + m^2) (n + 2) = 1 := by sorry

end NUMINAMATH_CALUDE_gcd_cube_plus_square_and_linear_l2329_232973


namespace NUMINAMATH_CALUDE_smallest_number_l2329_232982

def A : ℕ := 36

def B : ℕ := 27 + 5

def C : ℕ := 3 * 10

def D : ℕ := 40 - 3

theorem smallest_number (h : A = 36 ∧ B = 27 + 5 ∧ C = 3 * 10 ∧ D = 40 - 3) :
  C ≤ A ∧ C ≤ B ∧ C ≤ D :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_l2329_232982


namespace NUMINAMATH_CALUDE_equation_solution_l2329_232911

theorem equation_solution : ∃ x : ℝ, (x / 2 - 1 = 3) ∧ (x = 8) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2329_232911


namespace NUMINAMATH_CALUDE_circle_tangency_l2329_232949

/-- Two circles are tangent internally if the distance between their centers
    equals the difference of their radii -/
def are_tangent_internally (c1_center c2_center : ℝ × ℝ) (r1 r2 : ℝ) : Prop :=
  (c1_center.1 - c2_center.1)^2 + (c1_center.2 - c2_center.2)^2 = (r1 - r2)^2

theorem circle_tangency (m : ℝ) :
  are_tangent_internally (m, 0) (-1, 2*m) 2 3 →
  m = 0 ∨ m = -2/5 := by
  sorry

end NUMINAMATH_CALUDE_circle_tangency_l2329_232949


namespace NUMINAMATH_CALUDE_solution_set_equals_expected_solutions_l2329_232972

/-- The set of solutions to the system of equations -/
def SolutionSet : Set (ℝ × ℝ × ℝ) :=
  {(x, y, z) | 3 * (x^2 + y^2 + z^2) = 1 ∧ x^2*y^2 + y^2*z^2 + z^2*x^2 = x*y*z*(x + y + z)^3}

/-- The set of expected solutions -/
def ExpectedSolutions : Set (ℝ × ℝ × ℝ) :=
  {(1/3, 1/3, 1/3), (-1/3, -1/3, -1/3), (1/Real.sqrt 3, 0, 0), (0, 1/Real.sqrt 3, 0), (0, 0, 1/Real.sqrt 3)}

/-- Theorem stating that the solution set is equal to the expected solutions -/
theorem solution_set_equals_expected_solutions : SolutionSet = ExpectedSolutions := by
  sorry


end NUMINAMATH_CALUDE_solution_set_equals_expected_solutions_l2329_232972


namespace NUMINAMATH_CALUDE_complex_number_opposites_l2329_232926

theorem complex_number_opposites (b : ℝ) : 
  let z : ℂ := (2 - b * Complex.I) / (1 + 2 * Complex.I)
  (z.re = -z.im) → b = -2/3 := by
sorry

end NUMINAMATH_CALUDE_complex_number_opposites_l2329_232926


namespace NUMINAMATH_CALUDE_correct_division_result_l2329_232947

theorem correct_division_result (student_divisor student_quotient correct_divisor : ℕ) 
  (h1 : student_divisor = 63)
  (h2 : student_quotient = 24)
  (h3 : correct_divisor = 36) :
  (student_divisor * student_quotient) / correct_divisor = 42 :=
by sorry

end NUMINAMATH_CALUDE_correct_division_result_l2329_232947


namespace NUMINAMATH_CALUDE_spherical_segment_volume_l2329_232960

/-- Given a sphere of radius 10 cm, prove that a spherical segment with a ratio of 10:7 for its curved surface area to base area has a volume of 288π cm³ -/
theorem spherical_segment_volume (r : ℝ) (m : ℝ) (h_r : r = 10) 
  (h_ratio : (2 * r * m) / (m * (2 * r - m)) = 10 / 7) : 
  (m^2 * π / 3) * (3 * r - m) = 288 * π := by
  sorry

#check spherical_segment_volume

end NUMINAMATH_CALUDE_spherical_segment_volume_l2329_232960
