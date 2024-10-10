import Mathlib

namespace power_zero_eq_one_l1780_178015

theorem power_zero_eq_one (x : ℝ) (h : x ≠ 0) : x^0 = 1 := by
  sorry

end power_zero_eq_one_l1780_178015


namespace difference_of_numbers_l1780_178056

theorem difference_of_numbers (x y : ℝ) (h1 : x + y = 30) (h2 : x * y = 200) : 
  |x - y| = 10 := by
sorry

end difference_of_numbers_l1780_178056


namespace mean_temperature_l1780_178020

def temperatures : List ℚ := [-6, -3, -3, -4, 2, 4, 1]

def mean (list : List ℚ) : ℚ :=
  (list.sum) / list.length

theorem mean_temperature : mean temperatures = -6/7 := by
  sorry

end mean_temperature_l1780_178020


namespace smallest_with_16_divisors_l1780_178006

def divisor_count (n : ℕ+) : ℕ := (Nat.divisors n.val).card

def has_16_divisors (n : ℕ+) : Prop := divisor_count n = 16

theorem smallest_with_16_divisors : 
  ∃ (n : ℕ+), has_16_divisors n ∧ ∀ (m : ℕ+), has_16_divisors m → n ≤ m :=
by
  use 216
  sorry

end smallest_with_16_divisors_l1780_178006


namespace division_power_eq_inv_pow_l1780_178027

/-- Division power of a rational number -/
def division_power (a : ℚ) (n : ℕ) : ℚ :=
  if n = 0 then 1
  else if n = 1 then a
  else a / (division_power a (n-1))

/-- Theorem: Division power equals inverse raised to power (n-2) -/
theorem division_power_eq_inv_pow (a : ℚ) (n : ℕ) (h1 : a ≠ 0) (h2 : n ≥ 2) :
  division_power a n = (a⁻¹) ^ (n - 2) :=
by sorry

end division_power_eq_inv_pow_l1780_178027


namespace fraction_meaningful_l1780_178002

theorem fraction_meaningful (a : ℝ) : 
  (∃ x : ℝ, x = 1 / (a + 3)) ↔ a ≠ -3 := by
sorry

end fraction_meaningful_l1780_178002


namespace expression_evaluation_l1780_178014

theorem expression_evaluation : 2 + 3 * 4 - 5 + 6 / 2 = 12 := by
  sorry

end expression_evaluation_l1780_178014


namespace allocation_methods_count_l1780_178049

/-- Represents the number of male students -/
def num_males : ℕ := 4

/-- Represents the number of female students -/
def num_females : ℕ := 3

/-- Represents the total number of students -/
def total_students : ℕ := num_males + num_females

/-- Represents the minimum number of students in each group -/
def min_group_size : ℕ := 2

/-- Calculates the number of ways to divide students into two groups -/
def num_allocation_methods : ℕ := sorry

/-- Theorem stating that the number of allocation methods is 52 -/
theorem allocation_methods_count : num_allocation_methods = 52 := by sorry

end allocation_methods_count_l1780_178049


namespace root_sum_reciprocals_l1780_178058

-- Define the polynomial
def f (x : ℝ) := x^3 - x + 2

-- Define the roots
def a : ℝ := sorry
def b : ℝ := sorry
def c : ℝ := sorry

-- State the theorem
theorem root_sum_reciprocals :
  f a = 0 ∧ f b = 0 ∧ f c = 0 →
  1 / (a + 2) + 1 / (b + 2) + 1 / (c + 2) = 11 / 4 := by sorry

end root_sum_reciprocals_l1780_178058


namespace gwen_total_books_l1780_178033

/-- The number of books on each shelf -/
def books_per_shelf : ℕ := 9

/-- The number of shelves for mystery books -/
def mystery_shelves : ℕ := 3

/-- The number of shelves for picture books -/
def picture_shelves : ℕ := 5

/-- The total number of books Gwen has -/
def total_books : ℕ := books_per_shelf * (mystery_shelves + picture_shelves)

theorem gwen_total_books : total_books = 72 := by sorry

end gwen_total_books_l1780_178033


namespace farm_field_calculation_correct_l1780_178091

/-- Represents the farm field ploughing problem -/
structure FarmField where
  initialCapacityA : ℝ
  initialCapacityB : ℝ
  reducedCapacityA : ℝ
  reducedCapacityB : ℝ
  extraDays : ℕ
  unattendedArea : ℝ

/-- Calculates the area of the farm field and the initially planned work days -/
def calculateFarmFieldResult (f : FarmField) : ℝ × ℕ :=
  let initialTotalCapacity := f.initialCapacityA + f.initialCapacityB
  let reducedTotalCapacity := f.reducedCapacityA + f.reducedCapacityB
  let area := 6600
  let initialDays := 30
  (area, initialDays)

/-- Theorem stating the correctness of the farm field calculation -/
theorem farm_field_calculation_correct (f : FarmField) 
  (h1 : f.initialCapacityA = 120)
  (h2 : f.initialCapacityB = 100)
  (h3 : f.reducedCapacityA = f.initialCapacityA * 0.9)
  (h4 : f.reducedCapacityB = 90)
  (h5 : f.extraDays = 3)
  (h6 : f.unattendedArea = 60) :
  calculateFarmFieldResult f = (6600, 30) := by
  sorry

#eval calculateFarmFieldResult {
  initialCapacityA := 120,
  initialCapacityB := 100,
  reducedCapacityA := 108,
  reducedCapacityB := 90,
  extraDays := 3,
  unattendedArea := 60
}

end farm_field_calculation_correct_l1780_178091


namespace smallest_n_divisibility_l1780_178035

theorem smallest_n_divisibility : ∃ (n : ℕ), n > 0 ∧ 
  45 ∣ n^2 ∧ 720 ∣ n^3 ∧ 
  ∀ (m : ℕ), m > 0 → 45 ∣ m^2 → 720 ∣ m^3 → n ≤ m :=
by
  use 60
  sorry

end smallest_n_divisibility_l1780_178035


namespace sqrt_D_always_odd_l1780_178045

theorem sqrt_D_always_odd (x : ℤ) : 
  let a : ℤ := x
  let b : ℤ := x + 1
  let c : ℤ := a * b
  let D : ℤ := a^2 + b^2 + c^2
  ∃ (k : ℤ), D = (2 * k + 1)^2 := by sorry

end sqrt_D_always_odd_l1780_178045


namespace max_children_count_max_children_is_26_l1780_178047

def initial_apples : ℕ := 55
def initial_cookies : ℕ := 114
def initial_chocolates : ℕ := 83

def remaining_apples : ℕ := 3
def remaining_cookies : ℕ := 10
def remaining_chocolates : ℕ := 5

def distributed_apples : ℕ := initial_apples - remaining_apples
def distributed_cookies : ℕ := initial_cookies - remaining_cookies
def distributed_chocolates : ℕ := initial_chocolates - remaining_chocolates

theorem max_children_count : ℕ → Prop :=
  fun n =>
    n > 0 ∧
    distributed_apples % n = 0 ∧
    distributed_cookies % n = 0 ∧
    distributed_chocolates % n = 0 ∧
    ∀ m : ℕ, m > n →
      (distributed_apples % m ≠ 0 ∨
       distributed_cookies % m ≠ 0 ∨
       distributed_chocolates % m ≠ 0)

theorem max_children_is_26 : max_children_count 26 := by sorry

end max_children_count_max_children_is_26_l1780_178047


namespace sum_of_powers_l1780_178030

theorem sum_of_powers (x y : ℝ) (h1 : (x + y)^2 = 7) (h2 : (x - y)^2 = 3) :
  (x^2 + y^2 = 5) ∧ (x^4 + y^4 = 23) ∧ (x^6 + y^6 = 110) := by
  sorry

end sum_of_powers_l1780_178030


namespace opposite_face_is_blue_l1780_178063

/-- Represents the colors of the squares --/
inductive Color
  | R | B | O | Y | G | W

/-- Represents a square with colors on both sides --/
structure Square where
  front : Color
  back : Color

/-- Represents the cube formed by folding the squares --/
structure Cube where
  squares : List Square
  white_face : Color
  opposite_face : Color

/-- Axiom: The cube is formed by folding six hinged squares --/
axiom cube_formation (c : Cube) : c.squares.length = 6

/-- Axiom: The white face exists in the cube --/
axiom white_face_exists (c : Cube) : c.white_face = Color.W

/-- Theorem: The face opposite to the white face is blue --/
theorem opposite_face_is_blue (c : Cube) : c.opposite_face = Color.B := by
  sorry

end opposite_face_is_blue_l1780_178063


namespace tie_shirt_ratio_l1780_178034

/-- Represents the cost of a school uniform -/
structure UniformCost where
  pants : ℝ
  shirt : ℝ
  tie : ℝ
  socks : ℝ

/-- Calculates the total cost of a given number of uniforms -/
def totalCost (u : UniformCost) (n : ℕ) : ℝ :=
  n * (u.pants + u.shirt + u.tie + u.socks)

/-- Theorem: The ratio of tie cost to shirt cost is 1:5 given the uniform pricing conditions -/
theorem tie_shirt_ratio :
  ∀ (u : UniformCost),
    u.pants = 20 →
    u.shirt = 2 * u.pants →
    u.socks = 3 →
    totalCost u 5 = 355 →
    u.tie / u.shirt = 1 / 5 := by
  sorry


end tie_shirt_ratio_l1780_178034


namespace max_groups_is_nine_l1780_178060

/-- Represents the number of singers for each voice type -/
structure ChoirComposition :=
  (sopranos : ℕ)
  (altos : ℕ)
  (tenors : ℕ)
  (basses : ℕ)

/-- Represents the ratio of voice types required in each group -/
structure GroupRatio :=
  (soprano_ratio : ℕ)
  (alto_ratio : ℕ)
  (tenor_ratio : ℕ)
  (bass_ratio : ℕ)

/-- Function to calculate the maximum number of complete groups -/
def maxCompleteGroups (choir : ChoirComposition) (ratio : GroupRatio) : ℕ :=
  min (choir.sopranos / ratio.soprano_ratio)
      (min (choir.altos / ratio.alto_ratio)
           (min (choir.tenors / ratio.tenor_ratio)
                (choir.basses / ratio.bass_ratio)))

/-- Theorem stating that the maximum number of complete groups is 9 -/
theorem max_groups_is_nine :
  let choir := ChoirComposition.mk 10 15 12 18
  let ratio := GroupRatio.mk 1 1 1 2
  maxCompleteGroups choir ratio = 9 :=
by
  sorry

#check max_groups_is_nine

end max_groups_is_nine_l1780_178060


namespace min_value_theorem_l1780_178044

theorem min_value_theorem (m n : ℝ) (h1 : m * n > 0) 
  (h2 : -2 * m - n + 1 = 0) : 
  (2 / m + 1 / n) ≥ 9 := by
sorry

end min_value_theorem_l1780_178044


namespace one_sided_limits_arctg_reciprocal_l1780_178062

noncomputable def f (x : ℝ) : ℝ := Real.arctan (1 / (x - 1))

theorem one_sided_limits_arctg_reciprocal :
  (∀ ε > 0, ∃ δ > 0, ∀ x > 1, |x - 1| < δ → |f x - π/2| < ε) ∧
  (∀ ε > 0, ∃ δ > 0, ∀ x < 1, |x - 1| < δ → |f x + π/2| < ε) :=
sorry

end one_sided_limits_arctg_reciprocal_l1780_178062


namespace fib_10_calls_l1780_178023

def FIB : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | n+2 => FIB (n+1) + FIB n

def count_calls : ℕ → ℕ
  | 0 => 0
  | 1 => 0
  | n+2 => count_calls (n+1) + count_calls n + 2

theorem fib_10_calls : count_calls 10 = 176 := by
  sorry

end fib_10_calls_l1780_178023


namespace g_zero_at_negative_one_l1780_178086

def g (x s : ℝ) : ℝ := 3 * x^4 + x^3 - 2 * x^2 - 4 * x + s

theorem g_zero_at_negative_one (s : ℝ) : g (-1) s = 0 ↔ s = -4 := by
  sorry

end g_zero_at_negative_one_l1780_178086


namespace bobsQuestionsRatio_l1780_178092

/-- Represents the number of questions created in each hour -/
structure HourlyQuestions where
  first : ℕ
  second : ℕ
  third : ℕ

/-- Represents the conditions of the problem -/
def bobsQuestions : HourlyQuestions → Prop
  | ⟨first, second, third⟩ =>
    first = 13 ∧
    third = 2 * second ∧
    first + second + third = 91

/-- The theorem to be proved -/
theorem bobsQuestionsRatio (q : HourlyQuestions) :
  bobsQuestions q → q.second / q.first = 2 := by
  sorry

end bobsQuestionsRatio_l1780_178092


namespace smallest_n_multiple_of_seven_l1780_178095

theorem smallest_n_multiple_of_seven (x y : ℤ) 
  (hx : (x - 4) % 7 = 0) 
  (hy : (y + 4) % 7 = 0) : 
  (∃ n : ℕ+, (x^2 - x*y + y^2 + n) % 7 = 0 ∧ 
    ∀ m : ℕ+, (x^2 - x*y + y^2 + m) % 7 = 0 → n ≤ m) → 
  (∃ n : ℕ+, (x^2 - x*y + y^2 + n) % 7 = 0 ∧ 
    ∀ m : ℕ+, (x^2 - x*y + y^2 + m) % 7 = 0 → n ≤ m) ∧ 
  (∀ n : ℕ+, (x^2 - x*y + y^2 + n) % 7 = 0 ∧ 
    (∀ m : ℕ+, (x^2 - x*y + y^2 + m) % 7 = 0 → n ≤ m) → n = 1) :=
sorry

end smallest_n_multiple_of_seven_l1780_178095


namespace uma_income_l1780_178042

/-- Represents the income and expenditure of a person -/
structure Person where
  income : ℝ
  expenditure : ℝ

/-- The problem setup -/
def problem_setup (uma bala : Person) : Prop :=
  -- Income ratio
  uma.income / bala.income = 8 / 7 ∧
  -- Expenditure ratio
  uma.expenditure / bala.expenditure = 7 / 6 ∧
  -- Savings
  uma.income - uma.expenditure = 2000 ∧
  bala.income - bala.expenditure = 2000

/-- The theorem to prove -/
theorem uma_income (uma bala : Person) :
  problem_setup uma bala → uma.income = 8000 / 7.5 := by
  sorry

end uma_income_l1780_178042


namespace odd_function_property_l1780_178038

-- Define an odd function on ℝ
def odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f x

-- State the theorem
theorem odd_function_property (f : ℝ → ℝ) (h1 : odd_function f) (h2 : f 2 = 2) :
  f (-2) = -2 := by
  sorry

end odd_function_property_l1780_178038


namespace angle_ratio_l1780_178052

-- Define the points
variable (A B C P Q M : Point)

-- Define the angles
def angle (P Q R : Point) : ℝ := sorry

-- State the conditions
axiom trisect : angle A C P = angle P C Q ∧ angle P C Q = angle Q C B
axiom bisect : angle P C M = angle M C Q

-- State the theorem
theorem angle_ratio : 
  (angle M C Q) / (angle A C Q) = 1 / 4 := by sorry

end angle_ratio_l1780_178052


namespace chickens_in_coop_l1780_178057

theorem chickens_in_coop (coop run free_range : ℕ) : 
  run = 2 * coop →
  free_range = 2 * run - 4 →
  free_range = 52 →
  coop = 14 := by
sorry

end chickens_in_coop_l1780_178057


namespace optimal_purchase_is_maximal_l1780_178050

/-- The cost of a red pencil in kopecks -/
def red_cost : ℕ := 27

/-- The cost of a blue pencil in kopecks -/
def blue_cost : ℕ := 23

/-- The maximum total cost in kopecks -/
def max_cost : ℕ := 940

/-- The maximum allowed difference between the number of blue and red pencils -/
def max_diff : ℕ := 10

/-- Represents a valid purchase of pencils -/
structure PencilPurchase where
  red : ℕ
  blue : ℕ
  total_cost_valid : red * red_cost + blue * blue_cost ≤ max_cost
  diff_valid : blue ≤ red + max_diff

/-- The optimal purchase of pencils -/
def optimal_purchase : PencilPurchase :=
  { red := 14
  , blue := 24
  , total_cost_valid := by sorry
  , diff_valid := by sorry }

/-- Theorem stating that the optimal purchase maximizes the total number of pencils -/
theorem optimal_purchase_is_maximal :
  ∀ p : PencilPurchase, p.red + p.blue ≤ optimal_purchase.red + optimal_purchase.blue :=
by sorry

end optimal_purchase_is_maximal_l1780_178050


namespace book_pages_count_l1780_178076

/-- The number of pages Liam read in a week-long reading assignment -/
def totalPages (firstThreeDaysAvg : ℕ) (nextThreeDaysAvg : ℕ) (lastDayPages : ℕ) : ℕ :=
  3 * firstThreeDaysAvg + 3 * nextThreeDaysAvg + lastDayPages

/-- Theorem stating that the total number of pages in the book is 310 -/
theorem book_pages_count :
  totalPages 45 50 25 = 310 := by
  sorry

end book_pages_count_l1780_178076


namespace range_of_a_l1780_178028

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, x^2 - 8*x - 20 < 0 → x^2 - 2*x + 1 - a^2 ≤ 0) ∧ 
  (∃ x : ℝ, x^2 - 2*x + 1 - a^2 ≤ 0 ∧ x^2 - 8*x - 20 ≥ 0) ∧
  (a > 0) →
  a ≥ 9 := by
sorry

end range_of_a_l1780_178028


namespace sum_of_unknown_angles_l1780_178040

/-- A six-sided polygon with two right angles and three known angles -/
structure HexagonWithKnownAngles where
  -- The three known angles
  angle_P : ℝ
  angle_Q : ℝ
  angle_R : ℝ
  -- Conditions on the known angles
  angle_P_eq : angle_P = 30
  angle_Q_eq : angle_Q = 60
  angle_R_eq : angle_R = 34
  -- The polygon has two right angles
  has_two_right_angles : True

/-- The sum of the two unknown angles in the hexagon is 124° -/
theorem sum_of_unknown_angles (h : HexagonWithKnownAngles) :
  ∃ x y, x + y = 124 := by sorry

end sum_of_unknown_angles_l1780_178040


namespace part1_part2_l1780_178024

-- Define a "three times angle triangle"
def is_three_times_angle_triangle (a b c : ℝ) : Prop :=
  (a + b + c = 180) ∧ (a = 3 * b ∨ b = 3 * c ∨ c = 3 * a)

-- Part 1
theorem part1 : is_three_times_angle_triangle 35 40 105 := by sorry

-- Part 2
theorem part2 (a b c : ℝ) (h : is_three_times_angle_triangle a b c) (hb : b = 60) :
  (min a (min b c) = 20) ∨ (min a (min b c) = 30) := by sorry

end part1_part2_l1780_178024


namespace hyperbola_equilateral_triangle_l1780_178064

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x * y = 1

-- Define the two branches of the hyperbola
def branch1 (x y : ℝ) : Prop := hyperbola x y ∧ x > 0
def branch2 (x y : ℝ) : Prop := hyperbola x y ∧ x < 0

-- Define an equilateral triangle
def is_equilateral_triangle (P Q R : ℝ × ℝ) : Prop :=
  let (px, py) := P
  let (qx, qy) := Q
  let (rx, ry) := R
  (px - qx)^2 + (py - qy)^2 = (qx - rx)^2 + (qy - ry)^2 ∧
  (qx - rx)^2 + (qy - ry)^2 = (rx - px)^2 + (ry - py)^2

-- Main theorem
theorem hyperbola_equilateral_triangle :
  ∀ (Q R : ℝ × ℝ),
  hyperbola (-1) (-1) →
  branch2 (-1) (-1) →
  branch1 Q.1 Q.2 →
  branch1 R.1 R.2 →
  is_equilateral_triangle (-1, -1) Q R →
  (Q = (2 - Real.sqrt 3, 2 + Real.sqrt 3) ∧ R = (2 + Real.sqrt 3, 2 - Real.sqrt 3)) ∨
  (Q = (2 + Real.sqrt 3, 2 - Real.sqrt 3) ∧ R = (2 - Real.sqrt 3, 2 + Real.sqrt 3)) :=
by sorry

end hyperbola_equilateral_triangle_l1780_178064


namespace eccentricity_range_l1780_178093

/-- Represents an ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_pos : 0 < b ∧ b < a

/-- Represents a point on the coordinate plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Defines the property of a point P being inside the ellipse -/
def inside_ellipse (P : Point) (E : Ellipse) : Prop :=
  (P.x^2 / E.a^2) + (P.y^2 / E.b^2) < 1

/-- Defines the condition that the dot product of vectors PF₁ and PF₂ is zero -/
def orthogonal_foci (P : Point) (E : Ellipse) : Prop :=
  ∃ (F₁ F₂ : Point), (P.x - F₁.x) * (P.x - F₂.x) + (P.y - F₁.y) * (P.y - F₂.y) = 0

/-- Theorem stating the range of eccentricity for the ellipse -/
theorem eccentricity_range (E : Ellipse) 
  (h : ∀ P : Point, orthogonal_foci P E → inside_ellipse P E) :
  ∃ e : ℝ, 0 < e ∧ e < Real.sqrt 2 / 2 ∧ e^2 = (E.a^2 - E.b^2) / E.a^2 := by
  sorry

end eccentricity_range_l1780_178093


namespace circle_line_intersection_chord_length_l1780_178054

/-- A circle in the xy-plane -/
structure Circle where
  equation : ℝ → ℝ → ℝ → Prop

/-- A line in the xy-plane -/
structure Line where
  equation : ℝ → ℝ → Prop

/-- The length of a chord formed by the intersection of a circle and a line -/
def chordLength (c : Circle) (l : Line) : ℝ := sorry

theorem circle_line_intersection_chord_length (a : ℝ) :
  let c : Circle := { equation := fun x y z => x^2 + y^2 + 2*x - 2*y + z = 0 }
  let l : Line := { equation := fun x y => x + y + 2 = 0 }
  chordLength c l = 4 → a = -4 := by sorry

end circle_line_intersection_chord_length_l1780_178054


namespace green_toads_in_shrublands_l1780_178061

/-- Represents the different types of toads -/
inductive ToadType
| Green
| Brown
| Blue
| Red

/-- Represents the different habitats -/
inductive Habitat
| Wetlands
| Forests
| Grasslands
| Marshlands
| Shrublands

/-- The population ratio of toads -/
def populationRatio : ToadType → ℕ
| ToadType.Green => 1
| ToadType.Brown => 25
| ToadType.Blue => 10
| ToadType.Red => 20

/-- The proportion of brown toads that are spotted -/
def spottedBrownProportion : ℚ := 1/4

/-- The proportion of blue toads that are striped -/
def stripedBlueProportion : ℚ := 1/3

/-- The proportion of red toads with star pattern -/
def starPatternRedProportion : ℚ := 1/2

/-- The density of specific toad types in each habitat -/
def specificToadDensity : Habitat → ℚ
| Habitat.Wetlands => 60  -- spotted brown toads
| Habitat.Forests => 45   -- camouflaged blue toads
| Habitat.Grasslands => 100  -- star pattern red toads
| Habitat.Marshlands => 120  -- plain brown toads
| Habitat.Shrublands => 35   -- striped blue toads

/-- Theorem: The number of green toads per acre in Shrublands is 10.5 -/
theorem green_toads_in_shrublands :
  let totalBlueToads : ℚ := specificToadDensity Habitat.Shrublands / stripedBlueProportion
  let greenToads : ℚ := totalBlueToads / populationRatio ToadType.Blue
  greenToads = 10.5 := by sorry

end green_toads_in_shrublands_l1780_178061


namespace point_coordinates_l1780_178059

/-- A point in the Cartesian coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- Checks if a point is in the fourth quadrant -/
def is_fourth_quadrant (p : Point) : Prop :=
  p.x > 0 ∧ p.y < 0

/-- Distance of a point to the x-axis -/
def distance_to_x_axis (p : Point) : ℝ :=
  |p.y|

/-- Distance of a point to the y-axis -/
def distance_to_y_axis (p : Point) : ℝ :=
  |p.x|

theorem point_coordinates (P : Point) 
  (h1 : is_fourth_quadrant P)
  (h2 : distance_to_x_axis P = 2)
  (h3 : distance_to_y_axis P = 5) :
  P.x = 5 ∧ P.y = -2 := by
  sorry

end point_coordinates_l1780_178059


namespace max_value_abc_l1780_178029

theorem max_value_abc (a b c : ℝ) (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : 0 ≤ c) (h4 : a + b + c = 1) :
  ∃ (max : ℝ), max = Real.sqrt 3 / 18 ∧ 
  ∀ (x y z : ℝ), 0 ≤ x ∧ 0 ≤ y ∧ 0 ≤ z ∧ x + y + z = 1 → 
  x^2 * (y - z) + y^2 * (z - x) + z^2 * (x - y) ≤ max :=
by sorry

end max_value_abc_l1780_178029


namespace oranges_picked_total_l1780_178008

/-- The number of oranges Mary picked -/
def mary_oranges : ℕ := 122

/-- The number of oranges Jason picked -/
def jason_oranges : ℕ := 105

/-- The total number of oranges picked -/
def total_oranges : ℕ := mary_oranges + jason_oranges

theorem oranges_picked_total :
  total_oranges = 227 := by sorry

end oranges_picked_total_l1780_178008


namespace product_of_roots_l1780_178031

theorem product_of_roots : ∃ (r₁ r₂ : ℝ), 
  (r₁^2 + 18*r₁ + 30 = 2 * Real.sqrt (r₁^2 + 18*r₁ + 45)) ∧
  (r₂^2 + 18*r₂ + 30 = 2 * Real.sqrt (r₂^2 + 18*r₂ + 45)) ∧
  r₁ ≠ r₂ ∧
  r₁ * r₂ = 20 := by
  sorry

end product_of_roots_l1780_178031


namespace circle_area_from_circumference_l1780_178075

/-- Given a circle with circumference 36 cm, its area is 324/π square centimeters. -/
theorem circle_area_from_circumference :
  ∀ (r : ℝ), 2 * π * r = 36 → π * r^2 = 324 / π := by
  sorry

end circle_area_from_circumference_l1780_178075


namespace sandwich_cost_is_four_l1780_178069

/-- The cost of Karen's fast food order --/
def fast_food_order (burger_cost smoothie_cost sandwich_cost : ℚ) : Prop :=
  burger_cost = 5 ∧
  smoothie_cost = 4 ∧
  burger_cost + 2 * smoothie_cost + sandwich_cost = 17

theorem sandwich_cost_is_four :
  ∀ (burger_cost smoothie_cost sandwich_cost : ℚ),
    fast_food_order burger_cost smoothie_cost sandwich_cost →
    sandwich_cost = 4 := by
  sorry

end sandwich_cost_is_four_l1780_178069


namespace pen_price_calculation_l1780_178072

theorem pen_price_calculation (num_pens : ℕ) (num_pencils : ℕ) (total_cost : ℚ) (pencil_price : ℚ) :
  num_pens = 30 →
  num_pencils = 75 →
  total_cost = 690 →
  pencil_price = 2 →
  (total_cost - num_pencils * pencil_price) / num_pens = 18 := by
sorry

end pen_price_calculation_l1780_178072


namespace polygon_missing_angle_l1780_178016

theorem polygon_missing_angle (n : ℕ) (sum_n_minus_1 : ℝ) (h1 : n > 2) (h2 : sum_n_minus_1 = 2843) : 
  (n - 2) * 180 - sum_n_minus_1 = 37 := by
  sorry

end polygon_missing_angle_l1780_178016


namespace monochromatic_equilateral_triangle_l1780_178080

-- Define a type for colors
inductive Color
| White
| Black

-- Define a type for points in the plane
structure Point where
  x : ℝ
  y : ℝ

-- Define a coloring function
def coloring : Point → Color := sorry

-- Define the distance between two points
def distance (p q : Point) : ℝ := sorry

-- Define an equilateral triangle
structure EquilateralTriangle where
  a : Point
  b : Point
  c : Point
  eq_sides : distance a b = distance b c ∧ distance b c = distance c a

-- Theorem statement
theorem monochromatic_equilateral_triangle :
  ∃ (t : EquilateralTriangle),
    (distance t.a t.b = 1 ∨ distance t.a t.b = Real.sqrt 3) ∧
    (coloring t.a = coloring t.b ∧ coloring t.b = coloring t.c) := by
  sorry

end monochromatic_equilateral_triangle_l1780_178080


namespace sandbox_volume_calculation_l1780_178037

/-- The volume of a rectangular box with given dimensions -/
def sandbox_volume (length width depth : ℝ) : ℝ :=
  length * width * depth

/-- Theorem: The volume of the sandbox is 3,429,000 cubic centimeters -/
theorem sandbox_volume_calculation :
  sandbox_volume 312 146 75 = 3429000 := by
  sorry

end sandbox_volume_calculation_l1780_178037


namespace all_black_after_two_rotations_l1780_178025

/-- Represents a 4x4 grid where each cell can be either black or white -/
def Grid := Fin 4 → Fin 4 → Bool

/-- Returns true if the given position is on a diagonal of a 4x4 grid -/
def isDiagonal (row col : Fin 4) : Bool :=
  row = col ∨ row + col = 3

/-- Initial grid configuration with black diagonals -/
def initialGrid : Grid :=
  fun row col => isDiagonal row col

/-- Rotates a position 90 degrees clockwise in a 4x4 grid -/
def rotate (row col : Fin 4) : Fin 4 × Fin 4 :=
  (col, 3 - row)

/-- Applies the transformation rule after rotation -/
def transform (g : Grid) : Grid :=
  fun row col =>
    let (oldRow, oldCol) := rotate row col
    g row col ∨ initialGrid oldRow oldCol

/-- Applies two consecutive 90° rotations and transformations -/
def finalGrid : Grid :=
  transform (transform initialGrid)

/-- Theorem stating that all squares in the final grid are black -/
theorem all_black_after_two_rotations :
  ∀ row col, finalGrid row col = true := by sorry

end all_black_after_two_rotations_l1780_178025


namespace lottery_tickets_bought_l1780_178082

theorem lottery_tickets_bought (total_won : ℕ) (winning_number_value : ℕ) (winning_numbers_per_ticket : ℕ) : 
  total_won = 300 →
  winning_number_value = 20 →
  winning_numbers_per_ticket = 5 →
  (total_won / winning_number_value) / winning_numbers_per_ticket = 3 :=
by sorry

end lottery_tickets_bought_l1780_178082


namespace arithmetic_sequence_8th_term_l1780_178039

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d

theorem arithmetic_sequence_8th_term 
  (a : ℕ → ℝ) 
  (h_arithmetic : arithmetic_sequence a) 
  (h_sum : a 7 + a 8 + a 9 = 21) : 
  a 8 = 7 := by
sorry

end arithmetic_sequence_8th_term_l1780_178039


namespace negation_of_existence_proposition_l1780_178026

theorem negation_of_existence_proposition :
  (¬∃ x : ℝ, Real.exp x - x - 1 ≤ 0) ↔ (∀ x : ℝ, Real.exp x - x - 1 > 0) :=
by sorry

end negation_of_existence_proposition_l1780_178026


namespace f_is_odd_and_increasing_l1780_178070

-- Define the function f(x) = x|x|
def f (x : ℝ) : ℝ := x * abs x

-- Define what it means for a function to be odd
def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

-- Define what it means for a function to be increasing
def is_increasing (f : ℝ → ℝ) : Prop := ∀ a b, a < b → f a < f b

-- Theorem statement
theorem f_is_odd_and_increasing : is_odd f ∧ is_increasing f := by
  sorry

end f_is_odd_and_increasing_l1780_178070


namespace dogs_equal_initial_l1780_178019

/-- Calculates the remaining number of dogs in an animal rescue center after a series of events. -/
def remaining_dogs (initial : ℕ) (moved_in : ℕ) (first_adoption : ℕ) (second_adoption : ℕ) : ℕ :=
  initial + moved_in - first_adoption - second_adoption

/-- Theorem stating that the number of remaining dogs equals the initial number under specific conditions. -/
theorem dogs_equal_initial 
  (initial : ℕ) (moved_in : ℕ) (first_adoption : ℕ) (second_adoption : ℕ) 
  (h1 : initial = 200) 
  (h2 : moved_in = 100) 
  (h3 : first_adoption = 40) 
  (h4 : second_adoption = 60) : 
  remaining_dogs initial moved_in first_adoption second_adoption = initial :=
by sorry

end dogs_equal_initial_l1780_178019


namespace millet_majority_day_four_l1780_178046

/-- Amount of millet in the feeder on day n -/
def millet_amount (n : ℕ) : ℝ :=
  if n = 0 then 0
  else 0.4 + 0.7 * millet_amount (n - 1)

/-- Total amount of seeds in the feeder on day n -/
def total_seeds (n : ℕ) : ℝ := 1

theorem millet_majority_day_four :
  (∀ k < 4, millet_amount k ≤ 0.5) ∧ millet_amount 4 > 0.5 := by sorry

end millet_majority_day_four_l1780_178046


namespace balloon_distribution_l1780_178041

theorem balloon_distribution (total_balloons : ℕ) (num_friends : ℕ) (take_back : ℕ) 
  (h1 : total_balloons = 250)
  (h2 : num_friends = 5)
  (h3 : take_back = 11) :
  (total_balloons / num_friends) - take_back = 39 := by
  sorry

end balloon_distribution_l1780_178041


namespace smaller_number_proof_l1780_178005

theorem smaller_number_proof (x y : ℝ) (h1 : x + y = 15) (h2 : 3 * x = 5 * y - 11) : 
  min x y = 8 := by
sorry

end smaller_number_proof_l1780_178005


namespace mary_cut_roses_l1780_178017

/-- The number of roses Mary cut from her garden -/
def roses_cut (initial final : ℕ) : ℕ := final - initial

theorem mary_cut_roses : roses_cut 6 16 = 10 := by sorry

end mary_cut_roses_l1780_178017


namespace subset_relation_l1780_178079

theorem subset_relation (x : ℝ) : x^2 - x < 0 → x < 1 := by
  sorry

end subset_relation_l1780_178079


namespace total_price_calculation_l1780_178007

def jewelry_original_price : ℝ := 30
def painting_original_price : ℝ := 100
def jewelry_price_increase : ℝ := 10
def painting_price_increase_percentage : ℝ := 0.20
def jewelry_sales_tax : ℝ := 0.06
def painting_sales_tax : ℝ := 0.08
def discount_percentage : ℝ := 0.10
def discount_min_amount : ℝ := 800
def jewelry_quantity : ℕ := 2
def painting_quantity : ℕ := 5

def jewelry_new_price : ℝ := jewelry_original_price + jewelry_price_increase
def painting_new_price : ℝ := painting_original_price * (1 + painting_price_increase_percentage)

def jewelry_price_with_tax : ℝ := jewelry_new_price * (1 + jewelry_sales_tax)
def painting_price_with_tax : ℝ := painting_new_price * (1 + painting_sales_tax)

def total_price : ℝ := jewelry_price_with_tax * jewelry_quantity + painting_price_with_tax * painting_quantity

theorem total_price_calculation :
  total_price = 732.80 ∧ total_price < discount_min_amount :=
sorry

end total_price_calculation_l1780_178007


namespace equal_cost_at_40_bookshelves_l1780_178053

/-- The number of bookcases to be purchased -/
def num_bookcases : ℕ := 20

/-- The cost of a bookcase in dollars -/
def bookcase_cost : ℕ := 300

/-- The cost of a bookshelf in dollars -/
def bookshelf_cost : ℕ := 100

/-- The discount rate at supermarket B as a fraction -/
def discount_rate : ℚ := 1/5

/-- Calculate the cost at supermarket A -/
def cost_A (x : ℕ) : ℕ := num_bookcases * bookcase_cost + bookshelf_cost * (x - num_bookcases)

/-- Calculate the cost at supermarket B -/
def cost_B (x : ℕ) : ℚ := (1 - discount_rate) * (num_bookcases * bookcase_cost + x * bookshelf_cost)

theorem equal_cost_at_40_bookshelves :
  ∃ x : ℕ, x ≥ num_bookcases ∧ (cost_A x : ℚ) = cost_B x ∧ x = 40 := by
  sorry

end equal_cost_at_40_bookshelves_l1780_178053


namespace pears_left_theorem_l1780_178073

/-- The number of pears Keith and Mike are left with after Keith gives away some pears -/
def pears_left (keith_picked : ℕ) (mike_picked : ℕ) (keith_gave_away : ℕ) : ℕ :=
  (keith_picked - keith_gave_away) + mike_picked

/-- Theorem stating that Keith and Mike are left with 13 pears -/
theorem pears_left_theorem : pears_left 47 12 46 = 13 := by
  sorry

end pears_left_theorem_l1780_178073


namespace cubic_curve_tangent_line_bc_product_l1780_178083

/-- Given a cubic curve y = x³ + bx + c passing through (1, 2) with tangent line y = x + 1 at that point, 
    the product bc equals -6. -/
theorem cubic_curve_tangent_line_bc_product (b c : ℝ) : 
  (1^3 + b*1 + c = 2) →   -- Point (1, 2) is on the curve
  (3*1^2 + b = 1) →       -- Derivative at x = 1 is 1 (from tangent line y = x + 1)
  b * c = -6 := by sorry

end cubic_curve_tangent_line_bc_product_l1780_178083


namespace value_of_y_l1780_178067

theorem value_of_y : (2010^2 - 2010 + 1) / 2010 = 2009 + 1/2010 := by
  sorry

end value_of_y_l1780_178067


namespace margin_calculation_l1780_178018

-- Define the sheet dimensions and side margin
def sheet_width : ℝ := 20
def sheet_length : ℝ := 30
def side_margin : ℝ := 2

-- Define the percentage of the page used for typing
def typing_percentage : ℝ := 0.64

-- Define the function to calculate the typing area
def typing_area (top_bottom_margin : ℝ) : ℝ :=
  (sheet_width - 2 * side_margin) * (sheet_length - 2 * top_bottom_margin)

-- Define the theorem
theorem margin_calculation :
  ∃ (top_bottom_margin : ℝ),
    typing_area top_bottom_margin = typing_percentage * sheet_width * sheet_length ∧
    top_bottom_margin = 3 := by
  sorry

end margin_calculation_l1780_178018


namespace scientific_notation_of_229000_l1780_178013

theorem scientific_notation_of_229000 :
  229000 = 2.29 * (10 ^ 5) := by
  sorry

end scientific_notation_of_229000_l1780_178013


namespace plastic_rings_weight_l1780_178048

/-- The weight of the orange ring in ounces -/
def orange_weight : ℝ := 0.08333333333333333

/-- The weight of the purple ring in ounces -/
def purple_weight : ℝ := 0.3333333333333333

/-- The weight of the white ring in ounces -/
def white_weight : ℝ := 0.4166666666666667

/-- The total weight of all rings in ounces -/
def total_weight : ℝ := orange_weight + purple_weight + white_weight

theorem plastic_rings_weight : total_weight = 0.8333333333333333 := by
  sorry

end plastic_rings_weight_l1780_178048


namespace remainder_seven_count_l1780_178078

theorem remainder_seven_count : ∃! k : ℕ, k = (Finset.filter (fun n => 61 % n = 7) (Finset.range 62)).card := by
  sorry

end remainder_seven_count_l1780_178078


namespace solution_satisfies_system_l1780_178003

theorem solution_satisfies_system :
  let solutions : List (Int × Int) := [(-3, -1), (-1, -3), (1, 3), (3, 1)]
  ∀ (x y : Int), (x, y) ∈ solutions →
    (x^2 - x*y + y^2 = 7 ∧ x^4 + x^2*y^2 + y^4 = 91) := by
  sorry

end solution_satisfies_system_l1780_178003


namespace dani_pants_per_pair_l1780_178066

/-- Calculates the number of pants in each pair given the initial number of pants,
    the number of pants after a certain number of years, the number of pairs received each year,
    and the number of years. -/
def pants_per_pair (initial_pants : ℕ) (final_pants : ℕ) (pairs_per_year : ℕ) (years : ℕ) : ℕ :=
  let total_pairs := pairs_per_year * years
  let total_new_pants := final_pants - initial_pants
  total_new_pants / total_pairs

theorem dani_pants_per_pair :
  pants_per_pair 50 90 4 5 = 2 := by
  sorry

end dani_pants_per_pair_l1780_178066


namespace integral_equals_six_implies_b_equals_e_to_four_l1780_178009

theorem integral_equals_six_implies_b_equals_e_to_four (b : ℝ) :
  (∫ (x : ℝ) in e..b, 2 / x) = 6 → b = Real.exp 4 := by
  sorry

end integral_equals_six_implies_b_equals_e_to_four_l1780_178009


namespace derivative_of_f_l1780_178077

noncomputable def f (x : ℝ) : ℝ := 2^x - Real.log x / Real.log 3

theorem derivative_of_f (x : ℝ) (h : x > 0) :
  deriv f x = 2^x * Real.log 2 - 1 / (x * Real.log 3) :=
by sorry

end derivative_of_f_l1780_178077


namespace initial_orchids_count_l1780_178090

theorem initial_orchids_count (initial_roses : ℕ) (final_roses : ℕ) (final_orchids : ℕ) (total_flowers : ℕ) : 
  initial_roses = 13 →
  final_roses = 14 →
  final_orchids = 91 →
  total_flowers = 105 →
  final_roses + final_orchids = total_flowers →
  final_orchids = initial_roses + final_orchids - total_flowers + final_roses :=
by
  sorry

#check initial_orchids_count

end initial_orchids_count_l1780_178090


namespace estimated_probability_is_correct_l1780_178087

/-- Represents the result of a single trial in the traffic congestion simulation -/
structure TrialResult :=
  (days_with_congestion : Nat)
  (h_valid : days_with_congestion ≤ 3)

/-- The simulation data -/
def simulation_data : Finset TrialResult := sorry

/-- The total number of trials in the simulation -/
def total_trials : Nat := 20

/-- The number of trials with exactly two days of congestion -/
def trials_with_two_congestion : Nat := 5

/-- The estimated probability of having exactly two days of congestion in three days -/
def estimated_probability : ℚ := trials_with_two_congestion / total_trials

theorem estimated_probability_is_correct :
  estimated_probability = 1/4 := by sorry

end estimated_probability_is_correct_l1780_178087


namespace base5_44_equals_binary_10111_l1780_178022

-- Define a function to convert a base-5 number to decimal
def base5ToDecimal (n : ℕ) : ℕ :=
  (n / 10) * 5 + (n % 10)

-- Define a function to convert a decimal number to binary
def decimalToBinary (n : ℕ) : List ℕ :=
  if n = 0 then [0]
  else
    let rec toBinary (m : ℕ) (acc : List ℕ) : List ℕ :=
      if m = 0 then acc
      else toBinary (m / 2) ((m % 2) :: acc)
    toBinary n []

-- Theorem stating that (44)₅ in base-5 is equal to (10111)₂ in binary
theorem base5_44_equals_binary_10111 :
  decimalToBinary (base5ToDecimal 44) = [1, 0, 1, 1, 1] := by
  sorry

end base5_44_equals_binary_10111_l1780_178022


namespace elevator_problem_l1780_178088

theorem elevator_problem (initial_people : ℕ) (remaining_people : ℕ) 
  (h1 : initial_people = 18) 
  (h2 : remaining_people = 11) : 
  initial_people - remaining_people = 7 := by
  sorry

end elevator_problem_l1780_178088


namespace total_books_count_l1780_178099

theorem total_books_count (T : ℕ) : 
  (T = (1/4 : ℚ) * T + 10 + 
       (3/5 : ℚ) * (T - ((1/4 : ℚ) * T + 10)) - 5 + 
       12 + 13) → 
  T = 80 := by
  sorry

end total_books_count_l1780_178099


namespace product_sum_theorem_l1780_178068

theorem product_sum_theorem (a b c : ℝ) 
  (h1 : a^2 + b^2 + c^2 = 138) 
  (h2 : a + b + c = 20) : 
  a*b + b*c + c*a = 131 := by
sorry

end product_sum_theorem_l1780_178068


namespace complement_union_M_N_l1780_178036

open Set

def U : Set ℕ := {1,2,3,4,5,6,7,8}
def M : Set ℕ := {1,3,5,7}
def N : Set ℕ := {5,6,7}

theorem complement_union_M_N : 
  (U \ (M ∪ N)) = {2,4,8} := by sorry

end complement_union_M_N_l1780_178036


namespace sophie_journey_l1780_178010

/-- Proves that given the specified journey conditions, the walking distance is 5.1 km -/
theorem sophie_journey (d : ℝ) 
  (h1 : (2/3 * d) / 20 + (1/3 * d) / 4 = 1.8) : 
  1/3 * d = 5.1 := by
  sorry

end sophie_journey_l1780_178010


namespace stationery_cost_theorem_l1780_178011

/-- The cost of pencils and notebooks given specific quantities -/
structure StationeryCost where
  pencil_cost : ℝ
  notebook_cost : ℝ

/-- The conditions from the problem -/
def problem_conditions (c : StationeryCost) : Prop :=
  4 * c.pencil_cost + 5 * c.notebook_cost = 3.35 ∧
  6 * c.pencil_cost + 4 * c.notebook_cost = 3.16

/-- The theorem to prove -/
theorem stationery_cost_theorem (c : StationeryCost) :
  problem_conditions c →
  20 * c.pencil_cost + 13 * c.notebook_cost = 10.29 := by
  sorry

end stationery_cost_theorem_l1780_178011


namespace machine_production_theorem_l1780_178051

def pair_productions : List ℕ := [35, 39, 40, 49, 44, 46, 30, 41, 32, 36]

def individual_productions : List ℕ := [13, 17, 19, 22, 27]

def valid_pair_production (p : ℕ × ℕ) : Prop :=
  p.1 ∈ individual_productions ∧ p.2 ∈ individual_productions ∧ p.1 + p.2 ∈ pair_productions

theorem machine_production_theorem :
  (∀ p ∈ pair_productions, ∃ (x y : ℕ), x ∈ individual_productions ∧ y ∈ individual_productions ∧ x + y = p) ∧
  (∀ (x y : ℕ), x ∈ individual_productions → y ∈ individual_productions → x ≠ y → x + y ∈ pair_productions) ∧
  (individual_productions.length = 5) :=
sorry

end machine_production_theorem_l1780_178051


namespace trailing_zeros_2006_factorial_trailing_zeros_2006_factorial_is_500_l1780_178055

theorem trailing_zeros_2006_factorial : Nat → Nat
| n => (n / 5) + (n / 25) + (n / 125) + (n / 625) + (n / 3125)

theorem trailing_zeros_2006_factorial_is_500 :
  trailing_zeros_2006_factorial 2006 = 500 := by
  sorry

end trailing_zeros_2006_factorial_trailing_zeros_2006_factorial_is_500_l1780_178055


namespace sum_of_powers_of_i_l1780_178089

-- Define i as a complex number with i² = -1
def i : ℂ := Complex.I

-- State the theorem
theorem sum_of_powers_of_i :
  (Finset.range 603).sum (fun k => i ^ k) = i - 1 := by
  sorry

-- Note: The proof is omitted as per your instructions.

end sum_of_powers_of_i_l1780_178089


namespace luke_money_lasted_nine_weeks_l1780_178043

/-- The number of weeks Luke's money lasted given his earnings and spending -/
def weeks_money_lasted (mowing_earnings weed_eating_earnings weekly_spending : ℕ) : ℕ :=
  (mowing_earnings + weed_eating_earnings) / weekly_spending

/-- Theorem: Given Luke's earnings and spending, his money lasted 9 weeks -/
theorem luke_money_lasted_nine_weeks :
  weeks_money_lasted 9 18 3 = 9 := by
  sorry

end luke_money_lasted_nine_weeks_l1780_178043


namespace alice_prob_after_three_turns_l1780_178021

/-- Represents the player who has the ball -/
inductive Player
  | Alice
  | Bob

/-- The game state after each turn -/
def GameState := List Player

/-- The probability of Alice keeping the ball on her turn -/
def aliceKeepProb : ℚ := 2/3

/-- The probability of Bob keeping the ball on his turn -/
def bobKeepProb : ℚ := 2/3

/-- The initial game state with Alice having the ball -/
def initialState : GameState := [Player.Alice]

/-- Calculates the probability of a specific game state after three turns -/
def stateProb (state : GameState) : ℚ :=
  match state with
  | [Player.Alice, Player.Alice, Player.Alice, Player.Alice] => aliceKeepProb * aliceKeepProb * aliceKeepProb
  | [Player.Alice, Player.Alice, Player.Bob, Player.Alice] => aliceKeepProb * (1 - aliceKeepProb) * (1 - bobKeepProb)
  | [Player.Alice, Player.Bob, Player.Alice, Player.Alice] => (1 - aliceKeepProb) * (1 - bobKeepProb) * aliceKeepProb
  | [Player.Alice, Player.Bob, Player.Bob, Player.Alice] => (1 - aliceKeepProb) * bobKeepProb * (1 - bobKeepProb)
  | _ => 0

/-- All possible game states after three turns where Alice ends up with the ball -/
def validStates : List GameState := [
  [Player.Alice, Player.Alice, Player.Alice, Player.Alice],
  [Player.Alice, Player.Alice, Player.Bob, Player.Alice],
  [Player.Alice, Player.Bob, Player.Alice, Player.Alice],
  [Player.Alice, Player.Bob, Player.Bob, Player.Alice]
]

/-- The main theorem: probability of Alice having the ball after three turns is 14/27 -/
theorem alice_prob_after_three_turns :
  (validStates.map stateProb).sum = 14/27 := by
  sorry


end alice_prob_after_three_turns_l1780_178021


namespace function_range_theorem_l1780_178065

theorem function_range_theorem (a : ℝ) :
  (∃ x : ℝ, (|2*x + 1| + |2*x - 3| < |a - 1|)) →
  (a < -3 ∨ a > 5) :=
by sorry

end function_range_theorem_l1780_178065


namespace johns_investment_l1780_178081

theorem johns_investment (total_interest rate1 rate_difference investment1 : ℝ) 
  (h1 : total_interest = 1282)
  (h2 : rate1 = 0.11)
  (h3 : rate_difference = 0.015)
  (h4 : investment1 = 4000) : 
  ∃ investment2 : ℝ, 
    investment2 = 6736 ∧ 
    total_interest = investment1 * rate1 + investment2 * (rate1 + rate_difference) :=
by
  sorry

end johns_investment_l1780_178081


namespace consecutive_three_digit_prime_factors_l1780_178085

theorem consecutive_three_digit_prime_factors :
  ∀ n : ℕ, 
    100 ≤ n ∧ n + 9 ≤ 999 →
    ∃ (S : Finset ℕ),
      (∀ p ∈ S, Nat.Prime p) ∧
      (Finset.card S ≤ 23) ∧
      (∀ k : ℕ, n ≤ k ∧ k ≤ n + 9 → ∀ p : ℕ, Nat.Prime p → p ∣ k → p ∈ S) :=
by sorry

end consecutive_three_digit_prime_factors_l1780_178085


namespace right_triangle_c_squared_l1780_178071

theorem right_triangle_c_squared (a b c : ℝ) : 
  a = 9 → b = 12 → (c^2 = a^2 + b^2 ∨ b^2 = a^2 + c^2) → c^2 = 225 ∨ c^2 = 63 := by
  sorry

end right_triangle_c_squared_l1780_178071


namespace drums_per_day_l1780_178096

/-- Given that 90 drums are filled in 5 days, prove that 18 drums are filled per day -/
theorem drums_per_day (total_drums : ℕ) (total_days : ℕ) (h1 : total_drums = 90) (h2 : total_days = 5) :
  total_drums / total_days = 18 := by
  sorry

end drums_per_day_l1780_178096


namespace smallest_positive_omega_l1780_178094

/-- Given a function f(x) = sin(ωx + π/3), if f(x - π/3) = -f(x) for all x, 
    then the smallest positive value of ω is 3. -/
theorem smallest_positive_omega (ω : ℝ) : 
  (∀ x, Real.sin (ω * (x - π/3) + π/3) = -Real.sin (ω * x + π/3)) → 
  (∀ δ > 0, δ < ω → δ ≤ 3) ∧ ω = 3 := by
  sorry

end smallest_positive_omega_l1780_178094


namespace arithmetic_sequence_properties_l1780_178012

-- Define the arithmetic sequence a_n
def a (n : ℕ) : ℚ := sorry

-- Define the sequence b_n
def b (n : ℕ) : ℚ := 1 / (n.cast * a n)

-- Define the sum of the first n terms of b_n
def S (n : ℕ) : ℚ := sorry

theorem arithmetic_sequence_properties :
  (∀ n : ℕ, a (n + 1) - a n = a 2 - a 1) ∧  -- arithmetic sequence property
  a 7 = 4 ∧                                 -- given condition
  a 19 = 2 * a 9 ∧                          -- given condition
  (∀ n : ℕ, a n = (1 + n.cast) / 2) ∧       -- general formula for a_n
  (∀ n : ℕ, S n = (2 * n.cast) / (n.cast + 1)) -- sum of first n terms of b_n
  := by sorry

end arithmetic_sequence_properties_l1780_178012


namespace james_february_cost_l1780_178074

/-- Calculates the total cost for a streaming service based on the given parameters. -/
def streaming_cost (base_cost : ℝ) (free_hours : ℕ) (extra_hour_cost : ℝ) 
                   (movie_rental_cost : ℝ) (hours_streamed : ℕ) (movies_rented : ℕ) : ℝ :=
  let extra_hours := max (hours_streamed - free_hours) 0
  base_cost + (extra_hours : ℝ) * extra_hour_cost + (movies_rented : ℝ) * movie_rental_cost

/-- Theorem stating that James' streaming cost in February is $24. -/
theorem james_february_cost :
  streaming_cost 15 50 2 0.1 53 30 = 24 := by
  sorry

end james_february_cost_l1780_178074


namespace hotel_problem_l1780_178001

theorem hotel_problem (n : ℕ) : n = 9 :=
  let total_spent : ℚ := 29.25
  let standard_meal_cost : ℚ := 3
  let standard_meal_count : ℕ := 8
  let extra_cost : ℚ := 2

  have h1 : n > 0 := by sorry
  have h2 : (n : ℚ) * (total_spent / n) = total_spent := by sorry
  have h3 : standard_meal_count * standard_meal_cost + (total_spent / n + extra_cost) = total_spent := by sorry

  sorry

end hotel_problem_l1780_178001


namespace tile_relationship_l1780_178000

theorem tile_relationship (r : ℕ) (w : ℕ) : 
  (3 ≤ r ∧ r ≤ 7) → 
  (
    (r = 3 ∧ w = 6) ∨
    (r = 4 ∧ w = 8) ∨
    (r = 5 ∧ w = 10) ∨
    (r = 6 ∧ w = 12) ∨
    (r = 7 ∧ w = 14)
  ) →
  w = 2 * r :=
by sorry

end tile_relationship_l1780_178000


namespace sum_of_integers_l1780_178004

theorem sum_of_integers : (-9) + 18 + 2 + (-1) = 10 := by
  sorry

end sum_of_integers_l1780_178004


namespace range_of_a_l1780_178084

theorem range_of_a (a : ℝ) : 
  (∀ x > 0, Real.log x - a * x ≤ 2 * a^2 - 3) → a ≥ 1 := by
  sorry

end range_of_a_l1780_178084


namespace ladder_movement_l1780_178098

theorem ladder_movement (ladder_length : ℝ) (initial_distance : ℝ) (slide_down : ℝ) : 
  ladder_length = 25 →
  initial_distance = 7 →
  slide_down = 4 →
  ∃ (final_distance : ℝ),
    final_distance > initial_distance ∧
    final_distance ^ 2 + (ladder_length - slide_down) ^ 2 = ladder_length ^ 2 ∧
    final_distance - initial_distance = 8 :=
by sorry

end ladder_movement_l1780_178098


namespace valentine_biscuits_l1780_178097

theorem valentine_biscuits (total_biscuits : ℕ) (num_dogs : ℕ) (biscuits_per_dog : ℕ) :
  total_biscuits = 6 →
  num_dogs = 2 →
  total_biscuits = num_dogs * biscuits_per_dog →
  biscuits_per_dog = 3 := by
  sorry

end valentine_biscuits_l1780_178097


namespace pete_walked_3350_miles_l1780_178032

/-- Represents a pedometer with a maximum reading before reset --/
structure Pedometer :=
  (max_reading : ℕ)

/-- Represents Pete's walking data for a year --/
structure YearlyWalkingData :=
  (pedometer : Pedometer)
  (resets : ℕ)
  (final_reading : ℕ)
  (steps_per_mile : ℕ)

/-- Calculates the total miles walked based on the yearly walking data --/
def total_miles_walked (data : YearlyWalkingData) : ℕ :=
  ((data.resets * (data.pedometer.max_reading + 1) + data.final_reading) / data.steps_per_mile)

/-- Theorem stating that Pete walked 3350 miles given the problem conditions --/
theorem pete_walked_3350_miles :
  let petes_pedometer : Pedometer := ⟨99999⟩
  let petes_data : YearlyWalkingData := ⟨petes_pedometer, 50, 25000, 1500⟩
  total_miles_walked petes_data = 3350 := by
  sorry


end pete_walked_3350_miles_l1780_178032
