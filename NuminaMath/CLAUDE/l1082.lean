import Mathlib

namespace move_fulcrum_towards_wood_l1082_108251

/-- Represents the material of a sphere -/
inductive Material
| CastIron
| Wood

/-- Properties of a sphere -/
structure Sphere where
  material : Material
  density : ℝ
  volume : ℝ
  mass : ℝ

/-- The setup of the balance problem -/
structure BalanceSetup where
  airDensity : ℝ
  castIronSphere : Sphere
  woodenSphere : Sphere
  fulcrumPosition : ℝ  -- 0 means middle, negative means towards cast iron, positive means towards wood

/-- Conditions for the balance problem -/
def validSetup (setup : BalanceSetup) : Prop :=
  setup.castIronSphere.material = Material.CastIron ∧
  setup.woodenSphere.material = Material.Wood ∧
  setup.castIronSphere.density > setup.woodenSphere.density ∧
  setup.castIronSphere.density > setup.airDensity ∧
  setup.woodenSphere.density > setup.airDensity ∧
  setup.castIronSphere.volume < setup.woodenSphere.volume ∧
  setup.castIronSphere.mass < setup.woodenSphere.mass

/-- The balance condition when the fulcrum is in the middle -/
def balanceCondition (setup : BalanceSetup) : Prop :=
  (setup.castIronSphere.density - setup.airDensity) * setup.castIronSphere.volume =
  (setup.woodenSphere.density - setup.airDensity) * setup.woodenSphere.volume

/-- Theorem stating that the fulcrum needs to be moved towards the wooden sphere -/
theorem move_fulcrum_towards_wood (setup : BalanceSetup) :
  validSetup setup → balanceCondition setup → setup.fulcrumPosition > 0 := by
  sorry

end move_fulcrum_towards_wood_l1082_108251


namespace trajectory_of_P_l1082_108275

-- Define the line l
def line_l (θ : ℝ) (x y : ℝ) : Prop := x * Real.cos θ + y * Real.sin θ = 1

-- Define the perpendicularity condition
def perpendicular_to_l (x y : ℝ) : Prop := ∃ θ, line_l θ x y

-- Define the origin
def origin : ℝ × ℝ := (0, 0)

-- Theorem statement
theorem trajectory_of_P : ∀ x y : ℝ, perpendicular_to_l x y → x^2 + y^2 = 1 := by
  sorry

end trajectory_of_P_l1082_108275


namespace union_of_M_and_N_l1082_108246

def M : Set ℤ := {-1, 0, 2, 4}
def N : Set ℤ := {0, 2, 3, 4}

theorem union_of_M_and_N : M ∪ N = {-1, 0, 2, 3, 4} := by sorry

end union_of_M_and_N_l1082_108246


namespace circle_intersection_length_l1082_108252

-- Define the right triangle ABC
structure RightTriangle where
  A : Real
  B : Real
  C : Real
  angle_A : A = 30 * Real.pi / 180
  hypotenuse : C = 2 * A
  right_angle : B = 90 * Real.pi / 180

-- Define the circle and point K
structure CircleAndPoint (t : RightTriangle) where
  K : Real
  on_hypotenuse : K ≤ t.C ∧ K ≥ 0
  diameter : t.A = 2

-- Theorem statement
theorem circle_intersection_length (t : RightTriangle) (c : CircleAndPoint t) :
  let CK := Real.sqrt (t.A * (t.C - c.K))
  CK = 1 := by sorry

end circle_intersection_length_l1082_108252


namespace initial_rope_length_correct_l1082_108212

/-- The initial length of rope before decorating trees -/
def initial_rope_length : ℝ := 8.9

/-- The length of string used to decorate one tree -/
def string_per_tree : ℝ := 0.84

/-- The number of trees decorated -/
def num_trees : ℕ := 10

/-- The length of rope remaining after decorating trees -/
def remaining_rope : ℝ := 0.5

/-- Theorem stating that the initial rope length is correct -/
theorem initial_rope_length_correct :
  initial_rope_length = string_per_tree * num_trees + remaining_rope :=
by sorry

end initial_rope_length_correct_l1082_108212


namespace peter_age_approx_l1082_108254

def cindy_age : ℕ := 5

def jan_age : ℕ := cindy_age + 2

def marcia_age : ℕ := 2 * jan_age

def greg_age : ℕ := marcia_age + 2

def bobby_age : ℕ := (3 * greg_age) / 2

noncomputable def peter_age : ℝ := 2 * Real.sqrt (bobby_age : ℝ)

theorem peter_age_approx : 
  ∀ ε > 0, |peter_age - 10| < ε := by sorry

end peter_age_approx_l1082_108254


namespace smallest_difference_l1082_108259

def digits : List Nat := [2, 4, 5, 6, 9]

def is_valid_arrangement (a b : Nat) : Prop :=
  ∃ (x y z u v : Nat),
    x ∈ digits ∧ y ∈ digits ∧ z ∈ digits ∧ u ∈ digits ∧ v ∈ digits ∧
    x ≠ y ∧ x ≠ z ∧ x ≠ u ∧ x ≠ v ∧
    y ≠ z ∧ y ≠ u ∧ y ≠ v ∧
    z ≠ u ∧ z ≠ v ∧
    u ≠ v ∧
    a = 100 * x + 10 * y + z ∧
    b = 10 * u + v

theorem smallest_difference :
  ∀ a b : Nat,
    is_valid_arrangement a b →
    a > b →
    a - b ≥ 149 :=
by sorry

end smallest_difference_l1082_108259


namespace trapezoid_determines_unique_plane_l1082_108295

/-- A point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- A trapezoid in 3D space -/
structure Trapezoid where
  p1 : Point3D
  p2 : Point3D
  p3 : Point3D
  p4 : Point3D
  is_trapezoid : ∃ (a b : ℝ), a ≠ b ∧
    (p2.x - p1.x) * (p4.y - p3.y) = (p2.y - p1.y) * (p4.x - p3.x) ∧
    (p3.x - p2.x) * (p1.y - p4.y) = (p3.y - p2.y) * (p1.x - p4.x)

/-- A plane in 3D space -/
structure Plane where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

/-- Definition of a point lying on a plane -/
def Point3D.on_plane (p : Point3D) (plane : Plane) : Prop :=
  plane.a * p.x + plane.b * p.y + plane.c * p.z + plane.d = 0

/-- Theorem: A trapezoid determines a unique plane -/
theorem trapezoid_determines_unique_plane (t : Trapezoid) :
  ∃! (plane : Plane), t.p1.on_plane plane ∧ t.p2.on_plane plane ∧
                      t.p3.on_plane plane ∧ t.p4.on_plane plane :=
sorry

end trapezoid_determines_unique_plane_l1082_108295


namespace investment_dividend_l1082_108221

/-- Calculates the total dividend received from an investment in shares -/
theorem investment_dividend (investment : ℝ) (share_value : ℝ) (premium_rate : ℝ) (dividend_rate : ℝ) :
  investment = 14400 →
  share_value = 100 →
  premium_rate = 0.20 →
  dividend_rate = 0.06 →
  let share_cost := share_value * (1 + premium_rate)
  let num_shares := investment / share_cost
  let dividend_per_share := share_value * dividend_rate
  dividend_per_share * num_shares = 720 := by
  sorry

end investment_dividend_l1082_108221


namespace b_is_geometric_sequence_l1082_108228

-- Define the geometric sequence a_n
def a (n : ℕ) (a₁ q : ℝ) : ℝ := a₁ * q^(n - 1)

-- Define the sequence b_n
def b (n : ℕ) (a₁ q : ℝ) : ℝ := a (3*n - 2) a₁ q + a (3*n - 1) a₁ q + a (3*n) a₁ q

-- Theorem statement
theorem b_is_geometric_sequence (a₁ q : ℝ) (hq : q ≠ 1) :
  ∀ n : ℕ, b (n + 1) a₁ q = (b n a₁ q) * q^3 :=
sorry

end b_is_geometric_sequence_l1082_108228


namespace sum_of_reciprocal_extrema_l1082_108267

theorem sum_of_reciprocal_extrema (x y : ℝ) : 
  (4 * x^2 - 5 * x * y + 4 * y^2 = 5) → 
  let S := x^2 + y^2
  ∃ (S_max S_min : ℝ), 
    (∀ (x' y' : ℝ), (4 * x'^2 - 5 * x' * y' + 4 * y'^2 = 5) → x'^2 + y'^2 ≤ S_max) ∧
    (∀ (x' y' : ℝ), (4 * x'^2 - 5 * x' * y' + 4 * y'^2 = 5) → S_min ≤ x'^2 + y'^2) ∧
    (1 / S_max + 1 / S_min = 8 / 5) :=
by sorry

end sum_of_reciprocal_extrema_l1082_108267


namespace k_value_in_set_union_l1082_108201

theorem k_value_in_set_union (A B : Set ℕ) (k : ℕ) :
  A = {1, 2, k} →
  B = {1, 2, 3, 5} →
  A ∪ B = {1, 2, 3, 5} →
  k = 3 ∨ k = 5 := by
  sorry

end k_value_in_set_union_l1082_108201


namespace intersection_complement_equality_l1082_108203

-- Define the universe set U as the set of real numbers
def U : Set ℝ := Set.univ

-- Define set M
def M : Set ℝ := {x | x ≤ 2}

-- Define set N
def N : Set ℝ := {x | -1 ≤ x ∧ x ≤ 1}

-- State the theorem
theorem intersection_complement_equality :
  M ∩ (U \ N) = {x : ℝ | x < -1 ∨ (1 < x ∧ x ≤ 2)} := by sorry

end intersection_complement_equality_l1082_108203


namespace square_eq_four_solutions_l1082_108241

theorem square_eq_four_solutions (x : ℝ) : (x - 1)^2 = 4 ↔ x = 3 ∨ x = -1 := by
  sorry

end square_eq_four_solutions_l1082_108241


namespace class_size_l1082_108237

/-- The number of students who borrowed at least 3 books -/
def R : ℕ := 16

/-- The total number of students in the class -/
def S : ℕ := 42

theorem class_size :
  (∃ (R : ℕ),
    (0 * 2 + 1 * 12 + 2 * 12 + 3 * R) / S = 2 ∧
    S = 2 + 12 + 12 + R) →
  S = 42 := by sorry

end class_size_l1082_108237


namespace john_house_planks_l1082_108208

theorem john_house_planks :
  ∀ (total_nails nails_per_plank additional_nails : ℕ),
    total_nails = 11 →
    nails_per_plank = 3 →
    additional_nails = 8 →
    ∃ (num_planks : ℕ),
      num_planks * nails_per_plank + additional_nails = total_nails ∧
      num_planks = 1 := by
sorry

end john_house_planks_l1082_108208


namespace fixed_point_of_line_l1082_108231

/-- The fixed point through which the line (2k+1)x+(k-1)y+(7-k)=0 passes for all real k -/
theorem fixed_point_of_line (k : ℝ) : 
  ∃! p : ℝ × ℝ, ∀ k : ℝ, (2*k + 1) * p.1 + (k - 1) * p.2 + (7 - k) = 0 :=
by sorry

end fixed_point_of_line_l1082_108231


namespace isosceles_triangle_perimeter_l1082_108276

theorem isosceles_triangle_perimeter (m : ℝ) :
  (3 : ℝ) ^ 2 - (m + 1) * 3 + 2 * m = 0 →
  ∃ (a b : ℝ),
    a ^ 2 - (m + 1) * a + 2 * m = 0 ∧
    b ^ 2 - (m + 1) * b + 2 * m = 0 ∧
    ((a = b ∧ a + a + b = 10) ∨ (a ≠ b ∧ a + a + b = 11)) :=
by sorry

end isosceles_triangle_perimeter_l1082_108276


namespace pancake_fundraiser_l1082_108282

/-- The civic league's pancake breakfast fundraiser --/
theorem pancake_fundraiser 
  (pancake_price : ℝ) 
  (bacon_price : ℝ) 
  (pancake_stacks_sold : ℕ) 
  (bacon_slices_sold : ℕ) 
  (h1 : pancake_price = 4)
  (h2 : bacon_price = 2)
  (h3 : pancake_stacks_sold = 60)
  (h4 : bacon_slices_sold = 90) :
  pancake_price * pancake_stacks_sold + bacon_price * bacon_slices_sold = 420 :=
by sorry

end pancake_fundraiser_l1082_108282


namespace greatest_abdba_divisible_by_13_l1082_108211

def is_valid_abdba (n : ℕ) : Prop :=
  100000 ≤ n ∧ n < 1000000 ∧
  ∃ (a b d : ℕ),
    a < 10 ∧ b < 10 ∧ d < 10 ∧
    a ≠ b ∧ b ≠ d ∧ a ≠ d ∧
    n = a * 10000 + b * 1000 + d * 100 + b * 10 + a

theorem greatest_abdba_divisible_by_13 :
  ∀ n : ℕ, is_valid_abdba n → n % 13 = 0 → n ≤ 96769 :=
by sorry

end greatest_abdba_divisible_by_13_l1082_108211


namespace eight_power_32_sum_equals_2_power_99_l1082_108298

theorem eight_power_32_sum_equals_2_power_99 :
  (8:ℕ)^32 + (8:ℕ)^32 + (8:ℕ)^32 + (8:ℕ)^32 + 
  (8:ℕ)^32 + (8:ℕ)^32 + (8:ℕ)^32 + (8:ℕ)^32 = (2:ℕ)^99 :=
by sorry

end eight_power_32_sum_equals_2_power_99_l1082_108298


namespace circle_line_distance_range_l1082_108219

theorem circle_line_distance_range (b : ℝ) : 
  (∃! (p q : ℝ × ℝ), 
    p.1^2 + p.2^2 = 4 ∧ 
    q.1^2 + q.2^2 = 4 ∧ 
    (p ≠ q) ∧
    (|p.2 - p.1 - b| / Real.sqrt 2 = 1) ∧
    (|q.2 - q.1 - b| / Real.sqrt 2 = 1)) →
  (b < -Real.sqrt 2 ∧ b > -3 * Real.sqrt 2) ∨ 
  (b > Real.sqrt 2 ∧ b < 3 * Real.sqrt 2) :=
sorry

end circle_line_distance_range_l1082_108219


namespace kishore_savings_theorem_l1082_108229

/-- Represents Mr. Kishore's financial situation --/
structure KishoreFinances where
  rent : ℕ
  milk : ℕ
  groceries : ℕ
  education : ℕ
  petrol : ℕ
  miscellaneous : ℕ
  savings_percentage : ℚ

/-- Calculates the total expenses --/
def total_expenses (k : KishoreFinances) : ℕ :=
  k.rent + k.milk + k.groceries + k.education + k.petrol + k.miscellaneous

/-- Calculates the monthly salary --/
def monthly_salary (k : KishoreFinances) : ℚ :=
  (total_expenses k : ℚ) / (1 - k.savings_percentage)

/-- Calculates the savings amount --/
def savings_amount (k : KishoreFinances) : ℚ :=
  k.savings_percentage * monthly_salary k

/-- Theorem: Mr. Kishore's savings are approximately 2683.33 Rs. --/
theorem kishore_savings_theorem (k : KishoreFinances) 
  (h1 : k.rent = 5000)
  (h2 : k.milk = 1500)
  (h3 : k.groceries = 4500)
  (h4 : k.education = 2500)
  (h5 : k.petrol = 2000)
  (h6 : k.miscellaneous = 5650)
  (h7 : k.savings_percentage = 1/10) :
  ∃ (ε : ℚ), abs (savings_amount k - 2683.33) < ε ∧ ε < 1/100 := by
  sorry

end kishore_savings_theorem_l1082_108229


namespace base_ten_proof_l1082_108214

/-- Converts a number from base b to decimal --/
def to_decimal (n : ℕ) (b : ℕ) : ℕ := sorry

/-- Converts a number from decimal to base b --/
def from_decimal (n : ℕ) (b : ℕ) : ℕ := sorry

/-- Checks if the equation 162_b + 235_b = 407_b holds for a given base b --/
def equation_holds (b : ℕ) : Prop :=
  to_decimal 162 b + to_decimal 235 b = to_decimal 407 b

theorem base_ten_proof :
  ∃! b : ℕ, b > 1 ∧ equation_holds b ∧ b = 10 :=
sorry

end base_ten_proof_l1082_108214


namespace order_of_abc_l1082_108233

theorem order_of_abc (a b c : ℝ) (ha : a = 2^(4/3)) (hb : b = 3^(2/3)) (hc : c = 25^(1/3)) :
  b < a ∧ a < c := by
  sorry

end order_of_abc_l1082_108233


namespace hyperbola_sum_l1082_108220

/-- Represents a hyperbola with center (h, k), focus (h, f), and vertex (h, v) -/
structure Hyperbola where
  h : ℝ
  k : ℝ
  f : ℝ
  v : ℝ

/-- The equation of the hyperbola is (y - k)²/a² - (x - h)²/b² = 1 -/
def hyperbola_equation (hyp : Hyperbola) (a b : ℝ) : Prop :=
  ∀ x y : ℝ, (y - hyp.k)^2 / a^2 - (x - hyp.h)^2 / b^2 = 1

/-- The theorem to be proved -/
theorem hyperbola_sum (hyp : Hyperbola) (a b : ℝ) :
  hyp.h = 1 ∧ hyp.k = 1 ∧ hyp.f = 7 ∧ hyp.v = -2 ∧ 
  hyperbola_equation hyp a b →
  hyp.h + hyp.k + a + b = 5 + 3 * Real.sqrt 3 := by
  sorry

end hyperbola_sum_l1082_108220


namespace perfect_power_sequence_exists_l1082_108289

theorem perfect_power_sequence_exists : ∃ a : ℕ+, ∀ k ∈ Set.Icc 2015 2558, 
  ∃ (b : ℕ+) (n : ℕ), n ≥ 2 ∧ (k : ℝ) * a.val = b.val ^ n :=
sorry

end perfect_power_sequence_exists_l1082_108289


namespace graph_passes_through_second_and_fourth_quadrants_l1082_108288

-- Define the function
def f (x : ℝ) : ℝ := -3 * x

-- State the theorem
theorem graph_passes_through_second_and_fourth_quadrants :
  (∃ x y, x < 0 ∧ y > 0 ∧ f x = y) ∧ 
  (∃ x y, x > 0 ∧ y < 0 ∧ f x = y) :=
by sorry

end graph_passes_through_second_and_fourth_quadrants_l1082_108288


namespace jackson_money_l1082_108265

theorem jackson_money (williams_money : ℝ) (h1 : williams_money > 0) 
  (h2 : williams_money + 5 * williams_money = 150) : 
  5 * williams_money = 125 := by
sorry

end jackson_money_l1082_108265


namespace polynomial_relationship_l1082_108230

def x : Fin 5 → ℕ
  | 0 => 1
  | 1 => 2
  | 2 => 3
  | 3 => 4
  | 4 => 5

def y : Fin 5 → ℕ
  | 0 => 1
  | 1 => 4
  | 2 => 9
  | 3 => 16
  | 4 => 25

theorem polynomial_relationship : ∀ i : Fin 5, y i = (x i) ^ 2 := by
  sorry

end polynomial_relationship_l1082_108230


namespace smoking_lung_cancer_study_l1082_108206

theorem smoking_lung_cancer_study (confidence : Real) 
  (h1 : confidence = 0.99) : 
  let error_probability := 1 - confidence
  error_probability ≤ 0.01 := by
sorry

end smoking_lung_cancer_study_l1082_108206


namespace arithmetic_sequence_sum_problem_solution_l1082_108296

-- Define the arithmetic sequence and its sum
def arithmetic_sequence (a₁ d : ℤ) (n : ℕ) : ℤ := a₁ + d * (n - 1)
def sum_arithmetic_sequence (a₁ d : ℤ) (n : ℕ) : ℤ := n * a₁ + n * (n - 1) * d / 2

theorem arithmetic_sequence_sum (a₁ : ℤ) :
  ∃ d : ℤ, 
    (sum_arithmetic_sequence a₁ d 6 - 2 * sum_arithmetic_sequence a₁ d 3 = 18) → 
    (sum_arithmetic_sequence a₁ d 2017 = 2017) := by
  sorry

-- Main theorem
theorem problem_solution : 
  ∃ d : ℤ, 
    (sum_arithmetic_sequence (-2015) d 6 - 2 * sum_arithmetic_sequence (-2015) d 3 = 18) → 
    (sum_arithmetic_sequence (-2015) d 2017 = 2017) := by
  sorry

end arithmetic_sequence_sum_problem_solution_l1082_108296


namespace socks_needed_to_triple_wardrobe_l1082_108255

/-- Represents the number of items in Jonas' wardrobe -/
structure Wardrobe where
  socks : ℕ
  shoes : ℕ
  pants : ℕ
  tshirts : ℕ
  hats : ℕ
  jackets : ℕ

/-- Calculates the total number of individual items in the wardrobe -/
def totalItems (w : Wardrobe) : ℕ :=
  w.socks * 2 + w.shoes * 2 + w.pants + w.tshirts + w.hats + w.jackets

/-- Jonas' current wardrobe -/
def jonasWardrobe : Wardrobe :=
  { socks := 20
    shoes := 5
    pants := 10
    tshirts := 10
    hats := 6
    jackets := 4 }

/-- Theorem: Jonas needs to buy 80 pairs of socks to triple his wardrobe -/
theorem socks_needed_to_triple_wardrobe :
  let current := totalItems jonasWardrobe
  let target := current * 3
  let difference := target - current
  difference / 2 = 80 := by sorry

end socks_needed_to_triple_wardrobe_l1082_108255


namespace necessary_but_not_sufficient_l1082_108250

theorem necessary_but_not_sufficient : 
  (∀ x : ℝ, x > 3 → x > 1) ∧ 
  (∃ x : ℝ, x > 1 ∧ ¬(x > 3)) := by
  sorry

end necessary_but_not_sufficient_l1082_108250


namespace parabola_focus_l1082_108215

/-- The focus of the parabola x^2 = 8y has coordinates (0, 2) -/
theorem parabola_focus (x y : ℝ) :
  (x^2 = 8*y) → (∃ p : ℝ, p = 2 ∧ (0, p) = (0, 2)) :=
by sorry

end parabola_focus_l1082_108215


namespace ramanujan_number_l1082_108239

def hardy : ℂ := Complex.mk 7 4

theorem ramanujan_number (r : ℂ) : r * hardy = Complex.mk 60 (-18) → r = Complex.mk (174/65) (-183/65) := by
  sorry

end ramanujan_number_l1082_108239


namespace factorization_problems_l1082_108266

variable (m a b : ℝ)

theorem factorization_problems :
  (ma^2 - mb^2 = m*(a+b)*(a-b)) ∧
  ((a+b) - 2*a*(a+b) + a^2*(a+b) = (a+b)*(a-1)^2) :=
by sorry

end factorization_problems_l1082_108266


namespace homeless_families_donation_l1082_108281

theorem homeless_families_donation (total spent first_set second_set : ℝ) 
  (h1 : total = 900)
  (h2 : first_set = 325)
  (h3 : second_set = 260) :
  total - (first_set + second_set) = 315 := by
sorry

end homeless_families_donation_l1082_108281


namespace f_has_one_real_root_l1082_108209

-- Define the polynomial
def f (x : ℝ) : ℝ := (x - 4) * (x^2 + 4*x + 5)

-- Theorem statement
theorem f_has_one_real_root : ∃! x : ℝ, f x = 0 := by
  sorry

end f_has_one_real_root_l1082_108209


namespace problem_solution_l1082_108279

theorem problem_solution : 
  ∀ M : ℚ, (5 + 7 + 9) / 3 = (2005 + 2007 + 2009) / M → M = 860 := by
  sorry

end problem_solution_l1082_108279


namespace money_distribution_l1082_108216

theorem money_distribution (a b c : ℕ) 
  (total : a + b + c = 700)
  (ac_sum : a + c = 300)
  (bc_sum : b + c = 600) :
  c = 200 := by
  sorry

end money_distribution_l1082_108216


namespace wall_building_time_l1082_108253

theorem wall_building_time
  (workers_initial : ℕ)
  (days_initial : ℕ)
  (workers_new : ℕ)
  (h1 : workers_initial = 60)
  (h2 : days_initial = 3)
  (h3 : workers_new = 30)
  (h4 : workers_initial > 0)
  (h5 : workers_new > 0)
  (h6 : days_initial > 0) :
  let days_new := workers_initial * days_initial / workers_new
  days_new = 6 :=
by sorry

end wall_building_time_l1082_108253


namespace volunteers_distribution_l1082_108247

theorem volunteers_distribution (n : ℕ) (k : ℕ) : 
  n = 6 → k = 4 → (k^n - k * (k-1)^n + (k.choose 2) * (k-2)^n) = 1564 := by
  sorry

end volunteers_distribution_l1082_108247


namespace permutation_remainders_l1082_108207

theorem permutation_remainders (a : Fin 11 → Fin 11) (h : Function.Bijective a) :
  ∃ i j : Fin 11, i ≠ j ∧ (i.val + 1) * (a i).val ≡ (j.val + 1) * (a j).val [ZMOD 11] := by
  sorry

end permutation_remainders_l1082_108207


namespace min_value_theorem_l1082_108202

theorem min_value_theorem (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  (x + 1/y) * (x + 1/y - 1000) + (y + 1/x) * (y + 1/x - 1000) ≥ -500000 := by
  sorry

end min_value_theorem_l1082_108202


namespace sum_of_cyclic_relations_l1082_108280

theorem sum_of_cyclic_relations (p q r : ℝ) : 
  p ≠ q ∧ q ≠ r ∧ r ≠ p →
  q = p * (4 - p) →
  r = q * (4 - q) →
  p = r * (4 - r) →
  p + q + r = 6 ∨ p + q + r = 7 := by
sorry

end sum_of_cyclic_relations_l1082_108280


namespace athlete_heartbeats_l1082_108285

/-- Calculates the total number of heartbeats during a race --/
def total_heartbeats (heart_rate : ℕ) (race_distance : ℕ) (running_pace : ℕ) (resting_time : ℕ) : ℕ :=
  let running_time := race_distance * running_pace
  let total_time := running_time + resting_time
  total_time * heart_rate

/-- Theorem: The athlete's heart beats 29250 times during the race --/
theorem athlete_heartbeats : 
  total_heartbeats 150 30 6 15 = 29250 := by
  sorry

end athlete_heartbeats_l1082_108285


namespace smallest_self_descriptive_number_l1082_108205

/-- Represents the value of a letter in the alphabet (A=1, B=2, ..., Z=26) -/
def letter_value (c : Char) : ℕ :=
  match c with
  | 'A' => 1 | 'B' => 2 | 'C' => 3 | 'D' => 4 | 'E' => 5 | 'F' => 6 | 'G' => 7
  | 'H' => 8 | 'I' => 9 | 'J' => 10 | 'K' => 11 | 'L' => 12 | 'M' => 13 | 'N' => 14
  | 'O' => 15 | 'P' => 16 | 'Q' => 17 | 'R' => 18 | 'S' => 19 | 'T' => 20
  | 'U' => 21 | 'V' => 22 | 'W' => 23 | 'X' => 24 | 'Y' => 25 | 'Z' => 26
  | _ => 0

/-- Calculates the sum of letter values in a string -/
def string_value (s : String) : ℕ :=
  s.toList.map letter_value |>.sum

/-- Converts a number to its written-out form in French -/
def number_to_french (n : ℕ) : String :=
  match n with
  | 222 => "DEUXCENTVINGTDEUX"
  | _ => ""  -- We only need to define 222 for this problem

theorem smallest_self_descriptive_number :
  ∀ n : ℕ, n < 222 → string_value (number_to_french n) ≠ n ∧
  string_value (number_to_french 222) = 222 := by
  sorry

#eval string_value (number_to_french 222)  -- Should output 222

end smallest_self_descriptive_number_l1082_108205


namespace max_value_on_interval_l1082_108299

-- Define the function
def f (x : ℝ) : ℝ := x^4 - 2*x^2 + 5

-- State the theorem
theorem max_value_on_interval :
  ∃ (c : ℝ), c ∈ Set.Icc (-2 : ℝ) (2 : ℝ) ∧
  ∀ (x : ℝ), x ∈ Set.Icc (-2 : ℝ) (2 : ℝ) → f x ≤ f c ∧
  f c = 13 :=
sorry

end max_value_on_interval_l1082_108299


namespace jellybean_difference_l1082_108291

/-- The number of jellybeans each person has -/
structure JellybeanCount where
  tino : ℕ
  lee : ℕ
  arnold : ℕ

/-- The conditions of the jellybean problem -/
def jellybean_problem (j : JellybeanCount) : Prop :=
  j.tino > j.lee ∧
  j.arnold = j.lee / 2 ∧
  j.arnold = 5 ∧
  j.tino = 34

/-- The theorem stating the difference between Tino's and Lee's jellybean counts -/
theorem jellybean_difference (j : JellybeanCount) 
  (h : jellybean_problem j) : j.tino - j.lee = 24 := by
  sorry

end jellybean_difference_l1082_108291


namespace pigeonhole_birthday_birthday_problem_l1082_108261

theorem pigeonhole_birthday (n : ℕ) (m : ℕ) (h : n > m) :
  ∀ f : Fin n → Fin m, ∃ i j : Fin n, i ≠ j ∧ f i = f j := by
  sorry

theorem birthday_problem :
  ∀ f : Fin 367 → Fin 366, ∃ i j : Fin 367, i ≠ j ∧ f i = f j := by
  exact pigeonhole_birthday 367 366 (by norm_num)

end pigeonhole_birthday_birthday_problem_l1082_108261


namespace fish_pond_problem_l1082_108283

/-- Represents the number of fish in a pond. -/
def N : ℕ := sorry

/-- The number of fish initially tagged and released. -/
def tagged_fish : ℕ := 40

/-- The number of fish caught in the second catch. -/
def second_catch : ℕ := 40

/-- The number of tagged fish found in the second catch. -/
def tagged_in_second_catch : ℕ := 2

/-- The fraction of tagged fish in the second catch. -/
def fraction_tagged_in_catch : ℚ := tagged_in_second_catch / second_catch

/-- The fraction of tagged fish in the pond. -/
def fraction_tagged_in_pond : ℚ := tagged_fish / N

theorem fish_pond_problem :
  fraction_tagged_in_catch = fraction_tagged_in_pond →
  N = 800 :=
by sorry

end fish_pond_problem_l1082_108283


namespace hundred_to_fifty_zeros_l1082_108235

theorem hundred_to_fifty_zeros (n : ℕ) : 100^50 = 10^100 := by
  sorry

end hundred_to_fifty_zeros_l1082_108235


namespace max_value_of_f_l1082_108232

/-- Given a function f(x) = x^3 - 3ax + 2 where x = 2 is an extremum point,
    prove that the maximum value of f(x) is 18 -/
theorem max_value_of_f (a : ℝ) (f : ℝ → ℝ) (h1 : f = fun x ↦ x^3 - 3*a*x + 2) 
    (h2 : ∃ (ε : ℝ), ε > 0 ∧ ∀ x ∈ Set.Ioo (2 - ε) (2 + ε), f x ≥ f 2 ∨ f x ≤ f 2) :
  (⨆ x, f x) = 18 := by
  sorry


end max_value_of_f_l1082_108232


namespace special_function_is_identity_l1082_108272

/-- A function satisfying the given conditions -/
def special_function (f : ℝ → ℝ) : Prop :=
  f 1 = 1 ∧ ∀ x y : ℝ, f (x^2 - y^2) = (x - y) * (f x + f y)

/-- Theorem stating that any function satisfying the conditions is the identity function -/
theorem special_function_is_identity (f : ℝ → ℝ) (h : special_function f) : 
  ∀ x : ℝ, f x = x := by sorry

end special_function_is_identity_l1082_108272


namespace trig_identity_l1082_108240

theorem trig_identity (α : ℝ) (h : Real.tan α = 3) :
  2 * (Real.sin α)^2 - (Real.sin α) * (Real.cos α) + (Real.cos α)^2 = 8/5 := by
  sorry

end trig_identity_l1082_108240


namespace complex_expression_equality_l1082_108227

theorem complex_expression_equality : 
  (Real.sqrt 3 + 5) * (5 - Real.sqrt 3) - 
  (Real.sqrt 8 + 2 * Real.sqrt (1/2)) / Real.sqrt 2 + 
  Real.sqrt ((Real.sqrt 5 - 3)^2) = 22 - Real.sqrt 5 := by
  sorry

end complex_expression_equality_l1082_108227


namespace triangle_properties_l1082_108210

theorem triangle_properties (a b c : ℝ) (A B C : ℝ) :
  a + b = 10 →
  c = 2 * Real.sqrt 7 →
  c * Real.sin B = Real.sqrt 3 * b * Real.cos C →
  C = π / 3 ∧ 
  (1/2) * a * b * Real.sin C = 6 * Real.sqrt 3 :=
by sorry

end triangle_properties_l1082_108210


namespace negation_of_universal_proposition_l1082_108213

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x > 0 → x^2 - 3*x + 2 < 0) ↔ (∃ x : ℝ, x > 0 ∧ x^2 - 3*x + 2 ≥ 0) :=
by sorry

end negation_of_universal_proposition_l1082_108213


namespace t_range_l1082_108242

theorem t_range (t α β a : ℝ) :
  (t = Real.cos β ^ 3 + (α / 2) * Real.cos β) →
  (a ≤ t) →
  (t ≤ α - 5 * Real.cos β) →
  (-2/3 ≤ t ∧ t ≤ 1) := by
  sorry

end t_range_l1082_108242


namespace tan_value_for_given_sum_l1082_108260

theorem tan_value_for_given_sum (x : ℝ) 
  (h1 : Real.sin x + Real.cos x = 1/5)
  (h2 : 0 ≤ x ∧ x < π) : 
  Real.tan x = -4/3 := by
  sorry

end tan_value_for_given_sum_l1082_108260


namespace sum_of_reciprocals_l1082_108236

theorem sum_of_reciprocals (x y : ℝ) (h1 : x + y = 15) (h2 : x * y = 36) : 
  1 / x + 1 / y = 5 / 12 := by
  sorry

end sum_of_reciprocals_l1082_108236


namespace eel_fat_l1082_108277

/-- The amount of fat in ounces for each type of fish --/
structure FishFat where
  herring : ℝ
  eel : ℝ
  pike : ℝ

/-- The number of each type of fish cooked --/
def fish_count : ℝ := 40

/-- The total amount of fat served in ounces --/
def total_fat : ℝ := 3600

/-- Theorem stating the amount of fat in an eel --/
theorem eel_fat (f : FishFat) 
  (herring_fat : f.herring = 40)
  (pike_fat : f.pike = f.eel + 10)
  (total_fat_eq : fish_count * (f.herring + f.eel + f.pike) = total_fat) :
  f.eel = 20 := by
  sorry

end eel_fat_l1082_108277


namespace augmented_matrix_solution_l1082_108293

/-- Given an augmented matrix representing a system of linear equations with a known solution,
    prove that the difference between certain elements of the augmented matrix is 16. -/
theorem augmented_matrix_solution (c₁ c₂ : ℝ) : 
  (∃ (x y : ℝ), x = 3 ∧ y = 5 ∧ 
   2 * x + 3 * y = c₁ ∧
   y = c₂) →
  c₁ - c₂ = 16 := by
sorry

end augmented_matrix_solution_l1082_108293


namespace place_value_ratio_l1082_108218

-- Define the number
def number : ℝ := 58624.0791

-- Define the place value of 6 (thousands)
def place_value_6 : ℝ := 1000

-- Define the place value of 7 (tenths)
def place_value_7 : ℝ := 0.1

-- Theorem statement
theorem place_value_ratio :
  place_value_6 / place_value_7 = 10000 := by
  sorry

end place_value_ratio_l1082_108218


namespace two_digit_number_theorem_l1082_108262

/-- A two-digit natural number -/
def TwoDigitNumber (n : ℕ) : Prop := n ≥ 10 ∧ n ≤ 99

/-- The first digit of a two-digit number -/
def firstDigit (n : ℕ) : ℕ := n / 10

/-- The second digit of a two-digit number -/
def secondDigit (n : ℕ) : ℕ := n % 10

/-- The condition given in the problem -/
def satisfiesCondition (n : ℕ) : Prop :=
  4 * (firstDigit n) + 2 * (secondDigit n) = n / 2

theorem two_digit_number_theorem (n : ℕ) :
  TwoDigitNumber n ∧ satisfiesCondition n → n = 32 ∨ n = 64 ∨ n = 96 := by
  sorry

end two_digit_number_theorem_l1082_108262


namespace metal_collection_contest_solution_l1082_108268

/-- Represents the metal collection contest between boys and girls -/
structure MetalContest where
  totalMetal : ℕ
  boyAverage : ℕ
  girlAverage : ℕ
  numBoys : ℕ
  numGirls : ℕ

/-- Checks if the given numbers satisfy the contest conditions -/
def isValidContest (contest : MetalContest) : Prop :=
  contest.boyAverage * contest.numBoys + contest.girlAverage * contest.numGirls = contest.totalMetal

/-- Checks if boys won the contest -/
def boysWon (contest : MetalContest) : Prop :=
  contest.boyAverage * contest.numBoys > contest.girlAverage * contest.numGirls

/-- Theorem stating the solution to the metal collection contest -/
theorem metal_collection_contest_solution :
  ∃ (contest : MetalContest),
    contest.totalMetal = 2831 ∧
    contest.boyAverage = 95 ∧
    contest.girlAverage = 74 ∧
    contest.numBoys = 15 ∧
    contest.numGirls = 19 ∧
    isValidContest contest ∧
    boysWon contest :=
  sorry

end metal_collection_contest_solution_l1082_108268


namespace smallest_integer_with_given_remainders_l1082_108290

theorem smallest_integer_with_given_remainders :
  ∀ x : ℕ,
  (x % 6 = 5 ∧ x % 7 = 6 ∧ x % 8 = 7) →
  (∀ y : ℕ, y > 0 ∧ y % 6 = 5 ∧ y % 7 = 6 ∧ y % 8 = 7 → x ≤ y) →
  x = 167 :=
by sorry

end smallest_integer_with_given_remainders_l1082_108290


namespace train_crossing_time_l1082_108225

/-- Calculates the time for a train to cross a signal pole given its length and the time it takes to cross a platform of equal length. -/
theorem train_crossing_time (train_length platform_length : ℝ) (platform_crossing_time : ℝ) 
  (h1 : train_length = 300)
  (h2 : platform_length = 300)
  (h3 : platform_crossing_time = 36) :
  let total_distance := train_length + platform_length
  let train_speed := total_distance / platform_crossing_time
  train_length / train_speed = 18 := by sorry

end train_crossing_time_l1082_108225


namespace rationalize_and_simplify_l1082_108269

theorem rationalize_and_simplify :
  (Real.sqrt 8 + Real.sqrt 3) / (Real.sqrt 5 + Real.sqrt 3) =
  Real.sqrt 10 - Real.sqrt 6 + (Real.sqrt 15) / 2 - 3 / 2 := by
  sorry

end rationalize_and_simplify_l1082_108269


namespace typing_time_is_35_minutes_l1082_108292

/-- Represents the typing scenario with given conditions -/
structure TypingScenario where
  barbaraMaxSpeed : ℕ
  barbaraInjuryReduction : ℕ
  barbaraFatigueReduction : ℕ
  barbaraFatigueInterval : ℕ
  jimSpeed : ℕ
  jimTime : ℕ
  monicaSpeed : ℕ
  monicaTime : ℕ
  breakDuration : ℕ
  breakInterval : ℕ
  documentLength : ℕ

/-- Calculates the minimum time required to type the document -/
def minTypingTime (scenario : TypingScenario) : ℕ :=
  sorry

/-- Theorem stating that the minimum typing time for the given scenario is 35 minutes -/
theorem typing_time_is_35_minutes (scenario : TypingScenario) 
  (h1 : scenario.barbaraMaxSpeed = 212)
  (h2 : scenario.barbaraInjuryReduction = 40)
  (h3 : scenario.barbaraFatigueReduction = 5)
  (h4 : scenario.barbaraFatigueInterval = 15)
  (h5 : scenario.jimSpeed = 100)
  (h6 : scenario.jimTime = 20)
  (h7 : scenario.monicaSpeed = 150)
  (h8 : scenario.monicaTime = 10)
  (h9 : scenario.breakDuration = 5)
  (h10 : scenario.breakInterval = 25)
  (h11 : scenario.documentLength = 3440) :
  minTypingTime scenario = 35 :=
by sorry

end typing_time_is_35_minutes_l1082_108292


namespace no_polyhedron_with_area_2015_l1082_108200

/-- Represents a polyhedron constructed from unit cubes -/
structure Polyhedron where
  num_cubes : ℕ
  num_glued_faces : ℕ

/-- Calculates the surface area of a polyhedron -/
def surface_area (p : Polyhedron) : ℕ :=
  6 * p.num_cubes - 2 * p.num_glued_faces

/-- Theorem stating the impossibility of constructing a polyhedron with surface area 2015 -/
theorem no_polyhedron_with_area_2015 :
  ∀ p : Polyhedron, surface_area p ≠ 2015 := by
  sorry


end no_polyhedron_with_area_2015_l1082_108200


namespace liangliang_speed_l1082_108258

/-- The walking speeds of Mingming and Liangliang -/
structure WalkingSpeeds where
  mingming : ℝ
  liangliang : ℝ

/-- The initial and final distances between Mingming and Liangliang -/
structure Distances where
  initial : ℝ
  final : ℝ

/-- The time elapsed between the initial and final measurements -/
def elapsedTime : ℝ := 20

/-- The theorem stating the possible walking speeds of Liangliang -/
theorem liangliang_speed 
  (speeds : WalkingSpeeds) 
  (distances : Distances) 
  (h1 : speeds.mingming = 80) 
  (h2 : distances.initial = 3000) 
  (h3 : distances.final = 2900) :
  speeds.liangliang = 85 ∨ speeds.liangliang = 75 :=
sorry

end liangliang_speed_l1082_108258


namespace two_white_balls_possible_l1082_108263

/-- Represents the contents of the box -/
structure BoxContents where
  black : ℕ
  white : ℕ

/-- Represents a single replacement rule -/
inductive ReplacementRule
  | ThreeBlack
  | TwoBlackOneWhite
  | OneBlackTwoWhite
  | ThreeWhite

/-- Applies a single replacement rule to the box contents -/
def applyRule (contents : BoxContents) (rule : ReplacementRule) : BoxContents :=
  match rule with
  | ReplacementRule.ThreeBlack => 
      ⟨contents.black - 2, contents.white⟩
  | ReplacementRule.TwoBlackOneWhite => 
      ⟨contents.black - 1, contents.white⟩
  | ReplacementRule.OneBlackTwoWhite => 
      ⟨contents.black - 1, contents.white⟩
  | ReplacementRule.ThreeWhite => 
      ⟨contents.black + 1, contents.white - 2⟩

/-- Applies a sequence of replacement rules to the box contents -/
def applyRules (initial : BoxContents) (rules : List ReplacementRule) : BoxContents :=
  rules.foldl applyRule initial

theorem two_white_balls_possible : 
  ∃ (rules : List ReplacementRule), 
    (applyRules ⟨100, 100⟩ rules).white = 2 :=
  sorry


end two_white_balls_possible_l1082_108263


namespace problem_1_l1082_108248

theorem problem_1 : Real.sin (30 * π / 180) + |(-1)| - (Real.sqrt 3 - Real.pi)^0 = 1/2 := by
  sorry

end problem_1_l1082_108248


namespace factor_t_squared_minus_81_l1082_108278

theorem factor_t_squared_minus_81 (t : ℝ) : t^2 - 81 = (t - 9) * (t + 9) := by
  sorry

end factor_t_squared_minus_81_l1082_108278


namespace all_ap_lines_pass_through_point_l1082_108222

/-- A line in the form ax + by = c where a, b, and c form an arithmetic progression -/
structure APLine where
  a : ℝ
  d : ℝ
  eq : ℝ × ℝ → Prop := fun (x, y) ↦ a * x + (a + d) * y = a + 2 * d

/-- The theorem stating that all APLines pass through the point (-1, 2) -/
theorem all_ap_lines_pass_through_point :
  ∀ (l : APLine), l.eq (-1, 2) :=
sorry

end all_ap_lines_pass_through_point_l1082_108222


namespace gcd_xyz_square_l1082_108243

theorem gcd_xyz_square (x y z : ℕ) (h : (1 : ℚ) / x - (1 : ℚ) / y = (1 : ℚ) / z) :
  ∃ k : ℕ, Nat.gcd x (Nat.gcd y z) * x * y * z = k ^ 2 := by
sorry

end gcd_xyz_square_l1082_108243


namespace slope_determines_y_coordinate_l1082_108226

/-- Given two points R and S in a coordinate plane, if the slope of the line through R and S
    is equal to -4/3, then the y-coordinate of S is -8/3. -/
theorem slope_determines_y_coordinate (x_R y_R x_S : ℚ) : 
  let R : ℚ × ℚ := (x_R, y_R)
  let S : ℚ × ℚ := (x_S, y_S)
  x_R = -3 →
  y_R = 8 →
  x_S = 5 →
  (y_S - y_R) / (x_S - x_R) = -4/3 →
  y_S = -8/3 := by
sorry

end slope_determines_y_coordinate_l1082_108226


namespace f_properties_l1082_108249

noncomputable def f (x : ℝ) : ℝ := x^3 - 1/x^3

theorem f_properties :
  (∀ x > 0, f (-x) = -f x) ∧
  (∀ a b, 0 < a → a < b → f a < f b) := by
  sorry

end f_properties_l1082_108249


namespace quarter_circles_sum_approaches_circumference_l1082_108244

/-- The sum of quarter-circle arc lengths approaches the original circle's circumference as n approaches infinity -/
theorem quarter_circles_sum_approaches_circumference (R : ℝ) (h : R > 0) :
  let C := 2 * Real.pi * R
  let quarter_circle_sum (n : ℕ) := 2 * n * (Real.pi * C) / (2 * n)
  ∀ ε > 0, ∃ N, ∀ n ≥ N, |quarter_circle_sum n - C| < ε :=
by sorry

end quarter_circles_sum_approaches_circumference_l1082_108244


namespace population_reaches_max_in_180_years_l1082_108224

-- Define the initial conditions
def initial_year : ℕ := 2023
def island_area : ℕ := 31500
def land_per_person : ℕ := 2
def initial_population : ℕ := 250
def doubling_period : ℕ := 30

-- Define the maximum sustainable population
def max_population : ℕ := island_area / land_per_person

-- Define the population growth function
def population_after_years (years : ℕ) : ℕ :=
  initial_population * (2 ^ (years / doubling_period))

-- Theorem statement
theorem population_reaches_max_in_180_years :
  ∃ (years : ℕ), years = 180 ∧ 
  population_after_years years ≥ max_population ∧
  population_after_years (years - doubling_period) < max_population :=
sorry

end population_reaches_max_in_180_years_l1082_108224


namespace ceiling_floor_product_range_l1082_108257

theorem ceiling_floor_product_range (y : ℝ) : 
  y < -1 → ⌈y⌉ * ⌊y⌋ = 72 → -9 < y ∧ y < -8 := by
  sorry

end ceiling_floor_product_range_l1082_108257


namespace extended_pattern_ratio_l1082_108264

/-- Represents a square tile pattern -/
structure TilePattern :=
  (side : ℕ)
  (black_tiles : ℕ)
  (white_tiles : ℕ)

/-- Extends a tile pattern by adding a border of white tiles -/
def extend_pattern (p : TilePattern) : TilePattern :=
  { side := p.side + 2,
    black_tiles := p.black_tiles,
    white_tiles := p.white_tiles + (p.side + 2)^2 - p.side^2 }

/-- The ratio of black tiles to white tiles in a pattern -/
def tile_ratio (p : TilePattern) : ℚ :=
  p.black_tiles / p.white_tiles

theorem extended_pattern_ratio :
  let original := TilePattern.mk 5 13 12
  let extended := extend_pattern original
  tile_ratio extended = 13 / 36 := by sorry

end extended_pattern_ratio_l1082_108264


namespace complex_multiplication_l1082_108287

theorem complex_multiplication (z : ℂ) (h : z = 1 + Complex.I) : (1 + z) * z = 1 + 3 * Complex.I := by
  sorry

end complex_multiplication_l1082_108287


namespace dark_tile_fraction_is_five_sixteenths_l1082_108238

/-- Represents a tiled floor with a repeating pattern of dark tiles -/
structure TiledFloor :=
  (size : ℕ)  -- Size of the square floor (number of tiles per side)
  (dark_tiles_per_section : ℕ)  -- Number of dark tiles in each 4x4 section
  (total_tiles_per_section : ℕ)  -- Total number of tiles in each 4x4 section

/-- The fraction of dark tiles on the floor -/
def dark_tile_fraction (floor : TiledFloor) : ℚ :=
  (floor.dark_tiles_per_section : ℚ) / (floor.total_tiles_per_section : ℚ)

/-- Theorem stating that the fraction of dark tiles is 5/16 -/
theorem dark_tile_fraction_is_five_sixteenths (floor : TiledFloor) 
  (h1 : floor.size > 0)
  (h2 : floor.dark_tiles_per_section = 5)
  (h3 : floor.total_tiles_per_section = 16) : 
  dark_tile_fraction floor = 5 / 16 := by
  sorry

end dark_tile_fraction_is_five_sixteenths_l1082_108238


namespace uncorrelated_variables_l1082_108273

/-- Represents a variable in our correlation problem -/
structure Variable where
  name : String

/-- Represents a pair of variables -/
structure VariablePair where
  var1 : Variable
  var2 : Variable

/-- Defines what it means for two variables to be correlated -/
def are_correlated (pair : VariablePair) : Prop :=
  sorry  -- The actual definition would go here

/-- The list of variable pairs we're considering -/
def variable_pairs : List VariablePair :=
  [ { var1 := { name := "Grain yield" }, var2 := { name := "Amount of fertilizer used" } },
    { var1 := { name := "College entrance examination scores" }, var2 := { name := "Time spent on review" } },
    { var1 := { name := "Sales of goods" }, var2 := { name := "Advertising expenses" } },
    { var1 := { name := "Number of books sold at fixed price" }, var2 := { name := "Sales revenue" } } ]

/-- The theorem we want to prove -/
theorem uncorrelated_variables : 
  ∃ (pair : VariablePair), pair ∈ variable_pairs ∧ ¬(are_correlated pair) :=
sorry


end uncorrelated_variables_l1082_108273


namespace square_eq_nine_solutions_l1082_108271

theorem square_eq_nine_solutions (x : ℝ) : x^2 = 9 ↔ x = 3 ∨ x = -3 := by sorry

end square_eq_nine_solutions_l1082_108271


namespace quadratic_root_value_l1082_108217

theorem quadratic_root_value (n : ℝ) : n^2 - 5*n + 4 = 0 → n^2 - 5*n = -4 := by
  sorry

end quadratic_root_value_l1082_108217


namespace christopher_karen_difference_l1082_108245

/-- Proves that Christopher has $8.00 more than Karen given their quarter counts -/
theorem christopher_karen_difference :
  let karen_quarters : ℕ := 32
  let christopher_quarters : ℕ := 64
  let quarter_value : ℚ := 1/4
  let karen_money := karen_quarters * quarter_value
  let christopher_money := christopher_quarters * quarter_value
  christopher_money - karen_money = 8 :=
by sorry

end christopher_karen_difference_l1082_108245


namespace benny_baseball_gear_spending_l1082_108274

/-- The amount Benny spent on baseball gear --/
def amount_spent (initial_amount left_over : ℕ) : ℕ :=
  initial_amount - left_over

/-- Theorem: Benny spent $47 on baseball gear --/
theorem benny_baseball_gear_spending :
  amount_spent 79 32 = 47 := by
  sorry

end benny_baseball_gear_spending_l1082_108274


namespace pedestrians_meeting_l1082_108286

/-- The problem of two pedestrians meeting --/
theorem pedestrians_meeting 
  (distance : ℝ) 
  (initial_meeting_time : ℝ) 
  (adjusted_meeting_time : ℝ) 
  (speed_multiplier_1 : ℝ) 
  (speed_multiplier_2 : ℝ) 
  (h1 : distance = 105) 
  (h2 : initial_meeting_time = 7.5) 
  (h3 : adjusted_meeting_time = 8 + 1/13) 
  (h4 : speed_multiplier_1 = 1.5) 
  (h5 : speed_multiplier_2 = 0.5) :
  ∃ (speed1 speed2 : ℝ), 
    speed1 = 8 ∧ 
    speed2 = 6 ∧ 
    initial_meeting_time * (speed1 + speed2) = distance ∧ 
    adjusted_meeting_time * (speed_multiplier_1 * speed1 + speed_multiplier_2 * speed2) = distance :=
by sorry


end pedestrians_meeting_l1082_108286


namespace multiples_between_200_and_500_l1082_108204

def count_multiples (lower upper lcm : ℕ) : ℕ :=
  (upper / lcm) - ((lower - 1) / lcm)

theorem multiples_between_200_and_500 : count_multiples 200 500 36 = 8 := by
  sorry

end multiples_between_200_and_500_l1082_108204


namespace book_cost_proof_l1082_108284

theorem book_cost_proof (initial_amount : ℕ) (num_books : ℕ) (remaining_amount : ℕ) 
  (h1 : initial_amount = 79)
  (h2 : num_books = 9)
  (h3 : remaining_amount = 16) :
  (initial_amount - remaining_amount) / num_books = 7 :=
by
  sorry

end book_cost_proof_l1082_108284


namespace fraction_non_negative_iff_positive_denominator_l1082_108294

theorem fraction_non_negative_iff_positive_denominator :
  ∀ x : ℝ, (2 / x ≥ 0) ↔ (x > 0) := by sorry

end fraction_non_negative_iff_positive_denominator_l1082_108294


namespace pants_cut_amount_l1082_108297

def skirt_cut : ℝ := 0.75
def difference : ℝ := 0.25

theorem pants_cut_amount : ∃ (x : ℝ), x = skirt_cut - difference := by sorry

end pants_cut_amount_l1082_108297


namespace solve_for_y_l1082_108223

theorem solve_for_y (x y : ℝ) (h1 : x^(3*y) = 8) (h2 : x = 2) : y = 1 := by
  sorry

end solve_for_y_l1082_108223


namespace total_candies_l1082_108270

theorem total_candies (linda_candies chloe_candies : ℕ) 
  (h1 : linda_candies = 34) 
  (h2 : chloe_candies = 28) : 
  linda_candies + chloe_candies = 62 := by
  sorry

end total_candies_l1082_108270


namespace remainder_problem_l1082_108256

theorem remainder_problem (m n : ℕ) (h1 : m % n = 2) (h2 : (3 * m) % n = 1) : n = 5 := by
  sorry

end remainder_problem_l1082_108256


namespace container_fill_fraction_l1082_108234

theorem container_fill_fraction (initial_percentage : ℝ) (added_water : ℝ) (capacity : ℝ) : 
  initial_percentage = 0.3 →
  added_water = 27 →
  capacity = 60 →
  (initial_percentage * capacity + added_water) / capacity = 0.75 := by
sorry

end container_fill_fraction_l1082_108234
