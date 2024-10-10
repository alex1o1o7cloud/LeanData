import Mathlib

namespace rotten_eggs_calculation_l17_1784

/-- The percentage of spoiled milk bottles -/
def spoiled_milk_percentage : ℝ := 0.20

/-- The percentage of flour canisters with weevils -/
def weevil_flour_percentage : ℝ := 0.25

/-- The probability of all three ingredients being good -/
def all_good_probability : ℝ := 0.24

/-- The percentage of rotten eggs -/
def rotten_eggs_percentage : ℝ := 0.60

theorem rotten_eggs_calculation :
  (1 - spoiled_milk_percentage) * (1 - rotten_eggs_percentage) * (1 - weevil_flour_percentage) = all_good_probability :=
by sorry

end rotten_eggs_calculation_l17_1784


namespace only_one_correct_proposition_l17_1715

/-- A proposition about the relationship between lines and planes in 3D space -/
inductive GeometryProposition
  | InfinitePointsImpliesParallel
  | ParallelToPlaneImpliesParallelToLines
  | ParallelLineImpliesParallelToPlane
  | ParallelToPlaneImpliesNoIntersection

/-- Predicate to check if a geometry proposition is correct -/
def is_correct_proposition (p : GeometryProposition) : Prop :=
  match p with
  | GeometryProposition.InfinitePointsImpliesParallel => False
  | GeometryProposition.ParallelToPlaneImpliesParallelToLines => False
  | GeometryProposition.ParallelLineImpliesParallelToPlane => False
  | GeometryProposition.ParallelToPlaneImpliesNoIntersection => True

/-- Theorem stating that only one of the geometry propositions is correct -/
theorem only_one_correct_proposition :
  ∃! (p : GeometryProposition), is_correct_proposition p :=
sorry

end only_one_correct_proposition_l17_1715


namespace smallest_prime_factor_of_2551_l17_1737

theorem smallest_prime_factor_of_2551 :
  ∃ (p : ℕ), Nat.Prime p ∧ p ∣ 2551 ∧ ∀ (q : ℕ), Nat.Prime q → q ∣ 2551 → p ≤ q :=
by sorry

end smallest_prime_factor_of_2551_l17_1737


namespace arithmetic_sequence_sum_l17_1728

/-- An arithmetic sequence is a sequence where the difference between
    any two consecutive terms is constant. -/
def is_arithmetic_sequence (a : ℕ → ℚ) : Prop :=
  ∃ d : ℚ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℚ) :
  is_arithmetic_sequence a → a 2 = 5 → a 5 = 33 → a 3 + a 4 = 38 := by
  sorry

end arithmetic_sequence_sum_l17_1728


namespace complex_number_problem_l17_1717

def z (b : ℝ) : ℂ := 3 + b * Complex.I

theorem complex_number_problem (b : ℝ) 
  (h : ∃ k : ℝ, (1 + 3 * Complex.I) * z b = k * Complex.I) :
  z b = 3 + Complex.I ∧ Complex.abs ((z b) / (2 + Complex.I)) = Real.sqrt 2 := by
  sorry

end complex_number_problem_l17_1717


namespace neither_sufficient_nor_necessary_l17_1795

theorem neither_sufficient_nor_necessary (a b : ℝ) :
  (∃ x y : ℝ, x - y > 0 ∧ x^2 - y^2 ≤ 0) ∧
  (∃ x y : ℝ, x - y ≤ 0 ∧ x^2 - y^2 > 0) :=
sorry

end neither_sufficient_nor_necessary_l17_1795


namespace solution_set_characterization_l17_1798

open Set

def solution_set (f : ℝ → ℝ) : Set ℝ := {x | x > 0 ∧ f x ≤ Real.log x}

theorem solution_set_characterization
  (f : ℝ → ℝ) (hf : Differentiable ℝ f)
  (h1 : f 1 = 0)
  (h2 : ∀ x > 0, x * (deriv f x) > 1) :
  solution_set f = Ioc 0 1 := by
  sorry

end solution_set_characterization_l17_1798


namespace boat_speed_in_still_water_l17_1786

/-- Proves that a boat's speed in still water is 42 km/hr given specific conditions -/
theorem boat_speed_in_still_water 
  (current_speed : ℝ) 
  (downstream_distance : ℝ) 
  (downstream_time : ℝ) 
  (h1 : current_speed = 3) 
  (h2 : downstream_distance = 33) 
  (h3 : downstream_time = 44 / 60) : 
  ∃ (still_water_speed : ℝ), 
    still_water_speed = 42 ∧ 
    downstream_distance = (still_water_speed + current_speed) * downstream_time :=
by sorry

end boat_speed_in_still_water_l17_1786


namespace parallel_lines_in_plane_l17_1766

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the relations
variable (parallel : Line → Plane → Prop)
variable (parallel_lines : Line → Line → Prop)
variable (intersect : Plane → Plane → Line → Prop)
variable (contained_in : Line → Plane → Prop)

-- State the theorem
theorem parallel_lines_in_plane 
  (α β : Plane) (a b c : Line) :
  parallel a α →
  parallel b α →
  intersect β α c →
  contained_in a β →
  contained_in b β →
  parallel_lines a b :=
sorry

end parallel_lines_in_plane_l17_1766


namespace sum_of_roots_equals_one_l17_1775

theorem sum_of_roots_equals_one :
  ∀ x₁ x₂ : ℝ,
  (x₁ + 3) * (x₁ - 4) = 18 →
  (x₂ + 3) * (x₂ - 4) = 18 →
  x₁ + x₂ = 1 := by
  sorry

end sum_of_roots_equals_one_l17_1775


namespace no_solution_iff_m_zero_l17_1731

theorem no_solution_iff_m_zero (m : ℝ) : 
  (∀ x : ℝ, x ≠ 1 → (2 - x) / (1 - x) ≠ (m + x) / (1 - x) + 1) ↔ m = 0 := by
  sorry

end no_solution_iff_m_zero_l17_1731


namespace irrational_sqrt_10_and_others_rational_l17_1778

theorem irrational_sqrt_10_and_others_rational : 
  (Irrational (Real.sqrt 10)) ∧ 
  (¬ Irrational (1 / 7 : ℝ)) ∧ 
  (¬ Irrational (3.5 : ℝ)) ∧ 
  (¬ Irrational (-0.3030030003 : ℝ)) := by
  sorry

end irrational_sqrt_10_and_others_rational_l17_1778


namespace perfect_square_trinomial_k_l17_1734

/-- A polynomial is a perfect square trinomial if it can be expressed as (x + a)^2 for some real number a. -/
def IsPerfectSquareTrinomial (p : ℝ → ℝ) : Prop :=
  ∃ a : ℝ, ∀ x : ℝ, p x = (x + a)^2

/-- Given that x^2 + kx + 25 is a perfect square trinomial, prove that k = 10 or k = -10. -/
theorem perfect_square_trinomial_k (k : ℝ) :
  IsPerfectSquareTrinomial (fun x => x^2 + k*x + 25) →
  k = 10 ∨ k = -10 := by
  sorry


end perfect_square_trinomial_k_l17_1734


namespace quadratic_inequality_solution_l17_1792

theorem quadratic_inequality_solution (a b : ℝ) : 
  (∀ x, x^2 < a*x + b ↔ 1 < x ∧ x < 3) → b^a = 81 := by
sorry

end quadratic_inequality_solution_l17_1792


namespace vector_perpendicular_l17_1790

/-- Given vectors a, b, and c in ℝ², prove that (a-b) is perpendicular to c -/
theorem vector_perpendicular (a b c : ℝ × ℝ) 
  (ha : a = (0, 5)) 
  (hb : b = (4, -3)) 
  (hc : c = (-2, -1)) : 
  (a.1 - b.1) * c.1 + (a.2 - b.2) * c.2 = 0 := by
  sorry

end vector_perpendicular_l17_1790


namespace f_symmetric_about_pi_third_l17_1718

/-- A function is symmetric about a point (a, 0) if f(a + x) = -f(a - x) for all x in the domain of f -/
def SymmetricAboutPoint (f : ℝ → ℝ) (a : ℝ) : Prop :=
  ∀ x, f (a + x) = -f (a - x)

/-- The tangent function -/
noncomputable def tan (x : ℝ) : ℝ := Real.tan x

/-- The given function f(x) = tan(x + π/6) -/
noncomputable def f (x : ℝ) : ℝ := tan (x + Real.pi / 6)

/-- Theorem stating that f(x) = tan(x + π/6) is symmetric about the point (π/3, 0) -/
theorem f_symmetric_about_pi_third : SymmetricAboutPoint f (Real.pi / 3) := by
  sorry

end f_symmetric_about_pi_third_l17_1718


namespace differential_equation_solution_l17_1707

/-- The differential equation dy/dx + xy = x^2 has the general solution y(x) = x^3/4 + C/x -/
theorem differential_equation_solution (x : ℝ) (C : ℝ) :
  let y : ℝ → ℝ := λ x => x^3 / 4 + C / x
  let dy_dx : ℝ → ℝ := λ x => 3 * x^2 / 4 - C / x^2
  ∀ x ≠ 0, dy_dx x + x * y x = x^2 := by
sorry

end differential_equation_solution_l17_1707


namespace max_popsicles_for_8_dollars_l17_1754

/-- Represents the different popsicle purchase options -/
inductive PopsicleOption
  | Single
  | Box3
  | Box5

/-- Returns the cost of a given popsicle option -/
def cost (option : PopsicleOption) : ℕ :=
  match option with
  | .Single => 1
  | .Box3 => 2
  | .Box5 => 3

/-- Returns the number of popsicles in a given option -/
def popsicles (option : PopsicleOption) : ℕ :=
  match option with
  | .Single => 1
  | .Box3 => 3
  | .Box5 => 5

/-- Represents a purchase of popsicles -/
structure Purchase where
  singles : ℕ
  box3s : ℕ
  box5s : ℕ

/-- Calculates the total cost of a purchase -/
def totalCost (p : Purchase) : ℕ :=
  p.singles * cost PopsicleOption.Single +
  p.box3s * cost PopsicleOption.Box3 +
  p.box5s * cost PopsicleOption.Box5

/-- Calculates the total number of popsicles in a purchase -/
def totalPopsicles (p : Purchase) : ℕ :=
  p.singles * popsicles PopsicleOption.Single +
  p.box3s * popsicles PopsicleOption.Box3 +
  p.box5s * popsicles PopsicleOption.Box5

/-- Theorem: The maximum number of popsicles that can be purchased with $8 is 13 -/
theorem max_popsicles_for_8_dollars :
  ∀ p : Purchase, totalCost p ≤ 8 → totalPopsicles p ≤ 13 ∧
  ∃ p' : Purchase, totalCost p' = 8 ∧ totalPopsicles p' = 13 :=
sorry

end max_popsicles_for_8_dollars_l17_1754


namespace tv_sets_in_shop_d_l17_1709

theorem tv_sets_in_shop_d (total_shops : Nat) (avg_tv_sets : Nat)
  (shop_a shop_b shop_c shop_e : Nat) :
  total_shops = 5 →
  avg_tv_sets = 48 →
  shop_a = 20 →
  shop_b = 30 →
  shop_c = 60 →
  shop_e = 50 →
  ∃ shop_d : Nat, shop_d = 80 ∧
    avg_tv_sets * total_shops = shop_a + shop_b + shop_c + shop_d + shop_e :=
by sorry

end tv_sets_in_shop_d_l17_1709


namespace brick_packing_theorem_l17_1761

/-- Represents the dimensions of a rectangular parallelepiped -/
structure Dimensions where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Calculates the volume of a rectangular parallelepiped -/
def volume (d : Dimensions) : ℕ := d.length * d.width * d.height

theorem brick_packing_theorem (box : Dimensions) 
  (brick1 brick2 : Dimensions) 
  (h_box : box = ⟨10, 11, 14⟩) 
  (h_brick1 : brick1 = ⟨2, 5, 8⟩) 
  (h_brick2 : brick2 = ⟨2, 3, 7⟩) :
  ∃ (x y : ℕ), 
    x * volume brick1 + y * volume brick2 = volume box ∧ 
    x + y = 24 ∧ 
    ∀ (a b : ℕ), a * volume brick1 + b * volume brick2 = volume box → a + b ≥ 24 := by
  sorry

end brick_packing_theorem_l17_1761


namespace mans_wage_to_womans_wage_ratio_l17_1743

/-- Prove that the ratio of a man's daily wage to a woman's daily wage is 4:1 -/
theorem mans_wage_to_womans_wage_ratio :
  ∀ (man_wage woman_wage : ℚ),
  (∃ k : ℚ, man_wage = k * woman_wage) →  -- Man's wage is some multiple of woman's wage
  (8 * 25 * man_wage = 14400) →           -- 8 men working for 25 days earn Rs. 14400
  (40 * 30 * woman_wage = 21600) →        -- 40 women working for 30 days earn Rs. 21600
  man_wage / woman_wage = 4 / 1 := by
sorry

end mans_wage_to_womans_wage_ratio_l17_1743


namespace companion_point_on_hyperbola_companion_point_on_line_companion_point_line_equation_l17_1771

/-- Definition of companion point -/
def is_companion_point (P Q : ℝ × ℝ) : Prop :=
  Q.1 = P.1 + 2 ∧ Q.2 = P.2 - 4

/-- Theorem 1: The companion point of P(2,-1) lies on y = -20/x -/
theorem companion_point_on_hyperbola :
  ∀ Q : ℝ × ℝ, is_companion_point (2, -1) Q → Q.2 = -20 / Q.1 :=
sorry

/-- Theorem 2: If P(a,b) lies on y = x+5 and (-1,-2) is its companion point, then P = (-3,2) -/
theorem companion_point_on_line :
  ∀ P : ℝ × ℝ, P.2 = P.1 + 5 → is_companion_point P (-1, -2) → P = (-3, 2) :=
sorry

/-- Theorem 3: If P(a,b) lies on y = 2x+3, then its companion point Q lies on y = 2x-5 -/
theorem companion_point_line_equation :
  ∀ P Q : ℝ × ℝ, P.2 = 2 * P.1 + 3 → is_companion_point P Q → Q.2 = 2 * Q.1 - 5 :=
sorry

end companion_point_on_hyperbola_companion_point_on_line_companion_point_line_equation_l17_1771


namespace project_men_count_l17_1745

/-- The number of men originally working on the project -/
def original_men : ℕ := 110

/-- The number of days it takes the original number of men to complete the work -/
def original_days : ℕ := 100

/-- The reduction in the number of men -/
def men_reduction : ℕ := 10

/-- The increase in days when the number of men is reduced -/
def days_increase : ℕ := 10

theorem project_men_count :
  (original_men * original_days = (original_men - men_reduction) * (original_days + days_increase)) →
  original_men = 110 := by
  sorry

end project_men_count_l17_1745


namespace proportional_relationship_and_point_value_l17_1760

/-- Given that y is directly proportional to x-1 and y = 4 when x = 3,
    prove that the relationship between y and x is y = 2x - 2,
    and when the point (-1,m) lies on this graph, m = -4. -/
theorem proportional_relationship_and_point_value 
  (y : ℝ → ℝ) 
  (h1 : ∃ k : ℝ, ∀ x, y x = k * (x - 1)) 
  (h2 : y 3 = 4) :
  (∀ x, y x = 2*x - 2) ∧ 
  y (-1) = -4 := by
sorry

end proportional_relationship_and_point_value_l17_1760


namespace school_garden_flowers_l17_1748

theorem school_garden_flowers :
  let green_flowers : ℕ := 9
  let red_flowers : ℕ := 3 * green_flowers
  let yellow_flowers : ℕ := 12
  let total_flowers : ℕ := green_flowers + red_flowers + yellow_flowers + (green_flowers + red_flowers + yellow_flowers)
  total_flowers = 96 := by
  sorry

end school_garden_flowers_l17_1748


namespace distance_to_y_axis_l17_1742

def point_A (x : ℝ) : ℝ × ℝ := (x - 4, 2 * x + 3)

theorem distance_to_y_axis (x : ℝ) : 
  (|x - 4| = 1) ↔ (x = 5 ∨ x = 3) :=
sorry

end distance_to_y_axis_l17_1742


namespace tax_free_amount_satisfies_equation_l17_1733

/-- The tax-free amount for goods purchased in country B -/
def tax_free_amount : ℝ :=
  -- We define the tax-free amount, but don't provide its value
  -- as it needs to be proved
  sorry

/-- The total value of goods purchased -/
def total_value : ℝ := 1720

/-- The tax rate as a decimal -/
def tax_rate : ℝ := 0.11

/-- The amount of tax paid -/
def tax_paid : ℝ := 123.2

/-- Theorem stating that the tax-free amount satisfies the given equation -/
theorem tax_free_amount_satisfies_equation :
  tax_rate * (total_value - tax_free_amount) = tax_paid := by
  sorry

end tax_free_amount_satisfies_equation_l17_1733


namespace ndoti_winning_strategy_l17_1701

/-- Represents a point on the plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a square on the plane -/
structure Square where
  a : Point
  b : Point
  c : Point
  d : Point

/-- Represents a quadrilateral on the plane -/
structure Quadrilateral where
  x : Point
  y : Point
  z : Point
  w : Point

/-- Calculates the area of a square -/
def squareArea (s : Square) : ℝ := sorry

/-- Calculates the area of a quadrilateral -/
def quadrilateralArea (q : Quadrilateral) : ℝ := sorry

/-- Checks if a point is on a side of the square -/
def isOnSquareSide (p : Point) (s : Square) : Prop := sorry

/-- Ndoti's strategy function -/
def ndotiStrategy (s : Square) (x : Point) : Quadrilateral := sorry

/-- The main theorem stating Ndoti's winning strategy -/
theorem ndoti_winning_strategy (s : Square) :
  ∀ x : Point, isOnSquareSide x s →
    quadrilateralArea (ndotiStrategy s x) < (1/2) * squareArea s :=
by sorry

end ndoti_winning_strategy_l17_1701


namespace circle_radius_when_area_circumference_ratio_is_ten_l17_1723

/-- Given a circle with area M cm² and circumference N cm, if M/N = 10, then the radius is 20 cm -/
theorem circle_radius_when_area_circumference_ratio_is_ten
  (M N : ℝ) -- M is the area, N is the circumference
  (h1 : M > 0) -- area is positive
  (h2 : N > 0) -- circumference is positive
  (h3 : M = π * (N / (2 * π))^2) -- area formula
  (h4 : M / N = 10) -- given ratio
  : N / (2 * π) = 20 := by
  sorry

end circle_radius_when_area_circumference_ratio_is_ten_l17_1723


namespace filter_kit_price_l17_1741

theorem filter_kit_price :
  let individual_prices : List ℝ := [12.45, 12.45, 14.05, 14.05, 11.50]
  let total_individual_price := individual_prices.sum
  let discount_percentage : ℝ := 11.03448275862069 / 100
  let kit_price := total_individual_price * (1 - discount_percentage)
  kit_price = 57.382758620689655 := by
sorry

end filter_kit_price_l17_1741


namespace lucys_fish_count_l17_1735

theorem lucys_fish_count (initial_fish : ℕ) 
  (h1 : initial_fish + 68 = 280) : initial_fish = 212 := by
  sorry

end lucys_fish_count_l17_1735


namespace question_1_question_2_l17_1755

-- Define the propositions p and q
def p (x a : ℝ) : Prop := x - a < 0
def q (x : ℝ) : Prop := x^2 - 4*x + 3 ≤ 0

-- Define the range of x for question 1
def range_x : Set ℝ := {x | 1 ≤ x ∧ x < 2}

-- Define the range of a for question 2
def range_a : Set ℝ := {a | a > 3}

-- Theorem for question 1
theorem question_1 (a : ℝ) (h : a = 2) :
  {x : ℝ | p x a ∧ q x} = range_x := by sorry

-- Theorem for question 2
theorem question_2 :
  (∀ x, q x → p x a) ∧ (∃ x, p x a ∧ ¬q x) →
  a ∈ range_a := by sorry

end question_1_question_2_l17_1755


namespace geometric_sequence_property_l17_1788

/-- Represents a geometric sequence with first term a and common ratio q -/
structure GeometricSequence (α : Type*) [Field α] where
  a : α
  q : α

/-- Sum of first n terms of a geometric sequence -/
def sumGeometric {α : Type*} [Field α] (seq : GeometricSequence α) (n : ℕ) : α :=
  seq.a * (1 - seq.q ^ n) / (1 - seq.q)

theorem geometric_sequence_property {α : Type*} [Field α] (seq : GeometricSequence α) :
  (sumGeometric seq 3 + sumGeometric seq 6 = 2 * sumGeometric seq 9) →
  (seq.a * seq.q + seq.a * seq.q^4 = 4) →
  seq.a * seq.q^7 = 2 := by
  sorry

end geometric_sequence_property_l17_1788


namespace dance_ratio_l17_1730

/-- Given the conditions of a dance, prove the ratio of boys to girls -/
theorem dance_ratio :
  ∀ (boys girls teachers : ℕ),
  girls = 60 →
  teachers = boys / 5 →
  boys + girls + teachers = 114 →
  ∃ (a b : ℕ), a ≠ 0 ∧ b ≠ 0 ∧ boys * b = girls * a ∧ a = 3 ∧ b = 4 :=
by
  sorry

end dance_ratio_l17_1730


namespace february_warmer_than_january_l17_1793

/-- The average temperature in January 2023 in Taiyuan City (in °C) -/
def jan_temp : ℝ := -12

/-- The average temperature in February 2023 in Taiyuan City (in °C) -/
def feb_temp : ℝ := -6

/-- The difference in average temperature between February and January 2023 in Taiyuan City -/
def temp_difference : ℝ := feb_temp - jan_temp

theorem february_warmer_than_january : temp_difference = 6 := by
  sorry

end february_warmer_than_january_l17_1793


namespace arithmetic_sequence_61st_term_l17_1751

/-- An arithmetic sequence is a sequence where the difference between
    successive terms is constant. -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_61st_term
  (a : ℕ → ℝ)
  (h_arith : ArithmeticSequence a)
  (h_5 : a 5 = 33)
  (h_45 : a 45 = 153) :
  a 61 = 201 :=
sorry

end arithmetic_sequence_61st_term_l17_1751


namespace ognev_phone_number_l17_1782

/-- Represents a surname -/
structure Surname :=
  (name : String)

/-- Calculates the length of a surname -/
def surname_length (s : Surname) : Nat :=
  s.name.length

/-- Gets the position of a character in the alphabet (A=1, B=2, etc.) -/
def char_position (c : Char) : Nat :=
  if 'A' ≤ c ∧ c ≤ 'Z' then c.toNat - 'A'.toNat + 1
  else if 'a' ≤ c ∧ c ≤ 'z' then c.toNat - 'a'.toNat + 1
  else 0

/-- Calculates the phone number for a given surname -/
def phone_number (s : Surname) : Nat :=
  let len := surname_length s
  let first_pos := char_position s.name.front
  let last_pos := char_position s.name.back
  len * 1000 + first_pos * 100 + last_pos

/-- The theorem to be proved -/
theorem ognev_phone_number :
  phone_number { name := "Ognev" } = 5163 := by
  sorry

end ognev_phone_number_l17_1782


namespace line_plane_relationship_l17_1779

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relationships between lines and planes
variable (perpendicular : Line → Line → Prop)
variable (parallel_line_plane : Line → Plane → Prop)
variable (intersects : Line → Plane → Prop)
variable (contained_in : Line → Plane → Prop)
variable (parallel_line_plane : Line → Plane → Prop)

-- State the theorem
theorem line_plane_relationship 
  (a b : Line) (α : Plane) 
  (h1 : perpendicular a b) 
  (h2 : parallel_line_plane a α) : 
  intersects b α ∨ contained_in b α ∨ parallel_line_plane b α :=
sorry

end line_plane_relationship_l17_1779


namespace binomial_coeff_not_arithmetic_progression_l17_1776

theorem binomial_coeff_not_arithmetic_progression (n k : ℕ) (h1 : k ≤ n - 3) :
  ¬∃ (a d : ℤ), 
    (Nat.choose n k : ℤ) = a ∧
    (Nat.choose n (k + 1) : ℤ) = a + d ∧
    (Nat.choose n (k + 2) : ℤ) = a + 2*d ∧
    (Nat.choose n (k + 3) : ℤ) = a + 3*d :=
by sorry

end binomial_coeff_not_arithmetic_progression_l17_1776


namespace greatest_measuring_length_l17_1780

theorem greatest_measuring_length
  (length1 length2 length3 : ℕ)
  (h1 : length1 = 1234)
  (h2 : length2 = 898)
  (h3 : length3 > 0)
  (h4 : Nat.gcd length1 (Nat.gcd length2 length3) = 1) :
  ∀ (measuring_length : ℕ),
    (measuring_length > 0 ∧
     length1 % measuring_length = 0 ∧
     length2 % measuring_length = 0 ∧
     length3 % measuring_length = 0) →
    measuring_length = 1 :=
by sorry

end greatest_measuring_length_l17_1780


namespace trig_identity_l17_1791

theorem trig_identity (α : ℝ) : 
  (Real.sin α)^2 + (Real.cos (30 * π / 180 - α))^2 - 
  (Real.sin α) * (Real.cos (30 * π / 180 - α)) = 3/4 := by
  sorry

end trig_identity_l17_1791


namespace gift_wrapping_expenses_l17_1772

def total_spent : ℝ := 700
def gift_cost : ℝ := 561

theorem gift_wrapping_expenses : total_spent - gift_cost = 139 := by
  sorry

end gift_wrapping_expenses_l17_1772


namespace sum_of_roots_squared_equation_l17_1738

theorem sum_of_roots_squared_equation (x : ℝ) :
  (x - 3)^2 = 16 → ∃ y : ℝ, (y - 3)^2 = 16 ∧ x + y = 6 :=
by sorry

end sum_of_roots_squared_equation_l17_1738


namespace associated_number_equality_l17_1716

-- Define the associated number function
def associated_number (x : ℚ) : ℚ :=
  if x ≥ 0 then 2 * x - 1 else -2 * x + 1

-- State the theorem
theorem associated_number_equality (a b : ℚ) (ha : a > 0) (hb : b < 0) 
  (h_eq : associated_number a = associated_number b) : 
  (a + b)^2 - 2*a - 2*b = -1 := by sorry

end associated_number_equality_l17_1716


namespace sugar_price_reduction_l17_1711

/-- Calculates the percentage reduction in sugar price given the original price and the amount that can be bought after reduction. -/
theorem sugar_price_reduction 
  (original_price : ℝ) 
  (budget : ℝ) 
  (extra_amount : ℝ) 
  (h1 : original_price = 8) 
  (h2 : budget = 120) 
  (h3 : extra_amount = 1) 
  (h4 : budget / original_price + extra_amount = budget / (budget / (budget / original_price + extra_amount))) : 
  (original_price - budget / (budget / original_price + extra_amount)) / original_price * 100 = 6.25 := by
sorry

end sugar_price_reduction_l17_1711


namespace inverse_power_function_at_4_l17_1752

-- Define a power function
def isPowerFunction (f : ℝ → ℝ) : Prop :=
  ∃ a : ℝ, ∀ x : ℝ, f x = x ^ a

-- State the theorem
theorem inverse_power_function_at_4 (f : ℝ → ℝ) 
  (h1 : isPowerFunction f) 
  (h2 : f 2 = Real.sqrt 2) :
  Function.invFun f 4 = 16 := by
sorry

end inverse_power_function_at_4_l17_1752


namespace candy_distribution_l17_1757

theorem candy_distribution (total_candies : ℕ) (num_bags : ℕ) (candies_per_bag : ℕ) :
  total_candies = 15 →
  num_bags = 5 →
  total_candies = num_bags * candies_per_bag →
  candies_per_bag = 3 := by
sorry

end candy_distribution_l17_1757


namespace total_deduction_is_137_5_l17_1789

/-- Represents David's hourly wage in dollars -/
def hourly_wage : ℝ := 25

/-- Represents the local tax rate as a decimal -/
def local_tax_rate : ℝ := 0.025

/-- Represents the retirement fund contribution rate as a decimal -/
def retirement_rate : ℝ := 0.03

/-- Converts dollars to cents -/
def dollars_to_cents (dollars : ℝ) : ℝ := dollars * 100

/-- Calculates the total deduction in cents -/
def total_deduction : ℝ :=
  dollars_to_cents (hourly_wage * local_tax_rate + hourly_wage * retirement_rate)

/-- Theorem stating that the total deduction is 137.5 cents -/
theorem total_deduction_is_137_5 : total_deduction = 137.5 := by
  sorry


end total_deduction_is_137_5_l17_1789


namespace exists_monochromatic_isosceles_right_triangle_l17_1721

-- Define a type for colors
inductive Color
| Red
| Blue

-- Define a type for points in the plane
structure Point where
  x : ℝ
  y : ℝ

-- Define a coloring function
def coloring : Point → Color := sorry

-- Define an isosceles right triangle
def isIsoscelesRightTriangle (a b c : Point) : Prop := sorry

-- Theorem statement
theorem exists_monochromatic_isosceles_right_triangle :
  ∃ a b c : Point, isIsoscelesRightTriangle a b c ∧ 
    coloring a = coloring b ∧ coloring b = coloring c := by sorry

end exists_monochromatic_isosceles_right_triangle_l17_1721


namespace value_of_x_l17_1762

theorem value_of_x : ∃ x : ℝ, (3 * x + 15 = (1 / 3) * (6 * x + 45)) ∧ (x = 0) := by
  sorry

end value_of_x_l17_1762


namespace square_rectangle_equal_area_l17_1713

theorem square_rectangle_equal_area (a b c : ℝ) (h : a > 0 ∧ b > 0 ∧ c > 0) :
  a^2 = b * c → a = Real.sqrt (b * c) := by
  sorry

end square_rectangle_equal_area_l17_1713


namespace parabola_min_distance_sum_l17_1703

/-- The minimum distance sum from a point on a parabola to its focus and an external point -/
theorem parabola_min_distance_sum (F : ℝ × ℝ) (B : ℝ × ℝ) :
  let parabola := {P : ℝ × ℝ | P.2^2 = 4 * P.1}
  F = (1, 0) →
  B = (3, 4) →
  (∃ (min : ℝ), ∀ (P : ℝ × ℝ), P ∈ parabola →
    Real.sqrt ((P.1 - B.1)^2 + (P.2 - B.2)^2) +
    Real.sqrt ((P.1 - F.1)^2 + (P.2 - F.2)^2) ≥ min ∧
    min = 2 * Real.sqrt 5) :=
by sorry

end parabola_min_distance_sum_l17_1703


namespace rectangle_cutting_l17_1765

theorem rectangle_cutting (m : ℕ) (h : m > 12) : 
  ∃ (x y : ℕ), x > 0 ∧ y > 0 ∧ x * y > m ∧ x * (y - 1) < m :=
by sorry

end rectangle_cutting_l17_1765


namespace basketball_game_scores_l17_1744

/-- Represents the scores of a team in a basketball game -/
structure TeamScores where
  q1 : ℕ
  q2 : ℕ
  q3 : ℕ
  q4 : ℕ

/-- Calculates the total score of a team -/
def totalScore (scores : TeamScores) : ℕ :=
  scores.q1 + scores.q2 + scores.q3 + scores.q4

/-- Calculates the score for the first half -/
def firstHalfScore (scores : TeamScores) : ℕ :=
  scores.q1 + scores.q2

/-- Calculates the score for the second half -/
def secondHalfScore (scores : TeamScores) : ℕ :=
  scores.q3 + scores.q4

/-- Checks if the scores form an increasing geometric sequence -/
def isGeometricSequence (scores : TeamScores) : Prop :=
  ∃ r : ℚ, r > 1 ∧ 
    scores.q2 = scores.q1 * r ∧
    scores.q3 = scores.q2 * r ∧
    scores.q4 = scores.q3 * r

/-- Checks if the scores form an increasing arithmetic sequence -/
def isArithmeticSequence (scores : TeamScores) : Prop :=
  ∃ d : ℕ, d > 0 ∧
    scores.q2 = scores.q1 + d ∧
    scores.q3 = scores.q2 + d ∧
    scores.q4 = scores.q3 + d

theorem basketball_game_scores 
  (eagles : TeamScores) (tigers : TeamScores) 
  (h1 : isGeometricSequence eagles)
  (h2 : isArithmeticSequence tigers)
  (h3 : firstHalfScore eagles = firstHalfScore tigers)
  (h4 : totalScore eagles = totalScore tigers + 2)
  (h5 : totalScore eagles ≤ 100)
  (h6 : totalScore tigers ≤ 100) :
  secondHalfScore eagles + secondHalfScore tigers = 116 := by
  sorry

end basketball_game_scores_l17_1744


namespace n_ge_digit_product_eq_digit_product_iff_eq_four_l17_1758

/-- Function that returns the product of digits of a positive integer -/
def digit_product (n : ℕ+) : ℕ :=
  sorry

/-- Theorem stating that n is greater than or equal to the product of its digits -/
theorem n_ge_digit_product (n : ℕ+) : (n : ℕ) ≥ digit_product n :=
  sorry

/-- Theorem stating that n^2 - 17n + 56 equals the product of digits of n if and only if n = 4 -/
theorem eq_digit_product_iff_eq_four (n : ℕ+) : 
  (n : ℕ)^2 - 17*(n : ℕ) + 56 = digit_product n ↔ n = 4 :=
  sorry

end n_ge_digit_product_eq_digit_product_iff_eq_four_l17_1758


namespace unique_solution_xyz_squared_l17_1720

theorem unique_solution_xyz_squared (x y z : ℕ) :
  x^2 + y^2 + z^2 = 2*x*y*z → x = 0 ∧ y = 0 ∧ z = 0 :=
by sorry

end unique_solution_xyz_squared_l17_1720


namespace bacteria_growth_30_min_l17_1710

/-- Calculates the bacterial population after a given number of 5-minute intervals -/
def bacterial_population (initial_population : ℕ) (num_intervals : ℕ) : ℕ :=
  initial_population * (3 ^ num_intervals)

/-- Theorem stating the bacterial population after 30 minutes -/
theorem bacteria_growth_30_min (initial_population : ℕ) 
  (h1 : initial_population = 50) : 
  bacterial_population initial_population 6 = 36450 := by
  sorry

#eval bacterial_population 50 6

end bacteria_growth_30_min_l17_1710


namespace open_box_volume_l17_1747

/-- The volume of an open box formed by cutting squares from the corners of a rectangular sheet. -/
theorem open_box_volume 
  (sheet_length : ℝ) 
  (sheet_width : ℝ) 
  (cut_length : ℝ) 
  (h1 : sheet_length = 48) 
  (h2 : sheet_width = 36) 
  (h3 : cut_length = 8) : 
  (sheet_length - 2 * cut_length) * (sheet_width - 2 * cut_length) * cut_length = 5120 := by
  sorry

#check open_box_volume

end open_box_volume_l17_1747


namespace rectangle_y_value_l17_1702

theorem rectangle_y_value (y : ℝ) : 
  y > 0 →  -- y is positive
  (5 - (-3)) * (y - 2) = 64 →  -- area of rectangle is 64
  y = 10 := by
sorry

end rectangle_y_value_l17_1702


namespace least_three_digit_with_digit_product_8_l17_1763

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

def digit_product (n : ℕ) : ℕ :=
  (n / 100) * ((n / 10) % 10) * (n % 10)

theorem least_three_digit_with_digit_product_8 :
  ∀ n : ℕ, is_three_digit n → digit_product n = 8 → 118 ≤ n :=
by sorry

end least_three_digit_with_digit_product_8_l17_1763


namespace percentage_decrease_l17_1722

/-- Proves that for an original number of 40, if the difference between its value 
    increased by 25% and its value decreased by x% is 22, then x = 30. -/
theorem percentage_decrease (x : ℝ) : 
  (40 + 0.25 * 40) - (40 - 0.01 * x * 40) = 22 → x = 30 := by
  sorry

end percentage_decrease_l17_1722


namespace right_triangle_from_equation_l17_1785

theorem right_triangle_from_equation (a b c : ℝ) 
  (triangle_sides : a > 0 ∧ b > 0 ∧ c > 0)
  (triangle_inequality : a + b > c ∧ b + c > a ∧ c + a > b)
  (h : (a - 6)^2 + Real.sqrt (b - 8) + |c - 10| = 0) : 
  a^2 + b^2 = c^2 := by
sorry

end right_triangle_from_equation_l17_1785


namespace impossibility_of_arrangement_l17_1704

theorem impossibility_of_arrangement : ¬ ∃ (a b : Fin 1986 → ℕ), 
  (∀ k : Fin 1986, b k - a k = k.val + 1) ∧
  (∀ i j : Fin 1986, i ≠ j → (a i ≠ a j ∧ a i ≠ b j ∧ b i ≠ b j)) ∧
  (∀ n : ℕ, n ∈ Set.range a ∪ Set.range b → n ≤ 2 * 1986) :=
by sorry

end impossibility_of_arrangement_l17_1704


namespace proposition_is_false_l17_1797

theorem proposition_is_false : 
  ¬(∀ x : ℝ, (x ≠ 1 ∨ x ≠ 2) → (x^2 - 3*x + 2 ≠ 0)) :=
by sorry

end proposition_is_false_l17_1797


namespace sledding_problem_l17_1729

/-- Sledding problem -/
theorem sledding_problem (mary_hill_length : ℝ) (mary_speed : ℝ) (ann_speed : ℝ) (time_difference : ℝ)
  (h1 : mary_hill_length = 630)
  (h2 : mary_speed = 90)
  (h3 : ann_speed = 40)
  (h4 : time_difference = 13) :
  let mary_time := mary_hill_length / mary_speed
  let ann_time := mary_time + time_difference
  ann_speed * ann_time = 800 :=
by sorry

end sledding_problem_l17_1729


namespace all_values_equal_l17_1773

-- Define the type for coordinates
def Coord := ℤ × ℤ

-- Define the type for the value assignment function
def ValueAssignment := Coord → ℕ

-- Define the property that each value is the average of its neighbors
def IsAverageOfNeighbors (f : ValueAssignment) : Prop :=
  ∀ (x y : ℤ), f (x, y) * 4 = f (x + 1, y) + f (x - 1, y) + f (x, y + 1) + f (x, y - 1)

-- State the theorem
theorem all_values_equal (f : ValueAssignment) (h : IsAverageOfNeighbors f) :
  ∀ (x₁ y₁ x₂ y₂ : ℤ), f (x₁, y₁) = f (x₂, y₂) := by
  sorry

end all_values_equal_l17_1773


namespace sin_390_degrees_l17_1753

theorem sin_390_degrees : Real.sin (390 * π / 180) = 1 / 2 := by
  have h1 : ∀ x, Real.sin (x + 2 * π) = Real.sin x := by sorry
  have h2 : Real.sin (π / 6) = 1 / 2 := by sorry
  sorry

end sin_390_degrees_l17_1753


namespace base13_addition_proof_l17_1724

/-- Represents a digit in base 13 -/
inductive Base13Digit
  | D0 | D1 | D2 | D3 | D4 | D5 | D6 | D7 | D8 | D9 | A | B | C

/-- Represents a number in base 13 -/
def Base13Number := List Base13Digit

/-- Addition of two Base13Numbers -/
def add_base13 : Base13Number → Base13Number → Base13Number
  | _, _ => sorry  -- Implementation details omitted

/-- Conversion of a natural number to Base13Number -/
def nat_to_base13 : Nat → Base13Number
  | _ => sorry  -- Implementation details omitted

theorem base13_addition_proof :
  add_base13 (nat_to_base13 528) (nat_to_base13 274) =
  [Base13Digit.D7, Base13Digit.A, Base13Digit.C] :=
by sorry

end base13_addition_proof_l17_1724


namespace second_number_value_l17_1768

theorem second_number_value (a b c : ℝ) : 
  a + b + c = 550 →
  a = 2 * b →
  c = (1 / 3) * a →
  b = 150 := by
sorry

end second_number_value_l17_1768


namespace john_and_alice_money_sum_l17_1705

theorem john_and_alice_money_sum :
  let john_money : ℚ := 5 / 8
  let alice_money : ℚ := 7 / 20
  (john_money + alice_money : ℚ) = 39 / 40 := by
  sorry

end john_and_alice_money_sum_l17_1705


namespace point_not_in_second_quadrant_l17_1700

theorem point_not_in_second_quadrant (a : ℝ) :
  ¬(a < 0 ∧ 2*a - 1 > 0) :=
by sorry

end point_not_in_second_quadrant_l17_1700


namespace bouquet_stamens_l17_1764

/-- Proves that the total number of stamens in a bouquet is 216 --/
theorem bouquet_stamens :
  ∀ (black_roses crimson_flowers : ℕ),
  (4 * black_roses + 8 * crimson_flowers) - (2 * black_roses + 3 * crimson_flowers) = 108 →
  4 * black_roses + 10 * crimson_flowers = 216 :=
by
  sorry

end bouquet_stamens_l17_1764


namespace last_three_digits_are_427_l17_1725

/-- A function that generates the nth digit in the list of increasing positive integers 
    starting with 2 and containing all numbers with a first digit of 2 -/
def nthDigitInList (n : ℕ) : ℕ := sorry

/-- The last three digits of the 2000-digit sequence -/
def lastThreeDigits : ℕ × ℕ × ℕ := (nthDigitInList 1998, nthDigitInList 1999, nthDigitInList 2000)

theorem last_three_digits_are_427 : lastThreeDigits = (4, 2, 7) := by sorry

end last_three_digits_are_427_l17_1725


namespace unique_x_value_l17_1736

/-- Binary operation ⋆ on pairs of integers -/
def star : (ℤ × ℤ) → (ℤ × ℤ) → (ℤ × ℤ) := 
  fun (a, b) (c, d) ↦ (a + c, b - d)

/-- Theorem stating the unique value of x satisfying the equation -/
theorem unique_x_value : 
  ∃! x : ℤ, star (x, 4) (2, 1) = star (5, 2) (1, -3) := by sorry

end unique_x_value_l17_1736


namespace cube_root_three_irrational_l17_1706

theorem cube_root_three_irrational :
  ¬ (∃ (p q : ℤ), q ≠ 0 ∧ (3 : ℝ) ^ (1/3 : ℝ) = (p : ℝ) / (q : ℝ)) :=
by sorry

end cube_root_three_irrational_l17_1706


namespace negation_of_existence_proposition_l17_1750

theorem negation_of_existence_proposition :
  (¬ ∃ x : ℝ, x > 0 ∧ Real.log x ≤ x + 1) ↔ (∀ x : ℝ, x > 0 → Real.log x > x + 1) := by
  sorry

end negation_of_existence_proposition_l17_1750


namespace jason_attended_twelve_games_l17_1732

/-- The number of games Jason attended given his planned and missed games -/
def games_attended (planned_this_month : ℕ) (planned_last_month : ℕ) (missed : ℕ) : ℕ :=
  (planned_this_month + planned_last_month) - missed

/-- Theorem stating that Jason attended 12 games given the problem conditions -/
theorem jason_attended_twelve_games :
  games_attended 11 17 16 = 12 := by
  sorry

end jason_attended_twelve_games_l17_1732


namespace divisors_of_12m_squared_l17_1769

def is_even (n : ℕ) : Prop := ∃ k, n = 2 * k

def count_divisors (n : ℕ) : ℕ := (Finset.filter (· ∣ n) (Finset.range (n + 1))).card

theorem divisors_of_12m_squared (m : ℕ) 
  (h_even : is_even m) 
  (h_divisors : count_divisors m = 7) : 
  count_divisors (12 * m^2) = 30 := by
  sorry

end divisors_of_12m_squared_l17_1769


namespace decimal_to_fraction_sum_l17_1777

theorem decimal_to_fraction_sum (a b : ℕ+) : 
  (a : ℚ) / (b : ℚ) = 0.425875 ∧ 
  ∀ (c d : ℕ+), (c : ℚ) / (d : ℚ) = 0.425875 → c ≤ a ∧ d ≤ b → 
  (a : ℕ) + (b : ℕ) = 11407 := by
sorry

end decimal_to_fraction_sum_l17_1777


namespace triangle_inequality_expression_l17_1770

theorem triangle_inequality_expression (a b c : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (hab : a + b > c) (hbc : b + c > a) (hca : c + a > b) :
  a^4 + b^4 + c^4 - 2*a^2*b^2 - 2*b^2*c^2 - 2*c^2*a^2 < 0 := by
  sorry

end triangle_inequality_expression_l17_1770


namespace shaded_area_problem_l17_1796

theorem shaded_area_problem (diagonal : ℝ) (num_squares : ℕ) : 
  diagonal = 10 → num_squares = 25 → 
  (diagonal^2 / 2) = (num_squares : ℝ) * (diagonal^2 / (2 * num_squares : ℝ)) := by
  sorry

end shaded_area_problem_l17_1796


namespace trajectory_and_intersection_l17_1708

-- Define the circles
def circle1 (x y : ℝ) : Prop := x^2 + y^2 - 2*y = 0
def circle2 (x y : ℝ) : Prop := x^2 + (y+1)^2 = 4

-- Define the centers of the circles
def center1 : ℝ × ℝ := (0, 1)
def center2 : ℝ × ℝ := (0, -1)

-- Define the condition for point P
def point_condition (x y : ℝ) : Prop :=
  x ≠ 0 → ((y - 1) / x) * ((y + 1) / x) = -1/2

-- Define the trajectory equation
def trajectory (x y : ℝ) : Prop := x^2/2 + y^2 = 1

-- State the theorem
theorem trajectory_and_intersection :
  -- Part 1: Trajectory equation
  (∀ x y : ℝ, point_condition x y → trajectory x y) ∧
  -- Part 2: Line x = 0 intersects at two points with equal distance from C₁
  (∃ C D : ℝ × ℝ,
    C.1 = 0 ∧ D.1 = 0 ∧
    C ≠ D ∧
    trajectory C.1 C.2 ∧
    trajectory D.1 D.2 ∧
    (C.1 - center1.1)^2 + (C.2 - center1.2)^2 =
    (D.1 - center1.1)^2 + (D.2 - center1.2)^2) :=
by sorry

end trajectory_and_intersection_l17_1708


namespace min_value_a_plus_b_l17_1739

theorem min_value_a_plus_b (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 3 * a + b = 2 * a * b) :
  ∀ x y, x > 0 → y > 0 → 3 * x + y = 2 * x * y → a + b ≤ x + y ∧ a + b = 2 + Real.sqrt 3 :=
sorry

end min_value_a_plus_b_l17_1739


namespace sixth_day_work_time_l17_1794

def work_time (n : ℕ) : ℕ := 15 * 2^(n - 1)

theorem sixth_day_work_time :
  work_time 6 = 8 * 60 := by
  sorry

end sixth_day_work_time_l17_1794


namespace jelly_bean_probability_l17_1783

theorem jelly_bean_probability (p_red p_orange p_yellow p_green : ℝ) :
  p_red = 0.15 →
  p_orange = 0.35 →
  p_red + p_orange + p_yellow + p_green = 1 →
  p_yellow + p_green = 0.5 := by
sorry

end jelly_bean_probability_l17_1783


namespace complex_fraction_equality_l17_1774

theorem complex_fraction_equality : (1 / (1 + 1 / (2 + 1 / 3))) = 7 / 10 := by
  sorry

end complex_fraction_equality_l17_1774


namespace guest_author_payment_l17_1759

theorem guest_author_payment (B : ℕ) (h1 : B < 10) (h2 : B > 0) 
  (h3 : (200 + 10 * B) % 14 = 0) : B = 8 := by
  sorry

end guest_author_payment_l17_1759


namespace total_jogging_distance_l17_1726

def monday_distance : ℕ := 2
def tuesday_distance : ℕ := 5
def wednesday_distance : ℕ := 9

theorem total_jogging_distance :
  monday_distance + tuesday_distance + wednesday_distance = 16 := by
  sorry

end total_jogging_distance_l17_1726


namespace line_equation_slope_intercept_l17_1799

/-- Given a line equation, prove its slope and y-intercept -/
theorem line_equation_slope_intercept :
  ∀ (x y : ℝ),
  (3 : ℝ) * (x - 2) + (-4 : ℝ) * (y - 8) = 0 →
  ∃ (m b : ℝ), m = 3/4 ∧ b = 13/2 ∧ y = m * x + b :=
by sorry

end line_equation_slope_intercept_l17_1799


namespace ellipse_focal_property_l17_1749

/-- The ellipse with semi-major axis 13 and semi-minor axis 12 -/
def Ellipse : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1^2 / 169) + (p.2^2 / 144) = 1}

/-- The foci of the ellipse -/
def Foci : Set (ℝ × ℝ) := sorry

/-- The distance between two points in ℝ² -/
def distance (p q : ℝ × ℝ) : ℝ := sorry

theorem ellipse_focal_property (P : ℝ × ℝ) (F₁ F₂ : ℝ × ℝ) :
  P ∈ Ellipse → F₁ ∈ Foci → F₂ ∈ Foci → distance P F₁ = 4 →
  distance P F₂ = 22 := by sorry

end ellipse_focal_property_l17_1749


namespace candy_distribution_l17_1787

theorem candy_distribution (total_candies : ℕ) (sour_percentage : ℚ) (num_people : ℕ) :
  total_candies = 300 →
  sour_percentage = 40 / 100 →
  num_people = 3 →
  (total_candies - (sour_percentage * total_candies).floor) / num_people = 60 := by
sorry

end candy_distribution_l17_1787


namespace seating_arrangement_for_100_people_l17_1714

/-- Represents a seating arrangement with rows of 9 or 10 people -/
structure SeatingArrangement where
  rows_of_10 : ℕ
  rows_of_9 : ℕ

/-- The total number of people in the seating arrangement -/
def total_people (s : SeatingArrangement) : ℕ :=
  10 * s.rows_of_10 + 9 * s.rows_of_9

/-- The theorem stating that for 100 people, there are 10 rows of 10 people -/
theorem seating_arrangement_for_100_people :
  ∃ (s : SeatingArrangement), total_people s = 100 ∧ s.rows_of_10 = 10 := by
  sorry

end seating_arrangement_for_100_people_l17_1714


namespace right_triangles_count_l17_1756

/-- Represents a geometric solid with front, top, and side views -/
structure GeometricSolid where
  front_view : Set (Point × Point)
  top_view : Set (Point × Point)
  side_view : Set (Point × Point)

/-- Counts the number of unique right-angled triangles in a geometric solid -/
def count_right_triangles (solid : GeometricSolid) : ℕ :=
  sorry

/-- Theorem stating that the number of right-angled triangles is 3 -/
theorem right_triangles_count (solid : GeometricSolid) :
  count_right_triangles solid = 3 :=
sorry

end right_triangles_count_l17_1756


namespace y_intercept_for_specific_line_l17_1712

/-- A line in a 2D plane. -/
structure Line where
  slope : ℝ
  x_intercept : ℝ × ℝ

/-- The y-intercept of a line. -/
def y_intercept (l : Line) : ℝ × ℝ := sorry

/-- Theorem: For a line with slope -3 and x-intercept (7,0), the y-intercept is (0,21). -/
theorem y_intercept_for_specific_line :
  let l : Line := { slope := -3, x_intercept := (7, 0) }
  y_intercept l = (0, 21) := by sorry

end y_intercept_for_specific_line_l17_1712


namespace power_difference_in_set_l17_1781

theorem power_difference_in_set (m n : ℕ) :
  (3 ^ m - 2 ^ n ∈ ({-1, 5, 7} : Set ℤ)) ↔ 
  ((m, n) ∈ ({(0, 1), (1, 2), (2, 2), (2, 1)} : Set (ℕ × ℕ))) := by
  sorry

end power_difference_in_set_l17_1781


namespace quadratic_root_range_l17_1767

theorem quadratic_root_range (k : ℝ) : 
  (∃ α β : ℝ, 
    (7 * α^2 - (k + 13) * α + k^2 - k - 2 = 0) ∧ 
    (7 * β^2 - (k + 13) * β + k^2 - k - 2 = 0) ∧ 
    (0 < α) ∧ (α < 1) ∧ (1 < β) ∧ (β < 2)) →
  ((3 < k ∧ k < 4) ∨ (-2 < k ∧ k < -1)) :=
by sorry

end quadratic_root_range_l17_1767


namespace birds_cannot_all_be_on_same_tree_l17_1727

/-- Represents the state of birds on trees -/
structure BirdState where
  white : Nat -- Number of birds on white trees
  black : Nat -- Number of birds on black trees

/-- A move represents two birds switching to neighboring trees -/
def move (state : BirdState) : BirdState :=
  { white := state.white, black := state.black }

theorem birds_cannot_all_be_on_same_tree :
  ∀ (n : Nat), n > 0 →
  let initial_state : BirdState := { white := 3, black := 3 }
  let final_state := (move^[n]) initial_state
  (final_state.white ≠ 0 ∧ final_state.black ≠ 6) ∧
  (final_state.white ≠ 6 ∧ final_state.black ≠ 0) :=
by sorry

end birds_cannot_all_be_on_same_tree_l17_1727


namespace robin_water_consumption_l17_1746

def bottles_morning : ℕ := sorry
def bottles_afternoon : ℕ := sorry
def total_bottles : ℕ := 14

theorem robin_water_consumption :
  (bottles_morning = bottles_afternoon) →
  (bottles_morning + bottles_afternoon = total_bottles) →
  bottles_morning = 7 := by
  sorry

end robin_water_consumption_l17_1746


namespace point_in_fourth_quadrant_l17_1719

theorem point_in_fourth_quadrant (a : ℝ) (h : a > 1) :
  let P : ℝ × ℝ := (1 + a, 1 - a)
  P.1 > 0 ∧ P.2 < 0 :=
by sorry

end point_in_fourth_quadrant_l17_1719


namespace left_handed_to_non_throwers_ratio_l17_1740

/- Define the football team -/
def total_players : ℕ := 70
def throwers : ℕ := 37
def right_handed : ℕ := 59

/- Theorem to prove the ratio -/
theorem left_handed_to_non_throwers_ratio :
  let non_throwers := total_players - throwers
  let right_handed_non_throwers := right_handed - throwers
  let left_handed := non_throwers - right_handed_non_throwers
  (left_handed : ℚ) / non_throwers = 1 / 3 := by
  sorry

end left_handed_to_non_throwers_ratio_l17_1740
