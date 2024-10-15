import Mathlib

namespace NUMINAMATH_CALUDE_m_greater_than_n_l520_52079

theorem m_greater_than_n (a : ℝ) : 2 * a * (a - 2) + 7 > (a - 2) * (a - 3) := by
  sorry

end NUMINAMATH_CALUDE_m_greater_than_n_l520_52079


namespace NUMINAMATH_CALUDE_F_of_2_f_of_3_equals_341_l520_52039

-- Define the functions f and F
def f (a : ℝ) : ℝ := a^2 - 2
def F (a b : ℝ) : ℝ := b^3 - a

-- Theorem statement
theorem F_of_2_f_of_3_equals_341 : F 2 (f 3) = 341 := by
  sorry

end NUMINAMATH_CALUDE_F_of_2_f_of_3_equals_341_l520_52039


namespace NUMINAMATH_CALUDE_prime_sum_theorem_l520_52013

theorem prime_sum_theorem (a b c : ℕ) : 
  Nat.Prime a → Nat.Prime b → Nat.Prime c → 
  b + c = 13 → c^2 - a^2 = 72 → 
  a + b + c = 20 := by
  sorry

end NUMINAMATH_CALUDE_prime_sum_theorem_l520_52013


namespace NUMINAMATH_CALUDE_parallelepiped_properties_l520_52083

/-- Represents an oblique parallelepiped with given properties -/
structure ObliqueParallelepiped where
  lateral_edge_projection : ℝ
  height : ℝ
  rhombus_area : ℝ
  rhombus_diagonal : ℝ

/-- Calculates the lateral surface area of the parallelepiped -/
def lateral_surface_area (p : ObliqueParallelepiped) : ℝ := sorry

/-- Calculates the volume of the parallelepiped -/
def volume (p : ObliqueParallelepiped) : ℝ := sorry

/-- Theorem stating the lateral surface area and volume of the given parallelepiped -/
theorem parallelepiped_properties :
  let p : ObliqueParallelepiped := {
    lateral_edge_projection := 5,
    height := 12,
    rhombus_area := 24,
    rhombus_diagonal := 8
  }
  lateral_surface_area p = 260 ∧ volume p = 312 := by sorry

end NUMINAMATH_CALUDE_parallelepiped_properties_l520_52083


namespace NUMINAMATH_CALUDE_inequality_proof_l520_52014

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h : a * b + b * c + c * a + 2 * a * b * c = 1) :
  Real.sqrt (a * b) + Real.sqrt (b * c) + Real.sqrt (c * a) ≤ 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l520_52014


namespace NUMINAMATH_CALUDE_trapezoid_side_length_l520_52078

/-- Represents a trapezoid ABCD with specific properties -/
structure Trapezoid where
  -- Length of side AB
  ab : ℝ
  -- Length of side CD
  cd : ℝ
  -- The ratio of the area of triangle ABC to the area of triangle ADC is 4:1
  area_ratio : ab / cd = 4
  -- The sum of AB and CD is 250
  sum_sides : ab + cd = 250

/-- 
Theorem: In a trapezoid ABCD, if the ratio of the area of triangle ABC 
to the area of triangle ADC is 4:1, and AB + CD = 250 cm, then AB = 200 cm.
-/
theorem trapezoid_side_length (t : Trapezoid) : t.ab = 200 := by
  sorry

end NUMINAMATH_CALUDE_trapezoid_side_length_l520_52078


namespace NUMINAMATH_CALUDE_first_system_solution_second_system_solution_l520_52012

-- First system of equations
theorem first_system_solution :
  ∃ (x y : ℝ), 3 * x + 2 * y = 5 ∧ y = 2 * x - 8 ∧ x = 3 ∧ y = -2 := by
sorry

-- Second system of equations
theorem second_system_solution :
  ∃ (x y : ℝ), 2 * x - y = 10 ∧ 2 * x + 3 * y = 2 ∧ x = 4 ∧ y = -2 := by
sorry

end NUMINAMATH_CALUDE_first_system_solution_second_system_solution_l520_52012


namespace NUMINAMATH_CALUDE_position_determination_in_plane_l520_52034

theorem position_determination_in_plane :
  ∀ (P : ℝ × ℝ), ∃! (θ : ℝ) (r : ℝ), 
    P.1 = r * Real.cos θ ∧ P.2 = r * Real.sin θ ∧ r ≥ 0 :=
by sorry

end NUMINAMATH_CALUDE_position_determination_in_plane_l520_52034


namespace NUMINAMATH_CALUDE_solve_for_k_l520_52042

theorem solve_for_k (x y k : ℝ) : 
  x = -3 ∧ y = 2 ∧ 2 * x + k * y = 6 → k = 6 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_k_l520_52042


namespace NUMINAMATH_CALUDE_problem_solution_l520_52040

def f (a : ℝ) (x : ℝ) : ℝ := |2*x - a| + |2*x + 3|
def g (x : ℝ) : ℝ := |x - 1| + 3

theorem problem_solution :
  (∀ x : ℝ, |g x| < 5 ↔ x ∈ Set.Ioo (-1) 3) ∧
  (∀ a : ℝ, (∀ x₁ : ℝ, ∃ x₂ : ℝ, f a x₁ = g x₂) →
    a ∈ Set.Iic (-6) ∪ Set.Ici 0) :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l520_52040


namespace NUMINAMATH_CALUDE_polynomial_product_l520_52091

theorem polynomial_product (a a₁ a₂ a₃ a₄ a₅ : ℝ) :
  (∀ x : ℝ, (1 - x)^2 = a + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5) →
  (a + a₂ + a₄) * (a₁ + a₃ + a₅) = -256 := by
sorry

end NUMINAMATH_CALUDE_polynomial_product_l520_52091


namespace NUMINAMATH_CALUDE_equation_solution_l520_52051

theorem equation_solution : ∃ (x : ℝ), (3 / (x - 2) - 1 = 1 / (2 - x)) ∧ (x = 6) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l520_52051


namespace NUMINAMATH_CALUDE_partnership_profit_l520_52074

/-- The total profit of a business partnership --/
def total_profit (a_investment b_investment : ℤ) (management_fee_percent : ℚ) (a_total_received : ℤ) : ℚ :=
  let total_investment := a_investment + b_investment
  let a_share_percent := a_investment / total_investment
  let remaining_profit_percent := 1 - management_fee_percent
  let a_total_percent := management_fee_percent + (a_share_percent * remaining_profit_percent)
  (a_total_received : ℚ) / a_total_percent

/-- The proposition that the total profit is 9600 given the specified conditions --/
theorem partnership_profit : 
  total_profit 15000 25000 (1/10) 4200 = 9600 := by
  sorry

end NUMINAMATH_CALUDE_partnership_profit_l520_52074


namespace NUMINAMATH_CALUDE_a_seven_minus_a_two_l520_52050

def S (n : ℕ) : ℤ := 2 * n^2 - 3 * n

def a (n : ℕ) : ℤ := S n - S (n - 1)

theorem a_seven_minus_a_two : a 7 - a 2 = 20 := by
  sorry

end NUMINAMATH_CALUDE_a_seven_minus_a_two_l520_52050


namespace NUMINAMATH_CALUDE_symmetry_of_exponential_graphs_l520_52043

theorem symmetry_of_exponential_graphs :
  ∀ a : ℝ, 
  let f : ℝ → ℝ := λ x => 3^x
  let g : ℝ → ℝ := λ x => -(3^(-x))
  (f a = 3^a ∧ g (-a) = -3^a) ∧ 
  ((-a, -f a) = (-1 : ℝ) • (a, f a)) := by sorry

end NUMINAMATH_CALUDE_symmetry_of_exponential_graphs_l520_52043


namespace NUMINAMATH_CALUDE_perpendicular_vectors_x_value_l520_52061

theorem perpendicular_vectors_x_value 
  (x : ℝ) 
  (a b : ℝ × ℝ) 
  (ha : a = (x, 3)) 
  (hb : b = (2, x - 5)) 
  (h_perp : a.1 * b.1 + a.2 * b.2 = 0) : 
  x = 3 := by
sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_x_value_l520_52061


namespace NUMINAMATH_CALUDE_ten_students_both_activities_l520_52004

/-- Calculates the number of students who can do both swimming and gymnastics -/
def students_both_activities (total : ℕ) (swim : ℕ) (gym : ℕ) (neither : ℕ) : ℕ :=
  total - (total - swim + total - gym - neither)

/-- Theorem stating that 10 students can do both swimming and gymnastics -/
theorem ten_students_both_activities :
  students_both_activities 60 27 28 15 = 10 := by
  sorry

end NUMINAMATH_CALUDE_ten_students_both_activities_l520_52004


namespace NUMINAMATH_CALUDE_solve_exponential_equation_l520_52097

theorem solve_exponential_equation :
  ∃ x : ℝ, (2 : ℝ) ^ (x - 3) = 4 ^ (x + 1) ∧ x = -5 := by
  sorry

end NUMINAMATH_CALUDE_solve_exponential_equation_l520_52097


namespace NUMINAMATH_CALUDE_cyclic_sum_inequality_l520_52069

open Real

/-- The cyclic sum of a function over five variables -/
def cyclicSum (f : ℝ → ℝ → ℝ → ℝ → ℝ → ℝ) (a b c d e : ℝ) : ℝ :=
  f a b c d e + f b c d e a + f c d e a b + f d e a b c + f e a b c d

/-- Theorem: For positive real numbers a, b, c, d, e satisfying abcde = 1,
    the cyclic sum of (a + abc)/(1 + ab + abcd) is greater than or equal to 10/3 -/
theorem cyclic_sum_inequality (a b c d e : ℝ) 
    (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) (he : e > 0)
    (h_prod : a * b * c * d * e = 1) :
    cyclicSum (fun a b c d e => (a + a*b*c)/(1 + a*b + a*b*c*d)) a b c d e ≥ 10/3 := by
  sorry

end NUMINAMATH_CALUDE_cyclic_sum_inequality_l520_52069


namespace NUMINAMATH_CALUDE_A_not_necessary_for_B_A_not_sufficient_for_B_A_neither_necessary_nor_sufficient_for_B_l520_52092

-- Define condition A
def condition_A (x y : ℝ) : Prop := x ≠ 1 ∧ y ≠ 2

-- Define condition B
def condition_B (x y : ℝ) : Prop := x + y ≠ 3

-- Theorem stating that A is not necessary for B
theorem A_not_necessary_for_B : ¬∀ x y : ℝ, condition_B x y → condition_A x y := by
  sorry

-- Theorem stating that A is not sufficient for B
theorem A_not_sufficient_for_B : ¬∀ x y : ℝ, condition_A x y → condition_B x y := by
  sorry

-- Main theorem combining the above results
theorem A_neither_necessary_nor_sufficient_for_B :
  (¬∀ x y : ℝ, condition_B x y → condition_A x y) ∧
  (¬∀ x y : ℝ, condition_A x y → condition_B x y) := by
  sorry

end NUMINAMATH_CALUDE_A_not_necessary_for_B_A_not_sufficient_for_B_A_neither_necessary_nor_sufficient_for_B_l520_52092


namespace NUMINAMATH_CALUDE_functional_equation_solution_l520_52009

/-- Given functions f, g, h: ℝ → ℝ satisfying the functional equation
    f(x) - g(y) = (x-y) · h(x+y) for all x, y ∈ ℝ,
    prove that there exist constants d, c ∈ ℝ such that
    f(x) = g(x) = dx² + c for all x ∈ ℝ. -/
theorem functional_equation_solution
  (f g h : ℝ → ℝ)
  (h_eq : ∀ x y : ℝ, f x - g y = (x - y) * h (x + y)) :
  ∃ d c : ℝ, ∀ x : ℝ, f x = d * x^2 + c ∧ g x = d * x^2 + c :=
sorry

end NUMINAMATH_CALUDE_functional_equation_solution_l520_52009


namespace NUMINAMATH_CALUDE_remainder_of_12345678_div_9_l520_52048

theorem remainder_of_12345678_div_9 : 12345678 % 9 = 0 := by
  sorry

end NUMINAMATH_CALUDE_remainder_of_12345678_div_9_l520_52048


namespace NUMINAMATH_CALUDE_final_pet_count_l520_52070

/-- Represents the number of pets in the pet center -/
structure PetCount where
  dogs : ℕ
  cats : ℕ
  rabbits : ℕ
  birds : ℕ

/-- Calculates the total number of pets -/
def totalPets (pets : PetCount) : ℕ :=
  pets.dogs + pets.cats + pets.rabbits + pets.birds

/-- Initial pet count -/
def initialPets : PetCount :=
  { dogs := 36, cats := 29, rabbits := 15, birds := 10 }

/-- First adoption -/
def firstAdoption (pets : PetCount) : PetCount :=
  { dogs := pets.dogs - 20, cats := pets.cats, rabbits := pets.rabbits - 5, birds := pets.birds }

/-- New pets added -/
def newPetsAdded (pets : PetCount) : PetCount :=
  { dogs := pets.dogs, cats := pets.cats + 12, rabbits := pets.rabbits + 8, birds := pets.birds + 5 }

/-- Second adoption -/
def secondAdoption (pets : PetCount) : PetCount :=
  { dogs := pets.dogs, cats := pets.cats - 10, rabbits := pets.rabbits, birds := pets.birds - 4 }

/-- The main theorem stating the final number of pets -/
theorem final_pet_count :
  totalPets (secondAdoption (newPetsAdded (firstAdoption initialPets))) = 76 := by
  sorry

end NUMINAMATH_CALUDE_final_pet_count_l520_52070


namespace NUMINAMATH_CALUDE_nut_storage_impact_l520_52038

/-- Represents the types of nuts found in Mason's car -/
inductive NutType
  | Almond
  | Walnut
  | Hazelnut

/-- Represents the squirrels and their nut-storing behavior -/
structure Squirrel where
  nutType : NutType
  count : Nat
  nutsPerDay : Nat
  days : Nat

/-- Calculates the total number of nuts stored by a group of squirrels -/
def totalNuts (s : Squirrel) : Nat :=
  s.count * s.nutsPerDay * s.days

/-- Calculates the weight of a single nut in grams -/
def nutWeight (n : NutType) : Rat :=
  match n with
  | NutType.Almond => 1/2
  | NutType.Walnut => 10
  | NutType.Hazelnut => 2

/-- Calculates the total weight of nuts stored by a group of squirrels -/
def totalWeight (s : Squirrel) : Rat :=
  (totalNuts s : Rat) * nutWeight s.nutType

/-- Calculates the efficiency reduction based on the total weight of nuts -/
def efficiencyReduction (totalWeight : Rat) : Rat :=
  min 100 (totalWeight / 100)

/-- The main theorem stating the total weight of nuts and efficiency reduction -/
theorem nut_storage_impact (almondSquirrels walnutSquirrels hazelnutSquirrels : Squirrel) 
    (h1 : almondSquirrels = ⟨NutType.Almond, 2, 30, 35⟩)
    (h2 : walnutSquirrels = ⟨NutType.Walnut, 3, 20, 40⟩)
    (h3 : hazelnutSquirrels = ⟨NutType.Hazelnut, 1, 10, 45⟩) :
    totalWeight almondSquirrels + totalWeight walnutSquirrels + totalWeight hazelnutSquirrels = 25950 ∧
    efficiencyReduction (totalWeight almondSquirrels + totalWeight walnutSquirrels + totalWeight hazelnutSquirrels) = 100 := by
  sorry


end NUMINAMATH_CALUDE_nut_storage_impact_l520_52038


namespace NUMINAMATH_CALUDE_base7_subtraction_l520_52007

/-- Converts a base-7 number to decimal --/
def toDecimal (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * 7^i) 0

/-- Converts a decimal number to base-7 --/
def toBase7 (n : Nat) : List Nat :=
  if n = 0 then [0] else
  let rec aux (m : Nat) (acc : List Nat) :=
    if m = 0 then acc else aux (m / 7) ((m % 7) :: acc)
  aux n []

theorem base7_subtraction :
  let a := [5, 5, 2, 1]  -- 1255 in base 7
  let b := [2, 3, 4]     -- 432 in base 7
  let c := [1, 2, 5]     -- 521 in base 7
  toBase7 (toDecimal a - toDecimal b) = c := by sorry

end NUMINAMATH_CALUDE_base7_subtraction_l520_52007


namespace NUMINAMATH_CALUDE_percentage_increase_l520_52000

theorem percentage_increase (B C : ℝ) (h1 : C = B - 30) : 
  let A := 3 * B
  100 * (A - C) / C = 200 + 9000 / C := by
  sorry

end NUMINAMATH_CALUDE_percentage_increase_l520_52000


namespace NUMINAMATH_CALUDE_system_solution_l520_52087

theorem system_solution :
  ∃ (x y : ℚ), 4 * x - 3 * y = 2 ∧ 5 * x + y = (3 / 2) ∧ x = (13 / 38) ∧ y = (-4 / 19) := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l520_52087


namespace NUMINAMATH_CALUDE_roots_property_l520_52062

-- Define the quadratic equation
def quadratic_eq (x : ℝ) : Prop := 3 * x^2 + 5 * x - 7 = 0

-- Define the theorem
theorem roots_property (p q : ℝ) (hp : quadratic_eq p) (hq : quadratic_eq q) :
  (p - 2) * (q - 2) = 5 := by
  sorry

end NUMINAMATH_CALUDE_roots_property_l520_52062


namespace NUMINAMATH_CALUDE_chef_guests_problem_l520_52046

theorem chef_guests_problem (adults children seniors : ℕ) : 
  children = adults - 35 →
  seniors = 2 * children →
  adults + children + seniors = 127 →
  adults = 58 := by
  sorry

end NUMINAMATH_CALUDE_chef_guests_problem_l520_52046


namespace NUMINAMATH_CALUDE_no_14_cents_combination_l520_52072

/-- Represents the types of coins available -/
inductive Coin
  | Penny
  | Nickel
  | Dime
  | Quarter

/-- Returns the value of a coin in cents -/
def coinValue (c : Coin) : ℕ :=
  match c with
  | Coin.Penny => 1
  | Coin.Nickel => 5
  | Coin.Dime => 10
  | Coin.Quarter => 25

/-- A selection of coins is represented as a list of Coins -/
def CoinSelection := List Coin

/-- Calculates the total value of a coin selection in cents -/
def totalValue (selection : CoinSelection) : ℕ :=
  selection.map coinValue |>.sum

/-- Theorem stating that it's impossible to select 6 coins totaling 14 cents -/
theorem no_14_cents_combination :
  ∀ (selection : CoinSelection),
    selection.length = 6 →
    totalValue selection ≠ 14 :=
by sorry

end NUMINAMATH_CALUDE_no_14_cents_combination_l520_52072


namespace NUMINAMATH_CALUDE_geraldo_tea_consumption_l520_52022

-- Define the conversion factor from gallons to pints
def gallons_to_pints : ℝ := 8

-- Define the total amount of tea in gallons
def total_tea : ℝ := 20

-- Define the number of containers
def num_containers : ℝ := 80

-- Define the number of containers Geraldo drank
def containers_drunk : ℝ := 3.5

-- Theorem statement
theorem geraldo_tea_consumption :
  (total_tea / num_containers) * containers_drunk * gallons_to_pints = 7 := by
  sorry

end NUMINAMATH_CALUDE_geraldo_tea_consumption_l520_52022


namespace NUMINAMATH_CALUDE_max_median_is_four_point_five_l520_52093

/-- Represents the soda shop scenario -/
structure SodaShop where
  total_cans : ℕ
  total_customers : ℕ
  min_cans_per_customer : ℕ
  h_total_cans : total_cans = 310
  h_total_customers : total_customers = 120
  h_min_cans : min_cans_per_customer = 1

/-- Calculates the maximum possible median number of cans bought per customer -/
def max_median_cans (shop : SodaShop) : ℚ :=
  sorry

/-- Theorem stating that the maximum possible median is 4.5 -/
theorem max_median_is_four_point_five (shop : SodaShop) :
  max_median_cans shop = 4.5 := by
  sorry

end NUMINAMATH_CALUDE_max_median_is_four_point_five_l520_52093


namespace NUMINAMATH_CALUDE_grape_pickers_l520_52015

/-- Given information about grape pickers and their work rate, calculate the number of pickers. -/
theorem grape_pickers (total_drums : ℕ) (total_days : ℕ) (drums_per_day : ℕ) :
  total_drums = 90 →
  total_days = 6 →
  drums_per_day = 15 →
  (total_drums / total_days : ℚ) = drums_per_day →
  drums_per_day / drums_per_day = 1 :=
by sorry

end NUMINAMATH_CALUDE_grape_pickers_l520_52015


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l520_52057

/-- The equation of a potential hyperbola with parameter k -/
def hyperbola_equation (k : ℝ) (x y : ℝ) : Prop :=
  x^2 / (k - 3) - y^2 / (k + 3) = 1

/-- Predicate to check if an equation represents a hyperbola -/
def is_hyperbola (k : ℝ) : Prop :=
  ∃ x y, hyperbola_equation k x y ∧ (k - 3) * (k + 3) > 0

/-- Statement: k > 3 is a sufficient but not necessary condition for the equation to represent a hyperbola -/
theorem sufficient_not_necessary_condition :
  (∀ k : ℝ, k > 3 → is_hyperbola k) ∧
  ¬(∀ k : ℝ, is_hyperbola k → k > 3) :=
sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l520_52057


namespace NUMINAMATH_CALUDE_work_scaling_l520_52029

theorem work_scaling (people₁ work₁ days : ℕ) (people₂ : ℕ) :
  people₁ > 0 →
  work₁ > 0 →
  days > 0 →
  (people₁ * work₁ = people₁ * people₁) →
  people₂ = people₁ * (people₂ / people₁) →
  (people₂ / people₁ : ℚ) * work₁ = people₂ / people₁ * people₁ :=
by sorry

end NUMINAMATH_CALUDE_work_scaling_l520_52029


namespace NUMINAMATH_CALUDE_equation_solution_l520_52067

theorem equation_solution : ∃! x : ℝ, 
  Real.sqrt x + Real.sqrt (x + 9) + 3 * Real.sqrt (x^2 + 9*x) + Real.sqrt (3*x + 27) = 45 - 3*x ∧ 
  x = 729/144 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l520_52067


namespace NUMINAMATH_CALUDE_symmetric_points_difference_l520_52065

/-- Given two points P₁ and P₂ that are symmetric with respect to the origin,
    prove that m - n = 8. -/
theorem symmetric_points_difference (m n : ℝ) : 
  (∃ (P₁ P₂ : ℝ × ℝ), 
    P₁ = (2 - m, 5) ∧ 
    P₂ = (3, 2*n + 1) ∧ 
    P₁.1 = -P₂.1 ∧ 
    P₁.2 = -P₂.2) → 
  m - n = 8 := by
sorry

end NUMINAMATH_CALUDE_symmetric_points_difference_l520_52065


namespace NUMINAMATH_CALUDE_division_result_l520_52056

theorem division_result : (3486 : ℝ) / 189 = 18.444444444444443 := by
  sorry

end NUMINAMATH_CALUDE_division_result_l520_52056


namespace NUMINAMATH_CALUDE_sector_area_l520_52066

/-- Given a circular sector with circumference 6 and central angle 1 radian, its area is 2 -/
theorem sector_area (circumference : ℝ) (central_angle : ℝ) (area : ℝ) :
  circumference = 6 →
  central_angle = 1 →
  area = 2 :=
by sorry

end NUMINAMATH_CALUDE_sector_area_l520_52066


namespace NUMINAMATH_CALUDE_problem_statement_l520_52047

/-- A normal distribution with given mean and standard deviation -/
structure NormalDistribution where
  mean : ℝ
  std_dev : ℝ
  std_dev_pos : 0 < std_dev

/-- The value that is a given number of standard deviations below the mean -/
def value_below_mean (d : NormalDistribution) (n : ℝ) : ℝ :=
  d.mean - n * d.std_dev

/-- The problem statement -/
theorem problem_statement (d : NormalDistribution) 
  (h1 : d.mean = 17.5)
  (h2 : d.std_dev = 2.5) :
  value_below_mean d 2.7 = 10.75 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l520_52047


namespace NUMINAMATH_CALUDE_saturday_visitors_200_l520_52089

/-- Calculates the number of visitors on Saturday given the ticket price, 
    weekday visitors, Sunday visitors, and total revenue -/
def visitors_on_saturday (ticket_price : ℕ) (weekday_visitors : ℕ) 
  (sunday_visitors : ℕ) (total_revenue : ℕ) : ℕ :=
  (total_revenue / ticket_price) - (5 * weekday_visitors) - sunday_visitors

/-- Proves that the number of visitors on Saturday is 200 given the specified conditions -/
theorem saturday_visitors_200 : 
  visitors_on_saturday 3 100 300 3000 = 200 := by
  sorry

end NUMINAMATH_CALUDE_saturday_visitors_200_l520_52089


namespace NUMINAMATH_CALUDE_intersection_implies_a_values_l520_52096

def A (a : ℝ) : Set ℝ := {a^2, a+1, 3}
def B (a : ℝ) : Set ℝ := {a-3, 2*a-1, a^2+1}

theorem intersection_implies_a_values :
  ∀ a : ℝ, A a ∩ B a = {3} → a = 6 ∨ a = Real.sqrt 2 ∨ a = -Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_intersection_implies_a_values_l520_52096


namespace NUMINAMATH_CALUDE_cake_problem_l520_52075

/-- Proves that the initial number of cakes is 12, given the conditions of the problem. -/
theorem cake_problem (total : ℕ) (fallen : ℕ) (undamaged : ℕ) (destroyed : ℕ) 
  (h1 : fallen = total / 2)
  (h2 : undamaged = fallen / 2)
  (h3 : destroyed = 3)
  (h4 : fallen = undamaged + destroyed) :
  total = 12 := by
  sorry

end NUMINAMATH_CALUDE_cake_problem_l520_52075


namespace NUMINAMATH_CALUDE_no_numbers_satisfying_conditions_l520_52085

def is_in_range (n : ℕ) : Prop := 7 ≤ n ∧ n ≤ 49

def divisible_by_6 (n : ℕ) : Prop := n % 6 = 0

def remainder_3_mod_5 (n : ℕ) : Prop := n % 5 = 3

-- We don't need to define primality as it's already in Mathlib

theorem no_numbers_satisfying_conditions :
  ¬∃ n : ℕ, is_in_range n ∧ divisible_by_6 n ∧ remainder_3_mod_5 n ∧ Nat.Prime n :=
sorry

end NUMINAMATH_CALUDE_no_numbers_satisfying_conditions_l520_52085


namespace NUMINAMATH_CALUDE_rabbit_chicken_problem_l520_52027

theorem rabbit_chicken_problem (total : ℕ) (rabbits chickens : ℕ → ℕ) :
  total = 40 →
  (∀ x : ℕ, rabbits x + chickens x = total) →
  (∀ x : ℕ, 4 * rabbits x = 10 * 2 * chickens x - 8) →
  (∃ x : ℕ, rabbits x = 33) :=
by sorry

end NUMINAMATH_CALUDE_rabbit_chicken_problem_l520_52027


namespace NUMINAMATH_CALUDE_range_of_a_l520_52005

-- Define the function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 1 then (1 - 2*a)^x else Real.log x / Real.log a + 1/3

-- State the theorem
theorem range_of_a (a : ℝ) :
  (∀ x₁ x₂ : ℝ, x₁ ≠ x₂ → (f a x₁ - f a x₂) / (x₁ - x₂) < 0) ↔ 
  (0 < a ∧ a ≤ 1/3) :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l520_52005


namespace NUMINAMATH_CALUDE_sod_area_second_section_l520_52025

/-- Given the total area of sod needed and the area of the first section,
    prove that the area of the second section is 4800 square feet. -/
theorem sod_area_second_section
  (total_sod_squares : ℕ)
  (sod_square_size : ℕ)
  (first_section_length : ℕ)
  (first_section_width : ℕ)
  (h1 : total_sod_squares = 1500)
  (h2 : sod_square_size = 4)
  (h3 : first_section_length = 30)
  (h4 : first_section_width = 40) :
  total_sod_squares * sod_square_size - first_section_length * first_section_width = 4800 :=
by sorry

end NUMINAMATH_CALUDE_sod_area_second_section_l520_52025


namespace NUMINAMATH_CALUDE_heart_then_face_prob_l520_52045

/-- A standard deck of cards -/
structure Deck :=
  (cards : Finset (Fin 52))
  (card_count : cards.card = 52)

/-- The suit of a card -/
inductive Suit
  | Hearts | Diamonds | Clubs | Spades

/-- The rank of a card -/
inductive Rank
  | Ace | Two | Three | Four | Five | Six | Seven | Eight | Nine | Ten | Jack | Queen | King

/-- A playing card -/
structure Card :=
  (suit : Suit)
  (rank : Rank)

/-- Definition of a face card -/
def isFaceCard (c : Card) : Prop :=
  c.rank = Rank.Jack ∨ c.rank = Rank.Queen ∨ c.rank = Rank.King ∨ c.rank = Rank.Ace

/-- The probability of drawing a heart as the first card and a face card as the second -/
def heartThenFaceProbability (d : Deck) : ℚ :=
  5 / 86

/-- Theorem stating the probability of drawing a heart then a face card -/
theorem heart_then_face_prob (d : Deck) :
  heartThenFaceProbability d = 5 / 86 := by
  sorry


end NUMINAMATH_CALUDE_heart_then_face_prob_l520_52045


namespace NUMINAMATH_CALUDE_school_class_average_difference_l520_52055

theorem school_class_average_difference :
  let total_students : ℕ := 200
  let total_teachers : ℕ := 5
  let class_sizes : List ℕ := [80, 60, 40, 15, 5]
  
  let t : ℚ := (class_sizes.sum : ℚ) / total_teachers
  
  let s : ℚ := (class_sizes.map (λ size => size * size)).sum / total_students
  
  t - s = -19.25 := by sorry

end NUMINAMATH_CALUDE_school_class_average_difference_l520_52055


namespace NUMINAMATH_CALUDE_triangle_cosine_theorem_l520_52030

/-- Given a triangle ABC with sides BC = 5 and AC = 4, and cos(A - B) = 7/8, prove that cos C = -1/4 -/
theorem triangle_cosine_theorem (A B C : ℝ) (h1 : BC = 5) (h2 : AC = 4) (h3 : Real.cos (A - B) = 7/8) :
  Real.cos C = -1/4 := by
  sorry

end NUMINAMATH_CALUDE_triangle_cosine_theorem_l520_52030


namespace NUMINAMATH_CALUDE_stingray_count_shark_stingray_relation_total_fish_count_l520_52052

/-- The number of stingrays in an aquarium -/
def num_stingrays : ℕ := 28

/-- The number of sharks in the aquarium -/
def num_sharks : ℕ := 2 * num_stingrays

/-- The total number of fish in the aquarium -/
def total_fish : ℕ := 84

/-- Theorem stating that the number of stingrays is 28 -/
theorem stingray_count : num_stingrays = 28 := by
  sorry

/-- Theorem verifying the relationship between sharks and stingrays -/
theorem shark_stingray_relation : num_sharks = 2 * num_stingrays := by
  sorry

/-- Theorem verifying the total number of fish -/
theorem total_fish_count : num_stingrays + num_sharks = total_fish := by
  sorry

end NUMINAMATH_CALUDE_stingray_count_shark_stingray_relation_total_fish_count_l520_52052


namespace NUMINAMATH_CALUDE_line_through_origin_and_intersection_l520_52071

-- Define the two lines
def line1 (x y : ℝ) : Prop := 2*x + 3*y + 8 = 0
def line2 (x y : ℝ) : Prop := x - y - 1 = 0

-- Define the intersection point
def intersection_point : ℝ × ℝ := (x, y) where
  x := -1
  y := -2

-- Define the line l
def line_l (x y : ℝ) : Prop := 2*x - y = 0

-- Theorem statement
theorem line_through_origin_and_intersection :
  ∃ (x y : ℝ),
    line1 x y ∧ 
    line2 x y ∧ 
    line_l 0 0 ∧ 
    line_l (intersection_point.1) (intersection_point.2) ∧
    ∀ (a b : ℝ), line_l a b ↔ 2*a - b = 0 :=
sorry

end NUMINAMATH_CALUDE_line_through_origin_and_intersection_l520_52071


namespace NUMINAMATH_CALUDE_shirt_original_price_l520_52053

/-- Calculates the original price of an item given its discounted price and discount percentage. -/
def originalPrice (discountedPrice : ℚ) (discountPercentage : ℚ) : ℚ :=
  discountedPrice / (1 - discountPercentage / 100)

/-- Theorem stating that if a shirt is sold at Rs. 780 after a 20% discount, 
    then the original price of the shirt was Rs. 975. -/
theorem shirt_original_price : 
  originalPrice 780 20 = 975 := by
  sorry

end NUMINAMATH_CALUDE_shirt_original_price_l520_52053


namespace NUMINAMATH_CALUDE_megan_initial_cupcakes_l520_52001

/-- The number of cupcakes Todd ate -/
def todd_ate : ℕ := 43

/-- The number of packages Megan could make with the remaining cupcakes -/
def num_packages : ℕ := 4

/-- The number of cupcakes in each package -/
def cupcakes_per_package : ℕ := 7

/-- The initial number of cupcakes Megan baked -/
def initial_cupcakes : ℕ := todd_ate + num_packages * cupcakes_per_package

theorem megan_initial_cupcakes : initial_cupcakes = 71 := by
  sorry

end NUMINAMATH_CALUDE_megan_initial_cupcakes_l520_52001


namespace NUMINAMATH_CALUDE_sphere_surface_area_of_circumscribed_rectangular_solid_l520_52064

/-- The surface area of a sphere circumscribing a rectangular solid with dimensions √3, √2, and 1 is 6π. -/
theorem sphere_surface_area_of_circumscribed_rectangular_solid :
  let length : ℝ := Real.sqrt 3
  let width : ℝ := Real.sqrt 2
  let height : ℝ := 1
  let diagonal : ℝ := Real.sqrt (length ^ 2 + width ^ 2 + height ^ 2)
  let radius : ℝ := diagonal / 2
  let surface_area : ℝ := 4 * Real.pi * radius ^ 2
  surface_area = 6 * Real.pi := by
sorry

end NUMINAMATH_CALUDE_sphere_surface_area_of_circumscribed_rectangular_solid_l520_52064


namespace NUMINAMATH_CALUDE_first_sequence_general_term_second_sequence_general_term_l520_52026

/-- First sequence -/
def S₁ (n : ℕ) : ℚ := n^2 + (1/2) * n

/-- Second sequence -/
def S₂ (n : ℕ) : ℚ := (1/4) * n^2 + (2/3) * n + 3

/-- General term of the first sequence -/
def a₁ (n : ℕ) : ℚ := 2 * n - 1/2

/-- General term of the second sequence -/
def a₂ (n : ℕ) : ℚ :=
  if n = 1 then 47/12 else (6 * n + 5) / 12

theorem first_sequence_general_term (n : ℕ) :
  S₁ (n + 1) - S₁ n = a₁ (n + 1) :=
sorry

theorem second_sequence_general_term (n : ℕ) :
  S₂ (n + 1) - S₂ n = a₂ (n + 1) :=
sorry

end NUMINAMATH_CALUDE_first_sequence_general_term_second_sequence_general_term_l520_52026


namespace NUMINAMATH_CALUDE_problem_solution_l520_52024

def U : Set ℕ := {2, 3, 4, 5, 6}

def A : Set ℕ := {x ∈ U | x^2 - 6*x + 8 = 0}

def B : Set ℕ := {2, 5, 6}

theorem problem_solution : (U \ A) ∪ B = {2, 3, 5, 6} := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l520_52024


namespace NUMINAMATH_CALUDE_min_stable_stories_l520_52008

/-- Represents a domino placement on a rectangular grid --/
structure DominoPlacement :=
  (width : Nat) -- Width of the rectangle
  (height : Nat) -- Height of the rectangle
  (dominoes : Nat) -- Number of dominoes per story

/-- Represents a tower of domino placements --/
structure DominoTower :=
  (base : DominoPlacement)
  (stories : Nat)

/-- Defines when a domino tower is considered stable --/
def isStable (tower : DominoTower) : Prop :=
  ∀ (x y : ℚ), 0 ≤ x ∧ x < tower.base.width ∧ 0 ≤ y ∧ y < tower.base.height →
    ∃ (s : Nat), s < tower.stories ∧ 
      ∃ (dx dy : ℚ), (0 ≤ dx ∧ dx < 2 ∧ 0 ≤ dy ∧ dy < 1) ∧
        (⌊x⌋ ≤ x - dx ∧ x - dx < ⌊x⌋ + 1) ∧
        (⌊y⌋ ≤ y - dy ∧ y - dy < ⌊y⌋ + 1)

/-- The main theorem stating the minimum number of stories for a stable tower --/
theorem min_stable_stories (tower : DominoTower) 
  (h_width : tower.base.width = 10)
  (h_height : tower.base.height = 11)
  (h_dominoes : tower.base.dominoes = 55) :
  (isStable tower ↔ tower.stories ≥ 5) :=
sorry

end NUMINAMATH_CALUDE_min_stable_stories_l520_52008


namespace NUMINAMATH_CALUDE_temperature_drop_l520_52023

/-- Given an initial temperature and a temperature drop, calculates the final temperature -/
def finalTemperature (initial : Int) (drop : Int) : Int :=
  initial - drop

/-- Theorem: If the initial temperature is -6°C and it drops by 5°C, then the final temperature is -11°C -/
theorem temperature_drop : finalTemperature (-6) 5 = -11 := by
  sorry

end NUMINAMATH_CALUDE_temperature_drop_l520_52023


namespace NUMINAMATH_CALUDE_distance_calculation_l520_52018

def speed : Real := 20
def time : Real := 8
def distance : Real := speed * time

theorem distance_calculation : distance = 160 := by
  sorry

end NUMINAMATH_CALUDE_distance_calculation_l520_52018


namespace NUMINAMATH_CALUDE_min_triangles_for_G_2008_l520_52036

/-- Represents a point in a 2D grid --/
structure GridPoint where
  x : Nat
  y : Nat

/-- Defines the grid G_n --/
def G (n : Nat) : Set GridPoint :=
  {p : GridPoint | p.x ≥ 1 ∧ p.x ≤ n ∧ p.y ≥ 1 ∧ p.y ≤ n}

/-- Minimum number of triangles needed to cover a grid --/
def minTriangles (n : Nat) : Nat :=
  if n = 2 then 1
  else if n = 3 then 2
  else (n * n) / 3 * 2

/-- Theorem stating the minimum number of triangles needed to cover G_2008 --/
theorem min_triangles_for_G_2008 :
  minTriangles 2008 = 1338 :=
sorry

end NUMINAMATH_CALUDE_min_triangles_for_G_2008_l520_52036


namespace NUMINAMATH_CALUDE_negation_of_quadratic_inequality_l520_52035

theorem negation_of_quadratic_inequality (p : Prop) : 
  (p ↔ ∀ x : ℝ, x^2 + x + 1 ≠ 0) → 
  (¬p ↔ ∃ x : ℝ, x^2 + x + 1 = 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_quadratic_inequality_l520_52035


namespace NUMINAMATH_CALUDE_complex_modulus_equality_l520_52073

theorem complex_modulus_equality : 
  Complex.abs ((7 - 5*Complex.I)*(3 + 4*Complex.I) + (4 - 3*Complex.I)*(2 + 7*Complex.I)) = Real.sqrt 6073 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_equality_l520_52073


namespace NUMINAMATH_CALUDE_quadratic_inequality_l520_52028

theorem quadratic_inequality (a b : ℝ) (h : ∃ x y : ℝ, x ≠ y ∧ x^2 + a*x + b = 0 ∧ y^2 + a*y + b = 0) :
  ∃ n : ℤ, |n^2 + a*n + b| ≤ max (1/4 : ℝ) ((1/2 : ℝ) * Real.sqrt (a^2 - 4*b)) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_l520_52028


namespace NUMINAMATH_CALUDE_division_with_remainder_l520_52019

theorem division_with_remainder (x y : ℕ+) : 
  (x : ℝ) / (y : ℝ) = 96.12 →
  (x : ℝ) % (y : ℝ) = 5.76 →
  y = 100 := by
sorry

end NUMINAMATH_CALUDE_division_with_remainder_l520_52019


namespace NUMINAMATH_CALUDE_sum_six_consecutive_integers_l520_52011

theorem sum_six_consecutive_integers (n : ℤ) :
  n + (n + 1) + (n + 2) + (n + 3) + (n + 4) + (n + 5) = 6 * n + 15 := by
  sorry

end NUMINAMATH_CALUDE_sum_six_consecutive_integers_l520_52011


namespace NUMINAMATH_CALUDE_dot_product_result_l520_52077

theorem dot_product_result :
  let a : ℝ × ℝ := (2 * Real.sin (35 * π / 180), 2 * Real.cos (35 * π / 180))
  let b : ℝ × ℝ := (Real.cos (5 * π / 180), -Real.sin (5 * π / 180))
  (a.1 * b.1 + a.2 * b.2) = 1 := by
  sorry

end NUMINAMATH_CALUDE_dot_product_result_l520_52077


namespace NUMINAMATH_CALUDE_polynomial_composition_l520_52031

/-- Given a function f and a polynomial g, proves that g satisfies the given condition -/
theorem polynomial_composition (f g : ℝ → ℝ) : 
  (∀ x, f x = x^2) →
  (∀ x, f (g x) = 4*x^2 + 4*x + 1) →
  (∀ x, g x = 2*x + 1 ∨ g x = -2*x - 1) :=
by sorry

end NUMINAMATH_CALUDE_polynomial_composition_l520_52031


namespace NUMINAMATH_CALUDE_daughters_and_granddaughters_without_children_l520_52084

/-- Represents Bertha's family structure -/
structure BerthaFamily where
  daughters : ℕ
  granddaughters : ℕ
  total_descendants : ℕ
  daughters_with_children : ℕ

/-- The conditions of Bertha's family -/
def bertha_conditions : BerthaFamily where
  daughters := 8
  granddaughters := 32
  total_descendants := 40
  daughters_with_children := 8

/-- Theorem stating the number of daughters and granddaughters without children -/
theorem daughters_and_granddaughters_without_children 
  (family : BerthaFamily) 
  (h1 : family.daughters = bertha_conditions.daughters)
  (h2 : family.total_descendants = bertha_conditions.total_descendants)
  (h3 : family.granddaughters = family.total_descendants - family.daughters)
  (h4 : family.daughters_with_children * 4 = family.granddaughters)
  (h5 : family.daughters_with_children ≤ family.daughters) :
  family.total_descendants - family.daughters_with_children = 32 := by
  sorry

end NUMINAMATH_CALUDE_daughters_and_granddaughters_without_children_l520_52084


namespace NUMINAMATH_CALUDE_abc_inequalities_l520_52076

theorem abc_inequalities (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (habc : a * b * c = 1) : 
  ((1 + a) * (1 + b) * (1 + c) ≥ 8) ∧ 
  (Real.sqrt a + Real.sqrt b + Real.sqrt c ≤ 1/a + 1/b + 1/c) := by
  sorry

end NUMINAMATH_CALUDE_abc_inequalities_l520_52076


namespace NUMINAMATH_CALUDE_complex_product_real_implies_a_equals_one_l520_52021

theorem complex_product_real_implies_a_equals_one (a : ℝ) :
  ((1 + Complex.I) * (1 - a * Complex.I)).im = 0 → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_product_real_implies_a_equals_one_l520_52021


namespace NUMINAMATH_CALUDE_hike_duration_is_one_hour_l520_52010

/-- Represents the hike scenario with given conditions -/
structure HikeScenario where
  total_distance : Real
  initial_water : Real
  final_water : Real
  leak_rate : Real
  last_mile_consumption : Real
  first_three_miles_rate : Real

/-- Calculates the duration of the hike based on given conditions -/
def hike_duration (scenario : HikeScenario) : Real :=
  -- The actual calculation is not implemented here
  sorry

/-- Theorem stating that the hike duration is 1 hour for the given scenario -/
theorem hike_duration_is_one_hour (scenario : HikeScenario) 
  (h1 : scenario.total_distance = 4)
  (h2 : scenario.initial_water = 6)
  (h3 : scenario.final_water = 1)
  (h4 : scenario.leak_rate = 1)
  (h5 : scenario.last_mile_consumption = 1)
  (h6 : scenario.first_three_miles_rate = 0.6666666666666666) :
  hike_duration scenario = 1 := by
  sorry

end NUMINAMATH_CALUDE_hike_duration_is_one_hour_l520_52010


namespace NUMINAMATH_CALUDE_hyperbola_sufficient_not_necessary_l520_52002

/-- Hyperbola equation -/
def is_hyperbola (x y a b : ℝ) : Prop :=
  x^2 / a^2 - y^2 / b^2 = 1

/-- Asymptotes equation -/
def is_asymptote (x y a b : ℝ) : Prop :=
  y = b/a * x ∨ y = -b/a * x

/-- The hyperbola equation is a sufficient but not necessary condition for the asymptotes equation -/
theorem hyperbola_sufficient_not_necessary (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∀ x y, is_hyperbola x y a b → is_asymptote x y a b) ∧
  ¬(∀ x y, is_asymptote x y a b → is_hyperbola x y a b) :=
sorry

end NUMINAMATH_CALUDE_hyperbola_sufficient_not_necessary_l520_52002


namespace NUMINAMATH_CALUDE_complex_square_root_of_18i_l520_52090

theorem complex_square_root_of_18i :
  ∀ (z : ℂ), (∃ (x y : ℝ), z = x + y * I ∧ x > 0 ∧ z^2 = 18 * I) → z = 3 + 3 * I :=
by
  sorry

end NUMINAMATH_CALUDE_complex_square_root_of_18i_l520_52090


namespace NUMINAMATH_CALUDE_cyclic_sum_inequality_l520_52094

theorem cyclic_sum_inequality (a b c d : ℝ) 
  (pos_a : 0 < a) (pos_b : 0 < b) (pos_c : 0 < c) (pos_d : 0 < d)
  (sum_eq_one : a + b + c + d = 1) :
  (a / (a^2 + 1)) + (b / (b^2 + 1)) + (c / (c^2 + 1)) + (d / (d^2 + 1)) ≤ 16/17 := by
  sorry

end NUMINAMATH_CALUDE_cyclic_sum_inequality_l520_52094


namespace NUMINAMATH_CALUDE_geometric_sequence_properties_l520_52086

theorem geometric_sequence_properties (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h_geom : ∃ q : ℝ, q > 0 ∧ b = a * q ∧ c = a * q^2) :
  ∃ r : ℝ, r > 0 ∧
    (a + b + c) = (Real.sqrt (3 * (a * b + b * c + c * a))) * r ∧
    (Real.sqrt (3 * (a * b + b * c + c * a))) = (27 * a * b * c)^(1/3) * r :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_properties_l520_52086


namespace NUMINAMATH_CALUDE_triangle_area_change_l520_52044

/-- Theorem: Effect on triangle area when height is decreased by 40% and base is increased by 40% -/
theorem triangle_area_change (base height : ℝ) (base_new height_new area area_new : ℝ) 
  (h1 : base_new = base * 1.4)
  (h2 : height_new = height * 0.6)
  (h3 : area = (base * height) / 2)
  (h4 : area_new = (base_new * height_new) / 2) :
  area_new = area * 0.84 := by
sorry

end NUMINAMATH_CALUDE_triangle_area_change_l520_52044


namespace NUMINAMATH_CALUDE_median_is_90_l520_52054

/-- Represents the score distribution of students -/
structure ScoreDistribution where
  score_70 : Nat
  score_80 : Nat
  score_90 : Nat
  score_100 : Nat

/-- Calculates the total number of students -/
def total_students (sd : ScoreDistribution) : Nat :=
  sd.score_70 + sd.score_80 + sd.score_90 + sd.score_100

/-- Defines the median score for a given score distribution -/
def median_score (sd : ScoreDistribution) : Nat :=
  if sd.score_70 + sd.score_80 ≥ (total_students sd + 1) / 2 then 80
  else if sd.score_70 + sd.score_80 + sd.score_90 ≥ (total_students sd + 1) / 2 then 90
  else 100

/-- Theorem stating that the median score for the given distribution is 90 -/
theorem median_is_90 (sd : ScoreDistribution) 
  (h1 : sd.score_70 = 1)
  (h2 : sd.score_80 = 6)
  (h3 : sd.score_90 = 5)
  (h4 : sd.score_100 = 3) :
  median_score sd = 90 := by
  sorry

end NUMINAMATH_CALUDE_median_is_90_l520_52054


namespace NUMINAMATH_CALUDE_geometric_sequence_103rd_term_l520_52041

/-- Given a geometric sequence with first term a and common ratio r,
    this function returns the nth term of the sequence. -/
def geometric_sequence (a : ℝ) (r : ℝ) (n : ℕ) : ℝ :=
  a * r^(n - 1)

theorem geometric_sequence_103rd_term :
  let a : ℝ := 4
  let r : ℝ := -3
  geometric_sequence a r 103 = 4 * 3^102 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_103rd_term_l520_52041


namespace NUMINAMATH_CALUDE_A_star_B_equality_l520_52059

def A : Set ℝ := {x : ℝ | 0 ≤ x ∧ x ≤ 3}
def B : Set ℝ := {x : ℝ | x ≥ 1}

def star_operation (A B : Set ℝ) : Set ℝ := (A ∪ B) \ (A ∩ B)

theorem A_star_B_equality : 
  star_operation A B = {x : ℝ | (0 ≤ x ∧ x < 1) ∨ x > 3} :=
by sorry

end NUMINAMATH_CALUDE_A_star_B_equality_l520_52059


namespace NUMINAMATH_CALUDE_unique_students_count_unique_students_is_34_l520_52016

/-- The number of unique students in a mathematics contest at Gauss High School --/
theorem unique_students_count : ℕ :=
  let euclid_class : ℕ := 12
  let raman_class : ℕ := 10
  let pythagoras_class : ℕ := 15
  let euclid_raman_overlap : ℕ := 3
  euclid_class + raman_class + pythagoras_class - euclid_raman_overlap

/-- Proof that the number of unique students is 34 --/
theorem unique_students_is_34 : unique_students_count = 34 := by
  sorry

end NUMINAMATH_CALUDE_unique_students_count_unique_students_is_34_l520_52016


namespace NUMINAMATH_CALUDE_range_of_2x_plus_y_range_of_c_l520_52006

-- Define the circle
def Circle (x y : ℝ) : Prop := x^2 + y^2 = 2*y

-- Statement for the range of 2x + y
theorem range_of_2x_plus_y (x y : ℝ) (h : Circle x y) :
  -1 - Real.sqrt 5 ≤ 2*x + y ∧ 2*x + y ≤ 1 + Real.sqrt 5 := by sorry

-- Statement for the range of c
theorem range_of_c (c : ℝ) (h : ∀ x y : ℝ, Circle x y → x + y + c > 0) :
  c ≥ -1 := by sorry

end NUMINAMATH_CALUDE_range_of_2x_plus_y_range_of_c_l520_52006


namespace NUMINAMATH_CALUDE_complex_equality_condition_l520_52049

theorem complex_equality_condition (a b c d : ℝ) : 
  let z1 : ℂ := Complex.mk a b
  let z2 : ℂ := Complex.mk c d
  (z1 = z2 → a = c) ∧ 
  ∃ a b c d : ℝ, a = c ∧ Complex.mk a b ≠ Complex.mk c d :=
by sorry

end NUMINAMATH_CALUDE_complex_equality_condition_l520_52049


namespace NUMINAMATH_CALUDE_circle_diameter_endpoint_l520_52033

/-- Given a circle with center (1, -2) and one endpoint of a diameter at (4, 3),
    the other endpoint of the diameter is at (7, 3). -/
theorem circle_diameter_endpoint :
  let center : ℝ × ℝ := (1, -2)
  let endpoint1 : ℝ × ℝ := (4, 3)
  let endpoint2 : ℝ × ℝ := (7, 3)
  (endpoint1.1 - center.1 = center.1 - endpoint2.1 ∧
   endpoint1.2 - center.2 = center.2 - endpoint2.2) :=
by sorry

end NUMINAMATH_CALUDE_circle_diameter_endpoint_l520_52033


namespace NUMINAMATH_CALUDE_shane_photos_february_l520_52020

/-- The number of photos Shane takes in the first two months of the year -/
def total_photos : ℕ := 146

/-- The number of photos Shane takes each day in January -/
def photos_per_day_january : ℕ := 2

/-- The number of days in January -/
def days_in_january : ℕ := 31

/-- The number of weeks in February -/
def weeks_in_february : ℕ := 4

/-- Calculate the number of photos Shane takes each week in February -/
def photos_per_week_february : ℕ :=
  (total_photos - photos_per_day_january * days_in_january) / weeks_in_february

theorem shane_photos_february :
  photos_per_week_february = 21 := by
  sorry

end NUMINAMATH_CALUDE_shane_photos_february_l520_52020


namespace NUMINAMATH_CALUDE_exists_x_where_exp_leq_x_plus_one_l520_52060

theorem exists_x_where_exp_leq_x_plus_one : ∃ x : ℝ, Real.exp x ≤ x + 1 := by
  sorry

end NUMINAMATH_CALUDE_exists_x_where_exp_leq_x_plus_one_l520_52060


namespace NUMINAMATH_CALUDE_max_elements_l520_52058

structure RelationSystem where
  S : Type
  rel : S → S → Prop
  distinct_relation : ∀ a b : S, a ≠ b → (rel a b ∨ rel b a) ∧ ¬(rel a b ∧ rel b a)
  transitivity : ∀ a b c : S, a ≠ b → b ≠ c → a ≠ c → rel a b → rel b c → rel c a

theorem max_elements (R : RelationSystem) : 
  ∃ (n : ℕ), ∀ (m : ℕ), (∃ (f : Fin m → R.S), Function.Injective f) → m ≤ n :=
sorry

end NUMINAMATH_CALUDE_max_elements_l520_52058


namespace NUMINAMATH_CALUDE_deleted_pictures_l520_52017

theorem deleted_pictures (zoo_pics museum_pics remaining_pics : ℕ) 
  (h1 : zoo_pics = 24)
  (h2 : museum_pics = 12)
  (h3 : remaining_pics = 22) :
  zoo_pics + museum_pics - remaining_pics = 14 := by
  sorry

end NUMINAMATH_CALUDE_deleted_pictures_l520_52017


namespace NUMINAMATH_CALUDE_max_value_of_f_on_interval_l520_52037

def f (x : ℝ) : ℝ := 2 * x^3 - 3 * x^2 - 12 * x + 5

theorem max_value_of_f_on_interval :
  ∃ (x : ℝ), x ∈ Set.Icc 0 3 ∧ f x = 5 ∧ ∀ (y : ℝ), y ∈ Set.Icc 0 3 → f y ≤ f x :=
sorry

end NUMINAMATH_CALUDE_max_value_of_f_on_interval_l520_52037


namespace NUMINAMATH_CALUDE_cosine_inequality_l520_52088

theorem cosine_inequality (x y : Real) : 
  x ∈ Set.Icc 0 (Real.pi / 2) →
  y ∈ Set.Icc 0 (Real.pi / 2) →
  Real.cos (x - y) ≥ Real.cos x - Real.cos y := by
sorry

end NUMINAMATH_CALUDE_cosine_inequality_l520_52088


namespace NUMINAMATH_CALUDE_diamond_five_three_l520_52032

-- Define the operation ⋄
def diamond (a b : ℕ) : ℕ := 4 * a + 6 * b

-- Theorem statement
theorem diamond_five_three : diamond 5 3 = 38 := by
  sorry

end NUMINAMATH_CALUDE_diamond_five_three_l520_52032


namespace NUMINAMATH_CALUDE_max_students_distribution_l520_52063

def stationery_A : ℕ := 38
def stationery_B : ℕ := 78
def stationery_C : ℕ := 128

def remaining_A : ℕ := 2
def remaining_B : ℕ := 6
def remaining_C : ℕ := 20

def distributed_A : ℕ := stationery_A - remaining_A
def distributed_B : ℕ := stationery_B - remaining_B
def distributed_C : ℕ := stationery_C - remaining_C

theorem max_students_distribution :
  ∃ (n : ℕ), n > 0 ∧ 
    distributed_A % n = 0 ∧
    distributed_B % n = 0 ∧
    distributed_C % n = 0 ∧
    ∀ (m : ℕ), m > n →
      (distributed_A % m ≠ 0 ∨
       distributed_B % m ≠ 0 ∨
       distributed_C % m ≠ 0) →
    n = 36 :=
by sorry

end NUMINAMATH_CALUDE_max_students_distribution_l520_52063


namespace NUMINAMATH_CALUDE_unique_prime_triple_l520_52003

theorem unique_prime_triple (p : ℤ) : 
  (Nat.Prime p.natAbs ∧ Nat.Prime (p + 2).natAbs ∧ Nat.Prime (p + 4).natAbs) ↔ p = 3 :=
sorry

end NUMINAMATH_CALUDE_unique_prime_triple_l520_52003


namespace NUMINAMATH_CALUDE_soccer_team_size_l520_52098

theorem soccer_team_size (total_goals : ℕ) (games_played : ℕ) (goals_other_players : ℕ) :
  total_goals = 150 →
  games_played = 15 →
  goals_other_players = 30 →
  ∃ (team_size : ℕ),
    team_size > 0 ∧
    (team_size / 3 : ℚ) * games_played + goals_other_players = total_goals ∧
    team_size = 24 :=
by
  sorry

end NUMINAMATH_CALUDE_soccer_team_size_l520_52098


namespace NUMINAMATH_CALUDE_tims_takeout_cost_l520_52082

/-- The total cost of Tim's Chinese take-out -/
def total_cost : ℝ := 50

/-- The percentage of the cost that went to entrees -/
def entree_percentage : ℝ := 0.8

/-- The number of appetizers Tim bought -/
def num_appetizers : ℕ := 2

/-- The cost of a single appetizer -/
def appetizer_cost : ℝ := 5

theorem tims_takeout_cost :
  total_cost = (num_appetizers : ℝ) * appetizer_cost / (1 - entree_percentage) :=
by sorry

end NUMINAMATH_CALUDE_tims_takeout_cost_l520_52082


namespace NUMINAMATH_CALUDE_infinitely_many_solutions_l520_52095

theorem infinitely_many_solutions (d : ℝ) : 
  (∀ x : ℝ, 3 * (5 + d * x) = 15 * x + 15) ↔ d = 5 :=
sorry

end NUMINAMATH_CALUDE_infinitely_many_solutions_l520_52095


namespace NUMINAMATH_CALUDE_arithmetic_sequence_fifth_term_l520_52080

def arithmetic_sequence (a : ℕ → ℝ) := ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_fifth_term
  (a : ℕ → ℝ)
  (h_arithmetic : arithmetic_sequence a)
  (h_sum_1_2 : a 1 + a 2 = 1)
  (h_sum_3_4 : a 3 + a 4 = 5) :
  a 5 = 4 :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_fifth_term_l520_52080


namespace NUMINAMATH_CALUDE_karls_savings_l520_52081

/-- The problem of calculating Karl's savings --/
theorem karls_savings :
  let folder_price : ℚ := 5/2
  let pen_price : ℚ := 1
  let folder_count : ℕ := 7
  let pen_count : ℕ := 10
  let folder_discount : ℚ := 3/10
  let pen_discount : ℚ := 15/100
  
  let folder_savings := folder_count * (folder_price * folder_discount)
  let pen_savings := pen_count * (pen_price * pen_discount)
  
  folder_savings + pen_savings = 27/4 := by
  sorry

end NUMINAMATH_CALUDE_karls_savings_l520_52081


namespace NUMINAMATH_CALUDE_ratio_difference_l520_52068

theorem ratio_difference (a b c : ℝ) (h1 : a / b = 3 / 5) (h2 : b / c = 5 / 7) (h3 : c = 56) : c - a = 32 := by
  sorry

end NUMINAMATH_CALUDE_ratio_difference_l520_52068


namespace NUMINAMATH_CALUDE_f_monotone_decreasing_iff_a_in_range_l520_52099

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then 2 * x^2 - 8 * a * x + 3 else Real.log x / Real.log a

def monotone_decreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x ≤ y → f y ≤ f x

theorem f_monotone_decreasing_iff_a_in_range (a : ℝ) :
  (monotone_decreasing (f a)) ↔ (1/2 ≤ a ∧ a ≤ 5/8) :=
sorry

end NUMINAMATH_CALUDE_f_monotone_decreasing_iff_a_in_range_l520_52099
