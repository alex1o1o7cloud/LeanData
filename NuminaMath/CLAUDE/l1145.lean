import Mathlib

namespace speed_of_northern_cyclist_l1145_114548

/-- Theorem: Speed of northern cyclist
Given two cyclists starting from the same place in opposite directions,
with one going north at speed v kmph and the other going south at 40 kmph,
if they are 50 km apart after 0.7142857142857143 hours, then v = 30 kmph. -/
theorem speed_of_northern_cyclist (v : ℝ) : 
  v > 0 → -- Assuming positive speed
  50 = (v + 40) * 0.7142857142857143 →
  v = 30 := by
  sorry

end speed_of_northern_cyclist_l1145_114548


namespace sum_of_roots_quadratic_sum_of_roots_specific_equation_l1145_114534

theorem sum_of_roots_quadratic (a b c : ℝ) (h : a ≠ 0) :
  let f := fun x : ℝ => a * x^2 + b * x + c
  (∃ x₁ x₂ : ℝ, f x₁ = 0 ∧ f x₂ = 0 ∧ x₁ ≠ x₂) →
  (∃ s : ℝ, ∀ x : ℝ, f x = 0 → (∃ y : ℝ, f y = 0 ∧ y ≠ x ∧ x + y = s)) →
  s = -b / a :=
by sorry

theorem sum_of_roots_specific_equation :
  let f := fun x : ℝ => x^2 - 5*x + 6 - 9
  (∃ x₁ x₂ : ℝ, f x₁ = 0 ∧ f x₂ = 0 ∧ x₁ ≠ x₂) →
  (∃ s : ℝ, ∀ x : ℝ, f x = 0 → (∃ y : ℝ, f y = 0 ∧ y ≠ x ∧ x + y = s)) →
  s = 5 :=
by sorry

end sum_of_roots_quadratic_sum_of_roots_specific_equation_l1145_114534


namespace distinct_values_of_3_3_3_3_l1145_114565

-- Define a function to represent the expression with different parenthesizations
def exprParenthesization : List (ℕ → ℕ → ℕ → ℕ) :=
  [ (λ a b c => a^(b^(c^c))),
    (λ a b c => a^((b^c)^c)),
    (λ a b c => ((a^b)^c)^c),
    (λ a b c => (a^(b^c))^c),
    (λ a b c => (a^b)^(c^c)) ]

-- Define a function to evaluate the expression for a given base
def evaluateExpr (base : ℕ) : List ℕ :=
  exprParenthesization.map (λ f => f base base base)

-- Theorem statement
theorem distinct_values_of_3_3_3_3 :
  (evaluateExpr 3).toFinset.card = 3 := by sorry


end distinct_values_of_3_3_3_3_l1145_114565


namespace binomial_coefficient_18_8_l1145_114500

theorem binomial_coefficient_18_8 (h1 : Nat.choose 16 6 = 8008)
                                  (h2 : Nat.choose 16 7 = 11440)
                                  (h3 : Nat.choose 16 8 = 12870) :
  Nat.choose 18 8 = 43758 := by
  sorry

end binomial_coefficient_18_8_l1145_114500


namespace circle_equation_correct_l1145_114535

-- Define the center of the circle
def center : ℝ × ℝ := (1, -2)

-- Define the radius of the circle
def radius : ℝ := 4

-- Define the equation of the circle
def circle_equation (x y : ℝ) : Prop :=
  (x - center.1)^2 + (y - center.2)^2 = radius^2

-- Theorem statement
theorem circle_equation_correct :
  ∀ x y : ℝ, circle_equation x y ↔ ((x - 1)^2 + (y + 2)^2 = 16) :=
by sorry

end circle_equation_correct_l1145_114535


namespace prime_divisors_of_50_factorial_l1145_114568

theorem prime_divisors_of_50_factorial (n : ℕ) : n = 50 → 
  (Finset.filter Nat.Prime (Finset.range (n + 1))).card = 15 := by
  sorry

end prime_divisors_of_50_factorial_l1145_114568


namespace spa_nail_polish_problem_l1145_114558

/-- The number of girls who went to the spa -/
def num_girls : ℕ := 8

/-- The number of fingers on each limb -/
def fingers_per_limb : ℕ := 5

/-- The total number of fingers polished -/
def total_fingers_polished : ℕ := 40

/-- The number of limbs polished per girl -/
def limbs_per_girl : ℕ := total_fingers_polished / (num_girls * fingers_per_limb)

theorem spa_nail_polish_problem :
  limbs_per_girl = 1 :=
by sorry

end spa_nail_polish_problem_l1145_114558


namespace least_common_denominator_l1145_114564

theorem least_common_denominator : Nat.lcm 2 (Nat.lcm 3 (Nat.lcm 4 (Nat.lcm 5 (Nat.lcm 6 (Nat.lcm 7 (Nat.lcm 8 9)))))) = 2520 := by
  sorry

end least_common_denominator_l1145_114564


namespace movie_profit_calculation_l1145_114545

def actor_cost : ℕ := 1200
def num_people : ℕ := 50
def food_cost_per_person : ℕ := 3
def movie_selling_price : ℕ := 10000

def total_food_cost : ℕ := num_people * food_cost_per_person
def actors_and_food_cost : ℕ := actor_cost + total_food_cost
def equipment_rental_cost : ℕ := 2 * actors_and_food_cost
def total_cost : ℕ := actors_and_food_cost + equipment_rental_cost

theorem movie_profit_calculation :
  movie_selling_price - total_cost = 5950 := by
  sorry

end movie_profit_calculation_l1145_114545


namespace berries_to_buy_l1145_114552

def total_needed : ℕ := 36
def strawberries : ℕ := 4
def blueberries : ℕ := 8
def raspberries : ℕ := 3
def blackberries : ℕ := 5

theorem berries_to_buy (total_needed strawberries blueberries raspberries blackberries : ℕ) :
  total_needed - (strawberries + blueberries + raspberries + blackberries) = 16 := by
  sorry

end berries_to_buy_l1145_114552


namespace students_with_both_pets_l1145_114583

theorem students_with_both_pets (total : ℕ) (dog_owners : ℕ) (cat_owners : ℕ) 
  (h1 : total = 50)
  (h2 : dog_owners = 35)
  (h3 : cat_owners = 40)
  (h4 : dog_owners + cat_owners - total ≤ dog_owners)
  (h5 : dog_owners + cat_owners - total ≤ cat_owners) :
  dog_owners + cat_owners - total = 25 := by
  sorry

end students_with_both_pets_l1145_114583


namespace photo_arrangements_l1145_114505

/-- The number of male students -/
def num_male_students : ℕ := 4

/-- The number of female students -/
def num_female_students : ℕ := 3

/-- The total number of students -/
def total_students : ℕ := num_male_students + num_female_students

theorem photo_arrangements :
  /- (1) Arrangements with male student A at one of the ends -/
  (∃ (n : ℕ), n = 1440 ∧ 
    n = 2 * (Nat.factorial (total_students - 1))) ∧
  /- (2) Arrangements where female students B and C are not next to each other -/
  (∃ (m : ℕ), m = 3600 ∧ 
    m = (Nat.factorial (total_students - 2)) * (total_students * (total_students - 1) / 2)) ∧
  /- (3) Arrangements where female student B is not at the ends and C is not in the middle -/
  (∃ (k : ℕ), k = 3120 ∧ 
    k = (Nat.factorial (total_students - 2)) * 4 + (Nat.factorial (total_students - 2)) * 4 * 5) :=
by sorry

end photo_arrangements_l1145_114505


namespace angle_sum_around_point_l1145_114522

theorem angle_sum_around_point (y : ℝ) : 
  (6*y + 3*y + 4*y + 2*y = 360) → y = 24 := by
sorry

end angle_sum_around_point_l1145_114522


namespace jim_gas_cost_l1145_114594

/-- The total amount spent on gas by Jim -/
def total_gas_cost (nc_gallons : ℕ) (nc_price : ℚ) (va_gallons : ℕ) (va_price_increase : ℚ) : ℚ :=
  (nc_gallons : ℚ) * nc_price + (va_gallons : ℚ) * (nc_price + va_price_increase)

/-- Theorem stating that Jim's total gas cost is $50.00 -/
theorem jim_gas_cost :
  total_gas_cost 10 2 10 1 = 50 := by
  sorry

end jim_gas_cost_l1145_114594


namespace shop_b_better_l1145_114502

/-- Represents a costume rental shop -/
structure Shop where
  name : String
  base_price : ℕ
  discount_rate : ℚ
  discount_threshold : ℕ
  additional_discount : ℕ

/-- Calculates the number of sets that can be rented from a shop given a budget -/
def sets_rentable (shop : Shop) (budget : ℕ) : ℚ :=
  if budget / shop.base_price > shop.discount_threshold
  then (budget + shop.additional_discount) / (shop.base_price * (1 - shop.discount_rate))
  else budget / shop.base_price

/-- The main theorem proving Shop B offers more sets than Shop A -/
theorem shop_b_better (shop_a shop_b : Shop) (budget : ℕ) :
  shop_a.name = "A" →
  shop_b.name = "B" →
  shop_b.base_price = shop_a.base_price + 10 →
  400 / shop_a.base_price = 500 / shop_b.base_price →
  shop_b.discount_rate = 1/5 →
  shop_b.discount_threshold = 100 →
  shop_b.additional_discount = 200 →
  budget = 5000 →
  sets_rentable shop_b budget > sets_rentable shop_a budget :=
by
  sorry

end shop_b_better_l1145_114502


namespace combination_permutation_ratio_l1145_114516

def binomial_coefficient (n k : ℕ) : ℕ := (Nat.factorial n) / ((Nat.factorial k) * (Nat.factorial (n - k)))

def permutation (n k : ℕ) : ℕ := (Nat.factorial n) / (Nat.factorial (n - k))

theorem combination_permutation_ratio (x y : ℕ) (h : y > x) :
  (binomial_coefficient y x : ℚ) / (binomial_coefficient (y + 2) x : ℚ) = 1 / 3 ∧
  (permutation y x : ℚ) / (binomial_coefficient y x : ℚ) = 24 ↔
  x = 4 ∧ y = 8 := by
  sorry

end combination_permutation_ratio_l1145_114516


namespace fixed_point_symmetric_coordinates_l1145_114525

/-- Given a line that always passes through a fixed point P and P is symmetric about x + y = 0, prove P's coordinates -/
theorem fixed_point_symmetric_coordinates :
  ∀ (k : ℝ), 
  (∃ (P : ℝ × ℝ), ∀ (x y : ℝ), k * x - y + k - 2 = 0 → (x, y) = P) →
  (∃ (P' : ℝ × ℝ), 
    (P'.1 + P'.2 = 0) ∧ 
    (P'.1 - P.1)^2 + (P'.2 - P.2)^2 = 2 * ((P.1 + P.2) / 2)^2) →
  P = (2, 1) :=
by sorry


end fixed_point_symmetric_coordinates_l1145_114525


namespace cement_mixture_weight_l1145_114569

theorem cement_mixture_weight (sand_ratio : ℚ) (water_ratio : ℚ) (gravel_weight : ℚ) 
  (h1 : sand_ratio = 1/2)
  (h2 : water_ratio = 1/5)
  (h3 : gravel_weight = 15) :
  ∃ (total_weight : ℚ), 
    sand_ratio * total_weight + water_ratio * total_weight + gravel_weight = total_weight ∧
    total_weight = 50 := by
  sorry

end cement_mixture_weight_l1145_114569


namespace greatest_multiple_of_four_l1145_114530

theorem greatest_multiple_of_four (x : ℕ) : 
  x > 0 → x % 4 = 0 → x^2 < 2000 → x ≤ 44 ∧ ∃ y : ℕ, y > 0 ∧ y % 4 = 0 ∧ y^2 < 2000 ∧ y = 44 := by
  sorry

end greatest_multiple_of_four_l1145_114530


namespace calculation_proof_l1145_114554

theorem calculation_proof : (-36) / (-1/2 + 1/6 - 1/3) = 54 := by
  sorry

end calculation_proof_l1145_114554


namespace no_three_color_solution_exists_seven_color_solution_l1145_114532

-- Define a circle
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define an equilateral triangle
structure EqTriangle where
  vertices : Fin 3 → ℝ × ℝ

-- Define a coloring function type
def ColoringFunction (n : ℕ) := ℝ × ℝ → Fin n

-- Define what it means for a circle to be contained in another circle
def containedIn (c1 c2 : Circle) : Prop :=
  (c1.radius + ((c1.center.1 - c2.center.1)^2 + (c1.center.2 - c2.center.2)^2).sqrt ≤ c2.radius)

-- Define what it means for a triangle to be inscribed in a circle
def inscribedIn (t : EqTriangle) (c : Circle) : Prop := sorry

-- Define what it means for a coloring to be good for a circle
def goodColoring (f : ColoringFunction n) (c : Circle) : Prop :=
  ∀ t : EqTriangle, inscribedIn t c → (f (t.vertices 0) ≠ f (t.vertices 1) ∧ 
                                       f (t.vertices 0) ≠ f (t.vertices 2) ∧ 
                                       f (t.vertices 1) ≠ f (t.vertices 2))

-- Main theorem for part 1
theorem no_three_color_solution :
  ¬ ∃ (f : ColoringFunction 3), 
    ∀ (c : Circle), 
      containedIn c (Circle.mk (0, 0) 2) → c.radius ≥ 1 → goodColoring f c :=
sorry

-- Main theorem for part 2
theorem exists_seven_color_solution :
  ∃ (g : ColoringFunction 7), 
    ∀ (c : Circle), 
      containedIn c (Circle.mk (0, 0) 2) → c.radius ≥ 1 → goodColoring g c :=
sorry

end no_three_color_solution_exists_seven_color_solution_l1145_114532


namespace quadratic_inequality_l1145_114544

theorem quadratic_inequality (x : ℝ) : x^2 - 3*x - 40 > 0 ↔ x < -5 ∨ x > 8 := by
  sorry

end quadratic_inequality_l1145_114544


namespace same_type_ab_squared_and_neg_two_ab_squared_l1145_114588

/-- A polynomial type representing terms of the form c * a^m * b^n where c is a constant -/
structure PolynomialTerm (α : Type*) [CommRing α] where
  coeff : α
  a_exp : ℕ
  b_exp : ℕ

/-- The degree of a polynomial term -/
def PolynomialTerm.degree {α : Type*} [CommRing α] (term : PolynomialTerm α) : ℕ :=
  term.a_exp + term.b_exp

/-- Check if two polynomial terms are of the same type -/
def same_type {α : Type*} [CommRing α] (t1 t2 : PolynomialTerm α) : Prop :=
  t1.a_exp = t2.a_exp ∧ t1.b_exp = t2.b_exp

theorem same_type_ab_squared_and_neg_two_ab_squared 
  {α : Type*} [CommRing α] (a b : α) : 
  let ab_squared : PolynomialTerm α := ⟨1, 1, 2⟩
  let neg_two_ab_squared : PolynomialTerm α := ⟨-2, 1, 2⟩
  same_type ab_squared neg_two_ab_squared ∧ 
  ab_squared.degree = 3 :=
by sorry

end same_type_ab_squared_and_neg_two_ab_squared_l1145_114588


namespace set_intersection_problem_l1145_114580

def A (a : ℝ) : Set ℝ := {3, 4, a^2 - 3*a - 1}
def B (a : ℝ) : Set ℝ := {2*a, -3}

theorem set_intersection_problem (a : ℝ) :
  (A a ∩ B a = {-3}) → a = 1 := by
  sorry

end set_intersection_problem_l1145_114580


namespace solve_for_y_l1145_114556

theorem solve_for_y (x y : ℤ) (h1 : x^2 + 3*x + 6 = y - 2) (h2 : x = -5) : y = 18 := by
  sorry

end solve_for_y_l1145_114556


namespace product_xy_equals_one_l1145_114520

theorem product_xy_equals_one (x y : ℝ) 
  (h : (Real.sqrt (x^2 + 1) - x + 1) * (Real.sqrt (y^2 + 1) - y + 1) = 2) : 
  x * y = 1 := by
  sorry

end product_xy_equals_one_l1145_114520


namespace largest_positive_root_bound_l1145_114586

theorem largest_positive_root_bound (b c : ℝ) (hb : |b| ≤ 3) (hc : |c| ≤ 3) :
  let r := (3 + Real.sqrt 21) / 2
  ∀ x : ℝ, x > 0 → x^2 + b*x + c = 0 → x ≤ r :=
by sorry

end largest_positive_root_bound_l1145_114586


namespace repeating_decimal_one_point_foursix_equals_fraction_l1145_114581

/-- Represents a repeating decimal with a whole number part and a repeating fractional part. -/
structure RepeatingDecimal where
  whole : ℕ
  repeating : ℕ

/-- Converts a RepeatingDecimal to its rational representation. -/
def repeating_decimal_to_rational (d : RepeatingDecimal) : ℚ :=
  d.whole + (d.repeating : ℚ) / (99 : ℚ)

theorem repeating_decimal_one_point_foursix_equals_fraction :
  repeating_decimal_to_rational ⟨1, 46⟩ = 145 / 99 := by
  sorry

end repeating_decimal_one_point_foursix_equals_fraction_l1145_114581


namespace regular_polygon_exterior_angle_l1145_114578

theorem regular_polygon_exterior_angle (n : ℕ) (n_pos : 0 < n) :
  (360 : ℝ) / n = 72 → n = 5 := by
  sorry

end regular_polygon_exterior_angle_l1145_114578


namespace triangle_area_with_given_sides_l1145_114579

theorem triangle_area_with_given_sides :
  let a : ℝ := 65
  let b : ℝ := 60
  let c : ℝ := 25
  let s : ℝ := (a + b + c) / 2
  let area : ℝ := Real.sqrt (s * (s - a) * (s - b) * (s - c))
  area = 750 := by sorry

end triangle_area_with_given_sides_l1145_114579


namespace distance_sum_between_22_and_23_l1145_114521

/-- Given points A, B, and D in a 2D plane, prove that the sum of distances AD and BD 
    is between 22 and 23. -/
theorem distance_sum_between_22_and_23 :
  let A : ℝ × ℝ := (15, 0)
  let B : ℝ × ℝ := (0, 0)
  let D : ℝ × ℝ := (6, 8)
  let distance (p q : ℝ × ℝ) := Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)
  22 < distance A D + distance B D ∧ distance A D + distance B D < 23 := by
  sorry


end distance_sum_between_22_and_23_l1145_114521


namespace min_total_length_l1145_114599

/-- A set of arcs on a circle -/
structure ArcSet :=
  (n : ℕ)                    -- number of arcs
  (arcs : Fin n → ℝ)         -- length of each arc in degrees
  (total_length : ℝ)         -- total length of all arcs
  (rotation_overlap : ∀ θ : ℝ, ∃ i : Fin n, ∃ j : Fin n, (arcs i + θ) % 360 = arcs j)
                             -- for any rotation, there's an overlap

/-- The minimum total length of arcs in an ArcSet is 360/n -/
theorem min_total_length (F : ArcSet) : F.total_length ≥ 360 / F.n :=
sorry

end min_total_length_l1145_114599


namespace cubic_congruence_solutions_l1145_114528

theorem cubic_congruence_solutions :
  ∀ (a b : ℤ),
    (a^3 ≡ b^3 [ZMOD 121] ↔ (a ≡ b [ZMOD 121] ∨ 11 ∣ a ∧ 11 ∣ b)) ∧
    (a^3 ≡ b^3 [ZMOD 169] ↔ (a ≡ b [ZMOD 169] ∨ a ≡ 22*b [ZMOD 169] ∨ a ≡ 146*b [ZMOD 169] ∨ 13 ∣ a ∧ 13 ∣ b)) :=
by sorry

end cubic_congruence_solutions_l1145_114528


namespace star_operation_result_l1145_114566

-- Define the set of elements
inductive Element : Type
  | one : Element
  | two : Element
  | three : Element
  | four : Element

-- Define the operation
def star : Element → Element → Element
  | Element.one, Element.one => Element.four
  | Element.one, Element.two => Element.one
  | Element.one, Element.three => Element.two
  | Element.one, Element.four => Element.three
  | Element.two, Element.one => Element.one
  | Element.two, Element.two => Element.three
  | Element.two, Element.three => Element.four
  | Element.two, Element.four => Element.two
  | Element.three, Element.one => Element.two
  | Element.three, Element.two => Element.four
  | Element.three, Element.three => Element.one
  | Element.three, Element.four => Element.three
  | Element.four, Element.one => Element.three
  | Element.four, Element.two => Element.two
  | Element.four, Element.three => Element.three
  | Element.four, Element.four => Element.four

theorem star_operation_result :
  star (star Element.three Element.one) (star Element.four Element.two) = Element.three := by
  sorry

end star_operation_result_l1145_114566


namespace colored_points_theorem_l1145_114513

theorem colored_points_theorem (r b g : ℕ) (d_rb d_rg d_bg : ℝ) : 
  r + b + g = 15 →
  (r : ℝ) * (b : ℝ) * d_rb = 51 →
  (r : ℝ) * (g : ℝ) * d_rg = 39 →
  (b : ℝ) * (g : ℝ) * d_bg = 1 →
  d_rb > 0 →
  d_rg > 0 →
  d_bg > 0 →
  ((r = 13 ∧ b = 1 ∧ g = 1) ∨ (r = 8 ∧ b = 4 ∧ g = 3)) := by
sorry

end colored_points_theorem_l1145_114513


namespace min_beta_delta_sum_min_beta_delta_value_l1145_114550

open Complex

/-- The function g(z) as defined in the problem -/
def g (β δ : ℂ) (z : ℂ) : ℂ := (3 + 2*I)*z^2 + β*z + δ

/-- The theorem stating the minimum value of |β| + |δ| -/
theorem min_beta_delta_sum :
  ∃ (β δ : ℂ), 
    (g β δ 1).im = 0 ∧ 
    (g β δ (-I)).im = 0 ∧
    ∀ (β' δ' : ℂ), (g β' δ' 1).im = 0 → (g β' δ' (-I)).im = 0 → 
      abs β + abs δ ≤ abs β' + abs δ' :=
by
  sorry

/-- The actual minimum value of |β| + |δ| -/
theorem min_beta_delta_value :
  ∃ (β δ : ℂ), 
    (g β δ 1).im = 0 ∧ 
    (g β δ (-I)).im = 0 ∧
    abs β + abs δ = Real.sqrt 40 :=
by
  sorry

end min_beta_delta_sum_min_beta_delta_value_l1145_114550


namespace inequality_solution_range_l1145_114515

theorem inequality_solution_range (a : ℝ) : 
  (∃ x : ℝ, |x + 2| + |x - 3| ≤ a) → a ≥ 5 := by
  sorry

end inequality_solution_range_l1145_114515


namespace donation_ratio_l1145_114598

theorem donation_ratio (margo_donation julie_donation : ℕ) 
  (h1 : margo_donation = 4300)
  (h2 : julie_donation = 4700) :
  (julie_donation - margo_donation) / (margo_donation + julie_donation) = 2 / 45 := by
  sorry

end donation_ratio_l1145_114598


namespace paper_clip_distribution_l1145_114543

theorem paper_clip_distribution (total_clips : ℕ) (clips_per_box : ℕ) (h1 : total_clips = 81) (h2 : clips_per_box = 9) :
  total_clips / clips_per_box = 9 := by
  sorry

end paper_clip_distribution_l1145_114543


namespace linear_function_intersection_l1145_114576

/-- A linear function y = kx + 2 intersects the x-axis at a point 2 units away from the origin if and only if k = ±1 -/
theorem linear_function_intersection (k : ℝ) : 
  (∃ x : ℝ, k * x + 2 = 0 ∧ |x| = 2) ↔ (k = 1 ∨ k = -1) := by
  sorry

end linear_function_intersection_l1145_114576


namespace neither_odd_nor_even_and_increasing_l1145_114584

-- Define the function f(x) = |x + 1|
def f (x : ℝ) : ℝ := |x + 1|

-- State the theorem
theorem neither_odd_nor_even_and_increasing :
  (¬ ∀ x, f (-x) = -f x) ∧  -- not odd
  (¬ ∀ x, f (-x) = f x) ∧  -- not even
  (∀ x y, 0 < x → x < y → f x < f y) -- monotonically increasing on (0, +∞)
  := by sorry

end neither_odd_nor_even_and_increasing_l1145_114584


namespace parabola_no_x_intersection_l1145_114529

/-- The parabola defined by y = -2x^2 + x - 1 has no intersection with the x-axis -/
theorem parabola_no_x_intersection :
  ∀ x : ℝ, -2 * x^2 + x - 1 ≠ 0 := by
  sorry

end parabola_no_x_intersection_l1145_114529


namespace f5_computation_l1145_114595

/-- A function that represents a boolean operation (AND or OR) -/
def BoolOp : Type := Bool → Bool → Bool

/-- Compute f₅ using only 5 boolean operations -/
def compute_f5 (x₁ x₂ x₃ x₄ x₅ : Bool) (op₁ op₂ op₃ op₄ op₅ : BoolOp) : Bool :=
  let x₆ := op₁ x₁ x₃
  let x₇ := op₂ x₂ x₆
  let x₈ := op₃ x₃ x₅
  let x₉ := op₄ x₄ x₈
  op₅ x₇ x₉

/-- Theorem: f₅ can be computed using only 5 operations of conjunctions and disjunctions -/
theorem f5_computation (x₁ x₂ x₃ x₄ x₅ : Bool) :
  ∃ (op₁ op₂ op₃ op₄ op₅ : BoolOp),
    (∀ a b, op₁ a b = a ∨ b ∨ op₁ a b = a ∧ b) ∧
    (∀ a b, op₂ a b = a ∨ b ∨ op₂ a b = a ∧ b) ∧
    (∀ a b, op₃ a b = a ∨ b ∨ op₃ a b = a ∧ b) ∧
    (∀ a b, op₄ a b = a ∨ b ∨ op₄ a b = a ∧ b) ∧
    (∀ a b, op₅ a b = a ∨ b ∨ op₅ a b = a ∧ b) :=
by
  sorry


end f5_computation_l1145_114595


namespace y_sqrt_x_plus_one_l1145_114510

theorem y_sqrt_x_plus_one (x y k : ℝ) : 
  (y * (Real.sqrt x + 1) = k) →
  (x = 1 ∧ y = 5 → k = 10) ∧
  (y = 2 → x = 16) := by
sorry

end y_sqrt_x_plus_one_l1145_114510


namespace parallel_vectors_x_value_l1145_114514

theorem parallel_vectors_x_value (x : ℝ) 
  (h1 : x > 0) 
  (a : ℝ × ℝ) 
  (b : ℝ × ℝ) 
  (h2 : a = (8 + x/2, x)) 
  (h3 : b = (x + 1, 2)) 
  (h4 : ∃ (k : ℝ), a = k • b) : x = 4 := by
  sorry

end parallel_vectors_x_value_l1145_114514


namespace fraction_subtraction_simplification_l1145_114517

theorem fraction_subtraction_simplification :
  ∃ (a b : ℚ), a = 9/19 ∧ b = 5/57 ∧ a - b = 22/57 ∧ (∀ (c d : ℤ), c ≠ 0 → 22/57 = c/d → (c = 22 ∧ d = 57 ∨ c = -22 ∧ d = -57)) :=
by sorry

end fraction_subtraction_simplification_l1145_114517


namespace pets_remaining_l1145_114533

theorem pets_remaining (initial_pets : ℕ) (lost_pets : ℕ) (death_rate : ℚ) : 
  initial_pets = 16 → 
  lost_pets = 6 → 
  death_rate = 1/5 → 
  initial_pets - lost_pets - (death_rate * (initial_pets - lost_pets : ℚ)).floor = 8 := by
  sorry

end pets_remaining_l1145_114533


namespace last_two_digits_of_7_2017_l1145_114577

-- Define a function to get the last two digits of a number
def lastTwoDigits (n : ℕ) : ℕ := n % 100

-- Define the pattern for the last two digits of powers of 7
def powerOf7Pattern (k : ℕ) : ℕ :=
  match k % 4 with
  | 0 => 01
  | 1 => 07
  | 2 => 49
  | 3 => 43
  | _ => 0  -- This case should never occur

theorem last_two_digits_of_7_2017 :
  lastTwoDigits (7^2017) = 07 :=
by sorry

end last_two_digits_of_7_2017_l1145_114577


namespace function_properties_l1145_114590

/-- Given two real numbers p1 and p2, where p1 ≠ p2, we define two functions f and g. -/
theorem function_properties (p1 p2 : ℝ) (h : p1 ≠ p2) :
  let f := fun x : ℝ => (3 : ℝ) ^ (|x - p1|)
  let g := fun x : ℝ => (3 : ℝ) ^ (|x - p2|)
  -- 1. f can be obtained by translating g
  (∃ k : ℝ, ∀ x : ℝ, f x = g (x + k)) ∧
  -- 2. f + g is symmetric about x = (p1 + p2) / 2
  (∀ x : ℝ, f x + g x = f (p1 + p2 - x) + g (p1 + p2 - x)) ∧
  -- 3. f - g is symmetric about the point ((p1 + p2) / 2, 0)
  (∀ x : ℝ, f x - g x = -(f (p1 + p2 - x) - g (p1 + p2 - x))) :=
by sorry

end function_properties_l1145_114590


namespace inequality_implies_a_range_l1145_114527

theorem inequality_implies_a_range (a : ℝ) : 
  (∀ θ : ℝ, 0 ≤ θ ∧ θ ≤ π/2 → 
    Real.sqrt 2 * (2*a + 3) * Real.cos (θ - π/4) + 6 / (Real.sin θ + Real.cos θ) - 2 * Real.sin (2*θ) < 3*a + 6) 
  → a > 3 := by
  sorry

end inequality_implies_a_range_l1145_114527


namespace sample_represents_knowledge_l1145_114572

/-- Represents the population of teachers and students -/
def Population : ℕ := 1500

/-- Represents the sample size -/
def SampleSize : ℕ := 150

/-- Represents an individual in the population -/
structure Individual where
  id : ℕ
  isTeacher : Bool

/-- Represents the survey sample -/
structure Sample where
  individuals : Finset Individual
  size : ℕ
  h_size : size = SampleSize

/-- Represents the national security knowledge of an individual -/
def NationalSecurityKnowledge : Type := ℕ

/-- The theorem stating what the sample represents in the survey -/
theorem sample_represents_knowledge (sample : Sample) :
  ∃ (knowledge : Individual → NationalSecurityKnowledge),
    (∀ i ∈ sample.individuals, knowledge i ∈ Set.range knowledge) ∧
    (∀ i ∉ sample.individuals, knowledge i ∉ Set.range knowledge) :=
sorry

end sample_represents_knowledge_l1145_114572


namespace distinct_z_values_l1145_114518

def is_valid_number (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

def swap_digits (x : ℕ) : ℕ :=
  let a := x / 100
  let b := (x / 10) % 10
  let c := x % 10
  100 * b + 10 * a + c

def z (x : ℕ) : ℕ := Int.natAbs (x - swap_digits x)

theorem distinct_z_values (x : ℕ) (hx : is_valid_number x) : 
  ∃ (S : Finset ℕ), (∀ n, n ∈ S ↔ ∃ y, is_valid_number y ∧ z y = n) ∧ Finset.card S = 9 :=
sorry

end distinct_z_values_l1145_114518


namespace gcd_90_252_l1145_114537

theorem gcd_90_252 : Nat.gcd 90 252 = 18 := by
  sorry

end gcd_90_252_l1145_114537


namespace equal_perimeters_rectangle_square_l1145_114593

/-- Given two equal lengths of wire, one formed into a rectangle and one formed into a square,
    the perimeters of the resulting shapes are equal. -/
theorem equal_perimeters_rectangle_square (wire_length : ℝ) (h : wire_length > 0) :
  ∃ (rect_width rect_height square_side : ℝ),
    rect_width > 0 ∧ rect_height > 0 ∧ square_side > 0 ∧
    2 * (rect_width + rect_height) = wire_length ∧
    4 * square_side = wire_length ∧
    2 * (rect_width + rect_height) = 4 * square_side :=
by sorry

end equal_perimeters_rectangle_square_l1145_114593


namespace expression_equality_l1145_114553

theorem expression_equality (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (hxyz : x = y * z) :
  (x + y + z)⁻¹ * (x⁻¹ + y⁻¹ + z⁻¹) * (x + y + y * z)⁻¹ * ((x * y)⁻¹ + (y * z)⁻¹ + (x * z)⁻¹) = 
  1 / (y^3 * z^3 * (y + 1)^2) :=
sorry

end expression_equality_l1145_114553


namespace segment_point_difference_l1145_114508

/-- Given a line segment PQ with endpoints P(6,-2) and Q(-3,10), and a point R(a,b) on PQ such that
    the distance from P to R is one-third the distance from P to Q, prove that b-a = -1. -/
theorem segment_point_difference (a b : ℝ) : 
  let p : ℝ × ℝ := (6, -2)
  let q : ℝ × ℝ := (-3, 10)
  let r : ℝ × ℝ := (a, b)
  (r.1 - p.1) / (q.1 - p.1) = (r.2 - p.2) / (q.2 - p.2) ∧  -- R is on PQ
  (r.1 - p.1)^2 + (r.2 - p.2)^2 = (1/9) * ((q.1 - p.1)^2 + (q.2 - p.2)^2) -- PR = (1/3)PQ
  →
  b - a = -1 := by
sorry

end segment_point_difference_l1145_114508


namespace fractional_equation_range_l1145_114506

theorem fractional_equation_range (x a : ℝ) : 
  (2 * x - a) / (x + 1) = 1 → x > 0 → a > -1 := by sorry

end fractional_equation_range_l1145_114506


namespace percentage_problem_l1145_114538

theorem percentage_problem : 
  ∃ P : ℝ, (0.45 * 60 = P * 40 + 13) ∧ P = 0.35 := by
  sorry

end percentage_problem_l1145_114538


namespace smallest_result_is_zero_l1145_114539

def S : Set ℕ := {2, 4, 6, 8, 10, 12}

def operation (a b c : ℕ) : ℕ := ((a + b - c) * c)

theorem smallest_result_is_zero :
  ∃ (a b c : ℕ), a ∈ S ∧ b ∈ S ∧ c ∈ S ∧ a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
  operation a b c = 0 ∧
  ∀ (x y z : ℕ), x ∈ S → y ∈ S → z ∈ S → x ≠ y → y ≠ z → x ≠ z →
  operation x y z ≥ 0 :=
sorry

end smallest_result_is_zero_l1145_114539


namespace max_salary_specific_team_l1145_114523

/-- Represents a basketball team -/
structure BasketballTeam where
  players : Nat
  minSalary : Nat
  salaryCap : Nat

/-- Calculates the maximum possible salary for a single player in a basketball team -/
def maxPlayerSalary (team : BasketballTeam) : Nat :=
  team.salaryCap - (team.players - 1) * team.minSalary

/-- Theorem stating the maximum possible salary for a single player
    in a specific basketball team configuration -/
theorem max_salary_specific_team :
  let team : BasketballTeam := {
    players := 25,
    minSalary := 18000,
    salaryCap := 900000
  }
  maxPlayerSalary team = 468000 := by
  sorry

#eval maxPlayerSalary {players := 25, minSalary := 18000, salaryCap := 900000}

end max_salary_specific_team_l1145_114523


namespace binary_1101011_equals_107_l1145_114562

def binary_to_decimal (binary : List Bool) : ℕ :=
  binary.enum.foldl (fun acc (i, b) => acc + if b then 2^i else 0) 0

theorem binary_1101011_equals_107 :
  binary_to_decimal [true, true, false, true, false, true, true] = 107 := by
  sorry

end binary_1101011_equals_107_l1145_114562


namespace modulus_of_complex_fraction_l1145_114541

theorem modulus_of_complex_fraction : 
  Complex.abs (2 / (1 + Complex.I * Real.sqrt 3)) = 1 := by
  sorry

end modulus_of_complex_fraction_l1145_114541


namespace series_divergence_l1145_114574

theorem series_divergence (a : ℕ → ℝ) 
  (h1 : ∀ n : ℕ, 0 < a n)
  (h2 : ∀ n : ℕ, a n ≤ a (2 * n) + a (2 * n + 1)) :
  ¬ (Summable a) :=
sorry

end series_divergence_l1145_114574


namespace cos_difference_equals_half_l1145_114591

theorem cos_difference_equals_half : 
  Real.cos (24 * π / 180) * Real.cos (36 * π / 180) - 
  Real.cos (66 * π / 180) * Real.cos (54 * π / 180) = 1/2 := by
  sorry

end cos_difference_equals_half_l1145_114591


namespace equidistant_points_on_line_l1145_114570

theorem equidistant_points_on_line :
  ∃ (x₁ y₁ x₂ y₂ : ℝ),
    (4 * x₁ + 3 * y₁ = 12) ∧
    (4 * x₂ + 3 * y₂ = 12) ∧
    (|x₁| = |y₁|) ∧
    (|x₂| = |y₂|) ∧
    (x₁ > 0 ∧ y₁ > 0) ∧
    (x₂ > 0 ∧ y₂ < 0) ∧
    ¬∃ (x₃ y₃ : ℝ),
      (4 * x₃ + 3 * y₃ = 12) ∧
      (|x₃| = |y₃|) ∧
      ((x₃ < 0 ∧ y₃ > 0) ∨ (x₃ < 0 ∧ y₃ < 0)) :=
by sorry

end equidistant_points_on_line_l1145_114570


namespace lcm_from_product_and_hcf_l1145_114519

theorem lcm_from_product_and_hcf (a b : ℕ+) :
  a * b = 84942 ∧ Nat.gcd a b = 33 → Nat.lcm a b = 2574 := by
  sorry

end lcm_from_product_and_hcf_l1145_114519


namespace swim_time_ratio_l1145_114567

/-- The ratio of time taken to swim upstream vs downstream -/
theorem swim_time_ratio 
  (Vm : ℝ) 
  (Vs : ℝ) 
  (h1 : Vm = 5) 
  (h2 : Vs = 1.6666666666666667) : 
  (Vm + Vs) / (Vm - Vs) = 2 := by
  sorry

end swim_time_ratio_l1145_114567


namespace divisor_between_l1145_114509

theorem divisor_between (n a b : ℕ) (hn : n > 8) (ha : a > 0) (hb : b > 0) 
  (hab : a < b) (hdiv_a : a ∣ n) (hdiv_b : b ∣ n) (heq : n = a^2 + b) : 
  ∃ d : ℕ, d ∣ n ∧ a < d ∧ d < b :=
sorry

end divisor_between_l1145_114509


namespace sixth_salary_l1145_114597

def salary_problem (salaries : List ℝ) (mean : ℝ) : Prop :=
  let n : ℕ := salaries.length + 1
  let total : ℝ := salaries.sum
  salaries.length = 5 ∧
  mean * n = total + (n - salaries.length) * (mean * n - total)

theorem sixth_salary :
  ∀ (salaries : List ℝ) (mean : ℝ),
  salary_problem salaries mean →
  (mean * (salaries.length + 1) - salaries.sum) = 2500 :=
by sorry

#check sixth_salary

end sixth_salary_l1145_114597


namespace impossibility_of_forming_parallelepiped_l1145_114504

/-- Represents the dimensions of a rectangular parallelepiped -/
structure Dimensions where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Checks if a parallelepiped can be formed from smaller parallelepipeds -/
def can_form_parallelepiped (large : Dimensions) (small : Dimensions) : Prop :=
  ∃ (n : ℕ), 
    n * (small.length * small.width * small.height) = large.length * large.width * large.height ∧
    ∀ (face : ℕ), face ∈ 
      [large.length * large.width, large.width * large.height, large.length * large.height] →
      ∃ (a b : ℕ), a * small.length * small.width + b * small.length * small.height + 
                   (n - a - b) * small.width * small.height = face

theorem impossibility_of_forming_parallelepiped : 
  ¬ can_form_parallelepiped 
    (Dimensions.mk 3 4 5) 
    (Dimensions.mk 2 2 1) := by
  sorry

end impossibility_of_forming_parallelepiped_l1145_114504


namespace caitlin_bracelets_l1145_114571

/-- The number of bracelets Caitlin can make -/
def num_bracelets : ℕ := 11

/-- The total number of beads Caitlin has -/
def total_beads : ℕ := 528

/-- The number of large beads per bracelet -/
def large_beads_per_bracelet : ℕ := 12

/-- The ratio of small beads to large beads in each bracelet -/
def small_to_large_ratio : ℕ := 2

theorem caitlin_bracelets :
  (total_beads / 2) / (large_beads_per_bracelet * small_to_large_ratio) = num_bracelets :=
sorry

end caitlin_bracelets_l1145_114571


namespace cake_cutting_l1145_114511

theorem cake_cutting (cake_side : ℝ) (num_pieces : ℕ) : 
  cake_side = 15 → num_pieces = 9 → 
  ∃ (piece_side : ℝ), piece_side = 5 ∧ 
  cake_side = piece_side * Real.sqrt (num_pieces : ℝ) := by
sorry

end cake_cutting_l1145_114511


namespace largest_prime_factor_l1145_114561

def expression : ℕ := 16^4 + 2 * 16^2 + 1 - 15^4

theorem largest_prime_factor :
  ∃ (p : ℕ), Nat.Prime p ∧ p ∣ expression ∧ ∀ (q : ℕ), Nat.Prime q → q ∣ expression → q ≤ p :=
by sorry

end largest_prime_factor_l1145_114561


namespace initial_strawberry_jelly_beans_proof_initial_strawberry_jelly_beans_l1145_114547

theorem initial_strawberry_jelly_beans : ℕ → ℕ → Prop :=
  fun s g =>
    s = 3 * g ∧ (s - 15 = 4 * (g - 15)) → s = 135

-- The proof is omitted
theorem proof_initial_strawberry_jelly_beans :
  ∀ s g : ℕ, initial_strawberry_jelly_beans s g :=
sorry

end initial_strawberry_jelly_beans_proof_initial_strawberry_jelly_beans_l1145_114547


namespace largest_class_size_l1145_114551

theorem largest_class_size (total_students : ℕ) (num_classes : ℕ) (diff : ℕ) : 
  total_students = 100 → 
  num_classes = 5 → 
  diff = 2 → 
  (∃ x : ℕ, total_students = x + (x - diff) + (x - 2*diff) + (x - 3*diff) + (x - 4*diff)) → 
  ∃ x : ℕ, x = 24 ∧ total_students = x + (x - diff) + (x - 2*diff) + (x - 3*diff) + (x - 4*diff) :=
by sorry

end largest_class_size_l1145_114551


namespace subset_divisibility_property_l1145_114542

theorem subset_divisibility_property (A : Finset ℕ) (hA : A.card = 3) :
  ∃ B : Finset ℕ, B ⊆ A ∧ B.card = 2 ∧
  ∀ (x y : ℕ) (hx : x ∈ B) (hy : y ∈ B) (m n : ℕ) (hm : Odd m) (hn : Odd n),
  (10 : ℕ) ∣ (x^m * y^n - x^n * y^m) :=
sorry

end subset_divisibility_property_l1145_114542


namespace eleventh_tenth_square_difference_l1145_114524

/-- The side length of the nth square in the sequence -/
def squareSideLength (n : ℕ) : ℕ := 3 + 2 * (n - 1)

/-- The number of tiles in the nth square -/
def squareTiles (n : ℕ) : ℕ := (squareSideLength n) ^ 2

/-- The difference in tiles between the nth and (n-1)th squares -/
def tileDifference (n : ℕ) : ℕ := squareTiles n - squareTiles (n - 1)

theorem eleventh_tenth_square_difference :
  tileDifference 11 = 88 := by sorry

end eleventh_tenth_square_difference_l1145_114524


namespace cheryl_mms_l1145_114563

/-- The number of m&m's Cheryl ate after lunch -/
def lunch_mms : ℕ := 7

/-- The number of m&m's Cheryl ate after dinner -/
def dinner_mms : ℕ := 5

/-- The number of m&m's Cheryl gave to her sister -/
def sister_mms : ℕ := 13

/-- The total number of m&m's Cheryl had at the beginning -/
def total_mms : ℕ := lunch_mms + dinner_mms + sister_mms

theorem cheryl_mms : total_mms = 25 := by
  sorry

end cheryl_mms_l1145_114563


namespace profit_maximization_l1145_114582

/-- The profit function for a product with cost 20 yuan per kilogram -/
def profit_function (x : ℝ) : ℝ := (x - 20) * (-x + 150)

/-- The sales volume function -/
def sales_volume (x : ℝ) : ℝ := -x + 150

theorem profit_maximization :
  ∃ (max_price max_profit : ℝ),
    (∀ x : ℝ, 20 ≤ x ∧ x ≤ 90 → profit_function x ≤ max_profit) ∧
    max_price = 85 ∧
    max_profit = 4225 ∧
    profit_function max_price = max_profit :=
by sorry

end profit_maximization_l1145_114582


namespace derivative_ln_2x_plus_1_l1145_114589

open Real

theorem derivative_ln_2x_plus_1 (x : ℝ) :
  deriv (fun x => Real.log (2 * x + 1)) x = 2 / (2 * x + 1) := by
  sorry

end derivative_ln_2x_plus_1_l1145_114589


namespace no_real_distinct_roots_l1145_114546

theorem no_real_distinct_roots (k : ℝ) : 
  ¬∃ (x y : ℝ), x ≠ y ∧ x^2 + 2*k*x + 3*k^2 = 0 ∧ y^2 + 2*k*y + 3*k^2 = 0 := by
  sorry

end no_real_distinct_roots_l1145_114546


namespace min_value_reciprocal_sum_l1145_114555

theorem min_value_reciprocal_sum (t q r₁ r₂ : ℝ) : 
  (∀ x, x^2 - t*x + q = 0 ↔ x = r₁ ∨ x = r₂) →
  r₁ + r₂ = r₁^2 + r₂^2 →
  r₁^2 + r₂^2 = r₁^4 + r₂^4 →
  ∃ (min : ℝ), min = 2 ∧ ∀ (s t : ℝ), 
    (∀ x, x^2 - s*x + t = 0 ↔ x = r₁ ∨ x = r₂) →
    r₁ + r₂ = r₁^2 + r₂^2 →
    r₁^2 + r₂^2 = r₁^4 + r₂^4 →
    min ≤ 1/r₁^5 + 1/r₂^5 :=
by sorry

end min_value_reciprocal_sum_l1145_114555


namespace arrange_products_eq_eight_l1145_114503

/-- The number of ways to arrange 4 different products in a row,
    with products A and B both to the left of product C -/
def arrange_products : ℕ :=
  let n : ℕ := 4  -- Total number of products
  let ways_to_arrange_AB : ℕ := 2  -- Number of ways to arrange A and B
  let positions_for_last_product : ℕ := 4  -- Possible positions for the last product
  ways_to_arrange_AB * positions_for_last_product

/-- Theorem stating that the number of arrangements is 8 -/
theorem arrange_products_eq_eight : arrange_products = 8 := by
  sorry

end arrange_products_eq_eight_l1145_114503


namespace banana_group_size_l1145_114592

theorem banana_group_size 
  (total_bananas : ℕ) 
  (num_groups : ℕ) 
  (h1 : total_bananas = 392) 
  (h2 : num_groups = 196) : 
  total_bananas / num_groups = 2 := by
sorry

end banana_group_size_l1145_114592


namespace revenue_maximizing_price_l1145_114560

/-- Revenue function for the bookstore --/
def revenue (p : ℝ) : ℝ := p * (200 - 6 * p)

/-- The maximum price constraint --/
def max_price : ℝ := 30

/-- Theorem stating the price that maximizes revenue --/
theorem revenue_maximizing_price :
  ∃ (p : ℝ), p ≤ max_price ∧ 
  ∀ (q : ℝ), q ≤ max_price → revenue q ≤ revenue p ∧
  p = 50 / 3 := by
  sorry

end revenue_maximizing_price_l1145_114560


namespace coefficient_without_x_is_70_l1145_114540

/-- The coefficient of the term without x in (xy - 1/x)^8 -/
def coefficientWithoutX : ℕ :=
  Nat.choose 8 4

/-- Theorem: The coefficient of the term without x in (xy - 1/x)^8 is 70 -/
theorem coefficient_without_x_is_70 : coefficientWithoutX = 70 := by
  sorry

end coefficient_without_x_is_70_l1145_114540


namespace skirt_cut_amount_l1145_114587

/-- The amount cut off the pants in inches -/
def pants_cut : ℝ := 0.5

/-- The additional amount cut off the skirt compared to the pants in inches -/
def additional_skirt_cut : ℝ := 0.25

/-- The total amount cut off the skirt in inches -/
def skirt_cut : ℝ := pants_cut + additional_skirt_cut

theorem skirt_cut_amount : skirt_cut = 0.75 := by sorry

end skirt_cut_amount_l1145_114587


namespace side_c_value_l1145_114575

/-- Given an acute triangle ABC with sides a = 4, b = 5, and area 5√3, 
    prove that the length of side c is √21 -/
theorem side_c_value (A B C : ℝ) (a b c : ℝ) (h_acute : A + B + C = π) 
  (h_a : a = 4) (h_b : b = 5) (h_area : (1/2) * a * b * Real.sin C = 5 * Real.sqrt 3) :
  c = Real.sqrt 21 := by
  sorry


end side_c_value_l1145_114575


namespace min_correct_answers_l1145_114501

/-- Represents the scoring system and conditions of the IQ test -/
structure IQTest where
  total_questions : ℕ
  correct_points : ℕ
  wrong_points : ℕ
  unanswered : ℕ
  min_score : ℕ

/-- Calculates the score based on the number of correct answers -/
def calculate_score (test : IQTest) (correct_answers : ℕ) : ℤ :=
  (correct_answers : ℤ) * test.correct_points - 
  ((test.total_questions - test.unanswered - correct_answers) : ℤ) * test.wrong_points

/-- Theorem stating the minimum number of correct answers needed to achieve the minimum score -/
theorem min_correct_answers (test : IQTest) : 
  test.total_questions = 20 ∧ 
  test.correct_points = 5 ∧ 
  test.wrong_points = 2 ∧ 
  test.unanswered = 2 ∧ 
  test.min_score = 60 →
  (∀ x : ℕ, x < 14 → calculate_score test x < test.min_score) ∧
  calculate_score test 14 ≥ test.min_score := by
  sorry


end min_correct_answers_l1145_114501


namespace vertical_motion_time_relation_l1145_114531

/-- Represents the vertical motion of a ball thrown upward and returning to its starting point. -/
structure VerticalMotion where
  V₀ : ℝ  -- Initial velocity
  g : ℝ   -- Gravitational acceleration
  t₁ : ℝ  -- Time to reach maximum height
  H : ℝ   -- Maximum height
  t : ℝ   -- Total time of motion

/-- The theorem stating the relationship between initial velocity, gravity, and total time of motion. -/
theorem vertical_motion_time_relation (motion : VerticalMotion)
  (h_positive_V₀ : 0 < motion.V₀)
  (h_positive_g : 0 < motion.g)
  (h_max_height : motion.H = (1/2) * motion.g * motion.t₁^2)
  (h_symmetry : motion.t = 2 * motion.t₁) :
  motion.t = 2 * motion.V₀ / motion.g :=
by sorry

end vertical_motion_time_relation_l1145_114531


namespace equation_solutions_l1145_114573

theorem equation_solutions :
  (∃ (x : ℝ), (1/3) * (x - 3)^2 = 12 ↔ x = 9 ∨ x = -3) ∧
  (∃ (x : ℝ), (2*x - 1)^2 = (1 - x)^2 ↔ x = 2/3 ∨ x = 0) := by
  sorry

end equation_solutions_l1145_114573


namespace dans_initial_money_l1145_114559

def candy_price : ℕ := 2
def chocolate_price : ℕ := 3

theorem dans_initial_money :
  ∀ (initial_money : ℕ),
  (initial_money = candy_price + chocolate_price) ∧
  (chocolate_price - candy_price = 1) →
  initial_money = 5 := by
sorry

end dans_initial_money_l1145_114559


namespace electrician_wage_l1145_114526

theorem electrician_wage (total_hours : ℝ) (bricklayer_wage : ℝ) (total_payment : ℝ) (individual_hours : ℝ)
  (h1 : total_hours = 90)
  (h2 : bricklayer_wage = 12)
  (h3 : total_payment = 1350)
  (h4 : individual_hours = 22.5) :
  (total_payment - bricklayer_wage * individual_hours) / individual_hours = 48 := by
sorry

end electrician_wage_l1145_114526


namespace extreme_value_condition_l1145_114507

/-- The cubic function with parameter a -/
def f (a : ℝ) (x : ℝ) : ℝ := x^3 + a*x^2 + (a+6)*x + 1

/-- The derivative of f with respect to x -/
def f' (a : ℝ) (x : ℝ) : ℝ := 3*x^2 + 2*a*x + (a+6)

/-- Theorem stating the condition for f to have extreme values on ℝ -/
theorem extreme_value_condition (a : ℝ) : 
  (∃ x : ℝ, (f' a x = 0 ∧ ∀ y : ℝ, f' a y = 0 → y = x) → False) ↔ (a > 6 ∨ a < -3) :=
sorry

end extreme_value_condition_l1145_114507


namespace average_marks_combined_l1145_114557

theorem average_marks_combined (n1 n2 : ℕ) (avg1 avg2 : ℚ) :
  n1 = 30 →
  n2 = 50 →
  avg1 = 40 →
  avg2 = 70 →
  (n1 * avg1 + n2 * avg2) / (n1 + n2 : ℚ) = 58.75 := by
  sorry

end average_marks_combined_l1145_114557


namespace greatest_prime_factor_of_sum_factorials_l1145_114512

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

theorem greatest_prime_factor_of_sum_factorials :
  ∃ p : ℕ, Nat.Prime p ∧ p ∣ (factorial 15 + factorial 18) ∧
  ∀ q : ℕ, Nat.Prime q → q ∣ (factorial 15 + factorial 18) → q ≤ p :=
by sorry

end greatest_prime_factor_of_sum_factorials_l1145_114512


namespace exists_abc_sum_product_l1145_114549

def NatPos := {n : ℕ | n > 0}

def A : Set ℕ := {x | ∃ m ∈ NatPos, x = 3 * m}
def B : Set ℕ := {x | ∃ m ∈ NatPos, x = 3 * m - 1}
def C : Set ℕ := {x | ∃ m ∈ NatPos, x = 3 * m - 2}

theorem exists_abc_sum_product (a : ℕ) (b : ℕ) (c : ℕ) 
  (ha : a ∈ A) (hb : b ∈ B) (hc : c ∈ C) :
  ∃ a b c, a ∈ A ∧ b ∈ B ∧ c ∈ C ∧ 2006 = a + b * c :=
by sorry

end exists_abc_sum_product_l1145_114549


namespace rectangle_dimensions_l1145_114585

/-- Given a rectangle where the length is twice the width and the perimeter in inches
    equals the area in square inches, prove that the width is 3 inches and the length is 6 inches. -/
theorem rectangle_dimensions (w : ℝ) (h1 : w > 0) :
  (6 * w = 2 * w^2) → (w = 3 ∧ 2 * w = 6) := by
  sorry

end rectangle_dimensions_l1145_114585


namespace veranda_area_l1145_114536

/-- Given a rectangular room with length 17 m and width 12 m, surrounded by a veranda of width 2 m on all sides, the area of the veranda is 132 m². -/
theorem veranda_area (room_length : ℝ) (room_width : ℝ) (veranda_width : ℝ) :
  room_length = 17 →
  room_width = 12 →
  veranda_width = 2 →
  (room_length + 2 * veranda_width) * (room_width + 2 * veranda_width) - room_length * room_width = 132 :=
by sorry

end veranda_area_l1145_114536


namespace problem_1_l1145_114596

theorem problem_1 : Real.sqrt 8 / Real.sqrt 2 + (Real.sqrt 5 + 3) * (Real.sqrt 5 - 3) = -2 := by
  sorry

end problem_1_l1145_114596
