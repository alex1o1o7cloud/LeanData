import Mathlib

namespace right_angled_projection_l1931_193199

structure Plane where
  α : Type

structure Point where
  x : ℝ
  y : ℝ
  z : ℝ

def Triangle (A B C : Point) := True

def RightAngledTriangle (A B C : Point) := Triangle A B C

def IsInPlane (p : Point) (α : Plane) : Prop := sorry

def IsOutsidePlane (p : Point) (α : Plane) : Prop := sorry

def Projection (p : Point) (α : Plane) : Point := sorry

def IsOn (p : Point) (A B : Point) : Prop := sorry

theorem right_angled_projection 
  (α : Plane) (A B C C1 : Point) : 
  RightAngledTriangle A B C →
  IsInPlane A α →
  IsInPlane B α →
  IsOutsidePlane C α →
  C1 = Projection C α →
  ¬IsOn C1 A B →
  RightAngledTriangle A B C1 := by sorry

end right_angled_projection_l1931_193199


namespace arithmetic_sequence_ratio_l1931_193132

/-- Given two arithmetic sequences {a_n} and {b_n} with sums A_n and B_n,
    if A_n / B_n = (7n + 45) / (n + 3) for all n, then a_5 / b_5 = 9 -/
theorem arithmetic_sequence_ratio (a b : ℕ → ℚ) (A B : ℕ → ℚ) :
  (∀ n, A n = (n / 2) * (a 1 + a n)) →
  (∀ n, B n = (n / 2) * (b 1 + b n)) →
  (∀ n, A n / B n = (7 * n + 45) / (n + 3)) →
  a 5 / b 5 = 9 := by
  sorry

end arithmetic_sequence_ratio_l1931_193132


namespace correct_mean_after_error_correction_l1931_193136

theorem correct_mean_after_error_correction (n : ℕ) (incorrect_mean correct_value incorrect_value : ℝ) :
  n = 30 →
  incorrect_mean = 250 →
  correct_value = 165 →
  incorrect_value = 135 →
  (n : ℝ) * incorrect_mean + (correct_value - incorrect_value) = n * 251 := by
  sorry

end correct_mean_after_error_correction_l1931_193136


namespace functions_are_odd_l1931_193176

-- Define the property for functions f and g
def has_property (f g : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (g x) = g (f x) ∧ f (g x) = -x

-- Define what it means for a function to be odd
def is_odd (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f x

-- Theorem statement
theorem functions_are_odd (f g : ℝ → ℝ) (h : has_property f g) :
  is_odd f ∧ is_odd g :=
sorry

end functions_are_odd_l1931_193176


namespace total_apples_is_45_l1931_193122

/-- The number of apples given to each person -/
def apples_per_person : ℝ := 15.0

/-- The number of people who received apples -/
def number_of_people : ℝ := 3.0

/-- The total number of apples given -/
def total_apples : ℝ := apples_per_person * number_of_people

/-- Theorem stating that the total number of apples is 45.0 -/
theorem total_apples_is_45 : total_apples = 45.0 := by
  sorry

end total_apples_is_45_l1931_193122


namespace unique_base_thirteen_l1931_193128

/-- Converts a digit character to its numeric value -/
def char_to_digit (c : Char) : ℕ :=
  if c.isDigit then c.toNat - '0'.toNat
  else if c = 'A' then 10
  else if c = 'B' then 11
  else if c = 'C' then 12
  else 0

/-- Converts a string representation of a number in base a to its decimal value -/
def to_decimal (s : String) (a : ℕ) : ℕ :=
  s.foldr (fun c acc => char_to_digit c + a * acc) 0

/-- Checks if the equation 375_a + 592_a = 9C7_a is satisfied for a given base a -/
def equation_satisfied (a : ℕ) : Prop :=
  to_decimal "375" a + to_decimal "592" a = to_decimal "9C7" a

theorem unique_base_thirteen :
  ∃! a : ℕ, a > 12 ∧ equation_satisfied a ∧ char_to_digit 'C' = 12 :=
sorry

end unique_base_thirteen_l1931_193128


namespace intersection_complement_M_and_N_l1931_193170

def M : Set ℝ := {x : ℝ | x^2 - 3*x - 4 ≥ 0}
def N : Set ℝ := {x : ℝ | 1 < x ∧ x < 5}

theorem intersection_complement_M_and_N :
  (Mᶜ ∩ N) = {x : ℝ | 1 < x ∧ x < 4} :=
sorry

end intersection_complement_M_and_N_l1931_193170


namespace road_signs_ratio_l1931_193193

/-- Represents the number of road signs at each intersection -/
structure RoadSigns where
  s1 : ℕ  -- First intersection
  s2 : ℕ  -- Second intersection
  s3 : ℕ  -- Third intersection
  s4 : ℕ  -- Fourth intersection

/-- Theorem stating the ratio of road signs at the third to second intersection -/
theorem road_signs_ratio 
  (signs : RoadSigns) 
  (h1 : signs.s1 = 40)
  (h2 : signs.s2 = signs.s1 + signs.s1 / 4)
  (h3 : signs.s4 = signs.s3 - 20)
  (h4 : signs.s1 + signs.s2 + signs.s3 + signs.s4 = 270) :
  signs.s3 / signs.s2 = 2 := by
  sorry

#eval (100 : ℚ) / 50  -- Expected output: 2

end road_signs_ratio_l1931_193193


namespace value_of_a_l1931_193181

theorem value_of_a (a b c : ℝ) 
  (eq1 : a + b = c) 
  (eq2 : b + c = 6) 
  (eq3 : c = 4) : 
  a = 2 := by
sorry

end value_of_a_l1931_193181


namespace polynomial_expansion_l1931_193190

theorem polynomial_expansion (x : ℝ) : 
  (7 * x^2 + 5 - 3 * x) * (4 * x^3) = 28 * x^5 - 12 * x^4 + 20 * x^3 := by
  sorry

end polynomial_expansion_l1931_193190


namespace susan_coins_value_l1931_193186

theorem susan_coins_value :
  ∀ (n d : ℕ),
  n + d = 30 →
  5 * n + 10 * d + 90 = 10 * n + 5 * d →
  5 * n + 10 * d = 180 := by
sorry

end susan_coins_value_l1931_193186


namespace spotted_fluffy_cats_l1931_193196

def village_cats : ℕ := 120

def spotted_fraction : ℚ := 1/3

def fluffy_spotted_fraction : ℚ := 1/4

theorem spotted_fluffy_cats :
  (village_cats : ℚ) * spotted_fraction * fluffy_spotted_fraction = 10 := by
  sorry

end spotted_fluffy_cats_l1931_193196


namespace gp_sum_ratio_l1931_193137

/-- For a geometric progression with common ratio 3, the ratio of the sum
    of the first 6 terms to the sum of the first 3 terms is 28. -/
theorem gp_sum_ratio (a : ℝ) : 
  let r := 3
  let S₃ := a * (1 - r^3) / (1 - r)
  let S₆ := a * (1 - r^6) / (1 - r)
  S₆ / S₃ = 28 := by
sorry


end gp_sum_ratio_l1931_193137


namespace sphere_volume_to_surface_area_l1931_193155

theorem sphere_volume_to_surface_area :
  ∀ (r : ℝ), 
    (4 / 3 * π * r^3 = 32 * π / 3) →
    (4 * π * r^2 = 16 * π) :=
by sorry

end sphere_volume_to_surface_area_l1931_193155


namespace profit_at_45_price_for_1200_profit_l1931_193160

/-- Represents the craft selling scenario -/
structure CraftSelling where
  cost_price : ℕ
  base_price : ℕ
  base_volume : ℕ
  price_volume_ratio : ℕ
  max_price : ℕ

/-- Calculates the daily sales volume based on the selling price -/
def daily_volume (cs : CraftSelling) (price : ℕ) : ℤ :=
  cs.base_volume - cs.price_volume_ratio * (price - cs.base_price)

/-- Calculates the daily profit based on the selling price -/
def daily_profit (cs : CraftSelling) (price : ℕ) : ℤ :=
  (price - cs.cost_price) * daily_volume cs price

/-- The craft selling scenario for the given problem -/
def craft_scenario : CraftSelling := {
  cost_price := 30
  base_price := 40
  base_volume := 80
  price_volume_ratio := 2
  max_price := 55
}

/-- Theorem for the daily sales profit at 45 yuan -/
theorem profit_at_45 : daily_profit craft_scenario 45 = 1050 := by sorry

/-- Theorem for the selling price that achieves 1200 yuan daily profit -/
theorem price_for_1200_profit :
  ∃ (price : ℕ), price ≤ craft_scenario.max_price ∧ daily_profit craft_scenario price = 1200 ∧
  ∀ (p : ℕ), p ≤ craft_scenario.max_price → daily_profit craft_scenario p = 1200 → p = price := by sorry

end profit_at_45_price_for_1200_profit_l1931_193160


namespace f_of_3_equals_10_l1931_193121

def f (x : ℝ) : ℝ := 3 * x + 1

theorem f_of_3_equals_10 : f 3 = 10 := by
  sorry

end f_of_3_equals_10_l1931_193121


namespace reading_homework_pages_l1931_193129

theorem reading_homework_pages
  (math_pages : ℕ)
  (problems_per_page : ℕ)
  (total_problems : ℕ)
  (h1 : math_pages = 6)
  (h2 : problems_per_page = 3)
  (h3 : total_problems = 30) :
  (total_problems - math_pages * problems_per_page) / problems_per_page = 4 :=
by sorry

end reading_homework_pages_l1931_193129


namespace no_finite_k_with_zero_difference_l1931_193131

def u (n : ℕ) : ℕ := n^4 + n^2

def Δ : (ℕ → ℕ) → (ℕ → ℕ)
  | f => fun n => f (n + 1) - f n

def iteratedΔ : ℕ → (ℕ → ℕ) → (ℕ → ℕ)
  | 0 => id
  | k + 1 => Δ ∘ iteratedΔ k

theorem no_finite_k_with_zero_difference :
  ∀ k : ℕ, ∃ n : ℕ, (iteratedΔ k u) n ≠ 0 := by sorry

end no_finite_k_with_zero_difference_l1931_193131


namespace new_average_salary_l1931_193169

/-- Calculates the new average monthly salary after a change in supervisor --/
theorem new_average_salary
  (num_people : ℕ)
  (num_workers : ℕ)
  (old_average : ℚ)
  (old_supervisor_salary : ℚ)
  (new_supervisor_salary : ℚ)
  (h_num_people : num_people = 9)
  (h_num_workers : num_workers = 8)
  (h_old_average : old_average = 430)
  (h_old_supervisor : old_supervisor_salary = 870)
  (h_new_supervisor : new_supervisor_salary = 960) :
  (num_people * old_average - old_supervisor_salary + new_supervisor_salary) / num_people = 440 :=
sorry

end new_average_salary_l1931_193169


namespace chicken_rabbit_problem_l1931_193184

theorem chicken_rabbit_problem :
  ∀ (chickens rabbits : ℕ),
    chickens + rabbits = 15 →
    2 * chickens + 4 * rabbits = 40 →
    chickens = 10 ∧ rabbits = 5 := by
  sorry

end chicken_rabbit_problem_l1931_193184


namespace shirt_markup_l1931_193140

theorem shirt_markup (P : ℝ) (h : 2 * P - 1.8 * P = 5) : 1.8 * P = 45 := by
  sorry

end shirt_markup_l1931_193140


namespace problem_statement_l1931_193163

theorem problem_statement (a b m n : ℝ) : 
  a * m^2001 + b * n^2001 = 3 →
  a * m^2002 + b * n^2002 = 7 →
  a * m^2003 + b * n^2003 = 24 →
  a * m^2004 + b * n^2004 = 102 →
  m^2 * (n - 1) = 6 := by
sorry

end problem_statement_l1931_193163


namespace total_points_target_l1931_193174

def average_points_after_two_games : ℝ := 61.5
def points_in_game_three : ℕ := 47
def additional_points_needed : ℕ := 330

theorem total_points_target :
  (2 * average_points_after_two_games + points_in_game_three + additional_points_needed : ℝ) = 500 := by
  sorry

end total_points_target_l1931_193174


namespace y_to_x_equals_25_l1931_193102

theorem y_to_x_equals_25 (x y : ℝ) (h : |x - 2| + (y + 5)^2 = 0) : y^x = 25 := by
  sorry

end y_to_x_equals_25_l1931_193102


namespace geometric_jump_sequence_ratio_l1931_193156

/-- A sequence is a jump sequence if (a_i - a_i+2)(a_i+2 - a_i+1) > 0 for any three consecutive terms -/
def is_jump_sequence (a : ℕ → ℝ) : Prop :=
  ∀ i : ℕ, (a i - a (i + 2)) * (a (i + 2) - a (i + 1)) > 0

/-- A sequence is geometric with ratio q if a_(n+1) = q * a_n for all n -/
def is_geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = q * a n

theorem geometric_jump_sequence_ratio {a : ℕ → ℝ} {q : ℝ} 
  (h_geometric : is_geometric_sequence a q)
  (h_jump : is_jump_sequence a) :
  -1 < q ∧ q < 0 :=
sorry

end geometric_jump_sequence_ratio_l1931_193156


namespace line_slope_l1931_193151

/-- Given a line l with equation y = (1/2)x + 1, its slope is 1/2 -/
theorem line_slope (l : Set (ℝ × ℝ)) (h : l = {(x, y) | y = (1/2) * x + 1}) :
  (∃ m : ℝ, ∀ (x₁ y₁ x₂ y₂ : ℝ), (x₁, y₁) ∈ l → (x₂, y₂) ∈ l → x₁ ≠ x₂ → 
    m = (y₂ - y₁) / (x₂ - x₁)) ∧ 
  (∀ m : ℝ, (∀ (x₁ y₁ x₂ y₂ : ℝ), (x₁, y₁) ∈ l → (x₂, y₂) ∈ l → x₁ ≠ x₂ → 
    m = (y₂ - y₁) / (x₂ - x₁)) → m = 1/2) :=
by sorry

end line_slope_l1931_193151


namespace unique_function_satisfying_conditions_l1931_193104

noncomputable def f (x : ℝ) : ℝ :=
  if 0 ≤ x ∧ x < 2 then 2 / (2 - x) else 0

theorem unique_function_satisfying_conditions :
  (∀ x y, x ≥ 0 → y ≥ 0 → f (x * f y) * f y = f (x + y)) ∧
  (f 2 = 0) ∧
  (∀ x, 0 ≤ x → x < 2 → f x ≠ 0) ∧
  (∀ g : ℝ → ℝ, (∀ x y, x ≥ 0 → y ≥ 0 → g (x * g y) * g y = g (x + y)) →
    (g 2 = 0) →
    (∀ x, 0 ≤ x → x < 2 → g x ≠ 0) →
    (∀ x, x ≥ 0 → g x = f x)) :=
by sorry

end unique_function_satisfying_conditions_l1931_193104


namespace garden_ratio_l1931_193171

/-- A rectangular garden with given perimeter and length has a specific length-to-width ratio -/
theorem garden_ratio (perimeter length width : ℝ) : 
  perimeter = 300 →
  length = 100 →
  perimeter = 2 * length + 2 * width →
  length / width = 2 := by
  sorry

end garden_ratio_l1931_193171


namespace gcd_45_75_l1931_193103

theorem gcd_45_75 : Nat.gcd 45 75 = 15 := by
  sorry

end gcd_45_75_l1931_193103


namespace mary_farm_animals_l1931_193115

-- Define the initial state and transactions
def initial_lambs : ℕ := 18
def initial_alpacas : ℕ := 5
def lamb_babies : ℕ := 7 * 4
def traded_lambs : ℕ := 8
def traded_alpacas : ℕ := 2
def gained_goats : ℕ := 3
def gained_chickens : ℕ := 10
def alpacas_from_chickens : ℕ := 2
def additional_lambs : ℕ := 20
def additional_alpacas : ℕ := 6

-- Define the theorem
theorem mary_farm_animals :
  let lambs := initial_lambs + lamb_babies - traded_lambs + additional_lambs
  let alpacas := initial_alpacas - traded_alpacas + alpacas_from_chickens + additional_alpacas
  let goats := gained_goats
  let chickens := gained_chickens / 2
  (lambs = 58 ∧ alpacas = 11 ∧ goats = 3 ∧ chickens = 5) := by
  sorry

end mary_farm_animals_l1931_193115


namespace inequality_constraint_l1931_193114

theorem inequality_constraint (a b : ℝ) : 
  (∀ x : ℝ, |a * Real.sin x + b * Real.sin (2 * x)| ≤ 1) →
  |a| + |b| ≥ 2 / Real.sqrt 3 →
  ((a = 4 / (3 * Real.sqrt 3) ∧ b = 2 / (3 * Real.sqrt 3)) ∨
   (a = -4 / (3 * Real.sqrt 3) ∧ b = -2 / (3 * Real.sqrt 3)) ∨
   (a = 4 / (3 * Real.sqrt 3) ∧ b = -2 / (3 * Real.sqrt 3)) ∨
   (a = -4 / (3 * Real.sqrt 3) ∧ b = 2 / (3 * Real.sqrt 3))) :=
by sorry


end inequality_constraint_l1931_193114


namespace tan_105_degrees_l1931_193187

theorem tan_105_degrees : Real.tan (105 * π / 180) = -2 - Real.sqrt 3 := by
  sorry

end tan_105_degrees_l1931_193187


namespace tess_decoration_l1931_193133

/-- The number of heart stickers Tess has -/
def heart_stickers : ℕ := 120

/-- The number of star stickers Tess has -/
def star_stickers : ℕ := 81

/-- The number of smiley stickers Tess has -/
def smiley_stickers : ℕ := 45

/-- The greatest number of pages Tess can decorate -/
def max_pages : ℕ := Nat.gcd (Nat.gcd heart_stickers star_stickers) smiley_stickers

theorem tess_decoration :
  max_pages = 3 ∧
  heart_stickers % max_pages = 0 ∧
  star_stickers % max_pages = 0 ∧
  smiley_stickers % max_pages = 0 ∧
  ∀ n : ℕ, n > max_pages →
    (heart_stickers % n = 0 ∧ star_stickers % n = 0 ∧ smiley_stickers % n = 0) → False :=
by sorry

end tess_decoration_l1931_193133


namespace symmetry_axis_implies_line_l1931_193107

/-- Given a function f(x) = a*sin(x) + b*cos(x) where x is real,
    if x₀ is an axis of symmetry for f(x) and tan(x₀) = 2,
    then the point (a,b) lies on the line x - 2y = 0. -/
theorem symmetry_axis_implies_line (a b x₀ : ℝ) :
  let f := fun (x : ℝ) ↦ a * Real.sin x + b * Real.cos x
  (∀ x, f (x₀ + x) = f (x₀ - x)) →  -- x₀ is an axis of symmetry
  Real.tan x₀ = 2 →
  a - 2 * b = 0 := by
  sorry

end symmetry_axis_implies_line_l1931_193107


namespace product_of_cosines_l1931_193175

theorem product_of_cosines : 
  (1 + Real.cos (π / 9)) * (1 + Real.cos (2 * π / 9)) * 
  (1 + Real.cos (8 * π / 9)) * (1 + Real.cos (7 * π / 9)) = 3 / 16 := by
  sorry

end product_of_cosines_l1931_193175


namespace min_value_theorem_l1931_193101

theorem min_value_theorem (x : ℝ) (h : x > 0) :
  x^4 + 16*x + 256/x^6 ≥ 56 ∧
  (x^4 + 16*x + 256/x^6 = 56 ↔ x = 2) := by
  sorry

end min_value_theorem_l1931_193101


namespace min_value_complex_expression_l1931_193192

theorem min_value_complex_expression (z : ℂ) (h : Complex.abs z = 1) :
  Complex.abs (z^2 - 2*z + 3) ≥ 2 * Real.sqrt 6 / 3 := by
  sorry

end min_value_complex_expression_l1931_193192


namespace sum_of_fractions_l1931_193105

theorem sum_of_fractions : 
  (2 : ℚ) / 100 + 5 / 1000 + 8 / 10000 + 6 / 100000 = 0.02586 := by
  sorry

end sum_of_fractions_l1931_193105


namespace parallelepiped_height_l1931_193141

/-- The surface area of a rectangular parallelepiped -/
def surface_area (l w h : ℝ) : ℝ := 2*l*w + 2*l*h + 2*w*h

/-- Theorem: The height of a rectangular parallelepiped with given dimensions -/
theorem parallelepiped_height (w l : ℝ) (h : ℝ) :
  w = 7 → l = 8 → surface_area l w h = 442 → h = 11 := by
  sorry

end parallelepiped_height_l1931_193141


namespace total_balls_l1931_193119

theorem total_balls (S V B : ℕ) : 
  S = 68 ∧ 
  S = V - 12 ∧ 
  S = B + 23 → 
  S + V + B = 193 := by
sorry

end total_balls_l1931_193119


namespace rectangular_field_with_pond_l1931_193139

theorem rectangular_field_with_pond (l w : ℝ) : 
  l = 2 * w →                    -- length is double the width
  l * w = 8 * 49 →               -- area of field is 8 times area of pond (7^2 = 49)
  l = 28 := by
sorry

end rectangular_field_with_pond_l1931_193139


namespace polynomial_equality_l1931_193189

theorem polynomial_equality (p : ℝ → ℝ) :
  (∀ x, p x + (2*x^6 + 4*x^4 + 6*x^2) = 8*x^4 + 27*x^3 + 33*x^2 + 15*x + 5) →
  (∀ x, p x = -2*x^6 + 4*x^4 + 27*x^3 + 27*x^2 + 15*x + 5) := by
sorry

end polynomial_equality_l1931_193189


namespace fourth_intersection_point_l1931_193142

/-- A point in the 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- The hyperbola xy = 1 -/
def hyperbola (p : Point) : Prop := p.x * p.y = 1

/-- Four points lie on the same circle -/
def on_same_circle (p1 p2 p3 p4 : Point) : Prop := 
  ∃ (h k s : ℝ), 
    (p1.x - h)^2 + (p1.y - k)^2 = s^2 ∧
    (p2.x - h)^2 + (p2.y - k)^2 = s^2 ∧
    (p3.x - h)^2 + (p3.y - k)^2 = s^2 ∧
    (p4.x - h)^2 + (p4.y - k)^2 = s^2

theorem fourth_intersection_point : 
  let p1 : Point := ⟨3, 1/3⟩
  let p2 : Point := ⟨-4, -1/4⟩
  let p3 : Point := ⟨1/6, 6⟩
  let p4 : Point := ⟨-1/2, -2⟩
  hyperbola p1 ∧ hyperbola p2 ∧ hyperbola p3 ∧ hyperbola p4 ∧
  on_same_circle p1 p2 p3 p4 := by
  sorry

end fourth_intersection_point_l1931_193142


namespace tent_donation_problem_l1931_193116

theorem tent_donation_problem (total_tents : ℕ) (total_value : ℕ) 
  (cost_A : ℕ) (cost_B : ℕ) :
  total_tents = 300 →
  total_value = 260000 →
  cost_A = 800 →
  cost_B = 1000 →
  ∃ (num_A num_B : ℕ),
    num_A + num_B = total_tents ∧
    num_A * cost_A + num_B * cost_B = total_value ∧
    num_A = 200 ∧
    num_B = 100 :=
by sorry

end tent_donation_problem_l1931_193116


namespace market_qualified_product_probability_l1931_193165

theorem market_qualified_product_probability :
  let market_share_A : ℝ := 0.8
  let market_share_B : ℝ := 0.2
  let qualification_rate_A : ℝ := 0.75
  let qualification_rate_B : ℝ := 0.8
  market_share_A * qualification_rate_A + market_share_B * qualification_rate_B = 0.76 :=
by sorry

end market_qualified_product_probability_l1931_193165


namespace probability_at_least_one_white_ball_l1931_193138

theorem probability_at_least_one_white_ball (total_balls : ℕ) (red_balls : ℕ) (white_balls : ℕ) :
  total_balls = red_balls + white_balls →
  red_balls = 5 →
  white_balls = 4 →
  (1 - (red_balls / total_balls * (red_balls - 1) / (total_balls - 1))) = 13 / 18 := by
  sorry

end probability_at_least_one_white_ball_l1931_193138


namespace no_tetrahedron_with_heights_1_2_3_6_l1931_193106

/-- Represents a tetrahedron with four heights -/
structure Tetrahedron where
  h1 : ℝ
  h2 : ℝ
  h3 : ℝ
  h4 : ℝ

/-- The property that the sum of areas of any three faces is greater than the area of the fourth face -/
def validTetrahedron (t : Tetrahedron) : Prop :=
  ∀ (v : ℝ), v > 0 →
    (3 * v / t.h1 < 3 * v / t.h2 + 3 * v / t.h3 + 3 * v / t.h4) ∧
    (3 * v / t.h2 < 3 * v / t.h1 + 3 * v / t.h3 + 3 * v / t.h4) ∧
    (3 * v / t.h3 < 3 * v / t.h1 + 3 * v / t.h2 + 3 * v / t.h4) ∧
    (3 * v / t.h4 < 3 * v / t.h1 + 3 * v / t.h2 + 3 * v / t.h3)

/-- Theorem stating that no tetrahedron exists with heights 1, 2, 3, and 6 -/
theorem no_tetrahedron_with_heights_1_2_3_6 :
  ¬ ∃ (t : Tetrahedron), t.h1 = 1 ∧ t.h2 = 2 ∧ t.h3 = 3 ∧ t.h4 = 6 ∧ validTetrahedron t :=
sorry

end no_tetrahedron_with_heights_1_2_3_6_l1931_193106


namespace unknown_number_value_l1931_193154

theorem unknown_number_value (y : ℝ) : (12 : ℝ)^3 * y^4 / 432 = 5184 → y = 2 := by
  sorry

end unknown_number_value_l1931_193154


namespace fraction_zero_implies_a_equals_two_l1931_193173

theorem fraction_zero_implies_a_equals_two (a : ℝ) : 
  (a^2 - 4) / (a + 2) = 0 → a = 2 := by
  sorry

end fraction_zero_implies_a_equals_two_l1931_193173


namespace molecular_weight_constant_l1931_193153

-- Define the molecular weight of Aluminum carbonate
def aluminum_carbonate_mw : ℝ := 233.99

-- Define temperature and pressure
def temperature : ℝ := 298
def pressure : ℝ := 1

-- Define compressibility and thermal expansion coefficients
-- (We don't use these in the theorem, but they're mentioned in the problem)
def compressibility : ℝ := sorry
def thermal_expansion : ℝ := sorry

-- Theorem stating that the molecular weight remains constant
theorem molecular_weight_constant (T P : ℝ) :
  aluminum_carbonate_mw = 233.99 := by
  sorry

end molecular_weight_constant_l1931_193153


namespace solve_equation_l1931_193100

theorem solve_equation :
  ∃ y : ℚ, (2 * y + 3 * y = 500 - (4 * y + 5 * y)) ∧ y = 250 / 7 := by
sorry

end solve_equation_l1931_193100


namespace least_positive_integer_divisible_by_some_but_not_all_l1931_193158

/-- A function that checks if a number is divisible by some but not all integers from 1 to 10 -/
def isDivisibleBySomeButNotAll (m : ℕ) : Prop :=
  (∃ k : ℕ, 1 ≤ k ∧ k ≤ 10 ∧ m % k = 0) ∧
  (∃ k : ℕ, 1 ≤ k ∧ k ≤ 10 ∧ m % k ≠ 0)

/-- The main theorem stating that 3 is the least positive integer satisfying the condition -/
theorem least_positive_integer_divisible_by_some_but_not_all :
  (∀ n : ℕ, 0 < n ∧ n < 3 → ¬isDivisibleBySomeButNotAll (n^2 - n)) ∧
  isDivisibleBySomeButNotAll (3^2 - 3) := by
  sorry

end least_positive_integer_divisible_by_some_but_not_all_l1931_193158


namespace probability_of_nine_in_three_elevenths_l1931_193144

def decimal_representation (n d : ℕ) : List ℕ := sorry

def count_digit (l : List ℕ) (digit : ℕ) : ℕ := sorry

def probability_of_digit (n d digit : ℕ) : ℚ :=
  let rep := decimal_representation n d
  (count_digit rep digit : ℚ) / (rep.length : ℚ)

theorem probability_of_nine_in_three_elevenths :
  probability_of_digit 3 11 9 = 0 := by sorry

end probability_of_nine_in_three_elevenths_l1931_193144


namespace square_ratio_sum_l1931_193120

theorem square_ratio_sum (area_ratio : ℚ) (a b c : ℕ) : 
  area_ratio = 300 / 75 →
  (a : ℚ) * Real.sqrt b / c = Real.sqrt area_ratio →
  a + b + c = 4 := by
  sorry

end square_ratio_sum_l1931_193120


namespace perpendicular_vectors_x_value_l1931_193146

/-- Given two vectors a and b in R², prove that if they are perpendicular and have specific coordinates, then the x-coordinate of a is -2. -/
theorem perpendicular_vectors_x_value (a b : ℝ × ℝ) :
  a.1 = x ∧ a.2 = 1 ∧ b = (3, 6) ∧ a.1 * b.1 + a.2 * b.2 = 0 → x = -2 := by
  sorry

end perpendicular_vectors_x_value_l1931_193146


namespace logarithm_comparison_l1931_193135

theorem logarithm_comparison : ∃ (a b c : ℝ),
  a = Real.log 2 / Real.log 3 ∧
  b = Real.log 2 / Real.log 5 ∧
  c = Real.log 3 / Real.log 2 ∧
  c > a ∧ a > b := by
  sorry

end logarithm_comparison_l1931_193135


namespace star_commutative_l1931_193130

variable {M : Type*} [Nonempty M]
variable (star : M → M → M)

axiom left_inverse : ∀ a b : M, star (star a b) b = a
axiom right_inverse : ∀ a b : M, star a (star a b) = b

theorem star_commutative : ∀ a b : M, star a b = star b a := by sorry

end star_commutative_l1931_193130


namespace vector_computation_l1931_193191

theorem vector_computation : 
  (4 : ℝ) • (![2, -9] : Fin 2 → ℝ) - (3 : ℝ) • (![(-1), -6] : Fin 2 → ℝ) = ![11, -18] :=
by sorry

end vector_computation_l1931_193191


namespace shortest_side_of_right_triangle_l1931_193124

/-- Given a right triangle with sides of length 6 and 8, 
    the length of the third side is 2√7 -/
theorem shortest_side_of_right_triangle : ∃ (a b c : ℝ), 
  a = 6 ∧ b = 8 ∧ c = 2 * Real.sqrt 7 ∧ 
  a^2 + c^2 = b^2 ∧ 
  ∀ (x : ℝ), (x^2 + a^2 = b^2 → x ≥ c) :=
by sorry

end shortest_side_of_right_triangle_l1931_193124


namespace triangle_area_is_12_l1931_193127

/-- The area of a triangular region bounded by the x-axis, y-axis, and the line 3x + 2y = 12 -/
def triangleArea : ℝ := 12

/-- The equation of the line bounding the triangular region -/
def lineEquation (x y : ℝ) : Prop := 3 * x + 2 * y = 12

/-- The x-intercept of the line -/
def xIntercept : ℝ := 4

/-- The y-intercept of the line -/
def yIntercept : ℝ := 6

theorem triangle_area_is_12 :
  triangleArea = (1 / 2) * xIntercept * yIntercept :=
sorry

end triangle_area_is_12_l1931_193127


namespace exponential_inequality_l1931_193161

theorem exponential_inequality (a b : ℝ) (h1 : a > b) (h2 : b > 0) :
  (1.5 : ℝ) ^ a > (1.5 : ℝ) ^ b := by
  sorry

end exponential_inequality_l1931_193161


namespace sin_330_degrees_l1931_193162

theorem sin_330_degrees : Real.sin (330 * π / 180) = -1/2 := by
  sorry

end sin_330_degrees_l1931_193162


namespace right_triangles_with_sqrt1001_leg_l1931_193113

theorem right_triangles_with_sqrt1001_leg :
  ∃! (n : ℕ), n > 0 ∧ n = (Finset.filter 
    (fun t : ℕ × ℕ × ℕ => 
      t.1 * t.1 + 1001 = t.2.2 * t.2.2 ∧ 
      t.2.1 * t.2.1 = 1001 ∧
      t.1 > 0 ∧ t.2.1 > 0 ∧ t.2.2 > 0)
    (Finset.product (Finset.range 1000) (Finset.product (Finset.range 1000) (Finset.range 1000)))).card ∧
  n = 4 :=
sorry

end right_triangles_with_sqrt1001_leg_l1931_193113


namespace calculation_proof_l1931_193134

theorem calculation_proof : 5 * 7 * 11 + 21 / 7 - 3 = 385 := by
  sorry

end calculation_proof_l1931_193134


namespace principal_calculation_l1931_193143

/-- Given a principal amount P at simple interest for 3 years, 
    if increasing the interest rate by 1% results in Rs. 72 more interest, 
    then P = 2400. -/
theorem principal_calculation (P : ℝ) (R : ℝ) : 
  (P * (R + 1) * 3) / 100 - (P * R * 3) / 100 = 72 → P = 2400 := by
  sorry

end principal_calculation_l1931_193143


namespace calculate_expression_l1931_193198

theorem calculate_expression : -2^3 / (-2) + (-2)^2 * (-5) = -16 := by
  sorry

end calculate_expression_l1931_193198


namespace smallest_possible_S_l1931_193178

/-- The number of faces on each die -/
def num_faces : ℕ := 8

/-- The target sum we're comparing to -/
def target_sum : ℕ := 3000

/-- The function to calculate the smallest possible value of S -/
def smallest_S (n : ℕ) : ℕ := 9 * n - target_sum

/-- The theorem stating the smallest possible value of S -/
theorem smallest_possible_S :
  ∃ (n : ℕ), 
    (n * num_faces ≥ target_sum) ∧ 
    (∀ m : ℕ, m < n → m * num_faces < target_sum) ∧
    (smallest_S n = 375) := by
  sorry

#check smallest_possible_S

end smallest_possible_S_l1931_193178


namespace points_on_angle_bisector_l1931_193111

/-- Given two points A and B, proves that if they lie on the angle bisector of the first and third quadrants, their coordinates satisfy specific conditions. -/
theorem points_on_angle_bisector 
  (a b : ℝ) 
  (A : ℝ × ℝ) 
  (B : ℝ × ℝ) 
  (h1 : A = (a - 1, 2)) 
  (h2 : B = (-3, b + 1)) 
  (h3 : (a - 1) = 2 ∧ (b + 1) = -3) : 
  a = 3 ∧ b = -4 := by
  sorry

end points_on_angle_bisector_l1931_193111


namespace prob_at_least_one_X_correct_l1931_193110

/-- Represents the probability of selecting at least one person who used model X
    when randomly selecting 2 people from a group of 5, where 3 used model X and 2 used model Y. -/
def prob_at_least_one_X : ℚ := 9 / 10

/-- The total number of people in the experience group -/
def total_people : ℕ := 5

/-- The number of people who used model X bicycles -/
def model_X_users : ℕ := 3

/-- The number of people who used model Y bicycles -/
def model_Y_users : ℕ := 2

/-- The number of ways to select 2 people from the group -/
def total_selections : ℕ := total_people.choose 2

/-- The number of ways to select 2 people who both used model Y -/
def both_Y_selections : ℕ := model_Y_users.choose 2

theorem prob_at_least_one_X_correct :
  prob_at_least_one_X = 1 - (both_Y_selections : ℚ) / total_selections :=
by sorry

end prob_at_least_one_X_correct_l1931_193110


namespace extended_line_segment_vector_representation_l1931_193172

/-- Given a line segment AB extended to point P such that AP:PB = 7:5,
    prove that the position vector of P can be expressed as 
    P = (5/12)A + (7/12)B -/
theorem extended_line_segment_vector_representation 
  (A B P : ℝ × ℝ) -- A, B, and P are points in 2D space
  (h : (dist A P) / (dist P B) = 7 / 5) : -- AP:PB = 7:5
  ∃ (t u : ℝ), t = 5/12 ∧ u = 7/12 ∧ P = t • A + u • B :=
by sorry

end extended_line_segment_vector_representation_l1931_193172


namespace original_length_is_one_meter_l1931_193108

/-- The length of the line after erasing part of it, in centimeters -/
def remaining_length : ℝ := 76

/-- The length that was erased from the line, in centimeters -/
def erased_length : ℝ := 24

/-- The number of centimeters in one meter -/
def cm_per_meter : ℝ := 100

/-- The theorem stating that the original length of the line was 1 meter -/
theorem original_length_is_one_meter : 
  (remaining_length + erased_length) / cm_per_meter = 1 := by sorry

end original_length_is_one_meter_l1931_193108


namespace triangle_side_angle_equivalence_l1931_193147

-- Define a triangle with sides a, b, c and angles A, B, C
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the theorem
theorem triangle_side_angle_equivalence (t : Triangle) :
  (t.a / Real.cos t.A = t.b / Real.cos t.B) ↔ (t.a = t.b) := by
  sorry

end triangle_side_angle_equivalence_l1931_193147


namespace arithmetic_operations_l1931_193164

theorem arithmetic_operations :
  ((-20) - (-14) + (-18) - 13 = -37) ∧
  (((-3/4) + (1/6) - (5/8)) / (-1/24) = 29) ∧
  ((-3^2) + (-3)^2 + 3*2 + |(-4)| = 10) ∧
  (16 / (-2)^3 - (-1/6) * (-4) + (-1)^2024 = -5/3) := by
  sorry

end arithmetic_operations_l1931_193164


namespace range_of_m_l1931_193194

theorem range_of_m : ∃ (a b : ℝ), a = 1 ∧ b = 3 ∧
  ∀ m : ℝ, (∀ x : ℝ, |m - x| < 2 → -1 < x ∧ x < 5) →
  a ≤ m ∧ m ≤ b :=
by sorry

end range_of_m_l1931_193194


namespace problem_solution_l1931_193182

theorem problem_solution : 
  let tan60 := Real.sqrt 3
  |Real.sqrt 2 - Real.sqrt 3| - tan60 + 1 / Real.sqrt 2 = -(Real.sqrt 2 / 2) := by
  sorry

end problem_solution_l1931_193182


namespace stone_price_calculation_l1931_193183

/-- The price per stone when selling a collection of precious stones -/
def price_per_stone (total_amount : ℕ) (num_stones : ℕ) : ℚ :=
  (total_amount : ℚ) / (num_stones : ℚ)

/-- Theorem stating that the price per stone is $1785 when 8 stones are sold for $14280 -/
theorem stone_price_calculation :
  price_per_stone 14280 8 = 1785 := by
  sorry

end stone_price_calculation_l1931_193183


namespace speed_adjustment_l1931_193166

/-- Given a constant distance traveled at 10 km/h in 6 minutes,
    the speed required to travel the same distance in 8 minutes is 7.5 km/h. -/
theorem speed_adjustment (initial_speed initial_time new_time : ℝ) :
  initial_speed = 10 →
  initial_time = 6 / 60 →
  new_time = 8 / 60 →
  let distance := initial_speed * initial_time
  let new_speed := distance / new_time
  new_speed = 7.5 := by
sorry

end speed_adjustment_l1931_193166


namespace min_value_of_sum_l1931_193150

theorem min_value_of_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) :
  ∃ m : ℝ, m = 5 ∧ ∀ x y : ℝ, x > 0 → y > 0 → x + y = 1 → x + 1/x + y + 1/y ≥ m :=
by sorry

end min_value_of_sum_l1931_193150


namespace sixtieth_element_is_2064_l1931_193157

/-- The set of sums of powers of 2 with natural number exponents where the first exponent is less than the second -/
def PowerSumSet : Set ℕ :=
  {n | ∃ (x y : ℕ), x < y ∧ n = 2^x + 2^y}

/-- The 60th element in the ascending order of PowerSumSet -/
def sixtieth_element : ℕ := sorry

/-- Theorem stating that the 60th element of PowerSumSet is 2064 -/
theorem sixtieth_element_is_2064 : sixtieth_element = 2064 := by sorry

end sixtieth_element_is_2064_l1931_193157


namespace circle_radius_l1931_193149

-- Define the circle equation
def circle_equation (x y : ℝ) : Prop :=
  x^2 - 8*x + y^2 - 4*y + 16 = 0

-- State the theorem
theorem circle_radius : ∃ (h k r : ℝ), r = 2 ∧
  ∀ (x y : ℝ), circle_equation x y ↔ (x - h)^2 + (y - k)^2 = r^2 :=
sorry

end circle_radius_l1931_193149


namespace probability_one_unit_apart_l1931_193159

/-- The number of points around the square -/
def num_points : ℕ := 12

/-- The number of pairs of points that are one unit apart -/
def favorable_pairs : ℕ := 12

/-- The total number of ways to choose two points from num_points -/
def total_pairs : ℕ := num_points.choose 2

/-- The probability of choosing two points one unit apart -/
def probability : ℚ := favorable_pairs / total_pairs

theorem probability_one_unit_apart : probability = 2 / 11 := by sorry

end probability_one_unit_apart_l1931_193159


namespace oliver_candy_boxes_l1931_193177

theorem oliver_candy_boxes (initial_boxes final_boxes : ℕ) : 
  initial_boxes = 8 → final_boxes = 6 → initial_boxes + final_boxes = 14 :=
by sorry

end oliver_candy_boxes_l1931_193177


namespace bird_nest_theorem_l1931_193112

/-- Represents a bird's trip information -/
structure BirdTrip where
  trips_to_x : ℕ
  trips_to_y : ℕ
  trips_to_z : ℕ
  distance_to_x : ℕ
  distance_to_y : ℕ
  distance_to_z : ℕ
  time_to_x : ℕ
  time_to_y : ℕ
  time_to_z : ℕ

def bird_a : BirdTrip :=
  { trips_to_x := 15
  , trips_to_y := 0
  , trips_to_z := 10
  , distance_to_x := 300
  , distance_to_y := 0
  , distance_to_z := 400
  , time_to_x := 30
  , time_to_y := 0
  , time_to_z := 40 }

def bird_b : BirdTrip :=
  { trips_to_x := 0
  , trips_to_y := 20
  , trips_to_z := 5
  , distance_to_x := 0
  , distance_to_y := 500
  , distance_to_z := 600
  , time_to_x := 0
  , time_to_y := 60
  , time_to_z := 50 }

def total_distance (bird : BirdTrip) : ℕ :=
  2 * (bird.trips_to_x * bird.distance_to_x +
       bird.trips_to_y * bird.distance_to_y +
       bird.trips_to_z * bird.distance_to_z)

def total_time (bird : BirdTrip) : ℕ :=
  bird.trips_to_x * bird.time_to_x +
  bird.trips_to_y * bird.time_to_y +
  bird.trips_to_z * bird.time_to_z

theorem bird_nest_theorem :
  total_distance bird_a + total_distance bird_b = 43000 ∧
  total_time bird_a + total_time bird_b = 2300 := by
  sorry

end bird_nest_theorem_l1931_193112


namespace square_congruent_one_count_l1931_193148

/-- For n ≥ 2, the number of integers x with 0 ≤ x < n such that x² ≡ 1 (mod n) 
    is equal to 2 times the number of pairs (a, b) such that ab = n and gcd(a, b) = 1 -/
theorem square_congruent_one_count (n : ℕ) (h : n ≥ 2) :
  (Finset.filter (fun x => x^2 % n = 1) (Finset.range n)).card =
  2 * (Finset.filter (fun p : ℕ × ℕ => p.1 * p.2 = n ∧ Nat.gcd p.1 p.2 = 1) 
    (Finset.product (Finset.range (n + 1)) (Finset.range (n + 1)))).card := by
  sorry

end square_congruent_one_count_l1931_193148


namespace special_trapezoid_area_ratios_l1931_193195

/-- A trapezoid with a diagonal forming a 45° angle with the base, 
    and both inscribed and circumscribed circles -/
structure SpecialTrapezoid where
  -- Base lengths
  a : ℝ
  b : ℝ
  -- Height
  h : ℝ
  -- Diagonal forms 45° angle with base
  diagonal_angle : Real.cos (45 * π / 180) = h / (a - b)
  -- Inscribed circle exists
  inscribed_circle_exists : ∃ r : ℝ, r > 0 ∧ r = h / 2
  -- Circumscribed circle exists
  circumscribed_circle_exists : ∃ R : ℝ, R > 0 ∧ R = h / Real.sqrt 2

/-- The main theorem about the area ratios -/
theorem special_trapezoid_area_ratios (t : SpecialTrapezoid) : 
  (t.a + t.b) * t.h / (π * (t.h / 2)^2) = 4 / π ∧
  (t.a + t.b) * t.h / (π * (t.h / Real.sqrt 2)^2) = 2 / π := by
  sorry

end special_trapezoid_area_ratios_l1931_193195


namespace units_digit_is_nine_l1931_193123

/-- The product of digits of a two-digit number -/
def P (n : ℕ) : ℕ := (n / 10) * (n % 10)

/-- The sum of digits of a two-digit number -/
def S (n : ℕ) : ℕ := (n / 10) + (n % 10)

/-- A two-digit number is between 10 and 99, inclusive -/
def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

theorem units_digit_is_nine (N : ℕ) (h1 : is_two_digit N) (h2 : N = P N + S N) :
  N % 10 = 9 := by
  sorry

end units_digit_is_nine_l1931_193123


namespace composition_ratio_l1931_193126

def f (x : ℝ) : ℝ := 3 * x + 5

def g (x : ℝ) : ℝ := 4 * x - 3

theorem composition_ratio :
  (f (g (f (g 3)))) / (g (f (g (f 3)))) = 380 / 653 := by
  sorry

end composition_ratio_l1931_193126


namespace bits_of_88888_base16_l1931_193145

/-- The number of bits required to represent 88888₁₆ in base-2 is 20. -/
theorem bits_of_88888_base16 : ∃ (n : ℕ), n = 20 ∧ 
  (∀ m : ℕ, 2^m > 88888 * 16^4 + 88888 * 16^3 + 88888 * 16^2 + 88888 * 16 + 88888 → m ≥ n) ∧
  2^n > 88888 * 16^4 + 88888 * 16^3 + 88888 * 16^2 + 88888 * 16 + 88888 :=
by sorry

end bits_of_88888_base16_l1931_193145


namespace perfect_square_values_l1931_193197

theorem perfect_square_values (x : ℕ) : 
  (x = 0 ∨ x = 9 ∨ x = 12) → 
  ∃ y : ℕ, 2^6 + 2^10 + 2^x = y^2 :=
by sorry

end perfect_square_values_l1931_193197


namespace total_spots_granger_and_cisco_l1931_193180

/-- The number of spots Rover has -/
def rover_spots : ℕ := 46

/-- The number of spots Cisco has -/
def cisco_spots : ℕ := rover_spots / 2 - 5

/-- The number of spots Granger has -/
def granger_spots : ℕ := 5 * cisco_spots

/-- Theorem stating the total number of spots Granger and Cisco have combined -/
theorem total_spots_granger_and_cisco : 
  granger_spots + cisco_spots = 108 := by sorry

end total_spots_granger_and_cisco_l1931_193180


namespace smallest_number_remainder_l1931_193179

theorem smallest_number_remainder (n : ℕ) : 
  (n = 197) → 
  (∀ m : ℕ, m < n → m % 13 ≠ 2 ∨ m % 16 ≠ 5) → 
  n % 13 = 2 → 
  n % 16 = 5 := by
  sorry

end smallest_number_remainder_l1931_193179


namespace arithmetic_sequence_solve_y_l1931_193118

/-- Given that 1/3, y-2, and 4y are consecutive terms of an arithmetic sequence, prove that y = -13/6 -/
theorem arithmetic_sequence_solve_y (y : ℚ) : 
  (y - 2 - (1/3 : ℚ) = 4*y - (y - 2)) → y = -13/6 := by
sorry

end arithmetic_sequence_solve_y_l1931_193118


namespace max_value_quarter_l1931_193188

def f (a b : ℕ) : ℚ := (a : ℚ) / (10 * b + a) + (b : ℚ) / (10 * a + b)

theorem max_value_quarter (a b : ℕ) (ha : 2 ≤ a ∧ a ≤ 8) (hb : 2 ≤ b ∧ b ≤ 8) :
  f a b ≤ 1/4 := by
  sorry

#eval f 2 2  -- To check the function definition

end max_value_quarter_l1931_193188


namespace matching_socks_probability_theorem_l1931_193125

/-- The number of different pairs of socks -/
def num_pairs : ℕ := 5

/-- The number of days socks are selected -/
def num_days : ℕ := 5

/-- The probability of wearing matching socks on both the third and fifth day -/
def matching_socks_probability : ℚ := 1 / 63

/-- Theorem stating the probability of wearing matching socks on both the third and fifth day -/
theorem matching_socks_probability_theorem :
  let total_socks := 2 * num_pairs
  let favorable_outcomes := num_pairs * (num_pairs - 1) * (Nat.choose (total_socks - 4) 2) * (Nat.choose (total_socks - 6) 2) * (Nat.choose (total_socks - 8) 2)
  let total_outcomes := (Nat.choose total_socks 2) * (Nat.choose (total_socks - 2) 2) * (Nat.choose (total_socks - 4) 2) * (Nat.choose (total_socks - 6) 2) * (Nat.choose (total_socks - 8) 2)
  (favorable_outcomes : ℚ) / total_outcomes = matching_socks_probability :=
by sorry

#check matching_socks_probability_theorem

end matching_socks_probability_theorem_l1931_193125


namespace opposite_of_negative_2023_l1931_193167

theorem opposite_of_negative_2023 :
  ∃ x : ℤ, x + (-2023) = 0 ∧ x = 2023 := by sorry

end opposite_of_negative_2023_l1931_193167


namespace school_pupils_count_l1931_193168

theorem school_pupils_count (girls boys : ℕ) (h1 : girls = 542) (h2 : boys = 387) :
  girls + boys = 929 := by
  sorry

end school_pupils_count_l1931_193168


namespace supplement_bottles_sum_l1931_193109

/-- Given 5 supplement bottles, where 2 bottles have 30 pills each, and after using 70 pills,
    350 pills remain, prove that the sum of pills in the other 3 bottles is 360. -/
theorem supplement_bottles_sum (total_bottles : Nat) (small_bottles : Nat) (pills_per_small_bottle : Nat)
  (pills_used : Nat) (pills_remaining : Nat) :
  total_bottles = 5 →
  small_bottles = 2 →
  pills_per_small_bottle = 30 →
  pills_used = 70 →
  pills_remaining = 350 →
  ∃ (a b c : Nat), a + b + c = 360 :=
by sorry

end supplement_bottles_sum_l1931_193109


namespace max_value_sqrt_product_max_value_achieved_l1931_193117

theorem max_value_sqrt_product (a b c d : Real) 
  (ha : 0 ≤ a ∧ a ≤ 1) 
  (hb : 0 ≤ b ∧ b ≤ 1) 
  (hc : 0 ≤ c ∧ c ≤ 1) 
  (hd : 0 ≤ d ∧ d ≤ 1) : 
  (Real.sqrt (a * b * c * d) + Real.sqrt ((1 - a) * (1 - b) * (1 - c) * (1 - d))) ≤ 1 :=
by sorry

theorem max_value_achieved (a b c d : Real) :
  (a = 0 ∧ b = 0 ∧ c = 0 ∧ d = 0) ∨ (a = 1 ∧ b = 1 ∧ c = 1 ∧ d = 1) →
  Real.sqrt (a * b * c * d) + Real.sqrt ((1 - a) * (1 - b) * (1 - c) * (1 - d)) = 1 :=
by sorry

end max_value_sqrt_product_max_value_achieved_l1931_193117


namespace distance_to_larger_section_specific_case_l1931_193152

/-- Represents a right triangular pyramid with two parallel cross sections -/
structure RightTriangularPyramid where
  /-- Area of the smaller cross section -/
  area_small : ℝ
  /-- Area of the larger cross section -/
  area_large : ℝ
  /-- Distance between the two cross sections -/
  cross_section_distance : ℝ

/-- Calculates the distance from the apex to the larger cross section -/
def distance_to_larger_section (p : RightTriangularPyramid) : ℝ :=
  sorry

/-- Theorem stating the distance to the larger cross section for specific conditions -/
theorem distance_to_larger_section_specific_case :
  let p : RightTriangularPyramid := {
    area_small := 150 * Real.sqrt 3,
    area_large := 300 * Real.sqrt 3,
    cross_section_distance := 10
  }
  distance_to_larger_section p = 10 * Real.sqrt 2 := by
  sorry

end distance_to_larger_section_specific_case_l1931_193152


namespace jamal_has_one_black_marble_l1931_193185

/-- Represents the bag of marbles with different colors. -/
structure MarbleBag where
  yellow : ℕ
  blue : ℕ
  green : ℕ
  black : ℕ

/-- The probability of drawing a black marble. -/
def blackProbability : ℚ := 1 / 28

/-- Jamal's bag of marbles. -/
def jamalsBag : MarbleBag := {
  yellow := 12,
  blue := 10,
  green := 5,
  black := 1  -- We'll prove this is correct
}

/-- The total number of marbles in the bag. -/
def totalMarbles (bag : MarbleBag) : ℕ :=
  bag.yellow + bag.blue + bag.green + bag.black

/-- Theorem stating that Jamal's bag contains exactly one black marble. -/
theorem jamal_has_one_black_marble :
  jamalsBag.black = 1 ∧
  (jamalsBag.black : ℚ) / (totalMarbles jamalsBag : ℚ) = blackProbability :=
by sorry

end jamal_has_one_black_marble_l1931_193185
