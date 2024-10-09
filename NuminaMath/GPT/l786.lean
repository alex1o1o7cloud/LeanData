import Mathlib

namespace fraction_multiplier_l786_78641

theorem fraction_multiplier (x y : ℝ) :
  (3 * x * 3 * y) / (3 * x + 3 * y) = 3 * (x * y) / (x + y) :=
by
  sorry

end fraction_multiplier_l786_78641


namespace find_number_l786_78622

theorem find_number :
  let f_add (a b : ℝ) : ℝ := a * b
  let f_sub (a b : ℝ) : ℝ := a + b
  let f_mul (a b : ℝ) : ℝ := a / b
  let f_div (a b : ℝ) : ℝ := a - b
  (f_div 9 8) * (f_mul 7 some_number) + (f_sub some_number 10) = 13.285714285714286 :=
  let some_number := 5
  sorry

end find_number_l786_78622


namespace sum_y_coordinates_of_other_vertices_l786_78611

theorem sum_y_coordinates_of_other_vertices (x1 y1 x2 y2 : ℤ) 
  (h1 : (x1, y1) = (2, 10)) (h2 : (x2, y2) = (-6, -6)) :
  (∃ y3 y4 : ℤ, (4 : ℤ) = y3 + y4) :=
by
  sorry

end sum_y_coordinates_of_other_vertices_l786_78611


namespace cafe_purchase_l786_78664

theorem cafe_purchase (s d : ℕ) (h_d : d ≥ 2) (h_cost : 5 * s + 125 * d = 4000) :  s + d = 11 :=
    -- Proof steps go here
    sorry

end cafe_purchase_l786_78664


namespace a2b_sub_ab2_eq_neg16sqrt5_l786_78660

noncomputable def a : ℝ := 4 + 2 * Real.sqrt 5
noncomputable def b : ℝ := 4 - 2 * Real.sqrt 5

theorem a2b_sub_ab2_eq_neg16sqrt5 : a^2 * b - a * b^2 = -16 * Real.sqrt 5 :=
by
  sorry

end a2b_sub_ab2_eq_neg16sqrt5_l786_78660


namespace area_of_segment_l786_78649

theorem area_of_segment (R : ℝ) (hR : R > 0) (h_perimeter : 4 * R = 2 * R + 2 * R) :
  (1 - (1 / 2) * Real.sin 2) * R^2 = (fun R => (1 - (1 / 2) * Real.sin 2) * R^2) R :=
by
  sorry

end area_of_segment_l786_78649


namespace hydrogen_atoms_in_compound_l786_78640

theorem hydrogen_atoms_in_compound :
  ∀ (molecular_weight_of_compound atomic_weight_Al atomic_weight_O atomic_weight_H : ℕ)
    (num_Al num_O num_H : ℕ),
    molecular_weight_of_compound = 78 →
    atomic_weight_Al = 27 →
    atomic_weight_O = 16 →
    atomic_weight_H = 1 →
    num_Al = 1 →
    num_O = 3 →
    molecular_weight_of_compound = 
      (num_Al * atomic_weight_Al) + (num_O * atomic_weight_O) + (num_H * atomic_weight_H) →
    num_H = 3 := by
  intros
  sorry

end hydrogen_atoms_in_compound_l786_78640


namespace total_income_l786_78652

variable (I : ℝ)

/-- A person distributed 20% of his income to his 3 children each. -/
def distributed_children (I : ℝ) : ℝ := 3 * 0.20 * I

/-- He deposited 30% of his income to his wife's account. -/
def deposited_wife (I : ℝ) : ℝ := 0.30 * I

/-- The total percentage of his income that was given away is 90%. -/
def total_given_away (I : ℝ) : ℝ := distributed_children I + deposited_wife I 

/-- The remaining income after giving away 90%. -/
def remaining_income (I : ℝ) : ℝ := I - total_given_away I

/-- He donated 5% of the remaining income to the orphan house. -/
def donated_orphan_house (remaining : ℝ) : ℝ := 0.05 * remaining

/-- Finally, he has $40,000 left, which is 95% of the remaining income. -/
def final_amount (remaining : ℝ) : ℝ := 0.95 * remaining

theorem total_income (I : ℝ) (h : final_amount (remaining_income I) = 40000) :
  I = 421052.63 := 
  sorry

end total_income_l786_78652


namespace joseph_investment_after_two_years_l786_78630

noncomputable def initial_investment : ℝ := 1000
noncomputable def monthly_addition : ℝ := 100
noncomputable def yearly_interest_rate : ℝ := 0.10
noncomputable def time_in_years : ℕ := 2

theorem joseph_investment_after_two_years :
  let first_year_total := initial_investment + 12 * monthly_addition
  let first_year_interest := first_year_total * yearly_interest_rate
  let end_of_first_year_total := first_year_total + first_year_interest
  let second_year_total := end_of_first_year_total + 12 * monthly_addition
  let second_year_interest := second_year_total * yearly_interest_rate
  let end_of_second_year_total := second_year_total + second_year_interest
  end_of_second_year_total = 3982 := 
by
  sorry

end joseph_investment_after_two_years_l786_78630


namespace total_profit_or_loss_is_negative_175_l786_78659

theorem total_profit_or_loss_is_negative_175
    (price_A price_B selling_price : ℝ)
    (profit_A loss_B : ℝ)
    (h1 : selling_price = 2100)
    (h2 : profit_A = 0.2)
    (h3 : loss_B = 0.2)
    (hA : price_A * (1 + profit_A) = selling_price)
    (hB : price_B * (1 - loss_B) = selling_price) :
    (selling_price + selling_price) - (price_A + price_B) = -175 := 
by 
  -- The proof is omitted
  sorry

end total_profit_or_loss_is_negative_175_l786_78659


namespace hyperbola_smaller_focus_l786_78612

noncomputable def smaller_focus_coordinates : ℝ × ℝ :=
  let h := 5
  let k := 20
  let a := 3
  let b := 7
  let c := Real.sqrt (a^2 + b^2)
  (h - c, k)

theorem hyperbola_smaller_focus :
  (smaller_focus_coordinates = (Real.sqrt 58 - 2.62, 20)) :=
by
  sorry

end hyperbola_smaller_focus_l786_78612


namespace area_of_trapezium_l786_78617

-- Definitions
def length_parallel_side_1 : ℝ := 4
def length_parallel_side_2 : ℝ := 5
def perpendicular_distance : ℝ := 6

-- Statement
theorem area_of_trapezium :
  (1 / 2) * (length_parallel_side_1 + length_parallel_side_2) * perpendicular_distance = 27 :=
by
  sorry

end area_of_trapezium_l786_78617


namespace rectangle_length_l786_78637

theorem rectangle_length : 
  ∃ l b : ℝ, 
    (l = 2 * b) ∧ 
    (20 < l ∧ l < 50) ∧ 
    (10 < b ∧ b < 30) ∧ 
    ((l - 5) * (b + 5) = l * b + 75) ∧ 
    (l = 40) :=
sorry

end rectangle_length_l786_78637


namespace rectangular_garden_length_l786_78642

theorem rectangular_garden_length (L P B : ℕ) (h1 : P = 600) (h2 : B = 150) (h3 : P = 2 * (L + B)) : L = 150 :=
by
  sorry

end rectangular_garden_length_l786_78642


namespace largest_binomial_coefficient_term_largest_coefficient_term_remainder_mod_7_l786_78671

-- Definitions and conditions
def binomial_expansion (x : ℕ) (n : ℕ) : ℕ := (x + (3 * x^2))^n

-- Problem statements
theorem largest_binomial_coefficient_term :
  ∀ x n,
  (x + 3 * x^2)^n = (x + 3 * x^2)^7 →
  (2^n = 128) →
  ∃ t : ℕ, t = 2835 * x^11 := 
by sorry

theorem largest_coefficient_term :
  ∀ x n,
  (x + 3 * x^2)^n = (x + 3 * x^2)^7 →
  exists t, t = 5103 * x^13 :=
by sorry

theorem remainder_mod_7 :
  ∀ x n,
  x = 3 →
  n = 2016 →
  (x + (3 * x^2))^n % 7 = 1 :=
by sorry

end largest_binomial_coefficient_term_largest_coefficient_term_remainder_mod_7_l786_78671


namespace coefficient_j_l786_78604

theorem coefficient_j (j k : ℝ) (p : Polynomial ℝ) (h : p = Polynomial.C 400 + Polynomial.X * Polynomial.C k + Polynomial.X^2 * Polynomial.C j + Polynomial.X^4) :
  (∃ a d : ℝ, (d ≠ 0) ∧ (0 > (4*a + 6*d)) ∧ (p.eval a = 0) ∧ (p.eval (a + d) = 0) ∧ (p.eval (a + 2*d) = 0) ∧ (p.eval (a + 3*d) = 0)) → 
  j = -40 :=
by
  sorry

end coefficient_j_l786_78604


namespace totalPearsPicked_l786_78632

-- Define the number of pears picked by each individual
def jasonPears : ℕ := 46
def keithPears : ℕ := 47
def mikePears : ℕ := 12

-- State the theorem to prove the total number of pears picked
theorem totalPearsPicked : jasonPears + keithPears + mikePears = 105 := 
by
  -- The proof is omitted
  sorry

end totalPearsPicked_l786_78632


namespace prism_diagonal_and_surface_area_l786_78625

/-- 
  A rectangular prism has dimensions of 12 inches, 16 inches, and 21 inches.
  Prove that the length of the diagonal is 29 inches, 
  and the total surface area of the prism is 1560 square inches.
-/
theorem prism_diagonal_and_surface_area :
  let a := 12
  let b := 16
  let c := 21
  let d := Real.sqrt (a^2 + b^2 + c^2)
  let S := 2 * (a * b + b * c + c * a)
  d = 29 ∧ S = 1560 := by
  let a := 12
  let b := 16
  let c := 21
  let d := Real.sqrt (a^2 + b^2 + c^2)
  let S := 2 * (a * b + b * c + c * a)
  sorry

end prism_diagonal_and_surface_area_l786_78625


namespace eccentricity_of_hyperbola_l786_78690

noncomputable def hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) (c : ℝ)
  (hc : c^2 = a^2 + b^2) : ℝ :=
  (1 + Real.sqrt 5) / 2

theorem eccentricity_of_hyperbola (a b c e : ℝ)
  (ha : a > 0) (hb : b > 0) (h_hyperbola : c^2 = a^2 + b^2)
  (h_eccentricity : e = (1 + Real.sqrt 5) / 2) :
  e = hyperbola_eccentricity a b ha hb c h_hyperbola :=
by
  sorry

end eccentricity_of_hyperbola_l786_78690


namespace joey_read_percentage_l786_78678

theorem joey_read_percentage : 
  ∀ (total_pages read_after_break : ℕ), 
  total_pages = 30 → read_after_break = 9 → 
  ( (total_pages - read_after_break : ℕ) / (total_pages : ℕ) * 100 ) = 70 :=
by
  intros total_pages read_after_break h_total h_after
  sorry

end joey_read_percentage_l786_78678


namespace Tim_income_percentage_less_than_Juan_l786_78675

-- Definitions for the problem
variables (T M J : ℝ)

-- Conditions based on the problem
def condition1 : Prop := M = 1.60 * T
def condition2 : Prop := M = 0.80 * J

-- Goal statement
theorem Tim_income_percentage_less_than_Juan :
  condition1 T M ∧ condition2 M J → T = 0.50 * J :=
by sorry

end Tim_income_percentage_less_than_Juan_l786_78675


namespace race_distance_l786_78662

theorem race_distance (D : ℝ) (h1 : ∀ t : ℝ, t = 30 → D / 30 = D / t)
                      (h2 : ∀ t : ℝ, t = 45 → D / 45 = D / t)
                      (h3 : ∀ d : ℝ, d = 33.333333333333336 → D - (D / 45) * 30 = d) :
  D = 100 :=
sorry

end race_distance_l786_78662


namespace ken_gave_manny_10_pencils_l786_78677

theorem ken_gave_manny_10_pencils (M : ℕ) 
  (ken_pencils : ℕ := 50)
  (ken_kept : ℕ := 20)
  (ken_distributed : ℕ := ken_pencils - ken_kept)
  (nilo_pencils : ℕ := M + 10)
  (distribution_eq : M + nilo_pencils = ken_distributed) : 
  M = 10 :=
by
  sorry

end ken_gave_manny_10_pencils_l786_78677


namespace find_side_length_l786_78698

noncomputable def cos (x : ℝ) := Real.cos x

theorem find_side_length
  (A : ℝ) (c : ℝ) (b : ℝ) (a : ℝ)
  (hA : A = Real.pi / 3)
  (hc : c = Real.sqrt 3)
  (hb : b = 2 * Real.sqrt 3) :
  a = 3 := 
sorry

end find_side_length_l786_78698


namespace fruit_prices_l786_78602

theorem fruit_prices :
  (∃ x y : ℝ, 60 * x + 40 * y = 1520 ∧ 30 * x + 50 * y = 1360 ∧ x = 12 ∧ y = 20) :=
sorry

end fruit_prices_l786_78602


namespace earnings_from_jam_l786_78609

def betty_strawberries : ℕ := 16
def matthew_additional_strawberries : ℕ := 20
def jar_strawberries : ℕ := 7
def jar_price : ℕ := 4

theorem earnings_from_jam :
  let matthew_strawberries := betty_strawberries + matthew_additional_strawberries
  let natalie_strawberries := matthew_strawberries / 2
  let total_strawberries := betty_strawberries + matthew_strawberries + natalie_strawberries
  let total_jars := total_strawberries / jar_strawberries
  let total_money := total_jars * jar_price
  total_money = 40 :=
by
  let matthew_strawberries := betty_strawberries + matthew_additional_strawberries
  let natalie_strawberries := matthew_strawberries / 2
  let total_strawberries := betty_strawberries + matthew_strawberries + natalie_strawberries
  let total_jars := total_strawberries / jar_strawberries
  let total_money := total_jars * jar_price
  show total_money = 40
  sorry

end earnings_from_jam_l786_78609


namespace exists_even_function_b_l786_78658

-- Define the function f(x) = 2x^2 - b*x
def f (b x : ℝ) : ℝ := 2 * x^2 - b * x

-- Define the condition for f being an even function: f(-x) = f(x)
def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = f x

-- The main theorem stating the existence of a b in ℝ such that f is an even function
theorem exists_even_function_b :
  ∃ b : ℝ, is_even_function (f b) :=
by
  sorry

end exists_even_function_b_l786_78658


namespace correct_addition_l786_78618

-- Define the initial conditions and goal
theorem correct_addition (x : ℕ) : (x + 26 = 61) → (x + 62 = 97) :=
by
  intro h
  -- Proof steps would be provided here
  sorry

end correct_addition_l786_78618


namespace no_grasshopper_at_fourth_vertex_l786_78645

-- Definitions based on given conditions
def is_vertex_of_square (x : ℝ) (y : ℝ) : Prop :=
  (x = 0 ∨ x = 1) ∧ (y = 0 ∨ y = 1)

def distance (a b : ℝ × ℝ) : ℝ :=
  (a.1 - b.1) ^ 2 + (a.2 - b.2) ^ 2

def leapfrog_jump (a b : ℝ × ℝ) : ℝ × ℝ :=
  (2 * b.1 - a.1, 2 * b.2 - a.2)

-- Problem statement
theorem no_grasshopper_at_fourth_vertex (a b c : ℝ × ℝ) :
  is_vertex_of_square a.1 a.2 ∧ is_vertex_of_square b.1 b.2 ∧ is_vertex_of_square c.1 c.2 →
  ∃ d : ℝ × ℝ, is_vertex_of_square d.1 d.2 ∧ d ≠ a ∧ d ≠ b ∧ d ≠ c →
  ∀ (n : ℕ) (pos : ℕ → ℝ × ℝ → ℝ × ℝ → ℝ × ℝ), (pos 0 a b = leapfrog_jump a b) ∧
    (pos n a b = leapfrog_jump (pos (n-1) a b) (pos (n-1) b c)) →
    (pos n a b).1 ≠ (d.1) ∨ (pos n a b).2 ≠ (d.2) :=
sorry

end no_grasshopper_at_fourth_vertex_l786_78645


namespace similar_triangles_legs_sum_l786_78667

theorem similar_triangles_legs_sum (a b : ℕ) (h1 : a * b = 18) (h2 : a^2 + b^2 = 25) (bigger_area : ℕ) (smaller_area : ℕ) (hypotenuse : ℕ) 
  (h_similar : bigger_area = 225) 
  (h_smaller_area : smaller_area = 9) 
  (h_hypotenuse : hypotenuse = 5) 
  (h_non_3_4_5 : ¬ (a = 3 ∧ b = 4 ∨ a = 4 ∧ b = 3)) : 
  5 * (a + b) = 45 := 
by sorry

end similar_triangles_legs_sum_l786_78667


namespace meadow_grazing_days_l786_78661

theorem meadow_grazing_days 
    (a b x : ℝ) 
    (h1 : a + 6 * b = 27 * 6 * x)
    (h2 : a + 9 * b = 23 * 9 * x)
    : ∃ y : ℝ, (a + y * b = 21 * y * x) ∧ y = 12 := 
by
    sorry

end meadow_grazing_days_l786_78661


namespace expansion_correct_l786_78634

noncomputable def P (x y : ℝ) : ℝ := 2 * x^25 - 5 * x^8 + 2 * x * y^3 - 9

noncomputable def M (x : ℝ) : ℝ := 3 * x^7

theorem expansion_correct (x y : ℝ) :
  (P x y) * (M x) = 6 * x^32 - 15 * x^15 + 6 * x^8 * y^3 - 27 * x^7 :=
by
  sorry

end expansion_correct_l786_78634


namespace tangent_line_circle_intersection_l786_78682

open Real

noncomputable def is_tangent (θ : ℝ) : Prop :=
  abs (4 * tan θ) / sqrt ((tan θ) ^ 2 + 1) = 2

theorem tangent_line_circle_intersection (θ : ℝ) (hθ : 0 ≤ θ ∧ θ < π) :
  is_tangent θ ↔ θ = π / 6 ∨ θ = 5 * π / 6 :=
sorry

end tangent_line_circle_intersection_l786_78682


namespace original_fraction_l786_78621

theorem original_fraction (x y : ℝ) (hxy : x / y = 5 / 7)
  (hx : 1.20 * x / (0.90 * y) = 20 / 21) : x / y = 5 / 7 :=
by {
  sorry
}

end original_fraction_l786_78621


namespace no_isosceles_triangle_exists_l786_78631

-- Define the grid size
def grid_size : ℕ := 5

-- Define points A and B such that AB is three units horizontally
structure Point where
  x : ℕ
  y : ℕ

-- Define specific points A and B
def A : Point := ⟨2, 2⟩
def B : Point := ⟨5, 2⟩

-- Define a function to check if a triangle is isosceles
def is_isosceles (p1 p2 p3 : Point) : Prop :=
  (p1.x - p2.x)^2 + (p1.y - p2.y)^2 = (p1.x - p3.x)^2 + (p1.y - p3.y)^2 ∨
  (p1.x - p2.x)^2 + (p1.y - p2.y)^2 = (p2.x - p3.x)^2 + (p2.y - p3.y)^2 ∨
  (p1.x - p3.x)^2 + (p1.y - p3.y)^2 = (p2.x - p3.x)^2 + (p2.y - p3.y)^2

-- Prove that there are no points C that make triangle ABC isosceles
theorem no_isosceles_triangle_exists :
  ¬ ∃ C : Point, C.x ≤ grid_size ∧ C.y ≤ grid_size ∧ is_isosceles A B C :=
by
  sorry

end no_isosceles_triangle_exists_l786_78631


namespace greatest_divisor_remainders_l786_78691

theorem greatest_divisor_remainders (d : ℤ) :
  d > 0 → (1657 % d = 10) → (2037 % d = 7) → d = 1 :=
by
  intros hdg h1657 h2037
  sorry

end greatest_divisor_remainders_l786_78691


namespace prop_D_l786_78676

variable (a b : ℝ)

theorem prop_D (a b : ℝ) (h : a > |b|) : a^2 > b^2 :=
  by
    sorry

end prop_D_l786_78676


namespace max_intersection_points_circles_lines_l786_78619

-- Definitions based on the conditions
def num_circles : ℕ := 3
def num_lines : ℕ := 2

-- Function to calculate the number of points of intersection
def max_points_of_intersection (num_circles num_lines : ℕ) : ℕ :=
  (num_circles * (num_circles - 1) / 2) * 2 + 
  num_circles * num_lines * 2 + 
  (num_lines * (num_lines - 1) / 2)

-- The proof statement
theorem max_intersection_points_circles_lines :
  max_points_of_intersection num_circles num_lines = 19 :=
by
  sorry

end max_intersection_points_circles_lines_l786_78619


namespace tan_double_beta_alpha_value_l786_78626

open Real

-- Conditions
def alpha_in_interval (α : ℝ) : Prop := 0 < α ∧ α < π / 2
def beta_in_interval (β : ℝ) : Prop := π / 2 < β ∧ β < π
def cos_beta (β : ℝ) : Prop := cos β = -1 / 3
def sin_alpha_plus_beta (α β : ℝ) : Prop := sin (α + β) = (4 - sqrt 2) / 6

-- Proof problem 1: Prove that tan 2β = 4√2 / 7 given the conditions
theorem tan_double_beta (α β : ℝ) (h1 : alpha_in_interval α) (h2 : beta_in_interval β)
  (h3 : cos_beta β) (h4 : sin_alpha_plus_beta α β) :
  tan (2 * β) = (4 * sqrt 2) / 7 :=
by sorry

-- Proof problem 2: Prove that α = π / 4 given the conditions
theorem alpha_value (α β : ℝ) (h1 : alpha_in_interval α) (h2 : beta_in_interval β)
  (h3 : cos_beta β) (h4 : sin_alpha_plus_beta α β) :
  α = π / 4 :=
by sorry

end tan_double_beta_alpha_value_l786_78626


namespace monotonically_increasing_interval_l786_78657

noncomputable def f (x : ℝ) : ℝ :=
  Real.cos x ^ 2 + Real.sqrt 3 * Real.sin x * Real.cos x

theorem monotonically_increasing_interval :
  ∀ x, 0 < x ∧ x ≤ π / 6 → ∀ y, x ≤ y ∧ y < π / 2 → f x ≤ f y :=
by
  intro x hx y hy
  sorry

end monotonically_increasing_interval_l786_78657


namespace volume_cone_equals_cylinder_minus_surface_area_l786_78693

theorem volume_cone_equals_cylinder_minus_surface_area (r h : ℝ) :
  let V_cyl := π * r^2 * h
  let V_cone := (1 / 3) * π * r^2 * h
  let S_lateral_cyl := 2 * π * r * h
  V_cone = V_cyl - (1 / 3) * S_lateral_cyl * r := by
  let V_cyl := π * r^2 * h
  let V_cone := (1 / 3) * π * r^2 * h
  let S_lateral_cyl := 2 * π * r * h
  sorry

end volume_cone_equals_cylinder_minus_surface_area_l786_78693


namespace gcd_3570_4840_l786_78669

-- Define the numbers
def num1 : Nat := 3570
def num2 : Nat := 4840

-- Define the problem statement
theorem gcd_3570_4840 : Nat.gcd num1 num2 = 10 := by
  sorry

end gcd_3570_4840_l786_78669


namespace unique_real_solution_k_l786_78603

-- Definitions corresponding to problem conditions:
def is_real_solution (a b k : ℤ) : Prop :=
  (a > 0) ∧ (b > 0) ∧ (∃ (x y : ℝ), x * x = a - 1 ∧ y * y = b - 1 ∧ x + y = Real.sqrt (a * b + k))

-- Theorem statement:
theorem unique_real_solution_k (k : ℤ) : (∀ a b : ℤ, is_real_solution a b k → (a = 2 ∧ b = 2)) ↔ k = 0 :=
sorry

end unique_real_solution_k_l786_78603


namespace dogs_not_doing_anything_l786_78644

def total_dogs : ℕ := 500
def dogs_running : ℕ := 18 * total_dogs / 100
def dogs_playing_with_toys : ℕ := (3 * total_dogs) / 20
def dogs_barking : ℕ := 7 * total_dogs / 100
def dogs_digging_holes : ℕ := total_dogs / 10
def dogs_competing : ℕ := 12
def dogs_sleeping : ℕ := (2 * total_dogs) / 25
def dogs_eating_treats : ℕ := total_dogs / 5

def dogs_doing_anything : ℕ := dogs_running + dogs_playing_with_toys + dogs_barking + dogs_digging_holes + dogs_competing + dogs_sleeping + dogs_eating_treats

theorem dogs_not_doing_anything : total_dogs - dogs_doing_anything = 98 :=
by
  -- proof steps would go here
  sorry

end dogs_not_doing_anything_l786_78644


namespace find_f_values_l786_78694

def func_property1 (f : ℕ → ℕ) : Prop := 
  ∀ a b : ℕ, a ≠ b → a * f a + b * f b > a * f b + b * f a

def func_property2 (f : ℕ → ℕ) : Prop := 
  ∀ n : ℕ, f (f n) = 3 * n

theorem find_f_values (f : ℕ → ℕ) (h1 : func_property1 f) (h2 : func_property2 f) : 
  f 1 + f 6 + f 28 = 66 :=
sorry

end find_f_values_l786_78694


namespace exists_An_Bn_l786_78627

theorem exists_An_Bn (n : ℕ) : ∃ (A_n B_n : ℕ), (3 - Real.sqrt 7) ^ n = A_n - B_n * Real.sqrt 7 := by
  sorry

end exists_An_Bn_l786_78627


namespace first_book_length_l786_78673

-- Statement of the problem
theorem first_book_length
  (x : ℕ) -- Number of pages in the first book
  (total_pages : ℕ)
  (days_in_two_weeks : ℕ)
  (pages_per_day : ℕ)
  (second_book_pages : ℕ := 100) :
  pages_per_day = 20 ∧ days_in_two_weeks = 14 ∧ total_pages = 280 ∧ total_pages = pages_per_day * days_in_two_weeks ∧ total_pages = x + second_book_pages → x = 180 :=
by
  sorry

end first_book_length_l786_78673


namespace total_highlighters_l786_78686

def num_pink_highlighters := 9
def num_yellow_highlighters := 8
def num_blue_highlighters := 5

theorem total_highlighters : 
  num_pink_highlighters + num_yellow_highlighters + num_blue_highlighters = 22 :=
by
  sorry

end total_highlighters_l786_78686


namespace sin_330_l786_78643

theorem sin_330 : Real.sin (330 * Real.pi / 180) = -1 / 2 := 
by
  sorry

end sin_330_l786_78643


namespace inequality_holds_for_all_x_l786_78679

variable (a x : ℝ)

theorem inequality_holds_for_all_x (h : a ∈ Set.Ioc (-2 : ℝ) 4): ∀ x : ℝ, (x^2 - a*x + 9 > 0) :=
sorry

end inequality_holds_for_all_x_l786_78679


namespace probability_black_pen_l786_78601

-- Define the total number of pens and the specific counts
def total_pens : ℕ := 5 + 6 + 7
def green_pens : ℕ := 5
def black_pens : ℕ := 6
def red_pens : ℕ := 7

-- Define the probability calculation
def probability (total : ℕ) (count : ℕ) : ℚ := count / total

-- State the theorem
theorem probability_black_pen :
  probability total_pens black_pens = 1 / 3 :=
by sorry

end probability_black_pen_l786_78601


namespace largest_cube_volume_l786_78666

theorem largest_cube_volume (width length height : ℕ) (h₁ : width = 15) (h₂ : length = 12) (h₃ : height = 8) :
  ∃ V, V = 512 := by
  use 8^3
  sorry

end largest_cube_volume_l786_78666


namespace geometric_seq_a7_l786_78638

-- Definitions for the geometric sequence and conditions
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q

variables {a : ℕ → ℝ}
axiom a1 : a 1 = 2
axiom a3 : a 3 = 4
axiom geom_seq : geometric_sequence a

-- Statement to prove
theorem geometric_seq_a7 : a 7 = 16 :=
by
  -- proof will be filled in here
  sorry

end geometric_seq_a7_l786_78638


namespace inequality_proof_l786_78620

variables {a b c : ℝ}

theorem inequality_proof (h_pos : a > 0 ∧ b > 0 ∧ c > 0) 
  (h_min : min (min (a * b) (b * c)) (c * a) ≥ 1) :
  (↑((a^2 + 1) * (b^2 + 1) * (c^2 + 1)) ^ (1 / 3 : ℝ)) ≤ ((a + b + c) / 3) ^ 2 + 1 :=
by
  sorry

end inequality_proof_l786_78620


namespace skittles_total_l786_78650

-- Define the conditions
def skittles_per_friend : ℝ := 40.0
def number_of_friends : ℝ := 5.0

-- Define the target statement using the conditions
theorem skittles_total : (skittles_per_friend * number_of_friends = 200.0) :=
by 
  -- Using sorry to placeholder the proof
  sorry

end skittles_total_l786_78650


namespace Roberto_outfit_count_l786_78653

theorem Roberto_outfit_count :
  let trousers := 5
  let shirts := 6
  let jackets := 3
  let ties := 2
  trousers * shirts * jackets * ties = 180 :=
by
  sorry

end Roberto_outfit_count_l786_78653


namespace every_integer_as_sum_of_squares_l786_78696

theorem every_integer_as_sum_of_squares (n : ℤ) : ∃ x y z : ℕ, 0 < x ∧ 0 < y ∧ 0 < z ∧ n = (x^2 : ℤ) + (y^2 : ℤ) - (z^2 : ℤ) :=
by sorry

end every_integer_as_sum_of_squares_l786_78696


namespace find_a_l786_78629

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x^3 + 4 * x^2 + 3 * x
noncomputable def f_prime (a : ℝ) (x : ℝ) : ℝ := 3 * a * x^2 + 8 * x + 3

theorem find_a (a : ℝ) : f_prime a 1 = 2 → a = -3 := by
  intros h
  -- skipping the proof, as it is not required
  sorry

end find_a_l786_78629


namespace expression_range_l786_78684

open Real -- Open the real number namespace

theorem expression_range (x y : ℝ) (h : (x - 1)^2 + (y - 4)^2 = 1) : 
  0 ≤ (x * y - x) / (x^2 + (y - 1)^2) ∧ (x * y - x) / (x^2 + (y - 1)^2) ≤ 12 / 25 :=
sorry -- Proof to be filled in.

end expression_range_l786_78684


namespace college_girls_count_l786_78606

theorem college_girls_count 
  (B G : ℕ)
  (h1 : B / G = 8 / 5)
  (h2 : B + G = 455) : 
  G = 175 := 
sorry

end college_girls_count_l786_78606


namespace students_speaking_both_languages_l786_78628

theorem students_speaking_both_languages:
  ∀ (total E T N B : ℕ),
    total = 150 →
    E = 55 →
    T = 85 →
    N = 30 →
    (total - N) = 120 →
    (E + T - B) = 120 → B = 20 :=
by
  intros total E T N B h_total h_E h_T h_N h_langs h_equiv
  sorry

end students_speaking_both_languages_l786_78628


namespace prove_range_of_a_l786_78670

noncomputable def f (x a : ℝ) := x^2 + (a + 1) * x + Real.log (abs (a + 2))

def is_increasing (f : ℝ → ℝ) (interval : Set ℝ) :=
 ∀ ⦃x y⦄, x ∈ interval → y ∈ interval → x ≤ y → f x ≤ f y

def g (x a : ℝ) := (a + 1) * x
def is_decreasing (g : ℝ → ℝ) :=
 ∀ ⦃x y⦄, x ≤ y → g y ≤ g x

def proposition_p (a : ℝ) : Prop :=
  is_increasing (f a) (Set.Ici ((a + 1)^2))

def proposition_q (a : ℝ) : Prop :=
  is_decreasing (g a)

theorem prove_range_of_a (a : ℝ) (h : ¬ (proposition_p a ↔ proposition_q a)) :
  a > -3 / 2 :=
sorry

end prove_range_of_a_l786_78670


namespace suff_and_nec_eq_triangle_l786_78648

noncomputable def triangle (A B C: ℝ) (a b c : ℝ) : Prop :=
(B + C = 2 * A) ∧ (b + c = 2 * a)

theorem suff_and_nec_eq_triangle (A B C a b c : ℝ) (h : triangle A B C a b c) :
  A = 60 ∧ B = 60 ∧ C = 60 ∧ a = b ∧ b = c :=
sorry

end suff_and_nec_eq_triangle_l786_78648


namespace smallest_d_for_inverse_g_l786_78610

def g (x : ℝ) := (x - 3)^2 - 8

theorem smallest_d_for_inverse_g : ∃ d : ℝ, (∀ x y : ℝ, x ≠ y → x ≥ d → y ≥ d → g x ≠ g y) ∧ ∀ d' : ℝ, d' < 3 → ∃ x y : ℝ, x ≠ y ∧ x ≥ d' ∧ y ≥ d' ∧ g x = g y :=
by
  sorry

end smallest_d_for_inverse_g_l786_78610


namespace proper_sampling_method_l786_78689

-- Definitions for conditions
def large_bulbs : ℕ := 120
def medium_bulbs : ℕ := 60
def small_bulbs : ℕ := 20
def sample_size : ℕ := 25

-- Definition for the proper sampling method to use
def sampling_method : String := "Stratified sampling"

-- Theorem statement to prove the sampling method
theorem proper_sampling_method :
  ∃ method : String, 
  method = sampling_method ∧
  sampling_method = "Stratified sampling" := by
    sorry

end proper_sampling_method_l786_78689


namespace chameleons_changed_color_l786_78639

-- Define a structure to encapsulate the conditions
structure ChameleonProblem where
  total_chameleons : ℕ
  initial_blue : ℕ -> ℕ
  remaining_blue : ℕ -> ℕ
  red_after_change : ℕ -> ℕ

-- Provide the specific problem instance
def chameleonProblemInstance : ChameleonProblem := {
  total_chameleons := 140,
  initial_blue := λ x => 5 * x,
  remaining_blue := id, -- remaining_blue(x) = x
  red_after_change := λ x => 3 * (140 - 5 * x)
}

-- Define the main theorem
theorem chameleons_changed_color (x : ℕ) :
  (chameleonProblemInstance.initial_blue x - chameleonProblemInstance.remaining_blue x) = 80 :=
by
  sorry

end chameleons_changed_color_l786_78639


namespace darius_scores_less_l786_78688

variable (D M Ma : ℕ)

-- Conditions
def condition1 := D = 10
def condition2 := Ma = D + 3
def condition3 := D + M + Ma = 38

-- Theorem to prove
theorem darius_scores_less (D M Ma : ℕ) (h1 : condition1 D) (h2 : condition2 D Ma) (h3 : condition3 D M Ma) : M - D = 5 :=
by
  sorry

end darius_scores_less_l786_78688


namespace fifteen_pow_mn_eq_PnQm_l786_78695

-- Definitions
def P (m : ℕ) := 3^m
def Q (n : ℕ) := 5^n

-- Theorem statement
theorem fifteen_pow_mn_eq_PnQm (m n : ℕ) : 15^(m * n) = (P m)^n * (Q n)^m :=
by
  -- Placeholder for the proof, which isn't required
  sorry

end fifteen_pow_mn_eq_PnQm_l786_78695


namespace trimino_tilings_greater_l786_78646

noncomputable def trimino_tilings (n : ℕ) : ℕ := sorry
noncomputable def domino_tilings (n : ℕ) : ℕ := sorry

theorem trimino_tilings_greater (n : ℕ) (h : n > 1) : trimino_tilings (3 * n) > domino_tilings (2 * n) :=
sorry

end trimino_tilings_greater_l786_78646


namespace cost_of_balls_max_basketball_count_l786_78624

-- Define the prices of basketball and soccer ball
variables (x y : ℕ)

-- Define the conditions given in the problem
def condition1 : Prop := 2 * x + 3 * y = 310
def condition2 : Prop := 5 * x + 2 * y = 500

-- Proving the cost of each basketball and soccer ball
theorem cost_of_balls (h1 : condition1 x y) (h2 : condition2 x y) : x = 80 ∧ y = 50 :=
sorry

-- Define the total number of balls and the inequality constraint
variable (m : ℕ)
def total_balls_condition : Prop := m + (60 - m) = 60
def cost_constraint : Prop := 80 * m + 50 * (60 - m) ≤ 4000

-- Proving the maximum number of basketballs
theorem max_basketball_count (hc : cost_constraint m) (ht : total_balls_condition m) : m ≤ 33 :=
sorry

end cost_of_balls_max_basketball_count_l786_78624


namespace abs_five_minus_e_l786_78635

noncomputable def e : ℝ := Real.exp 1

theorem abs_five_minus_e : |5 - e| = 5 - e := by
  sorry

end abs_five_minus_e_l786_78635


namespace company_KW_price_percentage_l786_78647

theorem company_KW_price_percentage
  (A B : ℝ)
  (h1 : ∀ P: ℝ, P = 1.9 * A)
  (h2 : ∀ P: ℝ, P = 2 * B) :
  Price = 131.034 / 100 * (A + B) := 
by
  sorry

end company_KW_price_percentage_l786_78647


namespace train_length_l786_78656

theorem train_length (speed_first_train speed_second_train : ℝ) (length_second_train : ℝ) (cross_time : ℝ) (L1 : ℝ) : 
  speed_first_train = 100 ∧ 
  speed_second_train = 60 ∧ 
  length_second_train = 300 ∧ 
  cross_time = 18 → 
  L1 = 420 :=
by
  sorry

end train_length_l786_78656


namespace range_of_u_l786_78681

variable (a b u : ℝ)

theorem range_of_u (h1 : a > 0) (h2 : b > 0) (h3 : (1 / a) + (9 / b) = 1) : u ≤ 16 :=
by
  sorry

end range_of_u_l786_78681


namespace marketing_survey_l786_78600

theorem marketing_survey
  (H_neither : Nat := 80)
  (H_only_A : Nat := 60)
  (H_ratio_Both_to_Only_B : Nat := 3)
  (H_both : Nat := 25) :
  H_neither + H_only_A + (H_ratio_Both_to_Only_B * H_both) + H_both = 240 := 
sorry

end marketing_survey_l786_78600


namespace determine_rectangle_R_area_l786_78608

def side_length_large_square (s : ℕ) : Prop :=
  s = 4

def area_rectangle_R (s : ℕ) (area_R : ℕ) : Prop :=
  s * s - (1 * 4 + 1 * 1) = area_R

theorem determine_rectangle_R_area :
  ∃ (s : ℕ) (area_R : ℕ), side_length_large_square s ∧ area_rectangle_R s area_R :=
by {
  sorry
}

end determine_rectangle_R_area_l786_78608


namespace sum_of_x_intersections_is_zero_l786_78692

-- Definition of an even function
def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

-- Definition for the x-coordinates of the intersection points with x-axis
def intersects_x_axis (f : ℝ → ℝ) (x_coords : List ℝ) : Prop :=
  (∀ x ∈ x_coords, f x = 0) ∧ (x_coords.length = 4)

-- Main theorem
theorem sum_of_x_intersections_is_zero 
  (f : ℝ → ℝ)
  (x_coords : List ℝ)
  (h1 : is_even_function f)
  (h2 : intersects_x_axis f x_coords) : 
  x_coords.sum = 0 :=
sorry

end sum_of_x_intersections_is_zero_l786_78692


namespace percentage_of_first_to_second_l786_78685

theorem percentage_of_first_to_second (X : ℝ) (first second : ℝ) :
  first = 0.06 * X →
  second = 0.30 * X →
  (first / second) * 100 = 20 :=
by
  intros h1 h2
  sorry

end percentage_of_first_to_second_l786_78685


namespace pow_sum_ge_mul_l786_78607

theorem pow_sum_ge_mul (m n : ℕ) : 2^(m + n - 2) ≥ m * n := 
sorry

end pow_sum_ge_mul_l786_78607


namespace value_of_y_l786_78623

theorem value_of_y (x y : ℤ) (h1 : 1.5 * (x : ℝ) = 0.25 * (y : ℝ)) (h2 : x = 24) : y = 144 :=
  sorry

end value_of_y_l786_78623


namespace stormi_mowing_charge_l786_78699

theorem stormi_mowing_charge (cars_washed : ℕ) (car_wash_price : ℕ) (lawns_mowed : ℕ) (bike_cost : ℕ) (money_needed_more : ℕ) 
  (total_from_cars : ℕ := cars_washed * car_wash_price)
  (total_earned : ℕ := bike_cost - money_needed_more)
  (earned_from_lawns : ℕ := total_earned - total_from_cars) :
  cars_washed = 3 → car_wash_price = 10 → lawns_mowed = 2 → bike_cost = 80 → money_needed_more = 24 → earned_from_lawns / lawns_mowed = 13 := 
by
  sorry

end stormi_mowing_charge_l786_78699


namespace yan_distance_ratio_l786_78665

theorem yan_distance_ratio (w x y : ℝ) (h1 : w > 0) (h2 : x > 0) (h3 : y > 0)
(h4 : y / w = x / w + (x + y) / (5 * w)) : x / y = 2 / 3 :=
by
  sorry

end yan_distance_ratio_l786_78665


namespace solve_for_x_l786_78633

theorem solve_for_x (x y : ℚ) (h1 : x - y = 8) (h2 : x + 2 * y = 10) : x = 26 / 3 := by
  sorry

end solve_for_x_l786_78633


namespace vitya_knows_answers_29_attempts_vitya_knows_answers_24_attempts_l786_78614

/-- The test consists of 30 questions, each with two possible answers (one correct and one incorrect). 
    Vitya can proceed in such a way that he can guarantee to know all the correct answers no later than:
    (a) after the 29th attempt (and answer all questions correctly on the 30th attempt)
    (b) after the 24th attempt (and answer all questions correctly on the 25th attempt)
    - Vitya initially does not know any of the answers.
    - The test is always the same.
-/
def vitya_test (k : Nat) : Prop :=
  k = 30 ∧ (∀ (attempts : Fin 30 → Bool), attempts 30 = attempts 29 ∧ attempts 30)

theorem vitya_knows_answers_29_attempts :
  vitya_test 30 :=
by 
  sorry

theorem vitya_knows_answers_24_attempts :
  vitya_test 25 :=
by 
  sorry

end vitya_knows_answers_29_attempts_vitya_knows_answers_24_attempts_l786_78614


namespace sufficient_but_not_necessary_l786_78697

theorem sufficient_but_not_necessary (x : ℝ) : ((0 < x) → (|x-1| - |x| ≤ 1)) ∧ ((|x-1| - |x| ≤ 1) → True) ∧ ¬((|x-1| - |x| ≤ 1) → (0 < x)) := sorry

end sufficient_but_not_necessary_l786_78697


namespace ellipse_foci_coordinates_l786_78616

theorem ellipse_foci_coordinates :
  (∀ x y : ℝ, x^2 / 9 + y^2 / 5 = 1 → (x = 2 ∧ y = 0) ∨ (x = -2 ∧ y = 0)) :=
by
  sorry

end ellipse_foci_coordinates_l786_78616


namespace stadium_capacity_l786_78680

theorem stadium_capacity 
  (C : ℕ)
  (entry_fee : ℕ := 20)
  (three_fourth_full_fees : ℕ := 3 / 4 * C * entry_fee)
  (full_fees : ℕ := C * entry_fee)
  (fee_difference : ℕ := full_fees - three_fourth_full_fees)
  (h : fee_difference = 10000) :
  C = 2000 :=
by
  sorry

end stadium_capacity_l786_78680


namespace trajectory_eq_l786_78683

theorem trajectory_eq :
  ∀ (x y : ℝ), abs x * abs y = 1 → (x * y = 1 ∨ x * y = -1) :=
by
  intro x y h
  sorry

end trajectory_eq_l786_78683


namespace uv_divisible_by_3_l786_78605

theorem uv_divisible_by_3
  {u v : ℤ}
  (h : 9 ∣ (u^2 + u * v + v^2)) :
  3 ∣ u ∧ 3 ∣ v :=
sorry

end uv_divisible_by_3_l786_78605


namespace no_rational_roots_of_odd_l786_78663

theorem no_rational_roots_of_odd (m n : ℤ) (hm : m % 2 = 1) (hn : n % 2 = 1) : ¬ ∃ x : ℚ, x^2 + 2 * m * x + 2 * n = 0 :=
sorry

end no_rational_roots_of_odd_l786_78663


namespace range_of_f_l786_78655

def f (x : ℤ) := x + 1

theorem range_of_f : 
  (∀ x ∈ ({-1, 0, 1, 2} : Set ℤ), f x ∈ ({0, 1, 2, 3} : Set ℤ)) ∧ 
  (∀ y ∈ ({0, 1, 2, 3} : Set ℤ), ∃ x ∈ ({-1, 0, 1, 2} : Set ℤ), f x = y) := 
by 
  sorry

end range_of_f_l786_78655


namespace power_of_7_mod_10_l786_78674

theorem power_of_7_mod_10 (k : ℕ) (h : 7^4 ≡ 1 [MOD 10]) : 7^150 ≡ 9 [MOD 10] :=
sorry

end power_of_7_mod_10_l786_78674


namespace solution_set_of_inequality_l786_78636

noncomputable def f (x : ℝ) : ℝ :=
if x ≥ 0 then x^2 - 4 * x else (-(x^2 - 4 * x))

theorem solution_set_of_inequality :
  {x : ℝ | f (x - 2) < 5} = {x : ℝ | -3 < x ∧ x < 7} := by
  sorry

end solution_set_of_inequality_l786_78636


namespace find_locus_of_P_l786_78672

theorem find_locus_of_P:
  ∃ x y: ℝ, (x - 1)^2 + y^2 = 9 ∧ y ≠ 0 ∧
          ((x + 2)^2 + y^2 + (x - 4)^2 + y^2 = 36) :=
sorry

end find_locus_of_P_l786_78672


namespace min_value_of_expression_l786_78613

noncomputable def problem_statement : Prop :=
  ∃ (x y z : ℝ), x > 0 ∧ y > 0 ∧ z > 0 ∧ ((1/x) + (1/y) + (1/z) = 9) ∧ (x^2 * y^3 * z^2 = 1/2268)

theorem min_value_of_expression :
  problem_statement := 
sorry

end min_value_of_expression_l786_78613


namespace theater_price_balcony_l786_78651

theorem theater_price_balcony 
  (price_orchestra : ℕ) (total_tickets : ℕ) (total_revenue : ℕ) (extra_balcony_tickets : ℕ) (price_balcony : ℕ) 
  (h1 : price_orchestra = 12) 
  (h2 : total_tickets = 380) 
  (h3 : total_revenue = 3320) 
  (h4 : extra_balcony_tickets = 240) 
  (h5 : ∃ (O : ℕ), O + (O + extra_balcony_tickets) = total_tickets ∧ (price_orchestra * O) + (price_balcony * (O + extra_balcony_tickets)) = total_revenue) : 
  price_balcony = 8 := 
by
  sorry

end theater_price_balcony_l786_78651


namespace intervals_of_increase_l786_78668

def f (x : ℝ) : ℝ := 2*x^3 - 6*x^2 + 7

theorem intervals_of_increase : 
  ∀ x : ℝ, (x < 0 ∨ x > 2) → (6*x^2 - 12*x > 0) :=
by
  -- Placeholder for proof
  sorry

end intervals_of_increase_l786_78668


namespace chandler_weeks_to_save_l786_78687

theorem chandler_weeks_to_save :
  let birthday_money := 50 + 35 + 15 + 20
  let weekly_earnings := 18
  let bike_cost := 650
  ∃ x : ℕ, (birthday_money + x * weekly_earnings) ≥ bike_cost ∧ (birthday_money + (x - 1) * weekly_earnings) < bike_cost := 
by
  sorry

end chandler_weeks_to_save_l786_78687


namespace sufficient_but_not_necessary_l786_78615

theorem sufficient_but_not_necessary (x : ℝ) : (x < -1) → (x < -1 ∨ x > 1) ∧ ¬(∀ y : ℝ, (x < -1 ∨ y > 1) → (y < -1)) :=
by
  -- This means we would prove that if x < -1, then x < -1 ∨ x > 1 holds (sufficient),
  -- and show that there is a case (x > 1) where x < -1 is not necessary for x < -1 ∨ x > 1. 
  sorry

end sufficient_but_not_necessary_l786_78615


namespace smallest_number_multiple_of_45_and_exceeds_100_by_multiple_of_7_l786_78654

theorem smallest_number_multiple_of_45_and_exceeds_100_by_multiple_of_7 :
  ∃ n : ℕ, n % 45 = 0 ∧ (n - 100) % 7 = 0 ∧ n = 135 :=
sorry

end smallest_number_multiple_of_45_and_exceeds_100_by_multiple_of_7_l786_78654
