import Mathlib

namespace parabola_coefficients_l844_84443

theorem parabola_coefficients 
  (a b c : ℝ) 
  (h_vertex : ∀ x : ℝ, (2 - (-2))^2 * a + (-2 * 2 * a + b) * (2 - (-2)) + (c - 5) = 0)
  (h_point : 9 = a * (2:ℝ)^2 + b * (2:ℝ) + c) : 
  a = 1 / 4 ∧ b = 1 ∧ c = 6 := 
by 
  sorry

end parabola_coefficients_l844_84443


namespace inequality_for_pos_reals_equality_condition_l844_84422

open Real

theorem inequality_for_pos_reals (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  a / c + c / b ≥ 4 * a / (a + b) :=
by
  -- Theorem Statement Proof Skeleton
  sorry

theorem equality_condition (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a / c + c / b = 4 * a / (a + b)) ↔ (a = b ∧ b = c) :=
by
  -- Theorem Statement Proof Skeleton
  sorry

end inequality_for_pos_reals_equality_condition_l844_84422


namespace cosine_identity_l844_84423

theorem cosine_identity (α : ℝ) (h : Real.sin (Real.pi / 6 - α) = 1 / 4) : 
  Real.cos (2 * α - Real.pi / 3) = 7 / 8 := by 
  sorry

end cosine_identity_l844_84423


namespace average_age_of_dance_group_l844_84410

theorem average_age_of_dance_group
  (avg_age_children : ℕ)
  (avg_age_adults : ℕ)
  (num_children : ℕ)
  (num_adults : ℕ)
  (total_num_members : ℕ)
  (total_sum_ages : ℕ)
  (average_age : ℚ)
  (h_children : avg_age_children = 12)
  (h_adults : avg_age_adults = 40)
  (h_num_children : num_children = 8)
  (h_num_adults : num_adults = 12)
  (h_total_members : total_num_members = 20)
  (h_total_ages : total_sum_ages = 576)
  (h_average_age : average_age = 28.8) :
  average_age = (total_sum_ages : ℚ) / total_num_members :=
by
  sorry

end average_age_of_dance_group_l844_84410


namespace fewer_bees_than_flowers_l844_84425

theorem fewer_bees_than_flowers :
  (5 - 3 = 2) :=
by
  sorry

end fewer_bees_than_flowers_l844_84425


namespace minimum_value_expression_l844_84424

theorem minimum_value_expression {x1 x2 x3 x4 : ℝ} (hx1 : 0 < x1) (hx2 : 0 < x2) (hx3 : 0 < x3) (hx4 : 0 < x4) (h_sum : x1 + x2 + x3 + x4 = Real.pi) :
  (2 * (Real.sin x1)^2 + 1 / (Real.sin x1)^2) * (2 * (Real.sin x2)^2 + 1 / (Real.sin x2)^2) * (2 * (Real.sin x3)^2 + 1 / (Real.sin x3)^2) * (2 * (Real.sin x4)^2 + 1 / (Real.sin x4)^2) ≥ 81 :=
by {
  sorry
}

end minimum_value_expression_l844_84424


namespace relationship_cannot_be_determined_l844_84427

noncomputable def point_on_parabola (a b c x y : ℝ) : Prop :=
  y = a * x^2 + b * x + c

theorem relationship_cannot_be_determined
  (a b c x1 y1 x2 y2 x3 y3 x4 y4 : ℝ) (h1 : a ≠ 0) 
  (h2 : point_on_parabola a b c x1 y1) 
  (h3 : point_on_parabola a b c x2 y2) 
  (h4 : point_on_parabola a b c x3 y3) 
  (h5 : point_on_parabola a b c x4 y4)
  (h6 : x1 + x4 - x2 + x3 = 0) : 
  ¬( ∃ m n : ℝ, ((y4 - y1) / (x4 - x1) = m ∧ (y2 - y3) / (x2 - x3) = m) ∨ 
                     ((y4 - y1) / (x4 - x1) * (y2 - y3) / (x2 - x3) = -1) ∨ 
                     ((y4 - y1) / (x4 - x1) ≠ m ∧ (y2 - y3) / (x2 - x3) ≠ m ∧ 
                      (y4 - y1) / (x4 - x1) * (y2 - y3) / (x2 - x3) ≠ -1)) :=
sorry

end relationship_cannot_be_determined_l844_84427


namespace unoccupied_volume_l844_84476

/--
Given:
1. Three congruent cones, each with a radius of 8 cm and a height of 8 cm.
2. The cones are enclosed within a cylinder such that the bases of two cones are at each base of the cylinder, and one cone is inverted in the middle touching the other two cones at their vertices.
3. The height of the cylinder is 16 cm.

Prove:
The volume of the cylinder not occupied by the cones is 512π cubic cm.
-/
theorem unoccupied_volume 
  (r h : ℝ) 
  (hr : r = 8) 
  (hh_cone : h = 8) 
  (hh_cyl : h_cyl = 16) 
  : (π * r^2 * h_cyl) - (3 * (1/3 * π * r^2 * h)) = 512 * π := 
by 
  sorry

end unoccupied_volume_l844_84476


namespace average_student_age_before_leaving_l844_84459

theorem average_student_age_before_leaving
  (A : ℕ)
  (student_count : ℕ := 30)
  (leaving_student_age : ℕ := 11)
  (teacher_age : ℕ := 41)
  (new_avg_age : ℕ := 11)
  (new_total_students : ℕ := 30)
  (initial_total_age : ℕ := 30 * A)
  (remaining_students : ℕ := 29)
  (total_age_after_leaving : ℕ := initial_total_age - leaving_student_age)
  (total_age_including_teacher : ℕ := total_age_after_leaving + teacher_age) :
  total_age_including_teacher / new_total_students = new_avg_age → A = 10 := 
  by
    intros h
    sorry

end average_student_age_before_leaving_l844_84459


namespace red_notebooks_count_l844_84458

variable (R B : ℕ)

-- Conditions
def cost_condition : Prop := 4 * R + 4 + 3 * B = 37
def count_condition : Prop := R + 2 + B = 12
def blue_notebooks_expr : Prop := B = 10 - R

-- Prove the number of red notebooks
theorem red_notebooks_count : cost_condition R B ∧ count_condition R B ∧ blue_notebooks_expr R B → R = 3 := by
  sorry

end red_notebooks_count_l844_84458


namespace ratio_XY_7_l844_84486

variable (Z : ℕ)
variable (population_Z : ℕ := Z)
variable (population_Y : ℕ := 2 * Z)
variable (population_X : ℕ := 14 * Z)

theorem ratio_XY_7 :
  population_X / population_Y = 7 := by
  sorry

end ratio_XY_7_l844_84486


namespace rotation_problem_l844_84493

-- Define the coordinates of the points in the plane
structure Point where
  x : ℝ
  y : ℝ

-- Define the triangles with given vertices
def P : Point := {x := 0, y := 0}
def Q : Point := {x := 0, y := 13}
def R : Point := {x := 17, y := 0}

def P' : Point := {x := 34, y := 26}
def Q' : Point := {x := 46, y := 26}
def R' : Point := {x := 34, y := 0}

-- Rotation parameters
variables (n : ℝ) (x y : ℝ) (h₀ : 0 < n) (h₁ : n < 180)

-- The mathematical proof problem
theorem rotation_problem :
  n + x + y = 180 := by
  sorry

end rotation_problem_l844_84493


namespace recurring_decimal_mul_seven_l844_84456

-- Declare the repeating decimal as a definition
def recurring_decimal_0_3 : ℚ := 1 / 3

-- Theorem stating that the product of 0.333... and 7 is 7/3
theorem recurring_decimal_mul_seven : recurring_decimal_0_3 * 7 = 7 / 3 :=
by
  -- Insert proof here
  sorry

end recurring_decimal_mul_seven_l844_84456


namespace total_books_l844_84478

def books_per_shelf : ℕ := 78
def number_of_shelves : ℕ := 15

theorem total_books : books_per_shelf * number_of_shelves = 1170 := 
by
  sorry

end total_books_l844_84478


namespace area_of_pentagon_AEDCB_l844_84421

structure Rectangle (A B C D : Type) :=
  (AB BC AD CD : ℕ)

def is_perpendicular (A E E' D : Type) : Prop := sorry

def area_of_triangle (AE DE : ℕ) : ℕ :=
  (AE * DE) / 2

def area_of_rectangle (length width : ℕ) : ℕ :=
  length * width

def area_of_pentagon (area_rect area_triangle : ℕ) : ℕ :=
  area_rect - area_triangle

theorem area_of_pentagon_AEDCB
  (A B C D E : Type)
  (h_rectangle : Rectangle A B C D)
  (h_perpendicular : is_perpendicular A E E D)
  (AE DE : ℕ)
  (h_ae : AE = 9)
  (h_de : DE = 12)
  : area_of_pentagon (area_of_rectangle 15 12) (area_of_triangle AE DE) = 126 := 
  sorry

end area_of_pentagon_AEDCB_l844_84421


namespace find_k_l844_84400

-- Given: The polynomial x^2 - 3k * x * y - 3y^2 + 6 * x * y - 8
-- We want to prove the value of k such that the polynomial does not contain the term "xy".

theorem find_k (k : ℝ) : 
  (∀ x y : ℝ, (x^2 - 3 * k * x * y - 3 * y^2 + 6 * x * y - 8) = x^2 - 3 * y^2 - 8) → 
  k = 2 := 
by
  intro h
  have h_coeff := h 1 1
  -- We should observe that the polynomial should not contain the xy term
  sorry

end find_k_l844_84400


namespace true_product_of_two_digit_number_l844_84469

theorem true_product_of_two_digit_number (a b : ℕ) (h1 : b = 2 * a) (h2 : 136 * (10 * b + a) = 136 * (10 * a + b) + 1224) : 136 * (10 * a + b) = 1632 := 
by sorry

end true_product_of_two_digit_number_l844_84469


namespace simplify_polynomial_l844_84470

variable {R : Type} [CommRing R] (s : R)

theorem simplify_polynomial :
  (2 * s^2 + 5 * s - 3) - (2 * s^2 + 9 * s - 4) = -4 * s + 1 :=
by
  sorry

end simplify_polynomial_l844_84470


namespace greatest_percentage_increase_l844_84407

def pop1970_F := 30000
def pop1980_F := 45000
def pop1970_G := 60000
def pop1980_G := 75000
def pop1970_H := 40000
def pop1970_I := 20000
def pop1980_combined_H := 70000
def pop1970_J := 90000
def pop1980_J := 120000

def percentage_increase (pop1970 pop1980 : ℕ) : ℚ :=
  ((pop1980 - pop1970 : ℚ) / pop1970) * 100

theorem greatest_percentage_increase :
  ∀ (city : ℕ), (city = pop1970_F -> percentage_increase pop1970_F pop1980_F >= percentage_increase pop1970_G pop1980_G) ∧
                (city = pop1970_F -> percentage_increase pop1970_F pop1980_F >= percentage_increase (pop1970_H + pop1970_I) pop1980_combined_H) ∧
                (city = pop1970_F -> percentage_increase pop1970_F pop1980_F >= percentage_increase pop1970_J pop1980_J) := by 
  sorry

end greatest_percentage_increase_l844_84407


namespace Alan_age_is_29_l844_84479

/-- Alan and Chris ages problem -/
theorem Alan_age_is_29
    (A C : ℕ)
    (h1 : A + C = 52)
    (h2 : C = A / 3 + 2 * (A - C)) :
    A = 29 :=
by
  sorry

end Alan_age_is_29_l844_84479


namespace cost_price_radio_l844_84481

theorem cost_price_radio (SP : ℝ) (loss_percentage : ℝ) (C : ℝ) 
  (h1 : SP = 1305) 
  (h2 : loss_percentage = 0.13) 
  (h3 : SP = C * (1 - loss_percentage)) :
  C = 1500 := 
by 
  sorry

end cost_price_radio_l844_84481


namespace items_left_in_store_l844_84434

def restocked : ℕ := 4458
def sold : ℕ := 1561
def storeroom : ℕ := 575

theorem items_left_in_store : restocked - sold + storeroom = 3472 := by
  sorry

end items_left_in_store_l844_84434


namespace hank_donated_percentage_l844_84436

variable (A_c D_c A_b D_b A_l D_t D_l p : ℝ) (h1 : A_c = 100) (h2 : D_c = 0.90 * A_c)
variable (h3 : A_b = 80) (h4 : D_b = 0.75 * A_b) (h5 : A_l = 50) (h6 : D_t = 200)

theorem hank_donated_percentage :
  D_l = D_t - (D_c + D_b) → 
  p = (D_l / A_l) * 100 → 
  p = 100 :=
by
  sorry

end hank_donated_percentage_l844_84436


namespace solve_for_x_l844_84438

theorem solve_for_x (x : ℝ) (h : (x + 6) / (x - 3) = 4) : x = 6 :=
by
  sorry

end solve_for_x_l844_84438


namespace jerrie_minutes_l844_84440

-- Define the conditions
def barney_situps_per_minute := 45
def carrie_situps_per_minute := 2 * barney_situps_per_minute
def jerrie_situps_per_minute := carrie_situps_per_minute + 5
def barney_total_situps := 1 * barney_situps_per_minute
def carrie_total_situps := 2 * carrie_situps_per_minute
def combined_total_situps := 510

-- Define the question and required proof
theorem jerrie_minutes :
  ∃ J : ℕ, barney_total_situps + carrie_total_situps + J * jerrie_situps_per_minute = combined_total_situps ∧ J = 3 :=
  by
  sorry

end jerrie_minutes_l844_84440


namespace totalSandwiches_l844_84437

def numberOfPeople : ℝ := 219.0
def sandwichesPerPerson : ℝ := 3.0

theorem totalSandwiches : numberOfPeople * sandwichesPerPerson = 657.0 := by
  -- Proof goes here
  sorry

end totalSandwiches_l844_84437


namespace science_books_have_9_copies_l844_84420

theorem science_books_have_9_copies :
  ∃ (A B C D : ℕ), A + B + C + D = 35 ∧ A + B = 17 ∧ B + C = 16 ∧ A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D ∧ B = 9 :=
by
  sorry

end science_books_have_9_copies_l844_84420


namespace xy_product_range_l844_84415

theorem xy_product_range (x y : ℝ) (h : x^2 * y^2 + x^2 - 10 * x * y - 8 * x + 16 = 0) :
  0 ≤ x * y ∧ x * y ≤ 10 := 
sorry

end xy_product_range_l844_84415


namespace boris_neighbors_l844_84403

-- Define the people
inductive Person
| Arkady | Boris | Vera | Galya | Danya | Egor
deriving DecidableEq

open Person

-- Define the circular arrangement
def next_to (p1 p2 : Person) : Prop :=
p1 = Vera ∧ p2 = Danya ∨
p1 = Danya ∧ p2 = Egor ∨
p1 = Egor ∧ p2 = Vera ∨
p1 = Boris ∧ p2 = Galya ∨
p1 = Galya ∧ p2 = Boris ∨
p1 = Boris ∧ p2 = Arkady ∨
p1 = Arkady ∧ p2 = Boris

axiom danya_next_to_vera : next_to Danya Vera
axiom galya_opposite_egor : ∀ p, (p = Galya) = (p ≠ Egor) ∧ (next_to Egor Danya)  
axiom egor_next_to_danya : next_to Egor Danya
axiom arkady_not_next_to_galya : ¬ next_to Arkady Galya

theorem boris_neighbors : next_to Boris Arkady ∧ next_to Boris Galya :=
by {
  sorry
}

end boris_neighbors_l844_84403


namespace solve_equation_l844_84460

theorem solve_equation (a b n : ℕ) (p : ℕ) [hp : Fact (Nat.Prime p)] :
  (a > 0) → (b > 0) → (n > 0) → (a ^ 2013 + b ^ 2013 = p ^ n) ↔ 
  ∃ k : ℕ, a = 2 ^ k ∧ b = 2 ^ k ∧ p = 2 ∧ n = 2013 * k + 1 :=
by
  sorry

end solve_equation_l844_84460


namespace max_area_and_length_l844_84430

def material_cost (x y : ℝ) : ℝ :=
  900 * x + 400 * y + 200 * x * y

def area (x y : ℝ) : ℝ := x * y

theorem max_area_and_length (x y : ℝ) (h₁ : material_cost x y ≤ 32000) :
  ∃ (S : ℝ) (x : ℝ), S = 100 ∧ x = 20 / 3 :=
sorry

end max_area_and_length_l844_84430


namespace circumscribed_circle_radius_l844_84432

noncomputable def radius_of_circumscribed_circle (a b : ℝ) : ℝ :=
  (Real.sqrt (a^2 + b^2)) / 2

theorem circumscribed_circle_radius (a r l b R : ℝ)
  (h1 : r = 1)
  (h2 : a = 2 * Real.sqrt 3)
  (h3 : b = 3)
  (h4 : l = a)
  (h5 : R = radius_of_circumscribed_circle l b) :
  R = Real.sqrt 21 / 2 :=
by
  sorry

end circumscribed_circle_radius_l844_84432


namespace measure_of_angle_l844_84426

theorem measure_of_angle (x : ℝ) (h1 : 90 - x = 3 * x - 10) : x = 25 :=
by
  sorry

end measure_of_angle_l844_84426


namespace find_5_minus_c_l844_84471

theorem find_5_minus_c (c d : ℤ) (h₁ : 5 + c = 6 - d) (h₂ : 3 + d = 8 + c) : 5 - c = 7 := by
  sorry

end find_5_minus_c_l844_84471


namespace slope_of_line_inclination_l844_84487

theorem slope_of_line_inclination (α : ℝ) (h1 : 0 ≤ α) (h2 : α < 180) 
  (h3 : Real.tan (α * Real.pi / 180) = Real.sqrt 3 / 3) : α = 30 :=
by
  sorry

end slope_of_line_inclination_l844_84487


namespace mean_combined_set_l844_84477

noncomputable def mean (s : Finset ℚ) : ℚ :=
  (s.sum id) / s.card

theorem mean_combined_set :
  ∀ (s1 s2 : Finset ℚ),
  s1.card = 7 →
  s2.card = 8 →
  mean s1 = 15 →
  mean s2 = 18 →
  mean (s1 ∪ s2) = 249 / 15 :=
by
  sorry

end mean_combined_set_l844_84477


namespace number_of_attempted_problems_l844_84406

-- Lean statement to define the problem setup
def student_assignment_problem (x y : ℕ) : Prop :=
  8 * x - 5 * y = 13 ∧ x + y ≤ 20

-- The Lean statement asserting the solution to the problem
theorem number_of_attempted_problems : ∃ x y : ℕ, student_assignment_problem x y ∧ x + y = 13 := 
by
  sorry

end number_of_attempted_problems_l844_84406


namespace application_methods_l844_84409

variables (students : Fin 6) (colleges : Fin 3)

def total_applications_without_restriction : ℕ := 3^6
def applications_missing_one_college : ℕ := 2^6
def overcounted_applications_missing_two_college : ℕ := 1

theorem application_methods (h1 : total_applications_without_restriction = 729)
    (h2 : applications_missing_one_college = 64)
    (h3 : overcounted_applications_missing_two_college = 1) :
    ∀ (students : Fin 6), ∀ (colleges : Fin 3),
      (total_applications_without_restriction - 3 * applications_missing_one_college + 3 * overcounted_applications_missing_two_college = 540) :=
by {
  sorry
}

end application_methods_l844_84409


namespace f_at_count_l844_84495

def f (a b c : ℕ) : ℕ := (a * b * c) / (Nat.gcd (Nat.gcd a b) c * Nat.lcm (Nat.lcm a b) c)

def is_f_at (n : ℕ) : Prop :=
  ∃ (x y z : ℕ), x ≠ y ∧ y ≠ z ∧ z ≠ x ∧ x ≤ 60 ∧ y ≤ 60 ∧ z ≤ 60 ∧ f x y z = n

theorem f_at_count : ∃ (n : ℕ), n = 70 ∧ ∀ k, is_f_at k → k ≤ 70 := 
sorry

end f_at_count_l844_84495


namespace ab2c_value_l844_84480

theorem ab2c_value (a b c : ℚ) (h₁ : |a + 1| + (b - 2)^2 = 0) (h₂ : |c| = 3) :
  a + b + 2 * c = 7 ∨ a + b + 2 * c = -5 := sorry

end ab2c_value_l844_84480


namespace empty_set_l844_84492

def setA := {x : ℝ | x^2 - 4 = 0}
def setB := {x : ℝ | x > 9 ∨ x < 3}
def setC := {p : ℝ × ℝ | p.1^2 + p.2^2 = 0}
def setD := {x : ℝ | x > 9 ∧ x < 3}

theorem empty_set : setD = ∅ := 
  sorry

end empty_set_l844_84492


namespace inverse_value_exists_l844_84451

noncomputable def f (a x : ℝ) := a^x - 1

theorem inverse_value_exists (a : ℝ) (h : f a 1 = 1) : (f a)⁻¹ 3 = 2 :=
by
  sorry

end inverse_value_exists_l844_84451


namespace negation_proof_l844_84412

theorem negation_proof (a b : ℝ) : 
  (¬ (a > b → 2 * a > 2 * b - 1)) = (a ≤ b → 2 * a ≤ 2 * b - 1) :=
by
  sorry

end negation_proof_l844_84412


namespace prove_p_false_and_q_true_l844_84467

variables (p q : Prop)

theorem prove_p_false_and_q_true (h1 : p ∨ q) (h2 : ¬p) : ¬p ∧ q :=
by {
  -- proof placeholder
  sorry
}

end prove_p_false_and_q_true_l844_84467


namespace monthly_price_reduction_rate_l844_84431

-- Let's define the given conditions
def initial_price_March : ℝ := 23000
def price_in_May : ℝ := 16000

-- Define the monthly average price reduction rate
variable (x : ℝ)

-- Define the statement to be proven
theorem monthly_price_reduction_rate :
  23 * (1 - x) ^ 2 = 16 :=
sorry

end monthly_price_reduction_rate_l844_84431


namespace solution_l844_84462

variable (x y z : ℝ)

noncomputable def problem := 
  ∀ (x y z : ℝ), x > 0 → y > 0 → z > 0 →
  x^2 + x * y + y^2 = 48 →
  y^2 + y * z + z^2 = 25 →
  z^2 + z * x + x^2 = 73 →
  x * y + y * z + z * x = 40

theorem solution : problem := by
  intros
  sorry

end solution_l844_84462


namespace central_angle_of_regular_hexagon_l844_84419

theorem central_angle_of_regular_hexagon:
  ∀ (α : ℝ), 
  (∃ n : ℕ, n = 6 ∧ n * α = 360) →
  α = 60 :=
by
  sorry

end central_angle_of_regular_hexagon_l844_84419


namespace rationalize_simplify_l844_84449

theorem rationalize_simplify :
  3 / (Real.sqrt 75 + Real.sqrt 3) = Real.sqrt 3 / 6 :=
by
  sorry

end rationalize_simplify_l844_84449


namespace incorrect_connection_probability_l844_84450

noncomputable def probability_of_incorrect_connection (p : ℝ) : ℝ :=
  let r2 := 1 / 9
  let r3 := (8 / 9) * (1 / 9)
  (3 * p^2 * (1 - p) * r2) + (1 * p^3 * r3)

theorem incorrect_connection_probability : probability_of_incorrect_connection 0.02 = 0.000131 :=
by
  sorry

end incorrect_connection_probability_l844_84450


namespace circle_placement_possible_l844_84454

theorem circle_placement_possible
  (length : ℕ)
  (width : ℕ)
  (n : ℕ)
  (area_ci : ℕ)
  (ne_int_lt : length = 20)
  (ne_wid_lt : width = 25)
  (ne_squares : n = 120)
  (sm_area_lt : area_ci = 456) :
  120 * (1 + (Real.pi / 4)) < area_ci :=
by sorry

end circle_placement_possible_l844_84454


namespace consecutive_numbers_count_l844_84488

theorem consecutive_numbers_count (n x : ℕ) (h_avg : (2 * n * 20 = n * (2 * x + n - 1))) (h_largest : x + n - 1 = 23) : n = 7 :=
by
  sorry

end consecutive_numbers_count_l844_84488


namespace age_difference_l844_84445

-- Let D denote the daughter's age and M denote the mother's age
variable (D M : ℕ)

-- Conditions given in the problem
axiom h1 : M = 11 * D
axiom h2 : M + 13 = 2 * (D + 13)

-- The main proof statement to show the difference in their current ages
theorem age_difference : M - D = 40 :=
by
  sorry

end age_difference_l844_84445


namespace total_cost_is_correct_l844_84442

def cost_per_pound : ℝ := 0.45
def weight_sugar : ℝ := 40
def weight_flour : ℝ := 16

theorem total_cost_is_correct :
  weight_sugar * cost_per_pound + weight_flour * cost_per_pound = 25.20 :=
by
  sorry

end total_cost_is_correct_l844_84442


namespace radius_of_circumscribed_sphere_eq_a_l844_84418

-- Assume a to be a real number representing the side length of the base and height of the hexagonal pyramid
variables (a : ℝ)

-- Representing the base as a regular hexagon and the pyramid as having equal side length and height
def regular_hexagonal_pyramid (a : ℝ) : Type := {b : ℝ // b = a}

-- The radius of the circumscribed sphere to a given regular hexagonal pyramid
def radius_of_circumscribed_sphere (a : ℝ) : ℝ := a

-- Theorem stating that the radius of the sphere circumscribed around a regular hexagonal pyramid 
-- with side length and height both equal to a is a
theorem radius_of_circumscribed_sphere_eq_a (a : ℝ) :
  radius_of_circumscribed_sphere a = a :=
by {
  sorry
}

end radius_of_circumscribed_sphere_eq_a_l844_84418


namespace max_collisions_l844_84497

-- Define the problem
theorem max_collisions (n : ℕ) (hn : n > 0) : 
  ∃ C : ℕ, C = (n * (n - 1)) / 2 := 
sorry

end max_collisions_l844_84497


namespace complement_of_A_in_U_l844_84483

-- Define the universal set U as the set of integers
def U : Set ℤ := Set.univ

-- Define the set A as the set of odd integers
def A : Set ℤ := {x : ℤ | ∃ k : ℤ, x = 2 * k + 1}

-- Define the complement of A in U
def complement_A : Set ℤ := U \ A

-- State the equivalence to be proved
theorem complement_of_A_in_U :
  complement_A = {x : ℤ | ∃ k : ℤ, x = 2 * k} :=
by
  sorry

end complement_of_A_in_U_l844_84483


namespace find_x_squared_minus_y_squared_l844_84439

theorem find_x_squared_minus_y_squared 
  (x y : ℝ)
  (h1 : x + y = 5)
  (h2 : x - y = 1) :
  x^2 - y^2 = 5 := 
by
  sorry

end find_x_squared_minus_y_squared_l844_84439


namespace smallest_p_l844_84428

theorem smallest_p 
  (p q : ℕ) 
  (h1 : (5 : ℚ) / 8 < p / (q : ℚ) ∧ p / (q : ℚ) < 7 / 8)
  (h2 : p + q = 2005) : p = 772 :=
sorry

end smallest_p_l844_84428


namespace find_x_l844_84401

variables {K J : ℝ} {A B C A_star B_star C_star : Type*}

-- Define the triangles and areas
def triangle_area (K : ℝ) : Prop := K > 0

-- We know the fractions of segments in triangle
def segment_ratios (x : ℝ) : Prop :=
  0 < x ∧ x < 1 ∧
  ∀ (AA_star AB BB_star BC CC_star CA : ℝ),
    AA_star / AB = x ∧ BB_star / BC = x ∧ CC_star / CA = x

-- Area of the smaller inner triangle
def inner_triangle_area (x : ℝ) (K : ℝ) (J : ℝ) : Prop :=
  J = x * K

-- The theorem combining all to show x = 1/3
theorem find_x (x : ℝ) (K J : ℝ) (triangleAreaK : triangle_area K)
    (ratios : segment_ratios x)
    (innerArea : inner_triangle_area x K J) :
  x = 1 / 3 :=
by
  sorry

end find_x_l844_84401


namespace samantha_spends_36_dollars_l844_84435

def cost_per_toy : ℝ := 12.00
def discount_factor : ℝ := 0.5
def num_toys_bought : ℕ := 4

def total_spent (cost_per_toy : ℝ) (discount_factor : ℝ) (num_toys_bought : ℕ) : ℝ :=
  let pair_cost := cost_per_toy + (cost_per_toy * discount_factor)
  let num_pairs := num_toys_bought / 2
  num_pairs * pair_cost

theorem samantha_spends_36_dollars :
  total_spent cost_per_toy discount_factor num_toys_bought = 36.00 :=
sorry

end samantha_spends_36_dollars_l844_84435


namespace no_nontrivial_integer_solutions_l844_84452

theorem no_nontrivial_integer_solutions (x y z : ℤ) : x^3 + 2*y^3 + 4*z^3 - 6*x*y*z = 0 -> x = 0 ∧ y = 0 ∧ z = 0 :=
by
  sorry

end no_nontrivial_integer_solutions_l844_84452


namespace area_of_sector_l844_84444

theorem area_of_sector (r : ℝ) (θ : ℝ) (h1 : r = 10) (h2 : θ = π / 5) : 
  (1 / 2) * r * r * θ = 10 * π :=
by
  rw [h1, h2]
  sorry

end area_of_sector_l844_84444


namespace simplification_evaluation_l844_84461

theorem simplification_evaluation (x : ℝ) (h : x = Real.sqrt 5 + 2) :
  (x + 2) / (x - 1) / (x + 1 - 3 / (x - 1)) = Real.sqrt 5 / 5 :=
by
  sorry

end simplification_evaluation_l844_84461


namespace find_second_term_of_ratio_l844_84417

theorem find_second_term_of_ratio
  (a b c d : ℕ)
  (h1 : a = 6)
  (h2 : b = 7)
  (h3 : c = 3)
  (h4 : (a - c) * 4 < a * d) :
  d = 5 :=
by
  sorry

end find_second_term_of_ratio_l844_84417


namespace arrange_books_l844_84463

-- Given conditions
def math_books_count := 4
def history_books_count := 6

-- Question: How many ways can the books be arranged given the conditions?
theorem arrange_books (math_books_count history_books_count : ℕ) :
  math_books_count = 4 → 
  history_books_count = 6 →
  ∃ ways : ℕ, ways = 51840 :=
by
  sorry

end arrange_books_l844_84463


namespace arithmetic_sequence_sum_ratio_l844_84473

noncomputable def S (n : ℕ) (a_1 : ℚ) (d : ℚ) : ℚ :=
  n * a_1 + (n * (n - 1) / 2) * d

theorem arithmetic_sequence_sum_ratio (a_1 d : ℚ) (h : d ≠ 0) (h_ratio : (a_1 + 5 * d) / (a_1 + 2 * d) = 2) :
  S 6 a_1 d / S 3 a_1 d = 7 / 2 :=
by
  sorry

end arithmetic_sequence_sum_ratio_l844_84473


namespace negation_of_proposition_l844_84474

theorem negation_of_proposition (a b c : ℝ) (h1 : a + b + c ≥ 0) (h2 : abc ≤ 0) : 
  (∃ x y z : ℝ, (x < 0) ∧ (y < 0) ∧ (x = a ∨ x = b ∨ x = c) ∧ (y = a ∨ y = b ∨ y = c) ∧ (x ≠ y)) →
  ¬(∀ x y z : ℝ, (x < 0 ∨ y < 0 ∨ z < 0) → (x ≠ y → x ≠ z → y ≠ z → (x = a ∨ x = b ∨ x = c) ∧ (y = a ∨ y = b ∨ y = c) ∧ (z = a ∨ z = b ∨ z = c))) :=
sorry

end negation_of_proposition_l844_84474


namespace hilary_regular_toenails_in_jar_l844_84485

-- Conditions
def jar_capacity : Nat := 100
def big_toenail_size : Nat := 2
def num_big_toenails : Nat := 20
def remaining_regular_toenails_space : Nat := 20

-- Question & Answer
theorem hilary_regular_toenails_in_jar : 
  (jar_capacity - remaining_regular_toenails_space - (num_big_toenails * big_toenail_size)) = 40 :=
by
  sorry

end hilary_regular_toenails_in_jar_l844_84485


namespace total_money_l844_84446

theorem total_money 
  (n_pennies n_nickels n_dimes n_quarters n_half_dollars : ℝ) 
  (h_pennies : n_pennies = 9) 
  (h_nickels : n_nickels = 4) 
  (h_dimes : n_dimes = 3) 
  (h_quarters : n_quarters = 7) 
  (h_half_dollars : n_half_dollars = 5) : 
  0.01 * n_pennies + 0.05 * n_nickels + 0.10 * n_dimes + 0.25 * n_quarters + 0.50 * n_half_dollars = 4.84 :=
by 
  sorry

end total_money_l844_84446


namespace line_through_A_area_1_l844_84414

def line_equation : Prop :=
  ∃ k : ℚ, ∀ x y : ℚ, (y = k * (x + 2) + 2) ↔ 
    (x + 2 * y - 2 = 0 ∨ 2 * x + y + 2 = 0) ∧ 
    (2 * (k * 0 + 2) * (-2 - 2 / k) = 2)

theorem line_through_A_area_1 : line_equation :=
by
  sorry

end line_through_A_area_1_l844_84414


namespace intersection_point_on_y_axis_l844_84457

theorem intersection_point_on_y_axis (k : ℝ) :
  ∃ y : ℝ, 2 * 0 + 3 * y - k = 0 ∧ 0 - k * y + 12 = 0 ↔ k = 6 ∨ k = -6 :=
by
  sorry

end intersection_point_on_y_axis_l844_84457


namespace percent_increase_in_maintenance_time_l844_84494

theorem percent_increase_in_maintenance_time (original_time new_time : ℝ) (h1 : original_time = 25) (h2 : new_time = 30) : 
  ((new_time - original_time) / original_time) * 100 = 20 :=
by
  sorry

end percent_increase_in_maintenance_time_l844_84494


namespace simplify_and_evaluate_expression_l844_84416

theorem simplify_and_evaluate_expression (a b : ℝ) (h1 : a = -1) (h2 : a * b = 2) :
  3 * (2 * a^2 * b + a * b^2) - (3 * a * b^2 - a^2 * b) = -14 := by
  sorry

end simplify_and_evaluate_expression_l844_84416


namespace number_of_recipes_l844_84413

-- Let's define the necessary conditions.
def cups_per_recipe : ℕ := 2
def total_cups_needed : ℕ := 46

-- Prove that the number of recipes required is 23.
theorem number_of_recipes : total_cups_needed / cups_per_recipe = 23 :=
by
  sorry

end number_of_recipes_l844_84413


namespace percent_decrease_l844_84441

theorem percent_decrease (p_original p_sale : ℝ) (h₁ : p_original = 100) (h₂ : p_sale = 50) :
  ((p_original - p_sale) / p_original * 100) = 50 := by
  sorry

end percent_decrease_l844_84441


namespace min_value_of_expression_l844_84472

theorem min_value_of_expression (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (h : x * y * z = 27) 
  : x + 3 * y + 9 * z ≥ 27 :=
sorry

end min_value_of_expression_l844_84472


namespace Sara_house_size_l844_84447

theorem Sara_house_size (nada_size : ℕ) (h1 : nada_size = 450) (h2 : Sara_size = 2 * nada_size + 100) : Sara_size = 1000 :=
by sorry

end Sara_house_size_l844_84447


namespace vector_projection_unique_l844_84402

theorem vector_projection_unique (a : ℝ) (c d : ℝ) (h : c + 3 * d = 0) :
    ∃ p : ℝ × ℝ, (∀ a : ℝ, ∀ (v : ℝ × ℝ) (w : ℝ × ℝ), 
      v = (a, 3 * a - 2) → 
      w = (c, d) → 
      ∃ p : ℝ × ℝ, p = (3 / 5, -1 / 5)) :=
sorry

end vector_projection_unique_l844_84402


namespace minimize_sum_pos_maximize_product_pos_l844_84405

def N : ℕ := 10^1001 - 1

noncomputable def find_min_sum_position : ℕ := 996

noncomputable def find_max_product_position : ℕ := 995

theorem minimize_sum_pos :
  ∀ m : ℕ, (m ≠ find_min_sum_position) → 
      (2 * 10^m + 10^(1001-m) - 10) ≥ (2 * 10^find_min_sum_position + 10^(1001-find_min_sum_position) - 10) := 
sorry

theorem maximize_product_pos :
  ∀ m : ℕ, (m ≠ find_max_product_position) → 
      ((2 * 10^m - 1) * (10^(1001 - m) - 9)) ≤ ((2 * 10^find_max_product_position - 1) * (10^(1001 - find_max_product_position) - 9)) :=
sorry

end minimize_sum_pos_maximize_product_pos_l844_84405


namespace inequality_holds_for_real_numbers_l844_84496

theorem inequality_holds_for_real_numbers (a1 a2 a3 a4 : ℝ) (h1 : 1 < a1) 
  (h2 : 1 < a2) (h3 : 1 < a3) (h4 : 1 < a4) : 
  8 * (a1 * a2 * a3 * a4 + 1) ≥ (1 + a1) * (1 + a2) * (1 + a3) * (1 + a4) :=
by sorry

end inequality_holds_for_real_numbers_l844_84496


namespace find_value_of_m_l844_84453

theorem find_value_of_m (m : ℝ) :
  (∃ (x y : ℝ), x^2 + y^2 - 4*x + 2*y + m = 0 ∧ (x - 2)^2 + (y + 1)^2 = 4) →
  m = 1 :=
sorry

end find_value_of_m_l844_84453


namespace election_total_valid_votes_l844_84455

theorem election_total_valid_votes (V B : ℝ) 
    (hA : 0.45 * V = B * V + 250) 
    (hB : 2.5 * B = 62.5) :
    V = 1250 :=
by
  sorry

end election_total_valid_votes_l844_84455


namespace avg_time_stopped_per_hour_l844_84464

-- Definitions and conditions
def avgSpeedInMotion : ℝ := 75
def overallAvgSpeed : ℝ := 40

-- Statement to prove
theorem avg_time_stopped_per_hour :
  (1 - overallAvgSpeed / avgSpeedInMotion) * 60 = 28 := 
by
  sorry

end avg_time_stopped_per_hour_l844_84464


namespace percentage_passed_all_three_l844_84433

variable (F_H F_E F_M F_HE F_EM F_HM F_HEM : ℝ)

theorem percentage_passed_all_three :
  F_H = 0.46 →
  F_E = 0.54 →
  F_M = 0.32 →
  F_HE = 0.18 →
  F_EM = 0.12 →
  F_HM = 0.1 →
  F_HEM = 0.06 →
  (100 - (F_H + F_E + F_M - F_HE - F_EM - F_HM + F_HEM)) = 2 :=
by sorry

end percentage_passed_all_three_l844_84433


namespace find_number_l844_84491

theorem find_number (x : ℝ) (h : 0.5 * x = 0.1667 * x + 10) : x = 30 :=
by {
  sorry
}

end find_number_l844_84491


namespace train_length_is_200_l844_84490

noncomputable def train_length 
  (speed_kmh : ℕ) 
  (time_s: ℕ) : ℕ := 
  ((speed_kmh * 1000) / 3600) * time_s

theorem train_length_is_200
  (h_speed : 40 = 40)
  (h_time : 18 = 18) :
  train_length 40 18 = 200 :=
sorry

end train_length_is_200_l844_84490


namespace trains_pass_time_l844_84468

def length_train1 : ℕ := 200
def length_train2 : ℕ := 280

def speed_train1_kmph : ℕ := 42
def speed_train2_kmph : ℕ := 30

def kmph_to_mps (speed_kmph : ℕ) : ℚ :=
  speed_kmph * 1000 / 3600

def relative_speed_mps : ℚ :=
  kmph_to_mps (speed_train1_kmph + speed_train2_kmph)

def total_length : ℕ :=
  length_train1 + length_train2

def time_to_pass_trains : ℚ :=
  total_length / relative_speed_mps

theorem trains_pass_time :
  time_to_pass_trains = 24 := by
  sorry

end trains_pass_time_l844_84468


namespace evaluateExpression_correct_l844_84429

open Real

noncomputable def evaluateExpression : ℝ :=
  (-2)^2 + 2 * sin (π / 3) - tan (π / 3)

theorem evaluateExpression_correct : evaluateExpression = 4 :=
  sorry

end evaluateExpression_correct_l844_84429


namespace probability_exactly_two_singers_same_province_l844_84411

-- Defining the number of provinces and number of singers per province
def num_provinces : ℕ := 6
def singers_per_province : ℕ := 2

-- Total number of singers
def num_singers : ℕ := num_provinces * singers_per_province

-- Define the total number of ways to choose 4 winners from 12 contestants
def total_combinations : ℕ := Nat.choose num_singers 4

-- Define the number of favorable ways to select exactly two singers from the same province and two from two other provinces
def favorable_combinations : ℕ := 
  (Nat.choose num_provinces 1) *  -- Choose one province for the pair
  (Nat.choose (num_provinces - 1) 2) *  -- Choose two remaining provinces
  (Nat.choose singers_per_province 1) *
  (Nat.choose singers_per_province 1)

-- Calculate the probability
def probability : ℚ := favorable_combinations / total_combinations

-- Stating the theorem to be proved
theorem probability_exactly_two_singers_same_province : probability = 16 / 33 :=
by
  sorry

end probability_exactly_two_singers_same_province_l844_84411


namespace magnitude_squared_complex_l844_84448

noncomputable def complex_number := Complex.mk 3 (-4)
noncomputable def squared_complex := complex_number * complex_number

theorem magnitude_squared_complex : Complex.abs squared_complex = 25 :=
by
  sorry

end magnitude_squared_complex_l844_84448


namespace sports_day_results_l844_84482

-- Conditions and questions
variables (a b c : ℕ)
variables (class1_score class2_score class3_score class4_score : ℕ)

-- Conditions given in the problem
axiom pos_a : 0 < a
axiom pos_b : 0 < b
axiom pos_c : 0 < c
axiom a_gt_b_gt_c : a > b ∧ b > c
axiom no_ties : (class1_score ≠ class2_score) ∧ (class2_score ≠ class3_score) ∧ (class3_score ≠ class4_score) ∧ (class1_score ≠ class3_score) ∧ (class1_score ≠ class4_score) ∧ (class2_score ≠ class4_score)
axiom class_scores : class1_score + class2_score + class3_score + class4_score = 40

-- To prove
theorem sports_day_results : a + b + c = 8 ∧ a = 5 :=
by
  sorry

end sports_day_results_l844_84482


namespace polynomial_product_is_square_l844_84475

theorem polynomial_product_is_square (x a : ℝ) :
  (x + a) * (x + 2 * a) * (x + 3 * a) * (x + 4 * a) + a^4 = (x^2 + 5 * a * x + 5 * a^2)^2 :=
by
  sorry

end polynomial_product_is_square_l844_84475


namespace problem_statement_l844_84484

-- Defining the properties of the function
def even_function (f : ℝ → ℝ) := ∀ x, f x = f (-x)
def symmetric_about_2 (f : ℝ → ℝ) := ∀ x, f (2 + (2 - x)) = f x

-- Given the function f, even function, and symmetric about line x = 2,
-- and given that f(3) = 3, we need to prove f(-1) = 3.
theorem problem_statement (f : ℝ → ℝ) 
  (h1 : even_function f) 
  (h2 : symmetric_about_2 f) 
  (h3 : f 3 = 3) : 
  f (-1) = 3 := 
sorry

end problem_statement_l844_84484


namespace inequality_abc_l844_84499

theorem inequality_abc (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a + b) / (a^2 + b^2) + (b + c) / (b^2 + c^2) + (c + a) / (c^2 + a^2) ≤ 1/a + 1/b + 1/c := 
by
  sorry

end inequality_abc_l844_84499


namespace trigonometric_bound_l844_84408

open Real

theorem trigonometric_bound (x y : ℝ) : 
  -1/2 ≤ (x + y) * (1 - x * y) / ((1 + x^2) * (1 + y^2)) ∧ 
  (x + y) * (1 - x * y) / ((1 + x^2) * (1 + y^2)) ≤ 1/2 :=
by 
  sorry

end trigonometric_bound_l844_84408


namespace vectors_parallel_x_squared_eq_two_l844_84404

theorem vectors_parallel_x_squared_eq_two (x : ℝ) 
  (a : ℝ × ℝ := (x+2, 1+x)) 
  (b : ℝ × ℝ := (x-2, 1-x)) 
  (parallel : (a.1 * b.2 - a.2 * b.1) = 0) : x^2 = 2 :=
sorry

end vectors_parallel_x_squared_eq_two_l844_84404


namespace solve_fraction_l844_84465

theorem solve_fraction : (3.242 * 10) / 100 = 0.3242 := by
  sorry

end solve_fraction_l844_84465


namespace inequality_must_hold_l844_84498

section
variables {a b c : ℝ}

theorem inequality_must_hold (h : a > b) : (a - b) * c^2 ≥ 0 :=
sorry
end

end inequality_must_hold_l844_84498


namespace license_plate_combinations_l844_84489

theorem license_plate_combinations :
  let num_consonants := 21
  let num_vowels := 5
  let num_digits := 10
  num_consonants^2 * num_vowels^2 * num_digits = 110250 :=
by
  let num_consonants := 21
  let num_vowels := 5
  let num_digits := 10
  sorry

end license_plate_combinations_l844_84489


namespace measure_of_angle_C_l844_84466

variable (C D : ℕ)
variable (h1 : C + D = 180)
variable (h2 : C = 5 * D)

theorem measure_of_angle_C : C = 150 :=
by
  sorry

end measure_of_angle_C_l844_84466
