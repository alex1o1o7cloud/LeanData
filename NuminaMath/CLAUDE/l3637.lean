import Mathlib

namespace right_triangle_inequality_l3637_363708

theorem right_triangle_inequality (a b c : ℝ) (h1 : 0 < a) (h2 : a ≤ b) (h3 : b < c)
  (h4 : a^2 + b^2 = c^2) :
  (1/a) + (1/b) + (1/c) ≥ (5 + 3 * Real.sqrt 2) / (a + b + c) := by
  sorry

end right_triangle_inequality_l3637_363708


namespace choose_six_three_equals_twenty_l3637_363742

theorem choose_six_three_equals_twenty : Nat.choose 6 3 = 20 := by
  sorry

end choose_six_three_equals_twenty_l3637_363742


namespace regular_polygon_interior_angle_sum_l3637_363713

theorem regular_polygon_interior_angle_sum (exterior_angle : ℝ) : 
  exterior_angle = 72 → (360 / exterior_angle - 2) * 180 = 540 := by
  sorry

end regular_polygon_interior_angle_sum_l3637_363713


namespace employee_pay_l3637_363763

theorem employee_pay (total : ℝ) (x y : ℝ) (h1 : total = 770) (h2 : x = 1.2 * y) (h3 : x + y = total) : y = 350 := by
  sorry

end employee_pay_l3637_363763


namespace chemistry_class_size_l3637_363717

theorem chemistry_class_size :
  -- Total number of students
  let total : ℕ := 120
  -- Students in both chemistry and biology
  let chem_bio : ℕ := 35
  -- Students in both biology and physics
  let bio_phys : ℕ := 15
  -- Students in both chemistry and physics
  let chem_phys : ℕ := 10
  -- Function to calculate total students in a class
  let class_size (only : ℕ) (with_other1 : ℕ) (with_other2 : ℕ) := only + with_other1 + with_other2
  -- Constraint: Chemistry class is four times as large as biology class
  ∀ (bio_only : ℕ) (chem_only : ℕ) (phys_only : ℕ),
    class_size chem_only chem_bio chem_phys = 4 * class_size bio_only chem_bio bio_phys →
    -- Constraint: No student takes all three classes
    class_size bio_only chem_bio bio_phys + class_size chem_only chem_bio chem_phys + class_size phys_only bio_phys chem_phys = total →
    -- Conclusion: Chemistry class size is 198
    class_size chem_only chem_bio chem_phys = 198 :=
by
  sorry

end chemistry_class_size_l3637_363717


namespace parallel_line_and_plane_existence_l3637_363743

-- Define a line in 3D space
structure Line3D where
  point : ℝ × ℝ × ℝ
  direction : ℝ × ℝ × ℝ

-- Define a plane in 3D space
structure Plane3D where
  point : ℝ × ℝ × ℝ
  normal : ℝ × ℝ × ℝ

-- Define parallelism between lines
def parallel_lines (l1 l2 : Line3D) : Prop := sorry

-- Define parallelism between a line and a plane
def parallel_line_plane (l : Line3D) (p : Plane3D) : Prop := sorry

-- Define a point not on a line
def point_not_on_line (p : ℝ × ℝ × ℝ) (l : Line3D) : Prop := sorry

theorem parallel_line_and_plane_existence 
  (l : Line3D) (p : ℝ × ℝ × ℝ) (h : point_not_on_line p l) : 
  (∃! l' : Line3D, parallel_lines l l' ∧ l'.point = p) ∧ 
  (∃ f : ℕ → Plane3D, (∀ n : ℕ, parallel_line_plane l (f n) ∧ (f n).point = p) ∧ 
                      (∀ n m : ℕ, n ≠ m → f n ≠ f m)) := by
  sorry

end parallel_line_and_plane_existence_l3637_363743


namespace motel_rent_is_400_l3637_363788

/-- The total rent charged by a motel on a Saturday night. -/
def totalRent (r50 r60 : ℕ) : ℝ := 50 * r50 + 60 * r60

/-- The rent after changing 10 rooms from $60 to $50. -/
def newRent (r50 r60 : ℕ) : ℝ := 50 * (r50 + 10) + 60 * (r60 - 10)

/-- The theorem stating that the total rent is $400. -/
theorem motel_rent_is_400 (r50 r60 : ℕ) : 
  (∃ (r50 r60 : ℕ), totalRent r50 r60 = 400 ∧ 
    newRent r50 r60 = 0.75 * totalRent r50 r60) := by
  sorry

end motel_rent_is_400_l3637_363788


namespace grade_distribution_l3637_363710

theorem grade_distribution (total_students : ℕ) 
  (fraction_A : ℚ) (fraction_C : ℚ) (num_D : ℕ) :
  total_students = 800 →
  fraction_A = 1/5 →
  fraction_C = 1/2 →
  num_D = 40 →
  (total_students : ℚ) * (1 - fraction_A - fraction_C) - num_D = (1/4 : ℚ) * total_students :=
by
  sorry

end grade_distribution_l3637_363710


namespace inequality_proof_l3637_363773

theorem inequality_proof (n : ℕ) (h : n > 1) : 1 + n * 2^((n-1)/2) < 2^n := by
  sorry

end inequality_proof_l3637_363773


namespace typing_time_together_l3637_363754

/-- Given Meso's and Tyler's typing speeds, calculate the time it takes them to type 40 pages together -/
theorem typing_time_together 
  (meso_pages : ℕ) (meso_time : ℕ) (tyler_pages : ℕ) (tyler_time : ℕ) (total_pages : ℕ) :
  meso_pages = 15 →
  meso_time = 5 →
  tyler_pages = 15 →
  tyler_time = 3 →
  total_pages = 40 →
  (total_pages : ℚ) / ((meso_pages : ℚ) / (meso_time : ℚ) + (tyler_pages : ℚ) / (tyler_time : ℚ)) = 5 := by
  sorry

end typing_time_together_l3637_363754


namespace extremum_implies_a_eq_neg_two_l3637_363751

/-- The function f(x) = a ln x + x^2 has an extremum at x = 1 -/
def has_extremum_at_one (a : ℝ) : Prop :=
  let f := fun x : ℝ => a * Real.log x + x^2
  ∃ ε > 0, ∀ x ∈ Set.Ioo (1 - ε) (1 + ε), f x ≤ f 1 ∨ f x ≥ f 1

/-- If f(x) = a ln x + x^2 has an extremum at x = 1, then a = -2 -/
theorem extremum_implies_a_eq_neg_two (a : ℝ) :
  has_extremum_at_one a → a = -2 :=
by
  sorry

end extremum_implies_a_eq_neg_two_l3637_363751


namespace f_g_zero_range_l3637_363756

def f (x : ℝ) : ℝ := sorry

def g (x : ℝ) : ℝ := f x

theorem f_g_zero_range (π : ℝ) (h_π : π > 0) :
  (∀ x ∈ Set.Icc (1 / π) π, f x = f (1 / x)) →
  (∀ x ∈ Set.Icc (1 / π) 1, f x = Real.log x) →
  (∃ x ∈ Set.Icc (1 / π) π, g x = 0) →
  Set.Icc (-π * Real.log π) 0 = {a | g a = 0} := by sorry

end f_g_zero_range_l3637_363756


namespace sqrt_18_div_sqrt_8_l3637_363760

theorem sqrt_18_div_sqrt_8 : Real.sqrt 18 / Real.sqrt 8 = 3 / 2 := by
  sorry

end sqrt_18_div_sqrt_8_l3637_363760


namespace nested_bracket_value_l3637_363740

-- Define the operation [a,b,c]
def bracket (a b c : ℚ) : ℚ := (a + b) / c

-- State the theorem
theorem nested_bracket_value :
  bracket (bracket 80 40 120) (bracket 4 2 6) (bracket 50 25 75) = 2 := by
  sorry

end nested_bracket_value_l3637_363740


namespace chess_tournament_games_l3637_363785

theorem chess_tournament_games (n : ℕ) 
  (total_players : ℕ) (total_games : ℕ) 
  (h1 : total_players = 17) 
  (h2 : total_games = 272) : n = 2 := by
  sorry

end chess_tournament_games_l3637_363785


namespace kosher_meals_count_l3637_363707

/-- Calculates the number of kosher meals given the total number of clients,
    number of vegan meals, number of both vegan and kosher meals,
    and number of meals that are neither vegan nor kosher. -/
def kosher_meals (total : ℕ) (vegan : ℕ) (both : ℕ) (neither : ℕ) : ℕ :=
  total - neither - (vegan - both)

/-- Proves that the number of clients needing kosher meals is 8,
    given the specific conditions from the problem. -/
theorem kosher_meals_count :
  kosher_meals 30 7 3 18 = 8 := by
  sorry

end kosher_meals_count_l3637_363707


namespace max_no_draw_participants_16_550_l3637_363744

/-- Represents a tic-tac-toe tournament -/
structure Tournament where
  participants : ℕ
  total_points : ℕ
  win_points : ℕ
  draw_points : ℕ
  loss_points : ℕ

/-- Calculates the total number of games in a tournament -/
def total_games (t : Tournament) : ℕ :=
  t.participants * (t.participants - 1) / 2

/-- Calculates the maximum number of participants who could have played without a draw -/
def max_no_draw_participants (t : Tournament) : ℕ :=
  sorry

/-- Theorem stating the maximum number of participants without a draw in the given tournament -/
theorem max_no_draw_participants_16_550 :
  let t : Tournament := {
    participants := 16,
    total_points := 550,
    win_points := 5,
    draw_points := 2,
    loss_points := 0
  }
  max_no_draw_participants t = 5 := by
  sorry

end max_no_draw_participants_16_550_l3637_363744


namespace exactly_three_special_triangles_l3637_363799

/-- A right-angled triangle with integer sides where the area is twice the perimeter -/
structure SpecialTriangle where
  a : ℕ+
  b : ℕ+
  c : ℕ+
  right_angled : a.val^2 + b.val^2 = c.val^2
  area_perimeter : (a.val * b.val : ℕ) = 4 * (a.val + b.val + c.val)

/-- There are exactly three special triangles -/
theorem exactly_three_special_triangles : 
  ∃! (list : List SpecialTriangle), list.length = 3 ∧ 
  (∀ t : SpecialTriangle, t ∈ list) ∧
  (∀ t ∈ list, t ∈ [⟨9, 40, 41, sorry, sorry⟩, ⟨10, 24, 26, sorry, sorry⟩, ⟨12, 16, 20, sorry, sorry⟩]) :=
sorry

end exactly_three_special_triangles_l3637_363799


namespace fraction_equality_l3637_363728

theorem fraction_equality (x y : ℚ) (hx : x = 3/5) (hy : y = 7/9) :
  (5*x + 9*y) / (45*x*y) = 10/21 := by
  sorry

end fraction_equality_l3637_363728


namespace g_max_min_l3637_363718

noncomputable def g (x : ℝ) : ℝ := Real.sin x ^ 8 + 8 * Real.cos x ^ 8

theorem g_max_min :
  (∀ x, g x ≤ 8) ∧ (∃ x, g x = 8) ∧ (∀ x, 8/27 ≤ g x) ∧ (∃ x, g x = 8/27) :=
sorry

end g_max_min_l3637_363718


namespace complex_in_fourth_quadrant_l3637_363722

/-- The complex number z = (2-i)/(1+i) is located in the fourth quadrant of the complex plane. -/
theorem complex_in_fourth_quadrant : ∃ (x y : ℝ), 
  (x > 0 ∧ y < 0) ∧ 
  (Complex.I : ℂ) * ((2 : ℂ) - Complex.I) = ((1 : ℂ) + Complex.I) * (x + y * Complex.I) := by
  sorry

end complex_in_fourth_quadrant_l3637_363722


namespace logical_equivalences_l3637_363719

theorem logical_equivalences :
  (∀ A B C : Prop,
    ((A ∨ B) → C) ↔ ((A → C) ∧ (B → C))) ∧
  (∀ A B C : Prop,
    (A → (B ∧ C)) ↔ ((A → B) ∧ (A → C))) := by
  sorry

end logical_equivalences_l3637_363719


namespace function_continuity_l3637_363712

-- Define a function f on the real line
variable (f : ℝ → ℝ)

-- Define the condition that f(x) + f(ax) is continuous for any a > 1
def condition (f : ℝ → ℝ) : Prop :=
  ∀ a : ℝ, a > 1 → Continuous (fun x ↦ f x + f (a * x))

-- Theorem statement
theorem function_continuity (h : condition f) : Continuous f := by
  sorry

end function_continuity_l3637_363712


namespace water_bucket_addition_l3637_363757

theorem water_bucket_addition (initial_water : ℝ) (added_water : ℝ) :
  initial_water = 3 → added_water = 6.8 → initial_water + added_water = 9.8 :=
by
  sorry

end water_bucket_addition_l3637_363757


namespace problem_statement_l3637_363796

theorem problem_statement (a b : ℝ) (h : |a + 1| + (b - 2)^2 = 0) : 
  (a + b)^2021 + a^2022 = 2 := by
sorry

end problem_statement_l3637_363796


namespace no_real_roots_composite_l3637_363715

-- Define the function f
def f (b c x : ℝ) : ℝ := x^2 + b*x + c

-- State the theorem
theorem no_real_roots_composite (b c : ℝ) :
  (∀ x : ℝ, f b c x ≠ x) →
  (∀ x : ℝ, f b c (f b c x) ≠ x) :=
by sorry

end no_real_roots_composite_l3637_363715


namespace cytoplasm_distribution_in_cell_division_l3637_363791

/-- Represents a cell in a diploid organism -/
structure DiploidCell where
  cytoplasm : Set ℝ
  deriving Inhabited

/-- Represents the process of cell division -/
def cell_division (parent : DiploidCell) : DiploidCell × DiploidCell :=
  sorry

/-- The distribution of cytoplasm during cell division is random -/
def is_random_distribution (division : DiploidCell → DiploidCell × DiploidCell) : Prop :=
  sorry

/-- The distribution of cytoplasm during cell division is unequal -/
def is_unequal_distribution (division : DiploidCell → DiploidCell × DiploidCell) : Prop :=
  sorry

/-- Theorem: In diploid organism cells, the distribution of cytoplasm during cell division is random and unequal -/
theorem cytoplasm_distribution_in_cell_division :
  is_random_distribution cell_division ∧ is_unequal_distribution cell_division :=
sorry

end cytoplasm_distribution_in_cell_division_l3637_363791


namespace clock_angle_theorem_l3637_363782

/-- The angle in radians through which the minute hand of a clock turns from 1:00 to 3:20 -/
def clock_angle_radians : ℝ := sorry

/-- The angle in degrees that the minute hand turns per minute -/
def minute_hand_degrees_per_minute : ℝ := 6

/-- The time difference in minutes from 1:00 to 3:20 -/
def time_difference_minutes : ℕ := 2 * 60 + 20

theorem clock_angle_theorem : 
  clock_angle_radians = -(minute_hand_degrees_per_minute * time_difference_minutes * (π / 180)) := by
  sorry

end clock_angle_theorem_l3637_363782


namespace inequality_preservation_l3637_363732

theorem inequality_preservation (a b : ℝ) (h : a > b) : a / 3 > b / 3 := by
  sorry

end inequality_preservation_l3637_363732


namespace min_value_z_l3637_363727

theorem min_value_z (x y : ℝ) : 3 * x^2 + 5 * y^2 + 8 * x - 10 * y + 4 * x * y + 35 ≥ 251 / 9 := by
  sorry

end min_value_z_l3637_363727


namespace shortest_side_in_triangle_l3637_363793

theorem shortest_side_in_triangle (A B C : Real) (a b c : Real) :
  B = 45 * π / 180 →  -- Convert 45° to radians
  C = 60 * π / 180 →  -- Convert 60° to radians
  c = 1 →
  A + B + C = π →     -- Sum of angles in a triangle
  a / Real.sin A = b / Real.sin B →  -- Law of Sines
  b / Real.sin B = c / Real.sin C →  -- Law of Sines
  b < a ∧ b < c →     -- b is the shortest side
  b = Real.sqrt 6 / 3 :=
by sorry

end shortest_side_in_triangle_l3637_363793


namespace stationery_store_problem_l3637_363762

/-- Represents the weekly sales volume as a function of the selling price -/
def sales_volume (x : ℝ) : ℝ := -2 * x + 80

/-- Represents the weekly profit as a function of the selling price -/
def profit (x : ℝ) : ℝ := (x - 20) * (sales_volume x)

theorem stationery_store_problem 
  (h_price_range : ∀ x, 20 ≤ x ∧ x ≤ 28 → x ∈ Set.Icc 20 28)
  (h_sales_22 : sales_volume 22 = 36)
  (h_sales_24 : sales_volume 24 = 32) :
  (∀ x, sales_volume x = -2 * x + 80) ∧
  (∃ x ∈ Set.Icc 20 28, profit x = 150 ∧ x = 25) ∧
  (∀ x ∈ Set.Icc 20 28, profit x ≤ profit 28 ∧ profit 28 = 192) := by
  sorry

#check stationery_store_problem

end stationery_store_problem_l3637_363762


namespace locus_of_vertex_A_l3637_363741

-- Define the triangle ABC
def Triangle (A B C : ℝ × ℝ) : Prop := True

-- Define the median CM
def Median (C M : ℝ × ℝ) : Prop := True

-- Define the constant length of CM
def ConstantLength (CM : ℝ) : Prop := True

-- Define the midpoint of BC
def Midpoint (K B C : ℝ × ℝ) : Prop := 
  K.1 = (B.1 + C.1) / 2 ∧ K.2 = (B.2 + C.2) / 2

-- Define a circle
def Circle (center : ℝ × ℝ) (radius : ℝ) (point : ℝ × ℝ) : Prop :=
  (point.1 - center.1)^2 + (point.2 - center.2)^2 = radius^2

-- Theorem statement
theorem locus_of_vertex_A (A B C : ℝ × ℝ) (K : ℝ × ℝ) (CM : ℝ) :
  Triangle A B C →
  Midpoint K B C →
  ConstantLength CM →
  ∀ M, Median C M →
  ∃ center radius, Circle center radius A ∧ 
    center = K ∧ 
    radius = 2 * CM ∧
    ¬(A.1 = B.1 ∧ A.2 = B.2) ∧ 
    ¬(A.1 = C.1 ∧ A.2 = C.2) :=
by sorry

end locus_of_vertex_A_l3637_363741


namespace inequality_and_equality_condition_l3637_363721

theorem inequality_and_equality_condition (a b c : ℝ) (α : ℝ) 
    (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  a * b * c * (a^α + b^α + c^α) ≥ 
    a^(α+2) * (-a + b + c) + b^(α+2) * (a - b + c) + c^(α+2) * (a + b - c) ∧
  (a * b * c * (a^α + b^α + c^α) = 
    a^(α+2) * (-a + b + c) + b^(α+2) * (a - b + c) + c^(α+2) * (a + b - c) ↔
   a = b ∧ b = c) :=
by sorry

end inequality_and_equality_condition_l3637_363721


namespace fifth_root_of_x_times_fourth_root_l3637_363747

theorem fifth_root_of_x_times_fourth_root (x : ℝ) (hx : x > 0) :
  (x * x^(1/4))^(1/5) = x^(1/4) := by
  sorry

end fifth_root_of_x_times_fourth_root_l3637_363747


namespace curves_intersection_equality_l3637_363778

-- Define the four curves
def C₁ (x y : ℝ) : Prop := x^2 - y^2 = x / (x^2 + y^2)
def C₂ (x y : ℝ) : Prop := 2*x*y + y / (x^2 + y^2) = 3
def C₃ (x y : ℝ) : Prop := x^3 - 3*x*y^2 + 3*y = 1
def C₄ (x y : ℝ) : Prop := 3*y*x^2 - 3*x - y^3 = 0

-- State the theorem
theorem curves_intersection_equality :
  ∀ (x y : ℝ), (C₁ x y ∧ C₂ x y) ↔ (C₃ x y ∧ C₄ x y) := by sorry

end curves_intersection_equality_l3637_363778


namespace perimeter_division_ratio_l3637_363783

/-- A square with a point on its diagonal and a line passing through that point. -/
structure SquareWithLine where
  /-- Side length of the square -/
  a : ℝ
  /-- Point M divides diagonal AC in ratio 2:1 -/
  m : ℝ × ℝ
  /-- The line divides the square's area in ratio 9:31 -/
  areaRatio : ℝ × ℝ
  /-- Conditions -/
  h1 : a > 0
  h2 : m = (2*a/3, 2*a/3)
  h3 : areaRatio = (9, 31)

/-- The theorem to be proved -/
theorem perimeter_division_ratio (s : SquareWithLine) :
  let p1 := (9 : ℝ) / 10 * (4 * s.a)
  let p2 := (31 : ℝ) / 10 * (4 * s.a)
  (p1, p2) = (9, 31) := by sorry

end perimeter_division_ratio_l3637_363783


namespace seq1_infinitely_many_composites_seq2_infinitely_many_composites_l3637_363772

-- Define the first sequence
def seq1 (n : ℕ) : ℕ :=
  3^n * 10^n + 7

-- Define the second sequence
def seq2 (n : ℕ) : ℕ :=
  3^n * 10^n + 31

-- Statement for the first sequence
theorem seq1_infinitely_many_composites :
  ∃ (S : Set ℕ), Set.Infinite S ∧ ∀ (n : ℕ), n ∈ S → ¬ Nat.Prime (seq1 n) :=
sorry

-- Statement for the second sequence
theorem seq2_infinitely_many_composites :
  ∃ (S : Set ℕ), Set.Infinite S ∧ ∀ (n : ℕ), n ∈ S → ¬ Nat.Prime (seq2 n) :=
sorry

end seq1_infinitely_many_composites_seq2_infinitely_many_composites_l3637_363772


namespace power_of_power_l3637_363700

theorem power_of_power : (3^4)^2 = 6561 := by
  sorry

end power_of_power_l3637_363700


namespace camel_cost_is_5200_l3637_363736

-- Define the cost of each animal
def camel_cost : ℝ := 5200
def elephant_cost : ℝ := 13000
def ox_cost : ℝ := 8666.67
def horse_cost : ℝ := 2166.67

-- Define the relationships between animal costs
axiom camel_horse_ratio : 10 * camel_cost = 24 * horse_cost
axiom horse_ox_ratio : ∃ x : ℕ, x * horse_cost = 4 * ox_cost
axiom ox_elephant_ratio : 6 * ox_cost = 4 * elephant_cost
axiom elephant_total_cost : 10 * elephant_cost = 130000

-- Theorem to prove
theorem camel_cost_is_5200 : camel_cost = 5200 := by
  sorry

end camel_cost_is_5200_l3637_363736


namespace square_division_perimeter_l3637_363730

theorem square_division_perimeter (s : ℝ) (h1 : s > 0) : 
  let square_perimeter := 4 * s
  let rectangle_length := s
  let rectangle_width := s / 2
  let rectangle_perimeter := 2 * (rectangle_length + rectangle_width)
  square_perimeter = 200 → rectangle_perimeter = 150 := by
  sorry

end square_division_perimeter_l3637_363730


namespace union_determines_x_l3637_363774

theorem union_determines_x (A B : Set ℕ) (x : ℕ) :
  A = {1, 2, x} →
  B = {2, 4, 5} →
  A ∪ B = {1, 2, 3, 4, 5} →
  x = 3 := by
sorry

end union_determines_x_l3637_363774


namespace mohamed_donated_three_bags_l3637_363768

/-- The number of bags Leila donated -/
def leila_bags : ℕ := 2

/-- The number of toys in each of Leila's bags -/
def leila_toys_per_bag : ℕ := 25

/-- The number of toys in each of Mohamed's bags -/
def mohamed_toys_per_bag : ℕ := 19

/-- The difference between Mohamed's and Leila's toy donations -/
def toy_difference : ℕ := 7

/-- Calculates the total number of toys Leila donated -/
def leila_total_toys : ℕ := leila_bags * leila_toys_per_bag

/-- Calculates the total number of toys Mohamed donated -/
def mohamed_total_toys : ℕ := leila_total_toys + toy_difference

/-- The number of bags Mohamed donated -/
def mohamed_bags : ℕ := mohamed_total_toys / mohamed_toys_per_bag

theorem mohamed_donated_three_bags : mohamed_bags = 3 := by
  sorry

end mohamed_donated_three_bags_l3637_363768


namespace complex_equation_solution_l3637_363706

theorem complex_equation_solution (z : ℂ) : z * (1 + 2*I) = 3 + I → z = 1 - I := by
  sorry

end complex_equation_solution_l3637_363706


namespace price_per_apple_l3637_363765

/-- Calculate the price per apple given the orchard layout, apple production, and total revenue -/
theorem price_per_apple (rows : ℕ) (columns : ℕ) (apples_per_tree : ℕ) (total_revenue : ℚ) : 
  rows = 3 → columns = 4 → apples_per_tree = 5 → total_revenue = 30 →
  total_revenue / (rows * columns * apples_per_tree) = 0.5 := by
sorry

end price_per_apple_l3637_363765


namespace certain_number_divisibility_l3637_363739

theorem certain_number_divisibility : ∃ (k : ℕ), k = 65 ∧ 
  (∀ (n : ℕ), n < 6 → ¬(k ∣ 11 * n - 1)) ∧ 
  (k ∣ 11 * 6 - 1) := by
  sorry

end certain_number_divisibility_l3637_363739


namespace segment_length_l3637_363753

theorem segment_length : 
  let endpoints := {x : ℝ | |x - (27 : ℝ)^(1/3)| = 5}
  ∃ a b : ℝ, a ∈ endpoints ∧ b ∈ endpoints ∧ |a - b| = 10 :=
by sorry

end segment_length_l3637_363753


namespace sequence_sum_l3637_363749

theorem sequence_sum (a b c d : ℕ) (h1 : 0 < a ∧ a < b ∧ b < c ∧ c < d)
  (h2 : b - a = c - b) (h3 : c * c = b * d) (h4 : d - a = 30) :
  a + b + c + d = 129 := by
sorry

end sequence_sum_l3637_363749


namespace oxen_grazing_problem_l3637_363733

theorem oxen_grazing_problem (total_rent : ℕ) (a_months b_oxen b_months c_oxen c_months : ℕ) (c_share : ℕ) :
  total_rent = 175 →
  a_months = 7 →
  b_oxen = 12 →
  b_months = 5 →
  c_oxen = 15 →
  c_months = 3 →
  c_share = 45 →
  ∃ a_oxen : ℕ, a_oxen * a_months + b_oxen * b_months + c_oxen * c_months = total_rent ∧ a_oxen = 10 := by
  sorry


end oxen_grazing_problem_l3637_363733


namespace inverse_proportion_translation_l3637_363734

/-- Given a non-zero constant k and a function f(x) = k/(x+1) - 2,
    if f(-3) = 1, then k = -6 -/
theorem inverse_proportion_translation (k : ℝ) (hk : k ≠ 0) :
  let f : ℝ → ℝ := λ x => k / (x + 1) - 2
  f (-3) = 1 → k = -6 := by
sorry

end inverse_proportion_translation_l3637_363734


namespace carlys_running_schedule_l3637_363725

/-- Carly's running schedule over four weeks -/
theorem carlys_running_schedule (x : ℝ) : 
  (∃ week2 week3 : ℝ,
    week2 = 2*x + 3 ∧ 
    week3 = (9/7) * week2 ∧ 
    week3 - 5 = 4) → 
  x = 2 := by sorry

end carlys_running_schedule_l3637_363725


namespace complex_number_quadrant_l3637_363759

theorem complex_number_quadrant : 
  let z : ℂ := (3 + Complex.I) * (1 - Complex.I)
  (z.re > 0 ∧ z.im < 0) :=
by
  sorry

end complex_number_quadrant_l3637_363759


namespace paint_cans_calculation_l3637_363767

theorem paint_cans_calculation (initial_cans : ℚ) : 
  (initial_cans / 2 - (initial_cans / 6 + 5) = 5) → initial_cans = 30 := by
  sorry

end paint_cans_calculation_l3637_363767


namespace largest_value_when_x_is_9_l3637_363705

theorem largest_value_when_x_is_9 :
  let x : ℝ := 9
  (x / 2 > Real.sqrt x) ∧
  (x / 2 > x - 5) ∧
  (x / 2 > 40 / x) ∧
  (x / 2 > x^2 / 20) := by
  sorry

end largest_value_when_x_is_9_l3637_363705


namespace apple_distribution_l3637_363786

/-- The number of ways to distribute n apples among k people, with each person receiving at least m apples -/
def distribute_apples (n k m : ℕ) : ℕ :=
  Nat.choose (n - k * m + k - 1) (k - 1)

/-- Theorem stating that there are 253 ways to distribute 30 apples among 3 people, with each person receiving at least 3 apples -/
theorem apple_distribution : distribute_apples 30 3 3 = 253 := by
  sorry

end apple_distribution_l3637_363786


namespace total_watching_time_l3637_363737

/-- Calculates the total watching time for two people watching multiple videos at different speeds -/
theorem total_watching_time
  (video_length : ℝ)
  (num_videos : ℕ)
  (speed_ratio_1 : ℝ)
  (speed_ratio_2 : ℝ)
  (h1 : video_length = 100)
  (h2 : num_videos = 6)
  (h3 : speed_ratio_1 = 2)
  (h4 : speed_ratio_2 = 1) :
  (num_videos * video_length / speed_ratio_1) + (num_videos * video_length / speed_ratio_2) = 900 := by
  sorry

#check total_watching_time

end total_watching_time_l3637_363737


namespace gcd_power_sum_l3637_363789

theorem gcd_power_sum (n : ℕ) (h : n > 32) :
  Nat.gcd (n^5 + 5^3) (n + 5) = if n % 5 = 0 then 5 else 1 := by sorry

end gcd_power_sum_l3637_363789


namespace five_students_two_groups_l3637_363776

/-- The number of ways to assign n students to k groups, where each student
    must be assigned to exactly one group. -/
def assignmentWays (n k : ℕ) : ℕ := k^n

/-- Theorem: The number of ways to assign 5 students to 2 groups is 32. -/
theorem five_students_two_groups : assignmentWays 5 2 = 32 := by
  sorry

end five_students_two_groups_l3637_363776


namespace total_insects_eaten_l3637_363770

theorem total_insects_eaten (num_geckos : ℕ) (insects_per_gecko : ℕ) (num_lizards : ℕ) : 
  num_geckos = 5 → insects_per_gecko = 6 → num_lizards = 3 → 
  num_geckos * insects_per_gecko + num_lizards * (2 * insects_per_gecko) = 66 := by
  sorry

#check total_insects_eaten

end total_insects_eaten_l3637_363770


namespace lcm_gcd_product_9_10_l3637_363777

theorem lcm_gcd_product_9_10 : Nat.lcm 9 10 * Nat.gcd 9 10 = 90 := by
  sorry

end lcm_gcd_product_9_10_l3637_363777


namespace symmetry_implies_m_and_n_l3637_363781

/-- Two points are symmetric about the x-axis if their x-coordinates are equal and their y-coordinates are opposites -/
def symmetric_about_x_axis (a b : ℝ × ℝ) : Prop :=
  a.1 = b.1 ∧ a.2 = -b.2

/-- The theorem stating that if A(-4, m-3) and B(2n, 1) are symmetric about the x-axis, then m = 2 and n = -2 -/
theorem symmetry_implies_m_and_n (m n : ℝ) :
  symmetric_about_x_axis (-4, m - 3) (2*n, 1) → m = 2 ∧ n = -2 := by
  sorry

end symmetry_implies_m_and_n_l3637_363781


namespace pages_copied_for_15_dollars_l3637_363795

/-- The number of pages that can be copied given a certain amount of money and cost per page. -/
def pages_copied (total_money : ℚ) (cost_per_page : ℚ) : ℚ :=
  (total_money * 100) / cost_per_page

/-- Theorem stating that with $15 and a cost of 3 cents per page, 500 pages can be copied. -/
theorem pages_copied_for_15_dollars : 
  pages_copied 15 3 = 500 := by
  sorry

end pages_copied_for_15_dollars_l3637_363795


namespace unique_solution_l3637_363729

def digit_product (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) * digit_product (n / 10)

theorem unique_solution : ∃! x : ℕ, x > 0 ∧ digit_product x = x^2 - 10*x - 22 :=
sorry

end unique_solution_l3637_363729


namespace compare_negative_fractions_l3637_363766

theorem compare_negative_fractions : -2/3 > -3/4 := by
  sorry

end compare_negative_fractions_l3637_363766


namespace triangle_area_l3637_363748

/-- The area of a triangle formed by the points (0,0), (1,1), and (2,1) is 1/2. -/
theorem triangle_area : 
  let A : ℝ × ℝ := (0, 0)
  let B : ℝ × ℝ := (1, 1)
  let C : ℝ × ℝ := (2, 1)
  let area := (1/2) * abs (A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2))
  area = 1/2 := by sorry

end triangle_area_l3637_363748


namespace video_watching_time_l3637_363702

theorem video_watching_time (video_length : ℕ) (num_videos : ℕ) : 
  video_length = 100 → num_videos = 6 → 
  (num_videos * video_length / 2 + num_videos * video_length) = 900 := by
  sorry

end video_watching_time_l3637_363702


namespace brother_age_relation_l3637_363784

theorem brother_age_relation : 
  let current_age_older : ℕ := 15
  let current_age_younger : ℕ := 5
  let years_passed : ℕ := 5
  (current_age_older + years_passed) = 2 * (current_age_younger + years_passed) :=
by sorry

end brother_age_relation_l3637_363784


namespace sine_transformation_l3637_363723

theorem sine_transformation (ω A a φ : Real) 
  (h_ω : ω > 0) (h_A : A > 0) (h_a : a > 0) (h_φ : 0 < φ ∧ φ < π) :
  (∀ x, A * Real.sin (ω * x - φ) + a = 3 * Real.sin (2 * x - π / 6) + 1) →
  A + a + ω + φ = 16 / 3 + 11 * π / 12 := by
sorry

end sine_transformation_l3637_363723


namespace major_axis_length_for_specific_cylinder_l3637_363775

/-- The length of the major axis of an ellipse formed by intersecting a right circular cylinder --/
def majorAxisLength (cylinderRadius : ℝ) (majorMinorRatio : ℝ) : ℝ :=
  2 * cylinderRadius * majorMinorRatio

/-- Theorem: The major axis length for the given conditions --/
theorem major_axis_length_for_specific_cylinder :
  majorAxisLength 3 1.2 = 7.2 :=
by
  sorry

end major_axis_length_for_specific_cylinder_l3637_363775


namespace positive_integer_division_problem_l3637_363738

theorem positive_integer_division_problem (a b : ℕ) : 
  a > 1 → b > 1 → (∃k : ℕ, b + 1 = k * a) → (∃l : ℕ, a^3 - 1 = l * b) →
  ((b = a - 1) ∨ (∃p : ℕ, p = 1 ∨ p = 2 ∧ a = a^p ∧ b = a^3 - 1)) :=
by sorry

end positive_integer_division_problem_l3637_363738


namespace sum_of_i_powers_l3637_363769

-- Define the complex number i
def i : ℂ := Complex.I

-- State the theorem
theorem sum_of_i_powers : i^13 + i^18 + i^23 + i^28 + i^33 = i := by sorry

end sum_of_i_powers_l3637_363769


namespace triangle_equilateral_l3637_363731

/-- Given a triangle ABC with angle C = 60° and c² = ab, prove that ABC is equilateral -/
theorem triangle_equilateral (a b c : ℝ) (angleC : ℝ) :
  angleC = π / 3 →  -- 60° in radians
  c^2 = a * b →
  a > 0 → b > 0 → c > 0 →
  a = b ∧ b = c :=
by sorry

end triangle_equilateral_l3637_363731


namespace celeste_opod_probability_l3637_363792

/-- Represents the duration of a song in seconds -/
def SongDuration := ℕ

/-- Represents the o-Pod with its songs -/
structure OPod where
  songs : List SongDuration
  favorite_song : SongDuration

/-- Creates an o-Pod with 10 songs, where each song is 30 seconds longer than the previous one -/
def create_opod (favorite_duration : ℕ) : OPod :=
  { songs := List.range 10 |>.map (fun i => 30 * (i + 1)),
    favorite_song := favorite_duration }

/-- Calculates the probability of not hearing the entire favorite song within a given time -/
noncomputable def prob_not_hear_favorite (opod : OPod) (total_time : ℕ) : ℚ :=
  sorry

theorem celeste_opod_probability :
  let celeste_opod := create_opod 210
  prob_not_hear_favorite celeste_opod 270 = 79 / 90 := by
  sorry

end celeste_opod_probability_l3637_363792


namespace product_remainder_l3637_363724

/-- The number of times 23 is repeated in the product -/
def n : ℕ := 23

/-- The divisor -/
def m : ℕ := 32

/-- Function to calculate the remainder of the product of n 23's when divided by m -/
def f (n m : ℕ) : ℕ := (23^n) % m

theorem product_remainder : f n m = 19 := by sorry

end product_remainder_l3637_363724


namespace arithmetic_sequence_problem_l3637_363787

/-- Proves that for an arithmetic sequence with given properties, m = 2 when S_m is the arithmetic mean of a_m and a_{m+1} -/
theorem arithmetic_sequence_problem (m : ℕ) : 
  let a : ℕ → ℤ := λ n => 2*n - 1
  let S : ℕ → ℤ := λ n => n^2
  (S m = (a m + a (m+1)) / 2) → m = 2 :=
by
  sorry


end arithmetic_sequence_problem_l3637_363787


namespace cow_selling_price_l3637_363764

/-- Calculates the selling price of a cow given the initial cost, daily food cost,
    vaccination and deworming cost, number of days, and profit made. -/
theorem cow_selling_price
  (initial_cost : ℕ)
  (daily_food_cost : ℕ)
  (vaccination_cost : ℕ)
  (num_days : ℕ)
  (profit : ℕ)
  (h1 : initial_cost = 600)
  (h2 : daily_food_cost = 20)
  (h3 : vaccination_cost = 500)
  (h4 : num_days = 40)
  (h5 : profit = 600) :
  initial_cost + num_days * daily_food_cost + vaccination_cost + profit = 2500 :=
by
  sorry

end cow_selling_price_l3637_363764


namespace special_number_theorem_l3637_363711

/-- The type of positive integers with at least seven divisors -/
def HasAtLeastSevenDivisors (n : ℕ) : Prop :=
  ∃ (d₁ d₂ d₃ d₄ d₅ d₆ d₇ : ℕ), d₁ < d₂ ∧ d₂ < d₃ ∧ d₃ < d₄ ∧ d₄ < d₅ ∧ d₅ < d₆ ∧ d₆ < d₇ ∧
    d₁ ∣ n ∧ d₂ ∣ n ∧ d₃ ∣ n ∧ d₄ ∣ n ∧ d₅ ∣ n ∧ d₆ ∣ n ∧ d₇ ∣ n

/-- The property that n + 1 is equal to the sum of squares of its 6th and 7th divisors -/
def SumOfSquaresProperty (n : ℕ) : Prop :=
  ∃ (d₆ d₇ : ℕ), d₆ < d₇ ∧ d₆ ∣ n ∧ d₇ ∣ n ∧
    (∀ d : ℕ, d ∣ n → d < d₆ ∨ d = d₆ ∨ d = d₇ ∨ d₇ < d) ∧
    n + 1 = d₆^2 + d₇^2

theorem special_number_theorem (n : ℕ) 
  (h1 : HasAtLeastSevenDivisors n)
  (h2 : SumOfSquaresProperty n) :
  n = 144 ∨ n = 1984 :=
sorry

end special_number_theorem_l3637_363711


namespace no_real_roots_quadratic_l3637_363716

theorem no_real_roots_quadratic (m : ℝ) : 
  (∀ x : ℝ, -2 * x^2 + 6 * x + m ≠ 0) → m < -4.5 := by
  sorry

end no_real_roots_quadratic_l3637_363716


namespace consecutive_odd_product_divisibility_l3637_363709

theorem consecutive_odd_product_divisibility :
  ∀ (a b c : ℕ), 
    (a > 0) → 
    (b > 0) → 
    (c > 0) → 
    (Odd a) → 
    (b = a + 2) → 
    (c = b + 2) → 
    (∃ (k : ℕ), a * b * c = 3 * k) ∧ 
    (∀ (m : ℕ), m > 3 → ¬(∀ (x y z : ℕ), 
      (x > 0) → 
      (y > 0) → 
      (z > 0) → 
      (Odd x) → 
      (y = x + 2) → 
      (z = y + 2) → 
      (∃ (n : ℕ), x * y * z = m * n))) :=
by sorry

end consecutive_odd_product_divisibility_l3637_363709


namespace simplest_quadratic_radical_l3637_363704

-- Define the concept of a quadratic radical
def QuadraticRadical (x : ℝ) : Prop := ∃ y : ℝ, x = y^2

-- Define what it means for a quadratic radical to be simplest
def SimplestQuadraticRadical (x : ℝ) : Prop :=
  QuadraticRadical x ∧ 
  ∀ y : ℝ, (∃ z : ℝ, y = z^2 ∧ x = y * z) → y = 1

-- State the theorem
theorem simplest_quadratic_radical :
  SimplestQuadraticRadical 6 ∧
  ¬SimplestQuadraticRadical 12 ∧
  ¬SimplestQuadraticRadical 0.3 ∧
  ¬SimplestQuadraticRadical (1/2) :=
sorry

end simplest_quadratic_radical_l3637_363704


namespace arithmetic_calculations_l3637_363779

theorem arithmetic_calculations :
  ((-20) + (-14) - (-18) - 13 = -29) ∧
  (-24 * ((-1/2) + (3/4) - (1/3)) = 2) ∧
  (-49 * (24/25) * 10 = -499.6) ∧
  (-(3^2) + (((-1/3) * (-3)) - ((8/5) / (2^2))) = -(8 + 2/5)) := by
  sorry

end arithmetic_calculations_l3637_363779


namespace triangle_ratio_greater_than_two_l3637_363794

/-- In a right triangle ABC with ∠BAC = 90°, AB = 5, BC = 6, and point K dividing AC in ratio 3:1 from A,
    the ratio BK/AH is greater than 2, where AH is the altitude from A to BC. -/
theorem triangle_ratio_greater_than_two (A B C K H : ℝ × ℝ) : 
  -- Triangle ABC is right-angled at A
  (A.1 = 0 ∧ A.2 = 0) → 
  (B.1 = 5 ∧ B.2 = 0) → 
  (C.1 = 0 ∧ C.2 = 6) → 
  -- K divides AC in ratio 3:1 from A
  (K.1 = (3/4) * C.1 ∧ K.2 = (3/4) * C.2) →
  -- H is the foot of the altitude from A to BC
  (H.1 = 0 ∧ H.2 = 30 / Real.sqrt 61) →
  -- The ratio BK/AH is greater than 2
  Real.sqrt ((K.1 - B.1)^2 + (K.2 - B.2)^2) / Real.sqrt (H.1^2 + H.2^2) > 2 := by
  sorry

end triangle_ratio_greater_than_two_l3637_363794


namespace fifteenth_entry_is_22_l3637_363761

/-- r_7(n) represents the remainder when n is divided by 7 -/
def r_7 (n : ℕ) : ℕ := n % 7

/-- The list of nonnegative integers n that satisfy r_7(3n) ≤ 3 -/
def satisfying_list : List ℕ :=
  (List.range (100 : ℕ)).filter (fun n => r_7 (3 * n) ≤ 3)

theorem fifteenth_entry_is_22 : satisfying_list[14] = 22 := by
  sorry

end fifteenth_entry_is_22_l3637_363761


namespace inequality_solution_l3637_363726

noncomputable def f (x : ℝ) : ℝ := x^4 + Real.exp (abs x)

theorem inequality_solution :
  let S := {t : ℝ | 2 * f (Real.log t) - f (Real.log (1 / t)) ≤ f 2}
  S = {t : ℝ | Real.exp (-2) ≤ t ∧ t ≤ Real.exp 2} :=
by sorry

end inequality_solution_l3637_363726


namespace decagon_perimeter_l3637_363797

/-- A decagon is a polygon with 10 sides -/
def Decagon := Nat

/-- The number of sides in a decagon -/
def decagon_sides : Nat := 10

/-- The length of each side of our specific decagon -/
def side_length : ℝ := 3

/-- The perimeter of a polygon is the sum of the lengths of all its sides -/
def perimeter (n : Nat) (s : ℝ) : ℝ := n * s

/-- Theorem: The perimeter of a decagon with sides of length 3 units is 30 units -/
theorem decagon_perimeter : perimeter decagon_sides side_length = 30 := by
  sorry

end decagon_perimeter_l3637_363797


namespace no_rational_cos_sqrt2_l3637_363780

theorem no_rational_cos_sqrt2 : ¬∃ (x : ℝ), (∃ (a b : ℚ), (Real.cos x + Real.sqrt 2 = a) ∧ (Real.cos (2 * x) + Real.sqrt 2 = b)) := by
  sorry

end no_rational_cos_sqrt2_l3637_363780


namespace equal_elements_from_inequalities_l3637_363752

theorem equal_elements_from_inequalities (a : Fin 100 → ℝ)
  (h : ∀ i : Fin 100, a i - 3 * a (i + 1) + 2 * a (i + 2) ≥ 0) :
  ∀ i j : Fin 100, a i = a j :=
sorry

end equal_elements_from_inequalities_l3637_363752


namespace f_properties_l3637_363758

-- Define the function f(x) = -x^3 + 12x
def f (x : ℝ) : ℝ := -x^3 + 12*x

-- Define the interval [-3, 1]
def interval : Set ℝ := Set.Icc (-3) 1

theorem f_properties :
  -- f(x) is decreasing on [-3, -2] and increasing on [-2, 1]
  (∀ x ∈ Set.Icc (-3) (-2), ∀ y ∈ Set.Icc (-3) (-2), x < y → f x > f y) ∧
  (∀ x ∈ Set.Ioo (-2) 1, ∀ y ∈ Set.Ioo (-2) 1, x < y → f x < f y) ∧
  -- The maximum value is 11
  (∃ x ∈ interval, f x = 11 ∧ ∀ y ∈ interval, f y ≤ 11) ∧
  -- The minimum value is -16
  (∃ x ∈ interval, f x = -16 ∧ ∀ y ∈ interval, f y ≥ -16) := by
  sorry


end f_properties_l3637_363758


namespace gain_percent_when_selling_price_twice_cost_price_l3637_363746

/-- If the selling price of an item is twice its cost price, then the gain percent is 100% -/
theorem gain_percent_when_selling_price_twice_cost_price 
  (cost : ℝ) (selling : ℝ) (h : selling = 2 * cost) : 
  (selling - cost) / cost * 100 = 100 :=
sorry

end gain_percent_when_selling_price_twice_cost_price_l3637_363746


namespace pet_shop_dogs_count_l3637_363790

/-- Represents the number of animals of each type in the pet shop -/
structure PetShop where
  dogs : ℕ
  cats : ℕ
  bunnies : ℕ

/-- The ratio of dogs to cats to bunnies -/
def ratio : Fin 3 → ℕ
| 0 => 3  -- dogs
| 1 => 7  -- cats
| 2 => 12 -- bunnies

theorem pet_shop_dogs_count (shop : PetShop) :
  (ratio 0 : ℚ) / shop.dogs = (ratio 1 : ℚ) / shop.cats ∧
  (ratio 0 : ℚ) / shop.dogs = (ratio 2 : ℚ) / shop.bunnies ∧
  shop.dogs + shop.bunnies = 375 →
  shop.dogs = 75 := by
sorry

end pet_shop_dogs_count_l3637_363790


namespace line_reflection_l3637_363703

-- Define the slope of the original line
def k : ℝ := sorry

-- Define the reflection line
def reflection_line (x y : ℝ) : Prop := x + y = 1

-- Define the original line
def original_line (x y : ℝ) : Prop := y = k * x

-- Define the resulting line after reflection
def reflected_line (x y : ℝ) : Prop := y = (1 / k) * x + (k - 1) / k

-- State the theorem
theorem line_reflection (h1 : k ≠ 0) (h2 : k ≠ -1) :
  ∀ x y : ℝ, reflected_line x y ↔ 
  ∃ x' y' : ℝ, original_line x' y' ∧ 
  reflection_line ((x + x') / 2) ((y + y') / 2) :=
sorry

end line_reflection_l3637_363703


namespace complex_number_simplification_l3637_363735

theorem complex_number_simplification :
  (7 - 3*Complex.I) - 3*(2 - 5*Complex.I) + 4*Complex.I = 1 + 16*Complex.I :=
by sorry

end complex_number_simplification_l3637_363735


namespace base_8_5214_equals_2700_l3637_363771

def base_8_to_10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (8 ^ (digits.length - 1 - i))) 0

theorem base_8_5214_equals_2700 :
  base_8_to_10 [5, 2, 1, 4] = 2700 := by
  sorry

end base_8_5214_equals_2700_l3637_363771


namespace final_position_is_correct_l3637_363798

/-- Movement pattern A: 1 unit up, 2 units right -/
def pattern_a : ℤ × ℤ := (2, 1)

/-- Movement pattern B: 3 units left, 2 units down -/
def pattern_b : ℤ × ℤ := (-3, -2)

/-- Calculate the position after n movements -/
def position_after_n_movements (n : ℕ) : ℤ × ℤ :=
  let a_count := (n + 1) / 2
  let b_count := n / 2
  (a_count * pattern_a.1 + b_count * pattern_b.1,
   a_count * pattern_a.2 + b_count * pattern_b.2)

/-- The final position after 15 movements -/
def final_position : ℤ × ℤ := position_after_n_movements 15

theorem final_position_is_correct : final_position = (-5, -6) := by
  sorry

end final_position_is_correct_l3637_363798


namespace village_population_l3637_363714

theorem village_population (P : ℕ) : 
  (P : ℝ) * 0.9 * 0.8 = 4554 → P = 6325 := by
  sorry

end village_population_l3637_363714


namespace cubic_quadratic_inequality_l3637_363755

theorem cubic_quadratic_inequality (x y z : ℝ) (hx : x ≥ 0) (hy : y ≥ 0) (hz : z ≥ 0) :
  2 * (x^3 + y^3 + z^3) ≥ x^2*y + x^2*z + y^2*z + y^2*x + z^2*x + z^2*y :=
by sorry

end cubic_quadratic_inequality_l3637_363755


namespace secretary_work_time_l3637_363701

theorem secretary_work_time (t1 t2 t3 : ℕ) : 
  t1 + t2 + t3 = 110 ∧ 
  t3 = 55 ∧ 
  2 * t2 = 3 * t1 ∧ 
  5 * t1 = 3 * t3 :=
by sorry

end secretary_work_time_l3637_363701


namespace max_value_on_ellipse_l3637_363750

def ellipse (b : ℝ) (x y : ℝ) : Prop := x^2/4 + y^2/b^2 = 1

theorem max_value_on_ellipse (b : ℝ) (h : b > 0) :
  (∃ (x y : ℝ), ellipse b x y ∧ 
    ∀ (x' y' : ℝ), ellipse b x' y' → x^2 + 2*y ≥ x'^2 + 2*y') ∧
  (∃ (max : ℝ), 
    (0 < b ∧ b ≤ 4 → max = b^2/4 + 4) ∧
    (b > 4 → max = 2*b) ∧
    ∀ (x y : ℝ), ellipse b x y → x^2 + 2*y ≤ max) :=
by sorry

end max_value_on_ellipse_l3637_363750


namespace five_digit_number_divisible_by_9_l3637_363720

theorem five_digit_number_divisible_by_9 (a b c d e : ℕ) : 
  (10000 * a + 1000 * b + 100 * c + 10 * d + e) % 9 = 0 →
  (100 * a + 10 * c + e) - (100 * b + 10 * d + a) = 760 →
  10000 ≤ (10000 * a + 1000 * b + 100 * c + 10 * d + e) →
  (10000 * a + 1000 * b + 100 * c + 10 * d + e) < 100000 →
  10000 * a + 1000 * b + 100 * c + 10 * d + e = 81828 := by
  sorry

end five_digit_number_divisible_by_9_l3637_363720


namespace negation_of_existence_negation_of_universal_negation_of_implication_l3637_363745

-- 1. Negation of existence
theorem negation_of_existence :
  (¬ ∃ x : ℤ, x^2 - 2*x - 3 = 0) ↔ (∀ x : ℤ, x^2 - 2*x - 3 ≠ 0) :=
by sorry

-- 2. Negation of universal quantification
theorem negation_of_universal :
  (¬ ∀ x : ℝ, x^2 + 3 ≥ 2*x) ↔ (∃ x : ℝ, x^2 + 3 < 2*x) :=
by sorry

-- 3. Negation of implication
theorem negation_of_implication :
  (¬ (∀ x y : ℝ, x > 1 ∧ y > 1 → x + y > 2)) ↔
  (∃ x y : ℝ, (x ≤ 1 ∨ y ≤ 1) ∧ x + y ≤ 2) :=
by sorry

end negation_of_existence_negation_of_universal_negation_of_implication_l3637_363745
