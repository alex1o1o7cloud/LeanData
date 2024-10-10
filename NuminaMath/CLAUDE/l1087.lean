import Mathlib

namespace semester_weeks_calculation_l1087_108774

/-- The number of weeks in a semester before midterms -/
def weeks_in_semester : ℕ := 6

/-- The number of hours Annie spends on extracurriculars per week -/
def hours_per_week : ℕ := 13

/-- The number of weeks Annie takes off sick -/
def sick_weeks : ℕ := 2

/-- The total number of hours Annie spends on extracurriculars before midterms -/
def total_hours : ℕ := 52

theorem semester_weeks_calculation :
  weeks_in_semester * hours_per_week - sick_weeks * hours_per_week = total_hours := by
  sorry

end semester_weeks_calculation_l1087_108774


namespace tara_road_trip_cost_l1087_108771

/-- Represents a gas station with a price per gallon -/
structure GasStation :=
  (price : ℚ)

/-- Calculates the total cost of gas for a road trip -/
def total_gas_cost (tank_capacity : ℚ) (stations : List GasStation) : ℚ :=
  stations.map (λ station => station.price * tank_capacity) |>.sum

/-- Theorem: The total cost of gas for Tara's road trip is $180 -/
theorem tara_road_trip_cost :
  let tank_capacity : ℚ := 12
  let stations : List GasStation := [
    { price := 3 },
    { price := 7/2 },
    { price := 4 },
    { price := 9/2 }
  ]
  total_gas_cost tank_capacity stations = 180 := by
  sorry

end tara_road_trip_cost_l1087_108771


namespace max_value_symmetric_function_l1087_108789

def f (a b x : ℝ) : ℝ := (1 + 2*x) * (x^2 + a*x + b)

theorem max_value_symmetric_function (a b : ℝ) :
  (∀ x : ℝ, f a b (1 - x) = f a b (1 + x)) →
  (∃ x₀ : ℝ, x₀ ∈ Set.Icc (-1 : ℝ) 1 ∧
    ∀ x : ℝ, x ∈ Set.Icc (-1 : ℝ) 1 → f a b x ≤ f a b x₀) →
  (∃ x₀ : ℝ, x₀ ∈ Set.Icc (-1 : ℝ) 1 ∧
    f a b x₀ = 3 * Real.sqrt 3 / 2 ∧
    ∀ x : ℝ, x ∈ Set.Icc (-1 : ℝ) 1 → f a b x ≤ f a b x₀) :=
by sorry

end max_value_symmetric_function_l1087_108789


namespace distance_between_three_points_l1087_108775

-- Define a line
structure Line where
  -- Add any necessary properties for a line

-- Define a point on a line
structure Point (l : Line) where
  -- Add any necessary properties for a point on a line

-- Define the distance between two points on a line
def distance (l : Line) (p q : Point l) : ℝ :=
  sorry

-- Theorem statement
theorem distance_between_three_points (l : Line) (A B C : Point l) :
  distance l A B = 5 ∧ distance l B C = 3 →
  distance l A C = 8 ∨ distance l A C = 2 :=
sorry

end distance_between_three_points_l1087_108775


namespace length_lost_per_knot_l1087_108726

/-- Given a set of ropes and the total length after tying, calculate the length lost per knot -/
theorem length_lost_per_knot (rope_lengths : List ℝ) (total_length_after_tying : ℝ) : 
  rope_lengths = [8, 20, 2, 2, 2, 7] ∧ 
  total_length_after_tying = 35 → 
  (rope_lengths.sum - total_length_after_tying) / (rope_lengths.length - 1) = 1.2 := by
  sorry

end length_lost_per_knot_l1087_108726


namespace sum_of_evens_1_to_101_l1087_108740

theorem sum_of_evens_1_to_101 : 
  (Finset.range 51).sum (fun i => 2 * i) = 2550 := by
  sorry

end sum_of_evens_1_to_101_l1087_108740


namespace least_integer_square_condition_l1087_108783

theorem least_integer_square_condition : ∃ x : ℤ, x^2 = 3*x + 12 ∧ ∀ y : ℤ, y^2 = 3*y + 12 → x ≤ y :=
by sorry

end least_integer_square_condition_l1087_108783


namespace playground_area_l1087_108713

theorem playground_area (perimeter width length : ℝ) (h1 : perimeter = 80) 
  (h2 : length = 3 * width) (h3 : perimeter = 2 * (length + width)) : 
  length * width = 300 := by
  sorry

end playground_area_l1087_108713


namespace cone_volume_from_cylinder_volume_l1087_108786

/-- The volume of a cone with the same radius and height as a cylinder with volume 81π cm³ is 27π cm³ -/
theorem cone_volume_from_cylinder_volume (r h : ℝ) (h1 : r > 0) (h2 : h > 0) :
  π * r^2 * h = 81 * π → (1/3) * π * r^2 * h = 27 * π := by
  sorry

end cone_volume_from_cylinder_volume_l1087_108786


namespace sum_of_reciprocal_roots_l1087_108735

theorem sum_of_reciprocal_roots (a b : ℝ) : 
  a^2 - 6*a + 4 = 0 → b^2 - 6*b + 4 = 0 → a ≠ b → (1/a + 1/b) = 3/2 := by
  sorry

end sum_of_reciprocal_roots_l1087_108735


namespace cubic_root_ratio_l1087_108725

theorem cubic_root_ratio (a b c d : ℝ) (h : ∀ x : ℝ, a * x^3 + b * x^2 + c * x + d = 0 ↔ x = 2 ∨ x = 4 ∨ x = 5) :
  c / d = 19 / 20 := by
sorry

end cubic_root_ratio_l1087_108725


namespace solution_set_inequality_l1087_108708

theorem solution_set_inequality (x : ℝ) : (3*x - 1) / (2 - x) ≥ 1 ↔ 3/4 ≤ x ∧ x < 2 := by
  sorry

end solution_set_inequality_l1087_108708


namespace sammys_homework_l1087_108718

theorem sammys_homework (total : ℕ) (completed : ℕ) (h1 : total = 9) (h2 : completed = 2) :
  total - completed = 7 := by sorry

end sammys_homework_l1087_108718


namespace division_problem_l1087_108727

theorem division_problem (n : ℕ) : 
  let first_part : ℕ := 19
  let second_part : ℕ := 36 - first_part
  n * first_part + 3 * second_part = 203 → n = 8 := by
sorry

end division_problem_l1087_108727


namespace carlotta_time_theorem_l1087_108766

theorem carlotta_time_theorem (n : ℝ) :
  let s : ℝ := 6
  let p : ℝ := 2 * n * s
  let t : ℝ := 3 * n * s + s
  let C : ℝ := p + t + s
  C = 30 * n + 12 := by sorry

end carlotta_time_theorem_l1087_108766


namespace sum_of_fractions_l1087_108743

theorem sum_of_fractions : 
  (1 / (2 * 3 * 4 : ℚ)) + (1 / (3 * 4 * 5 : ℚ)) + (1 / (4 * 5 * 6 : ℚ)) + 
  (1 / (5 * 6 * 7 : ℚ)) + (1 / (6 * 7 * 8 : ℚ)) = 3 / 16 := by
  sorry

end sum_of_fractions_l1087_108743


namespace tan_sum_simplification_l1087_108765

theorem tan_sum_simplification : 
  (1 + Real.tan (30 * π / 180)) * (1 + Real.tan (15 * π / 180)) = 2 := by
  sorry

end tan_sum_simplification_l1087_108765


namespace green_shirt_percentage_l1087_108716

theorem green_shirt_percentage 
  (total_students : ℕ) 
  (blue_percent : ℚ) 
  (red_percent : ℚ) 
  (other_count : ℕ) 
  (h1 : total_students = 900) 
  (h2 : blue_percent = 44/100) 
  (h3 : red_percent = 28/100) 
  (h4 : other_count = 162) :
  (total_students - (blue_percent * total_students + red_percent * total_students + other_count : ℚ)) / total_students = 1/10 := by
  sorry

end green_shirt_percentage_l1087_108716


namespace pass_through_walls_technique_l1087_108734

theorem pass_through_walls_technique (n : ℕ) :
  10 * Real.sqrt (10 / n) = Real.sqrt (10 * (10 / n)) ↔ n = 99 :=
sorry

end pass_through_walls_technique_l1087_108734


namespace not_p_sufficient_not_necessary_for_q_l1087_108702

-- Define the conditions p and q
def p (x : ℝ) : Prop := (1 - x) * (x + 3) < 0
def q (x : ℝ) : Prop := 5 * x - 6 ≤ x^2

-- Define not_p
def not_p (x : ℝ) : Prop := ¬(p x)

-- Theorem statement
theorem not_p_sufficient_not_necessary_for_q :
  (∀ x, not_p x → q x) ∧ 
  ¬(∀ x, q x → not_p x) :=
sorry

end not_p_sufficient_not_necessary_for_q_l1087_108702


namespace positive_integer_divisibility_l1087_108700

theorem positive_integer_divisibility (n : ℕ) :
  (n + 2009 ∣ n^2 + 2009) ∧ (n + 2010 ∣ n^2 + 2010) → n = 0 ∨ n = 1 :=
by sorry

end positive_integer_divisibility_l1087_108700


namespace trajectory_curve_intersection_range_l1087_108794

-- Define the points A and B
def A : ℝ × ℝ := (-1, 0)
def B : ℝ × ℝ := (1, 0)

-- Define the moving point M and its projection N on AB
def M : ℝ × ℝ → ℝ × ℝ := fun (x, y) ↦ (x, y)
def N : ℝ × ℝ → ℝ × ℝ := fun (x, y) ↦ (x, 0)

-- Define vectors
def vec_MN (m : ℝ × ℝ) : ℝ × ℝ := (0, -(m.2))
def vec_AN (n : ℝ × ℝ) : ℝ × ℝ := (n.1 + 1, 0)
def vec_BN (n : ℝ × ℝ) : ℝ × ℝ := (n.1 - 1, 0)

-- Define dot product
def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

-- Define the condition for point M
def condition (m : ℝ × ℝ) : Prop :=
  let n := N m
  (vec_MN m).1^2 + (vec_MN m).2^2 = dot_product (vec_AN n) (vec_BN n)

-- Define the trajectory curve E
def curve_E (p : ℝ × ℝ) : Prop := p.1^2 - p.2^2 = 1

-- Define the line l
def line_l (k : ℝ) (p : ℝ × ℝ) : Prop := p.2 = k * p.1 - 1

-- Theorem statements
theorem trajectory_curve : ∀ m : ℝ × ℝ, condition m ↔ curve_E m := by sorry

theorem intersection_range : ∀ k : ℝ,
  (∃ p : ℝ × ℝ, curve_E p ∧ line_l k p) ↔ -Real.sqrt 2 ≤ k ∧ k ≤ Real.sqrt 2 := by sorry

end trajectory_curve_intersection_range_l1087_108794


namespace range_of_a_l1087_108756

def P (a : ℝ) := {x : ℝ | a - 4 < x ∧ x < a + 4}

def Q := {x : ℝ | x^2 - 4*x + 3 < 0}

theorem range_of_a :
  (∀ x, x ∈ Q → x ∈ P a) ↔ -1 ≤ a ∧ a ≤ 5 := by sorry

end range_of_a_l1087_108756


namespace abs_sum_zero_implies_diff_l1087_108762

theorem abs_sum_zero_implies_diff (a b : ℝ) : 
  |a - 2| + |b + 3| = 0 → a - b = 5 := by
sorry

end abs_sum_zero_implies_diff_l1087_108762


namespace quadratic_root_value_l1087_108711

theorem quadratic_root_value (m : ℝ) : 
  m^2 + m - 1 = 0 → 2*m^2 + 2*m + 2025 = 2027 := by
  sorry

end quadratic_root_value_l1087_108711


namespace xiao_ming_score_l1087_108784

/-- Calculates the comprehensive score based on individual scores and weights -/
def comprehensive_score (written_score practical_score publicity_score : ℝ)
  (written_weight practical_weight publicity_weight : ℝ) : ℝ :=
  written_score * written_weight + practical_score * practical_weight + publicity_score * publicity_weight

/-- Theorem stating that Xiao Ming's comprehensive score is 97 -/
theorem xiao_ming_score :
  let written_score : ℝ := 96
  let practical_score : ℝ := 98
  let publicity_score : ℝ := 96
  let written_weight : ℝ := 0.30
  let practical_weight : ℝ := 0.50
  let publicity_weight : ℝ := 0.20
  comprehensive_score written_score practical_score publicity_score
    written_weight practical_weight publicity_weight = 97 :=
by sorry


end xiao_ming_score_l1087_108784


namespace chocolates_not_in_box_l1087_108705

theorem chocolates_not_in_box (initial_chocolates : ℕ) (initial_boxes : ℕ) 
  (additional_chocolates : ℕ) (additional_boxes : ℕ) :
  initial_chocolates = 50 →
  initial_boxes = 3 →
  additional_chocolates = 25 →
  additional_boxes = 2 →
  ∃ (chocolates_per_box : ℕ),
    chocolates_per_box * (initial_boxes + additional_boxes) = initial_chocolates + additional_chocolates →
    initial_chocolates - (chocolates_per_box * initial_boxes) = 5 := by
  sorry

end chocolates_not_in_box_l1087_108705


namespace expand_polynomial_l1087_108796

theorem expand_polynomial (x : ℝ) : (1 + x^3) * (1 - x^4 + x^5) = 1 + x^3 - x^4 + x^5 - x^7 + x^8 := by
  sorry

end expand_polynomial_l1087_108796


namespace calculation_proof_l1087_108701

theorem calculation_proof : ((5 + 7 + 3) * 2 - 4) / 2 - 5 / 2 = 21 / 2 := by
  sorry

end calculation_proof_l1087_108701


namespace least_product_of_distinct_primes_l1087_108785

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

def distinct_primes_product (p q : ℕ) : Prop :=
  is_prime p ∧ is_prime q ∧ p ≠ q ∧ 50 < p ∧ p < 100 ∧ 50 < q ∧ q < 100

theorem least_product_of_distinct_primes :
  ∃ p q : ℕ, distinct_primes_product p q ∧
    p * q = 3127 ∧
    ∀ r s : ℕ, distinct_primes_product r s → p * q ≤ r * s :=
sorry

end least_product_of_distinct_primes_l1087_108785


namespace small_cuboid_height_l1087_108763

/-- Represents the dimensions of a cuboid -/
structure CuboidDimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the volume of a cuboid given its dimensions -/
def cuboidVolume (d : CuboidDimensions) : ℝ :=
  d.length * d.width * d.height

theorem small_cuboid_height
  (large : CuboidDimensions)
  (small_length : ℝ)
  (small_width : ℝ)
  (num_small_cuboids : ℕ)
  (h_large : large = { length := 16, width := 10, height := 12 })
  (h_small_length : small_length = 5)
  (h_small_width : small_width = 4)
  (h_num_small : num_small_cuboids = 32) :
  ∃ (small_height : ℝ),
    cuboidVolume large = num_small_cuboids * (small_length * small_width * small_height) ∧
    small_height = 3 := by
  sorry


end small_cuboid_height_l1087_108763


namespace sufficient_material_l1087_108710

-- Define the surface area of a rectangular box
def surface_area (l w h : ℝ) : ℝ := 2 * (l * w + l * h + w * h)

-- Define the volume of a rectangular box
def volume (l w h : ℝ) : ℝ := l * w * h

-- Theorem statement
theorem sufficient_material :
  ∃ (l w h : ℝ), l > 0 ∧ w > 0 ∧ h > 0 ∧ 
  surface_area l w h = 958 ∧ 
  volume l w h ≥ 1995 := by
  sorry

end sufficient_material_l1087_108710


namespace parabola_point_coordinates_l1087_108715

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define the focus of the parabola
def focus : ℝ × ℝ := (1, 0)

-- Define a point on the parabola
def point_on_parabola (A : ℝ × ℝ) : Prop :=
  parabola A.1 A.2

-- Define the dot product condition
def dot_product_condition (A : ℝ × ℝ) : Prop :=
  let O := (0, 0)
  let F := focus
  (A.1 - O.1) * (F.1 - A.1) + (A.2 - O.2) * (F.2 - A.2) = -4

-- Theorem statement
theorem parabola_point_coordinates :
  ∀ A : ℝ × ℝ,
  point_on_parabola A →
  dot_product_condition A →
  (A = (1, 2) ∨ A = (1, -2)) :=
sorry

end parabola_point_coordinates_l1087_108715


namespace boys_in_class_l1087_108724

theorem boys_in_class (total : ℕ) (girls_ratio : ℕ) (boys_ratio : ℕ) (boys : ℕ) : 
  total = 56 → 
  girls_ratio = 4 →
  boys_ratio = 3 →
  girls_ratio + boys_ratio = 7 →
  (girls_ratio : ℚ) / (boys_ratio : ℚ) = 4 / 3 →
  boys = (boys_ratio : ℚ) / (girls_ratio + boys_ratio : ℚ) * total →
  boys = 24 := by
sorry

end boys_in_class_l1087_108724


namespace power_of_product_l1087_108751

theorem power_of_product (x : ℝ) : (2 * x)^3 = 8 * x^3 := by
  sorry

end power_of_product_l1087_108751


namespace bicycle_spokes_front_wheel_l1087_108777

/-- Proves that a bicycle with 60 total spokes and twice as many spokes on the back wheel as on the front wheel has 20 spokes on the front wheel. -/
theorem bicycle_spokes_front_wheel : 
  ∀ (front back : ℕ), 
  front + back = 60 → 
  back = 2 * front → 
  front = 20 := by
sorry

end bicycle_spokes_front_wheel_l1087_108777


namespace three_times_a_plus_square_of_b_l1087_108723

/-- The algebraic expression "three times a plus the square of b" is equivalent to 3a + b² -/
theorem three_times_a_plus_square_of_b (a b : ℝ) : 3 * a + b^2 = 3 * a + b^2 := by
  sorry

end three_times_a_plus_square_of_b_l1087_108723


namespace x_equals_2_valid_l1087_108769

/-- Represents an assignment statement -/
inductive AssignmentStatement
| constant : ℕ → AssignmentStatement
| variable : String → ℕ → AssignmentStatement
| consecutive : String → String → ℕ → AssignmentStatement
| expression : String → String → ℕ → AssignmentStatement

/-- Checks if an assignment statement is valid -/
def isValidAssignment (stmt : AssignmentStatement) : Prop :=
  match stmt with
  | AssignmentStatement.variable _ _ => True
  | _ => False

theorem x_equals_2_valid :
  isValidAssignment (AssignmentStatement.variable "x" 2) = True :=
by sorry

end x_equals_2_valid_l1087_108769


namespace sequence_property_l1087_108792

def sequence_sum (a : ℕ+ → ℚ) (n : ℕ+) : ℚ :=
  (Finset.range n).sum (fun i => a ⟨i + 1, Nat.succ_pos i⟩)

theorem sequence_property (a : ℕ+ → ℚ) 
    (h : ∀ n : ℕ+, a n + sequence_sum a n = 2 * n.val + 1) :
  (∀ n : ℕ+, a n = 2 - 1 / (2 ^ n.val)) ∧
  (∀ n : ℕ+, (Finset.range n).sum (fun i => 1 / (2 ^ (i + 1) * a ⟨i + 1, Nat.succ_pos i⟩ * a ⟨i + 2, Nat.succ_pos (i + 1)⟩)) < 1 / 3) :=
by sorry

end sequence_property_l1087_108792


namespace greatest_q_minus_r_l1087_108754

theorem greatest_q_minus_r : ∃ (q r : ℕ+), 
  (1025 = 23 * q + r) ∧ 
  (∀ (q' r' : ℕ+), 1025 = 23 * q' + r' → q' - r' ≤ q - r) ∧
  (q - r = 27) := by
sorry

end greatest_q_minus_r_l1087_108754


namespace litter_patrol_problem_l1087_108788

/-- The Litter Patrol problem -/
theorem litter_patrol_problem (total_litter aluminum_cans : ℕ) 
  (h1 : total_litter = 18)
  (h2 : aluminum_cans = 8)
  (h3 : total_litter = aluminum_cans + glass_bottles) :
  glass_bottles = 10 := by
  sorry

end litter_patrol_problem_l1087_108788


namespace fourth_rectangle_area_l1087_108722

/-- Represents a rectangle divided into four smaller rectangles -/
structure DividedRectangle where
  area1 : ℝ
  area2 : ℝ
  area3 : ℝ
  area4 : ℝ

/-- The theorem stating that if three of the areas are 24, 15, and 9, then the fourth is 15 -/
theorem fourth_rectangle_area (rect : DividedRectangle) 
  (h1 : rect.area1 = 24)
  (h2 : rect.area2 = 15)
  (h3 : rect.area3 = 9) :
  rect.area4 = 15 := by
  sorry

end fourth_rectangle_area_l1087_108722


namespace last_four_digits_of_5_pow_2019_l1087_108752

theorem last_four_digits_of_5_pow_2019 (h5 : 5^5 % 10000 = 3125)
                                       (h6 : 5^6 % 10000 = 5625)
                                       (h7 : 5^7 % 10000 = 8125)
                                       (h8 : 5^8 % 10000 = 0625) :
  5^2019 % 10000 = 8125 := by
  sorry

end last_four_digits_of_5_pow_2019_l1087_108752


namespace max_value_reciprocal_sum_l1087_108787

theorem max_value_reciprocal_sum (x y a b : ℝ) 
  (ha : a > 1) (hb : b > 1) 
  (hx : a^x = 3) (hy : b^y = 3) 
  (hab : a + b = 2 * Real.sqrt 3) : 
  (∀ x' y' : ℝ, a^x' = 3 → b^y' = 3 → 1/x' + 1/y' ≤ 1) ∧ 
  (∃ x'' y'' : ℝ, a^x'' = 3 ∧ b^y'' = 3 ∧ 1/x'' + 1/y'' = 1) :=
sorry

end max_value_reciprocal_sum_l1087_108787


namespace repeating_decimal_to_fraction_l1087_108795

theorem repeating_decimal_to_fraction : 
  ∃ (x : ℚ), x = 57 / 99 ∧ (∀ n : ℕ, (x * 10^(2*n+2) - ⌊x * 10^(2*n+2)⌋ : ℚ) = 57 / 100) := by
  sorry

end repeating_decimal_to_fraction_l1087_108795


namespace lattice_point_theorem_l1087_108746

/-- A lattice point in 2D space -/
structure LatticePoint where
  x : ℤ
  y : ℤ

/-- The set of all lattice points -/
def L : Set LatticePoint := Set.univ

/-- Check if a line segment between two lattice points contains no other lattice points -/
def noInteriorLatticePoints (a b : LatticePoint) : Prop :=
  ∀ p : LatticePoint, p ∈ L → p ≠ a → p ≠ b → ¬(∃ t : ℚ, 0 < t ∧ t < 1 ∧ 
    p.x = a.x + t * (b.x - a.x) ∧ p.y = a.y + t * (b.y - a.y))

theorem lattice_point_theorem :
  (∀ a b c : LatticePoint, a ∈ L → b ∈ L → c ∈ L → a ≠ b → b ≠ c → a ≠ c →
    ∃ d : LatticePoint, d ∈ L ∧ d ≠ a ∧ d ≠ b ∧ d ≠ c ∧
      noInteriorLatticePoints a d ∧ noInteriorLatticePoints b d ∧ noInteriorLatticePoints c d) ∧
  (∃ a b c d : LatticePoint, a ∈ L ∧ b ∈ L ∧ c ∈ L ∧ d ∈ L ∧
    a ≠ b ∧ b ≠ c ∧ c ≠ d ∧ d ≠ a ∧
    (¬noInteriorLatticePoints a b ∨ ¬noInteriorLatticePoints b c ∨
     ¬noInteriorLatticePoints c d ∨ ¬noInteriorLatticePoints d a)) :=
by sorry

end lattice_point_theorem_l1087_108746


namespace fixed_point_exponential_function_l1087_108768

theorem fixed_point_exponential_function (a : ℝ) (ha : a > 0) (ha' : a ≠ 1) :
  let f : ℝ → ℝ := λ x ↦ a^(x - 1) + 1
  f 1 = 2 := by
  sorry

end fixed_point_exponential_function_l1087_108768


namespace sum_remainder_l1087_108760

theorem sum_remainder (m : ℤ) : (10 - 3*m + (5*m + 6)) % 8 = (2*m) % 8 := by
  sorry

end sum_remainder_l1087_108760


namespace x_is_irrational_l1087_108707

/-- Representation of the digits of 1987^k -/
def digits (k : ℕ) : List ℕ :=
  sorry

/-- Construct the number x as described in the problem -/
def x : ℝ :=
  sorry

/-- Theorem stating that x is irrational -/
theorem x_is_irrational : Irrational x := by
  sorry

end x_is_irrational_l1087_108707


namespace expression_not_prime_l1087_108767

def expression (x y : ℕ) : ℕ :=
  x^8 - x^7*y + x^6*y^2 - x^5*y^3 + x^4*y^4 - x^3*y^5 + x^2*y^6 - x*y^7 + y^8

theorem expression_not_prime (x y : ℕ) :
  ¬(Nat.Prime (expression x y)) :=
sorry

end expression_not_prime_l1087_108767


namespace benny_eggs_dozens_l1087_108782

/-- The number of eggs in a dozen -/
def eggs_per_dozen : ℕ := 12

/-- The total number of eggs Benny bought -/
def total_eggs : ℕ := 84

/-- The number of dozens of eggs Benny bought -/
def dozens_bought : ℕ := total_eggs / eggs_per_dozen

theorem benny_eggs_dozens : dozens_bought = 7 := by
  sorry

end benny_eggs_dozens_l1087_108782


namespace ratio_problem_l1087_108728

theorem ratio_problem (a b c d : ℚ) 
  (hab : a / b = 5 / 4)
  (hcd : c / d = 4 / 3)
  (hdb : d / b = 1 / 5) :
  a / c = 75 / 16 := by
sorry

end ratio_problem_l1087_108728


namespace system_solution_l1087_108732

theorem system_solution (x y : ℝ) : 
  (x / y + y / x = 173 / 26 ∧ 1 / x + 1 / y = 15 / 26) → 
  ((x = 13 ∧ y = 2) ∨ (x = 2 ∧ y = 13)) := by
sorry

end system_solution_l1087_108732


namespace part_one_part_two_l1087_108791

-- Define the function f
def f (x : ℝ) (a : ℝ) : ℝ := |x + 1| - a * |x - 1|

-- Part I
theorem part_one : 
  ∀ x : ℝ, (|x + 1| + 2 * |x - 1| > 5) ↔ (x < -4/3 ∨ x > 2) :=
sorry

-- Part II
theorem part_two :
  (∃ a : ℝ, ∀ x : ℝ, f x a ≤ a * |x + 3|) ∧
  (∀ b : ℝ, (∀ x : ℝ, f x b ≤ b * |x + 3|) → b ≥ 1/2) :=
sorry

end part_one_part_two_l1087_108791


namespace complex_fraction_simplification_l1087_108779

/-- Given that i² = -1, prove that (1 - i) / (2 + 3i) = -1/13 - 5/13 * i -/
theorem complex_fraction_simplification (i : ℂ) (h : i^2 = -1) :
  (1 - i) / (2 + 3*i) = -1/13 - 5/13 * i :=
by sorry

end complex_fraction_simplification_l1087_108779


namespace fourth_degree_polynomial_roots_l1087_108730

theorem fourth_degree_polynomial_roots :
  let p : ℝ → ℝ := λ x => 3*x^4 - 19*x^3 + 34*x^2 - 19*x + 3
  (∀ x : ℝ, p x = 0 ↔ x = 2 + Real.sqrt 3 ∨ 
                      x = 2 - Real.sqrt 3 ∨ 
                      x = (7 + Real.sqrt 13) / 6 ∨ 
                      x = (7 - Real.sqrt 13) / 6) :=
by sorry

end fourth_degree_polynomial_roots_l1087_108730


namespace quadratic_root_form_l1087_108717

/-- The quadratic equation 2x^2 - 5x - 4 = 0 -/
def quadratic_equation (x : ℝ) : Prop := 2 * x^2 - 5 * x - 4 = 0

/-- The roots of the equation in the form (m ± √n) / p -/
def root_form (m n p : ℕ) (x : ℝ) : Prop :=
  ∃ (sign : Bool), x = (m + if sign then 1 else -1 * Real.sqrt n) / p

/-- m, n, and p are coprime -/
def coprime (m n p : ℕ) : Prop := Nat.gcd m (Nat.gcd n p) = 1

theorem quadratic_root_form :
  ∃ (m n p : ℕ), 
    (∀ x : ℝ, quadratic_equation x → root_form m n p x) ∧
    coprime m n p ∧
    n = 57 := by sorry

end quadratic_root_form_l1087_108717


namespace polynomial_division_remainder_l1087_108721

theorem polynomial_division_remainder : ∃ (q r : Polynomial ℝ),
  x^4 + 4*x^2 + 20*x + 1 = (x^2 - 2*x + 7) * q + r ∧
  r.degree < (x^2 - 2*x + 7).degree ∧
  r = 8*x - 6 := by
  sorry

end polynomial_division_remainder_l1087_108721


namespace fraction_sum_equality_l1087_108733

theorem fraction_sum_equality : (2 / 10 : ℚ) + (7 / 100 : ℚ) + (3 / 1000 : ℚ) + (8 / 10000 : ℚ) = 2738 / 10000 := by
  sorry

end fraction_sum_equality_l1087_108733


namespace min_value_sqrt_and_reciprocal_l1087_108736

theorem min_value_sqrt_and_reciprocal (x : ℝ) (h : x > 0) :
  4 * Real.sqrt x + 4 / x ≥ 8 ∧ ∃ y > 0, 4 * Real.sqrt y + 4 / y = 8 :=
by sorry

end min_value_sqrt_and_reciprocal_l1087_108736


namespace simplify_expression_l1087_108744

theorem simplify_expression (x : ℝ) : (5 - 4 * x) - (2 + 5 * x) = 3 - 9 * x := by
  sorry

end simplify_expression_l1087_108744


namespace pairball_playing_time_l1087_108750

theorem pairball_playing_time (num_children : ℕ) (total_time : ℕ) (pair_size : ℕ) : 
  num_children = 6 →
  pair_size = 2 →
  total_time = 120 →
  (total_time * pair_size) / num_children = 40 := by
sorry

end pairball_playing_time_l1087_108750


namespace tan_alpha_value_l1087_108719

theorem tan_alpha_value (α : ℝ) (h : (Real.sin α - 2 * Real.cos α) / (2 * Real.sin α + 3 * Real.cos α) = 2) :
  Real.tan α = -8/3 := by sorry

end tan_alpha_value_l1087_108719


namespace cube_split_l1087_108703

theorem cube_split (m : ℕ) (h1 : m > 1) : 
  (∃ k : ℕ, k ≥ 0 ∧ k < m ∧ m^2 - m + 1 + 2*k = 73) → m = 9 := by
  sorry

end cube_split_l1087_108703


namespace train_crossing_cars_l1087_108799

/-- Represents the properties of a train passing through a crossing -/
structure TrainCrossing where
  cars_in_sample : ℕ
  sample_time : ℕ
  total_time : ℕ

/-- Calculates the number of cars in the train, rounded to the nearest multiple of 10 -/
def cars_in_train (tc : TrainCrossing) : ℕ :=
  let rate := tc.cars_in_sample / tc.sample_time
  let total_cars := rate * tc.total_time
  ((total_cars + 5) / 10) * 10

/-- Theorem stating that for the given train crossing scenario, the number of cars is 120 -/
theorem train_crossing_cars :
  let tc : TrainCrossing := { cars_in_sample := 9, sample_time := 15, total_time := 210 }
  cars_in_train tc = 120 := by
  sorry

end train_crossing_cars_l1087_108799


namespace multiply_add_equality_l1087_108737

theorem multiply_add_equality : (-3) * 2 + 4 = -2 := by
  sorry

end multiply_add_equality_l1087_108737


namespace two_digit_number_difference_l1087_108714

theorem two_digit_number_difference (x y : ℕ) : 
  x < 10 ∧ y < 10 ∧ y = 2 * x ∧ (10 * x + y) - (x + y) = 8 → 
  (10 * y + x) - (10 * x + y) = 9 := by
sorry

end two_digit_number_difference_l1087_108714


namespace f_value_at_2_l1087_108747

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

def f_neg (x : ℝ) : ℝ := -x^2 + x

theorem f_value_at_2 (f : ℝ → ℝ) 
    (h_odd : is_odd_function f)
    (h_neg : ∀ x < 0, f x = f_neg x) : 
  f 2 = 6 := by
sorry

end f_value_at_2_l1087_108747


namespace sector_central_angle_l1087_108761

/-- Given a circular sector with circumference 6 and area 2, prove that its central angle in radians is either 1 or 4. -/
theorem sector_central_angle (r l : ℝ) (h1 : 2 * r + l = 6) (h2 : (1 / 2) * l * r = 2) :
  let α := l / r
  α = 1 ∨ α = 4 := by sorry

end sector_central_angle_l1087_108761


namespace smallest_perfect_square_divisible_by_2_3_5_l1087_108759

theorem smallest_perfect_square_divisible_by_2_3_5 :
  ∀ n : ℕ, n > 0 → n^2 % 2 = 0 → n^2 % 3 = 0 → n^2 % 5 = 0 → n^2 ≥ 900 :=
by
  sorry

end smallest_perfect_square_divisible_by_2_3_5_l1087_108759


namespace expression_evaluation_l1087_108712

theorem expression_evaluation :
  (3^2 - 3*2) - (4^2 - 4*2) + (5^2 - 5*2) - (6^2 - 6*2) = -14 := by
  sorry

end expression_evaluation_l1087_108712


namespace eliza_dress_ironing_time_l1087_108704

/-- Represents the time in minutes it takes Eliza to iron a dress -/
def dress_ironing_time : ℕ := sorry

/-- Represents the time in minutes it takes Eliza to iron a blouse -/
def blouse_ironing_time : ℕ := 15

/-- Represents the total time in minutes Eliza spends ironing blouses -/
def total_blouse_ironing_time : ℕ := 2 * 60

/-- Represents the total time in minutes Eliza spends ironing dresses -/
def total_dress_ironing_time : ℕ := 3 * 60

/-- Represents the total number of clothes Eliza ironed -/
def total_clothes : ℕ := 17

theorem eliza_dress_ironing_time :
  (total_blouse_ironing_time / blouse_ironing_time) +
  (total_dress_ironing_time / dress_ironing_time) = total_clothes →
  dress_ironing_time = 20 := by sorry

end eliza_dress_ironing_time_l1087_108704


namespace min_value_reciprocal_sum_l1087_108738

/-- The minimum value of 1/m + 1/n given the conditions -/
theorem min_value_reciprocal_sum (a m n : ℝ) (ha : a > 0) (ha_neq : a ≠ 1)
  (h_line : 2*m + 2*n = 1) (h_positive : m*n > 0) :
  1/m + 1/n ≥ 8 ∧ ∃ (m n : ℝ), 2*m + 2*n = 1 ∧ m*n > 0 ∧ 1/m + 1/n = 8 :=
by sorry

end min_value_reciprocal_sum_l1087_108738


namespace positive_numbers_inequality_l1087_108793

theorem positive_numbers_inequality (a b c : ℝ) 
  (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0)
  (h_eq : 4 * a^2 + b^2 + 16 * c^2 = 1) : 
  (0 < a * b ∧ a * b < 1/4) ∧ 
  (1/a^2 + 1/b^2 + 1/(4*a*b*c^2) > 49) := by
  sorry

end positive_numbers_inequality_l1087_108793


namespace arithmetic_mean_of_integers_from_neg3_to_6_l1087_108739

def integers_range : List ℤ := List.range 10 |>.map (λ i => i - 3)

theorem arithmetic_mean_of_integers_from_neg3_to_6 :
  (integers_range.sum : ℚ) / integers_range.length = 3/2 := by
  sorry

end arithmetic_mean_of_integers_from_neg3_to_6_l1087_108739


namespace sum_of_divisors_1184_l1087_108749

def sum_of_divisors (n : ℕ) : ℕ := (Finset.filter (· ∣ n) (Finset.range (n + 1))).sum id

theorem sum_of_divisors_1184 : sum_of_divisors 1184 = 2394 := by
  sorry

end sum_of_divisors_1184_l1087_108749


namespace inequality_solution_l1087_108772

theorem inequality_solution (a : ℝ) :
  (a > 0 → (∀ x : ℝ, 6 * x^2 + a * x - a^2 < 0 ↔ -a/2 < x ∧ x < a/3)) ∧
  (a = 0 → ¬ ∃ x : ℝ, 6 * x^2 + a * x - a^2 < 0) ∧
  (a < 0 → (∀ x : ℝ, 6 * x^2 + a * x - a^2 < 0 ↔ a/3 < x ∧ x < -a/2)) :=
by sorry

end inequality_solution_l1087_108772


namespace apollonius_circle_l1087_108773

/-- The locus of points with a fixed distance ratio from two given points is a circle -/
theorem apollonius_circle (A B : ℝ × ℝ) (m n : ℝ) (h_pos : m > 0 ∧ n > 0) :
  ∃ (C : ℝ × ℝ) (r : ℝ), ∀ M : ℝ × ℝ,
    (dist M A) / (dist M B) = m / n ↔ dist M C = r :=
  sorry

end apollonius_circle_l1087_108773


namespace not_always_geometric_b_l1087_108798

/-- A sequence is geometric if there exists a common ratio q such that a(n+1) = q * a(n) for all n -/
def IsGeometric (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = q * a n

/-- Definition of the sequence b_n in terms of a_n -/
def B (a : ℕ → ℝ) (n : ℕ) : ℝ := a (2*n - 1) + a (2*n)

theorem not_always_geometric_b (a : ℕ → ℝ) :
  IsGeometric a → ¬ (∀ a : ℕ → ℝ, IsGeometric a → IsGeometric (B a)) :=
by
  sorry

end not_always_geometric_b_l1087_108798


namespace triangle_perimeter_increase_l1087_108770

/-- The growth factor between consecutive triangles -/
def growthFactor : ℝ := 1.2

/-- The number of triangles -/
def numTriangles : ℕ := 5

/-- Calculates the percent increase between the first and last triangle -/
def percentIncrease : ℝ := (growthFactor ^ (numTriangles - 1) - 1) * 100

theorem triangle_perimeter_increase :
  ∃ ε > 0, ε < 0.1 ∧ |percentIncrease - 107.4| < ε :=
sorry

end triangle_perimeter_increase_l1087_108770


namespace charles_chocolate_syrup_l1087_108781

/-- The amount of chocolate milk in each glass (in ounces) -/
def glass_size : ℝ := 8

/-- The amount of milk in each glass (in ounces) -/
def milk_per_glass : ℝ := 6.5

/-- The amount of chocolate syrup in each glass (in ounces) -/
def syrup_per_glass : ℝ := 1.5

/-- The total amount of milk Charles has (in ounces) -/
def total_milk : ℝ := 130

/-- The total amount of chocolate milk Charles will drink (in ounces) -/
def total_drink : ℝ := 160

/-- The theorem stating the amount of chocolate syrup Charles has -/
theorem charles_chocolate_syrup : 
  ∃ (syrup : ℝ), 
    (total_drink / glass_size) * syrup_per_glass = syrup ∧ 
    syrup = 30 := by sorry

end charles_chocolate_syrup_l1087_108781


namespace second_month_sale_l1087_108731

/-- Given the sales of a grocer for four months, prove that the sale in the second month is 4000 --/
theorem second_month_sale
  (sale1 : ℕ)
  (sale3 : ℕ)
  (sale4 : ℕ)
  (average : ℕ)
  (h1 : sale1 = 2500)
  (h2 : sale3 = 3540)
  (h3 : sale4 = 1520)
  (h4 : average = 2890)
  (h5 : (sale1 + sale3 + sale4 + (4 * average - sale1 - sale3 - sale4)) / 4 = average) :
  4 * average - sale1 - sale3 - sale4 = 4000 := by
sorry

end second_month_sale_l1087_108731


namespace ivanov_petrov_probability_l1087_108778

/-- The number of people in the group -/
def n : ℕ := 11

/-- The number of people that should be between Ivanov and Petrov -/
def k : ℕ := 3

/-- The probability of exactly k people sitting between two specific people
    in a random circular arrangement of n people -/
def probability (n k : ℕ) : ℚ :=
  if n > k + 1 then 1 / (n - 1) else 0

theorem ivanov_petrov_probability :
  probability n k = 1 / 10 := by sorry

end ivanov_petrov_probability_l1087_108778


namespace rectangle_longest_side_l1087_108709

theorem rectangle_longest_side (l w : ℝ) : 
  l > 0 ∧ w > 0 ∧                 -- positive dimensions
  2 * (l + w) = 180 ∧             -- perimeter is 180 feet
  l * w = 8 * 180 →               -- area is 8 times perimeter
  max l w = 72 := by sorry

end rectangle_longest_side_l1087_108709


namespace goals_tied_in_june_l1087_108780

def ronaldo_goals : List Nat := [2, 9, 14, 8, 7, 11, 12]
def messi_goals : List Nat := [5, 8, 18, 6, 10, 9, 9]

def cumulative_sum (xs : List Nat) : List Nat :=
  List.scanl (·+·) 0 xs

def first_equal_index (xs ys : List Nat) : Option Nat :=
  (List.zip xs ys).findIdx (fun (x, y) => x = y)

def months : List String := ["January", "February", "March", "April", "May", "June", "July"]

theorem goals_tied_in_june :
  first_equal_index (cumulative_sum ronaldo_goals) (cumulative_sum messi_goals) = some 5 :=
by sorry

end goals_tied_in_june_l1087_108780


namespace divisibility_of_fourth_power_sum_l1087_108720

theorem divisibility_of_fourth_power_sum (a b c n : ℤ) 
  (h1 : n ∣ (a + b + c)) 
  (h2 : n ∣ (a^2 + b^2 + c^2)) : 
  n ∣ (a^4 + b^4 + c^4) := by
sorry

end divisibility_of_fourth_power_sum_l1087_108720


namespace wood_gathering_proof_l1087_108758

/-- The number of pieces of wood that can be contained in one sack -/
def pieces_per_sack : ℕ := 20

/-- The number of sacks filled -/
def num_sacks : ℕ := 4

/-- The total number of pieces of wood gathered -/
def total_pieces : ℕ := pieces_per_sack * num_sacks

theorem wood_gathering_proof :
  total_pieces = 80 :=
by sorry

end wood_gathering_proof_l1087_108758


namespace min_lines_is_seven_l1087_108790

/-- A line in a 2D Cartesian coordinate system -/
structure Line where
  k : ℝ
  b : ℝ
  k_nonzero : k ≠ 0

/-- The set of quadrants a line passes through -/
def quadrants (l : Line) : Set (Fin 4) :=
  sorry

/-- The minimum number of lines needed to guarantee two lines pass through the same quadrants -/
def min_lines_same_quadrants : ℕ :=
  sorry

/-- Theorem stating that the minimum number of lines is 7 -/
theorem min_lines_is_seven : min_lines_same_quadrants = 7 := by
  sorry

end min_lines_is_seven_l1087_108790


namespace square_field_area_l1087_108741

theorem square_field_area (wire_length : ℝ) (wire_turns : ℕ) (field_side : ℝ) : 
  wire_length = (4 * field_side * wire_turns) → 
  wire_length = 15840 → 
  wire_turns = 15 → 
  field_side * field_side = 69696 := by
sorry

end square_field_area_l1087_108741


namespace min_students_with_blue_shirt_and_red_shoes_l1087_108706

theorem min_students_with_blue_shirt_and_red_shoes
  (n : ℕ)  -- Total number of students
  (blue_shirt : ℕ)  -- Number of students wearing blue shirts
  (red_shoes : ℕ)  -- Number of students wearing red shoes
  (h1 : blue_shirt = n * 3 / 7)  -- 3/7 of students wear blue shirts
  (h2 : red_shoes = n * 4 / 9)  -- 4/9 of students wear red shoes
  : ∃ (both : ℕ), both ≥ 8 ∧ blue_shirt + red_shoes - both = n :=
sorry

end min_students_with_blue_shirt_and_red_shoes_l1087_108706


namespace hdtv_horizontal_length_l1087_108729

theorem hdtv_horizontal_length :
  ∀ (diagonal : ℝ) (aspect_width aspect_height : ℕ),
    diagonal = 42 →
    aspect_width = 16 →
    aspect_height = 9 →
    ∃ (horizontal : ℝ),
      horizontal = (aspect_width : ℝ) * diagonal / Real.sqrt ((aspect_width ^ 2 : ℝ) + (aspect_height ^ 2 : ℝ)) ∧
      horizontal = 672 / Real.sqrt 337 := by
  sorry

end hdtv_horizontal_length_l1087_108729


namespace angle_difference_range_l1087_108776

theorem angle_difference_range (α β : ℝ) 
  (h1 : -π < α ∧ α < β ∧ β < π) : 
  -2*π < α - β ∧ α - β < 0 := by
  sorry

end angle_difference_range_l1087_108776


namespace prob_same_suit_is_one_seventeenth_l1087_108764

/-- Represents a standard deck of cards -/
structure Deck :=
  (cards : Fin 52)

/-- Represents the suit of a card -/
inductive Suit
  | Spades | Hearts | Diamonds | Clubs

/-- Represents the rank of a card -/
inductive Rank
  | Ace | Two | Three | Four | Five | Six | Seven | Eight | Nine | Ten | Jack | Queen | King

/-- A function that returns the suit of a card given its index in the deck -/
def cardSuit (card : Fin 52) : Suit :=
  sorry

/-- The probability of drawing two cards of the same suit from a standard deck -/
def probabilitySameSuit : ℚ :=
  1 / 17

/-- Theorem stating that the probability of drawing two cards of the same suit is 1/17 -/
theorem prob_same_suit_is_one_seventeenth :
  probabilitySameSuit = 1 / 17 := by
  sorry

end prob_same_suit_is_one_seventeenth_l1087_108764


namespace max_value_of_s_l1087_108748

theorem max_value_of_s (p q r s : ℝ) 
  (sum_eq : p + q + r + s = 8)
  (sum_prod_eq : p*q + p*r + p*s + q*r + q*s + r*s = 12) :
  s ≤ 2 + 3 * Real.sqrt 2 :=
sorry

end max_value_of_s_l1087_108748


namespace solve_for_z_l1087_108742

theorem solve_for_z (x y z : ℚ) : x = 11 → y = -8 → 2 * x - 3 * z = 5 * y → z = 62 / 3 := by
  sorry

end solve_for_z_l1087_108742


namespace hcf_from_lcm_and_product_l1087_108757

/-- Given two positive integers with LCM 560 and product 42000, their HCF is 75 -/
theorem hcf_from_lcm_and_product (A B : ℕ+) 
  (h_lcm : Nat.lcm A B = 560)
  (h_product : A * B = 42000) :
  Nat.gcd A B = 75 := by
  sorry

end hcf_from_lcm_and_product_l1087_108757


namespace repeating_decimal_to_fraction_l1087_108753

theorem repeating_decimal_to_fraction :
  ∃ (x : ℚ), x = 4 + 35 / 99 ∧ x = 431 / 99 := by
  sorry

end repeating_decimal_to_fraction_l1087_108753


namespace max_value_constraint_l1087_108755

theorem max_value_constraint (a b c : ℝ) (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : 0 ≤ c) 
  (h4 : a^2 + b^2 + c^2 = 1) :
  ∃ (max : ℝ), max = Real.sqrt 3 + Real.sqrt 2 / 3 ∧ 
    ∀ (x y z : ℝ), 0 ≤ x → 0 ≤ y → 0 ≤ z → x^2 + y^2 + z^2 = 1 → 
      x * y * Real.sqrt 3 + y * z * Real.sqrt 3 + z * x * Real.sqrt 2 ≤ max :=
by
  sorry

end max_value_constraint_l1087_108755


namespace acute_triangle_count_l1087_108745

/-- Count of integers satisfying acute triangle conditions --/
theorem acute_triangle_count : 
  (Finset.filter (fun x : ℕ => 
    x > 0 ∧ 
    18 + 36 > x ∧ 
    18 + x > 36 ∧ 
    36 + x > 18 ∧ 
    (x > 36 → x^2 < 18^2 + 36^2) ∧ 
    (x ≤ 36 → 36^2 < 18^2 + x^2))
    (Finset.range 55)).card = 9 := by
  sorry

end acute_triangle_count_l1087_108745


namespace calories_burned_l1087_108797

/-- The number of times players run up and down the bleachers -/
def num_runs : ℕ := 40

/-- The number of stairs in one direction -/
def stairs_one_way : ℕ := 32

/-- The number of calories burned per stair -/
def calories_per_stair : ℕ := 2

/-- The total number of calories burned during the exercise -/
def total_calories : ℕ := num_runs * (2 * stairs_one_way) * calories_per_stair

theorem calories_burned :
  total_calories = 5120 :=
by sorry

end calories_burned_l1087_108797
